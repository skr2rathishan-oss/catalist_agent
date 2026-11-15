"""Discover/probe generation models and route bounded automatic fallbacks."""

from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
import time

from app.agents.prompts import GENERATION_CHECK_PROMPT
from app.config import (EMBEDDING_USAGE_FILE, GENERATION_PROVIDERS, GENERATION_USAGE_FILE,
                        read_env_file, load_generation_settings)
from app.generation import ChatGenerationClient, GenerationError

RATE_LIMIT_STATUSES = (429, 402)


def key_status():
    """Report configured key names, never credential values."""

    values = read_env_file()
    return {provider: bool(os.environ.get(spec["key"], values.get(spec["key"], "")))
            for provider, spec in GENERATION_PROVIDERS.items()}


def usage_summary(generation_path=GENERATION_USAGE_FILE, embedding_path=EMBEDDING_USAGE_FILE,
                  window_minutes=60, tail=5000, now=None):
    """Summarise recent request outcomes per model from the key-free usage logs.

    Returns one row per (kind, provider, model), newest activity first, so a
    dashboard can show which routes are currently rate limited.
    """

    now = now or datetime.now(timezone.utc)
    since = now - timedelta(minutes=window_minutes)
    rows = {}
    for kind, path in (("generation", generation_path), ("embedding", embedding_path)):
        try:
            with open(path, encoding="utf-8") as file:
                lines = deque(file, maxlen=tail)
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
                stamp = datetime.fromisoformat(record["time"])
            except (ValueError, KeyError, TypeError):
                continue
            if stamp < since:
                continue
            provider = record.get("provider", "nvidia")
            key = (kind, provider, record.get("model", "?"))
            row = rows.setdefault(key, {"kind": kind, "provider": provider, "model": key[2],
                                        "requests": 0, "ok": 0, "rate_limited": 0,
                                        "last_status": None, "last_time": None, "last_rate_limit": None})
            status = record.get("status")
            row["requests"] += 1
            row["ok"] += status == "ok"
            if status in RATE_LIMIT_STATUSES:
                row["rate_limited"] += 1
                row["last_rate_limit"] = stamp
            if row["last_time"] is None or stamp >= row["last_time"]:
                row["last_time"], row["last_status"] = stamp, status
    result = sorted(rows.values(), key=lambda row: row["last_time"], reverse=True)
    for row in result:
        row["state"] = ("rate limited" if row["last_status"] in RATE_LIMIT_STATUSES
                        else "ok" if row["last_status"] == "ok" else f"error ({row['last_status']})")
    return result


def create_client(provider, model=None, **settings):
    """Construct a client with a provider-specific credential and fixed endpoint."""

    config = replace(load_generation_settings(provider=provider, model=model), **settings)
    return ChatGenerationClient(config)


def discover_models(provider="nvidia", client=None):
    """Return NVIDIA text-model candidates; presence alone never means inference passed."""

    client = client or create_client(provider, timeout=20)
    rows = []
    excluded = ("embed", "rerank", "reward", "guard", "safety", "whisper", "tts", "audio",
                "speech", "video", "diffusion", "parse", "nvclip", "deplot", "compound",
                "flux", "sdxl", "cosmos", "fuyu", "kosmos", "riva", "translate")
    for item in client.model_catalog():
        identifier = item["id"]
        if item.get("active") is False or any(word in identifier.lower() for word in excluded):
            continue
        architecture = item.get("architecture") or {}
        if architecture.get("output_modalities") and "text" not in architecture["output_modalities"]:
            continue
        if architecture.get("input_modalities") and "text" not in architecture["input_modalities"]:
            continue
        context = item.get("context_length", item.get("context_window"))
        if isinstance(context, (int, float)) and context < 32768:
            continue
        rows.append({"provider": provider, "model": identifier, "context_length": context,
                     "status": "catalog-listed; not tested"})
    return sorted(rows, key=lambda row: row["model"])


def probe_model(provider, model, client=None):
    """Make one fresh tiny request, not a cached success or a full RAG benchmark."""

    client = client or create_client(provider, model, timeout=25, max_retries=0, max_tokens=256)
    record = {"provider": provider, "model": model, "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = client.generate_json(GENERATION_CHECK_PROMPT, {"operation": "model_probe", "nonce": time.time_ns()})
        if set(response) != {"ok"} or response["ok"] is not True:
            raise GenerationError("Model did not return the required JSON object")
        record["status"] = "JSON check passed"
    except GenerationError as error:
        record.update(status="rate limited" if error.status == 429 else "check failed", reason=str(error))
    return record


class GenerationRouter:
    """Sticky fallback after exhausted transient failures, through selected routes."""

    def __init__(self, routes, on_event=None, client_factory=create_client):

        if not 1 <= len(routes) <= 3 or len(set(routes)) != len(routes):
            raise ValueError("Choose one primary and at most two distinct backup models")
        if any(provider not in GENERATION_PROVIDERS or not isinstance(model, str) or not model.strip()
               for provider, model in routes):
            raise ValueError("Each route requires a supported provider and explicit model ID")
        self.clients = [client_factory(provider, model, max_retries=1) for provider, model in routes]
        self.index = 0
        self.on_event = on_event
        self.history = [{"provider": provider, "model": model} for provider, model in routes[:1]]

    @property
    def settings(self):
        return self.clients[self.index].settings

    @property
    def api_requests(self):
        return sum(client.api_requests for client in self.clients)

    @property
    def cache_hits(self):
        return sum(client.cache_hits for client in self.clients)

    def generate_json(self, system, payload):

        while True:
            client = self.clients[self.index]
            try:
                return client.generate_json(system, payload)
            except GenerationError as error:
                fallback_reasons = {429: "HTTP 429", 408: "HTTP 408", 500: "HTTP 500",
                                    502: "HTTP 502", 503: "HTTP 503", 504: "HTTP 504",
                                    "network_error": "network error"}
                if error.status not in fallback_reasons or self.index + 1 >= len(self.clients):
                    raise
                previous = self.history[-1]
                self.index += 1
                selected = self.settings
                current = {"provider": selected.provider, "model": selected.model}
                self.history.append(current)
                if self.on_event:
                    self.on_event({"stage": "model_fallback", "details": {
                        "from": previous, "to": current, "reason": fallback_reasons[error.status]}})
