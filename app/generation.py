"""Cached JSON generation through NVIDIA's OpenAI-compatible chat endpoint."""

from datetime import datetime, timezone
import hashlib
import json
import socket
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import GENERATION_CACHE_FILE, GENERATION_USAGE_FILE, GENERATION_PROVIDERS, load_generation_settings, project_path

# Identify the client explicitly; some API fronts reject the default "Python-urllib" agent.
USER_AGENT = "meeting-rag/1.0"


class GenerationError(RuntimeError):
    """A sanitized provider, authentication, or structured-response failure."""

    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class NvidiaGenerationClient:
    """Provider adapter; the historical name remains compatible with existing imports."""

    def __init__(self, settings=None, cache_path=GENERATION_CACHE_FILE,
                 usage_path=GENERATION_USAGE_FILE, transport=None, sleep=None):

        self.settings = settings or load_generation_settings()
        self.provider = self.settings.provider
        self.base_url = GENERATION_PROVIDERS[self.provider]["url"]
        self.cache_path = project_path(cache_path)
        self.usage_path = project_path(usage_path)
        self.transport = transport or urlopen
        self.sleep = sleep or time.sleep
        self.last_request = None
        self.api_requests = 0
        self.cache_hits = 0
        # Optional observer for rate-limit retries; receives {"stage", "details"} dicts.
        self.on_event = None

    def generate_json(self, system: str, payload: dict) -> dict:
        """Return a JSON object; never interpret markdown or partial responses."""

        if not isinstance(system, str) or not system.strip() or not isinstance(payload, dict):
            raise ValueError("Generation requires instructions and a JSON object")
        body = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "temperature": 1.0, "top_p": 0.95,
            "max_tokens": self.settings.max_tokens, "stream": False,
            # NVIDIA's OpenAI-compatible endpoint documents this as JSON mode.
            # The strict parser below remains the final validation boundary.
            "response_format": {"type": "json_object"},
        }
        if self.settings.model in ("nvidia/nemotron-3-super-120b-a12b",
                                   "nvidia/nemotron-3.5-lightning-30b-a3b"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        encoded = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_path)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, response TEXT NOT NULL)")
            row = connection.execute("SELECT response FROM responses WHERE cache_key = ?", (key,)).fetchone()
            if row:
                try:
                    result = json.loads(row[0])
                    if isinstance(result, dict):
                        self.cache_hits += 1
                        return result
                except (ValueError, TypeError):
                    pass
            if not self.settings.api_key:
                raise GenerationError(GENERATION_PROVIDERS[self.provider]["key"] + " is required for uncached generation")
            result = self.request(encoded)
            connection.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(result)))
            connection.commit()
            return result
        finally:
            connection.close()

    def available_models(self) -> list[str]:
        """Read the authenticated catalog for diagnostics, without generating text."""

        return sorted({item["id"] for item in self.model_catalog()})

    def model_catalog(self) -> list[dict]:
        """Keep provider metadata so the UI can filter text models and pricing."""

        if not self.settings.api_key:
            raise GenerationError(GENERATION_PROVIDERS[self.provider]["key"] + " is required to check model availability")
        request = Request(self.base_url + "/models",
                          headers={"Authorization": "Bearer " + self.settings.api_key,
                                   "Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with self.transport(request, timeout=self.settings.timeout) as response:
                data = json.load(response)["data"]
            if not isinstance(data, list) or any(not isinstance(item, dict)
                                               or not isinstance(item.get("id"), str) for item in data):
                raise ValueError("Invalid model catalog")
            return data
        except HTTPError as error:
            status = error.code
            error.close()
            raise GenerationError(f"{self.provider} model catalog request failed ({status}); check key and service access", status=status) from None
        except (URLError, TimeoutError, socket.timeout, ConnectionError):
            raise GenerationError(f"Cannot connect to the {self.provider} model catalog; check network access") from None
        except (ValueError, TypeError, KeyError, AttributeError):
            raise GenerationError(f"{self.provider} returned an invalid model catalog") from None

    def request(self, encoded: bytes) -> dict:
        """Retry transient HTTP/network errors and malformed model responses."""

        for attempt in range(self.settings.max_retries + 1):
            if self.last_request is not None:
                delay = 60 / self.settings.requests_per_minute - (time.monotonic() - self.last_request)
                if delay > 0:
                    self.sleep(delay)
            started = time.monotonic()
            self.last_request = started
            self.api_requests += 1
            status, retry_after = "network_error", None
            request = Request(self.base_url + "/chat/completions", data=encoded,
                              headers={"Authorization": "Bearer " + self.settings.api_key,
                                       "Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT})
            try:
                with self.transport(request, timeout=self.settings.timeout) as response:
                    data = json.load(response)
                choice = data["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ValueError("Incomplete response")
                result = json.loads(choice["message"]["content"])
                if not isinstance(result, dict):
                    raise ValueError("Expected JSON object")
                self.log_usage(started, "ok", data.get("usage"))
                return result
            except HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After") if error.headers else None
                error.close()
                retryable = status in (408, 429, 500, 502, 503, 504)
            except (URLError, TimeoutError, socket.timeout, ConnectionError):
                retryable = True
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                status, retryable = "invalid_response", True
            self.log_usage(started, status, None)
            if status == 429 and attempt == self.settings.max_retries:
                raise GenerationError(
                    f"{self.provider} rate limit reached for '{self.settings.model}' after "
                    f"{attempt + 1} attempt(s). Change the model or add a backup model, "
                    "or wait for the quota to reset.", status=429, retry_after=retry_after,
                ) from None
            if status == 402:
                raise GenerationError("NVIDIA account quota or credit restriction reached. "
                                      "Choose another model or wait for the quota to reset.", status=402) from None
            if not retryable or attempt == self.settings.max_retries:
                if status == "invalid_response":
                    raise GenerationError(
                        f"{self.provider} returned invalid or incomplete JSON after {attempt + 1} attempt(s); "
                        "check model and token limit", status="invalid_response",
                    ) from None
                if status == 404:
                    raise GenerationError(
                        f"{self.provider} returned 404 for generation model '{self.settings.model}'. "
                        "The model or its serving route is unavailable for this request. "
                        "Run scripts/run.py --check-generation; select another model with "
                        "--generation-model or GENERATION_MODEL in .env. "
                        "A model listed in the catalog can still fail at inference time.",
                        status=404,
                    ) from None
                raise GenerationError(f"{self.provider} generation request failed ({status}); check key, model access, and quota", status=status) from None
            delay = min(2 ** attempt, 30)
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(int(retry_after), 60))
            if status == 429 and self.on_event is not None:
                self.on_event({"stage": "rate_limit_retry", "details": {
                    "provider": self.provider, "model": self.settings.model, "wait_seconds": delay,
                    "attempt": attempt + 1, "max_attempts": self.settings.max_retries + 1}})
            self.sleep(delay)
        raise GenerationError("Generation retry limit reached")

    def log_usage(self, started, status, usage):
        """Log token counts and elapsed time without prompts, responses, or keys."""

        tokens = {}
        if isinstance(usage, dict):
            tokens = {name: value for name, value in usage.items()
                      if name in ("prompt_tokens", "completion_tokens", "total_tokens")
                      and type(value) is int and value >= 0}
        record = {"time": datetime.now(timezone.utc).isoformat(), "provider": self.provider, "model": self.settings.model,
                  "status": status, "seconds": round(time.monotonic() - started, 3),
                  "usage": tokens or None, "estimated_cost_usd": None}
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


# Preserve existing imports while exposing a provider-neutral name.
ChatGenerationClient = NvidiaGenerationClient
