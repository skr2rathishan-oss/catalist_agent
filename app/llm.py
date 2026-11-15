"""NVIDIA embedding adapter with a resumable cache and bounded network retries."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import socket
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import EMBEDDING_CACHE_FILE, EMBEDDING_USAGE_FILE, load_embedding_settings, project_path
from app.embeddings import FORMAT_VERSION, normalise_vector, prepare_embedding_text

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Plain, credential-free error from the embedding provider."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NvidiaEmbeddingClient:
    """Use NVIDIA's text batches with explicit passage and query input types."""

    def __init__(self, settings=None, cache_path=EMBEDDING_CACHE_FILE,
                 usage_path=EMBEDDING_USAGE_FILE, transport=None, sleep=None):

        self.settings = settings or load_embedding_settings()
        self.cache_path = project_path(cache_path)
        self.usage_path = project_path(usage_path)
        self.transport = transport or urlopen
        self.sleep = sleep or time.sleep
        self.last_request = None
        self.api_requests = 0
        self.cache_hits = 0

    def embed(self, texts, task="document"):
        """Embed a list of passages or questions, using cache before API access."""

        if not isinstance(texts, list):
            raise ValueError("texts must be a list")
        prepared = [prepare_embedding_text(text, task) for text in texts]
        if not prepared:
            return []
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_path)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
            keys = []
            vectors = {}
            missing = {}
            for text in prepared:
                identity = [self.settings.model, self.settings.dimensions, FORMAT_VERSION, task, text]
                key = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()
                keys.append(key)
                if key in vectors or key in missing:
                    continue
                cached = connection.execute("SELECT vector FROM embeddings WHERE cache_key = ?", (key,)).fetchone()
                if cached:
                    try:
                        vectors[key] = normalise_vector(json.loads(cached[0]), self.settings.dimensions)
                        self.cache_hits += 1
                        continue
                    except (ValueError, TypeError):
                        pass  # A damaged cache entry is regenerated, never trusted.
                missing[key] = text
            pending = list(missing)
            for start in range(0, len(pending), self.settings.batch_size):
                batch_keys = pending[start:start + self.settings.batch_size]
                batch_texts = [missing[key] for key in batch_keys]
                batch_vectors = self.request_batch(batch_texts, task)
                for key, vector in zip(batch_keys, batch_vectors):
                    connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(vector)))
                    vectors[key] = vector
                # Preserve completed batches even if a later batch fails.
                connection.commit()
            return [vectors[key] for key in keys]
        finally:
            connection.close()

    def request_batch(self, texts, task):
        """Call NVIDIA's embeddings endpoint, disabling silent input truncation."""

        if not self.settings.api_key:
            raise EmbeddingError("Set NVIDIA_API_KEY in .env before generating embeddings")
        body = {
            "model": self.settings.model,
            "input": texts,
            "input_type": "passage" if task == "document" else "query",
            "dimensions": self.settings.dimensions,
            "encoding_format": "float",
            "truncate": "NONE",
        }
        request = Request(
            "https://integrate.api.nvidia.com/v1/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.settings.api_key}"},
            method="POST",
        )
        for attempt in range(self.settings.max_retries + 1):
            self.wait_for_rate_limit()
            started = time.monotonic()
            self.last_request = started
            self.api_requests += 1
            status = "network_error"
            retry_after = None
            try:
                with self.transport(request, timeout=self.settings.timeout) as response:
                    data = json.load(response)
                embeddings = data.get("data", [])
                if len(embeddings) != len(texts):
                    raise ValueError("Unexpected embedding count")
                # Providers may return embeddings out of order; index must map
                # every input exactly once, or evidence would get the wrong vector.
                indices = [item["index"] for item in embeddings]
                if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(texts))):
                    raise ValueError("Invalid embedding indices")
                embeddings.sort(key=lambda item: item["index"])
                vectors = []
                for embedding in embeddings:
                    vectors.append(normalise_vector(embedding.get("embedding"), self.settings.dimensions))
                self.log_usage(task, len(texts), started, "ok", data.get("usage"))
                return vectors
            except HTTPError as error:
                status = error.code
                if error.headers:
                    retry_after = error.headers.get("Retry-After")
                error.close()
                retryable = status in (408, 429, 500, 502, 503, 504)
            except (URLError, TimeoutError, socket.timeout, ConnectionError):
                retryable = True
            except (ValueError, TypeError, AttributeError, KeyError):
                self.log_usage(task, len(texts), started, "invalid_response", None)
                raise EmbeddingError("NVIDIA returned an invalid embedding response") from None
            self.log_usage(task, len(texts), started, status, None)
            if not retryable or attempt == self.settings.max_retries:
                raise EmbeddingError(f"NVIDIA embedding request failed ({status}); check key, model access, and quota",
                                     status=status if isinstance(status, int) else None) from None
            delay = min(2 ** attempt, 30)
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(int(retry_after), 60))
            self.sleep(delay)
        raise EmbeddingError("Embedding retry limit reached")

    def wait_for_rate_limit(self):
        """Space requests according to this account's configured request budget."""

        if self.last_request is not None:
            interval = 60 / self.settings.requests_per_minute
            delay = interval - (time.monotonic() - self.last_request)
            if delay > 0:
                self.sleep(delay)

    def log_usage(self, task, items, started, status, usage):
        """Log actual provider usage when supplied; never log text or credentials."""

        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "model": self.settings.model, "dimensions": self.settings.dimensions,
            "task": task, "items": items, "status": status,
            "seconds": round(time.monotonic() - started, 3),
            "usage": usage, "estimated_cost_usd": None,
        }
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        logger.info("NVIDIA embedding batch: status=%s items=%s seconds=%s", status, items, record["seconds"])
