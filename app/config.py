"""Shared settings. Paths are independent of the shell working directory."""

from pathlib import Path
from dataclasses import dataclass, field
import os
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
ANSWERS_FILE = PROJECT_ROOT / "outputs" / "answers.json"
CHUNK_TARGET = 400
CHUNK_MAXIMUM = 500
CHUNK_OVERLAP = 50
BM25_INDEX_FILE = PROCESSED_DIR / "bm25.sqlite3"
SEARCH_LIMIT = 5
# Special team value: search every team's meetings instead of one team.
ALL_TEAMS = "ALL"
BUILD_LOG_FILE = PROCESSED_DIR / "build.log"
VECTOR_INDEX_FILE = PROCESSED_DIR / "vectors.sqlite3"
EMBEDDING_CACHE_FILE = CACHE_DIR / "embeddings.sqlite3"
EMBEDDING_USAGE_FILE = PROCESSED_DIR / "embedding_usage.jsonl"
ENV_FILE = PROJECT_ROOT / ".env"
EVAL_QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "my_questions.json"
EVAL_RESULTS_FILE = PROJECT_ROOT / "evaluation" / "results.md"
EVAL_RUNS_DIR = PROCESSED_DIR / "evaluation_runs"
MEMORY_CONTEXT_MAX_WORDS = 2000
GENERATION_CACHE_FILE = CACHE_DIR / "generation.sqlite3"
GENERATION_USAGE_FILE = PROCESSED_DIR / "generation_usage.jsonl"
MODEL_CHECKS_DIR = PROCESSED_DIR / "model_checks"
# Generation uses NVIDIA only (the same key as embeddings).
GENERATION_PROVIDERS = {
    "nvidia": {"url": "https://integrate.api.nvidia.com/v1", "key": "NVIDIA_API_KEY",
               "model": "nvidia/nemotron-3.5-lightning-30b-a3b"},
}


def project_path(path: str | Path) -> Path:
    """Resolve a relative project path without depending on the current directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_env_file(path=ENV_FILE):
    """Read simple KEY=value settings without changing process environment."""

    values = {}
    path = Path(path)
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator:
            values[name.strip()] = value.strip().strip("\"'")
    return values


@dataclass(frozen=True)
class EmbeddingSettings:
    """NVIDIA embedding settings; secrets are excluded from representations."""

    api_key: str = field(default="", repr=False)
    model: str = "nvidia/nemotron-3.5-embed-8b"
    dimensions: int = 4096
    batch_size: int = 16
    requests_per_minute: int = 10
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self):

        if self.model != "nvidia/nemotron-3.5-embed-8b":
            raise ValueError("This adapter supports nvidia/nemotron-3.5-embed-8b only")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 4096:
            raise ValueError("Embedding dimensions must be between 1 and 4096")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 100:
            raise ValueError("Embedding batch_size must be between 1 and 100")
        if type(self.requests_per_minute) is not int or self.requests_per_minute < 1:
            raise ValueError("Embedding requests_per_minute must be positive")
        if not math.isfinite(self.timeout) or self.timeout <= 0 or type(self.max_retries) is not int or not 0 <= self.max_retries <= 5:
            raise ValueError("Invalid embedding timeout or retry count")


def load_embedding_settings(env_file=ENV_FILE):
    """Load settings with process variables taking precedence over .env."""

    values = read_env_file(env_file)

    def setting(name, default):

        return os.environ.get(name, values.get(name, default))

    return EmbeddingSettings(
        api_key=setting("NVIDIA_API_KEY", ""),
        model=setting("EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-8b"),
        dimensions=int(setting("EMBEDDING_DIMENSIONS", "4096")),
        batch_size=int(setting("EMBEDDING_BATCH_SIZE", "16")),
        requests_per_minute=int(setting("EMBEDDING_REQUESTS_PER_MINUTE", "10")),
        timeout=float(setting("EMBEDDING_TIMEOUT", "60")),
        max_retries=int(setting("EMBEDDING_MAX_RETRIES", "3")),
    )


@dataclass(frozen=True)
class GenerationSettings:
    """Settings for bounded provider chat requests, separate from embeddings."""

    api_key: str = field(default="", repr=False)
    model: str = "nvidia/nemotron-3-super-120b-a12b"
    max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 2
    requests_per_minute: int = 10
    provider: str = "nvidia"

    def __post_init__(self):

        if self.provider not in GENERATION_PROVIDERS:
            raise ValueError("Generation provider must be nvidia")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Generation model must be nonempty")
        if type(self.max_tokens) is not int or not 256 <= self.max_tokens <= 8192:
            raise ValueError("Generation max_tokens must be between 256 and 8192")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Generation timeout must be positive")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 5:
            raise ValueError("Generation max_retries must be between zero and five")
        if type(self.requests_per_minute) is not int or self.requests_per_minute < 1:
            raise ValueError("Generation requests_per_minute must be positive")


def load_generation_settings(env_file=ENV_FILE, provider=None, model=None):
    """Load NVIDIA generation settings; process variables take precedence over .env."""

    values = read_env_file(env_file)

    def setting(name, default):

        return os.environ.get(name, values.get(name, default))

    selected = provider or "nvidia"
    if selected not in GENERATION_PROVIDERS:
        raise ValueError("Only the nvidia generation provider is supported")
    spec = GENERATION_PROVIDERS[selected]
    return GenerationSettings(
        provider=selected,
        api_key=setting(spec["key"], ""),
        model=model or setting("GENERATION_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
        max_tokens=int(setting("GENERATION_MAX_TOKENS", "2048")),
        timeout=float(setting("GENERATION_TIMEOUT", "120")),
        max_retries=int(setting("GENERATION_MAX_RETRIES", "2")),
        requests_per_minute=int(setting("GENERATION_REQUESTS_PER_MINUTE", "10")),
    )
