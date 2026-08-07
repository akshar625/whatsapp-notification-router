"""Central configuration. Every tunable is env-var driven; secrets are env-only.

MODEL STRINGS LIVE HERE AND NOWHERE ELSE. No other module may hardcode one.

Nothing in this module reads a secret at import time - the `require_*_key()`
helpers are called lazily so the deterministic pipeline runs with no credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader. Real env vars always win over the file."""
    p = path or (REPO_ROOT / ".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


def _path(env_var: str, default: str) -> Path:
    p = Path(os.environ.get(env_var, default))
    return p if p.is_absolute() else REPO_ROOT / p


def _float(env_var: str, default: float) -> float:
    try:
        return float(os.environ[env_var])
    except (KeyError, ValueError):
        return default


def _int(env_var: str, default: int) -> int:
    try:
        return int(os.environ[env_var])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Paths:
    dataset: Path = field(default_factory=lambda: _path("DATASET_DIR", "dataset"))
    cache: Path = field(default_factory=lambda: _path("CACHE_DIR", "cache"))
    recon: Path = field(default_factory=lambda: _path("RECON_DIR", "recon"))
    eval: Path = field(default_factory=lambda: _path("EVAL_DIR", "eval"))
    output_csv: Path = field(default_factory=lambda: _path("OUTPUT_CSV", "output.csv"))

    @property
    def media_images(self) -> Path:
        return self.dataset / "media" / "images"

    @property
    def media_audio(self) -> Path:
        return self.dataset / "media" / "audio"

    @property
    def media_cache(self) -> Path:
        """Media artifacts, keyed by SHA-256 of file bytes (never by filename)."""
        return self.cache / "media"

    def csv(self, name: str) -> Path:
        return self.dataset / name

    def ensure(self) -> None:
        for d in (self.cache, self.media_cache, self.recon, self.eval):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Models:
    """The single source of truth for model identifiers."""

    # --- Media enrichment: vision + any media reasoning. Gemini free tier. ---
    # NOTE: `gemini-2.5-flash` returns 404 "no longer available to new users" on
    # this key. gemini-3.6-flash is the newest stable Flash-series model the key
    # can reach. Pro-series is paid-only and will fail - do not substitute one.
    vlm: str = field(default_factory=lambda: os.environ.get("VLM_MODEL", "gemini-3.6-flash"))

    # The free tier's binding limit is NOT the documented 15 RPM - it is
    # `GenerateRequestsPerDayPerProjectPerModel-FreeTier` = 20 requests per day,
    # *per model*. 33 media artifacts therefore cannot be done on one model in one
    # day. Each model carries its own daily bucket, so we rotate through these
    # (all Flash-series, all free tier, all verified reachable on this key) when a
    # per-day quota is exhausted. Pro-series is deliberately absent: it is
    # paid-only and 404s/429s on this key.
    vlm_fallbacks: tuple = field(default_factory=lambda: tuple(
        s.strip() for s in os.environ.get(
            "VLM_FALLBACK_MODELS",
            "gemini-3.5-flash,gemini-flash-latest,gemini-3-flash-preview,"
            "gemini-3.1-flash-lite,gemini-3.1-flash-lite-preview,gemini-3.5-flash-lite",
        ).split(",") if s.strip()))

    @property
    def vlm_chain(self) -> tuple:
        """Primary model first, then the daily-quota fallbacks, de-duplicated."""
        seen, out = set(), []
        for m in (self.vlm, *self.vlm_fallbacks):
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return tuple(out)

    # --- Decision engine (later stage). Anthropic. ---
    llm: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "claude-sonnet-4-6"))

    # --- Local speech-to-text. No network, no key. ---
    whisper: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL", "large-v3"))
    whisper_device: str = field(default_factory=lambda: os.environ.get("WHISPER_DEVICE", "cpu"))
    whisper_compute: str = field(
        default_factory=lambda: os.environ.get("WHISPER_COMPUTE_TYPE", "int8"))

    # --- Dense retrieval. Local sentence-transformer. ---
    embedding: str = field(default_factory=lambda: os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))


@dataclass(frozen=True)
class RateLimits:
    """Gemini free tier is 15 requests/minute. Exceeding it returns 429."""
    gemini_rpm: int = field(default_factory=lambda: _int("GEMINI_RPM", 15))
    max_retries: int = field(default_factory=lambda: _int("GEMINI_MAX_RETRIES", 6))
    backoff_base_s: float = field(default_factory=lambda: _float("GEMINI_BACKOFF_BASE", 2.0))
    backoff_cap_s: float = field(default_factory=lambda: _float("GEMINI_BACKOFF_CAP", 90.0))

    @property
    def min_interval_s(self) -> float:
        return 60.0 / max(1, self.gemini_rpm)


@dataclass(frozen=True)
class Thresholds:
    near_dup_jaccard: float = field(default_factory=lambda: _float("NEAR_DUP_JACCARD", 0.8))
    retrieval_top_k: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 8))
    hybrid_dense_weight: float = field(default_factory=lambda: _float("HYBRID_DENSE_WEIGHT", 0.5))
    min_candidate_score: float = field(default_factory=lambda: _float("MIN_CANDIDATE_SCORE", 0.05))
    # Below this top hybrid score we emit "none" rather than cite weak evidence.
    min_evidence_score: float = field(default_factory=lambda: _float("MIN_EVIDENCE_SCORE", 0.15))


def _bool(env_var: str, default: bool) -> bool:
    v = os.environ.get(env_var)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    rate: RateLimits = field(default_factory=RateLimits)
    thresholds: Thresholds = field(default_factory=Thresholds)

    # The deterministic fast path (F1/F2) is DISABLED by default.
    # Measured on all 8 rows it covered (eval/fastpath_audit.json): action and
    # message_type identical to the LLM 8/8, driver strictly less specific 8/8.
    # Since `reason` is a scored column, the fast path could only lose points on
    # one axis while gaining nothing on the other two. The rules are kept intact
    # behind this flag so the ablation stays reproducible.
    enable_fast_path: bool = field(default_factory=lambda: _bool("ENABLE_FAST_PATH", False))

    # Label vocabulary, taken verbatim from dataset/sample_messages.csv.
    actions: tuple = ("notify", "digest", "mute")
    message_types: tuple = (
        "urgent", "event", "personal", "business_update", "promotion",
        "scam", "spam", "forward", "greeting", "unknown",
    )

    # Closed vocabulary for the vision extractor.
    poster_categories: tuple = (
        "promotion_sale", "event_announcement", "society_notice", "school_notice",
        "payment_request", "health_notice", "personal_photo", "screenshot_of_chat",
        "news_forward", "religious_greeting", "other",
    )
    visual_qualities: tuple = (
        "clean_professional", "amateur", "heavily_compressed", "screenshot",
    )

    @staticmethod
    def require_gemini_key() -> str:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in. "
                "Secrets are read from the environment only."
            )
        return key

    @staticmethod
    def require_anthropic_key() -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in. "
                "Secrets are read from the environment only."
            )
        return key


CONFIG = Config()

DATASET_FILES = (
    "messages.csv", "sample_messages.csv", "output.csv", "users.csv", "groups.csv",
    "group_members.csv", "business_accounts.csv", "user_business_history.csv",
    "message_history.csv", "message_events.csv", "images.csv", "voice_notes.csv",
    "daily_notification_summary.csv",
)
