"""Stage 3: history retrieval.

For each incoming message we build a candidate pool from `message_history`,
scoped to the SAME user, then narrowed by same sender / same group / same
business. On top of that pool:

  a) near-duplicate detection (exact normalized match + token-Jaccard), joined
     to message_events so the decision layer can see how this user *reacted*
     the previous times they saw near-identical text;
  b) hybrid ranking - BM25 (or TF-IDF) lexical + sentence-transformer dense
     cosine - returning the top K with scores;
  c) every candidate carries its full message_events row.

Evidence ids may ONLY ever come from message_history: recon confirmed all 31
gold evidence ids resolve there and 0 resolve into messages.csv.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from context import Dataset, _f, _i, _s
from schema import (
    ContextPack, EventRecord, HistoryCandidate, NearDuplicateStats, RetrievalBundle,
)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def query_text(pack: ContextPack) -> str:
    """The text to retrieve against.

    A voice note carries NO message_text at all - 8 of the 110 incoming messages
    and 3 of the 30 solved samples are in that state. Ranking on the empty string
    scores every candidate at zero, which then trips the min-evidence gate and
    suppresses evidence that was actually sitting at rank 1. The cached
    transcript (or OCR text) is the message's real content, so it is the query.
    """
    if pack.message.text and pack.message.text.strip():
        return pack.message.text
    md = pack.media or {}
    return (md.get("transcript_raw") or md.get("ocr_text") or "").strip()


# ------------------------------------------------------------------- lexical
class BM25:
    """Okapi BM25. Deterministic, no training, no network."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = (sum(len(d) for d in corpus) / self.N) if self.N else 0.0
        self.tfs = [Counter(d) for d in corpus]
        df: Counter = Counter()
        for d in corpus:
            df.update(set(d))
        # +1 inside the log keeps idf non-negative for terms in every document
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype=float)
        if not self.N:
            return out
        for i, tf in enumerate(self.tfs):
            dl = len(self.corpus[i]) or 1
            s = 0.0
            for t in query:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# --------------------------------------------------------------------- dense
class DenseEncoder:
    """Sentence-transformer cosine. Multilingual so Hinglish/Devanagari embed
    sensibly. Degrades to lexical-only if the model cannot be loaded offline."""

    def __init__(self):
        self._model = None
        self._failed = False
        self._cache: dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        return not self._failed and self._load() is not None

    def _load(self):
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(CONFIG.models.embedding)
            except Exception as exc:  # noqa: BLE001
                print(f"[retrieval] dense encoder unavailable ({type(exc).__name__}); "
                      f"falling back to lexical-only ranking")
                self._failed = True
        return self._model

    def encode(self, texts: list[str]) -> Optional[np.ndarray]:
        m = self._load()
        if m is None:
            return None
        todo = [t for t in texts if t not in self._cache]
        if todo:
            vecs = m.encode(todo, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(todo, vecs):
                self._cache[t] = np.asarray(v, dtype=float)
        return np.vstack([self._cache[t] for t in texts]) if texts else None


# ----------------------------------------------------------------- retriever
class HistoryRetriever:
    def __init__(self, ds: Dataset, encoder: Optional[DenseEncoder] = None):
        self.ds = ds
        self.encoder = encoder if encoder is not None else DenseEncoder()
        self._index()

    def _index(self) -> None:
        hist = self.ds.message_history
        self.rows: dict[str, pd.Series] = {}
        self.by_user: dict[str, list[str]] = defaultdict(list)
        self.toks: dict[str, list[str]] = {}
        self.tokset: dict[str, set[str]] = {}
        self.norm: dict[str, str] = {}
        for _, r in hist.iterrows():
            mid = str(r["message_id"])
            self.rows[mid] = r
            self.by_user[str(r["user_id"])].append(mid)
            text = r["message_text"] if isinstance(r["message_text"], str) else ""
            tk = tokens(text)
            self.toks[mid] = tk
            self.tokset[mid] = set(tk)
            self.norm[mid] = normalize(text)

    # -- events -------------------------------------------------------------
    def events_for(self, history_id: str) -> EventRecord:
        row = self.ds._events.get(history_id)
        if row is None:
            return EventRecord.absent(history_id)
        return EventRecord(
            message_id=history_id, user_id=_s(row.get("user_id")),
            message_opened=_i(row.get("message_opened")),
            message_replied=_i(row.get("message_replied")),
            reaction_time_minutes=_f(row.get("reaction_time_minutes")),
            notification_dismissed=_i(row.get("notification_dismissed")),
            muted_after_message=_i(row.get("muted_after_message")),
            message_reported=_i(row.get("message_reported")),
            has_record=True,
        )

    # -- pool ---------------------------------------------------------------
    def candidate_pool(self, pack: ContextPack) -> tuple[list[str], dict[str, int],
                                                         list[str]]:
        """Same user, then scoped by same sender OR same group OR same business.

        Returns (scoped_ids, scope_counts, all_ids_for_this_user).
        """
        user_ids = self.by_user.get(pack.user_id, [])
        sender = pack.message.sender_user_id
        gid = pack.group.group_id if pack.group else None
        bid = pack.business.business_id if pack.business else None

        scoped: list[str] = []
        counts = {"sender": 0, "group": 0, "business": 0}
        for mid in user_ids:
            r = self.rows[mid]
            hit = None
            if sender and _s(r.get("sender_user_id")) == sender:
                hit = "sender"
            elif gid and _s(r.get("group_id")) == gid:
                hit = "group"
            elif bid and _s(r.get("business_id")) == bid:
                hit = "business"
            if hit:
                counts[hit] += 1
                scoped.append(mid)
        return scoped, counts, user_ids

    # -- near duplicates ----------------------------------------------------
    def near_duplicates(self, pack: ContextPack, scoped: list[str],
                        user_all: list[str]) -> NearDuplicateStats:
        """Recon found 19 exact text duplicates between messages.csv and
        message_history.csv - the strongest single signal in the dataset."""
        thr = CONFIG.thresholds.near_dup_jaccard
        qt = query_text(pack)
        q_norm = normalize(qt)
        q_set = set(tokens(qt))

        def matches(ids: list[str]) -> list[tuple[str, float, bool]]:
            out = []
            if not q_norm:
                return out
            for mid in ids:
                exact = self.norm[mid] == q_norm
                j = 1.0 if exact else jaccard(q_set, self.tokset[mid])
                if exact or j >= thr:
                    out.append((mid, j, exact))
            return out

        scoped_hits = matches(scoped)
        global_hits = matches(user_all)

        stats = NearDuplicateStats(
            occurrences=len(scoped_hits),
            matched_history_ids=[m for m, _, _ in scoped_hits],
            max_jaccard=round(max((j for _, j, _ in scoped_hits), default=0.0), 4),
            exact_matches=sum(1 for _, _, e in scoped_hits if e),
            times_opened=0, times_replied=0, times_dismissed=0,
            times_muted_after=0, times_reported=0, events_available=0,
            occurrences_user_global=len(global_hits),
            matched_history_ids_user_global=[m for m, _, _ in global_hits],
        )
        # Aggregate the user's actual reactions across every prior occurrence.
        for mid, _, _ in global_hits:
            ev = self.events_for(mid)
            if not ev.has_record:
                continue
            stats.events_available += 1
            stats.times_opened += ev.message_opened or 0
            stats.times_replied += ev.message_replied or 0
            stats.times_dismissed += ev.notification_dismissed or 0
            stats.times_muted_after += ev.muted_after_message or 0
            stats.times_reported += ev.message_reported or 0
        return stats

    # -- ranking ------------------------------------------------------------
    def rank(self, pack: ContextPack, scoped: list[str],
             nd: NearDuplicateStats) -> list[HistoryCandidate]:
        if not scoped:
            return []
        q_tokens = tokens(query_text(pack))
        corpus = [self.toks[m] for m in scoped]

        lex = BM25(corpus).scores(q_tokens) if q_tokens else np.zeros(len(scoped))
        lex_n = _minmax(lex)

        dense_n = np.zeros(len(scoped))
        if q_tokens and self.encoder.available:
            texts = [query_text(pack)] + [
                (self.rows[m]["message_text"] if isinstance(self.rows[m]["message_text"], str)
                 else "") for m in scoped]
            emb = self.encoder.encode(texts)
            if emb is not None:
                dense_n = np.clip(emb[1:] @ emb[0], 0.0, 1.0)

        w = CONFIG.thresholds.hybrid_dense_weight
        if not self.encoder.available:
            hybrid = lex_n                     # lexical-only fallback
        else:
            hybrid = w * dense_n + (1 - w) * lex_n

        q_set = set(q_tokens)
        dup = set(nd.matched_history_ids)
        cands: list[HistoryCandidate] = []
        for i, mid in enumerate(scoped):
            r = self.rows[mid]
            text = r["message_text"] if isinstance(r["message_text"], str) else ""
            scope = ("sender" if _s(r.get("sender_user_id")) == pack.message.sender_user_id
                     else "group" if pack.group and _s(r.get("group_id")) == pack.group.group_id
                     else "business" if pack.business
                     and _s(r.get("business_id")) == pack.business.business_id
                     else "user_global")
            cands.append(HistoryCandidate(
                history_message_id=mid, scope=scope,
                conversation_type=_s(r.get("conversation_type")),
                created_at=_s(r.get("created_at")), text=text,
                media_type=_s(r.get("media_type")),
                forwarded_count=_i(r.get("forwarded_count")),
                lexical_score=round(float(lex_n[i]), 4),
                dense_score=round(float(dense_n[i]), 4),
                hybrid_score=round(float(hybrid[i]), 4),
                jaccard=round(jaccard(q_set, self.tokset[mid]), 4),
                is_near_duplicate=mid in dup,
                events=self.events_for(mid),
            ))

        # Near-duplicates always outrank on ties: they are the strongest signal.
        cands.sort(key=lambda c: (c.is_near_duplicate, c.hybrid_score, c.jaccard),
                   reverse=True)
        keep = [c for c in cands
                if c.is_near_duplicate
                or c.hybrid_score >= CONFIG.thresholds.min_candidate_score]
        return (keep or cands)[:CONFIG.thresholds.retrieval_top_k]

    # -- bundle -------------------------------------------------------------
    def retrieve(self, pack: ContextPack) -> RetrievalBundle:
        scoped, counts, user_all = self.candidate_pool(pack)
        nd = self.near_duplicates(pack, scoped, user_all)
        cands = self.rank(pack, scoped, nd)
        return RetrievalBundle(
            message_id=pack.message_id, user_id=pack.user_id,
            pool_size=len(scoped), pool_scopes=counts,
            user_history_size=len(user_all), near_duplicates=nd, candidates=cands,
        )

    def retrieve_all(self, packs: list[ContextPack]) -> dict[str, RetrievalBundle]:
        return {p.message_id: self.retrieve(p) for p in packs}
