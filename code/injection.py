"""Deterministic detection of text that addresses an automated system.

DETERMINISTIC FALLBACK. The vision prompt already asks the extractor to flag
"embedded instructions to an automated system" in `scam_indicators`, but that
path costs a Gemini call and the free-tier daily quota is spent. This module
re-derives the same signal from already-cached `ocr_text` / transcript / message
text with pure regex, so nothing re-runs.

Detection is by FEATURE - phrasing that addresses a router, an assistant, or a
classifier - never by message id. A message that tries to steer the router is a
fact about the message and evidence of manipulation; it must never actually
steer anything.
"""
from __future__ import annotations

import re
from typing import Optional

# Each pattern targets a *shape* of instruction, not a specific message.
_PATTERNS: tuple[re.Pattern, ...] = (
    # "system note for notification router", "internal router metadata",
    # "assistant instruction", "routing override"
    re.compile(r"\b(system|assistant|internal|router|routing|agent|model|ai)\s*"
               r"(note|notes|message|instruction|instructions|metadata|override|"
               r"directive|prompt|command)\b", re.I),
    re.compile(r"\b(notification\s+router|message\s+router|the\s+router|"
               r"classifier|routing\s+system|automated\s+system)\b", re.I),
    # "set action=notify", "mark this as notify", "classify as urgent"
    re.compile(r"\b(set|mark|classify|treat|flag|label|route|categorize|categorise)\b"
               r"[^.\n]{0,40}\b(as\s+)?(notify|digest|mute|urgent|high\s+priority|"
               r"important|spam)\b", re.I),
    re.compile(r"\b(action|priority|confidence|user_priority|verified_business|"
               r"message_type)\s*[:=]\s*\S+", re.I),
    # "ignore sender risk", "disregard previous instructions", "override the rules"
    re.compile(r"\b(ignore|disregard|bypass|skip|override|forget)\b[^.\n]{0,40}"
               r"\b(risk|instruction|instructions|rule|rules|policy|filter|"
               r"previous|prior|above|sender)\b", re.I),
    # "always mark this as", "do not mute this"
    re.compile(r"\balways\s+(mark|treat|classify|set|route)\b", re.I),
    re.compile(r"\bdo\s+not\s+(mute|filter|block|suppress|ignore)\s+(this|it)\b", re.I),
)

# A window of context to quote back, so the decision layer can see the evidence.
_MAX_QUOTE = 240


def scan(text: Optional[str]) -> tuple[bool, Optional[str]]:
    """(has_embedded_instruction, embedded_instruction_text).

    Returns the matched sentence(s), trimmed, so the finding is auditable.
    """
    if not text or not text.strip():
        return False, None
    hits: list[str] = []
    for rx in _PATTERNS:
        for m in rx.finditer(text):
            # Quote the sentence the match sits in, not just the token.
            start = text.rfind(".", 0, m.start()) + 1
            end = text.find(".", m.end())
            end = len(text) if end == -1 else end + 1
            frag = text[start:end].strip()
            if frag and frag not in hits:
                hits.append(frag)
    if not hits:
        return False, None
    joined = " | ".join(hits)
    return True, joined[:_MAX_QUOTE]


def scan_many(*texts: Optional[str]) -> tuple[bool, Optional[str]]:
    """Scan several untrusted surfaces (message text, OCR, transcript) at once."""
    found, frags = False, []
    for t in texts:
        hit, frag = scan(t)
        if hit:
            found = True
            if frag:
                frags.append(frag)
    if not found:
        return False, None
    return True, " || ".join(frags)[:_MAX_QUOTE]
