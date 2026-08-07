"""Deadline extraction and temporal validity.

Media extraction surfaced expired content - `deadline_or_time` values that
predate the message's own `created_at`. This module turns those strings, plus
any date/time expression in `message_text`, into a resolved deadline.

Two rules that matter:

* Relative expressions ("aaj 5 baje", "kal", "tonight", "in 10 minutes") are
  resolved against the MESSAGE'S OWN `created_at`, never against today's real
  clock. Routing must be reproducible whenever the pipeline is re-run.
* For a range ("24-30 JUNE", "2-5 PM", "9 AM to 11 AM") the END of the range is
  the deadline - that is the moment the window actually shuts.

This is a feature extractor. It makes no routing decision.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------- vocabulary
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

# Hindi/Hinglish day words. "kal" is genuinely ambiguous (yesterday OR tomorrow);
# in this corpus it is always forward-looking ("kal milte hain"), so it resolves
# to +1 day. Documented rather than silently assumed.
_DAY_OFFSETS = {
    "today": 0, "aaj": 0, "tonight": 0, "aaj raat": 0, "this evening": 0,
    "is shaam": 0, "abhi": 0, "now": 0, "turant": 0,
    "tomorrow": 1, "kal": 1, "agle din": 1, "tomorrow morning": 1,
    "day after tomorrow": 2, "parso": 2,
}
# Words that pin a time of day when no clock time is given.
_DAYPARTS = {
    "morning": 9, "subah": 9, "noon": 12, "dopahar": 12, "afternoon": 15,
    "evening": 18, "shaam": 18, "night": 21, "raat": 21, "midnight": 23,
    "eod": 18, "end of day": 18,
}

_TIME_RE = re.compile(
    r"\b(?P<h>\d{1,2})[:.](?P<m>\d{2})\s*(?P<ap>am|pm)?\b|"
    r"\b(?P<h2>\d{1,2})\s*(?P<ap2>am|pm)\b|"
    r"\b(?P<h3>\d{1,2})\s*baje\b",
    re.IGNORECASE)

# "24-30 JUNE" / "24 - 30 June"
_RANGE_DM = re.compile(
    r"\b(?P<d1>\d{1,2})\s*[-–—]\s*(?P<d2>\d{1,2})\s+(?P<mon>[A-Za-z]{3,9})\b")
# "November 24-30"
_RANGE_MD = re.compile(
    r"\b(?P<mon>[A-Za-z]{3,9})\s+(?P<d1>\d{1,2})\s*[-–—]\s*(?P<d2>\d{1,2})\b")
# "15th November, 2025" / "15 Nov 2025" / "15 November"
_DMY = re.compile(
    r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3,9})\.?"
    r"(?:,?\s*(?P<y>\d{4}))?\b")
# "November 15, 2025" / "Nov 15"
_MDY = re.compile(
    r"\b(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?\s*(?P<y>\d{4}))?\b")
# "2026-07-30" / "30/07/2026"
_ISO = re.compile(r"\b(?P<y>\d{4})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})\b")
_DMY_NUM = re.compile(r"\b(?P<d>\d{1,2})/(?P<mo>\d{1,2})/(?P<y>\d{2,4})\b")
# "in 10 minutes" / "10 min me" / "next 30 minutes" / "in 2 hours"
_REL_MIN = re.compile(
    r"\b(?:in|next|within|agle)?\s*(?P<n>\d{1,3})\s*"
    r"(?P<unit>min(?:ute)?s?|hours?|hrs?|ghante?|din|days?)\b(?:\s*(?:me|mein))?",
    re.IGNORECASE)


@dataclass
class TemporalValidity:
    """Resolved deadline for a message. Observation only."""
    parsed_deadline: Optional[str]        # ISO 8601, or None
    deadline_source: str                  # "media" | "text" | "none"
    validity: str                         # future | today | past | unparseable
    days_until_deadline: Optional[int]    # signed; negative = already expired
    raw_expression: Optional[str] = None  # the substring we parsed


def _mk(base: datetime, *, month=None, day=None, year=None,
        hour=None, minute=0) -> Optional[datetime]:
    try:
        return datetime(year or base.year, month or base.month, day or base.day,
                        hour if hour is not None else 23,
                        minute if hour is not None else 59)
    except ValueError:
        return None


def _clock(text: str) -> Optional[tuple[int, int]]:
    """Last clock time in the string, as (hour, minute) on a 24h scale."""
    found = None
    for m in _TIME_RE.finditer(text):
        if m.group("h") is not None:
            h, mi, ap = int(m.group("h")), int(m.group("m")), m.group("ap")
        elif m.group("h2") is not None:
            h, mi, ap = int(m.group("h2")), 0, m.group("ap2")
        else:
            h, mi, ap = int(m.group("h3")), 0, None       # "5 baje"
        if h > 23:
            continue
        if ap:
            ap = ap.lower()
            if ap == "pm" and h < 12:
                h += 12
            elif ap == "am" and h == 12:
                h = 0
        elif h <= 11:
            # No meridiem. Society/school/work notices in this corpus mean
            # daytime-to-evening, so 1-11 without am/pm reads as PM. "3.40"
            # in "please reach by 3.40" is 15:40, not 03:40.
            if h >= 1:
                h += 12
        found = (h, mi)
    return found


def _daypart(text: str) -> Optional[int]:
    low = text.lower()
    for word, hour in _DAYPARTS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            return hour
    return None


def parse_deadline(text: str, created_at: datetime) -> tuple[Optional[datetime], Optional[str]]:
    """Best-effort deadline. Returns (datetime, matched_expression)."""
    if not text or not text.strip():
        return None, None
    low = text.lower()

    # 1. Explicit day ranges -> END of the range.
    for rx, order in ((_RANGE_DM, "dm"), (_RANGE_MD, "md")):
        m = rx.search(text)
        if m:
            mon = _MONTHS.get(m.group("mon").lower().rstrip("."))
            if mon:
                d2 = int(m.group("d2"))
                hm = _clock(text)
                dt = _mk(created_at, month=mon, day=d2,
                         hour=hm[0] if hm else None, minute=hm[1] if hm else 0)
                if dt:
                    return dt, m.group(0)

    # 2. Absolute numeric dates.
    m = _ISO.search(text) or _DMY_NUM.search(text)
    if m:
        g = m.groupdict()
        y = int(g["y"]) if g.get("y") else created_at.year
        if y < 100:
            y += 2000
        hm = _clock(text)
        dt = _mk(created_at, year=y, month=int(g["mo"]), day=int(g["d"]),
                 hour=hm[0] if hm else None, minute=hm[1] if hm else 0)
        if dt:
            return dt, m.group(0)

    # 3. Day-month / month-day with an optional year.
    for rx in (_DMY, _MDY):
        for m in rx.finditer(text):
            mon = _MONTHS.get(m.group("mon").lower().rstrip("."))
            if not mon:
                continue
            y = int(m.group("y")) if m.group("y") else created_at.year
            hm = _clock(text)
            dt = _mk(created_at, year=y, month=mon, day=int(m.group("d")),
                     hour=hm[0] if hm else None, minute=hm[1] if hm else 0)
            if dt:
                return dt, m.group(0)

    # 4. Relative offsets in minutes/hours/days.
    m = _REL_MIN.search(low)
    if m:
        n, unit = int(m.group("n")), m.group("unit").lower()
        if unit.startswith(("min", "ghant")) or unit.startswith("hour") or unit.startswith("hr"):
            delta = (timedelta(minutes=n) if unit.startswith("min")
                     else timedelta(hours=n))
            return created_at + delta, m.group(0)
        if unit.startswith(("day", "din")):
            return created_at + timedelta(days=n), m.group(0)

    # 5. Relative day words, optionally with a clock time or daypart.
    best: Optional[tuple[int, str]] = None
    for word, off in _DAY_OFFSETS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            if best is None or len(word) > len(best[1]):
                best = (off, word)
    hm = _clock(text)
    if best is not None:
        day = created_at + timedelta(days=best[0])
        if hm:
            h, mi = hm
        else:
            dp = _daypart(text)
            h, mi = (dp, 0) if dp is not None else (23, 59)
        return day.replace(hour=h, minute=mi, second=0, microsecond=0), best[1]

    # 6. A bare clock time means "today" relative to the message.
    if hm:
        return created_at.replace(hour=hm[0], minute=hm[1], second=0,
                                  microsecond=0), f"{hm[0]:02d}:{hm[1]:02d}"

    # 7. A bare daypart word.
    dp = _daypart(text)
    if dp is not None:
        return created_at.replace(hour=dp, minute=0, second=0, microsecond=0), None

    return None, None


def build_temporal_validity(message_text: str, media_deadline: Optional[str],
                            created_at: Optional[datetime]) -> TemporalValidity:
    """Media deadline wins when present; it is an explicit, extracted field."""
    if created_at is None:
        return TemporalValidity(None, "none", "unparseable", None)

    dt, expr, source = None, None, "none"
    if media_deadline and media_deadline.strip():
        dt, expr = parse_deadline(media_deadline, created_at)
        if dt is not None:
            source = "media"
        else:
            # A deadline string was present but could not be resolved - that is
            # "unparseable", which is different from "no deadline at all".
            return TemporalValidity(None, "media", "unparseable", None,
                                    media_deadline.strip()[:120])
    if dt is None and message_text:
        dt, expr = parse_deadline(message_text, created_at)
        if dt is not None:
            source = "text"

    if dt is None:
        return TemporalValidity(None, "none", "unparseable", None)

    delta_days = (dt.date() - created_at.date()).days
    if dt.date() == created_at.date():
        validity = "today"
    elif dt < created_at:
        validity = "past"
    else:
        validity = "future"

    return TemporalValidity(
        parsed_deadline=dt.isoformat(timespec="minutes"),
        deadline_source=source, validity=validity,
        days_until_deadline=delta_days,
        raw_expression=(expr or "")[:120] or None,
    )
