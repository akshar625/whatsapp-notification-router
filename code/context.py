"""Stage 0: load the 13 CSVs, join them, and emit one ContextPack per message.

Deterministic and side-effect free apart from an optional on-disk media probe
cache. No network, no model calls.
"""
from __future__ import annotations

import contextlib
import json
import hashlib
import re
import subprocess
import wave
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import injection
from config import CONFIG
from schema import (
    BaselineLoad, BusinessContext, ContextPack, GroupContext, MessageContext,
    RelationshipContext, TemplateStats, TemporalContext, TemporalValidityContext,
    UserContext,
)
from temporal import build_temporal_validity

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# ----------------------------------------------------------------- small helpers
def _s(v: Any) -> Optional[str]:
    """CSV cell -> clean str or None. pandas NaN is not a string."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def _i(v: Any) -> Optional[int]:
    s = _s(v)
    if s is None:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _f(v: Any) -> Optional[float]:
    s = _s(v)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _b(v: Any) -> Optional[bool]:
    i = _i(v)
    return None if i is None else bool(i)


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Safe ratio. Returns None when the denominator is unusable - never 0.0,
    which would be indistinguishable from a real zero rate."""
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 4)


def parse_quiet_window(raw: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """'22:00-07:00' -> (22, 7). Wrap-around across midnight is handled at use site."""
    if not raw:
        return None, None
    m = re.match(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", raw)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(3))


def in_quiet_window(hour: Optional[int], start: Optional[int], end: Optional[int]) -> Optional[bool]:
    if hour is None or start is None or end is None:
        return None
    if start == end:
        return False
    if start < end:                     # e.g. 00:00-07:00
        return start <= hour < end
    return hour >= start or hour < end  # wraps midnight, e.g. 22:00-07:00


def _norm_brand(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def domain_assessment(brand: Optional[str], official: Optional[str],
                      used: Optional[str]) -> tuple[Optional[bool], Optional[bool], Optional[str]]:
    """Compare the domain the sender actually used against the brand's official one.

    Returns (matches_official, is_lookalike, note). A verified account can still
    send from a non-official domain, so this is computed independently of the
    `verified` flag.
    """
    if used is None:
        return None, None, None
    if official is None:
        return None, None, "no_official_domain_on_record"
    if official.lower() == used.lower():
        return True, False, "domain_matches_official"

    brand_tok = _norm_brand(brand)
    used_norm = _norm_brand(used)
    official_root = _norm_brand(official.split(".")[0])
    lookalike = bool(brand_tok and brand_tok in used_norm) or bool(
        official_root and official_root in used_norm)
    note = ("lookalike_domain_embeds_brand_name" if lookalike
            else "domain_differs_from_official")
    return False, lookalike, note


# ------------------------------------------------------------------- media probe
def _probe_media(path: Path) -> dict[str, Any]:
    """True container + duration. The `.jpg`/`.mp3` extensions in this dataset are
    unreliable: 10/20 images and 9/13 audio files are a different format entirely."""
    magic = subprocess.run(["file", "-b", str(path)], capture_output=True,
                           text=True).stdout.strip()
    out: dict[str, Any] = {"magic": magic, "true_format": None, "duration_s": None}
    try:
        if magic.startswith("RIFF"):
            with contextlib.closing(wave.open(str(path), "rb")) as wf:
                out["true_format"] = "WAV/PCM"
                out["duration_s"] = round(wf.getnframes() / wf.getframerate(), 2)
        elif path.parent.name == "audio":
            import mutagen
            a = mutagen.File(str(path))
            out["true_format"] = type(a).__name__
            out["duration_s"] = round(a.info.length, 2)
        else:
            from PIL import Image
            with Image.open(path) as im:
                out["true_format"] = im.format
                out["width"], out["height"] = im.width, im.height
    except Exception as exc:                      # noqa: BLE001 - probe must never crash a run
        out["true_format"] = f"probe_error:{type(exc).__name__}"
    return out


class MediaIndex:
    """media_id -> on-disk facts, cached to cache/media_probe.json."""

    def __init__(self, images: pd.DataFrame, voice: pd.DataFrame):
        self.map: dict[str, str] = {}
        for _, r in images.iterrows():
            self.map[str(r["image_id"])] = str(r["file_path"])
        for _, r in voice.iterrows():
            self.map[str(r["voice_note_id"])] = str(r["file_path"])

        self._cache_file = CONFIG.paths.cache / "media_probe.json"
        self._probes: dict[str, dict] = {}
        if self._cache_file.exists():
            self._probes = json.loads(self._cache_file.read_text())

    def resolve(self, media_id: Optional[str]) -> dict[str, Any]:
        if not media_id:
            return {"path": None, "exists": None, "true_format": None, "duration_s": None}
        rel = self.map.get(media_id)
        if rel is None:
            return {"path": None, "exists": False, "true_format": None, "duration_s": None}
        full = CONFIG.paths.dataset / rel
        if not full.exists():
            return {"path": rel, "exists": False, "true_format": None, "duration_s": None}
        if media_id not in self._probes:
            self._probes[media_id] = _probe_media(full)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(self._probes, indent=1, sort_keys=True))
        p = self._probes[media_id]
        return {"path": rel, "exists": True, "true_format": p.get("true_format"),
                "duration_s": p.get("duration_s")}


# --------------------------------------------------------------------- dataset
class Dataset:
    """Eager loader for the 13 participant-facing CSVs."""

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dir = Path(dataset_dir) if dataset_dir else CONFIG.paths.dataset
        r = lambda n: pd.read_csv(self.dir / n)  # noqa: E731
        self.messages = r("messages.csv")
        self.sample_messages = r("sample_messages.csv")
        self.users = r("users.csv")
        self.groups = r("groups.csv")
        self.group_members = r("group_members.csv")
        self.business_accounts = r("business_accounts.csv")
        self.user_business_history = r("user_business_history.csv")
        self.message_history = r("message_history.csv")
        self.message_events = r("message_events.csv")
        self.images = r("images.csv")
        self.voice_notes = r("voice_notes.csv")
        self.daily_notification_summary = r("daily_notification_summary.csv")

        self.media = MediaIndex(self.images, self.voice_notes)
        self._index()
        self._index_templates()

    def _index(self) -> None:
        self._users = {str(r["user_id"]): r for _, r in self.users.iterrows()}
        self._groups = {str(r["group_id"]): r for _, r in self.groups.iterrows()}
        self._biz = {str(r["business_id"]): r for _, r in self.business_accounts.iterrows()}
        self._gm = {(str(r["group_id"]), str(r["user_id"])): r
                    for _, r in self.group_members.iterrows()}
        self._ubh = {(str(r["user_id"]), str(r["business_id"])): r
                     for _, r in self.user_business_history.iterrows()}
        self._events = {str(r["message_id"]): r for _, r in self.message_events.iterrows()}
        self._baselines = self._compute_baselines()

    # -- daily_notification_summary, aggregated per user over its own window ------
    def _compute_baselines(self) -> dict[str, BaselineLoad]:
        d = self.daily_notification_summary.copy()
        d["notifications_sent"] = pd.to_numeric(d["notifications_sent"], errors="coerce")
        d["notifications_dismissed"] = pd.to_numeric(d["notifications_dismissed"],
                                                     errors="coerce")
        g = d.groupby("user_id").agg(
            days_observed=("date", "nunique"),
            window_start=("date", "min"),
            window_end=("date", "max"),
            mean_daily_sent=("notifications_sent", "mean"),
            median_daily_sent=("notifications_sent", "median"),
            mean_daily_dismissed=("notifications_dismissed", "mean"),
            total_sent=("notifications_sent", "sum"),
            total_dismissed=("notifications_dismissed", "sum"),
        )
        # Percentile of this user's mean daily load against every other user.
        g["percentile_vs_all_users"] = (
            g["mean_daily_sent"].rank(pct=True) * 100).round(1)

        out: dict[str, BaselineLoad] = {}
        for uid, row in g.iterrows():
            out[str(uid)] = BaselineLoad(
                window_start=str(row["window_start"]),
                window_end=str(row["window_end"]),
                days_observed=int(row["days_observed"]),
                mean_daily_sent=round(float(row["mean_daily_sent"]), 3),
                median_daily_sent=round(float(row["median_daily_sent"]), 3),
                mean_daily_dismissed=round(float(row["mean_daily_dismissed"]), 3),
                total_sent=int(row["total_sent"]),
                total_dismissed=int(row["total_dismissed"]),
                dismissal_rate=_ratio(float(row["total_dismissed"]), float(row["total_sent"])),
                percentile_vs_all_users=float(row["percentile_vs_all_users"]),
                has_record=True,
            )
        return out

    # -- cross-user template frequency ---------------------------------------
    def _index_templates(self) -> None:
        """Normalized history text -> (occurrences, distinct recipients).

        Deliberately GLOBAL, across every user. This is the one place the
        pipeline looks outside the recipient's own history, and it feeds a
        feature only - evidence ids remain user-scoped everywhere.
        """
        import re as _re
        import unicodedata as _ud
        punct = _re.compile(r"[^\w\s]", _re.UNICODE)
        ws = _re.compile(r"\s+")

        def norm(t: Any) -> str:
            if not isinstance(t, str):
                return ""
            t = _ud.normalize("NFKC", t).lower()
            return ws.sub(" ", punct.sub(" ", t)).strip()

        self._norm_text = norm
        counts: dict[str, int] = {}
        recips: dict[str, set] = {}
        for _, r in self.message_history.iterrows():
            n = norm(r.get("message_text"))
            if not n:
                continue
            counts[n] = counts.get(n, 0) + 1
            recips.setdefault(n, set()).add(str(r.get("user_id")))
        self._template_counts = counts
        self._template_recipients = recips

    def template_stats(self, text: Any) -> TemplateStats:
        n = self._norm_text(text)
        if not n:
            return TemplateStats(0, 0)
        return TemplateStats(
            template_global_occurrences=self._template_counts.get(n, 0),
            template_distinct_recipients=len(self._template_recipients.get(n, ())),
        )

    def baseline_for(self, user_id: str) -> BaselineLoad:
        b = self._baselines.get(user_id)
        if b is not None:
            return b
        return BaselineLoad(window_start=None, window_end=None, days_observed=0,
                            mean_daily_sent=None, median_daily_sent=None,
                            mean_daily_dismissed=None, total_sent=None,
                            total_dismissed=None, dismissal_rate=None,
                            percentile_vs_all_users=None, has_record=False)


# ------------------------------------------------------------------- assembly
class ContextBuilder:
    def __init__(self, ds: Dataset):
        self.ds = ds
        try:
            from media import load_media_index
            self._media_idx = load_media_index()
        except Exception:  # noqa: BLE001 - media enrichment is optional
            self._media_idx = {"audio": {}, "images": {}}

    def media_record(self, media_type: Optional[str],
                     media_id: Optional[str]) -> Optional[dict]:
        if not media_id:
            return None
        kind = "images" if media_type == "image" else "audio"
        return self._media_idx.get(kind, {}).get(media_id)

    # -- user ------------------------------------------------------------------
    def build_user(self, user_id: str) -> UserContext:
        row = self.ds._users.get(user_id)
        if row is None:
            return UserContext(user_id=user_id, quiet_hours_raw=None, quiet_start_hour=None,
                               quiet_end_hour=None, messages_opened_30d=None,
                               messages_replied_30d=None, notifications_dismissed_30d=None,
                               messages_reported_30d=None, open_rate_proxy=None,
                               reply_rate_proxy=None, report_rate_proxy=None,
                               baseline_daily_load=self.ds.baseline_for(user_id))
        raw = _s(row.get("do_not_disturb_window"))
        qs, qe = parse_quiet_window(raw)
        opened = _i(row.get("messages_opened_30d"))
        replied = _i(row.get("messages_replied_30d"))
        dismissed = _i(row.get("notifications_dismissed_30d"))
        reported = _i(row.get("messages_reported_30d"))
        denom = None if opened is None or dismissed is None else opened + dismissed
        return UserContext(
            user_id=user_id, quiet_hours_raw=raw, quiet_start_hour=qs, quiet_end_hour=qe,
            messages_opened_30d=opened, messages_replied_30d=replied,
            notifications_dismissed_30d=dismissed, messages_reported_30d=reported,
            open_rate_proxy=_ratio(opened, denom),
            reply_rate_proxy=_ratio(replied, opened),
            report_rate_proxy=_ratio(reported, opened),
            baseline_daily_load=self.ds.baseline_for(user_id),
        )

    # -- group -----------------------------------------------------------------
    def build_group(self, group_id: Optional[str], user_id: str,
                    sender_user_id: Optional[str]) -> Optional[GroupContext]:
        if not group_id:
            return None
        g = self.ds._groups.get(group_id)
        recip = self.ds._gm.get((group_id, user_id))
        sender = self.ds._gm.get((group_id, sender_user_id)) if sender_user_id else None
        if g is None and recip is None:
            return GroupContext(group_id=group_id, group_name=None, group_type=None,
                                member_count=None, admin_count=None, created_at=None,
                                messages_30d=None, recipient_role=None,
                                recipient_joined_at=None, recipient_messages_sent_30d=None,
                                recipient_messages_read_30d=None,
                                recipient_replies_sent_30d=None, recipient_dismissals_30d=None,
                                is_muted_by_recipient=None, recipient_read_rate=None,
                                recipient_reply_rate=None, sender_role=None,
                                sender_is_admin=None, sender_messages_sent_30d=None,
                                has_record=False)
        g = g if g is not None else {}
        read = _i(recip.get("messages_read_30d")) if recip is not None else None
        replies = _i(recip.get("replies_sent_30d")) if recip is not None else None
        grp_msgs = _i(g.get("messages_30d")) if len(g) else None
        srole = _s(sender.get("role")) if sender is not None else None
        return GroupContext(
            group_id=group_id,
            group_name=_s(g.get("group_name")) if len(g) else None,
            group_type=_s(g.get("group_type")) if len(g) else None,
            member_count=_i(g.get("member_count")) if len(g) else None,
            admin_count=_i(g.get("admin_count")) if len(g) else None,
            created_at=_s(g.get("created_at")) if len(g) else None,
            messages_30d=grp_msgs,
            recipient_role=_s(recip.get("role")) if recip is not None else None,
            recipient_joined_at=_s(recip.get("joined_at")) if recip is not None else None,
            recipient_messages_sent_30d=_i(recip.get("messages_sent_30d")) if recip is not None else None,
            recipient_messages_read_30d=read,
            recipient_replies_sent_30d=replies,
            recipient_dismissals_30d=_i(recip.get("notifications_dismissed_30d")) if recip is not None else None,
            is_muted_by_recipient=_b(recip.get("group_muted_by_user")) if recip is not None else None,
            recipient_read_rate=_ratio(read, grp_msgs),
            recipient_reply_rate=_ratio(replies, read),
            sender_role=srole,
            sender_is_admin=(srole == "admin") if srole is not None else None,
            sender_messages_sent_30d=_i(sender.get("messages_sent_30d")) if sender is not None else None,
            has_record=True,
        )

    # -- business --------------------------------------------------------------
    def build_business(self, business_id: Optional[str]) -> Optional[BusinessContext]:
        if not business_id:
            return None
        b = self.ds._biz.get(business_id)
        if b is None:
            return BusinessContext(business_id=business_id, display_name=None, brand_name=None,
                                   category=None, verified=None, official_domain=None,
                                   domain_used_by_sender=None, account_age_days=None,
                                   messages_sent_30d=None, user_reports_30d=None,
                                   domain_used_by_sender_age_days=None,
                                   domain_matches_official=None, domain_is_lookalike=None,
                                   domain_risk_note=None, is_young_account=None,
                                   is_young_domain=None, report_rate_per_1k_sent=None,
                                   has_record=False)
        brand = _s(b.get("brand_name"))
        official = _s(b.get("official_domain"))
        used = _s(b.get("domain_used_by_sender"))
        matches, lookalike, note = domain_assessment(brand, official, used)
        age = _i(b.get("account_age_days"))
        dage = _i(b.get("domain_used_by_sender_age_days"))
        sent = _i(b.get("messages_sent_30d"))
        reports = _i(b.get("user_reports_30d"))
        return BusinessContext(
            business_id=business_id,
            display_name=_s(b.get("display_name")), brand_name=brand,
            category=_s(b.get("category")), verified=_b(b.get("verified")),
            official_domain=official, domain_used_by_sender=used,
            account_age_days=age, messages_sent_30d=sent, user_reports_30d=reports,
            domain_used_by_sender_age_days=dage,
            domain_matches_official=matches, domain_is_lookalike=lookalike,
            domain_risk_note=note,
            is_young_account=None if age is None else age < 90,
            is_young_domain=None if dage is None else dage < 90,
            report_rate_per_1k_sent=None if (reports is None or not sent) else round(
                1000 * reports / sent, 3),
            has_record=True,
        )

    # -- relationship ----------------------------------------------------------
    def build_relationship(self, user_id: str,
                           business_id: Optional[str]) -> RelationshipContext:
        if not business_id:
            return RelationshipContext(has_prior_relationship=False,
                                       relationship_status="not_a_business_message")
        row = self.ds._ubh.get((user_id, business_id))
        if row is None:
            # 36.7% of business messages land here. Explicit absence - never zeros.
            return RelationshipContext.absent()

        opened = _i(row.get("messages_opened_30d"))
        dismissed = _i(row.get("messages_dismissed_30d"))
        opted_out_at = _s(row.get("promotions_opted_out_at"))
        allows = _b(row.get("allows_promotions"))
        denom = None if opened is None or dismissed is None else opened + dismissed
        if opted_out_at:
            status = "opted_out"
        elif allows:
            status = "active_opted_in"
        else:
            status = "known_no_promo_consent"
        return RelationshipContext(
            has_prior_relationship=True, relationship_status=status,
            why_user_knows_account=_s(row.get("why_user_knows_account")),
            last_activity_at=_s(row.get("last_activity_at")),
            allows_promotions=allows, promotions_opted_out_at=opted_out_at,
            has_opted_out=bool(opted_out_at),
            activity_count_180d=_i(row.get("activity_count_180d")),
            messages_opened_30d=opened, messages_dismissed_30d=dismissed,
            messages_replied_30d=_i(row.get("messages_replied_30d")),
            last_reply_at=_s(row.get("last_reply_at")),
            engagement_ratio=_ratio(opened, denom),
        )

    # -- temporal --------------------------------------------------------------
    def build_temporal(self, created_at: str, user: UserContext) -> TemporalContext:
        dt = pd.to_datetime(created_at, errors="coerce")
        if pd.isna(dt):
            return TemporalContext(created_at_raw=created_at, hour=None, minute=None,
                                   day_of_week=None, is_weekend=None,
                                   in_quiet_hours=None, date=None)
        return TemporalContext(
            created_at_raw=created_at, hour=int(dt.hour), minute=int(dt.minute),
            day_of_week=_DAYS[int(dt.dayofweek)], is_weekend=int(dt.dayofweek) >= 5,
            in_quiet_hours=in_quiet_window(int(dt.hour), user.quiet_start_hour,
                                           user.quiet_end_hour),
            date=dt.strftime("%Y-%m-%d"),
        )

    # -- message ---------------------------------------------------------------
    def build_message(self, row: pd.Series,
                      media_rec: Optional[dict] = None) -> MessageContext:
        text = _s(row.get("message_text")) or ""
        media_id = _s(row.get("media_id"))
        m = self.ds.media.resolve(media_id)
        # Scan every untrusted surface this message carries: its own text plus
        # any OCR / transcript recovered from its attachment.
        surfaces = [text]
        if media_rec:
            surfaces += [media_rec.get("ocr_text"), media_rec.get("transcript_raw")]
        has_inj, inj_text = injection.scan_many(*surfaces)
        return MessageContext(
            message_id=str(row["message_id"]),
            conversation_type=_s(row.get("conversation_type")) or "unknown",
            sender_user_id=_s(row.get("sender_user_id")),
            text=text, text_is_empty=not text.strip(),
            char_len=len(text), word_len=len(text.split()),
            forwarded_count=_i(row.get("forwarded_count")) or 0,
            media_type=_s(row.get("media_type")), media_id=media_id,
            media_path=m["path"], media_exists=m["exists"],
            media_true_format=m["true_format"], media_duration_s=m["duration_s"],
            has_embedded_instruction=has_inj, embedded_instruction_text=inj_text,
        )

    # -- temporal validity -----------------------------------------------------
    def build_temporal_validity(self, row: pd.Series,
                                media_rec: Optional[dict]) -> TemporalValidityContext:
        created = pd.to_datetime(row.get("created_at"), errors="coerce")
        created = None if pd.isna(created) else created.to_pydatetime()
        media_deadline = (media_rec or {}).get("deadline_or_time")
        text = _s(row.get("message_text")) or ""
        # A voice note has no text of its own; its transcript is the text.
        if media_rec and not text.strip():
            text = media_rec.get("transcript_raw") or ""
        tv = build_temporal_validity(text, media_deadline, created)
        return TemporalValidityContext(
            parsed_deadline=tv.parsed_deadline, deadline_source=tv.deadline_source,
            validity=tv.validity, days_until_deadline=tv.days_until_deadline,
            raw_expression=tv.raw_expression,
        )

    # -- pack ------------------------------------------------------------------
    def build(self, row: pd.Series) -> ContextPack:
        user_id = str(row["user_id"])
        media_rec = self.media_record(_s(row.get("media_type")), _s(row.get("media_id")))
        msg = self.build_message(row, media_rec)
        user = self.build_user(user_id)
        return ContextPack(
            message_id=msg.message_id, user_id=user_id, message=msg, user=user,
            temporal=self.build_temporal(str(row["created_at"]), user),
            group=self.build_group(_s(row.get("group_id")), user_id, msg.sender_user_id),
            business=self.build_business(_s(row.get("business_id"))),
            relationship=self.build_relationship(user_id, _s(row.get("business_id"))),
            temporal_validity=self.build_temporal_validity(row, media_rec),
            template=self.ds.template_stats(row.get("message_text")),
            media=media_rec,
        )

    def build_all(self, frame: Optional[pd.DataFrame] = None) -> list[ContextPack]:
        df = self.ds.messages if frame is None else frame
        return [self.build(r) for _, r in df.iterrows()]


def load_context_packs(dataset_dir: Optional[Path] = None) -> tuple[Dataset, list[ContextPack]]:
    CONFIG.paths.ensure()
    ds = Dataset(dataset_dir)
    return ds, ContextBuilder(ds).build_all()
