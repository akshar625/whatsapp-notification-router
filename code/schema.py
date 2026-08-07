"""Typed containers for the routing pipeline.

Design rule that runs through this whole module: **absence is represented, never
imputed**. Only 63.3% of business messages have a `user_business_history` row, so
a missing relationship is modelled as an explicit `none_on_record` state rather
than as zeros, which would read downstream as "known but disengaged".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# --------------------------------------------------------------------------- user
@dataclass
class UserContext:
    user_id: str
    quiet_hours_raw: Optional[str]              # e.g. "22:00-07:00", verbatim from users.csv
    quiet_start_hour: Optional[int]
    quiet_end_hour: Optional[int]
    messages_opened_30d: Optional[int]
    messages_replied_30d: Optional[int]
    notifications_dismissed_30d: Optional[int]
    messages_reported_30d: Optional[int]
    open_rate_proxy: Optional[float]            # opened / (opened + dismissed)
    reply_rate_proxy: Optional[float]           # replied / opened
    report_rate_proxy: Optional[float]          # reported / opened

    # Aggregated over the full daily_notification_summary window (2026-07-04..07-17).
    # Deliberately NOT date-joined to messages.csv - the ranges do not overlap.
    baseline_daily_load: "BaselineLoad" = None


@dataclass
class BaselineLoad:
    """Per-user notification load, aggregated over the summary's own date window."""
    window_start: Optional[str]
    window_end: Optional[str]
    days_observed: int
    mean_daily_sent: Optional[float]
    median_daily_sent: Optional[float]
    mean_daily_dismissed: Optional[float]
    total_sent: Optional[int]
    total_dismissed: Optional[int]
    dismissal_rate: Optional[float]
    # Where this user sits against every other user in the table, 0-100.
    percentile_vs_all_users: Optional[float]
    has_record: bool = True


# -------------------------------------------------------------------------- group
@dataclass
class GroupContext:
    group_id: Optional[str]
    group_name: Optional[str]
    group_type: Optional[str]                   # family / society / coworker / marketplace / ...
    member_count: Optional[int]
    admin_count: Optional[int]
    created_at: Optional[str]
    messages_30d: Optional[int]

    # Recipient's own standing in this group.
    recipient_role: Optional[str]
    recipient_joined_at: Optional[str]
    recipient_messages_sent_30d: Optional[int]
    recipient_messages_read_30d: Optional[int]
    recipient_replies_sent_30d: Optional[int]
    recipient_dismissals_30d: Optional[int]
    is_muted_by_recipient: Optional[bool]
    recipient_read_rate: Optional[float]        # read / group messages_30d
    recipient_reply_rate: Optional[float]       # replies / read

    # Sender's standing, joined on (group_id, sender_user_id).
    sender_role: Optional[str]
    sender_is_admin: Optional[bool]
    sender_messages_sent_30d: Optional[int]

    has_record: bool = True


# ----------------------------------------------------------------------- business
@dataclass
class BusinessContext:
    business_id: Optional[str]
    display_name: Optional[str]
    brand_name: Optional[str]
    category: Optional[str]
    verified: Optional[bool]
    official_domain: Optional[str]
    domain_used_by_sender: Optional[str]
    account_age_days: Optional[int]
    messages_sent_30d: Optional[int]
    user_reports_30d: Optional[int]
    domain_used_by_sender_age_days: Optional[int]

    # Derived trust signals.
    domain_matches_official: Optional[bool]
    domain_is_lookalike: Optional[bool]         # brand token present in a non-official domain
    domain_risk_note: Optional[str]
    is_young_account: Optional[bool]            # account_age_days < 90
    is_young_domain: Optional[bool]             # sender-domain age < 90 days
    report_rate_per_1k_sent: Optional[float]

    has_record: bool = True


# -------------------------------------------------------------------- relationship
@dataclass
class RelationshipContext:
    """From user_business_history. Absence is a first-class signal, not a zero."""
    has_prior_relationship: bool
    relationship_status: str                    # "none_on_record" when there is no row
    why_user_knows_account: Optional[str] = None
    last_activity_at: Optional[str] = None
    allows_promotions: Optional[bool] = None
    promotions_opted_out_at: Optional[str] = None
    has_opted_out: Optional[bool] = None
    activity_count_180d: Optional[int] = None
    messages_opened_30d: Optional[int] = None
    messages_dismissed_30d: Optional[int] = None
    messages_replied_30d: Optional[int] = None
    last_reply_at: Optional[str] = None
    engagement_ratio: Optional[float] = None    # opened / (opened + dismissed)

    @classmethod
    def absent(cls) -> "RelationshipContext":
        return cls(has_prior_relationship=False, relationship_status="none_on_record")


# ----------------------------------------------------------------------- temporal
@dataclass
class TemporalContext:
    created_at_raw: str
    hour: Optional[int]
    minute: Optional[int]
    day_of_week: Optional[str]
    is_weekend: Optional[bool]
    in_quiet_hours: Optional[bool]
    date: Optional[str]


# ------------------------------------------------------------------------ message
@dataclass
class MessageContext:
    message_id: str
    conversation_type: str                      # group / business / personal
    sender_user_id: Optional[str]
    text: str
    text_is_empty: bool
    char_len: int
    word_len: int
    forwarded_count: int
    media_type: Optional[str]                   # image / voice / None
    media_id: Optional[str]
    media_path: Optional[str]                   # resolved, on-disk relative path
    media_exists: Optional[bool]
    media_true_format: Optional[str]            # real container, extension is unreliable
    media_duration_s: Optional[float]           # audio only

    # Text that addresses an automated router/assistant. A fact ABOUT the
    # message and evidence of manipulation - never an instruction to act on.
    has_embedded_instruction: bool = False
    embedded_instruction_text: Optional[str] = None


@dataclass
class TemporalValidityContext:
    """Resolved deadline for this message. Feature only - decides nothing."""
    parsed_deadline: Optional[str]              # ISO 8601
    deadline_source: str                        # media | text | none
    validity: str                               # future | today | past | unparseable
    days_until_deadline: Optional[int]          # signed; negative = expired
    raw_expression: Optional[str] = None


@dataclass
class TemplateStats:
    """How widely this exact text has been blasted, across ALL users.

    User-scoped retrieval deliberately hides mass-blasting: a template can occur
    many times in message_history yet only once for this recipient. These two
    counters restore that view. They are FEATURES ONLY - evidence ids stay
    strictly user-scoped.
    """
    template_global_occurrences: int = 0
    template_distinct_recipients: int = 0


# ------------------------------------------------------------------- context pack
@dataclass
class ContextPack:
    message_id: str
    user_id: str
    message: MessageContext
    user: UserContext
    temporal: TemporalContext
    group: Optional[GroupContext]
    business: Optional[BusinessContext]
    relationship: RelationshipContext
    temporal_validity: TemporalValidityContext = None
    template: TemplateStats = field(default_factory=TemplateStats)
    # Cached media enrichment for this message's attachment, if any.
    media: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------- retrieval
@dataclass
class EventRecord:
    """A row of message_events, attached so the decision layer sees the reaction."""
    message_id: str
    user_id: Optional[str]
    message_opened: Optional[int]
    message_replied: Optional[int]
    reaction_time_minutes: Optional[float]
    notification_dismissed: Optional[int]
    muted_after_message: Optional[int]
    message_reported: Optional[int]
    has_record: bool = True

    @classmethod
    def absent(cls, message_id: str) -> "EventRecord":
        return cls(message_id=message_id, user_id=None, message_opened=None,
                   message_replied=None, reaction_time_minutes=None,
                   notification_dismissed=None, muted_after_message=None,
                   message_reported=None, has_record=False)


@dataclass
class HistoryCandidate:
    history_message_id: str
    scope: str                                  # sender / group / business / user_global
    conversation_type: Optional[str]
    created_at: Optional[str]
    text: str
    media_type: Optional[str]
    forwarded_count: Optional[int]
    lexical_score: float
    dense_score: float
    hybrid_score: float
    jaccard: float
    is_near_duplicate: bool
    events: EventRecord


@dataclass
class NearDuplicateStats:
    """Aggregate reaction across every prior occurrence of near-identical text."""
    occurrences: int
    matched_history_ids: list[str]
    max_jaccard: float
    exact_matches: int
    times_opened: int
    times_replied: int
    times_dismissed: int
    times_muted_after: int
    times_reported: int
    events_available: int
    # Same, but across the user's entire history rather than just the scoped pool.
    occurrences_user_global: int = 0
    matched_history_ids_user_global: list[str] = field(default_factory=list)


@dataclass
class RetrievalBundle:
    message_id: str
    user_id: str
    pool_size: int
    pool_scopes: dict[str, int]
    user_history_size: int
    near_duplicates: NearDuplicateStats
    candidates: list[HistoryCandidate]

    @property
    def evidence_message_ids(self) -> list[str]:
        """Evidence may only ever cite message_history ids."""
        return [c.history_message_id for c in self.candidates]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- decision
@dataclass
class Decision:
    message_id: str
    action: str                                 # notify | digest | mute
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]

    def to_row(self) -> dict[str, str]:
        ev = ";".join(self.evidence_message_ids) if self.evidence_message_ids else "none"
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": ev,
        }
