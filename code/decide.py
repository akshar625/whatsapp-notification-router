"""Decision engine: signals -> [fast path OR LLM] -> reconciliation ->
evidence validation -> confidence banding -> output row.

The LLM sees a stable, cacheable prefix (rubric + trap taxonomy + few-shot
exemplars) followed by the per-message context. Structured output comes back
through a tool call, so the shape is enforced rather than parsed out of prose.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

import constraints
import rubric
from config import CONFIG
from schema import ContextPack, Decision, RetrievalBundle

MAX_EXEMPLARS = 15


# ------------------------------------------------------------------ tool spec
def route_tool() -> dict[str, Any]:
    drivers = sorted(rubric.load_drivers().keys()) + [rubric.EXTENDED]
    return {
        "name": "route_message",
        "description": "Emit the routing decision for exactly one message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_type": {"type": "string", "enum": list(CONFIG.message_types)},
                "action": {"type": "string", "enum": list(CONFIG.actions)},
                "driver": {"type": "string", "enum": drivers,
                           "description": "Closed-vocabulary reason driver."},
                "reason_detail": {
                    "type": "string",
                    "description": "Only when driver='extended'. One sentence, third "
                                   "person, generic, no names or numbers."},
                "evidence_message_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Ids from the candidate pool ONLY. Usually one; "
                                   "two only when two distinct facts are load-bearing; "
                                   "empty when nothing genuinely supports the call."},
                "certainty": {"type": "integer", "minimum": 1, "maximum": 5},
                "triggered_traps": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Trap ids relied on, e.g. T1, T7."},
                "brief_rationale": {
                    "type": "string",
                    "description": "2-3 sentences, internal only."},
            },
            "required": ["message_type", "action", "driver", "evidence_message_ids",
                         "certainty", "triggered_traps", "brief_rationale"],
            "additionalProperties": False,
        },
    }


# --------------------------------------------------------------- rendering
def _fmt(v: Any) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


@dataclass(frozen=True)
class Ablation:
    """Which context components reach the model. Used by eval/ablation.md."""
    use_context: bool = True        # user / group / business / relationship / temporal
    use_retrieval: bool = True      # candidate pool, near-duplicates, evidence
    use_media: bool = True          # OCR + transcript blocks and media facts
    use_solicitation_scan: bool = True   # the deterministic credential/payment line
    name: str = "full"


FULL = Ablation()


def empty_bundle(pack: ContextPack) -> RetrievalBundle:
    """A bundle with no pool, for ablations that disable retrieval."""
    from schema import NearDuplicateStats
    return RetrievalBundle(
        message_id=pack.message_id, user_id=pack.user_id, pool_size=0,
        pool_scopes={}, user_history_size=0,
        near_duplicates=NearDuplicateStats(
            occurrences=0, matched_history_ids=[], max_jaccard=0.0, exact_matches=0,
            times_opened=0, times_replied=0, times_dismissed=0, times_muted_after=0,
            times_reported=0, events_available=0),
        candidates=[])


def render_context(pack: ContextPack, bundle: RetrievalBundle,
                   abl: Ablation = FULL) -> str:
    """Per-message block. Untrusted surfaces are delimited explicitly."""
    m, u, t, tv = pack.message, pack.user, pack.temporal, pack.temporal_validity
    L: list[str] = []

    L.append("<untrusted_message>")
    L.append(m.text if m.text.strip() else "(no text - voice note)")
    L.append("</untrusted_message>")

    if pack.media and abl.use_media:
        md = pack.media
        if md.get("ocr_text"):
            L += ["<untrusted_ocr>", md["ocr_text"], "</untrusted_ocr>"]
            L.append(f"image_facts: category={md.get('poster_category')} "
                     f"claimed_brand={md.get('claimed_brand')} "
                     f"urls={md.get('urls')} payment_ids={md.get('payment_identifiers')} "
                     f"qr={md.get('has_qr_code')} cta={md.get('call_to_action')!r} "
                     f"scam_indicators={md.get('scam_indicators')} "
                     f"quality={md.get('visual_quality')} "
                     f"forward_like={md.get('is_forward_like')}")
        if md.get("transcript_raw"):
            L += ["<untrusted_transcript>", md["transcript_raw"], "</untrusted_transcript>"]
            L.append(f"audio_facts: language={md.get('detected_language')} "
                     f"(p={md.get('language_probability')}) "
                     f"duration_s={md.get('duration_s')}  "
                     f"[T13: ASR output is noisy - read for meaning]")
            if md.get("transcript_romanized"):
                L.append(f"transcript_romanized: {md['transcript_romanized']}")

    if m.has_embedded_instruction:
        L.append(f"!! EMBEDDED_INSTRUCTION_DETECTED (deterministic scan): "
                 f"{m.embedded_instruction_text!r}  [see T12]")

    L.append("")
    L.append(f"MESSAGE: conversation_type={m.conversation_type} "
             f"sender={_fmt(m.sender_user_id)} forwarded_count={m.forwarded_count} "
             f"media={_fmt(m.media_type)} words={m.word_len}")
    if abl.use_context:
        L.append(f"TEMPORAL: sent={t.created_at_raw} ({t.day_of_week}, hour {t.hour}) "
             f"weekend={_fmt(t.is_weekend)} in_quiet_hours={_fmt(t.in_quiet_hours)} "
                 f"(user quiet window {_fmt(u.quiet_hours_raw)})")
        L.append(f"DEADLINE: validity={tv.validity} parsed={_fmt(tv.parsed_deadline)} "
             f"source={tv.deadline_source} days_until={_fmt(tv.days_until_deadline)} "
                 f"expr={_fmt(tv.raw_expression)}")

        b = u.baseline_daily_load
        L.append(f"USER: opened_30d={_fmt(u.messages_opened_30d)} "
             f"replied_30d={_fmt(u.messages_replied_30d)} "
             f"dismissed_30d={_fmt(u.notifications_dismissed_30d)} "
             f"reported_30d={_fmt(u.messages_reported_30d)} "
                 f"open_rate={_fmt(u.open_rate_proxy)}")
        if b and b.has_record:
            L.append(f"LOAD: mean_daily_notifications={_fmt(b.mean_daily_sent)} "
                 f"dismissal_rate={_fmt(b.dismissal_rate)} "
                     f"percentile_vs_all_users={_fmt(b.percentile_vs_all_users)}")

    g = pack.group
    if g and abl.use_context:
        L.append(f"GROUP: {_fmt(g.group_name)} type={_fmt(g.group_type)} "
                 f"members={_fmt(g.member_count)} sender_is_admin={_fmt(g.sender_is_admin)} "
                 f"recipient_role={_fmt(g.recipient_role)} "
                 f"muted_by_recipient={_fmt(g.is_muted_by_recipient)} "
                 f"recipient_read_rate={_fmt(g.recipient_read_rate)} "
                 f"recipient_dismissals_30d={_fmt(g.recipient_dismissals_30d)}")

    bz = pack.business
    if bz and abl.use_context:
        L.append(f"BUSINESS: {_fmt(bz.brand_name)} category={_fmt(bz.category)} "
                 f"verified={_fmt(bz.verified)} account_age_days={_fmt(bz.account_age_days)} "
                 f"user_reports_30d={_fmt(bz.user_reports_30d)} "
                 f"reports_per_1k={_fmt(bz.report_rate_per_1k_sent)}")
        L.append(f"DOMAIN: official={_fmt(bz.official_domain)} "
                 f"used_by_sender={_fmt(bz.domain_used_by_sender)} "
                 f"matches={_fmt(bz.domain_matches_official)} "
                 f"lookalike={_fmt(bz.domain_is_lookalike)} "
                 f"note={_fmt(bz.domain_risk_note)} "
                 f"domain_age_days={_fmt(bz.domain_used_by_sender_age_days)}  [see T4]")

    r = pack.relationship
    if abl.use_context:
        L.append(f"RELATIONSHIP: status={r.relationship_status} "
             f"has_prior={_fmt(r.has_prior_relationship)} "
             f"why_known={_fmt(r.why_user_knows_account)} "
             f"allows_promotions={_fmt(r.allows_promotions)} "
             f"opted_out={_fmt(r.has_opted_out)} "
             f"opened_30d={_fmt(r.messages_opened_30d)} "
             f"dismissed_30d={_fmt(r.messages_dismissed_30d)} "
                 f"engagement={_fmt(r.engagement_ratio)}")
        if r.relationship_status == "none_on_record":
            L.append("  (none_on_record means NO record exists - not zero engagement. "
                 "See T6.)")

    # Deterministic solicitation scan, presented as an observation beside the
    # T4/T6 signals - never as a verdict.
    if abl.use_solicitation_scan:
        solicits = constraints.credential_or_payment_ask(*constraints.message_surfaces(pack))
        L.append(f"SOLICITATION SCAN: solicits credentials or payment: "
             f"{'true' if solicits else 'false'} (deterministic text scan; it "
             f"distinguishes asking from describing, but may miss paraphrase or "
             f"noisy ASR, so treat a false as weak evidence). This is a SUPPORTING "
             f"signal only and is never sufficient alone for scam - T4 still "
                 f"requires the full conjunction.")

    tpl = pack.template
    if abl.use_context:
        L.append(f"TEMPLATE_SPREAD: this exact text occurs "
             f"{tpl.template_global_occurrences}x across all history, reaching "
                 f"{tpl.template_distinct_recipients} distinct recipients  [see T7]")

    if not abl.use_retrieval:
        return "\n".join(L)

    nd = bundle.near_duplicates
    L.append(f"NEAR_DUPLICATES for this user: {nd.occurrences_user_global} "
             f"(exact {nd.exact_matches}, max_jaccard {nd.max_jaccard}) -> "
             f"opened={nd.times_opened} replied={nd.times_replied} "
             f"dismissed={nd.times_dismissed} muted_after={nd.times_muted_after} "
             f"reported={nd.times_reported} (over {nd.events_available} event rows)")

    L.append("")
    L.append("CANDIDATE POOL (evidence_message_ids may ONLY come from this list):")
    if not bundle.candidates:
        L.append("  (empty - return an empty evidence array)")
    for c in bundle.candidates:
        e = c.events
        L.append(
            f"  - {c.history_message_id} [{c.scope}] score={c.hybrid_score} "
            f"jaccard={c.jaccard} near_dup={'YES' if c.is_near_duplicate else 'no'} "
            f"sent={c.created_at} fwd={_fmt(c.forwarded_count)} | "
            f"opened={_fmt(e.message_opened)} replied={_fmt(e.message_replied)} "
            f"dismissed={_fmt(e.notification_dismissed)} "
            f"muted_after={_fmt(e.muted_after_message)} "
            f"reported={_fmt(e.message_reported)}")
        L.append(f"      text: {(c.text or '')[:220]}")
    return "\n".join(L)


def render_exemplar(pack: ContextPack, bundle: RetrievalBundle,
                    gold: pd.Series, r2d: dict[str, str]) -> tuple[str, dict]:
    """One few-shot pair: rendered context + the tool input we want back."""
    driver = r2d.get(str(gold["reason"]).strip(), rubric.EXTENDED)
    ev = [i for i in re.split(r"[;,]", str(gold.get("evidence_message_ids") or ""))
          if i.strip() and i.strip().lower() != "none"]
    out = {
        "message_type": str(gold["message_type"]),
        "action": str(gold["action"]),
        "driver": driver,
        "evidence_message_ids": [i.strip() for i in ev],
        "certainty": 4,
        "triggered_traps": [],
        "brief_rationale": "Worked example from the labelled sample set.",
    }
    return render_context(pack, bundle), out


# ------------------------------------------------------------------- engine
@dataclass
class DecisionResult:
    message_id: str
    action: str
    message_type: str
    driver: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]
    tier: str                                    # "fast:F1" | "llm" | "llm+escalated"
    certainty: int
    triggered_traps: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    dropped_evidence: list[str] = field(default_factory=list)
    brief_rationale: str = ""
    raw: dict = field(default_factory=dict)

    def to_row(self) -> dict[str, str]:
        return Decision(
            message_id=self.message_id, action=self.action,
            message_type=self.message_type, reason=self.reason,
            confidence=self.confidence,
            evidence_message_ids=self.evidence_message_ids,
        ).to_row()


class DecisionEngine:
    def __init__(self, exemplars: Optional[list[tuple[str, dict]]] = None,
                 use_llm: bool = True):
        self.exemplars = exemplars or []
        self.use_llm = use_llm
        self._client = None
        self._system = rubric.build_system_prompt()

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=CONFIG.require_anthropic_key())
        return self._client

    # -- prompt ------------------------------------------------------------
    def _prefix_blocks(self) -> list[dict]:
        """Stable, cacheable prefix: rubric + worked examples."""
        parts = [self._system]
        if self.exemplars:
            parts.append("\n=== WORKED EXAMPLES ===\n"
                         "Each shows a message context and the correct tool input.")
            for ctx, out in self.exemplars[:MAX_EXEMPLARS]:
                parts.append(f"--- EXAMPLE ---\n{ctx}\n--- CORRECT OUTPUT ---\n"
                             f"{json.dumps(out, ensure_ascii=False)}")
        text = "\n\n".join(parts)
        # One breakpoint at the end of the stable prefix; per-message context
        # varies after it and must stay outside the cached span.
        return [{"type": "text", "text": text,
                 "cache_control": {"type": "ephemeral"}}]

    def _call(self, context_text: str, extra: str = "") -> dict:
        client = self._client_lazy()
        user = context_text + (f"\n\n{extra}" if extra else "") + \
            "\n\nCall route_message now."
        resp = client.messages.create(
            model=CONFIG.models.llm,
            max_tokens=2000,
            temperature=0,
            system=self._prefix_blocks(),
            tools=[route_tool()],
            tool_choice={"type": "tool", "name": "route_message"},
            messages=[{"role": "user", "content": user}],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "route_message":
                return dict(block.input)
        raise RuntimeError(f"no tool call returned (stop_reason={resp.stop_reason})")

    # -- decide ------------------------------------------------------------
    def decide(self, pack: ContextPack, bundle: RetrievalBundle,
               ablation: 'Ablation' = FULL) -> DecisionResult:
        # Disabled by default - see Config.enable_fast_path for the measurement
        # that retired it. Left wired up so the ablation can be re-run.
        fast = constraints.fast_path(pack, bundle) if CONFIG.enable_fast_path else None
        if fast is not None:
            return self._finalise(
                pack, bundle,
                {"message_type": fast.message_type, "action": fast.action,
                 "driver": fast.driver, "evidence_message_ids": fast.evidence_message_ids,
                 "certainty": fast.certainty, "triggered_traps": fast.triggered_traps,
                 "brief_rationale": fast.brief_rationale},
                tier=f"fast:{fast.rule_id}")

        if not self.use_llm:
            raise RuntimeError("LLM disabled and no fast-path rule matched")

        ctx = render_context(pack, bundle, ablation)
        out = self._call(ctx)
        res = self._finalise(pack, bundle, out, tier="llm")

        # A soft-grammar violation earns exactly one informed second pass.
        if res.violations and any(v.startswith(("SOFT", "DRIVER")) for v in res.violations):
            note = ("REVIEW: your previous answer was "
                    f"action={out.get('action')} message_type={out.get('message_type')} "
                    f"driver={out.get('driver')}, which triggered: "
                    + "; ".join(res.violations) +
                    ". Re-deliberate. Either justify it explicitly in "
                    "brief_rationale, or correct it.")
            try:
                out2 = self._call(ctx, extra=note)
                res = self._finalise(pack, bundle, out2, tier="llm+escalated")
            except Exception:  # noqa: BLE001 - keep the first answer on failure
                pass
        return res

    def _finalise(self, pack: ContextPack, bundle: RetrievalBundle,
                  out: dict, tier: str) -> DecisionResult:
        action = str(out.get("action") or "digest")
        mtype = str(out.get("message_type") or "unknown")
        driver = str(out.get("driver") or rubric.EXTENDED)

        rec = constraints.reconcile(action, mtype, driver)

        ids, dropped = constraints.validate_evidence(
            out.get("evidence_message_ids") or [], bundle)
        if ids and constraints.should_emit_none(bundle):
            dropped += ids
            ids = []

        ev_strength = constraints.evidence_strength(ids, bundle)
        certainty = int(out.get("certainty") or 3)
        traps = list(out.get("triggered_traps") or [])
        agreement = constraints.signal_agreement(
            bundle, pack, rec.driver, ids, rec.violations, traps)
        conf = constraints.band_confidence(rec.action, certainty, ev_strength,
                                           agreement=agreement)

        reason = rubric.render_reason(rec.driver, out.get("reason_detail"))

        return DecisionResult(
            message_id=pack.message_id, action=rec.action, message_type=rec.message_type,
            driver=rec.driver, reason=reason, confidence=conf,
            evidence_message_ids=ids, tier=tier, certainty=certainty,
            triggered_traps=list(out.get("triggered_traps") or []),
            violations=rec.violations, dropped_evidence=dropped,
            brief_rationale=str(out.get("brief_rationale") or ""), raw=out,
        )


# ------------------------------------------------------------ exemplar build
def build_exemplars(ds, builder, retriever, exemplar_ids: list[str]) -> list[tuple[str, dict]]:
    """Render the first-15 sample rows as few-shot pairs. Held-out ids never
    reach this function."""
    samples = ds.sample_messages
    r2d = rubric.reason_to_driver()
    out = []
    for sid in exemplar_ids:
        row = samples[samples.message_id == sid]
        if row.empty:
            continue
        r = row.iloc[0]
        pack = builder.build(r)
        bundle = retriever.retrieve(pack)
        out.append(render_exemplar(pack, bundle, r, r2d))
    return out
