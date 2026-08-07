"""Deterministic pre/post layer around the LLM.

Pre  : a narrow fast path that skips the LLM entirely on unambiguous cases.
Post : type x action grammar reconciliation, evidence validation, confidence
       banding.

Every rule here is expressed over FEATURES. No message_id appears in any
condition, so each rule generalises to messages the system has not seen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from config import CONFIG
from rubric import CONFIDENCE_BANDS, HARD_TYPE_ACTION, SOFT_FORBIDDEN, load_drivers
from schema import ContextPack, RetrievalBundle

# ------------------------------------------------------------------ features
# Detecting a credential/payment SOLICITATION.
#
# Three defects fixed here after an audit (see README -> Corpus observations):
#   1. Every alternation now has a trailing \b. Previously `fee` matched inside
#      "feedback", so a cinema feedback request scored as a payment ask.
#   2. A bare noun is no longer enough. "Your payment was received" is a receipt,
#      not an ask; "No payment or OTP is required" is an explicit denial. An ask
#      now needs a solicitation VERB bound to a sensitive OBJECT.
#   3. The mandatory article is gone, so "verify wallet and card details" matches
#      where `verify (your|the) wallet` did not.
#
# Rule 2 is the same describing-vs-soliciting distinction T12b enforces at the
# LLM layer; keeping the two layers consistent matters because a false positive
# pushes a legitimate business message toward scam - a mute where notify or
# digest was right, which is the expensive direction.

_SENSITIVE = (r"otp\b|one[\s-]?time\s+password\b|verification\s+code\b|"
              r"login\s+code\b|security\s+code\b|access\s+code\b|\bpin\b|"
              r"password\b|cvv\b|card\s+(number|details?)\b|"
              r"account\s+(number|details?)\b|bank\s+details?\b|"
              r"wallet\s+(pin|details?)\b|kyc\b|"
              r"\b6\s*digit\s+(login\s+)?code\b")
_MONEY = (r"fee\b|fees\b|payment\b|amount\b|charges?\b|deposit\b|"
          r"token\s+amount\b|processing\s+fee\b|clearance\s+amount\b|"
          r"reattempt\b|upi\b|qr\b")
_ASK_VERB = (r"share|send|provide|give|enter|submit|fill|tell|reply\s+with|"
             r"forward|confirm|verify|complete|re[\s-]?verify|batao|bhejo|daal|"
             r"do\s+not\s+delay")
_PAY_VERB = (r"pay|transfer|deposit|remit|scan\s+(and\s+pay|the\s+qr)|"
             r"complete\s+the\s+payment|clear\s+the")

_ASK_PATTERNS = tuple(re.compile(p, re.I) for p in (
    # verb ... sensitive object   ("share the OTP", "confirm your card details")
    rf"\b({_ASK_VERB})\b[^.\n]{{0,45}}?({_SENSITIVE})",
    # sensitive object ... verb   ("OTP abhi batao", "card details send karo")
    rf"({_SENSITIVE})[^.\n]{{0,35}}?\b({_ASK_VERB})\b",
    # payment verb ... money/link ("pay the processing fee", "scan and pay")
    rf"\b({_PAY_VERB})\b[^.\n]{{0,45}}?({_MONEY}|link\b|now\b|today\b)",
    # money ... payment verb      ("processing fee pay today")
    rf"({_MONEY})[^.\n]{{0,35}}?\b({_PAY_VERB})\b",
    # prize/benefit claim flows
    r"\bclaim\b[^.\n]{0,30}\b(reward|benefit|refund|voucher|prize|amount)\b",
    r"\brelease\s+the\s+amount\b",
))

# An advisory or denial ABOUT credentials is not a request FOR them.
_NEGATION = re.compile(
    r"\b(no|never|not|non?e|do\s*n[o']?t|does\s*n[o']?t|will\s+never|"
    r"nahi|kabhi\s+nahi)\b[^.\n]{0,45}?"
    rf"({_SENSITIVE}|{_MONEY}|ask\b|share\b|required\b)"
    r"|\b(never|do\s*n[o']?t|don't)\s+(ask|share|disclose|reveal)\b"
    r"|\bno\s+(payment|otp|fee|charge)[^.\n]{0,25}\b(required|needed|asked)\b"
    r"|\bis\s+not\s+required\b", re.I)

_PRESSURE = re.compile(
    r"\b(immediately|urgently|right\s+now|within\s+\d+\s*(min|hour)|"
    r"before\s+(midnight|tonight|it\s+expires)|will\s+be\s+(blocked|locked|"
    r"restricted|suspended|closed)|final\s+reminder|last\s+chance|"
    r"account\s+block|jaldi|turant|warna)\b", re.I)

_CLAUSE = re.compile(r"[.!?\n]+")


def credential_or_payment_ask(*texts: Optional[str]) -> bool:
    """True only when the text SOLICITS a credential or a payment.

    Evaluated clause by clause so a denial in one sentence does not suppress a
    genuine ask in another, and a genuine-looking noun in a denial clause does
    not count as an ask.
    """
    for t in texts:
        if not t:
            continue
        for clause in _CLAUSE.split(t):
            if not clause.strip():
                continue
            if _NEGATION.search(clause):
                continue                      # describing, not soliciting
            if any(rx.search(clause) for rx in _ASK_PATTERNS):
                return True
    return False


def urgency_pressure(*texts: Optional[str]) -> bool:
    return any(t and _PRESSURE.search(t) for t in texts)


def message_surfaces(pack: ContextPack) -> list[str]:
    """Every untrusted text surface this message carries."""
    out = [pack.message.text or ""]
    if pack.media:
        out.append(pack.media.get("ocr_text") or "")
        out.append(pack.media.get("transcript_raw") or "")
    return [s for s in out if s]


# ----------------------------------------------------------------- fast path
@dataclass
class FastDecision:
    rule_id: str
    action: str
    message_type: str
    driver: str
    evidence_message_ids: list[str]
    certainty: int
    triggered_traps: list[str]
    brief_rationale: str


# Marketing shape: an unsubscribe affordance, a discount, or offer language.
_PROMOTIONAL = re.compile(
    r"\b(unsubscribe|reply\s+stop|opt\s*out|\d{1,3}\s*%\s*off|discount|offer|"
    r"deal|sale|coupon|promo|cashback|limited\s+time|shop\s+now|book\s+now|"
    r"t&c\s+apply)\b", re.I)


def _promotional_shape(pack: ContextPack) -> bool:
    """Feature test - is this message marketing-shaped?"""
    if pack.media and pack.media.get("poster_category") == "promotion_sale":
        return True
    return any(_PROMOTIONAL.search(s) for s in message_surfaces(pack))


def fast_path(pack: ContextPack, bundle: RetrievalBundle) -> Optional[FastDecision]:
    """Deliberately narrow. Better to fire rarely and correctly than often."""
    # A deterministic rule must never pre-empt a MORE specific classification.
    # A message carrying text that addresses an automated router is first and
    # foremost an injection attempt, and T12 has a dedicated driver saying so.
    # Letting F2 answer "first contact asking for something sensitive" instead
    # is true but strictly less informative, so injection rows always go to the
    # LLM. Same principle that keeps F1 from guessing a message_type.
    if pack.message.has_embedded_instruction:
        return None

    nd = bundle.near_duplicates

    # F1 - exact near-duplicate, seen at least twice before, and the user
    # dismissed or muted EVERY prior occurrence without ever opening one.
    #
    # message_type is NOT assumed. A repeatedly-dismissed template could equally
    # be a greeting chain, a forward, or a business_update, and a confident wrong
    # type is worse than deferring. F1 therefore only fires where the type is
    # unambiguous from features - a business conversation carrying marketing
    # shape - and hands everything else to the LLM.
    if (nd.exact_matches >= 1 and nd.max_jaccard >= 1.0
            and nd.occurrences_user_global >= 2
            and nd.events_available >= 2
            and nd.times_opened == 0
            and (nd.times_dismissed + nd.times_muted_after) >= nd.events_available
            and pack.message.conversation_type == "business"
            and _promotional_shape(pack)):
        ev = [c.history_message_id for c in bundle.candidates if c.is_near_duplicate][:1]
        return FastDecision(
            rule_id="F1",
            action="mute", message_type="promotion",
            driver="similar_historically_ignored",
            evidence_message_ids=ev, certainty=5,
            triggered_traps=["T7"],
            brief_rationale=(
                f"Exact marketing template seen {nd.occurrences_user_global}x by "
                f"this user in a business conversation; {nd.times_dismissed} "
                f"dismissed, {nd.times_muted_after} muted, 0 opened."),
        )

    # F2 - the full scam conjunction demanded by T4: unverified AND a
    # brand-lookalike domain AND no prior relationship AND a credential/payment ask.
    b, rel = pack.business, pack.relationship
    if (b is not None and b.has_record
            and b.verified is False
            and b.domain_is_lookalike is True
            and rel.relationship_status == "none_on_record"
            and credential_or_payment_ask(*message_surfaces(pack))):
        return FastDecision(
            rule_id="F2",
            action="mute", message_type="scam",
            driver="first_contact_sensitive_ask",
            evidence_message_ids=[], certainty=5,
            triggered_traps=["T4", "T6", "T11"],
            brief_rationale=(
                f"Unverified {b.brand_name!r} sending from lookalike domain "
                f"{b.domain_used_by_sender!r} (official {b.official_domain!r}), "
                f"no prior relationship, and it solicits credentials or payment."),
        )
    return None


# ------------------------------------------------------ grammar reconciliation
@dataclass
class Reconciliation:
    action: str
    message_type: str
    driver: str
    violations: list[str] = field(default_factory=list)
    escalate: bool = False


def reconcile(action: str, message_type: str, driver: str) -> Reconciliation:
    """Enforce HARD grammar; flag SOFT violations for re-deliberation.

    A soft violation is never silently rewritten - it is logged and escalated so
    the model gets a second, informed pass.
    """
    r = Reconciliation(action=action, message_type=message_type, driver=driver)

    required = HARD_TYPE_ACTION.get(message_type)
    if required and action != required:
        r.violations.append(
            f"HARD: message_type={message_type} requires action={required}, got {action}")
        r.action = required

    if message_type in SOFT_FORBIDDEN and r.action in SOFT_FORBIDDEN[message_type]:
        r.violations.append(
            f"SOFT: message_type={message_type} with action={r.action} is unusual")
        r.escalate = True

    # The driver's own action must agree with the final action.
    d = load_drivers().get(r.driver)
    if d is not None and d.action != r.action:
        r.violations.append(
            f"DRIVER: {r.driver} implies action={d.action}, final action={r.action}")
        r.escalate = True
    return r


# -------------------------------------------------------- evidence validation
def validate_evidence(ids: list[str], bundle: RetrievalBundle) -> tuple[list[str], list[str]]:
    """Drop anything the model did not actually see. Never allow fabrication."""
    pool = {c.history_message_id for c in bundle.candidates}
    kept, dropped, seen = [], [], set()
    for i in ids or []:
        i = str(i).strip()
        if not i or i in seen:
            continue
        seen.add(i)
        (kept if i in pool else dropped).append(i)
    return kept[:2], dropped


def evidence_strength(ids: list[str], bundle: RetrievalBundle) -> float:
    """0..1 - how strongly the cited evidence supports anything at all."""
    if not ids:
        return 0.0
    by_id = {c.history_message_id: c for c in bundle.candidates}
    best = 0.0
    for i in ids:
        c = by_id.get(i)
        if c is None:
            continue
        s = 1.0 if c.is_near_duplicate else min(1.0, max(0.0, c.hybrid_score))
        best = max(best, s)
    return best


def should_emit_none(bundle: RetrievalBundle) -> bool:
    if not bundle.candidates:
        return True
    return max(c.hybrid_score for c in bundle.candidates) < CONFIG.thresholds.min_evidence_score


# ------------------------------------------------------------------ confidence
def signal_agreement(bundle: RetrievalBundle, pack: ContextPack, driver: str,
                     ids: list[str], violations: list[str],
                     traps: list[str]) -> float:
    """Deterministic 0..1 estimate of how much the FEATURES agree with each other.

    Self-reported certainty is a generated token, not a posterior, and in
    practice it compresses to 4-5 regardless of how the scale is described in
    the prompt. So the position inside the band is driven mostly by things we
    can measure: whether cited evidence corroborates the driver, whether the
    user's prior reactions to near-identical text were unanimous or mixed,
    whether any constraint fired, and how many competing traps were in play.
    """
    import rubric as _rb
    score = 0.5
    by_id = {c.history_message_id: c for c in bundle.candidates}

    # Evidence: does the cited candidate's reaction record back the driver?
    if ids:
        backed = sum(1 for i in ids
                     if i in by_id and _rb.events_support_driver(driver, by_id[i].events))
        score += 0.18 if backed == len(ids) else (0.04 if backed else -0.10)
    else:
        score -= 0.10                       # decided with nothing to point at

    # Prior reactions to near-identical text: unanimous is strong, mixed is not.
    nd = bundle.near_duplicates
    if nd.events_available >= 2:
        fatigue = nd.times_dismissed + nd.times_muted_after
        if nd.times_opened == 0 and fatigue >= nd.events_available:
            score += 0.18
        elif nd.times_opened >= nd.events_available and fatigue == 0:
            score += 0.12
        else:
            score -= 0.06                   # the history disagrees with itself
    elif nd.occurrences_user_global == 0:
        score -= 0.04                       # no precedent at all

    # A constraint firing means the model's own output was internally inconsistent.
    if violations:
        score -= 0.18

    # Many traps in play means many competing considerations.
    n_traps = len(traps or [])
    score += 0.05 if n_traps <= 1 else (-0.10 if n_traps >= 4 else 0.0)

    # An unknown counterparty is genuinely less decidable.
    if pack.relationship.relationship_status == "none_on_record":
        score -= 0.06

    return max(0.0, min(1.0, score))


def band_confidence(action: str, certainty: int, ev_strength: float,
                    agreement: Optional[float] = None) -> float:
    """Map a 1-5 certainty monotonically onto the action's band.

    We never emit the model's self-reported probability: that is a generated
    token, not a posterior, and it clusters on 0.9/0.95.

    The mapping is RECENTRED so the band is actually used: certainty 1 lands on
    the band floor, 3 on the band MEAN, 5 on the ceiling. Evidence strength then
    perturbs slightly within the band, breaking ties without disturbing the
    monotonic ordering by certainty. The band means (notify 0.88, mute 0.84,
    digest 0.81) sit within ~0.006 of the gold means, so a model that reports
    certainty honestly lands on the gold calibration by construction.
    """
    lo, hi = CONFIDENCE_BANDS.get(action, (0.78, 0.84))
    c = max(1, min(5, int(certainty or 3)))
    model_pos = (c - 1) / 4.0                            # 1->0.0, 3->0.5, 5->1.0
    if agreement is None:
        pos = model_pos + (max(0.0, min(1.0, ev_strength)) - 0.5) * 0.10
    else:
        # Equal weight: the model's own read matters, but so does whether the
        # features actually corroborate it. Neither alone decides the position.
        pos = 0.5 * model_pos + 0.5 * max(0.0, min(1.0, agreement))
        pos += (max(0.0, min(1.0, ev_strength)) - 0.5) * 0.06
    pos = max(0.0, min(1.0, pos))
    return round(lo + pos * (hi - lo), 2)
