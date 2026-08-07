#!/usr/bin/env python3
"""CLI entrypoint.

    python code/run.py media     [--force]   # transcribe + extract ALL media
    python code/run.py context              # build ContextPacks, print a summary
    python code/run.py retrieve             # build retrieval bundles, print a summary
    python code/run.py report   [ids...]    # write recon/CONTEXT_REPORT.md

Everything is deterministic and idempotent. `media` is the only stage that
touches the network, and it is fully cached by SHA-256 of the file bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIG                                     # noqa: E402
from context import ContextBuilder, Dataset                   # noqa: E402
from retrieval import HistoryRetriever                        # noqa: E402


# --------------------------------------------------------------- report bits
def _fmt(v, none="-"):
    if v is None:
        return none
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _kv_block(title: str, obj, skip: tuple = ()) -> str:
    if obj is None:
        return f"**{title}:** _not applicable_\n"
    d = obj if isinstance(obj, dict) else asdict(obj)
    lines = [f"**{title}**", "", "| field | value |", "|---|---|"]
    for k, v in d.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"| `{k}` | {_fmt(v)} |")
    return "\n".join(lines) + "\n"


def pick_five(ds: Dataset) -> list[tuple[str, str]]:
    """The five hand-picked messages, per the spec, resolved against the data."""
    msgs = ds.messages
    picks: list[tuple[str, str]] = [
        ("msg_037", "Hinglish water-tanker group message (code-mixed text, society admin)"),
        ("msg_023", "Repeated bank template — this exact text occurs 4x across "
                    "message_history, but only 1x for THIS recipient, so the "
                    "user-scoped pool correctly surfaces one near-duplicate"),
    ]
    taken = {m for m, _ in picks}

    def has_rel(mid: str) -> bool:
        r = msgs[msgs.message_id == mid].iloc[0]
        bid = r["business_id"]
        if not isinstance(bid, str):
            return False
        return (str(r["user_id"]), bid) in ds._ubh

    biz = msgs[msgs.business_id.notna()]
    # Skip anything already chosen, so the report shows five DISTINCT messages.
    with_rel = next((m for m in biz.message_id if m not in taken and has_rel(m)), None)
    without_rel = next((m for m in biz.message_id if m not in taken and not has_rel(m)), None)
    if with_rel:
        picks.append((with_rel, "Business message WITH a user_business_history record"))
        taken.add(with_rel)
    if without_rel:
        picks.append((without_rel,
                      "Business message WITHOUT a relationship record (none_on_record)"))
        taken.add(without_rel)

    # A group message whose recipient has muted that group, if one exists.
    muted = None
    for _, r in msgs[msgs.group_id.notna()].iterrows():
        if str(r["message_id"]) in taken:
            continue
        gm = ds._gm.get((str(r["group_id"]), str(r["user_id"])))
        if gm is not None and int(gm.get("group_muted_by_user") or 0) == 1:
            muted = str(r["message_id"])
            break
    if muted:
        picks.append((muted, "Group message from a group the recipient has MUTED"))
    else:
        picks.append((None, "No muted-group message exists in messages.csv"))
    return picks


def write_report(ids: list[str] | None = None) -> Path:
    CONFIG.paths.ensure()
    ds = Dataset()
    builder = ContextBuilder(ds)
    retr = HistoryRetriever(ds)

    if ids:
        chosen = [(i, "requested on the command line") for i in ids]
    else:
        chosen = pick_five(ds)

    out = [
        "# Context Assembly — Sanity Report",
        "",
        "Stage 0 (`context.py`) and stage 3 (`retrieval.py`) output for five "
        "hand-picked messages, so the joins can be eyeballed before the decision "
        "layer is built.",
        "",
        f"- ContextPacks built: **{len(ds.messages)}** (one per `messages.csv` row)",
        f"- History rows indexed: **{len(ds.message_history)}**; "
        f"event rows: **{len(ds.message_events)}**",
        f"- Vision model: `{CONFIG.models.vlm}` · ASR: `{CONFIG.models.whisper}` · "
        f"decision (later): `{CONFIG.models.llm}`",
        "",
        "**Reading note.** `relationship.relationship_status = none_on_record` is a "
        "real state, not a gap: 36.7% of business messages have no "
        "`user_business_history` row and nothing is imputed for them.",
        "",
        "---",
        "",
    ]

    from media import load_media_index
    media_idx = load_media_index()

    for mid, why in chosen:
        if mid is None:
            out += [f"## (skipped) {why}", ""]
            continue
        row = ds.messages[ds.messages.message_id == mid]
        if row.empty:
            out += [f"## {mid} — NOT FOUND", ""]
            continue
        pack = builder.build(row.iloc[0])
        bundle = retr.retrieve(pack)

        out += [f"## `{mid}` — {why}", ""]
        out += ["**Raw message text**", "", "```",
                pack.message.text or "<<EMPTY - voice note, no text at all>>", "```", ""]
        out += [_kv_block("MESSAGE", pack.message), ""]
        out += [_kv_block("USER", pack.user, skip=("baseline_daily_load",)), ""]
        out += [_kv_block("USER · baseline_daily_load (aggregated over "
                          "daily_notification_summary's own 2026-07-04..07-17 window; "
                          "deliberately NOT date-joined to messages.csv)",
                          pack.user.baseline_daily_load), ""]
        out += [_kv_block("TEMPORAL", pack.temporal), ""]
        out += [_kv_block("GROUP", pack.group), ""]
        out += [_kv_block("BUSINESS", pack.business), ""]
        out += [_kv_block("RELATIONSHIP", pack.relationship), ""]

        if pack.message.media_id:
            kind = "images" if pack.message.media_type == "image" else "audio"
            rec = media_idx.get(kind, {}).get(pack.message.media_id)
            if rec:
                out += [_kv_block(f"MEDIA ENRICHMENT ({kind}: "
                                  f"`{pack.message.media_id}`)", rec,
                                  skip=("segments",)), ""]

        nd = bundle.near_duplicates
        out += [
            "**RETRIEVAL — pool**", "",
            f"- scoped pool size: **{bundle.pool_size}** "
            f"(by scope: {json.dumps(bundle.pool_scopes)})",
            f"- this user's full history: **{bundle.user_history_size}** rows",
            "",
            "**RETRIEVAL — near duplicates** "
            f"(Jaccard ≥ {CONFIG.thresholds.near_dup_jaccard})", "",
            f"- occurrences in scoped pool: **{nd.occurrences}** "
            f"(exact: {nd.exact_matches}, max Jaccard: {nd.max_jaccard})",
            f"- occurrences across this user's whole history: "
            f"**{nd.occurrences_user_global}**",
            f"- matched history ids: "
            f"`{', '.join(nd.matched_history_ids_user_global) or 'none'}`",
            "",
            "How this user reacted the previous times they saw near-identical text:",
            "",
            "| opened | replied | dismissed | muted after | reported | event rows |",
            "|---|---|---|---|---|---|",
            f"| {nd.times_opened} | {nd.times_replied} | {nd.times_dismissed} | "
            f"{nd.times_muted_after} | {nd.times_reported} | {nd.events_available} |",
            "",
        ]

        out += ["**RETRIEVAL — top candidates (hybrid: BM25 + dense cosine)**", ""]
        if not bundle.candidates:
            out += ["_No candidates: this user has no history scoped to this "
                    "sender / group / business._", ""]
        else:
            out += ["| # | history_id | scope | hybrid | lex | dense | jacc | dup | "
                    "opened | replied | dismissed | muted | reported |", "|" + "---|" * 13]
            for i, c in enumerate(bundle.candidates, 1):
                e = c.events
                out.append(
                    f"| {i} | `{c.history_message_id}` | {c.scope} | {c.hybrid_score} | "
                    f"{c.lexical_score} | {c.dense_score} | {c.jaccard} | "
                    f"{'YES' if c.is_near_duplicate else ''} | "
                    f"{_fmt(e.message_opened)} | {_fmt(e.message_replied)} | "
                    f"{_fmt(e.notification_dismissed)} | {_fmt(e.muted_after_message)} | "
                    f"{_fmt(e.message_reported)} |")
            out += ["", "Candidate text:", ""]
            for i, c in enumerate(bundle.candidates, 1):
                out += [f"*{i}. `{c.history_message_id}` "
                        f"({c.scope}, {c.created_at}, fwd={_fmt(c.forwarded_count)})*",
                        "```", (c.text or "<<empty>>")[:600], "```"]
            out += [""]
        out += [f"**evidence_message_ids this bundle would offer:** "
                f"`{';'.join(bundle.evidence_message_ids) or 'none'}`", "", "---", ""]

    path = CONFIG.paths.recon / "CONTEXT_REPORT.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Message Notification Router")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("media", help="transcribe + extract ALL media (cached)")
    m.add_argument("--force", action="store_true", help="ignore the cache")
    m.add_argument("--images-only", action="store_true")
    m.add_argument("--audio-only", action="store_true")

    sub.add_parser("context", help="build ContextPacks and print a summary")
    sub.add_parser("retrieve", help="build retrieval bundles and print a summary")

    r = sub.add_parser("report", help="write recon/CONTEXT_REPORT.md")
    r.add_argument("ids", nargs="*", help="message_ids (default: the 5 hand-picked)")

    a = ap.parse_args()
    CONFIG.paths.ensure()

    if a.cmd == "media":
        from media import enrich_all
        enrich_all(force=a.force, images=not a.audio_only, audio=not a.images_only)

    elif a.cmd == "context":
        ds = Dataset()
        packs = ContextBuilder(ds).build_all()
        print(f"built {len(packs)} context packs")
        no_rel = sum(1 for p in packs
                     if p.relationship.relationship_status == "none_on_record")
        print(f"  business messages with no relationship record: {no_rel}")
        print(f"  messages inside the user's quiet window: "
              f"{sum(1 for p in packs if p.temporal.in_quiet_hours)}")
        print(f"  messages from a muted group: "
              f"{sum(1 for p in packs if p.group and p.group.is_muted_by_recipient)}")
        print(f"  business senders using a non-official domain: "
              f"{sum(1 for p in packs if p.business and p.business.domain_matches_official is False)}")

    elif a.cmd == "retrieve":
        ds = Dataset()
        packs = ContextBuilder(ds).build_all()
        retr = HistoryRetriever(ds)
        bundles = retr.retrieve_all(packs)
        empty = sum(1 for b in bundles.values() if not b.candidates)
        dup = sum(1 for b in bundles.values() if b.near_duplicates.occurrences_user_global)
        print(f"built {len(bundles)} retrieval bundles")
        print(f"  with >=1 near-duplicate in the user's history: {dup}")
        print(f"  with an empty candidate pool: {empty}")
        print(f"  mean candidates: "
              f"{sum(len(b.candidates) for b in bundles.values())/len(bundles):.2f}")

    elif a.cmd == "report":
        write_report(a.ids or None)


if __name__ == "__main__":
    main()
