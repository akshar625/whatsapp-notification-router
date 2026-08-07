#!/usr/bin/env python3
"""Component ablation over the 15 held-out sample rows.

Answers which components actually carry the accuracy, and records the retired
fast path as a measured negative result rather than an opinion.

    python code/ablate.py        # runs every configuration, writes eval/ablation.md

Held-out rows never enter any prompt in any configuration; exemplars are
re-rendered under each ablation so the prompt stays internally consistent.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rubric                                                    # noqa: E402
from config import CONFIG                                        # noqa: E402
from context import ContextBuilder, Dataset                      # noqa: E402
from decide import (Ablation, DecisionEngine, empty_bundle,      # noqa: E402
                    render_context, render_exemplar)
from retrieval import HistoryRetriever                           # noqa: E402

CONFIGS = [
    ("a", Ablation(False, False, False, False, "rubric only"),
     "Message text + rubric. No context features, no retrieval, no media."),
    ("b", Ablation(True, False, False, False, "+ context features"),
     "Adds user / group / business / relationship / temporal / deadline / template."),
    ("c", Ablation(True, True, False, False, "+ retrieval & evidence"),
     "Adds the candidate pool, near-duplicate aggregates and evidence citation."),
    ("d", Ablation(True, True, True, False, "+ media enrichment"),
     "Adds OCR text, ASR transcripts and the extracted media facts."),
    ("e", Ablation(True, True, True, True, "full (as shipped)"),
     "Adds the deterministic credential/payment solicitation scan."),
    ("f", Ablation(True, True, True, True, "full + fast path (retired)"),
     "As shipped, but with ENABLE_FAST_PATH=True. The retired configuration."),
]


def gold_evidence(v) -> list[str]:
    if not isinstance(v, str):
        return []
    return [i.strip() for i in re.split(r"[;,]", v)
            if i.strip() and i.strip().lower() != "none"]


def run_config(key, abl, ds, builder, retriever, split, r2d, fast_path: bool):
    """Score one configuration over the held-out 15."""
    import config as cfg_mod
    object.__setattr__(CONFIG, "enable_fast_path", fast_path)

    # Exemplars rendered under the SAME ablation, so the prompt is consistent.
    samples = ds.sample_messages
    exemplars = []
    for sid in split["exemplars"]:
        row = samples[samples.message_id == sid]
        if row.empty:
            continue
        r = row.iloc[0]
        pack = builder.build(r)
        bundle = retriever.retrieve(pack) if abl.use_retrieval else empty_bundle(pack)
        ctx, out = render_exemplar(pack, bundle, r, r2d)
        exemplars.append((render_context(pack, bundle, abl), out))

    engine = DecisionEngine(exemplars=exemplars)
    rows = []
    for n, sid in enumerate(split["held_out"], 1):
        row = samples[samples.message_id == sid]
        if row.empty:
            continue
        g = row.iloc[0]
        pack = builder.build(g)
        bundle = retriever.retrieve(pack) if abl.use_retrieval else empty_bundle(pack)
        print(f"    [{key}] {n}/15 {sid}", flush=True)
        # Route through decide() - NOT _call() - so the fast path is actually
        # consulted. Calling _call directly makes config (f) identical to (e)
        # by construction and silently invalidates the fast-path measurement.
        res = engine.decide(pack, bundle, ablation=abl)
        rows.append({
            "tier": res.tier,
            "message_id": sid,
            "pred_action": res.action, "gold_action": str(g["action"]),
            "pred_type": res.message_type, "gold_type": str(g["message_type"]),
            "pred_driver": res.driver,
            "gold_driver": r2d.get(str(g["reason"]).strip(), "?"),
            "pred_evidence": res.evidence_message_ids,
            "gold_evidence": gold_evidence(g.get("evidence_message_ids")),
            "conf": res.confidence,
        })
    object.__setattr__(CONFIG, "enable_fast_path", False)
    return rows


def score(rows):
    n = len(rows) or 1
    return {
        "n": len(rows),
        "action": sum(r["pred_action"] == r["gold_action"] for r in rows),
        "type": sum(r["pred_type"] == r["gold_type"] for r in rows),
        "driver": sum(r["pred_driver"] == r["gold_driver"] for r in rows),
        "both": sum(r["pred_action"] == r["gold_action"]
                    and r["pred_type"] == r["gold_type"] for r in rows),
        "evidence_cited": sum(bool(r["pred_evidence"]) for r in rows),
    }


def main() -> None:
    CONFIG.paths.ensure()
    ds = Dataset()
    builder = ContextBuilder(ds)
    retriever = HistoryRetriever(ds)
    split = rubric.load_split()
    r2d = rubric.reason_to_driver()

    results = {}
    for key, abl, desc in CONFIGS:
        print(f"\n=== {key}: {abl.name} ===", flush=True)
        rows = run_config(key, abl, ds, builder, retriever, split, r2d,
                          fast_path=(key == "f"))
        results[key] = {"abl": abl.name, "desc": desc, "rows": rows,
                        "score": score(rows)}
        s = results[key]["score"]
        print(f"    -> action {s['action']}/{s['n']}  type {s['type']}/{s['n']}  "
              f"driver {s['driver']}/{s['n']}")

    (CONFIG.paths.eval / "ablation.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(results)


def write_md(results) -> None:
    L = [
        "# Component Ablation",
        "",
        "Each configuration is scored over the **same 15 held-out rows** from "
        "`sample_messages.csv` (the last 15 by `message_id`; see `eval/split.json`). "
        "Held-out rows never enter any prompt in any configuration, and the 15 "
        "exemplars are re-rendered under each ablation so the prompt stays "
        "internally consistent.",
        "",
        f"Decision model: `{CONFIG.models.llm}`, temperature 0, structured tool output.",
        "",
        "| # | configuration | action | message_type | driver | both a+t | rows citing evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, _, _ in CONFIGS:
        r = results[key]
        s = r["score"]
        L.append(f"| {key} | {r['abl']} | **{s['action']}/{s['n']}** | "
                 f"**{s['type']}/{s['n']}** | {s['driver']}/{s['n']} | "
                 f"{s['both']}/{s['n']} | {s['evidence_cited']}/{s['n']} |")
    L += ["", "## What each configuration adds", ""]
    for key, _, desc in CONFIGS:
        L.append(f"- **{key} — {results[key]['abl']}**: {desc}")
    L += ["", "## Per-row detail", ""]
    for key, _, _ in CONFIGS:
        r = results[key]
        L += [f"### {key} — {r['abl']}", "",
              "| message_id | action | type | driver |", "|---|---|---|---|"]
        for row in r["rows"]:
            a = "OK" if row["pred_action"] == row["gold_action"] else "**MISS**"
            t = "OK" if row["pred_type"] == row["gold_type"] else "**MISS**"
            d = "OK" if row["pred_driver"] == row["gold_driver"] else "-"
            L.append(f"| {row['message_id']} | {a} {row['pred_action']}"
                     f"/{row['gold_action']} | {t} {row['pred_type']}"
                     f"/{row['gold_type']} | {d} |")
        L.append("")
    path = CONFIG.paths.eval / "ablation.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
