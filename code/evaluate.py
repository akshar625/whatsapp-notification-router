"""Score the router against the HELD-OUT half of sample_messages.csv.

The exemplar/held-out split is written by rubric.load_split(): sorted by
message_id, first 15 are prompting exemplars, last 15 are held out. Held-out
rows never enter any prompt.

Gold labels are used for SCORING ONLY. No message_id -> label mapping is ever
consulted at decision time.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rubric                                            # noqa: E402
from config import CONFIG                                # noqa: E402
from context import ContextBuilder, Dataset              # noqa: E402
from decide import DecisionEngine, build_exemplars       # noqa: E402
from retrieval import HistoryRetriever                   # noqa: E402


def gold_evidence(v) -> list[str]:
    if not isinstance(v, str):
        return []
    return [i.strip() for i in re.split(r"[;,]", v)
            if i.strip() and i.strip().lower() != "none"]


def evidence_support(res, bundle) -> dict:
    """EVIDENCE SUPPORT SCORE - computed WITHOUT reference to gold ids.

    Exact-match against gold is unattainable by construction (see README ->
    Corpus observations), so defensibility is the target instead: every cited id
    must be one the model actually saw, its recorded reaction must corroborate
    the driver it was cited for, and a repetition-based driver must cite an
    actual near-duplicate.
    """
    by_id = {c.history_message_id: c for c in bundle.candidates}
    ids = res.evidence_message_ids
    in_pool = all(i in by_id for i in ids)
    supported = [i for i in ids
                 if rubric.events_support_driver(res.driver, by_id[i].events)]
    needs_dup = res.driver in rubric.REPETITION_DRIVERS
    dup_ok = (not needs_dup) or any(by_id[i].is_near_duplicate for i in ids) or not ids
    return {
        "n_ids": len(ids),
        "in_pool": in_pool,
        "driver_class": rubric.driver_class(res.driver),
        "events_support": len(supported),
        "events_support_all": bool(ids) and len(supported) == len(ids),
        "repetition_driver": needs_dup,
        "near_dup_ok": dup_ok,
        "defensible": bool(in_pool and (not ids or supported) and dup_ok),
    }


def run(limit: int | None = None) -> dict:
    CONFIG.paths.ensure()
    split = rubric.load_split()
    ds = Dataset()
    builder = ContextBuilder(ds)
    retriever = HistoryRetriever(ds)

    print(f"[split] exemplars={len(split['exemplars'])} held_out={len(split['held_out'])}")
    print(f"[split] written to {CONFIG.paths.eval / 'split.json'}")
    print(f"[model] decision engine: {CONFIG.models.llm}")

    exemplars = build_exemplars(ds, builder, retriever, split["exemplars"])
    print(f"[prompt] {len(exemplars)} worked examples in the cacheable prefix")

    engine = DecisionEngine(exemplars=exemplars)
    samples = ds.sample_messages
    r2d = rubric.reason_to_driver()

    held = split["held_out"][:limit] if limit else split["held_out"]
    rows = []
    for n, sid in enumerate(held, 1):
        row = samples[samples.message_id == sid]
        if row.empty:
            continue
        g = row.iloc[0]
        pack = builder.build(g)
        bundle = retriever.retrieve(pack)
        print(f"  [{n}/{len(held)}] {sid} ...", flush=True)
        res = engine.decide(pack, bundle)
        rows.append({
            "support": evidence_support(res, bundle),
            "message_id": sid,
            "pred_action": res.action, "gold_action": str(g["action"]),
            "pred_type": res.message_type, "gold_type": str(g["message_type"]),
            "pred_driver": res.driver,
            "gold_driver": r2d.get(str(g["reason"]).strip(), "?"),
            "pred_evidence": res.evidence_message_ids,
            "gold_evidence": gold_evidence(g.get("evidence_message_ids")),
            "pred_conf": res.confidence, "gold_conf": float(g["confidence"]),
            "tier": res.tier, "certainty": res.certainty,
            "traps": res.triggered_traps, "violations": res.violations,
            "dropped_evidence": res.dropped_evidence,
            "rationale": res.brief_rationale,
        })

    out = CONFIG.paths.eval / "heldout_results.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    report(rows)
    print(f"\n[written] {out}")
    return {"rows": rows}


def report(rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        print("no rows scored")
        return
    act = sum(r["pred_action"] == r["gold_action"] for r in rows)
    typ = sum(r["pred_type"] == r["gold_type"] for r in rows)
    both = sum(r["pred_action"] == r["gold_action"] and r["pred_type"] == r["gold_type"]
               for r in rows)
    ev_exact = sum(set(r["pred_evidence"]) == set(r["gold_evidence"]) for r in rows)
    ev_overlap = sum(bool(set(r["pred_evidence"]) & set(r["gold_evidence"])) for r in rows)

    print("\n" + "=" * 108)
    print("HELD-OUT RESULTS (15 rows never shown to the model)")
    print("=" * 108)
    hdr = (f"{'message_id':16s} {'action pred/gold':22s} {'type pred/gold':26s} "
           f"{'conf':11s} {'tier':12s} {'ev':4s}")
    print(hdr)
    print("-" * 108)
    for r in rows:
        a = f"{r['pred_action']}/{r['gold_action']}"
        t = f"{r['pred_type']}/{r['gold_type']}"
        ok_a = "OK " if r["pred_action"] == r["gold_action"] else "XX "
        ok_t = "OK " if r["pred_type"] == r["gold_type"] else "XX "
        ev = ("=" if set(r["pred_evidence"]) == set(r["gold_evidence"])
              else ("~" if set(r["pred_evidence"]) & set(r["gold_evidence"]) else "X"))
        print(f"{r['message_id']:16s} {ok_a}{a:19s} {ok_t}{t:23s} "
              f"{r['pred_conf']:.2f}/{r['gold_conf']:.2f}  {r['tier']:12s} {ev:4s}")
        print(f"{'':16s}   driver: {r['pred_driver']}  |  gold: {r['gold_driver']}")
        print(f"{'':16s}   evid  : {r['pred_evidence'] or 'none'}  |  gold: "
              f"{r['gold_evidence'] or 'none'}  traps={r['traps']}")
        if r["violations"]:
            print(f"{'':16s}   !! violations: {r['violations']}")
        if r["dropped_evidence"]:
            print(f"{'':16s}   !! dropped (not in pool): {r['dropped_evidence']}")

    drv = sum(r["pred_driver"] == r["gold_driver"] for r in rows)
    print("-" * 108)
    print("PRIMARY METRICS")
    print(f"  action accuracy      : {act}/{n} = {act/n:.0%}")
    print(f"  message_type accuracy: {typ}/{n} = {typ/n:.0%}")
    print(f"  driver accuracy      : {drv}/{n} = {drv/n:.0%}")
    print(f"  both action+type     : {both}/{n} = {both/n:.0%}")
    print(f"  tier                 : {dict(Counter(r['tier'] for r in rows))}")
    print(f"  extended driver used : {sum(r['pred_driver']=='extended' for r in rows)}")

    # ---- evidence support score (no gold ids involved) ----------------------
    sup = [r["support"] for r in rows]
    cited = [s for s in sup if s["n_ids"] > 0]
    print("\nEVIDENCE SUPPORT SCORE (computed without gold ids)")
    print(f"  cited id in candidate pool   : "
          f"{sum(s['in_pool'] for s in sup)}/{n} = "
          f"{sum(s['in_pool'] for s in sup)/n:.0%}  (must be 100%)")
    print(f"  fabricated ids dropped       : "
          f"{sum(len(r['dropped_evidence']) for r in rows)}")
    if cited:
        print(f"  events corroborate driver    : "
              f"{sum(s['events_support_all'] for s in cited)}/{len(cited)} of rows that cited")
    rep = [s for s in sup if s["repetition_driver"]]
    if rep:
        print(f"  repetition driver cites a near-duplicate: "
              f"{sum(s['near_dup_ok'] for s in rep)}/{len(rep)}")
    print(f"  fully defensible rows        : "
          f"{sum(s['defensible'] for s in sup)}/{n} = "
          f"{sum(s['defensible'] for s in sup)/n:.0%}")
    dist = Counter(s["n_ids"] for s in sup)
    gold_d = rubric.GOLD_EVIDENCE_COUNT_DIST
    tot_gold = sum(gold_d.values())
    print("  evidence-count distribution vs gold norm:")
    for k in (1, 2, 0):
        ours = dist.get(k, 0)
        print(f"    {k} id(s): ours {ours}/{n} = {ours/n:5.0%}   "
              f"gold {gold_d.get(k,0)}/{tot_gold} = {gold_d.get(k,0)/tot_gold:5.0%}")
    print("  [diagnostic only, artifact-of-construction] exact id match vs gold: "
          f"{ev_exact}/{n} (any overlap {ev_overlap}/{n})")

    # ---- confidence calibration -------------------------------------------
    print("\nCONFIDENCE CALIBRATION")
    ok_const, ok_mean = True, True
    for a in ("notify", "digest", "mute"):
        vals = [r["pred_conf"] for r in rows if r["pred_action"] == a]
        if not vals:
            print(f"  {a:7s} (none predicted)")
            continue
        uniq = sorted(set(vals))
        mean = sum(vals) / len(vals)
        target = rubric.GOLD_CONFIDENCE_MEAN[a]
        delta = mean - target
        const_flag = ("OK" if len(uniq) > 1
                      else ("SINGLE-ROW" if len(vals) == 1 else "CONSTANT!"))
        if const_flag == "CONSTANT!":
            ok_const = False
        within = abs(delta) <= 0.02
        ok_mean &= within
        print(f"  {a:7s} n={len(vals):2d} values={uniq}  mean={mean:.3f} "
              f"gold={target:.3f} delta={delta:+.3f} "
              f"{'PASS' if within else 'FAIL (>0.02)'}  spread={const_flag}")
    print(f"  assert non-constant : {'PASS' if ok_const else 'FAIL'}")
    print(f"  assert mean +/-0.02 : {'PASS' if ok_mean else 'FAIL'}")
    print(f"  certainty histogram : "
          f"{dict(sorted(Counter(r['certainty'] for r in rows).items()))}")

    print("\nconfusion (gold -> pred), action")
    c = Counter((r["gold_action"], r["pred_action"]) for r in rows)
    for (g, p), k in sorted(c.items()):
        mark = "" if g == p else "   <-- miss"
        print(f"  {g:7s} -> {p:7s} : {k}{mark}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    run(limit=lim)
