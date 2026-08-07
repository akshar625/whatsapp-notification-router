#!/usr/bin/env python3
"""Run the full pipeline over dataset/messages.csv and write output.csv.

    python code/predict.py            # run all 110 and write output.csv
    python code/predict.py --report   # re-report from cache/predictions.json

Every decision is cached to cache/predictions.json keyed by message_id, so a
re-run resumes rather than re-billing. Validation and reporting read the CSV
back off disk, so what is checked is what will actually be submitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rubric                                            # noqa: E402
from config import CONFIG                                # noqa: E402
from context import ContextBuilder, Dataset              # noqa: E402
from decide import DecisionEngine, build_exemplars       # noqa: E402
from retrieval import HistoryRetriever                   # noqa: E402

OUT_COLUMNS = ["message_id", "action", "message_type", "reason",
               "confidence", "evidence_message_ids"]
CACHE = CONFIG.paths.cache / "predictions.json"


# ------------------------------------------------------------------- running
def run(force: bool = False) -> list[dict]:
    CONFIG.paths.ensure()
    ds = Dataset()
    builder = ContextBuilder(ds)
    retriever = HistoryRetriever(ds)
    split = rubric.load_split()

    # Same 15 exemplars used for the held-out evaluation, so the configuration
    # that was measured is the configuration that ships.
    exemplars = build_exemplars(ds, builder, retriever, split["exemplars"])
    engine = DecisionEngine(exemplars=exemplars)
    print(f"[model] {CONFIG.models.llm}  |  {len(exemplars)} cached exemplars")

    done: dict[str, dict] = {}
    if CACHE.exists() and not force:
        done = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"[cache] resuming with {len(done)} decisions already made")

    msgs = ds.messages
    t0 = time.time()
    for n, (_, row) in enumerate(msgs.iterrows(), 1):
        mid = str(row["message_id"])
        if mid in done and not force:
            continue
        pack = builder.build(row)
        bundle = retriever.retrieve(pack)
        try:
            res = engine.decide(pack, bundle)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [{n}/{len(msgs)}] {mid} ERROR {type(exc).__name__}: {exc}",
                  flush=True)
            time.sleep(3)
            res = engine.decide(pack, bundle)          # one retry, then propagate
        done[mid] = {
            "message_id": mid, "action": res.action, "message_type": res.message_type,
            "driver": res.driver, "reason": res.reason, "confidence": res.confidence,
            "evidence_message_ids": res.evidence_message_ids, "tier": res.tier,
            "certainty": res.certainty, "traps": res.triggered_traps,
            "violations": res.violations, "dropped_evidence": res.dropped_evidence,
            "rationale": res.brief_rationale,
            "pool_size": bundle.pool_size,
            "in_quiet_hours": pack.temporal.in_quiet_hours,
            "temporal_validity": pack.temporal_validity.validity,
            "relationship_status": pack.relationship.relationship_status,
            "has_embedded_instruction": pack.message.has_embedded_instruction,
        }
        CACHE.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
        if n % 10 == 0 or n == len(msgs):
            print(f"  [{n}/{len(msgs)}] {mid} -> {res.action}/{res.message_type} "
                  f"({res.tier})  [{time.time()-t0:.0f}s]", flush=True)

    rows = [done[str(m)] for m in msgs["message_id"]]
    write_csv(rows)
    return rows


def write_csv(rows: list[dict]) -> Path:
    """Write output.csv with the exact required columns, in order."""
    path = CONFIG.paths.output_csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, quoting=csv.QUOTE_MINIMAL,
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            ev = r["evidence_message_ids"]
            w.writerow({
                "message_id": r["message_id"],
                "action": r["action"],
                "message_type": r["message_type"],
                "reason": r["reason"],
                "confidence": f"{float(r['confidence']):.2f}",
                "evidence_message_ids": ";".join(ev) if ev else "none",
            })
    print(f"[written] {path}")
    return path


# ---------------------------------------------------------------- validation
def validate(rows: list[dict]) -> bool:
    ds = Dataset()
    expected = [str(m) for m in ds.messages["message_id"]]
    hist_ids = {str(m) for m in ds.message_history["message_id"]}
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))

    print("\n" + "=" * 100)
    print("OUTPUT VALIDATION (read back from output.csv on disk)")
    print("=" * 100)

    # Round-trip through pandas: the business templates are multi-line, so the
    # quoting has to survive a real parse, not just look right in the file.
    df = pd.read_csv(CONFIG.paths.output_csv, dtype=str, keep_default_na=False)
    check("round-trips through pandas.read_csv", len(df) == 110,
          f"parsed {len(df)} rows")
    check("columns exact and in order", list(df.columns) == OUT_COLUMNS,
          str(list(df.columns)))
    check("row count is exactly 110", len(df) == 110, f"{len(df)}")

    got = list(df["message_id"])
    check("one row per message_id, no missing", set(got) == set(expected),
          f"missing={sorted(set(expected)-set(got))[:5]} extra={sorted(set(got)-set(expected))[:5]}")
    check("no duplicate message_id", len(got) == len(set(got)),
          f"{len(got)-len(set(got))} dupes")
    check("row order matches messages.csv", got == expected)

    bad_a = sorted(set(df["action"]) - set(CONFIG.actions))
    check("every action in {notify,digest,mute}", not bad_a, str(bad_a))
    bad_t = sorted(set(df["message_type"]) - set(CONFIG.message_types))
    check("every message_type in the allowed 11", not bad_t, str(bad_t))

    conf_ok, bad_c = True, []
    for v in df["confidence"]:
        try:
            f = float(v)
            if not (0.0 <= f <= 1.0):
                conf_ok = False
                bad_c.append(v)
        except ValueError:
            conf_ok = False
            bad_c.append(v)
    check("every confidence numeric in [0,1]", conf_ok, str(bad_c[:5]))

    empties = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in OUT_COLUMNS}
    check("no empty cells", sum(empties.values()) == 0, str(empties))
    nans = {c: int((df[c].astype(str).str.lower().isin(["nan", "none-nan"])).sum())
            for c in OUT_COLUMNS if c != "evidence_message_ids"}
    check("no literal NaN values", sum(nans.values()) == 0, str(nans))

    # Evidence: separator, sentinel, resolvability, no fabrication.
    bad_ids, sep_ok, counts = [], True, Counter()
    for v in df["evidence_message_ids"]:
        s = str(v).strip()
        if s == "none":
            counts[0] += 1
            continue
        if "," in s:
            sep_ok = False
        ids = [i for i in s.split(";") if i]
        counts[len(ids)] += 1
        bad_ids += [i for i in ids if i not in hist_ids]
    check("evidence is semicolon-separated (no commas)", sep_ok)
    check('empty evidence written as literal "none"',
          counts[0] == int((df["evidence_message_ids"] == "none").sum()))
    check("every evidence id resolves to message_history.csv", not bad_ids,
          str(sorted(set(bad_ids))[:5]))
    check("zero fabricated ids (dropped during validation)",
          sum(len(r["dropped_evidence"]) for r in rows) == 0,
          f"{sum(len(r['dropped_evidence']) for r in rows)} dropped")

    raw = CONFIG.paths.output_csv.read_text(encoding="utf-8")
    check("multi-line fields are quoted, file parses to 110 data rows",
          raw.count("\n") >= 110 and len(df) == 110,
          f"{raw.count(chr(10))} newlines in file, {len(df)} parsed rows")

    print(f"\n  OVERALL: {'ALL CHECKS PASS' if ok else 'FAILURES ABOVE'}")
    return ok


# ---------------------------------------------------------------- reporting
SAMPLE_ACTION_PCT = {"digest": 36.7, "mute": 33.3, "notify": 30.0}


def report(rows: list[dict]) -> None:
    n = len(rows)
    ds = Dataset()
    samp = ds.sample_messages
    print("\n" + "=" * 100)
    print("DISTRIBUTIONS  (no labels exist for these 110 rows - outliers only, not a target)")
    print("=" * 100)

    ac = Counter(r["action"] for r in rows)
    print("\naction:")
    for a in ("notify", "digest", "mute"):
        print(f"  {a:8s} {ac[a]:3d}  {100*ac[a]/n:5.1f}%   sample {SAMPLE_ACTION_PCT[a]:5.1f}%")

    tc = Counter(r["message_type"] for r in rows)
    sc = Counter(samp["message_type"])
    print("\nmessage_type:")
    for t in CONFIG.message_types:
        sp = 100 * sc.get(t, 0) / len(samp)
        print(f"  {t:16s} {tc.get(t,0):3d}  {100*tc.get(t,0)/n:5.1f}%   sample {sp:5.1f}%")

    print("\nconfidence per action (gold means: notify 0.874, mute 0.836, digest 0.816):")
    for a in ("notify", "digest", "mute"):
        v = [float(r["confidence"]) for r in rows if r["action"] == a]
        if not v:
            continue
        g = rubric.GOLD_CONFIDENCE_MEAN[a]
        print(f"  {a:8s} n={len(v):3d}  mean={sum(v)/len(v):.3f}  "
              f"range=[{min(v):.2f},{max(v):.2f}]  gold={g:.3f}  "
              f"delta={sum(v)/len(v)-g:+.3f}")

    ec = Counter(len(r["evidence_message_ids"]) for r in rows)
    gold = rubric.GOLD_EVIDENCE_COUNT_DIST
    tg = sum(gold.values())
    print("\nevidence count (gold norm: 1 id 83%, 2 ids 10%, none 7%):")
    for k in (1, 2, 0):
        print(f"  {k} id(s): {ec.get(k,0):3d}  {100*ec.get(k,0)/n:5.1f}%   "
              f"gold {100*gold.get(k,0)/tg:5.1f}%")
    if any(k > 2 for k in ec):
        print(f"  !! more than 2 ids: {{k:v for k,v in ec.items() if k>2}}")

    tier = Counter(r["tier"] for r in rows)
    print(f"\ntier: {dict(tier)}")
    fast = [r for r in rows if r["tier"].startswith("fast")]
    print(f"  fast-path hits ({len(fast)}):")
    for r in fast:
        print(f"    {r['message_id']:10s} {r['tier']:10s} -> {r['action']}/{r['message_type']}")
    print(f"  extended driver used: {sum(r['driver']=='extended' for r in rows)}")

    traps = Counter()
    for r in rows:
        for t in r["traps"]:
            traps[str(t).strip().upper()] += 1
    print("\ntraps triggered:")
    for t in [f"T{i}" for i in range(1, 15)] + ["T12B"]:
        if traps.get(t):
            print(f"  {t:5s} {traps[t]:3d} rows")
    unknown = {k: v for k, v in traps.items()
               if k not in {f"T{i}" for i in range(1, 15)} | {"T12B"}}
    if unknown:
        print(f"  (unrecognised trap labels: {unknown})")


def flags(rows: list[dict]) -> None:
    by_id = {r["message_id"]: r for r in rows}
    ds = Dataset()
    builder = ContextBuilder(ds)
    print("\n" + "=" * 100)
    print("FLAGGED FOR REVIEW")
    print("=" * 100)

    hard = [r for r in rows if any(v.startswith("HARD") for v in r["violations"])]
    print(f"\n1. HARD constraint fired (model output overridden): {len(hard)}")
    for r in hard:
        print(f"   {r['message_id']:10s} -> {r['action']}/{r['message_type']}  {r['violations']}")

    soft = [r for r in rows if r["tier"] == "llm+escalated"
            or any(v.startswith(("SOFT", "DRIVER")) for v in r["violations"])]
    print(f"\n2. SOFT violation escalated for re-deliberation: {len(soft)}")
    for r in soft:
        print(f"   {r['message_id']:10s} -> {r['action']}/{r['message_type']} "
              f"tier={r['tier']}  {r['violations']}")

    inj = [r for r in rows if r["has_embedded_instruction"]]
    print(f"\n3. Injection rows (deterministic scan): {len(inj)}")
    for r in inj:
        good = r["action"] == "mute" and r["message_type"] in ("spam", "scam")
        print(f"   {r['message_id']:10s} -> {r['action']}/{r['message_type']} "
              f"driver={r['driver']}  {'OK' if good else '<-- REVIEW'}")

    print("\n4. Expired posters shared across opted-in vs opted-out users (T3 x T1):")
    packs = {}
    for _, row in ds.messages.iterrows():
        p = builder.build(row)
        if p.temporal_validity.validity == "past":
            packs[p.message_id] = p
    for mid, p in packs.items():
        r = by_id[mid]
        print(f"   {mid:10s} user={p.user_id} media={p.message.media_id} "
              f"deadline={p.temporal_validity.raw_expression!r} "
              f"({p.temporal_validity.days_until_deadline}d) "
              f"rel={p.relationship.relationship_status} -> "
              f"{r['action']}/{r['message_type']} driver={r['driver']}")
    if len(packs) == 2:
        a, b = list(packs)
        print(f"   diverge? {by_id[a]['action']} vs {by_id[b]['action']} -> "
              f"{'YES' if by_id[a]['action'] != by_id[b]['action'] else 'NO - both same'}")

    quiet = [r for r in rows if r["action"] == "notify" and r["in_quiet_hours"]]
    print(f"\n5. action=notify inside the user's quiet hours: {len(quiet)}")
    for r in quiet:
        print(f"   {r['message_id']:10s} -> {r['message_type']} driver={r['driver']} "
              f"conf={r['confidence']}")

    empty = [r for r in rows if r["pool_size"] == 0]
    print(f"\n6. Empty candidate pool: {len(empty)}")
    for r in empty:
        ev = r["evidence_message_ids"] or "none"
        print(f"   {r['message_id']:10s} pool=0 evidence={ev} "
              f"{'OK' if not r['evidence_message_ids'] else '<-- REVIEW'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="re-report from cache")
    ap.add_argument("--force", action="store_true", help="ignore the decision cache")
    a = ap.parse_args()

    if a.report:
        done = json.loads(CACHE.read_text(encoding="utf-8"))
        ds = Dataset()
        rows = [done[str(m)] for m in ds.messages["message_id"] if str(m) in done]
        write_csv(rows)
    else:
        rows = run(force=a.force)

    validate(rows)
    report(rows)
    flags(rows)


if __name__ == "__main__":
    main()
