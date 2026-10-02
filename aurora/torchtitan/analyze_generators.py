#!/usr/bin/env python3
"""Per-tile generator breakdown from a run's structured logs.

The console metrics pool every generator rank together, so they cannot show a
slow tile or separate decode work from weight-sync cost. The per-rank JSONL
logs can: each generator rank records `vllm_engine_step` spans (engine
iterations, i.e. real decode work) and `pull_model_state_dict_copy` spans
(fetching updated weights from TorchStore).

    python3 analyze_generators.py <dump_folder>/structured_logs [--prefix TS]

`--prefix` selects one run when a dump folder holds several, matching the
timestamp in the filename, e.g. --prefix 20260930-2329.
"""

import argparse
import glob
import json
import os
import statistics as st

SPANS = ("vllm_engine_step", "pull_model_state_dict_copy")


def rank_of(path: str) -> int:
    return int(path.split("global_rank_")[1].split(".")[0])


def summarize(path: str) -> dict:
    open_spans: dict[tuple[str, str], int] = {}
    durs: dict[str, list[float]] = {name: [] for name in SPANS}
    host = None
    first = last = None
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            host = host or rec.get("host_name")
            t = rec.get("time_us")
            if t:
                first = t if first is None else first
                last = t
            name = rec.get("log_type_name") or ""
            for span in SPANS:
                if name == f"{span}_start":
                    open_spans[(span, rec.get("task_name"))] = t
                elif name == f"{span}_end":
                    start = open_spans.pop((span, rec.get("task_name")), None)
                    if start is not None:
                        durs[span].append((t - start) / 1e6)
    return {
        "rank": rank_of(path),
        "host": host,
        "span_s": (last - first) / 1e6 if first and last else 0.0,
        **{k: v for k, v in durs.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("structured_logs")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    pattern = os.path.join(
        args.structured_logs, f"rl_generator.global_rank_*{args.prefix}*.jsonl"
    )
    files = sorted(glob.glob(pattern), key=rank_of)
    if not files:
        raise SystemExit(f"no generator logs matching {pattern}")

    rows = [summarize(f) for f in files]

    print(
        f"{'rank':>4} {'host':<15} {'engine_steps':>12} {'engine_s':>9} "
        f"{'pulls':>6} {'pull_s':>8} {'pull_max_s':>10} {'span_s':>8} {'busy%':>6}"
    )
    for r in rows:
        eng, pull = r["vllm_engine_step"], r["pull_model_state_dict_copy"]
        eng_s, pull_s = sum(eng), sum(pull)
        busy = 100 * eng_s / r["span_s"] if r["span_s"] else 0
        print(
            f"{r['rank']:>4} {str(r['host']):<15} {len(eng):>12} {eng_s:9.1f} "
            f"{len(pull):>6} {pull_s:8.1f} {max(pull, default=0):10.2f} "
            f"{r['span_s']:8.1f} {busy:6.1f}"
        )

    eng_tot = [sum(r["vllm_engine_step"]) for r in rows]
    pull_tot = [sum(r["pull_model_state_dict_copy"]) for r in rows]
    npulls = max((len(r["pull_model_state_dict_copy"]) for r in rows), default=0)
    print()
    print(f"ranks {len(rows)}")
    print(
        f"engine_s  min {min(eng_tot):.1f}  max {max(eng_tot):.1f}  "
        f"spread {max(eng_tot) - min(eng_tot):.1f}   <- a straggler tile shows up here"
    )
    print(
        f"pull_s    min {min(pull_tot):.1f}  max {max(pull_tot):.1f}  "
        f"mean {st.mean(pull_tot):.1f}"
        + (f"   ({st.mean(pull_tot) / npulls:.2f} s per pull)" if npulls else "")
    )


if __name__ == "__main__":
    main()
