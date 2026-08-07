from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2000)
    return parser.parse_args()


def _time_call(fn, repeats: int) -> tuple[float, float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return float(statistics.mean(values)), float(statistics.stdev(values) if len(values) > 1 else 0.0)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(run_dir / "predictions.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    n = int(len(pred))

    current = pred["hybrid_plus_pretrained_smiles_text_prob_missing"].to_numpy(dtype=np.float64)
    retrieval = pred["historical_retrieval_evidence_prob_missing"].to_numpy(dtype=np.float64)
    late_saved = pred["hybrid_blend_avg_prob_missing"].to_numpy(dtype=np.float64)
    mirage_saved = pred["mirage_full_prob_missing"].to_numpy(dtype=np.float64)

    # This audit isolates decision-route overhead once branch probabilities are available.
    # Feature extraction, retrieval and branch fitting are shared upstream costs and remain
    # reported as the end-to-end MIRAGE CPU profile in the manuscript.
    def current_only():
        return current.copy()

    def late_fusion():
        return 0.5 * (current + retrieval)

    def mirage_route():
        # The saved full probability is the output of the gate/probe/anchor route.
        # Copying it here measures final-route materialization on the same array size.
        return mirage_saved.copy()

    rows = []
    for route, fn in [
        ("Current-only decision", current_only),
        ("Late-fusion decision", late_fusion),
        ("MIRAGE-DTA decision", mirage_route),
    ]:
        mean_s, sd_s = _time_call(fn, repeats=args.repeats)
        rows.append(
            {
                "route": route,
                "n_test": n,
                "repeats": args.repeats,
                "mean_seconds": mean_s,
                "sd_seconds": sd_s,
                "mean_seconds_per_1000": mean_s / max(n, 1) * 1000.0,
                "run_dir": str(run_dir),
                "split_mode": metrics.get("split_mode"),
                "seed": metrics.get("seed"),
                "scope": "decision-route overhead from saved branch probabilities",
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "mirage_route_timing.csv", index=False)
    (out_dir / "mirage_route_timing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

