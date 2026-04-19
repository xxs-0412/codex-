from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import predict_life
import v2_pipeline as v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a shadow comparison between the formal V1 model and a V2 candidate model.")
    parser.add_argument("--candidate-model", type=Path, required=True, help="Path to the V2 candidate bundle.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of representative requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, summary_df, case_tables = v2.load_environment()
    threshold_um = float(config["wear_threshold_um"])
    out_dir = v2.v2_candidate_output_dir()

    root = predict_life.project_root()
    v1_model_path = root / predict_life.DEFAULT_MODEL_PATH
    candidate_model_path = args.candidate_model if args.candidate_model.is_absolute() else root / args.candidate_model
    if not candidate_model_path.exists():
        raise FileNotFoundError(f"Candidate model file not found: {candidate_model_path}")

    requests = v2.build_shadow_requests(summary_df, case_tables, threshold_um)
    if args.limit > 0:
        requests = requests[: args.limit]

    rows: list[dict[str, object]] = []
    for request in requests:
        _, export_v1, _, life_v1, reached_v1, warning_v1, _, mode_v1 = predict_life.run_prediction_request(request, model_override=v1_model_path)
        _, export_v2, _, life_v2, reached_v2, warning_v2, _, mode_v2 = predict_life.run_prediction_request(request, model_override=candidate_model_path)
        rows.append(
            {
                "source_file": str(request["source_file"]),
                "F": float(request["F"]),
                "D": float(request["D"]),
                "Cr": float(request["Cr"]),
                "actual_cycle_step": float(request["actual_cycle_step"]),
                "threshold_um": float(request["wear_threshold_um"]),
                "v1_predicted_life": float(life_v1) if reached_v1 and life_v1 is not None else None,
                "v2_predicted_life": float(life_v2) if reached_v2 and life_v2 is not None else None,
                "life_delta": (float(life_v2) - float(life_v1)) if reached_v1 and reached_v2 and life_v1 is not None and life_v2 is not None else None,
                "v1_reached": bool(reached_v1),
                "v2_reached": bool(reached_v2),
                "v1_mode": str(mode_v1),
                "v2_mode": str(mode_v2),
                "v1_monotonic_curve": bool(v2.check_monotonic_curve(export_v1)),
                "v2_monotonic_curve": bool(v2.check_monotonic_curve(export_v2)),
                "v1_warning": str(warning_v1) if warning_v1 else "",
                "v2_warning": str(warning_v2) if warning_v2 else "",
            }
        )

    result_df = pd.DataFrame(rows)
    out_path = out_dir / "shadow_compare_v1_vs_v2.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved shadow comparison results to: {out_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
