from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import v2_pipeline as v2


def assert_full_benchmark_mode(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark mode file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mode = str(payload.get("mode", "")).strip().lower()
    if mode != "full":
        raise ValueError("Candidate deployment training requires full benchmark results. Lite screening outputs are not allowed.")


def load_recommended_candidate(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Recommended candidate file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("deployment_ready", False)):
        raise ValueError("No deployment-ready V2 candidate is available. Keep V1 as the formal model.")
    return payload


def main() -> None:
    _, device_label = v2.resolve_training_device()
    print(f"device: {device_label}")
    config, summary_df, case_tables = v2.load_environment()
    threshold_um = float(config["wear_threshold_um"])
    real_k = float(config["real_wear_coeff_mpa_inv"])
    out_dir = v2.v2_candidate_output_dir()

    assert_full_benchmark_mode(out_dir / "benchmark_mode.json")
    candidate_payload = load_recommended_candidate(out_dir / "recommended_candidate.json")
    candidate_config = v2.config_from_summary_row(candidate_payload)
    final_epochs = int(round(float(candidate_payload["median_best_epoch"])))

    measured_df = v2.measured_summary(summary_df)
    measured_tables = {str(row.file_name): case_tables[str(row.file_name)] for row in measured_df.itertuples(index=False)}
    train_dataset = v2.build_raw_sequence_dataset(measured_tables, candidate_config.sequence_length)
    train_config = candidate_config.with_updates(use_validation=False)
    runtime = v2.train_candidate_model(
        train_dataset=train_dataset,
        val_dataset=None,
        config=train_config,
        real_k=real_k,
        forced_epochs=final_epochs,
    )

    training_wear_limit_um = max(
        float(threshold_um),
        max(float(table["wear_depth"].max()) * 1000.0 for table in measured_tables.values()),
    )
    model_name = f"{v2.safe_stem(candidate_config.name)}_candidate.pt"
    model_path = v2.save_candidate_bundle(
        runtime=runtime,
        train_summary=measured_df,
        case_tables=measured_tables,
        training_wear_limit_um=training_wear_limit_um,
        model_filename=model_name,
        copy_to_software=False,
    )

    train_pressure = v2.evaluate_pressure(runtime, train_dataset)
    metrics_df = pd.DataFrame(
        [
            {
                **candidate_config.to_record(),
                "forced_epochs": int(final_epochs),
                "training_case_count": int(len(measured_df)),
                "training_sample_count": int(len(train_dataset)),
                "pressure_mae": float(train_pressure["pressure_mae"]),
                "pressure_rmse": float(train_pressure["pressure_rmse"]),
                "pressure_mape": float(train_pressure["pressure_mape"]),
                "training_wear_limit_um": float(training_wear_limit_um),
                "candidate_model_path": str(model_path),
            }
        ]
    )
    metrics_path = out_dir / "v2_candidate_training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"Saved candidate deployment model to: {model_path}")
    print(f"Saved candidate training metrics to: {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
