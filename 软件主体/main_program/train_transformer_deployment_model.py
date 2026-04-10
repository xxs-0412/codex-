from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch

import benchmark_network_architectures as bench
from train_real_wear_models import (
    comparison_dir,
    load_cases,
    load_processing_config,
    recommended_cycle_step,
    software_model_path,
    trained_model_dir,
)


DEPLOY_EPOCHS = 1800
DEPLOY_MODEL_NAME = "Transformer"


def save_transformer_bundle(
    model: torch.nn.Module,
    seq_scaler: bench.FeatureScaler,
    target_scaler: bench.TargetScaler,
    summary_df: pd.DataFrame,
    recommended_step: float,
    training_wear_limit_um: float,
) -> Path:
    bundle = {
        "model_state_dict": model.state_dict(),
        "x_mean": seq_scaler.mean,
        "x_std": seq_scaler.std,
        "y_mean": target_scaler.mean,
        "y_std": target_scaler.std,
        "feature_order": ["F", "D", "Cr", "actual_cycle", "wear_depth"],
        "training_wear_limit_um": float(training_wear_limit_um),
        "training_actual_life_max": float(summary_df["actual_life"].max()),
        "training_actual_life_mean": float(summary_df["actual_life"].mean()),
        "recommended_cycle_step": float(recommended_step),
        "available_coating": "DLC",
        "model_architecture": DEPLOY_MODEL_NAME,
        "sequence_length": int(bench.SEQ_LEN),
    }
    model_path = trained_model_dir() / "Transformer部署模型.pt"
    torch.save(bundle, model_path)
    software_model_path().parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, software_model_path())
    return model_path


def main() -> None:
    bench.set_seed(bench.SEED)
    bench.EPOCHS = DEPLOY_EPOCHS

    config = load_processing_config()
    summary_df, case_tables = load_cases()
    measured_summary = summary_df[summary_df["has_measured_pressure"].astype(bool)].reset_index(drop=True)
    measured_tables = {str(row.file_name): case_tables[str(row.file_name)] for row in measured_summary.itertuples(index=False)}

    if not measured_tables:
        raise ValueError("No measured-pressure cases available for deployment training.")

    train_seq, train_y = bench.build_sequence_dataset(measured_tables, bench.SEQ_LEN)
    model, seq_scaler, target_scaler = bench.train_model(DEPLOY_MODEL_NAME, train_seq, train_y)
    train_metrics = bench.evaluate_pressure(model, seq_scaler, target_scaler, train_seq, train_y)
    recommended_step = recommended_cycle_step(measured_tables)
    max_train_wear_um = max(float(table["wear_depth"].max()) * 1000.0 for table in measured_tables.values())
    model_path = save_transformer_bundle(
        model,
        seq_scaler,
        target_scaler,
        measured_summary,
        recommended_step,
        max(float(config["wear_threshold_um"]), max_train_wear_um),
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": DEPLOY_MODEL_NAME,
                "training_case_count": int(len(measured_summary)),
                "training_sample_count": int(len(train_seq)),
                "pressure_mae": float(train_metrics["pressure_mae"]),
                "pressure_rmse": float(train_metrics["pressure_rmse"]),
                "pressure_mape": float(train_metrics["pressure_mape"]),
                "recommended_cycle_step": float(recommended_step),
                "training_wear_limit_um": max(float(config["wear_threshold_um"]), max_train_wear_um),
            }
        ]
    )
    metrics_path = comparison_dir() / "Transformer部署模型指标.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    log_lines = [
        "Transformer deployment-model training complete.",
        f"Measured training cases: {len(measured_summary)}",
        f"Sequence samples: {len(train_seq)}",
        f"Pressure MAE={train_metrics['pressure_mae']:.4f} MPa",
        f"Pressure RMSE={train_metrics['pressure_rmse']:.4f} MPa",
        f"Pressure MAPE={train_metrics['pressure_mape']:.4%}",
        f"Recommended cycle step={recommended_step:.2f}",
        f"Saved deployment model: {model_path}",
        f"Software model synced to: {software_model_path()}",
        f"Metrics saved to: {metrics_path}",
    ]
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
