from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch


APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent
COMMON_DIR = PROJECT_ROOT / "工具和杂项" / "4.25测试集选定_论文结构重跑"
MODEL_PATH = APP_ROOT / "storage" / "training_validation" / "current_enhanced_transformer_model.pt"
METRICS_PATH = APP_ROOT / "storage" / "training_validation" / "current_enhanced_transformer_metrics.csv"

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import common_fixed_split as common  # noqa: E402


def load_deployment_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_df, case_tables = common.load_data()
    eligible_df = common.eligible_cases(summary_df)
    train_tables = {
        str(row.file_name): case_tables[str(row.file_name)]
        for row in eligible_df.itertuples(index=False)
    }
    if not train_tables:
        raise RuntimeError("No eligible cases found for deployment training.")
    return eligible_df, train_tables


def train_and_save(epochs: int, seed: int) -> Path:
    common.set_seed(seed)
    summary_df, train_tables = load_deployment_data()
    config = common.TransformerConfig(
        seq_len=12,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_ff=64,
        dropout=0.0,
        epochs=int(epochs),
        learning_rate=1e-3,
        weight_decay=1e-6,
    )
    feature_spec = common.FEATURE_R2_SPEC
    shape_config = common.ShapeLossConfig(
        name="S1_slow_abs_0p01",
        mono_lambda=0.0,
        slow_lambda=0.01,
        mono_tolerance_ratio=0.0,
    )

    train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(
        train_tables,
        config.seq_len,
    )
    model, seq_scaler, target_scaler = common.train_model(
        "Transformer",
        train_raw_seq,
        train_y,
        feature_spec,
        config,
        shape_config=shape_config,
        train_case_names=train_case_names,
        train_step_indices=train_step_indices,
    )
    metrics = common.evaluate_pressure(model, seq_scaler, target_scaler, train_raw_seq, train_y, feature_spec)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state_dict": model.state_dict(),
        "x_mean": seq_scaler.mean,
        "x_std": seq_scaler.std,
        "y_mean": target_scaler.mean,
        "y_std": target_scaler.std,
        "feature_columns": list(feature_spec.columns),
        "feature_spec_name": feature_spec.name,
        "feature_variant": "M1_R2_log_cycle_replace",
        "shape_variant": shape_config.name,
        "model_name": "Transformer",
        "model_architecture": "Enhanced Transformer",
        "transformer_config": asdict(config),
        "sequence_length": int(config.seq_len),
        "training_case_count": int(len(summary_df)),
        "training_sample_count": int(len(train_raw_seq)),
        "training_wear_limit_um": float(max(common.WEAR_THRESHOLD_UM, summary_df["final_wear_um"].astype(float).max())),
        "training_actual_life_max": float(summary_df["actual_life"].astype(float).max()),
        "training_actual_life_mean": float(summary_df["actual_life"].astype(float).mean()),
        "recommended_cycle_step": 250.0,
        "base_wear_coeff": float(common.REAL_WEAR_COEFF_MPA_INV),
        "available_coating": "DLC",
        "seed": int(seed),
        "metrics": metrics,
    }
    torch.save(bundle, MODEL_PATH)

    metrics_row = {
        "model": "Enhanced Transformer",
        "seed": int(seed),
        "epochs": int(epochs),
        "feature_variant": "M1_R2_log_cycle_replace",
        "shape_variant": shape_config.name,
        "training_case_count": int(len(summary_df)),
        "training_sample_count": int(len(train_raw_seq)),
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    summary_payload = {
        key: value
        for key, value in bundle.items()
        if key not in {"model_state_dict", "x_mean", "x_std", "y_mean", "y_std"}
    }
    MODEL_PATH.with_suffix(".json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Train current Enhanced Transformer deployment model.")
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260425)
    args = parser.parse_args()
    model_path = train_and_save(args.epochs, args.seed)
    print(f"Saved current Enhanced Transformer model: {model_path}")
    print(f"Saved training metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
