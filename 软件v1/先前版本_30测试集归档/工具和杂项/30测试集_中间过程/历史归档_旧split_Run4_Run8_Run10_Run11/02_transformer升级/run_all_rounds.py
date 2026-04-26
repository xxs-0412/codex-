from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
COMMON_DIR = ROOT_DIR / "工具和杂项" / "30测试集_中间过程"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import common_fixed_split as common


BASELINE_RESULT_DIR = ROOT_DIR / "结果" / "30测试集" / "01_基准测试"
DETAIL_ALL_FILENAME = "候选汇总.csv"
NOTES_FILENAME = "运行说明.txt"
BEST_CONFIG_FILENAME = "最优配置.json"


@dataclass(frozen=True)
class RoundConfig:
    round_id: str
    description: str
    seq_len: int
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 64
    dropout: float = 0.0
    epochs: int = 1200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_norm: float | None = None
    use_validation: bool = False
    lr_patience: int = 80
    lr_factor: float = 0.5
    es_patience: int = 220
    es_min_delta: float = 1e-5


ROUND_CONFIGS = [
    RoundConfig("baseline_seq6", "基准 Transformer 配置，seq_len=6", seq_len=6),
    RoundConfig("seq6_regularized", "seq_len=6 + dropout/weight_decay/grad_clip", seq_len=6, dropout=0.1, epochs=2200, learning_rate=5e-4, weight_decay=1e-4, grad_clip_norm=1.0),
    RoundConfig("seq6_low_lr", "seq_len=6 + 低学习率长训练", seq_len=6, epochs=2200, learning_rate=3e-4, weight_decay=1e-5, grad_clip_norm=1.0),
    RoundConfig("seq8_base", "seq_len=8 基础配置", seq_len=8),
    RoundConfig("seq8_mild", "seq_len=8 + 温和训练策略", seq_len=8, epochs=1800, learning_rate=5e-4, weight_decay=1e-5, grad_clip_norm=1.0),
    RoundConfig("seq8_d64_ff128", "seq_len=8, d_model=64, ff=128, 2层", seq_len=8, d_model=64, num_layers=2, dim_ff=128),
    RoundConfig("seq10_base", "seq_len=10 基础配置", seq_len=10),
    RoundConfig("seq10_mild", "seq_len=10 + 温和训练策略", seq_len=10, epochs=1800, learning_rate=5e-4, weight_decay=1e-5, grad_clip_norm=1.0),
    RoundConfig("seq10_d48_ff96", "seq_len=10, d_model=48, ff=96, 2层", seq_len=10, d_model=48, num_layers=2, dim_ff=96),
    RoundConfig("seq10_d64_ff128_l2", "seq_len=10, d_model=64, ff=128, 2层", seq_len=10, d_model=64, num_layers=2, dim_ff=128),
    RoundConfig("seq10_d64_ff256_l3", "seq_len=10, d_model=64, ff=256, 3层", seq_len=10, d_model=64, num_layers=3, dim_ff=256),
    RoundConfig("seq10_d96_ff192_l2", "seq_len=10, d_model=96, ff=192, 2层", seq_len=10, d_model=96, num_layers=2, dim_ff=192),
    RoundConfig("seq10_d64_dropout", "seq_len=10, d_model=64 + dropout=0.1", seq_len=10, d_model=64, num_layers=2, dim_ff=128, dropout=0.1, epochs=1800, learning_rate=5e-4, weight_decay=1e-4, grad_clip_norm=1.0),
    RoundConfig("seq10_val_regularized", "seq_len=10 + validation/early-stop 正则分支", seq_len=10, dropout=0.1, epochs=3000, learning_rate=5e-4, weight_decay=1e-4, grad_clip_norm=1.0, use_validation=True),
    RoundConfig("seq12_base", "seq_len=12 基础配置", seq_len=12),
    RoundConfig("seq12_mild", "seq_len=12 + 温和训练策略", seq_len=12, epochs=1800, learning_rate=5e-4, weight_decay=1e-5, grad_clip_norm=1.0),
    RoundConfig("seq12_d64_ff128_l2", "seq_len=12, d_model=64, ff=128, 2层", seq_len=12, d_model=64, num_layers=2, dim_ff=128),
    RoundConfig("seq12_d64_ff256_l3", "seq_len=12, d_model=64, ff=256, 3层", seq_len=12, d_model=64, num_layers=3, dim_ff=256),
    RoundConfig("seq12_d96_ff192_l2", "seq_len=12, d_model=96, ff=192, 2层", seq_len=12, d_model=96, num_layers=2, dim_ff=192),
    RoundConfig("seq12_val_mild", "seq_len=12 + validation/early-stop 温和分支", seq_len=12, dropout=0.05, epochs=2600, learning_rate=5e-4, weight_decay=1e-5, grad_clip_norm=1.0, use_validation=True),
    RoundConfig("seq15_d64_ff128_l2", "seq_len=15, d_model=64, ff=128, 2层", seq_len=15, d_model=64, num_layers=2, dim_ff=128),
    RoundConfig("seq15_d64_ff256_l3", "seq_len=15, d_model=64, ff=256, 3层", seq_len=15, d_model=64, num_layers=3, dim_ff=256),
    RoundConfig("seq20_d64_ff128_l2", "seq_len=20, d_model=64, ff=128, 2层", seq_len=20, d_model=64, num_layers=2, dim_ff=128),
    RoundConfig("seq20_mild", "seq_len=20 + 温和训练策略", seq_len=20, d_model=64, num_layers=2, dim_ff=128, epochs=1800, learning_rate=5e-4, weight_decay=1e-5, grad_clip_norm=1.0),
]


def _stable_round_seed(name: str) -> int:
    total = 0
    for idx, ch in enumerate(str(name)):
        total += (idx + 1) * ord(ch)
    return int(common.SEED + (total % 100000))


def to_transformer_config(config: RoundConfig) -> common.TransformerConfig:
    return common.TransformerConfig(
        seq_len=config.seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_ff=config.dim_ff,
        dropout=config.dropout,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def pick_validation_case(train_df: pd.DataFrame) -> str:
    sorted_df = train_df.sort_values("actual_life").reset_index(drop=True)
    return str(sorted_df.iloc[len(sorted_df) // 2]["file_name"])


def train_model(
    config: RoundConfig,
    train_tables: dict[str, pd.DataFrame],
    val_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[nn.Module, common.FeatureScaler, common.TargetScaler, dict[str, float | str]]:
    train_raw_seq, train_y, _, _ = common.build_raw_sequence_dataset(train_tables, config.seq_len)
    train_input = common.transform_raw_sequences(train_raw_seq, common.LEGACY_FEATURE_SPEC)

    seq_scaler = common.FeatureScaler().fit(train_input.reshape(-1, train_input.shape[-1]))
    target_scaler = common.TargetScaler().fit(train_y)
    train_x_scaled = common.to_tensor(common.scale_sequences(train_input, seq_scaler))
    train_y_scaled = common.to_tensor(target_scaler.transform(train_y))
    raw_seq_t = common.to_tensor(train_raw_seq)

    val_x_scaled = None
    val_y_scaled = None
    if val_tables:
        val_raw_seq, val_y, _, _ = common.build_raw_sequence_dataset(val_tables, config.seq_len)
        val_input = common.transform_raw_sequences(val_raw_seq, common.LEGACY_FEATURE_SPEC)
        val_x_scaled = common.to_tensor(common.scale_sequences(val_input, seq_scaler))
        val_y_scaled = common.to_tensor(target_scaler.transform(val_y))

    model = common.TransformerNet(5, to_transformer_config(config)).to(common.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None
    if val_x_scaled is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=1e-7,
        )
    mse = nn.MSELoss()
    best_score = float("inf")
    best_state = None
    epochs_no_improve = 0
    actual_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        pred = model(train_x_scaled)
        data_loss = mse(pred, train_y_scaled)
        wear_pen, load_pen, clearance_pen = common.monotonic_penalty(model, raw_seq_t, seq_scaler, common.LEGACY_FEATURE_SPEC)
        loss = (
            data_loss
            + common.MONO_LAMBDA_WEAR * wear_pen
            + common.MONO_LAMBDA_LOAD * load_pen
            + common.MONO_LAMBDA_CLEARANCE * clearance_pen
        )

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        current_score = float(loss.item())
        if val_x_scaled is not None and val_y_scaled is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x_scaled)
                current_score = float(mse(val_pred, val_y_scaled).item())
            if scheduler is not None:
                scheduler.step(current_score)

        if current_score + (config.es_min_delta if val_x_scaled is not None else 0.0) < best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        actual_epochs = epoch
        if epoch == 1 or epoch % 300 == 0:
            print(
                f"    epoch {epoch:4d} | train_loss={float(loss.item()):.6e}"
                + (f" | val_loss={current_score:.6e}" if val_x_scaled is not None else "")
            )

        if val_x_scaled is not None and epochs_no_improve >= config.es_patience:
            print(f"    early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info: dict[str, float | str] = {
        "actual_epochs": float(actual_epochs),
        "best_score": float(best_score),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "validation_mode": "with_validation" if val_x_scaled is not None else "no_validation",
    }
    return model, seq_scaler, target_scaler, train_info


def run_round(
    config: RoundConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
) -> dict[str, float | str | bool]:
    detail_path = SCRIPT_DIR / f"详细结果_{config.round_id}.csv"
    summary_path = SCRIPT_DIR / f"汇总_{config.round_id}.csv"
    print()
    print("=" * 72)
    print(f"Running {config.round_id}: {config.description}")
    print(
        f"  seq_len={config.seq_len}, d_model={config.d_model}, layers={config.num_layers}, "
        f"ff={config.dim_ff}, dropout={config.dropout}, epochs={config.epochs}, "
        f"lr={config.learning_rate}, wd={config.weight_decay}, val={config.use_validation}"
    )
    print("=" * 72)

    if detail_path.exists():
        existing_df = pd.read_csv(detail_path)
        rows = existing_df.to_dict("records")
        completed_cases = {str(row["test_case"]) for _, row in existing_df.iterrows()}
        print(f"Resume enabled for {config.round_id}: loaded {len(existing_df)}/{len(test_df)} test cases")
        if summary_path.exists() and len(existing_df) >= len(test_df):
            return pd.read_csv(summary_path).iloc[0].to_dict()
    else:
        rows = []
        completed_cases: set[str] = set()

    val_tables = None
    val_file = ""
    if config.use_validation:
        val_file = pick_validation_case(train_df)
        train_files = [str(name) for name in train_df["file_name"] if str(name) != val_file]
        val_tables = {val_file: case_tables[val_file]}
    else:
        train_files = [str(name) for name in train_df["file_name"]]

    train_tables = {name: case_tables[name] for name in train_files}
    common.set_seed(_stable_round_seed(config.round_id))
    model, seq_scaler, target_scaler, train_info = train_model(config, train_tables, val_tables)

    for idx, test_row in test_df.iterrows():
        test_file = str(test_row["file_name"])
        test_source = str(test_row["source_file"])
        if test_source in completed_cases:
            print(f"[{idx + 1}/{len(test_df)}] test={test_source} | skip existing")
            continue

        print(f"[{idx + 1}/{len(test_df)}] test={test_source}")
        test_table = case_tables[test_file]
        test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, config.seq_len)
        true_curve_df = common.threshold_ground_truth(test_table, common.WEAR_THRESHOLD_UM, test_row)
        true_life_actual = float(test_row["actual_life"])

        pressure_metrics = common.evaluate_pressure(
            model,
            seq_scaler,
            target_scaler,
            test_raw_seq,
            test_y,
            common.LEGACY_FEATURE_SPEC,
        )
        rollout_df, predicted_life = common.rollout_case(
            model,
            seq_scaler,
            target_scaler,
            test_table,
            common.WEAR_THRESHOLD_UM,
            common.REAL_WEAR_COEFF_MPA_INV,
            true_life_actual,
            common.LEGACY_FEATURE_SPEC,
            config.seq_len,
        )
        curve_mae = common.wear_curve_mae(true_curve_df, rollout_df)

        rows.append(
            {
                "round_id": config.round_id,
                "test_case": test_source,
                "pressure_mae": pressure_metrics["pressure_mae"],
                "pressure_rmse": pressure_metrics["pressure_rmse"],
                "pressure_mape": pressure_metrics["pressure_mape"],
                "wear_mae_um": curve_mae,
                "predicted_life": predicted_life,
                "true_life": true_life_actual,
                "life_abs_error": abs(predicted_life - true_life_actual),
                "life_rel_error": abs(predicted_life - true_life_actual) / max(true_life_actual, common.EPS),
                "actual_epochs": train_info["actual_epochs"],
                "best_score": train_info["best_score"],
                "final_lr": train_info["final_lr"],
                "validation_mode": train_info["validation_mode"],
                "validation_file": val_file,
            }
        )
        completed_cases.add(test_source)
        pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    detail_df = pd.DataFrame(rows)
    summary_row = {
        "round_id": config.round_id,
        "description": config.description,
        "seq_len": config.seq_len,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_ff": config.dim_ff,
        "dropout": config.dropout,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip_norm": config.grad_clip_norm if config.grad_clip_norm is not None else float("nan"),
        "use_validation": config.use_validation,
        "mean_pressure_mae": float(detail_df["pressure_mae"].mean()),
        "mean_pressure_rmse": float(detail_df["pressure_rmse"].mean()),
        "mean_pressure_mape": float(detail_df["pressure_mape"].mean()),
        "mean_wear_mae_um": float(detail_df["wear_mae_um"].mean()),
        "mean_life_abs_error": float(detail_df["life_abs_error"].mean()),
        "median_life_abs_error": float(detail_df["life_abs_error"].median()),
        "mean_life_rel_error": float(detail_df["life_rel_error"].mean()),
        "max_life_abs_error": float(detail_df["life_abs_error"].max()),
        "min_life_abs_error": float(detail_df["life_abs_error"].min()),
        "mean_actual_epochs": float(detail_df["actual_epochs"].mean()),
    }
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(
        f"Completed {config.round_id}: life_abs_error={summary_row['mean_life_abs_error']:.0f}, "
        f"pressure_MAE={summary_row['mean_pressure_mae']:.2f}, wear_MAE={summary_row['mean_wear_mae_um']:.3f}"
    )
    return summary_row


def write_notes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    baseline_summary_path = BASELINE_RESULT_DIR / "汇总_各模型平均指标.csv"
    lines = [
        "30测试集 Transformer 候选搜索说明",
        "=" * 72,
        f"Data source: {common.DATA_DIR}",
        f"Split source: {common.SPLIT_CSV}",
        f"Train runs: {', '.join(train_df['source_file'].astype(str).tolist())}",
        f"Test runs: {', '.join(test_df['source_file'].astype(str).tolist())}",
        "Candidate count: " + str(len(ROUND_CONFIGS)),
        "公平性口径: 固定测试集不变，训练集 26 run，候选统一使用原始 5 维输入和原始 rollout 步长；本轮只比较 Transformer 参数。",
        "候选覆盖: seq_len=6/8/10/12/15/20，d_model=32/48/64/96，2/3层，ff=64/96/128/192/256，温和正则、低学习率和 validation 分支。",
        "",
        "CUDA:",
    ]
    lines.extend([f"  {line}" for line in common.device_summary_lines()])
    if baseline_summary_path.exists():
        baseline_summary = pd.read_csv(baseline_summary_path)
        transformer_row = baseline_summary[baseline_summary["model"] == "Transformer"]
        if not transformer_row.empty:
            row = transformer_row.iloc[0]
            lines.extend(
                [
                    "",
                    "30测试集 baseline Transformer:",
                    f"  pressure_MAE={float(row['mean_pressure_mae']):.2f}",
                    f"  wear_MAE={float(row['mean_wear_mae_um']):.3f}",
                    f"  life_abs_error={float(row['mean_life_abs_error']):.0f}",
                ]
            )
    (SCRIPT_DIR / NOTES_FILENAME).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    common.require_cuda()
    common.set_seed(common.SEED)
    for line in common.device_summary_lines():
        print(line)

    summary_df, case_tables = common.load_data()
    train_df, test_df = common.load_fixed_split(summary_df)
    print(f"Train runs: {len(train_df)} | Test runs: {len(test_df)}")
    write_notes(train_df, test_df)

    summary_rows = [run_round(config, train_df, test_df, case_tables) for config in ROUND_CONFIGS]
    all_summary_df = pd.DataFrame(summary_rows).sort_values("mean_life_abs_error").reset_index(drop=True)
    all_summary_df.to_csv(SCRIPT_DIR / DETAIL_ALL_FILENAME, index=False, encoding="utf-8-sig")

    best_row = all_summary_df.iloc[0].to_dict()
    best_config = next(config for config in ROUND_CONFIGS if config.round_id == str(best_row["round_id"]))
    (SCRIPT_DIR / BEST_CONFIG_FILENAME).write_text(
        json.dumps(asdict(best_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("Candidate summary")
    print("=" * 72)
    print(all_summary_df[["round_id", "mean_pressure_mae", "mean_wear_mae_um", "mean_life_abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
