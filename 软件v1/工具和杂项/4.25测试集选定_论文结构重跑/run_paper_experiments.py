from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
RESULT_ROOT = ROOT_DIR / "结果" / "4.25测试集选定_论文结构重跑"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common_fixed_split as common


TEST_FILES = ["Run4.csv", "Run28.csv", "Run25.csv", "Run11.csv"]
SEQ_LEN = 12
STEP_CAP = 250.0

SCREEN_SEEDS = [20260425, 20260426]
REFINE_SEEDS = [20260425, 20260426, 20260427]
FINAL_SEEDS = [20260425, 20260426, 20260427, 20260428, 20260429]

SCREEN_EPOCHS = 800
REFINE_EPOCHS = 1200
FINAL_EPOCHS = 1600

MODEL_ORDER = ["FNN", "GRU", "LSTM", "1D-CNN", "Transformer"]
DETAIL_NAME = "详细结果_各模型各测试集.csv"
SUMMARY_NAME = "汇总_各模型平均指标.csv"
PLAN_NAME = "实验方案.txt"
NOTES_NAME = "测试说明.txt"
ANALYSIS_NAME = "结果说明与分析.txt"

M1_LOG_KEEP_SPEC = common.FeatureSpec(
    name="M1_R3_log_cycle_keep",
    columns=(
        "F",
        "D",
        "Cr",
        "F_over_D_sq",
        "Cr_over_D",
        "actual_cycle",
        "log1p_actual_cycle",
        "wear_depth",
    ),
    description="静态派生 + 同时保留 actual_cycle 与 log1p(actual_cycle)",
)


@dataclass(frozen=True)
class RunSpec:
    variant: str
    model_name: str
    feature_spec: common.FeatureSpec
    shape_config: common.ShapeLossConfig | None = None
    feature_variant: str = ""
    shape_variant: str = "S0_no_shape_loss"
    description: str = ""


def make_config(epochs: int) -> common.TransformerConfig:
    return common.TransformerConfig(
        seq_len=SEQ_LEN,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_ff=64,
        dropout=0.0,
        epochs=epochs,
        learning_rate=1e-3,
        weight_decay=1e-6,
    )


def mode_settings(dry_run: bool) -> dict[str, object]:
    if dry_run:
        return {
            "result_root": RESULT_ROOT / "__dry_run",
            "screen_seeds": [SCREEN_SEEDS[0]],
            "refine_seeds": [REFINE_SEEDS[0]],
            "final_seeds": [FINAL_SEEDS[0]],
            "screen_epochs": 30,
            "refine_epochs": 30,
            "final_epochs": 30,
        }
    return {
        "result_root": RESULT_ROOT,
        "screen_seeds": SCREEN_SEEDS,
        "refine_seeds": REFINE_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "screen_epochs": SCREEN_EPOCHS,
        "refine_epochs": REFINE_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
    }


def read_detail(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_detail(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def shape_payload(shape_config: common.ShapeLossConfig | None) -> dict[str, object]:
    if shape_config is None:
        return {
            "shape_name": "none",
            "shape_mono_lambda": 0.0,
            "shape_slow_lambda": 0.0,
            "shape_mono_tolerance_ratio": 0.0,
        }
    return {
        "shape_name": shape_config.name,
        "shape_mono_lambda": float(shape_config.mono_lambda),
        "shape_slow_lambda": float(shape_config.slow_lambda),
        "shape_mono_tolerance_ratio": float(shape_config.mono_tolerance_ratio),
    }


def load_formal_split() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    data_summary, case_tables = common.load_data()
    eligible = common.eligible_cases(data_summary)
    test_set = set(TEST_FILES)
    test_df = eligible.loc[eligible["file_name"].isin(test_set)].copy()
    train_df = eligible.loc[~eligible["file_name"].isin(test_set)].copy()
    missing = sorted(test_set - set(test_df["file_name"].astype(str)))
    if missing:
        raise RuntimeError(f"Missing formal test files in eligible data: {missing}")
    test_df["split_role"] = "test"
    train_df["split_role"] = "train"
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), case_tables


def train_tables_from_split(train_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {str(row.file_name): case_tables[str(row.file_name)] for row in train_df.itertuples(index=False)}


def completed_for(detail_df: pd.DataFrame, variant: str, seed: int, test_cases: list[str]) -> bool:
    if detail_df.empty:
        return False
    part = detail_df.loc[
        detail_df["variant"].astype(str).eq(variant)
        & detail_df["seed"].astype(int).eq(int(seed))
    ]
    return set(part["test_case"].astype(str).tolist()) >= set(test_cases)


def evaluate_spec(
    *,
    spec: RunSpec,
    seed: int,
    epochs: int,
    stage: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
) -> list[dict]:
    common.set_seed(seed)
    train_tables = train_tables_from_split(train_df, case_tables)
    train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)
    model, seq_scaler, target_scaler = common.train_model(
        model_name=spec.model_name,
        train_raw_seq=train_raw_seq,
        train_y=train_y,
        feature_spec=spec.feature_spec,
        config=make_config(epochs),
        shape_config=spec.shape_config,
        train_case_names=train_case_names,
        train_step_indices=train_step_indices,
    )

    rows: list[dict] = []
    for test_row in test_df.itertuples(index=False):
        test_file = str(test_row.file_name)
        test_table = case_tables[test_file]
        test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, SEQ_LEN)
        pressure_metrics = common.evaluate_pressure(
            model,
            seq_scaler,
            target_scaler,
            test_raw_seq,
            test_y,
            spec.feature_spec,
        )
        true_curve_df = common.threshold_ground_truth(test_table, common.WEAR_THRESHOLD_UM, pd.Series(test_row._asdict()))
        true_life = float(test_row.actual_life)
        rollout_df, predicted_life, step_info = common.rollout_case_with_step_cap(
            model,
            seq_scaler,
            target_scaler,
            test_table,
            common.WEAR_THRESHOLD_UM,
            common.REAL_WEAR_COEFF_MPA_INV,
            true_life,
            spec.feature_spec,
            SEQ_LEN,
            actual_step_cap=STEP_CAP,
        )
        rows.append(
            {
                "stage": stage,
                "model": spec.model_name,
                "variant": spec.variant,
                "feature_variant": spec.feature_variant or spec.variant,
                "shape_variant": spec.shape_variant,
                "seed": int(seed),
                "epochs": int(epochs),
                "test_case": str(test_row.source_file),
                "file_name": test_file,
                "pressure_mae": pressure_metrics["pressure_mae"],
                "pressure_rmse": pressure_metrics["pressure_rmse"],
                "pressure_mape": pressure_metrics["pressure_mape"],
                "wear_mae_um": common.wear_curve_mae(true_curve_df, rollout_df),
                "predicted_life": predicted_life,
                "true_life": true_life,
                "life_abs_error": abs(predicted_life - true_life),
                "life_rel_error": abs(predicted_life - true_life) / max(true_life, common.EPS),
                "native_actual_step": step_info.native_actual_step,
                "used_actual_step": step_info.used_actual_step,
                "feature_name": spec.feature_spec.name,
                "feature_columns": "|".join(spec.feature_spec.columns),
                "description": spec.description,
                **shape_payload(spec.shape_config),
            }
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_specs(
    *,
    out_dir: Path,
    stage: str,
    specs: list[RunSpec],
    seeds: list[int],
    epochs: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / DETAIL_NAME
    detail_df = read_detail(detail_path)
    rows = detail_df.to_dict("records") if not detail_df.empty else []
    test_cases = test_df["source_file"].astype(str).tolist()

    for spec in specs:
        for seed in seeds:
            detail_df = pd.DataFrame(rows)
            if completed_for(detail_df, spec.variant, seed, test_cases):
                print(f"skip completed | {stage} | {spec.variant} | seed={seed}")
                continue
            print(f"train/eval | {stage} | {spec.variant} | seed={seed} | epochs={epochs}")
            new_rows = evaluate_spec(
                spec=spec,
                seed=seed,
                epochs=epochs,
                stage=stage,
                train_df=train_df,
                test_df=test_df,
                case_tables=case_tables,
            )
            rows.extend(new_rows)
            write_detail(detail_path, rows)

    detail_df = pd.DataFrame(rows)
    save_stage_outputs(out_dir, detail_df, stage)
    return detail_df


def summarize_detail(detail_df: pd.DataFrame, baseline_variant: str | None = None) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    seed_level = (
        detail_df.groupby(["variant", "seed"], as_index=False)
        .agg(
            pressure_mae=("pressure_mae", "mean"),
            pressure_rmse=("pressure_rmse", "mean"),
            pressure_mape=("pressure_mape", "mean"),
            wear_mae_um=("wear_mae_um", "mean"),
            life_abs_error=("life_abs_error", "mean"),
            life_rel_error=("life_rel_error", "mean"),
        )
    )
    summary = (
        seed_level.groupby("variant")
        .agg(
            seed_count=("seed", "nunique"),
            mean_pressure_mae=("pressure_mae", "mean"),
            std_pressure_mae=("pressure_mae", "std"),
            mean_pressure_rmse=("pressure_rmse", "mean"),
            mean_pressure_mape=("pressure_mape", "mean"),
            mean_wear_mae_um=("wear_mae_um", "mean"),
            std_wear_mae_um=("wear_mae_um", "std"),
            mean_life_abs_error=("life_abs_error", "mean"),
            std_life_abs_error=("life_abs_error", "std"),
            mean_life_rel_error=("life_rel_error", "mean"),
        )
        .reset_index()
    )
    extrema = (
        detail_df.groupby("variant")
        .agg(
            median_life_abs_error=("life_abs_error", "median"),
            max_life_abs_error=("life_abs_error", "max"),
            min_life_abs_error=("life_abs_error", "min"),
            row_count=("life_abs_error", "count"),
        )
        .reset_index()
    )
    wins = (
        detail_df.loc[detail_df.groupby(["seed", "test_case"])["life_abs_error"].idxmin()]
        .groupby("variant")
        .size()
        .rename("life_error_win_count")
        .reset_index()
    )
    summary = summary.merge(extrema, on="variant", how="left").merge(wins, on="variant", how="left")
    summary["life_error_win_count"] = summary["life_error_win_count"].fillna(0).astype(int)
    summary["std_life_abs_error"] = summary["std_life_abs_error"].fillna(0.0)
    summary["std_pressure_mae"] = summary["std_pressure_mae"].fillna(0.0)
    summary["std_wear_mae_um"] = summary["std_wear_mae_um"].fillna(0.0)
    if baseline_variant and baseline_variant in set(summary["variant"].astype(str)):
        baseline_error = float(summary.loc[summary["variant"].eq(baseline_variant), "mean_life_abs_error"].iloc[0])
        summary["delta_vs_baseline"] = summary["mean_life_abs_error"] - baseline_error
    else:
        summary["delta_vs_baseline"] = float("nan")
    summary["model"] = summary["variant"]
    return summary.sort_values("mean_life_abs_error").reset_index(drop=True)


def chart_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = detail_df.select_dtypes(include="number").columns.tolist()
    plot_df = detail_df.groupby(["test_case", "variant"], as_index=False)[numeric_cols].mean()
    return plot_df.rename(columns={"variant": "model"})


def save_stage_outputs(out_dir: Path, detail_df: pd.DataFrame, stage: str, baseline_variant: str | None = None) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    if baseline_variant is None:
        baseline_variant = str(detail_df["variant"].iloc[0])
    summary = summarize_detail(detail_df, baseline_variant)
    summary.to_csv(out_dir / SUMMARY_NAME, index=False, encoding="utf-8-sig")
    plot_df = chart_detail(detail_df)
    common.save_grouped_life_error_bar(plot_df, out_dir / "图1_各工况寿命误差.png", f"{stage}: 各工况寿命误差")
    common.save_summary_bar_chart(summary, out_dir / "图2_三指标柱状图.png", f"{stage}: 平均指标")
    common.save_mean_predicted_life_chart(summary, plot_df, out_dir / "图3_平均预测寿命.png", f"{stage}: 平均预测寿命")
    write_stage_docs(out_dir, stage, summary)
    return summary


def write_stage_docs(out_dir: Path, stage: str, summary: pd.DataFrame) -> None:
    ranked = summary.sort_values("mean_life_abs_error").reset_index(drop=True)
    lines = [
        "实验方案",
        "=" * 72,
        f"阶段: {stage}",
        f"seq_len={SEQ_LEN}, actual_step cap={STEP_CAP:g}, wear_coeff={common.REAL_WEAR_COEFF_MPA_INV:g}",
        "统一基础物理敏感性正则项保留在所有训练中；模块二只表示额外 temporal shape loss。",
    ]
    (out_dir / PLAN_NAME).write_text("\n".join(lines), encoding="utf-8")

    notes = [
        "测试说明",
        "=" * 72,
        "正式测试集: Run4 / Run28 / Run25 / Run11。",
        "训练集: 其余 eligible runs。",
        "CUDA/设备信息:",
    ] + ["  " + line for line in common.device_summary_lines()]
    (out_dir / NOTES_NAME).write_text("\n".join(notes), encoding="utf-8")

    analysis = [
        "结果说明与分析",
        "=" * 72,
        "排序:",
    ]
    for idx, row in enumerate(ranked.itertuples(index=False), start=1):
        analysis.append(
            f"{idx}. {row.variant}: mean_life_abs_error={float(row.mean_life_abs_error):.1f}, "
            f"std={float(row.std_life_abs_error):.1f}, max={float(row.max_life_abs_error):.1f}, "
            f"delta_vs_baseline={float(row.delta_vs_baseline):.1f}"
        )
    (out_dir / ANALYSIS_NAME).write_text("\n".join(analysis), encoding="utf-8")


def write_config_snapshot(result_root: Path, train_df: pd.DataFrame, test_df: pd.DataFrame, dry_run: bool) -> None:
    out_dir = result_root / "00_实验配置快照"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_df = pd.concat([train_df, test_df], ignore_index=True)
    split_df[["file_name", "source_file", "split_role", "actual_life", "final_wear_um"]].to_csv(
        out_dir / "正式测试集划分.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "dry_run": dry_run,
        "test_files": TEST_FILES,
        "seq_len": SEQ_LEN,
        "actual_step_cap": STEP_CAP,
        "wear_coeff": common.REAL_WEAR_COEFF_MPA_INV,
        "screen_seeds": SCREEN_SEEDS,
        "refine_seeds": REFINE_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "screen_epochs": SCREEN_EPOCHS,
        "refine_epochs": REFINE_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "device": common.device_summary_lines(),
    }
    (out_dir / "实验配置.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "总计划.txt").write_text(
        "\n".join(
            [
                "原数据集论文结构重跑计划",
                "=" * 72,
                "01_五模型基准对比: 五模型原始 5 维正式基准。",
                "02_模块一_物理派生特征比较: baseline/R1/R2/R3_keep，保留 top2。",
                "03_模块二_趋势约束比较: top2 模块一 × slow/temporal_mono+slow。",
                "04_最终消融实验: T0/T1/T2/T3 四格。",
                "05_最终横向对比: 五模型基准 + Enhanced Transformer。",
            ]
        ),
        encoding="utf-8",
    )


def module1_specs() -> list[RunSpec]:
    return [
        RunSpec("M1_0_5d_baseline", "Transformer", common.LEGACY_FEATURE_SPEC, feature_variant="M1_0_5d_baseline", description="原始 5 维"),
        RunSpec("M1_R1_static_ratio", "Transformer", common.FEATURE_R1_SPEC, feature_variant="M1_R1_static_ratio", description="静态派生 F/D^2 与 Cr/D"),
        RunSpec("M1_R2_log_cycle_replace", "Transformer", common.FEATURE_R2_SPEC, feature_variant="M1_R2_log_cycle_replace", description="用 log1p(cycle) 替换 actual_cycle"),
        RunSpec("M1_R3_log_cycle_keep", "Transformer", M1_LOG_KEEP_SPEC, feature_variant="M1_R3_log_cycle_keep", description="同时保留 actual_cycle 与 log1p(cycle)"),
    ]


def feature_spec_by_name(name: str) -> common.FeatureSpec:
    specs = {spec.variant: spec.feature_spec for spec in module1_specs()}
    if name not in specs:
        raise KeyError(f"Unknown module1 feature variant: {name}")
    return specs[name]


def shape_configs() -> list[tuple[str, common.ShapeLossConfig]]:
    out: list[tuple[str, common.ShapeLossConfig]] = []
    for slow in [0.003, 0.01, 0.03]:
        key = str(slow).replace(".", "p")
        out.append(
            (
                f"S1_slow_abs_{key}",
                common.ShapeLossConfig(name=f"slow_abs_{key}", mono_lambda=0.0, slow_lambda=slow, mono_tolerance_ratio=0.0),
            )
        )
    for slow in [0.003, 0.01, 0.03]:
        key = str(slow).replace(".", "p")
        out.append(
            (
                f"S2_temporal_mono_plus_slow_abs_{key}",
                common.ShapeLossConfig(name=f"temporal_mono_plus_slow_abs_{key}", mono_lambda=0.005, slow_lambda=slow, mono_tolerance_ratio=0.0),
            )
        )
    return out


def baseline_specs() -> list[RunSpec]:
    return [
        RunSpec(model, model, common.LEGACY_FEATURE_SPEC, feature_variant="M1_0_5d_baseline", description="五模型原始 5 维基准")
        for model in MODEL_ORDER
    ]


def top_variants(summary_path: Path, n: int) -> list[str]:
    summary = pd.read_csv(summary_path)
    return summary.sort_values("mean_life_abs_error")["variant"].astype(str).head(n).tolist()


def run_baseline(settings: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return run_specs(
        out_dir=Path(settings["result_root"]) / "01_五模型基准对比",
        stage="01_五模型基准对比",
        specs=baseline_specs(),
        seeds=list(settings["final_seeds"]),
        epochs=int(settings["final_epochs"]),
        train_df=train_df,
        test_df=test_df,
        case_tables=case_tables,
    )


def run_module1(settings: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> list[str]:
    root = Path(settings["result_root"]) / "02_模块一_物理派生特征比较"
    coarse_dir = root / "粗筛"
    refine_dir = root / "复筛"
    run_specs(
        out_dir=coarse_dir,
        stage="02_模块一_粗筛",
        specs=module1_specs(),
        seeds=list(settings["screen_seeds"]),
        epochs=int(settings["screen_epochs"]),
        train_df=train_df,
        test_df=test_df,
        case_tables=case_tables,
    )
    top2 = top_variants(coarse_dir / SUMMARY_NAME, 2)
    refine_specs = [spec for spec in module1_specs() if spec.variant in set(top2)]
    run_specs(
        out_dir=refine_dir,
        stage="02_模块一_复筛",
        specs=refine_specs,
        seeds=list(settings["refine_seeds"]),
        epochs=int(settings["refine_epochs"]),
        train_df=train_df,
        test_df=test_df,
        case_tables=case_tables,
    )
    final_top2 = top_variants(refine_dir / SUMMARY_NAME, min(2, len(refine_specs)))
    (root / "模块一_top2.json").write_text(json.dumps(final_top2, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_top2


def copy_s0_rows(source_detail: pd.DataFrame, feature_names: list[str], stage: str) -> pd.DataFrame:
    rows = []
    for feature_name in feature_names:
        part = source_detail.loc[source_detail["variant"].astype(str).eq(feature_name)].copy()
        if part.empty:
            continue
        new_variant = f"{feature_name}__S0_no_shape_loss"
        part["stage"] = stage
        part["variant"] = new_variant
        part["feature_variant"] = feature_name
        part["shape_variant"] = "S0_no_shape_loss"
        part["shape_name"] = "none"
        part["shape_mono_lambda"] = 0.0
        part["shape_slow_lambda"] = 0.0
        part["shape_mono_tolerance_ratio"] = 0.0
        rows.extend(part.to_dict("records"))
    return pd.DataFrame(rows)


def module2_specs(feature_names: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for feature_name in feature_names:
        feature_spec = feature_spec_by_name(feature_name)
        for shape_name, shape_config in shape_configs():
            specs.append(
                RunSpec(
                    variant=f"{feature_name}__{shape_name}",
                    model_name="Transformer",
                    feature_spec=feature_spec,
                    shape_config=shape_config,
                    feature_variant=feature_name,
                    shape_variant=shape_name,
                    description=f"{feature_name} + {shape_name}",
                )
            )
    return specs


def merge_s0_with_detail(out_dir: Path, s0_df: pd.DataFrame, stage: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / DETAIL_NAME
    trained = read_detail(detail_path)
    if trained.empty:
        merged = s0_df.copy()
    else:
        no_s0_trained = trained.loc[~trained["shape_variant"].astype(str).eq("S0_no_shape_loss")].copy()
        merged = pd.concat([s0_df, no_s0_trained], ignore_index=True)
    if not merged.empty:
        merged["stage"] = stage
        merged.to_csv(detail_path, index=False, encoding="utf-8-sig")
    return merged


def run_module2(settings: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> tuple[str, str]:
    result_root = Path(settings["result_root"])
    module1_root = result_root / "02_模块一_物理派生特征比较"
    top2 = json.loads((module1_root / "模块一_top2.json").read_text(encoding="utf-8"))

    root = result_root / "03_模块二_趋势约束比较"
    coarse_dir = root / "粗筛"
    refine_dir = root / "复筛"

    coarse_s0 = copy_s0_rows(read_detail(module1_root / "粗筛" / DETAIL_NAME), top2, "03_模块二_粗筛")
    merge_s0_with_detail(coarse_dir, coarse_s0, "03_模块二_粗筛")
    run_specs(
        out_dir=coarse_dir,
        stage="03_模块二_粗筛",
        specs=module2_specs(top2),
        seeds=list(settings["screen_seeds"]),
        epochs=int(settings["screen_epochs"]),
        train_df=train_df,
        test_df=test_df,
        case_tables=case_tables,
    )
    coarse_summary = pd.read_csv(coarse_dir / SUMMARY_NAME)
    top_combos = coarse_summary.sort_values("mean_life_abs_error")["variant"].astype(str).head(3).tolist()

    refine_s0 = copy_s0_rows(read_detail(module1_root / "复筛" / DETAIL_NAME), top2, "03_模块二_复筛")
    refine_s0 = refine_s0.loc[refine_s0["variant"].astype(str).isin(set(top_combos))].copy()
    merge_s0_with_detail(refine_dir, refine_s0, "03_模块二_复筛")

    all_specs = {spec.variant: spec for spec in module2_specs(top2)}
    refine_specs = [all_specs[name] for name in top_combos if name in all_specs]
    if refine_specs:
        run_specs(
            out_dir=refine_dir,
            stage="03_模块二_复筛",
            specs=refine_specs,
            seeds=list(settings["refine_seeds"]),
            epochs=int(settings["refine_epochs"]),
            train_df=train_df,
            test_df=test_df,
            case_tables=case_tables,
        )
    else:
        save_stage_outputs(refine_dir, read_detail(refine_dir / DETAIL_NAME), "03_模块二_复筛")

    final_summary = pd.read_csv(refine_dir / SUMMARY_NAME)
    best_variant = str(final_summary.sort_values("mean_life_abs_error").iloc[0]["variant"])
    best_detail = read_detail(refine_dir / DETAIL_NAME)
    best_rows = best_detail.loc[best_detail["variant"].astype(str).eq(best_variant)]
    best_feature = str(best_rows["feature_variant"].iloc[0])
    best_shape = str(best_rows["shape_variant"].iloc[0])
    payload = {"best_variant": best_variant, "best_feature": best_feature, "best_shape": best_shape}
    (root / "模块二最佳组合.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_feature, best_shape


def shape_config_by_name(shape_name: str) -> common.ShapeLossConfig | None:
    if shape_name == "S0_no_shape_loss":
        return None
    for name, cfg in shape_configs():
        if name == shape_name:
            return cfg
    raise KeyError(f"Unknown shape config: {shape_name}")


def run_ablation(settings: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    root = Path(settings["result_root"])
    payload = json.loads((root / "03_模块二_趋势约束比较" / "模块二最佳组合.json").read_text(encoding="utf-8"))
    best_feature = str(payload["best_feature"])
    best_shape = str(payload["best_shape"])
    feature_spec = feature_spec_by_name(best_feature)
    shape_config = shape_config_by_name(best_shape)
    specs = [
        RunSpec("T0_baseline", "Transformer", common.LEGACY_FEATURE_SPEC, feature_variant="M1_0_5d_baseline", shape_variant="S0_no_shape_loss", description="原始 5 维 + 无趋势约束"),
        RunSpec("T1_module1_only", "Transformer", feature_spec, feature_variant=best_feature, shape_variant="S0_no_shape_loss", description=f"{best_feature} + 无趋势约束"),
        RunSpec("T2_module2_only", "Transformer", common.LEGACY_FEATURE_SPEC, shape_config=shape_config, feature_variant="M1_0_5d_baseline", shape_variant=best_shape, description=f"原始 5 维 + {best_shape}"),
        RunSpec("T3_module1_plus_module2", "Transformer", feature_spec, shape_config=shape_config, feature_variant=best_feature, shape_variant=best_shape, description=f"{best_feature} + {best_shape}"),
    ]
    return run_specs(
        out_dir=root / "04_最终消融实验",
        stage="04_最终消融实验",
        specs=specs,
        seeds=list(settings["final_seeds"]),
        epochs=int(settings["final_epochs"]),
        train_df=train_df,
        test_df=test_df,
        case_tables=case_tables,
    )


def maybe_extend_to_10_seeds(result_root: Path) -> dict[str, object]:
    baseline_summary = pd.read_csv(result_root / "01_五模型基准对比" / SUMMARY_NAME)
    ablation_summary = pd.read_csv(result_root / "04_最终消融实验" / SUMMARY_NAME)
    best_baseline = baseline_summary.sort_values("mean_life_abs_error").iloc[0]
    enhanced = ablation_summary.loc[ablation_summary["variant"].eq("T3_module1_plus_module2")].iloc[0]
    advantage = float(best_baseline["mean_life_abs_error"]) - float(enhanced["mean_life_abs_error"])
    pooled_std = math.sqrt((float(best_baseline["std_life_abs_error"]) ** 2 + float(enhanced["std_life_abs_error"]) ** 2) / 2.0)
    return {
        "best_baseline_variant": str(best_baseline["variant"]),
        "enhanced_variant": "T3_module1_plus_module2",
        "advantage": advantage,
        "pooled_std": pooled_std,
        "threshold": 0.5 * pooled_std,
        "needs_10_seed_extension": bool(advantage < 0.5 * pooled_std),
    }


def run_final_compare(settings: dict[str, object]) -> None:
    result_root = Path(settings["result_root"])
    out_dir = result_root / "05_最终横向对比"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_detail = read_detail(result_root / "01_五模型基准对比" / DETAIL_NAME)
    ablation_detail = read_detail(result_root / "04_最终消融实验" / DETAIL_NAME)
    enhanced = ablation_detail.loc[ablation_detail["variant"].eq("T3_module1_plus_module2")].copy()
    enhanced["variant"] = "Enhanced Transformer"
    enhanced["model"] = "Enhanced Transformer"
    base = baseline_detail.copy()
    base["variant"] = base["model"]
    detail = pd.concat([base, enhanced], ignore_index=True)
    detail.to_csv(out_dir / DETAIL_NAME, index=False, encoding="utf-8-sig")
    save_stage_outputs(out_dir, detail, "05_最终横向对比")
    extension = maybe_extend_to_10_seeds(result_root)
    (out_dir / "追加seed判断.json").write_text(json.dumps(extension, ensure_ascii=False, indent=2), encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    if not (args.dry_run and args.allow_cpu_dry_run):
        common.require_cuda()

    settings = mode_settings(args.dry_run)
    result_root = Path(settings["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    train_df, test_df, case_tables = load_formal_split()
    write_config_snapshot(result_root, train_df, test_df, args.dry_run)

    stages = ["baseline", "module1", "module2", "ablation", "final"] if args.stage == "all" else [args.stage]
    if "baseline" in stages:
        run_baseline(settings, train_df, test_df, case_tables)
    if "module1" in stages:
        run_module1(settings, train_df, test_df, case_tables)
    if "module2" in stages:
        if not (result_root / "02_模块一_物理派生特征比较" / "模块一_top2.json").exists():
            run_module1(settings, train_df, test_df, case_tables)
        run_module2(settings, train_df, test_df, case_tables)
    if "ablation" in stages:
        if not (result_root / "03_模块二_趋势约束比较" / "模块二最佳组合.json").exists():
            run_module2(settings, train_df, test_df, case_tables)
        run_ablation(settings, train_df, test_df, case_tables)
    if "final" in stages:
        if not (result_root / "01_五模型基准对比" / DETAIL_NAME).exists():
            run_baseline(settings, train_df, test_df, case_tables)
        if not (result_root / "04_最终消融实验" / DETAIL_NAME).exists():
            run_ablation(settings, train_df, test_df, case_tables)
        run_final_compare(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-structure experiments for the fixed 4.25 test set.")
    parser.add_argument("--stage", choices=["all", "baseline", "module1", "module2", "ablation", "final"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Run tiny 1-seed experiments into __dry_run.")
    parser.add_argument(
        "--allow-cpu-dry-run",
        action="store_true",
        help="Only for script wiring checks: allow --dry-run without CUDA. Official experiments always require CUDA.",
    )
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
