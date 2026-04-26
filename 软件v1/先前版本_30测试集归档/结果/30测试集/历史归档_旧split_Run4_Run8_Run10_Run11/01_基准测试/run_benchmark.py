from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
COMMON_DIR = ROOT_DIR / "工具和杂项" / "30测试集_中间过程"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import common_fixed_split as common


EXPERIMENT_NAME = "30测试集 基准测试"
EXPERIMENT_DATE = "2026-04-24"

SEQ_LEN = 6
BASELINE_CONFIG = common.TransformerConfig(
    seq_len=SEQ_LEN,
    d_model=32,
    nhead=4,
    num_layers=2,
    dim_ff=64,
    dropout=0.0,
    epochs=1200,
    learning_rate=1e-3,
    weight_decay=1e-6,
)

MODEL_ORDER = ["FNN", "GRU", "LSTM", "1D-CNN", "Transformer"]

DETAIL_FILENAME = "详细结果_各模型各测试集.csv"
SUMMARY_FILENAME = "汇总_各模型平均指标.csv"
CHART1_FILENAME = "图1_各工况寿命误差分组柱状图.png"
CHART2_FILENAME = "图2_模型对比三指标柱状图.png"
CHART3_FILENAME = "图3_平均预测寿命柱状图.png"
NOTES_FILENAME = "测试说明.txt"
ANALYSIS_FILENAME = "结果说明与分析.txt"


def build_notes(train_df: pd.DataFrame, test_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    lines = [
        "测试说明",
        "=" * 60,
        f"实验名称: {EXPERIMENT_NAME}",
        f"测试日期: {EXPERIMENT_DATE}",
        f"数据源: {common.DATA_DIR}",
        f"测试集清单: {common.SPLIT_CSV}",
        f"训练集数量: {len(train_df)}",
        f"测试集数量: {len(test_df)}",
        f"测试集 run: {', '.join(test_df['source_file'].astype(str).tolist())}",
        f"序列长度: {SEQ_LEN}",
        f"训练轮次: {BASELINE_CONFIG.epochs}",
        "",
        "实验目的:",
        "  在完整 30 组数据、固定测试集 Run4 / Run8 / Run10 / Run11 上建立五模型共同基线；测试集沿用旧主线，训练集由 13 个 run 扩展为 26 个 run，用于公平观察完整数据集带来的效果变化。",
        "",
        "实验方案:",
        "  所有模型统一使用 26 个训练 run 训练一次，再在固定的 4 个测试 run 上统一评估；模型结构、序列长度、训练轮次、闭环 rollout 口径保持与 17测试集基准一致。",
        "",
        "CUDA:",
    ]
    lines.extend([f"  {line}" for line in common.device_summary_lines()])
    lines.extend(
        [
            "",
            "输出文件:",
            f"  {DETAIL_FILENAME}",
            f"  {SUMMARY_FILENAME}",
            f"  {CHART1_FILENAME}",
            f"  {CHART2_FILENAME}",
            f"  {CHART3_FILENAME}",
            f"  {NOTES_FILENAME}",
            f"  {ANALYSIS_FILENAME}",
            "",
            "模型排序（按寿命绝对误差）:",
        ]
    )
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"  {row.model}: life_abs_error={float(row.mean_life_abs_error):.0f}, "
            f"pressure_MAE={float(row.mean_pressure_mae):.2f}, wear_MAE={float(row.mean_wear_mae_um):.3f}"
        )
    return "\n".join(lines)


def build_analysis(summary_df: pd.DataFrame, detail_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    ranked = summary_df.sort_values("mean_life_abs_error").reset_index(drop=True)
    best = ranked.iloc[0]
    worst_case = detail_df.sort_values("life_abs_error", ascending=False).iloc[0]
    lines = [
        "结果说明与分析",
        "=" * 72,
        "",
        "一、实验设置",
        "-" * 72,
        f"本轮固定测试集共 4 个 run: {', '.join(test_df['source_file'].astype(str).tolist())}",
        f"训练集共 {len(train_df)} 个 run，测试集共 {len(test_df)} 个 run。",
        "公平性说明：测试集仍固定为 Run4 / Run8 / Run10 / Run11，模型结构、seq_len、训练轮次、闭环 rollout 口径均沿用 17测试集基准；本轮只改变训练数据规模。",
        "",
        "二、总体排序",
        "-" * 72,
    ]
    for idx, row in enumerate(ranked.itertuples(index=False), start=1):
        lines.append(
            f"{idx}. {row.model} | pressure_MAE={float(row.mean_pressure_mae):.2f} | "
            f"wear_MAE={float(row.mean_wear_mae_um):.3f} | life_abs_error={float(row.mean_life_abs_error):.0f}"
        )
    lines.extend(
        [
            "",
            "三、关键观察",
            "-" * 72,
            f"1. 本轮固定测试集最佳模型为 {best['model']}，平均寿命绝对误差约 {float(best['mean_life_abs_error']):.0f}。",
            f"2. 单个误差最大的测试工况为 {worst_case['test_case']} / {worst_case['model']}，寿命绝对误差约 {float(worst_case['life_abs_error']):.0f}。",
            "3. 这组结果是完整 30 组数据下后续 Transformer 参数升级、Shape Loss、特征构造和组合实验的统一对照基线。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    common.require_cuda()
    common.set_seed(common.SEED)
    for line in common.device_summary_lines():
        print(line)
    print(f"{EXPERIMENT_NAME}")
    print(f"  seq_len={SEQ_LEN}, epochs={BASELINE_CONFIG.epochs}")
    print(f"  split={common.SPLIT_CSV}")
    print()

    summary_df, case_tables = common.load_data()
    train_df, test_df = common.load_fixed_split(summary_df)
    print(f"Train runs: {len(train_df)}")
    for _, row in train_df.iterrows():
        print(f"  train | {row['source_file']:20s} | life={float(row['actual_life']):10.1f}")
    print(f"Test runs: {len(test_df)}")
    for _, row in test_df.iterrows():
        print(f"  test  | {row['source_file']:20s} | life={float(row['actual_life']):10.1f}")
    print()

    train_tables = {str(row["file_name"]): case_tables[str(row["file_name"])] for _, row in train_df.iterrows()}
    train_raw_seq, train_y, _, _ = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)

    detail_path = SCRIPT_DIR / DETAIL_FILENAME
    if detail_path.exists():
        existing_df = pd.read_csv(detail_path)
        scan_rows = existing_df.to_dict("records")
        completed_pairs = {(str(row["test_case"]), str(row["model"])) for _, row in existing_df.iterrows()}
        trained_models = {str(row["model"]) for _, row in existing_df.iterrows()}
        print(f"Resume enabled: loaded {len(existing_df)} existing rows.")
    else:
        scan_rows = []
        completed_pairs: set[tuple[str, str]] = set()
        trained_models: set[str] = set()

    model_cache: dict[str, tuple] = {}
    for model_name in MODEL_ORDER:
        if model_name not in trained_models:
            print(f"Training model once on 26-run train pool: {model_name}")
            model_cache[model_name] = common.train_model(
                model_name=model_name,
                train_raw_seq=train_raw_seq,
                train_y=train_y,
                feature_spec=common.LEGACY_FEATURE_SPEC,
                config=BASELINE_CONFIG,
            )

        for test_idx, test_row in test_df.iterrows():
            test_case = str(test_row["source_file"])
            if (test_case, model_name) in completed_pairs:
                print(f"skip existing | test={test_case} | model={model_name}")
                continue

            if model_name not in model_cache:
                # Existing detail file may already contain some rows for this model; retrain once if needed.
                print(f"Re-training cached model for resume: {model_name}")
                model_cache[model_name] = common.train_model(
                    model_name=model_name,
                    train_raw_seq=train_raw_seq,
                    train_y=train_y,
                    feature_spec=common.LEGACY_FEATURE_SPEC,
                    config=BASELINE_CONFIG,
                )

            model, seq_scaler, target_scaler = model_cache[model_name]
            test_file = str(test_row["file_name"])
            test_table = case_tables[test_file]
            test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, SEQ_LEN)
            true_curve_df = common.threshold_ground_truth(test_table, common.WEAR_THRESHOLD_UM, test_row)
            true_life_actual = float(test_row["actual_life"])

            print(f"evaluate | test={test_case} | model={model_name}")
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
                SEQ_LEN,
            )
            curve_mae = common.wear_curve_mae(true_curve_df, rollout_df)
            scan_rows.append(
                {
                    "test_case": test_case,
                    "model": model_name,
                    "pressure_mae": pressure_metrics["pressure_mae"],
                    "pressure_rmse": pressure_metrics["pressure_rmse"],
                    "pressure_mape": pressure_metrics["pressure_mape"],
                    "wear_mae_um": curve_mae,
                    "predicted_life": predicted_life,
                    "true_life": true_life_actual,
                    "life_abs_error": abs(predicted_life - true_life_actual),
                    "life_rel_error": abs(predicted_life - true_life_actual) / max(true_life_actual, common.EPS),
                }
            )
            completed_pairs.add((test_case, model_name))
            pd.DataFrame(scan_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    scan_df = pd.DataFrame(scan_rows)
    summary_path = SCRIPT_DIR / SUMMARY_FILENAME
    chart1_path = SCRIPT_DIR / CHART1_FILENAME
    chart2_path = SCRIPT_DIR / CHART2_FILENAME
    chart3_path = SCRIPT_DIR / CHART3_FILENAME
    notes_path = SCRIPT_DIR / NOTES_FILENAME
    analysis_path = SCRIPT_DIR / ANALYSIS_FILENAME

    summary_stats = (
        scan_df.groupby("model")
        .agg(
            mean_pressure_mae=("pressure_mae", "mean"),
            mean_pressure_rmse=("pressure_rmse", "mean"),
            mean_pressure_mape=("pressure_mape", "mean"),
            mean_wear_mae_um=("wear_mae_um", "mean"),
            mean_life_abs_error=("life_abs_error", "mean"),
            median_life_abs_error=("life_abs_error", "median"),
            mean_life_rel_error=("life_rel_error", "mean"),
            max_life_abs_error=("life_abs_error", "max"),
            min_life_abs_error=("life_abs_error", "min"),
        )
        .reset_index()
        .sort_values("mean_life_abs_error")
        .reset_index(drop=True)
    )
    win_table = (
        scan_df.loc[scan_df.groupby("test_case")["life_abs_error"].idxmin()]
        .groupby("model")
        .size()
        .rename("life_error_win_count")
        .reset_index()
    )
    summary_stats = summary_stats.merge(win_table, on="model", how="left").fillna({"life_error_win_count": 0})
    summary_stats["life_error_win_count"] = summary_stats["life_error_win_count"].astype(int)
    summary_stats.to_csv(summary_path, index=False, encoding="utf-8-sig")

    common.save_grouped_life_error_bar(scan_df, chart1_path, "30测试集: 各工况寿命误差分组对比")
    common.save_summary_bar_chart(summary_stats, chart2_path, "30测试集: 五模型固定测试集对比")
    common.save_mean_predicted_life_chart(summary_stats, scan_df, chart3_path, "30测试集: 平均预测寿命对比")

    notes_path.write_text(build_notes(train_df, test_df, summary_stats), encoding="utf-8")
    analysis_path.write_text(build_analysis(summary_stats, scan_df, train_df, test_df), encoding="utf-8")

    print()
    print("=" * 60)
    print("30测试集 基准测试完成")
    print("=" * 60)
    print(summary_stats.to_string(index=False))


if __name__ == "__main__":
    main()


