from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
COMMON_DIR = ROOT_DIR / "工具和杂项" / "30测试集_中间过程"
SEARCH_DIR = COMMON_DIR / "02_transformer升级"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import common_fixed_split as common


EXPERIMENT_NAME = "30测试集 Transformer 升级测试"
EXPERIMENT_DATE = "2026-04-24"

DETAIL_FILENAME = "详细结果_各测试集.csv"
SUMMARY_FILENAME = "汇总_各模型平均指标.csv"
COMPARE_FILENAME = "与基准五模型对比.csv"
CHART1_FILENAME = "图1_Transformer_v2各工况寿命误差.png"
CHART2_FILENAME = "图2_与五模型基准三指标对比.png"
CHART3_FILENAME = "图3_与五模型基准平均预测寿命对比.png"
NOTES_FILENAME = "测试说明.txt"
UPGRADE_FILENAME = "升级方案.txt"
ANALYSIS_FILENAME = "结果说明与分析.txt"
CONFIG_FILENAME = "最终配置.json"


def load_best_config_payload() -> dict[str, object]:
    config_path = SEARCH_DIR / "最优配置.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing best config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def to_config(payload: dict[str, object]) -> common.TransformerConfig:
    return common.TransformerConfig(
        seq_len=int(payload["seq_len"]),
        d_model=int(payload["d_model"]),
        nhead=int(payload["nhead"]),
        num_layers=int(payload["num_layers"]),
        dim_ff=int(payload["dim_ff"]),
        dropout=float(payload["dropout"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
    )


def build_upgrade_text(candidate_df: pd.DataFrame, baseline_summary: pd.DataFrame) -> str:
    baseline_transformer = baseline_summary[baseline_summary["model"] == "Transformer"].iloc[0]
    best_round = str(candidate_df.iloc[0]["round_id"])
    lines = [
        "升级方案",
        "=" * 72,
        "",
        "一、基线",
        "-" * 72,
        "本轮基线来自 `30测试集/01_基准测试` 中的 Transformer 结果。",
        f"基线指标: pressure_MAE={float(baseline_transformer['mean_pressure_mae']):.2f}, "
        f"wear_MAE={float(baseline_transformer['mean_wear_mae_um']):.3f}, "
        f"life_abs_error={float(baseline_transformer['mean_life_abs_error']):.0f}",
        "",
        "二、候选搜索范围",
        "-" * 72,
        "本轮候选比 17测试集旧版更完整，覆盖 seq_len=6/8/10/12/15/20、d_model=32/48/64/96、2/3层、不同 ff 宽度、dropout/weight_decay、低学习率和 validation 分支。",
        "所有候选使用同一固定测试集、同一 26-run 训练池、同一 5 维输入和同一 rollout 口径，未混入 shape loss、特征构造或 250 步长改动。",
        "",
        "三、候选结果与取舍",
        "-" * 72,
    ]
    for row in candidate_df.itertuples(index=False):
        delta_vs_baseline = float(row.mean_life_abs_error) - float(baseline_transformer["mean_life_abs_error"])
        lines.extend(
            [
                f"{row.round_id}: {row.description}",
                f"  配置: seq_len={int(row.seq_len)}, d_model={int(row.d_model)}, layers={int(row.num_layers)}, ff={int(row.dim_ff)}, "
                f"dropout={float(row.dropout):.2f}, epochs={int(row.epochs)}, lr={float(row.learning_rate):.1e}, wd={float(row.weight_decay):.1e}, val={bool(row.use_validation)}",
                f"  指标: pressure_MAE={float(row.mean_pressure_mae):.2f}, wear_MAE={float(row.mean_wear_mae_um):.3f}, life_abs_error={float(row.mean_life_abs_error):.0f}",
                f"  相对 baseline 寿命误差变化: {delta_vs_baseline:+.0f}",
                f"  结论: {'保留' if row.round_id == best_round else '不保留'}",
            ]
        )
    lines.extend(["", "四、最终选定", "-" * 72, f"固定测试集下候选搜索最佳配置选为 {best_round}。"])
    return "\n".join(lines)


def build_notes(config: common.TransformerConfig, payload: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame, baseline_summary: pd.DataFrame) -> str:
    baseline_transformer = baseline_summary[baseline_summary["model"] == "Transformer"].iloc[0]
    lines = [
        "测试说明",
        "=" * 60,
        f"实验名称: {EXPERIMENT_NAME}",
        f"测试日期: {EXPERIMENT_DATE}",
        f"数据源: {common.DATA_DIR}",
        f"测试集清单: {common.SPLIT_CSV}",
        f"训练集 run: {', '.join(train_df['source_file'].astype(str).tolist())}",
        f"测试集 run: {', '.join(test_df['source_file'].astype(str).tolist())}",
        f"最终候选: {payload['round_id']} - {payload['description']}",
        f"最终配置: seq_len={config.seq_len}, d_model={config.d_model}, nhead={config.nhead}, layers={config.num_layers}, ff={config.dim_ff}",
        f"训练参数: epochs={config.epochs}, lr={config.learning_rate}, weight_decay={config.weight_decay}, dropout={config.dropout}",
        "",
        "实验目的:",
        "  在完整 30 组数据、固定测试集 Run4 / Run8 / Run10 / Run11 上重新探索 Transformer 最优参数，并与五模型基准进行统一对比。",
        "",
        "公平性说明:",
        "  本轮只比较 Transformer 参数，不改变数据 split、不改变 5 维输入、不加入 shape loss/特征构造、不加入 250 步长改动。",
        "",
        "30测试集 baseline Transformer:",
        f"  pressure_MAE={float(baseline_transformer['mean_pressure_mae']):.2f}",
        f"  wear_MAE={float(baseline_transformer['mean_wear_mae_um']):.3f}",
        f"  life_abs_error={float(baseline_transformer['mean_life_abs_error']):.0f}",
        "",
        "CUDA:",
    ]
    lines.extend([f"  {line}" for line in common.device_summary_lines()])
    return "\n".join(lines)


def build_analysis(summary_v2: pd.DataFrame, baseline_summary: pd.DataFrame, candidate_df: pd.DataFrame, compare_df: pd.DataFrame) -> str:
    baseline_transformer = baseline_summary[baseline_summary["model"] == "Transformer"].iloc[0]
    baseline_lstm = baseline_summary[baseline_summary["model"] == "LSTM"].iloc[0]
    final_row = summary_v2.iloc[0]
    best_candidate = candidate_df.iloc[0]
    ranking_df = compare_df.sort_values("mean_life_abs_error").reset_index(drop=True)
    lines = [
        "结果说明与分析",
        "=" * 72,
        "",
        "一、最终结果",
        "-" * 72,
        f"30测试集 baseline Transformer: pressure_MAE={float(baseline_transformer['mean_pressure_mae']):.2f}, "
        f"wear_MAE={float(baseline_transformer['mean_wear_mae_um']):.3f}, life_abs_error={float(baseline_transformer['mean_life_abs_error']):.0f}",
        f"Transformer_v2 最终结果: pressure_MAE={float(final_row['mean_pressure_mae']):.2f}, "
        f"wear_MAE={float(final_row['mean_wear_mae_um']):.3f}, life_abs_error={float(final_row['mean_life_abs_error']):.0f}",
        f"相对 baseline Transformer 寿命误差变化: {float(final_row['mean_life_abs_error']) - float(baseline_transformer['mean_life_abs_error']):+.0f}",
        f"相对当前 LSTM 基准寿命误差变化: {float(final_row['mean_life_abs_error']) - float(baseline_lstm['mean_life_abs_error']):+.0f}",
        "",
        "二、后验排名（含五模型基准）",
        "-" * 72,
    ]
    for idx, row in enumerate(ranking_df.itertuples(index=False), start=1):
        lines.append(
            f"{idx}. {row.model} | pressure_MAE={float(row.mean_pressure_mae):.2f} | "
            f"wear_MAE={float(row.mean_wear_mae_um):.3f} | life_abs_error={float(row.mean_life_abs_error):.0f}"
        )
    lines.extend(
        [
            "",
            "三、候选链路观察",
            "-" * 72,
            f"1. 本轮完整搜索最佳候选为 {best_candidate['round_id']}，搜索阶段 life_abs_error 约 {float(best_candidate['mean_life_abs_error']):.0f}。",
            f"2. 最终正式复训一次后得到 life_abs_error 约 {float(final_row['mean_life_abs_error']):.0f}，用于和五模型基准横向比较。",
            "3. 如果最终结果仍未超过 LSTM，则说明完整 30 组数据下基础架构暂时仍应保留 LSTM 与 Transformer 双线，而不是只押 Transformer。",
            "4. 后续 shape loss 和特征构造应继承本轮完整数据下重新筛出的 Transformer 配置，再分别测试模块贡献。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    common.require_cuda()
    common.set_seed(common.SEED)
    payload = load_best_config_payload()
    config = to_config(payload)
    for line in common.device_summary_lines():
        print(line)
    print(f"{EXPERIMENT_NAME}")
    print(
        f"  best={payload['round_id']} | seq_len={config.seq_len}, d_model={config.d_model}, nhead={config.nhead}, "
        f"layers={config.num_layers}, ff={config.dim_ff}"
    )
    print(f"  epochs={config.epochs}, lr={config.learning_rate}, wd={config.weight_decay}")
    print()

    summary_df, case_tables = common.load_data()
    train_df, test_df = common.load_fixed_split(summary_df)
    baseline_detail_path = ROOT_DIR / "结果" / "30测试集" / "01_基准测试" / "详细结果_各模型各测试集.csv"
    baseline_summary_path = ROOT_DIR / "结果" / "30测试集" / "01_基准测试" / "汇总_各模型平均指标.csv"
    baseline_summary = pd.read_csv(baseline_summary_path)
    candidate_df = pd.read_csv(SEARCH_DIR / "候选汇总.csv").sort_values("mean_life_abs_error").reset_index(drop=True)

    train_tables = {str(row["file_name"]): case_tables[str(row["file_name"])] for _, row in train_df.iterrows()}
    train_raw_seq, train_y, _, _ = common.build_raw_sequence_dataset(train_tables, config.seq_len)

    detail_path = SCRIPT_DIR / DETAIL_FILENAME
    if detail_path.exists():
        existing_df = pd.read_csv(detail_path)
        scan_rows = existing_df.to_dict("records")
        completed_cases = {str(row["test_case"]) for _, row in existing_df.iterrows()}
        print(f"Resume enabled: loaded {len(existing_df)} completed test cases.")
    else:
        scan_rows = []
        completed_cases: set[str] = set()

    model, seq_scaler, target_scaler = common.train_model(
        model_name="Transformer_v2",
        train_raw_seq=train_raw_seq,
        train_y=train_y,
        feature_spec=common.LEGACY_FEATURE_SPEC,
        config=config,
    )

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
        scan_rows.append(
            {
                "test_case": test_source,
                "model": "Transformer_v2",
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
        completed_cases.add(test_source)
        pd.DataFrame(scan_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    scan_df = pd.DataFrame(scan_rows)
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
    summary_stats.to_csv(SCRIPT_DIR / SUMMARY_FILENAME, index=False, encoding="utf-8-sig")

    compare_df = pd.concat([baseline_summary, summary_stats], ignore_index=True).sort_values("mean_life_abs_error").reset_index(drop=True)
    compare_df.to_csv(SCRIPT_DIR / COMPARE_FILENAME, index=False, encoding="utf-8-sig")

    common.save_grouped_life_error_bar(scan_df, SCRIPT_DIR / CHART1_FILENAME, "30测试集: Transformer_v2 各工况寿命误差")
    common.save_summary_bar_chart(compare_df, SCRIPT_DIR / CHART2_FILENAME, "30测试集: Transformer_v2 与五模型基准三指标对比")
    common.save_mean_predicted_life_chart(compare_df, pd.concat([pd.read_csv(baseline_detail_path), scan_df], ignore_index=True), SCRIPT_DIR / CHART3_FILENAME, "30测试集: Transformer_v2 与五模型基准平均预测寿命")

    (SCRIPT_DIR / NOTES_FILENAME).write_text(build_notes(config, payload, train_df, test_df, baseline_summary), encoding="utf-8")
    (SCRIPT_DIR / UPGRADE_FILENAME).write_text(build_upgrade_text(candidate_df, baseline_summary), encoding="utf-8")
    (SCRIPT_DIR / ANALYSIS_FILENAME).write_text(build_analysis(summary_stats, baseline_summary, candidate_df, compare_df), encoding="utf-8")
    (SCRIPT_DIR / CONFIG_FILENAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 72)
    print("30测试集 Transformer 升级测试完成")
    print("=" * 72)
    print(compare_df[["model", "mean_pressure_mae", "mean_wear_mae_um", "mean_life_abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
