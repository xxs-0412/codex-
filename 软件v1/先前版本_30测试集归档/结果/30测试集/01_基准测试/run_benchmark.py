from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
COMMON_DIR = ROOT_DIR / "工具和杂项" / "30测试集_中间过程"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import common_fixed_split as common


EXPERIMENT_NAME = "30测试集 基准测试 seq12 步长对照"
EXPERIMENT_DATE = "2026-04-24"
SEQ_LEN = 12
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
STEP_MODES = [
    {"name": "默认步长_seq12", "label": "默认步长", "actual_step_cap": None},
    {"name": "固定250_seq12", "label": "固定250", "actual_step_cap": 250.0},
]

DETAIL_FILENAME = "详细结果_各模型各测试集.csv"
SUMMARY_FILENAME = "汇总_各模型平均指标.csv"
CHART1_FILENAME = "图1_各工况寿命误差分组柱状图.png"
CHART2_FILENAME = "图2_模型对比三指标柱状图.png"
CHART3_FILENAME = "图3_平均预测寿命柱状图.png"
NOTES_FILENAME = "测试说明.txt"
PLAN_FILENAME = "实验方案.txt"
ANALYSIS_FILENAME = "结果说明与分析.txt"
COMPARE_FILENAME = "默认_vs_固定250_对比汇总.csv"
COMPARE_CHART_FILENAME = "图_默认_vs_固定250_寿命误差对比.png"


def mode_dir(mode: dict) -> Path:
    path = SCRIPT_DIR / str(mode["name"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_plan(train_df: pd.DataFrame, test_df: pd.DataFrame, mode: dict) -> str:
    cap = mode["actual_step_cap"]
    step_text = "每个测试 run 的 native median actual_cycle step" if cap is None else f"actual_step cap 到 {cap:g}"
    return "\n".join(
        [
            "实验方案",
            "=" * 60,
            f"实验名称: {EXPERIMENT_NAME} - {mode['label']}",
            f"测试日期: {EXPERIMENT_DATE}",
            f"数据源: {common.DATA_DIR}",
            f"测试集清单: {common.SPLIT_CSV}",
            f"测试集 run: {', '.join(test_df['source_file'].astype(str).tolist())}",
            f"训练集数量: {len(train_df)}",
            f"测试集数量: {len(test_df)}",
            f"seq_len: {SEQ_LEN}",
            f"epochs: {BASELINE_CONFIG.epochs}",
            f"rollout步长口径: {step_text}",
            "",
            "实验目的:",
            "  在完整 30 组数据的新固定测试集 Run4 / Run8 / Run11 / Run30 上，重新建立五模型 seq_len=12 基准；同时比较默认步长与固定250步长对闭环寿命预测的影响。",
            "",
            "公平性设计:",
            "  父级脚本只训练每个模型一次，然后用同一组模型权重分别评估默认步长和固定250，避免两次训练随机波动影响步长对比。",
            "  固定250只改变闭环 rollout 和磨损积分步长，不对训练集原始序列做重采样。",
            "",
            "模型范围:",
            "  FNN / GRU / LSTM / 1D-CNN / Transformer。",
        ]
    )


def build_notes(train_df: pd.DataFrame, test_df: pd.DataFrame, summary_df: pd.DataFrame, mode: dict) -> str:
    lines = [
        "测试说明",
        "=" * 60,
        f"实验名称: {EXPERIMENT_NAME} - {mode['label']}",
        f"测试日期: {EXPERIMENT_DATE}",
        f"数据源: {common.DATA_DIR}",
        f"测试集清单: {common.SPLIT_CSV}",
        f"训练集数量: {len(train_df)}",
        f"测试集数量: {len(test_df)}",
        f"测试集 run: {', '.join(test_df['source_file'].astype(str).tolist())}",
        f"序列长度: {SEQ_LEN}",
        f"训练轮次: {BASELINE_CONFIG.epochs}",
        f"actual_step_cap: {mode['actual_step_cap']}",
        "",
        "实验目的:",
        "  用完整30组数据和新四测试集重新建立 seq_len=12 五模型基线，并观察闭环步长从默认 native 改为固定250后，寿命误差、压力误差和磨损曲线误差是否变化。",
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
            f"  {PLAN_FILENAME}",
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


def build_analysis(summary_df: pd.DataFrame, detail_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, mode: dict) -> str:
    ranked = summary_df.sort_values("mean_life_abs_error").reset_index(drop=True)
    best = ranked.iloc[0]
    worst_case = detail_df.sort_values("life_abs_error", ascending=False).iloc[0]
    step_desc = "默认 native 步长" if mode["actual_step_cap"] is None else "固定250步长"
    lines = [
        "结果说明与分析",
        "=" * 72,
        "",
        "一、实验设置",
        "-" * 72,
        f"本轮固定测试集共 4 个 run：{', '.join(test_df['source_file'].astype(str).tolist())}。",
        f"训练集共 {len(train_df)} 个 run，测试集共 {len(test_df)} 个 run。",
        f"本目录口径：{step_desc}；seq_len={SEQ_LEN}；五模型共用同一训练集和同一组模型权重。",
        "固定250口径只改变闭环 rollout 和磨损积分步长，不重采样训练数据。" if mode["actual_step_cap"] is not None else "默认步长口径沿用每个测试 run 的 native median actual_cycle step。",
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
            f"1. 本口径最佳模型为 {best['model']}，平均寿命绝对误差约 {float(best['mean_life_abs_error']):.0f}。",
            f"2. 单个误差最大的测试工况为 {worst_case['test_case']} / {worst_case['model']}，寿命绝对误差约 {float(worst_case['life_abs_error']):.0f}。",
            f"3. 本结果应与另一子目录的步长口径结果配套阅读，重点判断固定250是否比默认 native 步长更稳。",
        ]
    )
    return "\n".join(lines)


def summarize(scan_df: pd.DataFrame) -> pd.DataFrame:
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
            mean_native_actual_step=("native_actual_step", "mean"),
            mean_used_actual_step=("used_actual_step", "mean"),
            step_capped_count=("step_capped", "sum"),
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
    return summary_stats


def save_mode_outputs(mode: dict, scan_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    out_dir = mode_dir(mode)
    detail_path = out_dir / DETAIL_FILENAME
    summary_path = out_dir / SUMMARY_FILENAME
    chart1_path = out_dir / CHART1_FILENAME
    chart2_path = out_dir / CHART2_FILENAME
    chart3_path = out_dir / CHART3_FILENAME

    scan_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df = summarize(scan_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    common.save_grouped_life_error_bar(scan_df, chart1_path, f"30测试集 seq12 {mode['label']}: 各工况寿命误差")
    common.save_summary_bar_chart(summary_df, chart2_path, f"30测试集 seq12 {mode['label']}: 五模型对比")
    common.save_mean_predicted_life_chart(summary_df, scan_df, chart3_path, f"30测试集 seq12 {mode['label']}: 平均预测寿命")
    (out_dir / PLAN_FILENAME).write_text(build_plan(train_df, test_df, mode), encoding="utf-8")
    (out_dir / NOTES_FILENAME).write_text(build_notes(train_df, test_df, summary_df, mode), encoding="utf-8")
    (out_dir / ANALYSIS_FILENAME).write_text(build_analysis(summary_df, scan_df, train_df, test_df, mode), encoding="utf-8")
    return summary_df


def save_comparison(default_summary: pd.DataFrame, fixed_summary: pd.DataFrame) -> pd.DataFrame:
    merged = default_summary.merge(fixed_summary, on="model", suffixes=("_default", "_fixed250"))
    rows = []
    for row in merged.itertuples(index=False):
        default_life = float(row.mean_life_abs_error_default)
        fixed_life = float(row.mean_life_abs_error_fixed250)
        rows.append(
            {
                "model": row.model,
                "default_mean_life_abs_error": default_life,
                "fixed250_mean_life_abs_error": fixed_life,
                "fixed250_minus_default_life_abs_error": fixed_life - default_life,
                "fixed250_life_error_change_pct": (fixed_life - default_life) / max(default_life, common.EPS),
                "default_mean_wear_mae_um": float(row.mean_wear_mae_um_default),
                "fixed250_mean_wear_mae_um": float(row.mean_wear_mae_um_fixed250),
                "fixed250_minus_default_wear_mae_um": float(row.mean_wear_mae_um_fixed250) - float(row.mean_wear_mae_um_default),
                "mean_pressure_mae": float(row.mean_pressure_mae_default),
            }
        )
    compare_df = pd.DataFrame(rows).sort_values("fixed250_minus_default_life_abs_error").reset_index(drop=True)
    compare_df.to_csv(SCRIPT_DIR / COMPARE_FILENAME, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    plot_df = compare_df.set_index("model").loc[MODEL_ORDER].reset_index()
    x = range(len(plot_df))
    width = 0.36
    ax.bar([i - width / 2 for i in x], plot_df["default_mean_life_abs_error"], width=width, label="默认步长", color="#64748b")
    ax.bar([i + width / 2 for i in x], plot_df["fixed250_mean_life_abs_error"], width=width, label="固定250", color="#0f766e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["model"], rotation=0)
    ax.set_ylabel("mean_life_abs_error")
    ax.set_title("默认步长 vs 固定250：平均寿命绝对误差")
    ax.legend()
    common.style_axes(ax)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / COMPARE_CHART_FILENAME, dpi=220)
    plt.close(fig)

    best_default = default_summary.sort_values("mean_life_abs_error").iloc[0]
    best_fixed = fixed_summary.sort_values("mean_life_abs_error").iloc[0]
    best_change = compare_df.iloc[0]
    text = "\n".join(
        [
            "seq12 默认步长 vs 固定250 对比说明",
            "=" * 72,
            "",
            "一、实验口径",
            "-" * 72,
            "本轮使用同一组训练完成的五模型权重，同时评估默认 native 步长和固定250步长。",
            "固定250只改变闭环 rollout 和磨损积分步长，不对训练样本做重采样。",
            "当前测试集为 Run4 / Run8 / Run11 / Run30，seq_len=12。",
            "",
            "二、总体结论",
            "-" * 72,
            f"默认步长最佳模型：{best_default['model']}，mean_life_abs_error≈{float(best_default['mean_life_abs_error']):.0f}。",
            f"固定250最佳模型：{best_fixed['model']}，mean_life_abs_error≈{float(best_fixed['mean_life_abs_error']):.0f}。",
            f"固定250相对默认步长改善最大的模型：{best_change['model']}，寿命误差变化≈{float(best_change['fixed250_minus_default_life_abs_error']):.0f}。",
            "",
            "三、阅读建议",
            "-" * 72,
            "如果固定250普遍降低寿命误差，说明软件默认预测步长取 250 更稳；如果只对个别模型有效，则后续应把步长作为部署参数而不是训练结论的一部分。",
        ]
    )
    (SCRIPT_DIR / "结果说明与分析.txt").write_text(text, encoding="utf-8")
    return compare_df


def main() -> None:
    common.require_cuda()
    common.set_seed(common.SEED)
    for line in common.device_summary_lines():
        print(line)
    print(EXPERIMENT_NAME)
    print(f"  seq_len={SEQ_LEN}, epochs={BASELINE_CONFIG.epochs}")
    print(f"  split={common.SPLIT_CSV}")
    print("  step modes: 默认步长, 固定250")
    print()

    summary_df, case_tables = common.load_data()
    train_df, test_df = common.load_fixed_split(summary_df)
    print(f"Train runs: {len(train_df)}")
    print(f"Test runs: {len(test_df)}")
    for _, row in test_df.iterrows():
        test_table = case_tables[str(row["file_name"])]
        step = common.resolve_rollout_steps(test_table)
        print(
            f"  test | {row['source_file']:20s} | life={float(row['actual_life']):10.1f} | "
            f"native_step={step.native_actual_step:.1f}"
        )
    print()

    train_tables = {str(row["file_name"]): case_tables[str(row["file_name"])] for _, row in train_df.iterrows()}
    train_raw_seq, train_y, _, _ = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)
    mode_rows = {mode["name"]: [] for mode in STEP_MODES}

    for model_name in MODEL_ORDER:
        print(f"Training model once on 26-run train pool: {model_name}")
        model, seq_scaler, target_scaler = common.train_model(
            model_name=model_name,
            train_raw_seq=train_raw_seq,
            train_y=train_y,
            feature_spec=common.LEGACY_FEATURE_SPEC,
            config=BASELINE_CONFIG,
        )

        for _, test_row in test_df.iterrows():
            test_case = str(test_row["source_file"])
            test_file = str(test_row["file_name"])
            test_table = case_tables[test_file]
            test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, SEQ_LEN)
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

            for mode in STEP_MODES:
                print(f"evaluate | mode={mode['label']} | test={test_case} | model={model_name}")
                rollout_df, predicted_life, step_info = common.rollout_case_with_step_cap(
                    model,
                    seq_scaler,
                    target_scaler,
                    test_table,
                    common.WEAR_THRESHOLD_UM,
                    common.REAL_WEAR_COEFF_MPA_INV,
                    true_life_actual,
                    common.LEGACY_FEATURE_SPEC,
                    SEQ_LEN,
                    actual_step_cap=mode["actual_step_cap"],
                )
                curve_mae = common.wear_curve_mae(true_curve_df, rollout_df)
                mode_rows[mode["name"]].append(
                    {
                        "test_case": test_case,
                        "model": model_name,
                        "step_mode": mode["label"],
                        "actual_step_cap": mode["actual_step_cap"],
                        "native_actual_step": step_info.native_actual_step,
                        "used_actual_step": step_info.used_actual_step,
                        "native_sim_step": step_info.native_sim_step,
                        "used_sim_step": step_info.used_sim_step,
                        "step_capped": int(step_info.capped),
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

    summaries = {}
    for mode in STEP_MODES:
        scan_df = pd.DataFrame(mode_rows[mode["name"]])
        summaries[mode["name"]] = save_mode_outputs(mode, scan_df, train_df, test_df)

    compare_df = save_comparison(summaries["默认步长_seq12"], summaries["固定250_seq12"])
    print()
    print("=" * 72)
    print("seq12 默认步长 / 固定250 基准测试完成")
    print("=" * 72)
    for mode in STEP_MODES:
        print(f"\n[{mode['label']}]")
        print(summaries[mode["name"]].to_string(index=False))
    print("\n[固定250 - 默认步长]")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
