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


EXPERIMENT_NAME = "30测试集 五模型统一增强"
EXPERIMENT_DATE = "2026-04-24"
SEQ_LEN = 12
BASELINE_DIR = ROOT_DIR / "结果" / "30测试集" / "01_基准测试"

CONFIG = common.TransformerConfig(
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
FEATURE_SPEC = common.FEATURE_R1_SPEC
SHAPE_CONFIG = common.ShapeLossConfig(
    name="slow_only",
    mono_lambda=0.0,
    slow_lambda=0.01,
    mono_tolerance_ratio=0.0,
)
MODEL_ORDER = ["FNN", "GRU", "LSTM", "1D-CNN", "Transformer"]
STEP_MODES = [
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
COMPARE_FILENAME = "固定250_增强_vs_基准_对比汇总.csv"


def mode_dir(mode: dict) -> Path:
    path = SCRIPT_DIR / str(mode["name"])
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def build_plan(train_df: pd.DataFrame, test_df: pd.DataFrame, mode: dict) -> str:
    step_text = "native median actual_cycle step" if mode["actual_step_cap"] is None else "actual_step cap 到 250"
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
            f"epochs: {CONFIG.epochs}",
            f"rollout步长口径: {step_text}",
            "",
            "增强包定义:",
            f"  特征工程: {FEATURE_SPEC.name}，{FEATURE_SPEC.description}。",
            "  特征列: " + ", ".join(FEATURE_SPEC.columns),
            f"  Shape Loss: {SHAPE_CONFIG.name}，mono_lambda={SHAPE_CONFIG.mono_lambda}, slow_lambda={SHAPE_CONFIG.slow_lambda}, mono_tolerance_ratio={SHAPE_CONFIG.mono_tolerance_ratio}。",
            "",
            "实验目的:",
            "  将此前表现最值得保留的特征工程和损失函数构造组合成统一增强包，直接加到 FNN / GRU / LSTM / 1D-CNN / Transformer 五个模型上，检验它是否是跨模型通用增强，而不是只对 Transformer 有效。",
            "",
            "公平性设计:",
            "  本轮与 01_基准测试使用相同 split、相同 seq_len、相同 epochs、相同五模型范围。",
            "  本轮只按固定250 rollout口径评估，与 01_基准测试/固定250_seq12 直接对比。",
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
        f"训练轮次: {CONFIG.epochs}",
        f"actual_step_cap: {mode['actual_step_cap']}",
        "",
        "实验目的:",
        "  在最新 30测试集 split 上，把 R1_静态派生 + slow_only 同时加到五个模型，观察增强包对寿命误差、压力误差和磨损曲线误差的影响。",
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
    lines = [
        "结果说明与分析",
        "=" * 72,
        "",
        "一、实验设置",
        "-" * 72,
        f"本轮固定测试集共 4 个 run：{', '.join(test_df['source_file'].astype(str).tolist())}。",
        f"训练集共 {len(train_df)} 个 run，测试集共 {len(test_df)} 个 run。",
        f"增强包为 R1_静态派生 + slow_only，seq_len={SEQ_LEN}，步长口径为 {mode['label']}。",
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
            "3. 本目录只说明增强模型自身排序；是否真的有效，需要结合父级 `固定250_增强_vs_基准_对比汇总.csv` 与 01_基准测试逐模型比较。",
        ]
    )
    return "\n".join(lines)


def save_mode_outputs(mode: dict, scan_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    out_dir = mode_dir(mode)
    detail_path = out_dir / DETAIL_FILENAME
    summary_path = out_dir / SUMMARY_FILENAME

    scan_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df = summarize(scan_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    common.save_grouped_life_error_bar(scan_df, out_dir / CHART1_FILENAME, f"30测试集 五模型统一增强 {mode['label']}: 各工况寿命误差")
    common.save_summary_bar_chart(summary_df, out_dir / CHART2_FILENAME, f"30测试集 五模型统一增强 {mode['label']}: 五模型对比")
    common.save_mean_predicted_life_chart(summary_df, scan_df, out_dir / CHART3_FILENAME, f"30测试集 五模型统一增强 {mode['label']}: 平均预测寿命")
    (out_dir / PLAN_FILENAME).write_text(build_plan(train_df, test_df, mode), encoding="utf-8")
    (out_dir / NOTES_FILENAME).write_text(build_notes(train_df, test_df, summary_df, mode), encoding="utf-8")
    (out_dir / ANALYSIS_FILENAME).write_text(build_analysis(summary_df, scan_df, train_df, test_df, mode), encoding="utf-8")
    return summary_df


def load_baseline_summary(mode_name: str) -> pd.DataFrame:
    path = BASELINE_DIR / mode_name / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {path}")
    return pd.read_csv(path)


def save_comparison(enhanced_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mode in STEP_MODES:
        mode_name = str(mode["name"])
        baseline = load_baseline_summary(mode_name)
        enhanced = enhanced_summaries[mode_name]
        merged = baseline.merge(enhanced, on="model", suffixes=("_baseline", "_enhanced"))
        for row in merged.itertuples(index=False):
            baseline_life = float(row.mean_life_abs_error_baseline)
            enhanced_life = float(row.mean_life_abs_error_enhanced)
            rows.append(
                {
                    "step_mode": mode["label"],
                    "model": row.model,
                    "baseline_mean_life_abs_error": baseline_life,
                    "enhanced_mean_life_abs_error": enhanced_life,
                    "enhanced_minus_baseline_life_abs_error": enhanced_life - baseline_life,
                    "enhanced_life_error_change_pct": (enhanced_life - baseline_life) / max(baseline_life, common.EPS),
                    "baseline_mean_pressure_mae": float(row.mean_pressure_mae_baseline),
                    "enhanced_mean_pressure_mae": float(row.mean_pressure_mae_enhanced),
                    "enhanced_minus_baseline_pressure_mae": float(row.mean_pressure_mae_enhanced) - float(row.mean_pressure_mae_baseline),
                    "baseline_mean_wear_mae_um": float(row.mean_wear_mae_um_baseline),
                    "enhanced_mean_wear_mae_um": float(row.mean_wear_mae_um_enhanced),
                    "enhanced_minus_baseline_wear_mae_um": float(row.mean_wear_mae_um_enhanced) - float(row.mean_wear_mae_um_baseline),
                }
            )
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(SCRIPT_DIR / COMPARE_FILENAME, index=False, encoding="utf-8-sig")


    fig, ax = plt.subplots(figsize=(11, 6))
    fixed_df = compare_df[compare_df["step_mode"].eq("固定250")].set_index("model").loc[MODEL_ORDER].reset_index()
    x = range(len(fixed_df))
    width = 0.36
    ax.bar([i - width / 2 for i in x], fixed_df["baseline_mean_life_abs_error"], width=width, label="基准", color="#64748b")
    ax.bar([i + width / 2 for i in x], fixed_df["enhanced_mean_life_abs_error"], width=width, label="统一增强", color="#0f766e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fixed_df["model"])
    ax.set_ylabel("mean_life_abs_error")
    ax.set_title("固定250: 五模型基准 vs 统一增强")
    ax.legend()
    common.style_axes(ax)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "图_固定250_增强_vs_基准.png", dpi=220)
    plt.close(fig)

    return compare_df


def build_parent_docs(compare_df: pd.DataFrame, enhanced_summaries: dict[str, pd.DataFrame]) -> None:
    lines = [
        "五模型统一增强 结果说明与分析",
        "=" * 72,
        "",
        "一、实验目的",
        "-" * 72,
        "本轮不是重新搜索模块，而是把此前最值得保留的 `R1_静态派生 + slow_only` 作为统一增强包，直接加到五个模型上，检验它是否具有跨模型通用性。",
        "",
        "二、当前口径",
        "-" * 72,
        "固定测试集：Run4 / Run8 / Run11 / Run30。",
        "模型范围：FNN / GRU / LSTM / 1D-CNN / Transformer。",
        "训练设置：seq_len=12，epochs=1200，原始训练序列不重采样。",
        "步长输出：只保留固定250口径；训练仍使用原始数据采样点，预测闭环递推和磨损积分使用 250。",
        "",
        "三、增强相对基准的逐模型变化",
        "-" * 72,
    ]
    part = compare_df[compare_df["step_mode"].eq("固定250")].sort_values("enhanced_minus_baseline_life_abs_error")
    for row in part.itertuples(index=False):
        delta = float(row.enhanced_minus_baseline_life_abs_error)
        sign = "改善" if delta < 0 else "退化"
        lines.append(
            f"{row.model}: 基准={float(row.baseline_mean_life_abs_error):.0f}, "
            f"增强={float(row.enhanced_mean_life_abs_error):.0f}, {sign}={abs(delta):.0f}"
        )
    lines.append("")

    fixed_summary = enhanced_summaries["固定250_seq12"].sort_values("mean_life_abs_error").reset_index(drop=True)
    lines.extend(
        [
            "四、增强模型自身排序",
            "-" * 72,
            f"固定250增强后最佳模型：{fixed_summary.iloc[0]['model']}，mean_life_abs_error≈{float(fixed_summary.iloc[0]['mean_life_abs_error']):.0f}。",
            "",
            "五、当前判断",
            "-" * 72,
            "如果某个模型增强后寿命误差下降，说明这套特征和趋势先验对它有帮助；如果寿命误差上升，则说明该模型可能已经用自己的结构学到了类似信息，或者增强包在当前数据和 split 下引入了额外偏置。",
            "最终是否保留这套增强包，应看固定250口径，因为这对应后续软件默认预测步长。",
        ]
    )
    (SCRIPT_DIR / ANALYSIS_FILENAME).write_text("\n".join(lines), encoding="utf-8")

    notes = "\n".join(
        [
            "测试说明",
            "=" * 72,
            f"实验名称：{EXPERIMENT_NAME}",
            f"测试日期：{EXPERIMENT_DATE}",
            "增强包：R1_静态派生 + slow_only。",
            "测试集：Run4 / Run8 / Run11 / Run30。",
            "seq_len：12。",
            "epochs：1200。",
            "正式子目录：固定250_seq12。",
            "父级对比文件：固定250_增强_vs_基准_对比汇总.csv。",
        ]
    )
    (SCRIPT_DIR / NOTES_FILENAME).write_text(notes, encoding="utf-8")


def main() -> None:
    common.require_cuda()
    common.set_seed(common.SEED)
    for line in common.device_summary_lines():
        print(line)
    print(EXPERIMENT_NAME)
    print(f"  seq_len={SEQ_LEN}, epochs={CONFIG.epochs}")
    print(f"  feature={FEATURE_SPEC.name}, shape={SHAPE_CONFIG.name}")
    print(f"  split={common.SPLIT_CSV}")
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
    train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)
    mode_rows = {mode["name"]: [] for mode in STEP_MODES}

    for model_name in MODEL_ORDER:
        print(f"Training enhanced model once on 26-run train pool: {model_name}")
        model, seq_scaler, target_scaler = common.train_model(
            model_name=model_name,
            train_raw_seq=train_raw_seq,
            train_y=train_y,
            feature_spec=FEATURE_SPEC,
            config=CONFIG,
            shape_config=SHAPE_CONFIG,
            train_case_names=train_case_names,
            train_step_indices=train_step_indices,
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
                FEATURE_SPEC,
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
                    FEATURE_SPEC,
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

    enhanced_summaries = {}
    for mode in STEP_MODES:
        scan_df = pd.DataFrame(mode_rows[mode["name"]])
        enhanced_summaries[mode["name"]] = save_mode_outputs(mode, scan_df, train_df, test_df)

    compare_df = save_comparison(enhanced_summaries)
    build_parent_docs(compare_df, enhanced_summaries)

    print()
    print("=" * 72)
    print("五模型统一增强完成")
    print("=" * 72)
    for mode in STEP_MODES:
        print(f"\n[{mode['label']}]")
        print(enhanced_summaries[mode["name"]].to_string(index=False))
    print("\n[增强 - 基准]")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()



