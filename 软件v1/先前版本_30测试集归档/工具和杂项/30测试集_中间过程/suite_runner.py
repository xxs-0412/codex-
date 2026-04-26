from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import common_fixed_split as common


DETAIL_FILENAME = "详细结果_各测试集.csv"
SUMMARY_FILENAME = "汇总_各模型平均指标.csv"
CHART1_FILENAME = "图1_各工况寿命误差分组柱状图.png"
CHART2_FILENAME = "图2_模型对比柱状图.png"
CHART3_FILENAME = "图3_平均预测寿命柱状图.png"
NOTES_FILENAME = "测试说明.txt"
PLAN_FILENAME = "实验方案.txt"
ANALYSIS_FILENAME = "结果说明与分析.txt"
BEST_FILENAME = "最优变体.json"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    model_name: str
    feature_spec: common.FeatureSpec
    train_config: common.TransformerConfig
    description: str
    shape_config: common.ShapeLossConfig | None = None


def _stable_variant_seed(name: str) -> int:
    total = 0
    for idx, ch in enumerate(str(name)):
        total += (idx + 1) * ord(ch)
    return int(common.SEED + (total % 100000))


def _variant_to_payload(variant: VariantSpec) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": variant.name,
        "model_name": variant.model_name,
        "feature_spec": {
            "name": variant.feature_spec.name,
            "columns": list(variant.feature_spec.columns),
            "description": variant.feature_spec.description,
        },
        "train_config": {
            "seq_len": variant.train_config.seq_len,
            "d_model": variant.train_config.d_model,
            "nhead": variant.train_config.nhead,
            "num_layers": variant.train_config.num_layers,
            "dim_ff": variant.train_config.dim_ff,
            "dropout": variant.train_config.dropout,
            "epochs": variant.train_config.epochs,
            "learning_rate": variant.train_config.learning_rate,
            "weight_decay": variant.train_config.weight_decay,
        },
        "description": variant.description,
    }
    if variant.shape_config is not None:
        payload["shape_config"] = {
            "name": variant.shape_config.name,
            "mono_lambda": variant.shape_config.mono_lambda,
            "slow_lambda": variant.shape_config.slow_lambda,
            "mono_tolerance_ratio": variant.shape_config.mono_tolerance_ratio,
        }
    return payload


def _effectiveness_label(summary_df: pd.DataFrame, baseline_name: str) -> pd.Series:
    baseline = summary_df[summary_df["model"] == baseline_name].iloc[0]
    labels: list[str] = []
    for row in summary_df.itertuples(index=False):
        if str(row.model) == baseline_name:
            labels.append("baseline")
            continue
        life_improved = float(row.mean_life_abs_error) < float(baseline["mean_life_abs_error"])
        pressure_reg = (float(row.mean_pressure_mae) - float(baseline["mean_pressure_mae"])) / max(float(baseline["mean_pressure_mae"]), common.EPS)
        wear_reg = (float(row.mean_wear_mae_um) - float(baseline["mean_wear_mae_um"])) / max(float(baseline["mean_wear_mae_um"]), common.EPS)
        if not life_improved:
            labels.append("无效")
        elif pressure_reg > 0.10 or wear_reg > 0.10:
            labels.append("条件有效")
        else:
            labels.append("有效")
    return pd.Series(labels)


def _build_notes(
    experiment_name: str,
    experiment_date: str,
    family_name: str,
    suite_kind: str,
    purpose: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variants: list[VariantSpec],
) -> str:
    lines = [
        "测试说明",
        "=" * 60,
        f"实验名称: {experiment_name}",
        f"测试日期: {experiment_date}",
        f"实验家族: {family_name}",
        f"实验类型: {suite_kind}",
        f"数据源: {common.DATA_DIR}",
        f"测试集清单: {common.SPLIT_CSV}",
        f"训练集 run: {', '.join(train_df['source_file'].astype(str).tolist())}",
        f"测试集 run: {', '.join(test_df['source_file'].astype(str).tolist())}",
        "",
        "实验目的:",
        f"  {purpose}",
        "",
        "当前数据阶段说明:",
        "  当前不是前 2/3 前缀观测实验，而是整体数据量尚未采满时的固定测试集实验。",
        "",
        "对照组设置:",
    ]
    for variant in variants:
        line = f"  {variant.name}: {variant.description}"
        if variant.shape_config is not None:
            line += (
                f" (mono={variant.shape_config.mono_lambda}, "
                f"slow={variant.shape_config.slow_lambda}, "
                f"mono_tolerance_ratio={variant.shape_config.mono_tolerance_ratio})"
            )
        lines.append(line)
    lines.extend(["", "CUDA:"])
    lines.extend([f"  {line}" for line in common.device_summary_lines()])
    return "\n".join(lines)


def _build_plan_text(
    family_name: str,
    suite_kind: str,
    baseline_source: str,
    variants: list[VariantSpec],
) -> str:
    lines = [
        "实验方案",
        "=" * 72,
        "",
        "一、基线继承关系",
        "-" * 72,
        baseline_source,
        "",
        "二、候选设置",
        "-" * 72,
    ]
    for variant in variants:
        lines.extend(
            [
                f"{variant.name}",
                f"  描述: {variant.description}",
                f"  模型: {variant.model_name}",
                f"  特征: {', '.join(variant.feature_spec.columns)}",
                f"  配置: seq_len={variant.train_config.seq_len}, epochs={variant.train_config.epochs}, "
                f"lr={variant.train_config.learning_rate}, wd={variant.train_config.weight_decay}",
            ]
        )
        if variant.shape_config is not None:
            lines.append(
                f"  Shape loss: mono={variant.shape_config.mono_lambda}, slow={variant.shape_config.slow_lambda}, "
                f"mono_tolerance_ratio={variant.shape_config.mono_tolerance_ratio}"
            )
    lines.extend(
        [
            "",
            "三、当前判优规则",
            "-" * 72,
            "先看 mean_life_abs_error 是否优于同家族 baseline。",
            "如果寿命误差改善，但 pressure 或 wear 明显恶化（超过 10%），记为条件有效。",
            "当前结果仅代表现阶段数据下的判断；未来约 30 组数据完成后需要按相同流程重新验证。",
        ]
    )
    return "\n".join(lines)


def _build_analysis(
    suite_kind: str,
    family_name: str,
    summary_df: pd.DataFrame,
    baseline_name: str,
) -> str:
    ranked = summary_df.sort_values("mean_life_abs_error").reset_index(drop=True)
    baseline = ranked[ranked["model"] == baseline_name].iloc[0]
    best = ranked.iloc[0]
    lines = [
        "结果说明与分析",
        "=" * 72,
        "",
        "一、总体排序",
        "-" * 72,
    ]
    for idx, row in enumerate(ranked.itertuples(index=False), start=1):
        lines.append(
            f"{idx}. {row.model} | pressure_MAE={float(row.mean_pressure_mae):.2f} | "
            f"wear_MAE={float(row.mean_wear_mae_um):.3f} | life_abs_error={float(row.mean_life_abs_error):.0f} | "
            f"判定={row.effectiveness}"
        )
    lines.extend(
        [
            "",
            "二、基线对照",
            "-" * 72,
            f"同家族 baseline 为 {baseline_name}，其寿命绝对误差约 {float(baseline['mean_life_abs_error']):.0f}。",
            f"当前最优候选为 {best['model']}，寿命绝对误差约 {float(best['mean_life_abs_error']):.0f}。",
        ]
    )

    if str(best["model"]) != baseline_name:
        delta = float(best["mean_life_abs_error"]) - float(baseline["mean_life_abs_error"])
        lines.append(f"相对 baseline 的寿命误差变化为 {delta:+.0f}。")
    else:
        lines.append("当前没有候选优于 baseline。")

    lines.extend(
        [
            "",
            "三、当前阶段解释",
            "-" * 72,
            f"1. 这轮 {suite_kind} 试跑是在固定测试集、数据尚未采满的阶段进行的。",
            "2. 当前结论主要用于判断这条增补模块是否值得继续，而不是最终定型。",
            "3. 等未来数据接近完整 30 组后，应按同一套固定测试集流程重新验证。",
            f"4. 后续跨家族比较时，应将 {family_name} 当前最优增补版与另一家族的最优增补版再做统一比较。",
        ]
    )
    return "\n".join(lines)


def run_variant_suite(
    *,
    output_dir: Path,
    experiment_name: str,
    experiment_date: str,
    family_name: str,
    suite_kind: str,
    purpose: str,
    baseline_source: str,
    variants: list[VariantSpec],
    chart_title_prefix: str,
) -> pd.DataFrame:
    common.require_cuda()
    common.set_seed(common.SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    for line in common.device_summary_lines():
        print(line)
    print(experiment_name)
    print(f"  family={family_name} | suite={suite_kind}")
    print()

    summary_df, case_tables = common.load_data()
    train_df, test_df = common.load_fixed_split(summary_df)
    train_tables = {str(row['file_name']): case_tables[str(row['file_name'])] for _, row in train_df.iterrows()}

    detail_path = output_dir / DETAIL_FILENAME
    if detail_path.exists():
        existing_df = pd.read_csv(detail_path)
        scan_rows = existing_df.to_dict("records")
        completed_pairs = {(str(row["test_case"]), str(row["model"])) for _, row in existing_df.iterrows()}
        print(f"Resume enabled: loaded {len(existing_df)} existing rows.")
    else:
        scan_rows = []
        completed_pairs: set[tuple[str, str]] = set()

    for variant in variants:
        variant_complete = all((str(test_row["source_file"]), variant.name) in completed_pairs for _, test_row in test_df.iterrows())
        if variant_complete:
            print(f"skip training | variant={variant.name} | all test cases already done")
            continue

        variant_seed = _stable_variant_seed(variant.name)
        common.set_seed(variant_seed)
        print(f"Training variant: {variant.name}")
        print(f"  seed={variant_seed}")
        train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, variant.train_config.seq_len)
        model, seq_scaler, target_scaler = common.train_model(
            model_name=variant.model_name,
            train_raw_seq=train_raw_seq,
            train_y=train_y,
            feature_spec=variant.feature_spec,
            config=variant.train_config,
            shape_config=variant.shape_config,
            train_case_names=train_case_names,
            train_step_indices=train_step_indices,
        )

        for idx, test_row in test_df.iterrows():
            test_case = str(test_row["source_file"])
            if (test_case, variant.name) in completed_pairs:
                print(f"skip existing | variant={variant.name} | test={test_case}")
                continue

            print(f"evaluate | variant={variant.name} | test={test_case}")
            test_file = str(test_row["file_name"])
            test_table = case_tables[test_file]
            test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, variant.train_config.seq_len)
            true_curve_df = common.threshold_ground_truth(test_table, common.WEAR_THRESHOLD_UM, test_row)
            true_life_actual = float(test_row["actual_life"])

            pressure_metrics = common.evaluate_pressure(
                model,
                seq_scaler,
                target_scaler,
                test_raw_seq,
                test_y,
                variant.feature_spec,
            )
            rollout_df, predicted_life = common.rollout_case(
                model,
                seq_scaler,
                target_scaler,
                test_table,
                common.WEAR_THRESHOLD_UM,
                common.REAL_WEAR_COEFF_MPA_INV,
                true_life_actual,
                variant.feature_spec,
                variant.train_config.seq_len,
            )
            curve_mae = common.wear_curve_mae(true_curve_df, rollout_df)
            scan_rows.append(
                {
                    "test_case": test_case,
                    "model": variant.name,
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
            completed_pairs.add((test_case, variant.name))
            pd.DataFrame(scan_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    detail_df = pd.DataFrame(scan_rows)
    summary_df_out = (
        detail_df.groupby("model")
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
    summary_df_out["effectiveness"] = _effectiveness_label(summary_df_out, variants[0].name)
    summary_df_out.to_csv(output_dir / SUMMARY_FILENAME, index=False, encoding="utf-8-sig")

    common.save_grouped_life_error_bar(detail_df, output_dir / CHART1_FILENAME, f"{chart_title_prefix}: Life Error by Test Case")
    common.save_summary_bar_chart(summary_df_out, output_dir / CHART2_FILENAME, f"{chart_title_prefix}: Summary Comparison")
    common.save_mean_predicted_life_chart(summary_df_out, detail_df, output_dir / CHART3_FILENAME, f"{chart_title_prefix}: Mean Predicted Life")

    (output_dir / NOTES_FILENAME).write_text(
        _build_notes(experiment_name, experiment_date, family_name, suite_kind, purpose, train_df, test_df, variants),
        encoding="utf-8",
    )
    (output_dir / PLAN_FILENAME).write_text(
        _build_plan_text(family_name, suite_kind, baseline_source, variants),
        encoding="utf-8",
    )
    (output_dir / ANALYSIS_FILENAME).write_text(
        _build_analysis(suite_kind, family_name, summary_df_out, variants[0].name),
        encoding="utf-8",
    )

    best_variant_name = str(summary_df_out.iloc[0]["model"])
    best_variant = next(variant for variant in variants if variant.name == best_variant_name)
    (output_dir / BEST_FILENAME).write_text(
        json.dumps(_variant_to_payload(best_variant), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_df_out
