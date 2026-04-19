from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchmark_network_architectures as bench
from train_real_wear_models import EPS, comparison_dir, load_cases, load_processing_config, threshold_ground_truth


MODEL_NAME = "Transformer"
SCAN_EPOCHS = 1500
HIGH_ERROR_COUNT = 4


def output_dir() -> Path:
    path = comparison_dir() / "transformer_cross_test_scan"
    path.mkdir(parents=True, exist_ok=True)
    return path


def eligible_test_rows(summary_df: pd.DataFrame, threshold_um: float) -> pd.DataFrame:
    mask = summary_df["has_measured_pressure"].astype(bool) & (summary_df["final_wear_um"].astype(float) >= float(threshold_um))
    return summary_df.loc[mask].sort_values("actual_life").reset_index(drop=True)


def coverage_features(train_rows: pd.DataFrame, test_row: pd.Series, threshold_um: float) -> dict:
    train_rows = train_rows.reset_index(drop=True)
    test_f = float(test_row["F"])
    test_d = float(test_row["D"])
    test_cr = float(test_row["Cr"])

    train_fdcr = train_rows[["F", "D", "Cr"]].to_numpy(dtype=float)
    test_vec = np.array([test_f, test_d, test_cr], dtype=float)
    scale = np.ptp(train_fdcr, axis=0)
    scale[scale < EPS] = 1.0
    nearest_distance = float(np.min(np.sqrt(np.sum(((train_fdcr - test_vec) / scale) ** 2, axis=1))))

    same_d = np.isclose(train_rows["D"].to_numpy(dtype=float), test_d)
    same_cr = np.isclose(train_rows["Cr"].to_numpy(dtype=float), test_cr)
    same_f = np.isclose(train_rows["F"].to_numpy(dtype=float), test_f)

    return {
        "train_case_count": int(len(train_rows)),
        "train_F_min": float(train_rows["F"].min()),
        "train_F_max": float(train_rows["F"].max()),
        "train_D_min": float(train_rows["D"].min()),
        "train_D_max": float(train_rows["D"].max()),
        "train_Cr_min": float(train_rows["Cr"].min()),
        "train_Cr_max": float(train_rows["Cr"].max()),
        "same_D_count": int(same_d.sum()),
        "same_Cr_count": int(same_cr.sum()),
        "same_F_count": int(same_f.sum()),
        "same_DCr_count": int((same_d & same_cr).sum()),
        "same_DF_count": int((same_d & same_f).sum()),
        "same_FCr_count": int((same_f & same_cr).sum()),
        "is_F_extrapolation": bool(test_f < float(train_rows["F"].min()) - EPS or test_f > float(train_rows["F"].max()) + EPS),
        "is_D_extrapolation": bool(test_d < float(train_rows["D"].min()) - EPS or test_d > float(train_rows["D"].max()) + EPS),
        "is_Cr_extrapolation": bool(test_cr < float(train_rows["Cr"].min()) - EPS or test_cr > float(train_rows["Cr"].max()) + EPS),
        "nearest_param_distance": nearest_distance,
        "threshold_margin_um": float(test_row["final_wear_um"]) - float(threshold_um),
    }


def reason_text(row: pd.Series, distance_cutoff: float) -> str:
    reasons: list[str] = []

    if bool(row["is_F_extrapolation"]) or bool(row["is_D_extrapolation"]) or bool(row["is_Cr_extrapolation"]):
        axes: list[str] = []
        if bool(row["is_F_extrapolation"]):
            axes.append("F")
        if bool(row["is_D_extrapolation"]):
            axes.append("D")
        if bool(row["is_Cr_extrapolation"]):
            axes.append("Cr")
        reasons.append(f"测试工况在 {'/'.join(axes)} 上超出训练范围")

    if int(row["same_DCr_count"]) == 0:
        reasons.append("训练集中没有同尺寸同间隙参照，局部磨损形状更难锁定")
    elif int(row["same_DCr_count"]) == 1:
        reasons.append("训练集中同尺寸同间隙参照只有 1 组，局部约束偏弱")

    if int(row["same_DF_count"]) == 0:
        reasons.append("训练集中缺少同尺寸同载荷对照，间隙影响与载荷影响不易分离")

    if int(row["same_D_count"]) <= 1:
        reasons.append("同尺寸样本数量偏少")

    if float(row["nearest_param_distance"]) >= float(distance_cutoff):
        reasons.append("该工况距离训练样本簇较远，属于弱插值或近外推区域")

    if float(row["threshold_margin_um"]) <= 0.30:
        reasons.append("最终磨损只略高于 5 微米，阈值寿命对尾段斜率更敏感")

    if not reasons:
        reasons.append("整体属于训练分布内部，误差更可能来自样本量偏少和压力序列噪声")
    return "；".join(reasons)


def save_error_bar(scan_df: pd.DataFrame, out_path: Path) -> None:
    ranked = scan_df.sort_values("life_abs_error", ascending=False).reset_index(drop=True)
    labels = ranked["test_case"].tolist()
    values = ranked["life_abs_error"].to_numpy(dtype=float)
    colors = np.where(ranked["error_rank"].to_numpy(dtype=int) <= HIGH_ERROR_COUNT, "#dc2626", "#2563eb")

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    fig.patch.set_facecolor("#f4f1eb")
    ax.set_facecolor("#fbfaf7")
    bars = ax.bar(labels, values, color=colors, edgecolor="#ffffff", linewidth=1.2, width=0.68)

    ax.set_title("Transformer Leave-One-Run Test Error", fontsize=14, fontweight="bold", color="#14202b", pad=12)
    ax.set_ylabel("Life absolute error (cycles)", fontsize=11, color="#14202b")
    ax.set_xlabel("Held-out test run", fontsize=10, color="#5f6b76")
    ax.grid(axis="y", color="#d8dee6", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9d1da")
    ax.spines["bottom"].set_color("#c9d1da")
    ax.tick_params(axis="x", rotation=18, labelsize=10, colors="#14202b")
    ax.tick_params(axis="y", labelsize=10, colors="#14202b")

    y_max = max(float(values.max()) if len(values) else 1.0, 1.0) * 1.15
    ax.set_ylim(0.0, y_max)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + y_max * 0.012,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#14202b",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    bench.set_seed(bench.SEED)
    bench.EPOCHS = SCAN_EPOCHS

    config = load_processing_config()
    threshold_um = float(config["wear_threshold_um"])
    real_k = float(config["real_wear_coeff_mpa_inv"])

    summary_df, case_tables = load_cases()
    test_rows = eligible_test_rows(summary_df, threshold_um)

    scan_rows: list[dict] = []
    for test_row in test_rows.itertuples(index=False):
        test_series = pd.Series(test_row._asdict())
        test_file = str(test_series["file_name"])
        test_source = str(test_series["source_file"])
        train_summary = summary_df.loc[summary_df["file_name"] != test_file].reset_index(drop=True)
        train_tables = {name: table for name, table in case_tables.items() if name != test_file}
        test_table = case_tables[test_file]

        train_seq, train_y = bench.build_sequence_dataset(train_tables, bench.SEQ_LEN)
        test_seq, test_y = bench.build_sequence_dataset({test_file: test_table}, bench.SEQ_LEN)
        true_curve_df = threshold_ground_truth(test_table, threshold_um, test_series)
        true_life_actual = float(test_series["actual_life"])

        print(f"Transformer leave-one-run training | test={test_source}")
        model, seq_scaler, target_scaler = bench.train_model(MODEL_NAME, train_seq, train_y)
        pressure_metrics = bench.evaluate_pressure(model, seq_scaler, target_scaler, test_seq, test_y)
        rollout_df, predicted_life = bench.rollout_case(model, seq_scaler, target_scaler, test_table, threshold_um, real_k, true_life_actual)
        wear_mae = bench.wear_curve_mae(true_curve_df, rollout_df)
        coverage = coverage_features(train_summary, test_series, threshold_um)

        scan_rows.append(
            {
                "test_case": test_source,
                "pressure_mae": pressure_metrics["pressure_mae"],
                "pressure_rmse": pressure_metrics["pressure_rmse"],
                "pressure_mape": pressure_metrics["pressure_mape"],
                "wear_mae_um": wear_mae,
                "predicted_life": predicted_life,
                "true_life": true_life_actual,
                "life_abs_error": abs(predicted_life - true_life_actual),
                "life_rel_error": abs(predicted_life - true_life_actual) / max(true_life_actual, EPS),
                **coverage,
            }
        )

    scan_df = pd.DataFrame(scan_rows).sort_values("true_life").reset_index(drop=True)
    scan_df["error_rank"] = scan_df["life_abs_error"].rank(method="dense", ascending=False).astype(int)
    distance_cutoff = float(scan_df["nearest_param_distance"].quantile(0.70)) if len(scan_df) > 1 else float(scan_df["nearest_param_distance"].max())
    scan_df["suspected_reason"] = scan_df.apply(lambda row: reason_text(row, distance_cutoff), axis=1)

    out_dir = output_dir()
    scan_df.to_csv(out_dir / "transformer_leave_one_run_errors.csv", index=False, encoding="utf-8-sig")

    brief_df = scan_df[
        [
            "test_case",
            "true_life",
            "predicted_life",
            "life_abs_error",
            "life_rel_error",
            "pressure_mae",
            "wear_mae_um",
            "suspected_reason",
        ]
    ].copy()
    brief_df.to_csv(out_dir / "transformer_leave_one_run_brief.csv", index=False, encoding="utf-8-sig")

    summary_stats = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "eligible_test_case_count": int(len(scan_df)),
                "mean_pressure_mae": float(scan_df["pressure_mae"].mean()),
                "mean_wear_mae_um": float(scan_df["wear_mae_um"].mean()),
                "mean_life_abs_error": float(scan_df["life_abs_error"].mean()),
                "median_life_abs_error": float(scan_df["life_abs_error"].median()),
                "mean_life_rel_error": float(scan_df["life_rel_error"].mean()),
            }
        ]
    )
    summary_stats.to_csv(out_dir / "transformer_leave_one_run_summary.csv", index=False, encoding="utf-8-sig")

    high_error_df = scan_df.sort_values("life_abs_error", ascending=False).head(HIGH_ERROR_COUNT).reset_index(drop=True)
    analysis_lines = [
        f"Transformer leave-one-run scan on measured runs with final wear >= {threshold_um:.2f} um.",
        f"Training epochs per fold: {SCAN_EPOCHS}",
        "",
        "High-error runs and suspected reasons:",
    ]
    for row in high_error_df.itertuples(index=False):
        analysis_lines.extend(
            [
                f"- {row.test_case}: |life error|={row.life_abs_error:.1f} cycles, wear MAE={row.wear_mae_um:.4f} um, pressure MAE={row.pressure_mae:.3f} MPa",
                f"  reason: {row.suspected_reason}",
            ]
        )
    (out_dir / "high_error_analysis.txt").write_text("\n".join(analysis_lines), encoding="utf-8")

    save_error_bar(scan_df, out_dir / "transformer_leave_one_run_life_error.png")

    print("Transformer leave-one-run scan complete.")
    print(summary_stats.to_string(index=False))
    print(scan_df[["test_case", "true_life", "predicted_life", "life_abs_error", "life_rel_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
