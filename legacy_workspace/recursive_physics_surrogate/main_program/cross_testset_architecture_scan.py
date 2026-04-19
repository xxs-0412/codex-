from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchmark_network_architectures as bench
from train_real_wear_models import load_cases, load_processing_config, threshold_ground_truth


SCAN_MODELS = ["FNN", "GRU", "Transformer"]
SCAN_EPOCHS = 1200


def output_dir() -> Path:
    path = bench.comparison_dir() / "cross_testset_scan"
    path.mkdir(parents=True, exist_ok=True)
    return path


def eligible_test_rows(summary_df: pd.DataFrame, threshold_um: float) -> pd.DataFrame:
    mask = summary_df["has_measured_pressure"].astype(bool) & (summary_df["final_wear_um"].astype(float) >= float(threshold_um))
    return summary_df.loc[mask].sort_values("actual_life").reset_index(drop=True)


def save_grouped_life_error_plot(scan_df: pd.DataFrame, out_path: Path) -> None:
    pivot = scan_df.pivot(index="test_case", columns="model", values="life_abs_error").sort_index()
    tests = pivot.index.tolist()
    models = pivot.columns.tolist()
    x = np.arange(len(tests), dtype=float)
    width = 0.23

    colors = [bench.MODEL_COLORS.get(model, "#64748b") for model in models]
    fig, ax = plt.subplots(figsize=(13.8, 5.8))
    fig.patch.set_facecolor("#f4f1eb")
    ax.set_facecolor("#fbfaf7")

    for idx, model in enumerate(models):
        vals = pivot[model].to_numpy(dtype=float)
        bars = ax.bar(x + (idx - (len(models) - 1) / 2.0) * width, vals, width=width, color=colors[idx], edgecolor="#ffffff", linewidth=1.0, label=model)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.0f}", ha="center", va="bottom", fontsize=7.8)

    ax.set_xticks(x)
    ax.set_xticklabels(tests, rotation=18)
    ax.set_ylabel("Life absolute error (cycles)")
    ax.set_title("Cross-Test-Set Comparison of Life Error")
    ax.grid(axis="y", color="#d8dee6", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=len(models), loc="upper center")
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
        test_file = str(test_row.file_name)
        test_source = str(test_row.source_file)
        train_tables = {name: table for name, table in case_tables.items() if name != test_file}
        test_table = case_tables[test_file]
        train_seq, train_y = bench.build_sequence_dataset(train_tables, bench.SEQ_LEN)
        test_seq, test_y = bench.build_sequence_dataset({test_file: test_table}, bench.SEQ_LEN)
        true_curve_df = threshold_ground_truth(test_table, threshold_um, pd.Series(test_row._asdict()))
        true_life_actual = float(test_row.actual_life)

        for model_name in SCAN_MODELS:
            print(f"test={test_source} | model={model_name}")
            model, seq_scaler, target_scaler = bench.train_model(model_name, train_seq, train_y)
            pressure_metrics = bench.evaluate_pressure(model, seq_scaler, target_scaler, test_seq, test_y)
            rollout_df, predicted_life = bench.rollout_case(model, seq_scaler, target_scaler, test_table, threshold_um, real_k, true_life_actual)
            wear_mae = bench.wear_curve_mae(true_curve_df, rollout_df)

            scan_rows.append(
                {
                    "test_case": test_source,
                    "model": model_name,
                    "pressure_mae": pressure_metrics["pressure_mae"],
                    "pressure_rmse": pressure_metrics["pressure_rmse"],
                    "pressure_mape": pressure_metrics["pressure_mape"],
                    "wear_mae_um": wear_mae,
                    "predicted_life": predicted_life,
                    "true_life": true_life_actual,
                    "life_abs_error": abs(predicted_life - true_life_actual),
                    "life_rel_error": abs(predicted_life - true_life_actual) / max(true_life_actual, bench.EPS),
                }
            )

    scan_df = pd.DataFrame(scan_rows)
    out_dir = output_dir()
    scan_df.to_csv(out_dir / "cross_testset_architecture_errors.csv", index=False)

    summary_stats = (
        scan_df.groupby("model")
        .agg(
            mean_pressure_mae=("pressure_mae", "mean"),
            mean_wear_mae_um=("wear_mae_um", "mean"),
            mean_life_abs_error=("life_abs_error", "mean"),
            median_life_abs_error=("life_abs_error", "median"),
            mean_life_rel_error=("life_rel_error", "mean"),
        )
        .reset_index()
        .sort_values("mean_life_abs_error")
    )
    win_table = (
        scan_df.loc[scan_df.groupby("test_case")["life_abs_error"].idxmin()]
        .groupby("model")
        .size()
        .rename("life_error_win_count")
        .reset_index()
    )
    summary_stats = summary_stats.merge(win_table, on="model", how="left").fillna({"life_error_win_count": 0})
    summary_stats.to_csv(out_dir / "cross_testset_architecture_summary.csv", index=False)

    save_grouped_life_error_plot(scan_df, out_dir / "cross_testset_life_error_grouped.png")

    notes = [
        f"Exploratory cross-test-set scan with {SCAN_MODELS}",
        f"Training epochs per model: {SCAN_EPOCHS}",
        "Purpose: check whether Transformer still keeps an advantage when the held-out test case changes.",
        "Only real measured data are used; no synthetic virtual samples are added.",
        "These results are for internal model selection, not the final presentation figure.",
    ]
    (out_dir / "scan_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    print("Cross-test-set scan complete.")
    print(summary_stats.to_string(index=False))


if __name__ == "__main__":
    main()
