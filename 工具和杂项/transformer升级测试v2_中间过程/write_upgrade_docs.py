from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MIDDLE_DIR = Path(__file__).resolve().parent
RESULT_DIR = ROOT_DIR / "结果" / "transformer升级测试v2"
BASELINE_DIR = ROOT_DIR / "结果" / "4.18v1基准测试"

CANDIDATE_SUMMARY = MIDDLE_DIR / "\u5019\u9009\u6c47\u603b.csv"
BASELINE_SUMMARY = BASELINE_DIR / "\u6c47\u603b_\u5404\u6a21\u578b\u5e73\u5747\u6307\u6807.csv"
FINAL_SUMMARY = RESULT_DIR / "\u6c47\u603b_\u5404\u6a21\u578b\u5e73\u5747\u6307\u6807.csv"

UPGRADE_PLAN = RESULT_DIR / "\u5347\u7ea7\u65b9\u6848.txt"
ANALYSIS_PATH = RESULT_DIR / "\u7ed3\u679c\u8bf4\u660e\u4e0e\u5206\u6790.txt"
TEST_NOTES = RESULT_DIR / "\u6d4b\u8bd5\u8bf4\u660e.txt"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(require_file(CANDIDATE_SUMMARY))
    baseline = pd.read_csv(require_file(BASELINE_SUMMARY))
    final_summary = pd.read_csv(require_file(FINAL_SUMMARY))
    return candidates, baseline, final_summary


def fmt_row(row: pd.Series) -> str:
    return (
        f"pressure_MAE={float(row['mean_pressure_mae']):.2f}, "
        f"wear_MAE={float(row['mean_wear_mae_um']):.3f}, "
        f"life_abs_error={float(row['mean_life_abs_error']):.0f}"
    )


def build_upgrade_plan(candidates: pd.DataFrame, baseline: pd.DataFrame, final_summary: pd.DataFrame) -> str:
    baseline_transformer = baseline[baseline["model"] == "Transformer"].iloc[0]
    final_row = final_summary.iloc[0]
    candidate_map = {row["round_id"]: row for _, row in candidates.iterrows()}

    comparisons = [
        ("round1_training_strategy", "\u8bad\u7ec3\u7b56\u7565/\u9a8c\u8bc1\u96c6/\u6b63\u5219\u5316\u5206\u652f", "baseline"),
        ("round1b_no_val", "\u8bad\u7ec3\u7b56\u7565\u5206\u652f\uff08\u53bb\u6389\u9a8c\u8bc1\u96c6\uff09", "baseline"),
        ("round2_seq10", "seq_len=10", "baseline"),
        ("round2b_seq10_mild", "seq_len=10 + \u6e29\u548c\u8bad\u7ec3\u7b56\u7565", "round2_seq10"),
        ("round3_capacity", "d_model=64, ff=128, 2\u5c42", "round2_seq10"),
        ("round3b_capacity", "d_model=64, ff=256, 3\u5c42", "round3_capacity"),
        ("round4_seq15", "seq_len=15 \u6700\u7ec8\u7ec4\u5408", "round3_capacity"),
    ]

    lines = [
        "\u5347\u7ea7\u65b9\u6848",
        "=" * 72,
        "",
        "\u4e00\u3001\u57fa\u7ebf",
        "-" * 72,
        "\u672c\u8f6e Transformer \u57fa\u7ebf\u6765\u81ea 4.18v1 \u65b0 17-run \u57fa\u51c6\u6d4b\u8bd5\u4e2d\u7684 Transformer v1 \u7ed3\u679c\u3002",
        f"\u57fa\u7ebf\u53c2\u6570: seq_len=6, d_model=32, nhead=4, layers=2, ff=64, epochs=1200, lr=1e-3, wd=1e-6",
        f"\u57fa\u7ebf\u6307\u6807: {fmt_row(baseline_transformer)}",
        "",
        "\u4e8c\u3001\u5019\u9009\u94fe\u8def\u4e0e\u53d6\u820d",
        "-" * 72,
    ]

    for round_id, desc, ref_id in comparisons:
        row = candidate_map[round_id]
        ref_row = baseline_transformer if ref_id == "baseline" else candidate_map[ref_id]
        delta = float(row["mean_life_abs_error"]) - float(ref_row["mean_life_abs_error"])
        keep = "\u4fdd\u7559" if delta < 0 else "\u4e0d\u4fdd\u7559"
        lines.extend(
            [
                f"{round_id}: {desc}",
                f"  \u914d\u7f6e: seq_len={int(row['seq_len'])}, d_model={int(row['d_model'])}, layers={int(row['num_layers'])}, ff={int(row['dim_ff'])}, "
                f"dropout={float(row['dropout']):.2f}, epochs={int(row['epochs'])}, lr={float(row['learning_rate']):.1e}, wd={float(row['weight_decay']):.1e}",
                f"  \u6307\u6807: {fmt_row(row)}",
                f"  \u53c2\u8003\u5bf9\u8c61: {'4.18v1 baseline Transformer' if ref_id == 'baseline' else ref_id}",
                f"  \u5bff\u547d\u8bef\u5dee\u53d8\u5316: {delta:+.0f}",
                f"  \u7ed3\u8bba: {keep}",
            ]
        )

    lines.extend(
        [
            "",
            "\u4e09\u3001\u6700\u7ec8\u5b9a\u7248",
            "-" * 72,
            "Transformer v2 \u6700\u7ec8\u53c2\u6570\u91c7\u7528 round4_seq15 \u914d\u7f6e\uff1a",
            "  seq_len=15, d_model=64, nhead=4, layers=2, ff=128, dropout=0.0, epochs=1200, lr=1e-3, wd=1e-6",
            f"\u6700\u7ec8\u6307\u6807: {fmt_row(final_row)}",
        ]
    )
    return "\n".join(lines)


def build_analysis(candidates: pd.DataFrame, baseline: pd.DataFrame, final_summary: pd.DataFrame) -> str:
    baseline_transformer = baseline[baseline["model"] == "Transformer"].iloc[0]
    final_row = final_summary.iloc[0]

    merged = baseline.copy()
    merged = pd.concat(
        [
            merged,
            pd.DataFrame(
                [
                    {
                        "model": "Transformer_v2",
                        "mean_pressure_mae": final_row["mean_pressure_mae"],
                        "mean_wear_mae_um": final_row["mean_wear_mae_um"],
                        "mean_life_abs_error": final_row["mean_life_abs_error"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ranking = merged.sort_values("mean_life_abs_error").reset_index(drop=True)

    best_candidate = candidates.sort_values("mean_life_abs_error").iloc[0]
    seq10 = candidates[candidates["round_id"] == "round2_seq10"].iloc[0]
    capacity = candidates[candidates["round_id"] == "round3_capacity"].iloc[0]

    lines = [
        "\u7ed3\u679c\u8bf4\u660e\u4e0e\u5206\u6790",
        "=" * 72,
        "",
        "\u4e00\u3001\u6700\u7ec8\u7ed3\u679c",
        "-" * 72,
        f"4.18v1 \u57fa\u7ebf Transformer: {fmt_row(baseline_transformer)}",
        f"Transformer v2 \u6700\u7ec8\u7ed3\u679c: {fmt_row(final_row)}",
        (
            f"\u5bff\u547d\u8bef\u5dee\u53d8\u5316: "
            f"{float(final_row['mean_life_abs_error']) - float(baseline_transformer['mean_life_abs_error']):+.0f}"
        ),
        "",
        "\u4e8c\u3001\u540e\u9a8c\u6392\u540d\uff08\u542b 4.18v1 \u4e94\u6a21\u578b\u57fa\u7ebf\uff09",
        "-" * 72,
    ]
    for idx, row in enumerate(ranking.itertuples(index=False), start=1):
        lines.append(
            f"{idx}. {row.model:<14s} | pressure_MAE={float(row.mean_pressure_mae):6.2f} | "
            f"wear_MAE={float(row.mean_wear_mae_um):6.3f} | life_abs_error={float(row.mean_life_abs_error):8.0f}"
        )

    lines.extend(
        [
            "",
            "\u4e09\u3001\u5347\u7ea7\u94fe\u8def\u89c2\u5bdf",
            "-" * 72,
            (
                f"1. seq_len \u662f\u6700\u5148\u89c1\u6548\u7684\u53d8\u91cf\uff1around2 \u76f8\u6bd4 baseline \u7684 "
                f"\u5bff\u547d\u8bef\u5dee\u53d8\u5316 {float(seq10['mean_life_abs_error']) - float(baseline_transformer['mean_life_abs_error']):+.0f}\u3002"
            ),
            (
                f"2. \u5bb9\u91cf\u5347\u7ea7\uff08round3\uff09\u5728 seq_len=10 \u57fa\u7840\u4e0a\u8fdb\u4e00\u6b65\u53d8\u5316 "
                f"{float(capacity['mean_life_abs_error']) - float(seq10['mean_life_abs_error']):+.0f}\u3002"
            ),
            (
                f"3. \u6700\u7ec8\u6700\u4f18\u5019\u9009\u662f {best_candidate['round_id']} "
                f"\uff0c\u5bf9\u5e94\u6307\u6807 {fmt_row(best_candidate)}\u3002"
            ),
            "4. \u8bad\u7ec3\u7b56\u7565\u548c\u6b63\u5219\u5316\u5206\u652f\u662f\u5728\u65b0 17-run \u6570\u636e\u4e0a\u91cd\u65b0\u8bc4\u4f30\u7684\uff0c\u672a\u76f4\u63a5\u6cbf\u7528\u65e7\u7248\u7ed3\u8bba\u3002",
            "",
            "\u56db\u3001\u6587\u4ef6\u8bf4\u660e",
            "-" * 72,
            "- run_benchmark.py: \u6700\u7ec8 Transformer v2 \u6b63\u5f0f\u8fd0\u884c\u811a\u672c",
            "- \u8be6\u7ec6\u7ed3\u679c_\u5404\u6298.csv: 17 \u6298\u6700\u7ec8 Transformer v2 \u6307\u6807",
            "- \u6c47\u603b_\u5404\u6a21\u578b\u5e73\u5747\u6307\u6807.csv: Transformer v2 \u6700\u7ec8\u6c47\u603b",
            "- \u56fe1_\u5404\u5de5\u51b5\u5bff\u547d\u8bef\u5dee\u5206\u7ec4\u67f1\u72b6\u56fe.png: \u4e0e 4.18v1 \u57fa\u51c6\u6a21\u578b\u7684\u5de5\u51b5\u7ea7\u5bf9\u6bd4",
            "- \u56fe2_\u6a21\u578b\u5bf9\u6bd4\u67f1\u72b6\u56fe.png: \u4e0e 4.18v1 \u4e94\u6a21\u578b\u7684\u603b\u4f53\u5bf9\u6bd4",
        ]
    )
    return "\n".join(lines)


def build_test_notes(candidates: pd.DataFrame, baseline: pd.DataFrame, final_summary: pd.DataFrame) -> str:
    baseline_transformer = baseline[baseline["model"] == "Transformer"].iloc[0]
    final_row = final_summary.iloc[0]
    lines = [
        "\u6d4b\u8bd5\u8bf4\u660e",
        "=" * 60,
        "Data source: \u63d0\u53d6\u7ed3\u679c_\u5904\u7406\u540e",
        "Validation: 17-fold leave-one-run",
        "Search chain: round1, round1b, round2, round2b, round3, round3b, round4",
        "Final config: seq_len=15, d_model=64, nhead=4, layers=2, ff=128, epochs=1200, lr=1e-3, wd=1e-6",
        "",
        f"Baseline transformer: {fmt_row(baseline_transformer)}",
        f"Final transformer v2: {fmt_row(final_row)}",
        "",
        "Candidate table path:",
        f"  {CANDIDATE_SUMMARY}",
        f"  total candidates={len(candidates)}",
    ]
    return "\n".join(lines)


def main() -> None:
    candidates, baseline, final_summary = load_tables()
    UPGRADE_PLAN.write_text(build_upgrade_plan(candidates, baseline, final_summary), encoding="utf-8")
    ANALYSIS_PATH.write_text(build_analysis(candidates, baseline, final_summary), encoding="utf-8")
    TEST_NOTES.write_text(build_test_notes(candidates, baseline, final_summary), encoding="utf-8")
    print(f"Wrote: {UPGRADE_PLAN}")
    print(f"Wrote: {ANALYSIS_PATH}")
    print(f"Wrote: {TEST_NOTES}")


if __name__ == "__main__":
    main()
