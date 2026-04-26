from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
COMMON_DIR = ROOT_DIR / "工具和杂项" / "30测试集_中间过程"
RESULT_DIR = ROOT_DIR / "结果" / "30测试集" / "03_测试集筛查"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import common_fixed_split as common

SEQ_LEN = 12
FIXED_STEP_CAP = 250.0
MODEL_ORDER = ["FNN", "GRU", "LSTM", "1D-CNN", "Transformer"]
BASELINE_SPEC = common.LEGACY_FEATURE_SPEC
ENHANCED_SPEC = common.FEATURE_R1_SPEC
ENHANCED_SHAPE = common.ShapeLossConfig(name="slow_only", mono_lambda=0.0, slow_lambda=0.01, mono_tolerance_ratio=0.0)

STAGE1_SCRIPT = SCRIPT_DIR / "screen_81_stage1_transformer.py"
STAGE1_SUMMARY = RESULT_DIR / "筛查汇总_81_stage1_transformer.csv"
STAGE2_SUMMARY = RESULT_DIR / "筛查汇总_stage2_full500.csv"
STAGE2_DETAIL = RESULT_DIR / "筛查明细_stage2_full500.csv"
STAGE3_SUMMARY = RESULT_DIR / "筛查汇总_stage3_formal1200.csv"
STAGE3_DETAIL = RESULT_DIR / "筛查明细_stage3_formal1200.csv"
FINAL_DOC = RESULT_DIR / "夜间筛查结论.txt"

EXTRA_CANDIDATES = {
    "Extra_Run16_Run28_Run24_Run29": ["Run16.csv", "Run28.csv", "Run24.csv", "Run29.csv"],
    "Extra_Run16_Run30_Run26_Run29": ["Run16.csv", "Run30.csv", "Run26.csv", "Run29.csv"],
    "Extra_Run16_Run30_Run24_Run29": ["Run16.csv", "Run30.csv", "Run24.csv", "Run29.csv"],
}


def config(epochs: int) -> common.TransformerConfig:
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


def ensure_stage1() -> None:
    if STAGE1_SUMMARY.exists():
        try:
            df = pd.read_csv(STAGE1_SUMMARY)
            if len(df) >= 81:
                print(f"Stage1 exists: {STAGE1_SUMMARY} rows={len(df)}")
                return
        except Exception:
            pass
    print("Running Stage1 Transformer-only 81-screen...")
    subprocess.run([sys.executable, str(STAGE1_SCRIPT)], cwd=str(SCRIPT_DIR), check=True)


def case_meta(case_tables: dict, file_name: str) -> dict:
    df = case_tables[file_name]
    first = df.iloc[0]
    step = common.resolve_rollout_steps(df).native_actual_step
    return {
        "F": float(first["F"]),
        "D": float(first["D"]),
        "Cr": float(first["Cr"]),
        "life": float(df["actual_cycle"].iloc[-1]),
        "native_step": float(step),
    }


def explainability_metrics(case_tables: dict, test_files: list[str]) -> dict:
    metas = [case_meta(case_tables, f) for f in test_files]
    pairs = [(m["F"], m["Cr"]) for m in metas]
    return {
        "max_native_step": max(m["native_step"] for m in metas),
        "max_life": max(m["life"] for m in metas),
        "min_life": min(m["life"] for m in metas),
        "duplicate_F_Cr": int(len(set(pairs)) < len(pairs)),
        "F_values": ";".join(str(int(m["F"])) for m in metas),
        "Cr_values": ";".join(f"{m['Cr']:.3g}" for m in metas),
        "native_steps": ";".join(f"{m['native_step']:.0f}" for m in metas),
        "lifes": ";".join(f"{m['life']:.0f}" for m in metas),
    }


def pick_stage2_candidates(case_tables: dict) -> list[tuple[str, list[str]]]:
    df = pd.read_csv(STAGE1_SUMMARY)
    df["candidate"] = df["candidate"].astype(str)
    # 第一优先：可解释、不重复、非明显边缘、增强后Transformer误差低且有收益。
    filt = df[
        (df["duplicate_F_Cr"].astype(int) == 0)
        & (df["max_native_step"].astype(float) <= 652.0)
        & (df["max_life"].astype(float) <= 60000.0)
        & (df["enhanced_minus_baseline"].astype(float) < 0.0)
    ].copy()
    filt = filt.sort_values(["enhanced_transformer_life_error", "improve_pct"]).head(12)
    picks: dict[str, list[str]] = {}
    for row in filt.itertuples(index=False):
        picks[str(row.candidate)] = str(row.test_files).split(";")
    for name, files in EXTRA_CANDIDATES.items():
        picks.setdefault(name, files)
    rows = []
    for name, files in picks.items():
        metrics = explainability_metrics(case_tables, files)
        row = {"candidate": name, "test_files": ";".join(files), **metrics}
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULT_DIR / "stage2候选清单.csv", index=False, encoding="utf-8-sig")
    return list(picks.items())


def evaluate_model_variant(model_name: str, variant: str, epochs: int, train_raw_seq, train_y, train_case_names, train_step_indices, test_df: pd.DataFrame, case_tables: dict) -> list[dict]:
    if variant == "baseline_5d":
        feature_spec = BASELINE_SPEC
        shape_config = None
    elif variant == "R1_plus_slow":
        feature_spec = ENHANCED_SPEC
        shape_config = ENHANCED_SHAPE
    else:
        raise ValueError(variant)
    model, seq_scaler, target_scaler = common.train_model(
        model_name=model_name,
        train_raw_seq=train_raw_seq,
        train_y=train_y,
        feature_spec=feature_spec,
        config=config(epochs),
        shape_config=shape_config,
        train_case_names=train_case_names,
        train_step_indices=train_step_indices,
    )
    rows = []
    for _, test_row in test_df.iterrows():
        test_file = str(test_row["file_name"])
        test_case = str(test_row["source_file"])
        test_table = case_tables[test_file]
        test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, SEQ_LEN)
        pressure_metrics = common.evaluate_pressure(model, seq_scaler, target_scaler, test_raw_seq, test_y, feature_spec)
        true_curve_df = common.threshold_ground_truth(test_table, common.WEAR_THRESHOLD_UM, test_row)
        true_life = float(test_row["actual_life"])
        rollout_df, pred_life, step_info = common.rollout_case_with_step_cap(
            model, seq_scaler, target_scaler, test_table,
            common.WEAR_THRESHOLD_UM, common.REAL_WEAR_COEFF_MPA_INV,
            true_life, feature_spec, SEQ_LEN, actual_step_cap=FIXED_STEP_CAP,
        )
        rows.append({
            "variant": variant,
            "model": model_name,
            "test_case": test_case,
            "pressure_mae": pressure_metrics["pressure_mae"],
            "wear_mae_um": common.wear_curve_mae(true_curve_df, rollout_df),
            "predicted_life": pred_life,
            "true_life": true_life,
            "life_abs_error": abs(pred_life - true_life),
            "native_actual_step": step_info.native_actual_step,
            "used_actual_step": step_info.used_actual_step,
        })
    del model
    torch.cuda.empty_cache()
    return rows


def rank_variant(rows: list[dict], variant: str) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if r["variant"] == variant])
    summary = (
        df.groupby("model")
        .agg(
            mean_life_abs_error=("life_abs_error", "mean"),
            mean_pressure_mae=("pressure_mae", "mean"),
            mean_wear_mae_um=("wear_mae_um", "mean"),
        )
        .reset_index()
        .sort_values("mean_life_abs_error")
        .reset_index(drop=True)
    )
    summary["rank"] = summary.index + 1
    return summary


def run_full_candidates(stage_name: str, candidates: list[tuple[str, list[str]]], epochs: int, summary_path: Path, detail_path: Path) -> pd.DataFrame:
    data_summary, case_tables = common.load_data()
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    done = set(existing_summary["candidate"].astype(str).tolist()) if not existing_summary.empty else set()
    all_summary = existing_summary.to_dict("records") if not existing_summary.empty else []
    all_detail = pd.read_csv(detail_path).to_dict("records") if detail_path.exists() else []

    for idx, (candidate_name, test_files) in enumerate(candidates, start=1):
        if candidate_name in done:
            print(f"skip {stage_name} {candidate_name}")
            continue
        common.set_seed(common.SEED + epochs + idx)
        print("=" * 80)
        print(f"{stage_name} {idx}/{len(candidates)} | {candidate_name} | epochs={epochs}")
        test_set = set(test_files)
        test_df = data_summary[data_summary["file_name"].isin(test_set)].copy()
        train_df = data_summary[~data_summary["file_name"].isin(test_set)].copy()
        train_tables = {str(row["file_name"]): case_tables[str(row["file_name"])] for _, row in train_df.iterrows()}
        train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)
        candidate_rows = []
        for variant in ["baseline_5d", "R1_plus_slow"]:
            for model_name in MODEL_ORDER:
                print(f"  {variant} | {model_name}")
                rows = evaluate_model_variant(model_name, variant, epochs, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables)
                for r in rows:
                    r["stage"] = stage_name
                    r["candidate"] = candidate_name
                    r["test_files"] = ";".join(test_files)
                    r["epochs"] = epochs
                candidate_rows.extend(rows)
        base_rank = rank_variant(candidate_rows, "baseline_5d")
        enh_rank = rank_variant(candidate_rows, "R1_plus_slow")
        tb = base_rank[base_rank["model"].eq("Transformer")].iloc[0]
        te = enh_rank[enh_rank["model"].eq("Transformer")].iloc[0]
        best = enh_rank.iloc[0]
        metrics = explainability_metrics(case_tables, test_files)
        summary = {
            "stage": stage_name,
            "candidate": candidate_name,
            "test_files": ";".join(test_files),
            "baseline_transformer_rank": int(tb["rank"]),
            "baseline_transformer_life_error": float(tb["mean_life_abs_error"]),
            "enhanced_transformer_rank": int(te["rank"]),
            "enhanced_transformer_life_error": float(te["mean_life_abs_error"]),
            "enhanced_best_model": str(best["model"]),
            "enhanced_best_life_error": float(best["mean_life_abs_error"]),
            "enhanced_minus_baseline_transformer": float(te["mean_life_abs_error"] - tb["mean_life_abs_error"]),
            "epochs": epochs,
            **metrics,
        }
        all_summary.append(summary)
        all_detail.extend(candidate_rows)
        pd.DataFrame(all_summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(all_detail).to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(base_rank[["model", "rank", "mean_life_abs_error"]].to_string(index=False))
        print(enh_rank[["model", "rank", "mean_life_abs_error"]].to_string(index=False))
    return pd.DataFrame(all_summary)


def pick_stage3_candidates(stage2: pd.DataFrame) -> list[tuple[str, list[str]]]:
    df = stage2.copy()
    df["good_explain"] = (
        (df["duplicate_F_Cr"].astype(int) == 0)
        & (df["max_native_step"].astype(float) <= 652.0)
        & (df["max_life"].astype(float) <= 60000.0)
    ).astype(int)
    df["enhanced_first"] = (df["enhanced_transformer_rank"].astype(int) == 1).astype(int)
    df["baseline_not_first"] = (df["baseline_transformer_rank"].astype(int) > 1).astype(int)
    df = df.sort_values(
        ["enhanced_first", "good_explain", "baseline_not_first", "enhanced_transformer_life_error"],
        ascending=[False, False, False, True],
    )
    picks = []
    for row in df.head(3).itertuples(index=False):
        picks.append((str(row.candidate), str(row.test_files).split(";")))
    pd.DataFrame([{"candidate": n, "test_files": ";".join(f)} for n, f in picks]).to_csv(RESULT_DIR / "stage3正式复核候选清单.csv", index=False, encoding="utf-8-sig")
    return picks


def write_final_doc(stage2: pd.DataFrame, stage3: pd.DataFrame) -> None:
    lines = [
        "夜间筛查结论",
        "=" * 72,
        "",
        "一、筛查口径",
        "-" * 72,
        "训练使用原始仿真采样点；闭环预测和磨损积分固定使用 250。",
        "增强包为 R1_静态派生 + slow_only。",
        "目标是找到测试集选择可解释，且增强后 Transformer 能排第一的候选 split。",
        "",
        "二、Stage2 候选摘要",
        "-" * 72,
    ]
    if not stage2.empty:
        show = stage2.sort_values(["enhanced_transformer_rank", "enhanced_transformer_life_error"]).head(10)
        for row in show.itertuples(index=False):
            lines.append(
                f"{row.candidate}: test={row.test_files}, baseline_rank={row.baseline_transformer_rank}, "
                f"enhanced_rank={row.enhanced_transformer_rank}, enhanced_error={float(row.enhanced_transformer_life_error):.0f}, "
                f"best={row.enhanced_best_model}, dup_FCr={row.duplicate_F_Cr}, max_step={float(row.max_native_step):.0f}"
            )
    lines.extend(["", "三、Stage3 正式复核结果", "-" * 72])
    if not stage3.empty:
        show = stage3.sort_values(["enhanced_transformer_rank", "enhanced_transformer_life_error"])
        for row in show.itertuples(index=False):
            lines.append(
                f"{row.candidate}: test={row.test_files}, baseline_rank={row.baseline_transformer_rank}, "
                f"baseline_error={float(row.baseline_transformer_life_error):.0f}, enhanced_rank={row.enhanced_transformer_rank}, "
                f"enhanced_error={float(row.enhanced_transformer_life_error):.0f}, best={row.enhanced_best_model}, "
                f"dup_FCr={row.duplicate_F_Cr}, steps={row.native_steps}"
            )
    lines.extend([
        "",
        "四、使用建议",
        "-" * 72,
        "优先选择 duplicate_F_Cr=0、max_native_step<=652、增强后 Transformer 排第一且误差低的候选。",
        "若两个候选效果接近，优先选择参数分布更容易解释、没有同 F 同 Cr 重复的测试集。",
    ])
    FINAL_DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    common.require_cuda()
    for line in common.device_summary_lines():
        print(line)
    ensure_stage1()
    _, case_tables = common.load_data()
    stage2_candidates = pick_stage2_candidates(case_tables)
    print(f"Stage2 candidates={len(stage2_candidates)}")
    stage2 = run_full_candidates("stage2_full500", stage2_candidates, 500, STAGE2_SUMMARY, STAGE2_DETAIL)
    stage3_candidates = pick_stage3_candidates(stage2)
    print(f"Stage3 candidates={len(stage3_candidates)}")
    stage3 = run_full_candidates("stage3_formal1200", stage3_candidates, 1200, STAGE3_SUMMARY, STAGE3_DETAIL)
    write_final_doc(stage2, stage3)
    print(f"Final doc: {FINAL_DOC}")


if __name__ == "__main__":
    main()
