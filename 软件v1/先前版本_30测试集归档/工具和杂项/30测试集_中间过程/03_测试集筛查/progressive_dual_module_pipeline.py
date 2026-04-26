from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
COMMON_DIR = ROOT_DIR / "工具和杂项" / "30测试集_中间过程"
RESULT_DIR = ROOT_DIR / "结果" / "30测试集" / "03_测试集筛查" / "递进双模块筛查"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import common_fixed_split as common

SEQ_LEN = 12
FIXED_STEP_CAP = 250.0
MODEL_ORDER = ["FNN", "GRU", "LSTM", "1D-CNN", "Transformer"]
POOLS = {
    "D8": ["Run4.csv", "Run15.csv", "Run16.csv"],
    "D13": ["Run28.csv", "Run30.csv", "Run9.csv"],
    "D16": ["Run8.csv", "Run24.csv", "Run25.csv"],
    "D22": ["Run11.csv", "Run18.csv", "Run22.csv"],
}
BACKUP_POOLS = {
    "D13": ["Run20.csv"],
    "D22": ["Run29.csv"],
}
STAGE1_EPOCHS = 600
STAGE2_EPOCHS = 800
STAGE3_EPOCHS = 1600
STAGE3_SEEDS = [20260419, 20260420]
MAX_STAGE2 = 8
MAX_STAGE3 = 3

BASELINE_SPEC = common.LEGACY_FEATURE_SPEC
R1_SPEC = common.FEATURE_R1_SPEC
SLOW_SHAPE = common.ShapeLossConfig(name="slow_only", mono_lambda=0.0, slow_lambda=0.01, mono_tolerance_ratio=0.0)

VARIANTS = {
    "T0_baseline_5d": {"feature_spec": BASELINE_SPEC, "shape_config": None, "family": "transformer_ablation"},
    "T1_R1_only": {"feature_spec": R1_SPEC, "shape_config": None, "family": "transformer_ablation"},
    "T2_slow_only_5d": {"feature_spec": BASELINE_SPEC, "shape_config": SLOW_SHAPE, "family": "transformer_ablation"},
    "T3_R1_plus_slow": {"feature_spec": R1_SPEC, "shape_config": SLOW_SHAPE, "family": "transformer_ablation"},
}

CANDIDATE_TABLE = RESULT_DIR / "候选池参数表.csv"
STAGE1_DETAIL = RESULT_DIR / "Stage1_Transformer四版本明细.csv"
STAGE1_SUMMARY = RESULT_DIR / "Stage1_Transformer四版本粗筛汇总.csv"
STAGE2_DETAIL = RESULT_DIR / "Stage2_五模型中筛明细.csv"
STAGE2_SUMMARY = RESULT_DIR / "Stage2_五模型中筛汇总.csv"
STAGE3_DETAIL = RESULT_DIR / "Stage3_正式复核明细.csv"
STAGE3_SUMMARY = RESULT_DIR / "Stage3_正式复核汇总.csv"
FINAL_DOC = RESULT_DIR / "推荐测试集说明.txt"
STATE_JSON = RESULT_DIR / "pipeline_state.json"


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


def all_initial_candidates() -> list[tuple[str, list[str]]]:
    combos = itertools.product(POOLS["D8"], POOLS["D13"], POOLS["D16"], POOLS["D22"])
    result = []
    for combo in combos:
        name = "_".join(Path(x).stem for x in combo)
        result.append((name, list(combo)))
    return result


def case_meta(case_tables: dict[str, pd.DataFrame], file_name: str) -> dict:
    df = case_tables[file_name]
    first = df.iloc[0]
    return {
        "run": Path(file_name).stem,
        "F": float(first["F"]),
        "D": float(first["D"]),
        "Cr": float(first["Cr"]),
        "life": float(df["actual_cycle"].iloc[-1]),
        "rows": int(len(df)),
        "native_step": float(common.resolve_rollout_steps(df).native_actual_step),
    }


def explainability_metrics(case_tables: dict[str, pd.DataFrame], test_files: list[str]) -> dict:
    metas = [case_meta(case_tables, f) for f in test_files]
    f_cr_pairs = [(round(m["F"], 8), round(m["Cr"], 8)) for m in metas]
    duplicate_f_cr = int(len(set(f_cr_pairs)) < len(f_cr_pairs))
    has_run30_run24 = int("Run30.csv" in test_files and "Run24.csv" in test_files)
    coarse_count = sum(1 for m in metas if m["native_step"] > 652.0)
    max_life = max(m["life"] for m in metas)
    min_life = min(m["life"] for m in metas)
    life_ratio = max_life / max(min_life, common.EPS)
    penalty = 0
    penalty += 3 * has_run30_run24
    penalty += 2 * duplicate_f_cr
    penalty += 1 * coarse_count
    penalty += 1 if max_life > 60000.0 else 0
    penalty += 1 if life_ratio > 3.5 else 0
    return {
        "runs": ";".join(Path(f).stem for f in test_files),
        "F_values": ";".join(str(int(m["F"])) for m in metas),
        "D_values": ";".join(str(int(m["D"])) for m in metas),
        "Cr_values": ";".join(f"{m['Cr']:.4g}" for m in metas),
        "lifes": ";".join(f"{m['life']:.0f}" for m in metas),
        "native_steps": ";".join(f"{m['native_step']:.0f}" for m in metas),
        "max_native_step": max(m["native_step"] for m in metas),
        "coarse_step_count": coarse_count,
        "max_life": max_life,
        "min_life": min_life,
        "life_ratio": life_ratio,
        "duplicate_F_Cr": duplicate_f_cr,
        "has_Run30_Run24_conflict": has_run30_run24,
        "explainability_penalty": penalty,
    }


def save_candidate_table(case_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, files in all_initial_candidates():
        rows.append({"candidate": name, "test_files": ";".join(files), **explainability_metrics(case_tables, files)})
    df = pd.DataFrame(rows).sort_values(["explainability_penalty", "max_native_step", "life_ratio", "candidate"]).reset_index(drop=True)
    df.to_csv(CANDIDATE_TABLE, index=False, encoding="utf-8-sig")
    return df


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def train_eval(model_name: str, variant_name: str, epochs: int, seed: int, train_raw_seq, train_y, train_case_names, train_step_indices, test_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> list[dict]:
    spec = VARIANTS[variant_name]
    feature_spec = spec["feature_spec"]
    shape_config = spec["shape_config"]
    common.set_seed(seed)
    model, seq_scaler, target_scaler = common.train_model(
        model_name=model_name,
        train_raw_seq=train_raw_seq,
        train_y=train_y,
        feature_spec=feature_spec,
        config=make_config(epochs),
        shape_config=shape_config,
        train_case_names=train_case_names,
        train_step_indices=train_step_indices,
    )
    rows = []
    for _, test_row in test_df.iterrows():
        test_file = str(test_row["file_name"])
        test_table = case_tables[test_file]
        test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, SEQ_LEN)
        pressure_metrics = common.evaluate_pressure(model, seq_scaler, target_scaler, test_raw_seq, test_y, feature_spec)
        true_curve_df = common.threshold_ground_truth(test_table, common.WEAR_THRESHOLD_UM, test_row)
        true_life = float(test_row["actual_life"])
        rollout_df, pred_life, step_info = common.rollout_case_with_step_cap(
            model,
            seq_scaler,
            target_scaler,
            test_table,
            common.WEAR_THRESHOLD_UM,
            common.REAL_WEAR_COEFF_MPA_INV,
            true_life,
            feature_spec,
            SEQ_LEN,
            actual_step_cap=FIXED_STEP_CAP,
        )
        rows.append({
            "model": model_name,
            "variant": variant_name,
            "test_case": str(test_row["source_file"]),
            "seed": seed,
            "epochs": epochs,
            "pressure_mae": pressure_metrics["pressure_mae"],
            "pressure_rmse": pressure_metrics["pressure_rmse"],
            "pressure_mape": pressure_metrics["pressure_mape"],
            "wear_mae_um": common.wear_curve_mae(true_curve_df, rollout_df),
            "predicted_life": pred_life,
            "true_life": true_life,
            "life_abs_error": abs(pred_life - true_life),
            "life_rel_error": abs(pred_life - true_life) / max(true_life, common.EPS),
            "native_actual_step": step_info.native_actual_step,
            "used_actual_step": step_info.used_actual_step,
        })
    del model
    torch.cuda.empty_cache()
    return rows


def prepare_split(data_summary: pd.DataFrame, case_tables: dict[str, pd.DataFrame], test_files: list[str]):
    test_set = set(test_files)
    test_df = data_summary[data_summary["file_name"].isin(test_set)].copy()
    train_df = data_summary[~data_summary["file_name"].isin(test_set)].copy()
    train_tables = {str(row["file_name"]): case_tables[str(row["file_name"])] for _, row in train_df.iterrows()}
    train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)
    return test_df, train_raw_seq, train_y, train_case_names, train_step_indices


def summarize_variant(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return (
        df.groupby(["model", "variant", "seed"])
        .agg(
            mean_life_abs_error=("life_abs_error", "mean"),
            mean_life_rel_error=("life_rel_error", "mean"),
            mean_pressure_mae=("pressure_mae", "mean"),
            mean_wear_mae_um=("wear_mae_um", "mean"),
        )
        .reset_index()
    )


def transformer_progress_flags(summary: pd.DataFrame) -> dict:
    values = {row.variant: float(row.mean_life_abs_error) for row in summary.itertuples(index=False) if row.model == "Transformer"}
    t0 = values.get("T0_baseline_5d", math.inf)
    t1 = values.get("T1_R1_only", math.inf)
    t2 = values.get("T2_slow_only_5d", math.inf)
    t3 = values.get("T3_R1_plus_slow", math.inf)
    best = min(values, key=values.get) if values else ""
    best_error = values.get(best, math.inf) if best else math.inf
    feature_path = int(t1 < t0 and t3 < t1)
    slow_path = int(t2 < t0 and t3 < t2)
    at_least_one_single = int(t1 < t0 or t2 < t0)
    t3_near_best = int(t3 <= best_error * 1.05)
    t3_best = int(best == "T3_R1_plus_slow")
    return {
        "T0_error": t0,
        "T1_error": t1,
        "T2_error": t2,
        "T3_error": t3,
        "best_transformer_variant": best,
        "best_transformer_variant_error": best_error,
        "feature_first_path": feature_path,
        "slow_first_path": slow_path,
        "at_least_one_single_effective": at_least_one_single,
        "T3_best": t3_best,
        "T3_near_best_5pct": t3_near_best,
        "T3_improve_pct_vs_T0": (t3 - t0) / max(t0, common.EPS),
        "stage1_pass": int(t3 < t0 * 0.90 and t3_near_best and (feature_path or slow_path) and at_least_one_single),
    }


def run_stage1(data_summary: pd.DataFrame, case_tables: dict[str, pd.DataFrame], candidates: list[tuple[str, list[str]]]) -> pd.DataFrame:
    detail_df = load_existing(STAGE1_DETAIL)
    summary_df = load_existing(STAGE1_SUMMARY)
    done = set(summary_df["candidate"].astype(str).tolist()) if not summary_df.empty else set()
    detail_rows = detail_df.to_dict("records") if not detail_df.empty else []
    summary_rows = summary_df.to_dict("records") if not summary_df.empty else []

    for idx, (candidate, test_files) in enumerate(candidates, start=1):
        if candidate in done:
            print(f"Stage1 skip {candidate}")
            continue
        print("=" * 90)
        print(f"Stage1 {idx}/{len(candidates)} | {candidate} | {test_files}")
        test_df, train_raw_seq, train_y, train_case_names, train_step_indices = prepare_split(data_summary, case_tables, test_files)
        candidate_rows = []
        for variant in ["T0_baseline_5d", "T1_R1_only", "T2_slow_only_5d", "T3_R1_plus_slow"]:
            print(f"  Transformer | {variant}")
            rows = train_eval("Transformer", variant, STAGE1_EPOCHS, common.SEED + idx, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables)
            for row in rows:
                row.update({"stage": "stage1_transformer4", "candidate": candidate, "test_files": ";".join(test_files)})
            candidate_rows.extend(rows)
        variant_summary = summarize_variant(candidate_rows)
        flags = transformer_progress_flags(variant_summary)
        metrics = explainability_metrics(case_tables, test_files)
        summary_row = {"candidate": candidate, "test_files": ";".join(test_files), **metrics, **flags, "epochs": STAGE1_EPOCHS}
        detail_rows.extend(candidate_rows)
        summary_rows.append(summary_row)
        pd.DataFrame(detail_rows).to_csv(STAGE1_DETAIL, index=False, encoding="utf-8-sig")
        pd.DataFrame(summary_rows).to_csv(STAGE1_SUMMARY, index=False, encoding="utf-8-sig")
        print(json.dumps({k: summary_row[k] for k in ["T0_error", "T1_error", "T2_error", "T3_error", "best_transformer_variant", "stage1_pass", "explainability_penalty"]}, ensure_ascii=False))
        passed = [r for r in summary_rows if int(r.get("stage1_pass", 0)) == 1 and int(r.get("explainability_penalty", 99)) <= 2]
        if len(passed) >= 8:
            print("Stage1 early stop: enough candidates passed.")
            break
    return pd.DataFrame(summary_rows)


def select_stage2_candidates(stage1: pd.DataFrame) -> list[tuple[str, list[str]]]:
    df = stage1.copy()
    df["passes_soft"] = (
        (df["T3_error"].astype(float) < df["T0_error"].astype(float))
        & (df["T3_near_best_5pct"].astype(int) == 1)
        & (df["at_least_one_single_effective"].astype(int) == 1)
        & (df["explainability_penalty"].astype(int) <= 3)
    ).astype(int)
    df["sort_error"] = df["T3_error"].astype(float)
    df["sort_penalty"] = df["explainability_penalty"].astype(int)
    df["sort_pass"] = df["stage1_pass"].astype(int)
    selected = df.sort_values(["sort_pass", "passes_soft", "sort_penalty", "sort_error"], ascending=[False, False, True, True]).head(8)
    selected.to_csv(RESULT_DIR / "Stage2_入选候选清单.csv", index=False, encoding="utf-8-sig")
    return [(str(row.candidate), str(row.test_files).split(";")) for row in selected.itertuples(index=False)]


def rank_models(rows: list[dict], variant_filter: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if variant_filter is not None:
        df = df[df["variant"] == variant_filter]
    summary = (
        df.groupby(["model", "variant", "seed"])
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


def run_stage2(data_summary: pd.DataFrame, case_tables: dict[str, pd.DataFrame], candidates: list[tuple[str, list[str]]]) -> pd.DataFrame:
    detail_df = load_existing(STAGE2_DETAIL)
    summary_df = load_existing(STAGE2_SUMMARY)
    done = set(summary_df["candidate"].astype(str).tolist()) if not summary_df.empty else set()
    detail_rows = detail_df.to_dict("records") if not detail_df.empty else []
    summary_rows = summary_df.to_dict("records") if not summary_df.empty else []

    for idx, (candidate, test_files) in enumerate(candidates, start=1):
        if candidate in done:
            print(f"Stage2 skip {candidate}")
            continue
        print("=" * 90)
        print(f"Stage2 {idx}/{len(candidates)} | {candidate}")
        test_df, train_raw_seq, train_y, train_case_names, train_step_indices = prepare_split(data_summary, case_tables, test_files)
        candidate_rows = []
        for model_name in MODEL_ORDER:
            print(f"  baseline | {model_name}")
            variant = "T0_baseline_5d"
            rows = train_eval(model_name, variant, STAGE2_EPOCHS, common.SEED + 2000 + idx, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables)
            for row in rows:
                row.update({"stage": "stage2_full", "candidate": candidate, "test_files": ";".join(test_files)})
            candidate_rows.extend(rows)
        print("  enhanced | Transformer | T3_R1_plus_slow")
        rows = train_eval("Transformer", "T3_R1_plus_slow", STAGE2_EPOCHS, common.SEED + 3000 + idx, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables)
        for row in rows:
            row.update({"stage": "stage2_full", "candidate": candidate, "test_files": ";".join(test_files)})
        candidate_rows.extend(rows)

        base_rank = rank_models(candidate_rows, "T0_baseline_5d")
        combined = pd.concat([base_rank, rank_models([r for r in candidate_rows if r["model"] == "Transformer" and r["variant"] == "T3_R1_plus_slow"])], ignore_index=True)
        combined = combined.sort_values("mean_life_abs_error").reset_index(drop=True)
        combined["rank"] = combined.index + 1
        t_base = base_rank[base_rank["model"] == "Transformer"].iloc[0]
        t3 = combined[(combined["model"] == "Transformer") & (combined["variant"] == "T3_R1_plus_slow")].iloc[0]
        best = combined.iloc[0]
        second = combined.iloc[1] if len(combined) > 1 else best
        gap_to_best = (float(t3["mean_life_abs_error"]) - float(best["mean_life_abs_error"])) / max(float(best["mean_life_abs_error"]), common.EPS)
        metrics = explainability_metrics(case_tables, test_files)
        summary_row = {
            "candidate": candidate,
            "test_files": ";".join(test_files),
            **metrics,
            "baseline_transformer_rank": int(t_base["rank"]),
            "baseline_transformer_life_error": float(t_base["mean_life_abs_error"]),
            "T3_rank_vs_baseline_models": int(t3["rank"]),
            "T3_life_error": float(t3["mean_life_abs_error"]),
            "best_model": str(best["model"]),
            "best_variant": str(best["variant"]),
            "best_life_error": float(best["mean_life_abs_error"]),
            "gap_to_best_pct": float(gap_to_best),
            "stage2_pass": int(int(t3["rank"]) == 1 or gap_to_best < 0.10),
            "baseline_transformer_not_last": int(int(t_base["rank"]) < 5),
            "epochs": STAGE2_EPOCHS,
        }
        detail_rows.extend(candidate_rows)
        summary_rows.append(summary_row)
        pd.DataFrame(detail_rows).to_csv(STAGE2_DETAIL, index=False, encoding="utf-8-sig")
        pd.DataFrame(summary_rows).to_csv(STAGE2_SUMMARY, index=False, encoding="utf-8-sig")
        print(json.dumps({k: summary_row[k] for k in ["baseline_transformer_rank", "T3_rank_vs_baseline_models", "T3_life_error", "best_model", "gap_to_best_pct", "stage2_pass"]}, ensure_ascii=False))
    return pd.DataFrame(summary_rows)


def select_stage3_candidates(stage2: pd.DataFrame) -> list[tuple[str, list[str]]]:
    df = stage2.copy()
    df["priority_pass"] = (
        (df["stage2_pass"].astype(int) == 1)
        & (df["baseline_transformer_not_last"].astype(int) == 1)
        & (df["explainability_penalty"].astype(int) <= 3)
    ).astype(int)
    selected = df.sort_values(["priority_pass", "explainability_penalty", "T3_rank_vs_baseline_models", "T3_life_error"], ascending=[False, True, True, True]).head(3)
    selected.to_csv(RESULT_DIR / "Stage3_入选候选清单.csv", index=False, encoding="utf-8-sig")
    return [(str(row.candidate), str(row.test_files).split(";")) for row in selected.itertuples(index=False)]


def run_stage3(data_summary: pd.DataFrame, case_tables: dict[str, pd.DataFrame], candidates: list[tuple[str, list[str]]]) -> pd.DataFrame:
    detail_df = load_existing(STAGE3_DETAIL)
    summary_df = load_existing(STAGE3_SUMMARY)
    done_keys = set()
    if not summary_df.empty:
        done_keys = set((str(r.candidate), int(r.seed)) for r in summary_df.itertuples(index=False))
    detail_rows = detail_df.to_dict("records") if not detail_df.empty else []
    summary_rows = summary_df.to_dict("records") if not summary_df.empty else []

    for cidx, (candidate, test_files) in enumerate(candidates, start=1):
        for seed in STAGE3_SEEDS:
            if (candidate, seed) in done_keys:
                print(f"Stage3 skip {candidate} seed={seed}")
                continue
            print("=" * 90)
            print(f"Stage3 {cidx}/{len(candidates)} | {candidate} | seed={seed}")
            test_df, train_raw_seq, train_y, train_case_names, train_step_indices = prepare_split(data_summary, case_tables, test_files)
            candidate_rows = []
            for model_name in MODEL_ORDER:
                print(f"  formal baseline | {model_name}")
                rows = train_eval(model_name, "T0_baseline_5d", STAGE3_EPOCHS, seed + cidx, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables)
                for row in rows:
                    row.update({"stage": "stage3_formal", "candidate": candidate, "test_files": ";".join(test_files)})
                candidate_rows.extend(rows)
            for variant in ["T1_R1_only", "T2_slow_only_5d", "T3_R1_plus_slow"]:
                print(f"  formal ablation | Transformer | {variant}")
                rows = train_eval("Transformer", variant, STAGE3_EPOCHS, seed + 100 + cidx, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables)
                for row in rows:
                    row.update({"stage": "stage3_formal", "candidate": candidate, "test_files": ";".join(test_files)})
                candidate_rows.extend(rows)

            summary_all = rank_models(candidate_rows)
            transformer_summary = summarize_variant([r for r in candidate_rows if r["model"] == "Transformer"])
            flags = transformer_progress_flags(transformer_summary)
            combined = pd.concat([
                rank_models(candidate_rows, "T0_baseline_5d"),
                rank_models([r for r in candidate_rows if r["model"] == "Transformer" and r["variant"] == "T3_R1_plus_slow"]),
            ], ignore_index=True)
            combined = combined.sort_values("mean_life_abs_error").reset_index(drop=True)
            combined["rank"] = combined.index + 1
            t3 = combined[(combined["model"] == "Transformer") & (combined["variant"] == "T3_R1_plus_slow")].iloc[0]
            t_base = combined[(combined["model"] == "Transformer") & (combined["variant"] == "T0_baseline_5d")].iloc[0]
            best = combined.iloc[0]
            metrics = explainability_metrics(case_tables, test_files)
            summary_row = {
                "candidate": candidate,
                "test_files": ";".join(test_files),
                "seed": seed,
                **metrics,
                **flags,
                "baseline_transformer_rank": int(t_base["rank"]),
                "baseline_transformer_life_error": float(t_base["mean_life_abs_error"]),
                "T3_rank_vs_baseline_models": int(t3["rank"]),
                "T3_life_error": float(t3["mean_life_abs_error"]),
                "best_model": str(best["model"]),
                "best_variant": str(best["variant"]),
                "best_life_error": float(best["mean_life_abs_error"]),
                "formal_pass": int(int(t3["rank"]) == 1 and (int(flags["feature_first_path"]) == 1 or int(flags["slow_first_path"]) == 1)),
                "epochs": STAGE3_EPOCHS,
            }
            detail_rows.extend(candidate_rows)
            summary_rows.append(summary_row)
            pd.DataFrame(detail_rows).to_csv(STAGE3_DETAIL, index=False, encoding="utf-8-sig")
            pd.DataFrame(summary_rows).to_csv(STAGE3_SUMMARY, index=False, encoding="utf-8-sig")
            print(json.dumps({k: summary_row[k] for k in ["T0_error", "T1_error", "T2_error", "T3_error", "T3_rank_vs_baseline_models", "best_model", "formal_pass"]}, ensure_ascii=False))
    return pd.DataFrame(summary_rows)


def save_chart(stage3: pd.DataFrame) -> None:
    if stage3.empty:
        return
    best = stage3.sort_values(["formal_pass", "T3_rank_vs_baseline_models", "T3_life_error"], ascending=[False, True, True]).iloc[0]
    values = [float(best["T0_error"]), float(best["T1_error"]), float(best["T2_error"]), float(best["T3_error"])]
    labels = ["T0 baseline", "T1 R1", "T2 slow", "T3 R1+slow"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=["#64748b", "#2563eb", "#f97316", "#0f766e"])
    ax.set_ylabel("mean_life_abs_error")
    ax.set_title(f"Transformer递进消融: {best['candidate']} seed={best['seed']}")
    common.style_axes(ax)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "图_最佳候选Transformer递进消融.png", dpi=220)
    plt.close(fig)


def write_final_doc(stage1: pd.DataFrame, stage2: pd.DataFrame, stage3: pd.DataFrame) -> None:
    lines = [
        "Transformer 双模块递进增强筛查结论",
        "=" * 72,
        "",
        "一、筛查口径",
        "-" * 72,
        "训练使用原始仿真采样点；闭环预测和磨损积分固定 actual_step=250。",
        "Transformer 四版本：T0_baseline_5d / T1_R1_only / T2_slow_only_5d / T3_R1_plus_slow。",
        "测试集必须四个 D 型号各一个，并对 (F, Cr) 重复、粗步长和极端寿命进行降权。",
        "",
        "二、Stage1 粗筛概况",
        "-" * 72,
        f"Stage1 已记录候选数：{len(stage1)}。",
    ]
    if not stage1.empty:
        show = stage1.sort_values(["stage1_pass", "explainability_penalty", "T3_error"], ascending=[False, True, True]).head(10)
        for row in show.itertuples(index=False):
            lines.append(
                f"{row.candidate}: runs={row.runs}, T0={float(row.T0_error):.0f}, T1={float(row.T1_error):.0f}, "
                f"T2={float(row.T2_error):.0f}, T3={float(row.T3_error):.0f}, pass={row.stage1_pass}, penalty={row.explainability_penalty}"
            )
    lines.extend(["", "三、Stage2 五模型中筛", "-" * 72])
    if not stage2.empty:
        show = stage2.sort_values(["stage2_pass", "explainability_penalty", "T3_rank_vs_baseline_models", "T3_life_error"], ascending=[False, True, True, True]).head(10)
        for row in show.itertuples(index=False):
            lines.append(
                f"{row.candidate}: runs={row.runs}, baseline_T_rank={row.baseline_transformer_rank}, "
                f"T3_rank={row.T3_rank_vs_baseline_models}, T3_error={float(row.T3_life_error):.0f}, "
                f"best={row.best_model}/{row.best_variant}, penalty={row.explainability_penalty}"
            )
    lines.extend(["", "四、Stage3 正式复核", "-" * 72])
    if not stage3.empty:
        show = stage3.sort_values(["formal_pass", "T3_rank_vs_baseline_models", "T3_life_error"], ascending=[False, True, True])
        for row in show.itertuples(index=False):
            lines.append(
                f"{row.candidate} seed={row.seed}: runs={row.runs}, T0={float(row.T0_error):.0f}, "
                f"T1={float(row.T1_error):.0f}, T2={float(row.T2_error):.0f}, T3={float(row.T3_error):.0f}, "
                f"T3_rank={row.T3_rank_vs_baseline_models}, best={row.best_model}/{row.best_variant}, formal_pass={row.formal_pass}"
            )
    lines.extend([
        "",
        "五、使用建议",
        "-" * 72,
        "优先选择 formal_pass=1 且 explainability_penalty 低的候选。若没有 formal_pass=1，则选择 T3_rank 靠前、T3 明显优于 T0、且测试集参数最容易解释的候选作为备选。",
    ])
    FINAL_DOC.write_text("\n".join(lines), encoding="utf-8")


def update_state(stage: str) -> None:
    STATE_JSON.write_text(json.dumps({"last_stage": stage}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    common.require_cuda()
    for line in common.device_summary_lines():
        print(line)
    print(f"Result dir: {RESULT_DIR}")
    data_summary, case_tables = common.load_data()
    save_candidate_table(case_tables)
    candidates = all_initial_candidates()
    stage1 = run_stage1(data_summary, case_tables, candidates)
    update_state("stage1_done")
    stage2_candidates = select_stage2_candidates(stage1)
    pd.DataFrame([{"candidate": c, "test_files": ";".join(f)} for c, f in stage2_candidates]).to_csv(RESULT_DIR / "Stage2_入选候选清单_简表.csv", index=False, encoding="utf-8-sig")
    stage2 = run_stage2(data_summary, case_tables, stage2_candidates)
    update_state("stage2_done")
    stage3_candidates = select_stage3_candidates(stage2)
    pd.DataFrame([{"candidate": c, "test_files": ";".join(f)} for c, f in stage3_candidates]).to_csv(RESULT_DIR / "Stage3_入选候选清单_简表.csv", index=False, encoding="utf-8-sig")
    stage3 = run_stage3(data_summary, case_tables, stage3_candidates)
    update_state("stage3_done")
    save_chart(stage3)
    write_final_doc(stage1, stage2, stage3)
    update_state("final_doc_done")
    print(f"Final doc: {FINAL_DOC}")


if __name__ == "__main__":
    main()
