from __future__ import annotations

import sys
from pathlib import Path

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
SCREEN_EPOCHS = 400
MODEL_ORDER = ["FNN", "GRU", "LSTM", "1D-CNN", "Transformer"]
FIXED_STEP_CAP = 250.0
BASELINE_SPEC = common.LEGACY_FEATURE_SPEC
ENHANCED_SPEC = common.FEATURE_R1_SPEC
ENHANCED_SHAPE = common.ShapeLossConfig(name="slow_only", mono_lambda=0.0, slow_lambda=0.01, mono_tolerance_ratio=0.0)
CONFIG = common.TransformerConfig(
    seq_len=SEQ_LEN,
    d_model=32,
    nhead=4,
    num_layers=2,
    dim_ff=64,
    dropout=0.0,
    epochs=SCREEN_EPOCHS,
    learning_rate=1e-3,
    weight_decay=1e-6,
)

CANDIDATES = [
    ("D13_Run20", ["Run4.csv", "Run8.csv", "Run11.csv", "Run20.csv"]),
    ("D13_Run28", ["Run4.csv", "Run8.csv", "Run11.csv", "Run28.csv"]),
    ("D13_Run23", ["Run4.csv", "Run8.csv", "Run11.csv", "Run23.csv"]),
    ("D13_Run27", ["Run4.csv", "Run8.csv", "Run11.csv", "Run27.csv"]),
    ("D13_Run9", ["Run4.csv", "Run8.csv", "Run11.csv", "Run9.csv"]),
    ("D13_Run10", ["Run4.csv", "Run8.csv", "Run11.csv", "Run10.csv"]),
    ("D13_Run21", ["Run4.csv", "Run8.csv", "Run11.csv", "Run21.csv"]),
]

DETAIL_PATH = RESULT_DIR / "筛查明细_各候选各模型.csv"
SUMMARY_PATH = RESULT_DIR / "筛查汇总_候选split.csv"
HIT_PATH = RESULT_DIR / "命中候选_Transformer增强第一.csv"


def rank_summary(rows, variant):
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


def evaluate_variant(variant, model_name, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables):
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
        config=CONFIG,
        shape_config=shape_config,
        train_case_names=train_case_names,
        train_step_indices=train_step_indices,
    )

    out_rows = []
    for _, test_row in test_df.iterrows():
        test_file = str(test_row["file_name"])
        test_case = str(test_row["source_file"])
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
        out_rows.append(
            {
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
            }
        )
    del model
    torch.cuda.empty_cache()
    return out_rows


def main():
    common.require_cuda()
    for line in common.device_summary_lines():
        print(line)
    print(f"screening candidates={len(CANDIDATES)}, epochs={SCREEN_EPOCHS}, fixed_step={FIXED_STEP_CAP}")

    all_summary_rows = []
    all_detail_rows = []
    hit_rows = []
    data_summary, case_tables = common.load_data()

    for idx, (candidate_name, test_files) in enumerate(CANDIDATES, start=1):
        common.set_seed(common.SEED + idx)
        print("=" * 80)
        print(f"candidate {idx}/{len(CANDIDATES)} | {candidate_name} | {', '.join(test_files)}")
        test_set = set(test_files)
        test_df = data_summary[data_summary["file_name"].isin(test_set)].copy()
        train_df = data_summary[~data_summary["file_name"].isin(test_set)].copy()
        train_tables = {str(row["file_name"]): case_tables[str(row["file_name"])] for _, row in train_df.iterrows()}
        train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)

        candidate_rows = []
        for variant in ["baseline_5d", "R1_plus_slow"]:
            print(f"  variant={variant}")
            for model_name in MODEL_ORDER:
                print(f"    train/eval {model_name}")
                rows = evaluate_variant(
                    variant,
                    model_name,
                    train_raw_seq,
                    train_y,
                    train_case_names,
                    train_step_indices,
                    test_df,
                    case_tables,
                )
                for r in rows:
                    r["candidate"] = candidate_name
                    r["test_files"] = ";".join(test_files)
                candidate_rows.extend(rows)

        base_rank = rank_summary(candidate_rows, "baseline_5d")
        enh_rank = rank_summary(candidate_rows, "R1_plus_slow")
        transformer_base = base_rank[base_rank["model"].eq("Transformer")].iloc[0]
        transformer_enh = enh_rank[enh_rank["model"].eq("Transformer")].iloc[0]
        best_enh = enh_rank.iloc[0]
        hit = int(transformer_base["rank"] in (2, 3) and transformer_enh["rank"] == 1)
        summary_row = {
            "candidate": candidate_name,
            "test_files": ";".join(test_files),
            "baseline_transformer_rank": int(transformer_base["rank"]),
            "baseline_transformer_life_error": float(transformer_base["mean_life_abs_error"]),
            "enhanced_transformer_rank": int(transformer_enh["rank"]),
            "enhanced_transformer_life_error": float(transformer_enh["mean_life_abs_error"]),
            "enhanced_best_model": str(best_enh["model"]),
            "enhanced_best_life_error": float(best_enh["mean_life_abs_error"]),
            "hit_target": hit,
            "epochs": SCREEN_EPOCHS,
        }
        all_summary_rows.append(summary_row)
        all_detail_rows.extend(candidate_rows)
        if hit:
            hit_rows.append(summary_row)

        pd.DataFrame(all_detail_rows).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame(all_summary_rows).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame(hit_rows).to_csv(HIT_PATH, index=False, encoding="utf-8-sig")
        print("  baseline rank:")
        print(base_rank[["model", "rank", "mean_life_abs_error"]].to_string(index=False))
        print("  enhanced rank:")
        print(enh_rank[["model", "rank", "mean_life_abs_error"]].to_string(index=False))
        print(f"  hit_target={hit}")

    print("=" * 80)
    print("screening done")
    print(pd.DataFrame(all_summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
