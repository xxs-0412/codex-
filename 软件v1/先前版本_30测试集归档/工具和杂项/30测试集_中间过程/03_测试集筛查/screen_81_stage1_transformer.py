from __future__ import annotations

import itertools
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
SCREEN_EPOCHS = 250
FIXED_STEP_CAP = 250.0
MODEL_NAME = "Transformer"
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

POOLS = {
    "D8": ["Run4.csv", "Run15.csv", "Run16.csv"],
    "D13": ["Run20.csv", "Run28.csv", "Run30.csv"],
    "D16": ["Run8.csv", "Run24.csv", "Run25.csv"],
    "D22": ["Run11.csv", "Run18.csv", "Run29.csv"],
}
OUT_PATH = RESULT_DIR / "筛查汇总_81_stage1_transformer.csv"
DETAIL_PATH = RESULT_DIR / "筛查明细_81_stage1_transformer.csv"


def case_meta(case_tables, file_name):
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


def eval_transformer(variant, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables):
    if variant == "baseline_5d":
        feature_spec = BASELINE_SPEC
        shape_config = None
    else:
        feature_spec = ENHANCED_SPEC
        shape_config = ENHANCED_SHAPE
    model, seq_scaler, target_scaler = common.train_model(
        model_name=MODEL_NAME,
        train_raw_seq=train_raw_seq,
        train_y=train_y,
        feature_spec=feature_spec,
        config=CONFIG,
        shape_config=shape_config,
        train_case_names=train_case_names,
        train_step_indices=train_step_indices,
    )
    rows=[]
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
            "variant": variant,
            "test_case": str(test_row["source_file"]),
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


def main():
    common.require_cuda()
    for line in common.device_summary_lines():
        print(line)
    data_summary, case_tables = common.load_data()
    combos = list(itertools.product(POOLS["D8"], POOLS["D13"], POOLS["D16"], POOLS["D22"]))
    print(f"stage1 Transformer-only combos={len(combos)}, epochs={SCREEN_EPOCHS}")
    summary_rows=[]
    detail_rows=[]
    for idx, test_files_tuple in enumerate(combos, start=1):
        common.set_seed(common.SEED + 1000 + idx)
        test_files = list(test_files_tuple)
        name = "_".join([f.replace('.csv','') for f in test_files])
        print("="*80)
        print(f"{idx}/{len(combos)} {name}")
        test_set=set(test_files)
        test_df=data_summary[data_summary["file_name"].isin(test_set)].copy()
        train_df=data_summary[~data_summary["file_name"].isin(test_set)].copy()
        train_tables={str(row["file_name"]):case_tables[str(row["file_name"])] for _,row in train_df.iterrows()}
        train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)
        combo_detail=[]
        for variant in ["baseline_5d", "R1_plus_slow"]:
            print(f"  {variant}")
            rows=eval_transformer(variant, train_raw_seq, train_y, train_case_names, train_step_indices, test_df, case_tables)
            for r in rows:
                r["candidate"] = name
                r["test_files"] = ";".join(test_files)
            combo_detail.extend(rows)
        base_df=pd.DataFrame([r for r in combo_detail if r["variant"]=="baseline_5d"])
        enh_df=pd.DataFrame([r for r in combo_detail if r["variant"]=="R1_plus_slow"])
        metas=[case_meta(case_tables, f) for f in test_files]
        pairs=[(m["F"], m["Cr"]) for m in metas]
        duplicate_F_Cr = len(set(pairs)) < len(pairs)
        summary={
            "candidate": name,
            "test_files": ";".join(test_files),
            "baseline_transformer_life_error": float(base_df["life_abs_error"].mean()),
            "enhanced_transformer_life_error": float(enh_df["life_abs_error"].mean()),
            "enhanced_minus_baseline": float(enh_df["life_abs_error"].mean() - base_df["life_abs_error"].mean()),
            "improve_pct": float((enh_df["life_abs_error"].mean() - base_df["life_abs_error"].mean()) / max(base_df["life_abs_error"].mean(), common.EPS)),
            "max_native_step": max(m["native_step"] for m in metas),
            "max_life": max(m["life"] for m in metas),
            "min_life": min(m["life"] for m in metas),
            "duplicate_F_Cr": int(duplicate_F_Cr),
            "F_values": ";".join(str(int(m["F"])) for m in metas),
            "Cr_values": ";".join(f"{m['Cr']:.3g}" for m in metas),
            "epochs": SCREEN_EPOCHS,
        }
        summary_rows.append(summary)
        detail_rows.extend(combo_detail)
        pd.DataFrame(summary_rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame(detail_rows).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
        print(f"  base={summary['baseline_transformer_life_error']:.1f}, enh={summary['enhanced_transformer_life_error']:.1f}, delta={summary['enhanced_minus_baseline']:.1f}, dupFCr={summary['duplicate_F_Cr']}")
    print("done")
    print(pd.DataFrame(summary_rows).sort_values(["enhanced_transformer_life_error", "improve_pct"]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
