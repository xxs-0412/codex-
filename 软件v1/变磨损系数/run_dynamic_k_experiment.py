# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SOURCE_TOOL_DIR = ROOT_DIR / "工具和杂项" / "4.25测试集选定_论文结构重跑"
GPR_DIR = ROOT_DIR / "磨损系数预测"

if str(SOURCE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_TOOL_DIR))

import common_fixed_split as common  # noqa: E402


TEST_FILES = ["Run4.csv", "Run28.csv", "Run25.csv", "Run11.csv"]
SEQ_LEN = 12
STEP_CAP = 250.0
DEFAULT_SEEDS = [20260425, 20260426, 20260427, 20260428, 20260429]
DEFAULT_EPOCHS = 1600

DETAIL_NAME = "detail_dynamic_k.csv"
SUMMARY_NAME = "summary_dynamic_k.csv"
PER_CASE_NAME = "per_case_dynamic_k.csv"
CONFIG_NAME = "config_dynamic_k.json"

CALIB_STRENGTHS = [0.25, 0.5, 0.75, 1.0]
CALIB_CLIP_WIDTHS = [0.05, 0.10, 0.15, 0.20]


@dataclass(frozen=True)
class DynamicKConfig:
    name: str
    strength: float
    clip_width: float
    late_mode: str = "freeze60"
    domain_tolerance: float = 0.10
    min_t_kcycles: float = 5.0
    late_t_kcycles: float = 60.0


def parse_float_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def safe_name(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def load_formal_split() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    data_summary, case_tables = common.load_data()
    eligible = common.eligible_cases(data_summary)
    test_set = set(TEST_FILES)
    test_df = eligible.loc[eligible["file_name"].isin(test_set)].copy()
    train_df = eligible.loc[~eligible["file_name"].isin(test_set)].copy()
    missing = sorted(test_set - set(test_df["file_name"].astype(str)))
    if missing:
        raise RuntimeError(f"Missing formal test files in eligible data: {missing}")
    test_df["split_role"] = "test"
    train_df["split_role"] = "train"
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), case_tables


def train_tables_from_split(train_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {str(row.file_name): case_tables[str(row.file_name)] for row in train_df.itertuples(index=False)}


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


def final_shape_config() -> common.ShapeLossConfig:
    return common.ShapeLossConfig(
        name="slow_abs_0p01",
        mono_lambda=0.0,
        slow_lambda=0.01,
        mono_tolerance_ratio=0.0,
    )


class GprWearFactor:
    def __init__(self, gpr_dir: Path) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)
        self.models = joblib.load(gpr_dir / "gpr_models.pkl")
        self.scalers = joblib.load(gpr_dir / "gpr_scalers.pkl")
        self.bounds = joblib.load(gpr_dir / "stage_bounds.pkl")
        self.stage_refs: dict[int, float] = {0: 1.0, 1: 1.0}
        self.stage_support: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for stage, model in self.models.items():
            X_train = self.scalers[stage].inverse_transform(model.X_train_)
            self.stage_support[int(stage)] = (X_train.min(axis=0), X_train.max(axis=0))

    def stable_stage(self, t_kcycles: float) -> int:
        q1 = float(self.bounds["q1"])
        q2 = float(self.bounds["q2"])
        if t_kcycles <= q1:
            return 0
        if t_kcycles <= q2:
            return 1
        return 1

    def raw_factor(self, F: float, D: float, elastic_modulus_gpa: float, t_kcycles: float) -> tuple[float, float, int]:
        t = max(float(t_kcycles), 5.0)
        t = min(t, 60.0)
        stage = self.stable_stage(t)
        X = np.array([[float(F) / 1000.0, float(elastic_modulus_gpa), float(D) / 2.0, t]], dtype=float)
        X_scaled = self.scalers[stage].transform(X)
        log_alpha, log_std = self.models[stage].predict(X_scaled, return_std=True)
        return float(np.exp(log_alpha[0])), float(log_std[0]), int(stage)

    def in_domain(
        self,
        F: float,
        D: float,
        elastic_modulus_gpa: float,
        t_kcycles: float,
        stage: int,
        tolerance: float,
    ) -> bool:
        if stage not in self.stage_support:
            return False
        low, high = self.stage_support[stage]
        x = np.array([float(F) / 1000.0, float(elastic_modulus_gpa), float(D) / 2.0, float(t_kcycles)], dtype=float)
        tol = max(float(tolerance), 0.0)
        lower = low * (1.0 - tol)
        upper = high * (1.0 + tol)
        return bool(np.all(x >= lower) and np.all(x <= upper))

    def fit_reference(self, train_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> dict[int, float]:
        values: dict[int, list[float]] = {0: [], 1: []}
        for row in train_df.itertuples(index=False):
            table = case_tables[str(row.file_name)]
            for point in table.itertuples(index=False):
                t_kcycles = float(point.actual_cycle) / 1000.0
                if t_kcycles > 60.0:
                    continue
                raw, _, stage = self.raw_factor(
                    F=float(point.F),
                    D=float(point.D),
                    elastic_modulus_gpa=float(point.elastic_modulus_GPa),
                    t_kcycles=t_kcycles,
                )
                if raw > 0.0 and np.isfinite(raw):
                    values[stage].append(raw)
        refs: dict[int, float] = {}
        for stage, vals in values.items():
            if vals:
                arr = np.asarray(vals, dtype=float)
                refs[stage] = float(np.exp(np.mean(np.log(np.clip(arr, common.EPS, None)))))
            else:
                refs[stage] = 1.0
        self.stage_refs = refs
        return refs

    def factor(
        self,
        *,
        F: float,
        D: float,
        elastic_modulus_gpa: float,
        actual_cycle: float,
        config: DynamicKConfig,
    ) -> tuple[float, float, float, int]:
        t_kcycles = float(actual_cycle) / 1000.0
        if t_kcycles > config.late_t_kcycles and config.late_mode == "unit_after60":
            return 1.0, 1.0, 0.0, 1
        t_guard = min(max(t_kcycles, config.min_t_kcycles), config.late_t_kcycles)
        stage = self.stable_stage(t_guard)
        if not self.in_domain(F, D, elastic_modulus_gpa, t_guard, stage, config.domain_tolerance):
            return 1.0, 1.0, 0.0, stage
        raw, log_std, stage = self.raw_factor(F, D, elastic_modulus_gpa, t_kcycles)
        ref = max(float(self.stage_refs.get(stage, 1.0)), common.EPS)
        centered = raw / ref
        factor = 1.0 + float(config.strength) * (centered - 1.0)
        low = max(0.01, 1.0 - float(config.clip_width))
        high = 1.0 + float(config.clip_width)
        factor = float(np.clip(factor, low, high))
        return factor, raw, log_std, stage


def dynamic_rollout_case(
    *,
    model: torch.nn.Module,
    seq_scaler: common.FeatureScaler,
    target_scaler: common.TargetScaler,
    case_df: pd.DataFrame,
    threshold_um: float,
    base_k: float,
    true_life_actual: float,
    feature_spec: common.FeatureSpec,
    seq_len: int,
    actual_step_cap: float | None,
    gpr_factor: GprWearFactor,
    dynamic_config: DynamicKConfig,
) -> tuple[pd.DataFrame, float, common.RolloutStepInfo]:
    step_info = common.resolve_rollout_steps(case_df, actual_step_cap=actual_step_cap)
    first = case_df.iloc[0]
    F = float(first["F"])
    D = float(first["D"])
    Cr = float(first["Cr"])
    elastic_modulus = float(first.get("elastic_modulus_GPa", 210.0))
    threshold_mm = threshold_um / 1000.0

    actual_cycle = 0.0
    sim_cycle = 0.0
    wear_depth = 0.0
    history: list[list[float]] = [
        common.build_history_row(F, D, Cr, actual_cycle, wear_depth, feature_spec, prev_stress=0.0)
    ]
    rows: list[dict[str, float]] = []
    predicted_life_actual = true_life_actual
    actual_step = step_info.used_actual_step
    sim_step = step_info.used_sim_step

    internal_limit = max(true_life_actual * 1.35, float(case_df["actual_cycle"].max()) * 1.15, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + common.MAX_EXTRA_STEPS

    for _ in range(max_steps):
        pred_stress = common.predict_pressure_from_history(
            model, seq_scaler, target_scaler, history, feature_spec, seq_len
        )
        k_factor, raw_factor, log_std, k_stage = gpr_factor.factor(
            F=F,
            D=D,
            elastic_modulus_gpa=elastic_modulus,
            actual_cycle=actual_cycle,
            config=dynamic_config,
        )
        k_eff = base_k * k_factor
        rows.append(
            {
                "sim_cycle": sim_cycle,
                "actual_cycle": actual_cycle,
                "pred_stress": pred_stress,
                "pred_wear_depth_um": wear_depth * 1000.0,
                "k_factor": k_factor,
                "raw_k_factor": raw_factor,
                "k_log_std": log_std,
                "k_stage": float(k_stage),
                "k_eff": k_eff,
            }
        )

        delta_s = actual_step * math.pi * D / 6.0
        delta_wear = k_eff * pred_stress * delta_s
        next_actual_cycle = actual_cycle + actual_step
        next_wear_depth = wear_depth + delta_wear
        next_sim_cycle = sim_cycle + sim_step

        if next_wear_depth >= threshold_mm:
            ratio = (threshold_mm - wear_depth) / max(delta_wear, common.EPS)
            predicted_life_actual = actual_cycle + ratio * actual_step
            rows.append(
                {
                    "sim_cycle": sim_cycle + ratio * sim_step,
                    "actual_cycle": predicted_life_actual,
                    "pred_stress": pred_stress,
                    "pred_wear_depth_um": threshold_um,
                    "k_factor": k_factor,
                    "raw_k_factor": raw_factor,
                    "k_log_std": log_std,
                    "k_stage": float(k_stage),
                    "k_eff": k_eff,
                }
            )
            break

        actual_cycle = next_actual_cycle
        sim_cycle = next_sim_cycle
        wear_depth = next_wear_depth
        history.append(common.build_history_row(F, D, Cr, actual_cycle, wear_depth, feature_spec, prev_stress=pred_stress))

    return pd.DataFrame(rows), float(predicted_life_actual), step_info


def fixed_factor_rollout_case(
    *,
    model: torch.nn.Module,
    seq_scaler: common.FeatureScaler,
    target_scaler: common.TargetScaler,
    case_df: pd.DataFrame,
    threshold_um: float,
    base_k: float,
    k_factor: float,
    true_life_actual: float,
    feature_spec: common.FeatureSpec,
    seq_len: int,
    actual_step_cap: float | None,
) -> tuple[pd.DataFrame, float, common.RolloutStepInfo]:
    step_info = common.resolve_rollout_steps(case_df, actual_step_cap=actual_step_cap)
    first = case_df.iloc[0]
    F = float(first["F"])
    D = float(first["D"])
    Cr = float(first["Cr"])
    threshold_mm = threshold_um / 1000.0

    actual_cycle = 0.0
    sim_cycle = 0.0
    wear_depth = 0.0
    history: list[list[float]] = [
        common.build_history_row(F, D, Cr, actual_cycle, wear_depth, feature_spec, prev_stress=0.0)
    ]
    rows: list[dict[str, float]] = []
    predicted_life_actual = true_life_actual
    actual_step = step_info.used_actual_step
    sim_step = step_info.used_sim_step
    k_eff = base_k * float(k_factor)

    internal_limit = max(true_life_actual * 1.35, float(case_df["actual_cycle"].max()) * 1.15, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + common.MAX_EXTRA_STEPS

    for _ in range(max_steps):
        pred_stress = common.predict_pressure_from_history(
            model, seq_scaler, target_scaler, history, feature_spec, seq_len
        )
        rows.append(
            {
                "sim_cycle": sim_cycle,
                "actual_cycle": actual_cycle,
                "pred_stress": pred_stress,
                "pred_wear_depth_um": wear_depth * 1000.0,
                "k_factor": float(k_factor),
                "raw_k_factor": float(k_factor),
                "k_log_std": 0.0,
                "k_stage": -2.0,
                "k_eff": k_eff,
            }
        )

        delta_s = actual_step * math.pi * D / 6.0
        delta_wear = k_eff * pred_stress * delta_s
        next_actual_cycle = actual_cycle + actual_step
        next_wear_depth = wear_depth + delta_wear
        next_sim_cycle = sim_cycle + sim_step

        if next_wear_depth >= threshold_mm:
            ratio = (threshold_mm - wear_depth) / max(delta_wear, common.EPS)
            predicted_life_actual = actual_cycle + ratio * actual_step
            rows.append(
                {
                    "sim_cycle": sim_cycle + ratio * sim_step,
                    "actual_cycle": predicted_life_actual,
                    "pred_stress": pred_stress,
                    "pred_wear_depth_um": threshold_um,
                    "k_factor": float(k_factor),
                    "raw_k_factor": float(k_factor),
                    "k_log_std": 0.0,
                    "k_stage": -2.0,
                    "k_eff": k_eff,
                }
            )
            break

        actual_cycle = next_actual_cycle
        sim_cycle = next_sim_cycle
        wear_depth = next_wear_depth
        history.append(common.build_history_row(F, D, Cr, actual_cycle, wear_depth, feature_spec, prev_stress=pred_stress))

    return pd.DataFrame(rows), float(predicted_life_actual), step_info


def calibration_features(case_df: pd.DataFrame, rollout_df: pd.DataFrame, predicted_life: float) -> np.ndarray:
    first = case_df.iloc[0]
    F = float(first["F"])
    D = float(first["D"])
    Cr = float(first["Cr"])
    pred_stress = rollout_df["pred_stress"].to_numpy(dtype=float)
    if len(pred_stress) == 0:
        pred_stress = np.array([0.0], dtype=float)
    start = float(pred_stress[0])
    mean = float(np.mean(pred_stress))
    end = float(pred_stress[-1])
    drop_ratio = (start - end) / max(start, common.EPS)
    return np.array(
        [
            F / 1000.0,
            D,
            Cr,
            F / max(D * D, common.EPS),
            Cr / max(D, common.EPS),
            math.log1p(max(float(predicted_life), 0.0)),
            start,
            mean,
            end,
            drop_ratio,
        ],
        dtype=float,
    )


def fit_closed_loop_k_calibrator(
    *,
    model: torch.nn.Module,
    seq_scaler: common.FeatureScaler,
    target_scaler: common.TargetScaler,
    train_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
    feature_spec: common.FeatureSpec,
    seq_len: int,
    actual_step_cap: float | None,
) -> tuple[object, pd.DataFrame]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    audit_rows: list[dict[str, float | str]] = []
    for row in train_df.itertuples(index=False):
        case_df = case_tables[str(row.file_name)]
        true_life = float(row.actual_life)
        rollout_df, predicted_life, _ = fixed_rollout_case(
            model=model,
            seq_scaler=seq_scaler,
            target_scaler=target_scaler,
            case_df=case_df,
            threshold_um=common.WEAR_THRESHOLD_UM,
            base_k=common.REAL_WEAR_COEFF_MPA_INV,
            true_life_actual=true_life,
            feature_spec=feature_spec,
            seq_len=seq_len,
            actual_step_cap=actual_step_cap,
        )
        target_factor = np.clip(predicted_life / max(true_life, common.EPS), 0.70, 1.30)
        X_rows.append(calibration_features(case_df, rollout_df, predicted_life))
        y_rows.append(float(np.log(target_factor)))
        audit_rows.append(
            {
                "file_name": str(row.file_name),
                "source_file": str(row.source_file),
                "predicted_life": float(predicted_life),
                "true_life": true_life,
                "target_k_factor": float(target_factor),
            }
        )
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(np.vstack(X_rows), np.asarray(y_rows, dtype=float))
    return reg, pd.DataFrame(audit_rows)


def fixed_rollout_case(
    *,
    model: torch.nn.Module,
    seq_scaler: common.FeatureScaler,
    target_scaler: common.TargetScaler,
    case_df: pd.DataFrame,
    threshold_um: float,
    base_k: float,
    true_life_actual: float,
    feature_spec: common.FeatureSpec,
    seq_len: int,
    actual_step_cap: float | None,
) -> tuple[pd.DataFrame, float, common.RolloutStepInfo]:
    rollout_df, predicted_life, step_info = common.rollout_case_with_step_cap(
        model,
        seq_scaler,
        target_scaler,
        case_df,
        threshold_um,
        base_k,
        true_life_actual,
        feature_spec,
        seq_len,
        actual_step_cap=actual_step_cap,
    )
    rollout_df = rollout_df.copy()
    rollout_df["k_factor"] = 1.0
    rollout_df["raw_k_factor"] = 1.0
    rollout_df["k_log_std"] = 0.0
    rollout_df["k_stage"] = -1.0
    rollout_df["k_eff"] = base_k
    return rollout_df, predicted_life, step_info


def summarize_detail(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fixed = detail_df.loc[detail_df["variant"].eq("T3_fixed_k")]
    fixed_mean = float(fixed["life_abs_error"].mean()) if not fixed.empty else float("nan")

    key_cols = ["seed", "test_case"]
    fixed_pair = fixed.set_index(key_cols)["life_abs_error"]
    for variant, part in detail_df.groupby("variant", sort=False):
        part = part.copy()
        pair = part.set_index(key_cols)["life_abs_error"]
        common_index = pair.index.intersection(fixed_pair.index)
        if variant == "T3_fixed_k" or len(common_index) == 0:
            win_rate = float("nan")
            paired_delta = float("nan")
        else:
            delta = pair.loc[common_index] - fixed_pair.loc[common_index]
            win_rate = float((delta < 0).mean())
            paired_delta = float(delta.mean())
        rows.append(
            {
                "variant": variant,
                "seed_count": int(part["seed"].nunique()),
                "case_count": int(part["test_case"].nunique()),
                "mean_pressure_mae": float(part["pressure_mae"].mean()),
                "mean_wear_mae_um": float(part["wear_mae_um"].mean()),
                "mean_life_abs_error": float(part["life_abs_error"].mean()),
                "std_life_abs_error": float(part["life_abs_error"].std(ddof=0)),
                "delta_vs_fixed_mean": float(part["life_abs_error"].mean() - fixed_mean),
                "paired_delta_vs_fixed": paired_delta,
                "win_rate_vs_fixed": win_rate,
                "mean_k_factor": float(part["mean_k_factor"].mean()),
                "min_k_factor": float(part["min_k_factor"].min()),
                "max_k_factor": float(part["max_k_factor"].max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_life_abs_error").reset_index(drop=True)

    per_case_rows = []
    for (variant, test_case), part in detail_df.groupby(["variant", "test_case"], sort=False):
        per_case_rows.append(
            {
                "variant": variant,
                "test_case": test_case,
                "seed_count": int(part["seed"].nunique()),
                "mean_predicted_life": float(part["predicted_life"].mean()),
                "true_life": float(part["true_life"].iloc[0]),
                "mean_life_abs_error": float(part["life_abs_error"].mean()),
                "mean_wear_mae_um": float(part["wear_mae_um"].mean()),
                "mean_k_factor": float(part["mean_k_factor"].mean()),
            }
        )
    per_case = pd.DataFrame(per_case_rows).sort_values(["variant", "test_case"]).reset_index(drop=True)
    return summary, per_case


def save_summary_chart(summary_df: pd.DataFrame, out_path: Path) -> None:
    top = summary_df.sort_values("mean_life_abs_error").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563eb" if v == "T3_fixed_k" else "#dc2626" for v in top["variant"]]
    ax.barh(top["variant"], top["mean_life_abs_error"], color=colors)
    ax.set_xlabel("Mean life absolute error (cycles)")
    ax.set_title("Dynamic wear coefficient sweep")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(top["mean_life_abs_error"]):
        ax.text(value + 10, idx, f"{value:.0f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def evaluate(
    *,
    out_dir: Path,
    seeds: list[int],
    epochs: int,
    strengths: list[float],
    clip_widths: list[float],
    late_mode: str,
    domain_tolerance: float,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment. Use E:\\AI\\cuda_env\\python.exe with CUDA.")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, case_tables = load_formal_split()
    train_tables = train_tables_from_split(train_df, case_tables)

    gpr_factor = GprWearFactor(GPR_DIR)
    stage_refs = gpr_factor.fit_reference(train_df, case_tables)

    dynamic_configs = [
        DynamicKConfig(
            name=f"T4_dynk_s{safe_name(strength)}_c{safe_name(width)}_d{safe_name(domain_tolerance)}",
            strength=strength,
            clip_width=width,
            late_mode=late_mode,
            domain_tolerance=domain_tolerance,
        )
        for strength in strengths
        for width in clip_widths
    ]

    payload = {
        "seeds": seeds,
        "epochs": epochs,
        "seq_len": SEQ_LEN,
        "actual_step_cap": STEP_CAP,
        "base_wear_coeff": common.REAL_WEAR_COEFF_MPA_INV,
        "test_files": TEST_FILES,
        "stage_refs": stage_refs,
        "dynamic_configs": [asdict(cfg) for cfg in dynamic_configs],
        "calibration": {
            "method": "Ridge on training closed-loop residuals",
            "strengths": CALIB_STRENGTHS,
            "clip_widths": CALIB_CLIP_WIDTHS,
            "target_factor": "clipped(predicted_life_fixed / true_life_train, 0.70, 1.30)",
        },
        "device": common.device_summary_lines(),
    }
    (out_dir / CONFIG_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    train_raw_seq, train_y, train_case_names, train_step_indices = common.build_raw_sequence_dataset(train_tables, SEQ_LEN)
    feature_spec = common.FEATURE_R2_SPEC
    shape_config = final_shape_config()

    detail_rows: list[dict[str, object]] = []
    for seed in seeds:
        print(f"[train] seed={seed}, epochs={epochs}", flush=True)
        common.set_seed(seed)
        model, seq_scaler, target_scaler = common.train_model(
            model_name="Transformer",
            train_raw_seq=train_raw_seq,
            train_y=train_y,
            feature_spec=feature_spec,
            config=make_config(epochs),
            shape_config=shape_config,
            train_case_names=train_case_names,
            train_step_indices=train_step_indices,
        )
        k_calibrator, calib_audit_df = fit_closed_loop_k_calibrator(
            model=model,
            seq_scaler=seq_scaler,
            target_scaler=target_scaler,
            train_df=train_df,
            case_tables=case_tables,
            feature_spec=feature_spec,
            seq_len=SEQ_LEN,
            actual_step_cap=STEP_CAP,
        )
        calib_audit_df.to_csv(out_dir / f"calibration_train_seed_{seed}.csv", index=False, encoding="utf-8-sig")

        for test_row in test_df.itertuples(index=False):
            test_file = str(test_row.file_name)
            test_table = case_tables[test_file]
            true_life = float(test_row.actual_life)
            true_curve_df = common.threshold_ground_truth(test_table, common.WEAR_THRESHOLD_UM, pd.Series(test_row._asdict()))
            test_raw_seq, test_y, _, _ = common.build_raw_sequence_dataset({test_file: test_table}, SEQ_LEN)
            pressure_metrics = common.evaluate_pressure(
                model,
                seq_scaler,
                target_scaler,
                test_raw_seq,
                test_y,
                feature_spec,
            )

            rollout_jobs: list[tuple[str, pd.DataFrame, float, common.RolloutStepInfo]] = []
            fixed_df, fixed_life, fixed_step = fixed_rollout_case(
                model=model,
                seq_scaler=seq_scaler,
                target_scaler=target_scaler,
                case_df=test_table,
                threshold_um=common.WEAR_THRESHOLD_UM,
                base_k=common.REAL_WEAR_COEFF_MPA_INV,
                true_life_actual=true_life,
                feature_spec=feature_spec,
                seq_len=SEQ_LEN,
                actual_step_cap=STEP_CAP,
            )
            rollout_jobs.append(("T3_fixed_k", fixed_df, fixed_life, fixed_step))

            x_calib = calibration_features(test_table, fixed_df, fixed_life).reshape(1, -1)
            raw_calib_factor = float(np.exp(k_calibrator.predict(x_calib)[0]))
            for strength in CALIB_STRENGTHS:
                for width in CALIB_CLIP_WIDTHS:
                    factor = float(np.exp(float(strength) * np.log(max(raw_calib_factor, common.EPS))))
                    factor = float(np.clip(factor, 1.0 - width, 1.0 + width))
                    rollout_df, predicted_life, step_info = fixed_factor_rollout_case(
                        model=model,
                        seq_scaler=seq_scaler,
                        target_scaler=target_scaler,
                        case_df=test_table,
                        threshold_um=common.WEAR_THRESHOLD_UM,
                        base_k=common.REAL_WEAR_COEFF_MPA_INV,
                        k_factor=factor,
                        true_life_actual=true_life,
                        feature_spec=feature_spec,
                        seq_len=SEQ_LEN,
                        actual_step_cap=STEP_CAP,
                    )
                    variant = f"T5_calibk_s{safe_name(strength)}_c{safe_name(width)}"
                    rollout_df["raw_k_factor"] = raw_calib_factor
                    rollout_jobs.append((variant, rollout_df, predicted_life, step_info))

            for cfg in dynamic_configs:
                rollout_df, predicted_life, step_info = dynamic_rollout_case(
                    model=model,
                    seq_scaler=seq_scaler,
                    target_scaler=target_scaler,
                    case_df=test_table,
                    threshold_um=common.WEAR_THRESHOLD_UM,
                    base_k=common.REAL_WEAR_COEFF_MPA_INV,
                    true_life_actual=true_life,
                    feature_spec=feature_spec,
                    seq_len=SEQ_LEN,
                    actual_step_cap=STEP_CAP,
                    gpr_factor=gpr_factor,
                    dynamic_config=cfg,
                )
                rollout_jobs.append((cfg.name, rollout_df, predicted_life, step_info))

            for variant, rollout_df, predicted_life, step_info in rollout_jobs:
                detail_rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "epochs": int(epochs),
                        "test_case": str(test_row.source_file),
                        "file_name": test_file,
                        "pressure_mae": pressure_metrics["pressure_mae"],
                        "pressure_rmse": pressure_metrics["pressure_rmse"],
                        "pressure_mape": pressure_metrics["pressure_mape"],
                        "wear_mae_um": common.wear_curve_mae(true_curve_df, rollout_df),
                        "predicted_life": predicted_life,
                        "true_life": true_life,
                        "life_abs_error": abs(predicted_life - true_life),
                        "life_rel_error": abs(predicted_life - true_life) / max(true_life, common.EPS),
                        "native_actual_step": step_info.native_actual_step,
                        "used_actual_step": step_info.used_actual_step,
                        "feature_variant": "M1_R2_log_cycle_replace",
                        "shape_variant": "S1_slow_abs_0p01",
                        "mean_k_factor": float(rollout_df["k_factor"].mean()),
                        "min_k_factor": float(rollout_df["k_factor"].min()),
                        "max_k_factor": float(rollout_df["k_factor"].max()),
                        "mean_raw_k_factor": float(rollout_df["raw_k_factor"].mean()),
                        "mean_k_log_std": float(rollout_df["k_log_std"].mean()),
                    }
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(out_dir / DETAIL_NAME, index=False, encoding="utf-8-sig")
        summary_df, per_case_df = summarize_detail(detail_df)
        summary_df.to_csv(out_dir / SUMMARY_NAME, index=False, encoding="utf-8-sig")
        per_case_df.to_csv(out_dir / PER_CASE_NAME, index=False, encoding="utf-8-sig")
        print(summary_df.head(8).to_string(index=False), flush=True)

    detail_df = pd.DataFrame(detail_rows)
    summary_df, per_case_df = summarize_detail(detail_df)
    detail_df.to_csv(out_dir / DETAIL_NAME, index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / SUMMARY_NAME, index=False, encoding="utf-8-sig")
    per_case_df.to_csv(out_dir / PER_CASE_NAME, index=False, encoding="utf-8-sig")
    save_summary_chart(summary_df, out_dir / "summary_dynamic_k.png")

    best = summary_df.iloc[0].to_dict()
    print("[done] best variant:", best["variant"], "mean_life_abs_error=", best["mean_life_abs_error"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results" / "dynamic_k_sweep")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--strengths", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--clip-widths", default="0.05,0.1,0.15,0.2")
    parser.add_argument("--late-mode", choices=["freeze60", "unit_after60"], default="freeze60")
    parser.add_argument("--domain-tolerance", type=float, default=0.10)
    args = parser.parse_args()

    evaluate(
        out_dir=args.out_dir,
        seeds=parse_int_list(args.seeds),
        epochs=int(args.epochs),
        strengths=parse_float_list(args.strengths),
        clip_widths=parse_float_list(args.clip_widths),
        late_mode=str(args.late_mode),
        domain_tolerance=float(args.domain_tolerance),
    )


if __name__ == "__main__":
    main()
