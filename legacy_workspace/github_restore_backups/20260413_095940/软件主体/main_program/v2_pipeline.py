from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from device_runtime import cpu_state_dict, infer_model_device, resolve_training_device
from train_real_wear_models import (
    EPS,
    comparison_dir,
    load_cases,
    load_processing_config,
    median_positive_diff,
    processing_config_path,
    recommended_cycle_step,
    software_model_path,
    threshold_ground_truth,
    trained_model_dir,
)


PRIMARY_SEED = 20260408
FINALIST_SEEDS = [20260408, 20260418, 20260428]

DEFAULT_MAX_EPOCHS = 600
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 60
DEFAULT_GRAD_CLIP = 1.0
MAX_EXTRA_STEPS = 400
LITE_MAX_EPOCHS = 200
LITE_PATIENCE = 20
LITE_SCREENING_FOLD_COUNT = 4

V1_SEQUENCE_LENGTH = 6
ROUND1_SEQUENCE_LENGTH = 10
ROUND2_SEQUENCE_LENGTHS = [6, 10, 15]
ROUND2_WEAR_LAMBDAS = [0.0, 0.1, 0.25]

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002

STATIC_FEATURE_ORDER = ["F", "D", "Cr", "F_over_D_sq", "Cr_over_D"]
STATIC_FEATURE_WITH_STEP_ORDER = STATIC_FEATURE_ORDER + ["actual_cycle_step"]
DYNAMIC_FEATURE_ORDER = ["log1p_actual_cycle", "wear_depth", "delta_cycle", "delta_wear"]
LEGACY_FEATURE_ORDER = ["F", "D", "Cr", "actual_cycle", "wear_depth"]


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    model_kind: str
    sequence_length: int
    seed: int = PRIMARY_SEED
    dropout: float = 0.0
    wear_consistency_lambda: float = 0.0
    run_stress_mono_lambda: float = 0.0
    run_stress_slow_lambda: float = 0.0
    use_cycle_step: bool = False
    use_validation: bool = True
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    grad_clip_norm: float = DEFAULT_GRAD_CLIP

    @property
    def feature_version(self) -> str:
        return "v2" if self.model_kind.startswith("v2_") else "v1"

    @property
    def static_feature_order(self) -> list[str]:
        if self.feature_version != "v2":
            return []
        if self.use_cycle_step:
            return STATIC_FEATURE_WITH_STEP_ORDER.copy()
        return STATIC_FEATURE_ORDER.copy()

    @property
    def dynamic_feature_order(self) -> list[str]:
        if self.feature_version != "v2":
            return []
        return DYNAMIC_FEATURE_ORDER.copy()

    @property
    def input_feature_order(self) -> list[str]:
        if self.model_kind == "v1_transformer":
            return LEGACY_FEATURE_ORDER.copy()
        if self.model_kind == "v2_single_branch":
            return self.static_feature_order + self.dynamic_feature_order
        return []

    def with_updates(self, **kwargs: Any) -> "CandidateConfig":
        return replace(self, **kwargs)

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_name": self.name,
            "model_kind": self.model_kind,
            "feature_version": self.feature_version,
            "sequence_length": int(self.sequence_length),
            "seed": int(self.seed),
            "dropout": float(self.dropout),
            "wear_consistency_lambda": float(self.wear_consistency_lambda),
            "run_stress_mono_lambda": float(self.run_stress_mono_lambda),
            "run_stress_slow_lambda": float(self.run_stress_slow_lambda),
            "use_cycle_step": bool(self.use_cycle_step),
            "use_validation": bool(self.use_validation),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "grad_clip_norm": float(self.grad_clip_norm),
        }


@dataclass
class SequenceDataset:
    raw_sequences: np.ndarray
    target_log: np.ndarray
    next_delta_wear: np.ndarray
    actual_step: np.ndarray
    case_names: list[str]
    step_index: np.ndarray

    def __len__(self) -> int:
        return int(self.raw_sequences.shape[0])

    def slice(self, indices: np.ndarray) -> "SequenceDataset":
        return SequenceDataset(
            raw_sequences=self.raw_sequences[indices],
            target_log=self.target_log[indices],
            next_delta_wear=self.next_delta_wear[indices],
            actual_step=self.actual_step[indices],
            case_names=[self.case_names[int(idx)] for idx in indices],
            step_index=self.step_index[indices],
        )


@dataclass
class RunShapeDataset:
    mono_raw_sequences: np.ndarray
    mono_actual_step: np.ndarray
    slow_raw_sequences: np.ndarray
    slow_actual_step: np.ndarray

    @property
    def mono_count(self) -> int:
        return int(self.mono_raw_sequences.shape[0])

    @property
    def slow_count(self) -> int:
        return int(self.slow_raw_sequences.shape[0])

    def slice(
        self,
        mono_indices: np.ndarray | None = None,
        slow_indices: np.ndarray | None = None,
        use_full_on_none: bool = True,
    ) -> "RunShapeDataset":
        mono_raw = self.mono_raw_sequences if mono_indices is None and use_full_on_none else self.mono_raw_sequences[:0]
        mono_step = self.mono_actual_step if mono_indices is None and use_full_on_none else self.mono_actual_step[:0]
        slow_raw = self.slow_raw_sequences if slow_indices is None and use_full_on_none else self.slow_raw_sequences[:0]
        slow_step = self.slow_actual_step if slow_indices is None and use_full_on_none else self.slow_actual_step[:0]
        if mono_indices is not None:
            mono_raw = self.mono_raw_sequences[mono_indices]
            mono_step = self.mono_actual_step[mono_indices]
        if slow_indices is not None:
            slow_raw = self.slow_raw_sequences[slow_indices]
            slow_step = self.slow_actual_step[slow_indices]
        return RunShapeDataset(
            mono_raw_sequences=mono_raw,
            mono_actual_step=mono_step,
            slow_raw_sequences=slow_raw,
            slow_actual_step=slow_step,
        )


@dataclass
class TrainedCandidate:
    config: CandidateConfig
    model: nn.Module
    input_scalers: dict[str, "ArrayScaler"]
    target_scaler: "ArrayScaler"
    best_epoch: int
    monitored_loss: float
    training_sample_count: int
    validation_sample_count: int


class ArrayScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, array: np.ndarray) -> "ArrayScaler":
        values = np.asarray(array, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def load(self, mean: np.ndarray, std: np.ndarray) -> "ArrayScaler":
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        return (values - self.mean) / self.std

    def inverse_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std_t = torch.tensor(self.std, dtype=torch.float32, device=tensor.device)
        return tensor * std_t + mean_t


class SequenceTransformerNet(nn.Module):
    def __init__(self, input_size: int, seq_len: int, d_model: int = 32, nhead: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence_x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(sequence_x) + self.pos_embed[:, : sequence_x.shape[1], :]
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class StaticDynamicTransformerNet(nn.Module):
    def __init__(
        self,
        static_input_size: int,
        dynamic_input_size: int,
        seq_len: int,
        d_model: int = 32,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(static_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.dynamic_proj = nn.Linear(dynamic_input_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, dynamic_x: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        static_ctx = self.static_net(static_x)
        dynamic_ctx = self.dynamic_proj(dynamic_x) + self.pos_embed[:, : dynamic_x.shape[1], :]
        dynamic_ctx = self.encoder(dynamic_ctx)
        fused = torch.cat([dynamic_ctx[:, -1, :], static_ctx], dim=-1)
        return self.head(fused)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, device_label = resolve_training_device()
    if device_label == "cuda":
        torch.cuda.manual_seed_all(seed)


def config_dir() -> Path:
    return processing_config_path().parent


def v2_candidate_output_dir() -> Path:
    path = comparison_dir() / "v2_candidate_benchmark"
    path.mkdir(parents=True, exist_ok=True)
    return path


def v2_lite_candidate_output_dir() -> Path:
    path = comparison_dir() / "v2_candidate_benchmark_lite"
    path.mkdir(parents=True, exist_ok=True)
    return path


def v2_shape_loss_output_dir() -> Path:
    path = comparison_dir() / "v2_shape_loss_benchmark"
    path.mkdir(parents=True, exist_ok=True)
    return path


def v2_lite_shape_loss_output_dir() -> Path:
    path = comparison_dir() / "v2_shape_loss_benchmark_lite"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fixed_split_manifest_path() -> Path:
    return config_dir() / "v2_fixed_splits.json"


def safe_stem(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "candidate"


def load_environment() -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    config = load_processing_config()
    summary_df, case_tables = load_cases()
    return config, summary_df, case_tables


def measured_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[summary_df["has_measured_pressure"].astype(bool)].reset_index(drop=True)


def eligible_summary(summary_df: pd.DataFrame, threshold_um: float) -> pd.DataFrame:
    measured_df = measured_summary(summary_df)
    return measured_df[measured_df["final_wear_um"].astype(float) >= float(threshold_um)].reset_index(drop=True)


def build_fixed_split_manifest(summary_df: pd.DataFrame, threshold_um: float) -> dict[str, Any]:
    measured_df = measured_summary(summary_df).sort_values(["actual_life", "file_name"]).reset_index(drop=True)
    eligible_names = set(eligible_summary(summary_df, threshold_um)["file_name"].astype(str).tolist())

    folds: list[dict[str, Any]] = []
    for _, test_row in measured_df.iterrows():
        test_file = str(test_row["file_name"])
        train_pool = measured_df[measured_df["file_name"] != test_file].sort_values(["actual_life", "file_name"]).reset_index(drop=True)
        if train_pool.empty:
            val_file = None
            train_runs: list[str] = []
        else:
            val_file = str(train_pool.iloc[len(train_pool) // 2]["file_name"])
            train_runs = [str(name) for name in train_pool["file_name"].astype(str).tolist() if str(name) != val_file]

        folds.append(
            {
                "test_run": test_file,
                "test_source_file": str(test_row["source_file"]),
                "eligible_for_life": bool(test_file in eligible_names),
                "val_run": val_file,
                "train_runs": train_runs,
            }
        )

    return {
        "version": 1,
        "threshold_um": float(threshold_um),
        "measured_runs": measured_df["file_name"].astype(str).tolist(),
        "eligible_life_runs": sorted([str(name) for name in eligible_names]),
        "folds": folds,
    }


def load_or_create_split_manifest(summary_df: pd.DataFrame, threshold_um: float) -> dict[str, Any]:
    manifest = build_fixed_split_manifest(summary_df, threshold_um)
    path = fixed_split_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _evenly_spaced_indices(total_count: int, target_count: int) -> list[int]:
    if total_count <= 0 or target_count <= 0:
        return []
    if total_count <= target_count:
        return list(range(total_count))
    raw = np.linspace(0, total_count - 1, num=target_count)
    indices: list[int] = []
    for value in raw:
        index = int(round(float(value)))
        if index not in indices:
            indices.append(index)
    for index in range(total_count):
        if len(indices) >= target_count:
            break
        if index not in indices:
            indices.append(index)
    return sorted(indices[:target_count])


def build_screening_manifest(
    manifest: dict[str, Any],
    summary_df: pd.DataFrame,
    threshold_um: float,
    fold_count: int = LITE_SCREENING_FOLD_COUNT,
) -> dict[str, Any]:
    eligible_df = eligible_summary(summary_df, threshold_um).sort_values(["actual_life", "file_name"]).reset_index(drop=True)
    source_df = eligible_df if not eligible_df.empty else measured_summary(summary_df).sort_values(["actual_life", "file_name"]).reset_index(drop=True)
    indices = _evenly_spaced_indices(len(source_df), fold_count)
    selected_runs = [str(source_df.iloc[index]["file_name"]) for index in indices]
    selected_set = set(selected_runs)
    screening_folds = [dict(fold) for fold in manifest["folds"] if str(fold["test_run"]) in selected_set]
    return {
        **manifest,
        "mode": "lite",
        "base_fold_count": int(len(manifest["folds"])),
        "screening_fold_count": int(len(screening_folds)),
        "screening_test_runs": selected_runs,
        "folds": screening_folds,
    }


def legacy_v1_baseline_config() -> CandidateConfig:
    return CandidateConfig(
        name="V1_frozen_baseline|seq=6",
        model_kind="v1_transformer",
        sequence_length=V1_SEQUENCE_LENGTH,
        seed=PRIMARY_SEED,
        dropout=0.0,
        wear_consistency_lambda=0.0,
        use_cycle_step=False,
        use_validation=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        batch_size=DEFAULT_BATCH_SIZE,
        max_epochs=1200,
        patience=DEFAULT_PATIENCE,
        grad_clip_norm=0.0,
    )


def make_v1_protocol_candidate(sequence_length: int = ROUND1_SEQUENCE_LENGTH, seed: int = PRIMARY_SEED) -> CandidateConfig:
    return CandidateConfig(
        name=f"V1_protocol|seq={sequence_length}",
        model_kind="v1_transformer",
        sequence_length=sequence_length,
        seed=seed,
        dropout=0.0,
        wear_consistency_lambda=0.0,
        use_cycle_step=False,
        use_validation=True,
        learning_rate=1e-3,
        weight_decay=1e-6,
        batch_size=DEFAULT_BATCH_SIZE,
        max_epochs=1200,
        patience=DEFAULT_PATIENCE,
        grad_clip_norm=0.0,
    )


def make_lite_config(config: CandidateConfig) -> CandidateConfig:
    return config.with_updates(
        max_epochs=LITE_MAX_EPOCHS,
        patience=LITE_PATIENCE,
    )


def uses_run_shape_losses(config: CandidateConfig) -> bool:
    return bool(config.run_stress_mono_lambda > 0.0 or config.run_stress_slow_lambda > 0.0)


def build_candidate_name(model_kind: str, sequence_length: int, wear_consistency_lambda: float, use_cycle_step: bool) -> str:
    if model_kind == "v2_single_branch":
        prefix = "V2_single_branch"
    elif model_kind == "v2_dual_branch":
        prefix = "V2_dual_branch"
    else:
        prefix = model_kind
    return (
        f"{prefix}|seq={int(sequence_length)}"
        f"|wear={float(wear_consistency_lambda):.2f}"
        f"|step={'on' if use_cycle_step else 'off'}"
    )


def make_v2_single_branch_candidate(
    sequence_length: int = ROUND1_SEQUENCE_LENGTH,
    wear_consistency_lambda: float = 0.0,
    run_stress_mono_lambda: float = 0.0,
    run_stress_slow_lambda: float = 0.0,
    use_cycle_step: bool = False,
    seed: int = PRIMARY_SEED,
) -> CandidateConfig:
    name = build_candidate_name("v2_single_branch", sequence_length, wear_consistency_lambda, use_cycle_step)
    if run_stress_mono_lambda > 0.0 or run_stress_slow_lambda > 0.0:
        name = (
            f"{name}"
            f"|runmono={float(run_stress_mono_lambda):.2f}"
            f"|runslow={float(run_stress_slow_lambda):.2f}"
        )
    return CandidateConfig(
        name=name,
        model_kind="v2_single_branch",
        sequence_length=sequence_length,
        seed=seed,
        dropout=0.1,
        wear_consistency_lambda=wear_consistency_lambda,
        run_stress_mono_lambda=run_stress_mono_lambda,
        run_stress_slow_lambda=run_stress_slow_lambda,
        use_cycle_step=use_cycle_step,
        use_validation=True,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        batch_size=DEFAULT_BATCH_SIZE,
        max_epochs=DEFAULT_MAX_EPOCHS,
        patience=DEFAULT_PATIENCE,
        grad_clip_norm=DEFAULT_GRAD_CLIP,
    )


def make_v2_dual_branch_candidate(
    sequence_length: int = ROUND1_SEQUENCE_LENGTH,
    wear_consistency_lambda: float = 0.0,
    run_stress_mono_lambda: float = 0.0,
    run_stress_slow_lambda: float = 0.0,
    use_cycle_step: bool = False,
    seed: int = PRIMARY_SEED,
) -> CandidateConfig:
    name = build_candidate_name("v2_dual_branch", sequence_length, wear_consistency_lambda, use_cycle_step)
    if run_stress_mono_lambda > 0.0 or run_stress_slow_lambda > 0.0:
        name = (
            f"{name}"
            f"|runmono={float(run_stress_mono_lambda):.2f}"
            f"|runslow={float(run_stress_slow_lambda):.2f}"
        )
    return CandidateConfig(
        name=name,
        model_kind="v2_dual_branch",
        sequence_length=sequence_length,
        seed=seed,
        dropout=0.1,
        wear_consistency_lambda=wear_consistency_lambda,
        run_stress_mono_lambda=run_stress_mono_lambda,
        run_stress_slow_lambda=run_stress_slow_lambda,
        use_cycle_step=use_cycle_step,
        use_validation=True,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        batch_size=DEFAULT_BATCH_SIZE,
        max_epochs=DEFAULT_MAX_EPOCHS,
        patience=DEFAULT_PATIENCE,
        grad_clip_norm=DEFAULT_GRAD_CLIP,
    )


def frozen_v1_baseline_definition() -> dict[str, Any]:
    config = legacy_v1_baseline_config()
    record = config.to_record()
    record.update(
        {
            "selection_mode": "best_training_loss",
            "feature_order": LEGACY_FEATURE_ORDER.copy(),
            "architecture": {
                "encoder_layers": 2,
                "d_model": 32,
                "nhead": 4,
                "dropout": 0.0,
                "activation": "gelu",
            },
        }
    )
    return record


def build_raw_sequence_dataset(case_tables: dict[str, pd.DataFrame], seq_len: int) -> SequenceDataset:
    sequences: list[np.ndarray] = []
    targets: list[list[float]] = []
    next_delta_wear: list[list[float]] = []
    actual_steps: list[list[float]] = []
    case_names: list[str] = []
    step_indices: list[int] = []

    for case_name, table in case_tables.items():
        case_df = table.reset_index(drop=True)
        if len(case_df) < 2:
            continue

        raw_values = case_df[["F", "D", "Cr", "actual_cycle", "wear_depth"]].to_numpy(dtype=np.float32)
        stress_log = np.log(np.clip(case_df["stress"].to_numpy(dtype=np.float32), EPS, None))
        wear_values = case_df["wear_depth"].to_numpy(dtype=np.float32)
        actual_step = median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float))

        for idx in range(len(case_df) - 1):
            start = max(0, idx - seq_len + 1)
            seq = raw_values[start : idx + 1]
            if len(seq) < seq_len:
                pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
                seq = np.vstack([pad, seq])

            sequences.append(seq.astype(np.float32))
            targets.append([float(stress_log[idx])])
            next_delta_wear.append([float(wear_values[idx + 1] - wear_values[idx])])
            actual_steps.append([float(actual_step)])
            case_names.append(str(case_name))
            step_indices.append(int(idx))

    if sequences:
        raw_sequences = np.asarray(sequences, dtype=np.float32)
        target_log = np.asarray(targets, dtype=np.float32)
        next_delta = np.asarray(next_delta_wear, dtype=np.float32)
        actual_step_array = np.asarray(actual_steps, dtype=np.float32)
    else:
        raw_sequences = np.zeros((0, seq_len, len(LEGACY_FEATURE_ORDER)), dtype=np.float32)
        target_log = np.zeros((0, 1), dtype=np.float32)
        next_delta = np.zeros((0, 1), dtype=np.float32)
        actual_step_array = np.zeros((0, 1), dtype=np.float32)
    return SequenceDataset(
        raw_sequences=raw_sequences,
        target_log=target_log,
        next_delta_wear=next_delta,
        actual_step=actual_step_array,
        case_names=case_names,
        step_index=np.asarray(step_indices, dtype=np.int64),
    )


def empty_run_shape_dataset(sequence_length: int, feature_count: int = len(LEGACY_FEATURE_ORDER)) -> RunShapeDataset:
    return RunShapeDataset(
        mono_raw_sequences=np.zeros((0, 2, sequence_length, feature_count), dtype=np.float32),
        mono_actual_step=np.zeros((0, 2, 1), dtype=np.float32),
        slow_raw_sequences=np.zeros((0, 3, sequence_length, feature_count), dtype=np.float32),
        slow_actual_step=np.zeros((0, 3, 1), dtype=np.float32),
    )


def build_run_shape_dataset(sequence_dataset: SequenceDataset, max_wear_mm: float = 0.005) -> RunShapeDataset:
    if len(sequence_dataset) == 0:
        sequence_length = int(sequence_dataset.raw_sequences.shape[1]) if sequence_dataset.raw_sequences.ndim >= 2 else 0
        feature_count = int(sequence_dataset.raw_sequences.shape[2]) if sequence_dataset.raw_sequences.ndim >= 3 else len(LEGACY_FEATURE_ORDER)
        return empty_run_shape_dataset(sequence_length=sequence_length, feature_count=feature_count)

    mono_sequences: list[np.ndarray] = []
    mono_steps: list[np.ndarray] = []
    slow_sequences: list[np.ndarray] = []
    slow_steps: list[np.ndarray] = []

    grouped_indices: dict[str, list[tuple[int, int]]] = {}
    for sample_idx, case_name in enumerate(sequence_dataset.case_names):
        grouped_indices.setdefault(str(case_name), []).append((int(sequence_dataset.step_index[sample_idx]), int(sample_idx)))

    for pairs in grouped_indices.values():
        ordered = sorted(pairs, key=lambda item: item[0])
        for left, right in zip(ordered[:-1], ordered[1:]):
            left_step, left_idx = left
            right_step, right_idx = right
            if right_step - left_step != 1:
                continue
            wear_pair = np.asarray(
                [
                    sequence_dataset.raw_sequences[left_idx, -1, 4],
                    sequence_dataset.raw_sequences[right_idx, -1, 4],
                ],
                dtype=np.float32,
            )
            if np.any(wear_pair > float(max_wear_mm)):
                continue
            mono_sequences.append(
                np.stack(
                    [
                        sequence_dataset.raw_sequences[left_idx],
                        sequence_dataset.raw_sequences[right_idx],
                    ],
                    axis=0,
                ).astype(np.float32)
            )
            mono_steps.append(
                np.stack(
                    [
                        sequence_dataset.actual_step[left_idx],
                        sequence_dataset.actual_step[right_idx],
                    ],
                    axis=0,
                ).astype(np.float32)
            )

        for first, second, third in zip(ordered[:-2], ordered[1:-1], ordered[2:]):
            first_step, first_idx = first
            second_step, second_idx = second
            third_step, third_idx = third
            if second_step - first_step != 1 or third_step - second_step != 1:
                continue
            wear_triple = np.asarray(
                [
                    sequence_dataset.raw_sequences[first_idx, -1, 4],
                    sequence_dataset.raw_sequences[second_idx, -1, 4],
                    sequence_dataset.raw_sequences[third_idx, -1, 4],
                ],
                dtype=np.float32,
            )
            if np.any(wear_triple > float(max_wear_mm)):
                continue
            slow_sequences.append(
                np.stack(
                    [
                        sequence_dataset.raw_sequences[first_idx],
                        sequence_dataset.raw_sequences[second_idx],
                        sequence_dataset.raw_sequences[third_idx],
                    ],
                    axis=0,
                ).astype(np.float32)
            )
            slow_steps.append(
                np.stack(
                    [
                        sequence_dataset.actual_step[first_idx],
                        sequence_dataset.actual_step[second_idx],
                        sequence_dataset.actual_step[third_idx],
                    ],
                    axis=0,
                ).astype(np.float32)
            )

    shape_dataset = empty_run_shape_dataset(
        sequence_length=int(sequence_dataset.raw_sequences.shape[1]),
        feature_count=int(sequence_dataset.raw_sequences.shape[2]),
    )
    if mono_sequences:
        shape_dataset.mono_raw_sequences = np.asarray(mono_sequences, dtype=np.float32)
        shape_dataset.mono_actual_step = np.asarray(mono_steps, dtype=np.float32)
    if slow_sequences:
        shape_dataset.slow_raw_sequences = np.asarray(slow_sequences, dtype=np.float32)
        shape_dataset.slow_actual_step = np.asarray(slow_steps, dtype=np.float32)
    return shape_dataset


def make_raw_sequence(history: list[list[float]], seq_len: int) -> np.ndarray:
    seq = np.asarray(history[-seq_len:], dtype=np.float32)
    if len(seq) < seq_len:
        pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
        seq = np.vstack([pad, seq])
    return seq.astype(np.float32)


def compute_dynamic_features(raw_sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_sequences, dtype=np.float32)
    cycles = values[:, :, 3]
    wear_depth = values[:, :, 4]
    delta_cycle = np.zeros_like(cycles, dtype=np.float32)
    delta_wear = np.zeros_like(wear_depth, dtype=np.float32)
    delta_cycle[:, 1:] = np.maximum(cycles[:, 1:] - cycles[:, :-1], 0.0)
    delta_wear[:, 1:] = np.maximum(wear_depth[:, 1:] - wear_depth[:, :-1], 0.0)
    return np.stack(
        [
            np.log1p(np.maximum(cycles, 0.0)),
            wear_depth,
            delta_cycle,
            delta_wear,
        ],
        axis=-1,
    ).astype(np.float32)


def compute_static_features(raw_sequences: np.ndarray, actual_step: np.ndarray, use_cycle_step: bool) -> np.ndarray:
    values = np.asarray(raw_sequences, dtype=np.float32)
    F = values[:, -1, 0]
    D = np.maximum(values[:, -1, 1], EPS)
    Cr = values[:, -1, 2]
    features = [
        F,
        D,
        Cr,
        F / np.maximum(D * D, EPS),
        Cr / D,
    ]
    if use_cycle_step:
        features.append(np.asarray(actual_step, dtype=np.float32).reshape(-1))
    return np.stack(features, axis=-1).astype(np.float32)


def build_single_branch_features(raw_sequences: np.ndarray, actual_step: np.ndarray, use_cycle_step: bool) -> np.ndarray:
    static_features = compute_static_features(raw_sequences, actual_step, use_cycle_step)
    dynamic_features = compute_dynamic_features(raw_sequences)
    static_seq = np.repeat(static_features[:, np.newaxis, :], dynamic_features.shape[1], axis=1)
    return np.concatenate([static_seq, dynamic_features], axis=-1).astype(np.float32)


def build_model_inputs(
    raw_sequences: np.ndarray,
    actual_step: np.ndarray,
    config: CandidateConfig,
    input_scalers: dict[str, ArrayScaler] | None = None,
    fit: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, ArrayScaler]]:
    scalers = {} if input_scalers is None else dict(input_scalers)

    if config.model_kind == "v1_transformer":
        sequence_x = np.asarray(raw_sequences, dtype=np.float32)
        if fit:
            scalers["sequence"] = ArrayScaler().fit(sequence_x.reshape(-1, sequence_x.shape[-1]))
        sequence_x = scalers["sequence"].transform(sequence_x)
        return {"sequence": sequence_x.astype(np.float32)}, scalers

    if config.model_kind == "v2_single_branch":
        sequence_x = build_single_branch_features(raw_sequences, actual_step, config.use_cycle_step)
        if fit:
            scalers["sequence"] = ArrayScaler().fit(sequence_x.reshape(-1, sequence_x.shape[-1]))
        sequence_x = scalers["sequence"].transform(sequence_x)
        return {"sequence": sequence_x.astype(np.float32)}, scalers

    static_x = compute_static_features(raw_sequences, actual_step, config.use_cycle_step)
    dynamic_x = compute_dynamic_features(raw_sequences)
    if fit:
        scalers["static"] = ArrayScaler().fit(static_x)
        scalers["dynamic"] = ArrayScaler().fit(dynamic_x.reshape(-1, dynamic_x.shape[-1]))
    static_x = scalers["static"].transform(static_x)
    dynamic_x = scalers["dynamic"].transform(dynamic_x)
    return {"static": static_x.astype(np.float32), "dynamic": dynamic_x.astype(np.float32)}, scalers


def instantiate_model(config: CandidateConfig) -> nn.Module:
    if config.model_kind == "v1_transformer":
        return SequenceTransformerNet(
            input_size=len(LEGACY_FEATURE_ORDER),
            seq_len=config.sequence_length,
            dropout=0.0,
        )
    if config.model_kind == "v2_single_branch":
        return SequenceTransformerNet(
            input_size=len(config.static_feature_order) + len(config.dynamic_feature_order),
            seq_len=config.sequence_length,
            dropout=config.dropout,
        )
    return StaticDynamicTransformerNet(
        static_input_size=len(config.static_feature_order),
        dynamic_input_size=len(config.dynamic_feature_order),
        seq_len=config.sequence_length,
        dropout=config.dropout,
    )


def to_tensor_inputs(inputs_np: dict[str, np.ndarray], device: Any) -> dict[str, torch.Tensor]:
    return {name: torch.tensor(values, dtype=torch.float32, device=device) for name, values in inputs_np.items()}


def forward_model(model: nn.Module, inputs_t: dict[str, torch.Tensor]) -> torch.Tensor:
    if "sequence" in inputs_t:
        return model(inputs_t["sequence"])
    return model(inputs_t["dynamic"], inputs_t["static"])


def slice_inputs(inputs_np: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in inputs_np.items()}


def batch_indices(sample_count: int, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    if sample_count <= 0:
        return []
    if batch_size <= 0 or batch_size >= sample_count:
        return [np.arange(sample_count, dtype=np.int64)]
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(sample_count)
    batches: list[np.ndarray] = []
    for start in range(0, sample_count, batch_size):
        batches.append(order[start : start + batch_size])
    return batches


def monotonic_penalty(
    model: nn.Module,
    raw_sequences: np.ndarray,
    actual_step: np.ndarray,
    config: CandidateConfig,
    input_scalers: dict[str, ArrayScaler],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = infer_model_device(model)
    wear_seq = raw_sequences.copy()
    wear_seq[:, :, 4] = wear_seq[:, :, 4] + WEAR_DELTA_MM
    load_seq = raw_sequences.copy()
    load_seq[:, :, 0] = load_seq[:, :, 0] * (1.0 + LOAD_DELTA_RATIO)
    clearance_seq = raw_sequences.copy()
    clearance_seq[:, :, 2] = clearance_seq[:, :, 2] + CLEARANCE_DELTA_MM

    base_inputs = to_tensor_inputs(build_model_inputs(raw_sequences, actual_step, config, input_scalers=input_scalers, fit=False)[0], device)
    wear_inputs = to_tensor_inputs(build_model_inputs(wear_seq, actual_step, config, input_scalers=input_scalers, fit=False)[0], device)
    load_inputs = to_tensor_inputs(build_model_inputs(load_seq, actual_step, config, input_scalers=input_scalers, fit=False)[0], device)
    clearance_inputs = to_tensor_inputs(build_model_inputs(clearance_seq, actual_step, config, input_scalers=input_scalers, fit=False)[0], device)

    base_pred = forward_model(model, base_inputs)
    wear_pred = forward_model(model, wear_inputs)
    load_pred = forward_model(model, load_inputs)
    clearance_pred = forward_model(model, clearance_inputs)

    wear_pen = torch.mean(torch.relu(wear_pred - base_pred) ** 2)
    load_pen = torch.mean(torch.relu(base_pred - load_pred) ** 2)
    clearance_pen = torch.mean(torch.relu(base_pred - clearance_pred) ** 2)
    return wear_pen, load_pen, clearance_pen


def wear_consistency_loss(
    pred_log_scaled: torch.Tensor,
    target_scaler: ArrayScaler,
    raw_sequences: np.ndarray,
    actual_step: np.ndarray,
    next_delta_wear: np.ndarray,
    real_k: float,
) -> torch.Tensor:
    pred_log = target_scaler.inverse_torch(pred_log_scaled)
    pred_stress = torch.exp(pred_log).reshape(-1)
    D_t = torch.tensor(raw_sequences[:, -1, 1], dtype=torch.float32, device=pred_stress.device)
    step_t = torch.tensor(actual_step.reshape(-1), dtype=torch.float32, device=pred_stress.device)
    delta_s = step_t * math.pi * D_t / 6.0
    pred_delta = pred_stress * delta_s * float(real_k)
    true_delta = torch.tensor(next_delta_wear.reshape(-1), dtype=torch.float32, device=pred_stress.device)
    return torch.mean((pred_delta - true_delta) ** 2)


def scaled_log_to_stress(pred_log_scaled: torch.Tensor, target_scaler: ArrayScaler) -> torch.Tensor:
    pred_log = target_scaler.inverse_torch(pred_log_scaled)
    return torch.exp(pred_log)


def run_stress_mono_penalty(pred_stress_pairs: torch.Tensor) -> torch.Tensor:
    if pred_stress_pairs.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_stress_pairs.device)
    return torch.mean(torch.relu(pred_stress_pairs[:, 1] - pred_stress_pairs[:, 0]))


def run_stress_slow_penalty(pred_stress_triples: torch.Tensor) -> torch.Tensor:
    if pred_stress_triples.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_stress_triples.device)
    drop_t = pred_stress_triples[:, 0] - pred_stress_triples[:, 1]
    drop_next = pred_stress_triples[:, 1] - pred_stress_triples[:, 2]
    return torch.mean(torch.relu(drop_next - drop_t))


def predict_grouped_stress(
    model: nn.Module,
    raw_sequences: np.ndarray,
    actual_step: np.ndarray,
    group_size: int,
    config: CandidateConfig,
    input_scalers: dict[str, ArrayScaler],
    target_scaler: ArrayScaler,
) -> torch.Tensor:
    device = infer_model_device(model)
    flat_raw = raw_sequences.reshape(-1, raw_sequences.shape[-2], raw_sequences.shape[-1])
    flat_step = actual_step.reshape(-1, actual_step.shape[-1])
    inputs_np, _ = build_model_inputs(
        flat_raw,
        flat_step,
        config,
        input_scalers=input_scalers,
        fit=False,
    )
    pred_scaled = forward_model(model, to_tensor_inputs(inputs_np, device))
    pred_stress = scaled_log_to_stress(pred_scaled, target_scaler).reshape(-1)
    return pred_stress.reshape(-1, group_size)


def run_stress_shape_losses(
    model: nn.Module,
    shape_dataset: RunShapeDataset | None,
    config: CandidateConfig,
    input_scalers: dict[str, ArrayScaler],
    target_scaler: ArrayScaler,
    compute_mono: bool = True,
    compute_slow: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = infer_model_device(model)
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    if shape_dataset is None:
        return zero, zero

    mono_pen = zero
    slow_pen = zero
    if compute_mono and shape_dataset.mono_count > 0:
        mono_stress = predict_grouped_stress(
            model=model,
            raw_sequences=shape_dataset.mono_raw_sequences,
            actual_step=shape_dataset.mono_actual_step,
            group_size=2,
            config=config,
            input_scalers=input_scalers,
            target_scaler=target_scaler,
        )
        mono_pen = run_stress_mono_penalty(mono_stress)
    if compute_slow and shape_dataset.slow_count > 0:
        slow_stress = predict_grouped_stress(
            model=model,
            raw_sequences=shape_dataset.slow_raw_sequences,
            actual_step=shape_dataset.slow_actual_step,
            group_size=3,
            config=config,
            input_scalers=input_scalers,
            target_scaler=target_scaler,
        )
        slow_pen = run_stress_slow_penalty(slow_stress)
    return mono_pen, slow_pen


def total_objective(
    model: nn.Module,
    inputs_np: dict[str, np.ndarray],
    target_scaled_np: np.ndarray,
    raw_sequences: np.ndarray,
    actual_step: np.ndarray,
    next_delta_wear: np.ndarray,
    input_scalers: dict[str, ArrayScaler],
    target_scaler: ArrayScaler,
    config: CandidateConfig,
    real_k: float,
    shape_dataset: RunShapeDataset | None = None,
) -> torch.Tensor:
    device = infer_model_device(model)
    mse = nn.MSELoss()
    inputs_t = to_tensor_inputs(inputs_np, device)
    target_t = torch.tensor(target_scaled_np, dtype=torch.float32, device=device)
    pred = forward_model(model, inputs_t)
    data_loss = mse(pred, target_t)
    wear_pen, load_pen, clearance_pen = monotonic_penalty(model, raw_sequences, actual_step, config, input_scalers)
    consistency_pen = torch.tensor(0.0, dtype=torch.float32, device=device)
    if config.wear_consistency_lambda > 0.0:
        consistency_pen = wear_consistency_loss(pred, target_scaler, raw_sequences, actual_step, next_delta_wear, real_k)
    run_mono_pen = torch.tensor(0.0, dtype=torch.float32, device=device)
    run_slow_pen = torch.tensor(0.0, dtype=torch.float32, device=device)
    if shape_dataset is not None and (config.run_stress_mono_lambda > 0.0 or config.run_stress_slow_lambda > 0.0):
        run_mono_pen, run_slow_pen = run_stress_shape_losses(
            model=model,
            shape_dataset=shape_dataset,
            config=config,
            input_scalers=input_scalers,
            target_scaler=target_scaler,
            compute_mono=config.run_stress_mono_lambda > 0.0,
            compute_slow=config.run_stress_slow_lambda > 0.0,
        )
    return (
        data_loss
        + config.wear_consistency_lambda * consistency_pen
        + config.run_stress_mono_lambda * run_mono_pen
        + config.run_stress_slow_lambda * run_slow_pen
        + MONO_LAMBDA_WEAR * wear_pen
        + MONO_LAMBDA_LOAD * load_pen
        + MONO_LAMBDA_CLEARANCE * clearance_pen
    )


def evaluate_loss(
    model: nn.Module,
    dataset: SequenceDataset | None,
    inputs_np: dict[str, np.ndarray] | None,
    target_scaled_np: np.ndarray | None,
    shape_dataset: RunShapeDataset | None,
    input_scalers: dict[str, ArrayScaler],
    target_scaler: ArrayScaler,
    config: CandidateConfig,
    real_k: float,
) -> float:
    if dataset is None or inputs_np is None or target_scaled_np is None or len(dataset) == 0:
        return float("inf")
    model.eval()
    with torch.no_grad():
        loss = total_objective(
            model=model,
            inputs_np=inputs_np,
            target_scaled_np=target_scaled_np,
            raw_sequences=dataset.raw_sequences,
            actual_step=dataset.actual_step,
            next_delta_wear=dataset.next_delta_wear,
            input_scalers=input_scalers,
            target_scaler=target_scaler,
            config=config,
            real_k=real_k,
            shape_dataset=shape_dataset,
        )
    return float(loss.item())


def train_candidate_model(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset | None,
    config: CandidateConfig,
    real_k: float,
    forced_epochs: int | None = None,
) -> TrainedCandidate:
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    set_seed(config.seed)
    train_inputs_np, input_scalers = build_model_inputs(
        train_dataset.raw_sequences,
        train_dataset.actual_step,
        config,
        fit=True,
    )
    train_shape_dataset = build_run_shape_dataset(train_dataset) if uses_run_shape_losses(config) else None
    target_scaler = ArrayScaler().fit(train_dataset.target_log)
    train_target_scaled = target_scaler.transform(train_dataset.target_log)

    val_inputs_np = None
    val_target_scaled = None
    val_shape_dataset = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_inputs_np, _ = build_model_inputs(
            val_dataset.raw_sequences,
            val_dataset.actual_step,
            config,
            input_scalers=input_scalers,
            fit=False,
        )
        val_target_scaled = target_scaler.transform(val_dataset.target_log)
        val_shape_dataset = build_run_shape_dataset(val_dataset) if uses_run_shape_losses(config) else None

    device, _ = resolve_training_device()
    model = instantiate_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        foreach=False,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float("inf")
    no_improve = 0
    total_epochs = int(forced_epochs) if forced_epochs is not None else int(config.max_epochs)

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        main_batches = batch_indices(len(train_dataset), config.batch_size, config.seed, epoch)
        mono_count = 0 if train_shape_dataset is None else int(train_shape_dataset.mono_count)
        slow_count = 0 if train_shape_dataset is None else int(train_shape_dataset.slow_count)
        mono_batch_size = min(int(config.batch_size), mono_count) if mono_count > 0 else 0
        slow_batch_size = min(int(config.batch_size), slow_count) if slow_count > 0 else 0
        mono_batches = batch_indices(mono_count, mono_batch_size, config.seed + 1000, epoch)
        slow_batches = batch_indices(slow_count, slow_batch_size, config.seed + 2000, epoch)

        for batch_pos, indices in enumerate(main_batches):
            batch_inputs_np = slice_inputs(train_inputs_np, indices)
            batch_target_scaled = train_target_scaled[indices]
            batch_raw = train_dataset.raw_sequences[indices]
            batch_step = train_dataset.actual_step[indices]
            batch_next_delta = train_dataset.next_delta_wear[indices]
            mono_indices = mono_batches[batch_pos % len(mono_batches)] if mono_batches else None
            slow_indices = slow_batches[batch_pos % len(slow_batches)] if slow_batches else None
            shape_batch = None
            if train_shape_dataset is not None:
                shape_batch = train_shape_dataset.slice(
                    mono_indices=mono_indices,
                    slow_indices=slow_indices,
                    use_full_on_none=False,
                )

            loss = total_objective(
                model=model,
                inputs_np=batch_inputs_np,
                target_scaled_np=batch_target_scaled,
                raw_sequences=batch_raw,
                actual_step=batch_step,
                next_delta_wear=batch_next_delta,
                input_scalers=input_scalers,
                target_scaler=target_scaler,
                config=config,
                real_k=real_k,
                shape_dataset=shape_batch,
            )
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_score = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        if forced_epochs is not None:
            if epoch == total_epochs:
                best_epoch = epoch
                best_score = train_score
                best_state = copy.deepcopy(model.state_dict())
            continue

        monitor_score = train_score
        if config.use_validation and val_dataset is not None and len(val_dataset) > 0:
            monitor_score = evaluate_loss(
                model=model,
                dataset=val_dataset,
                inputs_np=val_inputs_np,
                target_scaled_np=val_target_scaled,
                shape_dataset=val_shape_dataset,
                input_scalers=input_scalers,
                target_scaler=target_scaler,
                config=config,
                real_k=real_k,
            )

        if monitor_score < best_score - 1e-9:
            best_score = monitor_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if config.use_validation and no_improve >= int(config.patience):
            break

    model.load_state_dict(best_state)
    return TrainedCandidate(
        config=config,
        model=model,
        input_scalers=input_scalers,
        target_scaler=target_scaler,
        best_epoch=int(best_epoch),
        monitored_loss=float(best_score),
        training_sample_count=len(train_dataset),
        validation_sample_count=0 if val_dataset is None else len(val_dataset),
    )


def predict_log_scaled(
    runtime: TrainedCandidate,
    raw_sequences: np.ndarray,
    actual_step: np.ndarray,
) -> torch.Tensor:
    inputs_np, _ = build_model_inputs(
        raw_sequences,
        actual_step,
        runtime.config,
        input_scalers=runtime.input_scalers,
        fit=False,
    )
    device = infer_model_device(runtime.model)
    inputs_t = to_tensor_inputs(inputs_np, device)
    runtime.model.eval()
    with torch.no_grad():
        return forward_model(runtime.model, inputs_t)


def predict_pressure_from_history(
    runtime: TrainedCandidate,
    history: list[list[float]],
    actual_cycle_step: float,
) -> float:
    raw_seq = make_raw_sequence(history, runtime.config.sequence_length)[np.newaxis, :, :]
    actual_step = np.asarray([[float(actual_cycle_step)]], dtype=np.float32)
    pred_log_scaled = predict_log_scaled(runtime, raw_seq, actual_step)
    pred_log = runtime.target_scaler.inverse_torch(pred_log_scaled).cpu().numpy().reshape(-1)[0]
    return float(np.exp(pred_log))


def evaluate_pressure(runtime: TrainedCandidate, dataset: SequenceDataset) -> dict[str, float]:
    pred_log_scaled = predict_log_scaled(runtime, dataset.raw_sequences, dataset.actual_step)
    pred_log = runtime.target_scaler.inverse_torch(pred_log_scaled).cpu().numpy().reshape(-1)
    pred_stress = np.exp(pred_log)
    true_stress = np.exp(dataset.target_log.reshape(-1))
    rmse = float(np.sqrt(np.mean((pred_stress - true_stress) ** 2)))
    mae = float(np.mean(np.abs(pred_stress - true_stress)))
    mape = float(np.mean(np.abs(pred_stress - true_stress) / np.maximum(true_stress, EPS)))
    return {"pressure_rmse": rmse, "pressure_mae": mae, "pressure_mape": mape}


def rollout_case(
    runtime: TrainedCandidate,
    case_df: pd.DataFrame,
    threshold_um: float,
    real_k: float,
    true_life_actual: float,
) -> tuple[pd.DataFrame, float]:
    first = case_df.iloc[0]
    F = float(first["F"])
    D = float(first["D"])
    Cr = float(first["Cr"])
    actual_step = median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float))
    sim_step = median_positive_diff(case_df["sim_cycle"].to_numpy(dtype=float))
    threshold_mm = threshold_um / 1000.0

    actual_cycle = 0.0
    sim_cycle = 0.0
    wear_depth = 0.0
    history: list[list[float]] = [[F, D, Cr, actual_cycle, wear_depth]]
    rows: list[dict[str, float]] = []
    predicted_life_actual = true_life_actual

    internal_limit = max(true_life_actual * 1.35, float(case_df["actual_cycle"].max()) * 1.15, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + MAX_EXTRA_STEPS

    for _ in range(max_steps):
        pred_stress = predict_pressure_from_history(runtime, history, actual_step)
        rows.append(
            {
                "sim_cycle": sim_cycle,
                "actual_cycle": actual_cycle,
                "pred_stress": pred_stress,
                "pred_wear_depth_um": wear_depth * 1000.0,
            }
        )

        delta_s = actual_step * math.pi * D / 6.0
        delta_wear = real_k * pred_stress * delta_s
        next_actual_cycle = actual_cycle + actual_step
        next_wear_depth = wear_depth + delta_wear
        next_sim_cycle = sim_cycle + sim_step

        if next_wear_depth >= threshold_mm:
            ratio = (threshold_mm - wear_depth) / max(delta_wear, EPS)
            predicted_life_actual = actual_cycle + ratio * actual_step
            rows.append(
                {
                    "sim_cycle": sim_cycle + ratio * sim_step,
                    "actual_cycle": predicted_life_actual,
                    "pred_stress": pred_stress,
                    "pred_wear_depth_um": threshold_um,
                }
            )
            break

        actual_cycle = next_actual_cycle
        sim_cycle = next_sim_cycle
        wear_depth = next_wear_depth
        history.append([F, D, Cr, actual_cycle, wear_depth])

    return pd.DataFrame(rows), float(predicted_life_actual)


def wear_curve_mae(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    true_x = true_df["actual_cycle"].to_numpy(dtype=float)
    true_y = true_df["wear_depth_um"].to_numpy(dtype=float)
    pred_x = pred_df["actual_cycle"].to_numpy(dtype=float)
    pred_y = pred_df["pred_wear_depth_um"].to_numpy(dtype=float)
    pred_interp = np.interp(true_x, pred_x, pred_y)
    return float(np.mean(np.abs(pred_interp - true_y)))


def fold_tables(case_tables: dict[str, pd.DataFrame], file_names: list[str]) -> dict[str, pd.DataFrame]:
    return {str(name): case_tables[str(name)] for name in file_names if str(name) in case_tables}


def summarize_fold_metrics(fold_df: pd.DataFrame) -> dict[str, Any]:
    if fold_df.empty:
        raise ValueError("Fold dataframe is empty.")

    config_cols = [
        "candidate_name",
        "model_kind",
        "feature_version",
        "sequence_length",
        "seed",
        "dropout",
        "wear_consistency_lambda",
        "run_stress_mono_lambda",
        "run_stress_slow_lambda",
        "use_cycle_step",
        "use_validation",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "grad_clip_norm",
    ]
    base = {column: fold_df.iloc[0][column] for column in config_cols}
    life_df = fold_df[fold_df["eligible_for_life"].astype(bool)].reset_index(drop=True)
    return {
        **base,
        "fold_count": int(len(fold_df)),
        "life_fold_count": int(len(life_df)),
        "mean_pressure_mae": float(fold_df["pressure_mae"].mean()),
        "mean_pressure_rmse": float(fold_df["pressure_rmse"].mean()),
        "mean_pressure_mape": float(fold_df["pressure_mape"].mean()),
        "mean_best_epoch": float(fold_df["best_epoch"].mean()),
        "median_best_epoch": float(fold_df["best_epoch"].median()),
        "mean_life_rel_error": float(life_df["life_rel_error"].mean()) if len(life_df) else math.nan,
        "median_life_abs_error": float(life_df["life_abs_error"].median()) if len(life_df) else math.nan,
        "mean_life_abs_error": float(life_df["life_abs_error"].mean()) if len(life_df) else math.nan,
        "mean_wear_mae_um": float(life_df["wear_mae_um"].mean()) if len(life_df) else math.nan,
    }


def evaluate_candidate_on_manifest(
    config: CandidateConfig,
    manifest: dict[str, Any],
    summary_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
    threshold_um: float,
    real_k: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []

    for fold in manifest["folds"]:
        test_run = str(fold["test_run"])
        val_run = str(fold["val_run"]) if fold["val_run"] is not None else None
        train_runs = [str(name) for name in fold["train_runs"]]
        eligible_for_life = bool(fold["eligible_for_life"])

        train_dataset = build_raw_sequence_dataset(fold_tables(case_tables, train_runs), config.sequence_length)
        val_dataset = None
        if config.use_validation and val_run is not None:
            val_dataset = build_raw_sequence_dataset({val_run: case_tables[val_run]}, config.sequence_length)

        runtime = train_candidate_model(train_dataset, val_dataset, config, real_k)
        test_dataset = build_raw_sequence_dataset({test_run: case_tables[test_run]}, config.sequence_length)
        pressure_metrics = evaluate_pressure(runtime, test_dataset)

        test_summary_row = summary_df.loc[summary_df["file_name"].astype(str) == test_run].iloc[0]
        true_life_actual = float(test_summary_row["actual_life"])
        wear_mae_um = math.nan
        predicted_life = math.nan
        life_abs_error = math.nan
        life_rel_error = math.nan

        if eligible_for_life:
            true_curve_df = threshold_ground_truth(case_tables[test_run], threshold_um, test_summary_row)
            rollout_df, predicted_life = rollout_case(runtime, case_tables[test_run], threshold_um, real_k, true_life_actual)
            wear_mae_um = wear_curve_mae(true_curve_df, rollout_df)
            life_abs_error = abs(predicted_life - true_life_actual)
            life_rel_error = life_abs_error / max(true_life_actual, EPS)

        fold_rows.append(
            {
                **config.to_record(),
                "test_run": test_run,
                "test_source_file": str(test_summary_row["source_file"]),
                "eligible_for_life": eligible_for_life,
                "val_run": val_run,
                "train_run_count": int(len(train_runs)),
                "best_epoch": int(runtime.best_epoch),
                "pressure_mae": float(pressure_metrics["pressure_mae"]),
                "pressure_rmse": float(pressure_metrics["pressure_rmse"]),
                "pressure_mape": float(pressure_metrics["pressure_mape"]),
                "wear_mae_um": float(wear_mae_um) if not math.isnan(wear_mae_um) else math.nan,
                "predicted_life": float(predicted_life) if not math.isnan(predicted_life) else math.nan,
                "true_life": float(true_life_actual),
                "life_abs_error": float(life_abs_error) if not math.isnan(life_abs_error) else math.nan,
                "life_rel_error": float(life_rel_error) if not math.isnan(life_rel_error) else math.nan,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return pd.DataFrame([summarize_fold_metrics(fold_df)]), fold_df


def evaluate_candidate_suite(
    configs: list[CandidateConfig],
    manifest: dict[str, Any],
    summary_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
    threshold_um: float,
    real_k: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    for config in configs:
        summary_frame, fold_frame = evaluate_candidate_on_manifest(config, manifest, summary_df, case_tables, threshold_um, real_k)
        summary_frames.append(summary_frame)
        fold_frames.append(fold_frame)
    return pd.concat(summary_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True)


def rank_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked = ranked.sort_values(
        by=["mean_life_rel_error", "median_life_abs_error", "mean_pressure_mae"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def build_round1_candidates() -> list[CandidateConfig]:
    return [
        make_v1_protocol_candidate(sequence_length=ROUND1_SEQUENCE_LENGTH, seed=PRIMARY_SEED),
        make_v2_single_branch_candidate(sequence_length=ROUND1_SEQUENCE_LENGTH, wear_consistency_lambda=0.0, use_cycle_step=False, seed=PRIMARY_SEED),
        make_v2_dual_branch_candidate(sequence_length=ROUND1_SEQUENCE_LENGTH, wear_consistency_lambda=0.0, use_cycle_step=False, seed=PRIMARY_SEED),
    ]


def build_lite_round1_candidates() -> list[CandidateConfig]:
    return [
        make_lite_config(make_v1_protocol_candidate(sequence_length=ROUND1_SEQUENCE_LENGTH, seed=PRIMARY_SEED)),
        make_lite_config(
            make_v2_single_branch_candidate(
                sequence_length=ROUND1_SEQUENCE_LENGTH,
                wear_consistency_lambda=0.0,
                use_cycle_step=False,
                seed=PRIMARY_SEED,
            )
        ),
        make_lite_config(
            make_v2_dual_branch_candidate(
                sequence_length=ROUND1_SEQUENCE_LENGTH,
                wear_consistency_lambda=0.0,
                use_cycle_step=False,
                seed=PRIMARY_SEED,
            )
        ),
    ]


def build_round2_candidates(base_config: CandidateConfig) -> list[CandidateConfig]:
    configs: list[CandidateConfig] = []
    if base_config.model_kind == "v1_transformer":
        for seq_len in ROUND2_SEQUENCE_LENGTHS:
            configs.append(make_v1_protocol_candidate(sequence_length=seq_len, seed=PRIMARY_SEED))
        return configs

    for seq_len in ROUND2_SEQUENCE_LENGTHS:
        for wear_lambda in ROUND2_WEAR_LAMBDAS:
            for use_cycle_step in [False, True]:
                if base_config.model_kind == "v2_single_branch":
                    configs.append(
                        make_v2_single_branch_candidate(
                            sequence_length=seq_len,
                            wear_consistency_lambda=wear_lambda,
                            use_cycle_step=use_cycle_step,
                            seed=PRIMARY_SEED,
                        )
                    )
                else:
                    configs.append(
                        make_v2_dual_branch_candidate(
                            sequence_length=seq_len,
                            wear_consistency_lambda=wear_lambda,
                            use_cycle_step=use_cycle_step,
                            seed=PRIMARY_SEED,
                        )
                    )
    return configs


def select_top_candidates(summary_df: pd.DataFrame, count: int) -> list[str]:
    ranked = rank_summary_df(summary_df)
    return ranked["candidate_name"].head(count).astype(str).tolist()


def expand_configs_for_seeds(configs: list[CandidateConfig], seeds: list[int]) -> list[CandidateConfig]:
    expanded: list[CandidateConfig] = []
    for config in configs:
        for seed in seeds:
            expanded.append(config.with_updates(seed=seed))
    return expanded


def aggregate_seed_summaries(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "candidate_name",
        "model_kind",
        "feature_version",
        "sequence_length",
        "dropout",
        "wear_consistency_lambda",
        "run_stress_mono_lambda",
        "run_stress_slow_lambda",
        "use_cycle_step",
        "use_validation",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "grad_clip_norm",
    ]
    rows: list[dict[str, Any]] = []
    for _, group in seed_summary_df.groupby(group_cols, dropna=False):
        row = {column: group.iloc[0][column] for column in group_cols}
        life_mean = group["mean_life_rel_error"].to_numpy(dtype=float)
        row.update(
            {
                "seed_count": int(len(group)),
                "median_mean_life_rel_error": float(np.nanmedian(life_mean)),
                "median_median_life_abs_error": float(np.nanmedian(group["median_life_abs_error"].to_numpy(dtype=float))),
                "median_mean_pressure_mae": float(np.nanmedian(group["mean_pressure_mae"].to_numpy(dtype=float))),
                "median_mean_wear_mae_um": float(np.nanmedian(group["mean_wear_mae_um"].to_numpy(dtype=float))),
                "median_best_epoch": float(np.nanmedian(group["median_best_epoch"].to_numpy(dtype=float))),
                "life_rel_error_cv": float(np.nanstd(life_mean) / max(abs(float(np.nanmean(life_mean))), EPS)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_deployment_gate(finalist_df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    gated = finalist_df.copy()
    baseline_row = gated.loc[gated["candidate_name"].astype(str) == str(baseline_name)]
    if baseline_row.empty:
        gated["deployment_ready"] = False
        gated["gate_reason"] = "Missing baseline candidate."
        return gated

    baseline = baseline_row.iloc[0]
    base_life = float(baseline["median_mean_life_rel_error"])
    base_abs = float(baseline["median_median_life_abs_error"])
    base_pressure = float(baseline["median_mean_pressure_mae"])

    ready_flags: list[bool] = []
    ready_reasons: list[str] = []
    for row in gated.itertuples(index=False):
        if str(row.candidate_name) == str(baseline_name):
            ready_flags.append(False)
            ready_reasons.append("Baseline reference only.")
            continue

        life_improved = (base_life - float(row.median_mean_life_rel_error)) / max(base_life, EPS)
        abs_regression = (float(row.median_median_life_abs_error) - base_abs) / max(base_abs, EPS)
        pressure_regression = (float(row.median_mean_pressure_mae) - base_pressure) / max(base_pressure, EPS)
        stable = float(row.life_rel_error_cv) <= 0.15

        if life_improved < 0.05:
            ready_flags.append(False)
            ready_reasons.append("Life relative error improvement below 5%.")
        elif abs_regression > 0.02:
            ready_flags.append(False)
            ready_reasons.append("Median life absolute error regressed by more than 2%.")
        elif pressure_regression > 0.10:
            ready_flags.append(False)
            ready_reasons.append("Pressure MAE regressed by more than 10%.")
        elif not stable:
            ready_flags.append(False)
            ready_reasons.append("Three-seed life error variation is above 15%.")
        else:
            ready_flags.append(True)
            ready_reasons.append("Pass")

    gated["deployment_ready"] = ready_flags
    gated["gate_reason"] = ready_reasons
    return gated


def select_recommended_candidate(gated_df: pd.DataFrame) -> dict[str, Any] | None:
    ready_df = gated_df[gated_df["deployment_ready"].astype(bool)].copy()
    if ready_df.empty:
        return None
    ranked = ready_df.sort_values(
        by=["median_mean_life_rel_error", "median_median_life_abs_error", "median_mean_pressure_mae"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return ranked.iloc[0].to_dict()


def runtime_to_bundle(
    runtime: TrainedCandidate,
    train_summary: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
    training_wear_limit_um: float,
) -> dict[str, Any]:
    input_scaler_bundle = {
        name: {"mean": scaler.mean, "std": scaler.std}
        for name, scaler in runtime.input_scalers.items()
    }
    return {
        "feature_version": runtime.config.feature_version,
        "candidate_name": runtime.config.name,
        "model_kind": runtime.config.model_kind,
        "model_state_dict": cpu_state_dict(runtime.model),
        "input_scalers": input_scaler_bundle,
        "target_scaler": {"mean": runtime.target_scaler.mean, "std": runtime.target_scaler.std},
        "static_feature_order": runtime.config.static_feature_order,
        "dynamic_feature_order": runtime.config.dynamic_feature_order,
        "input_feature_order": runtime.config.input_feature_order,
        "training_wear_limit_um": float(training_wear_limit_um),
        "training_actual_life_max": float(train_summary["actual_life"].max()),
        "training_actual_life_mean": float(train_summary["actual_life"].mean()),
        "recommended_cycle_step": float(recommended_cycle_step(case_tables)),
        "available_coating": "DLC",
        "model_architecture": runtime.config.model_kind,
        "sequence_length": int(runtime.config.sequence_length),
        "dropout": float(runtime.config.dropout),
        "wear_consistency_lambda": float(runtime.config.wear_consistency_lambda),
        "run_stress_mono_lambda": float(runtime.config.run_stress_mono_lambda),
        "run_stress_slow_lambda": float(runtime.config.run_stress_slow_lambda),
        "use_cycle_step": bool(runtime.config.use_cycle_step),
        "use_validation": bool(runtime.config.use_validation),
        "learning_rate": float(runtime.config.learning_rate),
        "weight_decay": float(runtime.config.weight_decay),
        "batch_size": int(runtime.config.batch_size),
        "max_epochs": int(runtime.config.max_epochs),
        "patience": int(runtime.config.patience),
        "grad_clip_norm": float(runtime.config.grad_clip_norm),
        "best_epoch": int(runtime.best_epoch),
    }


def save_candidate_bundle(
    runtime: TrainedCandidate,
    train_summary: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
    training_wear_limit_um: float,
    model_filename: str,
    copy_to_software: bool = False,
) -> Path:
    bundle = runtime_to_bundle(runtime, train_summary, case_tables, training_wear_limit_um)
    model_path = trained_model_dir() / model_filename
    torch.save(bundle, model_path)
    if copy_to_software:
        software_model_path().parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, software_model_path())
    return model_path


def load_runtime_from_bundle(
    bundle: dict[str, Any],
    device: Any | None = None,
) -> tuple[nn.Module, dict[str, ArrayScaler], ArrayScaler, dict[str, Any], CandidateConfig]:
    config = CandidateConfig(
        name=str(bundle.get("candidate_name", "V2_candidate")),
        model_kind=str(bundle.get("model_kind", "v2_dual_branch")),
        sequence_length=int(bundle.get("sequence_length", ROUND1_SEQUENCE_LENGTH)),
        seed=int(bundle.get("seed", PRIMARY_SEED)),
        dropout=float(bundle.get("dropout", 0.1)),
        wear_consistency_lambda=float(bundle.get("wear_consistency_lambda", 0.0)),
        run_stress_mono_lambda=float(bundle.get("run_stress_mono_lambda", 0.0)),
        run_stress_slow_lambda=float(bundle.get("run_stress_slow_lambda", 0.0)),
        use_cycle_step=bool(bundle.get("use_cycle_step", False)),
        use_validation=bool(bundle.get("use_validation", False)),
        learning_rate=float(bundle.get("learning_rate", DEFAULT_LEARNING_RATE)),
        weight_decay=float(bundle.get("weight_decay", DEFAULT_WEIGHT_DECAY)),
        batch_size=int(bundle.get("batch_size", DEFAULT_BATCH_SIZE)),
        max_epochs=int(bundle.get("max_epochs", DEFAULT_MAX_EPOCHS)),
        patience=int(bundle.get("patience", DEFAULT_PATIENCE)),
        grad_clip_norm=float(bundle.get("grad_clip_norm", DEFAULT_GRAD_CLIP)),
    )
    runtime_device = torch.device("cpu") if device is None else device
    model = instantiate_model(config).to(runtime_device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    input_scalers: dict[str, ArrayScaler] = {}
    for name, values in dict(bundle.get("input_scalers", {})).items():
        input_scalers[str(name)] = ArrayScaler().load(values["mean"], values["std"])
    target_bundle = dict(bundle["target_scaler"])
    target_scaler = ArrayScaler().load(target_bundle["mean"], target_bundle["std"])

    metadata = {
        "feature_version": str(bundle.get("feature_version", "v2")),
        "training_wear_limit_um": float(bundle.get("training_wear_limit_um", 5.0)),
        "training_actual_life_max": float(bundle.get("training_actual_life_max", 150000.0)),
        "training_actual_life_mean": float(bundle.get("training_actual_life_mean", 120000.0)),
        "recommended_cycle_step": float(bundle.get("recommended_cycle_step", 1000.0)),
        "available_coating": str(bundle.get("available_coating", "DLC")),
        "model_architecture": str(bundle.get("model_architecture", config.model_kind)),
        "sequence_length": int(bundle.get("sequence_length", config.sequence_length)),
        "candidate_name": str(bundle.get("candidate_name", config.name)),
        "use_cycle_step": bool(bundle.get("use_cycle_step", False)),
        "run_stress_mono_lambda": float(bundle.get("run_stress_mono_lambda", 0.0)),
        "run_stress_slow_lambda": float(bundle.get("run_stress_slow_lambda", 0.0)),
        "static_feature_order": list(bundle.get("static_feature_order", config.static_feature_order)),
        "dynamic_feature_order": list(bundle.get("dynamic_feature_order", config.dynamic_feature_order)),
        "input_feature_order": list(bundle.get("input_feature_order", config.input_feature_order)),
        "best_epoch": int(bundle.get("best_epoch", 0)),
    }
    return model, input_scalers, target_scaler, metadata, config


def build_shadow_requests(summary_df: pd.DataFrame, case_tables: dict[str, pd.DataFrame], threshold_um: float) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for row in measured_summary(summary_df).itertuples(index=False):
        file_name = str(row.file_name)
        case_df = case_tables[file_name]
        requests.append(
            {
                "source_file": str(row.source_file),
                "coating_name": "DLC",
                "F": float(row.F),
                "D": float(row.D),
                "Cr": float(row.Cr),
                "elastic_modulus_GPa": float(getattr(row, "elastic_modulus_GPa", 210.0)),
                "actual_cycle_step": median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float)),
                "wear_threshold_um": float(threshold_um),
            }
        )
    return requests


def check_monotonic_curve(export_df: pd.DataFrame) -> bool:
    wear_values = export_df["wear_depth"].to_numpy(dtype=float)
    if len(wear_values) <= 1:
        return True
    return bool(np.all(np.diff(wear_values) >= -1e-10))


def run_feature_consistency_suite(case_tables: dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> pd.DataFrame:
    measured_df = measured_summary(summary_df)
    rows: list[dict[str, Any]] = []
    probe_configs = [
        make_v1_protocol_candidate(sequence_length=ROUND1_SEQUENCE_LENGTH),
        make_v2_single_branch_candidate(sequence_length=ROUND1_SEQUENCE_LENGTH),
        make_v2_dual_branch_candidate(sequence_length=ROUND1_SEQUENCE_LENGTH),
    ]

    for row in measured_df.itertuples(index=False):
        file_name = str(row.file_name)
        dataset = build_raw_sequence_dataset({file_name: case_tables[file_name]}, ROUND1_SEQUENCE_LENGTH)
        if len(dataset) == 0:
            continue
        probe_index = min(3, len(dataset) - 1)
        raw_sequence = dataset.raw_sequences[probe_index : probe_index + 1]
        actual_step = dataset.actual_step[probe_index : probe_index + 1]
        history = raw_sequence[0].tolist()

        for config in probe_configs:
            train_inputs, _ = build_model_inputs(raw_sequence, actual_step, config, fit=True)
            infer_raw = make_raw_sequence(history, config.sequence_length)[np.newaxis, :, :]
            infer_inputs, _ = build_model_inputs(infer_raw, actual_step, config, fit=True)
            matches = True
            for key in train_inputs:
                matches = matches and np.allclose(train_inputs[key], infer_inputs[key], atol=1e-6)

            rows.append(
                {
                    "test_run": file_name,
                    "candidate_name": config.name,
                    "sequence_length": int(config.sequence_length),
                    "feature_consistent": bool(matches),
                }
            )

    return pd.DataFrame(rows)


def config_from_summary_row(row: pd.Series | dict[str, Any]) -> CandidateConfig:
    values = dict(row)
    return CandidateConfig(
        name=str(values["candidate_name"]),
        model_kind=str(values["model_kind"]),
        sequence_length=int(values["sequence_length"]),
        seed=int(values.get("seed", PRIMARY_SEED)),
        dropout=float(values.get("dropout", 0.0)),
        wear_consistency_lambda=float(values.get("wear_consistency_lambda", 0.0)),
        run_stress_mono_lambda=float(values.get("run_stress_mono_lambda", 0.0)),
        run_stress_slow_lambda=float(values.get("run_stress_slow_lambda", 0.0)),
        use_cycle_step=bool(values.get("use_cycle_step", False)),
        use_validation=bool(values.get("use_validation", False)),
        learning_rate=float(values.get("learning_rate", DEFAULT_LEARNING_RATE)),
        weight_decay=float(values.get("weight_decay", DEFAULT_WEIGHT_DECAY)),
        batch_size=int(values.get("batch_size", DEFAULT_BATCH_SIZE)),
        max_epochs=int(values.get("max_epochs", DEFAULT_MAX_EPOCHS)),
        patience=int(values.get("patience", DEFAULT_PATIENCE)),
        grad_clip_norm=float(values.get("grad_clip_norm", DEFAULT_GRAD_CLIP)),
    )
