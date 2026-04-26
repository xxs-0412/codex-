from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


WEAR_THRESHOLD_UM = 5.0
REAL_WEAR_COEFF_MPA_INV = 1.84e-10
EPS = 1e-12

SEED = 20260419
MAX_EXTRA_STEPS = 400

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "数据快照" / "完整30run_4.24" / "处理后数据"
SPLIT_CSV = ROOT_DIR / "数据快照" / "30测试集" / "30测试集划分清单.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_COLORS = {
    "FNN": "#2563eb",
    "GRU": "#0f766e",
    "LSTM": "#7c3aed",
    "1D-CNN": "#d97706",
    "Transformer": "#dc2626",
    "Transformer_v2": "#b91c1c",
    "baseline": "#64748b",
    "shape_loss_v1": "#0ea5e9",
    "shape_loss_v2": "#ef4444",
    "mono_strict": "#0ea5e9",
    "slow_only": "#f97316",
    "mono_strict_plus_slow": "#ef4444",
    "mono_tolerant_plus_slow": "#10b981",
    "baseline_5d": "#64748b",
    "R1_静态派生_plus_slow_only": "#10b981",
    "R2_log_cycle_plus_slow_only": "#ef4444",
    "R1_静态派生": "#0f766e",
    "R2_log_cycle": "#2563eb",
    "R3_差分增强": "#b45309",
}

CARD_BG = "#fbfaf7"
GRID_COLOR = "#d8dee6"
TEXT_COLOR = "#14202b"
MUTED_COLOR = "#5f6b76"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    columns: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class TransformerConfig:
    seq_len: int
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 64
    dropout: float = 0.0
    epochs: int = 1200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6


@dataclass(frozen=True)
class ShapeLossConfig:
    name: str
    mono_lambda: float
    slow_lambda: float
    mono_tolerance_ratio: float = 0.0


@dataclass(frozen=True)
class RolloutStepInfo:
    native_actual_step: float
    native_sim_step: float
    used_actual_step: float
    used_sim_step: float
    actual_step_cap: float | None = None
    capped: bool = False


LEGACY_FEATURE_SPEC = FeatureSpec(
    name="legacy_5d",
    columns=("F", "D", "Cr", "actual_cycle", "wear_depth"),
    description="原始 5 维特征",
)

PREV_STRESS_6D_SPEC = FeatureSpec(
    name="prev_stress_6d",
    columns=("F", "D", "Cr", "actual_cycle", "wear_depth", "prev_stress"),
    description="原始 5 维 + 前一步应力",
)

FEATURE_R1_SPEC = FeatureSpec(
    name="R1_static_derived",
    columns=("F", "D", "Cr", "F_over_D_sq", "Cr_over_D", "actual_cycle", "wear_depth"),
    description="静态派生特征",
)

FEATURE_R2_SPEC = FeatureSpec(
    name="R2_log_cycle",
    columns=("F", "D", "Cr", "F_over_D_sq", "Cr_over_D", "log1p_actual_cycle", "wear_depth"),
    description="静态派生 + log1p(cycle)",
)

FEATURE_R3_SPEC = FeatureSpec(
    name="R3_delta_features",
    columns=("F", "D", "Cr", "F_over_D_sq", "Cr_over_D", "log1p_actual_cycle", "wear_depth", "delta_cycle", "delta_wear"),
    description="静态派生 + log1p(cycle) + 差分特征",
)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 17-test-set experiments.")


def feature_spec_uses_prev_stress(feature_spec: FeatureSpec) -> bool:
    return "prev_stress" in feature_spec.columns


def device_summary_lines() -> list[str]:
    lines = [
        f"Device: {DEVICE}",
        f"Torch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        lines.extend(
            [
                f"CUDA device index: {index}",
                f"CUDA device name: {torch.cuda.get_device_name(index)}",
                f"CUDA capability: {props.major}.{props.minor}",
                f"CUDA total memory (GB): {props.total_memory / (1024 ** 3):.2f}",
            ]
        )
    return lines


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class FeatureScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, array: np.ndarray) -> "FeatureScaler":
        values = np.asarray(array, dtype=np.float32)
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        return (values - self.mean) / self.std


class TargetScaler(FeatureScaler):
    def inverse_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std_t = torch.tensor(self.std, dtype=torch.float32, device=tensor.device)
        return tensor * std_t + mean_t


class FNNNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.net(x)


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class CNN1DNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.conv(z)
        z = z.reshape(z.shape[0], -1)
        return self.head(z)


class TransformerNet(nn.Module):
    def __init__(self, input_dim: int, config: TransformerConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        z = self.encoder(z)
        return self.head(z[:, -1, :])


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_path = DATA_DIR / "修改数据汇总.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    case_tables: dict[str, pd.DataFrame] = {}
    for row in summary_df.itertuples(index=False):
        case_tables[str(row.file_name)] = pd.read_csv(DATA_DIR / str(row.file_name))
    return summary_df, case_tables


def eligible_cases(summary_df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        summary_df["has_measured_pressure"].astype(bool)
        & (summary_df["final_wear_um"].astype(float) >= WEAR_THRESHOLD_UM)
    )
    return summary_df.loc[mask].sort_values("actual_life").reset_index(drop=True)


def load_fixed_split(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Missing split file: {SPLIT_CSV}")
    split_df = pd.read_csv(SPLIT_CSV)
    eligible = eligible_cases(summary_df)
    eligible_map = {str(row.file_name): row for row in eligible.itertuples(index=False)}

    rows: list[dict[str, object]] = []
    for row in split_df.itertuples(index=False):
        file_name = str(row.file_name)
        if file_name not in eligible_map:
            continue
        eligible_row = eligible_map[file_name]
        rows.append(
            {
                "file_name": file_name,
                "source_file": str(eligible_row.source_file),
                "role": str(row.role),
                "actual_life": float(eligible_row.actual_life),
                "final_wear_um": float(eligible_row.final_wear_um),
            }
        )

    merged = pd.DataFrame(rows).sort_values("actual_life").reset_index(drop=True)
    train_df = merged[merged["role"] == "train"].reset_index(drop=True)
    test_df = merged[merged["role"] == "test"].reset_index(drop=True)
    return train_df, test_df


def _build_raw_sequence_dataset(
    case_tables: dict[str, pd.DataFrame],
    seq_len: int,
    include_prev_stress: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    raw_sequences: list[np.ndarray] = []
    targets: list[list[float]] = []
    case_names: list[str] = []
    step_indices: list[int] = []

    for case_name, table in case_tables.items():
        rows = table.iloc[:-1].reset_index(drop=True)
        features = rows[["F", "D", "Cr", "actual_cycle", "wear_depth"]].to_numpy(dtype=np.float32)
        if include_prev_stress:
            prev_stress = np.zeros((len(rows), 1), dtype=np.float32)
            stress_values = rows["stress"].to_numpy(dtype=np.float32)
            if len(stress_values) > 1:
                prev_stress[1:, 0] = stress_values[:-1]
            features = np.concatenate([features, prev_stress], axis=1)
        targets_log = np.log(np.clip(rows["stress"].to_numpy(dtype=np.float32), EPS, None))
        for idx in range(len(rows)):
            start = max(0, idx - seq_len + 1)
            seq = features[start : idx + 1]
            if len(seq) < seq_len:
                pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            raw_sequences.append(seq.astype(np.float32))
            targets.append([float(targets_log[idx])])
            case_names.append(str(case_name))
            step_indices.append(int(idx))

    return (
        np.asarray(raw_sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        case_names,
        np.asarray(step_indices, dtype=np.int32),
    )


def build_raw_sequence_dataset(
    case_tables: dict[str, pd.DataFrame],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    return _build_raw_sequence_dataset(case_tables, seq_len, include_prev_stress=False)


def build_raw_sequence_dataset_prev_stress(
    case_tables: dict[str, pd.DataFrame],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    return _build_raw_sequence_dataset(case_tables, seq_len, include_prev_stress=True)


def _diff_with_zero(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        out = np.zeros_like(arr)
        if len(arr) > 1:
            out[1:] = arr[1:] - arr[:-1]
        return out
    out = np.zeros_like(arr)
    if arr.shape[-1] > 1:
        out[..., 1:] = arr[..., 1:] - arr[..., :-1]
    return out


def transform_raw_sequences(raw_sequences: np.ndarray, feature_spec: FeatureSpec) -> np.ndarray:
    raw = np.asarray(raw_sequences, dtype=np.float32)
    F = raw[..., 0]
    D = raw[..., 1]
    Cr = raw[..., 2]
    cycle = raw[..., 3]
    wear = raw[..., 4]
    prev_stress = raw[..., 5] if raw.shape[-1] > 5 else np.zeros_like(cycle)

    F_over_D_sq = F / np.maximum(D**2, EPS)
    Cr_over_D = Cr / np.maximum(D, EPS)
    log_cycle = np.log1p(np.maximum(cycle, 0.0))
    delta_cycle = _diff_with_zero(cycle)
    delta_wear = _diff_with_zero(wear)

    feature_map = {
        "F": F,
        "D": D,
        "Cr": Cr,
        "actual_cycle": cycle,
        "wear_depth": wear,
        "F_over_D_sq": F_over_D_sq,
        "Cr_over_D": Cr_over_D,
        "log1p_actual_cycle": log_cycle,
        "delta_cycle": delta_cycle,
        "delta_wear": delta_wear,
        "prev_stress": prev_stress,
    }
    stacked = np.stack([feature_map[column] for column in feature_spec.columns], axis=-1)
    return stacked.astype(np.float32)


def scale_sequences(sequences: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    flat = sequences.reshape(-1, sequences.shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(sequences.shape)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


def instantiate_model(model_name: str, input_dim: int, config: TransformerConfig) -> nn.Module:
    if model_name == "FNN":
        return FNNNet(input_dim).to(DEVICE)
    if model_name == "GRU":
        return GRUNet(input_dim).to(DEVICE)
    if model_name == "LSTM":
        return LSTMNet(input_dim).to(DEVICE)
    if model_name == "1D-CNN":
        return CNN1DNet(input_dim, config.seq_len).to(DEVICE)
    if model_name in {"Transformer", "Transformer_v2"}:
        return TransformerNet(input_dim, config).to(DEVICE)
    raise ValueError(f"Unsupported model: {model_name}")


def monotonic_penalty(
    model: nn.Module,
    raw_seq: torch.Tensor,
    seq_scaler: FeatureScaler,
    feature_spec: FeatureSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wear_seq = raw_seq.clone()
    wear_seq[:, :, 4] = wear_seq[:, :, 4] + WEAR_DELTA_MM
    load_seq = raw_seq.clone()
    load_seq[:, :, 0] = load_seq[:, :, 0] * (1.0 + LOAD_DELTA_RATIO)
    clearance_seq = raw_seq.clone()
    clearance_seq[:, :, 2] = clearance_seq[:, :, 2] + CLEARANCE_DELTA_MM

    base_features = transform_raw_sequences(raw_seq.detach().cpu().numpy(), feature_spec)
    wear_features = transform_raw_sequences(wear_seq.detach().cpu().numpy(), feature_spec)
    load_features = transform_raw_sequences(load_seq.detach().cpu().numpy(), feature_spec)
    clearance_features = transform_raw_sequences(clearance_seq.detach().cpu().numpy(), feature_spec)

    base_scaled = to_tensor(scale_sequences(base_features, seq_scaler))
    wear_scaled = to_tensor(scale_sequences(wear_features, seq_scaler))
    load_scaled = to_tensor(scale_sequences(load_features, seq_scaler))
    clearance_scaled = to_tensor(scale_sequences(clearance_features, seq_scaler))

    base_pred = model(base_scaled)
    wear_pred = model(wear_scaled)
    load_pred = model(load_scaled)
    clearance_pred = model(clearance_scaled)

    wear_pen = torch.mean(torch.relu(wear_pred - base_pred) ** 2)
    load_pen = torch.mean(torch.relu(base_pred - load_pred) ** 2)
    clearance_pen = torch.mean(torch.relu(base_pred - clearance_pred) ** 2)
    return wear_pen, load_pen, clearance_pen


def scaled_log_to_stress(pred_log_scaled: torch.Tensor, target_scaler: TargetScaler) -> torch.Tensor:
    pred_log = target_scaler.inverse_torch(pred_log_scaled)
    return torch.exp(pred_log)


def build_run_shape_pairs(
    raw_sequences: np.ndarray,
    case_names: list[str],
    step_indices: np.ndarray,
    max_wear_mm: float = 0.005,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    for case_name, step_index, raw_seq in zip(case_names, step_indices.tolist(), raw_sequences):
        grouped.setdefault(str(case_name), []).append((int(step_index), raw_seq))

    mono_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    slow_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for items in grouped.values():
        items = sorted(items, key=lambda item: item[0])
        for idx in range(len(items) - 1):
            curr_step, curr_seq = items[idx]
            next_step, next_seq = items[idx + 1]
            if next_step != curr_step + 1:
                continue
            if float(curr_seq[-1, 4]) <= max_wear_mm and float(next_seq[-1, 4]) <= max_wear_mm:
                mono_pairs.append((curr_seq.copy(), next_seq.copy()))

        for idx in range(len(items) - 2):
            step0, seq0 = items[idx]
            step1, seq1 = items[idx + 1]
            step2, seq2 = items[idx + 2]
            if not (step1 == step0 + 1 and step2 == step1 + 1):
                continue
            if (
                float(seq0[-1, 4]) <= max_wear_mm
                and float(seq1[-1, 4]) <= max_wear_mm
                and float(seq2[-1, 4]) <= max_wear_mm
            ):
                slow_triples.append((seq0.copy(), seq1.copy(), seq2.copy()))

    return mono_pairs, slow_triples


def run_stress_shape_losses(
    model: nn.Module,
    mono_pairs: list[tuple[np.ndarray, np.ndarray]],
    slow_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    feature_spec: FeatureSpec,
    mono_tolerance_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    mono_loss = zero
    slow_loss = zero

    if mono_pairs:
        left = np.asarray([pair[0] for pair in mono_pairs], dtype=np.float32)
        right = np.asarray([pair[1] for pair in mono_pairs], dtype=np.float32)
        left_scaled = to_tensor(scale_sequences(transform_raw_sequences(left, feature_spec), seq_scaler))
        right_scaled = to_tensor(scale_sequences(transform_raw_sequences(right, feature_spec), seq_scaler))
        s_t = scaled_log_to_stress(model(left_scaled), target_scaler)
        s_t1 = scaled_log_to_stress(model(right_scaled), target_scaler)
        tolerance = float(mono_tolerance_ratio) * torch.maximum(s_t, torch.tensor(EPS, dtype=torch.float32, device=DEVICE))
        mono_loss = torch.mean(torch.relu(s_t1 - s_t - tolerance))

    if slow_triples:
        a = np.asarray([triple[0] for triple in slow_triples], dtype=np.float32)
        b = np.asarray([triple[1] for triple in slow_triples], dtype=np.float32)
        c = np.asarray([triple[2] for triple in slow_triples], dtype=np.float32)
        a_scaled = to_tensor(scale_sequences(transform_raw_sequences(a, feature_spec), seq_scaler))
        b_scaled = to_tensor(scale_sequences(transform_raw_sequences(b, feature_spec), seq_scaler))
        c_scaled = to_tensor(scale_sequences(transform_raw_sequences(c, feature_spec), seq_scaler))
        s0 = scaled_log_to_stress(model(a_scaled), target_scaler)
        s1 = scaled_log_to_stress(model(b_scaled), target_scaler)
        s2 = scaled_log_to_stress(model(c_scaled), target_scaler)
        drop0 = s0 - s1
        drop1 = s1 - s2
        slow_loss = torch.mean(torch.relu(drop1 - drop0))

    return mono_loss, slow_loss


def train_model(
    model_name: str,
    train_raw_seq: np.ndarray,
    train_y: np.ndarray,
    feature_spec: FeatureSpec,
    config: TransformerConfig,
    shape_config: ShapeLossConfig | None = None,
    train_case_names: list[str] | None = None,
    train_step_indices: np.ndarray | None = None,
) -> tuple[nn.Module, FeatureScaler, TargetScaler]:
    input_seq = transform_raw_sequences(train_raw_seq, feature_spec)
    seq_scaler = FeatureScaler().fit(input_seq.reshape(-1, input_seq.shape[-1]))
    target_scaler = TargetScaler().fit(train_y)
    train_x_scaled = to_tensor(scale_sequences(input_seq, seq_scaler))
    train_y_scaled = to_tensor(target_scaler.transform(train_y))
    raw_seq_t = to_tensor(train_raw_seq)

    mono_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    slow_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if shape_config is not None and (shape_config.mono_lambda > 0.0 or shape_config.slow_lambda > 0.0):
        mono_pairs, slow_triples = build_run_shape_pairs(train_raw_seq, train_case_names or [], train_step_indices if train_step_indices is not None else np.zeros(len(train_raw_seq), dtype=np.int32))

    model = instantiate_model(model_name, input_seq.shape[-1], config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        pred = model(train_x_scaled)
        data_loss = mse(pred, train_y_scaled)
        wear_pen, load_pen, clearance_pen = monotonic_penalty(model, raw_seq_t, seq_scaler, feature_spec)
        loss = (
            data_loss
            + MONO_LAMBDA_WEAR * wear_pen
            + MONO_LAMBDA_LOAD * load_pen
            + MONO_LAMBDA_CLEARANCE * clearance_pen
        )
        if shape_config is not None and (shape_config.mono_lambda > 0.0 or shape_config.slow_lambda > 0.0):
            mono_loss, slow_loss = run_stress_shape_losses(
                model,
                mono_pairs,
                slow_triples,
                seq_scaler,
                target_scaler,
                feature_spec,
                shape_config.mono_tolerance_ratio,
            )
            loss = loss + shape_config.mono_lambda * mono_loss + shape_config.slow_lambda * slow_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 200 == 0:
            message = f"    epoch {epoch:4d} | loss={loss_value:.6e} | data={float(data_loss.item()):.6e}"
            if shape_config is not None and (shape_config.mono_lambda > 0.0 or shape_config.slow_lambda > 0.0):
                message += f" | mono={float(mono_loss.item()):.6e} | slow={float(slow_loss.item()):.6e}"
            print(message)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, seq_scaler, target_scaler


def evaluate_pressure(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    raw_seq_data: np.ndarray,
    target_log: np.ndarray,
    feature_spec: FeatureSpec,
) -> dict[str, float]:
    transformed = transform_raw_sequences(raw_seq_data, feature_spec)
    x_scaled = to_tensor(scale_sequences(transformed, seq_scaler))
    model.eval()
    with torch.no_grad():
        pred_log_scaled = model(x_scaled)
        pred_log = target_scaler.inverse_torch(pred_log_scaled).cpu().numpy().reshape(-1)
    pred_stress = np.exp(pred_log)
    true_stress = np.exp(target_log.reshape(-1))
    rmse = float(np.sqrt(np.mean((pred_stress - true_stress) ** 2)))
    mae = float(np.mean(np.abs(pred_stress - true_stress)))
    mape = float(np.mean(np.abs(pred_stress - true_stress) / np.maximum(true_stress, EPS)))
    return {"pressure_rmse": rmse, "pressure_mae": mae, "pressure_mape": mape}


def median_positive_diff(values: np.ndarray) -> float:
    diffs = np.diff(values.astype(np.float64))
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        raise ValueError("Expected a strictly increasing sequence.")
    return float(np.median(positive_diffs))


def resolve_rollout_steps(case_df: pd.DataFrame, actual_step_cap: float | None = None) -> RolloutStepInfo:
    native_actual_step = median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float))
    native_sim_step = median_positive_diff(case_df["sim_cycle"].to_numpy(dtype=float))

    if actual_step_cap is None:
        used_actual_step = native_actual_step
        capped = False
    else:
        if float(actual_step_cap) <= 0.0:
            raise ValueError("actual_step_cap must be positive.")
        used_actual_step = min(native_actual_step, float(actual_step_cap))
        capped = used_actual_step + EPS < native_actual_step

    used_sim_step = native_sim_step * used_actual_step / max(native_actual_step, EPS)
    return RolloutStepInfo(
        native_actual_step=float(native_actual_step),
        native_sim_step=float(native_sim_step),
        used_actual_step=float(used_actual_step),
        used_sim_step=float(used_sim_step),
        actual_step_cap=None if actual_step_cap is None else float(actual_step_cap),
        capped=bool(capped),
    )


def make_sequence(history: list[list[float]], seq_len: int) -> np.ndarray:
    seq = np.asarray(history[-seq_len:], dtype=np.float32)
    if len(seq) < seq_len:
        pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
        seq = np.vstack([pad, seq])
    return seq


def build_history_row(
    F: float,
    D: float,
    Cr: float,
    actual_cycle: float,
    wear_depth: float,
    feature_spec: FeatureSpec,
    prev_stress: float = 0.0,
) -> list[float]:
    row = [F, D, Cr, actual_cycle, wear_depth]
    if feature_spec_uses_prev_stress(feature_spec):
        row.append(prev_stress)
    return row


def predict_pressure_from_history(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    history: list[list[float]],
    feature_spec: FeatureSpec,
    seq_len: int,
) -> float:
    raw_seq = make_sequence(history, seq_len)
    input_seq = transform_raw_sequences(raw_seq[np.newaxis, :, :], feature_spec)
    x_scaled = to_tensor(scale_sequences(input_seq, seq_scaler))
    model.eval()
    with torch.no_grad():
        pred_log_scaled = model(x_scaled)
        pred_log = target_scaler.inverse_torch(pred_log_scaled).cpu().numpy().reshape(-1)[0]
    return float(np.exp(pred_log))


def threshold_ground_truth(case_df: pd.DataFrame, threshold_um: float, summary_row: pd.Series) -> pd.DataFrame:
    rows = [
        {
            "actual_cycle": float(row.actual_cycle),
            "wear_depth_um": float(row.wear_depth) * 1000.0,
            "stress": float(row.stress),
        }
        for row in case_df.itertuples(index=False)
    ]
    base_df = pd.DataFrame(rows)
    hit_idx = np.where(base_df["wear_depth_um"].to_numpy(dtype=float) >= threshold_um - EPS)[0]
    if len(hit_idx) > 0:
        idx = int(hit_idx[0])
        if idx == 0:
            return base_df.iloc[[0]].copy()
        if abs(base_df.iloc[idx]["wear_depth_um"] - threshold_um) < 1e-9:
            return base_df.iloc[: idx + 1].copy()
        prev_row = base_df.iloc[idx - 1]
        curr_row = base_df.iloc[idx]
        ratio = (threshold_um - float(prev_row["wear_depth_um"])) / max(float(curr_row["wear_depth_um"] - prev_row["wear_depth_um"]), EPS)
        extra = {
            "actual_cycle": float(prev_row["actual_cycle"] + ratio * (curr_row["actual_cycle"] - prev_row["actual_cycle"])),
            "wear_depth_um": float(threshold_um),
            "stress": float(prev_row["stress"] + ratio * (curr_row["stress"] - prev_row["stress"])),
        }
        return pd.concat([base_df.iloc[:idx], pd.DataFrame([extra])], ignore_index=True)
    extra = {
        "actual_cycle": float(summary_row["actual_life"]),
        "wear_depth_um": float(threshold_um),
        "stress": float(base_df.iloc[-1]["stress"]),
    }
    return pd.concat([base_df, pd.DataFrame([extra])], ignore_index=True)


def rollout_case(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    case_df: pd.DataFrame,
    threshold_um: float,
    real_k: float,
    true_life_actual: float,
    feature_spec: FeatureSpec,
    seq_len: int,
) -> tuple[pd.DataFrame, float]:
    step_info = resolve_rollout_steps(case_df)
    rollout_df, predicted_life_actual = _rollout_case_impl(
        model,
        seq_scaler,
        target_scaler,
        case_df,
        threshold_um,
        real_k,
        true_life_actual,
        feature_spec,
        seq_len,
        actual_step=step_info.used_actual_step,
        sim_step=step_info.used_sim_step,
    )
    return rollout_df, predicted_life_actual


def rollout_case_with_step_cap(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    case_df: pd.DataFrame,
    threshold_um: float,
    real_k: float,
    true_life_actual: float,
    feature_spec: FeatureSpec,
    seq_len: int,
    actual_step_cap: float | None,
) -> tuple[pd.DataFrame, float, RolloutStepInfo]:
    step_info = resolve_rollout_steps(case_df, actual_step_cap=actual_step_cap)
    rollout_df, predicted_life_actual = _rollout_case_impl(
        model,
        seq_scaler,
        target_scaler,
        case_df,
        threshold_um,
        real_k,
        true_life_actual,
        feature_spec,
        seq_len,
        actual_step=step_info.used_actual_step,
        sim_step=step_info.used_sim_step,
    )
    return rollout_df, predicted_life_actual, step_info


def _rollout_case_impl(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    case_df: pd.DataFrame,
    threshold_um: float,
    real_k: float,
    true_life_actual: float,
    feature_spec: FeatureSpec,
    seq_len: int,
    actual_step: float,
    sim_step: float,
) -> tuple[pd.DataFrame, float]:
    first = case_df.iloc[0]
    F = float(first["F"])
    D = float(first["D"])
    Cr = float(first["Cr"])
    threshold_mm = threshold_um / 1000.0

    actual_cycle = 0.0
    sim_cycle = 0.0
    wear_depth = 0.0
    history: list[list[float]] = [build_history_row(F, D, Cr, actual_cycle, wear_depth, feature_spec, prev_stress=0.0)]
    rows: list[dict[str, float]] = []
    predicted_life_actual = true_life_actual

    internal_limit = max(true_life_actual * 1.35, float(case_df["actual_cycle"].max()) * 1.15, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + MAX_EXTRA_STEPS

    for _ in range(max_steps):
        pred_stress = predict_pressure_from_history(model, seq_scaler, target_scaler, history, feature_spec, seq_len)
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
        history.append(build_history_row(F, D, Cr, actual_cycle, wear_depth, feature_spec, prev_stress=pred_stress))

    return pd.DataFrame(rows), float(predicted_life_actual)


def wear_curve_mae(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    true_x = true_df["actual_cycle"].to_numpy(dtype=float)
    true_y = true_df["wear_depth_um"].to_numpy(dtype=float)
    pred_x = pred_df["actual_cycle"].to_numpy(dtype=float)
    pred_y = pred_df["pred_wear_depth_um"].to_numpy(dtype=float)
    pred_interp = np.interp(true_x, pred_x, pred_y)
    return float(np.mean(np.abs(pred_interp - true_y)))


def style_axes(ax) -> None:
    ax.set_facecolor(CARD_BG)
    ax.grid(axis="x", alpha=0.0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9d1da")
    ax.spines["bottom"].set_color("#c9d1da")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)


def color_list(labels: list[str]) -> list[str]:
    return [MODEL_COLORS.get(label, "#64748b") for label in labels]


def save_grouped_life_error_bar(
    scan_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    pivot = scan_df.pivot(index="test_case", columns="model", values="life_abs_error").sort_index()
    tests = pivot.index.tolist()
    models = pivot.columns.tolist()
    x = np.arange(len(tests), dtype=float)
    width = 0.8 / max(len(models), 1)

    colors = [MODEL_COLORS.get(m, "#64748b") for m in models]
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#f4f1eb")
    ax.set_facecolor(CARD_BG)

    for idx, model in enumerate(models):
        vals = pivot[model].to_numpy(dtype=float)
        offset = (idx - (len(models) - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width=width, color=colors[idx], edgecolor="#ffffff", linewidth=0.9, label=model)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tests, rotation=15)
    ax.set_ylabel("Life absolute error (cycles)", fontsize=11, color=TEXT_COLOR)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.legend(frameon=False, ncol=max(1, len(models)), loc="upper center")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_summary_bar_chart(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    metric_specs = [
        ("mean_pressure_mae", "Mean Pressure MAE (MPa)", "{:.2f}"),
        ("mean_wear_mae_um", "Mean Wear-Curve MAE (um)", "{:.3f}"),
        ("mean_life_abs_error", "Mean Life Abs Error (cycles)", "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.patch.set_facecolor("#f4f1eb")

    for ax, (column, label, fmt) in zip(axes, metric_specs):
        ranked = summary_df.sort_values(column, ascending=True).reset_index(drop=True)
        labels = ranked["model"].astype(str).tolist()
        vals = ranked[column].to_numpy(dtype=float)
        bars = ax.barh(labels, vals, color=color_list(labels), edgecolor="#ffffff", linewidth=1.2, height=0.68)
        style_axes(ax)
        ax.set_title(label, fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
        ax.set_xlabel("Lower is better", fontsize=9.5, color=MUTED_COLOR, labelpad=8)
        ax.invert_yaxis()
        x_max = max(vals) * 1.18 if len(vals) else 1.0
        ax.set_xlim(0.0, x_max)
        best_value = float(np.min(vals)) if len(vals) else 0.0
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + x_max * 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                fmt.format(val),
                va="center",
                ha="left",
                fontsize=9.2,
                color=TEXT_COLOR,
                fontweight="bold" if abs(val - best_value) < 1e-9 else "normal",
            )

    fig.suptitle(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_mean_predicted_life_chart(summary_df: pd.DataFrame, scan_df: pd.DataFrame, out_path: Path, title: str) -> None:
    labels = summary_df["model"].astype(str).tolist()
    pred_values = scan_df.groupby("model")["predicted_life"].mean().reindex(labels).to_numpy(dtype=float)
    true_value = float(scan_df["true_life"].mean())

    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.patch.set_facecolor("#f4f1eb")
    bars = ax.bar(labels, pred_values, color=color_list(labels), edgecolor="#ffffff", linewidth=1.3, width=0.66)
    style_axes(ax)
    ax.set_ylabel("Mean predicted life (cycles)", fontsize=11, color=TEXT_COLOR)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=14)
    ax.tick_params(axis="x", rotation=12)
    ax.axhline(true_value, color="#111827", linewidth=1.8, linestyle="--", label=f"Mean FE truth = {true_value:.0f}")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)

    y_max = max(max(pred_values), true_value) * 1.16 if len(pred_values) else max(true_value, 1.0)
    ax.set_ylim(0.0, y_max)
    best_model = summary_df.sort_values("mean_life_abs_error", ascending=True).iloc[0]["model"]
    error_map = summary_df.set_index("model")["mean_life_abs_error"].to_dict()
    for label, bar, value in zip(labels, bars, pred_values):
        error = float(error_map[label])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + y_max * 0.015,
            f"{value:.0f}\n|err|={error:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=TEXT_COLOR,
            fontweight="bold" if label == best_model else "normal",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

