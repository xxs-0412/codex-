from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass

sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

WEAR_THRESHOLD_UM = 5.0
REAL_WEAR_COEFF_MPA_INV = 1.84e-10
EPS = 1e-12

SEED = 20260418
MAX_EXTRA_STEPS = 400

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "提取结果_处理后"
BASELINE_RESULT_DIR = ROOT_DIR / "结果" / "4.18v1基准测试"

DETAIL_ALL_FILENAME = "\u5019\u9009\u6c47\u603b.csv"
NOTES_FILENAME = "\u8fd0\u884c\u8bf4\u660e.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class RoundConfig:
    round_id: str
    description: str
    seq_len: int
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 64
    dropout: float = 0.0
    epochs: int = 1200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_norm: float | None = None
    use_validation: bool = False
    lr_patience: int = 80
    lr_factor: float = 0.5
    es_patience: int = 200
    es_min_delta: float = 1e-5


ROUND_CONFIGS = [
    RoundConfig(
        round_id="round1_training_strategy",
        description="validation+regularization branch",
        seq_len=6,
        dropout=0.1,
        epochs=3000,
        learning_rate=5e-4,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        use_validation=True,
    ),
    RoundConfig(
        round_id="round1b_no_val",
        description="regularization branch without validation split",
        seq_len=6,
        dropout=0.1,
        epochs=3000,
        learning_rate=5e-4,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
    ),
    RoundConfig(
        round_id="round2_seq10",
        description="seq_len=10",
        seq_len=10,
    ),
    RoundConfig(
        round_id="round2b_seq10_mild",
        description="seq_len=10 + mild training strategy",
        seq_len=10,
        epochs=1800,
        learning_rate=5e-4,
        weight_decay=1e-5,
        grad_clip_norm=1.0,
    ),
    RoundConfig(
        round_id="round3_capacity",
        description="seq_len=10, d_model=64, ff=128, 2 layers",
        seq_len=10,
        d_model=64,
        num_layers=2,
        dim_ff=128,
    ),
    RoundConfig(
        round_id="round3b_capacity",
        description="seq_len=10, d_model=64, ff=256, 3 layers",
        seq_len=10,
        d_model=64,
        num_layers=3,
        dim_ff=256,
    ),
    RoundConfig(
        round_id="round4_seq15",
        description="seq_len=15 final combo",
        seq_len=15,
        d_model=64,
        num_layers=2,
        dim_ff=128,
    ),
]


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transformer upgrade search.")


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


class TransformerNet(nn.Module):
    def __init__(self, config: RoundConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(5, config.d_model)
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


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_path = DATA_DIR / "\u4fee\u6539\u6570\u636e\u6c47\u603b.csv"
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


def build_sequence_dataset(case_tables: dict[str, pd.DataFrame], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[list[float]] = []
    for table in case_tables.values():
        rows = table.iloc[:-1].reset_index(drop=True)
        features = rows[["F", "D", "Cr", "actual_cycle", "wear_depth"]].to_numpy(dtype=np.float32)
        targets_log = np.log(np.clip(rows["stress"].to_numpy(dtype=np.float32), EPS, None))
        for idx in range(len(rows)):
            start = max(0, idx - seq_len + 1)
            seq = features[start : idx + 1]
            if len(seq) < seq_len:
                pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            sequences.append(seq.astype(np.float32))
            targets.append([float(targets_log[idx])])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def scale_sequences(sequences: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    flat = sequences.reshape(-1, sequences.shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(sequences.shape)


def monotonic_penalty(model: nn.Module, raw_seq: torch.Tensor, seq_scaler: FeatureScaler) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wear_seq = raw_seq.clone()
    wear_seq[:, :, 4] = wear_seq[:, :, 4] + WEAR_DELTA_MM
    load_seq = raw_seq.clone()
    load_seq[:, :, 0] = load_seq[:, :, 0] * (1.0 + LOAD_DELTA_RATIO)
    clearance_seq = raw_seq.clone()
    clearance_seq[:, :, 2] = clearance_seq[:, :, 2] + CLEARANCE_DELTA_MM

    base_scaled = to_tensor(scale_sequences(raw_seq.detach().cpu().numpy(), seq_scaler))
    wear_scaled = to_tensor(scale_sequences(wear_seq.detach().cpu().numpy(), seq_scaler))
    load_scaled = to_tensor(scale_sequences(load_seq.detach().cpu().numpy(), seq_scaler))
    clearance_scaled = to_tensor(scale_sequences(clearance_seq.detach().cpu().numpy(), seq_scaler))

    base_pred = model(base_scaled)
    wear_pred = model(wear_scaled)
    load_pred = model(load_scaled)
    clearance_pred = model(clearance_scaled)

    wear_pen = torch.mean(torch.relu(wear_pred - base_pred) ** 2)
    load_pen = torch.mean(torch.relu(base_pred - load_pred) ** 2)
    clearance_pen = torch.mean(torch.relu(base_pred - clearance_pred) ** 2)
    return wear_pen, load_pen, clearance_pen


def pick_validation_case(train_summary: pd.DataFrame) -> str:
    sorted_df = train_summary.sort_values("actual_life").reset_index(drop=True)
    return str(sorted_df.iloc[len(sorted_df) // 2]["file_name"])


def train_model(
    config: RoundConfig,
    train_tables: dict[str, pd.DataFrame],
    val_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[nn.Module, FeatureScaler, TargetScaler, dict[str, float | str]]:
    train_seq, train_y = build_sequence_dataset(train_tables, config.seq_len)
    seq_scaler = FeatureScaler().fit(train_seq.reshape(-1, train_seq.shape[-1]))
    target_scaler = TargetScaler().fit(train_y)
    train_x_scaled = to_tensor(scale_sequences(train_seq, seq_scaler))
    train_y_scaled = to_tensor(target_scaler.transform(train_y))
    raw_seq_t = to_tensor(train_seq)

    val_x_scaled = None
    val_y_scaled = None
    if val_tables:
        val_seq, val_y = build_sequence_dataset(val_tables, config.seq_len)
        val_x_scaled = to_tensor(scale_sequences(val_seq, seq_scaler))
        val_y_scaled = to_tensor(target_scaler.transform(val_y))

    model = TransformerNet(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None
    if val_x_scaled is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=1e-7,
        )
    mse = nn.MSELoss()
    best_score = float("inf")
    best_state = None
    epochs_no_improve = 0
    actual_epochs = 0
    best_val_loss = float("nan")

    for epoch in range(1, config.epochs + 1):
        model.train()
        pred = model(train_x_scaled)
        data_loss = mse(pred, train_y_scaled)
        wear_pen, load_pen, clearance_pen = monotonic_penalty(model, raw_seq_t, seq_scaler)
        loss = (
            data_loss
            + MONO_LAMBDA_WEAR * wear_pen
            + MONO_LAMBDA_LOAD * load_pen
            + MONO_LAMBDA_CLEARANCE * clearance_pen
        )

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        with torch.no_grad():
            current_score = float(loss.item())
            if val_x_scaled is not None and val_y_scaled is not None:
                model.eval()
                val_pred = model(val_x_scaled)
                current_score = float(mse(val_pred, val_y_scaled).item())
                best_val_loss = current_score
                if scheduler is not None:
                    scheduler.step(current_score)

        if current_score + (config.es_min_delta if val_x_scaled is not None else 0.0) < best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        actual_epochs = epoch
        if epoch == 1 or epoch % 200 == 0:
            print(
                f"    epoch {epoch:4d} | train_loss={float(loss.item()):.6e}"
                + (f" | val_loss={current_score:.6e}" if val_x_scaled is not None else "")
            )

        if val_x_scaled is not None and epochs_no_improve >= config.es_patience:
            print(f"    early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info: dict[str, float | str] = {
        "actual_epochs": float(actual_epochs),
        "best_score": float(best_score),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "validation_mode": "with_validation" if val_x_scaled is not None else "no_validation",
    }
    if val_x_scaled is not None:
        train_info["best_val_loss"] = float(best_score)
    return model, seq_scaler, target_scaler, train_info


def evaluate_pressure(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    seq_data: np.ndarray,
    target_log: np.ndarray,
) -> dict[str, float]:
    x_scaled = to_tensor(scale_sequences(seq_data, seq_scaler))
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


def make_sequence(history: list[list[float]], seq_len: int) -> np.ndarray:
    seq = np.asarray(history[-seq_len:], dtype=np.float32)
    if len(seq) < seq_len:
        pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
        seq = np.vstack([pad, seq])
    return seq


def predict_pressure_from_history(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    history: list[list[float]],
    seq_len: int,
) -> float:
    seq = make_sequence(history, seq_len)
    x_scaled = to_tensor(scale_sequences(seq[np.newaxis, :, :], seq_scaler))
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
    seq_len: int,
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
        pred_stress = predict_pressure_from_history(model, seq_scaler, target_scaler, history, seq_len)
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


def run_round(config: RoundConfig, eligible: pd.DataFrame, case_tables: dict[str, pd.DataFrame]) -> dict[str, float | str]:
    detail_path = SCRIPT_DIR / f"详细结果_{config.round_id}.csv"
    summary_path = SCRIPT_DIR / f"汇总_{config.round_id}.csv"
    print()
    print("=" * 72)
    print(f"Running {config.round_id}: {config.description}")
    print(
        f"  seq_len={config.seq_len}, d_model={config.d_model}, layers={config.num_layers}, "
        f"ff={config.dim_ff}, dropout={config.dropout}, epochs={config.epochs}, "
        f"lr={config.learning_rate}, wd={config.weight_decay}, val={config.use_validation}"
    )
    print("=" * 72)

    total_folds = len(eligible)
    if detail_path.exists():
        existing_detail_df = pd.read_csv(detail_path)
        rows = existing_detail_df.to_dict("records")
        completed_cases = {str(row["test_case"]) for _, row in existing_detail_df.iterrows()}
        print(f"Resume enabled for {config.round_id}: loaded {len(existing_detail_df)}/{total_folds} folds")
        if summary_path.exists() and len(existing_detail_df) >= total_folds:
            summary_df = pd.read_csv(summary_path)
            print(f"Reuse existing summary for {config.round_id}")
            return summary_df.iloc[0].to_dict()
    else:
        rows = []
        completed_cases: set[str] = set()

    for fold_idx, test_row in eligible.iterrows():
        test_file = str(test_row["file_name"])
        test_source = str(test_row["source_file"])
        if test_source in completed_cases:
            print(f"[{fold_idx + 1}/{total_folds}] test={test_source} | skip existing")
            continue
        remaining = eligible[eligible["file_name"] != test_file].reset_index(drop=True)

        val_tables = None
        val_file = ""
        if config.use_validation:
            val_file = pick_validation_case(remaining)
            train_files = [str(name) for name in remaining["file_name"] if str(name) != val_file]
            val_tables = {val_file: case_tables[val_file]}
        else:
            train_files = [str(name) for name in remaining["file_name"]]

        train_tables = {name: case_tables[name] for name in train_files}
        test_table = case_tables[test_file]

        print(f"[{fold_idx + 1}/{total_folds}] test={test_source}")
        model, seq_scaler, target_scaler, train_info = train_model(config, train_tables, val_tables)

        test_seq, test_y = build_sequence_dataset({test_file: test_table}, config.seq_len)
        true_curve_df = threshold_ground_truth(test_table, WEAR_THRESHOLD_UM, test_row)
        true_life_actual = float(test_row["actual_life"])
        pressure_metrics = evaluate_pressure(model, seq_scaler, target_scaler, test_seq, test_y)
        rollout_df, predicted_life = rollout_case(
            model,
            seq_scaler,
            target_scaler,
            test_table,
            WEAR_THRESHOLD_UM,
            REAL_WEAR_COEFF_MPA_INV,
            true_life_actual,
            config.seq_len,
        )
        curve_mae = wear_curve_mae(true_curve_df, rollout_df)

        row: dict[str, float | str] = {
            "round_id": config.round_id,
            "test_case": test_source,
            "pressure_mae": pressure_metrics["pressure_mae"],
            "pressure_rmse": pressure_metrics["pressure_rmse"],
            "pressure_mape": pressure_metrics["pressure_mape"],
            "wear_mae_um": curve_mae,
            "predicted_life": predicted_life,
            "true_life": true_life_actual,
            "life_abs_error": abs(predicted_life - true_life_actual),
            "life_rel_error": abs(predicted_life - true_life_actual) / max(true_life_actual, EPS),
            "actual_epochs": train_info["actual_epochs"],
            "best_score": train_info["best_score"],
            "final_lr": train_info["final_lr"],
            "validation_mode": train_info["validation_mode"],
            "validation_file": val_file,
        }
        rows.append(row)
        completed_cases.add(test_source)
        pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_row = {
        "round_id": config.round_id,
        "description": config.description,
        "seq_len": config.seq_len,
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "dim_ff": config.dim_ff,
        "dropout": config.dropout,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip_norm": config.grad_clip_norm if config.grad_clip_norm is not None else float("nan"),
        "use_validation": config.use_validation,
        "mean_pressure_mae": float(detail_df["pressure_mae"].mean()),
        "mean_pressure_rmse": float(detail_df["pressure_rmse"].mean()),
        "mean_pressure_mape": float(detail_df["pressure_mape"].mean()),
        "mean_wear_mae_um": float(detail_df["wear_mae_um"].mean()),
        "mean_life_abs_error": float(detail_df["life_abs_error"].mean()),
        "median_life_abs_error": float(detail_df["life_abs_error"].median()),
        "mean_life_rel_error": float(detail_df["life_rel_error"].mean()),
        "max_life_abs_error": float(detail_df["life_abs_error"].max()),
        "min_life_abs_error": float(detail_df["life_abs_error"].min()),
        "mean_actual_epochs": float(detail_df["actual_epochs"].mean()),
    }
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(
        f"Completed {config.round_id}: life_abs_error={summary_row['mean_life_abs_error']:.0f}, "
        f"pressure_MAE={summary_row['mean_pressure_mae']:.2f}, wear_MAE={summary_row['mean_wear_mae_um']:.3f}"
    )
    return summary_row


def write_notes(eligible: pd.DataFrame) -> None:
    lines = [
        "\u8fd0\u884c\u8bf4\u660e",
        "=" * 60,
        f"Data source: {DATA_DIR}",
        f"Eligible runs: {len(eligible)}",
        "Upgrade chain: round1, round1b, round2, round2b, round3, round3b, round4",
        "",
        "CUDA:",
    ]
    lines.extend([f"  {line}" for line in device_summary_lines()])
    (SCRIPT_DIR / NOTES_FILENAME).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    require_cuda()
    set_seed(SEED)
    for line in device_summary_lines():
        print(line)

    summary_df, case_tables = load_data()
    eligible = eligible_cases(summary_df)
    print(f"Eligible runs: {len(eligible)}")
    write_notes(eligible)

    baseline_summary_path = BASELINE_RESULT_DIR / "\u6c47\u603b_\u5404\u6a21\u578b\u5e73\u5747\u6307\u6807.csv"
    if baseline_summary_path.exists():
        baseline_summary = pd.read_csv(baseline_summary_path)
        transformer_row = baseline_summary[baseline_summary["model"] == "Transformer"]
        if not transformer_row.empty:
            print(
                "Baseline transformer from 4.18v1: "
                f"life_abs_error={float(transformer_row.iloc[0]['mean_life_abs_error']):.0f}, "
                f"pressure_MAE={float(transformer_row.iloc[0]['mean_pressure_mae']):.2f}, "
                f"wear_MAE={float(transformer_row.iloc[0]['mean_wear_mae_um']):.3f}"
            )

    summary_rows = [run_round(config, eligible, case_tables) for config in ROUND_CONFIGS]
    all_summary_df = pd.DataFrame(summary_rows).sort_values("mean_life_abs_error").reset_index(drop=True)
    all_summary_df.to_csv(SCRIPT_DIR / DETAIL_ALL_FILENAME, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("Candidate summary")
    print("=" * 72)
    print(all_summary_df[["round_id", "mean_pressure_mae", "mean_wear_mae_um", "mean_life_abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
