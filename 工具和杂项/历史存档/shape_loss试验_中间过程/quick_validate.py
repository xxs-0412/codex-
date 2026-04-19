from __future__ import annotations

import copy
import math
import sys

sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

WEAR_THRESHOLD_UM = 5.0
REAL_WEAR_COEFF_MPA_INV = 1.84e-10
ELASTIC_MODULUS_GPA = 210.0
EPS = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 20260413
SEQ_LEN = 15
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
EPOCHS = 1200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MAX_EXTRA_STEPS = 400

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002

RUN_STRESS_MONO_LAMBDA = 0.05
RUN_STRESS_SLOW_LAMBDA = 0.01
SHAPE_MAX_WEAR_MM = 0.005

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "磨损数据（改）"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, array: np.ndarray) -> FeatureScaler:
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
    def __init__(self, seq_len: int = SEQ_LEN, d_model: int = D_MODEL, nhead: int = NHEAD, num_layers: int = NUM_LAYERS, dim_feedforward: int = DIM_FF) -> None:
        super().__init__()
        self.input_proj = nn.Linear(5, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
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
    summary_path = DATA_DIR / "修改数据汇总.csv"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        sys.exit(1)
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


def build_sequence_dataset(case_tables: dict[str, pd.DataFrame], seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[list[float]] = []
    case_names: list[str] = []
    step_indices: list[int] = []
    for case_name, table in case_tables.items():
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
            case_names.append(case_name)
            step_indices.append(idx)
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        case_names,
        np.asarray(step_indices, dtype=np.int64),
    )


def build_run_shape_pairs(
    case_names: list[str],
    step_indices: np.ndarray,
    raw_sequences: np.ndarray,
    max_wear_mm: float = SHAPE_MAX_WEAR_MM,
) -> tuple[np.ndarray, np.ndarray]:
    from collections import defaultdict

    run_groups: dict[str, list[int]] = defaultdict(list)
    for sample_idx, (cn, si) in enumerate(zip(case_names, step_indices)):
        run_groups[cn].append((int(si), sample_idx))
    for cn in run_groups:
        run_groups[cn].sort(key=lambda x: x[0])

    mono_pairs: list[tuple[int, int]] = []
    slow_triples: list[tuple[int, int, int]] = []

    for cn, members in run_groups.items():
        sorted_members = [m for _, m in members]
        for i in range(len(sorted_members) - 1):
            idx_t = sorted_members[i]
            idx_t1 = sorted_members[i + 1]
            wear_t = float(raw_sequences[idx_t, -1, 4])
            wear_t1 = float(raw_sequences[idx_t1, -1, 4])
            if wear_t <= max_wear_mm and wear_t1 <= max_wear_mm:
                mono_pairs.append((idx_t, idx_t1))

        for i in range(len(sorted_members) - 2):
            idx_t = sorted_members[i]
            idx_t1 = sorted_members[i + 1]
            idx_t2 = sorted_members[i + 2]
            wear_t = float(raw_sequences[idx_t, -1, 4])
            wear_t1 = float(raw_sequences[idx_t1, -1, 4])
            wear_t2 = float(raw_sequences[idx_t2, -1, 4])
            if wear_t <= max_wear_mm and wear_t1 <= max_wear_mm and wear_t2 <= max_wear_mm:
                slow_triples.append((idx_t, idx_t1, idx_t2))

    mono_arr = np.asarray(mono_pairs, dtype=np.int64).reshape(-1, 2) if mono_pairs else np.empty((0, 2), dtype=np.int64)
    slow_arr = np.asarray(slow_triples, dtype=np.int64).reshape(-1, 3) if slow_triples else np.empty((0, 3), dtype=np.int64)
    return mono_arr, slow_arr


def scale_sequences(sequences: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    flat = sequences.reshape(-1, sequences.shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(sequences.shape)


def monotonic_penalty(
    model: nn.Module,
    raw_seq: torch.Tensor,
    seq_scaler: FeatureScaler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def scaled_log_to_stress(pred_log_scaled: torch.Tensor, target_scaler: TargetScaler) -> torch.Tensor:
    pred_log = target_scaler.inverse_torch(pred_log_scaled)
    return torch.exp(pred_log)


def run_stress_shape_losses(
    model: nn.Module,
    train_x_scaled: torch.Tensor,
    target_scaler: TargetScaler,
    mono_pairs: np.ndarray,
    slow_triples: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        all_pred_log_scaled = model(train_x_scaled)
    all_stress = scaled_log_to_stress(all_pred_log_scaled, target_scaler).squeeze(-1)

    mono_loss = torch.tensor(0.0, device=DEVICE)
    if len(mono_pairs) > 0:
        idx_t = torch.tensor(mono_pairs[:, 0], dtype=torch.long, device=DEVICE)
        idx_t1 = torch.tensor(mono_pairs[:, 1], dtype=torch.long, device=DEVICE)
        s_t = all_stress[idx_t]
        s_t1 = all_stress[idx_t1]
        mono_loss = torch.mean(torch.relu(s_t1 - s_t))

    slow_loss = torch.tensor(0.0, device=DEVICE)
    if len(slow_triples) > 0:
        idx_t = torch.tensor(slow_triples[:, 0], dtype=torch.long, device=DEVICE)
        idx_t1 = torch.tensor(slow_triples[:, 1], dtype=torch.long, device=DEVICE)
        idx_t2 = torch.tensor(slow_triples[:, 2], dtype=torch.long, device=DEVICE)
        s_t = all_stress[idx_t]
        s_t1 = all_stress[idx_t1]
        s_t2 = all_stress[idx_t2]
        drop_t = s_t - s_t1
        drop_t1 = s_t1 - s_t2
        slow_loss = torch.mean(torch.relu(drop_t1 - drop_t))

    return mono_loss, slow_loss


def train_model(
    train_seq: np.ndarray,
    train_y: np.ndarray,
    case_names: list[str],
    step_indices: np.ndarray,
    mono_lambda: float = 0.0,
    slow_lambda: float = 0.0,
) -> tuple[nn.Module, FeatureScaler, TargetScaler]:
    seq_scaler = FeatureScaler().fit(train_seq.reshape(-1, train_seq.shape[-1]))
    target_scaler = TargetScaler().fit(train_y)
    train_x_scaled = to_tensor(scale_sequences(train_seq, seq_scaler))
    train_y_scaled = to_tensor(target_scaler.transform(train_y))
    raw_seq_t = to_tensor(train_seq)

    mono_pairs, slow_triples = build_run_shape_pairs(case_names, step_indices, train_seq)
    print(f"    shape pairs: mono={len(mono_pairs)}, slow={len(slow_triples)}")

    model = TransformerNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pred = model(train_x_scaled)
        data_loss = mse(pred, train_y_scaled)
        wear_pen, load_pen, clearance_pen = monotonic_penalty(model, raw_seq_t, seq_scaler)

        mono_loss = torch.tensor(0.0, device=DEVICE)
        slow_loss = torch.tensor(0.0, device=DEVICE)
        if mono_lambda > 0.0 or slow_lambda > 0.0:
            mono_loss, slow_loss = run_stress_shape_losses(
                model, train_x_scaled, target_scaler, mono_pairs, slow_triples
            )

        loss = (
            data_loss
            + MONO_LAMBDA_WEAR * wear_pen
            + MONO_LAMBDA_LOAD * load_pen
            + MONO_LAMBDA_CLEARANCE * clearance_pen
            + mono_lambda * mono_loss
            + slow_lambda * slow_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 200 == 0:
            mono_val = float(mono_loss.item()) if mono_lambda > 0.0 else 0.0
            slow_val = float(slow_loss.item()) if slow_lambda > 0.0 else 0.0
            print(f"    epoch {epoch:4d} | loss={loss_value:.6e} | data={float(data_loss.item()):.6e} | mono={mono_val:.4f} | slow={slow_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, seq_scaler, target_scaler


def evaluate_pressure(
    model: nn.Module,
    seq_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    seq_data: np.ndarray,
    target_log: np.ndarray,
) -> dict:
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


def make_sequence(history: list[list[float]], seq_len: int = SEQ_LEN) -> np.ndarray:
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
) -> float:
    seq = make_sequence(history)
    x_scaled = to_tensor(scale_sequences(seq[np.newaxis, :, :], seq_scaler))
    model.eval()
    with torch.no_grad():
        pred_log_scaled = model(x_scaled)
        pred_log = target_scaler.inverse_torch(pred_log_scaled).cpu().numpy().reshape(-1)[0]
    return float(np.exp(pred_log))


def threshold_ground_truth(case_df: pd.DataFrame, threshold_um: float, summary_row: pd.Series) -> pd.DataFrame:
    target_mm = threshold_um / 1000.0
    rows: list[dict] = []
    for row in case_df.itertuples(index=False):
        rows.append(
            {
                "actual_cycle": float(row.actual_cycle),
                "wear_depth_um": float(row.wear_depth) * 1000.0,
                "stress": float(row.stress),
            }
        )
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
    rows: list[dict] = []
    predicted_life_actual = true_life_actual

    internal_limit = max(true_life_actual * 1.35, float(case_df["actual_cycle"].max()) * 1.15, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + MAX_EXTRA_STEPS

    for _ in range(max_steps):
        pred_stress = predict_pressure_from_history(model, seq_scaler, target_scaler, history)
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


def main() -> None:
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Shape Loss Quick Validation (2 folds only)")
    print(f"  seq_len={SEQ_LEN}, d_model={D_MODEL}, nhead={NHEAD}, layers={NUM_LAYERS}, ff={DIM_FF}")
    print(f"  mono_lambda={RUN_STRESS_MONO_LAMBDA}, slow_lambda={RUN_STRESS_SLOW_LAMBDA}")
    print()

    summary_df, case_tables = load_data()
    eligible = eligible_cases(summary_df)
    print(f"Eligible cases: {len(eligible)}")
    print()

    configs = [
        ("baseline", 0.0, 0.0),
        ("shape_v1", RUN_STRESS_MONO_LAMBDA, 0.0),
        ("shape_v2", RUN_STRESS_MONO_LAMBDA, RUN_STRESS_SLOW_LAMBDA),
    ]

    scan_rows: list[dict] = []
    test_folds = eligible.iloc[:2]
    total_runs = len(configs) * len(test_folds)
    run_count = 0

    for fold_idx, test_row in test_folds.iterrows():
        test_file = str(test_row["file_name"])
        test_source = str(test_row["source_file"])
        train_tables = {name: table for name, table in case_tables.items() if name != test_file}
        test_table = case_tables[test_file]

        train_seq, train_y, case_names, step_indices = build_sequence_dataset(train_tables, SEQ_LEN)
        test_seq, test_y, _, _ = build_sequence_dataset({test_file: test_table}, SEQ_LEN)
        true_curve_df = threshold_ground_truth(test_table, WEAR_THRESHOLD_UM, test_row)
        true_life_actual = float(test_row["actual_life"])

        for config_name, mono_l, slow_l in configs:
            run_count += 1
            print(f"[{run_count}/{total_runs}] fold={test_source} | config={config_name} (mono={mono_l}, slow={slow_l})")

            model, seq_scaler, target_scaler = train_model(
                train_seq, train_y, case_names, step_indices,
                mono_lambda=mono_l, slow_lambda=slow_l,
            )

            pressure_metrics = evaluate_pressure(model, seq_scaler, target_scaler, test_seq, test_y)
            rollout_df, predicted_life = rollout_case(model, seq_scaler, target_scaler, test_table, WEAR_THRESHOLD_UM, REAL_WEAR_COEFF_MPA_INV, true_life_actual)
            curve_mae = wear_curve_mae(true_curve_df, rollout_df)

            scan_rows.append(
                {
                    "test_case": test_source,
                    "config": config_name,
                    "mono_lambda": mono_l,
                    "slow_lambda": slow_l,
                    "pressure_mae": pressure_metrics["pressure_mae"],
                    "pressure_rmse": pressure_metrics["pressure_rmse"],
                    "pressure_mape": pressure_metrics["pressure_mape"],
                    "wear_mae_um": curve_mae,
                    "predicted_life": predicted_life,
                    "true_life": true_life_actual,
                    "life_abs_error": abs(predicted_life - true_life_actual),
                    "life_rel_error": abs(predicted_life - true_life_actual) / max(true_life_actual, EPS),
                }
            )
            print(f"    -> life_err={abs(predicted_life - true_life_actual):.0f}, pressure_mae={pressure_metrics['pressure_mae']:.2f}, wear_mae={curve_mae:.4f}")
            print()

    result_df = pd.DataFrame(scan_rows)
    result_df.to_csv(SCRIPT_DIR / "quick_validation.csv", index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("Quick Validation Complete")
    print("=" * 60)
    print(result_df[["test_case", "config", "life_abs_error", "pressure_mae", "wear_mae_um"]].to_string(index=False))


if __name__ == "__main__":
    main()
