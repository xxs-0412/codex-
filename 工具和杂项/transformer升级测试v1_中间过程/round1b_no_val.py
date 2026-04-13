from __future__ import annotations

import copy
import math
import sys

sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import matplotlib.pyplot as plt
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
SEQ_LEN = 6
EPOCHS = 3000
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EXTRA_STEPS = 400
GRAD_CLIP_NORM = 1.0

DROPOUT = 0.1

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "磨损数据（改）"

MODEL_COLORS = {"Transformer": "#dc2626"}
BASELINE_COLOR = "#2563eb"

CARD_BG = "#fbfaf7"
GRID_COLOR = "#d8dee6"
TEXT_COLOR = "#14202b"
MUTED_COLOR = "#5f6b76"


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
    def __init__(self, seq_len: int = SEQ_LEN, d_model: int = 32, nhead: int = 4, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.input_proj = nn.Linear(5, d_model)
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


def build_sequence_dataset(case_tables: dict[str, pd.DataFrame], seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
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


def train_model(train_seq: np.ndarray, train_y: np.ndarray) -> tuple[nn.Module, FeatureScaler, TargetScaler]:
    seq_scaler = FeatureScaler().fit(train_seq.reshape(-1, train_seq.shape[-1]))
    target_scaler = TargetScaler().fit(train_y)
    train_x_scaled = to_tensor(scale_sequences(train_seq, seq_scaler))
    train_y_scaled = to_tensor(target_scaler.transform(train_y))
    raw_seq_t = to_tensor(train_seq)

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
        loss = (
            data_loss
            + MONO_LAMBDA_WEAR * wear_pen
            + MONO_LAMBDA_LOAD * load_pen
            + MONO_LAMBDA_CLEARANCE * clearance_pen
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 500 == 0:
            print(f"    epoch {epoch:4d} | loss={loss_value:.6e} | data={float(data_loss.item()):.6e}")

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


def save_life_error_bar(scan_df: pd.DataFrame, baseline_df: pd.DataFrame, out_path: Path) -> None:
    cases = sorted(scan_df["test_case"].unique())
    x = np.arange(len(cases), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5.8))
    fig.patch.set_facecolor("#f4f1eb")
    ax.set_facecolor(CARD_BG)

    baseline_vals = [float(baseline_df.loc[baseline_df["test_case"] == c, "life_abs_error"].values[0]) for c in cases]
    upgrade_vals = [float(scan_df.loc[scan_df["test_case"] == c, "life_abs_error"].values[0]) for c in cases]

    bars1 = ax.bar(x - width / 2, baseline_vals, width=width, color=BASELINE_COLOR, edgecolor="#ffffff", linewidth=1.0, label="4.13v1 Baseline", alpha=0.75)
    bars2 = ax.bar(x + width / 2, upgrade_vals, width=width, color=MODEL_COLORS["Transformer"], edgecolor="#ffffff", linewidth=1.0, label="Round1b Upgrade")

    for bar, val in zip(bars1, baseline_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.0f}", ha="center", va="bottom", fontsize=7.0, color=BASELINE_COLOR)
    for bar, val in zip(bars2, upgrade_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.0f}", ha="center", va="bottom", fontsize=7.0, color=MODEL_COLORS["Transformer"])

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=18)
    ax.set_ylabel("Life absolute error (cycles)", fontsize=11, color=TEXT_COLOR)
    ax.set_title("Round1b vs Baseline: Life Error by Test Case", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_summary_comparison(summary_df: pd.DataFrame, baseline_summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.patch.set_facecolor("#f4f1eb")
    metric_specs = [
        ("mean_pressure_mae", "Mean Pressure MAE (MPa)", "{:.2f}"),
        ("mean_wear_mae_um", "Mean Wear-Curve MAE (um)", "{:.3f}"),
        ("mean_life_abs_error", "Mean Life Abs Error (cycles)", "{:.0f}"),
    ]

    for ax, (column, title, fmt) in zip(axes, metric_specs):
        labels = ["4.13v1 Baseline", "Round1b Upgrade"]
        vals = [
            float(baseline_summary[column].values[0]),
            float(summary_df[column].values[0]),
        ]
        colors = [BASELINE_COLOR, MODEL_COLORS["Transformer"]]
        bars = ax.barh(labels, vals, color=colors, edgecolor="#ffffff", linewidth=1.2, height=0.55)
        style_axes(ax)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
        ax.set_xlabel("Lower is better", fontsize=9.5, color=MUTED_COLOR, labelpad=8)
        ax.invert_yaxis()
        x_max = max(vals) * 1.18 if len(vals) else 1.0
        ax.set_xlim(0.0, x_max)

        for idx, (bar, val) in enumerate(zip(bars, vals)):
            fontweight = "bold" if (idx == 1 and vals[1] < vals[0]) or (idx == 0 and vals[0] < vals[1]) else "normal"
            ax.text(
                bar.get_width() + x_max * 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                fmt.format(val),
                va="center",
                ha="left",
                fontsize=9.5,
                color=TEXT_COLOR,
                fontweight=fontweight,
            )

    fig.suptitle("Round1b (No Val Set) vs 4.13v1 Baseline", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Round: 1b (Training Strategy - No Validation Set)")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Dropout: {DROPOUT}")
    print(f"Grad clip norm: {GRAD_CLIP_NORM}")
    print(f"Validation: no (use training loss for model selection)")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {SCRIPT_DIR}")
    print()

    summary_df, case_tables = load_data()
    eligible = eligible_cases(summary_df)
    print(f"Total cases: {len(summary_df)}")
    print(f"Eligible cases (wear >= {WEAR_THRESHOLD_UM} um): {len(eligible)}")
    for _, row in eligible.iterrows():
        print(f"  {row['source_file']:20s} | F={row['F']:8.1f} | D={row['D']:6.1f} | Cr={row['Cr']:7.4f} | life={row['actual_life']:10.1f} | wear={row['final_wear_um']:.2f} um")
    print()

    baseline_path = SCRIPT_DIR.parent.parent / "结果" / "4.13v1基准测试" / "详细结果_各折各模型.csv"
    baseline_df = None
    baseline_summary = None
    if baseline_path.exists():
        baseline_all = pd.read_csv(baseline_path)
        baseline_df = baseline_all[baseline_all["model"] == "Transformer"].reset_index(drop=True)
        baseline_summary_path = SCRIPT_DIR.parent.parent / "结果" / "4.13v1基准测试" / "汇总_各模型平均指标.csv"
        if baseline_summary_path.exists():
            baseline_summary_all = pd.read_csv(baseline_summary_path)
            baseline_summary = baseline_summary_all[baseline_summary_all["model"] == "Transformer"].reset_index(drop=True)
        print(f"Loaded baseline from: {baseline_path}")
        print(f"Baseline mean life_abs_error: {float(baseline_df['life_abs_error'].mean()):.0f}")
        print()

    scan_rows: list[dict] = []
    total_folds = len(eligible)
    run_count = 0

    for fold_idx, test_row in eligible.iterrows():
        test_file = str(test_row["file_name"])
        test_source = str(test_row["source_file"])
        train_tables = {name: table for name, table in case_tables.items() if name != test_file}
        test_table = case_tables[test_file]

        run_count += 1
        print(f"[{run_count}/{total_folds}] test={test_source} | train={len(train_tables)} cases")

        train_seq, train_y = build_sequence_dataset(train_tables, SEQ_LEN)
        test_seq, test_y = build_sequence_dataset({test_file: test_table}, SEQ_LEN)
        true_curve_df = threshold_ground_truth(test_table, WEAR_THRESHOLD_UM, test_row)
        true_life_actual = float(test_row["actual_life"])

        model, seq_scaler, target_scaler = train_model(train_seq, train_y)

        pressure_metrics = evaluate_pressure(model, seq_scaler, target_scaler, test_seq, test_y)
        rollout_df, predicted_life = rollout_case(model, seq_scaler, target_scaler, test_table, WEAR_THRESHOLD_UM, REAL_WEAR_COEFF_MPA_INV, true_life_actual)
        curve_mae = wear_curve_mae(true_curve_df, rollout_df)

        scan_rows.append(
            {
                "test_case": test_source,
                "model": "Transformer_R1b",
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

    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(SCRIPT_DIR / "详细结果_各折.csv", index=False, encoding="utf-8-sig")

    summary_stats = pd.DataFrame(
        [
            {
                "model": "Transformer_R1b",
                "mean_pressure_mae": float(scan_df["pressure_mae"].mean()),
                "mean_pressure_rmse": float(scan_df["pressure_rmse"].mean()),
                "mean_pressure_mape": float(scan_df["pressure_mape"].mean()),
                "mean_wear_mae_um": float(scan_df["wear_mae_um"].mean()),
                "mean_life_abs_error": float(scan_df["life_abs_error"].mean()),
                "median_life_abs_error": float(scan_df["life_abs_error"].median()),
                "mean_life_rel_error": float(scan_df["life_rel_error"].mean()),
                "max_life_abs_error": float(scan_df["life_abs_error"].max()),
                "min_life_abs_error": float(scan_df["life_abs_error"].min()),
            }
        ]
    )
    summary_stats.to_csv(SCRIPT_DIR / "汇总_指标.csv", index=False, encoding="utf-8-sig")

    if baseline_df is not None:
        save_life_error_bar(scan_df, baseline_df, SCRIPT_DIR / "图1_各工况寿命误差对比.png")
    if baseline_summary is not None:
        save_summary_comparison(summary_stats, baseline_summary, SCRIPT_DIR / "图2_指标对比柱状图.png")

    notes = [
        "Transformer 升级测试 v1 — 第一轮b：训练策略优化（无验证集）",
        "=" * 60,
        f"测试日期: 2026-04-13",
        f"设备: {DEVICE}",
        "",
        "本轮改动（相比4.13v1基线）：",
        f"  dropout        0.0  -> {DROPOUT}",
        f"  learning_rate  1e-3 -> {LEARNING_RATE}",
        f"  weight_decay   1e-6 -> {WEIGHT_DECAY}",
        f"  grad_clip_norm  无  -> {GRAD_CLIP_NORM}",
        f"  epochs         1200 -> {EPOCHS}",
        f"  验证集          无  -> 无（用训练loss选模型）",
        "",
        "模型架构（未改动）：",
        "  d_model=32, nhead=4, 2层encoder, dim_feedforward=64, gelu",
        f"  seq_len={SEQ_LEN}",
        "",
        f"磨损阈值: {WEAR_THRESHOLD_UM} um",
        f"磨损系数: {REAL_WEAR_COEFF_MPA_INV:.3e} MPa^-1",
        "",
        "4.13v1基线结果：",
        "  平均寿命绝对误差: 18673 转",
        "  压力MAE: 22.33 MPa",
        "  磨损MAE: 0.360 μm",
    ]
    (SCRIPT_DIR / "测试说明.txt").write_text("\n".join(notes), encoding="utf-8")

    print()
    print("=" * 60)
    print("Round 1b Complete: Training Strategy (No Val Set)")
    print("=" * 60)
    print(f"Mean life abs error: {float(scan_df['life_abs_error'].mean()):.0f} (baseline: 18673)")
    print(f"Mean pressure MAE: {float(scan_df['pressure_mae'].mean()):.2f} (baseline: 22.33)")
    print(f"Mean wear MAE: {float(scan_df['wear_mae_um'].mean()):.3f} (baseline: 0.360)")
    print()
    print(f"Results saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
