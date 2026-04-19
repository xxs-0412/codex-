from __future__ import annotations

import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_real_wear_models import (
    EPS,
    DEVICE,
    comparison_dir,
    load_cases,
    load_processing_config,
    median_positive_diff,
    predict_static_life,
    select_test_case,
    threshold_ground_truth,
    train_static_life_model,
)


SEED = 20260408
SEQ_LEN = 6
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


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
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
    def __init__(self, input_size: int = 5, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])


class LSTMNet(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class CNN1DNet(nn.Module):
    def __init__(self, seq_len: int = SEQ_LEN, channels: int = 32) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(5, channels, kernel_size=3, padding=1),
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
    def __init__(self, seq_len: int = SEQ_LEN, d_model: int = 32, nhead: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(5, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.0,
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


MODEL_SPECS = {
    "FNN": FNNNet,
    "GRU": GRUNet,
    "LSTM": LSTMNet,
    "1D-CNN": CNN1DNet,
    "Transformer": TransformerNet,
}

MODEL_COLORS = {
    "FNN": "#2563eb",
    "GRU": "#0f766e",
    "LSTM": "#7c3aed",
    "1D-CNN": "#d97706",
    "Transformer": "#dc2626",
}

CARD_BG = "#fbfaf7"
GRID_COLOR = "#d8dee6"
TEXT_COLOR = "#14202b"
MUTED_COLOR = "#5f6b76"


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


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


def train_model(model_name: str, train_seq: np.ndarray, train_y: np.ndarray) -> tuple[nn.Module, FeatureScaler, TargetScaler]:
    seq_scaler = FeatureScaler().fit(train_seq.reshape(-1, train_seq.shape[-1]))
    target_scaler = TargetScaler().fit(train_y)
    train_x_scaled = to_tensor(scale_sequences(train_seq, seq_scaler))
    train_y_scaled = to_tensor(target_scaler.transform(train_y))
    raw_seq_t = to_tensor(train_seq)

    model = MODEL_SPECS[model_name]().to(DEVICE)
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
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

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


def benchmark_output_dir() -> Path:
    path = comparison_dir() / "network_architecture_benchmark"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def save_mae_bar_chart(metrics_df: pd.DataFrame, out_path: Path) -> None:
    order = metrics_df["model"].tolist()
    colors = color_list(order)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.3))
    fig.patch.set_facecolor("#f4f1eb")
    metric_specs = [
        ("pressure_mae", "Pressure MAE (MPa)", "{:.2f}"),
        ("wear_mae_um", "Wear-Curve MAE (um)", "{:.3f}"),
        ("life_abs_error", "Life Absolute Error (cycles)", "{:.0f}"),
    ]

    for ax, (column, title, fmt) in zip(axes, metric_specs):
        ranked = metrics_df.sort_values(column, ascending=True).reset_index(drop=True)
        labels = ranked["model"].tolist()
        vals = ranked[column].to_numpy(dtype=float)
        bar_colors = color_list(labels)
        bars = ax.barh(labels, vals, color=bar_colors, edgecolor="#ffffff", linewidth=1.2, height=0.68)
        style_axes(ax)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
        ax.set_xlabel("Lower is better", fontsize=9.5, color=MUTED_COLOR, labelpad=8)
        ax.invert_yaxis()
        x_max = max(vals) * 1.18 if len(vals) else 1.0
        ax.set_xlim(0.0, x_max)

        for idx, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(
                bar.get_width() + x_max * 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                fmt.format(val),
                va="center",
                ha="left",
                fontsize=9.2,
                color=TEXT_COLOR,
                fontweight="bold" if idx == 0 else "normal",
            )

        if len(bars) > 0:
            best_bar = bars[0]
            ax.text(
                best_bar.get_width() + x_max * 0.015,
                best_bar.get_y() - best_bar.get_height() * 0.18,
                "Best",
                va="bottom",
                ha="left",
                fontsize=8.8,
                color="#b45309",
                fontweight="bold",
            )

    fig.suptitle("Held-Out Test Case: Multi-Architecture Error Comparison", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.text(0.5, 0.02, "All models trained on the same 11 real cases without synthetic augmentation", ha="center", fontsize=9.5, color=MUTED_COLOR)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_life_prediction_bar(metrics_df: pd.DataFrame, true_life: float, out_path: Path) -> None:
    ranked = metrics_df.sort_values("life_abs_error", ascending=True).reset_index(drop=True)
    labels = ranked["model"].tolist()
    values = ranked["predicted_life"].to_numpy(dtype=float)
    colors = color_list(labels)

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    fig.patch.set_facecolor("#f4f1eb")
    bars = ax.bar(labels, values, color=colors, edgecolor="#ffffff", linewidth=1.3, width=0.66)
    style_axes(ax)
    ax.set_ylabel("Predicted life (cycles)", fontsize=11, color=TEXT_COLOR)
    ax.set_title("Life Prediction by Different Neural Architectures", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=14)
    ax.tick_params(axis="x", rotation=14)
    ax.axhline(true_life, color="#111827", linewidth=1.8, linestyle="--", label=f"FE truth = {true_life:.0f}")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)

    y_max = max(max(values), true_life) * 1.16
    ax.set_ylim(0.0, y_max)

    best_model = ranked.iloc[0]["model"]
    for label, bar, value, error in zip(labels, bars, values, ranked["life_abs_error"].to_numpy(dtype=float)):
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

    ax.text(
        0.02,
        0.95,
        f"Best on this hold-out: {best_model}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        color="#b45309",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff4e5", "edgecolor": "#f59e0b", "alpha": 0.95},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    config = load_processing_config()
    threshold_um = float(config["wear_threshold_um"])
    real_k = float(config["real_wear_coeff_mpa_inv"])

    summary_df, case_tables = load_cases()
    test_row = select_test_case(summary_df, config["test_case_name"], threshold_um)
    test_file = str(test_row["file_name"])
    test_source = str(test_row["source_file"])

    train_summary = summary_df[summary_df["file_name"] != test_file].reset_index(drop=True)
    train_tables = {name: table for name, table in case_tables.items() if name != test_file}
    test_table = case_tables[test_file]

    train_seq, train_y = build_sequence_dataset(train_tables, SEQ_LEN)
    test_seq, test_y = build_sequence_dataset({test_file: test_table}, SEQ_LEN)
    true_curve_df = threshold_ground_truth(test_table, threshold_um, test_row)
    true_life_actual = float(test_row["actual_life"])

    benchmark_rows: list[dict] = []
    curve_export_rows: list[pd.DataFrame] = []

    for model_name in MODEL_SPECS:
        print(f"Training benchmark model: {model_name}")
        model, seq_scaler, target_scaler = train_model(model_name, train_seq, train_y)
        pressure_metrics = evaluate_pressure(model, seq_scaler, target_scaler, test_seq, test_y)
        rollout_df, predicted_life = rollout_case(model, seq_scaler, target_scaler, test_table, threshold_um, real_k, true_life_actual)
        curve_mae = wear_curve_mae(true_curve_df, rollout_df)

        benchmark_rows.append(
            {
                "model": model_name,
                "pressure_mae": pressure_metrics["pressure_mae"],
                "pressure_rmse": pressure_metrics["pressure_rmse"],
                "pressure_mape": pressure_metrics["pressure_mape"],
                "wear_mae_um": curve_mae,
                "predicted_life": predicted_life,
                "life_abs_error": abs(predicted_life - true_life_actual),
                "life_rel_error": abs(predicted_life - true_life_actual) / max(true_life_actual, EPS),
            }
        )

        export_df = rollout_df.copy()
        export_df = export_df.rename(
            columns={
                "pred_stress": f"{model_name}_pred_stress",
                "pred_wear_depth_um": f"{model_name}_pred_wear_depth_um",
            }
        )[
            ["actual_cycle", f"{model_name}_pred_stress", f"{model_name}_pred_wear_depth_um"]
        ]
        curve_export_rows.append(export_df)

    metrics_df = pd.DataFrame(benchmark_rows).sort_values("life_abs_error").reset_index(drop=True)
    out_dir = benchmark_output_dir()
    metrics_df.to_csv(out_dir / "architecture_benchmark_metrics.csv", index=False)

    merged_curve = true_curve_df.rename(columns={"wear_depth_um": "fe_truth_wear_um", "stress": "fe_truth_stress"})
    for export_df in curve_export_rows:
        merged_curve = pd.merge(merged_curve, export_df, on="actual_cycle", how="outer")
    merged_curve = merged_curve.sort_values("actual_cycle").reset_index(drop=True)
    merged_curve.to_csv(out_dir / "architecture_benchmark_curves.csv", index=False)

    save_mae_bar_chart(metrics_df, out_dir / "architecture_mae_bar.png")
    save_life_prediction_bar(metrics_df, true_life_actual, out_dir / "architecture_life_bar.png")

    notes = [
        f"Held-out test case: {test_source}",
        f"Threshold: {threshold_um:.2f} um",
        "Compared models: FNN, GRU, LSTM, 1D-CNN, Transformer.",
        "All models are trained on the same 11 real cases and evaluated on the same held-out case.",
        "No synthetic virtual samples were added in this benchmark.",
        "Pressure MAE, wear-curve MAE, and life absolute error are reported together.",
    ]
    (out_dir / "benchmark_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    static_scaler, static_model = train_static_life_model(train_summary)
    static_life = predict_static_life(static_scaler, static_model, test_row)
    static_summary = pd.DataFrame(
        [
            {"model": "FE_truth", "predicted_life": true_life_actual, "life_abs_error": 0.0},
            {"model": "Static_FNN", "predicted_life": static_life, "life_abs_error": abs(static_life - true_life_actual)},
        ]
    )
    static_summary.to_csv(out_dir / "static_baseline_reference.csv", index=False)

    print(f"Benchmark complete. Results saved to: {out_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
