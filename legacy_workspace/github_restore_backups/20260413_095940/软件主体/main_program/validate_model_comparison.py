from __future__ import annotations

import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from device_runtime import cpu_state_dict, resolve_training_device
from train_real_wear_models import (
    EPS,
    Standardizer,
    build_training_arrays,
    comparison_dir,
    load_cases,
    load_processing_config,
    median_positive_diff,
    recommended_cycle_step,
    rollout_case,
    save_curve_comparison,
    set_seed,
    threshold_ground_truth,
    trained_model_dir,
)

SEED = 20260408
DEVICE, DEVICE_LABEL = resolve_training_device()

TEST_RUNS = ["试验11.csv", "试验7.csv"]
TRAIN_EPOCHS = 2000
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEQ_LEN = 6

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002


class FNNNet(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)


class CNNet(nn.Module):
    def __init__(self, in_dim=5, seq_len=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x.permute(0, 2, 1))
        h = h.squeeze(-1)
        return self.head(h)


class GRUNet(nn.Module):
    def __init__(self, in_dim=5, hidden=48):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, h = self.gru(x)
        return self.head(h.squeeze(0))


class LSTMNet(nn.Module):
    def __init__(self, in_dim=5, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        return self.head(h.squeeze(0))


class TransformerNet(nn.Module):
    def __init__(self, in_dim=5, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.proj(x) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)
        return self.head(h[:, -1, :])


MODEL_REGISTRY = {
    "FNN": FNNNet,
    "1D-CNN": CNNet,
    "GRU": GRUNet,
    "LSTM": LSTMNet,
    "Transformer": TransformerNet,
}


def build_sequence_arrays(case_tables, seq_len=SEQ_LEN):
    all_seqs = []
    all_targets = []
    for table in case_tables.values():
        rows = table.iloc[:-1]
        feats = rows[["F", "D", "Cr", "actual_cycle", "wear_depth"]].to_numpy(dtype=np.float32)
        targets = np.log(np.maximum(rows["stress"].to_numpy(dtype=np.float32), EPS)).reshape(-1, 1)
        n = len(feats)
        for i in range(n):
            start = max(0, i - seq_len + 1)
            seq = feats[start:i + 1]
            pad_count = seq_len - len(seq)
            if pad_count > 0:
                pad = np.repeat(seq[:1], pad_count, axis=0)
                seq = np.vstack([pad, seq])
            all_seqs.append(seq)
            all_targets.append(targets[i])
    return np.asarray(all_seqs, dtype=np.float32), np.asarray(all_targets, dtype=np.float32)


def train_model(model, train_x, train_y, raw_x_np, x_scaler, y_scaler, epochs=TRAIN_EPOCHS):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    mse = nn.MSELoss()
    n = len(train_x)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n, device=DEVICE)
        epoch_losses = []

        for start in range(0, n, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            x_batch = train_x[idx]
            y_batch = train_y[idx]

            pred = model(x_batch)
            data_loss = mse(pred, y_batch)

            raw_batch = raw_x_np[idx.cpu().numpy()].copy()

            wear_shift = raw_batch.copy()
            wear_shift[:, -1] = wear_shift[:, -1] + WEAR_DELTA_MM
            raw_shift_t = torch.tensor(x_scaler.transform_np(wear_shift), dtype=torch.float32, device=DEVICE)
            pred_wear_plus = model(raw_shift_t)

            load_shift = raw_batch.copy()
            load_shift[:, 0] = load_shift[:, 0] * (1.0 + LOAD_DELTA_RATIO)
            load_shift_t = torch.tensor(x_scaler.transform_np(load_shift), dtype=torch.float32, device=DEVICE)
            pred_load_plus = model(load_shift_t)

            clearance_shift = raw_batch.copy()
            clearance_shift[:, 2] = clearance_shift[:, 2] + CLEARANCE_DELTA_MM
            clearance_shift_t = torch.tensor(x_scaler.transform_np(clearance_shift), dtype=torch.float32, device=DEVICE)
            pred_clearance_plus = model(clearance_shift_t)

            wear_penalty = torch.mean(torch.relu(pred_wear_plus - pred) ** 2)
            load_penalty = torch.mean(torch.relu(pred - pred_load_plus) ** 2)
            clearance_penalty = torch.mean(torch.relu(pred - pred_clearance_plus) ** 2)

            loss = data_loss + MONO_LAMBDA_WEAR * wear_penalty + MONO_LAMBDA_LOAD * load_penalty + MONO_LAMBDA_CLEARANCE * clearance_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        avg_loss = np.mean(epoch_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred_log = y_scaler.inverse_torch(model(train_x))
                true_log = y_scaler.inverse_torch(train_y)
                pred_s = torch.exp(pred_log)
                true_s = torch.exp(true_log)
                mape = torch.mean(torch.abs(pred_s - true_s) / (true_s + EPS)).item()
            print(f"  epoch {epoch:4d} | loss={avg_loss:.6e} | pressure_mape={mape:.4%}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_pressure_seq(model, x_scaler, y_scaler, F, D, Cr, actual_cycle, wear_depth):
    x = np.array([[F, D, Cr, actual_cycle, wear_depth]], dtype=np.float32)
    x_seq = np.repeat(x, SEQ_LEN, axis=0).reshape(1, SEQ_LEN, 5)
    x_t = torch.tensor(x_scaler.transform_np(x_seq), dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        pred_log_scaled = model(x_t)
        pred_log = y_scaler.inverse_torch(pred_log_scaled)
        pred_stress = torch.exp(pred_log)
    return float(pred_stress.item())


def rollout_seq_model(model, x_scaler, y_scaler, case_df, threshold_um, real_k, true_life):
    first = case_df.iloc[0]
    F, D, Cr = float(first["F"]), float(first["D"]), float(first["Cr"])
    actual_step = median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float))
    sim_step = median_positive_diff(case_df["sim_cycle"].to_numpy(dtype=float))
    threshold_mm = threshold_um / 1000.0
    actual_cycle = 0.0
    sim_cycle = 0.0
    wear_depth = 0.0
    predicted_life = true_life
    rows = []
    internal_limit = max(true_life * 1.4, float(case_df["actual_cycle"].max()) * 1.2, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + 400

    for _ in range(max_steps):
        pred_stress = predict_pressure_seq(model, x_scaler, y_scaler, F, D, Cr, actual_cycle, wear_depth)
        rows.append({"actual_cycle": actual_cycle, "pred_stress": pred_stress, "pred_wear_depth_um": wear_depth * 1000.0})
        delta_s = actual_step * math.pi * D / 6.0
        delta_wear = real_k * pred_stress * delta_s
        next_cycle = actual_cycle + actual_step
        next_wear = wear_depth + delta_wear
        if next_wear >= threshold_mm:
            ratio = (threshold_mm - wear_depth) / max(delta_wear, EPS)
            predicted_life = actual_cycle + ratio * actual_step
            rows.append({"actual_cycle": predicted_life, "pred_stress": pred_stress, "pred_wear_depth_um": threshold_um})
            break
        actual_cycle = next_cycle
        sim_cycle += sim_step
        wear_depth = next_wear

    return pd.DataFrame(rows), float(predicted_life)


def save_comparison_chart(out_path, results, test_runs, case_tables, threshold_um, summary_df):
    fig, axes = plt.subplots(1, len(test_runs), figsize=(7 * len(test_runs), 5.6))
    if len(test_runs) == 1:
        axes = [axes]

    for ax, test_file in zip(axes, test_runs):
        test_row = summary_df[summary_df["file_name"] == test_file].iloc[0]
        true_life = float(test_row["actual_life"])
        true_df = threshold_ground_truth(case_tables[test_file], threshold_um, test_row)
        ax.plot(true_df["actual_cycle"], true_df["wear_depth_um"], color="#111827", linewidth=2.3, label="FE truth")

        colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]
        for i, (name, res) in enumerate(results.items()):
            if test_file in res:
                rollout = res[test_file]["rollout"]
                pred_life = res[test_file]["predicted_life"]
                ax.plot(rollout["actual_cycle"], rollout["pred_wear_depth_um"],
                        color=colors[i % len(colors)], linewidth=1.8,
                        label=f"{name} ({pred_life:.0f})")

        ax.axhline(threshold_um, color="#dc2626", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Actual Cycle")
        ax.set_ylabel("Wear Depth (um)")
        ax.set_title(f"{test_file} | true={true_life:.0f}")
        ax.grid(alpha=0.28)
        ax.legend(fontsize=8)

    fig.suptitle("Multi-Model Wear Prediction Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_life_comparison_bar(out_path, results, test_runs, summary_df):
    model_names = list(results.keys())
    n_models = len(model_names)
    n_tests = len(test_runs)
    x = np.arange(n_tests)
    width = 0.8 / (n_models + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    true_lives = []
    for test_file in test_runs:
        row = summary_df[summary_df["file_name"] == test_file].iloc[0]
        true_lives.append(float(row["actual_life"]))

    ax.bar(x - width * n_models / 2, true_lives, width, color="#111827", label="FE truth")

    colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]
    for j, name in enumerate(model_names):
        pred_lives = []
        for test_file in test_runs:
            if test_file in results[name]:
                pred_lives.append(results[name][test_file]["predicted_life"])
            else:
                pred_lives.append(0)
        ax.bar(x - width * n_models / 2 + width * (j + 1), pred_lives, width,
               color=colors[j % len(colors)], label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(test_runs)
    ax.set_ylabel("Predicted Life (cycles)")
    ax.set_title("Life Prediction Comparison Across Models")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    set_seed(SEED)
    config = load_processing_config()
    threshold_um = float(config["wear_threshold_um"])
    real_k = float(config["real_wear_coeff_mpa_inv"])

    summary_df, case_tables = load_cases()
    train_files = [f for f in case_tables if f not in TEST_RUNS]
    train_tables = {f: case_tables[f] for f in train_files}

    print(f"device: {DEVICE_LABEL}")
    print(f"test runs: {TEST_RUNS}")
    print(f"train runs: {len(train_tables)}")

    out_dir = comparison_dir() / "model_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    for model_name, ModelClass in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        set_seed(SEED)
        model = ModelClass()

        train_x_np, train_y_np = build_sequence_arrays(train_tables, SEQ_LEN)
        n_samples, seq_len, feat_dim = train_x_np.shape

        flat_x = train_x_np.reshape(-1, feat_dim)
        x_scaler = Standardizer().fit(flat_x)
        y_scaler = Standardizer().fit(train_y_np)
        x_scaler.to_torch(DEVICE)
        y_scaler.to_torch(DEVICE)

        train_x_scaled = x_scaler.transform_np(train_x_np)
        train_x_t = torch.tensor(train_x_scaled, dtype=torch.float32, device=DEVICE)
        train_y_t = torch.tensor(y_scaler.transform_np(train_y_np), dtype=torch.float32, device=DEVICE)

        raw_flat = flat_x.copy()

        model = train_model(model, train_x_t, train_y_t, raw_flat, x_scaler, y_scaler, TRAIN_EPOCHS)

        model_results = {}
        for test_file in TEST_RUNS:
            test_table = case_tables[test_file]
            test_row = summary_df[summary_df["file_name"] == test_file].iloc[0]
            true_life = float(test_row["actual_life"])

            rollout_df, predicted_life = rollout_seq_model(
                model, x_scaler, y_scaler, test_table, threshold_um, real_k, true_life
            )

            life_abs_error = abs(predicted_life - true_life)
            life_rel_error = life_abs_error / max(true_life, EPS)

            model_results[test_file] = {
                "predicted_life": predicted_life,
                "true_life": true_life,
                "life_abs_error": life_abs_error,
                "life_rel_error": life_rel_error,
                "rollout": rollout_df,
            }

            print(f"  {test_file}: true={true_life:.0f} | pred={predicted_life:.0f} | "
                  f"abs_err={life_abs_error:.0f} | rel_err={life_rel_error:.2%}")

            summary_rows.append({
                "model": model_name,
                "test_run": test_file,
                "true_life": true_life,
                "predicted_life": predicted_life,
                "life_abs_error": life_abs_error,
                "life_rel_error": life_rel_error,
            })

        all_results[model_name] = model_results

    summary_df_out = pd.DataFrame(summary_rows)
    summary_df_out.to_csv(out_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to {out_dir / 'validation_summary.csv'}")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(summary_df_out.to_string(index=False))

    pivot = summary_df_out.pivot(index="model", columns="test_run", values="life_rel_error")
    print("\nRelative Error by Model x Test Run:")
    print(pivot.to_string())

    save_comparison_chart(
        out_dir / "wear_curve_comparison.png",
        all_results, TEST_RUNS, case_tables, threshold_um, summary_df
    )
    save_life_comparison_bar(
        out_dir / "life_comparison_bar.png",
        all_results, TEST_RUNS, summary_df
    )

    print(f"\nCharts saved to {out_dir}")


if __name__ == "__main__":
    main()
