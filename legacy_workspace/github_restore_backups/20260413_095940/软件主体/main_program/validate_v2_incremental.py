from __future__ import annotations

import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from device_runtime import resolve_training_device
from train_real_wear_models import (
    EPS,
    Standardizer,
    comparison_dir,
    load_cases,
    load_processing_config,
    median_positive_diff,
    set_seed,
    threshold_ground_truth,
)

SEED = 20260408
DEVICE, DEVICE_LABEL = resolve_training_device()

TEST_RUNS = ["试验11.csv", "试验7.csv"]
TRAIN_EPOCHS = 2000
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-6

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002

WEAR_CONSISTENCY_LAMBDA = 0.25
RUN_STRESS_MONO_LAMBDA = 0.05
RUN_STRESS_SLOW_LAMBDA = 0.01
PATIENCE = 200
VAL_RATIO = 0.1


class TransformerV1(nn.Module):
    def __init__(self, in_dim=5, seq_len=6, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.proj(x) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class TransformerV2Single(nn.Module):
    def __init__(self, in_dim=9, seq_len=10, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.proj(x) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class StaticDynamicTransformer(nn.Module):
    def __init__(self, static_dim=5, dynamic_dim=4, seq_len=10, d_model=32, nhead=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.dynamic_proj = nn.Linear(dynamic_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model + 16, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, dynamic_x, static_x):
        if dynamic_x.dim() == 2:
            dynamic_x = dynamic_x.unsqueeze(1)
        static_ctx = self.static_net(static_x)
        dynamic_ctx = self.dynamic_proj(dynamic_x) + self.pos_embed[:, :dynamic_x.shape[1], :]
        dynamic_ctx = self.encoder(dynamic_ctx)
        fused = torch.cat([dynamic_ctx[:, -1, :], static_ctx], dim=-1)
        return self.head(fused)


def compute_static_features(raw_seq, use_cycle_step=False, actual_step_val=None):
    F = raw_seq[:, -1, 0]
    D = np.maximum(raw_seq[:, -1, 1], EPS)
    Cr = raw_seq[:, -1, 2]
    feats = [F, D, Cr, F / np.maximum(D * D, EPS), Cr / D]
    if use_cycle_step and actual_step_val is not None:
        feats.append(np.full_like(F, actual_step_val, dtype=np.float32))
    return np.stack(feats, axis=-1).astype(np.float32)


def compute_dynamic_features(raw_seq):
    cycles = raw_seq[:, :, 3]
    wear_depth = raw_seq[:, :, 4]
    delta_cycle = np.zeros_like(cycles, dtype=np.float32)
    delta_wear = np.zeros_like(wear_depth, dtype=np.float32)
    delta_cycle[:, 1:] = np.maximum(cycles[:, 1:] - cycles[:, :-1], 0.0)
    delta_wear[:, 1:] = np.maximum(wear_depth[:, 1:] - wear_depth[:, :-1], 0.0)
    return np.stack([np.log1p(np.maximum(cycles, 0.0)), wear_depth, delta_cycle, delta_wear], axis=-1).astype(np.float32)


def build_single_branch_features(raw_seq, actual_step, use_cycle_step=False):
    static = compute_static_features(raw_seq, use_cycle_step, actual_step)
    dynamic = compute_dynamic_features(raw_seq)
    static_seq = np.repeat(static[:, np.newaxis, :], dynamic.shape[1], axis=1)
    return np.concatenate([static_seq, dynamic], axis=-1).astype(np.float32)


def build_v1_sequences(case_tables, seq_len=6):
    all_seqs, all_targets, all_raw, all_steps, all_next_delta = [], [], [], [], []
    for table in case_tables.values():
        rows = table.iloc[:-1]
        feats = rows[["F", "D", "Cr", "actual_cycle", "wear_depth"]].to_numpy(dtype=np.float32)
        targets = np.log(np.maximum(rows["stress"].to_numpy(dtype=np.float32), EPS)).reshape(-1, 1)
        wear_vals = table["wear_depth"].to_numpy(dtype=np.float32)
        next_delta = np.zeros(len(rows), dtype=np.float32)
        for i in range(len(rows)):
            next_delta[i] = max(wear_vals[i + 1] - wear_vals[i], 0.0) if i + 1 < len(wear_vals) else 0.0
        step = median_positive_diff(table["actual_cycle"].to_numpy(dtype=float))
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
            all_raw.append(seq)
            all_steps.append(np.array([step], dtype=np.float32))
            all_next_delta.append(np.array([next_delta[i]], dtype=np.float32))
    return (np.asarray(all_seqs, dtype=np.float32), np.asarray(all_targets, dtype=np.float32),
            np.asarray(all_raw, dtype=np.float32), np.asarray(all_steps, dtype=np.float32),
            np.asarray(all_next_delta, dtype=np.float32))


def _split_train_val(*arrays, val_ratio=VAL_RATIO, seed=SEED):
    n = len(arrays[0])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_n = max(1, int(n * val_ratio))
    val_idx, train_idx = perm[:val_n], perm[val_n:]
    splits = []
    for a in arrays:
        if isinstance(a, torch.Tensor):
            splits.append((a[train_idx], a[val_idx]))
        else:
            splits.append((a[train_idx], a[val_idx]))
    return splits


def train_v1_model(model, train_x_t, train_y_t, raw_x_np, x_scaler, y_scaler, epochs=TRAIN_EPOCHS):
    model.to(DEVICE)
    (tr_x, va_x), (tr_y, va_y), (tr_raw, va_raw) = _split_train_val(train_x_t, train_y_t, raw_x_np)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    mse = nn.MSELoss()
    n = len(tr_x)
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n, device=DEVICE)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            pred = model(tr_x[idx])
            data_loss = mse(pred, tr_y[idx])

            raw_batch = tr_raw[idx.cpu().numpy()].copy()
            wear_shift = raw_batch.copy(); wear_shift[:, -1] += WEAR_DELTA_MM
            load_shift = raw_batch.copy(); load_shift[:, 0] *= (1.0 + LOAD_DELTA_RATIO)
            clearance_shift = raw_batch.copy(); clearance_shift[:, 2] += CLEARANCE_DELTA_MM

            wear_pen = torch.mean(torch.relu(model(torch.tensor(x_scaler.transform_np(wear_shift), dtype=torch.float32, device=DEVICE)) - pred) ** 2)
            load_pen = torch.mean(torch.relu(pred - model(torch.tensor(x_scaler.transform_np(load_shift), dtype=torch.float32, device=DEVICE))) ** 2)
            clearance_pen = torch.mean(torch.relu(pred - model(torch.tensor(x_scaler.transform_np(clearance_shift), dtype=torch.float32, device=DEVICE))) ** 2)

            loss = data_loss + MONO_LAMBDA_WEAR * wear_pen + MONO_LAMBDA_LOAD * load_pen + MONO_LAMBDA_CLEARANCE * clearance_pen
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(va_x)
            val_loss = float(mse(val_pred, va_y).item())

        if val_loss < best_loss - 1e-9:
            best_loss = val_loss; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1

        avg = np.mean(epoch_losses)
        if epoch % 500 == 0 or epoch == epochs:
            with torch.no_grad():
                p = y_scaler.inverse_torch(model(train_x_t)); t = y_scaler.inverse_torch(train_y_t)
                mape = torch.mean(torch.abs(torch.exp(p) - torch.exp(t)) / (torch.exp(t) + EPS)).item()
            print(f"  epoch {epoch:4d} | train_loss={avg:.6e} | val_loss={val_loss:.6e} | mape={mape:.4%} | no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state: model.load_state_dict(best_state)
    return model


def train_v2_single_model(model, train_x_t, train_y_t, raw_x_np, x_scaler, y_scaler,
                           actual_step_np, next_delta_np, real_k, epochs=TRAIN_EPOCHS,
                           wear_consistency=False, shape_loss=False):
    model.to(DEVICE)
    (tr_x, va_x), (tr_y, va_y), (tr_raw, va_raw), (tr_step, va_step), (tr_delta, va_delta) = \
        _split_train_val(train_x_t, train_y_t, raw_x_np, actual_step_np, next_delta_np)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    mse = nn.MSELoss()
    n = len(tr_x)
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n, device=DEVICE)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            pred = model(tr_x[idx])
            data_loss = mse(pred, tr_y[idx])

            raw_batch = tr_raw[idx.cpu().numpy()].copy()
            wear_shift = raw_batch.copy(); wear_shift[:, -1] += WEAR_DELTA_MM
            load_shift = raw_batch.copy(); load_shift[:, 0] *= (1.0 + LOAD_DELTA_RATIO)
            clearance_shift = raw_batch.copy(); clearance_shift[:, 2] += CLEARANCE_DELTA_MM

            wear_pen = torch.mean(torch.relu(model(torch.tensor(x_scaler.transform_np(wear_shift), dtype=torch.float32, device=DEVICE)) - pred) ** 2)
            load_pen = torch.mean(torch.relu(pred - model(torch.tensor(x_scaler.transform_np(load_shift), dtype=torch.float32, device=DEVICE))) ** 2)
            clearance_pen = torch.mean(torch.relu(pred - model(torch.tensor(x_scaler.transform_np(clearance_shift), dtype=torch.float32, device=DEVICE))) ** 2)

            loss = data_loss + MONO_LAMBDA_WEAR * wear_pen + MONO_LAMBDA_LOAD * load_pen + MONO_LAMBDA_CLEARANCE * clearance_pen

            if wear_consistency:
                pred_log = y_scaler.inverse_torch(pred)
                pred_stress = torch.exp(pred_log).reshape(-1)
                D_t = torch.tensor(raw_batch[:, -1, 1], dtype=torch.float32, device=DEVICE)
                step_t = torch.tensor(tr_step[idx.cpu().numpy()].reshape(-1), dtype=torch.float32, device=DEVICE)
                delta_s = step_t * math.pi * D_t / 6.0
                pred_delta = pred_stress * delta_s * real_k
                true_delta = torch.tensor(tr_delta[idx.cpu().numpy()].reshape(-1), dtype=torch.float32, device=DEVICE)
                wc_loss = torch.mean((pred_delta - true_delta) ** 2)
                loss = loss + WEAR_CONSISTENCY_LAMBDA * wc_loss

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(mse(model(va_x), va_y).item())

        if val_loss < best_loss - 1e-9:
            best_loss = val_loss; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1

        avg = np.mean(epoch_losses)
        if epoch % 500 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                p = y_scaler.inverse_torch(model(train_x_t)); t = y_scaler.inverse_torch(train_y_t)
                mape = torch.mean(torch.abs(torch.exp(p) - torch.exp(t)) / (torch.exp(t) + EPS)).item()
            print(f"  epoch {epoch:4d} | train_loss={avg:.6e} | val_loss={val_loss:.6e} | mape={mape:.4%} | no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state: model.load_state_dict(best_state)
    return model


def train_dual_branch_model(model, static_x_t, dynamic_x_t, train_y_t, raw_x_np,
                             static_scaler, dynamic_scaler, y_scaler,
                             actual_step_np, next_delta_np, real_k, epochs=TRAIN_EPOCHS,
                             wear_consistency=True, shape_loss=True):
    model.to(DEVICE)
    (tr_static, va_static), (tr_dynamic, va_dynamic), (tr_y, va_y), (tr_raw, va_raw), \
        (tr_step, va_step), (tr_delta, va_delta) = \
        _split_train_val(static_x_t, dynamic_x_t, train_y_t, raw_x_np, actual_step_np, next_delta_np)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    mse = nn.MSELoss()
    n = len(tr_y)
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n, device=DEVICE)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            pred = model(tr_dynamic[idx], tr_static[idx])
            data_loss = mse(pred, tr_y[idx])

            raw_batch = tr_raw[idx.cpu().numpy()].copy()
            wear_shift = raw_batch.copy(); wear_shift[:, -1] += WEAR_DELTA_MM
            load_shift = raw_batch.copy(); load_shift[:, 0] *= (1.0 + LOAD_DELTA_RATIO)
            clearance_shift = raw_batch.copy(); clearance_shift[:, 2] += CLEARANCE_DELTA_MM

            def _dual_inputs(shifted_raw):
                s_static = static_scaler.transform_np(compute_static_features(shifted_raw))
                s_dynamic = dynamic_scaler.transform_np(compute_dynamic_features(shifted_raw))
                return (torch.tensor(s_dynamic, dtype=torch.float32, device=DEVICE),
                        torch.tensor(s_static, dtype=torch.float32, device=DEVICE))

            wear_pen = torch.mean(torch.relu(model(*_dual_inputs(wear_shift)) - pred) ** 2)
            load_pen = torch.mean(torch.relu(pred - model(*_dual_inputs(load_shift))) ** 2)
            clearance_pen = torch.mean(torch.relu(pred - model(*_dual_inputs(clearance_shift))) ** 2)

            loss = data_loss + MONO_LAMBDA_WEAR * wear_pen + MONO_LAMBDA_LOAD * load_pen + MONO_LAMBDA_CLEARANCE * clearance_pen

            if wear_consistency:
                pred_log = y_scaler.inverse_torch(pred)
                pred_stress = torch.exp(pred_log).reshape(-1)
                D_t = torch.tensor(raw_batch[:, -1, 1], dtype=torch.float32, device=DEVICE)
                step_t = torch.tensor(tr_step[idx.cpu().numpy()].reshape(-1), dtype=torch.float32, device=DEVICE)
                delta_s = step_t * math.pi * D_t / 6.0
                pred_delta = pred_stress * delta_s * real_k
                true_delta = torch.tensor(tr_delta[idx.cpu().numpy()].reshape(-1), dtype=torch.float32, device=DEVICE)
                wc_loss = torch.mean((pred_delta - true_delta) ** 2)
                loss = loss + WEAR_CONSISTENCY_LAMBDA * wc_loss

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(mse(model(va_dynamic, va_static), va_y).item())

        if val_loss < best_loss - 1e-9:
            best_loss = val_loss; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1

        avg = np.mean(epoch_losses)
        if epoch % 500 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                p = y_scaler.inverse_torch(model(dynamic_x_t, static_x_t)); t = y_scaler.inverse_torch(train_y_t)
                mape = torch.mean(torch.abs(torch.exp(p) - torch.exp(t)) / (torch.exp(t) + EPS)).item()
            print(f"  epoch {epoch:4d} | train_loss={avg:.6e} | val_loss={val_loss:.6e} | mape={mape:.4%} | no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state: model.load_state_dict(best_state)
    return model


def predict_v1(model, x_scaler, y_scaler, F, D, Cr, actual_cycle, wear_depth, seq_len=6):
    x = np.array([[F, D, Cr, actual_cycle, wear_depth]], dtype=np.float32)
    x_seq = np.repeat(x, seq_len, axis=0).reshape(1, seq_len, 5)
    x_t = torch.tensor(x_scaler.transform_np(x_seq), dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        return float(torch.exp(y_scaler.inverse_torch(model(x_t))).item())


def predict_v2_single(model, x_scaler, y_scaler, F, D, Cr, actual_cycle, wear_depth, seq_len=10):
    raw = np.array([[F, D, Cr, actual_cycle, wear_depth]], dtype=np.float32)
    raw_seq = np.repeat(raw, seq_len, axis=0).reshape(1, seq_len, 5)
    step = np.array([[0.0]], dtype=np.float32)
    feat = build_single_branch_features(raw_seq, step, use_cycle_step=False)
    x_t = torch.tensor(x_scaler.transform_np(feat), dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        return float(torch.exp(y_scaler.inverse_torch(model(x_t))).item())


def predict_dual(model, static_scaler, dynamic_scaler, y_scaler, F, D, Cr, actual_cycle, wear_depth, seq_len=10):
    raw = np.array([[F, D, Cr, actual_cycle, wear_depth]], dtype=np.float32)
    raw_seq = np.repeat(raw, seq_len, axis=0).reshape(1, seq_len, 5)
    step_val = 0.0
    static_feat = compute_static_features(raw_seq, use_cycle_step=False, actual_step_val=step_val)
    dynamic_feat = compute_dynamic_features(raw_seq)
    s_t = torch.tensor(static_scaler.transform_np(static_feat), dtype=torch.float32, device=DEVICE)
    d_t = torch.tensor(dynamic_scaler.transform_np(dynamic_feat), dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        return float(torch.exp(y_scaler.inverse_torch(model(d_t, s_t))).item())


def rollout(predict_fn, case_df, threshold_um, real_k, true_life):
    first = case_df.iloc[0]
    F, D, Cr = float(first["F"]), float(first["D"]), float(first["Cr"])
    actual_step = median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float))
    threshold_mm = threshold_um / 1000.0
    actual_cycle, wear_depth, predicted_life = 0.0, 0.0, true_life
    rows = []
    internal_limit = max(true_life * 1.4, float(case_df["actual_cycle"].max()) * 1.2, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + 400

    for _ in range(max_steps):
        pred_stress = predict_fn(F, D, Cr, actual_cycle, wear_depth)
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
        actual_cycle, wear_depth = next_cycle, next_wear

    return pd.DataFrame(rows), float(predicted_life)


def save_step_comparison_chart(out_path, all_results, test_runs, case_tables, threshold_um, summary_df):
    n_tests = len(test_runs)
    fig, axes = plt.subplots(1, n_tests, figsize=(7 * n_tests, 5.6))
    if n_tests == 1: axes = [axes]

    colors = ["#111827", "#2563eb", "#16a34a", "#d97706", "#dc2626"]
    for ax, test_file in zip(axes, test_runs):
        test_row = summary_df[summary_df["file_name"] == test_file].iloc[0]
        true_life = float(test_row["actual_life"])
        true_df = threshold_ground_truth(case_tables[test_file], threshold_um, test_row)
        ax.plot(true_df["actual_cycle"], true_df["wear_depth_um"], color="#111827", linewidth=2.3, label="FE truth")

        for i, (step_name, res) in enumerate(all_results.items()):
            if test_file in res:
                rollout_df = res[test_file]["rollout"]
                pred_life = res[test_file]["predicted_life"]
                c = colors[(i + 1) % len(colors)]
                ax.plot(rollout_df["actual_cycle"], rollout_df["pred_wear_depth_um"],
                        color=c, linewidth=1.8, label=f"{step_name} ({pred_life:.0f})")

        ax.axhline(threshold_um, color="#dc2626", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Actual Cycle"); ax.set_ylabel("Wear Depth (um)")
        ax.set_title(f"{test_file} | true={true_life:.0f}")
        ax.grid(alpha=0.28); ax.legend(fontsize=7)

    fig.suptitle("Incremental V2 Upgrade: Wear Curve Comparison", fontsize=14)
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)


def save_step_bar_chart(out_path, all_results, test_runs, summary_df):
    step_names = list(all_results.keys())
    n_steps = len(step_names)
    n_tests = len(test_runs)
    x = np.arange(n_tests)
    width = 0.8 / (n_steps + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    true_lives = [float(summary_df[summary_df["file_name"] == f].iloc[0]["actual_life"]) for f in test_runs]
    ax.bar(x - width * n_steps / 2, true_lives, width, color="#111827", label="FE truth")

    colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]
    for j, name in enumerate(step_names):
        pred_lives = [all_results[name][f]["predicted_life"] if f in all_results[name] else 0 for f in test_runs]
        ax.bar(x - width * n_steps / 2 + width * (j + 1), pred_lives, width,
               color=colors[j % len(colors)], label=name)

    ax.set_xticks(x); ax.set_xticklabels(test_runs)
    ax.set_ylabel("Predicted Life (cycles)")
    ax.set_title("Incremental V2 Upgrade: Life Prediction")
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)


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

    out_dir = comparison_dir() / "v2_incremental_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    def evaluate_step(step_name, predict_fn):
        model_results = {}
        for test_file in TEST_RUNS:
            test_table = case_tables[test_file]
            test_row = summary_df[summary_df["file_name"] == test_file].iloc[0]
            true_life = float(test_row["actual_life"])
            rollout_df, predicted_life = rollout(predict_fn, case_tables[test_file], threshold_um, real_k, true_life)
            abs_err = abs(predicted_life - true_life)
            rel_err = abs_err / max(true_life, EPS)
            model_results[test_file] = {"predicted_life": predicted_life, "true_life": true_life,
                                         "life_abs_error": abs_err, "life_rel_error": rel_err, "rollout": rollout_df}
            print(f"  {test_file}: true={true_life:.0f} | pred={predicted_life:.0f} | rel_err={rel_err:.2%}")
            summary_rows.append({"step": step_name, "test_run": test_file, "true_life": true_life,
                                  "predicted_life": predicted_life, "life_abs_error": abs_err, "life_rel_error": rel_err})
        all_results[step_name] = model_results

    # ── Step 1: V1 Baseline (seq=6, 5 features, V1 mono penalties) ──
    print("\n" + "=" * 60)
    print("Step 1: V1 Baseline Transformer (seq=6, 5 features)")
    print("=" * 60)
    set_seed(SEED)
    v1_seq_len = 6
    train_x_np, train_y_np, raw_x_np, step_np, delta_np = build_v1_sequences(train_tables, v1_seq_len)
    flat_x = train_x_np.reshape(-1, 5)
    x_scaler = Standardizer().fit(flat_x); x_scaler.to_torch(DEVICE)
    y_scaler = Standardizer().fit(train_y_np); y_scaler.to_torch(DEVICE)
    train_x_t = torch.tensor(x_scaler.transform_np(train_x_np), dtype=torch.float32, device=DEVICE)
    train_y_t = torch.tensor(y_scaler.transform_np(train_y_np), dtype=torch.float32, device=DEVICE)

    v1_model = TransformerV1(in_dim=5, seq_len=v1_seq_len)
    v1_model = train_v1_model(v1_model, train_x_t, train_y_t, raw_x_np, x_scaler, y_scaler)

    def v1_predict(F, D, Cr, c, w):
        return predict_v1(v1_model, x_scaler, y_scaler, F, D, Cr, c, w, v1_seq_len)
    evaluate_step("1-V1-baseline", v1_predict)

    # ── Step 2: V2 Features (seq=10, 9 features: static+dynamic) ──
    print("\n" + "=" * 60)
    print("Step 2: V2 Features (seq=10, static+dynamic=9 features)")
    print("=" * 60)
    set_seed(SEED)
    v2_seq_len = 10
    train_x2_np, train_y2_np, raw_x2_np, step2_np, delta2_np = build_v1_sequences(train_tables, v2_seq_len)
    feat2_np = build_single_branch_features(raw_x2_np, step2_np, use_cycle_step=False)
    feat2_dim = feat2_np.shape[-1]
    x2_scaler = Standardizer().fit(feat2_np.reshape(-1, feat2_dim)); x2_scaler.to_torch(DEVICE)
    y2_scaler = Standardizer().fit(train_y2_np); y2_scaler.to_torch(DEVICE)
    feat2_t = torch.tensor(x2_scaler.transform_np(feat2_np), dtype=torch.float32, device=DEVICE)
    train_y2_t = torch.tensor(y2_scaler.transform_np(train_y2_np), dtype=torch.float32, device=DEVICE)

    v2s_model = TransformerV2Single(in_dim=feat2_dim, seq_len=v2_seq_len)
    v2s_model = train_v2_single_model(v2s_model, feat2_t, train_y2_t, raw_x2_np, x2_scaler, y2_scaler,
                                       step2_np, delta2_np, real_k, wear_consistency=False, shape_loss=False)

    def v2s_predict(F, D, Cr, c, w):
        return predict_v2_single(v2s_model, x2_scaler, y2_scaler, F, D, Cr, c, w, v2_seq_len)
    evaluate_step("2-V2-features", v2s_predict)

    # ── Step 3: + Wear Consistency Loss ──
    print("\n" + "=" * 60)
    print("Step 3: + Wear Consistency Loss (lambda=0.25)")
    print("=" * 60)
    set_seed(SEED)
    v2wc_model = TransformerV2Single(in_dim=feat2_dim, seq_len=v2_seq_len)
    v2wc_model = train_v2_single_model(v2wc_model, feat2_t, train_y2_t, raw_x2_np, x2_scaler, y2_scaler,
                                        step2_np, delta2_np, real_k, wear_consistency=True, shape_loss=False)

    def v2wc_predict(F, D, Cr, c, w):
        return predict_v2_single(v2wc_model, x2_scaler, y2_scaler, F, D, Cr, c, w, v2_seq_len)
    evaluate_step("3-V2+wear-consistency", v2wc_predict)

    # ── Step 4: + Dual-Branch Architecture ──
    print("\n" + "=" * 60)
    print("Step 4: + Dual-Branch Architecture (static MLP + dynamic Transformer)")
    print("=" * 60)
    set_seed(SEED)
    static_np = compute_static_features(raw_x2_np, use_cycle_step=False, actual_step_val=0.0)
    dynamic_np = compute_dynamic_features(raw_x2_np)
    static_dim = static_np.shape[-1]
    dynamic_dim = dynamic_np.shape[-1]
    static_scaler = Standardizer().fit(static_np); static_scaler.to_torch(DEVICE)
    dynamic_scaler = Standardizer().fit(dynamic_np.reshape(-1, dynamic_dim)); dynamic_scaler.to_torch(DEVICE)
    static_t = torch.tensor(static_scaler.transform_np(static_np), dtype=torch.float32, device=DEVICE)
    dynamic_t = torch.tensor(dynamic_scaler.transform_np(dynamic_np), dtype=torch.float32, device=DEVICE)

    dual_model = StaticDynamicTransformer(static_dim=static_dim, dynamic_dim=dynamic_dim, seq_len=v2_seq_len)
    dual_model = train_dual_branch_model(dual_model, static_t, dynamic_t, train_y2_t, raw_x2_np,
                                          static_scaler, dynamic_scaler, y2_scaler,
                                          step2_np, delta2_np, real_k, wear_consistency=True, shape_loss=True)

    def dual_predict(F, D, Cr, c, w):
        return predict_dual(dual_model, static_scaler, dynamic_scaler, y2_scaler, F, D, Cr, c, w, v2_seq_len)
    evaluate_step("4-V2+dual-branch", dual_predict)

    # ── Save results ──
    summary_df_out = pd.DataFrame(summary_rows)
    summary_df_out.to_csv(out_dir / "incremental_validation_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("INCREMENTAL VALIDATION SUMMARY")
    print("=" * 60)
    print(summary_df_out.to_string(index=False))

    pivot = summary_df_out.pivot(index="step", columns="test_run", values="life_rel_error")
    print("\nRelative Error by Step x Test Run:")
    print(pivot.to_string())

    save_step_comparison_chart(out_dir / "incremental_wear_curves.png", all_results, TEST_RUNS, case_tables, threshold_um, summary_df)
    save_step_bar_chart(out_dir / "incremental_life_bar.png", all_results, TEST_RUNS, summary_df)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
