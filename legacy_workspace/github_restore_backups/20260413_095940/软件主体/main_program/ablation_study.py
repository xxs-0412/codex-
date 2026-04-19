from __future__ import annotations

import copy
import math
import sys
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
FINALIST_SEEDS = [20260408, 20260418, 20260428]
DEVICE, DEVICE_LABEL = resolve_training_device()

TEST_RUNS = ["试验11.csv", "试验7.csv"]
VAL_RUN = "试验12.csv"
TRAIN_EPOCHS = 2000
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE = 200

MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002
WEAR_CONSISTENCY_LAMBDA = 0.25


class TransformerSmall(nn.Module):
    def __init__(self, in_dim=5, seq_len=6, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.proj(x) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class TransformerLarge(nn.Module):
    def __init__(self, in_dim=9, seq_len=10, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.proj(x) + self.pos[:, :x.size(1), :]
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class FNNNet(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 3: x = x[:, -1, :]
        return self.net(x)


class CNNet(nn.Module):
    def __init__(self, in_dim=5, seq_len=6):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.head(self.conv(x.permute(0, 2, 1)).squeeze(-1))


class GRUNet(nn.Module):
    def __init__(self, in_dim=5, hidden=48):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        _, h = self.gru(x)
        return self.head(h.squeeze(0))


class LSTMNet(nn.Module):
    def __init__(self, in_dim=5, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        return self.head(h.squeeze(0))


def compute_v2_features(raw_seq, actual_step):
    F = raw_seq[:, :, 0]
    D = np.maximum(raw_seq[:, :, 1], EPS)
    Cr = raw_seq[:, :, 2]
    cycles = raw_seq[:, :, 3]
    wear_depth = raw_seq[:, :, 4]
    delta_cycle = np.zeros_like(cycles, dtype=np.float32)
    delta_wear = np.zeros_like(wear_depth, dtype=np.float32)
    delta_cycle[:, 1:] = np.maximum(cycles[:, 1:] - cycles[:, :-1], 0.0)
    delta_wear[:, 1:] = np.maximum(wear_depth[:, 1:] - wear_depth[:, :-1], 0.0)
    return np.stack([F, D, Cr, F / np.maximum(D * D, EPS), Cr / D,
                     np.log1p(np.maximum(cycles, 0.0)), wear_depth, delta_cycle, delta_wear], axis=-1).astype(np.float32)


def build_dataset(case_tables, seq_len, use_v2_features=False):
    all_seqs, all_targets, all_raw, all_steps, all_next_delta = [], [], [], [], []
    for table in case_tables.values():
        rows = table.iloc[:-1]
        v1_feats = rows[["F", "D", "Cr", "actual_cycle", "wear_depth"]].to_numpy(dtype=np.float32)
        targets = np.log(np.maximum(rows["stress"].to_numpy(dtype=np.float32), EPS)).reshape(-1, 1)
        wear_vals = table["wear_depth"].to_numpy(dtype=np.float32)
        next_delta = np.zeros(len(rows), dtype=np.float32)
        for i in range(len(rows)):
            next_delta[i] = max(wear_vals[i + 1] - wear_vals[i], 0.0) if i + 1 < len(wear_vals) else 0.0
        step = median_positive_diff(table["actual_cycle"].to_numpy(dtype=float))
        n = len(v1_feats)
        for i in range(n):
            start = max(0, i - seq_len + 1)
            seq = v1_feats[start:i + 1]
            pad_count = seq_len - len(seq)
            if pad_count > 0:
                pad = np.repeat(seq[:1], pad_count, axis=0)
                seq = np.vstack([pad, seq])
            all_seqs.append(seq)
            all_targets.append(targets[i])
            all_raw.append(seq)
            all_steps.append(np.array([step], dtype=np.float32))
            all_next_delta.append(np.array([next_delta[i]], dtype=np.float32))

    raw_seqs = np.asarray(all_seqs, dtype=np.float32)
    targets = np.asarray(all_targets, dtype=np.float32)
    raw_arr = np.asarray(all_raw, dtype=np.float32)
    steps = np.asarray(all_steps, dtype=np.float32)
    deltas = np.asarray(all_next_delta, dtype=np.float32)

    if use_v2_features:
        steps_expanded = np.repeat(steps, 1, axis=0)
        feats = compute_v2_features(raw_seqs, steps_expanded)
    else:
        feats = raw_seqs

    return feats, targets, raw_arr, steps, deltas


def split_by_run(case_tables, seq_len, use_v2_features, val_run, test_runs):
    train_tables = {f: t for f, t in case_tables.items() if f not in test_runs + [val_run]}
    val_tables = {f: t for f, t in case_tables.items() if f == val_run}

    tr_feats, tr_targets, tr_raw, tr_steps, tr_deltas = build_dataset(train_tables, seq_len, use_v2_features)
    va_feats, va_targets, va_raw, va_steps, va_deltas = build_dataset(val_tables, seq_len, use_v2_features)

    return (tr_feats, tr_targets, tr_raw, tr_steps, tr_deltas,
            va_feats, va_targets, va_raw, va_steps, va_deltas)


def _mono_penalties_single(model, raw_batch, x_scaler, pred, feat_fn=None):
    wear_shift = raw_batch.copy(); wear_shift[:, -1] += WEAR_DELTA_MM
    load_shift = raw_batch.copy(); load_shift[:, 0] *= (1.0 + LOAD_DELTA_RATIO)
    clearance_shift = raw_batch.copy(); clearance_shift[:, 2] += CLEARANCE_DELTA_MM

    if feat_fn is not None:
        def _call(s):
            step_zero = np.zeros((len(s), 1), dtype=np.float32)
            feat = feat_fn(s, step_zero)
            return model(torch.tensor(x_scaler.transform_np(feat), dtype=torch.float32, device=DEVICE))
    else:
        def _call(s):
            return model(torch.tensor(x_scaler.transform_np(s), dtype=torch.float32, device=DEVICE))

    wear_pen = torch.mean(torch.relu(_call(wear_shift) - pred) ** 2)
    load_pen = torch.mean(torch.relu(pred - _call(load_shift)) ** 2)
    clearance_pen = torch.mean(torch.relu(pred - _call(clearance_shift)) ** 2)
    return wear_pen, load_pen, clearance_pen


def _wc_loss(pred, y_scaler, raw_batch, step_batch, delta_batch, real_k):
    pred_log = y_scaler.inverse_torch(pred)
    pred_stress = torch.exp(pred_log).reshape(-1)
    D_t = torch.tensor(raw_batch[:, -1, 1], dtype=torch.float32, device=DEVICE)
    step_t = torch.tensor(step_batch.reshape(-1), dtype=torch.float32, device=DEVICE)
    delta_s = step_t * math.pi * D_t / 6.0
    pred_delta = pred_stress * delta_s * real_k
    true_delta = torch.tensor(delta_batch.reshape(-1), dtype=torch.float32, device=DEVICE)
    return torch.mean((pred_delta - true_delta) ** 2)


def train_model(model, tr_feats, tr_targets, tr_raw, tr_steps, tr_deltas,
                va_feats, va_targets, x_scaler, y_scaler, real_k,
                use_wc=False, feat_fn=None, seed=SEED):
    set_seed(seed)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    mse = nn.MSELoss()

    tr_x = torch.tensor(x_scaler.transform_np(tr_feats), dtype=torch.float32, device=DEVICE)
    tr_y = torch.tensor(y_scaler.transform_np(tr_targets), dtype=torch.float32, device=DEVICE)
    va_x = torch.tensor(x_scaler.transform_np(va_feats), dtype=torch.float32, device=DEVICE)
    va_y = torch.tensor(y_scaler.transform_np(va_targets), dtype=torch.float32, device=DEVICE)

    n = len(tr_x)
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        indices = torch.randperm(n, device=DEVICE)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            pred = model(tr_x[idx])
            data_loss = mse(pred, tr_y[idx])

            raw_batch = tr_raw[idx.cpu().numpy()]
            wear_pen, load_pen, clearance_pen = _mono_penalties_single(
                model, raw_batch, x_scaler, pred, feat_fn)

            loss = data_loss + MONO_LAMBDA_WEAR * wear_pen + MONO_LAMBDA_LOAD * load_pen + MONO_LAMBDA_CLEARANCE * clearance_pen

            if use_wc:
                loss = loss + WEAR_CONSISTENCY_LAMBDA * _wc_loss(
                    pred, y_scaler, raw_batch, tr_steps[idx.cpu().numpy()], tr_deltas[idx.cpu().numpy()], real_k)

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

        if epoch % 500 == 0 or epoch == TRAIN_EPOCHS:
            with torch.no_grad():
                p = y_scaler.inverse_torch(model(tr_x)); t = y_scaler.inverse_torch(tr_y)
                mape = torch.mean(torch.abs(torch.exp(p) - torch.exp(t)) / (torch.exp(t) + EPS)).item()
            print(f"    epoch {epoch:4d} | val_loss={val_loss:.6e} | mape={mape:.4%} | no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_single(model, x_scaler, y_scaler, F, D, Cr, c, w, seq_len, feat_fn=None):
    raw = np.array([[F, D, Cr, c, w]], dtype=np.float32)
    raw_seq = np.repeat(raw, seq_len, axis=0).reshape(1, seq_len, 5)
    if feat_fn is not None:
        step_zero = np.zeros((1, 1), dtype=np.float32)
        x_in = feat_fn(raw_seq, step_zero)
    else:
        x_in = raw_seq
    x_t = torch.tensor(x_scaler.transform_np(x_in), dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        return float(torch.exp(y_scaler.inverse_torch(model(x_t))).item())


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


def evaluate_on_tests(predict_fn, case_tables, summary_df, threshold_um, real_k):
    results = {}
    for test_file in TEST_RUNS:
        test_row = summary_df[summary_df["file_name"] == test_file].iloc[0]
        true_life = float(test_row["actual_life"])
        rollout_df, predicted_life = rollout(predict_fn, case_tables[test_file], threshold_um, real_k, true_life)
        abs_err = abs(predicted_life - true_life)
        rel_err = abs_err / max(true_life, EPS)
        results[test_file] = {"predicted_life": predicted_life, "true_life": true_life,
                               "life_abs_error": abs_err, "life_rel_error": rel_err, "rollout": rollout_df}
        print(f"    {test_file}: true={true_life:.0f} | pred={predicted_life:.0f} | rel_err={rel_err:.2%}")
    return results


def avg_rel_error(results):
    return np.mean([r["life_rel_error"] for r in results.values()])


def median_abs_error(results):
    return np.median([r["life_abs_error"] for r in results.values()])


def save_charts(out_dir, all_results, test_runs, case_tables, threshold_um, summary_df, prefix):
    n_tests = len(test_runs)
    fig, axes = plt.subplots(1, n_tests, figsize=(7 * n_tests, 5.6))
    if n_tests == 1: axes = [axes]
    colors = ["#111827", "#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#ec4899", "#059669", "#7c2d12", "#4338ca"]

    for ax, test_file in zip(axes, test_runs):
        test_row = summary_df[summary_df["file_name"] == test_file].iloc[0]
        true_life = float(test_row["actual_life"])
        true_df = threshold_ground_truth(case_tables[test_file], threshold_um, test_row)
        ax.plot(true_df["actual_cycle"], true_df["wear_depth_um"], color="#111827", linewidth=2.3, label="FE truth")
        for i, (name, res) in enumerate(all_results.items()):
            if test_file in res:
                r = res[test_file]
                c = colors[(i + 1) % len(colors)]
                ax.plot(r["rollout"]["actual_cycle"], r["rollout"]["pred_wear_depth_um"],
                        color=c, linewidth=1.8, label=f"{name} ({r['predicted_life']:.0f})")
        ax.axhline(threshold_um, color="#dc2626", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Actual Cycle"); ax.set_ylabel("Wear Depth (um)")
        ax.set_title(f"{test_file} | true={true_life:.0f}")
        ax.grid(alpha=0.28); ax.legend(fontsize=7)

    fig.suptitle(f"{prefix}: Wear Curve Comparison", fontsize=14)
    fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_wear_curves.png", dpi=300); plt.close(fig)

    names = list(all_results.keys())
    n_names = len(names)
    x = np.arange(n_tests)
    width = 0.8 / (n_names + 1)
    fig, ax = plt.subplots(figsize=(max(12, n_names), 6))
    true_lives = [float(summary_df[summary_df["file_name"] == f].iloc[0]["actual_life"]) for f in test_runs]
    ax.bar(x - width * n_names / 2, true_lives, width, color="#111827", label="FE truth")
    for j, name in enumerate(names):
        pred_lives = [all_results[name][f]["predicted_life"] if f in all_results[name] else 0 for f in test_runs]
        ax.bar(x - width * n_names / 2 + width * (j + 1), pred_lives, width,
               color=colors[(j + 1) % len(colors)], label=name)
    ax.set_xticks(x); ax.set_xticklabels(test_runs)
    ax.set_ylabel("Predicted Life (cycles)")
    ax.set_title(f"{prefix}: Life Prediction")
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_life_bar.png", dpi=300); plt.close(fig)


def main():
    config = load_processing_config()
    threshold_um = float(config["wear_threshold_um"])
    real_k = float(config["real_wear_coeff_mpa_inv"])
    summary_df, case_tables = load_cases()

    out_dir = comparison_dir() / "ablation_single_branch"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {DEVICE_LABEL}")
    print(f"test: {TEST_RUNS} | val: {VAL_RUN}")
    print(f"train: {len(case_tables) - len(TEST_RUNS) - 1} runs")

    # ═══════════════════════════════════════════════════════
    # PHASE 1: seq_len scan
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: seq_len scan (V1/V2 × seq_len=6/10/15)")
    print("=" * 70)

    phase1_results = {}
    phase1_rows = []

    for use_v2 in [False, True]:
        feat_label = "V2" if use_v2 else "V1"
        for seq_len in [6, 10, 15]:
            exp_name = f"{feat_label}-s{seq_len}"
            print(f"\n  --- {exp_name} ---")
            set_seed(SEED)

            tr_f, tr_y, tr_raw, tr_step, tr_delta, \
                va_f, va_y, va_raw, va_step, va_delta = \
                split_by_run(case_tables, seq_len, use_v2, VAL_RUN, TEST_RUNS)

            feat_dim = tr_f.shape[-1]
            x_scaler = Standardizer().fit(tr_f.reshape(-1, feat_dim)); x_scaler.to_torch(DEVICE)
            y_scaler = Standardizer().fit(tr_y); y_scaler.to_torch(DEVICE)

            model = TransformerSmall(in_dim=feat_dim, seq_len=seq_len)
            feat_fn = (lambda r, s: compute_v2_features(r, s)) if use_v2 else None
            model = train_model(model, tr_f, tr_y, tr_raw, tr_step, tr_delta,
                                va_f, va_y, x_scaler, y_scaler, real_k,
                                use_wc=False, feat_fn=feat_fn)

            def _make_pred(m, xs, ys, sl, ff):
                def fn(F, D, Cr, c, w):
                    return predict_single(m, xs, ys, F, D, Cr, c, w, sl, feat_fn=ff)
                return fn

            res = evaluate_on_tests(_make_pred(model, x_scaler, y_scaler, seq_len, feat_fn),
                                     case_tables, summary_df, threshold_um, real_k)
            phase1_results[exp_name] = res
            avg = avg_rel_error(res)
            med = median_abs_error(res)
            print(f"    => avg_rel_err={avg:.2%}, median_abs_err={med:.0f}")
            for tf in TEST_RUNS:
                phase1_rows.append({"exp": exp_name, "test_run": tf, "use_v2": use_v2,
                                     "seq_len": seq_len, "life_rel_error": res[tf]["life_rel_error"],
                                     "life_abs_error": res[tf]["life_abs_error"]})

    phase1_df = pd.DataFrame(phase1_rows)
    phase1_df.to_csv(out_dir / "seq_len_scan_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "-" * 70)
    print("PHASE 1 SUMMARY:")
    print("-" * 70)
    for exp_name, res in phase1_results.items():
        avg = avg_rel_error(res)
        med = median_abs_error(res)
        print(f"  {exp_name:12s}: avg_rel={avg:.2%}, median_abs={med:.0f}")

    best_seq = phase1_df.groupby("seq_len")["life_rel_error"].mean().idxmin()
    print(f"\n  => Best seq_len by avg rel_error: {best_seq}")

    best_v2_seq = phase1_df[phase1_df["use_v2"]].groupby("seq_len")["life_rel_error"].mean().idxmin()
    best_v1_seq = phase1_df[~phase1_df["use_v2"]].groupby("seq_len")["life_rel_error"].mean().idxmin()
    print(f"  => Best V2 seq_len: {best_v2_seq}, Best V1 seq_len: {best_v1_seq}")

    SEQ_LEN_STAR = best_v2_seq
    print(f"  => Using seq_len* = {SEQ_LEN_STAR} for Phase 2")

    save_charts(out_dir, phase1_results, TEST_RUNS, case_tables, threshold_um, summary_df, "phase1_seq_scan")

    # ═══════════════════════════════════════════════════════
    # PHASE 2: Main ablation (3-seed)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"PHASE 2: Main ablation (seq_len*={SEQ_LEN_STAR}, 3-seed)")
    print("=" * 70)

    ablation_configs = [
        ("Full(ABC)", True, True, True),
        ("-A(no V2)", False, True, True),
        ("-B(no Trans upg)", True, False, True),
        ("-C(no WC)", True, True, False),
        ("V1-baseline", False, False, False),
    ]

    phase2_results = {}
    phase2_rows = []

    for exp_name, use_v2, use_large, use_wc in ablation_configs:
        print(f"\n  --- {exp_name} ---")
        seed_results = {}

        for seed in FINALIST_SEEDS:
            print(f"    seed={seed}")
            set_seed(seed)

            sl = SEQ_LEN_STAR if use_v2 else (6 if SEQ_LEN_STAR == 6 else SEQ_LEN_STAR)
            tr_f, tr_y, tr_raw, tr_step, tr_delta, \
                va_f, va_y, va_raw, va_step, va_delta = \
                split_by_run(case_tables, sl, use_v2, VAL_RUN, TEST_RUNS)

            feat_dim = tr_f.shape[-1]
            x_scaler = Standardizer().fit(tr_f.reshape(-1, feat_dim)); x_scaler.to_torch(DEVICE)
            y_scaler = Standardizer().fit(tr_y); y_scaler.to_torch(DEVICE)

            if use_large:
                model = TransformerLarge(in_dim=feat_dim, seq_len=sl)
            else:
                model = TransformerSmall(in_dim=feat_dim, seq_len=sl)

            feat_fn = (lambda r, s: compute_v2_features(r, s)) if use_v2 else None
            model = train_model(model, tr_f, tr_y, tr_raw, tr_step, tr_delta,
                                va_f, va_y, x_scaler, y_scaler, real_k,
                                use_wc=use_wc, feat_fn=feat_fn, seed=seed)

            def _make_pred(m, xs, ys, s, ff):
                def fn(F, D, Cr, c, w):
                    return predict_single(m, xs, ys, F, D, Cr, c, w, s, feat_fn=ff)
                return fn

            res = evaluate_on_tests(_make_pred(model, x_scaler, y_scaler, sl, feat_fn),
                                     case_tables, summary_df, threshold_um, real_k)
            seed_results[seed] = res

            for tf in TEST_RUNS:
                phase2_rows.append({"exp": exp_name, "seed": seed, "test_run": tf,
                                     "use_v2": use_v2, "use_large": use_large, "use_wc": use_wc,
                                     "life_rel_error": res[tf]["life_rel_error"],
                                     "life_abs_error": res[tf]["life_abs_error"]})

        avg_rels = [avg_rel_error(sr) for sr in seed_results.values()]
        med_abss = [median_abs_error(sr) for sr in seed_results.values()]
        print(f"    3-seed avg_rel: {np.mean(avg_rels):.2%} ± {np.std(avg_rels):.2%}")
        print(f"    3-seed median_abs: {np.mean(med_abss):.0f} ± {np.std(med_abss):.0f}")

        phase2_results[exp_name] = seed_results[FINALIST_SEEDS[0]]

    phase2_df = pd.DataFrame(phase2_rows)
    phase2_df.to_csv(out_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "-" * 70)
    print("PHASE 2 SUMMARY (3-seed mean ± std):")
    print("-" * 70)
    for exp_name, _, _, _ in ablation_configs:
        rows = [r for r in phase2_rows if r["exp"] == exp_name]
        seed_avgs = []
        for seed in FINALIST_SEEDS:
            seed_rows = [r for r in rows if r["seed"] == seed]
            seed_avgs.append(np.mean([r["life_rel_error"] for r in seed_rows]))
        print(f"  {exp_name:18s}: {np.mean(seed_avgs):.2%} ± {np.std(seed_avgs):.2%}")

    full_avg = np.mean([np.mean([r["life_rel_error"] for r in phase2_rows if r["exp"] == "Full(ABC)" and r["seed"] == s])
                        for s in FINALIST_SEEDS])
    for exp_name, _, _, _ in ablation_configs[1:]:
        exp_avg = np.mean([np.mean([r["life_rel_error"] for r in phase2_rows if r["exp"] == exp_name and r["seed"] == s])
                           for s in FINALIST_SEEDS])
        delta = exp_avg - full_avg
        print(f"    {exp_name} delta from Full: {delta:+.2%}")

    save_charts(out_dir, phase2_results, TEST_RUNS, case_tables, threshold_um, summary_df, "phase2_ablation")

    # ═══════════════════════════════════════════════════════
    # PHASE 3: Final model vs other networks
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: Best V2 vs Other Networks")
    print("=" * 70)

    final_results = {"V2-Full(ABC)": phase2_results["Full(ABC)"],
                     "V1-baseline": phase2_results["V1-baseline"]}

    other_models = {"FNN": FNNNet, "1D-CNN": CNNet, "GRU": GRUNet, "LSTM": LSTMNet}
    final_rows = []

    for model_name, ModelClass in other_models.items():
        print(f"\n  Training {model_name}...")
        set_seed(SEED)
        sl = SEQ_LEN_STAR
        tr_f, tr_y, tr_raw, tr_step, tr_delta, \
            va_f, va_y, va_raw, va_step, va_delta = \
            split_by_run(case_tables, sl, False, VAL_RUN, TEST_RUNS)

        feat_dim = tr_f.shape[-1]
        x_scaler = Standardizer().fit(tr_f.reshape(-1, feat_dim)); x_scaler.to_torch(DEVICE)
        y_scaler = Standardizer().fit(tr_y); y_scaler.to_torch(DEVICE)

        m = ModelClass(in_dim=feat_dim, seq_len=sl)
        m = train_model(m, tr_f, tr_y, tr_raw, tr_step, tr_delta,
                         va_f, va_y, x_scaler, y_scaler, real_k, use_wc=False)

        def _make_pred(model, xs, ys, s):
            def fn(F, D, Cr, c, w):
                return predict_single(model, xs, ys, F, D, Cr, c, w, s)
            return fn

        res = evaluate_on_tests(_make_pred(m, x_scaler, y_scaler, sl),
                                 case_tables, summary_df, threshold_um, real_k)
        final_results[model_name] = res

        for tf in TEST_RUNS:
            final_rows.append({"name": model_name, "test_run": tf,
                                "life_rel_error": res[tf]["life_rel_error"],
                                "life_abs_error": res[tf]["life_abs_error"]})

    for name in ["V2-Full(ABC)", "V1-baseline"]:
        for tf in TEST_RUNS:
            final_rows.append({"name": name, "test_run": tf,
                                "life_rel_error": final_results[name][tf]["life_rel_error"],
                                "life_abs_error": final_results[name][tf]["life_abs_error"]})

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(out_dir / "final_model_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "-" * 70)
    print("PHASE 3 SUMMARY:")
    print("-" * 70)
    for name in ["V2-Full(ABC)", "V1-baseline", "FNN", "1D-CNN", "GRU", "LSTM"]:
        if name in final_results:
            avg = avg_rel_error(final_results[name])
            print(f"  {name:18s}: avg_rel_err = {avg:.2%}")

    save_charts(out_dir, final_results, TEST_RUNS, case_tables, threshold_um, summary_df, "phase3_final_comparison")
    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
