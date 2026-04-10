import glob
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = "./bearing_dataset"

# =========================
# 你每次主要改这三个参数
# =========================
PRED_F2 = 6000.0
PRED_D1 = 4.0
PRED_CR = 0.013

# =========================
# 物理/预测设置
# =========================
WEAR_UNIT = "mm"           # 当前 bearing_dataset 更像是 mm
WEAR_THRESHOLD_UM = 5.0    # 目标阈值：5 微米
TARGET_PLOT_POINTS = 80
PREDICT_DN = 500.0         # 内部递推步长
MAX_PREDICT_CYCLES = 3_000_000.0

# =========================
# 训练设置
# =========================
EPOCHS = 1500
LR = 1e-3
WEIGHT_DECAY = 1e-6
PHYSICS_LOSS_WEIGHT = 0.5
SEED = 42

EPS = 1e-12


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wear_threshold_in_data_unit():
    unit = WEAR_UNIT.strip().lower()
    if unit in {"mm", "millimeter", "millimetre"}:
        return WEAR_THRESHOLD_UM / 1000.0
    if unit in {"m", "meter", "metre"}:
        return WEAR_THRESHOLD_UM * 1e-6
    if unit in {"um", "μm", "micron", "micrometer", "micrometre"}:
        return WEAR_THRESHOLD_UM
    raise ValueError(f"不支持的磨损单位: {WEAR_UNIT}")


def wear_to_microns(values):
    unit = WEAR_UNIT.strip().lower()
    values = np.asarray(values, dtype=np.float32)
    if unit in {"mm", "millimeter", "millimetre"}:
        return values * 1000.0
    if unit in {"m", "meter", "metre"}:
        return values * 1e6
    if unit in {"um", "μm", "micron", "micrometer", "micrometre"}:
        return values
    raise ValueError(f"不支持的磨损单位: {WEAR_UNIT}")


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.mean_t = None
        self.std_t = None

    def fit(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        self.mean = array.mean(axis=0, keepdims=True)
        self.std = array.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def to_torch(self, device):
        self.mean_t = torch.tensor(self.mean, dtype=torch.float32, device=device)
        self.std_t = torch.tensor(self.std, dtype=torch.float32, device=device)
        return self

    def transform_np(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return (array - self.mean) / self.std

    def inverse_np(self, array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array * self.std + self.mean

    def transform_torch(self, tensor):
        return (tensor - self.mean_t) / self.std_t

    def inverse_torch(self, tensor):
        return tensor * self.std_t + self.mean_t


class StressNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class KappaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_sequences(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"在 {folder} 下没有找到 csv 文件")

    sequences = []
    required_cols = {"cycle", "stress", "F2_actual", "d1", "Cr", "wear_depth"}

    for file_path in files:
        df = pd.read_csv(file_path)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{file_path} 缺少列: {missing}")

        df = df.sort_values("cycle").reset_index(drop=True)
        cycle = df["cycle"].to_numpy(dtype=np.float32)
        stress = df["stress"].to_numpy(dtype=np.float32)
        wear = df["wear_depth"].to_numpy(dtype=np.float32)

        if len(df) < 2:
            raise ValueError(f"{file_path} 点数过少，至少需要 2 个点")
        if np.any(np.diff(cycle) <= 0):
            raise ValueError(f"{file_path} 的 cycle 不是严格递增")
        if np.any(stress <= 0):
            raise ValueError(f"{file_path} 含有非正 stress，当前脚本假定 stress > 0")
        if np.any(np.diff(wear) < -1e-10):
            raise ValueError(f"{file_path} 的 wear_depth 不是单调递增")

        sequences.append(
            {
                "cycle": cycle,
                "stress": stress,
                "wear": wear,
                "F": float(df["F2_actual"].iloc[0]),
                "d1": float(df["d1"].iloc[0]),
                "Cr": float(df["Cr"].iloc[0]),
                "file": os.path.basename(file_path),
            }
        )

    return sequences


def build_training_arrays(seqs):
    stress_x = []
    stress_y = []
    kappa_x = []
    kappa_y = []
    trans_stress_x = []
    trans_kappa_x = []
    trans_wear_now = []
    trans_wear_next = []
    trans_dN = []

    for seq in seqs:
        cycle = seq["cycle"]
        stress = seq["stress"]
        wear = seq["wear"]
        F = seq["F"]
        d1 = seq["d1"]
        Cr = seq["Cr"]

        for i in range(len(cycle)):
            stress_x.append([F, d1, Cr, cycle[i], wear[i]])
            stress_y.append([math.log(max(float(stress[i]), EPS))])

        for i in range(len(cycle) - 1):
            dN = float(cycle[i + 1] - cycle[i])
            kappa = (float(wear[i + 1]) - float(wear[i])) / (dN * max(float(stress[i]), EPS))
            kappa = max(kappa, EPS)

            kappa_x.append([F, d1, Cr, wear[i]])
            kappa_y.append([math.log(kappa)])

            trans_stress_x.append([F, d1, Cr, cycle[i], wear[i]])
            trans_kappa_x.append([F, d1, Cr, wear[i]])
            trans_wear_now.append([wear[i]])
            trans_wear_next.append([wear[i + 1]])
            trans_dN.append([dN])

    arrays = {
        "stress_x": np.asarray(stress_x, dtype=np.float32),
        "stress_y": np.asarray(stress_y, dtype=np.float32),
        "kappa_x": np.asarray(kappa_x, dtype=np.float32),
        "kappa_y": np.asarray(kappa_y, dtype=np.float32),
        "trans_stress_x": np.asarray(trans_stress_x, dtype=np.float32),
        "trans_kappa_x": np.asarray(trans_kappa_x, dtype=np.float32),
        "trans_wear_now": np.asarray(trans_wear_now, dtype=np.float32),
        "trans_wear_next": np.asarray(trans_wear_next, dtype=np.float32),
        "trans_dN": np.asarray(trans_dN, dtype=np.float32),
    }
    return arrays


def parameter_ranges(seqs):
    F_values = [seq["F"] for seq in seqs]
    d1_values = [seq["d1"] for seq in seqs]
    Cr_values = [seq["Cr"] for seq in seqs]
    return {
        "F": (min(F_values), max(F_values)),
        "d1": (min(d1_values), max(d1_values)),
        "Cr": (min(Cr_values), max(Cr_values)),
    }


def warn_if_outside_training_range(F, d1, Cr, ranges):
    warnings = []
    if not (ranges["F"][0] <= F <= ranges["F"][1]):
        warnings.append(f"F2={F} 超出训练范围 [{ranges['F'][0]}, {ranges['F'][1]}]")
    if not (ranges["d1"][0] <= d1 <= ranges["d1"][1]):
        warnings.append(f"d1={d1} 超出训练范围 [{ranges['d1'][0]}, {ranges['d1'][1]}]")
    if not (ranges["Cr"][0] <= Cr <= ranges["Cr"][1]):
        warnings.append(f"Cr={Cr} 超出训练范围 [{ranges['Cr'][0]}, {ranges['Cr'][1]}]")

    for message in warnings:
        print(f"warning: {message}")


def fit_standardizers(arrays):
    scalers = {
        "stress_x": Standardizer().fit(arrays["stress_x"]).to_torch(DEVICE),
        "stress_y": Standardizer().fit(arrays["stress_y"]).to_torch(DEVICE),
        "kappa_x": Standardizer().fit(arrays["kappa_x"]).to_torch(DEVICE),
        "kappa_y": Standardizer().fit(arrays["kappa_y"]).to_torch(DEVICE),
    }
    return scalers


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


def predict_stress_value(stress_net, scalers, F, d1, Cr, cycle, wear):
    x = np.array([[F, d1, Cr, cycle, wear]], dtype=np.float32)
    x_norm = scalers["stress_x"].transform_np(x)
    x_tensor = to_tensor(x_norm)

    with torch.no_grad():
        log_stress_norm = stress_net(x_tensor)
        log_stress = scalers["stress_y"].inverse_torch(log_stress_norm)
        stress = torch.exp(log_stress)

    return float(stress.item())


def predict_kappa_value(kappa_net, scalers, F, d1, Cr, wear):
    x = np.array([[F, d1, Cr, wear]], dtype=np.float32)
    x_norm = scalers["kappa_x"].transform_np(x)
    x_tensor = to_tensor(x_norm)

    with torch.no_grad():
        log_kappa_norm = kappa_net(x_tensor)
        log_kappa = scalers["kappa_y"].inverse_torch(log_kappa_norm)
        kappa = torch.exp(log_kappa)

    return float(kappa.item())


def train_model(seqs):
    arrays = build_training_arrays(seqs)
    scalers = fit_standardizers(arrays)

    stress_x = to_tensor(scalers["stress_x"].transform_np(arrays["stress_x"]))
    stress_y = to_tensor(scalers["stress_y"].transform_np(arrays["stress_y"]))
    kappa_x = to_tensor(scalers["kappa_x"].transform_np(arrays["kappa_x"]))
    kappa_y = to_tensor(scalers["kappa_y"].transform_np(arrays["kappa_y"]))
    trans_stress_x = to_tensor(scalers["stress_x"].transform_np(arrays["trans_stress_x"]))
    trans_kappa_x = to_tensor(scalers["kappa_x"].transform_np(arrays["trans_kappa_x"]))
    trans_wear_now = to_tensor(arrays["trans_wear_now"])
    trans_wear_next = to_tensor(arrays["trans_wear_next"])
    trans_dN = to_tensor(arrays["trans_dN"])

    stress_net = StressNet().to(DEVICE)
    kappa_net = KappaNet().to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(stress_net.parameters()) + list(kappa_net.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    mse = nn.MSELoss()

    wear_scale = max(
        float(arrays["trans_wear_next"].max()),
        wear_threshold_in_data_unit(),
        1e-6,
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        stress_net.train()
        kappa_net.train()

        pred_stress_norm = stress_net(stress_x)
        pred_kappa_norm = kappa_net(kappa_x)
        stress_loss = mse(pred_stress_norm, stress_y)
        kappa_loss = mse(pred_kappa_norm, kappa_y)

        trans_log_stress_norm = stress_net(trans_stress_x)
        trans_log_kappa_norm = kappa_net(trans_kappa_x)
        trans_log_stress = scalers["stress_y"].inverse_torch(trans_log_stress_norm)
        trans_log_kappa = scalers["kappa_y"].inverse_torch(trans_log_kappa_norm)
        trans_stress = torch.exp(trans_log_stress)
        trans_kappa = torch.exp(trans_log_kappa)
        wear_next_pred = trans_wear_now + trans_dN * trans_stress * trans_kappa
        physics_loss = torch.mean(((wear_next_pred - trans_wear_next) / wear_scale) ** 2)

        total_loss = stress_loss + kappa_loss + PHYSICS_LOSS_WEIGHT * physics_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_value = float(total_loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "stress_net": stress_net.state_dict(),
                "kappa_net": kappa_net.state_dict(),
            }

        if epoch == 1 or epoch % 100 == 0 or epoch == EPOCHS:
            with torch.no_grad():
                stress_real = torch.exp(scalers["stress_y"].inverse_torch(pred_stress_norm))
                stress_true = torch.exp(scalers["stress_y"].inverse_torch(stress_y))
                stress_rmse = torch.sqrt(torch.mean((stress_real - stress_true) ** 2)).item()

                kappa_real = torch.exp(scalers["kappa_y"].inverse_torch(pred_kappa_norm))
                kappa_true = torch.exp(scalers["kappa_y"].inverse_torch(kappa_y))
                kappa_rel = torch.mean(torch.abs(kappa_real - kappa_true) / (kappa_true + EPS)).item()

            print(
                f"epoch {epoch:4d} | total={loss_value:.6e} | "
                f"stress_loss={stress_loss.item():.6e} | "
                f"kappa_loss={kappa_loss.item():.6e} | "
                f"physics_loss={physics_loss.item():.6e} | "
                f"stress_rmse={stress_rmse:.4f} | "
                f"kappa_mape={kappa_rel:.4f}"
            )

    if best_state is not None:
        stress_net.load_state_dict(best_state["stress_net"])
        kappa_net.load_state_dict(best_state["kappa_net"])

    return stress_net, kappa_net, scalers


def compress_curve(cycle, stress, wear, target_points=80):
    n = len(cycle)
    if n <= target_points:
        return cycle, stress, wear

    idx = np.linspace(0, n - 1, target_points).astype(int)
    idx = np.unique(idx)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    return cycle[idx], stress[idx], wear[idx]


def predict_until_threshold(stress_net, kappa_net, scalers, F, d1, Cr):
    stress_net.eval()
    kappa_net.eval()

    wear_threshold = wear_threshold_in_data_unit()
    max_steps = int(MAX_PREDICT_CYCLES // PREDICT_DN)

    cycle_list = [0.0]
    wear_list = [0.0]
    stress_list = []

    current_cycle = 0.0
    current_wear = 0.0
    reached_threshold = False
    threshold_cycle = None

    for _ in range(max_steps):
        current_stress = predict_stress_value(
            stress_net, scalers, F, d1, Cr, current_cycle, current_wear
        )
        stress_list.append(current_stress)

        if current_wear >= wear_threshold:
            reached_threshold = True
            threshold_cycle = current_cycle
            break

        current_kappa = predict_kappa_value(kappa_net, scalers, F, d1, Cr, current_wear)
        next_cycle = current_cycle + PREDICT_DN
        next_wear = current_wear + PREDICT_DN * current_kappa * current_stress

        if next_wear >= wear_threshold:
            ratio = (wear_threshold - current_wear) / max(next_wear - current_wear, EPS)
            threshold_cycle = current_cycle + ratio * PREDICT_DN
            threshold_stress = predict_stress_value(
                stress_net, scalers, F, d1, Cr, threshold_cycle, wear_threshold
            )

            cycle_list.append(float(threshold_cycle))
            wear_list.append(float(wear_threshold))
            stress_list.append(float(threshold_stress))
            reached_threshold = True
            break

        current_cycle = next_cycle
        current_wear = next_wear
        cycle_list.append(float(current_cycle))
        wear_list.append(float(current_wear))

    if len(stress_list) < len(cycle_list):
        final_stress = predict_stress_value(
            stress_net, scalers, F, d1, Cr, cycle_list[-1], wear_list[-1]
        )
        stress_list.append(final_stress)

    cycle_arr = np.asarray(cycle_list, dtype=np.float32)
    stress_arr = np.asarray(stress_list[: len(cycle_arr)], dtype=np.float32)
    wear_arr = np.asarray(wear_list, dtype=np.float32)

    if threshold_cycle is None:
        threshold_cycle = float(cycle_arr[-1])

    return cycle_arr, stress_arr, wear_arr, float(threshold_cycle), reached_threshold


def plot_prediction(cycle, stress, wear, threshold_cycle, save_path="prediction_curve.png"):
    wear_um = wear_to_microns(wear)

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(cycle, stress, linewidth=2)
    ax1.axvline(threshold_cycle, linestyle="--", color="tab:red")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Stress")
    ax1.set_title("Predicted Stress Curve")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(cycle, wear_um, linewidth=2, label="Predicted wear")
    ax2.axhline(WEAR_THRESHOLD_UM, linestyle="--", color="tab:red", label="5 um threshold")
    ax2.axvline(threshold_cycle, linestyle="--", color="tab:orange", label="threshold cycle")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Wear Depth (um)")
    ax2.set_title("Predicted Wear Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"curve figure saved: {save_path}")


def main():
    set_seed(SEED)

    print(f"device: {DEVICE}")
    print("loading data...")
    seqs = load_sequences(DATA_FOLDER)
    ranges = parameter_ranges(seqs)
    print(f"sequence count: {len(seqs)}")
    print(
        f"training range: "
        f"F2={ranges['F'][0]}~{ranges['F'][1]}, "
        f"d1={ranges['d1'][0]}~{ranges['d1'][1]}, "
        f"Cr={ranges['Cr'][0]}~{ranges['Cr'][1]}"
    )
    print(
        f"wear threshold = {WEAR_THRESHOLD_UM:.3f} um "
        f"({wear_threshold_in_data_unit():.6f} in data unit: {WEAR_UNIT})"
    )

    for i, seq in enumerate(seqs[:5]):
        print(
            f"[{i}] file={seq['file']}, F={seq['F']}, d1={seq['d1']}, "
            f"Cr={seq['Cr']}, points={len(seq['cycle'])}, "
            f"wear_end={seq['wear'][-1]:.6f} {WEAR_UNIT}"
        )

    print("training surrogate model...")
    stress_net, kappa_net, scalers = train_model(seqs)
    print("training finished")

    print(
        f"predicting life for F2={PRED_F2}, d1={PRED_D1}, Cr={PRED_CR} "
        f"until wear reaches {WEAR_THRESHOLD_UM:.1f} um..."
    )
    warn_if_outside_training_range(PRED_F2, PRED_D1, PRED_CR, ranges)
    cycle_full, stress_full, wear_full, threshold_cycle, reached = predict_until_threshold(
        stress_net, kappa_net, scalers,
        PRED_F2, PRED_D1, PRED_CR,
    )

    cycle, stress, wear = compress_curve(
        cycle_full, stress_full, wear_full, target_points=TARGET_PLOT_POINTS
    )

    df = pd.DataFrame(
        {
            "cycle": cycle,
            "stress": stress,
            f"wear_{WEAR_UNIT}": wear,
            "wear_um": wear_to_microns(wear),
        }
    )
    df.to_csv("prediction_result.csv", index=False)
    print("prediction saved: prediction_result.csv")

    if reached:
        print(f"达到 5 微米时的转数约为: {threshold_cycle:.0f}")
    else:
        print(
            f"警告：在 {MAX_PREDICT_CYCLES:.0f} 转内未达到 5 微米，"
            f"当前最后磨损为 {wear_to_microns([wear_full[-1]])[0]:.3f} um"
        )

    plot_prediction(
        cycle,
        stress,
        wear,
        threshold_cycle=threshold_cycle,
        save_path="prediction_curve.png",
    )

    torch.save(
        {
            "stress_net": stress_net.state_dict(),
            "kappa_net": kappa_net.state_dict(),
            "wear_unit": WEAR_UNIT,
            "wear_threshold_um": WEAR_THRESHOLD_UM,
        },
        "surrogate_model.pt",
    )
    print("model saved: surrogate_model.pt")
    print("done")


if __name__ == "__main__":
    main()
