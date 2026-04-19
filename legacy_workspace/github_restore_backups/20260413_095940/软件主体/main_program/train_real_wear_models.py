from __future__ import annotations

import copy
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from device_runtime import cpu_state_dict, resolve_training_device


SEED = 20260408
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
EPOCHS = 3000
MAX_EXTRA_STEPS = 400
EPS = 1e-12
DEVICE, DEVICE_LABEL = resolve_training_device()
MONO_LAMBDA_WEAR = 0.08
MONO_LAMBDA_LOAD = 0.04
MONO_LAMBDA_CLEARANCE = 0.04
WEAR_DELTA_MM = 2.0e-4
LOAD_DELTA_RATIO = 0.05
CLEARANCE_DELTA_MM = 0.002
CASE_NAME_PATTERN = re.compile(r"^(?:run|试验)\s*(\d+)$", re.IGNORECASE)


class Standardizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.mean_t: torch.Tensor | None = None
        self.std_t: torch.Tensor | None = None

    def fit(self, array: np.ndarray) -> "Standardizer":
        values = np.asarray(array, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0
        return self

    def to_torch(self, device: torch.device) -> "Standardizer":
        self.mean_t = torch.tensor(self.mean, dtype=torch.float32, device=device)
        self.std_t = torch.tensor(self.std, dtype=torch.float32, device=device)
        return self

    def transform_np(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return (values - self.mean) / self.std

    def inverse_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std_t + self.mean_t


class StressNet(nn.Module):
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
        return self.net(x)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def workspace_root() -> Path:
    return project_root().parent


def processed_root() -> Path:
    path = workspace_root() / "磨损数据（改）"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir() -> Path:
    return processed_root()


def comparison_dir() -> Path:
    path = workspace_root() / "结果"
    path.mkdir(parents=True, exist_ok=True)
    return path


def trained_model_dir() -> Path:
    path = workspace_root() / "工具和杂项" / "训练模型"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processing_config_path() -> Path:
    path = workspace_root() / "工具和杂项" / "脚本与配置"
    path.mkdir(parents=True, exist_ok=True)
    return path / "处理参数配置.csv"


def logs_dir() -> Path:
    path = workspace_root() / "工具和杂项" / "运行日志"
    path.mkdir(parents=True, exist_ok=True)
    return path


def software_model_path() -> Path:
    return project_root() / "storage" / "training_validation" / "stressnet_model.pt"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE_LABEL == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_processing_config() -> dict:
    df = pd.read_csv(processing_config_path())
    if df.empty:
        raise ValueError(f"Empty processing config: {processing_config_path()}")
    row = df.iloc[0].to_dict()
    return {
        "real_wear_coeff_mpa_inv": float(row["real_wear_coeff_mpa_inv"]),
        "default_elastic_modulus_GPa": float(row["default_elastic_modulus_GPa"]),
        "wear_threshold_um": float(row["wear_threshold_um"]),
        "test_case_name": str(row["test_case_name"]).strip(),
    }


def load_cases() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_path = processed_dir() / "修改数据汇总.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Processed dataset summary not found: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    case_tables: dict[str, pd.DataFrame] = {}
    for row in summary_df.itertuples(index=False):
        case_tables[str(row.file_name)] = pd.read_csv(processed_dir() / str(row.file_name))
    return summary_df, case_tables


def case_token(text: str) -> str:
    stem = Path(str(text)).stem.strip().lower()
    match = CASE_NAME_PATTERN.match(stem)
    if match:
        return f"case_{int(match.group(1))}"
    return stem


def select_test_case(summary_df: pd.DataFrame, requested_name: str, threshold_um: float) -> pd.Series:
    requested_token = case_token(requested_name)
    for _, row in summary_df.iterrows():
        if requested_token in {case_token(row["source_file"]), case_token(row["file_name"])}:
            return row

    eligible = summary_df[
        (summary_df["has_measured_pressure"].astype(bool))
        & (summary_df["final_wear_um"].astype(float) >= float(threshold_um))
    ].sort_values("actual_life")
    if eligible.empty:
        raise ValueError("No eligible measured-pressure case found for holdout testing.")
    return eligible.iloc[len(eligible) // 2]


def build_training_arrays(case_tables: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    targets: list[list[float]] = []
    for table in case_tables.values():
        current_rows = table.iloc[:-1]
        for row in current_rows.itertuples(index=False):
            features.append([float(row.F), float(row.D), float(row.Cr), float(row.actual_cycle), float(row.wear_depth)])
            targets.append([math.log(max(float(row.stress), EPS))])
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


def train_stress_net(train_x: np.ndarray, train_y: np.ndarray) -> tuple[StressNet, Standardizer, Standardizer]:
    x_scaler = Standardizer().fit(train_x).to_torch(DEVICE)
    y_scaler = Standardizer().fit(train_y).to_torch(DEVICE)
    train_x_t = to_tensor(x_scaler.transform_np(train_x))
    train_y_t = to_tensor(y_scaler.transform_np(train_y))
    raw_x_t = to_tensor(train_x)

    model = StressNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=False)
    mse = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pred = model(train_x_t)

        wear_shift = raw_x_t.clone()
        wear_shift[:, 4] = wear_shift[:, 4] + WEAR_DELTA_MM
        wear_shift_t = to_tensor(x_scaler.transform_np(wear_shift.detach().cpu().numpy()))
        pred_wear_plus = model(wear_shift_t)

        load_shift = raw_x_t.clone()
        load_shift[:, 0] = load_shift[:, 0] * (1.0 + LOAD_DELTA_RATIO)
        load_shift_t = to_tensor(x_scaler.transform_np(load_shift.detach().cpu().numpy()))
        pred_load_plus = model(load_shift_t)

        clearance_shift = raw_x_t.clone()
        clearance_shift[:, 2] = clearance_shift[:, 2] + CLEARANCE_DELTA_MM
        clearance_shift_t = to_tensor(x_scaler.transform_np(clearance_shift.detach().cpu().numpy()))
        pred_clearance_plus = model(clearance_shift_t)

        data_loss = mse(pred, train_y_t)
        wear_penalty = torch.mean(torch.relu(pred_wear_plus - pred) ** 2)
        load_penalty = torch.mean(torch.relu(pred - pred_load_plus) ** 2)
        clearance_penalty = torch.mean(torch.relu(pred - pred_clearance_plus) ** 2)
        loss = (
            data_loss
            + MONO_LAMBDA_WEAR * wear_penalty
            + MONO_LAMBDA_LOAD * load_penalty
            + MONO_LAMBDA_CLEARANCE * clearance_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 500 == 0 or epoch == EPOCHS:
            with torch.no_grad():
                pred_log = y_scaler.inverse_torch(pred)
                true_log = y_scaler.inverse_torch(train_y_t)
                pred_stress = torch.exp(pred_log)
                true_stress = torch.exp(true_log)
                rmse = torch.sqrt(torch.mean((pred_stress - true_stress) ** 2)).item()
                mape = torch.mean(torch.abs(pred_stress - true_stress) / (true_stress + EPS)).item()
            print(
                f"epoch {epoch:4d} | loss={loss_value:.6e} | "
                f"data={float(data_loss.item()):.6e} | "
                f"pressure_rmse={rmse:.4f} MPa | pressure_mape={mape:.4%}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, x_scaler, y_scaler


def predict_pressure(
    model: StressNet,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    F: float,
    D: float,
    Cr: float,
    actual_cycle: float,
    wear_depth: float,
) -> float:
    x = np.array([[F, D, Cr, actual_cycle, wear_depth]], dtype=np.float32)
    x_t = to_tensor(x_scaler.transform_np(x))
    model.eval()
    with torch.no_grad():
        pred_log_scaled = model(x_t)
        pred_log = y_scaler.inverse_torch(pred_log_scaled)
        pred_stress = torch.exp(pred_log)
    return float(pred_stress.item())


def evaluate_pressure_fit(
    model: StressNet,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    features: np.ndarray,
    targets: np.ndarray,
) -> dict:
    x_t = to_tensor(x_scaler.transform_np(features))
    true_log_t = to_tensor(targets)

    model.eval()
    with torch.no_grad():
        pred_log_scaled = model(x_t)
        pred_log = y_scaler.inverse_torch(pred_log_scaled)
        pred_stress = torch.exp(pred_log).cpu().numpy().reshape(-1)
        true_stress = np.exp(true_log_t.cpu().numpy().reshape(-1))

    rmse = float(np.sqrt(np.mean((pred_stress - true_stress) ** 2)))
    mae = float(np.mean(np.abs(pred_stress - true_stress)))
    mape = float(np.mean(np.abs(pred_stress - true_stress) / np.maximum(true_stress, EPS)))
    return {"rmse": rmse, "mae": mae, "mape": mape}


def median_positive_diff(values: np.ndarray) -> float:
    diffs = np.diff(values.astype(np.float64))
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        raise ValueError("Expected a strictly increasing sequence.")
    return float(np.median(positive_diffs))


def recommended_cycle_step(case_tables: dict[str, pd.DataFrame]) -> float:
    steps = [median_positive_diff(table["actual_cycle"].to_numpy(dtype=float)) for table in case_tables.values()]
    return float(np.median(np.asarray(steps, dtype=np.float64)))


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
    model: StressNet,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    case_df: pd.DataFrame,
    threshold_um: float,
    real_k: float,
    true_life_actual: float,
) -> tuple[pd.DataFrame, float]:
    first = case_df.iloc[0]
    F = float(first["F"])
    D = float(first["D"])
    Cr = float(first["Cr"])
    elastic_modulus = float(first["elastic_modulus_GPa"])
    k_sim = float(first["k_sim"])
    actual_step = median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float))
    sim_step = median_positive_diff(case_df["sim_cycle"].to_numpy(dtype=float))

    actual_cycle = 0.0
    sim_cycle = 0.0
    wear_depth = 0.0
    predicted_life_actual = true_life_actual
    threshold_mm = threshold_um / 1000.0
    rows: list[dict] = []

    internal_limit = max(true_life_actual * 1.4, float(case_df["actual_cycle"].max()) * 1.2, actual_step * 20.0)
    max_steps = int(math.ceil(internal_limit / max(actual_step, 1.0))) + MAX_EXTRA_STEPS

    for _ in range(max_steps):
        pred_stress = predict_pressure(model, x_scaler, y_scaler, F, D, Cr, actual_cycle, wear_depth)
        rows.append(
            {
                "sim_cycle": sim_cycle,
                "actual_cycle": actual_cycle,
                "pred_stress": pred_stress,
                "F": F,
                "D": D,
                "Cr": Cr,
                "pred_wear_depth": wear_depth,
                "pred_wear_depth_um": wear_depth * 1000.0,
                "elastic_modulus_GPa": elastic_modulus,
                "k_sim": k_sim,
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
                    "pred_stress": predict_pressure(model, x_scaler, y_scaler, F, D, Cr, predicted_life_actual, threshold_mm),
                    "F": F,
                    "D": D,
                    "Cr": Cr,
                    "pred_wear_depth": threshold_mm,
                    "pred_wear_depth_um": threshold_um,
                    "elastic_modulus_GPa": elastic_modulus,
                    "k_sim": k_sim,
                }
            )
            break

        actual_cycle = next_actual_cycle
        sim_cycle = next_sim_cycle
        wear_depth = next_wear_depth

    return pd.DataFrame(rows), float(predicted_life_actual)


def train_static_life_model(train_summary: pd.DataFrame) -> tuple[StandardScaler, MLPRegressor]:
    features = train_summary[["F", "D", "Cr"]].to_numpy(dtype=np.float64)
    targets = np.log(train_summary["actual_life"].to_numpy(dtype=np.float64))
    scaler = StandardScaler().fit(features)
    x_train = scaler.transform(features)
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="lbfgs",
        alpha=1e-4,
        random_state=SEED,
        max_iter=5000,
    )
    model.fit(x_train, targets)
    return scaler, model


def predict_static_life(scaler: StandardScaler, model: MLPRegressor, row: pd.Series) -> float:
    x = np.array([[float(row["F"]), float(row["D"]), float(row["Cr"])]], dtype=np.float64)
    pred_log = float(model.predict(scaler.transform(x))[0])
    return float(np.exp(pred_log))


def save_curve_comparison(
    out_path: Path,
    true_df: pd.DataFrame,
    recursive_df: pd.DataFrame,
    threshold_um: float,
    static_life: float,
    recursive_life: float,
    true_life: float,
) -> pd.DataFrame:
    x_upper = max(true_life, recursive_life, static_life) * 1.05
    x_grid = np.linspace(0.0, x_upper, 400)
    true_curve = np.interp(x_grid, true_df["actual_cycle"], true_df["wear_depth_um"])
    recursive_curve = np.interp(x_grid, recursive_df["actual_cycle"], recursive_df["pred_wear_depth_um"])
    static_curve = threshold_um * np.clip(x_grid / max(static_life, EPS), 0.0, 1.0)

    merged = pd.DataFrame(
        {
            "actual_cycle": x_grid,
            "fe_truth_wear_um": true_curve,
            "recursive_pred_wear_um": recursive_curve,
            "static_reference_wear_um": static_curve,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.plot(true_df["actual_cycle"], true_df["wear_depth_um"], color="#111827", linewidth=2.3, label="FE / processed truth")
    ax.plot(recursive_df["actual_cycle"], recursive_df["pred_wear_depth_um"], color="#2563eb", linewidth=2.3, label="Recursive StressNet")
    ax.plot(x_grid, static_curve, color="#d97706", linewidth=2.0, linestyle="--", label="Static FNN (equiv. line)")
    ax.axhline(threshold_um, color="#dc2626", linestyle=":", linewidth=1.5, label=f"{threshold_um:.1f} um threshold")
    ax.set_xlabel("Actual Cycle")
    ax.set_ylabel("Wear Depth (um)")
    ax.set_title("Test-Case Wear Evolution Comparison")
    ax.grid(alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return merged


def save_life_bar_chart(
    out_path: Path,
    true_life: float,
    static_life: float,
    recursive_life: float,
) -> None:
    labels = ["FE truth", "Static FNN", "Recursive StressNet"]
    values = [true_life, static_life, recursive_life]
    colors = ["#111827", "#d97706", "#2563eb"]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylabel("Life at 5 um (cycles)")
    ax.set_title("Life Prediction Comparison on Held-Out Test Case")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.0f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_model_bundle(
    model: StressNet,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    train_summary: pd.DataFrame,
    recommended_step: float,
    training_wear_limit_um: float,
) -> Path:
    bundle = {
        "model_state_dict": cpu_state_dict(model),
        "x_mean": x_scaler.mean,
        "x_std": x_scaler.std,
        "y_mean": y_scaler.mean,
        "y_std": y_scaler.std,
        "feature_order": ["F", "D", "Cr", "actual_cycle", "wear_depth"],
        "training_wear_limit_um": float(training_wear_limit_um),
        "training_actual_life_max": float(train_summary["actual_life"].max()),
        "training_actual_life_mean": float(train_summary["actual_life"].mean()),
        "recommended_cycle_step": float(recommended_step),
        "available_coating": "DLC",
        "model_architecture": "FNN",
        "sequence_length": 1,
    }
    model_path = trained_model_dir() / "递推应力网络模型.pt"
    torch.save(bundle, model_path)
    software_model_path().parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, software_model_path())
    return model_path


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

    train_x, train_y = build_training_arrays(train_tables)
    test_x, test_y = build_training_arrays({test_file: test_table})

    print(f"device: {DEVICE_LABEL}")
    print(f"train cases: {len(train_tables)} | test case: {test_source}")
    print(f"train samples: {len(train_x)} | test samples: {len(test_x)}")

    model, x_scaler, y_scaler = train_stress_net(train_x, train_y)
    train_pressure_metrics = evaluate_pressure_fit(model, x_scaler, y_scaler, train_x, train_y)
    test_pressure_metrics = evaluate_pressure_fit(model, x_scaler, y_scaler, test_x, test_y)
    recommended_step = recommended_cycle_step(train_tables)

    true_life_actual = float(test_row["actual_life"])
    rollout_df, recursive_life = rollout_case(
        model,
        x_scaler,
        y_scaler,
        test_table,
        threshold_um,
        real_k,
        true_life_actual,
    )

    static_scaler, static_model = train_static_life_model(train_summary)
    static_life = predict_static_life(static_scaler, static_model, test_row)

    true_curve_df = threshold_ground_truth(test_table, threshold_um, test_row)
    curve_df = save_curve_comparison(
        comparison_dir() / "留出测试_磨损曲线对比.png",
        true_curve_df,
        rollout_df,
        threshold_um,
        static_life,
        recursive_life,
        true_life_actual,
    )
    save_life_bar_chart(
        comparison_dir() / "留出测试_寿命对比柱状图.png",
        true_life_actual,
        static_life,
        recursive_life,
    )

    curve_df.to_csv(comparison_dir() / "留出测试_磨损曲线对比数据.csv", index=False)
    rollout_df.to_csv(comparison_dir() / "留出测试_递推展开结果.csv", index=False)

    comparison_summary = pd.DataFrame(
        [
            {
                "model": "FE_truth",
                "test_case": test_source,
                "predicted_life": true_life_actual,
                "true_life": true_life_actual,
                "abs_error": 0.0,
                "rel_error": 0.0,
            },
            {
                "model": "Static_FNN",
                "test_case": test_source,
                "predicted_life": static_life,
                "true_life": true_life_actual,
                "abs_error": abs(static_life - true_life_actual),
                "rel_error": abs(static_life - true_life_actual) / max(true_life_actual, EPS),
            },
            {
                "model": "Recursive_StressNet",
                "test_case": test_source,
                "predicted_life": recursive_life,
                "true_life": true_life_actual,
                "abs_error": abs(recursive_life - true_life_actual),
                "rel_error": abs(recursive_life - true_life_actual) / max(true_life_actual, EPS),
            },
        ]
    )
    comparison_summary.to_csv(comparison_dir() / "留出测试_寿命对比表.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {"split": "train_pressure", **train_pressure_metrics},
            {"split": "test_pressure", **test_pressure_metrics},
        ]
    )
    metrics_df.to_csv(comparison_dir() / "留出测试_压力指标.csv", index=False)

    notes = [
        f"Test case: {test_source}",
        f"Threshold: {threshold_um:.4f} um",
        "Curve figure includes three lines:",
        "1. FE / processed truth",
        "2. Recursive StressNet rollout",
        "3. Static FNN equivalent linear reference",
        "",
        "Static FNN baseline predicts only terminal life from [F, D, Cr].",
        "Its wear curve is shown as a straight reference line to emphasize that it does not model wear evolution explicitly.",
    ]
    (comparison_dir() / "留出测试_说明.txt").write_text("\n".join(notes), encoding="utf-8")

    max_train_wear_um = max(float(table["wear_depth"].max()) * 1000.0 for table in train_tables.values())
    model_path = save_model_bundle(
        model,
        x_scaler,
        y_scaler,
        train_summary,
        recommended_step,
        max(threshold_um, max_train_wear_um),
    )

    log_lines = [
        "Real wear-model training complete.",
        f"Test case: {test_source}",
        f"Recommended cycle step: {recommended_step:.2f}",
        f"Recursive model saved to: {model_path}",
        f"Software model synced to: {software_model_path()}",
        f"Comparison results: {comparison_dir()}",
        "",
        "Pressure fit:",
        f"  train RMSE={train_pressure_metrics['rmse']:.4f} MPa, MAE={train_pressure_metrics['mae']:.4f} MPa, MAPE={train_pressure_metrics['mape']:.4%}",
        f"  test  RMSE={test_pressure_metrics['rmse']:.4f} MPa, MAE={test_pressure_metrics['mae']:.4f} MPa, MAPE={test_pressure_metrics['mape']:.4%}",
        "",
        "Held-out life comparison:",
        f"  FE truth={true_life_actual:.2f} cycles",
        f"  Static FNN={static_life:.2f} cycles | abs error={abs(static_life - true_life_actual):.2f}",
        f"  Recursive StressNet={recursive_life:.2f} cycles | abs error={abs(recursive_life - true_life_actual):.2f}",
    ]
    (logs_dir() / "模型训练日志.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
