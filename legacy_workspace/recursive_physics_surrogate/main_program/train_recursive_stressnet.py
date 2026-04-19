from __future__ import annotations

import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REAL_WEAR_COEFF_MPA_INV = 1.84e-10
WEAR_THRESHOLD_MM = 5e-3
TRAINING_WEAR_LIMIT_UM = 5.0
SEED = 20260330
EPOCHS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MAX_EXTRA_STEPS = 10
EPS = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def dataset_dir() -> Path:
    return project_root() / "data" / "current_dataset"


def output_dir() -> Path:
    path = project_root() / "storage" / "training_validation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_output_dir(path: Path) -> None:
    for file_path in path.iterdir():
        if file_path.is_file():
            file_path.unlink()


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cases() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_path = dataset_dir() / "dataset_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Dataset not found. Please run generate_dataset.py first.")

    summary_df = pd.read_csv(summary_path)
    case_tables: dict[str, pd.DataFrame] = {}
    for row in summary_df.itertuples(index=False):
        file_name = str(row.file_name)
        case_tables[file_name] = pd.read_csv(dataset_dir() / file_name)

    return summary_df, case_tables


def build_training_arrays(case_tables: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []

    for table in case_tables.values():
        current_rows = table.iloc[:-1]
        for row in current_rows.itertuples(index=False):
            features.append([row.F, row.D, row.Cr, row.actual_cycle, row.wear_depth])
            targets.append([math.log(max(float(row.stress), EPS))])

    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


def train_stress_net(
    train_x: np.ndarray,
    train_y: np.ndarray,
) -> tuple[StressNet, Standardizer, Standardizer]:
    x_scaler = Standardizer().fit(train_x).to_torch(DEVICE)
    y_scaler = Standardizer().fit(train_y).to_torch(DEVICE)

    train_x_t = to_tensor(x_scaler.transform_np(train_x))
    train_y_t = to_tensor(y_scaler.transform_np(train_y))

    model = StressNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pred = model(train_x_t)
        loss = mse(pred, train_y_t)

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
    train_x: np.ndarray,
    train_y: np.ndarray,
) -> dict:
    x_t = to_tensor(x_scaler.transform_np(train_x))
    true_log_t = to_tensor(train_y)

    model.eval()
    with torch.no_grad():
        pred_log_scaled = model(x_t)
        pred_log = y_scaler.inverse_torch(pred_log_scaled)
        pred_stress = torch.exp(pred_log).cpu().numpy().reshape(-1)
        true_stress = np.exp(true_log_t.cpu().numpy().reshape(-1))

    rmse = float(np.sqrt(np.mean((pred_stress - true_stress) ** 2)))
    mae = float(np.mean(np.abs(pred_stress - true_stress)))
    mape = float(np.mean(np.abs(pred_stress - true_stress) / true_stress))
    return {"pressure_rmse_mpa": rmse, "pressure_mae_mpa": mae, "pressure_mape": mape}


def median_positive_diff(values: np.ndarray) -> float:
    diffs = np.diff(values.astype(np.float64))
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        raise ValueError("Expected a strictly increasing sequence for rollout step estimation.")
    return float(np.median(positive_diffs))


def recommended_cycle_step(case_tables: dict[str, pd.DataFrame]) -> float:
    case_steps = []
    for case_df in case_tables.values():
        case_steps.append(median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float)))
    return float(np.median(np.asarray(case_steps, dtype=np.float64)))


def rollout_case(
    model: StressNet,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    case_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    first = case_df.iloc[0]
    F = float(first["F"])
    D = float(first["D"])
    Cr = float(first["Cr"])
    elastic_modulus = float(first["elastic_modulus_GPa"])
    k_sim = float(first["k_sim"])
    actual_step = median_positive_diff(case_df["actual_cycle"].to_numpy(dtype=float))
    sim_step = median_positive_diff(case_df["sim_cycle"].to_numpy(dtype=float))
    base_step_count = len(case_df) - 1
    true_life_actual = float(case_df.iloc[-1]["actual_cycle"])

    actual_cycle = 0.0
    sim_cycle = 0.0
    wear_depth = 0.0
    predicted_life_actual = true_life_actual
    rows = []

    for _ in range(base_step_count + MAX_EXTRA_STEPS + 1):
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
        delta_wear = REAL_WEAR_COEFF_MPA_INV * pred_stress * delta_s
        next_actual_cycle = actual_cycle + actual_step
        next_wear_depth = wear_depth + delta_wear
        next_sim_cycle = sim_cycle + sim_step

        if next_wear_depth >= WEAR_THRESHOLD_MM:
            ratio = (WEAR_THRESHOLD_MM - wear_depth) / max(delta_wear, EPS)
            predicted_life_actual = actual_cycle + ratio * actual_step
            break

        actual_cycle = next_actual_cycle
        sim_cycle = next_sim_cycle
        wear_depth = next_wear_depth

    return pd.DataFrame(rows), float(predicted_life_actual)


def save_demo_plot(
    out_dir: Path,
    ground_truth_df: pd.DataFrame,
    rollout_df: pd.DataFrame,
    file_name: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(ground_truth_df["actual_cycle"], ground_truth_df["stress"], label="Ground truth", linewidth=2)
    axes[0].plot(
        rollout_df["actual_cycle"],
        rollout_df["pred_stress"],
        label="StressNet recursive",
        linewidth=2,
        linestyle="--",
    )
    axes[0].set_ylabel("Stress (MPa)")
    axes[0].set_title(f"{file_name} Stress Evolution")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        ground_truth_df["actual_cycle"],
        ground_truth_df["wear_depth"] * 1000.0,
        label="Ground truth",
        linewidth=2,
    )
    axes[1].plot(
        rollout_df["actual_cycle"],
        rollout_df["pred_wear_depth_um"],
        label="StressNet recursive",
        linewidth=2,
        linestyle="--",
    )
    axes[1].axhline(TRAINING_WEAR_LIMIT_UM, color="tab:red", linestyle=":", label="5 um threshold")
    axes[1].set_xlabel("Actual Cycle")
    axes[1].set_ylabel("Wear Depth (um)")
    axes[1].set_title(f"{file_name} Wear Evolution")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(file_name).stem}_recursive_curve.png", dpi=300)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    out_dir = output_dir()
    reset_output_dir(out_dir)

    summary_df, case_tables = load_cases()
    train_x, train_y = build_training_arrays(case_tables)
    recommended_step = recommended_cycle_step(case_tables)

    print(f"device: {DEVICE}")
    print(f"training samples: {len(train_x)}")
    model, x_scaler, y_scaler = train_stress_net(train_x, train_y)
    pressure_metrics = evaluate_pressure_fit(model, x_scaler, y_scaler, train_x, train_y)

    rollout_rows = []

    for row in summary_df.itertuples(index=False):
        file_name = str(row.file_name)
        case_df = case_tables[file_name]
        rollout_df, predicted_life_actual = rollout_case(model, x_scaler, y_scaler, case_df)

        true_life_actual = float(case_df.iloc[-1]["actual_cycle"])
        life_abs_error = abs(predicted_life_actual - true_life_actual)
        life_relative_error = life_abs_error / true_life_actual

        rollout_rows.append(
            {
                "file_name": file_name,
                "true_life_actual": true_life_actual,
                "predicted_life_actual": predicted_life_actual,
                "life_abs_error": life_abs_error,
                "life_relative_error": life_relative_error,
            }
        )

        rollout_df.to_csv(out_dir / f"{Path(file_name).stem}_rollout.csv", index=False)

    rollout_summary = pd.DataFrame(rollout_rows)
    rollout_summary.to_csv(out_dir / "rollout_summary.csv", index=False)

    first_file = str(summary_df.iloc[0]["file_name"])
    save_demo_plot(
        out_dir,
        case_tables[first_file],
        pd.read_csv(out_dir / f"{Path(first_file).stem}_rollout.csv"),
        first_file,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
            "feature_order": ["F", "D", "Cr", "actual_cycle", "wear_depth"],
            "training_wear_limit_um": TRAINING_WEAR_LIMIT_UM,
            "training_actual_life_max": float(summary_df["actual_life"].max()),
            "training_actual_life_mean": float(summary_df["actual_life"].mean()),
            "recommended_cycle_step": recommended_step,
            "available_coating": "DLC",
        },
        out_dir / "stressnet_model.pt",
    )

    print("StressNet training complete.")
    print(
        "Pressure fit: "
        f"RMSE={pressure_metrics['pressure_rmse_mpa']:.4f} MPa, "
        f"MAE={pressure_metrics['pressure_mae_mpa']:.4f} MPa, "
        f"MAPE={pressure_metrics['pressure_mape']:.4%}"
    )
    print(
        "Recursive life rollout: "
        f"MAE={rollout_summary['life_abs_error'].mean():.2f} cycles, "
        f"MAPE={rollout_summary['life_relative_error'].mean():.4%}, "
        f"max abs error={rollout_summary['life_abs_error'].max():.2f} cycles"
    )
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
