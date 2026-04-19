from __future__ import annotations

import argparse
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import font_manager as fm
from scipy.optimize import curve_fit

import v2_pipeline as v2


REAL_WEAR_COEFF_MPA_INV = 1.84e-10
DEFAULT_MODEL_PATH = Path("storage") / "training_validation" / "stressnet_model.pt"
DEFAULT_INPUT_PATH = Path("app_workspace") / "prediction_input.csv"
DEFAULT_OUTPUT_DIR = Path("storage") / "predictions"
EPS = 1e-12
DEVICE = torch.device("cpu")
DEFAULT_COATING = "DLC"
FIT_TAIL_POINTS = 12
DEFAULT_SEQUENCE_LENGTH = 6
PLOT_FONT_CANDIDATES = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC"]


def select_plot_font():
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    for font_name in PLOT_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return fm.FontProperties(family=font_name)
    plt.rcParams["axes.unicode_minus"] = False
    return None


PLOT_FONT = select_plot_font()

COATING_LIBRARY = {
    "DLC": {
        "display_name": "DLC",
        "model_path": DEFAULT_MODEL_PATH,
        "wear_coeff_mpa_inv": REAL_WEAR_COEFF_MPA_INV,
        "description": "Default DLC coating model",
    }
}


class Standardizer:
    def __init__(self) -> None:
        self.mean_t: torch.Tensor | None = None
        self.std_t: torch.Tensor | None = None

    def load(self, mean: np.ndarray, std: np.ndarray, device: torch.device) -> "Standardizer":
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        std_arr[std_arr < 1e-8] = 1.0
        self.mean_t = torch.tensor(mean_arr, dtype=torch.float32, device=device)
        self.std_t = torch.tensor(std_arr, dtype=torch.float32, device=device)
        return self

    def transform_np(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        mean = self.mean_t.detach().cpu().numpy()
        std = self.std_t.detach().cpu().numpy()
        return (values - mean) / std

    def inverse_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std_t + self.mean_t


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


class TransformerNet(nn.Module):
    def __init__(self, seq_len: int = DEFAULT_SEQUENCE_LENGTH, d_model: int = 32, nhead: int = 4) -> None:
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


MODEL_FACTORY = {
    "FNN": lambda seq_len: FNNNet(),
    "Transformer": lambda seq_len: TransformerNet(seq_len=seq_len),
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def available_coatings() -> list[str]:
    return list(COATING_LIBRARY.keys())


def coating_config(coating_name: str) -> dict:
    if coating_name not in COATING_LIBRARY:
        raise KeyError(f"Unsupported coating: {coating_name}")
    return COATING_LIBRARY[coating_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bearing life and show wear-depth curve.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true", help="Save the plot only, do not open the window.")
    return parser.parse_args()


def default_request() -> dict:
    return {
        "coating_name": DEFAULT_COATING,
        "F": 4325.0,
        "D": 10.0,
        "Cr": 0.0500,
        "elastic_modulus_GPa": 210.0,
        "actual_cycle_step": 1672.0,
        "wear_threshold_um": 5.0,
    }


def normalize_request(request_like: dict) -> dict:
    defaults = default_request()
    request = {}
    request["coating_name"] = str(request_like.get("coating_name", defaults["coating_name"])).strip() or DEFAULT_COATING
    request["F"] = float(request_like.get("F", defaults["F"]))
    request["D"] = float(request_like.get("D", defaults["D"]))
    request["Cr"] = float(request_like.get("Cr", defaults["Cr"]))
    request["elastic_modulus_GPa"] = float(request_like.get("elastic_modulus_GPa", defaults["elastic_modulus_GPa"]))
    request["actual_cycle_step"] = float(request_like.get("actual_cycle_step", defaults["actual_cycle_step"]))
    request["wear_threshold_um"] = float(request_like.get("wear_threshold_um", defaults["wear_threshold_um"]))

    if request["coating_name"] not in COATING_LIBRARY:
        raise ValueError(f"当前未注册该涂层: {request['coating_name']}")
    if request["F"] <= 0 or request["D"] <= 0:
        raise ValueError("F 和 D 必须大于 0。")
    if request["actual_cycle_step"] <= 0:
        raise ValueError("递推步长必须大于 0。")
    if request["wear_threshold_um"] <= 0:
        raise ValueError("磨损阈值必须大于 0。")
    return request


def load_request(input_path: Path) -> dict:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No prediction row found in {input_path}")
    row = df.iloc[0].to_dict()
    return normalize_request(row)


def resolve_model_path(coating_name: str, model_override: Path | None = None) -> Path:
    if model_override is not None:
        return model_override
    return coating_config(coating_name)["model_path"]


def load_legacy_runtime(bundle: dict, architecture: str, seq_len: int) -> dict:
    if architecture not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model architecture in bundle: {architecture}")

    model = MODEL_FACTORY[architecture](seq_len).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    x_scaler = Standardizer().load(bundle["x_mean"], bundle["x_std"], DEVICE)
    y_scaler = Standardizer().load(bundle["y_mean"], bundle["y_std"], DEVICE)
    metadata = {
        "feature_version": "v1",
        "training_wear_limit_um": float(bundle.get("training_wear_limit_um", 5.0)),
        "training_actual_life_max": float(bundle.get("training_actual_life_max", 150000.0)),
        "training_actual_life_mean": float(bundle.get("training_actual_life_mean", 120000.0)),
        "recommended_cycle_step": float(bundle.get("recommended_cycle_step", 1000.0)),
        "available_coating": str(bundle.get("available_coating", DEFAULT_COATING)),
        "model_architecture": architecture,
        "sequence_length": seq_len,
        "candidate_name": str(bundle.get("candidate_name", architecture)),
    }
    return {
        "bundle_version": "v1",
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "metadata": metadata,
    }


def load_model_runtime(model_path: Path) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`.*")
    bundle = torch.load(model_path, map_location=DEVICE, weights_only=False)
    feature_version = str(bundle.get("feature_version", "v1"))
    if feature_version == "v2":
        model, input_scalers, target_scaler, metadata, config = v2.load_runtime_from_bundle(bundle, device=DEVICE)
        return {
            "bundle_version": "v2",
            "model": model,
            "input_scalers": input_scalers,
            "target_scaler": target_scaler,
            "metadata": metadata,
            "candidate_config": config,
        }

    architecture = str(bundle.get("model_architecture", "FNN"))
    seq_len = int(bundle.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    return load_legacy_runtime(bundle, architecture, seq_len)


def make_sequence(history: list[list[float]], seq_len: int) -> np.ndarray:
    seq = np.asarray(history[-seq_len:], dtype=np.float32)
    if len(seq) < seq_len:
        pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
        seq = np.vstack([pad, seq])
    return seq


def predict_pressure_from_history_v1(
    model: nn.Module,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    history: list[list[float]],
    sequence_length: int,
) -> float:
    x = make_sequence(history, sequence_length)[np.newaxis, :, :]
    x_norm = x_scaler.transform_np(x)
    x_t = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_log_scaled = model(x_t)
        pred_log = y_scaler.inverse_torch(pred_log_scaled)
        pred_stress = torch.exp(pred_log)
    return float(pred_stress.item())


def predict_pressure_from_runtime(runtime: dict, history: list[list[float]], actual_cycle_step: float) -> float:
    if runtime["bundle_version"] == "v2":
        trained_runtime = v2.TrainedCandidate(
            config=runtime["candidate_config"],
            model=runtime["model"],
            input_scalers=runtime["input_scalers"],
            target_scaler=runtime["target_scaler"],
            best_epoch=int(runtime["metadata"].get("best_epoch", 0)),
            monitored_loss=0.0,
            training_sample_count=0,
            validation_sample_count=0,
        )
        return v2.predict_pressure_from_history(trained_runtime, history, actual_cycle_step)

    metadata = runtime["metadata"]
    return predict_pressure_from_history_v1(
        runtime["model"],
        runtime["x_scaler"],
        runtime["y_scaler"],
        history,
        int(metadata.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)),
    )


def internal_cycle_limit(request: dict, model_metadata: dict) -> float:
    trained_limit = model_metadata["training_actual_life_max"]
    trained_wear_um = model_metadata["training_wear_limit_um"]
    threshold_scale = max(request["wear_threshold_um"] / max(trained_wear_um, EPS), 1.0)
    return max(220000.0, trained_limit * threshold_scale * 3.0)


def wear_fit_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return a * x + b * (1.0 - np.exp(-c * x))


def fit_wear_curve(actual_cycle: np.ndarray, wear_depth_mm: np.ndarray) -> tuple[np.ndarray, str]:
    x = np.asarray(actual_cycle, dtype=np.float64)
    y = np.asarray(wear_depth_mm, dtype=np.float64)

    if len(x) < FIT_TAIL_POINTS:
        raise RuntimeError("Not enough points for wear curve fitting.")

    tail_slope = max((y[-1] - y[-FIT_TAIL_POINTS]) / max(x[-1] - x[-FIT_TAIL_POINTS], EPS), EPS)
    p0 = [tail_slope, max(y[-1] * 0.2, EPS), 1.0 / max(x[-1], 1.0)]
    bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    try:
        params, _ = curve_fit(
            wear_fit_function,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
        )
        return params, "fit_extrapolation"
    except Exception:
        return np.array([tail_slope, y[-1] - tail_slope * x[-1], 0.0], dtype=np.float64), "linear_fallback"


def extrapolated_wear_value(cycle_value: float, params: np.ndarray, mode: str) -> float:
    if mode == "fit_extrapolation":
        return float(wear_fit_function(np.array([cycle_value]), *params)[0])
    slope, intercept, _ = params
    return float(slope * cycle_value + intercept)


def extrapolated_stress_value(base_df: pd.DataFrame, cycle_value: float) -> float:
    recent_df = base_df.tail(min(12, len(base_df)))
    x = recent_df["actual_cycle"].to_numpy(dtype=np.float64)
    y = recent_df["stress"].to_numpy(dtype=np.float64)
    if len(x) < 2:
        return float(y[-1])
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(max(slope * cycle_value + intercept, EPS))


def extend_prediction_with_fit(
    base_df: pd.DataFrame,
    request: dict,
    model_metadata: dict,
) -> tuple[pd.DataFrame, float | None, str]:
    target_wear_mm = request["wear_threshold_um"] / 1000.0
    trained_wear_mm = model_metadata["training_wear_limit_um"] / 1000.0
    actual_cycle_step = request["actual_cycle_step"]

    if target_wear_mm <= trained_wear_mm + EPS:
        return base_df, float(base_df["actual_cycle"].iloc[-1]), "recursive"

    fit_params, extension_mode = fit_wear_curve(
        base_df["actual_cycle"].to_numpy(dtype=np.float64),
        base_df["wear_depth"].to_numpy(dtype=np.float64),
    )

    extended_rows = []
    current_cycle = float(base_df["actual_cycle"].iloc[-1])
    max_actual_cycles = internal_cycle_limit(request, model_metadata)

    while current_cycle <= max_actual_cycles + EPS:
        current_cycle += actual_cycle_step
        wear_value = extrapolated_wear_value(current_cycle, fit_params, extension_mode)
        stress_value = extrapolated_stress_value(base_df, current_cycle)

        extended_rows.append(
            {
                "actual_cycle": int(round(current_cycle)),
                "stress": stress_value,
                "F": request["F"],
                "D": request["D"],
                "Cr": request["Cr"],
                "wear_depth": wear_value,
                "elastic_modulus_GPa": request["elastic_modulus_GPa"],
                "coating_name": request["coating_name"],
                "mode": extension_mode,
            }
        )

        if wear_value >= target_wear_mm:
            break

    if not extended_rows:
        return base_df, None, extension_mode

    extended_df = pd.DataFrame(extended_rows)
    merged_df = pd.concat([base_df, extended_df], ignore_index=True)

    threshold_candidates = merged_df[merged_df["wear_depth"] >= target_wear_mm]
    if threshold_candidates.empty:
        return merged_df, None, extension_mode

    threshold_row = threshold_candidates.iloc[0]
    return merged_df, float(threshold_row["actual_cycle"]), extension_mode


def build_warning_text(request: dict, model_metadata: dict, extension_mode: str | None) -> str | None:
    trained_wear_um = model_metadata["training_wear_limit_um"]
    if request["wear_threshold_um"] <= trained_wear_um + EPS:
        return None
    if extension_mode == "fit_extrapolation":
        return (
            f"当前阈值 {request['wear_threshold_um']:.2f} um 超过训练上限 {trained_wear_um:.2f} um，"
            "已启用曲线拟合外推: y = a*x + b*(1-exp(-c*x))。"
        )
    return (
        f"当前阈值 {request['wear_threshold_um']:.2f} um 超过训练上限 {trained_wear_um:.2f} um，"
        "曲线拟合未稳定，已自动退回尾段线性外推。"
    )


def rollout_prediction(
    runtime: dict,
    request: dict,
    model_metadata: dict,
) -> tuple[pd.DataFrame, float | None, bool, str | None, str]:
    F = request["F"]
    D = request["D"]
    Cr = request["Cr"]
    elastic_modulus_GPa = request["elastic_modulus_GPa"]
    actual_cycle_step = request["actual_cycle_step"]
    trained_limit_um = model_metadata["training_wear_limit_um"]
    recursive_target_um = min(request["wear_threshold_um"], trained_limit_um)
    wear_threshold_mm = recursive_target_um / 1000.0
    wear_depth_mm = 0.0
    coating_name = request["coating_name"]

    actual_cycle = 0.0
    rows = []
    life_actual_cycles = None
    reached = False
    max_actual_cycles = internal_cycle_limit(request, model_metadata)
    extension_mode = "recursive"
    actual_step = actual_cycle_step
    history: list[list[float]] = [[F, D, Cr, actual_cycle, wear_depth_mm]]

    while actual_cycle <= max_actual_cycles + EPS:
        pred_stress = predict_pressure_from_runtime(runtime, history, actual_step)
        rows.append(
            {
                "actual_cycle": int(round(actual_cycle)),
                "stress": pred_stress,
                "F": F,
                "D": D,
                "Cr": Cr,
                "wear_depth": wear_depth_mm,
                "elastic_modulus_GPa": elastic_modulus_GPa,
                "coating_name": coating_name,
                "mode": "recursive",
            }
        )

        delta_s_mm = actual_step * math.pi * D / 6.0
        delta_wear_mm = REAL_WEAR_COEFF_MPA_INV * pred_stress * delta_s_mm
        next_actual_cycle = actual_cycle + actual_step
        next_wear_depth_mm = wear_depth_mm + delta_wear_mm

        if next_wear_depth_mm >= wear_threshold_mm:
            ratio = (wear_threshold_mm - wear_depth_mm) / max(delta_wear_mm, EPS)
            threshold_cycle = actual_cycle + ratio * actual_step
            rows.append(
                {
                    "actual_cycle": int(round(threshold_cycle)),
                    "stress": pred_stress,
                    "F": F,
                    "D": D,
                    "Cr": Cr,
                    "wear_depth": wear_threshold_mm,
                    "elastic_modulus_GPa": elastic_modulus_GPa,
                    "coating_name": coating_name,
                    "mode": "recursive",
                }
            )
            life_actual_cycles = threshold_cycle
            reached = True
            break

        actual_cycle = next_actual_cycle
        wear_depth_mm = next_wear_depth_mm
        history.append([F, D, Cr, actual_cycle, wear_depth_mm])

    prediction_df = pd.DataFrame(rows)

    if request["wear_threshold_um"] > trained_limit_um + EPS and not prediction_df.empty:
        prediction_df, life_actual_cycles, extension_mode = extend_prediction_with_fit(
            prediction_df,
            request,
            model_metadata,
        )
        reached = life_actual_cycles is not None

    warning_text = build_warning_text(request, model_metadata, extension_mode if request["wear_threshold_um"] > trained_limit_um + EPS else None)
    return prediction_df, life_actual_cycles, reached, warning_text, extension_mode


def build_plot(
    prediction_df: pd.DataFrame,
    request: dict,
    life_actual_cycles: float | None,
    reached: bool,
    warning_text: str | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    if "mode" in prediction_df.columns:
        recursive_df = prediction_df[prediction_df["mode"] == "recursive"]
        extension_df = prediction_df[prediction_df["mode"] != "recursive"]
    else:
        recursive_df = prediction_df
        extension_df = prediction_df.iloc[0:0]

    if not recursive_df.empty:
        ax.plot(
            recursive_df["actual_cycle"],
            recursive_df["wear_depth"] * 1000.0,
            linewidth=2.5,
            color="#1f77b4",
            label="Recursive prediction",
        )
    if not extension_df.empty:
        ax.plot(
            extension_df["actual_cycle"],
            extension_df["wear_depth"] * 1000.0,
            linewidth=2.2,
            color="#ff7f0e",
            linestyle="--",
            label="Extrapolated segment",
        )

    ax.axhline(request["wear_threshold_um"], color="#d62728", linestyle="--", linewidth=1.5, label="Wear threshold")

    if reached and life_actual_cycles is not None:
        ax.axvline(life_actual_cycles, color="#2ca02c", linestyle=":", linewidth=1.5, label="Predicted life")
        box_text = (
            f"Predicted life: {life_actual_cycles:.0f} cycles\n"
            f"Threshold wear: {request['wear_threshold_um']:.2f} um\n"
            f"Coating: {request['coating_name']}\n"
            f"F: {request['F']:.1f} N, D: {request['D']:.2f} mm, Cr: {request['Cr']:.4f} mm"
        )
    else:
        box_text = (
            "Threshold not reached before internal safety limit\n"
            f"Current wear: {prediction_df['wear_depth'].iloc[-1] * 1000.0:.3f} um\n"
            f"Coating: {request['coating_name']}\n"
            f"F: {request['F']:.1f} N, D: {request['D']:.2f} mm, Cr: {request['Cr']:.4f} mm"
        )

    if warning_text:
        box_text = f"{box_text}\n{warning_text}"

    x_upper = float(prediction_df["actual_cycle"].max())
    if life_actual_cycles is not None:
        x_upper = max(x_upper, float(life_actual_cycles))
    x_upper *= 1.02

    y_upper = max(
        float((prediction_df["wear_depth"] * 1000.0).max()),
        float(request["wear_threshold_um"]),
    ) * 1.04

    ax.set_xlim(0.0, x_upper)
    ax.set_ylim(0.0, y_upper)
    ax.margins(x=0.0, y=0.0)
    ax.set_title("Bearing Wear Evolution Prediction", fontproperties=PLOT_FONT)
    ax.set_xlabel("Actual Cycle", fontproperties=PLOT_FONT)
    ax.set_ylabel("Wear Depth (um)", fontproperties=PLOT_FONT)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", prop=PLOT_FONT)

    threshold_ratio = min(max(float(request["wear_threshold_um"]) / max(y_upper, EPS), 0.0), 1.0)
    box_top = min(max(threshold_ratio - 0.045, 0.32), 0.90)
    ax.text(
        0.03,
        box_top,
        box_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontproperties=PLOT_FONT,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "gray", "alpha": 0.96},
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.20)
    return fig


def export_prediction_dataframe(prediction_df: pd.DataFrame) -> pd.DataFrame:
    export_df = prediction_df.copy()
    export_df["actual_cycle"] = export_df["actual_cycle"].round().astype(int)
    export_df = export_df[[
        "actual_cycle",
        "stress",
        "F",
        "D",
        "Cr",
        "wear_depth",
        "elastic_modulus_GPa",
        "coating_name",
    ]]
    return export_df


def timestamp_stem(request: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coating_name = str(request["coating_name"]).replace(" ", "_")
    return f"prediction_{coating_name}_{timestamp}"


def ensure_output_dir() -> Path:
    output_dir = project_root() / DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_prediction_artifacts(
    request: dict,
    export_df: pd.DataFrame,
    fig: plt.Figure,
    stem: str | None = None,
) -> tuple[Path, Path]:
    output_dir = ensure_output_dir()
    final_stem = stem or timestamp_stem(request)
    csv_path = output_dir / f"{final_stem}.csv"
    plot_path = output_dir / f"{final_stem}.png"
    export_df.to_csv(csv_path, index=False)
    fig.savefig(plot_path, dpi=300)
    return csv_path, plot_path


def run_prediction_request(request: dict, model_override: Path | None = None):
    normalized_request = normalize_request(request)
    model_path = resolve_model_path(normalized_request["coating_name"], model_override)
    root = project_root()
    model_path = model_path if model_path.is_absolute() else root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    runtime = load_model_runtime(model_path)
    metadata = runtime["metadata"]
    prediction_df, life_actual_cycles, reached, warning_text, extension_mode = rollout_prediction(
        runtime,
        normalized_request,
        metadata,
    )
    export_df = export_prediction_dataframe(prediction_df)
    fig = build_plot(prediction_df, normalized_request, life_actual_cycles, reached, warning_text)
    return normalized_request, export_df, fig, life_actual_cycles, reached, warning_text, metadata, extension_mode


def main() -> None:
    args = parse_args()
    root = project_root()
    input_path = args.input if args.input.is_absolute() else root / args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input file not found: {input_path}")

    request = load_request(input_path)
    normalized_request, export_df, fig, life_actual_cycles, reached, warning_text, _, extension_mode = run_prediction_request(
        request,
        args.model,
    )
    csv_path, plot_path = save_prediction_artifacts(normalized_request, export_df, fig)

    if reached and life_actual_cycles is not None:
        print(f"Predicted life: {life_actual_cycles:.2f} cycles")
    else:
        print("Threshold not reached before internal safety limit")
    print(f"Prediction mode: {extension_mode}")
    if warning_text:
        print(f"Warning: {warning_text}")
    print(f"Saved prediction curve to: {plot_path}")
    print(f"Saved prediction data to: {csv_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
