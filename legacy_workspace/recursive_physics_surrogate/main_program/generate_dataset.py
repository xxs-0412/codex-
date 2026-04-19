from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


REAL_WEAR_COEFF_MPA_INV = 1.84e-10
WEAR_THRESHOLD_MM = 5e-3
CASE_COUNT = 20
RNG_SEED = 20260330

DIAMETER_CANDIDATES_MM = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 17.0]
LOAD_RANGE_N = (1800.0, 6200.0)
CLEARANCE_RANGE_MM = (0.005, 0.050)
ELASTIC_MODULUS_RANGE_GPA = (180.0, 260.0)
SIM_LIFE_RANGE = (18.0, 30.0)
ROW_COUNT_RANGE = (50, 80)
ACTUAL_SWINGS_PER_SIM_CYCLE_RANGE = (3800, 5200)
ACTUAL_LIFE_RANGE = (104_000, 128_000)

ANCHOR_CASES = [
    {
        "F": 4325.0,
        "D": 10.0,
        "Cr": 0.0100,
        "E_GPa": 210.0,
        "row_count": 68,
        "actual_swings_per_sim_cycle": 4680,
        "actual_life_target": 116200,
        "phase": 0.18,
        "curve_bias": 0.010,
        "transient_boost": 0.74,
        "decay_rate": 5.20,
        "residual_floor": 0.47,
    },
    {
        "F": 4200.0,
        "D": 10.0,
        "Cr": 0.0200,
        "E_GPa": 208.0,
        "row_count": 64,
        "actual_swings_per_sim_cycle": 4640,
        "actual_life_target": 113400,
        "phase": 0.31,
        "curve_bias": -0.006,
        "transient_boost": 0.69,
        "decay_rate": 4.80,
        "residual_floor": 0.49,
    },
    {
        "F": 4325.0,
        "D": 10.0,
        "Cr": 0.0300,
        "E_GPa": 210.0,
        "row_count": 70,
        "actual_swings_per_sim_cycle": 4600,
        "actual_life_target": 110800,
        "phase": 0.47,
        "curve_bias": 0.014,
        "transient_boost": 0.78,
        "decay_rate": 5.40,
        "residual_floor": 0.46,
    },
    {
        "F": 4325.0,
        "D": 10.0,
        "Cr": 0.0400,
        "E_GPa": 210.0,
        "row_count": 62,
        "actual_swings_per_sim_cycle": 4580,
        "actual_life_target": 107800,
        "phase": 0.61,
        "curve_bias": -0.012,
        "transient_boost": 0.66,
        "decay_rate": 4.60,
        "residual_floor": 0.50,
    },
    {
        "F": 4325.0,
        "D": 10.0,
        "Cr": 0.0500,
        "E_GPa": 210.0,
        "row_count": 72,
        "actual_swings_per_sim_cycle": 4540,
        "actual_life_target": 104800,
        "phase": 0.73,
        "curve_bias": 0.008,
        "transient_boost": 0.70,
        "decay_rate": 5.00,
        "residual_floor": 0.48,
    },
    {
        "F": 2200.0,
        "D": 10.0,
        "Cr": 0.0100,
        "E_GPa": 210.0,
        "row_count": 66,
        "actual_swings_per_sim_cycle": 4720,
        "actual_life_target": 123200,
        "phase": 0.22,
        "curve_bias": 0.004,
        "transient_boost": 0.62,
        "decay_rate": 4.70,
        "residual_floor": 0.50,
    },
    {
        "F": 3000.0,
        "D": 10.0,
        "Cr": 0.0100,
        "E_GPa": 210.0,
        "row_count": 64,
        "actual_swings_per_sim_cycle": 4680,
        "actual_life_target": 120000,
        "phase": 0.36,
        "curve_bias": -0.004,
        "transient_boost": 0.66,
        "decay_rate": 4.95,
        "residual_floor": 0.49,
    },
    {
        "F": 3800.0,
        "D": 10.0,
        "Cr": 0.0100,
        "E_GPa": 210.0,
        "row_count": 62,
        "actual_swings_per_sim_cycle": 4640,
        "actual_life_target": 117200,
        "phase": 0.44,
        "curve_bias": 0.006,
        "transient_boost": 0.70,
        "decay_rate": 5.10,
        "residual_floor": 0.48,
    },
    {
        "F": 5000.0,
        "D": 10.0,
        "Cr": 0.0100,
        "E_GPa": 210.0,
        "row_count": 60,
        "actual_swings_per_sim_cycle": 4560,
        "actual_life_target": 110200,
        "phase": 0.58,
        "curve_bias": -0.008,
        "transient_boost": 0.74,
        "decay_rate": 5.30,
        "residual_floor": 0.47,
    },
    {
        "F": 5000.0,
        "D": 10.0,
        "Cr": 0.0200,
        "E_GPa": 210.0,
        "row_count": 60,
        "actual_swings_per_sim_cycle": 4540,
        "actual_life_target": 107400,
        "phase": 0.60,
        "curve_bias": -0.006,
        "transient_boost": 0.75,
        "decay_rate": 5.32,
        "residual_floor": 0.47,
    },
    {
        "F": 5000.0,
        "D": 10.0,
        "Cr": 0.0300,
        "E_GPa": 210.0,
        "row_count": 59,
        "actual_swings_per_sim_cycle": 4520,
        "actual_life_target": 106000,
        "phase": 0.62,
        "curve_bias": -0.006,
        "transient_boost": 0.75,
        "decay_rate": 5.35,
        "residual_floor": 0.47,
    },
    {
        "F": 5000.0,
        "D": 10.0,
        "Cr": 0.0400,
        "E_GPa": 210.0,
        "row_count": 58,
        "actual_swings_per_sim_cycle": 4510,
        "actual_life_target": 104800,
        "phase": 0.64,
        "curve_bias": -0.008,
        "transient_boost": 0.76,
        "decay_rate": 5.38,
        "residual_floor": 0.46,
    },
    {
        "F": 5000.0,
        "D": 10.0,
        "Cr": 0.0500,
        "E_GPa": 210.0,
        "row_count": 58,
        "actual_swings_per_sim_cycle": 4490,
        "actual_life_target": 104000,
        "phase": 0.66,
        "curve_bias": -0.010,
        "transient_boost": 0.77,
        "decay_rate": 5.42,
        "residual_floor": 0.46,
    },
    {
        "F": 6000.0,
        "D": 10.0,
        "Cr": 0.0100,
        "E_GPa": 210.0,
        "row_count": 58,
        "actual_swings_per_sim_cycle": 4480,
        "actual_life_target": 104600,
        "phase": 0.67,
        "curve_bias": -0.010,
        "transient_boost": 0.78,
        "decay_rate": 5.50,
        "residual_floor": 0.46,
    },
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def dataset_dir() -> Path:
    return project_root() / "data" / "current_dataset"


def reset_dataset_dir() -> None:
    dataset_dir().mkdir(parents=True, exist_ok=True)
    for csv_file in dataset_dir().glob("*.csv"):
        csv_file.unlink()


def finalize_case_parameters(base_params: dict) -> dict:
    actual_swings_per_sim_cycle = int(base_params["actual_swings_per_sim_cycle"])
    actual_life_target = int(base_params["actual_life_target"])
    sim_life = float(base_params.get("sim_life", np.round(actual_life_target / actual_swings_per_sim_cycle, 3)))
    return {
        "F": float(base_params["F"]),
        "D": float(base_params["D"]),
        "Cr": float(base_params["Cr"]),
        "E_GPa": float(base_params["E_GPa"]),
        "sim_life": sim_life,
        "row_count": int(base_params["row_count"]),
        "actual_swings_per_sim_cycle": actual_swings_per_sim_cycle,
        "actual_life_target": actual_life_target,
        "k_real": REAL_WEAR_COEFF_MPA_INV,
        "k_sim": REAL_WEAR_COEFF_MPA_INV * actual_swings_per_sim_cycle,
        "phase": float(base_params["phase"]),
        "curve_bias": float(base_params["curve_bias"]),
        "transient_boost": float(base_params.get("transient_boost", 0.66)),
        "decay_rate": float(base_params.get("decay_rate", 4.8)),
        "residual_floor": float(base_params.get("residual_floor", 0.50)),
    }


def anchored_case_parameters(case_index: int) -> dict | None:
    if 1 <= case_index <= len(ANCHOR_CASES):
        return finalize_case_parameters(ANCHOR_CASES[case_index - 1])
    return None


def severity_score(F: float, D: float, Cr: float) -> float:
    load_norm = (F - LOAD_RANGE_N[0]) / (LOAD_RANGE_N[1] - LOAD_RANGE_N[0])
    diameter_norm = (D - min(DIAMETER_CANDIDATES_MM)) / (
        max(DIAMETER_CANDIDATES_MM) - min(DIAMETER_CANDIDATES_MM)
    )
    clearance_norm = (Cr - CLEARANCE_RANGE_MM[0]) / (
        CLEARANCE_RANGE_MM[1] - CLEARANCE_RANGE_MM[0]
    )
    return 0.95 * load_norm + 1.15 * clearance_norm + 0.25 * (1.0 - diameter_norm)


def target_life_from_params(F: float, D: float, Cr: float, rng: np.random.Generator) -> int:
    severity = severity_score(F, D, Cr)
    base_life = 127_000.0 - 14_500.0 * severity
    noise = float(rng.normal(0.0, 250.0))
    target_life = int(round(base_life + noise))
    return int(np.clip(target_life, ACTUAL_LIFE_RANGE[0], ACTUAL_LIFE_RANGE[1]))


def sample_case_parameters(rng: np.random.Generator) -> dict:
    if rng.random() < 0.45:
        clearance_value = float(np.round(rng.uniform(0.028, CLEARANCE_RANGE_MM[1]), 4))
    else:
        clearance_value = float(np.round(rng.uniform(CLEARANCE_RANGE_MM[0], 0.028), 4))

    F = float(rng.integers(int(LOAD_RANGE_N[0]), int(LOAD_RANGE_N[1]) + 1))
    D = float(rng.choice(DIAMETER_CANDIDATES_MM))
    actual_life_target = target_life_from_params(F, D, clearance_value, rng)

    while True:
        row_count = int(rng.integers(ROW_COUNT_RANGE[0], ROW_COUNT_RANGE[1] + 1))
        actual_swings_per_sim_cycle = int(
            rng.integers(
                ACTUAL_SWINGS_PER_SIM_CYCLE_RANGE[0],
                ACTUAL_SWINGS_PER_SIM_CYCLE_RANGE[1] + 1,
            )
        )
        sim_life = actual_life_target / actual_swings_per_sim_cycle
        if SIM_LIFE_RANGE[0] <= sim_life <= SIM_LIFE_RANGE[1]:
            break

    return finalize_case_parameters(
        {
            "F": F,
            "D": D,
            "Cr": clearance_value,
            "E_GPa": float(np.round(rng.uniform(*ELASTIC_MODULUS_RANGE_GPA), 1)),
            "sim_life": float(np.round(sim_life, 3)),
            "row_count": row_count,
            "actual_swings_per_sim_cycle": actual_swings_per_sim_cycle,
            "actual_life_target": actual_life_target,
            "phase": float(rng.uniform(0.0, 1.0)),
            "curve_bias": float(rng.uniform(-0.04, 0.04)),
            "transient_boost": float(rng.uniform(0.52, 0.82)),
            "decay_rate": float(rng.uniform(4.0, 6.2)),
            "residual_floor": float(rng.uniform(0.44, 0.56)),
        }
    )


def static_factor(F: float, D: float, Cr: float) -> float:
    load_norm = (F - LOAD_RANGE_N[0]) / (LOAD_RANGE_N[1] - LOAD_RANGE_N[0])
    diameter_norm = (D - min(DIAMETER_CANDIDATES_MM)) / (
        max(DIAMETER_CANDIDATES_MM) - min(DIAMETER_CANDIDATES_MM)
    )
    clearance_norm = (Cr - CLEARANCE_RANGE_MM[0]) / (
        CLEARANCE_RANGE_MM[1] - CLEARANCE_RANGE_MM[0]
    )

    load_term = 0.90 + 0.35 * load_norm
    size_term = 1.08 - 0.20 * diameter_norm
    clearance_term = 0.94 + 0.18 * clearance_norm
    return load_term * size_term * clearance_term


def simulate_case_rows(scale_mpa: float, params: dict) -> list[dict]:
    sim_life = params["sim_life"]
    row_count = params["row_count"]
    actual_swings_per_sim_cycle = params["actual_swings_per_sim_cycle"]
    actual_life_target = params["actual_life_target"]
    F = params["F"]
    D = params["D"]
    Cr = params["Cr"]
    E_GPa = params["E_GPa"]
    k_real = params["k_real"]
    k_sim = params["k_sim"]
    curve_bias = params["curve_bias"]
    transient_boost = params["transient_boost"]
    decay_rate = params["decay_rate"]
    residual_floor = params["residual_floor"]

    sim_cycle_values = np.linspace(0.0, sim_life, row_count, dtype=np.float64)
    actual_cycle_values = np.rint(sim_cycle_values * actual_swings_per_sim_cycle).astype(np.int64)
    actual_cycle_values[-1] = actual_life_target

    pressure_base = static_factor(F, D, Cr)
    wear_depth = 0.0
    rows: list[dict] = []

    for index, sim_cycle in enumerate(sim_cycle_values):
        actual_cycle = int(actual_cycle_values[index])
        actual_norm = actual_cycle / actual_life_target

        effective_boost = max(0.18, transient_boost * (1.0 + 0.35 * curve_bias))
        effective_floor = max(0.52, residual_floor + 0.10 * curve_bias)
        dynamic_factor = effective_floor + effective_boost * math.exp(-decay_rate * actual_norm)
        stress = scale_mpa * pressure_base * dynamic_factor

        rows.append(
            {
                "sim_cycle": float(np.round(sim_cycle, 6)),
                "actual_cycle": actual_cycle,
                "stress": stress,
                "F": F,
                "D": D,
                "Cr": Cr,
                "wear_depth": wear_depth,
                "elastic_modulus_GPa": E_GPa,
                "k_sim": k_sim,
            }
        )

        if index < row_count - 1:
            next_actual_cycle = int(actual_cycle_values[index + 1])
            delta_actual_cycle = next_actual_cycle - actual_cycle
            delta_sliding_distance_mm = delta_actual_cycle * math.pi * D / 6.0
            delta_wear = k_real * stress * delta_sliding_distance_mm
            wear_depth += delta_wear

    return rows


def calibrate_pressure_scale(params: dict) -> list[dict]:
    lower = 5.0
    upper = 250.0

    for _ in range(80):
        middle = 0.5 * (lower + upper)
        rows = simulate_case_rows(middle, params)
        final_wear = rows[-1]["wear_depth"]
        if final_wear < WEAR_THRESHOLD_MM:
            lower = middle
        else:
            upper = middle

    return simulate_case_rows(upper, params)


def save_case(case_index: int, rows: list[dict]) -> dict:
    file_name = f"sim_{case_index}.csv"
    file_path = dataset_dir() / file_name
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)

    last_row = df.iloc[-1]
    return {
        "file_name": file_name,
        "F": float(last_row["F"]),
        "D": float(last_row["D"]),
        "Cr": float(last_row["Cr"]),
        "elastic_modulus_GPa": float(last_row["elastic_modulus_GPa"]),
        "k_sim": float(last_row["k_sim"]),
        "row_count": int(len(df)),
        "sim_life": float(last_row["sim_cycle"]),
        "actual_life": int(last_row["actual_cycle"]),
        "final_wear_depth": float(last_row["wear_depth"]),
        "final_wear_um": float(last_row["wear_depth"] * 1000.0),
        "stress_start": float(df.iloc[0]["stress"]),
        "stress_end": float(last_row["stress"]),
    }


def main() -> None:
    reset_dataset_dir()
    rng = np.random.default_rng(RNG_SEED)

    metadata_rows = []
    combined_rows = []

    for case_index in range(1, CASE_COUNT + 1):
        params = anchored_case_parameters(case_index) or sample_case_parameters(rng)
        rows = calibrate_pressure_scale(params)
        metadata_rows.append(save_case(case_index, rows))

        case_df = pd.DataFrame(rows).copy()
        case_df.insert(0, "file_name", f"sim_{case_index}.csv")
        combined_rows.append(case_df)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(dataset_dir() / "dataset_summary.csv", index=False)

    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_df.to_csv(dataset_dir() / "all_data.csv", index=False)

    print(f"Generated {CASE_COUNT} simple sim files in: {dataset_dir()}")
    print(
        "Actual life range at 5 um threshold: "
        f"{int(metadata_df['actual_life'].min())} - "
        f"{int(metadata_df['actual_life'].max())} cycles"
    )
    print(
        "Row count range: "
        f"{int(metadata_df['row_count'].min())} - {int(metadata_df['row_count'].max())}"
    )
    print(
        "Simulated wear coefficient range: "
        f"{metadata_df['k_sim'].min():.3e} - "
        f"{metadata_df['k_sim'].max():.3e} MPa^-1"
    )


if __name__ == "__main__":
    main()
