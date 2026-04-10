from __future__ import annotations

import argparse
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


DEFAULT_REAL_WEAR_COEFF_MPA_INV = 1.84e-10
DEFAULT_ELASTIC_MODULUS_GPA = 210.0
DEFAULT_WEAR_THRESHOLD_UM = 5.0
DEFAULT_TEST_CASE = "试验8"
EPS = 1e-12


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def workspace_root() -> Path:
    return project_root().parent


def default_source_dir() -> Path:
    return workspace_root() / "磨损数据"


def default_target_root() -> Path:
    return workspace_root() / "磨损数据（改）"


def comparison_dir() -> Path:
    return workspace_root() / "结果"


def trained_model_dir() -> Path:
    return workspace_root() / "工具和杂项" / "训练模型"


def logs_dir() -> Path:
    return workspace_root() / "工具和杂项" / "运行日志"


def default_config_path() -> Path:
    return workspace_root() / "工具和杂项" / "脚本与配置" / "处理参数配置.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw wear xlsx files into the standard recursive-training format.")
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--target-root", type=Path, default=default_target_root())
    parser.add_argument("--config", type=Path, default=default_config_path())
    return parser.parse_args()


def default_config_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "real_wear_coeff_mpa_inv": DEFAULT_REAL_WEAR_COEFF_MPA_INV,
                "default_elastic_modulus_GPa": DEFAULT_ELASTIC_MODULUS_GPA,
                "wear_threshold_um": DEFAULT_WEAR_THRESHOLD_UM,
                "test_case_name": DEFAULT_TEST_CASE,
            }
        ]
    )


def ensure_workspace(target_root: Path, config_path: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    comparison_dir().mkdir(parents=True, exist_ok=True)
    trained_model_dir().mkdir(parents=True, exist_ok=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        default_config_frame().to_csv(config_path, index=False)


def load_config(config_path: Path) -> dict:
    df = pd.read_csv(config_path)
    if df.empty:
        raise ValueError(f"Config file is empty: {config_path}")
    row = df.iloc[0].to_dict()
    return {
        "real_wear_coeff_mpa_inv": float(row.get("real_wear_coeff_mpa_inv", DEFAULT_REAL_WEAR_COEFF_MPA_INV)),
        "default_elastic_modulus_GPa": float(row.get("default_elastic_modulus_GPa", DEFAULT_ELASTIC_MODULUS_GPA)),
        "wear_threshold_um": float(row.get("wear_threshold_um", DEFAULT_WEAR_THRESHOLD_UM)),
        "test_case_name": str(row.get("test_case_name", DEFAULT_TEST_CASE)).strip() or DEFAULT_TEST_CASE,
    }


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    normalized = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    out = df.copy()
    out.columns = normalized
    return out


def read_text_table(source_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return normalize_headers(pd.read_csv(source_path, encoding=encoding))
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"Could not read text table: {source_path}") from last_error


def read_source_table(source_path: Path) -> pd.DataFrame:
    suffix = source_path.suffix.lower()
    header = source_path.read_bytes()[:4]
    is_zip_excel = header.startswith(b"PK\x03\x04")

    if suffix in {".xlsx", ".xlsm", ".xls"} and is_zip_excel:
        try:
            return normalize_headers(pd.read_excel(source_path, engine="openpyxl"))
        except (BadZipFile, ValueError):
            pass
    return read_text_table(source_path)


def metadata_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    columns = list(df.columns)
    for required in ["F", "D", "Cr"]:
        if required not in columns:
            raise KeyError(f"Missing required metadata column: {required}")
    f_idx = columns.index("F")
    if f_idx == 0:
        raise ValueError("Could not infer k_sim column because F is the first column.")
    return columns[f_idx - 1], "F", "D", "Cr"


def first_numeric_value(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric value found in column: {column}")
    return float(values.iloc[0])


def strict_round_cycles(values: np.ndarray) -> np.ndarray:
    rounded = np.rint(values).astype(np.int64)
    if len(rounded) == 0:
        return rounded
    rounded[0] = 0
    for idx in range(1, len(rounded)):
        rounded[idx] = max(int(rounded[idx]), int(rounded[idx - 1]) + 1)
    return rounded


def derive_segment_stress(sim_cycle: np.ndarray, wear_depth_mm: np.ndarray, diameter_mm: float, k_sim: float) -> np.ndarray:
    sim0 = np.concatenate([[0.0], sim_cycle.astype(float)])
    wear0 = np.concatenate([[0.0], wear_depth_mm.astype(float)])
    delta_sim = np.diff(sim0)
    delta_wear = np.diff(wear0)
    delta_s = delta_sim * math.pi * float(diameter_mm) / 6.0
    stress = delta_wear / np.maximum(float(k_sim) * delta_s, EPS)
    return np.clip(stress, EPS, None)


def wear_fit_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x + b * (1.0 - np.exp(-c * x))


def estimate_threshold_life(
    actual_cycle_float: np.ndarray,
    wear_depth_mm: np.ndarray,
    threshold_um: float,
) -> tuple[float, str]:
    target_mm = threshold_um / 1000.0
    actual = np.asarray(actual_cycle_float, dtype=float)
    wear = np.asarray(wear_depth_mm, dtype=float)

    if wear[0] > EPS:
        actual = np.concatenate([[0.0], actual])
        wear = np.concatenate([[0.0], wear])

    reached_idx = np.where(wear >= target_mm - EPS)[0]
    if len(reached_idx) > 0:
        idx = int(reached_idx[0])
        if idx == 0:
            return float(actual[0]), "measured"
        x0, x1 = float(actual[idx - 1]), float(actual[idx])
        y0, y1 = float(wear[idx - 1]), float(wear[idx])
        ratio = (target_mm - y0) / max(y1 - y0, EPS)
        return float(x0 + ratio * (x1 - x0)), "measured"

    try:
        tail_count = min(len(actual), 12)
        tail_slope = np.polyfit(actual[-tail_count:], wear[-tail_count:], 1)[0] if tail_count >= 2 else 1e-8
        p0 = [max(float(tail_slope), 1e-9), max(float(wear[-1] - wear[0]), 1e-6), 1e-5]
        params, _ = curve_fit(
            wear_fit_function,
            actual,
            wear,
            p0=p0,
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=50000,
        )
        lo = float(actual[-1])
        hi = max(lo * 1.5, lo + 1000.0)
        while wear_fit_function(np.array([hi]), *params)[0] < target_mm:
            hi *= 1.5
            if hi > max(lo * 200.0, lo + 5e7):
                break
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if wear_fit_function(np.array([mid]), *params)[0] >= target_mm:
                hi = mid
            else:
                lo = mid
        return float(hi), "fit"
    except Exception:
        tail_count = min(len(actual), 12)
        slope = np.polyfit(actual[-tail_count:], wear[-tail_count:], 1)[0] if tail_count >= 2 else 1e-8
        slope = max(float(slope), 1e-9)
        extra_cycle = max(target_mm - float(wear[-1]), 0.0) / slope
        return float(actual[-1] + extra_cycle), "linear"


def build_case_table(source_path: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    raw_df = read_source_table(source_path)
    columns = list(raw_df.columns)

    sim_column = "Step_Time"
    wear_column = "CWEAR" if "CWEAR" in columns else "Max_CWEAR"
    stress_column = "CPRESS" if "CPRESS" in columns else ("Max_CPRESS" if "Max_CPRESS" in columns else None)
    k_column, f_column, d_column, cr_column = metadata_columns(raw_df)

    sim_values = pd.to_numeric(raw_df[sim_column], errors="coerce").dropna().to_numpy(dtype=float)
    wear_values = pd.to_numeric(raw_df[wear_column], errors="coerce").dropna().to_numpy(dtype=float)
    if len(sim_values) != len(wear_values):
        raise ValueError(f"Length mismatch in {source_path.name}: sim={len(sim_values)}, wear={len(wear_values)}")

    k_sim = first_numeric_value(raw_df, k_column)
    load_f = first_numeric_value(raw_df, f_column)
    diameter_d = first_numeric_value(raw_df, d_column)
    clearance_cr = first_numeric_value(raw_df, cr_column)
    elastic_modulus = float(config["default_elastic_modulus_GPa"])
    real_k = float(config["real_wear_coeff_mpa_inv"])

    if stress_column is not None:
        stress_values = pd.to_numeric(raw_df[stress_column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(stress_values) != len(sim_values):
            raise ValueError(f"Length mismatch in {source_path.name}: sim={len(sim_values)}, stress={len(stress_values)}")
        stress_source = "measured_cpress"
    else:
        stress_values = derive_segment_stress(sim_values, wear_values, diameter_d, k_sim)
        stress_source = "derived_from_wear"

    sim_cycle = np.concatenate([[0.0], sim_values])
    wear_depth_mm = np.concatenate([[0.0], wear_values])
    stress = np.concatenate([[float(stress_values[0])], stress_values])
    actual_cycle_float = sim_cycle * k_sim / real_k
    actual_cycle = strict_round_cycles(actual_cycle_float)

    case_df = pd.DataFrame(
        {
            "sim_cycle": sim_cycle,
            "actual_cycle": actual_cycle,
            "stress": np.clip(stress, EPS, None),
            "F": load_f,
            "D": diameter_d,
            "Cr": clearance_cr,
            "wear_depth": wear_depth_mm,
            "elastic_modulus_GPa": elastic_modulus,
            "k_sim": k_sim,
            "source_file": source_path.name,
            "stress_source": stress_source,
        }
    )

    actual_life, life_mode = estimate_threshold_life(actual_cycle_float, wear_depth_mm, float(config["wear_threshold_um"]))
    sim_life = actual_life * real_k / k_sim

    summary_row = {
        "file_name": f"{source_path.stem}.csv",
        "source_file": source_path.name,
        "F": load_f,
        "D": diameter_d,
        "Cr": clearance_cr,
        "elastic_modulus_GPa": elastic_modulus,
        "k_sim": k_sim,
        "real_wear_coeff_mpa_inv": real_k,
        "row_count": int(len(case_df)),
        "sim_life": sim_life,
        "actual_life": actual_life,
        "actual_life_mode": life_mode,
        "final_sim_cycle": float(sim_cycle[-1]),
        "final_actual_cycle": float(actual_cycle_float[-1]),
        "final_wear_depth": float(wear_depth_mm[-1]),
        "final_wear_um": float(wear_depth_mm[-1] * 1000.0),
        "stress_start": float(case_df.iloc[0]["stress"]),
        "stress_end": float(case_df.iloc[-1]["stress"]),
        "has_measured_pressure": bool(stress_column is not None),
        "stress_source": stress_source,
        "wear_threshold_um": float(config["wear_threshold_um"]),
    }
    return case_df, summary_row


def main() -> None:
    args = parse_args()
    ensure_workspace(args.target_root, args.config)
    config = load_config(args.config)

    processed_dir = args.target_root
    for path in processed_dir.glob("*.csv"):
        path.unlink()

    summary_rows: list[dict] = []

    priorities = {".xlsx": 3, ".xlsm": 3, ".xls": 2, ".csv": 1}
    source_candidates = [
        path
        for path in args.source_dir.iterdir()
        if path.is_file() and not path.name.startswith("~$") and path.suffix.lower() in priorities
    ]
    selected_by_stem: dict[str, Path] = {}
    for path in sorted(source_candidates, key=lambda item: item.name.lower()):
        key = path.stem.lower()
        current = selected_by_stem.get(key)
        if current is None:
            selected_by_stem[key] = path
            continue
        current_rank = (priorities[current.suffix.lower()], current.stat().st_mtime)
        next_rank = (priorities[path.suffix.lower()], path.stat().st_mtime)
        if next_rank > current_rank:
            selected_by_stem[key] = path
    source_files = sorted(selected_by_stem.values(), key=lambda item: item.name.lower())
    if not source_files:
        raise FileNotFoundError(f"No supported source tables found in {args.source_dir}")

    for source_path in source_files:
        case_df, summary_row = build_case_table(source_path, config)
        case_path = processed_dir / summary_row["file_name"]
        case_df.to_csv(case_path, index=False)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("source_file").reset_index(drop=True)
    summary_df.to_csv(processed_dir / "修改数据汇总.csv", index=False)

    log_lines = [
        "修改后磨损数据处理完成。",
        f"原始数据目录: {args.source_dir}",
        f"修改数据目录: {processed_dir}",
        f"real_wear_coeff_mpa_inv={config['real_wear_coeff_mpa_inv']:.12e}",
        f"default_elastic_modulus_GPa={config['default_elastic_modulus_GPa']:.4f}",
        f"wear_threshold_um={config['wear_threshold_um']:.4f}",
        f"test_case_name={config['test_case_name']}",
        f"处理工况数量: {len(summary_df)}",
    ]
    (logs_dir() / "修改数据处理日志.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("\n".join(log_lines))
    print(summary_df[["source_file", "file_name", "actual_life", "actual_life_mode", "final_wear_um", "stress_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
