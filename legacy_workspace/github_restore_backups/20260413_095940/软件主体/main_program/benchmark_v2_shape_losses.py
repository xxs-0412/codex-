from __future__ import annotations

import argparse
import json
from pathlib import Path

import v2_pipeline as v2


DEFAULT_RUN_STRESS_MONO_LAMBDA = 0.05
DEFAULT_RUN_STRESS_SLOW_LAMBDA = 0.01
BASE_SHAPE_CONFIG = v2.make_v2_dual_branch_candidate(
    sequence_length=10,
    wear_consistency_lambda=0.25,
    use_cycle_step=True,
    seed=v2.PRIMARY_SEED,
)


def to_builtin(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=to_builtin), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V2 shape-loss benchmark in lite or full mode.")
    parser.add_argument("--mode", choices=["lite", "full"], default="lite", help="Benchmark mode. Defaults to lite screening.")
    parser.add_argument(
        "--screening-folds",
        type=int,
        default=v2.LITE_SCREENING_FOLD_COUNT,
        help="Number of representative folds to keep in lite mode.",
    )
    return parser.parse_args()


def build_shape_loss_configs(base_config: v2.CandidateConfig) -> list[v2.CandidateConfig]:
    base_name = str(base_config.name)
    return [
        base_config.with_updates(
            name=f"baseline|{base_name}",
            run_stress_mono_lambda=0.0,
            run_stress_slow_lambda=0.0,
        ),
        base_config.with_updates(
            name=f"shape_loss_v1|{base_name}",
            run_stress_mono_lambda=DEFAULT_RUN_STRESS_MONO_LAMBDA,
            run_stress_slow_lambda=0.0,
        ),
        base_config.with_updates(
            name=f"shape_loss_v2|{base_name}",
            run_stress_mono_lambda=DEFAULT_RUN_STRESS_MONO_LAMBDA,
            run_stress_slow_lambda=DEFAULT_RUN_STRESS_SLOW_LAMBDA,
        ),
    ]


def run_lite(
    summary_df,
    case_tables,
    threshold_um: float,
    real_k: float,
    full_manifest: dict,
    screening_folds: int,
) -> None:
    out_dir = v2.v2_lite_shape_loss_output_dir()
    screening_manifest = v2.build_screening_manifest(full_manifest, summary_df, threshold_um, screening_folds)
    base_config = v2.make_lite_config(BASE_SHAPE_CONFIG)
    configs = build_shape_loss_configs(base_config)

    write_json(
        out_dir / "benchmark_mode.json",
        {
            "mode": "lite",
            "screening_folds": int(screening_folds),
            "note": "Lite shape-loss screening only. Do not use these results for deployment replacement decisions.",
        },
    )
    write_json(out_dir / "shape_loss_manifest_full.json", full_manifest)
    write_json(out_dir / "shape_loss_manifest_lite.json", screening_manifest)
    write_json(out_dir / "shape_loss_base_config.json", base_config.to_record())

    summary_df_out, fold_df_out = v2.evaluate_candidate_suite(
        configs=configs,
        manifest=screening_manifest,
        summary_df=summary_df,
        case_tables=case_tables,
        threshold_um=threshold_um,
        real_k=real_k,
    )
    ranked_summary = v2.rank_summary_df(summary_df_out)
    ranked_summary.to_csv(out_dir / "shape_loss_summary.csv", index=False, encoding="utf-8-sig")
    fold_df_out.to_csv(out_dir / "shape_loss_folds.csv", index=False, encoding="utf-8-sig")

    winner_payload = dict(ranked_summary.iloc[0].to_dict())
    winner_payload.update(
        {
            "mode": "lite",
            "deployment_ready": False,
            "message": "Lite shape-loss winner only. Confirm in full benchmark before using it in deployment decisions.",
        }
    )
    write_json(out_dir / "screening_recommendation.json", winner_payload)

    print(f"Saved lite shape-loss benchmark artifacts to: {out_dir}")
    print(
        ranked_summary[
            [
                "rank",
                "candidate_name",
                "run_stress_mono_lambda",
                "run_stress_slow_lambda",
                "max_epochs",
                "patience",
                "mean_life_rel_error",
                "median_life_abs_error",
                "mean_pressure_mae",
            ]
        ].to_string(index=False)
    )
    print("Lite shape-loss result is for screening only and must not replace the formal software model.")


def run_full(summary_df, case_tables, threshold_um: float, real_k: float, manifest: dict) -> None:
    out_dir = v2.v2_shape_loss_output_dir()
    configs = build_shape_loss_configs(BASE_SHAPE_CONFIG)

    write_json(out_dir / "benchmark_mode.json", {"mode": "full"})
    write_json(out_dir / "shape_loss_manifest.json", manifest)
    write_json(out_dir / "shape_loss_base_config.json", BASE_SHAPE_CONFIG.to_record())

    summary_df_out, fold_df_out = v2.evaluate_candidate_suite(
        configs=configs,
        manifest=manifest,
        summary_df=summary_df,
        case_tables=case_tables,
        threshold_um=threshold_um,
        real_k=real_k,
    )
    ranked_summary = v2.rank_summary_df(summary_df_out)
    ranked_summary.to_csv(out_dir / "shape_loss_summary.csv", index=False, encoding="utf-8-sig")
    fold_df_out.to_csv(out_dir / "shape_loss_folds.csv", index=False, encoding="utf-8-sig")

    print(f"Saved full shape-loss benchmark artifacts to: {out_dir}")
    print(
        ranked_summary[
            [
                "rank",
                "candidate_name",
                "run_stress_mono_lambda",
                "run_stress_slow_lambda",
                "mean_life_rel_error",
                "median_life_abs_error",
                "mean_pressure_mae",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    _, device_label = v2.resolve_training_device()
    print(f"device: {device_label}")
    config, summary_df, case_tables = v2.load_environment()
    threshold_um = float(config["wear_threshold_um"])
    real_k = float(config["real_wear_coeff_mpa_inv"])
    manifest = v2.load_or_create_split_manifest(summary_df, threshold_um)

    if args.mode == "lite":
        run_lite(summary_df, case_tables, threshold_um, real_k, manifest, args.screening_folds)
        return
    run_full(summary_df, case_tables, threshold_um, real_k, manifest)


if __name__ == "__main__":
    main()
