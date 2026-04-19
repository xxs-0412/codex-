from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import v2_pipeline as v2


def to_builtin(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=to_builtin), encoding="utf-8")


def config_map(configs: list[v2.CandidateConfig]) -> dict[str, v2.CandidateConfig]:
    return {config.name: config for config in configs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V2 candidate benchmark in lite or full mode.")
    parser.add_argument("--mode", choices=["lite", "full"], default="lite", help="Benchmark mode. Defaults to lite screening.")
    parser.add_argument(
        "--screening-folds",
        type=int,
        default=v2.LITE_SCREENING_FOLD_COUNT,
        help="Number of representative folds to keep in lite mode.",
    )
    return parser.parse_args()


def run_lite(
    summary_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
    threshold_um: float,
    real_k: float,
    full_manifest: dict,
    screening_folds: int,
) -> None:
    out_dir = v2.v2_lite_candidate_output_dir()
    screening_manifest = v2.build_screening_manifest(full_manifest, summary_df, threshold_um, screening_folds)

    write_json(
        out_dir / "benchmark_mode.json",
        {
            "mode": "lite",
            "screening_folds": int(screening_folds),
            "note": "Lite screening only. Do not use these results for deployment replacement decisions.",
        },
    )
    write_json(out_dir / "fixed_split_manifest_full.json", full_manifest)
    write_json(out_dir / "fixed_split_manifest_lite.json", screening_manifest)

    consistency_df = v2.run_feature_consistency_suite(case_tables, summary_df)
    consistency_df.to_csv(out_dir / "feature_consistency_checks.csv", index=False, encoding="utf-8-sig")

    lite_configs = v2.build_lite_round1_candidates()
    lite_summary_df, lite_fold_df = v2.evaluate_candidate_suite(
        configs=lite_configs,
        manifest=screening_manifest,
        summary_df=summary_df,
        case_tables=case_tables,
        threshold_um=threshold_um,
        real_k=real_k,
    )
    lite_summary_df = v2.rank_summary_df(lite_summary_df)
    lite_summary_df.to_csv(out_dir / "lite_structure_summary.csv", index=False, encoding="utf-8-sig")
    lite_fold_df.to_csv(out_dir / "lite_structure_folds.csv", index=False, encoding="utf-8-sig")

    winner_row = lite_summary_df.iloc[0].to_dict()
    winner_payload = dict(winner_row)
    winner_payload.update(
        {
            "mode": "lite",
            "deployment_ready": False,
            "message": "Lite screening winner only. Run full benchmark before any deployment decision.",
        }
    )
    write_json(out_dir / "screening_recommendation.json", winner_payload)

    print(f"Saved lite V2 screening artifacts to: {out_dir}")
    print("Lite structure ranking:")
    print(
        lite_summary_df[
            [
                "rank",
                "candidate_name",
                "max_epochs",
                "patience",
                "mean_life_rel_error",
                "median_life_abs_error",
                "mean_pressure_mae",
            ]
        ].to_string(index=False)
    )
    print("Lite winner is for screening only and must not replace the formal software model.")


def run_full(
    summary_df: pd.DataFrame,
    case_tables: dict[str, pd.DataFrame],
    threshold_um: float,
    real_k: float,
    manifest: dict,
) -> None:
    out_dir = v2.v2_candidate_output_dir()

    write_json(out_dir / "benchmark_mode.json", {"mode": "full"})
    write_json(out_dir / "v1_frozen_baseline_config.json", v2.frozen_v1_baseline_definition())
    write_json(out_dir / "fixed_split_manifest.json", manifest)

    consistency_df = v2.run_feature_consistency_suite(case_tables, summary_df)
    consistency_df.to_csv(out_dir / "feature_consistency_checks.csv", index=False, encoding="utf-8-sig")

    frozen_config = v2.legacy_v1_baseline_config()
    frozen_summary_df, frozen_fold_df = v2.evaluate_candidate_suite(
        configs=[frozen_config],
        manifest=manifest,
        summary_df=summary_df,
        case_tables=case_tables,
        threshold_um=threshold_um,
        real_k=real_k,
    )
    frozen_summary_df = v2.rank_summary_df(frozen_summary_df)
    frozen_summary_df.to_csv(out_dir / "round0_v1_frozen_summary.csv", index=False, encoding="utf-8-sig")
    frozen_fold_df.to_csv(out_dir / "round0_v1_frozen_folds.csv", index=False, encoding="utf-8-sig")

    round1_configs = v2.build_round1_candidates()
    round1_summary_df, round1_fold_df = v2.evaluate_candidate_suite(
        configs=round1_configs,
        manifest=manifest,
        summary_df=summary_df,
        case_tables=case_tables,
        threshold_um=threshold_um,
        real_k=real_k,
    )
    round1_summary_df = v2.rank_summary_df(round1_summary_df)
    round1_summary_df.to_csv(out_dir / "round1_structure_summary.csv", index=False, encoding="utf-8-sig")
    round1_fold_df.to_csv(out_dir / "round1_structure_folds.csv", index=False, encoding="utf-8-sig")

    round1_winner_name = str(round1_summary_df.iloc[0]["candidate_name"])
    round1_winner_config = config_map(round1_configs)[round1_winner_name]
    round2_configs = v2.build_round2_candidates(round1_winner_config)
    round2_summary_df, round2_fold_df = v2.evaluate_candidate_suite(
        configs=round2_configs,
        manifest=manifest,
        summary_df=summary_df,
        case_tables=case_tables,
        threshold_um=threshold_um,
        real_k=real_k,
    )
    round2_summary_df = v2.rank_summary_df(round2_summary_df)
    round2_summary_df.to_csv(out_dir / "round2_tuning_summary.csv", index=False, encoding="utf-8-sig")
    round2_fold_df.to_csv(out_dir / "round2_tuning_folds.csv", index=False, encoding="utf-8-sig")

    finalists = v2.select_top_candidates(round2_summary_df, count=2)
    round2_config_lookup = config_map(round2_configs)
    final_reference = config_map(round1_configs)["V1_protocol|seq=10"]
    final_configs = [final_reference] + [round2_config_lookup[name] for name in finalists]

    final_seed_configs = v2.expand_configs_for_seeds(final_configs, v2.FINALIST_SEEDS)
    final_seed_summary_df, final_seed_fold_df = v2.evaluate_candidate_suite(
        configs=final_seed_configs,
        manifest=manifest,
        summary_df=summary_df,
        case_tables=case_tables,
        threshold_um=threshold_um,
        real_k=real_k,
    )
    final_seed_summary_df.to_csv(out_dir / "round3_finalist_seed_summary.csv", index=False, encoding="utf-8-sig")
    final_seed_fold_df.to_csv(out_dir / "round3_finalist_seed_folds.csv", index=False, encoding="utf-8-sig")

    final_aggregate_df = v2.aggregate_seed_summaries(final_seed_summary_df)
    final_aggregate_df = final_aggregate_df.sort_values(
        by=["median_mean_life_rel_error", "median_median_life_abs_error", "median_mean_pressure_mae"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    gated_df = v2.apply_deployment_gate(final_aggregate_df, baseline_name=final_reference.name)
    gated_df.to_csv(out_dir / "round3_finalist_gate_summary.csv", index=False, encoding="utf-8-sig")

    recommended = v2.select_recommended_candidate(gated_df)
    if recommended is None:
        write_json(
            out_dir / "recommended_candidate.json",
            {
                "deployment_ready": False,
                "message": "No V2 candidate passed the deployment gate. Keep V1 as the formal software model.",
            },
        )
    else:
        payload = dict(recommended)
        payload["deployment_ready"] = True
        write_json(out_dir / "recommended_candidate.json", payload)

    print(f"Saved full V2 candidate benchmark artifacts to: {out_dir}")
    print("Round 1 ranking:")
    print(round1_summary_df[["rank", "candidate_name", "mean_life_rel_error", "median_life_abs_error", "mean_pressure_mae"]].to_string(index=False))
    print("Round 2 ranking:")
    print(round2_summary_df[["rank", "candidate_name", "mean_life_rel_error", "median_life_abs_error", "mean_pressure_mae"]].head(8).to_string(index=False))
    print("Final gate summary:")
    print(gated_df[["candidate_name", "median_mean_life_rel_error", "median_median_life_abs_error", "median_mean_pressure_mae", "deployment_ready", "gate_reason"]].to_string(index=False))


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
