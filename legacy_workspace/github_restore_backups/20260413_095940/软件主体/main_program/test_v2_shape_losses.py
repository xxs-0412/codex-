from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn as nn

import v2_pipeline as v2


class LastActualCycleModel(nn.Module):
    def forward(self, sequence_x: torch.Tensor) -> torch.Tensor:
        return sequence_x[:, -1, 3:4]


def identity_scaler(width: int) -> v2.ArrayScaler:
    mean = np.zeros((1, width), dtype=np.float32)
    std = np.ones((1, width), dtype=np.float32)
    return v2.ArrayScaler().load(mean, std)


def make_sequence(log_stress: float, wear_mm: float, seq_len: int = 1) -> np.ndarray:
    row = np.array([1.0, 8.0, 0.01, float(log_stress), float(wear_mm)], dtype=np.float32)
    return np.repeat(row[np.newaxis, :], seq_len, axis=0).astype(np.float32)


def make_sequence_dataset(entries: list[tuple[str, int, float]], seq_len: int = 1) -> v2.SequenceDataset:
    raw_sequences = np.asarray(
        [make_sequence(log_stress=0.0, wear_mm=wear_mm, seq_len=seq_len) for _, _, wear_mm in entries],
        dtype=np.float32,
    )
    sample_count = len(entries)
    return v2.SequenceDataset(
        raw_sequences=raw_sequences,
        target_log=np.zeros((sample_count, 1), dtype=np.float32),
        next_delta_wear=np.zeros((sample_count, 1), dtype=np.float32),
        actual_step=np.ones((sample_count, 1), dtype=np.float32),
        case_names=[case_name for case_name, _, _ in entries],
        step_index=np.asarray([step_idx for _, step_idx, _ in entries], dtype=np.int64),
    )


class ShapeLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = v2.CandidateConfig(
            name="test_shape",
            model_kind="v1_transformer",
            sequence_length=1,
            run_stress_mono_lambda=0.05,
            run_stress_slow_lambda=0.01,
        )
        self.model = LastActualCycleModel()
        self.input_scalers = {"sequence": identity_scaler(len(v2.LEGACY_FEATURE_ORDER))}
        self.target_scaler = identity_scaler(1)

    def test_run_stress_mono_zero_when_monotonic_decreasing(self) -> None:
        shape_dataset = v2.RunShapeDataset(
            mono_raw_sequences=np.asarray(
                [[make_sequence(math.log(10.0), 0.001), make_sequence(math.log(8.0), 0.002)]],
                dtype=np.float32,
            ),
            mono_actual_step=np.ones((1, 2, 1), dtype=np.float32),
            slow_raw_sequences=np.zeros((0, 3, 1, 5), dtype=np.float32),
            slow_actual_step=np.zeros((0, 3, 1), dtype=np.float32),
        )
        mono_pen, _ = v2.run_stress_shape_losses(
            model=self.model,
            shape_dataset=shape_dataset,
            config=self.config,
            input_scalers=self.input_scalers,
            target_scaler=self.target_scaler,
            compute_slow=False,
        )
        self.assertAlmostEqual(float(mono_pen.item()), 0.0, places=7)

    def test_run_stress_mono_positive_when_sequence_rises(self) -> None:
        shape_dataset = v2.RunShapeDataset(
            mono_raw_sequences=np.asarray(
                [[make_sequence(math.log(10.0), 0.001), make_sequence(math.log(12.0), 0.002)]],
                dtype=np.float32,
            ),
            mono_actual_step=np.ones((1, 2, 1), dtype=np.float32),
            slow_raw_sequences=np.zeros((0, 3, 1, 5), dtype=np.float32),
            slow_actual_step=np.zeros((0, 3, 1), dtype=np.float32),
        )
        mono_pen, _ = v2.run_stress_shape_losses(
            model=self.model,
            shape_dataset=shape_dataset,
            config=self.config,
            input_scalers=self.input_scalers,
            target_scaler=self.target_scaler,
            compute_slow=False,
        )
        self.assertGreater(float(mono_pen.item()), 0.0)

    def test_run_stress_slow_zero_when_drop_decelerates(self) -> None:
        shape_dataset = v2.RunShapeDataset(
            mono_raw_sequences=np.zeros((0, 2, 1, 5), dtype=np.float32),
            mono_actual_step=np.zeros((0, 2, 1), dtype=np.float32),
            slow_raw_sequences=np.asarray(
                [[make_sequence(math.log(10.0), 0.001), make_sequence(math.log(8.0), 0.002), make_sequence(math.log(7.0), 0.003)]],
                dtype=np.float32,
            ),
            slow_actual_step=np.ones((1, 3, 1), dtype=np.float32),
        )
        _, slow_pen = v2.run_stress_shape_losses(
            model=self.model,
            shape_dataset=shape_dataset,
            config=self.config,
            input_scalers=self.input_scalers,
            target_scaler=self.target_scaler,
            compute_mono=False,
        )
        self.assertAlmostEqual(float(slow_pen.item()), 0.0, places=7)

    def test_run_stress_slow_positive_when_later_drop_is_steeper(self) -> None:
        shape_dataset = v2.RunShapeDataset(
            mono_raw_sequences=np.zeros((0, 2, 1, 5), dtype=np.float32),
            mono_actual_step=np.zeros((0, 2, 1), dtype=np.float32),
            slow_raw_sequences=np.asarray(
                [[make_sequence(math.log(10.0), 0.001), make_sequence(math.log(8.0), 0.002), make_sequence(math.log(5.0), 0.003)]],
                dtype=np.float32,
            ),
            slow_actual_step=np.ones((1, 3, 1), dtype=np.float32),
        )
        _, slow_pen = v2.run_stress_shape_losses(
            model=self.model,
            shape_dataset=shape_dataset,
            config=self.config,
            input_scalers=self.input_scalers,
            target_scaler=self.target_scaler,
            compute_mono=False,
        )
        self.assertGreater(float(slow_pen.item()), 0.0)

    def test_build_run_shape_dataset_filters_windows_above_5um(self) -> None:
        dataset = make_sequence_dataset(
            [
                ("run_a", 0, 0.001),
                ("run_a", 1, 0.004),
                ("run_a", 2, 0.006),
            ]
        )
        shape_dataset = v2.build_run_shape_dataset(dataset, max_wear_mm=0.005)
        self.assertEqual(shape_dataset.mono_count, 1)
        self.assertEqual(shape_dataset.slow_count, 0)

    def test_build_run_shape_dataset_respects_run_boundaries(self) -> None:
        dataset = make_sequence_dataset(
            [
                ("run_a", 0, 0.001),
                ("run_b", 1, 0.002),
            ]
        )
        shape_dataset = v2.build_run_shape_dataset(dataset, max_wear_mm=0.005)
        self.assertEqual(shape_dataset.mono_count, 0)
        self.assertEqual(shape_dataset.slow_count, 0)

    def test_build_run_shape_dataset_handles_short_runs(self) -> None:
        one_step_dataset = make_sequence_dataset([("run_a", 0, 0.001)])
        one_step_shape = v2.build_run_shape_dataset(one_step_dataset, max_wear_mm=0.005)
        self.assertEqual(one_step_shape.mono_count, 0)
        self.assertEqual(one_step_shape.slow_count, 0)

        two_step_dataset = make_sequence_dataset(
            [
                ("run_a", 0, 0.001),
                ("run_a", 1, 0.002),
            ]
        )
        two_step_shape = v2.build_run_shape_dataset(two_step_dataset, max_wear_mm=0.005)
        self.assertEqual(two_step_shape.mono_count, 1)
        self.assertEqual(two_step_shape.slow_count, 0)

    def test_total_objective_matches_old_behavior_when_shape_lambdas_are_zero(self) -> None:
        config = self.config.with_updates(run_stress_mono_lambda=0.0, run_stress_slow_lambda=0.0)
        raw_sequences = np.asarray(
            [
                make_sequence(math.log(10.0), 0.001),
                make_sequence(math.log(9.0), 0.002),
            ],
            dtype=np.float32,
        )
        actual_step = np.ones((2, 1), dtype=np.float32)
        target_log = np.asarray([[math.log(10.0)], [math.log(9.0)]], dtype=np.float32)
        input_scalers = {"sequence": identity_scaler(len(v2.LEGACY_FEATURE_ORDER))}
        inputs_np, _ = v2.build_model_inputs(raw_sequences, actual_step, config, input_scalers=input_scalers, fit=False)
        target_scaler = identity_scaler(1)
        target_scaled = target_scaler.transform(target_log)
        shape_dataset = v2.RunShapeDataset(
            mono_raw_sequences=np.asarray([[raw_sequences[0], raw_sequences[1]]], dtype=np.float32),
            mono_actual_step=np.ones((1, 2, 1), dtype=np.float32),
            slow_raw_sequences=np.zeros((0, 3, 1, 5), dtype=np.float32),
            slow_actual_step=np.zeros((0, 3, 1), dtype=np.float32),
        )

        without_shape = v2.total_objective(
            model=self.model,
            inputs_np=inputs_np,
            target_scaled_np=target_scaled,
            raw_sequences=raw_sequences,
            actual_step=actual_step,
            next_delta_wear=np.zeros((2, 1), dtype=np.float32),
            input_scalers=input_scalers,
            target_scaler=target_scaler,
            config=config,
            real_k=0.0,
            shape_dataset=None,
        )
        with_shape = v2.total_objective(
            model=self.model,
            inputs_np=inputs_np,
            target_scaled_np=target_scaled,
            raw_sequences=raw_sequences,
            actual_step=actual_step,
            next_delta_wear=np.zeros((2, 1), dtype=np.float32),
            input_scalers=input_scalers,
            target_scaler=target_scaler,
            config=config,
            real_k=0.0,
            shape_dataset=shape_dataset,
        )
        self.assertAlmostEqual(float(without_shape.item()), float(with_shape.item()), places=7)


if __name__ == "__main__":
    unittest.main()
