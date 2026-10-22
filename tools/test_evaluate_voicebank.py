"""Focused evaluator tests; run with the dedicated quality environment."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

import evaluate_voicebank as evaluator


class WaveformMetricsTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([1.0, -1.0, 1.0, -1.0])
        self.orthogonal = np.array([1.0, 1.0, -1.0, -1.0])

    def test_si_sdr_projects_zero_mean_and_ignores_scale_and_dc(self):
        estimate = self.reference + 0.5 * self.orthogonal
        expected = 10.0 * math.log10(4.0)
        self.assertAlmostEqual(evaluator.si_sdr(self.reference, estimate), expected)
        self.assertAlmostEqual(
            evaluator.si_sdr(self.reference + 9.0, 3.0 * estimate - 5.0), expected
        )

    def test_snr_retains_gain_error_and_dc(self):
        self.assertAlmostEqual(
            evaluator.snr(self.reference, 0.5 * self.reference),
            10.0 * math.log10(4.0),
        )
        self.assertAlmostEqual(evaluator.snr(self.reference, self.reference + 1.0), 0.0)

    def test_degenerate_or_nonfinite_metrics_raise_instead_of_flooring(self):
        for estimate in (self.reference, np.zeros(4), np.full(4, np.nan)):
            with self.subTest(estimate=estimate):
                with self.assertRaises(evaluator.EvaluationError):
                    evaluator.si_sdr(self.reference, estimate)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.snr(self.reference, self.reference)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.snr(np.zeros(4), np.ones(4))

    def test_levels_preserve_unclipped_peak_and_rms(self):
        levels = evaluator.audio_levels(np.array([-2.0, 0.0, 2.0, 0.0]))
        self.assertEqual(levels["peak"], 2.0)
        self.assertAlmostEqual(levels["rms"], math.sqrt(2.0))
        self.assertEqual(levels["samples_at_or_above_full_scale"], 2)


class AggregationTests(unittest.TestCase):
    def rows(self):
        # Real own-metric observations, not fabricated upstream PESQ/STOI outputs.
        reference = np.array([1.0, -1.0, 1.0, -1.0])
        error = np.array([1.0, 1.0, -1.0, -1.0])
        rows = []
        for filename in ("p232_001.wav", "p257_001.wav"):
            for model, factor in (("dfn3", 1.0), ("anr-lite", 0.5)):
                rows.append({
                    "filename": filename,
                    "model": model,
                    "samples": 4,
                    "si_sdr_db": evaluator.si_sdr(reference, reference + factor * error),
                    "snr_db": evaluator.snr(reference, reference + factor * error),
                })
        return rows

    def test_pairing_uses_filename_and_anr_minus_dfn3_direction(self):
        rows = self.rows()[::-1]
        summary, differences = evaluator.aggregate_rows(
            rows, ["p232_001.wav", "p257_001.wav"], ["anr-lite", "dfn3"],
            bootstrap_samples=100, seed=42, metrics=("si_sdr_db", "snr_db"),
        )
        self.assertEqual(summary["models"]["anr-lite"]["count"], 2)
        paired = summary["paired_anr_minus_dfn3"]
        self.assertEqual(paired["count"], 2)
        for metric in ("si_sdr_db", "snr_db"):
            result = paired["metrics"][metric]
            self.assertAlmostEqual(result["mean"], 6.020599913279624)
            np.testing.assert_allclose(result["ci95"], [6.020599913279624] * 2)
            self.assertAlmostEqual(differences[0][metric], 6.020599913279624)
        self.assertEqual(differences[0]["filename"], "p232_001.wav")

    def test_missing_duplicate_nonfinite_and_unexpected_rows_are_errors(self):
        rows = self.rows()
        cases = [rows[:-1], rows + [rows[0]], rows + [dict(rows[0], filename="extra.wav")]]
        bad = [dict(row) for row in rows]
        bad[0]["snr_db"] = float("nan")
        cases.append(bad)
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(evaluator.EvaluationError):
                    evaluator.aggregate_rows(
                        case, ["p232_001.wav", "p257_001.wav"], ["anr-lite", "dfn3"],
                        bootstrap_samples=10, seed=42, metrics=("si_sdr_db", "snr_db"),
                    )

    def test_bootstrap_is_deterministic_and_nonconstant(self):
        ci = evaluator.bootstrap_mean_ci(np.array([-3.0, 0.0, 7.0]), 300, 42)
        self.assertEqual(ci, evaluator.bootstrap_mean_ci(np.array([-3.0, 0.0, 7.0]), 300, 42))
        self.assertLess(ci[0], 4.0 / 3.0)
        self.assertGreater(ci[1], 4.0 / 3.0)


class FileContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clean = self.root / "clean"
        self.noisy = self.root / "noisy"
        self.enhanced = self.root / "enhanced"
        for directory in (self.clean, self.noisy, self.enhanced):
            directory.mkdir()
            for name in ("p232_002.wav", "p232_001.wav"):
                sf.write(directory / name, np.linspace(-0.2, 0.2, 481), 48000, subtype="FLOAT")
        self.models = {"noisy": self.noisy, "anr-lite": self.enhanced}

    def test_smoke_is_sorted_and_checks_whole_filename_set_before_limit(self):
        names, selected = evaluator.discover_inputs(self.clean, self.models, limit=1)
        self.assertEqual(names, ["p232_001.wav", "p232_002.wav"])
        self.assertEqual(selected, ["p232_001.wav"])
        (self.enhanced / "p232_002.wav").unlink()
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.discover_inputs(self.clean, self.models, limit=1)

    def test_full_run_requires_824_and_extra_names_are_rejected(self):
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.discover_inputs(self.clean, self.models)
        sf.write(self.enhanced / "extra.wav", np.zeros(481), 48000, subtype="FLOAT")
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.discover_inputs(self.clean, self.models, limit=1)

    def test_rejects_rate_channels_length_and_nonfinite_audio(self):
        path = self.enhanced / "p232_001.wav"
        cases = [
            (np.zeros(481), 16000),
            (np.zeros((481, 2)), 48000),
            (np.zeros(480), 48000),
            (np.full(481, np.nan), 48000),
        ]
        for audio, rate in cases:
            sf.write(path, audio, rate, subtype="FLOAT")
            with self.subTest(shape=audio.shape, rate=rate):
                with self.assertRaises(evaluator.EvaluationError):
                    evaluator.read_audio(path, expected_samples=481)

    def test_preflight_rejects_mismatch_before_scoring(self):
        sf.write(self.enhanced / "p232_002.wav", np.zeros(480), 48000, subtype="FLOAT")
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.validate_file_contracts(
                self.clean, self.models, ["p232_001.wav", "p232_002.wav"]
            )

    def test_float_waveform_is_not_normalized_or_clipped(self):
        audio = np.array([-1.5, 0.25, 2.0], dtype=np.float32)
        path = self.enhanced / "p232_001.wav"
        sf.write(path, audio, 48000, subtype="FLOAT")
        np.testing.assert_array_equal(evaluator.read_audio(path, expected_samples=3), audio)


if __name__ == "__main__":
    unittest.main()
