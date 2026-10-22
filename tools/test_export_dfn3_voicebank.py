"""Tests for DFN3 calibration, delay alignment, and audio validation."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from export_dfn3_voicebank import (
    command_groups,
    command_units,
    estimate_delay,
    make_high_band_probe,
    nominal_wet_delay,
    padded_length,
    read_wav,
    recover_dry,
    trim_delayed,
    verify_reconstruction,
)


class DelayTests(unittest.TestCase):
    def test_wet_nominal_delay_comes_from_history_geometry(self):
        self.assertEqual(nominal_wet_delay(1920), 1440)
        self.assertEqual(nominal_wet_delay(480), 0)
        for invalid in (0, 1919, -480):
            with self.assertRaises(ValueError):
                nominal_wet_delay(invalid)

    def test_high_band_probe_excludes_complex_df_band_and_verifies_lag(self):
        probe = make_high_band_probe(20003)
        spectrum = np.abs(np.fft.rfft(probe)) ** 2
        frequencies = np.fft.rfftfreq(probe.size, 1 / 48000)
        leakage = np.sum(spectrum[frequencies < 6000]) / np.sum(spectrum)
        self.assertLess(leakage, 0.001)
        rendered = np.pad(probe * 0.05, (1440, 1000))
        self.assertEqual(estimate_delay(probe, rendered, 2000)["delay_samples"], 1440)
        with self.assertRaises(ValueError):
            estimate_delay(probe, np.zeros(probe.size + 2000), 2000)

    def test_non_hop_delay_is_measured_without_gain_fitting(self):
        probe = np.random.default_rng(12).uniform(-0.4, 0.4, 5003).astype(np.float32)
        rendered = np.concatenate((np.zeros(731), probe * 0.3, np.zeros(1024)))
        measured = estimate_delay(probe, rendered, 1000)
        self.assertEqual(measured["delay_samples"], 731)
        self.assertGreater(measured["correlation"], 0.99999)
        with self.assertRaises(ValueError):
            verify_reconstruction(probe, rendered, 731)

    def test_delay_zero_and_exact_identity_are_supported(self):
        probe = np.random.default_rng(19).uniform(-0.2, 0.2, 1051).astype(np.float32)
        rendered = np.pad(probe, (0, 200))
        measured = estimate_delay(probe, rendered, 200)
        self.assertEqual(measured["delay_samples"], 0)
        stats = verify_reconstruction(probe, rendered, 0)
        self.assertEqual(stats["max_absolute_error"], 0.0)

    def test_search_rejects_truncation_boundary_and_ambiguous_signal(self):
        probe = np.random.default_rng(31).normal(size=1000)
        with self.assertRaises(ValueError):
            estimate_delay(probe, np.zeros(1099), 100)
        with self.assertRaises(ValueError):
            estimate_delay(probe, np.pad(probe, (100, 0)), 100)
        with self.assertRaises(ValueError):
            estimate_delay(np.zeros(1000), np.zeros(1200), 200)
        with self.assertRaises(ValueError):
            estimate_delay(np.ones(1000), np.ones(1200), 200)

    def test_reconstruction_checks_whole_probe_including_last_sample(self):
        probe = np.random.default_rng(3).uniform(-0.3, 0.3, 1001)
        rendered = np.pad(probe, (7, 100))
        verify_reconstruction(probe, rendered, 7)
        rendered[1007] += 0.01
        with self.assertRaises(ValueError):
            verify_reconstruction(probe, rendered, 7)

    def test_nonfinite_or_multichannel_arrays_are_rejected(self):
        for invalid in (np.array([0.0, np.nan]), np.ones((3, 2)), np.array([])):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    estimate_delay(invalid, np.ones(100), 10)


class GeometryTests(unittest.TestCase):
    def test_padding_flushes_fixed_delay_and_rounds_up_hops(self):
        self.assertEqual(padded_length(1001, 1440), 2880)
        self.assertEqual(padded_length(960, 1440), 2400)
        self.assertEqual(padded_length(1, 0), 480)

    def test_trim_preserves_signed_out_of_range_float_values_and_tail(self):
        raw = np.array([0, 0, -1.5, 0.25, 2.0, -0.75, 0, 0], dtype=np.float32)
        result = trim_delayed(raw, 4, 2, 8)
        np.testing.assert_array_equal(result, [-1.5, 0.25, 2.0, -0.75])
        self.assertEqual(result.dtype, np.float32)

    def test_trimming_rejects_truncated_and_excess_results(self):
        for actual in (7, 9):
            with self.assertRaises(ValueError):
                trim_delayed(np.zeros(actual), 4, 2, 8)
        with self.assertRaises(ValueError):
            trim_delayed(np.zeros(8), 7, 2, 8)
        with self.assertRaises(ValueError):
            trim_delayed(np.array([0.0, np.inf]), 1, 0, 2)
        for length, delay in ((0, 1), (1, -1)):
            with self.assertRaises(ValueError):
                padded_length(length, delay)

    def test_linear_mixture_recovers_dry_without_scale_adjustment(self):
        wet = np.array([0.2, -0.1, 0.05, 0.7], dtype=np.float32)
        dry = np.array([-0.4, 0.2, 0.0, 1.1], dtype=np.float32)
        alpha = 0.8912509381337456
        mixed = ((1 - alpha) * wet + alpha * dry).astype(np.float32)
        np.testing.assert_allclose(recover_dry(wet, mixed, alpha), dry, atol=1e-7)
        for invalid in (0.0, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                recover_dry(wet, mixed, invalid)
        with self.assertRaises(ValueError):
            recover_dry(wet[:-1], mixed, alpha)


class FileAndBatchTests(unittest.TestCase):
    def test_real_float_wav_roundtrip_keeps_unclipped_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.wav"
            expected = np.array([-1.4, 0.0, 1.7, -0.2], dtype=np.float32)
            sf.write(path, expected, 48000, subtype="FLOAT")
            np.testing.assert_array_equal(read_wav(path, require_float=True), expected)

    def test_real_wav_rejects_wrong_rate_stereo_and_nonfloat_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.wav"
            for samples, rate, subtype in (
                (np.ones(8), 16000, "FLOAT"),
                (np.ones((8, 2)), 48000, "FLOAT"),
                (np.ones(8), 48000, "PCM_16"),
            ):
                sf.write(path, samples, rate, subtype=subtype)
                with self.assertRaises(ValueError):
                    read_wav(path, require_float=True)

    def test_command_groups_preserve_paths_with_spaces_and_unicode(self):
        prefix = ["anr.exe", "--verbose", "denoise"]
        paths = [Path("scratch") / (f"{index:03d} " + "\u8bed" * 150 + ".wav") for index in range(200)]
        options = ["--output-dir", "results with spaces", "--strength", "80"]
        groups = command_groups(prefix, paths, options)
        self.assertGreater(len(groups), 1)
        flattened = []
        for group in groups:
            self.assertLessEqual(command_units(group), 24000)
            self.assertEqual(group[:3], prefix)
            self.assertEqual(group[-len(options):], options)
            flattened.extend(group[3:-len(options)])
        self.assertEqual(flattened, [str(path) for path in paths])
        with self.assertRaises(ValueError):
            command_groups(prefix, [Path("x" * 25000)], options)


if __name__ == "__main__":
    unittest.main()
