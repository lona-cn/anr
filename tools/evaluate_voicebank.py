"""Strict, offline VoiceBank-DEMAND evaluation of aligned 48 kHz waveforms.

Install tools/requirements-anr-lite-quality.txt in a dedicated environment first.
No installation or network access is performed by this scorer. Full runs require
824 matching filenames; --limit explicitly labels a sorted-prefix smoke run.
All metrics are higher-is-better, but no quality-improvement claim is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import importlib.metadata
import json
import math
import multiprocessing
from pathlib import Path
import platform
import re
import sys
import warnings

import numpy as np
from pesq import PesqError, pesq
from pystoi import stoi
from scipy.signal import resample_poly
import soundfile as sf


RATE = 48000
PERCEPTUAL_RATE = 16000
OFFICIAL_TEST_COUNT = 824
METRICS = ("wb_pesq", "stoi", "si_sdr_db", "snr_db")
LEVEL_FIELDS = ("peak", "rms", "samples_at_or_above_full_scale")
CSV_FIELDS = ("filename", "model", "samples", *METRICS, *LEVEL_FIELDS, "clean_peak", "clean_rms")


class EvaluationError(ValueError):
    """A corpus, waveform or metric violated the evaluation contract."""


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise EvaluationError(f"{label}: nonfinite value {value}")
    return value


def _waveform(audio: np.ndarray, label: str) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise EvaluationError(f"{label}: expected nonempty, finite mono waveform")
    return audio


def _pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = _waveform(reference, "reference")
    estimate = _waveform(estimate, "estimate")
    if reference.shape != estimate.shape:
        raise EvaluationError(f"length mismatch: {reference.size} versus {estimate.size}")
    return reference, estimate


def _energy_ratio_db(numerator: float, denominator: float, label: str) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        raise EvaluationError(f"{label}: nonfinite energy")
    if numerator <= 0.0 or denominator <= 0.0:
        raise EvaluationError(f"{label}: undefined zero-energy ratio ({numerator}, {denominator})")
    # Logarithms of energies avoid overflow in the division. No epsilon/floor.
    return _finite(10.0 * (math.log10(numerator) - math.log10(denominator)), label)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Zero-mean, projection-based SI-SDR in dB, without epsilon floors."""
    reference, estimate = _pair(reference, estimate)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= 0.0 or not math.isfinite(reference_energy):
        raise EvaluationError("SI-SDR: reference has no finite centered energy")
    projection = (np.dot(estimate, reference) / reference_energy) * reference
    residual = estimate - projection
    return _energy_ratio_db(float(np.dot(projection, projection)), float(np.dot(residual, residual)), "SI-SDR")


def snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-dependent waveform SNR in dB; no centering, gain or lag fitting."""
    reference, estimate = _pair(reference, estimate)
    residual = estimate - reference
    return _energy_ratio_db(float(np.dot(reference, reference)), float(np.dot(residual, residual)), "SNR")


def audio_levels(audio: np.ndarray) -> dict:
    audio = _waveform(audio, "levels")
    return {
        "peak": _finite(np.max(np.abs(audio)), "peak"),
        "rms": _finite(np.sqrt(np.mean(np.square(audio))), "RMS"),
        "samples_at_or_above_full_scale": int(np.count_nonzero(np.abs(audio) >= 1.0)),
    }


def _wav_names(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise EvaluationError(f"not an audio directory: {directory}")
    names = sorted(path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".wav")
    if not names:
        raise EvaluationError(f"no WAV files in {directory}")
    return names


def discover_inputs(clean_dir: Path, model_dirs: dict[str, Path], limit: int | None = None) -> tuple[list[str], list[str]]:
    if limit is not None and limit <= 0:
        raise EvaluationError("--limit must be positive")
    names = _wav_names(clean_dir)
    expected = set(names)
    for model, directory in model_dirs.items():
        actual = set(_wav_names(directory))
        if actual != expected:
            raise EvaluationError(
                f"{model}: filename mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    if limit is None and len(names) != OFFICIAL_TEST_COUNT:
        raise EvaluationError(f"full official test requires {OFFICIAL_TEST_COUNT} files; found {len(names)} (use --limit only for smoke)")
    return names, names if limit is None else names[:limit]


def _audio_info(path: Path):
    try:
        info = sf.info(str(path))
    except Exception as error:
        raise EvaluationError(f"{path}: cannot read WAV metadata: {error}") from error
    if info.format not in ("WAV", "WAVEX", "RF64"):
        raise EvaluationError(f"{path}: not a WAV container ({info.format})")
    if info.samplerate != RATE or info.channels != 1 or info.frames <= 0:
        raise EvaluationError(f"{path}: expected nonempty mono {RATE} Hz, got {info.frames} samples, {info.channels} channels, {info.samplerate} Hz")
    return info


def validate_file_contracts(clean_dir: Path, model_dirs: dict[str, Path], filenames: list[str]) -> None:
    """Check every selected header before starting expensive PESQ workers."""
    for filename in filenames:
        clean = _audio_info(clean_dir / filename)
        for model, directory in model_dirs.items():
            info = _audio_info(directory / filename)
            if info.frames != clean.frames:
                raise EvaluationError(f"{model}/{filename}: length mismatch: {info.frames} versus clean {clean.frames}")


def read_audio(path: Path, expected_samples: int | None = None) -> np.ndarray:
    info = _audio_info(path)
    if expected_samples is not None and info.frames != expected_samples:
        raise EvaluationError(f"{path}: length mismatch: {info.frames} versus clean {expected_samples}")
    try:
        audio, rate = sf.read(str(path), dtype="float64", always_2d=False)
    except Exception as error:
        raise EvaluationError(f"{path}: cannot decode WAV: {error}") from error
    audio = _waveform(audio, str(path))
    if rate != RATE or audio.size != info.frames:
        raise EvaluationError(f"{path}: decoded audio differs from preflight metadata")
    return audio


def score_utterance(job: tuple[str, str, dict[str, str]]) -> list[dict]:
    """One process reads clean once and scores all model outputs for a file."""
    filename, clean_dir, model_dirs = job
    try:
        clean = read_audio(Path(clean_dir) / filename)
        clean_levels = audio_levels(clean)
        # Identical resampling for every input: default Kaiser(5), constant-zero
        # padding, output length ceil(N/3). No independent normalization.
        clean_16k = resample_poly(clean, up=1, down=3)
        rows = []
        for model, directory in model_dirs.items():
            try:
                estimate = read_audio(Path(directory) / filename, expected_samples=clean.size)
                estimate_16k = resample_poly(estimate, up=1, down=3)
                # STOI warns for insufficient active speech; such a result is not
                # a valid substitute score. PESQ errors must raise, never codes.
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    perceptual = {
                        "wb_pesq": _finite(pesq(PERCEPTUAL_RATE, clean_16k, estimate_16k, "wb", on_error=PesqError.RAISE_EXCEPTION), "WB-PESQ"),
                        "stoi": _finite(stoi(clean_16k, estimate_16k, PERCEPTUAL_RATE, extended=False), "STOI"),
                    }
                rows.append({
                    "filename": filename,
                    "model": model,
                    "samples": int(clean.size),
                    **perceptual,
                    "si_sdr_db": si_sdr(clean, estimate),
                    "snr_db": snr(clean, estimate),
                    **audio_levels(estimate),
                    "clean_peak": clean_levels["peak"],
                    "clean_rms": clean_levels["rms"],
                })
            except Exception as error:
                raise EvaluationError(f"{model}/{filename}: scoring failed: {error}") from error
        return rows
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(f"{filename}: scoring failed: {error}") from error


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    values = _waveform(values, "bootstrap differences")
    if samples <= 0 or seed < 0:
        raise EvaluationError("bootstrap samples must be positive and seed nonnegative")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    # Bounded memory even for large bootstrap requests; generator ordering is
    # unchanged by chunking. The same seed pairs resamples across metrics.
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return [_finite(value, "bootstrap percentile") for value in np.quantile(means, [0.025, 0.975], method="linear")]


def aggregate_rows(rows: list[dict], filenames: list[str], models: list[str], bootstrap_samples: int, seed: int, metrics: tuple[str, ...] = METRICS) -> tuple[dict, list[dict]]:
    if not filenames or not models or len(set(filenames)) != len(filenames) or len(set(models)) != len(models):
        raise EvaluationError("aggregation requires unique nonempty filenames and models")
    expected = {(filename, model) for filename in filenames for model in models}
    indexed = {}
    for row in rows:
        key = (row["filename"], row["model"])
        if key not in expected or key in indexed:
            raise EvaluationError(f"unexpected or duplicate metric row: {key}")
        for metric in metrics:
            _finite(row[metric], f"{key}/{metric}")
        if not isinstance(row["samples"], int) or row["samples"] <= 0:
            raise EvaluationError(f"{key}: invalid sample count")
        indexed[key] = row
    if set(indexed) != expected:
        raise EvaluationError(f"missing metric rows: {sorted(expected - set(indexed))}")
    for filename in filenames:
        if len({indexed[filename, model]["samples"] for model in models}) != 1:
            raise EvaluationError(f"{filename}: sample counts differ across models")
    summary = {"models": {}}
    for model in models:
        selected = [indexed[filename, model] for filename in filenames]
        aggregate = {
            "count": len(selected),
            "samples": sum(row["samples"] for row in selected),
            "means": {metric: _finite(np.mean([row[metric] for row in selected]), f"{model}/{metric} mean") for metric in metrics},
        }
        if all(all(field in row for field in LEVEL_FIELDS) for row in selected):
            aggregate["audio_levels"] = {
                "mean_peak": _finite(np.mean([row["peak"] for row in selected]), "mean peak"),
                "max_peak": _finite(max(row["peak"] for row in selected), "max peak"),
                "mean_rms": _finite(np.mean([row["rms"] for row in selected]), "mean RMS"),
                "files_at_or_above_full_scale": sum(row["samples_at_or_above_full_scale"] > 0 for row in selected),
                "samples_at_or_above_full_scale": sum(row["samples_at_or_above_full_scale"] for row in selected),
            }
        summary["models"][model] = aggregate
    differences = []
    if "anr-lite" in models and "dfn3" in models:
        for filename in filenames:
            differences.append({
                "filename": filename,
                **{metric: _finite(indexed[filename, "anr-lite"][metric] - indexed[filename, "dfn3"][metric], f"{filename}/{metric} difference") for metric in metrics},
            })
        summary["paired_anr_minus_dfn3"] = {
            "available": True,
            "count": len(differences),
            "metrics": {
                metric: {
                    "mean": _finite(np.mean([row[metric] for row in differences]), f"{metric} paired mean"),
                    "ci95": bootstrap_mean_ci(np.array([row[metric] for row in differences]), bootstrap_samples, seed),
                }
                for metric in metrics
            },
        }
    else:
        summary["paired_anr_minus_dfn3"] = {
            "available": False, "count": 0,
            "reason": "both enhanced model names 'anr-lite' and 'dfn3' are required",
        }
    return summary, differences


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()})


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--clean-dir", required=True, type=Path)
    result.add_argument("--noisy-dir", required=True, type=Path)
    result.add_argument("--enhanced", required=True, action="append", metavar="NAME=DIR")
    result.add_argument("--output-dir", required=True, type=Path, help="new directory; never overwrites an existing run")
    result.add_argument("--workers", type=_positive, default=8)
    result.add_argument("--bootstrap-samples", type=_positive, default=2000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--limit", type=_positive, help="smoke-only sorted-prefix cap; omitted for all 824 files")
    return result


def run(args: argparse.Namespace) -> dict:
    clean_dir = args.clean_dir.resolve()
    model_dirs = {"noisy": args.noisy_dir.resolve()}
    for item in args.enhanced:
        name, separator, directory = item.partition("=")
        if not separator or not directory or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise EvaluationError(f"invalid enhanced argument {item!r}; expected NAME=DIR")
        if name in model_dirs or name == "clean":
            raise EvaluationError(f"duplicate or reserved model name: {name}")
        model_dirs[name] = Path(directory).resolve()
    model_dirs = {name: model_dirs[name] for name in ["noisy", *sorted(set(model_dirs) - {"noisy"})]}
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise EvaluationError(f"output directory already exists: {output_dir}")
    if args.seed < 0:
        raise EvaluationError("seed must be nonnegative")
    names, selected = discover_inputs(clean_dir, model_dirs, args.limit)
    validate_file_contracts(clean_dir, model_dirs, selected)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile", "pystoi", "pesq")}
    jobs = [(name, str(clean_dir), {model: str(path) for model, path in model_dirs.items()}) for name in selected]
    rows = []
    # Always processes, including --workers 1: no threaded/shared PESQ state.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        for utterance_rows in executor.map(score_utterance, jobs, chunksize=1):
            rows.extend(utterance_rows)
    summary, differences = aggregate_rows(rows, selected, list(model_dirs), args.bootstrap_samples, args.seed)
    samples = summary["models"]["noisy"]["samples"]
    summary.update({
        "schema_version": 1,
        "scope": "official_test_full_824" if args.limit is None else "smoke_sorted_prefix",
        "official_expected_files": OFFICIAL_TEST_COUNT,
        "source_files": len(names),
        "evaluated_files": len(selected),
        "not_evaluated_files": len(names) - len(selected),
        "limit": args.limit,
        "filenames": selected,
        "source_filenames": names,
        "speaker_counts": dict(sorted(Counter(name.split("_", 1)[0] for name in selected).items())),
        "metric_rows": len(rows),
        "expected_metric_rows": len(selected) * len(model_dirs),
        "metric_counts": {model: {metric: len(selected) for metric in METRICS} for model in model_dirs},
        "failed_files": 0,
        "skipped_files": 0,
        "samples_per_model": samples,
        "duration_seconds_per_model": samples / RATE,
        "clean_dir": str(clean_dir),
        "model_dirs": {model: str(path) for model, path in model_dirs.items()},
        "workers": args.workers,
        "dependencies": {**versions, "python": platform.python_version(), "libsndfile": sf.__libsndfile_version__},
        "platform": platform.platform(),
        "sampling": {
            "input_rate_hz": RATE, "channels": 1, "decode_dtype": "float64",
            "perceptual_rate_hz": PERCEPTUAL_RATE,
            "resampler": "scipy.signal.resample_poly(up=1, down=3, window=('kaiser', 5.0), padtype='constant', cval=0)",
            "perceptual_output_samples": "ceil(original_samples / 3)",
            "normalization": "none outside metric-internal standard processing",
            "alignment": "exact original lengths; no reference-fitted lag or gain; upstream alignment required",
        },
        "metric_conventions": {
            "wb_pesq": "pesq(16000, clean, estimate, mode='wb', on_error=RAISE_EXCEPTION)",
            "stoi": "pystoi.stoi(clean, estimate, 16000, extended=False)",
            "si_sdr_db": "48 kHz; subtract each waveform mean; project estimate onto clean; 10*log10(projected_energy/residual_energy)",
            "snr_db": "48 kHz; 10*log10(sum(clean**2)/sum((estimate-clean)**2)); no centering",
            "averaging": "unweighted arithmetic utterance means, not duration weighting",
            "nonfinite_policy": "raise; no epsilon floors, warning-derived scores, error codes or omitted files",
            "level_full_scale_threshold": "abs(sample) >= 1, counted without clipping; does not itself establish clipping",
            "direction": "all four metrics higher is better",
        },
        "bootstrap": {
            "method": "paired utterance resampling with replacement; percentile 95% CI; numpy quantile linear",
            "samples": args.bootstrap_samples, "seed": args.seed,
            "rng": "numpy.random.default_rng (PCG64)",
            "unit": "utterance (not independent speaker/domain sampling)",
            "comparison": "anr-lite minus dfn3, matched by filename",
        },
        "limitations": [
            "VoiceBank-DEMAND official test has two speakers; utterance bootstrap does not quantify unseen-speaker or domain generalization.",
            "DFN3 uses repository pretrained weights; their exact training corpus and overlap with this test set are not audited here. This is not a controlled architecture-only experiment.",
            "Objective metrics alone do not establish subjective listening quality or improvement.",
            "Filename/count checks require the supplied official test directories; corpus provenance is not authenticated by this scorer.",
        ],
    })
    # Publish only a complete result set; metric failures produce no success files.
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "per_file.csv", CSV_FIELDS, rows)
    _write_json(output_dir / "per_file.json", rows)
    _write_csv(output_dir / "paired_differences.csv", ("filename", *METRICS), differences)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parser().parse_args()
    try:
        summary = run(args)
    except Exception as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"scope": summary["scope"], "evaluated_files": summary["evaluated_files"], "metric_rows": summary["metric_rows"], "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
