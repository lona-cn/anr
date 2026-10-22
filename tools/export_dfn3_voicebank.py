"""Render the fixed DFN3 CLI with independent, fixed-lag calibration.

The CLI strength is NOT attenuation dB: it rounds strength * 0.6 to whole
dB. Consequently 1e-7 strength selects the wet path, not an identity path.
Calibration recovers the dry path from two source-defined linear mixtures
and verifies its unscaled reconstruction. Formal output is pure wet (strength
0, NOT bypass), aligned by the source-derived dry delay minus one hop. An
independent 8–18 kHz probe checks, but never defines, that nominal wet delay.
No clean speech is read by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.signal import correlate, fftconvolve, firwin
import soundfile as sf


SAMPLE_RATE = 48_000
HOP = 480
MAX_DELAY = 8 * HOP
CALIBRATION_STRENGTH = 1.6666667  # CLI rounds this to exactly 1 attenuation dB.
MODEL_SHA256 = "fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9"
COMMAND_UTF16_LIMIT = 24_000  # CreateProcess limit is 32,767 including terminator.


def nominal_wet_delay(dry_delay: int) -> int:
    """Source history geometry, NOT a speech correlation-optimized offset."""
    if dry_delay < HOP or dry_delay % HOP:
        raise ValueError("verified dry delay must be a positive whole number of hops")
    return dry_delay - HOP


def make_high_band_probe(samples: int = 2 * SAMPLE_RATE + 137) -> np.ndarray:
    if samples < 1:
        raise ValueError("probe sample count must be positive")
    taps = firwin(1025, [8000, 18000], pass_zero=False, fs=SAMPLE_RATE)
    noise = np.random.default_rng(20261023).uniform(-0.25, 0.25, samples + taps.size - 1)
    return fftconvolve(noise, taps, mode="valid").astype(np.float32)


def mono_array(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected nonempty finite mono samples")
    return value


def estimate_delay(probe: np.ndarray, rendered: np.ndarray, max_delay: int) -> dict:
    """One fixed nonnegative lag, using every probe sample and no fitted gain."""
    x = mono_array(probe, "probe").astype(np.float64)
    y = mono_array(rendered, "rendered probe").astype(np.float64)
    if max_delay < 1 or y.size < x.size + max_delay:
        raise ValueError("probe rendering is too short for the complete lag search")
    energy = float(np.dot(x, x))
    if energy <= 0:
        raise ValueError("silent probe cannot determine delay")
    scores = correlate(y[:x.size + max_delay], x, mode="valid", method="fft")
    cumulative = np.concatenate(([0.0], np.cumsum(y * y)))
    energies = cumulative[x.size:x.size + max_delay + 1] - cumulative[:max_delay + 1]
    denominator = np.sqrt(energy * np.maximum(energies, 0))
    correlation = np.divide(scores, denominator, out=np.zeros_like(scores), where=denominator > 0)
    lag = int(np.argmax(correlation))
    peak = float(correlation[lag])
    other = np.delete(correlation, lag)
    runner_up = float(np.max(other))
    if lag == max_delay or peak < 0.15 or peak - runner_up < 0.05:
        raise ValueError(f"ambiguous/out-of-range probe lag: lag={lag}, correlation={peak}, next={runner_up}")
    return {"delay_samples": lag, "correlation": peak, "runner_up_correlation": runner_up,
            "search_max_samples": max_delay, "comparison_samples": int(x.size)}


def padded_length(samples: int, delay: int) -> int:
    if samples < 1 or delay < 0:
        raise ValueError("sample count must be positive and delay nonnegative")
    return ((samples + delay + HOP - 1) // HOP) * HOP


def trim_delayed(rendered: np.ndarray, samples: int, delay: int, expected_samples: int) -> np.ndarray:
    result = mono_array(rendered, "rendered audio")
    if samples < 1 or delay < 0 or samples + delay > expected_samples:
        raise ValueError("invalid delay/length geometry")
    if result.size != expected_samples:
        raise ValueError(f"rendered length {result.size} != padded input {expected_samples}")
    return np.asarray(result[delay:delay + samples], dtype=np.float32).copy()


def verify_reconstruction(probe: np.ndarray, rendered: np.ndarray, delay: int) -> dict:
    x = mono_array(probe, "probe").astype(np.float64)
    y = mono_array(rendered, "dry reconstruction").astype(np.float64)
    if delay < 0 or y.size < delay + x.size:
        raise ValueError("dry reconstruction is truncated")
    error = y[delay:delay + x.size] - x
    rms = float(np.sqrt(np.mean(x * x)))
    if rms == 0:
        raise ValueError("cannot verify a silent probe")
    relative_rms = float(np.sqrt(np.mean(error * error)) / rms)
    peak = float(np.max(np.abs(error)))
    if relative_rms > 2e-5 or peak > 2e-5:
        raise ValueError(f"unscaled full-probe reconstruction failed: relative RMS={relative_rms}, peak={peak}")
    return {"relative_rms_error": relative_rms, "max_absolute_error": peak,
            "relative_rms_tolerance": 2e-5, "max_absolute_tolerance": 2e-5,
            "verified_samples": int(x.size), "gain_fit": False}


def recover_dry(wet: np.ndarray, limited: np.ndarray, alpha: float) -> np.ndarray:
    wet = mono_array(wet, "wet calibration").astype(np.float64)
    limited = mono_array(limited, "limited calibration").astype(np.float64)
    if wet.shape != limited.shape or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("dry recovery requires equal shapes and alpha in (0,1]")
    return (limited - (1.0 - alpha) * wet) / alpha


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_wav(path: Path, *, require_float: bool = False) -> np.ndarray:
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames < 1:
        raise ValueError(f"{path}: expected nonempty 48kHz mono WAV, got {info}")
    if info.format not in ("WAV", "WAVEX") or (require_float and info.subtype != "FLOAT"):
        raise ValueError(f"{path}: expected {'float32 ' if require_float else ''}WAV")
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE or data.size != info.frames:
        raise ValueError(f"{path}: decoder/sample-count mismatch")
    return mono_array(data, str(path))


def command_units(argv: list[str]) -> int:
    return len(subprocess.list2cmdline(argv).encode("utf-16-le")) // 2 + 1


def command_groups(prefix: list[str], inputs: list[Path], options: list[str]) -> list[list[str]]:
    groups = []
    current: list[str] = []
    for path in inputs:
        candidate = current + [str(path)]
        if command_units(prefix + candidate + options) > COMMAND_UTF16_LIMIT:
            if not current:
                raise ValueError(f"single input exceeds Windows command-line budget: {path}")
            groups.append(prefix + current + options)
            current = [str(path)]
            if command_units(prefix + current + options) > COMMAND_UTF16_LIMIT:
                raise ValueError(f"single input exceeds Windows command-line budget: {path}")
        else:
            current = candidate
    if current:
        groups.append(prefix + current + options)
    return groups


class CliRunner:
    def __init__(self, args: argparse.Namespace, logs: Path):
        self.args = args
        self.logs = logs
        self.invocations: list[dict] = []

    def run(self, argv: list[str], label: str) -> None:
        index = len(self.invocations)
        stem = f"{index:03d}-{label}"
        entry = {"argv": argv, "windows_command_utf16_units": command_units(argv),
                 "stdout": f"logs/{stem}.stdout.log", "stderr": f"logs/{stem}.stderr.log"}
        self.invocations.append(entry)
        write_json(self.logs / f"{stem}.argv.json", entry)
        start = time.perf_counter()
        with (self.logs / f"{stem}.stdout.log").open("xb") as stdout, \
                (self.logs / f"{stem}.stderr.log").open("xb") as stderr:
            process = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
        entry.update(returncode=process.returncode, elapsed_seconds=time.perf_counter() - start)
        write_json(self.logs / f"{stem}.argv.json", entry)
        if process.returncode != 0:
            raise RuntimeError(f"CLI exit {process.returncode}; see {entry['stderr']}")

    def render(self, inputs: list[Path], destination: Path, strength: float, label: str) -> None:
        destination.mkdir()
        options = ["--output-dir", str(destination), "--model", str(self.args.model),
                   "--backend", self.args.backend, "--strength", format(strength, ".9g"),
                   "--output-format", "wav", "--wav-format", "float32"]
        if self.args.provider is not None:
            options.extend(["--provider", self.args.provider])
        prefix = [str(self.args.executable), "--verbose", "denoise"]
        for group in command_groups(prefix, inputs, options):
            self.run(group, label)


def calibrate(runner: CliRunner, scratch: Path, diagnostic: dict) -> dict:
    diagnostic.update(
        status="running", stage="dry_probe_rendering",
        method="verified_algebraic_dry_recovery_then_source_nominal_wet_delay",
        formal_strength=0.0, formal_mode="pure_wet_not_bypass",
        alignment_definition="nominal_wet_delay = verified_dry_delay - 480; high-band lag only verifies this",
        probe_seed=20261022, wet_strength=0.0, limited_strength=CALIBRATION_STRENGTH,
        limited_attenuation_db=1.0,
        source_history_geometry={
            "source": "crates/anr-dfn3/src/shaders/dsp.wgsl",
            "history_shift": "lines72-79: state_out five-history=[S(k-4)..S(k)]; seven-history=[S(k-6)..S(k)]",
            "dry_read": "lines100-101: state_in[32480 + 2*962 + 2*f], five-history index2 = S(k-3)",
            "wet_coarse_mask_read": "lines84-86: state_out[37290 + 4*962 + 2*f], seven-history index4 = S(k-2), times real nonnegative band gain",
            "df_region": "lines87-96: complex five-tap DF only for f<96; at48k/FFT960 that is below4.8kHz",
            "synthesis": "lines118-123 and135-137: shared960-sample analysis/overlap-add, hop480; same STFT latency for dry and wet",
            "deduction": "dry source spectrum is one480-sample hop earlier than coarse wet source spectrum, so wet nominal delay=dry verified delay-480",
        },
        caveat="Low-frequency complex DF can change phase; no full-band correlation peak defines formal alignment. The nominal wet delay is frozen for all utterances; no clean target or gain fit is used.",
    )
    # Independent full-band random probe; reconstruction includes every sample,
    # including its non-hop-aligned end. Only appended terminal zeros are ignored.
    probe = np.random.default_rng(20261022).uniform(-0.25, 0.25, 2 * SAMPLE_RATE + 137).astype(np.float32)
    padded = np.pad(probe, (0, padded_length(probe.size, MAX_DELAY + HOP) - probe.size))
    probe_path = scratch / "independent-probe.wav"
    sf.write(probe_path, padded, SAMPLE_RATE, subtype="FLOAT")
    diagnostic.update(probe_sha256=sha256(probe_path), probe_samples=int(probe.size),
                      padded_probe_samples=int(padded.size))
    arrays = {}
    for label, value in (("probe-wet", 0.0), ("probe-limited", CALIBRATION_STRENGTH)):
        destination = scratch / label
        runner.render([probe_path], destination, value, label)
        arrays[label] = read_wav(destination / probe_path.name, require_float=True)
        if arrays[label].size != padded.size:
            raise ValueError(f"{label}: CLI did not preserve padded probe length")
    diagnostic["stage"] = "dry_reconstruction_verification"
    # model.rs:173 computes alpha in f32. Verified reconstruction catches any
    # material device-specific powf rounding or incorrect mixing assumption.
    alpha = float(np.float32(10.0 ** (-1.0 / 20.0)))
    diagnostic["limited_dry_alpha"] = alpha
    dry = recover_dry(arrays["probe-wet"], arrays["probe-limited"], alpha)
    sf.write(scratch / "recovered-dry.wav", dry.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    diagnostic["dry"] = estimate_delay(probe, dry, MAX_DELAY)
    dry_delay = diagnostic["dry"]["delay_samples"]
    # Record measured errors even when the strict reconstruction check rejects.
    difference = dry[dry_delay:dry_delay + probe.size] - probe.astype(np.float64)
    diagnostic["dry_reconstruction_measured"] = {
        "relative_rms_error": float(np.sqrt(np.mean(difference ** 2)) / np.sqrt(np.mean(probe.astype(np.float64) ** 2))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "verified_samples": int(probe.size), "gain_fit": False,
        "relative_rms_tolerance": 2e-5, "max_absolute_tolerance": 2e-5,
    }
    diagnostic["dry_reconstruction"] = verify_reconstruction(probe, dry, dry_delay)
    nominal = nominal_wet_delay(dry_delay)
    diagnostic.update(nominal_wet_delay_samples=nominal, delay_samples=nominal,
                      dry_minus_nominal_wet_delay_samples=HOP, stage="high_band_wet_verification")
    high = make_high_band_probe()
    high_padded = np.pad(high, (0, padded_length(high.size, MAX_DELAY + HOP) - high.size))
    high_path = scratch / "independent-high-band-probe.wav"
    sf.write(high_path, high_padded, SAMPLE_RATE, subtype="FLOAT")
    high_info = {
        "seed": 20261023, "passband_hz": [8000, 18000], "fir_taps": 1025,
        "strength": 0.0, "samples": int(high.size), "sha256": sha256(high_path),
        "purpose": "verify source-derived nominal delay in real positive coarse-mask region, not complex DF",
    }
    diagnostic["high_band_wet_verification"] = high_info
    high_dir = scratch / "probe-high-band-wet"
    runner.render([high_path], high_dir, 0.0, "probe-high-band-wet")
    high_raw = read_wav(high_dir / high_path.name, require_float=True)
    if high_raw.size != high_padded.size:
        raise ValueError("high-band probe: CLI did not preserve padded length")
    high_info["input_rms"] = float(np.sqrt(np.mean(high.astype(np.float64) ** 2)))
    aligned_high = high_raw[nominal:nominal + high.size].astype(np.float64)
    high_info["output_rms_at_nominal_delay"] = float(np.sqrt(np.mean(aligned_high ** 2)))
    try:
        high_info["measurement"] = estimate_delay(high, high_raw, MAX_DELAY)
    except ValueError as error:
        high_info["measurement_error"] = str(error)
        raise ValueError(f"high-band wet delay cannot be verified (possibly suppressed probe): {error}") from error
    if high_info["measurement"]["delay_samples"] != nominal:
        raise ValueError(f"high-band measured lag {high_info['measurement']['delay_samples']} disagrees with source-derived nominal lag {nominal}")
    high_info["matches_nominal_delay"] = True
    diagnostic.update(status="verified", stage="complete")
    return diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing paths are never reused")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--provider", help="ORT only; omitted ORT provider is recorded as cpu")
    parser.add_argument("--strength", type=float, default=0.0, help="formal pure-wet reference requires0; zero is NOT bypass")
    parser.add_argument("--limit", type=int, help="positive smoke-only cap; omit for formal export")
    args = parser.parse_args()
    if not math.isfinite(args.strength) or args.strength != 0.0:
        parser.error("--strength must be0 for the source-aligned pure-wet comparison;0 is not bypass")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive and is smoke-only")
    if args.provider is not None and args.backend != "ort":
        parser.error("--provider applies only to --backend ort")
    for key in ("executable", "model", "input_dir", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def export(args: argparse.Namespace) -> dict:
    if not math.isfinite(args.strength) or args.strength != 0.0:
        raise ValueError("formal reference export requires strength0: pure wet, not bypass")
    if not args.executable.is_file() or not args.model.is_file() or not args.input_dir.is_dir():
        raise ValueError("executable, model and input directory must exist")
    if args.output_dir.is_relative_to(args.input_dir):
        raise ValueError("output/scratch must be outside the read-only input corpus")
    model_hash = sha256(args.model)
    if model_hash != MODEL_SHA256:
        raise ValueError(f"not the fixed pretrained DFN3 model: SHA256={model_hash}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output namespace: {args.output_dir}")
    sources = sorted((p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"),
                     key=lambda p: p.name)
    if not sources or len({p.stem.casefold() for p in sources}) != len(sources):
        raise ValueError("input directory is empty or has case-insensitive output-name collisions")
    selected = sources if args.limit is None else sources[:args.limit]
    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "logs"
    logs.mkdir()
    scratch = args.output_dir / "scratch"
    scratch.mkdir()
    runner = CliRunner(args, logs)
    manifest = {"schema_version": 1, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "exporter_sha256": sha256(Path(__file__).resolve()),
                "model": str(args.model), "model_sha256": model_hash,
                "executable": str(args.executable), "executable_sha256": sha256(args.executable),
                "input_dir": str(args.input_dir), "source_count": len(sources), "evaluated_count": len(selected),
                "limit": args.limit, "smoke_only": args.limit is not None,
                "backend": args.backend, "provider": (args.provider or "cpu") if args.backend == "ort" else None,
                "strength": args.strength,
                "mode": "pure_wet_not_bypass",
                "strength_mapping": "attenuation_db=round_f32(strength/100*60), whole dB; zero disables attenuation limit, returns wet",
                "attenuation_db": math.floor(float(np.float32(np.float32(args.strength) / np.float32(100) * np.float32(60))) + 0.5),
                "sample_rate": SAMPLE_RATE, "channels": 1, "wav_subtype": "FLOAT",
                "normalization": "none", "clipping": False, "per_file_alignment": False,
                "dependencies": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                                 "soundfile": sf.__version__},
                "platform": platform.platform(), "invocations": runner.invocations, "files": [],
                "source_evidence": {
                    "strength": "crates/anr-dfn3/src/worker.rs:161-164 and crates/anr-ort/src/dfn3.rs:214-217: strength*0.6 rounded to whole dB; tiny positive strength is NOT identity",
                    "mixing": "crates/anr-dfn3/src/shaders/dsp.wgsl:82-108: db0 wet; limited=(1-alpha)*wet+alpha*dry",
                    "delay": "dsp.wgsl:72-79,84-85,100-101: wet state_out seven-history index4; dry state_in five-history index2",
                    "flush": "crates/anr-cli/src/audio/pipeline.rs:185-193: partial-hop padding only; no neural flush",
                    "reset": "crates/anr-cli/src/audio/pipeline.rs:55-56 resets each engine once per new stream"}}
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    start = time.perf_counter()
    try:
        runner.run([str(args.executable), "backends", "--json"], "backend-inventory")
        manifest["calibration"] = {}
        calibration = calibrate(runner, scratch, manifest["calibration"])
        delay = calibration["nominal_wet_delay_samples"]
        padded_dir = scratch / "padded-inputs"
        padded_dir.mkdir()
        padded_paths = []
        for path in selected:
            audio = read_wav(path)
            length = padded_length(audio.size, delay + HOP)
            padded = np.zeros(length, dtype=np.float32)
            padded[:audio.size] = audio
            prepared = padded_dir / (path.stem + ".wav")
            sf.write(prepared, padded, SAMPLE_RATE, subtype="FLOAT")
            padded_paths.append(prepared)
            manifest["files"].append({"filename": path.name, "source_sha256": sha256(path),
                                      "samples": int(audio.size), "padded_samples": length,
                                      "source_peak": float(np.max(np.abs(audio))),
                                      "source_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))})
        write_json(manifest_path, manifest)
        raw_dir = scratch / "raw-enhanced"
        runner.render(padded_paths, raw_dir, args.strength, "reference-batch")
        actual_names = {p.name for p in raw_dir.iterdir() if p.is_file()}
        if actual_names != {p.name for p in padded_paths}:
            raise ValueError("CLI output-name set does not exactly match the prepared inputs")
        for record, prepared in zip(manifest["files"], padded_paths):
            raw = read_wav(raw_dir / prepared.name, require_float=True)
            audio = trim_delayed(raw, record["samples"], delay, record["padded_samples"])
            destination = args.output_dir / record["filename"]
            sf.write(destination, audio, SAMPLE_RATE, subtype="FLOAT")
            record.update(output_sha256=sha256(destination), output_peak=float(np.max(np.abs(audio))),
                          output_rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))))
        manifest.update(status="complete", elapsed_seconds=time.perf_counter() - start,
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
        return manifest
    except BaseException as error:
        if "calibration" in manifest and manifest["calibration"].get("status") != "verified":
            manifest["calibration"].update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.perf_counter() - start)
        write_json(manifest_path, manifest)
        raise


def main() -> int:
    args = parse_args()
    try:
        result = export(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"DFN3 export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "smoke_only": result["smoke_only"],
                      "evaluated_count": result["evaluated_count"],
                      "delay_samples": result["calibration"]["delay_samples"],
                      "manifest": str(args.output_dir / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
