#!/usr/bin/env python3
"""Extract a release ZIP and exercise its CLI, runtime, and bundled DFN3 model."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import wave
import zipfile


MODEL_SHA256 = "fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9"
MODEL_FILES = {"dfn3.onnx", "LICENSE-MIT", "LICENSE-APACHE", "NOTICE.md"}
SEARCH_ENVIRONMENT = {
    "ORT_DYLIB_PATH",
    "ORT_LIB_LOCATION",
    "ORT_PREFER_DYNAMIC_LINK",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def extract_bundle(archive, destination):
    """Reject nonportable paths and links before extracting any archive members."""
    root_name = archive.stem
    require(root_name.startswith("anr-"), "release archive must be named anr-<label>-<host>.zip")
    with zipfile.ZipFile(archive) as bundle:
        members = []
        seen = set()
        for member in bundle.infolist():
            name = member.filename
            path = PurePosixPath(name)
            require(
                name and not path.is_absolute() and "\\" not in name
                and all(
                    part not in {"", ".", ".."} and not part.endswith((" ", "."))
                    and not PureWindowsPath(part).is_reserved()
                    and not any(character in part for character in '<>:"|?*\0')
                    for part in name.rstrip("/").split("/")
                ),
                f"unsafe ZIP member path: {name!r}",
            )
            require(path.parts[0] == root_name, f"ZIP member is outside {root_name!r}: {name!r}")
            require(len(path.parts) > 1 or member.is_dir(), "archive root must be a directory")
            # Reject case aliases even on a case-sensitive host: the same ZIP must be portable.
            key = str(path).casefold()
            require(key not in seen, f"duplicate ZIP member: {name!r}")
            seen.add(key)
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            require(
                kind in {0, stat.S_IFDIR if member.is_dir() else stat.S_IFREG},
                f"ZIP links and special files are not supported: {name!r}",
            )
            members.append((member, path, mode))
        require(members, "release ZIP is empty")
        for member, path, mode in members:
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            if os.name != "nt" and mode & 0o777:
                target.chmod(mode & 0o777)
    return destination / root_name


def required_file(path):
    require(path.is_file() and path.stat().st_size > 0, f"required bundle file missing or empty: {path}")
    return path


def verify_bundle(root):
    if sys.platform == "win32":
        binary_name, runtime_name = "anr.exe", "onnxruntime.dll"
    elif sys.platform == "linux":
        binary_name, runtime_name = "anr", "libonnxruntime.so"
    elif sys.platform == "darwin":
        binary_name, runtime_name = "anr", "libonnxruntime.dylib"
    else:
        raise RuntimeError(f"unsupported smoke-test platform: {sys.platform}")
    executable = required_file(root / binary_name)
    required_file(root / runtime_name)
    for relative in (
        "LICENSE-MIT", "LICENSE-APACHE", "README.md", "THIRD-PARTY-NOTICES.md",
        "licenses/onnxruntime/LICENSE", "licenses/onnxruntime/ThirdPartyNotices.txt",
    ):
        required_file(root / relative)
    if os.name != "nt":
        require(os.access(executable, os.X_OK), "ZIP executable mode is missing its executable bit")

    model_directory = root / "models"
    manifest_path = required_file(model_directory / "dfn3.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "bundled model manifest must be an object")
    entries = manifest.get("files")
    require(isinstance(entries, list), "bundled model manifest must contain a files array")
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict), "invalid bundled model manifest entry")
        name = entry.get("name")
        require(isinstance(name, str) and name in MODEL_FILES, f"unexpected model manifest file: {name!r}")
        require(name not in seen, f"duplicate model manifest file: {name}")
        seen.add(name)
        size, digest = entry.get("size"), entry.get("sha256")
        require(type(size) is int and size > 0, f"invalid manifest size for {name}")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"invalid manifest SHA256 for {name}")
        path = required_file(model_directory / name)
        require(path.stat().st_size == size, f"bundled model file size mismatch: {name}")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        require(actual == digest, f"bundled model file SHA256 mismatch: {name}")
        if name == "dfn3.onnx":
            require(actual == MODEL_SHA256, "bundle does not contain the specified real DFN3 model")
    require(seen == MODEL_FILES, f"bundled model manifest is missing: {', '.join(sorted(MODEL_FILES - seen))}")
    return executable, model_directory / "dfn3.onnx"


def invoke(executable, cwd, environment, *arguments, audio=None, timeout=60):
    """Capture stdout as bytes; failure reports must never print raw audio."""
    command = [str(executable), *arguments]
    try:
        result = subprocess.run(
            command, input=audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd, env=environment, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        stderr = (error.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"CLI command {arguments!r} timed out after {timeout}s\nstderr:\n{stderr}") from None
    stderr = result.stderr.decode("utf-8", errors="replace")
    require(result.returncode == 0, f"CLI command {arguments!r} failed with exit {result.returncode}\nstderr:\n{stderr}")
    return result.stdout


def check_samples(output, sample_count, label):
    require(len(output) == sample_count * 4, f"{label}: expected {sample_count * 4} output bytes, got {len(output)}")
    samples = struct.unpack(f"<{sample_count}f", output)
    require(all(math.isfinite(value) for value in samples), f"{label}: output contains non-finite samples")
    return samples


def smoke(executable, model, control_model, unrelated):
    environment = {key: value for key, value in os.environ.items() if key.upper() not in SEARCH_ENVIRONMENT}

    def run(*arguments, **options):
        return invoke(executable, unrelated, environment, *arguments, **options)

    run("--help")
    backends = json.loads(run("backends", "--json"))
    require(isinstance(backends, dict) and "backends" in backends, "backends --json omitted backends")

    # The real control graph doubles its second frame using carried model state.
    audio = struct.pack("<960f", *([0.125] * 960))
    output = run(
        "denoise", "-", "--input-format", "f32le", "--input-rate", "48000",
        "--input-channels", "1", "--output", "-", "--output-format", "f32le",
        "--model", str(control_model), "--strength", "100", audio=audio,
    )
    samples = check_samples(output, 960, "control graph")
    require(all(abs(value - 0.125) < 1e-5 for value in samples[:480]), "control graph first-frame values are incorrect")
    require(all(abs(value - 0.25) < 1e-5 for value in samples[480:]), "control graph carried-state values are incorrect")

    source = unrelated / "双声道 input.wav"
    target = unrelated / "双声道 output.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48000)
        writer.writeframes(struct.pack("<960h", *([4096, 0] * 480)))
    run("denoise", str(source), "--output", str(target), "--model", str(control_model))
    with wave.open(str(target), "rb") as reader:
        require(
            (reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth()) == (48000, 2, 480, 2),
            "stereo WAV output metadata is incorrect",
        )
        require(len(reader.readframes(480)) == 480 * 2 * 2, "stereo WAV output has an incorrect data length")
    info = json.loads(run("inspect", str(target), "--json"))
    require(
        isinstance(info, dict) and info.get("sample_rate") == 48000
        and info.get("channels") == 2 and info.get("frames") == 480,
        "inspect reported incorrect stereo WAV metadata",
    )

    # An incomplete final frame catches truncation/padding regressions; this is not a quality metric.
    count = 48001
    real_audio = struct.pack(f"<{count}f", *(0.1 * math.sin(2 * math.pi * 440 * index / 48000) for index in range(count)))
    real_output = run(
        "denoise", "-", "--input-format", "f32le", "--input-rate", "48000",
        "--input-channels", "1", "--output", "-", "--output-format", "f32le",
        "--model", str(model), "--strength", "100", audio=real_audio, timeout=180,
    )
    check_samples(real_output, count, "bundled real DFN3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--control-model", required=True, type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    control_model = args.control_model.resolve(strict=True)
    required_file(archive)
    required_file(control_model)
    with tempfile.TemporaryDirectory(prefix="anr-release-smoke-") as temporary:
        scratch = Path(temporary)
        extracted = scratch / "解压 bundle"
        extracted.mkdir()
        root = extract_bundle(archive, extracted)
        executable, model = verify_bundle(root)
        unrelated = scratch / "无关工作目录 with spaces"
        unrelated.mkdir()
        smoke(executable, model, control_model, unrelated)
    print("Release ZIP smoke passed: control-state values, stereo WAV, and real DFN3 length/finite output.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, wave.Error) as error:
        print(f"Release ZIP smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
