#!/usr/bin/env python3
"""Build a self-contained native CLI ZIP from verified release inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


REPOSITORY = Path(__file__).resolve().parents[2]
TARGETS = {
    "x86_64-pc-windows-msvc": "windows",
    "x86_64-unknown-linux-gnu": "linux",
    "aarch64-apple-darwin": "macos",
    "x86_64-apple-darwin": "macos",
}
REQUIRED_MODEL_FILES = {"dfn3.onnx", "LICENSE-MIT", "LICENSE-APACHE", "NOTICE.md"}
LIBRARY_PATTERNS = {
    "windows": re.compile(r"onnxruntime(?:[._-][A-Za-z0-9._-]+)?\.dll", re.IGNORECASE),
    "linux": re.compile(r"libonnxruntime(?:_[A-Za-z0-9_-]+)?\.so(?:\.[0-9]+)*"),
    "macos": re.compile(r"libonnxruntime(?:[._-][A-Za-z0-9._-]+)?\.dylib"),
}
PRIMARY_LIBRARY = {
    "windows": "onnxruntime.dll",
    "linux": "libonnxruntime.so",
    "macos": "libonnxruntime.dylib",
}


class PackagingError(Exception):
    pass


def regular_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise PackagingError(f"required nonempty file is missing: {path}")
    return path


def flat_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise PackagingError(f"not a portable flat filename: {value!r}")
    if value.endswith(".") or value.split(".", 1)[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise PackagingError(f"not a portable flat filename: {value!r}")
    return value


def model_inputs(model_dir: Path, manifest_path: Path) -> list[tuple[Path, str]]:
    regular_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise PackagingError("model manifest must be a JSON object")
    for field in ("repository", "tag", "license"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise PackagingError(f"model manifest requires a nonempty {field!r}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise PackagingError("model manifest requires a nonempty files array")
    inputs = []
    names = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PackagingError("model manifest file entries must be objects")
        name = flat_name(entry.get("name"))
        if name.casefold() in names or name.casefold() == "dfn3.json":
            raise PackagingError(f"duplicate or reserved model filename: {name}")
        names.add(name.casefold())
        size = entry.get("size")
        digest = entry.get("sha256")
        if type(size) is not int or size <= 0:
            raise PackagingError(f"model manifest has invalid size for {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise PackagingError(f"model manifest has invalid SHA256 for {name}")
        source = regular_file(model_dir / name)
        if source.stat().st_size != size:
            raise PackagingError(f"model size mismatch: {source}")
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest.lower():
            raise PackagingError(f"model SHA256 mismatch: {source}")
        inputs.append((source, name))
    actual_names = {name for _, name in inputs}
    missing = REQUIRED_MODEL_FILES - actual_names
    if missing:
        raise PackagingError(f"model manifest omits required files: {', '.join(sorted(missing))}")
    return inputs


def native_libraries(directory: Path, platform: str) -> list[Path]:
    if not directory.is_dir():
        raise PackagingError(f"runtime library directory is missing: {directory}")
    libraries = []
    names = set()
    for source in sorted(directory.iterdir()):
        if not LIBRARY_PATTERNS[platform].fullmatch(source.name):
            continue
        flat_name(source.name)
        if source.name.casefold() in names:
            raise PackagingError(f"colliding native library filename: {source.name}")
        names.add(source.name.casefold())
        regular_file(source)
        # Symlink aliases become independent regular files in the final ZIP.
        # Refuse links out of the explicitly supplied runtime directory.
        if source.resolve().parent != directory.resolve():
            raise PackagingError(f"runtime library resolves outside its input directory: {source}")
        libraries.append(source)
    if PRIMARY_LIBRARY[platform] not in {source.name for source in libraries}:
        raise PackagingError(f"runtime directory requires {PRIMARY_LIBRARY[platform]}: {directory}")
    return libraries


def copy_input(source: Path, destination: Path) -> None:
    regular_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)


def mac_tool(*arguments: str) -> str:
    try:
        result = subprocess.run(
            arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120,
        )
    except FileNotFoundError as error:
        raise PackagingError(f"macOS packaging requires Xcode command-line tool {arguments[0]}") from error
    except subprocess.CalledProcessError as error:
        raise PackagingError(
            f"macOS packaging command failed: {arguments!r}\n{error.stderr.strip()}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise PackagingError(f"macOS packaging command timed out: {arguments!r}") from error
    return result.stdout


def mac_dependencies(path: Path) -> list[str]:
    output = mac_tool("otool", "-L", str(path))
    dependencies = []
    for line in output.splitlines()[1:]:
        match = re.match(r"\s+(.+?)\s+\(compatibility version ", line)
        if match:
            dependencies.append(match.group(1))
        elif line.strip():
            raise PackagingError(f"cannot parse Mach-O dependency for {path}: {line}")
    return dependencies


def mac_rpaths(path: Path) -> set[str]:
    lines = mac_tool("otool", "-l", str(path)).splitlines()
    rpaths = set()
    for index, line in enumerate(lines):
        if line.strip() != "cmd LC_RPATH":
            continue
        for field in lines[index + 1:index + 4]:
            match = re.match(r"\s*path (.+) \(offset [0-9]+\)", field)
            if match:
                rpaths.add(match.group(1))
                break
        else:
            raise PackagingError(f"cannot parse LC_RPATH for {path}")
    return rpaths


def prepare_macos(executable: Path, libraries: list[Path]) -> None:
    if sys.platform != "darwin":
        raise PackagingError("macOS release packaging must run on its native macOS runner")
    bundled_names = {path.name for path in libraries}
    # Everything below acts exclusively on staged copies, including signatures.
    for path in [*libraries, executable]:
        path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        own_id = None
        if path != executable:
            identifiers = mac_tool("otool", "-D", str(path)).splitlines()[1:]
            if len(identifiers) != 1 or not identifiers[0].strip():
                raise PackagingError(f"expected one Mach-O dylib ID: {path}")
            own_id = identifiers[0].strip()
        arguments = ["install_name_tool"]
        if own_id is not None:
            arguments.extend(["-id", f"@rpath/{path.name}"])
        for dependency in mac_dependencies(path):
            if dependency == own_id:
                continue
            basename = dependency.rsplit("/", 1)[-1]
            if basename in bundled_names:
                replacement = f"@loader_path/{basename}"
                if dependency != replacement:
                    arguments.extend(["-change", dependency, replacement])
            elif not dependency.startswith(("/usr/lib/", "/System/Library/")):
                raise PackagingError(f"unbundled non-system Mach-O dependency in {path.name}: {dependency}")
        desired_rpath = "@executable_path" if path == executable else "@loader_path"
        existing_rpaths = mac_rpaths(path)
        for rpath in sorted(existing_rpaths - {desired_rpath}):
            arguments.extend(["-delete_rpath", rpath])
        if desired_rpath not in existing_rpaths:
            arguments.extend(["-add_rpath", desired_rpath])
        if len(arguments) > 1:
            mac_tool(*arguments, str(path))
        # Replacing install names invalidates the upstream/Rust signature. Sign
        # every flattened alias as well; none is left with a stale signature.
        mac_tool("codesign", "--force", "--sign", "-", "--timestamp=none", str(path))


def write_zip(root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                raise PackagingError(f"staging contains a non-regular file: {path}")
            name = path.relative_to(root.parent).as_posix()
            info = zipfile.ZipInfo.from_file(path, arcname=name, strict_timestamps=False)
            # Always emit Unix regular-file metadata, even when the packager is
            # running on Windows. No symlinks or host-specific absolute paths.
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | stat.S_IMODE(path.stat().st_mode)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def package(arguments: argparse.Namespace) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", arguments.label) or arguments.label.endswith("."):
        raise PackagingError("label must be a safe filename token starting with an ASCII letter or digit")
    platform = TARGETS[arguments.target]
    binary = regular_file(arguments.binary)
    if platform != "windows" and not binary.stat().st_mode & 0o111:
        raise PackagingError(f"Unix CLI input is not executable: {binary}")
    model_files = model_inputs(arguments.model_dir, arguments.model_manifest)
    libraries = native_libraries(arguments.runtime_lib_dir, platform)
    project_files = [
        (REPOSITORY / "LICENSE-MIT", "LICENSE-MIT"),
        (REPOSITORY / "LICENSE-APACHE", "LICENSE-APACHE"),
        (REPOSITORY / "crates/anr-cli/README.md", "README.md"),
        (REPOSITORY / "crates/anr-cli/THIRD-PARTY-NOTICES.md", "THIRD-PARTY-NOTICES.md"),
    ]
    notices = [(arguments.runtime_notice_dir / name, name) for name in ("LICENSE", "ThirdPartyNotices.txt")]
    for source, _ in [*project_files, *notices]:
        regular_file(source)
    stem = f"anr-{arguments.label}-{arguments.target}"
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    destination = arguments.output_dir / f"{stem}.zip"
    with tempfile.TemporaryDirectory(prefix="anr-package-") as temporary:
        root = Path(temporary) / stem
        root.mkdir()
        executable = root / ("anr.exe" if platform == "windows" else "anr")
        copy_input(binary, executable)
        for source, name in model_files:
            copy_input(source, root / "models" / name)
        copy_input(arguments.model_manifest, root / "models/dfn3.json")
        for source, name in project_files:
            copy_input(source, root / name)
        for source, name in notices:
            copy_input(source, root / "licenses/onnxruntime" / name)
        staged_libraries = []
        for source in libraries:
            staged = root / source.name
            copy_input(source, staged)
            staged_libraries.append(staged)
        if platform == "macos":
            prepare_macos(executable, staged_libraries)
        descriptor, partial_name = tempfile.mkstemp(prefix=f".{stem}-", suffix=".zip.tmp", dir=arguments.output_dir)
        os.close(descriptor)
        partial = Path(partial_name)
        try:
            write_zip(root, partial)
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--runtime-lib-dir", required=True, type=Path)
    parser.add_argument("--runtime-notice-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-manifest", type=Path, default=REPOSITORY / ".github/models/dfn3.json")
    arguments = parser.parse_args()
    try:
        destination = package(arguments)
    except (PackagingError, OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(destination)


if __name__ == "__main__":
    main()
