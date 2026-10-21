#!/usr/bin/env python3
"""Fetch immutable model assets and pinned CPU ONNX Runtime distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


REPOSITORY = Path(__file__).resolve().parents[2]
REQUIRED_MODEL_FILES = {"dfn3.onnx", "LICENSE-MIT", "LICENSE-APACHE", "NOTICE.md"}
RUNTIME_ARCHIVES = {
    "x86_64-pc-windows-msvc": "onnxruntime-win-x64-1.28.0.zip",
    "x86_64-unknown-linux-gnu": "onnxruntime-linux-x64-1.28.0.tgz",
    "aarch64-apple-darwin": "onnxruntime-osx-arm64-1.28.0.tgz",
}
LIBRARY_PATTERNS = {
    "x86_64-pc-windows-msvc": re.compile(r"onnxruntime(?:[._-][A-Za-z0-9._-]+)?\.dll", re.IGNORECASE),
    "x86_64-unknown-linux-gnu": re.compile(r"libonnxruntime(?:_[A-Za-z0-9_-]+)?\.so(?:\.[0-9]+)*"),
    "aarch64-apple-darwin": re.compile(r"libonnxruntime(?:[._-][A-Za-z0-9._-]+)?\.dylib"),
}
PRIMARY_LIBRARIES = {
    "x86_64-pc-windows-msvc": "onnxruntime.dll",
    "x86_64-unknown-linux-gnu": "libonnxruntime.so",
    "aarch64-apple-darwin": "libonnxruntime.dylib",
}
NOTICES = ("LICENSE", "ThirdPartyNotices.txt")
CHUNK_SIZE = 1024 * 1024
SOCKET_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 15 * 60
MAX_EXTRACTED_SIZE = 2 * 1024 * 1024 * 1024


class FetchError(Exception):
    pass


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        destination = urllib.parse.urlsplit(newurl)
        if destination.scheme != "https" or destination.username or destination.password:
            raise FetchError("refusing a non-HTTPS or credential-bearing download redirect")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def flat_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise FetchError(f"not a portable flat filename: {value!r}")
    if value.endswith(".") or value.split(".", 1)[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise FetchError(f"not a portable flat filename: {value!r}")
    return value


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise FetchError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def load_manifest(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as source:
        content = source.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise FetchError("manifest exceeds 1 MiB")
    manifest = json.loads(content.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(manifest, dict):
        raise FetchError("manifest must be a JSON object")
    return manifest, content


def pinned_asset(entry: object, name: str) -> tuple[int, str]:
    if not isinstance(entry, dict):
        raise FetchError(f"manifest entry for {name} must be an object")
    size = entry.get("size")
    digest = entry.get("sha256")
    if type(size) is not int or size <= 0:
        raise FetchError(f"manifest requires a positive integer size for {name}")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise FetchError(f"manifest requires a SHA256 digest for {name}")
    return size, digest.lower()


def release_url(repository: object, tag: object, filename: str) -> str:
    if not isinstance(repository, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository
    ):
        raise FetchError("manifest requires a GitHub OWNER/REPOSITORY")
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise FetchError("manifest requires a safe immutable release tag")
    return f"https://github.com/{repository}/releases/download/{urllib.parse.quote(tag, safe='')}/{urllib.parse.quote(filename, safe='')}"


def download(url: str, destination: Path, name: str, size: int, digest: str) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "anr-release-assets", "Accept-Encoding": "identity"}
    )
    # Public assets deliberately never use GH_TOKEN, gh authentication, or retries.
    opener = urllib.request.build_opener(HTTPSRedirectHandler())
    started = time.monotonic()
    actual_size = 0
    actual_digest = hashlib.sha256()
    try:
        with opener.open(request, timeout=SOCKET_TIMEOUT) as response:
            if response.status != 200:
                raise FetchError(f"download of {name} returned HTTP {response.status}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as error:
                    raise FetchError(f"download of {name} returned an invalid Content-Length") from error
                if declared_size != size:
                    raise FetchError(f"download size mismatch for {name}: expected {size}, HTTP reports {declared_size}")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise FetchError(f"download of {name} returned unexpected content encoding")
            with destination.open("xb") as output:
                while True:
                    if time.monotonic() - started > DOWNLOAD_TIMEOUT:
                        raise FetchError(f"download of {name} exceeded {DOWNLOAD_TIMEOUT} seconds")
                    chunk = response.read(min(CHUNK_SIZE, size - actual_size + 1))
                    if not chunk:
                        break
                    actual_size += len(chunk)
                    if actual_size > size:
                        raise FetchError(f"download of {name} exceeds pinned size {size}")
                    actual_digest.update(chunk)
                    output.write(chunk)
    except urllib.error.HTTPError as error:
        # Do not expose signed redirect URLs, headers, or authentication details.
        raise FetchError(f"download of {name} failed (HTTP {error.code})") from None
    except urllib.error.URLError:
        raise FetchError(f"download of {name} failed (connection or TLS error)") from None
    except TimeoutError:
        raise FetchError(f"download of {name} timed out") from None
    if actual_size != size:
        raise FetchError(f"download size mismatch for {name}: expected {size}, got {actual_size}")
    if actual_digest.hexdigest() != digest:
        raise FetchError(f"download SHA256 mismatch for {name}")


def verified_file(path: Path, size: int, digest: str) -> bool:
    if path.is_symlink():
        raise FetchError(f"refusing symlink output: {path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise FetchError(f"output is not a regular file: {path}")
    if path.stat().st_size != size:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == digest


def output_directory(path: Path) -> None:
    if path.is_symlink():
        raise FetchError(f"refusing symlink output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise FetchError(f"output is not a directory: {path}")


def install_file(source: Path, destination: Path) -> None:
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise FetchError(f"refusing non-regular output: {destination}")
    os.replace(source, destination)


def fetch_model(args: argparse.Namespace) -> None:
    manifest, content = load_manifest(args.manifest)
    if not isinstance(manifest.get("license"), str) or not manifest["license"].strip():
        raise FetchError("model manifest requires a nonempty license")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise FetchError("model manifest requires a nonempty files array")
    assets = []
    names = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise FetchError("model file entries must be objects")
        name = flat_name(entry.get("name"))
        if name.casefold() in names or name.casefold() == "dfn3.json":
            raise FetchError(f"duplicate or reserved model filename: {name}")
        names.add(name.casefold())
        size, digest = pinned_asset(entry, name)
        url = release_url(manifest.get("repository"), manifest.get("tag"), name)
        assets.append((name, size, digest, url))
    missing = REQUIRED_MODEL_FILES - {asset[0] for asset in assets}
    if missing:
        raise FetchError(f"model manifest omits required files: {', '.join(sorted(missing))}")
    output_directory(args.output_dir)
    with tempfile.TemporaryDirectory(prefix=".model-fetch-", dir=args.output_dir) as temporary:
        staged = Path(temporary)
        downloaded = []
        for name, size, digest, url in assets:
            if verified_file(args.output_dir / name, size, digest):
                print(f"Verified existing {name}")
                continue
            print(f"Downloading {name}")
            download(url, staged / name, name, size, digest)
            downloaded.append(name)
        # Every input is now verified; only then expose new final filenames.
        for name in downloaded:
            install_file(staged / name, args.output_dir / name)
        (staged / "dfn3.json").write_bytes(content)
        install_file(staged / "dfn3.json", args.output_dir / "dfn3.json")
    print(f"Verified model assets: {args.output_dir}")


def member_name(value: str) -> str:
    name = value.removesuffix("/")
    if not name or "\\" in name or ":" in name or "\x00" in name:
        raise FetchError(f"unsafe archive member name: {value!r}")
    parts = name.split("/")
    if name.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise FetchError(f"unsafe archive member name: {value!r}")
    return name


def selected_members(members: dict, root: str, target: str) -> dict[str, str]:
    selected = {}
    folded = set()
    for name in members:
        path = PurePosixPath(name)
        if path.parent == PurePosixPath(root) and path.name in NOTICES:
            destination = path.name
        elif path.parent == PurePosixPath(root, "lib") and LIBRARY_PATTERNS[target].fullmatch(path.name):
            flat_name(path.name)
            destination = f"lib/{path.name}"
        else:
            continue
        if destination.casefold() in folded:
            raise FetchError(f"colliding runtime output name: {destination}")
        folded.add(destination.casefold())
        selected[destination] = name
    for required in (*NOTICES, f"lib/{PRIMARY_LIBRARIES[target]}"):
        if required not in selected:
            raise FetchError(f"runtime archive is missing {root}/{required}")
    return selected


def copy_member(source, destination: Path, size: int, mode: int) -> None:
    if size <= 0 or size > MAX_EXTRACTED_SIZE:
        raise FetchError(f"invalid runtime member size: {destination.name}")
    remaining = size
    with destination.open("xb") as output:
        while remaining:
            chunk = source.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise FetchError(f"truncated runtime member: {destination.name}")
            output.write(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise FetchError(f"runtime member exceeds declared size: {destination.name}")
    destination.chmod(0o755 if mode & 0o111 else 0o644)


def extract_zip(archive: Path, stage: Path, root: str, target: str) -> None:
    with zipfile.ZipFile(archive) as package:
        members = {}
        for info in package.infolist():
            name = member_name(info.filename)
            if name in members:
                raise FetchError(f"duplicate runtime archive member: {name}")
            members[name] = info
        selected = selected_members(members, root, target)
        if sum(members[name].file_size for name in selected.values()) > MAX_EXTRACTED_SIZE:
            raise FetchError("runtime extraction exceeds 2 GiB")
        for destination, name in selected.items():
            info = members[name]
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if info.is_dir() or kind not in (0, stat.S_IFREG):
                raise FetchError(f"runtime ZIP member is not a regular file: {name}")
            with package.open(info) as source:
                copy_member(source, stage / destination, info.file_size, mode)


def resolve_tar_library(members: dict, name: str, root: str) -> tarfile.TarInfo:
    seen = set()
    while True:
        if name in seen:
            raise FetchError(f"cyclic runtime library alias: {name}")
        seen.add(name)
        info = members.get(name)
        if info is None:
            raise FetchError(f"runtime library alias has a missing target: {name}")
        if info.isfile():
            return info
        if not (info.issym() or info.islnk()):
            raise FetchError(f"runtime library is not a regular file or alias: {name}")
        link = info.linkname
        if not link or link.startswith("/") or "\\" in link or ":" in link or "\x00" in link:
            raise FetchError(f"unsafe runtime library alias: {name}")
        # Tar hard links name an archive member; symbolic links are parent-relative.
        name = posixpath.normpath(posixpath.join(posixpath.dirname(name), link) if info.issym() else link)
        if PurePosixPath(name).parent != PurePosixPath(root, "lib"):
            raise FetchError("runtime library alias resolves outside the archive lib directory")


def extract_tar(archive: Path, stage: Path, root: str, target: str) -> None:
    with tarfile.open(archive, mode="r:gz") as package:
        members = {}
        for info in package.getmembers():
            name = member_name(info.name)
            if name in members:
                raise FetchError(f"duplicate runtime archive member: {name}")
            members[name] = info
        selected = selected_members(members, root, target)
        resolved = {}
        for destination, name in selected.items():
            if destination.startswith("lib/"):
                info = resolve_tar_library(members, name, root)
                if not LIBRARY_PATTERNS[target].fullmatch(PurePosixPath(info.name).name):
                    raise FetchError(f"runtime library alias targets a non-library: {name}")
            else:
                info = members[name]
                if not info.isfile():
                    raise FetchError(f"runtime notice is not a regular file: {name}")
            resolved[destination] = info
        if sum(info.size for info in resolved.values()) > MAX_EXTRACTED_SIZE:
            raise FetchError("runtime extraction exceeds 2 GiB")
        for destination, info in resolved.items():
            source = package.extractfile(info)
            if source is None:
                raise FetchError(f"cannot read runtime archive member: {info.name}")
            with source:
                copy_member(source, stage / destination, info.size, info.mode)


def fetch_runtime(args: argparse.Namespace) -> None:
    if args.target == "x86_64-apple-darwin":
        raise FetchError("Intel macOS has no official ONNX Runtime 1.28.0 distribution; use the workflow's pinned source build and its actual LICENSE/ThirdPartyNotices.txt")
    if args.target not in RUNTIME_ARCHIVES:
        raise FetchError(f"unsupported runtime target: {args.target}")
    manifest, _ = load_manifest(args.manifest)
    if manifest.get("repository") != "microsoft/onnxruntime" or manifest.get("version") != "1.28.0":
        raise FetchError("runtime manifest must pin microsoft/onnxruntime version 1.28.0")
    targets = manifest.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(RUNTIME_ARCHIVES):
        raise FetchError("runtime manifest must contain exactly the three official CPU targets")
    for target, expected in RUNTIME_ARCHIVES.items():
        entry = targets[target]
        if not isinstance(entry, dict) or entry.get("archive") != expected:
            raise FetchError(f"runtime manifest archive mismatch for {target}")
        pinned_asset(entry, expected)
    entry = targets[args.target]
    name = flat_name(entry["archive"])
    size, digest = pinned_asset(entry, name)
    url = release_url(manifest["repository"], f"v{manifest['version']}", name)
    root = name[:-4]
    output_directory(args.output_dir)
    with tempfile.TemporaryDirectory(prefix=".runtime-fetch-", dir=args.output_dir) as temporary:
        temporary = Path(temporary)
        archive = temporary / name
        print(f"Downloading {name}")
        download(url, archive, name, size, digest)
        stage = temporary / "staged"
        (stage / "lib").mkdir(parents=True)
        if name.endswith(".zip"):
            extract_zip(archive, stage, root, args.target)
        else:
            extract_tar(archive, stage, root, args.target)
        library_dir = args.output_dir / "lib"
        output_directory(library_dir)
        expected_names = {path.name for path in (stage / "lib").iterdir()}
        unexpected = {path.name for path in library_dir.iterdir()} - expected_names
        if unexpected:
            raise FetchError(f"runtime output lib contains unexpected existing entries: {', '.join(sorted(unexpected))}")
        # Check every destination before installing any part of the verified archive.
        staged_files = sorted((stage / "lib").iterdir()) + [stage / notice for notice in NOTICES]
        for source in staged_files:
            destination = args.output_dir / source.relative_to(stage)
            if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                raise FetchError(f"refusing non-regular output: {destination}")
        for source in staged_files:
            install_file(source, args.output_dir / source.relative_to(stage))
    print(f"Verified runtime libraries and notices: {args.output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model", help="download and verify all immutable model assets")
    model.add_argument("--output-dir", type=Path, required=True)
    model.add_argument("--manifest", type=Path, default=REPOSITORY / ".github/models/dfn3.json")
    runtime = commands.add_parser("runtime", help="download and extract a pinned CPU runtime")
    runtime.add_argument("--target", required=True)
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument("--manifest", type=Path, default=REPOSITORY / ".github/onnxruntime.json")
    args = parser.parse_args()
    try:
        if args.command == "model":
            fetch_model(args)
        else:
            fetch_runtime(args)
    except (FetchError, OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, EOFError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
