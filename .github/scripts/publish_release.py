#!/usr/bin/env python3
"""Publish an exact four-platform release without replacing tags or assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import quote


TARGETS = (
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
)


class PublicationError(Exception):
    pass


@dataclass(frozen=True)
class Artifact:
    path: Path
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.name


def fingerprint(path: Path) -> Artifact:
    if path.is_symlink() or not path.is_file():
        raise PublicationError(f"required regular file is missing: {path}")
    before = path.stat()
    if before.st_size <= 0:
        raise PublicationError(f"required file is empty: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PublicationError(f"file changed while hashing: {path}")
    return Artifact(path, after.st_size, digest)


def validate_arguments(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", args.repository):
        raise PublicationError("--repository must be OWNER/REPO on github.com")
    if args.repository.split("/", 1)[1] in {".", ".."}:
        raise PublicationError("invalid repository name")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.commit):
        raise PublicationError("--commit must be a full 40-character hexadecimal SHA")
    args.commit = args.commit.lower()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", args.label):
        raise PublicationError("--label must be a portable filename token")
    branch = re.fullmatch(r"(main|master)-([0-9a-f]{12})", args.tag)
    if branch:
        if args.label != args.tag or branch[2] != args.commit[:12] or not args.prerelease:
            raise PublicationError("branch releases require matching branch-12sha label/tag and --prerelease")
        return
    version = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?",
        args.label,
    )
    if not version or args.tag != "v" + args.label:
        raise PublicationError("version releases require a semantic version label and matching v<label> tag")
    suffix = version[4]
    if suffix and any(not part or (part.isdigit() and len(part) > 1 and part[0] == "0")
                      for part in suffix.split(".")):
        raise PublicationError("invalid semantic version prerelease identifiers")
    if bool(suffix) != args.prerelease:
        raise PublicationError("--prerelease must agree with the version prerelease suffix")


def collect_artifacts(directory: Path, label: str) -> tuple[list[Artifact], bytes]:
    if not directory.is_dir():
        raise PublicationError(f"release directory is missing: {directory}")
    expected = {f"anr-{label}-{target}.zip" for target in TARGETS}
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.name.lower().endswith(".zip")}
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        extra = ", ".join(sorted(actual - expected)) or "none"
        raise PublicationError(f"release ZIP set mismatch; missing: {missing}; unexpected: {extra}")
    artifacts = [fingerprint(directory / name) for name in sorted(expected)]
    sums = "".join(f"{item.sha256}  {item.name}\n" for item in artifacts).encode("utf-8")
    existing = directory / "SHA256SUMS"
    if existing.exists() or existing.is_symlink():
        if existing.is_symlink() or not existing.is_file() or existing.read_bytes() != sums:
            raise PublicationError("existing local SHA256SUMS does not exactly match the four ZIPs")
    return artifacts, sums


class GitHub:
    """Run gh without shell interpolation or exposing its captured diagnostics."""

    def __init__(self, repository: str):
        self.repository = repository
        self.base = f"repos/{repository}"

    def run(self, command: list[str], *, cwd: Path | None = None,
            payload: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["gh", *command], cwd=cwd, input=payload, capture_output=True,
                encoding="utf-8", errors="replace", timeout=timeout, check=False,
            )
        except FileNotFoundError as error:
            raise PublicationError("installed authenticated GitHub CLI (gh) is required") from error
        except subprocess.TimeoutExpired as error:
            raise PublicationError("gh timed out; remote outcome may be unknown; inspect or rerun safely") from error

    def api(self, endpoint: str, *, method: str = "GET", payload: dict | None = None,
            allow_missing: bool = False, expect_list: bool = False,
            input_file: Path | None = None) -> dict | list | None:
        command = ["api", endpoint, "--hostname", "github.com", "--method", method,
                   "--include", "-H", "Accept: application/vnd.github+json",
                   "-H", "X-GitHub-Api-Version: 2022-11-28"]
        if payload is not None:
            command.extend(["--input", "-"])
        if input_file is not None:
            command.extend(["--input", str(input_file), "-H", "Content-Type: application/octet-stream"])
        result = self.run(command, payload=json.dumps(payload) if payload is not None else None,
                          timeout=900 if input_file is not None else 120)
        response = re.split(r"\r?\n\r?\n", result.stdout, maxsplit=1)
        status = re.match(r"HTTP/\S+\s+([0-9]{3})\b", response[0])
        code = int(status[1]) if status else None
        if code == 404 and allow_missing and method == "GET":
            return None
        if result.returncode != 0 or code is None or not 200 <= code < 300:
            detail = f"HTTP {code}" if code is not None else "no successful HTTP response"
            raise PublicationError(f"GitHub {method} {endpoint} failed ({detail}); no automatic retry")
        try:
            document = json.loads(response[1])
        except (IndexError, json.JSONDecodeError) as error:
            raise PublicationError("GitHub returned an invalid JSON response") from error
        if not isinstance(document, list if expect_list else dict):
            raise PublicationError("GitHub returned an unexpected JSON response")
        return document

    def resolve(self, ref: str, *, allow_missing: bool = False) -> str | None:
        result = self.api(f"{self.base}/commits/{quote(ref, safe='')}", allow_missing=allow_missing)
        if result is None:
            return None
        sha = result.get("sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            raise PublicationError("GitHub returned an invalid commit SHA")
        return sha.lower()

    def release(self, tag: str) -> dict | None:
        release = self.api(f"{self.base}/releases/tags/{quote(tag, safe='')}", allow_missing=True)
        if release is not None:
            return release
        # A draft need not have a Git ref yet and may not be exposed by the
        # release-by-tag endpoint. Authenticated release listing includes drafts.
        matches = []
        for page in range(1, 101):
            releases = self.api(f"{self.base}/releases?per_page=100&page={page}", expect_list=True)
            for candidate in releases:
                if not isinstance(candidate, dict):
                    raise PublicationError("release listing contains an invalid entry")
                if candidate.get("tag_name") == tag:
                    matches.append(candidate)
            if len(matches) > 1:
                raise PublicationError("multiple releases use the requested tag; refusing any mutation")
            if len(releases) < 100:
                return matches[0] if matches else None
        raise PublicationError("release listing exceeded 100 pages; refusing to assume no draft exists")

    def upload(self, release_id: int, artifact: Artifact) -> None:
        if fingerprint(artifact.path) != artifact:
            raise PublicationError(f"local artifact changed before upload: {artifact.name}")
        # Address the inspected draft by ID, not its possibly nonexistent tag.
        # POST has no replace operation: duplicate names fail without clobbering.
        endpoint = (f"https://uploads.github.com/{self.base}/releases/{release_id}/assets"
                    f"?name={quote(artifact.name, safe='')}")
        self.api(endpoint, method="POST", input_file=artifact.path)


def verify_tag(github: GitHub, args: argparse.Namespace, *, must_exist: bool = False) -> None:
    resolved = github.resolve(args.tag, allow_missing=True)
    if resolved is None:
        if must_exist or args.tag.startswith("v"):
            raise PublicationError("required existing version/release tag was not found")
    elif resolved != args.commit:
        raise PublicationError("existing tag resolves to another commit; refusing to move or reuse it")


def verify_release(github: GitHub, args: argparse.Namespace, release: dict,
                   artifacts: list[Artifact], *, release_id: int | None = None) -> list[Artifact]:
    identifier = release.get("id")
    if type(identifier) is not int or identifier <= 0 or (release_id is not None and identifier != release_id):
        raise PublicationError("release identity changed or is invalid")
    if release.get("tag_name") != args.tag:
        raise PublicationError("release tag does not match the requested tag")
    if type(release.get("draft")) is not bool or release.get("prerelease") is not args.prerelease:
        raise PublicationError("release draft/prerelease metadata does not match the requested release")
    target = release.get("target_commitish")
    if not isinstance(target, str) or not target:
        raise PublicationError("release target_commitish is missing")
    if release["draft"] and target.lower() != args.commit:
        raise PublicationError("draft target_commitish must be the exact requested commit SHA")
    target_sha = target.lower() if re.fullmatch(r"[0-9a-fA-F]{40}", target) else github.resolve(target)
    if target_sha != args.commit:
        raise PublicationError("release target_commitish resolves to another commit")
    verify_tag(github, args, must_exist=not release["draft"])
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise PublicationError("release assets are missing from the API response")
    expected = {item.name: item for item in artifacts}
    seen = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise PublicationError("release contains an invalid asset")
        name = asset.get("name")
        if not isinstance(name, str) or name not in expected or name in seen:
            raise PublicationError("release has unexpected or duplicate assets; refusing any mutation")
        seen.add(name)
        local = expected[name]
        digest = asset.get("digest")
        if (type(asset.get("size")) is not int or asset["size"] != local.size
                or not isinstance(digest, str) or digest.lower() != "sha256:" + local.sha256):
            raise PublicationError(f"existing asset {name} has a different or unknown size/SHA256 digest")
        if asset.get("state", "uploaded") != "uploaded":
            raise PublicationError(f"existing asset {name} is not fully uploaded")
    return [item for item in artifacts if item.name not in seen]


def release_body(args: argparse.Namespace, sums: bytes) -> str:
    return (
        f"ANR native CPU packages for commit {args.commit}.\n\n"
        "Four ZIPs provide Windows x64, Linux x64, macOS Apple Silicon, and macOS Intel builds. "
        "Each package bundles CPU ONNX Runtime 1.28.0 and models/dfn3.onnx beside the executable. "
        "From the extracted directory, run `./anr denoise input.wav --output clean.wav --model models/dfn3.onnx` "
        "(use `.\\anr.exe` on Windows). The --model argument remains required.\n\n"
        "Extract the entire ZIP and keep its native shared libraries beside the executable. "
        "Packages require a compatible OS and CPU architecture. Linux requires a compatible glibc; "
        "Windows may require the Microsoft Visual C++ runtime. macOS security policies may require "
        "explicit approval for downloaded, non-notarized binaries. GPU execution providers are not bundled.\n\n"
        "The supplied DFN3 model bytes are redistributed under MIT OR Apache-2.0; their license and "
        "notice files are in models/. This statement does not license user-supplied models. "
        "Project and ONNX Runtime license/notice files are included in each ZIP.\n\n"
        "Verify downloaded ZIPs against the attached SHA256SUMS file:\n\n```text\n"
        + sums.decode("utf-8") + "```\n"
    )


def publish(github: GitHub, args: argparse.Namespace, artifacts: list[Artifact], sums: bytes) -> None:
    if github.resolve(args.commit) != args.commit:
        raise PublicationError("requested commit does not resolve exactly in the target repository")
    verify_tag(github, args)
    with tempfile.TemporaryDirectory(prefix="anr-release-checksums-") as temporary:
        checksum_path = Path(temporary) / "SHA256SUMS"
        checksum_path.write_bytes(sums)
        expected = [*artifacts, fingerprint(checksum_path)]
        release = github.release(args.tag)
        if release is None:
            payload = {
                "tag_name": args.tag, "target_commitish": args.commit,
                "name": f"ANR {args.label}", "body": release_body(args, sums),
                "draft": True, "prerelease": args.prerelease,
            }
            if args.prerelease:
                payload["make_latest"] = "false"
            release = github.api(f"{github.base}/releases", method="POST", payload=payload)
        if release is None:
            raise PublicationError("release creation returned no release")
        missing = verify_release(github, args, release, expected)
        identifier = release["id"]
        if not release["draft"]:
            if missing:
                raise PublicationError("published release is incomplete; immutable public releases are never edited")
            print(f"Already complete: {args.repository} {args.tag}; no changes made.")
            return
        for artifact in missing:
            current = github.api(f"{github.base}/releases/{identifier}")
            if current is None:
                raise PublicationError("draft release disappeared")
            remaining = verify_release(github, args, current, expected, release_id=identifier)
            if not current["draft"]:
                raise PublicationError("release became public during upload; refusing further mutation")
            if artifact in remaining:
                github.upload(identifier, artifact)
        current = github.api(f"{github.base}/releases/{identifier}")
        if current is None:
            raise PublicationError("draft release disappeared")
        if verify_release(github, args, current, expected, release_id=identifier):
            raise PublicationError("draft is incomplete after upload; refusing publication")
        if current["draft"]:
            payload = {"draft": False}
            if args.prerelease:
                payload["make_latest"] = "false"
            published = github.api(f"{github.base}/releases/{identifier}", method="PATCH", payload=payload)
            if published is None or published.get("draft") is not False:
                raise PublicationError("publication was not confirmed; inspect the release before rerunning")
            if verify_release(github, args, published, expected, release_id=identifier):
                raise PublicationError("published release assets were not confirmed complete")
        print(f"Published complete release: {args.repository} {args.tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--prerelease", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and hash local inputs only; no gh calls or writes")
    args = parser.parse_args()
    try:
        validate_arguments(args)
        artifacts, sums = collect_artifacts(args.directory.resolve(), args.label)
        if args.dry_run:
            print(f"Local validation only: {args.repository} {args.tag} at {args.commit}")
            print("Remote tag/release state has not been checked. No files or GitHub resources were changed.")
            print(sums.decode("utf-8"), end="")
        else:
            publish(GitHub(args.repository), args, artifacts, sums)
    except (PublicationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
