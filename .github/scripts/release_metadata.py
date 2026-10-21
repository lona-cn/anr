"""Decide release identity before any native build or GitHub mutation."""

import argparse
import json
import os
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def release_metadata(event: str, ref: str, commit: str, version: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("release commit must be a complete lowercase Git SHA")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", version):
        raise ValueError("workspace version must be a safe Cargo version")
    result = {
        "publish": False,
        "tag": "",
        "label": f"ci-{commit[:12]}",
        "prerelease": True,
        "commit": commit,
    }
    if event != "push":
        return result
    if ref in ("refs/heads/main", "refs/heads/master"):
        label = f"{ref.removeprefix('refs/heads/')}-{commit[:12]}"
        result.update(publish=True, tag=label, label=label)
    elif ref.startswith("refs/tags/v"):
        tag = ref.removeprefix("refs/tags/")
        if tag != f"v{version}":
            raise ValueError(f"tag {tag!r} must match workspace version v{version}")
        result.update(publish=True, tag=tag, label=version, prerelease="-" in version.split("+", 1)[0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()
    with (ROOT / "Cargo.toml").open("rb") as source:
        version = tomllib.load(source)["workspace"]["package"]["version"]
    result = release_metadata(
        os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF"], os.environ["GITHUB_SHA"], version
    )
    runtime = json.loads((ROOT / ".github/onnxruntime.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", runtime["source_commit"]):
        raise ValueError("ONNX Runtime source commit must be a full Git SHA")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", runtime["version"]):
        raise ValueError("ONNX Runtime version must contain three numeric components")
    result.update(native_version=runtime["version"], native_commit=runtime["source_commit"])
    if args.output is not None:
        with args.output.open("a", encoding="utf-8", newline="\n") as output:
            for name, value in result.items():
                output.write(f"{name}={str(value).lower() if isinstance(value, bool) else value}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
