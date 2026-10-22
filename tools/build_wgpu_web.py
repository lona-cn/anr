"""Build browser WebGPU bindings with installed Cargo and wasm-bindgen tools.

Run from the workspace root: python tools/build_wgpu_web.py
No tools, targets, or models are installed or downloaded by this helper.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = "wasm32-unknown-unknown"


def installed_tool(name: str) -> str:
    local = ROOT / "target" / "web-tools" / "bin"
    executable = shutil.which(name, path=str(local)) or shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"Required tool '{name}' was not found in {local} or PATH. "
            "Install it before running this helper."
        )
    return executable


def run(command: list[str], *, capture: bool = False) -> str:
    print(f"+ {shlex.join(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as error:
        details = ""
        if capture:
            details = f"\n{error.stdout or ''}{error.stderr or ''}".rstrip()
        raise RuntimeError(
            f"Command failed with exit code {error.returncode}: "
            f"{shlex.join(command)}{details}"
        ) from error
    except OSError as error:
        raise RuntimeError(f"Could not execute {command[0]}: {error}") from error
    return result.stdout if capture else ""


def main() -> None:
    cargo = installed_tool("cargo")
    wasm_bindgen = installed_tool("wasm-bindgen")
    metadata = json.loads(run([
        cargo, "metadata", "--locked", "--no-deps", "--format-version", "1",
        "--manifest-path", str(ROOT / "Cargo.toml"),
    ], capture=True))
    target_directory = Path(metadata["target_directory"])
    run([
        cargo, "build", "--locked", "--release", "--target", TARGET, "-p", "anr-wgpu",
        "--manifest-path", str(ROOT / "Cargo.toml"),
    ])
    wasm = target_directory / TARGET / "release" / "anr_wgpu.wasm"
    if not wasm.is_file():
        raise RuntimeError(f"Cargo succeeded but the expected WASM artifact is missing: {wasm}")
    output = ROOT / "target" / "anr-wgpu-web"
    package = output / "pkg"
    package.mkdir(parents=True, exist_ok=True)
    run([
        wasm_bindgen, "--target", "web", "--out-dir", str(package),
        "--out-name", "anr_wgpu", str(wasm),
    ])
    shutil.copyfile(
        ROOT / "crates" / "anr-wgpu" / "examples" / "web" / "worker.js",
        output / "worker.js",
    )
    print(f"Browser package: {package / 'anr_wgpu.js'}")
    print(f"Module Worker: {output / 'worker.js'}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
