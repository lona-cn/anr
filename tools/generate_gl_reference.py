"""Regenerate compute-backend audio using ONNX's NumPy reference evaluator.

Development-only dependencies: onnx==1.19.1 numpy==2.2.6. No ONNX Runtime.
Run from the workspace root: python tools/generate_gl_reference.py
The fixed model SHA is intentionally checked before generating expectations.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator

MODEL_SHA = "fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9"
ROOT = Path(__file__).resolve().parents[1]


def reference_model(path: Path) -> onnx.ModelProto:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA:
        raise ValueError("unsupported model SHA-256")
    model = copy.deepcopy(onnx.load_model_from_string(data))
    nodes = []
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.domain == "com.microsoft" and node.op_type == "FusedConv":
            activation = attrs.pop("activation").decode("utf-8")
            temporary = node.output[0] + "__conv"
            nodes.append(helper.make_node("Conv", list(node.input), [temporary], **attrs))
            nodes.append(helper.make_node(activation, [temporary], list(node.output)))
        elif node.domain == "com.microsoft" and node.op_type == "FusedMatMul":
            if any(attrs.get(k, 0) for k in ("transA", "transB", "transBatchA", "transBatchB")):
                raise ValueError("unexpected fused matrix transpose")
            temporary = node.output[0] + "__matmul"
            scale = node.output[0] + "__alpha"
            model.graph.initializer.append(numpy_helper.from_array(np.array(attrs["alpha"], np.float32), scale))
            nodes.append(helper.make_node("MatMul", list(node.input), [temporary]))
            nodes.append(helper.make_node("Mul", [temporary, scale], list(node.output)))
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def main() -> None:
    evaluator = ReferenceEvaluator(reference_model(ROOT / "models/dfn3.onnx"))
    rng = np.random.default_rng(20261021)
    count = 60 * 480 + 13
    t = np.arange(count, dtype=np.float64) / 48000
    audio = (0.08 * np.sin(2 * np.pi * 440 * t) + 0.035 * rng.normal(size=count)).astype(np.float32)
    audio[:3 * 480] = 0
    audio[27 * 480:30 * 480] = 0
    state = np.zeros(45304, np.float32)
    output = []
    for index, start in enumerate(range(0, count, 480)):
        frame = np.zeros(480, np.float32)
        chunk = audio[start:start + 480]
        frame[:len(chunk)] = chunk
        db = 48 if index < 12 else 0 if index < 24 else 3 if index < 36 else 60
        enhanced, state, _ = evaluator.run(None, {
            "input_frame": frame, "states": state, "atten_lim_db": np.array(db, np.float32),
        })
        if not np.isfinite(enhanced).all() or not np.isfinite(state).all():
            raise ValueError(f"non-finite reference at frame {index}")
        output.append(enhanced[:len(chunk)])
    expected = np.concatenate(output).astype("<f4")
    for backend in ("gl", "gles", "vk", "wgpu", "dx"):
        directory = ROOT / f"crates/anr-{backend}/tests/fixtures"
        directory.mkdir(parents=True, exist_ok=True)
        audio.astype("<f4").tofile(directory / "reference_input.f32le")
        expected.tofile(directory / "reference_output.f32le")
    print(f"Generated {len(audio)} samples / {len(output)} stateful frames; output peak {np.max(np.abs(expected)):.9g}")


if __name__ == "__main__":
    main()
