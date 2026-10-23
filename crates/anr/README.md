# anr

[English](#english) | [简体中文](#简体中文)

## English

Backend selection and model discovery for DFN3 noise reduction. Native implementations share `NoiseReduction` and process **48 kHz mono `f32` audio** with equal-length input/output. Browser WebGPU has a separate asynchronous API. The library does not capture microphones, decode files, resample, or split channels. Requires the latest stable Rust toolchain.

### Backend and feature selection

**No execution backend is enabled by default.** Defaults are `download-models`, `copy-models`, and the weak `ort-download-binaries` option. Model preparation is independent of runtime selection.

| Backend feature | Public type | Native requirements |
| --- | --- | --- |
| `ort` | `anr::DFN3` / `anr::ort::DFN3` | Compatible ONNX Runtime; CPU convenience constructor, configurable providers. |
| `gl` | `anr::gl::DFN3` | Desktop OpenGL 4.3, Windows/WGL or Linux hardware EGL. |
| `gles` | `anr::gles::DFN3` | Real OpenGL ES 3.1 context, Windows/WGL or Linux hardware EGL/pbuffer. |
| `vk` | `anr::vk::DFN3` | Hardware Vulkan 1.1 on Windows/Linux. |
| `wgpu` | Native `anr::wgpu::DFN3`; WASM `anr::wgpu::web::DFN3` | Hardware Vulkan/DX12/Metal on Windows/Linux/macOS; WebGPU in browsers. |
| `dx` | `anr::dx::DFN3` | Windows hardware D3D12 and system D3DCompiler 47. No WARP. |
| `cuda` | `anr::cuda::DFN3` | Windows/Linux NVIDIA driver API 12.0+ and NVRTC 12/13; no build-time CUDA SDK. |
| `tensorrt` | `anr::tensorrt::DFN3` | Windows/Linux NVIDIA GPU; TensorRT/CUDA SDKs and C++ toolchain at build time. |
| `openvino` | `anr::openvino::{DFN3, Device}` | OpenVINO C++ headers/core link library and C++ toolchain at build time; Runtime, ONNX frontend and selected device plugin at runtime. |
| `opencl` | `anr::opencl::DFN3` | Windows/Linux hardware GPU ICD supporting OpenCL 1.2; no build-time SDK. |

Backends are additive Cargo features, not an automatic preference/fallback list. The application chooses the type to construct. Native `cuda` / `tensorrt` are independent implementations, not ORT's CUDA/TensorRT providers. `dx` is native D3D12, not ORT DirectML. GLES does not fall back to desktop GL; no Android, WebGL, or macOS GLES context is provided.

Additional facade features:

- `download-models` (default): downloads and verifies the pinned model unless `ANR_MODEL_PATH` supplies a local file.
- `copy-models` (default): copies that file beside build outputs; does not imply downloading.
- `ort-download-binaries` (default): forwards to `anr-ort/download-binaries` **only when ORT is enabled**; alone it does not activate `anr-ort`.
- `ort-load-dynamic`: **enables `ort`** and runtime shared-library loading.
- `ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`, `ort-xnnpack`: **enable `ort`** and the corresponding provider support. They do not select devices or install drivers/vendor libraries.

Cargo unifies features across dependencies. `default-features = false` applies to that dependency declaration, not to another crate's enabled features. An all-features build includes the native TensorRT/OpenVINO SDK requirements.

### ORT quickstart

For an external application, after `0.3.0` is published to crates.io:

```toml
[dependencies]
anr = { version = "0.3.0", features = ["ort"] }
```

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut reducer = DFN3::from_file(&environment, anr::default_model_path()?)?;
    reducer.set_factor(80.0);

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    reducer.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    reducer.reset();
    Ok(())
}
```

This processes a silent frame. Import `NoiseReduction` for its methods. `DFN3::from_file(&environment, path)` explicitly selects CPU, one intra-op thread, and no spinning; the application owns the `Environment`. For provider/options control, configure a session with `anr::ort::session_from_file` and wrap it with `DFN3::new(session)`.

For a supplied runtime instead of build-time runtime downloads:

```toml
anr = { version = "0.3.0", default-features = false, features = ["ort-load-dynamic"] }
```

Set `ORT_DYLIB_PATH` to a compatible shared-library file before `initialize_runtime()`. This configuration also disables automatic model preparation; pass a model path or bundle the model beside the executable. Cargo.lock pins `ort`/`ort-sys` `2.0.0-rc.15`, API 27, without `ndarray`. Its downloader uses ONNX Runtime 1.31.0; the repository's application release configuration separately pins 1.28.0 in `.github/onnxruntime.json`.

See the [ORT reference](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-ort/README.md) for the complete port schema, runtime setup, provider examples, and `prepare_model` / `ModelTarget` byte-buffer API. `session_from_file` prepares CUDA/TensorRT/DirectML graphs in memory once per session without rewriting source files. Direct upstream session loading bypasses preparation. GPU sessions may still place individual operators on CPU; provider registration failures remain errors.

From a source checkout, the minimal raw-audio example accepts 48 kHz mono little-endian IEEE 754 `f32` samples, not WAV:

```text
cargo run -p anr --features ort --example denoise -- input.f32le output.f32le
```

An optional model path precedes the input/output paths. The example refuses to overwrite an existing output file; use the CLI for decoded audio formats.

### Model preparation and discovery

The supported native model is [the pinned DFN3 artifact](https://github.com/lona-cn/anr/releases/download/model-dfn3-fe5eb64fa2e4/dfn3.onnx), SHA-256:

```text
fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9
```

Every non-ORT DFN3 backend, including browser WebGPU and native OpenVINO, validates this exact hash. ORT validates a port/semantic contract instead and can accept compatible alternate exports.

At build time, `download-models` verifies and reuses `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/dfn3.onnx` (`~/.cargo` by default). Set `ANR_MODEL_PATH` before building to select a local file and skip downloading. Absolute paths avoid ambiguity; the build script canonicalizes the path and records it. Local overrides are not checksum-verified by the downloader; compute constructors still enforce the fixed hash. Source-file changes trigger model preparation again.

`copy-models` writes `models/dfn3.onnx` into the Cargo profile directory and its `examples` and `deps` directories, including custom target/profile and cross-compilation layouts. With `copy-models` alone, `ANR_MODEL_PATH` is required. With `download-models` alone, only the persistent cache is used.

At runtime, `default_model_path()` checks:

1. `models/dfn3.onnx` beside the running executable.
2. The absolute model path recorded at build time, if it still exists.

It does not search the current directory, read a runtime `ANR_MODEL_PATH`, or download anything. The executable-relative model works even with all defaults disabled. Ship that model and its required notices with an application; build/cache paths are not portable. A persistent cache can outlive `cargo install`'s temporary build directory.

A cold cache with `CARGO_NET_OFFLINE=true`, download failure, or checksum mismatch fails the build. Supply a local model for offline preparation. Model downloads use native TLS; docs.rs builds skip preparation. See [model provenance and notices](https://github.com/lona-cn/anr/blob/v0.3.0/.github/models/NOTICE.md) before redistribution.

### Native streaming contract

All native DFN3 types implement `NoiseReduction: Send`; the trait supports `Box<dyn NoiseReduction>`. `Send` permits moving an instance between threads, not concurrent calls or nonblocking execution. Keep one reducer per continuous channel; native compute constructors own a worker thread and `run` waits for its result. Do not assume audio-callback safety.

- Constants: `SAMPLE_RATE = 48000`, `FRAME_SIZE = 480` (10 ms), `STATE_SIZE = 45304`.
- `run(sample_rate, input, output)` requires nonempty input at 48 kHz and equal-length output. Those validation errors leave output and state unchanged.
- Calls may contain any nonzero number of samples. A short final frame is zero-padded and advances a full frame of state, but only the requested samples are returned. Prefer multiples of 480 for continuous audio; many short calls are not equivalent to processing the same samples together.
- `reset()` clears audio/feature/network state and preserves strength. Native compute defers the reset until the next frame.
- No backend adds model-latency compensation or an end-of-stream flush.

An in-flight execution failure may follow successful frames and leave output/state partly advanced; there is no rollback. ORT callers should reset before reuse and recreate an unusable session. Native workers mark a reset pending after frame failure, but a failed runtime/device may require recreation. CUDA/OpenCL reject execution after a GPU error; Vulkan submission/fence/wait failures also invalidate the executor. Reset does not repair device loss. Shared error variants are documented in [`anr-core`](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-core/README.md).

Initial attenuation is **3 dB** (`factor() == 5`). `set_factor` clamps to `0..=100` and sets `round(factor / 100 * 60)` dB. `factor()` converts the actual setting back and rounds, so it may differ from the supplied value. NaN is ignored; positive/negative infinity saturates at the upper/lower bound. Only ORT additionally exposes fractional-dB `set_atten_lim_db` / `get_atten_lim_db` (`0..=60` dB).

Strength is an attenuation limit, not a dry/wet mix. Zero still executes the model and is not an unchanged-audio bypass.

### Native construction and deployment

For any native non-ORT backend, disable defaults if model preparation is not wanted and enable its feature:

```toml
anr = { version = "0.3.0", default-features = false, features = ["gles"] }
```

```rust,no_run
use anr::{NoiseReduction, Result, gles::DFN3};

fn main() -> Result<()> {
    let mut reducer = DFN3::from_file("models/dfn3.onnx")?;
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    reducer.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    Ok(())
}
```

The same one-path constructor applies to `gl`, `vk`, `wgpu`, `dx`, `cuda`, `tensorrt`, `opencl`, and default OpenVINO CPU. Advanced GL callers may choose `GlApi` through `anr::gl::DFN3::from_file_with_api(path, api)`; ordinary `gl::DFN3::from_file` remains desktop-only.

GL/GLES/Vulkan/wgpu/D3D12 use the fixed plan and WGSL kernels in `anr-dfn3`; Naga generates GLSL/GLSL ES/SPIR-V/HLSL where needed. The ONNX supplies parameters rather than a general-purpose runtime graph. No external model conversion, Python runtime, or shader compiler is required. CUDA/OpenCL execute the complete fixed FP32 schedule, including FFT, normalization, network inference, filtering, and overlap-add. These GPU backends do not substitute ORT or CPU inference.

**Vulkan:** specializes fixed Network modes/shapes and shares pipelines across matching dispatches, keeping buffer offsets and attenuation dynamic. GRUs use a private, once-transposed FP32 weight layout and parallel K reductions. Initialization measures widths 16/32/64; devices without GPU timestamps use width 32. No compilation or kernel selection occurs per frame, and `run` still waits for GPU completion/readback.

Readback prefers host-cached coherent memory when available, retaining the original compatible coherent memory type otherwise.

Dense `64→64` pointwise convolutions can optionally use `VK_KHR_cooperative_matrix`. Initialization checks a subgroup-scope `16×16×16` FP16-input/FP32-accumulator matrix, subgroup size 32 or 64, full subgroups and vote operations, 16,640 bytes of workgroup memory, five storage bindings, `shaderFloat16`, the Vulkan memory model, and Naga's required 16-bit storage features. GPU timing selects the matrix kernel only if it beats FP32 for that layer; missing support or timestamps retains the original FP32 pointwise path and Vulkan 1.1 minimum. The matrix path packs static weights once into an additional 128 KiB FP16 high/residual buffer and shares workgroup input tiles. Accumulation, original weights, activations and state remain FP32; out-of-half-range input tiles use the original convolution. Results are not promised to be bit-identical to FP32. Pipeline compilation and optional timing add initialization work.

**CUDA:** runtime-loads the Driver API and NVRTC, preferring major 13 then 12. Windows candidates are `nvrtc64_130_0.dll` and `nvrtc64_120_0.dll`; Linux candidates are `libnvrtc.so.13`, `.12`, and unversioned `libnvrtc.so`. Make NVRTC and its builtins dependencies discoverable through `PATH` / `LD_LIBRARY_PATH`. The driver must support the compiler's generated PTX and the compiler must support the GPU architecture; API 12.0 alone does not imply compatibility with NVRTC 13. Missing symbols/unsupported versions return errors. OpenCL instead requires the vendor's GPU ICD and does not select a CPU device.

**TensorRT:** builds two standard-layer FP32 neural engines with TF32 disabled, combined with CUDA DSP/FFT graphs. Initialization builds engines without a persistent engine cache. Building requires C++17, `TENSORRT_ROOT`, and `CUDA_ROOT` or `CUDA_PATH`; `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` override link directories.

**OpenVINO:** imports the ONNX through its native frontend and uses `MakeStateful` (`ReadValue`/`Assign`) with one `InferRequest` per stream. State remains in the selected plugin. Build with `OPENVINO_ROOT` or `OpenVINO_DIR` (CMake directory) pointing to the C++ headers and core link library; the ONNX frontend is needed at runtime for `Core::read_model`, not for compilation. Runtime also needs the selected device plugin. Official wheel `include/` / `libs/` layouts are accepted; Python is not a runtime dependency. `from_file` explicitly selects CPU. For another device, enable `openvino` and use:

```rust,no_run
use anr::{Result, openvino::{DFN3, Device}};

fn main() -> Result<()> {
    let reducer = DFN3::from_file_with_device("models/dfn3.onnx", Device::Gpu(0))?;
    println!("execution device: {}", reducer.execution_device());
    Ok(())
}
```

`Device::Cpu`, `Device::Gpu(n)` (`GPU.n`), and `Device::Npu` request exactly that plugin. Parsing accepts `CPU`, `GPU` (= `GPU.0`), `GPU.n`, and `NPU`, case-insensitively, not AUTO/HETERO or fallback lists. CPU/GPU request accuracy-oriented FP32; NPU may use FP16 and has no FP32-reference guarantee. Runtime/device/compilation failures never switch a requested GPU/NPU to CPU.

**Shared shim deployment:** TensorRT/OpenVINO builds compile private C++ shared libraries, loaded by Rust. Deploy `anr_tensorrt_native.dll` / `anr_openvino_native.dll` beside the executable (Linux: `libanr_tensorrt_native.so` / `libanr_openvino_native.so`; macOS OpenVINO: `libanr_openvino_native.dylib`), together with vendor runtimes, frontend/device plugins, and notices. Build-tree paths are not a deployment strategy.

- `ANR_TENSORRT_SHIM` / `ANR_OPENVINO_SHIM`: explicit shim files; otherwise executable and build directories are searched.
- Windows `ANR_TENSORRT_RUNTIME`: TensorRT DLL directory or file; `CUDA_ROOT` / `CUDA_PATH` locate CUDA DLLs. `PATH` also participates.
- `ANR_OPENVINO_RUNTIME`: explicit OpenVINO shared-library file.
- Platform loader paths must resolve transitive vendor dependencies, including Linux `LD_LIBRARY_PATH` where needed.

The CLI's `--provider` selects an ORT provider only; native selection uses `--backend`, with OpenVINO's `--openvino-device` separate. The CLI defaults to ORT even when another backend is compiled. File/pipe commands are in the [CLI reference](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-cli/README.md); historical timings, measured hardware, and reproduction rules are in [benchmark.md](https://github.com/lona-cn/anr/blob/v0.3.0/benchmark.md).

### Browser WebGPU

For Rust/WASM consumers, `default-features = false, features = ["wgpu"]` on `wasm32-unknown-unknown` exposes `anr::wgpu::web::DFN3`. It does **not** implement the synchronous `NoiseReduction: Send` trait:

```rust,ignore
use anr::wgpu::web::DFN3;

let mut reducer = DFN3::from_bytes(&model_bytes).await?;
reducer.set_factor(80.0);
let mut output = vec![0.0; input.len()];
reducer.run(DFN3::SAMPLE_RATE, &input, &mut output).await?;
reducer.reset();
```

Initialization and per-frame readback await WebGPU. Hash, framing, attenuation, and no-flush/no-latency-compensation rules remain the same. `reset` schedules state clearing before the next frame without submitting work. Invalid input changes nothing. A cancelled Rust run or recoverable execution error can follow completed frames and leaves a reset pending for the next frame. Device loss or a persistent uncaptured device error requires a new stream: that error is checked before reset, and `reset()` cannot repair the device. Worker clients should wait for the current request to end, then `dispose` and `init` again. No filesystem loading, WebGL, software adapter, or CPU fallback is provided. Use a WebGPU-capable browser in a secure context (HTTPS or localhost) with a hardware adapter.

The source checkout also builds direct JavaScript bindings. From the workspace root, install the target and a `wasm-bindgen-cli` matching `Cargo.lock`, then build and serve:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked --root target/web-tools
python tools/build_wgpu_web.py
python -m http.server 8000 --bind 127.0.0.1
```

The helper writes `target/anr-wgpu-web/pkg/anr_wgpu.js`, its WASM asset, and `target/anr-wgpu-web/worker.js`; it does not install tools or prepare model assets. Supply the hash-locked model yourself:

```javascript
import init, { DFN3 } from '/target/anr-wgpu-web/pkg/anr_wgpu.js';

await init();
const response = await fetch('/models/dfn3.onnx');
if (!response.ok) throw new Error(`Model request failed: HTTP ${response.status}`);
const reducer = await DFN3.fromBytes(new Uint8Array(await response.arrayBuffer()));
try {
  reducer.setFactor(80);
  const output = await reducer.run(48_000, new Float32Array(480));
  console.log(output.length); // 480
  reducer.reset();
} finally {
  reducer.free();
}
```

`run` returns `Promise<Float32Array>`. Overlapping runs are rejected; `factor`, `setFactor`, and `reset` also reject while a run is in flight. Await completion before calling them. Freeing the JS wrapper does not cancel an already-owned run.

The [module Worker example](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-wgpu/examples/web/worker.js) serializes requests with an `id` and `type`: `init` takes a model `ArrayBuffer`; `run` takes `sampleRate` and a `Float32Array`/`ArrayBuffer`; `setFactor` takes `factor`; `reset` and `dispose` take no payload. Replies contain `id`, `ok`, and `factor`, `output`, or `error` as applicable. Successful audio replies transfer `output.buffer`. Create it with `new Worker('/target/anr-wgpu-web/worker.js', { type: 'module' })`.

Microphone capture, resampling, AudioWorklet buffering, channel routing, and delay compensation belong to the host application. Default facade model-copy/download features do not supply browser assets.

### Regression references

Native numerical tests compare against checked-in `tests/fixtures/reference_input.f32le` and `reference_output.f32le`. Real-model tests are ignored by default: set `ANR_DFN3_MODEL` to the hash-locked model and select only crates supported by the host platform, hardware, and runtime. For example, on a desktop OpenGL 4.3 system, from the workspace root in PowerShell:

```powershell
$env:ANR_DFN3_MODEL = (Resolve-Path models/dfn3.onnx).Path
cargo test -p anr-gl --test dfn3 -- --ignored --nocapture
```

In a POSIX shell, set `export ANR_DFN3_MODEL="$PWD/models/dfn3.onnx"` before the same Cargo command. Substitute the selected crate or add supported `-p` selections; TensorRT/OpenVINO also need their build-time SDKs and runtime libraries. Do not run an unconditional all-backend set on unsupported hardware.

Fixtures come from ONNX's NumPy `ReferenceEvaluator`, not ONNX Runtime. To deliberately regenerate them, put the original model at `models/dfn3.onnx` and use an isolated development environment:

```sh
python -m pip install onnx==1.19.1 numpy==2.2.6
python tools/generate_gl_reference.py
```

The [generator](https://github.com/lona-cn/anr/blob/v0.3.0/tools/generate_gl_reference.py) checks the model SHA before evaluation and **overwrites both fixture files for all nine native backends**: GL, GLES, Vulkan, wgpu, D3D12, CUDA, TensorRT, OpenVINO, and OpenCL. Its deterministic stream includes silence, attenuation changes, and a 13-sample final frame. These Python packages are fixture-generation dependencies, not product runtime dependencies; normal regression runs use the existing fixtures.

For browser checks, build and serve the package as above, make `/models/dfn3.onnx` available, and run this in the browser console on that local server:

```javascript
const { runReference } = await import('/crates/anr-wgpu/tests/web/reference.js');
await runReference(
  '/target/anr-wgpu-web/pkg/anr_wgpu.js',
  '/models/dfn3.onnx',
  '/target/anr-wgpu-web/worker.js',
);
```

The [browser reference harness](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-wgpu/tests/web/reference.js) checks direct bindings and the module Worker against the same fixtures, including state/reset, partial-frame padding, rejected calls/concurrency, and stream lifecycle. It requires working hardware WebGPU; these numerical/API regressions are separate from performance benchmarks.

### License

MIT OR Apache-2.0, at your option: [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE). Check the model's provenance and terms separately; source-code licensing alone does not establish rights for every model export. ONNX Runtime and vendor provider libraries also carry their own distribution requirements.

## 简体中文

`anr` 提供 DFN3 后端选择和模型查找。原生实现共用 `NoiseReduction`，处理 **48 kHz、单声道 `f32` 音频**，输入输出等长；浏览器 WebGPU 使用独立异步 API。库不采集麦克风、不解码文件、不重采样，也不拆分声道。需要最新 Rust stable 工具链。

### 后端与 features

**默认不启用执行后端。** 默认项为 `download-models`、`copy-models` 和弱选项 `ort-download-binaries`；准备模型不等于选择运行库。

| 后端 feature | 公共类型 | 原生要求 |
| --- | --- | --- |
| `ort` | `anr::DFN3` / `anr::ort::DFN3` | 兼容 ONNX Runtime；便捷构造器使用 CPU，可自行配置 provider。 |
| `gl` | `anr::gl::DFN3` | Windows/WGL 或 Linux 硬件 EGL，桌面 OpenGL 4.3。 |
| `gles` | `anr::gles::DFN3` | Windows/WGL 或 Linux 硬件 EGL/pbuffer，真实 ES 3.1 context。 |
| `vk` | `anr::vk::DFN3` | Windows/Linux 硬件 Vulkan 1.1。 |
| `wgpu` | 原生 `anr::wgpu::DFN3`；WASM `anr::wgpu::web::DFN3` | Windows/Linux/macOS 硬件 Vulkan/DX12/Metal；浏览器 WebGPU。 |
| `dx` | `anr::dx::DFN3` | Windows 硬件 D3D12、系统 D3DCompiler 47，不用 WARP。 |
| `cuda` | `anr::cuda::DFN3` | Windows/Linux NVIDIA Driver API 12.0+、NVRTC 12/13；构建无需 CUDA SDK。 |
| `tensorrt` | `anr::tensorrt::DFN3` | Windows/Linux NVIDIA GPU；构建需要 TensorRT/CUDA SDK 和 C++ 工具链。 |
| `openvino` | `anr::openvino::{DFN3, Device}` | 构建需 OpenVINO C++ headers/core 链接库和 C++ 工具链；运行需 Runtime、ONNX frontend 及所选设备 plugin。 |
| `opencl` | `anr::opencl::DFN3` | Windows/Linux OpenCL 1.2 硬件 GPU ICD；构建无需 SDK。 |

Cargo features 是可叠加的编译选择，不是自动优先级或 fallback 列表，应用自行构造类型。原生 `cuda` / `tensorrt` 不是 ORT provider；`dx` 是独立 D3D12，不是 ORT DirectML。GLES 不退回桌面 GL，也不提供 Android、WebGL 或 macOS context。

其他 features：

- `download-models`（默认）：下载并校验固定模型；构建时设置 `ANR_MODEL_PATH` 则使用本地文件。
- `copy-models`（默认）：把选中模型复制到构建产物旁，不隐含下载。
- `ort-download-binaries`（默认）：仅在已启用 ORT 时转发 `anr-ort/download-binaries`，自身不激活后端。
- `ort-load-dynamic`：**同时启用 `ort`** 和运行时共享库加载。
- `ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`、`ort-xnnpack`：**同时启用 `ort`** 和对应 provider 支持，不自动选设备或安装驱动/厂商库。

Cargo 对依赖图中的 features 取并集；一处 `default-features = false` 不能撤销其他依赖启用的功能。全 features 构建也需要原生 TensorRT/OpenVINO SDK。

### ORT 入门

在工作区外的应用中，待 `0.3.0` 发布到 crates.io 后使用：

```toml
[dependencies]
anr = { version = "0.3.0", features = ["ort"] }
```

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut reducer = DFN3::from_file(&environment, anr::default_model_path()?)?;
    reducer.set_factor(80.0);
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    reducer.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    reducer.reset();
    Ok(())
}
```

示例处理一帧静音。需引入 `NoiseReduction` 才能使用 trait 方法。`DFN3::from_file(&environment, path)` 明确选择 CPU、一个 intra-op 线程并禁用忙等待，`Environment` 由应用持有。需要 provider/其他设置时，通过 `anr::ort::session_from_file` 创建 Session，再交给 `DFN3::new(session)`。

自备运行库、不在构建时下载：

```toml
anr = { version = "0.3.0", default-features = false, features = ["ort-load-dynamic"] }
```

调用 `initialize_runtime()` 前将 `ORT_DYLIB_PATH` 设为兼容的共享库文件。此配置也关闭自动模型准备，须显式指定路径或将模型放在程序旁。Cargo.lock 固定 `ort`/`ort-sys` `2.0.0-rc.15`、API 27，不启用 `ndarray`；下载器使用 ONNX Runtime 1.31.0，而应用发布配置 `.github/onnxruntime.json` 单独固定 1.28.0。

完整端口、运行库设置、provider 示例及 `prepare_model` / `ModelTarget` 字节 API 见 [ORT 参考](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-ort/README.md)。共用加载器对 CUDA/TensorRT/DirectML 在内存中准备图，每 Session 一次，不改写源文件；直接用上游加载会绕过准备。GPU Session 可以按算子使用 CPU，但 provider 注册失败仍报错。

源码工作区提供最小 raw 示例，输入为 48 kHz 单声道、小端 IEEE 754 `f32`，不是 WAV：

```text
cargo run -p anr --features ort --example denoise -- input.f32le output.f32le
```

可在输入/输出路径前另加模型路径；示例拒绝覆盖已有输出文件。需要解码音频格式时使用 CLI。

### 模型准备与查找

所有非 ORT DFN3 后端（含浏览器及原生 OpenVINO）要求[固定模型](https://github.com/lona-cn/anr/releases/download/model-dfn3-fe5eb64fa2e4/dfn3.onnx)的 SHA-256 为：

```text
fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9
```

ORT 不固定摘要，但模型必须满足端口和语义契约。

构建时，`download-models` 校验并复用 `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/dfn3.onnx`（默认 `~/.cargo`）。设置 `ANR_MODEL_PATH` 可改用本地文件、跳过下载；建议绝对路径，构建脚本会规范化并记录。下载器不校验本地 override 的摘要，Compute 构造器仍强制固定摘要。源文件改变会重新触发准备。

`copy-models` 将 `models/dfn3.onnx` 写入 Cargo profile 目录及其 `examples`、`deps`，支持自定义 target/profile 和交叉编译目录。仅启用复制时必须设置 `ANR_MODEL_PATH`；仅启用下载时只使用持久缓存。

运行时 `default_model_path()` 先查找可执行文件旁的 `models/dfn3.onnx`，再查找构建时记录且仍存在的绝对路径。它不搜索工作目录，不读取运行时 `ANR_MODEL_PATH`，也不联网；关闭全部 defaults 后仍可使用程序旁的模型。分发时须附模型及必要通知，不能依赖构建缓存路径。持久缓存可在 `cargo install` 临时目录删除后继续使用。

冷缓存且 `CARGO_NET_OFFLINE=true`、下载失败或摘要不符均会导致构建失败；离线准备请指定本地模型。下载使用 native TLS，docs.rs 跳过准备。分发前阅读[模型来源与通知](https://github.com/lona-cn/anr/blob/v0.3.0/.github/models/NOTICE.md)。

### 原生流处理契约

所有原生 DFN3 实现 `NoiseReduction: Send`，支持 `Box<dyn NoiseReduction>`。`Send` 允许在线程之间移交实例，不保证并行调用或非阻塞执行。每路连续音频/声道使用独立实例；原生 Compute 每实例持有工作线程，`run` 同步等待，不保证音频回调安全。

- 常量：`SAMPLE_RATE = 48000`、`FRAME_SIZE = 480`（10 ms）、`STATE_SIZE = 45304`。
- `run(sample_rate, input, output)` 要求输入非空、48 kHz、输出等长；这些校验失败不改变输出和状态。
- 接受任意非零长度。尾帧补零至 480 采样并推进完整一帧状态，只返回请求长度；连续调用尽量按 480 的整数倍，反复短调用不等同于整段处理。
- `reset()` 清除音频/特征/网络状态，保留强度；原生 Compute 在下一帧执行前落实 reset。
- 不补偿模型延迟，不提供流结束 flush。

执行错误可能发生在前几帧已写入输出、状态推进之后，不回滚。ORT 再次使用前 reset，Session 不可用时重建。原生 worker 在帧失败后标记下次 reset，但运行库/设备故障仍可能需要重建。CUDA/OpenCL 在 GPU 错误后拒绝继续执行；Vulkan 提交/fence/等待失败也会使执行器失效。reset 不能修复设备丢失。公共错误类型见 [`anr-core`](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-core/README.md)。

初始衰减为 **3 dB**（`factor() == 5`）。`set_factor` 将值限制到 `0..=100`，按 `round(factor / 100 * 60)` 设置整分贝；`factor()` 根据实际分贝反算并取整，可能与输入不同。NaN 不改设置，正负无穷分别饱和到上限和下限。仅 ORT 另提供保留小数的 `set_atten_lim_db` / `get_atten_lim_db`，范围 `0..=60` dB。

强度是衰减上限，不是干湿混合比例；零仍执行模型，不表示原音旁路。

### 原生构造与部署

不需要自动模型准备时关闭 defaults，启用所需 feature，例如：

```toml
anr = { version = "0.3.0", default-features = false, features = ["gles"] }
```

```rust,no_run
use anr::{NoiseReduction, Result, gles::DFN3};

fn main() -> Result<()> {
    let mut reducer = DFN3::from_file("models/dfn3.onnx")?;
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    reducer.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    Ok(())
}
```

`gl`、`vk`、`wgpu`、`dx`、`cuda`、`tensorrt`、`opencl` 和默认 OpenVINO CPU 同样使用单路径构造器。高级 GL 调用方可通过 `anr::gl::DFN3::from_file_with_api(path, api)` 选择 `GlApi`；普通 `gl::DFN3::from_file` 只创建桌面上下文。

GL/GLES/Vulkan/wgpu/D3D12 共用 `anr-dfn3` 固定计划和 WGSL 算子，按需由 Naga 生成 GLSL/GLSL ES/SPIR-V/HLSL。ONNX 只提供参数，不作为通用运行图；不需要外部模型转换、Python 运行库或 shader 编译器。CUDA/OpenCL 执行完整 FP32 调度，包括 FFT、归一化、神经网络、滤波和 overlap-add。这些 GPU 后端不替换为 ORT 或 CPU 推理。

**Vulkan：**为固定算子和 shape 编译 Network shader，相同形状复用 pipeline，buffer 偏移与衰减强度仍可动态读取。GRU 只在私有权重上传前重排一次，保持 FP32 并沿 K 维并行归约；初始化时实测 16/32/64 三种宽度，没有 GPU timestamp 时使用 32。逐帧处理不编译 shader、不重新选型，`run` 仍等待 GPU 完成和结果回读。

结果回读优先使用带 `HOST_CACHED` 的 coherent 内存，不支持时保留原来的兼容内存类型。

Dense `64→64` pointwise 卷积可选用 `VK_KHR_cooperative_matrix`。初始化时检查 subgroup 范围的 `16×16×16` FP16 输入、FP32 累加矩阵，32 或 64 的 subgroup 大小、完整 subgroup 与 vote 操作、16,640 字节 workgroup 内存、五个 storage binding、`shaderFloat16`、Vulkan memory model，以及 Naga 所需的 16-bit storage feature。仅在该层 GPU 实测快于 FP32 时选择矩阵路径；缺少支持或无法计时时保留原 FP32 pointwise 实现，最低要求仍为 Vulkan 1.1。矩阵路径只打包一次静态权重，新增 128 KiB FP16 高位/残差 buffer，并让 workgroup 共享输入 tile。累加、原权重、激活值和循环状态保持 FP32，超出 FP16 有限范围的输入 tile 使用原卷积。不保证逐位相同，初始化会增加 pipeline 编译与可选计时的工作。

**CUDA：**动态加载 Driver API 和 NVRTC，优先 major 13、再 12。Windows 候选为 `nvrtc64_130_0.dll`、`nvrtc64_120_0.dll`；Linux 为 `libnvrtc.so.13`、`.12`、`libnvrtc.so`。通过 `PATH` / `LD_LIBRARY_PATH` 提供 NVRTC 及 builtins 依赖。驱动须支持所选编译器的 PTX，编译器须支持 GPU 架构；Driver API 12.0 并不保证兼容 NVRTC 13。缺失符号或版本不支持正常报错。OpenCL 则需要厂商 GPU ICD，不选择 CPU device。

**TensorRT：**两个标准层 FP32 神经网络 engine 配合 CUDA DSP/FFT graph，禁用 TF32；初始化时构建，没有持久 engine cache。构建需要 C++17、`TENSORRT_ROOT`、`CUDA_ROOT` 或 `CUDA_PATH`，可用 `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` 覆盖链接目录。

**OpenVINO：**通过原生 frontend 导入 ONNX，使用 `MakeStateful`（`ReadValue`/`Assign`），每路流一个 `InferRequest`，状态保留在 plugin 内。构建通过 `OPENVINO_ROOT` 或 `OpenVINO_DIR`（CMake 目录）指定 C++ headers 和 core 链接库；编译不需要 ONNX frontend，运行时 `Core::read_model` 需要它。运行还需所选设备 plugin。接受官方 wheel 的 `include/`、`libs/` 布局，Python 不是运行依赖。`from_file` 明确选择 CPU；其他设备使用：

```rust,no_run
use anr::{Result, openvino::{DFN3, Device}};

fn main() -> Result<()> {
    let reducer = DFN3::from_file_with_device("models/dfn3.onnx", Device::Gpu(0))?;
    println!("execution device: {}", reducer.execution_device());
    Ok(())
}
```

`Device::Cpu`、`Device::Gpu(n)`（`GPU.n`）、`Device::Npu` 只请求对应 plugin。文本解析不区分大小写，接受 `CPU`、`GPU`（`GPU.0`）、`GPU.n`、`NPU`，不支持 AUTO/HETERO 或 fallback 列表。CPU/GPU 请求偏重精度的 FP32；NPU 可能使用 FP16，不保证满足 FP32 reference。运行库/设备缺失或编译失败均报错，不把 GPU/NPU 请求改成 CPU。

**共享 shim 部署：**TensorRT/OpenVINO 构建时编译私有 C++ 共享库，由 Rust 加载。将 `anr_tensorrt_native.dll` / `anr_openvino_native.dll` 放在程序旁，Linux 为 `libanr_tensorrt_native.so` / `libanr_openvino_native.so`，macOS OpenVINO 为 `libanr_openvino_native.dylib`；同时提供厂商 runtime、frontend/设备 plugin 和通知。构建目录路径不能替代部署。

- `ANR_TENSORRT_SHIM` / `ANR_OPENVINO_SHIM` 指定 shim 文件，否则查找程序目录及构建目录。
- Windows `ANR_TENSORRT_RUNTIME` 指定 TensorRT DLL 目录或文件，`CUDA_ROOT` / `CUDA_PATH` 定位 CUDA DLL，亦支持 `PATH`。
- `ANR_OPENVINO_RUNTIME` 指定 OpenVINO 共享库文件。
- 平台加载路径须能解析传递依赖，Linux 按需配置 `LD_LIBRARY_PATH`。

CLI `--provider` 仅选择 ORT provider；原生后端使用 `--backend`，OpenVINO 另用 `--openvino-device`。编入其他后端不改变 CLI 的 ORT 默认选择。文件/管道命令见 [CLI 参考](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-cli/README.md)，历史性能、实测硬件与复现规则见 [benchmark.md](https://github.com/lona-cn/anr/blob/v0.3.0/benchmark.md)。

### 浏览器 WebGPU

Rust/WASM 使用 `default-features = false, features = ["wgpu"]` 和 `wasm32-unknown-unknown`，通过 `anr::wgpu::web::DFN3` 调用。它**不实现**同步 `NoiseReduction: Send`：

```rust,ignore
use anr::wgpu::web::DFN3;

let mut reducer = DFN3::from_bytes(&model_bytes).await?;
reducer.set_factor(80.0);
let mut output = vec![0.0; input.len()];
reducer.run(DFN3::SAMPLE_RATE, &input, &mut output).await?;
reducer.reset();
```

初始化及逐帧回读异步等待 GPU。摘要、分帧、强度、不 flush/不补偿延迟的约定不变。`reset` 只标记下一帧前清空状态，不立即提交 GPU 操作。校验错误不改输出/状态。取消 Rust run 或可恢复的执行错误可能发生在已有帧完成后，并标记下一帧前 reset。设备丢失或持久的未捕获设备错误则必须重建流：程序在落实 reset 前就会返回该错误，`reset()` 不能修复设备。Worker 客户端应等当前请求结束，再依次 `dispose`、`init`。不提供文件加载、WebGL、软件 adapter 或 CPU fallback，需要安全上下文（HTTPS/localhost）及浏览器可用的硬件 WebGPU。

源码工作区也提供直接 JavaScript 绑定，构建流程如下；`wasm-bindgen-cli` 版本须与 `Cargo.lock` 匹配：

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked --root target/web-tools
python tools/build_wgpu_web.py
python -m http.server 8000 --bind 127.0.0.1
```

工具生成 `target/anr-wgpu-web/pkg/anr_wgpu.js`、WASM 资源及 `target/anr-wgpu-web/worker.js`，不安装工具、不准备模型资源。上方英文 JavaScript 示例可直接使用；应用须另提供固定摘要模型。

JS 提供 `DFN3.fromBytes`、`run`、`factor`、`setFactor`、`reset`、`free`。`run` 返回 `Promise<Float32Array>`，拒绝重叠运行；执行中 `factor`、`setFactor`、`reset` 也会报错，须先 await。释放 JS wrapper 不会取消已持有实例的 run。

[module Worker 示例](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-wgpu/examples/web/worker.js) 串行处理带 `id`、`type` 的消息：`init` 传模型 `ArrayBuffer`；`run` 传 `sampleRate` 和 `Float32Array`/`ArrayBuffer`；`setFactor` 传 `factor`；`reset`、`dispose` 无额外负载。回复含 `id`、`ok`，按操作附 `factor`、`output` 或 `error`；成功音频回复转移 `output.buffer`。用 `new Worker('/target/anr-wgpu-web/worker.js', { type: 'module' })` 创建。

麦克风采集、重采样、AudioWorklet 缓冲、声道路由和延迟补偿由应用负责；facade 默认下载/复制功能不会提供浏览器模型资源。

### 回归参考

原生数值测试使用仓库中的 `tests/fixtures/reference_input.f32le`、`reference_output.f32le`。真实模型测试默认忽略；将 `ANR_DFN3_MODEL` 指向固定摘要模型，只选择本机平台、硬件和运行库支持的 crate。例如，在支持桌面 OpenGL 4.3 的系统上，从工作区根目录用 PowerShell 执行：

```powershell
$env:ANR_DFN3_MODEL = (Resolve-Path models/dfn3.onnx).Path
cargo test -p anr-gl --test dfn3 -- --ignored --nocapture
```

POSIX shell 可先执行 `export ANR_DFN3_MODEL="$PWD/models/dfn3.onnx"`，再运行相同 Cargo 命令。按需替换 crate 或增加受支持的 `-p` 选择；TensorRT/OpenVINO 还需要构建 SDK 和运行库，不应无条件运行所有后端。

fixture 由 ONNX 的 NumPy `ReferenceEvaluator` 生成，不是 ORT。确需重新生成时，将原模型放在 `models/dfn3.onnx`，在隔离的开发环境中执行：

```sh
python -m pip install onnx==1.19.1 numpy==2.2.6
python tools/generate_gl_reference.py
```

[生成器](https://github.com/lona-cn/anr/blob/v0.3.0/tools/generate_gl_reference.py)先校验模型 SHA，再计算参考结果，**覆盖 GL、GLES、Vulkan、wgpu、D3D12、CUDA、TensorRT、OpenVINO、OpenCL 九个原生后端的两份 fixture 文件**。确定性输入覆盖静音、衰减切换及末尾 13 采样短帧。这些 Python 包仅用于生成开发数据，不是产品运行依赖；常规回归使用现有 fixture。

浏览器检查先按上文构建并启动本地服务器，提供 `/models/dfn3.onnx`，再在该站点的浏览器控制台执行：

```javascript
const { runReference } = await import('/crates/anr-wgpu/tests/web/reference.js');
await runReference(
  '/target/anr-wgpu-web/pkg/anr_wgpu.js',
  '/models/dfn3.onnx',
  '/target/anr-wgpu-web/worker.js',
);
```

[浏览器参考测试](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-wgpu/tests/web/reference.js)用相同 fixture 检查直接绑定和 module Worker，覆盖状态/reset、尾帧补零、无效调用/并发拒绝及实例生命周期。需要可用的硬件 WebGPU；数值/API 回归与性能 benchmark 分开进行。

### 许可

MIT OR Apache-2.0，任选其一：[LICENSE-MIT](LICENSE-MIT)、[LICENSE-APACHE](LICENSE-APACHE)。模型来源和许可须单独确认，源码许可不能证明所有模型导出的使用/分发权。ONNX Runtime 和厂商 provider 库也有各自的分发要求。
