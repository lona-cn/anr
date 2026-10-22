# anr

[English](#english) | [简体中文](#简体中文)

## English

`anr` is a Rust audio denoising library and command-line tool. It runs DeepFilterNet3 (DFN3) with ONNX Runtime, or independent model-specific OpenGL, OpenGL ES, Vulkan, wgpu, Direct3D 12 Compute, CUDA, TensorRT, OpenVINO, and OpenCL backends without ORT.

The library takes **48 kHz, mono `f32` samples**. For audio files, resampling, and multiple channels, use [`anr-cli`](crates/anr-cli/README.md). The CLI handles files, batches, and raw PCM streams; it does not provide microphone access, virtual audio devices, or drivers.

Source builds require the **latest stable Rust toolchain**. Default features download, verify, cache, and copy the supported DFN3 model at build time, including with `cargo install`; the model bytes are not embedded in Rust packages. GitHub application ZIPs bundle the same model. Check inputs and outputs before substituting another export.

### Choose a backend for the Rust library

The library supports `ort` (ONNX Runtime), `gl` (OpenGL), `gles` (OpenGL ES), `vk` (Vulkan), `wgpu`, `dx` (Direct3D 12), `cuda`, `tensorrt`, `openvino`, and `opencl`. No library backend is enabled by default; the default `ort-download-binaries` option does not turn on ORT.

For example, build with `wgpu` and without ORT:

```toml
[dependencies]
anr = { version = "0.1.1", default-features = false, features = ["wgpu"] }
```

In `Cargo.toml`, list the backends your program needs under `features`. If you list more than one, Cargo includes them all; it will not switch between them while the program runs. The non-ORT backends support only the project's specified DFN3 model, not arbitrary ONNX files. See the [library README](crates/anr/README.md) for backend requirements.

### Denoise a file

The [GitHub application releases](https://github.com/lona-cn/anr/releases) are built as native ZIPs with the CLI, CPU runtime, `models/dfn3.onnx`, and license files. Extract the whole ZIP and run from its directory:

```sh
./anr denoise voice.wav --output clean.wav
```

On Windows, use `.\anr.exe` instead of `./anr`. The CLI finds the bundled model beside the executable; `--model PATH` overrides it. Application ZIPs appear only after the release workflow passes; the separate [model asset release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) contains the model, not a CLI build. See the [CLI distribution notes](crates/anr-cli/README.md) for system runtime requirements.

Install the CLI from the workspace root. The executable is called `anr`. This default build uses ORT's prebuilt libraries where available; on Intel macOS, follow the [CLI instructions](crates/anr-cli/README.md) for supplying your own runtime instead.

```text
cargo install --path crates/anr-cli --locked
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav --json
anr backends --json
```

No manual model download or `--model` argument is needed with the default build. An explicit `--model PATH` still selects your own file. Once installed, `anr` does not need to run from the source directory. It uses CPU ORT by default, preserves the input sample rate, channel count, and frame count unless you request otherwise, and writes WAV or raw f32le.

See the [CLI README](crates/anr-cli/README.md) for raw stdin/stdout, how output files are saved, execution providers, model latency, and binary-safe shell examples.

The default build downloads the model and may download ONNX Runtime, so the first build needs network access and a supported platform toolchain. Intel macOS is not covered by the current default ORT prebuilt distribution. CI is configured to use Rust stable on native Windows x64, Linux x64, and macOS arm64/x64; the configuration itself is not evidence that those jobs have passed. See the CLI README for recorded local results and platforms not yet tested.

For a source build without ONNX Runtime, run the GLES backend directly from the workspace:

```text
cargo run -p anr-cli --release --no-default-features --features gles -- denoise voice.wav --model models/dfn3.onnx --backend gles --output clean.gles.wav
```

This requires a supported native ES 3.1+ driver and the exact model described below. `--backend gles` is required even in a GLES-only build: the CLI's default selection remains `ort`.

### Use the Rust library

The `anr` library does not enable a backend by default. Choose a feature from the backend table below; the following example explicitly enables ORT and its `DFN3` reexport:

```toml
[dependencies]
anr = { version = "0.1.1", features = ["ort"] }
```

The dependency examples use version `0.1.1`. To use a local checkout, replace the version dependency with a `path` dependency pointing to the crate directory.

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, anr::default_model_path()?)?;
    denoiser.set_factor(80.0);

    // Replace this silence with your own 48 kHz mono samples.
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;

    // Clear the previous recording's state; keep the strength setting.
    denoiser.reset();
    Ok(())
}
```

Before calling `run`:

- Supply at least one input sample.
- Use a sample rate of 48,000 Hz.
- Allocate an output slice with the same length as the input.

If any of these checks fail, the output and processing state stay unchanged. An error during model execution may leave both partly updated; it does not undo work already done.

#### Keep continuous audio in chunks of 480 samples

The model processes **480 samples at a time: 10 ms of audio**. `run` accepts any nonzero length, but pads a short final frame with zeros, processes the whole frame, and returns only the requested samples. That padding also advances the model state. Splitting a recording into many short calls can therefore produce different results from processing it in one call. For live audio, collect samples into multiples of 480 before passing them in.

Keep one `DFN3` instance per continuous stream; don't share it between unrelated recordings. The model uses information from earlier audio. Call `reset()` before switching recordings or restarting playback. This clears the audio state without changing the strength.

The library does not compensate for model latency and has no end-of-stream flush operation. Sound quality, latency, and suitability depend on the actual model; matching the required inputs and outputs does not establish any of them.

#### Adjust the strength

Every native DFN3 backend supports `NoiseReduction::set_factor(0..=100)`, mapped to dB with `round(factor / 100 * 60)`. Only the ORT concrete type additionally exposes `set_atten_lim_db` / `get_atten_lim_db` for direct `0..=60` dB access; these methods are not on the shared trait or compute frontends.

Out-of-range values are clamped. `NaN` leaves the setting unchanged; positive and negative infinity use the upper and lower bounds respectively.

`set_factor` rounds to whole dB; ORT's direct dB setter preserves fractional values. `factor()` converts the actual dB setting back and rounds it, so it may differ from the value passed in. This is an attenuation limit, not a dry/wet mix. Zero still runs the model and is not a bypass for the supplied DFN3 model.

#### Try the raw audio example

The following example uses ORT; run it from the workspace root. For compute backends and audio containers, use the CLI with an explicit `--backend` instead.

```text
cargo run -p anr --features ort --example denoise -- <input.f32le> <output.f32le>
```

The input is **48 kHz mono raw audio, not WAV**: four bytes per sample, stored as little-endian IEEE 754 `f32`. The output uses the same format. The default model is found automatically; prepend `<model.onnx>` to the arguments to select another file. Check that a replacement meets the requirements below and that you have permission to use it.

### ORT model inputs and outputs

The ORT adapter requires exactly these three inputs and three outputs, all `f32`. Names are matched when loading; input order does not matter. Brackets show tensor shapes: `[]` is a scalar, and `[1]` is a one-element vector. Compute backends require the fixed SHA-256 model in the next section, not merely matching port shapes.

| Direction | Name | Runtime shape |
| --- | --- | --- |
| Input | `input_frame` | `[480]` |
| Input | `states` | `[45304]` |
| Input | `atten_lim_db` | `[]` or `[1]` |
| Output | `enhanced_audio_frame` | `[480]` |
| Output | `new_states` | `[45304]` |
| Output | `lsnr` | `[]` or `[1]` |

A one-dimensional tensor may declare a symbolic length. Loading checks the rank and any fixed lengths; symbolic lengths do not change the required 480 audio samples and 45,304 state values at runtime. The constants are `DFN3::SAMPLE_RATE = 48000`, `FRAME_SIZE = 480`, and `STATE_SIZE = 45304`.

Audio and state outputs use preallocated buffers, and ONNX Runtime checks their actual lengths. For `lsnr`, only the declared type and shape are checked; its result is neither read nor returned. Extra inputs or outputs, different ranks, or a different meaning or ordering of state values require changes to the implementation.

### Select a backend and runtime

Adding `anr` alone does not enable model execution. Choose the Cargo features you need:

| Feature | What it does |
| --- | --- |
| `ort` | Provides `DFN3`. With default features enabled, the CPU runtime is downloaded and copied at build time. |
| `gl` | Provides `anr::gl::DFN3`, using native OpenGL 4.3 compute shaders without ORT. |
| `gles` | Provides `anr::gles::DFN3`, using a real native OpenGL ES 3.1+ compute context without ORT. |
| `vk` | Provides `anr::vk::DFN3`, using native Vulkan 1.1 compute without ORT or wgpu. |
| `wgpu` | Native `anr::wgpu::DFN3`, or asynchronous browser `anr::wgpu::web::DFN3` on `wasm32-unknown-unknown`, without ORT. |
| `dx` | Provides `anr::dx::DFN3`, using native Windows Direct3D 12 compute, not DirectML or wgpu. |
| `cuda` | Provides `anr::cuda::DFN3`, using CUDA Driver/NVRTC and replay graphs for complete FP32 inference, without ORT. |
| `tensorrt` | Provides `anr::tensorrt::DFN3`, using native TensorRT engines plus CUDA DSP/FFT; requires TensorRT/CUDA SDKs at build time. |
| `openvino` | Provides `anr::openvino::DFN3` and `Device`, importing the original ONNX with native OpenVINO Runtime; requires its C++ SDK. CPU is the explicit default. |
| `opencl` | Provides `anr::opencl::DFN3`, using complete FP32 OpenCL 1.2 GPU inference without ORT or CPU fallback. |
| `ort-download-binaries` | Default download option; takes effect only when `ort` is enabled and does not enable a backend by itself. |
| `download-models` | Enabled by default. Downloads and verifies the pinned DFN3 model into a persistent Cargo cache; `ANR_MODEL_PATH` overrides the build-time source. |
| `copy-models` | Enabled by default. Copies that model to `models/dfn3.onnx` beside Cargo binaries, examples, and tests; does not enable downloading by itself. |
| `ort-load-dynamic` | Loads your runtime. Use `default-features = false` with this feature to omit the downloader and its TLS dependencies; set `ORT_DYLIB_PATH` to the shared library file. |
| `ort-cuda` / `ort-tensorrt` | Compiles ORT execution-provider support, distinct from the independent `cuda` / `tensorrt` backends. |
| `ort-directml` / `ort-coreml` / `ort-xnnpack` | Compiles support for DirectML / CoreML / XNNPACK. |

`anr::default_model_path()` first checks beside the executable, then the absolute source path recorded at build time. It never downloads at runtime. Disable defaults to omit model downloading, or enable only `copy-models` and set `ANR_MODEL_PATH` to an absolute local file path. Cache location, offline builds, and distribution requirements are in the [model preparation guide](crates/anr/README.md#model-download-and-copying).

The project pins `ort = 2.0.0-rc.15` with API 27, without `ndarray`. Its default prebuilt runtime is 1.31.0; compatible self-provided API-27 runtimes, including 1.28.0, can also be used. Set `ORT_DYLIB_PATH` before initialization; it points to a library file, not a directory.

Call `anr::ort::initialize_runtime()` before building an environment so dynamic-library errors are returned normally. It loads the library but does not create or configure an environment. Build one `Environment` in your application and share it between denoisers. `DFN3::from_file(&environment, path)` explicitly uses the built-in CPU provider, one intra-op thread, and no thread-pool spinning. GPU features do not select a GPU or install drivers. For a custom provider, configure a `SessionBuilder`, call `anr::ort::session_from_file(builder, path, &providers)`, then pass the session to `DFN3::new(session)`.

The project's default ORT policy allows per-operator CPU execution alongside a GPU provider. CLI and benchmark keep ORT's normal placement rules for unsupported or CPU-preferred operators; this is not a temporary benchmark option. `DFN3::new(session)` preserves the supplied session's policy. A requested provider must still register successfully: missing GPU libraries or provider initialization failures remain errors. Native GL/GLES/Vulkan/wgpu/D3D12 implementations are unchanged and do not gain a CPU executor.

### Browser WebGPU (WASM)

On `wasm32-unknown-unknown`, the `wgpu` feature exposes the asynchronous Rust type `anr::wgpu::web::DFN3` (`anr_wgpu::web::DFN3` when using the backend directly). It loads model bytes with `from_bytes(...).await`, rather than reading a filesystem path. Its `run(sample_rate, input, output).await` is separate from the native synchronous `NoiseReduction` trait. The model hash, WGSL kernels, frame schedule, and strength/reset behavior are shared with the native implementation.

From a source checkout, install the target and a wasm-bindgen CLI matching `Cargo.lock`, then build the JavaScript/WASM package:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked --root target/web-tools
python tools/build_wgpu_web.py
python -m http.server 8000 --bind 127.0.0.1
```

The helper creates `target/anr-wgpu-web/pkg/` and stages a module Worker beside it. It does not download the model: serve the pinned `models/dfn3.onnx` separately. HTTPS or localhost and an available hardware WebGPU adapter are required; there is no WebGL or CPU fallback. Build-time model download/copy features do not embed or fetch the model in the browser.

For the source-root HTTP server, use this from a JavaScript module or browser console:

```js
const { default: init, DFN3 } = await import('/target/anr-wgpu-web/pkg/anr_wgpu.js');
await init();
const response = await fetch('/models/dfn3.onnx');
if (!response.ok) throw new Error(`Model request failed: HTTP ${response.status}`);
const stream = await DFN3.fromBytes(new Uint8Array(await response.arrayBuffer()));
try {
  stream.setFactor(80);
  const output = await stream.run(48_000, new Float32Array(480));
  console.log(output); // Replace the silence above with your own 48 kHz mono input.
} finally {
  stream.free();
}
```

Reuse one stream across calls. `run` returns a JS-owned `Float32Array`; overlapping runs reject, and `factor()`, `setFactor(...)`, and `reset()` report an error while a run is pending. `free()` releases the wrapper without invalidating an already returned Promise. Device loss requires a new stream; `reset()` only clears audio/model state.

For live audio, run inference in the [module Worker example](crates/anr-wgpu/examples/web/worker.js), not inside `AudioWorkletProcessor.process`. Requests carry `id` and `type`: `init` takes a model `ArrayBuffer`; `run` takes `sampleRate` and a `Float32Array` or `ArrayBuffer`; `setFactor` takes `factor`; `reset` and `dispose` need no payload. The Worker serializes requests and returns `{id, ok, output?, factor?, error?}`. Transfer model/input buffers and returned output buffers when passing messages; transferred buffers become detached in the sender.

Collect continuous audio into multiples of **480 samples** before submitting it. Only the final short frame should be zero-padded; it advances a full model frame. This backend does not provide microphone capture, resampling, AudioWorklet buffering, delay compensation, or an end flush.

The browser regression executes the generated WASM both directly and through the Worker, including reference output, reset, invalid input, partial-frame state, concurrent-call rejection, and freeing a wrapper during a run:

```js
const { runReference } = await import('/crates/anr-wgpu/tests/web/reference.js');
await runReference('/target/anr-wgpu-web/pkg/anr_wgpu.js',
  '/models/dfn3.onnx', '/target/anr-wgpu-web/worker.js');
```

Hardware WebGPU has been exercised on Windows / Edge 135 / NVIDIA Turing. Other browsers and mobile devices remain unverified; this is not a cross-device real-time guarantee. CI checks and links the WASM backend but does not run GPU inference.

If Windows `sccache` reports command-line error 206 while compiling `web-sys`, disable the wrapper only for this build process:

```sh
python -c "import os, runpy; os.environ['RUSTC_WRAPPER']=''; runpy.run_path('tools/build_wgpu_web.py', run_name='__main__')"
```

### Compute backends without ONNX Runtime

Build the CLI with `cargo build -p anr-cli --release --no-default-features --features gl,gles,vk,wgpu,dx`, then select `--backend gl`, `gles`, `vk`, `wgpu`, or `dx`. You can enable any subset. Do not pass an ORT `--provider`. The default CLI build, backend, and application release ZIPs remain ORT.

For example, a GLES-only library dependency:

```toml
anr = { version = "0.1.1", default-features = false, features = ["gles"] }
```

```rust,no_run
use anr::{gles::DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file("models/dfn3.onnx")?;
    denoiser.set_factor(80.0);
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

Construct `anr::gles::DFN3::from_file("models/dfn3.onnx")`, or the matching `gl`, `vk`, `wgpu`, or `dx` module, and use the same `NoiseReduction` trait. These shader backends need no ORT environment, runtime library, Python, model conversion, or external shader compiler. Their external ONNX file supplies parameters only, after SHA-256 validation against `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`; other exports are rejected. The new native backends use the same model hash, but OpenVINO imports and executes the original graph rather than treating ONNX only as a parameter container.

`anr-dfn3` shares the fixed execution plan, weight layout, worker, and WGSL operators. Naga translates to desktop GLSL 4.3, GLSL ES 310, SPIR-V, or HLSL; wgpu consumes WGSL directly. GLES reuses the executor in `anr-gl`, but requires an embedded context rather than accepting desktop ES compatibility. It maps only the completed output frame after a GPU fence because GLES has no `glGetBufferSubData`. The FFT needs 15,360 bytes of workgroup memory, within the ES 3.1 minimum of 16 KiB.

For native GL/GLES/Vulkan/wgpu/D3D12, all inference, including FFT, feature normalization, convolutions, grouped linear layers, five GRUs, spectral filtering, and overlap-add, runs in FP32 compute shaders. Weights and two state buffers remain on GPU. Each native reducer owns a worker thread; `run` waits for GPU output and is not a nonblocking audio-callback API. `lsnr` is not exposed, model latency is not compensated, and strength zero retains the attenuation branch rather than bypassing the model.

| Backend | Driver/platform requirement |
| --- | --- |
| `gl` | Windows desktop OpenGL 4.3; Linux hardware EGL device with desktop GL 4.3 and pbuffer, without X11 `DISPLAY`. Not macOS. |
| `gles` | Windows WGL driver with native ES context support; Linux hardware EGL device/pbuffer. Both require OpenGL ES 3.1+. No Android, WebGL, or macOS context is provided. |
| `vk` | Windows or Linux hardware Vulkan 1.1 device with compute queue. No window or surface. |
| `wgpu` | Native hardware Vulkan/DX12/Metal on Windows/Linux/macOS; browser WebGPU on `wasm32-unknown-unknown` in a secure context. No software fallback. |
| `dx` | Windows hardware D3D12 device and system D3DCompiler 47. No WARP, DirectML, or separately installed DXC. |

Initialization failures return errors rather than falling back to CPU. The five GL/GLES/Vulkan/wgpu/D3D12 backends passed streaming reference and raw/WAV/stereo/batch CLI tests on native Windows / RTX 2080 Ti. GLES-only compilation, non-GPU tests, and Clippy also passed in WSL Ubuntu 24.04. These results do not establish cross-device performance or sound quality; Linux GPU paths, Android, and macOS remain unverified.

Set `ANR_DFN3_MODEL` to the absolute model path and run `cargo test -p anr-gl -p anr-gles -p anr-vk -p anr-wgpu -p anr-dx -- --include-ignored --nocapture` on hardware supporting the selected packages. The reference covers strength changes, a partial final frame, reset, thread moves, and invalid-input state preservation. `tools/generate_gl_reference.py` regenerates the fixtures for all nine native backends using development-only `onnx==1.19.1` and `numpy==2.2.6`, not ONNX Runtime.

#### Native CUDA, TensorRT, OpenVINO, and OpenCL

Enable the identically named facade/CLI features explicitly. `cuda` and `opencl` execute the complete fixed FP32 schedule, including the 960-point FFT, recurrent network, filtering, and overlap-add, with GPU-resident weights/state. CUDA uses the Driver API, NVRTC, and replay graphs; OpenCL uses a hardware GPU ICD supporting OpenCL 1.2 and never substitutes a CPU device. TensorRT builds two real standard-layer neural-network engines and captures them with CUDA DSP/FFT into frame graphs: strongly typed FP32, TF32 disabled, no ORT, ONNX parser, or inference plugin.

OpenVINO instead directly imports the original ONNX through its native frontend, applies `MakeStateful` (`ReadValue`/`Assign`), and owns one `InferRequest` per stream. `anr::openvino::DFN3::from_file(path)` explicitly selects CPU. Use `from_file_with_device(path, anr::openvino::Device::Gpu(0))` or `Device::Npu` to request another device. CPU/GPU use accuracy-oriented FP32 hints; NPU hardware may use FP16 and has no verified FP32-reference guarantee. GPU/NPU initialization or compilation failures remain errors, never a switch to CPU; there is no AUTO/HETERO selection.

From the workspace root, with the required driver/runtime and model available:

```text
cargo run -p anr-cli --release --no-default-features --features cuda -- denoise voice.wav --model models/dfn3.onnx --backend cuda --output clean.cuda.wav
cargo run -p anr-cli --release --no-default-features --features opencl -- denoise voice.wav --model models/dfn3.onnx --backend opencl --output clean.opencl.wav
cargo run -p anr-cli --release --no-default-features --features tensorrt -- denoise voice.wav --model models/dfn3.onnx --backend tensorrt --output clean.tensorrt.wav
cargo run -p anr-cli --release --no-default-features --features openvino -- denoise voice.wav --model models/dfn3.onnx --backend openvino --openvino-device CPU --output clean.openvino.wav
```

`--openvino-device CPU|GPU|GPU.n|NPU` applies only to OpenVINO (`GPU` means `GPU.0`); `--provider` applies only to ORT. CLI default selection remains ORT, even when only another backend is compiled.

**SDK and deployment:** TensorRT needs a C++ toolchain plus `TENSORRT_ROOT` and `CUDA_ROOT` (or `CUDA_PATH`); `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` override link-library directories. OpenVINO needs `OPENVINO_ROOT` or `OpenVINO_DIR` (its CMake directory). The official OpenVINO wheel's `include/` and `libs/` can supply the SDK; Python is not a product runtime dependency. For example, in PowerShell, replace these conventional SDK locations with your extracted installations before running the commands above:

```powershell
$env:TENSORRT_ROOT = 'C:\SDKs\TensorRT-10.16.1'
$env:CUDA_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:OPENVINO_ROOT = 'C:\SDKs\openvino'
```

No system-wide installation is required. CUDA needs the NVIDIA driver and discoverable NVRTC/builtins libraries (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux); OpenCL needs a vendor GPU ICD. TensorRT/OpenVINO `build.rs` compile private shared C++ shims which Rust loads with `libloading`. Deploy `anr_tensorrt_native.dll` / `anr_openvino_native.dll` (Linux: `libanr_tensorrt_native.so` / `libanr_openvino_native.so`) beside the executable, together with the required vendor runtime, frontend, and device-plugin libraries and their notices. Build-tree paths are not a portable deployment. `ANR_TENSORRT_SHIM` and `ANR_OPENVINO_SHIM` select shim **files**; `ANR_TENSORRT_RUNTIME` selects a TensorRT DLL directory or file on Windows, while `ANR_OPENVINO_RUNTIME` selects the OpenVINO runtime library **file**. Standard loader paths must also resolve vendor dependencies. Default release ZIPs remain CPU ORT and do not bundle these SDKs or runtimes automatically.

The native `cuda` backend requires Driver API 12.0 or newer and NVRTC major 12 or 13. It probes the required driver symbols and returns ordinary errors for unsupported runtimes. It prefers NVRTC 13, then 12: Windows `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll`; Linux `libnvrtc.so.13` / `.12` / unversioned `libnvrtc.so`. The driver must support PTX produced by the selected NVRTC, and NVRTC must support the GPU architecture; NVRTC 13 is not promised to work with an arbitrary 12.x driver. Its builtins dependencies must also be discoverable.

**Recorded scope:** real streaming references and 10 raw/WAV/stereo/batch CLI checks passed for these four backends on Windows: CUDA/TensorRT/OpenCL on RTX 2080 Ti, OpenVINO CPU on AMD Ryzen 7 5800X, with CUDA 13.3, TensorRT 10.16.1, and OpenVINO 2026.4.0. OpenVINO GPU compilation failed selecting the original graph's `NonZero_count` kernel; NPU is untested. These are not cross-platform GPU or GPU/NPU success claims. The seven native benchmark choices (`gl,vk,dx,cuda,tensorrt,openvino,opencl`) completed 100 warmup / 1,000 measured frames each. TensorRT initialization took 17.342 s in that run; there is no persistent engine cache or fast-cold-start promise.

### Benchmark native compute and ORT GPU inference

`anr-bench` is a workspace-only tool. Its default build tests native compute without an ONNX Runtime dependency. Run from the repository root:

```sh
cargo run --release -p anr-bench -- --model models/dfn3.onnx --warmup 100 --frames 1000
cargo run --release -p anr-bench -- --backend vk,dx --json
```

By default it tests `gl,vk,dx` sequentially, warming up each backend for 100 frames and measuring 1,000 frames at strength 80. Use `--backend gl,vk` on Linux; D3D12 is Windows-only. `--backend` also accepts repeated flags and preserves their order. `--model` defaults to `models/dfn3.onnx`; `--warmup 0` is allowed, `--frames` must be positive, and `--strength` must be finite and within 0–100.

CUDA and OpenCL are compiled into the default native benchmark build, but are not added to the default selection. TensorRT and OpenVINO require opt-in SDK features:

```sh
cargo run --release -p anr-bench -- --backend cuda,opencl --warmup 100 --frames 1000 --json
cargo run --release -p anr-bench --features tensorrt,openvino -- --backend gl,vk,dx,cuda,tensorrt,openvino,opencl --openvino-device CPU --warmup 100 --frames 1000 --json
```

Supply the SDK/runtime paths described above before the second command. `cuda` / `tensorrt` are independent native backends, not aliases for `ort-cuda` / `ort-tensorrt`.

Optional `ort-cuda`, `ort-tensorrt`, and `ort-directml` features add identically named `--backend` choices. They are not added to the default backend list. DirectML requires Windows. Select a runtime source as well: `ort-download-binaries` uses the pinned prebuilt distribution, while `ort-load-dynamic` loads an API-27-compatible library file through `ORT_DYLIB_PATH`.

```sh
cargo run --release --target-dir target/benchmark-ort-gpu -p anr-bench --features ort-download-binaries,ort-cuda,ort-tensorrt,ort-directml -- --backend ort-cuda,ort-tensorrt,ort-directml --json
```

For a supplied shared runtime, set `ORT_DYLIB_PATH` before launch, replace `ort-download-binaries` with `ort-load-dynamic`, and use a separate target directory such as `target/benchmark-ort-dynamic`. GPU features do not install CUDA, cuDNN, TensorRT, or drivers. The pinned GPU distribution uses CUDA 13. Keep provider DLLs matched to the core runtime: reusing a target directory containing 1.28 provider DLLs with the 1.31 core caused a CUDA access violation during verification. A fresh target directory avoids changing other applications' cached files. A supplied runtime must include the requested providers; they are not necessarily all present in one runtime build.

ORT uses one environment across all selected backends, including after a session fails. Each session explicitly registers only the requested provider with registration errors enabled, disables environment providers and thread-pool spinning, and uses sequential execution with one intra-op thread. DirectML also disables memory patterns. Normal ORT placement may run individual operators on CPU; TensorRT does not silently add CUDA as a second provider.

**Model preparation:** CUDA/TensorRT use the shared `anr-ort` loader to rewrite an in-memory copy before session creation, without modifying the original ONNX or weights. CUDA expands the unsupported sigmoid `FusedConv`; TensorRT also replaces static `SplitToSequence`/`SequenceAt` state access with `Split` and safely lowers eligible tensor `If` branches to `Where`. CPU/DirectML and native compute retain their existing loading paths. Preparation is included in initialization, not repeated or timed per frame. Unsupported operators may still execute on CPU; TensorRT capability probing may print importer diagnostics for vendor fusions it cannot import. Actual initialization/inference errors still fail the backend.

The supplied model has passed CUDA/TensorRT/DirectML comparisons against the original CPU graph on Windows / RTX 2080 Ti with matching ORT 1.31 libraries, including live attenuation changes, reset, and a partial frame. The opt-in numerical regression is `ANR_DFN3_MODEL=/absolute/path/dfn3.onnx cargo test --release --target-dir target/benchmark-ort-gpu -p anr-ort --features cuda,tensorrt,directml --test compat -- --ignored`; keep matching provider DLLs accessible beside the test executable in `release/deps` as well as the benchmark executable.

Every backend receives the same one-second, fixed-seed mixture of 220 Hz / 880 Hz sine waves and noise, looped as a continuous 48 kHz mono stream in 480-sample frames. Input generation, model loading, shader compilation, warmup, result sorting, and reporting are outside the measured loop. Initialization is reported separately. Warmup state and input position are retained; there is no per-frame reset or benchmark-side allocation in the timed loop.

The table reports initialization, min/mean/max synchronous call latency, nearest-rank P50/P95/P99, frames per second, RTF, and frames taking **more than** the 10 ms audio budget. Latency includes backend execution, any CPU-assigned operators, and host/device transfers, not just GPU kernel time. Native GPU compute keeps recurrent state on GPU, and OpenVINO keeps state inside the selected plugin's request; the existing ORT adapter uses CPU-backed audio/state tensors, so its required transfers are measured too. FPS and `rtf` use the complete measured loop's wall time, including timing/bookkeeping overhead; `rtf = wall_seconds / audio_seconds`, so values below 1 are faster than real time.

`--json` writes configuration (including `openvino_device`), input seed, platform/build information, effective backend strength, timings, and `results[].status` (`ok` or `error`) to stdout; progress and debug-build warnings go to stderr. Successful OpenVINO results include `execution_device`. Successful results include `cpu_fallback_allowed`: true for ORT and false for native backends, **including explicitly selected OpenVINO CPU**. It describes fallback policy, not an operator count or a claim that all work ran on GPU. Strength follows the backend's existing attenuation rounding. A backend failure does not prevent later backends from running, but any failure produces exit code 1; invalid arguments produce exit code 2. Provider registration errors do not silently replace the requested provider with CPU. Apart from OpenVINO's explicit device selector, device selection remains each crate's own default; multiple-GPU machines must not assume every API chose the same adapter. Synthetic-input results are not sound-quality or representative-recording measurements.

### Develop ANR-Lite-48 with Burn

`anr-lite` is a separate Burn 0.22.1 model crate for **48 kHz, mono speech**. It implements the ANR-Lite-48 architecture and an optional VoiceBank-DEMAND training binary, but includes **no pretrained weights or automatic dataset downloads**. Random initialization is not a usable denoising model. It is not selected by the `anr` CLI and does not replace the hash-locked DFN3 export.

The network combines 64-band ERB envelope features with 96-bin complex low-frequency features. Both encoders use 16 channels; the shared temporal core has two 128-dimensional GRUs with four groups and dense inter-group mixing. The two heads predict full-band gains and five-tap complex residual filters. The residual head starts at zero. Batch normalization uses Burn's `train()` / `valid()` modes.

```sh
cargo run -p anr-lite --locked --example smoke
cargo run -p anr-lite --locked --features wgpu --example smoke -- --wgpu
cargo test -p anr-lite --locked --all-targets
```

The default `flex` feature enables Burn's CPU backend. The optional `wgpu` feature enables an explicitly selected discrete GPU; `--wgpu` does not silently switch to CPU. The smoke example generates its own signal, reports the untrained status, compares whole-waveform processing with streaming plus flush, and checks finite, nonzero input gradients without an optimizer update. Its first-call timing is not a steady-state benchmark or a sound-quality measurement.

```rust,no_run
use anr_lite::{AnrLite48, AudioProcessor};
use anr_lite::burn::{module::Module, tensor::{Device, Tensor}};

let device = Device::flex();
let model = AnrLite48::new(&device).valid();
let processor = AudioProcessor::new(&device);
// Replace with your own [batch, samples] f32 audio at 48 kHz.
let input = Tensor::<2>::zeros([1, 960], &device);
let output = processor.forward(&model, input);
```

`AudioProcessor::forward` starts a new batch of streams, pads the final hop, flushes, compensates the known delay, and returns the original sample count. For continuous audio, use `initial_state(batch)` and `forward_frame` with `[batch, 480]` tensors. Streaming returns delayed samples: `AUDIO_DELAY_SAMPLES = 1440` (30 ms), excluding collection of the current input block. At EOF, right-pad a partial hop and call `flush` once to emit the three delayed hops. Flush advances the stream; replacing its state starts a new recording.

For custom training loops, use an autodiff device and `model.train()`. Audio transforms, feature normalization, the network and spectral filtering retain their gradient graph across frames. Call `AudioState::detach()` or `ModelState::detach()` only at an intentional truncated-backpropagation boundary. Burn module records preserve model parameters and normalization state, not the live audio stream.

The 960-point differentiable FFT combines native 64-point FFTs with a fixed radix-15 stage because Burn's native FFT only accepts powers of two. Sparse ERB mapping avoids a dense frequency-by-band matrix. Model and DSP activations stay on the chosen device; this implementation does not provide allocation-free audio callbacks, custom fused native shaders, or verified speed/quality improvements over DFN3.

#### Train on VoiceBank-DEMAND 28-speaker

Enable `train` to build `anr-lite-train`. Run from the workspace root; `--dataset` defaults to `datasets/VoiceBank-DEMAND 28-speaker`. The extracted dataset must contain `clean_trainset_28spk_wav`, `noisy_trainset_28spk_wav`, `clean_testset_wav`, and `noisy_testset_wav`. The loader requires same-name, same-length mono 48 kHz PCM16 WAV pairs; it does not resample, normalize the pair independently, or modify dataset files. Text transcripts are not needed.

```sh
# CPU
cargo run -p anr-lite --release --locked --features train --bin anr-lite-train -- --device cpu
# Discrete GPU, explicitly selected
cargo run -p anr-lite --release --locked --features train,wgpu --bin anr-lite-train -- --device wgpu
# Bounded GPU smoke: real updates, limited validation and checkpoint reload/test
cargo run -p anr-lite --locked --features train,wgpu --bin anr-lite-train -- --device wgpu --epochs 1 --batch-size 2 --segment-seconds 0.04 --max-steps 2 --max-validation-items 1 --evaluate-test --max-test-items 1
```

Defaults are 50 epochs, batch size 8, 2-second random paired crops, AdamW at learning rate 0.001 with gradient clipping, and seed 42. Short files are zero-padded with valid-length masking. The loss combines compressed-magnitude MSE, compressed-complex MSE (compression 0.3), and 0.1-weight waveform L1; analysis uses a Hann window, FFT 960, and hop 480.

Speakers `p226,p287` are held out from the training directory by default; override with `--validation-speakers`. Validation uses complete utterances in inference mode. Only validation selects `best.bin`. The official test set is used only with `--evaluate-test`, after training, by loading that checkpoint. `--max-steps`, `--max-validation-items`, and `--max-test-items` are smoke limits: their results are not full-corpus evaluation or evidence of trained denoising quality.

Each run creates a new `target/anr-lite-training/<time>-<pid>` directory, or the new path supplied by `--output-dir`; an existing directory is refused. Artifacts are `config.json`, `splits.json`, `metrics.jsonl`, `summary.json`, `best.bin`, and `last.bin`. The records contain model parameters and BatchNorm state, **not optimizer/RNG state for exact training resume**. Load a trained record with `AnrLite48::new(&device).try_load_file(path)?.valid()`. Use `--help` for the complete argument list. FFmpeg is not required by the trainer.

### Which crate should I use?

Use `anr` in most Rust applications, or install `anr-cli` to work with files. Backend crates separate each implementation from the common interface.

| Crate | When to use it |
| --- | --- |
| `anr` | The usual starting point. Exports `NoiseReduction`, `Error`, and `Result`; select any backend feature listed above for its `DFN3` module. |
| `anr-core` | You need only `NoiseReduction`, `Error`, and `Result`, without a model runtime dependency. |
| `anr-ort` | You want the DFN3 ONNX Runtime implementation directly, or access to the same version of `ort`. |
| `anr-gl` | You want the fixed-model native OpenGL 4.3 DFN3 implementation without ORT. |
| `anr-gles` | Native OpenGL ES 3.1 DFN3, sharing the GL-family executor without desktop-GL or CPU fallback. |
| `anr-vk` | Native Vulkan compute DFN3, without ORT or wgpu. |
| `anr-wgpu` | wgpu compute DFN3, without ORT. |
| `anr-dx` | Native Windows Direct3D 12 compute DFN3, without ORT, DirectML, or wgpu. |
| `anr-cuda` | Complete fixed-model FP32 CUDA Driver/NVRTC inference with replay graphs, without ORT. |
| `anr-tensorrt` | Real native TensorRT neural-network engines plus CUDA DSP/FFT; TensorRT/CUDA SDKs required. |
| `anr-openvino` | Native OpenVINO ONNX import and stateful inference on explicitly selected CPU/GPU/NPU; C++ SDK required. |
| `anr-opencl` | Complete fixed-model FP32 OpenCL 1.2 hardware GPU inference without CPU fallback. |
| `anr-dfn3` | Shared fixed-model parameters, execution plan, worker, and shader sources; not a standalone device backend. |
| `anr-lite` | Develop and train ANR-Lite-48 in Burn, with differentiable streaming audio DSP, optional VoiceBank-DEMAND training, CPU and optional wgpu execution. No pretrained weights included. |
| `anr-cli` | You want the `anr` executable for files, batches, inspection, and raw PCM streams. CPU ORT is enabled by default. |
| `anr-bench` | Workspace-only streaming benchmark for native `gl`/`vk`/`dx`/`cuda`/`opencl`, opt-in native TensorRT/OpenVINO, and optional ORT GPU providers; table/JSON output, not included in application release ZIPs. |

`anr-dfn3`, `anr-lite`, and the native backend crates use this root README in their source packages. `anr-core`, `anr-ort`, the facade, and the CLI each have their own README. Native constructors take a model path (OpenVINO also offers explicit device selection), never an ORT environment or `Session`.

`NoiseReduction` works as `Box<dyn NoiseReduction>`, and reducers can be moved between threads (`Send`). The library crates do not manage audio devices, decode WAV, resample, or split multiple channels. The CLI handles file decoding, resampling, and channels outside the library. Automatic model preparation is provided by the `anr` facade, not the individual backend crates.

### Automated GitHub releases

The [workflow](.github/workflows/ci.yml) tests native Windows x64, Linux x64, and macOS arm64/x64. It downloads the pinned model and licenses, verifies their SHA-256 values, and runs both the control graph and real DFN3 from each extracted ZIP before publishing anything.

| Event | Result after all four platforms pass |
| --- | --- |
| Push to `main` or the current `master` branch | Immutable `main-<12-character SHA>` / `master-<12-character SHA>` prerelease; not marked latest. |
| Push `v<workspace version>` | Version release; a Cargo prerelease suffix remains a GitHub prerelease. |
| Pull request or manual CI | Tests and ZIP artifacts only; no Release. |

Publication uses Actions' built-in **`GITHUB_TOKEN`**, mapped to `GH_TOKEN`, with **Contents: write** granted only to the publication job. No custom repository secret is required; PR and build jobs remain read-only.

GitHub can reject a historical target whose workflow changes require **Workflows: write**, which the built-in token cannot obtain ([API restriction](https://docs.github.com/en/rest/releases/releases#create-a-release)). This accepted boundary produces an explicit failed release; the workflow does not substitute a newer commit, move tags, or overwrite assets.

Commit and push the workflow changes before pushing a version tag. For the current `Cargo.toml` version, the matching tag is `v0.1.1`; a mismatched tag fails rather than releasing mislabeled binaries. This workflow does not publish crates.io packages. Existing public releases and assets are never overwritten on rerun.

### License

The code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Including both full license texts in each standalone package is this project's distribution choice, not a Cargo requirement.

The supplied [model asset release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) is **MIT OR Apache-2.0**, as authorized by the maintainer for the exact SHA-256 in [its manifest](.github/models/dfn3.json); the ZIP carries the model's licenses and notice. Check other models separately: DeepFilterNet's source license does not establish rights for arbitrary exports or weights.

The CLI's Symphonia decoder and the ORT adapter's `onnx-protobuf` message definitions use MPL-2.0; rubato, Rust `ort`, ONNX Runtime, and execution-provider libraries have their own terms. See the [third-party notices](crates/anr-cli/THIRD-PARTY-NOTICES.md) for source, modification, and binary distribution obligations. The project's dual license does not replace those terms.

See [CHANGELOG.md](CHANGELOG.md) for version changes and [CI and Release](.github/workflows/ci.yml) for the maintained release procedure.

## 简体中文

`anr` 提供 Rust 音频降噪库和命令行工具，使用 DeepFilterNet3（DFN3）模型，支持 ONNX Runtime，以及不依赖 ORT 的独立模型专用 OpenGL、OpenGL ES、Vulkan、wgpu、Direct3D 12 Compute、CUDA、TensorRT、OpenVINO 和 OpenCL 后端。

库接口处理 **48 kHz、单声道的 `f32` 采样**。音频文件解码、采样率转换和多声道处理可以交给 [`anr-cli`](crates/anr-cli/README.md)。CLI 用于文件、批处理和 raw PCM 流，不提供麦克风访问、虚拟声卡或驱动。

从源码构建需要 **最新 Rust stable 工具链**。默认 features 会在构建时下载、校验、缓存并复制受支持的 DFN3 模型，`cargo install` 也适用；模型内容不嵌入 Rust 包。GitHub 应用 ZIP 附带同一模型。替换模型前仍需核对输入输出，名称相同不代表接口相同。

### 给 Rust 库选择后端

Rust 库支持 `ort`（ONNX Runtime）、`gl`（OpenGL）、`gles`（OpenGL ES）、`vk`（Vulkan）、`wgpu`、`dx`（Direct3D 12）、`cuda`、`tensorrt`、`openvino` 和 `opencl`。库默认不启用后端；默认的 `ort-download-binaries` 也不会单独启用 ORT。

例如，要使用 `wgpu` 且不启用 ORT，在 `Cargo.toml` 中这样写：

```toml
[dependencies]
anr = { version = "0.1.1", default-features = false, features = ["wgpu"] }
```

在 `Cargo.toml` 的 `features` 中列出要用的后端；写多个就会一起编入程序，程序运行时不会自动切换。非 ORT 后端只支持本项目指定的 DFN3 模型，不能用来运行任意 ONNX 文件。各后端的硬件和系统要求见 [anr crate 文档](crates/anr/README.md)。

### 直接处理音频文件

[GitHub 应用 Release](https://github.com/lona-cn/anr/releases) 由 CI 生成原生 ZIP，包含 CLI、CPU 运行库、`models/dfn3.onnx` 和许可文件。完整解压后，进入解压目录运行：

```sh
./anr denoise voice.wav --output clean.wav
```

Windows 将 `./anr` 换成 `.\anr.exe`；CLI 自动查找可执行文件旁的模型，`--model PATH` 可覆盖默认选择。应用 ZIP 需等发版工作流通过后才会出现；独立的 [模型资产 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 已提供模型，但不是 CLI 安装包。系统运行库要求见 [CLI 文档](crates/anr-cli/README.md)。

在源码工作区根目录安装 CLI，二进制名为 `anr`。默认构建方式使用 ORT 提供的预编译库；Intel macOS 请按 [CLI 文档](crates/anr-cli/README.md) 配置自备运行库。

```text
cargo install --path crates/anr-cli --locked
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav --json
anr backends --json
```

默认构建不需要手动下载模型或填写 `--model`；显式传入 `--model PATH` 仍可选择自己的文件。安装后不必从源码目录运行。CLI 默认使用 CPU ORT，默认保留输入采样率、声道数和帧数，输出 WAV 或 raw f32le。

raw stdin/stdout 的格式、输出文件的保存方式、执行 provider、模型延迟及不会损坏二进制音频的 shell 调用示例，见 [CLI 使用文档](crates/anr-cli/README.md)。

默认构建会下载模型，并可能下载 ONNX Runtime，首次构建需要网络和目标平台工具链。当前 ORT 默认预编译分发不包含 Intel macOS。CI 配置为使用 Rust stable 检查原生 Windows x64、Linux x64、macOS arm64/x64，但配置文件不代表这些任务已经通过；已有本地结果和未实测平台见 CLI 文档。

若要从源码运行不依赖 ONNX Runtime 的 GLES 版本，在工作区根目录执行：

```text
cargo run -p anr-cli --release --no-default-features --features gles -- denoise voice.wav --model models/dfn3.onnx --backend gles --output clean.gles.wav
```

需要受支持的原生 ES 3.1+ 驱动和后文指定的固定模型。即使只编入 GLES，仍须写 `--backend gles`，因为 CLI 的默认选择始终是 `ort`。

### 在 Rust 程序里使用

`anr` 库默认不启用后端，需从后文的后端表选择 feature。以下示例明确启用 ORT 及其 `DFN3` 导出：

```toml
[dependencies]
anr = { version = "0.1.1", features = ["ort"] }
```

依赖示例使用 `0.1.1`。使用本地源码时，可以把版本依赖改成指向对应 crate 目录的 `path` 依赖。

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, anr::default_model_path()?)?;
    denoiser.set_factor(80.0);

    // 用实际的 48 kHz 单声道采样替换这段静音。
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;

    // 清除上一段录音的处理状态，保留降噪强度。
    denoiser.reset();
    Ok(())
}
```

调用 `run` 前：

- 输入至少有一个采样。
- 采样率必须是 48,000 Hz。
- 输出数组由你分配，长度必须与输入相同。

这三项检查失败时，输出和处理状态都不会改变。如果运行模型时才出错，输出和状态可能已经部分更新，不会撤销已经完成的处理。

#### 连续音频每次尽量传入 480 的整数倍

模型一次处理 **480 个采样，也就是 10 ms 音频**。`run` 接受任意非零长度，但末尾不足 480 个采样时会补零，按完整的一帧处理，再只返回原有长度的结果。补上的零也会影响后续处理，所以把录音拆成许多短段，结果可能与整段处理不同。实时收到的零散采样，应先积攒到 480 的整数倍，再交给降噪器。

每路连续音频各用一个 `DFN3` 降噪器，不要让无关录音轮流共用。模型处理后面的音频时会用到前面的信息。换录音或重新从头播放前调用 `reset()`；它会清除音频状态，保留降噪强度。

库不会消除模型带来的时间延迟，也没有在录音结束后额外取出剩余结果的操作。降噪效果、延迟和适用场景取决于实际模型，输入输出匹配并不能证明这些结果。

#### 调整降噪强度

所有原生 DFN3 后端都支持 `NoiseReduction::set_factor(0..=100)`，按 `round(factor / 100 * 60)` 换算成分贝。只有 ORT 具体类型另外提供 `set_atten_lim_db` / `get_atten_lim_db`，可直接设置和读取 `0..=60` dB；它们不属于通用 trait，Compute 前端也不提供。

超出范围的值会被限制到最近的上限或下限；`NaN` 不改变当前设置，正无穷按上限处理，负无穷按下限处理。

`set_factor` 换算出的分贝值会取整，ORT 的直接分贝 setter 保留小数。`factor()` 根据实际分贝设置换算并取整，可能与传入值略有不同。这是衰减上限，不是原音与处理结果的混合比例。零仍会运行模型，对所提供的 DFN3 模型不表示旁路。

#### 用 raw 音频文件试运行

以下示例使用 ORT，需在工作区根目录运行。Compute 后端和带头音频文件应使用 CLI，并明确指定 `--backend`。

```text
cargo run -p anr --features ort --example denoise -- <input.f32le> <output.f32le>
```

输入文件是 **48 kHz、单声道的原始采样，不是 WAV**：每个采样占 4 字节，以小端序保存为 IEEE 754 `f32`。输出格式相同。默认自动查找模型；需要其他模型时，在输入、输出参数前加上 `<model.onnx>`。替换前请确认模型符合下面的要求，并且你有权使用它。

### ORT 模型输入和输出

ORT 适配器只接受下表的三个输入和三个输出，类型都必须为 `f32`。加载时按名称匹配，输入排列不限。方括号表示维度和长度；`[]` 是单个数，`[1]` 是单元素的一维数组。Compute 后端另要求下一节指定的模型 SHA-256，不能只凭端口形状相同判断可用。

| 方向 | 名称 | 运行时的大小 |
| --- | --- | --- |
| 输入 | `input_frame` | `[480]` |
| 输入 | `states` | `[45304]` |
| 输入 | `atten_lim_db` | `[]` 或 `[1]` |
| 输出 | `enhanced_audio_frame` | `[480]` |
| 输出 | `new_states` | `[45304]` |
| 输出 | `lsnr` | `[]` 或 `[1]` |

模型可以给一维数组的长度取名，而不写固定数值。加载时会检查维数和已经写明的长度；符号长度不能改变运行时需要的 480 个采样和 45,304 个状态值。对应常量是 `DFN3::SAMPLE_RATE = 48000`、`FRAME_SIZE = 480`、`STATE_SIZE = 45304`。

音频和状态输出会写入事先分配的空间，ONNX Runtime 会检查实际长度。`lsnr` 只检查模型声明的类型和大小，不读取或返回计算结果。如果模型有额外输入输出、数组维数不同，或状态值的含义和顺序不同，就需要修改实现，不能直接使用。

### 选择后端和运行库

只添加 `anr` 依赖不会启用模型处理功能。按需选择 Cargo features：

| 可选功能 | 作用 |
| --- | --- |
| `ort` | 提供 `DFN3`；保留默认 features 时，会在构建中下载并复制 CPU 运行库。 |
| `gl` | 提供 `anr::gl::DFN3`，直接使用原生 OpenGL 4.3 Compute Shader，不依赖 ORT。 |
| `gles` | 提供 `anr::gles::DFN3`，使用真实原生 OpenGL ES 3.1+ Compute 上下文，不依赖 ORT。 |
| `vk` | 提供 `anr::vk::DFN3`，使用原生 Vulkan 1.1 Compute，不依赖 ORT 或 wgpu。 |
| `wgpu` | 原生平台提供 `anr::wgpu::DFN3`；`wasm32-unknown-unknown` 提供异步 `anr::wgpu::web::DFN3`，均不依赖 ORT。 |
| `dx` | 提供 `anr::dx::DFN3`，使用原生 Windows Direct3D 12 Compute，不是 DirectML 或 wgpu。 |
| `cuda` | 提供 `anr::cuda::DFN3`，使用 CUDA Driver/NVRTC 与 replay graph 完成全流程 FP32 推理，不依赖 ORT。 |
| `tensorrt` | 提供 `anr::tensorrt::DFN3`，使用原生 TensorRT engines 和 CUDA DSP/FFT；构建需要 TensorRT/CUDA SDK。 |
| `openvino` | 提供 `anr::openvino::DFN3` 和 `Device`，原生 OpenVINO Runtime 直接导入原 ONNX；构建需要 C++ SDK，默认明确选择 CPU。 |
| `opencl` | 提供 `anr::opencl::DFN3`，使用 OpenCL 1.2 GPU 完成全流程 FP32 推理，不依赖 ORT，不退回 CPU。 |
| `ort-download-binaries` | 默认下载选项，只在启用 `ort` 后生效，本身不会启用后端。 |
| `download-models` | 默认启用，将固定 DFN3 模型下载并校验到持久 Cargo 缓存；构建时可用 `ANR_MODEL_PATH` 指定本地文件。 |
| `copy-models` | 默认启用，将选中的模型复制到 Cargo 程序、示例和测试旁的 `models/dfn3.onnx`，不会自动启用下载。 |
| `ort-load-dynamic` | 加载自备运行库；配合 `default-features = false` 可去掉下载器和 TLS 构建依赖，通过 `ORT_DYLIB_PATH` 指向共享库文件。 |
| `ort-cuda` / `ort-tensorrt` | 编入 ORT 执行 provider 支持，与独立的 `cuda` / `tensorrt` 后端不同。 |
| `ort-directml` / `ort-coreml` / `ort-xnnpack` | 编入 DirectML / CoreML / XNNPACK 支持。 |

`anr::default_model_path()` 先查找可执行文件旁的模型，再使用构建时记录的绝对路径，运行时不下载。关闭 defaults 可去掉模型下载，也可仅启用 `copy-models` 并将 `ANR_MODEL_PATH` 设为本地文件的绝对路径。缓存位置、离线构建和分发要求见 [模型准备说明](crates/anr/README.md#模型下载与复制)。

当前固定使用 `ort = 2.0.0-rc.15` 和 API 27，不启用 `ndarray`。默认预编译运行库为 1.31.0，也可使用兼容 API 27 的自备运行库，包括 1.28.0。动态构建需在初始化前设置 `ORT_DYLIB_PATH`，指向库文件而不是目录。

创建环境前先调用 `anr::ort::initialize_runtime()`，让动态库加载错误正常返回；这个函数只准备运行库，不创建或覆盖环境。应用创建一个 `Environment`，供多个降噪实例共享。`DFN3::from_file(&environment, path)` 显式选择内置 CPU provider，以一个线程执行算子并关闭线程池忙等待。GPU features 不会选择 GPU 或安装驱动；需要自定义 provider 时，先配置 `SessionBuilder`，调用 `anr::ort::session_from_file(builder, path, &providers)`，再传给 `DFN3::new(session)`。

项目的 ORT 默认策略允许 GPU provider 与 CPU 算子混合执行。CLI 和 benchmark 保留 ORT 对不支持或更适合 CPU 的算子的正常分配，不是 benchmark 的临时选项；`DFN3::new(session)` 保留调用方 Session 的策略。请求的 provider 仍须成功注册，缺少 GPU 库或 provider 初始化失败仍报错。原生 GL/GLES/Vulkan/wgpu/D3D12 实现不变，不因此新增 CPU 执行器。

### 浏览器 WebGPU（WASM）

`wasm32-unknown-unknown` 目标启用 `wgpu` 后，使用异步 Rust 类型 `anr::wgpu::web::DFN3`；直接依赖后端时为 `anr_wgpu::web::DFN3`。通过 `from_bytes(...).await` 加载模型字节，不读取文件路径；`run(sample_rate, input, output).await` 不实现原生同步 `NoiseReduction` trait。模型摘要、WGSL、帧调度及强度/reset 语义与原生实现共用。

在源码工作区安装 target 和与 `Cargo.lock` 匹配的 wasm-bindgen CLI，然后构建 JavaScript/WASM 包：

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked --root target/web-tools
python tools/build_wgpu_web.py
python -m http.server 8000 --bind 127.0.0.1
```

脚本生成 `target/anr-wgpu-web/pkg/`，并将 module Worker 放在其旁边；不会下载模型，需另行提供固定版本的 `models/dfn3.onnx`。浏览器须支持硬件 WebGPU，并运行在 HTTPS 或 localhost；不退回 WebGL 或 CPU。构建期模型下载/复制功能不会将模型嵌入 WASM，也不会在浏览器中下载模型。

以上 HTTP 服务以仓库根目录为起点，可在 JavaScript module 或浏览器控制台使用：

```js
const { default: init, DFN3 } = await import('/target/anr-wgpu-web/pkg/anr_wgpu.js');
await init();
const response = await fetch('/models/dfn3.onnx');
if (!response.ok) throw new Error(`模型请求失败：HTTP ${response.status}`);
const stream = await DFN3.fromBytes(new Uint8Array(await response.arrayBuffer()));
try {
  stream.setFactor(80);
  const output = await stream.run(48_000, new Float32Array(480));
  console.log(output); // 将上面的静音替换为自己的 48 kHz 单声道采样。
} finally {
  stream.free();
}
```

连续音频复用同一个实例。`run` 返回 JavaScript 自有的 `Float32Array`；重叠的 `run` 会拒绝，处理期间调用 `factor()`、`setFactor(...)` 或 `reset()` 会返回错误。`free()` 释放包装对象，但不会使此前已返回的 Promise 失效。设备丢失后须重新创建实例；`reset()` 只重置音频和模型状态。

实时音频应在 [module Worker 示例](crates/anr-wgpu/examples/web/worker.js) 中推理，不在 `AudioWorkletProcessor.process` 内等待 GPU。请求带 `id` 和 `type`：`init` 接收模型 `ArrayBuffer`；`run` 接收 `sampleRate` 和 `Float32Array` 或 `ArrayBuffer` 输入；`setFactor` 接收 `factor`；`reset`、`dispose` 无额外参数。Worker 串行处理，返回 `{id, ok, output?, factor?, error?}`。消息传递可转移模型、输入和输出缓冲，转移后发送方的缓冲会被分离，不能继续使用。

连续输入先累积成 **480 个采样的整数倍**，只有结束时的短帧才补零；补零也会推进完整一帧状态。本后端不提供麦克风采集、重采样、AudioWorklet 缓冲、延迟补偿或结束 flush。

浏览器回归运行真正生成的 WASM，分别通过直接调用和 Worker 检验 reference、reset、无效输入、尾帧状态、并发拒绝及处理期间释放包装对象：

```js
const { runReference } = await import('/crates/anr-wgpu/tests/web/reference.js');
await runReference('/target/anr-wgpu-web/pkg/anr_wgpu.js',
  '/models/dfn3.onnx', '/target/anr-wgpu-web/worker.js');
```

已在 Windows / Edge 135 / NVIDIA Turing 的硬件 WebGPU 上实跑；其他浏览器和移动设备未验证，不保证跨设备实时性。CI 检查并链接 WASM，不冒充 GPU 推理验收。

若 Windows 上的 `sccache` 编译 `web-sys` 时报告命令行长度错误 206，可仅为这次构建禁用 wrapper：

```sh
python -c "import os, runpy; os.environ['RUSTC_WRAPPER']=''; runpy.run_path('tools/build_wgpu_web.py', run_name='__main__')"
```

### 不依赖 ONNX Runtime 的 Compute 后端

CLI 使用 `cargo build -p anr-cli --release --no-default-features --features gl,gles,vk,wgpu,dx` 构建，也可只启用其中部分；运行时明确传 `--backend gl`、`gles`、`vk`、`wgpu` 或 `dx`，不要传 ORT 的 `--provider`。CLI 默认构建、默认后端和应用 Release ZIP 仍使用 ORT。

例如库依赖启用 `default-features = false, features = ["gles"]`，通过 `anr::gles::DFN3::from_file("models/dfn3.onnx")` 创建实例；其余对应 `gl`、`vk`、`wgpu`、`dx` 模块，继续使用 `NoiseReduction`。这些 shader 后端不需要 ORT 环境、运行库、Python、模型转换或外部 shader 编译器。仅从外置 ONNX 读取参数，并校验 SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`，拒绝其他导出版本。新增原生后端也校验同一模型摘要，但 OpenVINO 直接导入并执行原图，不能将所有非 ORT 后端都称为“ONNX 仅提供参数”。

```rust,no_run
use anr::{gles::DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file("models/dfn3.onnx")?;
    denoiser.set_factor(80.0);
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

`anr-dfn3` 共用固定执行顺序、权重布局、工作线程和 WGSL 算子；Naga 生成桌面 GLSL 4.3、GLSL ES 310、SPIR-V 或 HLSL，wgpu 直接使用 WGSL。GLES 复用 `anr-gl` 的执行器，但必须创建 embedded context，不接受桌面 GL 的 ES 兼容模式；因 GLES 没有 `glGetBufferSubData`，在 GPU fence 完成后只映射当前输出帧。FFT 需要 15,360 字节共享内存，低于 ES 3.1 的 16 KiB 最低要求。

对于原生 GL/GLES/Vulkan/wgpu/D3D12，FFT、特征归一化、卷积、分组线性层、五个 GRU、频谱滤波和 overlap-add 全部使用 FP32 Compute Shader，权重和双状态缓冲常驻 GPU。每个原生实例持有工作线程；`run` 等待 GPU 输出，不是非阻塞音频 callback。不返回 `lsnr`，不补偿模型延迟；强度为零保留衰减分支，不表示原音旁路。

| 后端 | 驱动与平台要求 |
| --- | --- |
| `gl` | Windows 桌面 OpenGL 4.3；Linux 硬件 EGL device 需支持桌面 GL 4.3 与 pbuffer，不依赖 X11 `DISPLAY`；不支持 macOS。 |
| `gles` | Windows WGL 驱动需支持原生 ES context；Linux 使用硬件 EGL device/pbuffer，均要求 OpenGL ES 3.1+。本后端未提供 Android、WebGL 或 macOS context。 |
| `vk` | Windows 或 Linux 的硬件 Vulkan 1.1 device，提供 compute queue；不创建窗口或 surface。 |
| `wgpu` | Windows/Linux/macOS 的原生硬件 Vulkan/DX12/Metal；`wasm32-unknown-unknown` 的浏览器 WebGPU 需安全上下文。不使用软件 fallback。 |
| `dx` | Windows 硬件 D3D12 device 和系统 D3DCompiler 47；不使用 WARP、DirectML，也不要求另装 DXC。 |

初始化失败直接返回错误，不退回 CPU。GL/GLES/Vulkan/wgpu/D3D12 这五种后端均已在原生 Windows / RTX 2080 Ti 上通过连续帧独立 ONNX 数值基准及 raw/WAV、立体声、批处理 CLI 验收；GLES-only 编译、非 GPU 测试与 Clippy 也已在 WSL Ubuntu 24.04 通过。不据此保证跨设备性能或音质；Linux GPU 路径、Android 和 macOS 尚未实跑。

将 `ANR_DFN3_MODEL` 设为模型绝对路径后，运行 `cargo test -p anr-gl -p anr-gles -p anr-vk -p anr-wgpu -p anr-dx -- --include-ignored --nocapture`，按实际硬件选择包。数值回归覆盖强度切换、最后不足一帧、reset、跨线程移交和无效输入不改变状态。`tools/generate_gl_reference.py` 使用开发依赖 `onnx==1.19.1`、`numpy==2.2.6` 生成全部九个原生后端的基准，不使用 ONNX Runtime。

#### 原生 CUDA、TensorRT、OpenVINO 和 OpenCL

facade 和 CLI 均须显式启用同名 feature。`cuda` 与 `opencl` 使用固定 FP32 调度执行包括 960 点 FFT、循环网络、滤波和 overlap-add 的完整流程，权重与状态常驻 GPU。CUDA 使用 Driver API、NVRTC 与 replay graph；OpenCL 使用支持 1.2 的硬件 GPU ICD，绝不替换为 CPU device。TensorRT 构建两个真正的标准层神经网络 engine，与 CUDA DSP/FFT 一起捕获到逐帧 graph：强类型 FP32、禁用 TF32，不使用 ORT、ONNX parser 或推理 plugin。

OpenVINO 通过原生 frontend 直接导入原 ONNX，使用 `MakeStateful`（`ReadValue`/`Assign`），每路流独占一个 `InferRequest`。`anr::openvino::DFN3::from_file(path)` 默认明确选择 CPU；其他设备用 `from_file_with_device(path, anr::openvino::Device::Gpu(0))` 或 `Device::Npu`。CPU/GPU 请求偏重精度的 FP32 hints；NPU 硬件可能使用 FP16，未承诺满足 FP32 reference 阈值。GPU/NPU 初始化或编译失败直接报错，不切换到 CPU，不使用 AUTO/HETERO。

准备驱动、运行库和模型后，在工作区根目录运行：

```text
cargo run -p anr-cli --release --no-default-features --features cuda -- denoise voice.wav --model models/dfn3.onnx --backend cuda --output clean.cuda.wav
cargo run -p anr-cli --release --no-default-features --features opencl -- denoise voice.wav --model models/dfn3.onnx --backend opencl --output clean.opencl.wav
cargo run -p anr-cli --release --no-default-features --features tensorrt -- denoise voice.wav --model models/dfn3.onnx --backend tensorrt --output clean.tensorrt.wav
cargo run -p anr-cli --release --no-default-features --features openvino -- denoise voice.wav --model models/dfn3.onnx --backend openvino --openvino-device CPU --output clean.openvino.wav
```

`--openvino-device CPU|GPU|GPU.n|NPU` 仅用于 OpenVINO（`GPU` 表示 `GPU.0`）；`--provider` 仅用于 ORT。即使只编入其他后端，CLI 默认选择也仍是 ORT。

**SDK 与部署：** TensorRT 需要 C++ 工具链及 `TENSORRT_ROOT`、`CUDA_ROOT`（或 `CUDA_PATH`），可用 `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` 覆盖链接库目录。OpenVINO 需要 `OPENVINO_ROOT` 或指向其 CMake 目录的 `OpenVINO_DIR`。官方 OpenVINO wheel 的 `include/`、`libs/` 也可作为 SDK；Python 不是产品运行依赖。例如在 PowerShell 中，将下列常规位置替换为自己的解压安装目录，再执行上面的命令：

```powershell
$env:TENSORRT_ROOT = 'C:\SDKs\TensorRT-10.16.1'
$env:CUDA_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:OPENVINO_ROOT = 'C:\SDKs\openvino'
```

无需系统级安装。CUDA 需要 NVIDIA 驱动，并让加载器找到 NVRTC/builtins 库（Windows `PATH`，Linux `LD_LIBRARY_PATH`）；OpenCL 需要厂商 GPU ICD。TensorRT/OpenVINO 的 `build.rs` 编译私有共享 C++ shim，Rust 通过 `libloading` 加载。分发时将 `anr_tensorrt_native.dll` / `anr_openvino_native.dll`（Linux 为 `libanr_tensorrt_native.so` / `libanr_openvino_native.so`）放在可执行文件旁，并附上所需厂商 runtime、frontend、设备 plugin 及相应许可说明。构建目录中的路径不能替代可移植部署。`ANR_TENSORRT_SHIM`、`ANR_OPENVINO_SHIM` 指向 shim **文件**；Windows 上 `ANR_TENSORRT_RUNTIME` 可指向 TensorRT DLL 目录或文件，`ANR_OPENVINO_RUNTIME` 则指向 OpenVINO runtime 库**文件**。厂商依赖也须能通过标准加载路径解析。默认 Release ZIP 仍是 CPU ORT，不会自动附带这些 SDK 或运行库。

原生 `cuda` 后端要求 Driver API 12.0 或更新版本，以及 major 12 或 13 的 NVRTC。它预检所需 driver 符号，不支持的运行库正常返回错误。加载优先 NVRTC 13、其次 12：Windows 为 `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll`；Linux 为 `libnvrtc.so.13` / `.12` / 无版本号的 `libnvrtc.so`。驱动必须支持所选 NVRTC 生成的 PTX，NVRTC 也须支持 GPU 架构；不保证 NVRTC 13 能搭配任意 12.x 驱动。其 builtins 依赖也必须可被加载器找到。

**已记录实测：** Windows 上四后端均通过真实连续帧 reference 和共 10 项 raw/WAV、立体声、批处理 CLI 检查；CUDA/TensorRT/OpenCL 使用 RTX 2080 Ti，OpenVINO CPU 使用 AMD Ryzen 7 5800X，SDK 版本为 CUDA 13.3、TensorRT 10.16.1、OpenVINO 2026.4.0。OpenVINO GPU 编译因原图 `NonZero_count` kernel 选择失败；NPU 未测。不能据此声称跨平台 GPU 或 OpenVINO GPU/NPU 验证通过。七个原生 benchmark 后端（`gl,vk,dx,cuda,tensorrt,openvino,opencl`）均完成各 100 帧预热、1000 帧正式测量；本次 TensorRT 初始化为 17.342 秒，没有持久 engine cache，不承诺快速冷启动。

### 测试原生 Compute 和 ORT GPU 推理性能

`anr-bench` 是工作区内的独立 benchmark，默认构建只测试原生 Compute，不依赖 ONNX Runtime。在仓库根目录运行：

```sh
cargo run --release -p anr-bench -- --model models/dfn3.onnx --warmup 100 --frames 1000
cargo run --release -p anr-bench -- --backend vk,dx --json
```

默认按 `gl,vk,dx` 顺序串行测试，每个后端预热 100 帧、正式测量 1000 帧，降噪强度为 80。Linux 使用 `--backend gl,vk`，D3D12 仅支持 Windows。`--backend` 可重复指定并保留顺序；`--model` 默认为 `models/dfn3.onnx`。允许 `--warmup 0`，`--frames` 必须大于零，`--strength` 必须是 0–100 的有限数值。

默认原生 benchmark 构建已编入 CUDA 和 OpenCL，但默认选择列表不变。TensorRT 与 OpenVINO 需要显式启用 SDK features：

```sh
cargo run --release -p anr-bench -- --backend cuda,opencl --warmup 100 --frames 1000 --json
cargo run --release -p anr-bench --features tensorrt,openvino -- --backend gl,vk,dx,cuda,tensorrt,openvino,opencl --openvino-device CPU --warmup 100 --frames 1000 --json
```

第二条命令运行前按上文配置 SDK 和运行库路径。`cuda` / `tensorrt` 是独立原生后端，不是 `ort-cuda` / `ort-tensorrt` 的别名。

可选的 `ort-cuda`、`ort-tensorrt`、`ort-directml` features 增加同名的 `--backend` 选项，不改变默认后端列表。DirectML 仅支持 Windows。还需选择运行库来源：`ort-download-binaries` 使用固定版本的预编译分发，`ort-load-dynamic` 通过 `ORT_DYLIB_PATH` 加载兼容 API 27 的共享库文件。

```sh
cargo run --release --target-dir target/benchmark-ort-gpu -p anr-bench --features ort-download-binaries,ort-cuda,ort-tensorrt,ort-directml -- --backend ort-cuda,ort-tensorrt,ort-directml --json
```

使用自备共享库时，在启动前设置 `ORT_DYLIB_PATH`，将 `ort-download-binaries` 替换为 `ort-load-dynamic`，并使用另一构建目录，如 `target/benchmark-ort-dynamic`。GPU features 不会安装 CUDA、cuDNN、TensorRT 或驱动；当前预编译 GPU 分发使用 CUDA 13。provider DLL 必须与 ORT 核心版本匹配：验收中，共享构建目录残留的 1.28 provider DLL 与 1.31 核心混用曾导致 CUDA 访问冲突。使用新构建目录可避免改动其他应用的缓存文件。自备运行库也必须实际包含请求的 provider，同一运行库不一定支持全部三种。

所有 ORT 后端共用一个 Environment，某个 Session 失败后也保留它。每个 Session 只显式注册指定 provider，注册失败即报错；禁用环境默认 provider 和线程池忙等待，使用单线程顺序执行。DirectML 另关闭 memory pattern。ORT 可以按正常规则将部分算子分配到 CPU；TensorRT 不会静默追加 CUDA provider。

**模型准备：** CUDA/TensorRT 通过共用的 `anr-ort` 加载入口，在创建 Session 前转换内存中的副本，不改写原 ONNX 或权重。CUDA 展开不受支持的 sigmoid `FusedConv`；TensorRT 另将静态 `SplitToSequence`/`SequenceAt` 状态访问改为 `Split`，并将能够安全展开的张量 `If` 分支改为 `Where`。CPU、DirectML 及原生 Compute 保留原有加载方式。转换计入初始化，不在每帧重复执行或计时。部分不支持的算子仍可交给 CPU；TensorRT 检查算子支持情况时，可能输出无法导入 vendor 融合算子的诊断，实际初始化或推理错误仍使该后端失败。

在 Windows / RTX 2080 Ti 和版本一致的 ORT 1.31 运行库上，原模型已通过 CUDA/TensorRT/DirectML 与原图 CPU 的数值对照，覆盖动态衰减量、reset 和不足一帧的输入。手动数值回归先设置 `ANR_DFN3_MODEL`，再运行 `cargo test --release --target-dir target/benchmark-ort-gpu -p anr-ort --features cuda,tensorrt,directml --test compat -- --ignored`；匹配的 provider DLL 除 benchmark 可执行文件旁外，也需位于测试程序的 `release/deps` 目录或对应加载路径。

所有后端共用固定种子生成的 1 秒合成音频，包含 220 Hz、880 Hz 正弦波与噪声，以 48 kHz 单声道、每帧 480 个采样连续循环。输入生成、模型读取、shader 编译、ORT Session 构建、预热、统计排序和报告输出不计入正式测量；初始化耗时单独列出。预热后保留模型状态与输入位置，不逐帧 reset，计时循环不由 benchmark 分配内存。

表格包含初始化耗时、同步 `run` 的最小/平均/最大延迟、nearest-rank P50/P95/P99、FPS、RTF，以及**超过** 10 ms 音频时长的帧数。单帧延迟包含后端执行、分配到 CPU 的算子和主机/设备间的数据传输，不是纯 GPU kernel 耗时。原生 GPU Compute 的状态常驻 GPU，OpenVINO 将状态保存在所选 plugin 的 request 内；现有 ORT 适配器使用 CPU 上的音频和状态张量，因此所需传输也计入耗时。FPS 与 `rtf` 使用正式循环的实际总耗时，包含计时和统计记录开销；`rtf = 实际耗时 / 音频时长`，小于 1 表示快于实时。

`--json` 将配置（包含 `openvino_device`）、输入种子、平台/构建信息、后端实际强度、耗时及 `results[].status`（`ok` 或 `error`）写入 stdout，进度与 debug 构建警告写入 stderr。OpenVINO 成功结果另含 `execution_device`。成功结果的 `cpu_fallback_allowed` 在 ORT 下为 true、原生后端下为 false，**明确选择 OpenVINO CPU 时也为 false**；它描述 fallback 策略，不统计算子数量，也不表示所有工作都在 GPU 上完成。强度沿用后端现有的衰减量取整规则。某个后端失败仍继续测试后续后端，但最终退出码为 1；参数错误返回 2。provider 注册失败不会静默换成 CPU。除 OpenVINO 的明确设备选择外，各 crate 仍按原有规则选设备，多 GPU 机器不能假定各 API 选中了同一块卡。合成输入结果不代表音质或真实录音的性能。

### 使用 Burn 开发 ANR-Lite-48

`anr-lite` 是独立的 Burn 0.22.1 模型库，输入为 **48 kHz 单声道语音**。已实现网络、音频处理与可选的 VoiceBank-DEMAND 训练 bin，但**不附带预训练权重，不自动下载数据集**。随机初始化模型不能直接用于有效降噪；`anr` CLI 不会选择它，也不会用它替换固定 SHA-256 的 DFN3 模型。

网络分别处理 64-band ERB 包络与最低 96 个频点的复数细节，编码器均为 16 通道；共享时序网络包含两层 128 维、四分组 GRU，每层后进行跨组混合。两个输出头分别预测全带增益和五 tap 复数残差滤波系数，残差头以零初始化。BatchNorm 沿用 Burn 的 `train()` / `valid()` 模式。

```sh
cargo run -p anr-lite --locked --example smoke
cargo run -p anr-lite --locked --features wgpu --example smoke -- --wgpu
cargo test -p anr-lite --locked --all-targets
```

默认 `flex` feature 使用 Burn CPU 后端；可选 `wgpu` feature 允许明确选择独立 GPU，`--wgpu` 不会静默退回 CPU。示例自行生成音频，明确报告“尚未训练”，比较整段处理与逐帧加 flush 的结果，并检查输入梯度有限且非零，不执行优化器更新。输出的首次调用耗时不是稳态 benchmark，也不代表音质。

```rust,no_run
use anr_lite::{AnrLite48, AudioProcessor};
use anr_lite::burn::{module::Module, tensor::{Device, Tensor}};

let device = Device::flex();
let model = AnrLite48::new(&device).valid();
let processor = AudioProcessor::new(&device);
// 换成自己的 [batch, samples]、48 kHz f32 音频。
let input = Tensor::<2>::zeros([1, 960], &device);
let output = processor.forward(&model, input);
```

`AudioProcessor::forward` 从新状态开始，处理尾帧补零、结束 flush 和已知延迟补偿，返回原始采样数。连续音频使用 `initial_state(batch)` 和接收 `[batch, 480]` 张量的 `forward_frame`。流式输出延迟为 `AUDIO_DELAY_SAMPLES = 1440`（30 ms），不包含收集当前输入帧的时间。结束时先补齐不足一帧的输入，再调用一次 `flush` 取出最后三个延迟帧；flush 会推进状态，开始另一段录音时应新建状态。

自行编写训练循环时，使用启用 autodiff 的设备与 `model.train()`。音频变换、特征归一化、网络及频谱滤波保留跨帧计算图；仅在有意截断反向传播时调用 `AudioState::detach()` 或 `ModelState::detach()`。Burn Module record 保存模型参数和归一化状态，不保存正在处理的音频流状态。

960 点可微 FFT 由原生 64 点 FFT 与固定 radix-15 阶段组合，因为 Burn 原生 FFT 只接受 2 的幂。稀疏 ERB 映射不构造稠密频率矩阵，模型与 DSP 激活不读回 CPU。本实现不是无分配音频 callback 或已融合的原生 shader，也没有证明比 DFN3 更快或音质更好。

#### 使用 VoiceBank-DEMAND 28-speaker 训练

启用 `train` feature 后可构建 `anr-lite-train`。从工作区根目录运行，`--dataset` 默认是 `datasets/VoiceBank-DEMAND 28-speaker`。已解压的数据集应包含 `clean_trainset_28spk_wav`、`noisy_trainset_28spk_wav`、`clean_testset_wav`、`noisy_testset_wav` 四个目录。加载器要求同名、等长的 48 kHz 单声道 PCM16 WAV 配对，不重采样、不分别归一化 noisy/clean，也不修改数据集文件；无需文本转录。

```sh
# CPU
cargo run -p anr-lite --release --locked --features train --bin anr-lite-train -- --device cpu
# 明确选择独立 GPU
cargo run -p anr-lite --release --locked --features train,wgpu --bin anr-lite-train -- --device wgpu
# 短跑验收：真实更新参数、限量验证，并重载 checkpoint 测试
cargo run -p anr-lite --locked --features train,wgpu --bin anr-lite-train -- --device wgpu --epochs 1 --batch-size 2 --segment-seconds 0.04 --max-steps 2 --max-validation-items 1 --evaluate-test --max-test-items 1
```

默认训练 50 轮、batch size 8、随机裁剪 2 秒的配对片段，使用学习率 0.001 的 AdamW 与梯度裁剪，随机种子为 42。短文件补零，并按实际有效长度计算损失。损失由压缩幅度谱 MSE、压缩复数谱 MSE（压缩指数 0.3）及权重为 0.1 的波形 L1 组成，分析使用 Hann 窗、960 点 FFT、480 点 hop。

默认从训练目录留出 `p226,p287` 两位说话人验证，可用 `--validation-speakers` 修改。验证以推理模式处理完整语句，只用验证损失选择 `best.bin`。只有显式添加 `--evaluate-test`，才会在训练结束后加载该 checkpoint 评估官方测试集。`--max-steps`、`--max-validation-items`、`--max-test-items` 用于短跑检查，其结果不代表完整数据集评估或已训练模型的降噪质量。

每次运行新建 `target/anr-lite-training/<time>-<pid>`，也可用 `--output-dir` 指定新路径；已有目录会被拒绝，不覆盖旧训练结果。输出包括 `config.json`、`splits.json`、`metrics.jsonl`、`summary.json`、`best.bin`、`last.bin`。模型 record 保存权重与 BatchNorm 状态，**不包含优化器和随机数状态，不能用于精确续训**。加载训练后的权重使用 `AnrLite48::new(&device).try_load_file(path)?.valid()`。完整参数见 `--help`；训练程序本身不依赖 FFmpeg。

### 各个包怎么选

Rust 应用通常使用 `anr`；要直接处理文件则安装 `anr-cli`。各后端包将具体实现与公共接口分开。

| 包 | 什么时候用 |
| --- | --- |
| `anr` | 一般使用。提供 `NoiseReduction`、`Error`、`Result`；按上表选择后端 feature 和对应 `DFN3` 模块。 |
| `anr-core` | 只需要 `NoiseReduction`、`Error` 和 `Result`，不希望引入模型运行库。 |
| `anr-ort` | 直接使用 DFN3 的 ONNX Runtime 实现，或使用它提供的同一版本 `ort`。 |
| `anr-gl` | 直接使用固定模型的原生 OpenGL 4.3 DFN3 实现，不依赖 ORT。 |
| `anr-gles` | 原生 OpenGL ES 3.1 DFN3，复用 GL 执行器，不退回桌面 GL 或 CPU。 |
| `anr-vk` | 直接使用原生 Vulkan Compute DFN3，不依赖 ORT 或 wgpu。 |
| `anr-wgpu` | 使用 wgpu Compute DFN3，不依赖 ORT。 |
| `anr-dx` | 使用原生 Windows Direct3D 12 Compute DFN3，不依赖 ORT、DirectML 或 wgpu。 |
| `anr-cuda` | 完整固定模型 FP32 CUDA Driver/NVRTC 推理及 replay graph，不依赖 ORT。 |
| `anr-tensorrt` | 真正的原生 TensorRT 神经网络 engines 与 CUDA DSP/FFT；需要 TensorRT/CUDA SDK。 |
| `anr-openvino` | 原生 OpenVINO ONNX 导入与有状态推理，明确选择 CPU/GPU/NPU；需要 C++ SDK。 |
| `anr-opencl` | 完整固定模型 FP32 OpenCL 1.2 硬件 GPU 推理，不退回 CPU。 |
| `anr-dfn3` | 固定模型参数、执行顺序、工作线程与 shader 源码；本身不提供设备后端。 |
| `anr-lite` | 使用 Burn 开发和训练 ANR-Lite-48，包含可微流式音频处理、可选的 VoiceBank-DEMAND 训练程序、CPU 与可选 wgpu 执行；不附带预训练权重。 |
| `anr-cli` | 安装名为 `anr` 的命令行工具，处理文件、批处理、音频信息查询和 raw PCM 流；默认启用 CPU ORT。 |
| `anr-bench` | 工作区内连续帧 benchmark，支持原生 `gl`/`vk`/`dx`/`cuda`/`opencl`、显式启用的原生 TensorRT/OpenVINO，以及可选 ORT GPU provider；输出表格或 JSON，不包含在应用 Release ZIP 中。 |

`anr-dfn3`、`anr-lite` 和原生后端 crate 在源码包中复用本根 README；`anr-core`、`anr-ort`、facade 和 CLI 各有独立 README。原生构造器传入模型路径（OpenVINO 另提供明确设备选择），不接收 ORT 的环境或 `Session`。

`NoiseReduction` 支持 `Box<dyn NoiseReduction>`，降噪器也可以移交给另一个线程（`Send`）。库包不负责音频设备管理、WAV 解码、采样率转换或多声道拆分；CLI 在库外处理文件解码、重采样和多声道。自动模型准备由 `anr` 入口 crate 提供，不放进各个后端 crate。

### 自动发布到 GitHub

[工作流](.github/workflows/ci.yml) 在 Windows x64、Linux x64、macOS arm64/x64 原生 runner 验证。它从固定 Release 下载模型及许可、核对 SHA-256，再解压各平台 ZIP，用控制图和真实 DFN3 实跑；四个平台全部通过后才公开发版。

| 触发事件 | 四个平台通过后的结果 |
| --- | --- |
| push 到 `main` 或现有 `master` | 生成不可变的 `main-<12位SHA>` / `master-<12位SHA>` 预发布，不设为 latest。 |
| push `v<工作区版本>` tag | 发布对应版本；Cargo 预发布后缀仍标为 GitHub prerelease。 |
| pull request 或手动 CI | 只测试和生成 ZIP artifacts，不发布 Release。 |

发布使用 Actions 自动提供的 **`GITHUB_TOKEN`**，映射为 `GH_TOKEN`，仅发布 job 授予 **Contents: write**；无需配置额外仓库 Secret，PR 与构建 job 保持只读。

部分历史提交的 workflow 变更可能触发 **Workflows: write** 授权检查，内置 token 无法获得该权限（[API 说明](https://docs.github.com/en/rest/releases/releases#create-a-release)）。当前策略接受这类发版明确失败，不换用较新的提交，也不移动 tag 或覆盖资产。

先提交并推送工作流改动，再推版本 tag。当前 `Cargo.toml` 对应 `v0.1.1`；tag 与版本不匹配会失败，不生成版本错误的包。该工作流不发布 crates.io，也不会在重跑时覆盖已有公开 Release 或资产。

### 许可

代码采用 **MIT OR Apache-2.0**，使用时任选其一。完整文本见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。每个独立发布包附带两份完整文本是本项目的分发选择，不是 Cargo 的强制要求。

本仓库提供的 [模型资产 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 已由维护者授权按 **MIT OR Apache-2.0** 分发，仅限 [清单](.github/models/dfn3.json) 中 SHA-256 对应的文件；ZIP 附带模型许可与说明。其他来源的模型仍需单独确认，不能从 DeepFilterNet 源码的许可推断任意导出文件或权重的权利。

CLI 的 Symphonia 解码器与 ORT 适配器使用的 `onnx-protobuf` 消息定义采用 MPL-2.0；rubato、Rust `ort`、ONNX Runtime 及执行 provider 所依赖的库也各有许可要求。源码获取、修改与二进制分发义务见 [第三方说明](crates/anr-cli/THIRD-PARTY-NOTICES.md)，项目双许可不替代这些条款。

版本变更见 [CHANGELOG.md](CHANGELOG.md)，持续维护的发版步骤见 [CI and Release](.github/workflows/ci.yml)。
