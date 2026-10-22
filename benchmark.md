# Benchmark

## 结果

| 后端 | 准备（ms） | 平均每帧（ms） | P99（ms） | FPS | RTF | 超过 10 ms 的帧数 / 1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gl` | 383.3 | 1.612 | 2.854 | 620.0 | 0.1613 | 0 |
| `vk` | 135.5 | 0.885 | 1.685 | 1129.3 | 0.0886 | 0 |
| `dx` | 1353.6 | 1.150 | 2.394 | 869.7 | 0.1150 | 0 |
| `cuda` | 284.7 | 0.942 | 1.787 | 1060.6 | 0.0943 | 1 |
| `tensorrt` | 19845.2 | 0.433 | 1.018 | 2310.9 | 0.0433 | 0 |
| `openvino` (CPU) | 1028.1 | 1.951 | 7.046 | 512.4 | 0.1951 | 1 |
| `opencl` | 128.4 | 0.913 | 1.708 | 1094.7 | 0.0913 | 0 |
| `ort-cuda` | 427.7 | 5.812 | 10.811 | 172.0 | 0.5812 | 19 |
| `ort-tensorrt` | 16826.9 | 4.800 | 7.908 | 208.3 | 0.4800 | 0 |
| `ort-directml` | 469.5 | 6.976 | 10.047 | 143.3 | 0.6976 | 11 |

每帧对应 10 ms 音频；RTF 为处理时长/音频时长，小于 1 表示快于实时。TensorRT 平均每帧最快，但准备时间最长；ORT CUDA 和 DirectML 有超时帧。准备时间不含 100 帧预热及首次推理中的延迟初始化。

## 测试条件

- DFN3：48 kHz 单声道，480 采样/帧，降噪强度 80；预热 100 帧，测量 1000 帧。
- 输入：220 Hz 和 880 Hz 音调加噪声，固定种子 `0x12345678`；连续处理，保留帧间状态。
- 单轮测试，顺序同表；未清理后台程序或驱动缓存。非音质测试。
- Windows x64（内部版本 26300）；AMD Ryzen 7 5800X；NVIDIA GeForce RTX 2080 Ti（驱动报告显存 22,528 MiB）。
- Rust 1.99.0（`--release`）；NVIDIA 驱动 616.64；CUDA 13.3.73、cuDNN 9.26、TensorRT 10.16.1.11；OpenVINO 2026.4.0；ONNX Runtime 1.31.0、`ort` 2.0.0-rc.15。
- 模型 `models/dfn3.onnx` SHA-256：`fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`。

OpenVINO 仅测 CPU；GPU 编译未成功，未测 NPU。GLES、wgpu、ORT CPU、CoreML 和 XNNPACK 没有测试入口。ORT TensorRT 测试虽有不支持部分计算的日志，仍正常完成，且允许 CPU 参与计算；它与 `tensorrt` 是不同后端。

## 复现

按 [README](README.md#native-cuda-tensorrt-openvino-and-opencl) 安装依赖并配置 DLL，然后在项目根目录运行：

```powershell
cargo run --locked --release `
  --target-dir target/benchmark-ort-gpu `
  -p anr-bench `
  --features "tensorrt,openvino,ort-download-binaries,ort-cuda,ort-tensorrt,ort-directml" -- `
  --model models/dfn3.onnx `
  --backend "gl,vk,dx,cuda,tensorrt,openvino,opencl,ort-cuda,ort-tensorrt,ort-directml" `
  --openvino-device CPU `
  --warmup 100 --frames 1000 --strength 80 --json
```