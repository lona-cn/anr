# 发布

GitHub Actions 构建并发布 CLI 应用包；crates.io 源码包由维护者手动发布。两者互相独立。以下命令均在仓库根目录执行。

## 准备版本

- 更新根 `Cargo.toml` 的 workspace 版本、内部依赖版本要求和 [CHANGELOG.md](CHANGELOG.md)。
- 内部依赖同时保留 `path` 和 `version`。除非计划升级 ORT，否则保留 `ort = "=2.0.0-rc.15"`。
- 使用最新 Rust stable。准备[固定 DFN3 模型](.github/models/dfn3.json)，将 `ANR_DFN3_MODEL` 设为模型文件的绝对路径，供真实模型测试使用。

## 运行发布检查

以下检查不需要 TensorRT/OpenVINO SDK，与 [CI workflow](.github/workflows/ci.yml) 的常规检查一致：

```text
cargo +stable fmt --all -- --check
cargo +stable build --locked -p anr-cli
cargo +stable test --locked --workspace --exclude anr-tensorrt --exclude anr-openvino --all-targets
cargo +stable test --locked -p anr-ort --test dfn3 -- --ignored
cargo +stable test --locked -p anr-cli --no-default-features
cargo +stable test --locked -p anr-cli --no-default-features --features gl,gles,vk,wgpu,dx,cuda,opencl
cargo +stable clippy --locked --workspace --exclude anr-tensorrt --exclude anr-openvino --all-targets -- -D warnings
cargo +stable doc --locked --workspace --exclude anr-tensorrt --exclude anr-openvino --no-deps --config "build.rustdocflags=['-D','warnings']"
```

CI 还检查明确列出的 provider features、动态运行库初始化和 WebAssembly 构建，完整参数见 workflow。不要改成 `--all-features`：那会同时启用需要厂商 SDK 的后端。

### 硬件后端

在支持相应 API 的设备上单独运行 ignored 测试，选择本机实际支持的 crate。TensorRT 需要 C++17、TensorRT 和 CUDA SDK；OpenVINO 需要 C++17 和 OpenVINO C++ SDK。SDK 路径、驱动及运行库设置见[库文档](crates/anr/README.md)。

配置好两个 SDK 后，可检查：

```text
cargo +stable check --locked -p anr-tensorrt -p anr-openvino
cargo +stable test --locked -p anr-tensorrt -p anr-openvino --test dfn3 -- --include-ignored --skip gpu_matches_streaming --nocapture
```

第二条运行 TensorRT GPU 和 OpenVINO CPU reference，跳过 OpenVINO GPU。需要发布 OpenVINO GPU 配置时，另行运行：

```text
cargo +stable test --locked -p anr-openvino --test dfn3 gpu_matches_streaming -- --include-ignored --nocapture
```

原图在部分 OpenVINO GPU 上会因 `NonZero_1_count` / `count_nonzero` 编译失败。该设备的检查未通过前，不要将它列为已支持的发布配置。

## GitHub 应用包

[release_metadata.py](.github/scripts/release_metadata.py) 根据触发事件决定是否发布：

| 触发方式 | 结果 |
| --- | --- |
| push 到 `main` / `master` | 创建 `<分支名>-<12位提交SHA>` tag 和 prerelease |
| push `v<workspace版本>` tag | 发布该版本；带预发布后缀的版本标为 prerelease |
| pull request 或手动运行 workflow | 上传 Actions artifacts，不创建 Release |

版本 tag 必须与 workspace 版本完全一致，否则 prepare job 会失败。例如工作区版本为 `0.1.1` 时：

```text
git tag v0.1.1
git push origin v0.1.1
```

发布 job 等待 Windows x64、Linux x64、macOS arm64、macOS x64 及 WebGPU 检查全部成功。它使用内置 `GITHUB_TOKEN`，只有发布 job 获得 `contents: write`。

应用包名为 `anr-<label>-<target>.zip`。流水线依次执行：

1. **获取固定资产**：[fetch_release_assets.py](.github/scripts/fetch_release_assets.py) 按模型和运行库 manifest 校验文件大小及 SHA-256。CPU ONNX Runtime 版本由 [onnxruntime.json](.github/onnxruntime.json) 固定，当前为 1.28.0；Intel macOS 从其中记录的提交构建，其余平台使用官方归档。
2. **构建与打包**：以 `--no-default-features --features ort-load-dynamic` 构建 CLI，再由 [package_release.py](.github/scripts/package_release.py) 收集文件。macOS 动态库会修正路径并作 ad-hoc 签名，未作 notarization。
3. **解压测试**：[smoke_release.py](.github/scripts/smoke_release.py) 从无关目录运行解压后的程序，检查模型与包内容、控制图状态、双声道 WAV/inspect，以及自动查找模型的 48,001-sample raw 推理。成功后上传 artifact。
4. **发布**：[publish_release.py](.github/scripts/publish_release.py) 收集四个平台的 ZIP，生成 `SHA256SUMS`，上传到 draft 并核验资产，最后公开。

已公开的版本和资产不覆盖，已有 tag 不移动。重跑遇到冲突时，先核对目标提交和资产摘要；不完整的公开 Release 需要维护者处理，脚本不会自动修补。

手动检查一个本机平台的包，可复用同一测试：

```text
python .github/scripts/smoke_release.py --archive <本平台ZIP路径> --control-model crates/anr-cli/tests/fixtures/control.onnx
```

系统运行要求见 [CLI 安装说明](crates/anr-cli/README.md)。WebAssembly 只参与 CI 检查，不在这些 ZIP 中分发。

## crates.io 源码包

首次发布前确认包名归属和发布权限。按下表从上到下发布，等待前置版本可被 registry 解析后再处理下一层：

| 顺序 | 包 |
| --- | --- |
| 1 | `anr-core` |
| 2 | `anr-ort`、`anr-dfn3` |
| 3 | `anr-gl`、`anr-vk`、`anr-wgpu`、`anr-dx`、`anr-cuda`、`anr-tensorrt`、`anr-openvino`、`anr-opencl` |
| 4 | `anr-gles` |
| 5 | `anr` |
| 6 | `anr-cli` |

`anr-lite` 可独立发布；`anr-bench` 设置了 `publish = false`。即使未启用某个后端，registry 也必须能解析 `anr` 的可选依赖。

对每个待发布的 crate 检查文件清单、解包构建和 dry-run。以下以 `anr-core` 为例：

```text
cargo package --locked -p anr-core --list
cargo package --locked -p anr-core
cargo publish --locked -p anr-core --dry-run
```

清单应包含源码、声明的 README、两份项目许可，以及该 crate 所需的 shader、kernel、native shim 源码和测试 fixture；CLI 另需 `THIRD-PARTY-NOTICES.md`。完整模型、录音、输出、SDK 和厂商运行库不进入源码包。`anr-lite` 没有显式 `include`，发布时需逐项检查其实际清单。

TensorRT/OpenVINO 的解包构建也需要 SDK。若前置依赖无法解析或构建失败，先解决错误，不用 `--no-verify` 跳过检查。脏工作区需要明确审查后再使用 `--allow-dirty`，不要为打包丢弃已有改动。

检查通过后上传：

```text
cargo publish --locked -p anr-core
```

GitHub workflow 不执行 `cargo publish`。同名同版本的 crate 一经发布不能覆盖。

## 分发文件与许可

默认应用 ZIP 使用 CPU ORT，应包含：

- CLI 可执行文件、README、项目的 MIT/Apache-2.0 许可及第三方说明。
- 可执行文件旁的 ORT 共享库，以及 `licenses/onnxruntime/` 中**实际运行库版本**的 `LICENSE` 和 `ThirdPartyNotices.txt`。
- `models/dfn3.onnx`、两份模型许可、NOTICE 和 `dfn3.json`。

打包脚本尚未收集完整的 Rust 依赖许可。发布者还需按 `Cargo.lock` 和启用的 features 补齐版权、许可、适用 NOTICE 及 MPL 源码获取说明，详见[第三方说明](crates/anr-cli/THIRD-PARTY-NOTICES.md)。

固定模型的分发授权见[模型 NOTICE](.github/models/NOTICE.md)，文件摘要以 [dfn3.json](.github/models/dfn3.json) 为准。更换模型时同时审查授权、更新 manifest 和代码中的固定摘要，并重跑数值回归。

普通构建的 `copy-models` 只复制模型，不复制许可或 NOTICE；自行分发构建产物时需要另附这些文件。定制 GPU/原生后端包还需收集私有 shim、厂商运行库和设备插件，具体布局见[库部署说明](crates/anr/README.md)，不要直接复制默认 CPU 包或整个 SDK 目录。

保存发布对应的提交、CI 运行链接和 `SHA256SUMS`。自定义后端包还应记录所用 SDK、运行库和驱动版本，便于重现用户的运行环境。
