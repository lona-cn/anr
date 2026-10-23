# 发布

GitHub Actions 构建并发布 CLI 应用包；crates.io 源码包由维护者手动发布。两者互相独立。以下命令均在仓库根目录执行。

## 准备版本

- 更新根 `Cargo.toml` 的 workspace 版本、内部依赖版本要求和 [CHANGELOG.md](CHANGELOG.md)，同步 `Cargo.lock` 中的工作区包版本；不要顺带升级第三方依赖。
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

版本 tag 必须与 workspace 版本完全一致，否则 prepare job 会失败。例如工作区版本为 `0.3.0` 时：

```text
git tag v0.3.0
git push origin v0.3.0
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

首次发布前确认包名归属和发布权限，验证维护者账号的邮箱，并按 [Cargo 发布说明](https://doc.rust-lang.org/cargo/reference/publishing.html#before-your-first-publish) 配置 token（`cargo login --registry crates-io`）。不要将 token 写进仓库、命令示例或日志。

发布命令显式指定 `--registry crates-io`，避免本机默认 registry 或 `replace-with` 镜像配置影响上传目标。dry-run 不上传，但仍可能需要联网读取 crates.io 配置与索引；没有相应缓存时不能加 `--offline`。

按下表从上到下发布，等待前置版本可被 registry 解析后再处理下一层：

| 顺序 | 包 |
| --- | --- |
| 1 | `anr-core` |
| 2 | `anr-ort`、`anr-dfn3` |
| 3 | `anr-gl`、`anr-vk`、`anr-wgpu`、`anr-dx`、`anr-cuda`、`anr-tensorrt`、`anr-openvino`、`anr-opencl` |
| 4 | `anr-gles`、`anr-bench` |
| 5 | `anr` |
| 6 | `anr-cli` |

`anr-lite` 可独立发布；`anr-bench` 在其原生后端及 ORT 依赖发布后发布。即使未启用某个后端，registry 也必须能解析 `anr` 和 `anr-bench` 的可选依赖。

对每个待发布的 crate 检查文件清单、解包构建和 dry-run。以下以 `anr-core` 为例：

```text
cargo package --locked -p anr-core --list
cargo package --locked -p anr-core
cargo publish --registry crates-io --locked -p anr-core --dry-run
```

清单应包含源码、声明的 README、两份项目许可，以及该 crate 所需的 shader、kernel、native shim 源码和测试 fixture；CLI 另需 `THIRD-PARTY-NOTICES.md`。完整模型、录音、输出、SDK 和厂商运行库不进入源码包。`anr-lite` 没有显式 `include`，发布时需逐项检查其实际清单。

TensorRT/OpenVINO 的解包构建也需要 SDK。若前置依赖无法解析或构建失败，先解决错误，不用 `--no-verify` 跳过检查。脏工作区需要明确审查后再使用 `--allow-dirty`，不要为打包丢弃已有改动。

检查通过后上传：

```text
cargo publish --registry crates-io --locked -p anr-core
```

GitHub workflow 不执行 `cargo publish`。同名同版本的 crate 一经发布不能覆盖。

当前发布版本以根 `Cargo.toml` 的 `[workspace.package].version` 为准；下节保留 v0.2.0 发布前的历史文档核查记录，不代表本次发布已经完成相同验证。

### v0.2.0 发布前文档核查

16 个 crate 均可发布，已声明 `description`、`license`、`repository` 和 `readme`；源码包应含 README 和两份项目许可，CLI 另含第三方说明。`anr-bench` 也提供独立 README、许可和包内测试 fixture，但不包含在应用 ZIP 中。现有说明已覆盖模型来源、默认 features、运行库/SDK、平台限制和发布顺序，不需要从头重写。

已完成两项高优先级文档工作：

- **源码包 README 链接**：十个复用仓库首页的 crate 及其余六个包，跨包和仓库文件链接使用对应版本的绝对地址；包内许可、示例和源文件保留相对链接。`anr-bench` 的新指南由其 crates.io 页面提供，不移动已经发布的 `v0.2.0` tag。
- **registry 使用入口**：仓库首页、CLI、库和 ANR-Lite 已补齐中英文安装或依赖示例，保留 checkout 安装、离线及 Intel macOS 配置。ANR-Lite 已区分安装后的训练/导出工具与必须 checkout 的 Python 评分和复现流程。

registry 命令适用于 `0.2.0` 发布后，版本固定的仓库链接需等 `v0.2.0` tag 发布后才能访问。本机已验证源码安装和独立工作目录下的工具使用，并通过本地 `[patch.crates-io]` 验证外部应用示例；这些检查不能代替 registry 安装验收。包含内部依赖的包在正式发布时，registry 必须能解析其依赖版本，不能用本机 workspace 构建代替。

本机 `cargo install --locked` 检查还报告 `Cargo.lock` 中的 `tokio 1.54.0` 已被 yank；安装成功，但正式发布前应核对并决定是否更新该依赖。本次文档修改没有升级依赖。

以下是仍建议补充的文档，不是 crates.io 上传接口的额外必填项：

| 优先级 | 工作 | 当前情况与具体修改 |
| --- | --- | --- |
| 中 | 集中说明从 0.1.x 升级的注意事项 | 变更记录已有详细条目，但应补一段面向已有用户的说明：默认模型下载/复制及其离线配置、库默认不启用执行后端、CLI 的 `--model` 变为可选、ORT provider 可按算子使用 CPU。不要把版本升级本身写成已完成 crates.io 发布。 |
| 低 | 补充搜索分类 | `anr-dfn3`、九个原生/图形计算后端和 `anr-lite`，共十一个包缺少 `keywords` / `categories`；按用途补充有效分类。这两项可选，不影响上传。 |

`description` 与 `license` / `license-file` 是 crates.io 要求的元数据；README、仓库地址和搜索信息用于帮助使用者。没有独立网站时不必重复填写 `homepage`；未显式设置 `documentation` 时，crates.io 会在 docs.rs 构建成功后自动提供文档入口。

### docs.rs 还有构建问题需要处理

`anr` 和 `anr-ort` 已配置关闭默认 features 并使用动态 ORT；`anr/build.rs` 也会在 `DOCS_RS` 环境跳过模型准备。本机以下严格文档构建已通过，但它不等同于 docs.rs 云端验收：

```text
cargo doc --locked --no-deps -p anr-core -p anr-ort -p anr --no-default-features --features anr/ort-load-dynamic --config "build.rustdocflags=['-D','warnings']"
```

`anr-tensorrt/build.rs` 与 `anr-openvino/build.rs` 仍无条件查找厂商 SDK，没有 `DOCS_RS` 分支。[INFERENCE] 在没有这些 SDK 的 docs.rs 沙箱中，文档构建预计失败；本次未运行云端构建。若要提供这两个 crate 的在线 API 文档，需要实现并验证无需 SDK 的文档构建方式，同时保留普通构建的 SDK 要求；只补 README 无法解决。其他平台相关后端也应明确文档目标和 features，不要统一启用 `all-features`。

官方依据：[Cargo 发布流程](https://doc.rust-lang.org/cargo/reference/publishing.html)、[manifest 字段](https://doc.rust-lang.org/cargo/reference/manifest.html)、[publish 选项](https://doc.rust-lang.org/cargo/commands/cargo-publish.html)、[docs.rs 沙箱与构建限制](https://docs.rs/about/builds)、[docs.rs 配置](https://docs.rs/about/metadata)。

## 分发文件与许可

默认应用 ZIP 使用 CPU ORT，应包含：

- CLI 可执行文件、README、项目的 MIT/Apache-2.0 许可及第三方说明。
- 可执行文件旁的 ORT 共享库，以及 `licenses/onnxruntime/` 中**实际运行库版本**的 `LICENSE` 和 `ThirdPartyNotices.txt`。
- `models/dfn3.onnx`、两份模型许可、NOTICE 和 `dfn3.json`。

打包脚本尚未收集完整的 Rust 依赖许可。发布者还需按 `Cargo.lock` 和启用的 features 补齐版权、许可、适用 NOTICE 及 MPL 源码获取说明，详见[第三方说明](crates/anr-cli/THIRD-PARTY-NOTICES.md)。

固定模型的分发授权见[模型 NOTICE](.github/models/NOTICE.md)，文件摘要以 [dfn3.json](.github/models/dfn3.json) 为准。更换模型时同时审查授权、更新 manifest 和代码中的固定摘要，并重跑数值回归。

普通构建的 `copy-models` 只复制模型，不复制许可或 NOTICE；自行分发构建产物时需要另附这些文件。定制 GPU/原生后端包还需收集私有 shim、厂商运行库和设备插件，具体布局见[库部署说明](crates/anr/README.md)，不要直接复制默认 CPU 包或整个 SDK 目录。

保存发布对应的提交、CI 运行链接和 `SHA256SUMS`。自定义后端包还应记录所用 SDK、运行库和驱动版本，便于重现用户的运行环境。
