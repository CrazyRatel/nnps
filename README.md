# nnps

一个使用 CUDA 和 CMake 构建的实验性 GPU 项目。

## 环境要求

- CMake 3.31 或更高版本
- 支持 C++17 的编译器
- NVIDIA CUDA Toolkit
- NVIDIA GPU 及匹配的驱动程序
- CLion（可选，也可以直接使用命令行构建）

当前工程通过 `project(nnps CUDA)` 启用 CUDA，并使用 C++17 标准。

### 平台说明

- Windows：需要 Visual Studio 的 MSVC 编译器，并确保 CLion 使用了正确的 Visual Studio Toolchain。`nvcc` 需要能够找到 `cl.exe`。
- Linux：需要 GCC 或 Clang、CUDA Toolkit，以及与 CUDA 版本兼容的 NVIDIA 驱动。
- macOS：现代 macOS 通常不支持 NVIDIA CUDA Toolkit，因此本项目的 CUDA 版本通常无法在 macOS 上直接编译或运行。若要支持 macOS，需要后续增加 CPU、Metal 或其他计算后端。

可以使用以下命令确认 CUDA 编译器可用：

```text
nvcc --version
```

Windows 下还应确认 MSVC 编译器可用：

```text
where cl.exe
```

Linux 下可以确认主机编译器可用：

```text
g++ --version
```

## 使用 CLion

1. 使用 CLion 打开项目根目录。
2. 选择已安装的 CMake、CUDA Toolkit 和 C++ Toolchain。
3. 等待 CMake 重新加载项目。
4. 构建并运行 `nnps` 配置。

Windows 用户建议从已加载 Visual Studio 开发环境的终端启动 CLion，或在 CLion 的 Toolchains 设置中正确配置 Visual Studio。

## 命令行构建

在项目根目录执行：

```text
cmake -S . -B build
cmake --build build --config Release
```

运行生成的 `nnps` 可执行文件。不同平台的可执行文件路径可能不同：

- Windows：`build/Release/nnps.exe` 或 `build/nnps.exe`
- Linux/macOS：`build/nnps`

## 换行符与构建目录

项目使用 `.gitattributes` 统一源码和构建文件的提交换行符为 LF，Windows 专用脚本保留 CRLF。请不要将 `cmake-build-*`、`build` 等本地构建目录提交到版本库。

如果项目已有文件需要按 `.gitattributes` 重新规范化，可以执行：

```text
git add --renormalize .
```

## 许可证

本项目以 GNU General Public License v3.0 发布，完整协议文本见 [LICENSE](LICENSE)。使用、修改或再发布本项目时，请遵守 GPL-3.0 的要求并保留协议声明。
