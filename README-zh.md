## 🌐 语言
[**中文**](README-zh.md) | [English](README.md)

# PASS（Particle Accelerator Simulation Studio）

[![文档](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://changmx.github.io/PASS/) [![许可证](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/changmx/PASS)](https://github.com/changmx/PASS/releases/latest)

PASS 是面向粒子加速器的多功能模拟平台，支持 Windows 和 Linux，提供 CPU 与 NVIDIA GPU 两种执行后端，旨在为束流动力学研究提供高性能、可扩展且可复现的六维粒子追踪与分析工具。

## 文档

完整文档发布在 [changmx.github.io/PASS](https://changmx.github.io/PASS/)，其中提供[中文文档](https://changmx.github.io/PASS/zh/)和[English documentation](https://changmx.github.io/PASS/en/)。文档网站包含物理模型、输入参数格式、支持的元件和监视器、坐标约定及完整示例。详细的使用说明和参考资料请以文档网站为准；README 仅保留项目概览。

## 安装

PASS 当前从源码目录安装，需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/changmx/PASS.git
cd PASS
python -m pip install --editable .
```

如需使用 GPU 追踪，可安装可选 CUDA 依赖：

```bash
python -m pip install --editable ".[cuda]"
```

CUDA 后端还需要 CUDA 工具包和兼容的 GPU；CPU 追踪不依赖 CUDA。

## 图形界面

PASS 提供可选的跨平台图形界面，包含独立的配置、运行和绘图页面。安装方式：

```bash
python -m pip install --editable ".[gui]"
```

随后可在仓库目录或已安装的环境中启动：

```bash
python -m PASS.gui
# 或
pass-gui
```

界面支持独立 JSON，也支持将多份输入 JSON、源文件、依赖和生成设置保存在单个 `.passproj` 项目中。文件菜单区分 JSON 保存、项目保存和导出。项目内容可以直接查看，参数或命令可连同依赖一起复制。紧凑界面提供深色/浅色/跟随系统主题、全部展开的属性字段，以及可拖动列宽和右键选择可选列的执行序列。

运行页面选择一份或两份输入，先生成固定输入快照，再用独立进程启动 PASS。导出的可运行输入包包含依赖和 `run.py` 启动脚本。绘图页面可加载 CSV/TFS，选择 X/Y 列并缩放。打包、源文件复用和输出位置见[项目流程说明](docs/source/zh/project_files.rst)。

**校验** 一键检查完整输入、全部序列模块、命令间依赖及输入 TFS 内容，并提供可筛选、定位和导出的报告。错误阻止运行，警告保留显示。初始化前也执行同一套检测；无 Qt 环境可运行 `python -m PASS.validation beam.json --report validation-report.json`。规则与范围见[输入检测说明](docs/source/zh/input_validation.rst)。

## 主要功能

- 在加速器束中进行六维粒子追踪；
- 支持逐元件追踪和基于 Twiss 的追踪流程；
- 可配置的注入过程和多束团粒子分布；
- 高频腔、磁铁、集体效应接口和束流监视器；
- 用于生成 JSON 输入文件及分析 TFS/CSV 输出的 Python 工具；
- CPU 以及可选的 CUDA 执行路径。

## 开发

在 VS Code 选中的 Python 环境中安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

用 VS Code 打开仓库根目录，并启用推荐的 YAPF 扩展。项目设置会在保存 Python
文件时自动格式化，新文件也适用。统一规则来自 `pyproject.toml`：YAPF 0.43.0，
基于 `pep8`，四空格缩进，行宽 150。不使用编辑器时，可以在仓库根目录运行：

```bash
python -m yapf --style pyproject.toml --in-place path/to/file.py
```

格式化器只负责排版；命名和注释约定保存在 `AGENTS.md`，仍需在开发和审查时遵循。

YAPF 不会格式化 Python 字符串中的 CUDA。包含 `__global__` 或 `__device__`
函数的三引号字符串，可使用项目工具处理：

```bash
python tools/format_cuda.py --check
python tools/format_cuda.py --write PASS/commands/solver/pic.py
```

不指定路径时扫描 `PASS/`，包括以后新增的 Python 文件。默认只检查，只有
`--write` 才修改文件。工具使用 `.clang-format` 中的规则，需要 clang-format
23.x（已用 23.1.0 验证）。它先搜索 `PATH`，再搜索标准 VS Code C/C++ 扩展目录；
其他安装位置可通过 `--clang-format /path/to/clang-format` 或环境变量
`CLANG_FORMAT` 指定。格式化不需要 CUDA 运行环境。

写入前会检查全部选中文件，核对 C++ 词法标记、预处理指令和字符串外的 Python
语法保持不变。动态 f-string 和单行片段会列出供人工检查；不含 CUDA 限定符的
字符串不在扫描范围内。退出码 `0` 表示无需修改或写入成功，`1` 表示检查发现
格式差异，`2` 表示工具或验证出错。检查成功不代表已验证列出的人工检查片段。

`tests/` 目录仅在本地维护，不纳入 Git 版本控制，新克隆的仓库不包含测试套件。
如果本地已有测试文件，以可编辑模式安装项目后，在仓库根目录运行测试：

```bash
python -m pytest
```

默认发现范围为 `tests/unit` 和 `tests/integration`，包含完整的空间电荷仿真。
分类批量运行、单例调用、保存场的分析复用和显式本地 Codex 回归命令，见
本地空间电荷测试指南 `tests/integration/space_charge/README.md`。

空间电荷通过 `Method` / `Solver` 支持 `pic`、`frozen` 和 `quasi-frozen` 跟踪。
运行 `python -m tests.integration.space_charge analytic` 可验证解析场、粒子 kick
与参数演化，并生成对照图。

欢迎通过 [GitHub issue 跟踪器](https://github.com/changmx/PASS/issues) 报告问题或提出功能建议。提交代码时请在适当情况下同时补充测试和文档。

## 许可证

PASS 使用 [Apache License 2.0](LICENSE) 发布。
