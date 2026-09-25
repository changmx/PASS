# PASS

[English](README.md) | [中文](README-zh.md)

PASS（Particle Accelerator Simulation Studio）是用于六维粒子跟踪与束流动力学分析的 Python 程序，支持基于元件或 Twiss 传输映射的跟踪、束流注入、射频系统、空间电荷、尾场及束流诊断，可使用 CPU 或 NVIDIA GPU 执行计算。

[使用手册](https://changmx.github.io/PASS/zh/index.html) · [示例](example) · [版本发布](https://github.com/changmx/PASS/releases) · [问题反馈](https://github.com/changmx/PASS/issues)

## 安装

需要 **Python 3.11 或更高版本**，建议使用虚拟环境。从源代码安装：

```bash
git clone https://github.com/changmx/PASS.git
cd PASS
python -m pip install --editable .
```

可选组件可以组合安装，例如 `python -m pip install --editable ".[gui,docs]"`。

| 可选组件 | 用途 |
| --- | --- |
| `gui` | 图形界面的配置、运行、绘图与计算工具 |
| `cuda` | GPU 跟踪；需要兼容的 NVIDIA GPU、驱动和 CUDA 环境 |
| `docs` | 在本地构建中英文手册 |
| `conversion` | 不安装 GUI 时使用 SDDS 格式转换；`gui` 已包含此依赖 |
| `dev` | 项目使用的 Python 格式化工具 |

CPU 跟踪不需要 CUDA。

## 运行模拟

**图形界面：**安装 `gui` 组件后启动：

```bash
python -m PASS.gui
```

打开或创建输入配置，应用参数修改，执行 **校验**，然后在 **运行** 页面启动模拟。计算结果可在 **绘图** 页面查看。`.passproj` 工程文件保存配置及其依赖的输入文件。具体操作见[图形界面](https://changmx.github.io/PASS/zh/gui.html)和[工程文件](https://changmx.github.io/PASS/zh/project_files.html)。

**Python 工作流：**[输入配置指南](https://changmx.github.io/PASS/zh/input_generation.html)说明输入生成、校验、执行和输出。以下命令从仓库根目录运行已有的分布生成示例：

```bash
cd example/01_generate_distribution
python generate_input.py --case longi-gaussian
python run_simulation.py --case longi-gaussian
python analyze_results.py --case longi-gaussian
```

生成脚本写入所选算例的输入文件，运行脚本将结果保存到该示例的 `output/` 目录，分析脚本读取该算例最近一次运行的结果。参数、其他算例及输出文件见[示例说明](example/01_generate_distribution/README.md)。

已有输入也可以先单独校验：

```bash
python -m PASS.validation path/to/beam.json --report validation-report.json
```

## 查阅手册

- [坐标约定与注入](https://changmx.github.io/PASS/zh/injection.html)
- [元件](https://changmx.github.io/PASS/zh/element/index.html)与 [Twiss 传输映射](https://changmx.github.io/PASS/zh/twiss.html)
- [空间电荷](https://changmx.github.io/PASS/zh/space_charge.html)与[尾场](https://changmx.github.io/PASS/zh/wake_field.html)
- [监视器与输出格式](https://changmx.github.io/PASS/zh/monitor/index.html)

安装 `docs` 组件后，在仓库根目录执行以下命令，同时构建中英文文档：

```bash
python -m sphinx -b html -W --keep-going docs/source docs/build/html
```

构建成功后，打开 `docs/build/html/index.html`。

## 参与开发

物理约定、文档及格式要求见 [AGENTS.md](AGENTS.md)。安装 `.[dev]` 后，使用 YAPF 0.43.0 和 `pyproject.toml` 格式化 Python；嵌入的 CUDA 代码通过 `python tools/format_cuda.py --check` 检查。`tests/` 在本地维护且不纳入 Git，重新克隆的仓库不包含测试集。

PASS 使用 [Apache License 2.0](LICENSE) 许可证。
