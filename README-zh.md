# PASS

[English](README.md) | [中文](README-zh.md)

PASS（Particle Accelerator Simulation Studio）是用于六维粒子跟踪与束流动力学分析的 Python 程序，支持基于元件或 Twiss 传输映射的跟踪、束流注入、射频系统、空间电荷、尾场及束流诊断，可使用 CPU 或 NVIDIA GPU 执行计算。

[使用手册](https://changmx.github.io/PASS/zh/index.html) · [示例](example) · [版本发布](https://github.com/changmx/PASS/releases) · [问题反馈](https://github.com/changmx/PASS/issues)

## 安装

需要 **Python 3.11 或更高版本**。先选择 PyPI 或 GitHub，再选择所需功能；每类中的三种配置任选一种。
完整安装包含 CPU、NVIDIA GPU 和 GUI；GPU 功能需要兼容的 NVIDIA GPU、驱动和 CUDA 环境。仅 CPU 和 GUI 安装不需要 CUDA。

### 从 PyPI 安装

**完整安装（CPU + GPU + GUI）：**

```bash
python -m pip install "pass-sim[gui,cuda]"
```

**仅安装 CPU 计算功能（命令行和 Python API）：**

```bash
python -m pip install pass-sim
```

**安装 GUI（包含 CPU 计算功能）：**

```bash
python -m pip install "pass-sim[gui]"
```

选择一种配置安装后，执行以下命令，查看已安装的 PASS 版本信息：

```bash
pass-run --version
```

安装了 GUI 后，使用 `pass-gui` 启动界面。

### 从 GitHub 安装

先克隆源码并进入仓库目录：

```bash
git clone https://github.com/changmx/PASS.git
cd PASS
```

也可以打开 [GitHub 仓库](https://github.com/changmx/PASS)，选择 **Code → Download ZIP** 下载并解压，然后进入包含 `pyproject.toml` 的目录。
在该目录中选择以下一种配置安装：

**完整安装（CPU + GPU + GUI）：**

```bash
python -m pip install --editable ".[gui,cuda]"
```

**仅安装 CPU 计算功能（命令行和 Python API）：**

```bash
python -m pip install --editable .
```

**安装 GUI（包含 CPU 计算功能）：**

```bash
python -m pip install --editable ".[gui]"
```

可编辑安装直接使用这份源码目录中的代码，修改 Python 源码后会在后续运行中生效。
安装后执行以下命令，查看已安装的 PASS 版本信息：

```bash
pass-run --version
```

安装了 GUI 后，使用 `pass-gui` 启动界面。

两种安装方式均支持 `python -m PASS --version` 查看版本信息。执行 `pass-run --help` 或 `pass-run -h` 可查看输入方式、选项、路径规则和示例。

## 运行模拟

**图形界面：**安装 `gui` 组件后启动：

```bash
pass-gui
```

`python -m PASS.gui` 会使用所选 Python 解释器启动同一个界面。

打开或创建输入配置，应用参数修改，执行 **校验**，然后在 **运行** 页面启动模拟。计算结果可在 **绘图** 页面查看。`.passproj` 工程文件保存配置及其依赖的输入文件。具体操作见[图形界面](https://changmx.github.io/PASS/zh/gui.html)和[工程文件](https://changmx.github.io/PASS/zh/project_files.html)。

**命令行：**安装 PASS 后，可直接运行 JSON 输入或已保存项目，无需 GUI 或 Qt：

```bash
# 使用已有 JSON 输入运行单束流。
pass-run path/to/beam0.json
# 使用两份 JSON 输入一起运行双束流。
pass-run path/to/beam0.json path/to/beam1.json
# 运行项目中保存的束流选择。
pass-run path/to/example.passproj
# 也可使用命名选项明确指定输入。
pass-run --beam0 path/to/beam0.json
pass-run --beam0 path/to/beam0.json --beam1 path/to/beam1.json
pass-run --passproj path/to/example.passproj
# 仅为本次运行指定其他输出根目录。
pass-run path/to/beam0.json --output results
pass-run path/to/example.passproj --output "results/project run"
```

命令接收输入文件路径；相对参数路径以终端当前工作目录为基准，路径包含空格时需要加引号。程序根据后缀选择 JSON 或项目读取方式。项目使用已保存的 Beam 0/Beam 1 选择，未保存 Beam 0 时使用当前活动输入，不会运行其中的全部配置，也不能再附加第二个输入参数。跟踪前会自动校验并归档所选输入及其依赖文件。

每条命令选择位置参数或命名输入选项中的一种，不能混用。`--beam0` 和 `--beam1` 接收 JSON 文件，`--beam1` 必须与 `--beam0` 一起使用。`--passproj` 接收一个 `.passproj` 文件，不能与两个束流选项中的任意一个组合使用。

`python -m PASS` 接受相同参数，并使用所选 Python 解释器执行同一套运行逻辑：

```bash
python -m PASS path/to/beam0.json
python -m PASS "path with spaces/example.passproj"
pass-run --help
pass-run -h
```

**输出目录：**通常默认文件夹仍为 `output`，其位置取决于输入和启动方式：

| 启动方式 | 输出根目录 | 根目录下的结果位置 |
| --- | --- | --- |
| 通过 `pass-run`、`python -m PASS` 或 Python API 运行 JSON | 第一份 JSON 的 `Output directory`，相对于该 JSON 所在目录；缺省或值为 `default`（不区分大小写）时，使用该 JSON 旁的 `output` | `<YYYY_MMDD>/<HHMM_SS>/` |
| 通过命令行运行已保存的 `.passproj` | 项目保存的运行输出目录，未设置时为 `output`；相对路径以项目所在目录为基准 | `<YYYY_MMDD>/<HHMM_SS>/` |
| GUI | **运行** 页的输出目录，初始值为 `output`；相对路径以 JSON／项目所在目录为基准，未保存项目使用当前工作目录 | `<YYYY_MMDD>/<HHMM_SS>/` |

`--output DIR` 可覆盖单份 JSON、双份 JSON 和 `.passproj` 的输出根目录，优先于输入配置或项目保存的设置，且仅对本次运行有效；相对路径以启动命令时终端的工作目录为基准。原始 JSON 或项目文件保持不变。`python -m PASS` 支持相同选项。

每次普通运行将输入快照保存在本次结果目录内，即 `<output>/<YYYY_MMDD>/<HHMM_SS>/input/`。其中包含原始配置值、执行 JSON、复制的输入依赖和 `run.json` 运行记录。JSON、项目和 GUI 运行统一使用按日期组织的结果目录，必要时添加后缀以避免复用已有目录。生成的 JSON 仍默认使用 `./output`。缺少 `Output directory` 或将其设为 `default` 的 JSON，现在统一使用第一份 JSON 旁的 `output`，不再使用仓库或安装目录。完整输出规则见[输入指南](https://changmx.github.io/PASS/zh/input_generation.html)和[工程文件](https://changmx.github.io/PASS/zh/project_files.html)。

**查看项目内容：**`.passproj` 是标准 ZIP 容器，可以用支持 ZIP 的压缩软件打开；也可以复制一份，把副本后缀改为 `.zip`，再解压。`manifest.json` 保存项目清单和运行设置，`configs/*.json` 是仿真输入，`assets/` 是依赖文件，`recipes/` 是生成设置。GUI 的 **项目内容** 也支持直接预览。修改后应通过 PASS 保存，以同步更新校验和与依赖索引。

**Python 工作流：**[输入配置指南](https://changmx.github.io/PASS/zh/input_generation.html)说明输入生成、校验、执行和输出。示例脚本位于源码仓库中。若通过 PyPI 安装，先下载或克隆仓库以获取示例，然后从仓库根目录运行分布生成示例：

```bash
cd example/01_generate_distribution
python generate_input.py --case longi-gaussian
pass-run --beam0 beam0_longi_gaussian.json
python analyze_results.py --case longi-gaussian
```

生成脚本写入所选算例的输入文件，`pass-run` 将结果保存到该示例的 `output/` 目录，分析脚本读取该算例最近一次运行的结果。分布和 RF 示例还提供 `run_simulation.py --case all`，通过同一 CLI 依次运行各自的预定义算例。参数、其他算例及输出文件见[示例说明](example/01_generate_distribution/README.md)。

已有输入也可以先单独校验：

```bash
python -m PASS.validation path/to/beam.json --report validation-report.json
```

## 查阅手册

- [坐标约定与注入](https://changmx.github.io/PASS/zh/injection.html)
- [束内散射](https://changmx.github.io/PASS/zh/ibs.html)：CPU/GPU 上的高斯增长率、动力学踢与局部二体碰撞
- [元件](https://changmx.github.io/PASS/zh/element/index.html)与 [Twiss 传输映射](https://changmx.github.io/PASS/zh/twiss.html)
- [空间电荷](https://changmx.github.io/PASS/zh/space_charge.html)、[尾场](https://changmx.github.io/PASS/zh/wake_field.html)与[电子云：静态踢角及预设束流驱动积累](https://changmx.github.io/PASS/zh/electron_cloud.html)
- [监视器与输出格式](https://changmx.github.io/PASS/zh/monitor/index.html)
- [频谱分析与 FMA](https://changmx.github.io/PASS/zh/spectral_analysis.html)：独立 NumPy 函数与 GUI **分析** 页面，支持 CSV/TSV/TXT/DAT、TFS、HDF5、NPY 和 NPZ 输入。

WakeField 文件模型仅接受标准尾场 TFS。CSV/TXT/HEADTAIL 数据需通过 `python -m PASS.tool.wake_conversion` 或 GUI 转换工具中的 **导入尾场…** 转换；单位、物理约定与示例见[尾场指南](https://changmx.github.io/PASS/zh/wake_field.html#wake-tfs-zh)。

安装 `docs` 组件后，在仓库根目录执行以下命令，同时构建中英文文档：

```bash
python -m sphinx -b html -W --keep-going docs/source docs/build/html
```

构建成功后，打开 `docs/build/html/index.html`。

## 参与开发

物理约定、文档及格式要求见 [AGENTS.md](AGENTS.md)。安装 `.[dev]` 后，使用 YAPF 0.43.0 和 `pyproject.toml` 格式化 Python；嵌入的 CUDA 代码通过 `python tools/format_cuda.py --check` 检查。`tests/` 在本地维护且不纳入 Git，重新克隆的仓库不包含测试集。

PASS 使用 [Apache License 2.0](LICENSE) 许可证。
