# PASS

[English](README.md) | [中文](README-zh.md)

PASS（Particle Accelerator Simulation Studio）是用于六维粒子跟踪与束流动力学分析的 Python 程序，支持基于元件或 Twiss 传输映射的跟踪、束流注入、射频系统、空间电荷、尾场及束流诊断，可使用 CPU 或 NVIDIA GPU 执行计算。

[使用手册](https://changmx.github.io/PASS/zh/index.html) · [示例](example) · [版本发布](https://github.com/changmx/PASS/releases) · [问题反馈](https://github.com/changmx/PASS/issues)

## 安装

需要 **Python 3.11 或更高版本**。默认安装包含 CPU 计算功能和 GUI。
`cpu` 选项明确表达 CPU 用途，安装内容与默认相同；`gpu` 选项增加 NVIDIA GPU 支持，同时保留 CPU 和 GUI 功能。
所有配置均安装 GUI 依赖（包括 Qt）。命令行 CPU 运行不加载 Qt，也不要求图形显示环境或 CUDA。

### 从 PyPI 安装

**默认安装（CPU + GUI）：**

```bash
python -m pip install pass-sim
```

**显式选择 CPU（同样包含 GUI）：**

```bash
python -m pip install "pass-sim[cpu]"
```

**GPU 安装（CPU + GPU + GUI）：**

```bash
python -m pip install "pass-sim[gpu]"
```

`python -m pip install "pass-sim[cpu,gpu]"` 与 GPU 安装等价。
不再提供原来的 `gui`、`cuda` 和 `conversion` 可选依赖入口，请使用上述命令。

安装后执行 `pass-run --version` 查看已安装的 PASS 版本，或使用 `pass-gui` 启动界面。

### 从 GitHub 安装

先克隆源码并进入仓库目录：

```bash
git clone https://github.com/changmx/PASS.git
cd PASS
```

也可以打开 [GitHub 仓库](https://github.com/changmx/PASS)，选择 **Code → Download ZIP** 下载并解压，然后进入包含 `pyproject.toml` 的目录。
在该目录中选择以下一种配置安装：

**默认安装（CPU + GUI）：**

```bash
python -m pip install --editable .
```

**显式选择 CPU（同样包含 GUI）：**

```bash
python -m pip install --editable ".[cpu]"
```

**GPU 安装（CPU + GPU + GUI）：**

```bash
python -m pip install --editable ".[gpu]"
```

`python -m pip install --editable ".[cpu,gpu]"` 与 GPU 安装等价。
可编辑安装直接使用这份源码目录中的代码，修改 Python 源码后会在后续运行中生效。
安装后执行 `pass-run --version` 查看已安装的 PASS 版本，或使用 `pass-gui` 启动界面。

两种安装方式均支持 `python -m PASS --version` 查看版本信息。执行 `pass-run --help` 或 `pass-run -h` 可查看输入方式、选项、路径规则和示例。

### GPU 环境要求与后端选择

安装 GPU 依赖不会自动选择模拟后端。请通过输入配置中的 `backend (gpu/cpu)` 选择 `cpu` 或 `gpu`。
GPU 运行需要兼容的 NVIDIA GPU、驱动和 CUDA 环境；当前 GPU 依赖面向 CUDA 13.x。
GPU 选项不安装 NVIDIA 驱动，也不保证安装完整的 CUDA Toolkit。
在支持的平台上，没有 GPU 也可能成功安装 GPU 软件包，但安装成功不代表 GPU 可用。
没有 CUDA 时仍可使用 CPU 计算和 GUI。显式选择 GPU 后，如果 CuPy 不可用、设备查询失败或设备初始化失败，PASS 会给出错误提示，不会静默回退到 CPU。
具体 GPU 功能首次使用时，还可能出现 CUDA 库加载或内核编译错误。

## 运行模拟

### 示例工程

以下示例假定 `/home/myprojpath` 中已有有效的 PASS 输入 `beam0.json`，并保存了下面的工程描述：

```text
/home/myprojpath/
├── myproject.passproj
├── beam0.json
├── quadrupole_ramp.tfs    # 可选：beam0.json 引用它时才需要
└── output/               # 运行模拟时创建
```

`myproject.passproj` 是普通 JSON 文本。单束流工程可以写成：

```json
{
  "format": "pass-project",
  "format_version": 3,
  "name": "我的模拟",
  "configs": ["beam0.json"],
  "active_config": "beam0.json",
  "run": {
    "beam0": "beam0.json",
    "beam1": null,
    "output_directory": "output"
  }
}
```

工程描述中的路径相对于 `/home/myprojpath`；`beam0.json` 内部的文件引用相对于该 JSON 所在目录。例如，`"Ramping file": "quadrupole_ramp.tfs"` 指向 `/home/myprojpath/quadrupole_ramp.tfs`。不需要建立 `configs/` 文件夹。实际使用时，将 `/home/myprojpath` 替换为你的工程路径。

### 命令行执行

支持 **三种输入方式**：单份 JSON 运行单束流、两份 JSON 运行双束流，以及按工程保存的选择运行。工程可以通过 `.passproj` 文件或所在文件夹指定。命令行执行不需要 GUI 或 Qt。

对于上述目录，以下每条命令都可以在 **任意工作目录** 执行：

```bash
# 直接运行 beam0.json，使用该 JSON 内的设置。
pass-run /home/myprojpath/beam0.json

# 使用 myproject.passproj 保存的束流选择及输出设置运行。
pass-run /home/myprojpath/myproject.passproj

# 打开该目录中的工程描述，再按工程保存的选择运行。
pass-run /home/myprojpath
```

指定文件夹时，PASS 优先读取其中的 `project.passproj`；若不存在，则读取该目录下唯一的 `.passproj` 文件。因此，上述示例会选中 `myproject.passproj`。如果没有 `project.passproj`，但存在多个 `.passproj` 文件，必须明确指定要运行的工程文件。程序不会递归搜索子目录。

先进入工程目录后，也可以使用等价的相对路径命令：

```bash
cd /home/myprojpath
pass-run beam0.json
pass-run myproject.passproj
pass-run .
```

这些命令是不同的启动选择，每次运行任选一条。工程根据 `run.beam0` 和可选的 `run.beam1` 选择输入；未设置 Beam 0 时使用 `active_config`，不会运行 `configs` 中的全部配置。跟踪前会自动校验并归档所选输入及其依赖文件。

命名选项提供同样功能的另一种写法：

```bash
pass-run --beam0 /home/myprojpath/beam0.json
pass-run --passproj /home/myprojpath/myproject.passproj
pass-run --passproj /home/myprojpath
```

如需 **双束流执行**，先准备第二份有效输入 `/home/myprojpath/beam1.json`，然后选择以下等价命令中的一条：

```bash
pass-run /home/myprojpath/beam0.json /home/myprojpath/beam1.json
pass-run --beam0 /home/myprojpath/beam0.json --beam1 /home/myprojpath/beam1.json
```

若要通过工程运行双束流，将 `"beam1.json"` 加入工程的 `configs` 列表，并把 `run.beam1` 改成 `"beam1.json"`，再执行 `pass-run /home/myprojpath/myproject.passproj`。工程自带束流选择，不能再附加第二个输入参数。每条命令选择位置参数或命名输入选项中的一种，不能混用；`--beam1` 必须与 `--beam0` 一起使用，`--passproj` 不能与两个束流选项中的任意一个组合使用。

`pass-run` 和 `python -m PASS` 是 **同一套执行逻辑的两个启动入口**，接受相同参数。需要明确使用某个 Python 环境时，可以用模块入口：

```bash
python -m PASS /home/myprojpath/beam0.json
python -m PASS /home/myprojpath/myproject.passproj
python -m PASS /home/myprojpath
```

命令行中的相对路径以终端当前工作目录为基准，路径包含空格时需要加引号。执行 `pass-run --help` 或 `python -m PASS --help` 可查看全部选项。

### 输出与编辑

按上述工程描述运行时，无论从哪个目录启动，结果默认保存在 `/home/myprojpath/output/<YYYY_MM_DD_HH_MM>/` 下。直接运行 `beam0.json` 时使用它自己的 `Output directory` 设置；如果缺省或值为 `default`（不区分大小写），输出根目录同样为 `/home/myprojpath/output`。

| 启动方式 | 输出根目录 | 根目录下的默认结果位置 |
| --- | --- | --- |
| 通过 `pass-run`、`python -m PASS` 或 Python API 运行 JSON | 第一份 JSON 的 `Output directory`，相对于该 JSON 所在目录；缺省或值为 `default`（不区分大小写）时，使用该 JSON 旁的 `output` | `<YYYY_MM_DD_HH_MM>/` |
| 通过命令行指定工程文件或工程目录 | 工程的 `run.output_directory`，未设置时为 `output`；相对路径以工程描述所在目录为基准 | `<YYYY_MM_DD_HH_MM>/` |
| GUI | **运行** 页的 **输出根目录**，初始值为 `output`；相对路径以 JSON／项目所在目录为基准，未保存项目使用当前工作目录 | `<YYYY_MM_DD_HH_MM>/` |

`--output DIR` 可以覆盖任意输入方式的输出根目录，且仅对本次运行有效。使用绝对路径时，可以从任意工作目录执行：

```bash
pass-run /home/myprojpath/myproject.passproj --output /home/myprojpath/results
```

此时输出根目录为 `/home/myprojpath/results`。相对的覆盖路径以终端当前工作目录为基准，例如：

```bash
cd /home/myprojpath
pass-run myproject.passproj --output "results/test run"
```

未设置自定义输出名称时，这会将结果写入 `/home/myprojpath/results/test run/<YYYY_MM_DD_HH_MM>/`。`--output` 优先于 JSON／工程中的设置，不会修改它们的文件内容。`python -m PASS` 支持相同选项。

若要用任务名称代替自动时间戳作为运行文件夹名，可在 Beam 0 JSON 中设置 `"Task name": "capture_sc"`（Python 对应 `MainConfig(task_name="capture_sc")`），设置工程的 `run.task_name`，或修改 **运行** 页的 **任务名称**。该字段初始直接显示实际值 `default`，已有自定义名称会保留；`Output directory` / `output_dir` 仍用于选择父级输出根目录。命令行覆盖方式为：

```bash
pass-run myproject.passproj --output results --task-name capture_sc
```

结果将保存在 `results/capture_sc/`，内部子目录和结果文件原有的 `HHMM_SS` 时间标记保持不变。任务名称省略、为 null、留空或为 `default` 时，使用精确到分钟的单层文件夹，例如 `output/2026_10_30_18_39/`。自动或自定义名称重名时均追加不补零的 `_1`、`_2`、`_3` 等后缀，例如 `2026_10_30_18_39_1` 或 `capture_sc_1`；`default` 在去除首尾空白后不区分大小写，例如 `" DEFAULT "` 具有相同效果。不会自动使用 Beam 0 文件名。存在 `run.task_name` 时，它优先于 Beam 0 字段；`--task-name` 优先于两者。名称只能是单个文件夹名，不能是路径。Python API 对应 `main(..., task_name="capture_sc")`；显式传入空字符串或 `default` 可覆盖已保存的自定义名称并恢复单层时间戳布局，例如 `--task-name default` 或 `main(..., task_name="default")`。

每次普通运行都在本次结果目录的 `input/` 子目录保存输入快照，其中包含原始配置值、执行 JSON、复制的依赖文件和 `run.json` 运行记录。JSON、工程和 GUI 运行共用这些布局规则。已有结果目录保持不变；旧输入或旧快照重新运行时，按新的单层布局创建新目录。完整规则见[输入指南](https://changmx.github.io/PASS/zh/input_generation.html)和[工程文件](https://changmx.github.io/PASS/zh/project_files.html)。

可以用文本编辑器直接修改 `myproject.passproj`、`beam0.json` 和文本依赖文件。在 GUI 中，外部修改后应先使用 **文件 → 重新加载项目**，再保存或运行；外部修改检查会避免覆盖修改或使用过期输入。移动或分享工程时，需要携带整个目录及其引用文件，单个 `.passproj` 描述文件并不包含完整工程。**导出可运行输入包** 可以将所选输入及依赖打包分享。

### 自动绘图

在束流输入中设置 `"Is plot figure": true`，即可在跟踪正常结束后自动绘图。每份 StatMonitor 历史数据生成九联图，每份分布快照生成六联图，每份 ParticleMonitor 文件生成十二联图，并使用全部已记录粒子及有效样本。图片保存在本次运行的 `plot/` 目录中。

也可以直接对已有结果目录绘图，无须重新运行模拟：

```bash
python -m PASS.plot output/2026_10_30_18_39
python -m PASS.plot output/2026_10_30_18_39 --format png pdf
```

默认输出 300 dpi PNG。再次绘图时使用未占用的文件名，保留已有图片。布局、单位、选项及绘图记录见[自动绘图](https://changmx.github.io/PASS/zh/monitor/plotting.html)。

### 图形界面

安装 `gui` 组件后，使用 `pass-gui` 或 `python -m PASS.gui` 启动界面。对于上述示例，可以通过 **文件 → 打开项目** 选择 `/home/myprojpath/myproject.passproj`，或通过 **文件 → 打开项目文件夹** 选择 `/home/myprojpath`。如果只编辑单份束流输入，使用 **文件 → 打开 JSON** 选择 `/home/myprojpath/beam0.json`。

应用参数修改，执行 **校验**，然后在 **运行** 页面启动模拟。计算结果可在 **绘图** 页面查看。具体操作见[图形界面](https://changmx.github.io/PASS/zh/gui.html)和[工程文件](https://changmx.github.io/PASS/zh/project_files.html)。

### Python 工作流

[输入配置指南](https://changmx.github.io/PASS/zh/input_generation.html)说明输入生成、校验、执行和输出。示例脚本位于源码仓库中。若通过 PyPI 安装，先下载或克隆仓库以获取示例，然后从仓库根目录运行分布生成示例：

```bash
cd example/01_generate_distribution
python generate_input.py --case longi-gaussian
pass-run --beam0 beam0_longi_gaussian.json
python analyze_results.py --case longi-gaussian
```

生成脚本写入所选算例的输入文件，`pass-run` 将结果保存到该示例的 `output/` 目录，分析脚本读取该算例最近一次运行的结果。分布和 RF 示例还提供 `run_simulation.py --case all`，通过同一 CLI 依次运行各自的预定义算例。参数、其他算例及输出文件见[示例说明](example/01_generate_distribution/README.md)。

已有输入也可以先单独校验：

```bash
python -m PASS.validation /home/myprojpath/beam0.json --report /home/myprojpath/validation-report.json
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

物理约定、文档及格式要求见 [AGENTS.md](AGENTS.md)。安装 `.[dev]` 后，使用 YAPF 0.43.0 和 `pyproject.toml` 格式化 Python；在源码仓库中运行 `python PASS/tool/format_cuda.py --check` 检查嵌入的 CUDA 代码。`tests/` 在本地维护且不纳入 Git，重新克隆的仓库不包含测试集。

PASS 使用 [Apache License 2.0](LICENSE) 许可证。
