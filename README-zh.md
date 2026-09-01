## 🌐 语言
[**中文**](README-zh.md) | [English](README.md)

# PASS（Particle Accelerator Simulation Studio）

[![文档](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://changmx.github.io/PASS/) [![许可证](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/changmx/PASS)](https://github.com/changmx/PASS/releases/latest)

PASS 是面向粒子加速器的多功能模拟平台，支持 Windows 和 Linux，提供 CPU 与 NVIDIA GPU 两种执行后端，旨在为束流动力学研究提供高性能、可扩展且可复现的六维粒子追踪与分析工具。

## 文档

完整文档发布在 [changmx.github.io/PASS](https://changmx.github.io/PASS/)，其中提供[中文文档](https://changmx.github.io/PASS/zh/)和[English documentation](https://changmx.github.io/PASS/en/)。文档网站包含物理模型、输入参数格式、支持的元件和监视器、坐标约定及完整示例。详细的使用说明和参考资料请以文档网站为准；README 仅保留项目概览。

## 安装

PASS 当前从源码目录安装，需要 Python 3.10 或更高版本。

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

## 主要功能

- 在加速器束中进行六维粒子追踪；
- 支持逐元件追踪和基于 Twiss 的追踪流程；
- 可配置的注入过程和多束团粒子分布；
- 高频腔、磁铁、集体效应接口和束流监视器；
- 用于生成 JSON 输入文件及分析 TFS/CSV 输出的 Python 工具；
- CPU 以及可选的 CUDA 执行路径。

## 开发

以可编辑模式安装项目后，在仓库根目录运行测试：

```bash
python -m pytest
```

欢迎通过 [GitHub issue 跟踪器](https://github.com/changmx/PASS/issues) 报告问题或提出功能建议。提交代码时请在适当情况下同时补充测试和文档。

## 许可证

PASS 使用 [Apache License 2.0](LICENSE) 发布。
