<h1 align="center">
  <img src="docs/assets/images/jimmytorch-tensor-prism.svg" width="240" alt="JimmyTorch 棱镜标志，代表数据集、模型与实验">
</h1>

<p align="center">
  一个紧凑的 PyTorch 科研框架，把数据集、模型和训练循环连接成可复现、可观察的实验。
</p>

<p align="center">
  <a href="docs/index.html"><img alt="文档" src="https://img.shields.io/badge/docs-HTML-54e0b1?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-framework-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <a href="LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-f5c76a?style=flat-square"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.cn.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/assets/images/architecture.svg" alt="JimmyTorch 架构：数据集经过模型和训练流程，形成持久化实验记录">
</p>

## 从科研代码到实验依据

JimmyTorch 通过少量明确的接口组织完整科研流程，而不是用庞大的训练平台隐藏研究代码：

| 构建 | 运行 | 观察 |
|---|---|---|
| 字典式数据集接口、可复用模型组件 | 代码化配置、混合精度、评估、调度与检查点 | SQLite 实验记录、实时进度、跨实验对比、图像、笔记与模型结构 |

训练循环不依赖浏览器服务。实验数据首先保存在本地，实时状态则通过异步、尽力而为的方式上报。

## 灵活的实验组合空间

<p align="center">
  <img src="docs/assets/images/experiment-space.svg" alt="JimmyTorch 实验组合空间：每个数据集都可与每个模型组合，每个组合支持多次运行，并可形成多个实验">
</p>

数据集和模型是可以自由复用的维度，而不是固定的一对一关系。每种组合都可以采用不同配置反复运行，并作为可对比的实验结果保留下来。

## Experiment Manager

<p align="center">
  <img src="docs/assets/images/experiment-manager.png" alt="经过脱敏的 JimmyTorch Experiment Manager，展示实时训练状态与多实验对比">
</p>

一个独立服务即可同时查看当前训练和历史实验，包括资源状态、运行时学习率控制、实验树、标量曲线对比、图像、摘要、笔记和模型结构视图。

## 快速开始

```bash
git clone https://github.com/JinmingWang/JimmyTorch.git
cd JimmyTorch

# 可选：在一个终端启动监控服务。
python -m Training.ExperimentManager

# 在另一个终端运行自带的 MNIST 示例。
python main.py
```

当前仓库直接从源码使用，尚未提供固定版本的依赖清单。请先为当前平台安装 PyTorch，再参考英文的[入门指南](docs/getting-started.html)安装运行时和可选依赖。

## 项目结构

```text
Datasets/                     批处理接口与数据预处理工具
Models/                       模型基类、网络层、损失和注意力模块
DiffusionModels/              DDPM 与 DDIM 过程
Training/ExperimentManager/   本地记录、实时客户端、服务端与可视化界面
JimmyExperiment.py            组件构建与实验生命周期
JimmyTrainer.py               训练、评估、调度与检查点
SKILL/jimmytorch/              面向 LLM 的项目指南
docs/                          面向开发者的英文文档
```

## 文档

- [架构概览](docs/index.html)
- [入门指南](docs/getting-started.html)
- [数据集开发](docs/datasets.html)
- [模型开发](docs/models.html)
- [训练与评估](docs/training.html)
- [Experiment Manager](docs/experiment-manager.html)
- [专项工具](docs/utilities.html)
- [API 参考](docs/api.html)

## 项目状态

JimmyTorch 是一个随着实际研究持续演进的个人科研框架。在其他项目中使用时，请结合当前源码和文档进行确认。

项目采用 [Apache License 2.0](LICENSE) 许可证。
