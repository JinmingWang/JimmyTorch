# JimmyTorch
我个人常用的深度学习实验工具，基于PyTorch实现，包含很多数据集、模型、训练、可视化等相关函数。因为是个人用的，因此内容不完善，用到什么东西就补充什么。

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![cn](https://img.shields.io/badge/lang-cn-red.svg)](README.cn.md)

## 使用方法

> ### 数据集
> 定义自己的数据集类，继承自 `JimmyDataset`
> 1. `__init__` 方法完成数据加载和转换为张量，并设置 `self.n_samples`。
> 2. `__getitem__` 方法返回一个包含数据的字典。

> ### 模型
> 定义自己的神经网络类，继承自 `JimmyModel`
> 1. `__init__` 方法与普通 PyTorch 模型类似，另外必须定义 `self.train_loss_names` 和 `self.eval_loss_names`。
> 2. `forward` 方法与普通 PyTorch 模型类似。
> 3. `trainStep` 方法处理模型前向传播、损失计算、损失反向传播、混合精度训练，并返回损失字典和输出字典。
> 4. `evalStep` 方法处理模型前向传播、损失计算，并返回损失字典和输出字典。

> ### 训练和实验
> 请参考 `JimmyTrainer.py` 中的训练流程，以及 `JimmyExperiment.py` 中的实验流程。最后，`main.py` 是入口。

## 模块介绍
> ### 数据集
- `JimmyDataset.py`: 数据集基类，可以通过继承此类定义自己的数据集。这个类更像是 PyTorch 中 Dataset + DataLoader 的结合。与 PyTorch 的数据加载过程不同，这个类假设数据已经处理成张量并加载到 GPU，因此 `__getitem__` 方法可以直接使用范围索引访问数据，这比多线程数据加载更高效。`__getitem__` 方法返回一个包含数据的字典，旨在保持不同模型和训练过程的一致性。
- `DatasetUtils.py`: 定义了 `DEVICE`，并导入了一些有用的数据集处理函数。
- `MultiThreadLoader.py`: 如果数据处理包含一些重的 CPU 计算，可以使用此类进行多线程数据加载。
- `TrajectoryUtils.py`: 我的研究主要是关于 GPS 轨迹的，因此此文件包含一些轨迹处理的实用函数，如轨迹插值、计算距离、裁剪、填充、翻转等。
- `SequenceUtils.py`: TODO，包含一些序列数据处理的实用函数。
- `TODO`: 未来可能会添加一些其他类型数据的实用函数，如图像、文本、图等。

> ### 模型
- `JimmyModel.py`: 模型基类，可以通过继承此类定义自己的模型。此类集成了训练、损失计算、优化和评估过程，并支持混合精度训练和 torch.compile。`train_loss_names` 和 `eval_loss_names` 用于告诉模型在训练和评估过程中将报告的损失和指标类型，这对于日志记录和可视化很有用。然后，`trainStep` 和 `evalStep` 方法用于处理单个训练或评估步骤，返回一个包含所有损失的字典和另一个包含所有输出的字典。所有损失的字典必须与 `train_loss_names` 和 `eval_loss_names` 匹配。此模块旨在保持不同模型的独特性，以便训练和评估过程可以在不同模型之间更具可重用性和一致性。
- `Basics.py`: 包含一些非常基础的模块，如归一化 + 激活 + 卷积的组合、MLP、位置编码、制作patch等。
- `Attentions.py`: 包含一些注意力模块，如多头自注意力（MHSA）、交叉注意力（CrossAttention）和 SE 层。
- `ModelUtils.py`: 包含一些非参数模块，如Transpose、Permute、Reshape、PrintShape、SequentialMultiIO、Rearrange。

> ### 训练和实验
- `ProgressManager.py`: 一个更好的进度可视化工具。
- `MovingAverage.py`: 计算标量的移动平均。
- `TensorBoardManager.py`: TensorBoard 的包装器，支持预注册标签进行日志记录。
- `JimmyTrainer.py`: 一个示例训练器类。
- `JimmyExperiment.py`: 一个示例实验类。
- `DynamicConfig.py`: 支持带有类及其初始化参数的配置，可以调用 `build` 方法创建类实例。