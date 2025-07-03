# JimmyTorch
我个人常用的深度学习实验工具，基于PyTorch实现，包含很多数据集、模型、训练、可视化等相关函数。

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