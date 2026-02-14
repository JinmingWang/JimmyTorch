# JimmyTorch

基于 PyTorch 的个人深度学习框架，集成数据集管理、模型训练、实验编排和可视化。针对轨迹/序列数据的研究工作流优化。

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![cn](https://img.shields.io/badge/lang-cn-red.svg)](README.cn.md)

---

## 1. 数据集 (Dataset)

### 快速使用

```python
from Datasets import JimmyDataset

class MyDataset(JimmyDataset):
    def __init__(self, batch_size: int, shuffle: bool = False):
        super().__init__(batch_size, drop_last=False, shuffle=shuffle)
        
        # 将所有数据加载并预处理为张量
        self.data = torch.randn(1000, 128).to(DEVICE)  # 示例：1000个样本
        self.labels = torch.randint(0, 10, (1000,)).to(DEVICE)
        self.n_samples = len(self.data)  # 必须设置 n_samples
    
    def __getitem__(self, idx):
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]
        
        return {
            'data': self.data[indices],
            'target': self.labels[indices]
        }

# 使用方法
train_set = MyDataset(batch_size=32, shuffle=True)
for batch_dict in train_set:
    data = batch_dict['data']
    target = batch_dict['target']
```

### 目录索引

| 路径 | 类 | 函数 | 描述 |
|------|-------|----------|-------------|
| `Datasets/JimmyDataset.py` | `JimmyDataset` | - | 基础数据集类，结合 Dataset + DataLoader 功能 |
| `Datasets/JimmyDataset.py` | `JimmyDataset` | `__init__(batch_size, drop_last, shuffle)` | 用批处理参数初始化数据集 |
| `Datasets/JimmyDataset.py` | `JimmyDataset` | `__getitem__(idx)` | 返回索引 idx 的批次字典（从1开始） |
| `Datasets/JimmyDataset.py` | `JimmyDataset` | `n_batches` | 根据 n_samples 计算批次数的属性 |
| `Datasets/DatasetUtils.py` | - | `DEVICE` | 全局设备变量 (cuda/cpu) |
| `Datasets/MultiThreadLoader.py` | `MultiThreadLoader` | `__init__(dataset, num_workers)` | CPU密集预处理的多线程数据加载 |
| `Datasets/TrajectoryUtils.py` | - | `computeDistance(trajs: BatchTraj \| Traj)` | 通过求和相邻点距离计算轨迹总距离 |
| `Datasets/TrajectoryUtils.py` | - | `cropPadTraj(traj, target_len, pad_value)` | 裁剪或填充轨迹至目标长度 |
| `Datasets/TrajectoryUtils.py` | - | `flipTrajWestEast(trajs)` | 通过取反经度值水平翻转轨迹 |
| `Datasets/TrajectoryUtils.py` | - | `centerTraj(trajs)` | 将轨迹中心化至原点 (0, 0) |
| `Datasets/TrajectoryUtils.py` | - | `zScoreTraj(trajs)` | 标准化轨迹至零均值单位方差 |
| `Datasets/TrajectoryUtils.py` | - | `rotateTraj(trajs, angles)` | 按给定角度（度）旋转轨迹 |
| `Datasets/TrajectoryUtils.py` | - | `interpTraj(trajs, num_points, mode)` | 将轨迹插值至指定点数 |
| `Datasets/TrajectoryUtils.py` | - | `geometricDistance(pred_points, gt_points, reduction)` | 计算GPS点间的Haversine距离（米） |
| `Datasets/TrajectoryUtils.py` | - | `computeJSD(dataset1, dataset2, num_bins)` | 计算轨迹分布的 Jensen-Shannon 散度 |
| `Datasets/TrajectoryUtils.py` | - | `plotTraj(ax, trajs, traj_lengths, color)` | 在 matplotlib 轴上绘制轨迹 |
| `Datasets/SequenceUtils.py` | - | - | (待完成) 序列数据工具函数 |
| `Datasets/MNISTDataset.py` | `MNISTSampleDataset` | - | MNIST 数据集示例实现 |

### 注意事项

- **预加载理念**：与 PyTorch 的 DataLoader 不同，`JimmyDataset` 假设所有数据在 `__init__` 时已预处理为张量并加载到 GPU。当数据可装入内存时，这消除了多线程加载开销。
- **字典返回**：`__getitem__` 必须返回字典。这统一了不同数据集和模型的接口。
- **索引约定**：`__getitem__(idx)` 使用从1开始的批次索引（idx=1 是第一个批次）。内部实现处理0索引转换。
- **轨迹类型**：`Traj = FT32[Tensor, "L 2"]`（单个轨迹），`BatchTraj = FT32[Tensor, "B L 2"]`（批量轨迹）。
- **MultiThreadLoader**：仅在预处理包含不可避免的CPU密集操作时使用（如文件I/O、复杂变换）。

---

## 2. 模型 (Model)

### 快速使用

```python
from Models import JimmyModel
import torch.nn as nn

class MyModel(JimmyModel):
    def __init__(self):
        super().__init__(
            optimizer_cls=torch.optim.AdamW,
            optimizer_args={"lr": 1e-4, "weight_decay": 0.01},
            mixed_precision=True,
            compile_model=False,
            clip_grad=1.0
        )
        
        # 定义用于日志记录的损失名称
        self.train_loss_names = ["Train/Loss", "Train/Acc"]
        self.eval_loss_names = ["Eval/Loss", "Eval/Acc"]
        
        # 定义模型架构
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def trainStep(self, data_dict):
        with getAutoCast(data_dict['data'], self.mixed_precision):
            output = self(data_dict['data'])
            loss = self.loss_fn(output, data_dict['target'])
            acc = (output.argmax(dim=-1) == data_dict['target']).float().mean()
        
        self.backwardOptimize(loss)
        
        return {"Train/Loss": loss.item(), "Train/Acc": acc.item()}, \
               {"output": output.detach()}
    
    def evalStep(self, data_dict):
        with torch.no_grad():
            output = self(data_dict['data'])
            loss = self.loss_fn(output, data_dict['target']).item()
            acc = (output.argmax(dim=-1) == data_dict['target']).float().mean().item()
        
        return {"Eval/Loss": loss, "Eval/Acc": acc}, \
               {"output": output.detach()}

# 使用方法
model = MyModel().to(DEVICE)
model.initialize()  # 创建优化器
```

### 目录索引

| 路径 | 类 | 函数 | 描述 |
|------|-------|----------|-------------|
| `Models/JimmyModel.py` | `JimmyModel` | `__init__(optimizer_cls, optimizer_args, mixed_precision, compile_model, clip_grad)` | 使用训练配置初始化模型 |
| `Models/JimmyModel.py` | `JimmyModel` | `initialize()` | 创建优化器，可选编译模型 |
| `Models/JimmyModel.py` | `JimmyModel` | `trainStep(data_dict)` | 执行一个训练步骤：前向、反向、优化。返回 loss_dict 和 output_dict |
| `Models/JimmyModel.py` | `JimmyModel` | `evalStep(data_dict)` | 执行一个评估步骤。返回 loss_dict 和 output_dict |
| `Models/JimmyModel.py` | `JimmyModel` | `testStep(data_dict)` | 执行一个测试步骤（默认：调用 evalStep） |
| `Models/JimmyModel.py` | `JimmyModel` | `backwardOptimize(loss)` | 使用混合精度和梯度裁剪执行反向传播 |
| `Models/JimmyModel.py` | `JimmyModel` | `saveTo(path)` | 保存模型 state_dict 到路径 |
| `Models/JimmyModel.py` | `JimmyModel` | `loadFrom(path)` | 从路径加载模型 state_dict，处理尺寸不匹配 |
| `Models/JimmyModel.py` | `JimmyModel` | `lr` | 返回当前学习率的属性 |
| `Models/Basics.py` | `Conv1DBnReLU` | `__init__(c_in, c_out, k, s, p, d, g)` | 1D 卷积 → 批归一化 → ReLU 块 |
| `Models/Basics.py` | `Conv2DBnGELU` | `__init__(c_in, c_out, k, s, p, d, g)` | 2D 卷积 → 批归一化 → GELU 块 |
| `Models/Basics.py` | `BnReLUConv1D` | `__init__(c_in, c_out, k, s, p, d, g)` | 批归一化 → ReLU → 1D 卷积块（预激活） |
| `Models/Basics.py` | `FCLayers` / `MLP` | `__init__(channel_list, act, final_act)` | 可配置激活的多层感知机 |
| `Models/Basics.py` | `PosEncoderSinusoidal` | `__init__(dim, max_len, merge_mode, d_pe)` | 正弦位置编码（加法/拼接） |
| `Models/Basics.py` | `PosEncoderLearned` | `__init__(dim, max_len, merge_mode)` | 可学习位置编码 |
| `Models/Basics.py` | `PosEncoderRotary` | `__init__(dim, max_len, base)` | 旋转位置编码 (RoPE) |
| `Models/Basics.py` | `PatchMaker1D` | `__init__(patch_size, stride, patch_as_vector)` | 从序列提取1D块 |
| `Models/Basics.py` | `PatchMaker2D` | `__init__(patch_size, stride, patch_as_vector, flatten)` | 从图像提取2D块 |
| `Models/Attentions.py` | `MHSA` | `__init__(d_in, num_heads, dropout)` | 带QKV投影的多头自注意力 |
| `Models/Attentions.py` | `CrossAttention` | `__init__(d_in, num_heads, dropout)` | 查询和键值对的交叉注意力 |
| `Models/Attentions.py` | `SELayer1D` | `__init__(c_in, reduction)` | 1D特征的挤压-激励层 |
| `Models/Attentions.py` | `SELayer2D` | `__init__(c_in, reduction)` | 2D特征的挤压-激励层 |
| `Models/ModelUtils.py` | `Transpose` | `__init__(dim1, dim2)` | 用于 nn.Sequential 的转置层 |
| `Models/ModelUtils.py` | `Permute` | `__init__(*dims)` | 用于 nn.Sequential 的排列层 |
| `Models/ModelUtils.py` | `Reshape` | `__init__(*shape)` | 用于 nn.Sequential 的重塑层 |
| `Models/ModelUtils.py` | `PrintShape` | `__init__(name)` | 用于打印张量形状的调试层 |
| `Models/ModelUtils.py` | `SequentialMultiIO` | `forward(*dynamic_inputs, **static_inputs)` | 支持多输入/输出的序列模块 |
| `Models/LossFunctions.py` | `MaskedLoss` | `__init__(base_loss)` | 仅对掩码元素应用损失 |
| `Models/LossFunctions.py` | `SequentialLossWithLength` | `__init__(base_loss)` | 对变长序列应用损失 |
| `Models/LossFunctions.py` | `RMSE` | - | 均方根误差损失 |
| `DiffusionModels/DDPM.py` | `DDPM` | - | 去噪扩散概率模型实现 |
| `DiffusionModels/DDIM.py` | `DDIM` | - | 去噪扩散隐式模型实现 |

### 注意事项

- **损失名称**：必须定义 `train_loss_names` 和 `eval_loss_names`。它们控制记录和可视化的指标。
- **返回格式**：`trainStep` 和 `evalStep` 必须返回 `(loss_dict, output_dict)`。`loss_dict` 中的键必须与声明的损失名称匹配。
- **混合精度**：自动处理梯度缩放。在前向传递中使用 `getAutoCast` 上下文。
- **梯度裁剪**：设置 `clip_grad > 0` 以启用。在混合精度中梯度反缩放后应用。
- **模型编译**：`compile_model=True` 使用 `torch.compile()` 进行优化（PyTorch 2.0+）。
- **检查点加载**：`loadFrom` 优雅地处理尺寸不匹配，跳过不兼容参数并报告。
- **Basics模块命名约定**：格式为 `[Norm][Act]Conv[ND]`（预激活）或 `Conv[ND][Norm][Act]`（后激活）。例如：`Conv2DBnGELU` = Conv2D → 批归一化 → GELU。
- **位置编码合并模式**：`"add"` 将编码加到输入；`"concat"` 沿特征维度拼接。

---

## 3. 训练与实验 (Training and Experiment)

### 快速使用

```python
from JimmyExperiment import JimmyExperiment
from DynamicConfig import DynamicConfig

# 定义实验配置
exp = JimmyExperiment(comments="使用AdamW的基线实验")

# 配置数据集
exp.dataset_cfg = DynamicConfig(
    MyDataset,
    batch_size=64,
    shuffle=True
)

# 配置模型
exp.model_cfg = DynamicConfig(
    MyModel,
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
    mixed_precision=True
)

# 配置学习率调度器
exp.lr_scheduler_cfg = DynamicConfig(
    JimmyLRScheduler,
    peak_lr=2e-4,
    min_lr=1e-7,
    warmup_count=10,
    patience=10,
    decay_rate=0.5
)

# 设置训练常量
exp.constants = {
    "n_epochs": 100,
    "moving_avg": 100,
    "eval_interval": 5
}

# 开始训练
trainer = exp.start(checkpoint=None)
```

### 目录索引

| 路径 | 类 | 函数 | 描述 |
|------|-------|----------|-------------|
| `JimmyTrainer.py` | `JimmyTrainer` | `__init__(train_set, eval_set, model, lr_scheduler, log_dir, save_dir, n_epochs, moving_avg, eval_interval)` | 使用数据集、模型和超参数初始化训练器 |
| `JimmyTrainer.py` | `JimmyTrainer` | `start()` | 执行完整训练循环，包含日志和检查点 |
| `JimmyTrainer.py` | `JimmyTrainer` | `evaluate(dataset, compute_avg)` | 在数据集上评估模型，返回损失字典（平均或每样本） |
| `JimmyExperiment.py` | `JimmyExperiment` | `__init__(comments)` | 使用描述字符串初始化实验 |
| `JimmyExperiment.py` | `JimmyExperiment` | `start(checkpoint)` | 从配置构建组件并启动训练 |
| `DynamicConfig.py` | `DynamicConfig` | `__init__(cls, **kwargs)` | 存储类和初始化参数 |
| `DynamicConfig.py` | `DynamicConfig` | `build()` | 使用存储的参数实例化类 |
| `DynamicConfig.py` | `DynamicConfig` | `add(**kwargs)` | 添加或更新参数 |
| `DynamicConfig.py` | `DynamicConfig` | `remove(key)` | 删除参数 |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `__init__(optimizer, peak_lr, min_lr, warmup_count, window_size, patience, decay_rate)` | 初始化自适应学习率调度器 |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `update(metric)` | 根据指标（损失）更新学习率 |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `phaseWarmUP()` | 计算预热阶段的LR（正弦上升） |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `phaseCosine()` | 对LR应用余弦退火 |
| `Training/JimmyLRScheduler.py` | `JimmyLRScheduler` | `phaseDecay()` | 当指标停滞时衰减LR |
| `Training/ProgressManager.py` | `ProgressManager` | `__init__(items_per_epoch, epochs, show_recent, refresh_interval, custom_fields)` | 使用自定义指标字段初始化进度可视化 |
| `Training/ProgressManager.py` | `ProgressManager` | `update(epoch, step, **kwargs)` | 使用当前指标更新进度显示 |
| `Training/ProgressManager.py` | `ProgressManager` | `close()` | 完成并关闭进度显示 |
| `Training/MovingAverage.py` | `MovingAvg` | `__init__(window_size)` | 初始化移动平均跟踪器 |
| `Training/MovingAverage.py` | `MovingAvg` | `update(value)` | 向移动平均添加新值 |
| `Training/MovingAverage.py` | `MovingAvg` | `get()` | 检索当前移动平均值 |
| `Training/TensorBoardManager.py` | `TensorBoardManager` | `__init__(log_dir, tags, value_types)` | 使用预注册标签初始化 TensorBoard 记录器 |
| `Training/TensorBoardManager.py` | `TensorBoardManager` | `log(step, **kwargs)` | 记录标量/直方图/图像值到 TensorBoard |
| `RunRecordManager.py` | `RunRecordManager` | - | 管理实验记录和结果（持久化层） |

### 注意事项

- **DynamicConfig模式**：延迟对象实例化直到调用 `build()`。允许在不重新创建对象的情况下运行时修改超参数。
- **JimmyLRScheduler阶段**：(1) 正弦上升到 peak_lr 的预热，(2) 带高频调制的余弦退火，(3) 检测到停滞时的指数衰减。
- **停滞检测**：LR调度器跟踪 `window_size` 个轮次的指标移动平均。如果 `patience` 个轮次无改善，触发衰减。
- **自动目录结构**：实验创建 `Runs/{数据集名}/{模型名}/{时间戳}/` 用于日志、检查点和配置。
- **检查点策略**：每 `eval_interval` 轮次保存 `best.pth`（最低评估损失）和 `last.pth`（最新）。
- **ProgressManager**：使用 Rich 库实时更新终端显示。显示最近N个轮次、ETA和自定义指标。
- **指标记录**：训练指标使用移动平均（减少噪声），评估指标是原始值。
- **PyTorch LR调度器兼容性**：`JimmyTrainer` 通过检测 `step()` 签名自动包装 PyTorch 调度器，创建兼容的 `update()` 方法。

---

## 4. 整体流程 (Overall Pipeline)

1. **实现数据集**：继承 `JimmyDataset`，在 `__init__` 中将所有数据加载为张量，设置 `self.n_samples`，实现返回字典的 `__getitem__`。

2. **实现模型**：继承 `JimmyModel`，在 `__init__` 中定义架构和 `train_loss_names`/`eval_loss_names`，实现 `forward`、`trainStep`、`evalStep`。

3. **配置实验**：创建 `JimmyExperiment` 实例，使用 `DynamicConfig` 配置 `dataset_cfg`、`model_cfg`、`lr_scheduler_cfg`，设置训练常量（`n_epochs`、`eval_interval` 等）。

4. **启动训练**：调用 `experiment.start(checkpoint=None)` 构建组件、创建目录并执行训练循环。

5. **监控进度**：通过 `ProgressManager` 在终端查看实时进度，在 `Runs/{数据集名}/{模型名}/{时间戳}/` 检查 TensorBoard 日志。

6. **评估和测试**：训练后，`JimmyExperiment` 自动在测试集上运行 `evaluate()` 并保存详细报告到 `test_report.csv`。

7. **针对新任务定制**：对于不同的训练流程（如GANs、强化学习），按照相同模式实现自定义 `Trainer` 和 `Experiment` 类。使用 `main.py` 作为入口脚本模板。

8. **使用构建块**：使用 `Models/Basics.py`（卷积块、MLP、位置编码）、`Models/Attentions.py`（MHSA、交叉注意力、SE层）和 `Models/ModelUtils.py`（形状操作层）中的组件组合模型。

9. **处理轨迹**：对于GPS轨迹任务，使用 `Datasets/TrajectoryUtils.py` 中的工具进行预处理（插值、旋转、归一化）和评估（几何距离、JSD）。

10. **利用高级特性**：启用混合精度训练（`mixed_precision=True`）、梯度裁剪（`clip_grad > 0`）、模型编译（`compile_model=True`）和自适应LR调度（`JimmyLRScheduler`）以实现高效训练。

---

## 核心设计原则

- **字典接口**：数据集返回字典，模型接受/返回字典。实现灵活组合，无需严格的参数顺序。
- **关注点分离**：数据集处理数据加载，模型处理计算和优化，训练器编排循环，实验管理配置。
- **GPU优先理念**：假设数据适合GPU内存。消除训练期间CPU-GPU传输开销。
- **配置即代码**：使用 `DynamicConfig` 以编程方式版本化和修改超参数。
- **可重现性**：实验日志自动保存模型架构、超参数和注释。
- **可扩展性**：核心类（`JimmyDataset`、`JimmyModel`、`JimmyTrainer`、`JimmyExperiment`）是模板。重写方法以实现特定任务逻辑。

---

## 快速开始示例

```python
# 1. 数据集
from Datasets import JimmyDataset
class MyData(JimmyDataset):
    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.X = torch.randn(1000, 10).to(DEVICE)
        self.y = torch.randint(0, 2, (1000,)).to(DEVICE)
        self.n_samples = 1000
    def __getitem__(self, idx):
        start, end = (idx-1)*self.batch_size, idx*self.batch_size
        return {'data': self.X[start:end], 'target': self.y[start:end]}

# 2. 模型
from Models import JimmyModel
import torch.nn as nn
class MyModel(JimmyModel):
    def __init__(self):
        super().__init__(optimizer_cls=torch.optim.Adam, optimizer_args={"lr": 1e-3})
        self.train_loss_names = ["Train/Loss"]
        self.eval_loss_names = ["Eval/Loss"]
        self.net = nn.Linear(10, 2)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x): return self.net(x)
    def trainStep(self, d):
        out = self(d['data'])
        loss = self.loss_fn(out, d['target'])
        self.backwardOptimize(loss)
        return {"Train/Loss": loss.item()}, {"output": out.detach()}
    def evalStep(self, d):
        with torch.no_grad():
            out = self(d['data'])
            loss = self.loss_fn(out, d['target'])
        return {"Eval/Loss": loss.item()}, {"output": out}

# 3. 实验
from JimmyExperiment import JimmyExperiment
from DynamicConfig import DynamicConfig
exp = JimmyExperiment("快速开始测试")
exp.dataset_cfg = DynamicConfig(MyData, batch_size=32)
exp.model_cfg = DynamicConfig(MyModel)
exp.constants = {"n_epochs": 10, "moving_avg": 10, "eval_interval": 2}
trainer = exp.start()
```

此示例用约40行代码演示了完整流程。
