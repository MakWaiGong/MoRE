# 预训练任务说明文档

本文档详细说明四个预训练任务的设计、数据格式、参数配置和训练机制。

---

## 📋 目录

1. [任务概览](#任务概览)
2. [BMP - 掩码语言模型](#1-bmp---掩码语言模型)
3. [Dist2NBS - 距离最近结合位点](#2-dist2nbs---距离最近结合位点)
4. [PPB - 蛋白质-肽结合](#3-ppb---蛋白质-肽结合)
5. [SWBindCount - 滑动窗口结合位点计数](#4-swbindcount---滑动窗口结合位点计数)
6. [训练机制](#训练机制)
7. [快速开始](#快速开始)

---

## 任务概览

| 任务 | 类型 | 目标 | 输出维度 | 最大长度 |
|------|------|------|----------|----------|
| **BMP** | 掩码语言模型 | 预测被掩码的氨基酸 | `vocab_size` (33) | 1024 |
| **Dist2NBS** | 回归 | 预测到最近结合位点的归一化距离 | 1 | 512 |
| **PPB** | 二分类 | 预测每个位置是否为结合位点 | 2 | 1024 |
| **SWBindCount** | 回归 | 预测滑动窗口内结合位点数量 | 1 | 1024 |

---

## 1. BMP - 掩码语言模型

### 🎯 任务目标

**Bidirectional Masked Prediction (BMP)** 是一个掩码语言模型任务，旨在学习蛋白质序列的上下文表示。模型需要预测被掩码位置的原始氨基酸。

### 📊 数据格式

**输入数据列**:
- `pdb_id`: 蛋白质ID
- `mask_seq`: 已掩码的序列（使用'X'表示掩码位置）
- `label`: 原始序列（作为预测目标）

**数据生成策略**:
1. **掩码位置选择**: 只掩码结合位点（label=1）和肽链全部位点
2. **掩码比例**: 默认50%（可配置）
3. **BERT式掩码策略**:
   - 80% 替换为 `[MASK]` (在数据中用'X'表示)
   - 10% 替换为随机氨基酸
   - 10% 保持不变

**示例**:
```
原始序列: MKTAYIAKQR...
掩码序列: MKTAYIAKXQ...  (第9位被掩码)
标签:     只在掩码位置保留真实氨基酸token_id，其他位置为-100
```

### ⚙️ 参数配置

**配置文件**: `pretrain_config.json` (合并配置文件，包含所有任务的配置)

```json
{
  "model": {
    "esm_model_path": "facebook/esm2_t36_3B_UR50D",
    "num_classes": 22,  // 实际使用时会自动替换为tokenizer.vocab_size
    "task_type": "masked_language_model",
    "hidden_dropout": 0.4
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["key", "value"],
    "lora_dropout": 0.3,
    "bias": "none"
  },
  "training": {
    "max_length": 1024,
    "default_epochs": 5,
    "default_batch_size": 8,
    "default_lr": 1e-3,
    "mask_ratio": 0.50
  }
}
```

### 🔗 Token对齐机制

**关键挑战**: 掩码序列和原始序列在tokenization后必须保持位置对齐。

**对齐方法**:
1. 分别对原始序列和掩码序列进行tokenization
2. 建立位置映射：
   - `original_seq_to_tokenized`: 原始序列位置 → tokenized位置
   - `mask_seq_to_tokenized`: 掩码序列位置 → tokenized位置
   - `mask_tokenized_to_original`: 掩码tokenized位置 → 原始序列位置
3. **严格验证**:
   - 验证原始序列和掩码序列长度一致
   - 验证tokenization后有效token数量一致
   - 验证掩码数量一致性

**代码位置**: `BMP_pretrain.py` 第102-146行

### 📈 损失函数

- **类型**: CrossEntropyLoss
- **忽略索引**: -100（非掩码位置）
- **计算方式**: 只在掩码位置计算损失

```python
loss = CrossEntropyLoss(ignore_index=-100)(
    logits.view(-1, vocab_size), 
    labels.view(-1)
)
# 只在掩码位置计算
loss = (loss_all * mask.view(-1)).sum() / (mask.sum() + 1e-8)
```

---

## 2. Dist2NBS - 距离最近结合位点

### 🎯 任务目标

**Distance to Nearest Binding Site (Dist2NBS)** 是一个回归任务，预测蛋白质序列中每个位置到最近结合位点的归一化距离。

### 📊 数据格式

**输入数据列**:
- `pdb_id`: 蛋白质ID
- `prot_seq`: 蛋白质序列
- `label`: 距离值（逗号分隔的浮点数，范围0-1）

**距离计算**:
1. 找到所有结合位点位置（label='1'）
2. 计算每个位置到最近结合位点的距离
3. 归一化到[0,1]范围：`normalized_dist = min(min_dist / (seq_len/2), 1.0)`
4. 结合位点本身距离为0.0

**示例**:
```
序列:     MKTAYIAKQR
结合位点: 0001001000  (第4和第7位是结合位点)
距离:     0.4,0.3,0.2,0.0,0.1,0.2,0.0,0.1,0.2,0.3
```

### ⚙️ 参数配置

**配置文件**: `pretrain_config.json` (合并配置文件)

```json
{
  "model": {
    "esm_model_path": "facebook/esm2_t36_3B_UR50D",
    "num_classes": 1,
    "task_type": "regression",
    "hidden_dropout": 0.4
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["key", "value"],
    "lora_dropout": 0.3,
    "bias": "none"
  },
  "training": {
    "max_length": 512,
    "default_epochs": 10,
    "default_batch_size": 4,
    "default_lr": 1e-4,
    "loss_function": "MSE"
  }
}
```

### 🔗 Token对齐机制

**对齐方法**:
1. 识别并跳过特殊token（cls, eos, sep, pad）
2. 按顺序将距离标签映射到序列token位置
3. 使用 `seq_token_positions` 列表建立一一对应关系

**代码位置**: `Dist2NBS_pretrain.py` 第89-99行

### 📈 损失函数

- **类型**: MSELoss
- **计算方式**: 只在有效位置（非特殊token）计算损失

```python
def mse_loss_with_mask(pred, target, mask):
    loss = MSELoss(reduction='none')(pred.squeeze(-1), target)
    return (loss * mask).sum() / (mask.sum() + 1e-8)
```

**模型结构**: 动态添加回归头 `distance_head: hidden_size → 1`

---

## 3. PPB - 蛋白质-肽结合

### 🎯 任务目标

**Protein-Peptide Binding (PPB)** 是一个二分类任务，预测蛋白质序列中每个位置是否为肽结合位点。

### 📊 数据格式

**输入数据列**:
- `pdb_id`: 蛋白质ID
- `prot_seq`: 蛋白质序列
- `pep_seq`: 肽链序列
- `label`: 二进制标签字符串（'1'表示结合位点，'0'表示非结合位点）

**示例**:
```
prot_seq: MKTAYIAKQR
pep_seq:  AYI
label:    0001110000  (第4-6位是结合位点)
```

### ⚙️ 参数配置

**配置文件**: `pretrain_config.json` (合并配置文件)

```json
{
  "model": {
    "esm_model_path": "facebook/esm2_t36_3B_UR50D",
    "num_classes": 2,
    "task_type": "classification",
    "hidden_dropout": 0.4
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["key", "value"],
    "lora_dropout": 0.3,
    "bias": "none"
  },
  "training": {
    "max_length": 1024,
    "default_epochs": 10,
    "default_batch_size": 4,
    "default_lr": 1e-4,
    "loss_function": "CrossEntropy"
  }
}
```

### 🔗 Token对齐机制

**对齐方法**:
1. 识别并跳过特殊token（cls, eos, sep, pad）
2. 按顺序将二进制标签映射到序列token位置
3. 标签值：0（非结合位点）或 1（结合位点）

**代码位置**: `PPB_pretrain.py` 第107-117行

### 📈 损失函数

- **类型**: CrossEntropyLoss
- **忽略索引**: -100（特殊token位置）
- **输出**: 2类（结合/非结合）

```python
loss = CrossEntropyLoss(ignore_index=-100)(
    logits.view(-1, 2), 
    labels.view(-1)
)
```

---

## 4. SWBindCount - 滑动窗口结合位点计数

### 🎯 任务目标

**Sliding Window Binding Count (SWBindCount)** 是一个回归任务，使用滑动窗口预测窗口内结合位点的数量。

### 📊 数据格式

**输入数据列**:
- `pdb_id`: 蛋白质ID
- `position`: 窗口起始位置（从1开始）
- `prot_seq`: 窗口序列（默认窗口大小30）
- `binding_site_count`: 窗口内结合位点数量（整数）

**数据生成策略**:
1. 使用滑动窗口扫描序列（默认窗口大小30，步长1）
2. 计算每个窗口内结合位点（label='1'）的数量
3. 如果序列长度小于窗口大小，跳过该样本

**示例**:
```
完整序列: MKTAYIAKQRSTUVWXYZ (长度18)
窗口大小: 5
窗口1 (pos=1): MKTAY → count=0
窗口2 (pos=2): KTAYI → count=1
窗口3 (pos=3): TAYIA → count=2
...
```

### ⚙️ 参数配置

**配置文件**: `pretrain_config.json` (合并配置文件)

```json
{
  "model": {
    "esm_model_path": "facebook/esm2_t36_3B_UR50D",
    "num_classes": 1,
    "task_type": "regression",
    "hidden_dropout": 0.4
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["key", "value"],
    "lora_dropout": 0.3,
    "bias": "none"
  },
  "training": {
    "max_length": 1024,
    "default_epochs": 10,
    "default_batch_size": 128,
    "default_lr": 1e-4,
    "loss_function": "Poisson",
    "loss_mode": "poisson"
  }
}
```

### 🔗 Token对齐机制

**对齐方法**: 
- 使用全局平均池化，不需要逐位置对齐
- 对有效位置（非padding）进行加权平均

**代码位置**: `SWBindCount_pretrain.py` 第262-266行

### 📈 损失函数

支持两种损失函数模式：

1. **MSE损失**（默认）:
```python
loss = MSELoss()(pred.squeeze(), target)
```

2. **Poisson损失**（推荐，适用于计数任务）:
```python
pred = torch.clamp(pred.squeeze(), min=1e-8)
loss = torch.mean(pred - target * torch.log(pred))
```

**模型结构**: 
- 全局平均池化: `[batch, seq_len, hidden] → [batch, hidden]`
- 回归头: `hidden → 64 → 1`

---

## 训练机制

### 📊 数据集划分

**所有任务使用相同的划分策略**:

- **训练集**: 95%
- **验证集**: 5%
- **测试集**: 无（如需可添加）

**划分方法**:
```python
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(
    range(len(full_dataset)), 
    test_size=0.05,  # 5%作为验证集
    random_state=42  # 固定随机种子，确保可复现
)
```

**代码位置**:
- BMP: `BMP_pretrain.py` 第337-344行
- Dist2NBS: `Dist2NBS_pretrain.py` 第190-200行
- PPB: `PPB_pretrain.py` 第212-223行
- SWBindCount: `SWBindCount_pretrain.py` 第183-195行

### ⏹️ 早停机制

**所有任务都实现了早停机制**:

- **Patience**: 3（验证损失连续3轮未改善则停止）
- **最大轮数**: 10（最多训练10个epoch）
- **最佳模型保存**: 自动保存验证损失最低的模型

**早停逻辑**:
```python
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(max_epochs):
    # ... 训练和验证 ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        saver.save_lora_adapter(model, output_dir)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('早停触发！')
            break
```

**代码位置**:
- BMP: `BMP_pretrain.py` 第394-485行
- Dist2NBS: `Dist2NBS_pretrain.py` 第216-318行
- PPB: `PPB_pretrain.py` 第235-320行
- SWBindCount: `SWBindCount_pretrain.py` 第222-332行

### 📈 训练过程可视化

**所有任务都保存训练曲线**:

1. **Loss曲线图**: `loss_curves.png`
   - 包含训练损失和验证损失曲线
   - 分辨率: 300 DPI
   - 格式: PNG

2. **Loss数据CSV**: `loss_data.csv`
   - 包含每个epoch的训练损失和验证损失
   - 列: `epoch`, `train_loss`, `val_loss`

**保存位置**: 各任务的输出目录（如 `lora/BMP/`）

**代码位置**:
- BMP: `BMP_pretrain.py` 第487-513行
- Dist2NBS: `Dist2NBS_pretrain.py` 第317-343行
- PPB: `PPB_pretrain.py` 第322-350行
- SWBindCount: `SWBindCount_pretrain.py` 第336-363行

---

## 快速开始

### 1. 生成预训练数据

```bash
cd Pretrain/data_process
python generate_pretrain_datasets.py \
    --input ../data/pretrain.csv \
    --output ../data \
    --mask_ratio 0.50 \
    --window_size 30 \
    --step_size 1 \
    --seed 42
```

这将生成四个CSV文件：
- `BMP.csv`
- `Dist2NBS.csv`
- `PPB.csv`
- `SWBindCount.csv`

### 2. 运行预训练

#### BMP预训练
```bash
cd Pretrain/src
python BMP_pretrain.py \
    --csv ../data/BMP.csv \
    --config pretrain_config.json \
    --epochs 5 \
    --batch_size 8 \
    --lr 1e-3 \
    --device cuda
```

#### Dist2NBS预训练
```bash
python Dist2NBS_pretrain.py \
    --data ../data/Dist2NBS.csv \
    --config pretrain_config.json \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --device cuda
```

#### PPB预训练
```bash
python PPB_pretrain.py \
    --data ../data/PPB.csv \
    --config pretrain_config.json \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --device cuda
```

#### SWBindCount预训练
```bash
python SWBindCount_pretrain.py \
    --data ../data/SWBindCount.csv \
    --config pretrain_config.json \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --loss_mode poisson \
    --device cuda
```

### 3. 检查训练结果

训练完成后，在输出目录（如 `lora/BMP/`）中会生成：

- `adapter_model.bin`: LoRA适配器权重
- `adapter_config.json`: LoRA配置
- `loss_curves.png`: 训练曲线图
- `loss_data.csv`: 损失数据
- `training_params.txt`: 训练参数记录

---

## 📝 注意事项

### Token对齐

1. **BMP任务**: 需要特别注意掩码序列和原始序列的token对齐
   - 代码中已添加严格验证机制
   - 如果出现对齐错误，会抛出异常或警告

2. **其他任务**: 通过跳过特殊token来确保对齐
   - PPB和Dist2NBS使用相同的对齐机制
   - SWBindCount使用全局池化，不涉及逐位置对齐

### 模型配置

1. **BMP任务**: `num_classes` 会自动替换为 `tokenizer.vocab_size`
   - 配置文件中的 `num_classes=22` 仅作参考
   - 实际运行时使用ESM tokenizer的词汇表大小（通常为33）

2. **动态回归头**: Dist2NBS和SWBindCount会在训练时动态添加回归头
   - 这些回归头会单独保存（`distance_head.pt`, `bind_count_head.pt`）

### 数据采样

- **SWBindCount**: 默认只使用10%的数据进行训练
  - 通过时间戳选择不同的fold，确保每次运行使用不同的数据子集
  - 如需使用全部数据，可以修改代码

---

## 🔗 相关文件

- **数据生成**: `data_process/generate_pretrain_datasets.py`
- **模型加载**: `src/get_model.py`
- **模型保存**: `src/model_saver.py`
- **配置文件**: `src/pretrain_config.json` (合并配置文件，包含所有任务配置)
- **训练脚本**: `src/*_pretrain.py`

---

## 📧 问题反馈

如遇到问题，请检查：
1. 数据格式是否正确
2. Token对齐是否正常（查看调试输出）
3. 模型输出维度是否匹配
4. 损失函数计算是否正确

---

**最后更新**: 2025-12-12
