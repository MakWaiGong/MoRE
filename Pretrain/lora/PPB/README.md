# PepLoe Model - LORA_ADAPTER

## 模型信息
- 模型类型: lora_adapter
- 基础模型: facebook/esm2_t36_3B_UR50D
- 分类数: 2
- LoRA配置: r=8, alpha=16

## 文件结构

- adapter_config.json: LoRA适配器配置
- adapter_model.bin: LoRA权重文件
- classifier.pth: 分类头权重
- model_config.json: 完整模型配置
- training_info.json: 训练信息（如果有）
- optimizer.pth: optimizer状态
- README.md: 本文件

## 加载方式
```python
from model_saver import LoRAModelSaver
saver = LoRAModelSaver(config_path)
model, tokenizer = saver.load_lora_adapter(save_path)
```
