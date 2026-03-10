import os
import json
import torch
import math
import time
from typing import Dict, Optional, Tuple
from peft import PeftModel, LoraConfig
from transformers import AutoModel, AutoTokenizer
from get_model import ESMWithClassification, load_config


def safe_json_serialize(obj):
    """安全的JSON序列化，处理Infinity、NaN值和不可序列化的对象"""
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(safe_json_serialize(item) for item in obj)
    elif isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        elif math.isnan(obj):
            return "NaN"
        else:
            return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    elif isinstance(obj, torch.Tensor):
        # PyTorch tensor不能直接序列化到JSON，跳过或转换
        return f"<Tensor: shape={list(obj.shape)}, dtype={str(obj.dtype)}>"
    else:
        # 对于其他不可序列化的对象，转换为字符串描述
        return f"<{type(obj).__name__}: {str(obj)[:100]}>"


def safe_json_deserialize(obj):
    """安全的JSON反序列化，将字符串形式的特殊值转换回浮点数"""
    if isinstance(obj, dict):
        return {k: safe_json_deserialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_deserialize(item) for item in obj]
    elif isinstance(obj, str):
        if obj == "Infinity":
            return float('inf')
        elif obj == "-Infinity":
            return float('-inf')
        elif obj == "NaN":
            return float('nan')
        else:
            return obj
    else:
        return obj


class LoRAModelSaver:
    """
    LoRA模型保存和加载管理器
    
    专门处理PEFT LoRA模型的保存、加载、合并等操作
    支持多种保存策略：
    1. 仅保存LoRA适配器（推荐，文件小）
    2. 保存完整模型状态
    3. 合并LoRA到基础模型并保存
    """
    
    def __init__(self, config_path: str, save_dir: str = "./checkpoints", task_name: str = None):
        """
        初始化模型保存器
        
        Args:
            config_path: 模型配置文件路径
            save_dir: 保存目录
            task_name: 任务名称（合并格式时必需）
        """
        self.config_path = config_path
        self.config = load_config(config_path, task_name)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_lora_adapter(
        self, 
        model: ESMWithClassification, 
        save_path: str,
        save_classifier: bool = True,
        additional_info: Optional[Dict] = None
    ) -> None:
        """
        保存LoRA适配器（推荐方式）
        
        这种方式只保存LoRA的权重，文件很小（通常几MB），
        加载时需要原始的基础模型。
        
        Args:
            model: 训练好的模型
            save_path: 保存路径（目录）
            save_classifier: 是否保存分类头
            additional_info: 额外信息（如训练指标、配置等）
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 保存LoRA适配器
        print(f"保存LoRA适配器到: {save_path}")
        model.base_model.save_pretrained(save_path)
        
        # 2. 保存分类头（如果需要）
        if save_classifier:
            classifier_path = os.path.join(save_path, "classifier.pth")
            torch.save({
                'classifier_state_dict': model.classifier.state_dict(),
                'num_classes': model.num_classes,
                'dropout_rate': model.dropout_rate
            }, classifier_path)
            print(f"分类头已保存: {classifier_path}")
        
        # 3. 保存模型配置
        config_save_path = os.path.join(save_path, "model_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 4. 保存额外信息
        if additional_info:
            # 分离出optimizer_state_dict，单独保存为.pth文件
            info_copy = additional_info.copy()
            optimizer_state_dict = info_copy.pop('optimizer_state_dict', None)
            
            # 使用安全的JSON序列化处理无效值
            try:
                safe_info = safe_json_serialize(info_copy)
                
                # 保存JSON格式的训练信息（不包含tensor）
                info_path = os.path.join(save_path, "training_info.json")
                with open(info_path, 'w') as f:
                    json.dump(safe_info, f, indent=2)
                print(f"训练信息已保存: {info_path}")
                        
            except Exception as e:
                print(f"警告: 保存训练信息失败: {str(e)}")
                # 创建一个最小的训练信息文件，避免完全失败
                minimal_info = {
                    'epoch': info_copy.get('epoch', 'unknown'),
                    'best_epoch': info_copy.get('best_epoch', 'unknown'),
                    'best_value': str(info_copy.get('best_value', 'unknown')),
                    'monitor': info_copy.get('monitor', 'val_loss'),
                    'mode': info_copy.get('mode', 'min'),
                    'training_completed': info_copy.get('training_completed', False),
                    'save_error': str(e),
                    'created_at': str(time.time())
                }
                info_path = os.path.join(save_path, "training_info.json")
                try:
                    with open(info_path, 'w') as f:
                        json.dump(minimal_info, f, indent=2)
                    print(f"最小训练信息已保存: {info_path}")
                except Exception as e2:
                    print(f"错误: 连最小训练信息也无法保存: {str(e2)}")
            
            # 单独保存optimizer状态
            if optimizer_state_dict is not None:
                try:
                    optimizer_path = os.path.join(save_path, "optimizer.pth")
                    torch.save(optimizer_state_dict, optimizer_path)
                    print(f"优化器状态已保存: {optimizer_path}")
                except Exception as e:
                    print(f"警告: 保存优化器状态失败: {str(e)}")
        
        # 5. 保存README
        self._create_model_readme(save_path, "lora_adapter")
        
        print(f"LoRA模型保存完成: {save_path}")
    
    def save_full_model(
        self, 
        model: ESMWithClassification, 
        save_path: str,
        additional_info: Optional[Dict] = None
    ) -> None:
        """
        保存完整模型状态（包含所有权重）
        
        这种方式保存所有权重，文件较大，但加载时不需要原始基础模型。
        适用于部署或者需要完全独立的模型文件的场景。
        
        Args:
            model: 训练好的模型
            save_path: 保存路径（文件）
            additional_info: 额外信息
        """
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': self.config,
            'model_class': 'ESMWithClassification'
        }
        
        if additional_info:
            # 分离optimizer_state_dict，保持与其他保存方法一致的处理方式
            info_copy = additional_info.copy()
            optimizer_state_dict = info_copy.pop('optimizer_state_dict', None)
            
            # 将non-tensor信息添加到save_data
            save_data.update(info_copy)
            
            # 如果有optimizer状态，单独添加（torch.save可以处理tensor）
            if optimizer_state_dict is not None:
                save_data['optimizer_state_dict'] = optimizer_state_dict
        
        torch.save(save_data, save_path)
        print(f"完整模型已保存: {save_path}")
    
    def save_merged_model(
        self, 
        model: ESMWithClassification, 
        save_path: str,
        additional_info: Optional[Dict] = None
    ) -> None:
        """
        合并LoRA到基础模型并保存
        
        将LoRA权重合并到基础模型中，得到一个标准的transformers模型。
        适用于推理部署，不再需要PEFT库。
        
        Args:
            model: 训练好的模型
            save_path: 保存路径（目录）
            additional_info: 额外信息
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 合并LoRA权重到基础模型
        print("正在合并LoRA权重到基础模型...")
        merged_model = model.base_model.merge_and_unload()
        
        # 2. 保存合并后的基础模型
        base_model_path = os.path.join(save_path, "merged_base_model")
        merged_model.save_pretrained(base_model_path)
        print(f"合并后的基础模型已保存: {base_model_path}")
        
        # 3. 保存分类头
        classifier_path = os.path.join(save_path, "classifier.pth")
        torch.save({
            'classifier_state_dict': model.classifier.state_dict(),
            'num_classes': model.num_classes,
            'dropout_rate': model.dropout_rate
        }, classifier_path)
        
        # 4. 保存配置
        config_save_path = os.path.join(save_path, "model_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 5. 保存额外信息
        if additional_info:
            # 分离出optimizer_state_dict，单独保存为.pth文件（与save_lora_adapter保持一致）
            info_copy = additional_info.copy()
            optimizer_state_dict = info_copy.pop('optimizer_state_dict', None)
            
            # 使用安全的JSON序列化处理无效值
            try:
                safe_info = safe_json_serialize(info_copy)
                
                # 保存JSON格式的训练信息（不包含tensor）
                info_path = os.path.join(save_path, "training_info.json")
                with open(info_path, 'w') as f:
                    json.dump(safe_info, f, indent=2)
                print(f"训练信息已保存: {info_path}")
                
            except Exception as e:
                print(f"警告: 保存训练信息失败: {str(e)}")
                # 创建一个最小的训练信息文件，避免完全失败
                minimal_info = {
                    'epoch': info_copy.get('epoch', 'unknown'),
                    'best_epoch': info_copy.get('best_epoch', 'unknown'),
                    'best_value': str(info_copy.get('best_value', 'unknown')),
                    'monitor': info_copy.get('monitor', 'val_loss'),
                    'mode': info_copy.get('mode', 'min'),
                    'training_completed': info_copy.get('training_completed', False),
                    'save_error': str(e),
                    'created_at': str(time.time())
                }
                info_path = os.path.join(save_path, "training_info.json")
                try:
                    with open(info_path, 'w') as f:
                        json.dump(minimal_info, f, indent=2)
                    print(f"最小训练信息已保存: {info_path}")
                except Exception as e2:
                    print(f"错误: 连最小训练信息也无法保存: {str(e2)}")
            
            # 单独保存optimizer状态
            if optimizer_state_dict is not None:
                try:
                    optimizer_path = os.path.join(save_path, "optimizer.pth")
                    torch.save(optimizer_state_dict, optimizer_path)
                    print(f"优化器状态已保存: {optimizer_path}")
                except Exception as e:
                    print(f"警告: 保存优化器状态失败: {str(e)}")
        
        # 6. 保存README
        self._create_model_readme(save_path, "merged_model")
        
        print(f"合并模型保存完成: {save_path}")
    
    def load_lora_adapter(
        self, 
        load_path: str,
        device: str = "cuda",
        load_classifier: bool = True
    ) -> Tuple[ESMWithClassification, AutoTokenizer]:
        """
        加载LoRA适配器模型
        
        Args:
            load_path: LoRA适配器保存路径
            device: 设备
            
        Returns:
            (model, tokenizer): 加载的模型和分词器
        """
        print(f"从LoRA适配器加载模型: {load_path}")
        
        # 1. 加载配置
        # 对于LoRA权重，使用预训练的配置来正确加载
        # 对于分类头，使用传入的配置来创建正确的输出维度
        config_path = os.path.join(load_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                pretrain_config = json.load(f)
            print(f"预训练配置中的num_classes: {pretrain_config['model']['num_classes']}")
        else:
            pretrain_config = self.config
        
        # 使用传入的配置（微调配置）来创建分类头
        finetune_config = self.config
        print(f"微调配置中的num_classes: {finetune_config['model']['num_classes']}")
        
        # 合并配置：LoRA部分使用预训练配置，分类头部分使用微调配置
        config = pretrain_config.copy()
        config['model']['num_classes'] = finetune_config['model']['num_classes']  # 使用微调的类别数
        print(f"最终使用的num_classes: {config['model']['num_classes']}")
        
        # 2. 加载基础模型和分词器
        # 根据设备类型选择合适的加载方式
        if device == "cpu" or not torch.cuda.is_available():
            # CPU模式：强制使用CPU
            base_model = AutoModel.from_pretrained(
                config['model']['esm_model_path'],
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            device = "cpu"  # 确保设备一致
        else:
            # GPU模式
            base_model = AutoModel.from_pretrained(config['model']['esm_model_path'])
        
        tokenizer = AutoTokenizer.from_pretrained(config['model']['esm_model_path'])
        
        # 3. 加载LoRA适配器
        base_model = PeftModel.from_pretrained(base_model, load_path)
        print("LoRA适配器加载成功")
        
        # 4. 创建完整模型
        hidden_size = base_model.config.hidden_size
        model = ESMWithClassification(
            base_model=base_model,
            hidden_size=hidden_size,
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['hidden_dropout']
        )
        
        # 5. 加载分类头（如果存在且需要加载）
        if load_classifier:
            classifier_path = os.path.join(load_path, "classifier.pth")
            if os.path.exists(classifier_path):
                classifier_data = torch.load(classifier_path, map_location=device)
                model.classifier.load_state_dict(classifier_data['classifier_state_dict'])
                print("分类头加载成功")
            else:
                print("警告: 未找到分类头文件，使用随机初始化的分类头")
        else:
            print("跳过分类头加载，使用随机初始化的分类头")
        
        model.to(device)
        print(f"模型加载完成，设备: {device}")
        
        return model, tokenizer
    
    def load_full_model(
        self, 
        load_path: str,
        device: str = "cuda"
    ) -> Tuple[ESMWithClassification, AutoTokenizer]:
        """
        加载完整模型
        
        Args:
            load_path: 完整模型保存路径
            device: 设备
            
        Returns:
            (model, tokenizer): 加载的模型和分词器
        """
        print(f"加载完整模型: {load_path}")
        
        # 加载保存的数据
        checkpoint = torch.load(load_path, map_location=device)
        config = checkpoint['model_config']
        
        # 重建模型（这里需要先创建基础架构）
        # 注意：完整模型加载比较复杂，因为需要重建PEFT结构
        # 推荐使用LoRA适配器方式
        raise NotImplementedError("完整模型加载功能待实现，推荐使用load_lora_adapter")
    
    def load_merged_model(
        self, 
        load_path: str,
        device: str = "cuda"
    ) -> Tuple[ESMWithClassification, AutoTokenizer]:
        """
        加载合并后的模型
        
        Args:
            load_path: 合并模型保存路径
            device: 设备
            
        Returns:
            (model, tokenizer): 加载的模型和分词器
        """
        print(f"加载合并模型: {load_path}")
        
        # 1. 加载配置
        config_path = os.path.join(load_path, "model_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 2. 加载合并后的基础模型
        base_model_path = os.path.join(load_path, "merged_base_model")
        base_model = AutoModel.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(config['model']['esm_model_path'])
        
        # 3. 创建完整模型
        hidden_size = base_model.config.hidden_size
        model = ESMWithClassification(
            base_model=base_model,
            hidden_size=hidden_size,
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['hidden_dropout']
        )
        
        # 4. 加载分类头
        classifier_path = os.path.join(load_path, "classifier.pth")
        classifier_data = torch.load(classifier_path, map_location=device)
        model.classifier.load_state_dict(classifier_data['classifier_state_dict'])
        
        model.to(device)
        print(f"合并模型加载完成，设备: {device}")
        
        return model, tokenizer
    
    def _create_model_readme(self, save_path: str, model_type: str) -> None:
        """创建模型说明文件"""
        readme_content = f"""# PepLoe Model - {model_type.upper()}

## 模型信息
- 模型类型: {model_type}
- 基础模型: {self.config['model']['esm_model_path']}
- 分类数: {self.config['model']['num_classes']}
- LoRA配置: r={self.config['lora']['r']}, alpha={self.config['lora']['lora_alpha']}

## 文件结构
"""
        
        if model_type == "lora_adapter":
            readme_content += """
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
"""
        elif model_type == "merged_model":
            readme_content += """
- merged_base_model/: 合并后的基础模型目录
- classifier.pth: 分类头权重
- model_config.json: 完整模型配置
- training_info.json: 训练信息（如果有）
- README.md: 本文件

## 加载方式
```python
from model_saver import LoRAModelSaver
saver = LoRAModelSaver(config_path)
model, tokenizer = saver.load_merged_model(save_path)
```
"""
        
        readme_path = os.path.join(save_path, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def get_model_info(self, model_path: str) -> Dict:
        """
        获取保存模型的信息
        
        Args:
            model_path: 模型路径
            
        Returns:
            模型信息字典
        """
        info = {}
        
        # 检查是否是LoRA适配器
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            info['type'] = 'lora_adapter'
            with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                info['adapter_config'] = json.load(f)
        
        # 检查是否是合并模型
        elif os.path.exists(os.path.join(model_path, "merged_base_model")):
            info['type'] = 'merged_model'
        
        # 检查训练信息
        training_info_path = os.path.join(model_path, "training_info.json")
        if os.path.exists(training_info_path):
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
                # 安全反序列化，处理特殊值
                info['training_info'] = safe_json_deserialize(training_info)
        
        # 检查模型配置
        config_path = os.path.join(model_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                info['model_config'] = json.load(f)
        
        return info


def create_model_saver(config_path: str, save_dir: str = "./checkpoints") -> LoRAModelSaver:
    """便捷函数：创建模型保存器"""
    return LoRAModelSaver(config_path, save_dir)


# 使用示例
if __name__ == "__main__":
    # 创建模型保存器
    config_path = "/public/home/lingwang/wow/PepLoe/src/model_config.json"
    saver = LoRAModelSaver(config_path, "./test_checkpoints")
    
    print("LoRA模型保存器创建成功")
    print(f"配置文件: {config_path}")
    print(f"保存目录: {saver.save_dir}") 