import os
import json
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

# 移除 HF_ENDPOINT 环境变量，使用官方 Hugging Face Hub
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]

class ESMWithClassification(nn.Module):
    """
    ESM模型加分类头
    参数维度说明:
    - input_ids: [batch_size, seq_len] 序列的token id
    - attention_mask: [batch_size, seq_len] 注意力掩码
    - base_model输出: [batch_size, seq_len, hidden_size] ESM编码后的序列表示
    - classifier输入: [batch_size, seq_len, hidden_size] 保留序列中每个位置的表示
    - 最终输出logits: [batch_size, seq_len, num_classes] 每个位置的分类预测分数
    """
    def __init__(self, base_model, hidden_size, num_classes, dropout_rate=0.3):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def forward(self, input_ids, attention_mask, output_attentions=False):
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=output_attentions  # 添加参数控制是否输出attention
        )
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        logits = self.classifier(sequence_output)    # [batch_size, seq_len, num_classes]
        
        result = {
            'logits': logits,
        }
        
        # 如果需要输出attention，则添加到结果中
        if output_attentions and hasattr(outputs, 'attentions') and outputs.attentions is not None:
            result['attentions'] = outputs.attentions  # tuple of tensors, 每层的attention weights
            
        return result


def load_config(config_path, task_name=None):
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        task_name: 任务名称（BMP, Dist2NBS, PPB, SWBindCount），如果为None则使用旧格式
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 检查是否是合并格式（包含 "tasks" 和 "common" 键）
    if "tasks" in config and "common" in config:
        if task_name is None:
            raise ValueError("检测到合并配置文件格式，请指定 task_name 参数（BMP, Dist2NBS, PPB, SWBindCount）")
        
        if task_name not in config["tasks"]:
            raise ValueError(f"任务名称 '{task_name}' 不存在。可用任务: {list(config['tasks'].keys())}")
        
        # 合并 common 和 task 特定配置
        task_config = config["tasks"][task_name]
        merged_config = {
            "kind": config.get("kind", {}),
            "model": {**config["common"]["model"], **task_config["model"]},
            "lora": config["common"]["lora"],
            "training": task_config["training"]
        }
        return merged_config
    else:
        # 旧格式，直接返回
        return config


def get_model(config_path, task_name=None):
    """根据配置文件加载和配置模型
    
    Args:
        config_path: 配置文件路径
        task_name: 任务名称（BMP, Dist2NBS, PPB, SWBindCount），合并格式时必需
        
    Returns:
        tuple: (model, tokenizer) 配置好的模型和分词器
    """
    # 首先确保移除 HF_ENDPOINT 环境变量（可能在运行时被重新设置）
    if "HF_ENDPOINT" in os.environ:
        old_value = os.environ["HF_ENDPOINT"]
        del os.environ["HF_ENDPOINT"]
        print(f"已移除 HF_ENDPOINT 环境变量 (原值: {old_value})")
    
    # 加载配置
    config = load_config(config_path, task_name)
    
    # 再次确保环境变量被移除（防止在加载配置过程中被重新设置）
    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]
    
    # 加载基础模型和分词器
    # 优先级：1. 本地路径（如果配置） 2. 本地缓存 3. 网络下载
    
    # 检查是否配置了本地模型路径
    local_model_path = config['model'].get('local_model_path', None)
    hf_model_path = config['model']['esm_model_path']  # Hugging Face路径
    
    # 确定使用的模型路径
    # 如果配置了本地路径，优先使用；否则使用模型ID，transformers会自动查找缓存
    if local_model_path and os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"✅ 使用配置的本地模型路径: {model_path}")
        use_local = True
    else:
        # 使用模型ID，transformers会自动在 ~/.cache/huggingface/hub/ 中查找
        model_path = hf_model_path
        if local_model_path:
            print(f"⚠️  警告：配置的本地路径不存在: {local_model_path}")
            print(f"将使用模型ID（自动查找本地缓存）: {model_path}")
        else:
            print(f"使用模型ID（自动查找本地缓存）: {model_path}")
        use_local = False
    
    # 确保环境变量被移除（可能在导入时被重新设置）
    if "HF_ENDPOINT" in os.environ:
        print(f"警告：检测到 HF_ENDPOINT={os.environ['HF_ENDPOINT']}，正在移除...")
        del os.environ["HF_ENDPOINT"]
    
    # 加载模型
    if use_local:
        # 直接从本地路径加载
        try:
            print(f"从本地路径加载模型: {model_path}")
            base_model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            print("模型加载成功（从本地路径）")
        except Exception as e:
            error_msg = f"""
❌ 从本地路径加载模型失败！

本地路径: {model_path}
错误详情: {type(e).__name__}: {str(e)[:500]}

请检查：
1. 路径是否正确
2. 模型文件是否完整（需要包含 config.json, pytorch_model.bin 等）
3. 是否有读取权限
"""
            print(error_msg)
            raise RuntimeError(f"无法从本地路径加载模型: {model_path}") from e
    else:
        # 使用模型ID，transformers会自动在本地缓存中查找
        # 强制只使用本地缓存，避免网络连接问题
        try:
            print(f"从本地缓存加载模型（模型ID: {model_path}）")
            print("transformers会自动在 ~/.cache/huggingface/hub/ 中查找")
            base_model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True  # 强制只使用本地缓存，不尝试网络下载
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            print("✅ 模型加载成功（从本地缓存）")
        except Exception as e:
            error_msg = f"""
❌ 无法从本地缓存加载模型！

模型ID: {model_path}
错误: {type(e).__name__}: {str(e)[:500]}

解决方案：
1. 在配置文件中添加 local_model_path，指向本地模型目录
   例如: "local_model_path": "/public/home/lingwang/.cache/huggingface/hub/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"

2. 或者确保模型已完整下载到本地缓存：
   - 缓存目录: ~/.cache/huggingface/hub/
   - 检查目录是否存在: models--facebook--esm2_t36_3B_UR50D/

3. 如果模型未下载，请先手动下载模型到本地
"""
            print(error_msg)
            raise RuntimeError(f"无法从本地缓存加载模型: {model_path}") from e
    
    # 配置LoRA
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    # 注入LoRA
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()
    
    # 创建完整模型
    hidden_size = base_model.config.hidden_size
    
    # 对于掩码语言模型任务（BMP），使用tokenizer的词汇表大小
    num_classes = config['model']['num_classes']
    if config['model'].get('task_type') == 'masked_language_model':
        # BMP任务需要预测整个词汇表，而不是固定的类别数
        num_classes = tokenizer.vocab_size
        print(f"检测到掩码语言模型任务，使用tokenizer词汇表大小: {num_classes}")
    
    model = ESMWithClassification(
        base_model=base_model,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout_rate=config['model']['hidden_dropout']
    )
    
    return model, tokenizer


def test_get_model():
    """测试get_model函数的功能
    
    测试要点:
    1. 检查模型和分词器是否正确加载
    2. 检查模型结构是否符合配置
    3. 检查LoRA配置是否正确注入
    4. 检查输出维度是否正确
    """
    # 准备测试用的配置文件路径
    config_path = "/public/home/lingwang/wow/PepLoe/src/model_config.json"
    
    try:
        # 调用get_model函数
        model, tokenizer = get_model(config_path)
        
        # 测试1: 检查返回值类型
        assert isinstance(model, ESMWithClassification), "模型类型不正确"
        assert isinstance(tokenizer, PreTrainedTokenizer), "分词器类型不正确"
        
        # 测试2: 检查模型结构
        config = load_config(config_path)
        assert model.num_classes == config['model']['num_classes'], "类别数量配置不正确"
        assert model.dropout_rate == config['model']['hidden_dropout'], "dropout率配置不正确"
        
        # 测试3: 检查LoRA配置
        lora_params = {n for n, _ in model.base_model.named_parameters() if "lora" in n.lower()}
        assert len(lora_params) > 0, "LoRA参数未被正确注入"
        
        # 测试4: 检查输出维度
        batch_size, seq_len = 2, 10
        vocab_size = tokenizer.vocab_size
        dummy_input = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        }
        
        # 测试不输出attention的情况
        output = model(**dummy_input, output_attentions=False)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        assert logits.shape == (batch_size, seq_len, config['model']['num_classes']), \
            f"输出维度不正确，期望 {(batch_size, seq_len, config['model']['num_classes'])}，实际 {logits.shape}"
        
        # 测试输出attention的情况
        output_with_attention = model(**dummy_input, output_attentions=True)
        if isinstance(output_with_attention, dict):
            logits = output_with_attention['logits']
            attentions = output_with_attention.get('attentions', None)
            assert attentions is not None, "未能获取attention weights"
            print(f"成功获取 {len(attentions)} 层的attention weights")
        else:
            logits = output_with_attention
        
        print("所有测试通过!")
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_get_model() 
    