#!/usr/bin/env python3
"""
验证脚本 - 加载训练好的模型进行验证
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加 PPI-site/src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PPI-site', 'src'))

from utils import set_seed, load_config
from metrics_utils import compute_metrics, apply_valid_mask
from model_utils import load_base_model_and_tokenizer, create_peft_model
from train_utils import get_criterion, save_metrics
from data_utils import get_data_loaders, DEFAULT_DATA_DIR
from more import (
    pHLAMoRA1_0Fusion,
    replace_lora_with_dynamic,
    eval_model
)

class ESMWithSequenceLabeling(nn.Module):
    """ESM序列标注模型包装器"""
    def __init__(self, base_model, fusion, hidden_size, num_classes, dropout_rate=0.3):
        super().__init__()
        self.base_model = base_model
        self.fusion = fusion
        print("[模型初始化] 开始动态LoRA替换...")
        replaced_count = replace_lora_with_dynamic(self.base_model, self.fusion)
        print(f"[模型初始化] 动态LoRA替换完成，替换了{replaced_count}个LoRA层")
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, prot_masks=None, output_attentions=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        return type("Outputs", (), {"logits": logits})()

def build_model(config, lora_paths, lora_keys, r_value, device):
    """构建模型"""
    model_path = config['model_path']
    tokenizer, base_model = load_base_model_and_tokenizer(model_path)
    
    lora_cfg = config.get('lora', {
        'r': r_value,
        'lora_alpha': 16,
        'target_modules': ['key', 'value'],
        'lora_dropout': 0.3,
        'bias': 'none'
    })
    
    peft_model = create_peft_model(base_model, lora_cfg)
    fusion = pHLAMoRA1_0Fusion(lora_paths, lora_keys, r=r_value, device=device)
    model = ESMWithSequenceLabeling(peft_model, fusion, base_model.config.hidden_size, 2)
    model = model.to(device)
    
    return tokenizer, model

def load_checkpoint(model, save_dir, device):
    """加载checkpoint"""
    print("加载checkpoint...")
    
    # 加载输入LoRA权重
    input_loras_dir = os.path.join(save_dir, 'input_loras')
    if os.path.exists(input_loras_dir):
        model.fusion.adaptive_fusion.lora_states = []
        lora_files = sorted([
            f for f in os.listdir(input_loras_dir) 
            if f.startswith('input_lora_') and f.endswith('.pth')
        ])
        for lora_file in lora_files:
            lora_path = os.path.join(input_loras_dir, lora_file)
            lora_state = torch.load(lora_path, weights_only=False)
            model.fusion.adaptive_fusion.lora_states.append(lora_state)
            print(f"  加载输入LoRA权重: {lora_path}")
        print(f"  总共加载了 {len(model.fusion.adaptive_fusion.lora_states)} 个输入LoRA权重")
    else:
        print(f"  警告: input_loras目录不存在: {input_loras_dir}")
        print(f"  将使用fusion模块初始化时加载的LoRA权重")
    
    # 加载fusion模块权重
    fusion_checkpoint = torch.load(
        os.path.join(save_dir, 'fusion.pth'), 
        weights_only=False
    )
    model.fusion.load_state_dict(fusion_checkpoint['fusion_state_dict'])
    
    # 恢复fusion模块的其他属性
    if 'V_r_cache' in fusion_checkpoint:
        model.fusion.adaptive_fusion.V_r_cache = fusion_checkpoint['V_r_cache']
    if 'v_mean_cache' in fusion_checkpoint:
        model.fusion.adaptive_fusion.v_mean_cache = fusion_checkpoint['v_mean_cache']
    if 'Z_init_cache' in fusion_checkpoint:
        model.fusion.adaptive_fusion.Z_init_cache = fusion_checkpoint['Z_init_cache']
    if 'detected_r' in fusion_checkpoint:
        model.fusion.adaptive_fusion.detected_r = fusion_checkpoint['detected_r']
    if 'hidden_dims' in fusion_checkpoint:
        model.fusion.adaptive_fusion.hidden_dims = fusion_checkpoint['hidden_dims']
    
    print(f"  加载fusion模块权重: {os.path.join(save_dir, 'fusion.pth')}")
    
    # 加载分类头权重
    classifier_checkpoint = torch.load(
        os.path.join(save_dir, 'classifier.pth'), 
        weights_only=False
    )
    classifier_state = classifier_checkpoint['classifier_state_dict']
    
    model_state = model.state_dict()
    for name, param in classifier_state.items():
        if name in model_state:
            model_state[name] = param
    model.load_state_dict(model_state)
    
    print(f"  加载分类头权重: {os.path.join(save_dir, 'classifier.pth')}")
    
    # 打印保存的配置信息
    if 'seed' in fusion_checkpoint:
        print(f"  保存的随机数种子: {fusion_checkpoint['seed']}")
    if 'config' in fusion_checkpoint:
        cfg = fusion_checkpoint['config']
        print(f"  保存的训练配置: epochs={cfg.get('epochs', 'N/A')}, "
              f"batch_size={cfg.get('batch_size', 'N/A')}, "
              f"lr={cfg.get('lr', 'N/A')}, r={cfg.get('r', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(description='验证脚本')
    parser.add_argument('--config', type=str, default='PPI-site/src/config.json', help='配置文件路径')
    parser.add_argument('--save_dir', type=str, required=True, help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default=None, help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--val_csv', type=str, default=None, help='验证数据路径')
    parser.add_argument('--r', type=int, default=4, help='PCA主成分数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 读取配置
    full_cfg = load_config(args.config)
    common_cfg = full_cfg.get('common', {})
    more_cfg = full_cfg.get('more', {})
    config = {**common_cfg, **more_cfg}
    
    if args.result_dir is None:
        args.result_dir = os.path.join(args.save_dir, 'val_results')
    os.makedirs(args.result_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取LoRA路径
    lora_paths = config['lora_paths']
    lora_keys = config['lora_keys']
    
    # 构建模型
    print("=" * 60)
    print("构建模型...")
    print("=" * 60)
    tokenizer, model = build_model(config, lora_paths, lora_keys, args.r, device)
    
    # 加载checkpoint
    print("\n" + "=" * 60)
    print("加载checkpoint...")
    print("=" * 60)
    load_checkpoint(model, args.save_dir, device)
    
    # 准备验证数据
    _, val_loader = get_data_loaders(
        tokenizer, args.batch_size, None, args.val_csv, seed=args.seed
    )
    
    criterion = get_criterion(device=device, csv_path=args.val_csv)
    
    # 验证
    print("\n" + "=" * 60)
    print("开始验证...")
    print("=" * 60)
    
    model.eval()
    val_loss, val_metrics = eval_model(model, val_loader, criterion, device, 0, "Val")
    
    print(f"\n验证结果:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Acc: {val_metrics.get('test_acc', 0.0):.4f}")
    print(f"  AUC: {val_metrics.get('test_auc', 0.0):.4f}")
    print(f"  PR-AUC: {val_metrics.get('test_pr_auc', 0.0):.4f}")
    
    # 保存验证指标
    val_save_path = os.path.join(args.result_dir, 'val_metrics.csv')
    save_metrics(val_metrics, val_save_path)
    print(f"\n验证指标已保存到: {val_save_path}")
    print("验证完成！")

if __name__ == "__main__":
    main()

