#!/usr/bin/env python3
"""
测试脚本 - 加载训练好的模型进行测试
支持多个测试集
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加 PPI-site/src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PPI-site', 'src'))

from utils import set_seed, load_config
from metrics_utils import compute_metrics, apply_valid_mask
from model_utils import load_base_model_and_tokenizer, create_peft_model
from train_utils import get_criterion, save_metrics, save_auc_variance, save_roc_pr_curves
from data_utils import get_test_loader, DEFAULT_DATA_DIR
from more import (
    pHLAMoRA1_0Fusion,
    replace_lora_with_dynamic
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
    parser = argparse.ArgumentParser(description='测试脚本')
    parser.add_argument('--config', type=str, default='PPI-site/src/config.json', help='配置文件路径')
    parser.add_argument('--save_dir', type=str, required=True, help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default=None, help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--test_csvs', nargs='+', default=None, help='测试数据路径（多个）')
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
        args.result_dir = os.path.join(args.save_dir, 'test_results')
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'test'), exist_ok=True)
    
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
    
    # 处理测试数据集
    if args.test_csvs:
        if isinstance(args.test_csvs, str):
            test_csvs = [args.test_csvs]
        else:
            test_csvs = args.test_csvs
        print(f" 使用命令行参数指定的测试数据集: {test_csvs}")
    else:
        # 默认4个测试集
        test_datasets = ["bitenet", "interpep", "pepbind", "pepnn"]
        test_csvs = [os.path.join(DEFAULT_DATA_DIR, f"{ds}_test.csv") for ds in test_datasets]
        print(f" 使用默认测试数据集: {test_csvs}")
    
    criterion = get_criterion(device=device)
    
    # 测试
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)
    
    for i, test_csv_path in enumerate(test_csvs):
        try:
            # 从路径中提取数据集名称
            dataset_name = os.path.basename(test_csv_path).replace('_test.csv', '').replace('.csv', '')
            print(f"\n测试数据集 {i+1}/{len(test_csvs)}: {dataset_name}")
            print(f"  路径: {test_csv_path}")
            
            test_loader = get_test_loader(test_csv_path, tokenizer, args.batch_size)
            
            model.eval()
            test_true, test_pred, test_prob, test_loss = [], [], [], 0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Test {dataset_name}"):
                    input_ids, prot_masks, attention_mask, labels = [x.to(device) for x in batch]
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        prot_masks=prot_masks
                    )
                    logits = outputs.logits  # [batch, seq_len, 2]
                    
                    # 使用统一掩码，只在蛋白质且非 padding 的位置计算
                    valid_logits, valid_labels, _ = apply_valid_mask(
                        logits, prot_masks, attention_mask, labels
                    )
                    if len(valid_labels) > 0:
                        loss = criterion(valid_logits, valid_labels)
                        test_loss += loss.item()
                        
                        preds = torch.argmax(valid_logits, dim=-1).detach().cpu().numpy()
                        probs = torch.softmax(valid_logits, dim=-1)[:, 1].detach().cpu().numpy()
                        test_pred.extend(preds)
                        test_prob.extend(probs)
                        test_true.extend(valid_labels.detach().cpu().numpy())
            
            test_loss /= len(test_loader) if len(test_loader) > 0 else 1
            y_true = np.array(test_true)
            y_prob = np.array(test_prob)
            test_metrics = compute_metrics(y_true, np.array(test_pred), y_prob)
            test_metrics['test_loss'] = test_loss
            
            # 为每个测试集创建单独的输出目录
            ds_result_dir = os.path.join(args.result_dir, 'test', dataset_name)
            os.makedirs(ds_result_dir, exist_ok=True)
            test_save_path = os.path.join(ds_result_dir, 'test_metrics.csv')
            save_metrics(test_metrics, test_save_path)
            
            # 计算并保存AUC方差估计
            auc_variance_path = os.path.join(ds_result_dir, 'auc_variance.txt')
            auc_delong, se_delong, ci_lower_delong, ci_upper_delong, \
            auc_hanley, se_hanley, ci_lower_hanley, ci_upper_hanley = save_auc_variance(
                y_true, y_prob, auc_variance_path
            )
            
            # 生成并保存ROC和PR曲线
            save_roc_pr_curves(y_true, y_prob, ds_result_dir, dataset_name, test_metrics)
            
            print(f"\n{dataset_name} 测试结果:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Acc: {test_metrics['test_acc']:.4f}")
            print(f"  AUC: {test_metrics['test_auc']:.4f}")
            print(f"  PR-AUC: {test_metrics.get('test_pr_auc', 0.0):.4f}")
            print(f"  DeLong AUC: {auc_delong:.4f} ± {se_delong:.4f}")
            print(f"  Hanley&MCN AUC: {auc_hanley:.4f} ± {se_hanley:.4f}")
            print(f"  结果已保存到: {ds_result_dir}")
            
        except Exception as e:
            print(f"测试数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

