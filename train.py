#!/usr/bin/env python3
"""
训练脚本 - 基于 more.py 的训练流程
支持分阶段训练：预训练阶段 + 联合训练阶段
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加 PPI-site/src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PPI-site', 'src'))

from utils import set_seed, load_config
from metrics_utils import compute_metrics, apply_valid_mask
from model_utils import load_base_model_and_tokenizer, create_peft_model
from train_utils import get_criterion, save_metrics
from data_utils import get_data_loaders, get_test_loader, DEFAULT_DATA_DIR
from more import (
    pHLAMoRA1_0Fusion,
    replace_lora_with_dynamic,
    freeze_fusion_lora,
    unfreeze_fusion_lora,
    train_one_epoch,
    eval_model,
    check_lora_keys
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
    
    print(f" LoRA注入配置:")
    print(f"  LoRA数量(固定): {len(lora_paths)}")
    print(f"  设定r值: {r_value}")
    
    if len(lora_paths) != 4:
        raise ValueError(f"当前实现只支持4个LoRA，但配置中给出了 {len(lora_paths)} 个")
    
    check_lora_keys(lora_paths, lora_keys)
    peft_model = create_peft_model(base_model, lora_cfg)
    
    fusion = pHLAMoRA1_0Fusion(lora_paths, lora_keys, r=r_value, device=device)
    
    model = ESMWithSequenceLabeling(peft_model, fusion, base_model.config.hidden_size, 2)
    model = model.to(device)
    
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--config', type=str, default='PPI-site/src/config.json', help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--pretrain_epochs', type=int, default=2, help='新LoRA预训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='联合训练阶段学习率')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='预训练阶段学习率')
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录')
    parser.add_argument('--result_dir', type=str, default=None, help='结果目录')
    parser.add_argument('--r', type=int, default=4, help='PCA主成分数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--train_csv', type=str, default=None, help='训练数据路径')
    parser.add_argument('--val_csv', type=str, default=None, help='验证数据路径')
    parser.add_argument('--test_csv', type=str, default=None, help='测试数据路径（用于监控）')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 读取配置
    full_cfg = load_config(args.config)
    common_cfg = full_cfg.get('common', {})
    more_cfg = full_cfg.get('more', {})
    config = {**common_cfg, **more_cfg}
    
    # 设置保存路径
    if args.save_dir is None:
        args.save_dir = config.get('save_dir', './results/more')
    if args.result_dir is None:
        args.result_dir = config.get('result_dir', './results/more')
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'train'), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取LoRA路径
    lora_paths = config['lora_paths']
    lora_keys = config['lora_keys']
    
    # 构建模型
    print("=" * 60)
    print("构建模型...")
    print("=" * 60)
    tokenizer, model = build_model(config, lora_paths, lora_keys, args.r, device)
    
    # 准备数据
    train_loader, val_loader = get_data_loaders(
        tokenizer, args.batch_size, args.train_csv, args.val_csv, seed=args.seed
    )
    
    # 加载测试集用于监控
    if args.test_csv:
        test_loader = get_test_loader(args.test_csv, tokenizer, args.batch_size)
    else:
        test_loader = get_test_loader(
            os.path.join(DEFAULT_DATA_DIR, 'merged_test.csv'), 
            tokenizer, args.batch_size
        )
    
    criterion = get_criterion(device=device, csv_path=args.train_csv)
    
    print("\n" + "=" * 60)
    print("开始分阶段训练...")
    print(f" 学习率设置:")
    print(f"  预训练阶段学习率: {args.pretrain_lr}")
    print(f"  联合训练阶段学习率: {args.lr}")
    print("=" * 60)
    
    best_val_auc = 0
    best_val_acc = 0
    train_metrics_list = []
    
    # 阶段1：新LoRA预训练
    print(f"\n{'='*60}")
    print(f" 阶段1：新LoRA预训练 (最多{args.pretrain_epochs}个epoch)")
    print(f"{'='*60}")
    
    freeze_fusion_lora(model)
    pretrain_optimizer = optim.AdamW(model.parameters(), lr=args.pretrain_lr)
    best_pretrain_val_auc = 0.0
    
    for epoch in range(args.pretrain_epochs):
        print(f"\n 预训练阶段 - Epoch {epoch+1}/{args.pretrain_epochs}")
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, pretrain_optimizer, criterion, device, epoch
        )
        val_loss, val_metrics = eval_model(
            model, val_loader, criterion, device, epoch, "Val"
        )
        test_loss, test_metrics = eval_model(
            model, test_loader, criterion, device, epoch, "Test"
        )
        
        current_val_auc = val_metrics.get('test_auc', 0.0)
        current_val_acc = val_metrics.get('test_acc', 0.0)
        
        print(f"  预训练 Epoch {epoch+1}/{args.pretrain_epochs}:")
        print(f"    Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"    Val Acc: {current_val_acc:.4f}, Val AUC: {current_val_auc:.4f}")
        
        if current_val_auc > best_pretrain_val_auc:
            best_pretrain_val_auc = current_val_auc
            best_val_auc = current_val_auc
            best_val_acc = current_val_acc
            torch.save(
                model.state_dict(), 
                os.path.join(args.save_dir, 'best_pretrain_model.pth')
            )
            print(f"     验证AUC提升！新的最佳预训练AUC: {best_pretrain_val_auc:.4f}")
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'stage': 'pretrain',
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            **{f'train_{k[5:]}': v for k, v in train_metrics.items()},
            **{f'val_{k[5:]}': v for k, v in val_metrics.items()},
            **{f'test_{k[5:]}': v for k, v in test_metrics.items()}
        }
        train_metrics_list.append(epoch_metrics)
    
    # 阶段2：联合训练
    actual_pretrain_epochs = len([m for m in train_metrics_list if m['stage'] == 'pretrain'])
    remaining_epochs = args.epochs - actual_pretrain_epochs
    
    print(f"\n{'='*60}")
    print(f" 阶段2：联合训练 (最多{remaining_epochs}个epoch)")
    print(f"{'='*60}")
    
    # 加载预训练最佳模型
    best_pretrain_model_path = os.path.join(args.save_dir, 'best_pretrain_model.pth')
    if os.path.exists(best_pretrain_model_path):
        print(f" 加载预训练最佳模型: {best_pretrain_model_path}")
        try:
            model.load_state_dict(torch.load(best_pretrain_model_path, map_location=device))
            print("  预训练最佳模型加载成功")
        except Exception as e:
            print(f"  预训练最佳模型加载失败: {e}")
    
    unfreeze_fusion_lora(model)
    joint_optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    for epoch in range(actual_pretrain_epochs, args.epochs):
        joint_epoch = epoch - actual_pretrain_epochs + 1
        print(f"\n 联合训练阶段 - Epoch {joint_epoch}/{remaining_epochs}")
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, joint_optimizer, criterion, device, epoch
        )
        val_loss, val_metrics = eval_model(
            model, val_loader, criterion, device, epoch, "Val"
        )
        test_loss, test_metrics = eval_model(
            model, test_loader, criterion, device, epoch, "Test"
        )
        
        current_val_auc = val_metrics.get('test_auc', 0.0)
        current_val_acc = val_metrics.get('test_acc', 0.0)
        
        print(f"  联合训练 Epoch {joint_epoch}/{remaining_epochs}:")
        print(f"    Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"    Val Acc: {current_val_acc:.4f}, Val AUC: {current_val_auc:.4f}")
        
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_val_acc = current_val_acc
            torch.save(
                model.state_dict(), 
                os.path.join(args.save_dir, 'best_model.pth')
            )
            print(f"     验证AUC提升！新的最佳联合训练AUC: {best_val_auc:.4f}")
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'stage': 'joint',
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            **{f'train_{k[5:]}': v for k, v in train_metrics.items()},
            **{f'val_{k[5:]}': v for k, v in val_metrics.items()},
            **{f'test_{k[5:]}': v for k, v in test_metrics.items()}
        }
        train_metrics_list.append(epoch_metrics)
    
    # 保存最终模型
    print(f"\n{'='*60}")
    print("保存最终模型...")
    print(f"{'='*60}")
    
    # 保存输入LoRA权重
    input_loras_dir = os.path.join(args.save_dir, 'input_loras')
    os.makedirs(input_loras_dir, exist_ok=True)
    for i, lora_state in enumerate(model.fusion.adaptive_fusion.lora_states):
        lora_path = os.path.join(input_loras_dir, f'input_lora_{i}.pth')
        torch.save(lora_state, lora_path)
        print(f"保存输入LoRA {i} 权重到: {lora_path}")
    
    # 保存fusion模块
    fusion_save_dict = {
        'fusion_state_dict': model.fusion.state_dict(),
        'adaptive_fusion_state_dict': model.fusion.adaptive_fusion.state_dict(),
        'new_lora_state_dict': model.fusion.new_lora.state_dict() if model.fusion.new_lora else {},
        'lora_keys': model.fusion.adaptive_fusion.lora_keys,
        'N': model.fusion.adaptive_fusion.N,
        'r': model.fusion.adaptive_fusion.r,
        'detected_r': model.fusion.adaptive_fusion.detected_r,
        'hidden_dims': model.fusion.adaptive_fusion.hidden_dims,
        'V_r_cache': model.fusion.adaptive_fusion.V_r_cache,
        'v_mean_cache': model.fusion.adaptive_fusion.v_mean_cache,
        'Z_init_cache': model.fusion.adaptive_fusion.Z_init_cache,
        'is_initialized': True,
        'epoch': args.epochs,
        'val_acc': best_val_acc,
        'val_auc': best_val_auc,
        'seed': args.seed,
        'config': {
            'epochs': args.epochs,
            'pretrain_epochs': args.pretrain_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'pretrain_lr': args.pretrain_lr,
            'r': args.r,
        }
    }
    torch.save(fusion_save_dict, os.path.join(args.save_dir, 'fusion.pth'))
    print(f"保存fusion模块到: {os.path.join(args.save_dir, 'fusion.pth')}")
    
    # 保存分类头
    classifier_state = {}
    for name, param in model.named_parameters():
        if 'classifier' in name.lower() or 'head' in name.lower():
            classifier_state[name] = param.data
    
    torch.save({
        'classifier_state_dict': classifier_state,
        'epoch': args.epochs,
        'val_acc': best_val_acc
    }, os.path.join(args.save_dir, 'classifier.pth'))
    print(f"保存分类头到: {os.path.join(args.save_dir, 'classifier.pth')}")
    
    # 保存训练指标
    train_df = pd.DataFrame(train_metrics_list)
    train_df.to_csv(
        os.path.join(args.result_dir, 'train', 'training_metrics.csv'), 
        index=False
    )
    print(f"训练指标已保存到 {os.path.join(args.result_dir, 'train', 'training_metrics.csv')}")
    print("训练完成！")

if __name__ == "__main__":
    main()

