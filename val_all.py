#!/usr/bin/env python3
"""
批量验证脚本 - 对四个数据集进行验证并汇总结果
方便核对指标
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加 PPI-site/src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PPI-site', 'src'))

from utils import set_seed, load_config
from metrics_utils import compute_metrics, apply_valid_mask
from model_utils import load_base_model_and_tokenizer, create_peft_model
from train_utils import get_criterion
from data_utils import get_test_loader, DEFAULT_DATA_DIR
from more import (
    pHLAMoRA1_0Fusion,
    replace_lora_with_dynamic,
    check_lora_keys
)

# 四个数据集名称
DEFAULT_DATASETS = ["bitenet", "interpep", "pepbind", "pepnn"]

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
    """构建模型 - 与原始more.py完全一致"""
    model_path = config['model_path']
    tokenizer, base_model = load_base_model_and_tokenizer(model_path)
    
    # 与原始more.py完全一致的lora_cfg配置
    lora_cfg = config.get('lora', {
        'r': r_value,
        'lora_alpha': 16,
        'target_modules': ['key', 'value'],
        'lora_dropout': 0.3,
        'bias': 'none'
    })
    print("   类别数: 2")
    
    # 检查LoRA keys（与原始more.py完全一致，顺序也要一致）
    print("\n[调试] LoRA权重文件列表:")
    for i, path in enumerate(lora_paths):
        print(f"  {i}: {path}")
    print("[调试] LoRA keys数量:", len(lora_keys))
    check_lora_keys(lora_paths, lora_keys)
    
    peft_model = create_peft_model(base_model, lora_cfg)
    
    # 创建fusion模块（与原始more.py完全一致）
    # 重要：现在固定使用4个LoRA和固定r，不再根据数量动态调整
    if len(lora_paths) != 4:
        raise ValueError(f"当前实现只支持4个LoRA，但配置中给出了 {len(lora_paths)} 个: {lora_paths}")
    fusion_r = r_value
    
    print(f" LoRA注入配置:")
    print(f"  LoRA数量(固定): {len(lora_paths)}")
    print(f"  设定r值: {r_value}")
    print(f"  实际r值(固定): {fusion_r}")
    
    fusion = pHLAMoRA1_0Fusion(lora_paths, lora_keys, r=fusion_r, device=device)
    model = ESMWithSequenceLabeling(peft_model, fusion, base_model.config.hidden_size, 2)
    model = model.to(device)
    
    return tokenizer, model

def load_checkpoint(model, save_dir, device):
    """加载checkpoint - 与原始more.py的测试逻辑完全一致"""
    print("加载最后一个epoch的模型进行验证（与more.py测试逻辑一致）...")
    
    # 关键说明：
    # more.py在测试时，model对象一直存在，base_model和peft_model的状态是最后一个epoch的状态
    # more.py只重新加载了input_loras、fusion.pth和classifier.pth（都是最后一个epoch的状态）
    # 所以more.py测试时使用的是最后一个epoch的完整模型状态（last_model）
    # 
    # val_all.py从头构建模型，需要恢复完整的last模型状态
    # 注意：more.py目前没有保存last_model.pth，只保存了best_model.pth（best epoch的状态）
    # 因此val_all.py优先尝试加载last_model.pth（如果存在），否则回退到best_model.pth
    # 然后加载fusion.pth和classifier.pth（最后一个epoch的状态）来覆盖fusion和classifier
    # 这样base_model和peft_model可能来自best epoch，fusion和classifier来自last epoch
    # 如果最后一个epoch恰好是best epoch，则完全一致；否则会有轻微差异
    
    # 优先加载last_model.pth（如果存在），否则回退到best_model.pth
    last_model_path = os.path.join(save_dir, 'last_model.pth')
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    if os.path.exists(last_model_path):
        print(f"  发现last_model.pth，加载完整的last模型状态（包含base_model和peft_model）...")
        try:
            last_state = torch.load(last_model_path, map_location=device, weights_only=False)
            model.load_state_dict(last_state, strict=False)  # strict=False允许部分匹配
            print(f"  成功加载last_model.pth的完整状态")
        except Exception as e:
            print(f"  警告: 加载last_model.pth失败: {e}，尝试使用best_model.pth")
            if os.path.exists(best_model_path):
                best_state = torch.load(best_model_path, map_location=device, weights_only=False)
                model.load_state_dict(best_state, strict=False)
                print(f"  已回退到best_model.pth")
            else:
                print(f"  警告：best_model.pth也不存在，base_model和peft_model将使用初始状态")
    elif os.path.exists(best_model_path):
        print(f"  未找到last_model.pth，使用best_model.pth作为回退...")
        print(f"  警告：这将导致base_model和peft_model使用best epoch的状态，而非last epoch")
        try:
            best_state = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(best_state, strict=False)
            print(f"  成功加载best_model.pth")
        except Exception as e:
            print(f"  警告: 加载best_model.pth失败: {e}，base_model和peft_model将使用初始状态")
    else:
        print(f"  警告：未找到last_model.pth和best_model.pth，base_model和peft_model将使用初始状态")
    
    # 加载每个输入LoRA权重（与原始more.py完全一致）
    input_loras_dir = os.path.join(save_dir, 'input_loras')
    if os.path.exists(input_loras_dir):
        model.fusion.adaptive_fusion.lora_states = []
        lora_files = sorted([f for f in os.listdir(input_loras_dir) if f.startswith('input_lora_') and f.endswith('.pth')])
        for lora_file in lora_files:
            lora_path = os.path.join(input_loras_dir, lora_file)
            lora_state = torch.load(lora_path, weights_only=False)
            model.fusion.adaptive_fusion.lora_states.append(lora_state)
            print(f"  加载输入LoRA权重: {lora_path}")
        print(f"  总共加载了 {len(model.fusion.adaptive_fusion.lora_states)} 个输入LoRA权重")
    else:
        print(f"  警告: input_loras目录不存在: {input_loras_dir}")
        print(f"  将使用fusion模块初始化时加载的LoRA权重")
    
    # 加载fusion模块权重（MLP参数和权重矩阵）（来自last epoch，会覆盖last_model.pth或best_model.pth中的fusion部分）
    fusion_checkpoint = torch.load(os.path.join(save_dir, 'fusion.pth'), weights_only=False)
    model.fusion.load_state_dict(fusion_checkpoint['fusion_state_dict'])
    
    # 恢复fusion模块的其他属性（与原始more.py完全一致）
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
    if 'is_initialized' in fusion_checkpoint:
        # 标记PCA已初始化，避免重复计算
        model.fusion.adaptive_fusion.is_initialized = fusion_checkpoint['is_initialized']
    
    print(f"  加载fusion模块权重: {os.path.join(save_dir, 'fusion.pth')}")
    print(f"  PCA缓存状态: {'已初始化' if len(model.fusion.adaptive_fusion.V_r_cache) > 0 else '未初始化'}")
    
    # 加载分类头权重（来自last epoch，会覆盖last_model.pth或best_model.pth中的classifier部分）
    classifier_checkpoint = torch.load(os.path.join(save_dir, 'classifier.pth'), weights_only=False)
    classifier_state = classifier_checkpoint['classifier_state_dict']
    
    # 只加载分类头参数（与原始more.py完全一致）
    model_state = model.state_dict()
    for name, param in classifier_state.items():
        if name in model_state:
            model_state[name] = param
    model.load_state_dict(model_state)
    
    print(f"  加载分类头权重: {os.path.join(save_dir, 'classifier.pth')}")
    
    # 打印保存的配置信息（与原始more.py完全一致）
    if 'seed' in fusion_checkpoint:
        print(f"  保存的随机数种子: {fusion_checkpoint['seed']}")
    if 'config' in fusion_checkpoint:
        config = fusion_checkpoint['config']
        print(f"  保存的训练配置: epochs={config.get('epochs', 'N/A')}, batch_size={config.get('batch_size', 'N/A')}, lr={config.get('lr', 'N/A')}, r={config.get('r', 'N/A')}")

def validate_dataset(model, tokenizer, dataset_name, data_dir, batch_size, device):
    """对单个数据集进行验证 - 与原始more.py的测试逻辑完全一致"""
    
    # 构建数据路径
    csv_path = os.path.join(data_dir, f"{dataset_name}_test.csv")
    
    if not os.path.exists(csv_path):
        print(f"  警告: 数据文件不存在: {csv_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"验证数据集: {dataset_name}")
    print(f"  数据路径: {csv_path}")
    print(f"{'='*60}")
    
    # 加载数据
    test_loader = get_test_loader(csv_path, tokenizer, batch_size)
    
    # 创建损失函数（仅用于计算loss，不影响其他指标）
    criterion = get_criterion(device=device)
    
    # 验证 - 完全按照原始more.py的测试逻辑
    # 注意：model.eval()在main()中已经调用，这里为了与more.py完全一致也调用一次
    model.eval()
    test_true, test_pred, test_prob, test_loss = [], [], [], 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Val {dataset_name}"):
            input_ids, prot_masks, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, prot_masks=prot_masks)
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
    
    test_loss /= len(test_loader)
    y_true = np.array(test_true)
    y_prob = np.array(test_prob)
    test_metrics = compute_metrics(y_true, np.array(test_pred), y_prob)
    test_metrics['test_loss'] = test_loss
    
    # 添加数据集名称
    result = {
        'dataset': dataset_name,
        'loss': test_loss,
        **{k: v for k, v in test_metrics.items()}
    }
    
    print(f"\n{dataset_name} 验证结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Acc: {test_metrics.get('test_acc', 0.0):.4f}")
    print(f"  AUC: {test_metrics.get('test_auc', 0.0):.4f}")
    print(f"  PR-AUC: {test_metrics.get('test_pr_auc', 0.0):.4f}")
    if 'test_f1' in test_metrics:
        print(f"  F1: {test_metrics.get('test_f1', 0.0):.4f}")
    if 'test_precision' in test_metrics:
        print(f"  Precision: {test_metrics.get('test_precision', 0.0):.4f}")
    if 'test_recall' in test_metrics:
        print(f"  Recall: {test_metrics.get('test_recall', 0.0):.4f}")
    
    return result

def main():
    # 默认配置
    config_path = 'config.json'  # 相对于PPI-site/src目录
    save_dir = 'PPI-site/results/more'  # 默认保存目录
    batch_size = 2
    data_dir = DEFAULT_DATA_DIR
    datasets = DEFAULT_DATASETS
    r_value = 4
    seed = 42  # 默认seed，如果checkpoint中有则使用checkpoint中的
    
    # 先读取配置，以便从checkpoint中读取seed
    full_cfg = load_config(config_path)
    common_cfg = full_cfg.get('common', {})
    more_cfg = full_cfg.get('more', {})
    config = {**common_cfg, **more_cfg}
    
    # 从配置中读取save_dir（如果配置中有）
    if 'save_dir' in more_cfg:
        save_dir = more_cfg['save_dir']
    
    # 从配置中读取r值（如果配置中有）
    if 'r' in more_cfg:
        r_value = more_cfg['r']
    
    # 关键：先从checkpoint中读取seed并设置（在构建模型之前）
    # 这样可以确保模型初始化和数据加载都使用相同的随机种子
    fusion_checkpoint_path = os.path.join(save_dir, 'fusion.pth')
    if os.path.exists(fusion_checkpoint_path):
        fusion_checkpoint = torch.load(fusion_checkpoint_path, weights_only=False)
        if 'seed' in fusion_checkpoint:
            seed = fusion_checkpoint['seed']
            print(f"从checkpoint读取随机数种子: {seed}")
        else:
            print(f"checkpoint中未找到seed，使用默认seed: {seed}")
    else:
        print(f"未找到fusion.pth，使用默认seed: {seed}")
    
    # 设置随机种子（在构建模型之前）
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取LoRA路径
    lora_paths = config['lora_paths']
    lora_keys = config['lora_keys']
    
    # 构建模型（只构建一次）
    print("=" * 60)
    print("构建模型...")
    print("=" * 60)
    tokenizer, model = build_model(config, lora_paths, lora_keys, r_value, device)
    
    # 加载checkpoint（只加载一次）
    print("\n" + "=" * 60)
    print("加载最佳模型进行验证...")
    print("=" * 60)
    
    load_checkpoint(model, save_dir, device)
    
    # 确保模型处于eval模式（与原始more.py保持一致）
    # 注意：在more.py中，测试时模型已经训练完成，所以已经是eval模式
    # 但为了确保一致性，这里显式调用一次
    model.eval()
    
    # 确保所有参数都不需要梯度（与测试时一致）
    for param in model.parameters():
        param.requires_grad = False
    
    # 对每个数据集进行验证
    print("\n" + "=" * 60)
    print(f"开始批量验证 ({len(datasets)} 个数据集)")
    print("=" * 60)
    
    all_results = []
    
    for dataset_name in datasets:
        try:
            result = validate_dataset(
                model, tokenizer, dataset_name, 
                data_dir, batch_size, device
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\n错误: 验证数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总结果并打印
    if all_results:
        print("\n" + "=" * 80)
        print("验证结果汇总")
        print("=" * 80)
        
        # 创建DataFrame
        df = pd.DataFrame(all_results)
        
        # 重新排列列顺序，将dataset放在第一列
        cols = ['dataset'] + [c for c in df.columns if c != 'dataset']
        df = df[cols]
        
        # 打印关键指标汇总表格
        print("\n关键指标汇总:")
        print("-" * 90)
        print(f"{'数据集':<12} {'Loss':<10} {'Acc':<10} {'AUC':<10} {'PR-AUC':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        print("-" * 90)
        for _, row in df.iterrows():
            print(f"{row['dataset']:<12} "
                  f"{row.get('loss', 0.0):<10.4f} "
                  f"{row.get('test_acc', 0.0):<10.4f} "
                  f"{row.get('test_auc', 0.0):<10.4f} "
                  f"{row.get('test_pr_auc', 0.0):<10.4f} "
                  f"{row.get('test_f1', 0.0):<10.4f} "
                  f"{row.get('test_precision', 0.0):<12.4f} "
                  f"{row.get('test_recall', 0.0):<10.4f}")
        print("-" * 90)
        
        # 打印详细指标（可选）
        print("\n详细指标:")
        print(df.to_string(index=False))
    else:
        print("\n警告: 没有成功验证任何数据集！")
    
    print("\n批量验证完成！")

if __name__ == "__main__":
    main()

