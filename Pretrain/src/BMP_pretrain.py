import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
pep_root = os.path.abspath(os.path.join(project_root, '../'))
src_path = os.path.join(pep_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 导入主模型相关模块
try:
    from get_model import get_model
    from model_saver import LoRAModelSaver
    print("成功导入主模型模块")
except ImportError as e:
    print(f"导入主模型模块失败: {e}")
    sys.exit(1)

# =====================
# 自定义Dataset
# =====================
class MaskedProteinDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=1024, mask_ratio=0.50):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        
        # 初始化时检查数据中的氨基酸类型
        print(f"检查数据集中的氨基酸类型...")
        all_aas = set()
        for _, row in self.data.head(100).iterrows():  # 只检查前100个样本
            # 数据生成文件格式：mask_seq, label
            mask_seq = str(row.get('mask_seq', ''))
            label_seq = str(row.get('label', ''))
            all_aas.update(mask_seq.replace('#', ''))
            all_aas.update(label_seq.replace('#', ''))
        
        print(f"  发现的所有氨基酸: {sorted(all_aas)}")
        print(f"  tokenizer词汇表大小: {self.tokenizer.vocab_size}")
        
        # 检查哪些氨基酸可能有问题
        problematic_aas = []
        for aa in all_aas:
            token_id = self.tokenizer.convert_tokens_to_ids(aa)
            if token_id is None or token_id < 0 or token_id >= self.tokenizer.vocab_size:
                problematic_aas.append(aa)
        
        if problematic_aas:
            print(f"发现可能有问题的氨基酸: {problematic_aas}")
            print("  这些氨基酸将被忽略（设为-100）")
        else:
            print("所有氨基酸都在有效范围内")
        print("---")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 数据生成文件格式：mask_seq (已掩码的序列), label (原始序列作为标签)
        mask_seq = str(row['mask_seq'])  # 已经掩码的序列
        label_seq = str(row['label'])    # 原始序列（作为标签）
        
        # 去除padding符号
        mask_seq = mask_seq.replace('#', '')
        actual_seq = label_seq.replace('#', '')  # 原始序列，用于生成标签
        
        # 数据生成文件使用'X'作为mask token，需要转换为ESM的mask token '<mask>'
        # 同时需要处理随机氨基酸替换的情况（10%概率）
        mask_seq_processed = mask_seq.replace('X', '<mask>')
        
        # 使用ESM tokenizer编码序列
        encoding = self.tokenizer(
            mask_seq_processed,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # 不padding
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 创建标签序列 - 只在掩码位置保留真实标签
        label_ids = torch.full((input_ids.size(0),), -100, dtype=torch.long)
        
        # 找到tokenizer中的mask token id
        mask_token_id = self.tokenizer.mask_token_id
        
        # 关键：确保索引一致性的正确位置映射
        # 方法：通过比较原始序列和掩码序列的tokenization来建立位置映射
        
        # 1. 先编码原始序列（无掩码）
        original_encoding = self.tokenizer(
            actual_seq,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors='pt'
        )
        original_input_ids = original_encoding['input_ids'].squeeze(0)
        original_attention_mask = original_encoding['attention_mask'].squeeze(0)
        
        # 2. 建立位置映射：mask_seq的tokenized位置 -> 原始序列位置
        # 对于ESM tokenizer，通常每个氨基酸对应一个token（除了特殊token）
        special_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            special_tokens.append(self.tokenizer.eos_token_id)
        
        # 找到原始序列中每个氨基酸在tokenized序列中的位置
        original_seq_to_tokenized = {}
        token_idx = 0
        for i, token_id in enumerate(original_input_ids):
            if token_id not in special_tokens and original_attention_mask[i] == 1:
                if token_idx < len(actual_seq):
                    original_seq_to_tokenized[token_idx] = i
                    token_idx += 1
        
        # 找到掩码序列中每个氨基酸在tokenized序列中的位置
        mask_seq_to_tokenized = {}
        token_idx = 0
        for i, token_id in enumerate(input_ids):
            if token_id not in special_tokens and attention_mask[i] == 1:
                if token_idx < len(mask_seq):
                    mask_seq_to_tokenized[token_idx] = i
                    token_idx += 1
        
        # 建立映射：mask_seq的tokenized位置 -> 原始序列位置
        # 关键：mask_seq和actual_seq的长度相同，位置一一对应
        # 验证：确保两个序列长度一致
        if len(actual_seq) != len(mask_seq):
            raise ValueError(
                f"样本 {idx}: 原始序列长度({len(actual_seq)})与掩码序列长度({len(mask_seq)})不一致！"
            )
        
        # 验证：确保tokenization后的有效token数量一致
        if len(original_seq_to_tokenized) != len(mask_seq_to_tokenized):
            print(f"警告：样本 {idx} tokenization后长度不一致！")
            print(f"  原始序列有效token数: {len(original_seq_to_tokenized)}")
            print(f"  掩码序列有效token数: {len(mask_seq_to_tokenized)}")
            print(f"  原始序列: {actual_seq[:50]}...")
            print(f"  掩码序列: {mask_seq[:50]}...")
        
        mask_tokenized_to_original = {}
        for seq_pos in range(min(len(actual_seq), len(mask_seq))):
            if seq_pos in mask_seq_to_tokenized:
                mask_tokenized_pos = mask_seq_to_tokenized[seq_pos]
                mask_tokenized_to_original[mask_tokenized_pos] = seq_pos
        
        # 调试信息：验证位置映射的正确性
        if idx < 5:  # 只对前5个样本打印调试信息
            print(f"样本 {idx}:")
            print(f"  原始序列长度: {len(actual_seq)}")
            print(f"  掩码序列长度: {len(mask_seq)}")
            print(f"  原始tokenized长度: {len(original_input_ids)}")
            print(f"  掩码tokenized长度: {len(input_ids)}")
            print(f"  原始位置映射数量: {len(original_seq_to_tokenized)}")
            print(f"  掩码位置映射数量: {len(mask_seq_to_tokenized)}")
            print(f"  掩码位置: {[i for i, t in enumerate(input_ids) if t == mask_token_id]}")
            print(f"  掩码序列中的'X'数量: {mask_seq.count('X')}")
            print(f"  掩码序列中的'<mask>'数量: {mask_seq_processed.count('<mask>')}")
            print(f"  tokenizer词汇表大小: {self.tokenizer.vocab_size}")
            print(f"  特殊token: cls={self.tokenizer.cls_token_id}, pad={self.tokenizer.pad_token_id}, mask={self.tokenizer.mask_token_id}")
            print("---")
        
        # 3. 为掩码位置设置正确的标签
        mask_count = 0
        for i, token_id in enumerate(input_ids):
            if token_id == mask_token_id:  # 这是掩码token
                # 找到这个掩码token对应的原始序列位置
                original_pos = mask_tokenized_to_original.get(i, None)
                
                if original_pos is not None and original_pos < len(actual_seq):
                    # 设置真实标签
                    original_aa = actual_seq[original_pos]
                    # 使用tokenizer的convert_tokens_to_ids方法
                    token_id = self.tokenizer.convert_tokens_to_ids(original_aa)
                    # 确保token_id在有效范围内
                    if token_id is not None and 0 <= token_id < self.tokenizer.vocab_size:
                        label_ids[i] = token_id
                        mask_count += 1
                    else:
                        print(f"警告：氨基酸 {original_aa} 的token_id {token_id} 无效")
                        print(f"  词汇表大小: {self.tokenizer.vocab_size}")
                        print(f"  有效范围: [0, {self.tokenizer.vocab_size})")
                        # 检查这个氨基酸是否在词汇表中
                        if hasattr(self.tokenizer, 'vocab'):
                            if original_aa in self.tokenizer.vocab:
                                print(f"  氨基酸 {original_aa} 在词汇表中，但token_id无效")
                            else:
                                print(f"  氨基酸 {original_aa} 不在词汇表中")
                        label_ids[i] = -100  # 设为ignore_index
                        
                        # 调试信息：验证标签设置的正确性
                        if idx < 5:
                            print(f"  掩码位置 {i} -> 原始位置 {original_pos} -> 氨基酸 {original_aa} -> token_id {label_ids[i]}")
        
        # 验证掩码数量的一致性
        expected_mask_count = len([i for i, t in enumerate(input_ids) if t == mask_token_id])
        if mask_count != expected_mask_count:
            print(f"警告：掩码数量不一致！期望 {expected_mask_count}，实际设置 {mask_count}")
        
        # 最终安全检查：确保所有标签都在有效范围内
        for i, label_id in enumerate(label_ids):
            if label_id != -100 and (label_id < 0 or label_id >= self.tokenizer.vocab_size):
                print(f"警告：位置 {i} 的标签 {label_id} 超出范围，设为ignore_index")
                label_ids[i] = -100
        
        # 额外的安全检查：确保tensor中的所有值都是有效的
        label_ids = torch.clamp(label_ids, min=-100, max=self.tokenizer.vocab_size-1)
        
        # 分析序列中的氨基酸类型
        if idx < 5:
            unique_aas = set(actual_seq)
            print(f"  序列中的唯一氨基酸: {sorted(unique_aas)}")
            # 检查哪些氨基酸可能有问题
            problematic_aas = []
            for aa in unique_aas:
                token_id = self.tokenizer.convert_tokens_to_ids(aa)
                if token_id is None or token_id < 0 or token_id >= self.tokenizer.vocab_size:
                    problematic_aas.append(aa)
            if problematic_aas:
                print(f"  可能有问题的氨基酸: {problematic_aas}")
            print("---")
        
        return {
            'input_ids': input_ids,
            'labels': label_ids,
            'mask': (input_ids == mask_token_id),  # 掩码位置
            'attention_mask': attention_mask,
            'pad_token_id': self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        }

def collate_fn(batch):
    # 找到batch中最长的序列长度
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # 动态padding到batch内最长长度
    padded_input_ids = []
    padded_labels = []
    padded_mask = []
    padded_attention_mask = []
    
    # 获取pad_token_id（如无则退化为0）
    pad_token_id = batch[0].get('pad_token_id', 0) if len(batch) > 0 else 0
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        # padding
        input_ids = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        labels = torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
        mask = torch.cat([item['mask'], torch.zeros(pad_len, dtype=torch.bool)])
        attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        
        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        padded_mask.append(mask)
        padded_attention_mask.append(attention_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels),
        'mask': torch.stack(padded_mask),
        'attention_mask': torch.stack(padded_attention_mask)
    }

# =====================
# 训练主流程
# =====================
def main():
    import argparse
    import json
    import re
    import os
    
    # 设置CUDA调试模式
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    parser = argparse.ArgumentParser(description='PepLoe BMP 掩码恢复预训练')
    # 默认使用最新生成的 BMP 数据集路径
    parser.add_argument('--csv', type=str, default=os.path.join(project_root, '../data/BMP.csv'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default=None)  # 改为None，自动生成
    parser.add_argument('--config', type=str, default=os.path.join(project_root, 'pretrain_config.json'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 自动生成包含参数信息的输出路径
    if args.save_dir is None:
        # 直接使用任务目录，不创建参数子目录
        args.save_dir = os.path.join("/public/home/lingwang/MoRE/Pretrain/lora", "BMP")
    
    print(f"开始BMP掩码恢复预训练")
    print(f"数据路径: {args.csv}")
    print(f"输出路径: {args.save_dir}")
    print(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"配置文件: {args.config}")

    # 自动创建输出目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存训练参数到txt文件
    params_txt_path = os.path.join(args.save_dir, 'training_params.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("BMP预训练参数配置\n")
        f.write("=" * 50 + "\n")
        f.write(f"任务类型: BMP (Bidirectional Masked Prediction)\n")
        f.write(f"数据路径: {args.csv}\n")
        f.write(f"输出路径: {args.save_dir}\n")
        f.write(f"配置文件: {args.config}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"设备: {args.device}\n")
        f.write(f"最大长度: 1024\n")
        f.write(f"掩码比例: 0.50\n")
        f.write(f"早停patience: 3\n")
        f.write(f"最大训练轮数: 10\n")
        f.write(f"验证集比例: 5%\n")
        f.write(f"随机种子: 42\n")
        f.write("=" * 50 + "\n")
        f.write(f"训练开始时间: {pd.Timestamp.now()}\n")
    
    print(f"训练参数已保存到: {params_txt_path}")

    # 加载模型和分词器（自动检测任务名称）
    model, tokenizer = get_model(args.config, task_name="BMP")
    model = model.to(args.device)
    print('模型和分词器加载完成')

    # 加载数据并划分训练集和验证集
    full_dataset = MaskedProteinDataset(args.csv, tokenizer, mask_ratio=0.50)
    
    # 划分训练集和验证集（5%作为验证集）
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.05, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    # 测试第一个batch的数据
    print("测试第一个batch的数据...")
    test_batch = next(iter(train_loader))
    print(f"  input_ids shape: {test_batch['input_ids'].shape}")
    print(f"  labels shape: {test_batch['labels'].shape}")
    print(f"  mask shape: {test_batch['mask'].shape}")
    print(f"  input_ids min/max: {test_batch['input_ids'].min().item()}/{test_batch['input_ids'].max().item()}")
    print(f"  labels min/max: {test_batch['labels'].min().item()}/{test_batch['labels'].max().item()}")
    print(f"  mask sum: {test_batch['mask'].sum().item()}")
    print(f"  tokenizer vocab size: {tokenizer.vocab_size}")
    
    # 检查是否有超出范围的标签
    invalid_labels = (test_batch['labels'] != -100) & ((test_batch['labels'] < 0) | (test_batch['labels'] >= tokenizer.vocab_size))
    if invalid_labels.any():
        print(f"发现 {invalid_labels.sum().item()} 个超出范围的标签!")
        invalid_indices = torch.where(invalid_labels)
        print(f"  位置: {invalid_indices}")
        print(f"  值: {test_batch['labels'][invalid_indices]}")
    else:
        print("所有标签都在有效范围内")
    
    # 测试模型输出
    print("测试模型输出...")
    model.eval()
    with torch.no_grad():
        test_input_ids = test_batch['input_ids'].to(args.device)
        test_attention_mask = test_batch['attention_mask'].to(args.device)
        outputs = model(test_input_ids, attention_mask=test_attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
        print(f"  logits shape: {logits.shape}")
        print(f"  logits last dim: {logits.size(-1)}")
        print(f"  tokenizer vocab size: {tokenizer.vocab_size}")
        
        if logits.size(-1) != tokenizer.vocab_size:
            print(f"警告：模型输出维度 {logits.size(-1)} 与词汇表大小 {tokenizer.vocab_size} 不匹配!")
        else:
            print("模型输出维度与词汇表大小匹配")
    print("---")

    # 损失函数（只在掩码位点计算loss）
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 早停机制
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    max_epochs = min(args.epochs, 10)  # 最多训练10轮
    
    # 记录loss曲线
    train_losses = []
    val_losses = []
    epochs_list = []
    
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        total_mask = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs} [Train]')
        for batch in pbar:
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            mask = batch['mask'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
            
            # 最终安全检查：确保标签在有效范围内
            labels_safe = labels.clone()
            invalid_mask = (labels_safe != -100) & ((labels_safe < 0) | (labels_safe >= logits.size(-1)))
            if invalid_mask.any():
                print(f"训练中发现 {invalid_mask.sum().item()} 个超出范围的标签，设为ignore_index")
                labels_safe[invalid_mask] = -100
            
            loss_all = criterion(logits.view(-1, logits.size(-1)), labels_safe.view(-1))
            loss = (loss_all * mask.view(-1)).sum() / (mask.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_mask += mask.sum().item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss = total_loss / (total_mask + 1e-8)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_mask = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{max_epochs} [Val]'):
                input_ids = batch['input_ids'].to(args.device)
                labels = batch['labels'].to(args.device)
                mask = batch['mask'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
                
                # 最终安全检查：确保标签在有效范围内
                labels_safe = labels.clone()
                invalid_mask = (labels_safe != -100) & ((labels_safe < 0) | (labels_safe >= logits.size(-1)))
                if invalid_mask.any():
                    print(f"验证中发现 {invalid_mask.sum().item()} 个超出范围的标签，设为ignore_index")
                    labels_safe[invalid_mask] = -100
                
                loss_all = criterion(logits.view(-1, logits.size(-1)), labels_safe.view(-1))
                loss = (loss_all * mask.view(-1)).sum() / (mask.sum() + 1e-8)
                val_loss += loss.item() * mask.sum().item()
                val_mask += mask.sum().item()
        
        val_loss = val_loss / (val_mask + 1e-8)
        
        # 记录loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_list.append(epoch + 1)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            saver = LoRAModelSaver(args.config, task_name="BMP")
            saver.save_lora_adapter(model, args.save_dir)
            print(f'保存最佳模型 (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'验证损失未改善 ({patience_counter}/{patience})')
            
        if patience_counter >= patience:
            print(f'早停触发！最佳验证损失: {best_val_loss:.4f}')
            break
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BMP Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存loss曲线图
    loss_curve_path = os.path.join(args.save_dir, 'loss_curves.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存loss数据到CSV
    loss_data = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_csv_path = os.path.join(args.save_dir, 'loss_data.csv')
    loss_data.to_csv(loss_csv_path, index=False)
    
    print(f'Loss曲线已保存到: {loss_curve_path}')
    print(f'Loss数据已保存到: {loss_csv_path}')

    print(f'BMP预训练完成，最佳验证损失: {best_val_loss:.4f}')

if __name__ == '__main__':
    main() 