import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
pep_root = os.path.abspath(os.path.join(project_root, '../'))
src_path = os.path.join(pep_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from get_model import get_model
    from model_saver import LoRAModelSaver
    print("成功导入主模型模块")
except ImportError as e:
    print(f"导入主模型模块失败: {e}")
    sys.exit(1)

class Dist2NBSDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = tokenizer.get_vocab()
        # 保存pad_token_id，用于collate_fn
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 处理不同的列名格式
        if 'prot_seq' in row:
            prot_seq = row['prot_seq']
        elif 'Prot_seq' in row:
            prot_seq = row['Prot_seq']
        else:
            raise KeyError("找不到蛋白质序列列，请检查CSV文件的列名")
        
        label_str = row['label']
        
        # 解析距离标签（逗号分隔的浮点数）
        distances = [float(x) for x in label_str.split(',')]
        
        # 去除padding符号，只保留实际序列
        actual_seq = prot_seq.replace('#', '')
        actual_distances = distances[:len(actual_seq)]
        
        # 编码序列，不padding（在collate_fn中动态padding）
        encoding = self.tokenizer(
            actual_seq,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 不padding
            return_tensors='pt'
        )
        
        # 创建距离标签
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 初始化距离标签，长度与input_ids一致
        distance_labels = torch.full((input_ids.size(0),), -100.0)
        
        # 识别特殊token（ESM通常在开头有<cls>，可能结尾有<eos>）
        special_token_ids = set()
        if hasattr(self.tokenizer, 'cls_token_id') and self.tokenizer.cls_token_id is not None:
            special_token_ids.add(self.tokenizer.cls_token_id)
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            special_token_ids.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'sep_token_id') and self.tokenizer.sep_token_id is not None:
            special_token_ids.add(self.tokenizer.sep_token_id)
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            special_token_ids.add(self.tokenizer.pad_token_id)
        
        # 找到实际序列token的位置（跳过特殊token）
        seq_token_positions = []
        for i in range(input_ids.size(0)):
            if attention_mask[i] == 1 and input_ids[i].item() not in special_token_ids:
                seq_token_positions.append(i)
        
        # 只对实际序列位置设置距离标签
        seq_len = min(len(actual_distances), len(seq_token_positions))
        for i in range(seq_len):
            token_pos = seq_token_positions[i]
            distance_labels[token_pos] = actual_distances[i]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'distance_labels': distance_labels,
            'mask': (distance_labels != -100).float(),  # 有效位置掩码
            'pad_token_id': self.pad_token_id
        }

def collate_fn(batch):
    # 找到batch中最长的序列长度
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # 获取pad_token_id（从batch中获取，如无则使用默认值0）
    pad_token_id = batch[0].get('pad_token_id', 0) if len(batch) > 0 else 0
    
    # 动态padding到batch内最长长度
    padded_input_ids = []
    padded_attention_mask = []
    padded_distance_labels = []
    padded_mask = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        # padding
        input_ids = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        distance_labels = torch.cat([item['distance_labels'], torch.full((pad_len,), -100.0)])
        mask = torch.cat([item['mask'], torch.zeros(pad_len, dtype=torch.bool)])
        
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_distance_labels.append(distance_labels)
        padded_mask.append(mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'distance_labels': torch.stack(padded_distance_labels),
        'mask': torch.stack(padded_mask)
    }

def train_dist2nbs(config_path, data_path, output_dir, epochs=10, batch_size=4, lr=1e-4, device='cuda'):
    # 自动生成包含参数信息的输出路径
    if output_dir is None:
        # 直接使用任务目录，不创建参数子目录
        output_dir = os.path.join("/public/home/lingwang/MoRE/Pretrain/lora", "Dist2NBS")
    
    print(f"开始Dist2NBS距离预测预训练")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_dir}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"配置文件: {config_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练参数到txt文件
    params_txt_path = os.path.join(output_dir, 'training_params.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("Dist2NBS预训练参数配置\n")
        f.write("=" * 50 + "\n")
        f.write(f"任务类型: Dist2NBS (Distance to Nearest Binding Site)\n")
        f.write(f"数据路径: {data_path}\n")
        f.write(f"输出路径: {output_dir}\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"学习率: {lr}\n")
        f.write(f"设备: {device}\n")
        f.write(f"最大长度: 512\n")  # Dist2NBS使用512
        f.write(f"早停patience: 3\n")
        f.write(f"最大训练轮数: 10\n")
        f.write(f"验证集比例: 5%\n")
        f.write(f"随机种子: 42\n")
        f.write("=" * 50 + "\n")
        f.write(f"训练开始时间: {pd.Timestamp.now()}\n")
    
    print(f"训练参数已保存到: {params_txt_path}")
    
    # 加载模型和分词器（自动检测任务名称）
    model, tokenizer = get_model(config_path, task_name="Dist2NBS")
    model = model.to(device)
    
    # 数据加载并划分训练集和验证集
    full_dataset = Dist2NBSDataset(data_path, tokenizer, max_length=512)
    
    # 划分训练集和验证集（5%作为验证集）
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.05, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 损失函数：MSE损失，忽略-100位置
    def mse_loss_with_mask(pred, target, mask):
        # pred: [batch_size, seq_len, 1]
        # target: [batch_size, seq_len]
        # mask: [batch_size, seq_len]
        loss = nn.MSELoss(reduction='none')(pred.squeeze(-1), target)
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss
    
    # 早停机制
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    max_epochs = min(epochs, 10)  # 最多训练10轮
    
    # 记录loss曲线
    train_losses = []
    val_losses = []
    epochs_list = []
    
    # 训练循环
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        total_mask = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            distance_labels = batch['distance_labels'].to(device)
            mask = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
            
            # 添加回归头：将logits转换为距离预测
            if not hasattr(model, 'distance_head'):
                hidden_size = logits.size(-1)
                model.distance_head = nn.Linear(hidden_size, 1).to(device)
                print(f"创建距离预测头: {hidden_size} -> 1")
            
            # 预测距离
            distance_pred = model.distance_head(logits)  # [batch_size, seq_len, 1]
            
            # 计算损失
            loss = mse_loss_with_mask(distance_pred, distance_labels, mask)
            
            # 反向传播
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
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                distance_labels = batch['distance_labels'].to(device)
                mask = batch['mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
                
                if hasattr(model, 'distance_head'):
                    distance_pred = model.distance_head(logits)
                    loss = mse_loss_with_mask(distance_pred, distance_labels, mask)
                    val_loss += loss.item() * mask.sum().item()
                    val_mask += mask.sum().item()
        
        val_loss = val_loss / (val_mask + 1e-8)
        
        # 记录loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_list.append(epoch + 1)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            saver = LoRAModelSaver(config_path, task_name="Dist2NBS")
            saver.save_lora_adapter(model, output_dir)
            if hasattr(model, 'distance_head'):
                torch.save(model.distance_head.state_dict(), os.path.join(output_dir, 'distance_head.pt'))
            print(f'保存最佳模型 (Val Loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            print(f'验证损失未改善 ({patience_counter}/{patience})')
            
        if patience_counter >= patience:
            print(f'早停触发！最佳验证损失: {best_val_loss:.6f}')
            break
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Dist2NBS Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存loss曲线图
    loss_curve_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存loss数据到CSV
    loss_data = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_csv_path = os.path.join(output_dir, 'loss_data.csv')
    loss_data.to_csv(loss_csv_path, index=False)
    
    print(f'Loss曲线已保存到: {loss_curve_path}')
    print(f'Loss数据已保存到: {loss_csv_path}')
    
    print(f"Dist2NBS预训练完成，最佳验证损失: {best_val_loss:.6f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dist2NBS距离预测预训练')
    parser.add_argument('--config', type=str, default=os.path.join(project_root, 'pretrain_config.json'))
    parser.add_argument('--data', type=str, default=os.path.join(project_root, '../data/Dist2NBS.csv'))
    parser.add_argument('--output', type=str, default=None) # Changed default to None
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    train_dist2nbs(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

if __name__ == '__main__':
    main() 