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

class SWBindCountDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = tokenizer.get_vocab()
        # 保存pad_token_id，用于collate_fn
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        print(f"加载SWBindCount数据集: {csv_file}, 样本数: {len(self.data)}")
        print(f"标签范围: {self.data['binding_site_count'].min()} 到 {self.data['binding_site_count'].max()}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pdb_id = row['pdb_id']
        position = row['position']
        prot_seq = row['prot_seq']
        label = int(row['binding_site_count'])  # 结合位点数量
        
        # 编码蛋白质序列
        encoding = self.tokenizer(
            prot_seq,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 不padding
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bind_count': torch.tensor(label, dtype=torch.float),
            'pdb_id': pdb_id,
            'position': position,
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
    bind_counts = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        # padding
        input_ids = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        bind_counts.append(item['bind_count'])
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'bind_count': torch.stack(bind_counts)
    }

def train_swbindcount(config_path, data_path, output_dir, epochs=10, batch_size=512, lr=1e-4, device='cuda', loss_mode='mse'):
    # 自动生成包含参数信息的输出路径
    if output_dir is None:
        # 直接使用任务目录，不创建参数子目录
        output_dir = os.path.join("/public/home/lingwang/MoRE/Pretrain/lora", "SWBindCount")
    
    print(f"开始SWBindCount结合位点数量预测预训练")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_dir}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"配置文件: {config_path}")
    
    # 使用固定种子42
    import random
    current_seed = 42
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练参数到txt文件
    params_txt_path = os.path.join(output_dir, 'training_params.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("SWBindCount预训练参数配置\n")
        f.write("=" * 50 + "\n")
        f.write(f"任务类型: SWBindCount (Sliding Window Binding Count)\n")
        f.write(f"数据路径: {data_path}\n")
        f.write(f"输出路径: {output_dir}\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"学习率: {lr}\n")
        f.write(f"设备: {device}\n")
        f.write(f"数据采样: 10% (随机采样)\n")
        f.write(f"最大长度: 1024\n")
        f.write(f"损失函数: {loss_mode}\n")
        f.write(f"早停patience: 3\n")
        f.write(f"最大训练轮数: 10\n")
        f.write(f"验证集比例: 5%\n")
        f.write(f"随机种子: {current_seed} (固定种子)\n")
        f.write("=" * 50 + "\n")
        f.write(f"训练开始时间: {pd.Timestamp.now()}\n")
    
    print(f"训练参数已保存到: {params_txt_path}")
    
    # 加载模型和分词器（自动检测任务名称）
    model, tokenizer = get_model(config_path, task_name="SWBindCount")
    model = model.to(device)
    
    # 数据加载 - 只使用10%的数据
    dataset = SWBindCountDataset(data_path, tokenizer)
    
    # 随机采样10%的数据
    total_samples = len(dataset)
    sample_size = int(total_samples * 0.1)
    print(f"原始数据集大小: {total_samples}")
    print(f"采样数据集大小: {sample_size} (10%)")
    
    # 使用固定种子42，通过索引分割确保每次的10%都不一样
    random.seed(current_seed)
    indices = list(range(total_samples))
    random.shuffle(indices)  # 同种子⇒同一次序
    
    # 分割成10个部分，每次取其中一部分
    size = total_samples // 10
    fold_indices = [indices[i*size : (i+1)*size] for i in range(10)]
    
    # 使用当前时间戳的最后一位来选择fold
    import time
    fold_idx = int(time.time()) % 10
    sample_indices = fold_indices[fold_idx]
    
    print(f"当前随机种子: {current_seed} (固定种子)")
    print(f"选择fold: {fold_idx}/10 (基于时间戳)")
    
    # 创建采样后的数据集
    from torch.utils.data import Subset
    sampled_dataset = Subset(dataset, sample_indices)
    
    # 划分训练集和验证集（5%作为验证集）
    train_indices, val_indices = train_test_split(
        range(len(sampled_dataset)), 
        test_size=0.05, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(sampled_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(sampled_dataset, val_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 损失函数：支持两种模式
    def mse_loss(pred, target):
        """均方误差损失（默认）"""
        return nn.MSELoss()(pred.squeeze(), target)
    
    def poisson_loss(pred, target):
        """泊松误差损失（适用于计数任务）"""
        # 确保预测值为正数（泊松分布要求）
        pred = torch.clamp(pred.squeeze(), min=1e-8)
        # 泊松负对数似然损失
        return torch.mean(pred - target * torch.log(pred))
    
    # 选择损失函数
    if loss_mode == 'mse':
        loss_fn = mse_loss
        print("使用MSE损失函数")
    elif loss_mode == 'poisson':
        loss_fn = poisson_loss
        print("使用泊松损失函数")
    else:
        raise ValueError(f"不支持的损失模式: {loss_mode}，请选择 'mse' 或 'poisson'")
    
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
        total_samples = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bind_count = batch['bind_count'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
            
            # 添加回归头：将logits转换为结合位点数量预测
            # 使用全局平均池化 + 线性层
            if not hasattr(model, 'bind_count_head'):
                hidden_size = logits.size(-1)
                model.bind_count_head = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)
                ).to(device)
                print(f"创建结合位点数量预测头: {hidden_size} -> 64 -> 1")
            
            # 全局平均池化
            # 只对有效位置（非padding）进行平均
            mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            masked_logits = logits * mask
            pooled = masked_logits.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [batch_size, hidden_size]
            
            # 预测结合位点数量
            bind_count_pred = model.bind_count_head(pooled)  # [batch_size, 1]
            
            # 计算损失
            loss = loss_fn(bind_count_pred, bind_count)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bind_count.size(0)
            total_samples += bind_count.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss = total_loss / total_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bind_count = batch['bind_count'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
                
                if hasattr(model, 'bind_count_head'):
                    mask = attention_mask.unsqueeze(-1)
                    masked_logits = logits * mask
                    pooled = masked_logits.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                    bind_count_pred = model.bind_count_head(pooled)
                    loss = loss_fn(bind_count_pred, bind_count)
                    val_loss += loss.item() * bind_count.size(0)
                    val_samples += bind_count.size(0)
        
        val_loss = val_loss / (val_samples + 1e-8)
        
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
            saver = LoRAModelSaver(config_path, task_name="SWBindCount")
            saver.save_lora_adapter(model, output_dir)
            if hasattr(model, 'bind_count_head'):
                torch.save(model.bind_count_head.state_dict(), os.path.join(output_dir, 'bind_count_head.pt'))
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
    plt.title('SWBindCount Training Progress')
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
    
    print(f"SWBindCount预训练完成，最佳验证损失: {best_val_loss:.6f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SWBindCount结合位点数量预测预训练')
    parser.add_argument('--config', type=str, default=os.path.join(project_root, 'pretrain_config.json'))
    parser.add_argument('--data', type=str, default=os.path.join(project_root, '../data/SWBindCount.csv'))
    parser.add_argument('--output', type=str, default=None) # 允许输出路径为None，由脚本内部生成
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--loss_mode', type=str, default='mse', choices=['mse', 'poisson'], 
                       help='损失函数模式: mse(均方误差) 或 poisson(泊松误差)')
    
    args = parser.parse_args()
    
    train_swbindcount(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        loss_mode=args.loss_mode
    )

if __name__ == '__main__':
    main() 