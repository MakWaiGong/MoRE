#!/usr/bin/env python3
"""
生成四个预训练任务的数据集
- BMP: 掩码语言模型
- Dist2NBS: 距离最近结合位点
- PPB: 蛋白质-肽结合
- SWBindCount: 滑动窗口结合位点计数
"""

import os
import pandas as pd
import random
from pathlib import Path
import argparse
import numpy as np


def preprocess_row(row: pd.Series) -> tuple:
    """
    预处理单行数据，提取并清理序列和标签
    返回: (pdb_id, actual_seq, actual_label, is_valid)
    """
    pdb_id = row['pdb_id']
    prot_seq = str(row['prot_seq']) if pd.notna(row['prot_seq']) else ''
    label_str = str(row['label']) if pd.notna(row['label']) else ''
    
    # 跳过空序列
    if not prot_seq:
        return (pdb_id, '', '', False)
    
    # 去除padding符号
    actual_seq = prot_seq.replace('#', '')
    
    # 跳过去除padding后为空的序列
    if not actual_seq:
        return (pdb_id, '', '', False)
    
    # 如果label比序列短，用'0'填充；如果长，截断
    if len(label_str) < len(actual_seq):
        actual_label = label_str + '0' * (len(actual_seq) - len(label_str))
    else:
        actual_label = label_str[:len(actual_seq)]
    
    return (pdb_id, actual_seq, actual_label, True)

def load_source_data(csv_path: str) -> pd.DataFrame:
    """加载源数据"""
    print(f"加载源数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   原始数据行数: {len(df)}")
    print(f"   列名: {list(df.columns)}")
    
    # 统一列名格式（处理大小写不一致的问题）
    if 'Prot_seq' in df.columns:
        df = df.rename(columns={'Prot_seq': 'prot_seq', 'Pep_seq': 'pep_seq'})
        print(f"   统一后的列名: {list(df.columns)}")
    
    # 检查必要的列是否存在
    required_columns = ['pdb_id', 'prot_seq', 'pep_seq', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    return df

def generate_bmp_data(df: pd.DataFrame, mask_ratio: float = 0.50) -> pd.DataFrame:
    """
    生成BMP数据集：掩码语言模型
    掩码策略：只掩码结合位点(label=1)和肽链全部位点
    输入: pdb_id, prot_seq, pep_seq, label
    输出: pdb_id, mask_seq, label
    """
    print(f"生成BMP数据集 (掩码比例: {mask_ratio})")
    print(f"   掩码策略: 只掩码结合位点(label=1)和肽链位置")
    print(f"   掩码方式: BERT式 (80%[MASK] + 10%随机氨基酸 + 10%保持不变)")
    
    bmp_data = []
    
    for idx, row in df.iterrows():
        pdb_id, actual_seq, actual_label, is_valid = preprocess_row(row)
        if not is_valid:
            continue
        
        # 处理肽链序列
        pep_seq = str(row['pep_seq']) if pd.notna(row['pep_seq']) else ''
        if not pep_seq:
            continue
        actual_pep_seq = pep_seq.replace('#', '')
        
        # 跳过去除padding后为空的肽链序列
        if not actual_pep_seq:
            continue
        
        # 添加padding到固定长度（与BMP预训练代码保持一致）
        max_len = 1024
        
        # 如果序列超过最大长度，先截断（确保掩码位置在有效范围内）
        if len(actual_seq) > max_len:
            actual_seq = actual_seq[:max_len]
            actual_label = actual_label[:max_len]
        
        # 解析标签，找到结合位点（label是二进制字符串格式）
        binding_positions = [i for i, char in enumerate(actual_label) if char == '1']
        
        # 找到肽链在蛋白质序列中的所有位置（可能多次出现）
        pep_positions = []
        start = 0
        while True:
            pep_start_pos = actual_seq.find(actual_pep_seq, start)
            if pep_start_pos == -1:
                break
            pep_positions.extend(range(pep_start_pos, pep_start_pos + len(actual_pep_seq)))
            start = pep_start_pos + 1
        
        # 合并掩码位置：结合位点 + 肽链位置
        mask_candidate_positions = list(set(binding_positions + pep_positions))
        
        # 从候选位置中随机选择掩码位置
        if mask_candidate_positions:
            num_mask = min(int(len(mask_candidate_positions) * mask_ratio), len(mask_candidate_positions))
            mask_positions = random.sample(mask_candidate_positions, num_mask)
        else:
            mask_positions = []
        
        # 创建掩码序列 - BERT式掩码策略
        mask_seq = list(actual_seq)
        for pos in mask_positions:
            # BERT式掩码：80%替换为[MASK]，10%替换为随机氨基酸，10%保持不变
            rand = random.random()
            if rand < 0.8:
                mask_seq[pos] = 'X'  # 80% 替换为[MASK]
            elif rand < 0.9:
                # 10% 替换为随机氨基酸
                random_aa = random.choice('ACDEFGHIKLMNPQRSTVWY')
                mask_seq[pos] = random_aa
            # 10% 保持不变 (不做任何操作)
        mask_seq = ''.join(mask_seq)
        
        # 添加padding到固定长度
        if len(mask_seq) < max_len:
            mask_seq += '#' * (max_len - len(mask_seq))
        else:
            mask_seq = mask_seq[:max_len]
            
        if len(actual_seq) < max_len:
            label_seq = actual_seq + '#' * (max_len - len(actual_seq))
        else:
            label_seq = actual_seq[:max_len]
        
        bmp_data.append({
            'pdb_id': pdb_id,
            'mask_seq': mask_seq,
            'label': label_seq
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"   处理进度: {idx + 1}/{len(df)}")
    
    bmp_df = pd.DataFrame(bmp_data)
    print(f"BMP数据集生成完成: {len(bmp_df)} 个样本")
    return bmp_df

def generate_dist2nbs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成Dist2NBS数据集：距离最近结合位点
    输入: pdb_id, prot_seq, pep_seq, label
    输出: pdb_id, prot_seq, label (距离值)
    """
    print(f"生成Dist2NBS数据集")
    
    dist2nbs_data = []
    
    for idx, row in df.iterrows():
        pdb_id, actual_seq, actual_label, is_valid = preprocess_row(row)
        if not is_valid:
            continue
        
        # 解析二进制标签，找到结合位点
        binding_positions = [i for i, char in enumerate(actual_label) if char == '1']
        
        # 计算每个位置到最近结合位点的距离
        distances = []
        for i in range(len(actual_label)):
            if i in binding_positions:
                distances.append(0.0)  # 在结合位点
            else:
                # 计算到最近结合位点的距离
                if binding_positions:
                    min_dist = min([abs(i - pos) for pos in binding_positions])
                    # 归一化到0-1范围：使用序列长度的一半作为最大可能距离
                    # 这样可以避免长序列中距离被过度归一化
                    max_possible_dist = len(actual_label) / 2.0
                    if max_possible_dist > 0:
                        normalized_dist = min(min_dist / max_possible_dist, 1.0)
                    else:
                        normalized_dist = 1.0
                else:
                    normalized_dist = 1.0  # 没有结合位点时设为最大距离
                distances.append(normalized_dist)
        
        # 添加padding到固定长度（与文档保持一致）
        max_len = 512
        if len(distances) < max_len:
            distances.extend([0.0] * (max_len - len(distances)))
        else:
            distances = distances[:max_len]
            
        if len(actual_seq) < max_len:
            padded_seq = actual_seq + '#' * (max_len - len(actual_seq))
        else:
            padded_seq = actual_seq[:max_len]
        
        # 将距离转换为逗号分隔的字符串，保留4位小数
        distance_str = ','.join([f"{d:.4f}" for d in distances])
        
        dist2nbs_data.append({
            'pdb_id': pdb_id,
            'prot_seq': padded_seq,
            'label': distance_str
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"   处理进度: {idx + 1}/{len(df)}")
    
    dist2nbs_df = pd.DataFrame(dist2nbs_data)
    print(f"Dist2NBS数据集生成完成: {len(dist2nbs_data)} 个样本")
    return dist2nbs_df

def generate_ppb_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成PPB数据集：蛋白质-肽结合
    输入: pdb_id, prot_seq, pep_seq, label
    输出: pdb_id, prot_seq, pep_seq, label (直接使用)
    """
    print(f"生成PPB数据集")
    
    # PPB直接使用原始数据，无需额外处理
    ppb_df = df.copy()
    print(f"PPB数据集生成完成: {len(ppb_df)} 个样本")
    return ppb_df

def generate_swbindcount_data(df: pd.DataFrame, window_size: int = 30, step_size: int = 1) -> pd.DataFrame:
    """
    生成SWBindCount数据集：滑动窗口结合位点计数
    输入: pdb_id, prot_seq, pep_seq, label
    输出: pdb_id, position, prot_seq, label
    """
    print(f"生成SWBindCount数据集 (窗口大小: {window_size}, 步长: {step_size})")
    
    swbindcount_data = []
    
    for idx, row in df.iterrows():
        pdb_id, actual_seq, actual_label, is_valid = preprocess_row(row)
        if not is_valid:
            continue
        
        # 滑动窗口扫描（如果序列长度小于窗口大小，跳过该样本）
        if len(actual_seq) < window_size:
            continue
        for start_pos in range(0, len(actual_seq) - window_size + 1, step_size):
            end_pos = start_pos + window_size
            
            # 提取窗口序列
            window_seq = actual_seq[start_pos:end_pos]
            window_label = actual_label[start_pos:end_pos]
            
            # 计算窗口内结合位点数量
            binding_count = window_label.count('1')
            
            swbindcount_data.append({
                'pdb_id': pdb_id,
                'position': start_pos + 1,  # 位置从1开始
                'prot_seq': window_seq,
                'binding_site_count': binding_count  # 使用binding_site_count列名，与预训练代码一致
            })
        
        if (idx + 1) % 1000 == 0:
            print(f"   处理进度: {idx + 1}/{len(df)}")
    
    swbindcount_df = pd.DataFrame(swbindcount_data)
    print(f"SWBindCount数据集生成完成: {len(swbindcount_df)} 个样本")
    return swbindcount_df

def save_datasets(datasets: dict, output_dir: str):
    """保存所有数据集"""
    print(f"保存数据集到: {output_dir}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for task_name, df in datasets.items():
        output_path = os.path.join(output_dir, f"{task_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"   {task_name}.csv: {len(df)} 个样本")

def main():
    parser = argparse.ArgumentParser(description="从merged_train.csv生成四个预训练任务的数据集")
    parser.add_argument("--input", type=str, default="../data/pretrain.csv", 
                       help="输入CSV文件路径")
    parser.add_argument("--output", type=str, default="../data", 
                       help="输出目录")
    parser.add_argument("--mask_ratio", type=float, default=0.40, 
                       help="BMP掩码比例 (0-1之间)")
    parser.add_argument("--window_size", type=int, default=30, 
                       help="SWBindCount窗口大小 (必须大于0)")
    parser.add_argument("--step_size", type=int, default=1, 
                       help="SWBindCount步长 (必须大于0)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子，确保结果可复现")
    
    args = parser.parse_args()
    
    # 参数验证
    if not (0.0 <= args.mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio必须在0-1之间，当前值: {args.mask_ratio}")
    if args.window_size <= 0:
        raise ValueError(f"window_size必须大于0，当前值: {args.window_size}")
    if args.step_size <= 0:
        raise ValueError(f"step_size必须大于0，当前值: {args.step_size}")
    
    # 设置随机种子确保可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("开始生成预训练数据集")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"掩码比例: {args.mask_ratio}")
    print(f"窗口大小: {args.window_size}")
    print(f"步长: {args.step_size}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 加载源数据
    df = load_source_data(args.input)
    
    # 生成四个数据集
    datasets = {}
    
    # 1. BMP数据集
    datasets['BMP'] = generate_bmp_data(df, args.mask_ratio)
    
    # 2. Dist2NBS数据集
    datasets['Dist2NBS'] = generate_dist2nbs_data(df)
    
    # 3. PPB数据集
    datasets['PPB'] = generate_ppb_data(df)
    
    # 4. SWBindCount数据集
    datasets['SWBindCount'] = generate_swbindcount_data(df, args.window_size, args.step_size)
    
    # 保存所有数据集
    save_datasets(datasets, args.output)
    
    print("\n所有数据集生成完成！")
    print("=" * 60)
    for task_name, df in datasets.items():
        print(f"{task_name}: {len(df)} 个样本")

if __name__ == "__main__":
    main()
