#!/usr/bin/env python3
"""
提取pMHC-TCR测试数据集信息
"""
import pandas as pd
import numpy as np
from collections import Counter

def extract_test_info():
    # 读取测试数据
    df = pd.read_csv('/public/home/lingwang/MoRE/pMHC-TCR/data/pmt_pmt/edges_pmt_test.csv')

    print("=== pMHC-TCR 测试数据集信息提取 ===\n")

    # 基本信息
    print(f"📊 数据集基本信息:")
    print(f"   总样本数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   列名: {list(df.columns)}\n")

    # 肽段统计
    peptide_counts = Counter(df['Peptide'])
    print("🧬 肽段(Peptide)统计:")
    print(f"   唯一肽段数: {len(peptide_counts)}")
    print("   Top 10 最频繁肽段:")
    for peptide, count in peptide_counts.most_common(10):
        print(f"     {peptide}: {count} 次")
    print()

    # MHC统计
    mhc_counts = Counter(df['MHC'])
    print("🧫 MHC统计:")
    print(f"   唯一MHC数: {len(mhc_counts)}")
    print("   MHC分布:")
    for mhc, count in sorted(mhc_counts.items()):
        print(f"     {mhc}: {count} 次")
    print()

    # TCR统计
    tcr_counts = Counter(df['TCR'])
    print("🦠 TCR统计:")
    print(f"   唯一TCR数: {len(tcr_counts)}")
    print("   Top 10 最频繁TCR:")
    for tcr, count in tcr_counts.most_common(10):
        print(f"     {tcr}: {count} 次")
    print()

    # 组合统计
    print("🔗 组合统计:")
    # Peptide-MHC组合
    pm_combinations = Counter([f"{row['Peptide']}-{row['MHC']}" for _, row in df.iterrows()])
    print(f"   Peptide-MHC唯一组合数: {len(pm_combinations)}")

    # Peptide-TCR组合
    pt_combinations = Counter([f"{row['Peptide']}-{row['TCR']}" for _, row in df.iterrows()])
    print(f"   Peptide-TCR唯一组合数: {len(pt_combinations)}")

    # MHC-TCR组合
    mt_combinations = Counter([f"{row['MHC']}-{row['TCR']}" for _, row in df.iterrows()])
    print(f"   MHC-TCR唯一组合数: {len(mt_combinations)}")
    print()

    # 最常见的组合
    print("🏆 最常见的组合:")
    print("   Top 5 Peptide-MHC组合:")
    for combo, count in pm_combinations.most_common(5):
        print(f"     {combo}: {count} 次")

    print("   Top 5 Peptide-TCR组合:")
    for combo, count in pt_combinations.most_common(5):
        print(f"     {combo}: {count} 次")

    print("   Top 5 MHC-TCR组合:")
    for combo, count in mt_combinations.most_common(5):
        print(f"     {combo}: {count} 次")

if __name__ == "__main__":
    extract_test_info()
