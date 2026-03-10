#!/usr/bin/env python3
"""
提取pMHC-TCR模型测试结果信息的脚本
"""
import pandas as pd
import numpy as np
import os

def extract_test_results():
    """提取和分析测试结果"""
    # 文件路径
    csv_file = "/public/home/lingwang/MoRE/pMHC-TCR/result/job_204492_seed42/training_metrics.csv"

    print("正在读取训练指标文件...")
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file, sep=';')
        print(f"成功读取文件，共 {len(df)} 行数据")

        # 提取所有test相关的列
        test_columns = [col for col in df.columns if col.startswith('test_')]

        print(f"\n找到 {len(test_columns)} 个测试相关指标：")
        for col in test_columns[:10]:  # 只显示前10个
            print(f"  - {col}")

        if len(test_columns) > 10:
            print(f"  ... 还有 {len(test_columns) - 10} 个指标")

        # 提取最后一行（通常是最终结果）
        final_results = df.iloc[-1]

        print("\n" + "="*60)
        print("最终测试结果汇总：")
        print("="*60)

        # 基本性能指标
        print("\n📊 基本性能指标:")
        metrics = ['test_acc', 'test_auc', 'test_pr_auc', 'test_precision',
                  'test_recall', 'test_f1', 'test_mcc', 'test_balanced_acc']

        for metric in metrics:
            if metric in df.columns:
                value = final_results[metric]
                print("15")

        # 混淆矩阵相关
        print("\n🔢 混淆矩阵统计:")
        confusion_metrics = ['test_tp', 'test_tn', 'test_fp', 'test_fn']

        for metric in confusion_metrics:
            if metric in df.columns:
                value = final_results[metric]
                print("8")

        # 计算额外的统计信息
        if all(metric in df.columns for metric in confusion_metrics):
            tp = final_results['test_tp']
            tn = final_results['test_tn']
            fp = final_results['test_fp']
            fn = final_results['test_fn']

            total = tp + tn + fp + fn
            print(f"  - 总样本数: {total}")
            # 计算精确率和召回率（验证计算是否正确）
            if tp + fp > 0:
                precision_calc = tp / (tp + fp)
                print(f"  - 计算得精确率: {precision_calc:.4f}")
            if tp + fn > 0:
                recall_calc = tp / (tp + fn)
                print(f"  - 计算得召回率: {recall_calc:.4f}")
        # 预测概率分析
        if 'test_probs' in df.columns:
            print("\n🎯 预测概率分析:")
            try:
                # 获取预测概率（这里只分析最后一个样本作为示例）
                probs_str = final_results['test_probs']
                if isinstance(probs_str, str):
                    # 解析概率字符串
                    probs = [float(x) for x in probs_str.strip('[]').split()]
                    print(f"  - 预测概率范围: [{min(probs):.4f}, {max(probs):.4f}]")
                    print(f"  - 平均预测概率: {np.mean(probs):.4f}")
                    print(f"  - 预测概率标准差: {np.std(probs):.4f}")

                    # 计算概率分布
                    high_prob = sum(1 for p in probs if p > 0.8)
                    med_prob = sum(1 for p in probs if 0.5 <= p <= 0.8)
                    low_prob = sum(1 for p in probs if p < 0.5)

                    print(f"  - 高置信度预测 (>0.8): {high_prob}")
                    print(f"  - 中等置信度预测 (0.5-0.8): {med_prob}")
                    print(f"  - 低置信度预测 (<0.5): {low_prob}")

            except Exception as e:
                print(f"  - 无法解析预测概率: {e}")

        # 保存结果到文件
        output_file = "test_results_summary.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("pMHC-TCR模型测试结果汇总\n")
            f.write("="*50 + "\n\n")

            f.write("基本性能指标:\n")
            for metric in metrics:
                if metric in df.columns:
                    f.write(f"  {metric}: {final_results[metric]:.4f}\n")
            f.write("\n")

            f.write("混淆矩阵统计:\n")
            for metric in confusion_metrics:
                if metric in df.columns:
                    f.write(f"  {metric}: {final_results[metric]}\n")
            f.write("\n")

            if all(metric in df.columns for metric in confusion_metrics):
                tp = final_results['test_tp']
                tn = final_results['test_tn']
                fp = final_results['test_fp']
                fn = final_results['test_fn']
                total = tp + tn + fp + fn
                f.write(f"总样本数: {total}\n")
                if tp + fp > 0:
                    precision_calc = tp / (tp + fp)
                    f.write(f"计算得精确率: {precision_calc:.4f}\n")
                if tp + fn > 0:
                    recall_calc = tp / (tp + fn)
                    f.write(f"计算得召回率: {recall_calc:.4f}\n")
        print(f"\n📄 结果已保存到: {output_file}")

        print("\n✅ 测试结果提取完成！")

    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

    return True

if __name__ == "__main__":
    success = extract_test_results()

    # 删除脚本文件
    if success:
        print("\n🗑️  清理脚本文件...")
        try:
            os.remove(__file__)
            print("✅ 脚本文件已删除")
        except Exception as e:
            print(f"⚠️  无法删除脚本文件: {e}")
