#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random
from tqdm import tqdm
from transformers import AutoTokenizer

# 设置随机种子确保可重复性
random.seed(42)

def load_jsonl(file_path):
    """加载JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_qwen_tokenizer():
    """创建Qwen3分词器"""
    try:
        # 使用Qwen3分词器
        return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
    except Exception as e:
        print(f"加载Qwen分词器失败: {e}")
        print("请确保已安装transformers库: pip install transformers")
        raise

def count_tokens(text, tokenizer):
    """使用Qwen分词器计算文本的token数量"""
    return len(tokenizer.encode(text))

def process_dataset(file_path, output_dir, sample_size=500, x_max=None, bin_size=20):
    """处理单个数据集，计算问题文本的token长度"""
    # 提取数据集名称
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n处理数据集: {dataset_name}")
    
    # 加载数据
    try:
        data = load_jsonl(file_path)
        print(f"成功加载 {len(data)} 条数据")
        
        # 采样
        if len(data) > sample_size:
            sampled_data = random.sample(data, sample_size)
            print(f"随机采样 {sample_size} 条数据")
        else:
            sampled_data = data
            print(f"数据集条目少于 {sample_size}，使用全部 {len(data)} 条数据")
        
        # 初始化Qwen分词器
        print("使用Qwen3分词器")
        tokenizer = create_qwen_tokenizer()
        
        # 统计token长度
        token_counts = []
        skipped_items = 0
        
        for item in tqdm(sampled_data, desc="计算token长度"):
            text = None
            # 根据数据集字段选择问题文本
            if 'problem' in item:
                text = item['problem']
            elif 'instruction' in item:
                text = item['instruction']
            
            if text:
                token_count = count_tokens(text, tokenizer)
                token_counts.append(token_count)
            else:
                skipped_items += 1
        
        # 计算统计信息
        stats = {
            'mean': float(np.mean(token_counts)),
            'median': float(np.median(token_counts)),
            'min': float(np.min(token_counts)),
            'max': float(np.max(token_counts)),
            'total': int(np.sum(token_counts)),
            'count': len(token_counts)
        }
        
        print(f"数据集 {dataset_name} 处理完成")
        print(f"统计信息:")
        print(f"  样本数: {stats['count']}")
        print(f"  平均token数: {stats['mean']:.2f}")
        print(f"  中位数token数: {stats['median']:.2f}")
        print(f"  最小token数: {stats['min']}")
        print(f"  最大token数: {stats['max']}")
        print(f"  总token数: {stats['total']}")
        if skipped_items > 0:
            print(f"  跳过: {skipped_items} 条数据 (无问题文本)")
        
        # 绘制分布图
        plt.figure(figsize=(10, 6))
        
        # 使用统一的横坐标范围和间隔
        if x_max:
            bins = range(0, x_max + bin_size, bin_size)
        else:
            # 如果未指定最大值，则使用当前数据集的最大值，向上取整到bin_size的倍数
            max_val = int(np.ceil(stats['max'] / bin_size) * bin_size)
            bins = range(0, max_val + bin_size, bin_size)
            
        plt.hist(token_counts, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Token Count', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Qwen3 Token Count Distribution - {dataset_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.tight_layout()
        img_file = os.path.join(output_dir, f"{dataset_name}_qwen3_token_distribution.png")
        plt.savefig(img_file, dpi=300)
        plt.close()
        
        return stats, token_counts
        
    except Exception as e:
        print(f"处理数据集 {file_path} 时出错: {str(e)}")
        return None, None

def save_summary(results, output_dir):
    """保存所有数据集的统计结果到文本文件"""
    output_file = os.path.join(output_dir, "qwen3_token_count_summary.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("数据集问题Qwen3 Token长度统计\n")
        f.write("=" * 40 + "\n\n")
        
        for dataset_name, stats in results.items():
            if stats:
                f.write(f"数据集: {dataset_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"  样本数: {stats['count']}\n")
                f.write(f"  平均token数: {stats['mean']:.2f}\n")
                f.write(f"  中位数token数: {stats['median']:.2f}\n")
                f.write(f"  最小token数: {stats['min']}\n")
                f.write(f"  最大token数: {stats['max']}\n")
                f.write(f"  总token数: {stats['total']}\n\n")
    
    print(f"汇总报告已保存至: {output_file}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用Qwen3分词器统计数据集中问题文本的token长度')
    parser.add_argument('--sample_size', type=int, default=500, help='每个数据集的采样数量')
    parser.add_argument('--output_dir', type=str, default="qwen3_token_analysis", help='输出目录')
    parser.add_argument('--bin_size', type=int, default=20, help='直方图每个bin的大小')
    args = parser.parse_args()
    
    # 数据集文件列表
    datasets = [
        "problemdata/1.math7500.jsonl",
        "problemdata/3.math500seed.jsonl",
        "problemdata/6.iter0.jsonl",
        "problemdata/7.iter1.jsonl",
        "problemdata/8.iter2.jsonl"
    ]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出结果将保存在: {args.output_dir}")
    print("使用Qwen3分词器进行token计数")
    
    # 处理所有数据集
    results = {}
    all_token_counts = []
    
    # 第一轮：处理所有数据集并收集信息
    for dataset_file in datasets:
        # 提取数据集名称
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        
        # 检查文件是否存在
        if not os.path.exists(dataset_file):
            print(f"警告: 文件 {dataset_file} 不存在，跳过")
            continue
        
        # 首先只收集统计信息和token数据
        stats, token_counts = process_dataset(
            file_path=dataset_file,
            output_dir=args.output_dir,
            sample_size=args.sample_size
        )
        
        if stats and token_counts:
            results[dataset_name] = stats
            all_token_counts.extend(token_counts)
    
    # 确定所有图表的统一最大x轴值
    if all_token_counts:
        global_max = max(all_token_counts)
        # 向上取整到bin_size的倍数
        x_max = int(np.ceil(global_max / args.bin_size) * args.bin_size)
        print(f"\n统一所有图表的横坐标最大值为: {x_max}")
        
        # 第二轮：使用统一的x轴范围重新绘制所有图表
        for dataset_file in datasets:
            dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
            
            if not os.path.exists(dataset_file) or dataset_name not in results:
                continue
                
            print(f"\n重新绘制 {dataset_name} 的分布图（统一横坐标）")
            process_dataset(
                file_path=dataset_file,
                output_dir=args.output_dir,
                sample_size=args.sample_size,
                x_max=x_max,
                bin_size=args.bin_size
            )
    
    # 保存汇总报告
    if results:
        save_summary(results, args.output_dir)
    else:
        print("没有成功处理任何数据集")

if __name__ == "__main__":
    main() 