#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_jsonl(file_path):
    """加载JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_response_lengths(file_path):
    """分析单个数据集的响应长度"""
    data = load_jsonl(file_path)
    
    # 提取响应长度
    response_lengths = []
    for item in data:
        if 'qwen_token_count' in item:
            response_lengths.append(item['qwen_token_count'])
    
    # 计算统计信息
    stats = {
        'mean': float(np.mean(response_lengths)),
        'median': float(np.median(response_lengths)),
        'std': float(np.std(response_lengths)),
        'min': float(np.min(response_lengths)),
        'max': float(np.max(response_lengths)),
        'count': len(response_lengths)
    }
    
    return stats

def create_comparison_plot(results, output_file):
    """创建数据集响应长度对比柱状图"""
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 数据准备
    datasets = ["MATH 7500", "MATH 500seed", "gendata0", "gendata1", "gendata2"]
    means = [results[d]['mean'] for d in results]
    
    # 设置柱状图位置
    x = np.arange(len(datasets))
    width = 0.35
    
    # 绘制柱状图 - 移除yerr参数
    bars = ax.bar(x, means, width,
                 color='skyblue', edgecolor='black', alpha=0.7)
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # 设置图表标题和标签
    ax.set_title('Response Length Comparison (Mean)', fontsize=14, pad=20)
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, fontsize=24)  # 增大数据集名称字体
    
    # 添加网格线
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PDF
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 数据集映射
    dataset_mapping = {
        "MATH 7500": "math7500_with_responses.jsonl",
        "MATH 500seed": "math500seed_with_responses.jsonl",
        "gendata0": "iter0_with_responses.jsonl",
        "gendata1": "iter1_with_responses.jsonl",
        "gendata2": "iter2_with_responses.jsonl"
    }
    
    # 分析结果
    results = {}
    
    # 处理每个数据集
    print("开始分析数据集响应长度...")
    for dataset_name, filename in dataset_mapping.items():
        print(f"\n处理数据集: {dataset_name}")
        file_path = os.path.join("response_length_analysis_full", filename)
        try:
            stats = analyze_response_lengths(file_path)
            results[dataset_name] = stats
            print(f"  样本数: {stats['count']}")
            print(f"  平均token数: {stats['mean']:.2f}")
            print(f"  中位数token数: {stats['median']:.2f}")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  最小/最大token数: {stats['min']:.0f}/{stats['max']:.0f}")
        except Exception as e:
            print(f"处理 {dataset_name} 时出错: {str(e)}")
    
    # 创建对比图
    if results:
        output_file = os.path.join("response_length_analysis_full", "response_length_comparison.pdf")
        create_comparison_plot(results, output_file)
        print(f"\n对比图已保存至: {output_file}")
    else:
        print("没有成功处理任何数据集")

if __name__ == "__main__":
    main() 