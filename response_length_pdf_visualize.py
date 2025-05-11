#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_results_from_jsonl(file_path):
    """从JSONL文件加载处理结果"""
    data = []
    response_lengths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if 'qwen_token_count' in item and item['qwen_token_count'] is not None:
                response_lengths.append(item['qwen_token_count'])
                data.append(item)
    
    # 计算统计信息
    stats = {
        'mean': float(np.mean(response_lengths)) if response_lengths else 0,
        'median': float(np.median(response_lengths)) if response_lengths else 0,
        'min': float(np.min(response_lengths)) if response_lengths else 0,
        'max': float(np.max(response_lengths)) if response_lengths else 0
    }
    
    return {
        'results': data,
        'response_lengths': response_lengths,
        'stats': stats
    }

def plot_length_distribution(result_data, output_filename, dataset_name):
    """绘制响应长度分布图，3:1比例，无标题，标签字体24"""
    response_lengths = result_data['response_lengths']
    
    if not response_lengths:
        print(f"警告：数据集 {dataset_name} 没有有效的响应长度数据，跳过绘图")
        return
    
    # 设置图形大小为3:1比例
    plt.figure(figsize=(12, 4))
    
    # 重置为默认字体大小
    plt.rcParams.update({'font.size': 12})
    
    # 创建直方图，使用40个bins
    max_length = max(response_lengths)
    bin_width = max_length / 40
    bins = range(0, int(max_length + bin_width), int(bin_width))
    
    plt.hist(response_lengths, bins=bins, color='skyblue', 
             edgecolor='black', alpha=0.7)
    
    # 只设置标签字体为24，无标题
    plt.xlabel('Response Length (tokens)', fontsize=24)
    plt.ylabel('# Problems', fontsize=24)
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.3)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存为PDF
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {output_filename}")
    
    # 关闭图形以释放内存
    plt.close()

def main():
    # 定义输入文件和数据集名称
    datasets = [
        {
            'input_file': 'math3977gen_with_responses.jsonl',
            'output_file': 'math3977gen_response_length_distribution.pdf',
            'name': 'Math3977gen'
        },
        {
            'input_file': 'math7507gen_with_responses.jsonl',
            'output_file': 'math7507gen_response_length_distribution.pdf',
            'name': 'Math7507gen'
        }
    ]
    
    # 处理每个数据集
    for dataset in datasets:
        print(f"Processing {dataset['name']}...")
        
        # 加载数据
        data = load_results_from_jsonl(dataset['input_file'])
        
        # 绘图并保存
        plot_length_distribution(
            data, 
            dataset['output_file'],
            dataset['name']
        )
    
    print("All visualizations completed successfully!")

if __name__ == "__main__":
    main() 