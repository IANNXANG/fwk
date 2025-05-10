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

def plot_distribution(result_data, dataset_name, output_dir='.', x_range=None):
    """绘制响应长度分布图，使用固定的横坐标范围，纵坐标自适应"""
    response_lengths = result_data['response_lengths']
    
    if not response_lengths:
        print(f"警告：数据集 {dataset_name} 没有有效的响应长度数据，跳过绘图")
        return
        
    plt.figure(figsize=(12, 7))
    
    # 创建直方图，固定刻度
    if x_range:
        bins = range(0, x_range + 50, 50)
    else:
        bins = range(0, max(response_lengths) + 50, 50)
    
    plt.hist(response_lengths, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 设置标题和标签（使用英文）
    plt.xlabel('Response Length (tokens)', fontsize=14)
    plt.ylabel('Sample Count', fontsize=14)
    plt.title(f'Distribution of Response Length for {dataset_name}', fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    
    # 只设置固定的X轴范围
    if x_range:
        plt.xlim(0, x_range)
    
    # 保存图片
    output_file = os.path.join(output_dir, f'{dataset_name}_response_length_distribution.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"分布图已保存至 {output_file}")
    plt.close()

def main():
    # 硬编码参数
    input_files = [
        "math7507gen_with_responses.jsonl",
        "math3977gen_with_responses.jsonl"
    ]
    dataset_names = [
        "Math 7507",
        "Math 3977"
    ]
    output_dir = "visualizations"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先读取所有数据集，找出最大的x值
    all_data = []
    max_x = 0
    
    for input_file in input_files:
        if os.path.exists(input_file):
            try:
                result_data = load_results_from_jsonl(input_file)
                all_data.append(result_data)
                
                # 更新最大x值
                if result_data['response_lengths']:
                    max_x = max(max_x, max(result_data['response_lengths']))
            except Exception as e:
                print(f"读取文件 {input_file} 时出错: {e}")
    
    # 向上取整到最接近的百或千
    max_x = int(np.ceil(max_x / 100.0)) * 100
    
    print(f"设置所有图表的X轴范围为: 0-{max_x}")
    print(f"Y轴范围将根据每个数据集自适应调整")
    
    # 使用相同的X轴范围绘制所有图表，Y轴自适应
    for i, (input_file, dataset_name) in enumerate(zip(input_files, dataset_names)):
        print(f"\n处理数据集可视化: {dataset_name}")
        try:
            if i < len(all_data):
                result_data = all_data[i]
                print(f"读取了 {len(result_data['results'])} 条数据，找到 {len(result_data['response_lengths'])} 个有效响应长度")
                plot_distribution(result_data, dataset_name, output_dir, max_x)
            else:
                print(f"数据集 {dataset_name} 的数据未能加载")
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")

if __name__ == "__main__":
    main() 