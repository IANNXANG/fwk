#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def load_results(file_path):
    """加载处理结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_distribution(result_data, dataset_name, output_dir='.'):
    """绘制响应长度分布图，不包含平均值和中位数线"""
    response_lengths = result_data['response_lengths']
    
    if not response_lengths:
        print(f"警告：数据集 {dataset_name} 没有有效的响应长度数据，跳过绘图")
        return
        
    plt.figure(figsize=(12, 7))
    
    # 创建直方图，固定刻度
    bins = range(0, max(response_lengths) + 50, 50)
    plt.hist(response_lengths, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 设置标题和标签（使用英文）
    plt.xlabel('Response Length (tokens)', fontsize=14)
    plt.ylabel('Sample Count', fontsize=14)
    plt.title(f'Distribution of Response Length for {dataset_name}', fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    
    # 保存图片
    output_file = os.path.join(output_dir, f'{dataset_name}_response_length_distribution.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"分布图已保存至 {output_file}")
    plt.close()

def main(args):
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个数据集并绘图
    for input_file, dataset_name in zip(args.input_files, args.dataset_names):
        print(f"\n处理数据集可视化: {dataset_name}")
        try:
            result_data = load_results(input_file)
            plot_distribution(result_data, dataset_name, args.output_dir)
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制响应长度分布图')
    parser.add_argument('--input_files', type=str, nargs='+', help='响应结果JSON文件路径列表')
    parser.add_argument('--dataset_names', type=str, nargs='+', help='数据集名称列表，与输入文件一一对应')
    parser.add_argument('--output_dir', type=str, default='.', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.input_files is None or args.dataset_names is None:
        print("请提供输入文件和数据集名称")
        parser.print_help()
        exit(1)
    
    if len(args.input_files) != len(args.dataset_names):
        print("输入文件数量必须与数据集名称数量相同")
        exit(1)
    
    main(args) 