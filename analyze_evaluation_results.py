#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import argparse
from collections import defaultdict

def load_jsonl(file_path):
    """加载JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_ratings(data):
    """分析评分数据，计算统计信息"""
    # 按维度分组评分
    ratings_by_dimension = defaultdict(list)
    for item in data:
        if item['rating'] is not None:  # 只统计有效评分
            ratings_by_dimension[item['dimension']].append(item['rating'])
    
    # 计算每个维度的统计信息
    stats = {}
    for dimension, ratings in ratings_by_dimension.items():
        ratings_array = np.array(ratings)
        stats[dimension] = {
            'mean': float(np.mean(ratings_array)),
            'median': float(np.median(ratings_array)),
            'std': float(np.std(ratings_array)),  # 添加标准差计算
            'min': float(np.min(ratings_array)),
            'max': float(np.max(ratings_array)),
            'count': len(ratings_array)
        }
    
    return stats

def generate_summary_report(all_results, output_dir):
    """生成评分结果的综合报告"""
    # 准备TXT报告内容
    txt_content = "数学问题评分分析报告\n"
    txt_content += "=" * 50 + "\n\n"
    
    # 准备CSV数据
    csv_rows = [["Dataset", "Dimension", "Mean", "Median", "Std Dev", "Min", "Max", "Count"]]
    
    # 处理每个数据集的结果
    for dataset_name, stats in all_results.items():
        txt_content += f"\n数据集: {dataset_name}\n"
        txt_content += "-" * 30 + "\n"
        
        for dimension, metrics in stats.items():
            # 添加到TXT报告
            txt_content += f"\n  维度: {dimension}\n"
            txt_content += f"    平均分: {metrics['mean']:.2f}\n"
            txt_content += f"    中位数: {metrics['median']:.2f}\n"
            txt_content += f"    标准差: {metrics['std']:.2f}\n"
            txt_content += f"    最低分: {metrics['min']:.2f}\n"
            txt_content += f"    最高分: {metrics['max']:.2f}\n"
            txt_content += f"    样本数: {metrics['count']}\n"
            
            # 添加到CSV数据
            csv_rows.append([
                dataset_name,
                dimension,
                f"{metrics['mean']:.2f}",
                f"{metrics['median']:.2f}",
                f"{metrics['std']:.2f}",
                f"{metrics['min']:.2f}",
                f"{metrics['max']:.2f}",
                str(metrics['count'])
            ])
    
    # 添加维度间的对比分析
    txt_content += "\n维度间对比分析\n"
    txt_content += "=" * 30 + "\n"
    
    # 计算所有数据集中每个维度的平均表现
    dimension_overall = defaultdict(list)
    for stats in all_results.values():
        for dimension, metrics in stats.items():
            dimension_overall[dimension].append({
                'mean': metrics['mean'],
                'std': metrics['std']
            })
    
    for dimension, metrics_list in dimension_overall.items():
        means = [m['mean'] for m in metrics_list]
        stds = [m['std'] for m in metrics_list]
        
        txt_content += f"\n{dimension}:\n"
        txt_content += f"  跨数据集平均分: {np.mean(means):.2f}\n"
        txt_content += f"  跨数据集平均标准差: {np.mean(stds):.2f}\n"
        txt_content += f"  数据集间的分数标准差: {np.std(means):.2f}\n"
    
    # 保存TXT报告
    txt_file = os.path.join(output_dir, "evaluation_analysis_report.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    
    # 保存CSV报告
    csv_file = os.path.join(output_dir, "evaluation_analysis_report.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"\n分析报告已生成:")
    print(f"  详细报告: {txt_file}")
    print(f"  CSV报告: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='分析数学问题评分结果')
    parser.add_argument('--input_dir', type=str, default="evaluation_results",
                      help='包含评分结果JSONL文件的目录')
    args = parser.parse_args()
    
    # 确保输入目录存在
    if not os.path.exists(args.input_dir):
        print(f"错误：目录 {args.input_dir} 不存在")
        return
    
    # 查找所有评分结果文件
    all_results = {}
    for file in os.listdir(args.input_dir):
        if file.endswith('_ratings.jsonl'):
            dataset_name = file.replace('_ratings.jsonl', '')
            file_path = os.path.join(args.input_dir, file)
            
            print(f"\n处理数据集: {dataset_name}")
            try:
                # 加载并分析数据
                data = load_jsonl(file_path)
                stats = analyze_ratings(data)
                all_results[dataset_name] = stats
                
                print(f"  成功分析 {len(data)} 条评分记录")
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue
    
    if all_results:
        # 生成综合报告
        generate_summary_report(all_results, args.input_dir)
    else:
        print("没有找到可分析的评分结果文件")

if __name__ == "__main__":
    main() 