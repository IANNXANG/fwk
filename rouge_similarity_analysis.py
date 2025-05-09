#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import tqdm
import matplotlib.font_manager as fm
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_jsonl(file_path):
    """加载JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def process_dataset(gen_file, seed_file, output_prefix):
    """处理单个数据集，计算相似度并保存中间结果"""
    # 加载两个文件
    print(f"加载数据文件: {gen_file}...")
    gen_data = load_jsonl(gen_file)
    seed_data = load_jsonl(seed_file)
    
    print(f"生成问题数量: {len(gen_data)}")
    print(f"种子问题数量: {len(seed_data)}")
    
    # 初始化ROUGE计算器
    print("初始化ROUGE评分器...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # 存储最大相似度
    max_similarities = []
    
    # 对每个生成的问题，计算与种子问题的最大相似度
    print(f"计算ROUGE-L相似度...")
    for i, gen_item in enumerate(tqdm.tqdm(gen_data)):
        # 获取problem字段
        if 'problem' not in gen_item:
            print(f"警告: 第{i}条数据中没有'problem'字段，跳过")
            continue
            
        gen_problem = gen_item['problem']
        
        # 计算与所有种子问题的相似度
        similarities = []
        for seed_item in seed_data:
            if 'instruction' not in seed_item:
                continue
                
            seed_instruction = seed_item['instruction']
            
            # 计算ROUGE-L评分
            try:
                scores = scorer.score(gen_problem, seed_instruction)
                similarities.append(scores['rougeL'].fmeasure)
            except Exception as e:
                print(f"计算第{i}条数据的ROUGE-L分数时出错: {e}")
                similarities.append(0)
        
        # 记录最大相似度
        if similarities:
            max_similarity = max(similarities)
            max_similarities.append(max_similarity)
    
    # 分析相似度分布
    print("分析相似度分布...")
    sim_array = np.array(max_similarities)
    
    print(f"总样本数: {len(max_similarities)}")
    print(f"平均相似度: {np.mean(sim_array):.4f}")
    print(f"中位数相似度: {np.median(sim_array):.4f}")
    print(f"最小相似度: {np.min(sim_array):.4f}")
    print(f"最大相似度: {np.max(sim_array):.4f}")
    
    # 分布统计
    bins = np.arange(0, 1.05, 0.05)
    hist, bin_edges = np.histogram(sim_array, bins=bins)
    
    for i in range(len(hist)):
        start, end = bin_edges[i], bin_edges[i+1]
        count = hist[i]
        percentage = (count / len(max_similarities)) * 100
        print(f"相似度 {start:.2f}-{end:.2f}: {count} 个样本 ({percentage:.2f}%)")
    
    # 保存相似度数据
    result_data = {
        'max_similarities': max_similarities,
        'stats': {
            'mean': float(np.mean(sim_array)),
            'median': float(np.median(sim_array)),
            'min': float(np.min(sim_array)),
            'max': float(np.max(sim_array))
        }
    }
    
    output_file = f'{output_prefix}_rouge_similarity_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    print(f"相似度数据已保存至 {output_file}")
    
    return result_data

def plot_distribution(datasets_results, dataset_names):
    """根据保存的中间结果绘制分布图"""
    for i, (result_data, dataset_name) in enumerate(zip(datasets_results, dataset_names)):
        max_similarities = result_data['max_similarities']
        sim_array = np.array(max_similarities)
        
        plt.figure(figsize=(12, 7))
        plt.hist(sim_array, bins=20, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('ROUGE-L Similarity', fontsize=14)
        plt.ylabel('Sample Count', fontsize=14)
        plt.title(f'Distribution of Maximum ROUGE-L Similarity for {dataset_name}', fontsize=16)
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0, 1)  # 固定横轴范围为0-1
        
        # 添加平均值和中位数线
        mean_val = result_data['stats']['mean']
        median_val = result_data['stats']['median']
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.4f}')
        
        # 添加图例
        plt.legend(fontsize=12)
        
        # 保存图片
        output_file = f'{dataset_name}_rouge_similarity_distribution.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"分布图已保存至 {output_file}")
        plt.close()

def main():
    # 数据集文件
    datasets = [
        {
            'gen_file': "problemdata/4.math7507gen.jsonl",
            'seed_file': "problemdata/3.math500seed.jsonl",
            'name': "math7507gen"
        },
        {
            'gen_file': "problemdata/5.math3977gen.jsonl",
            'seed_file': "problemdata/3.math500seed.jsonl",
            'name': "math3977gen"
        }
    ]
    
    # 确保输出目录存在
    os.makedirs("problemdata", exist_ok=True)
    
    # 处理每个数据集并保存中间结果
    results = []
    dataset_names = []
    
    for dataset in datasets:
        print(f"\n处理数据集: {dataset['name']}")
        result = process_dataset(dataset['gen_file'], dataset['seed_file'], dataset['name'])
        results.append(result)
        dataset_names.append(dataset['name'])
    
    # 根据中间结果绘图
    print("\n根据保存的结果绘制图表...")
    plot_distribution(results, dataset_names)

if __name__ == "__main__":
    main() 