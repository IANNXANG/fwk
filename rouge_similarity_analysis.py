#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import tqdm
import matplotlib.font_manager as fm

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

def main():
    # 加载两个文件
    print("加载数据文件...")
    gen_data = load_jsonl("problemdata/4.math7507gen.jsonl")
    seed_data = load_jsonl("problemdata/3.math500seed.jsonl")
    
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
    
    # 绘制分布图
    plt.figure(figsize=(12, 7))
    plt.hist(sim_array, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('ROUGE-L 相似度', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.title('生成问题与种子问题的最大ROUGE-L相似度分布', fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    
    # 添加平均值和中位数线
    mean_val = np.mean(sim_array)
    median_val = np.median(sim_array)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'平均值: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'中位数: {median_val:.4f}')
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('rouge_similarity_distribution.png', dpi=300)
    print("分布图已保存至 rouge_similarity_distribution.png")
    
    # 保存相似度数据
    with open('rouge_similarity_data.json', 'w', encoding='utf-8') as f:
        json.dump({
            'max_similarities': max_similarities,
            'stats': {
                'mean': float(np.mean(sim_array)),
                'median': float(np.median(sim_array)),
                'min': float(np.min(sim_array)),
                'max': float(np.max(sim_array))
            }
        }, f, indent=2)
    print("相似度数据已保存至 rouge_similarity_data.json")

if __name__ == "__main__":
    main() 