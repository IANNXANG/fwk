#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from collections import Counter
import re
import jieba
import matplotlib
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

# 创建输出目录
output_dir = "problem_visualizations"
os.makedirs(output_dir, exist_ok=True)

def load_data_from_jsonl(file_path, sample_size=100):
    """从JSONL文件加载数据并随机抽样，sample_size=float('inf')时加载所有数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 确保样本大小不超过数据集大小，sample_size为无穷大时加载所有数据
    if sample_size == float('inf'):
        sampled_data = data
    else:
        sample_size = min(sample_size, len(data))
        sampled_data = random.sample(data, sample_size)
    
    # 提取问题文本，优先使用problem字段，如果不存在则使用instruction字段
    problems = []
    for item in sampled_data:
        if 'problem' in item and item['problem']:
            problems.append(item['problem'])
        elif 'instruction' in item and item['instruction']:
            problems.append(item['instruction'])
        else:
            problems.append('')  # 如果两个字段都不存在，添加空字符串
    
    return problems

def extract_features_tfidf(problems):
    """使用TF-IDF提取特征"""
    # 对中文文本进行分词处理
    segmented_problems = []
    for problem in problems:
        words = jieba.cut(problem)
        segmented_problems.append(" ".join(words))
    
    # 使用TF-IDF提取特征
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(segmented_problems)
    return features.toarray()

def extract_features_bert(problems, device='cpu'):
    """使用BERT提取特征"""
    try:
        # 加载预训练的BERT模型
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
        model.to(device)
        model.eval()
        
        features = []
        
        # 使用tqdm显示进度
        for problem in tqdm(problems, desc="提取BERT特征"):
            inputs = tokenizer(problem, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用[CLS]标记的输出作为整个句子的表示
                features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten())
        
        return np.array(features)
    except Exception as e:
        print(f"BERT特征提取失败: {e}")
        return None

def apply_tsne(features, perplexity=30, n_components=2):
    """应用t-SNE进行降维"""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(features)

def plot_tsne_results(tsne_results, labels, title, output_file):
    """绘制t-SNE结果并保存可视化"""
    plt.figure(figsize=(12, 10))
    
    # 将结果转换为DataFrame以便于使用seaborn绘图
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'category': labels
    })
    
    # 使用seaborn绘制散点图
    sns.scatterplot(data=df, x='x', y='y', hue='category', palette='tab10', s=100, alpha=0.7)
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE维度1', fontsize=14)
    plt.ylabel('t-SNE维度2', fontsize=14)
    plt.legend(title='数据来源', fontsize=12)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file, dpi=300)
    plt.close()

def analyze_problem_lengths(all_problems, labels, output_file):
    """分析并可视化问题长度分布"""
    # 计算每个问题的长度
    problem_lengths = [len(problem) for problem in all_problems]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'length': problem_lengths,
        'category': labels
    })
    
    # 可视化各类别问题长度的分布
    plt.figure(figsize=(14, 8))
    
    # 绘制箱线图
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='category', y='length')
    plt.title('各类别问题长度的箱线图', fontsize=14)
    plt.xlabel('数据来源', fontsize=12)
    plt.ylabel('问题长度 (字符数)', fontsize=12)
    plt.xticks(rotation=45)
    
    # 绘制小提琴图
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x='category', y='length')
    plt.title('各类别问题长度的小提琴图', fontsize=14)
    plt.xlabel('数据来源', fontsize=12)
    plt.ylabel('问题长度 (字符数)', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # 输出各类别的平均长度
    avg_lengths = df.groupby('category')['length'].mean()
    return avg_lengths

def analyze_top_words(all_problems, labels, output_file, top_n=20):
    """分析并可视化问题文本中的高频词"""
    # 创建一个字典，按类别存储问题
    category_problems = {}
    unique_labels = list(set(labels))
    
    for label, problem in zip(labels, all_problems):
        if label not in category_problems:
            category_problems[label] = []
        category_problems[label].append(problem)
    
    # 对每个类别计算高频词
    category_top_words = {}
    
    for label in unique_labels:
        # 合并该类别的所有问题
        combined_text = " ".join(category_problems[label])
        # 使用jieba进行分词
        words = jieba.cut(combined_text)
        # 过滤掉停用词和标点符号
        filtered_words = [word for word in words if len(word) > 1 and not bool(re.match(r'[^\w\s]', word))]
        # 计算词频
        word_counts = Counter(filtered_words)
        # 保存top_n高频词
        category_top_words[label] = word_counts.most_common(top_n)
    
    # 可视化每个类别的高频词
    fig, axes = plt.subplots(len(unique_labels), 1, figsize=(12, 4 * len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        ax = axes[i] if len(unique_labels) > 1 else axes
        
        words = [item[0] for item in category_top_words[label]]
        counts = [item[1] for item in category_top_words[label]]
        
        # 创建水平条形图
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # 标签从顶部开始
        ax.set_title(f'类别 {label} 的前 {top_n} 个高频词')
        ax.set_xlabel('词频')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return category_top_words

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # JSONL文件路径
    jsonl_files = [
        "problemdata/1.math7500.jsonl",
        "problemdata/2.math4097.jsonl",
        "problemdata/3.math500seed.jsonl",
        "problemdata/4.math7507gen.jsonl",
        "problemdata/5.math3977gen.jsonl"
    ]
    
    # 从每个文件加载所有数据
    all_problems = []
    labels = []
    
    for i, file_path in enumerate(jsonl_files):
        print(f"加载文件: {file_path}")
        try:
            # 不限制sample_size，加载所有数据
            problems = load_data_from_jsonl(file_path, sample_size=float('inf'))
            all_problems.extend(problems)
            # 使用文件编号作为标签
            labels.extend([f"文件{i+1}" for _ in range(len(problems))])
            print(f"  成功加载 {len(problems)} 条记录")
        except Exception as e:
            print(f"  加载失败: {e}")
    
    if not all_problems:
        print("未加载任何数据，退出程序")
        return
    
    print(f"总共加载了 {len(all_problems)} 条问题记录")
    
    # 1. 使用TF-IDF特征进行t-SNE分析
    print("使用TF-IDF提取特征...")
    tfidf_features = extract_features_tfidf(all_problems)
    
    print("应用t-SNE降维...")
    tsne_results_tfidf = apply_tsne(tfidf_features)
    
    print("绘制TF-IDF特征的t-SNE可视化...")
    plot_tsne_results(
        tsne_results_tfidf, 
        labels, 
        "基于TF-IDF特征的问题文本t-SNE可视化", 
        os.path.join(output_dir, "tfidf_tsne_visualization.png")
    )
    
    # 2. 使用BERT特征进行t-SNE分析（如果可用）
    bert_features = extract_features_bert(all_problems)
    if bert_features is not None:
        print("应用t-SNE降维到BERT特征...")
        tsne_results_bert = apply_tsne(bert_features)
        
        print("绘制BERT特征的t-SNE可视化...")
        plot_tsne_results(
            tsne_results_bert, 
            labels, 
            "基于BERT特征的问题文本t-SNE可视化", 
            os.path.join(output_dir, "bert_tsne_visualization.png")
        )
    
    # 3. 分析问题长度
    print("分析问题长度分布...")
    avg_lengths = analyze_problem_lengths(
        all_problems, 
        labels, 
        os.path.join(output_dir, "problem_length_analysis.png")
    )
    print("各类别的平均问题长度:")
    for category, avg_length in avg_lengths.items():
        print(f"  {category}: {avg_length:.2f} 字符")
    
    # 4. 分析高频词
    print("分析问题中的高频词...")
    category_top_words = analyze_top_words(
        all_problems, 
        labels, 
        os.path.join(output_dir, "top_words_analysis.png")
    )
    
    print("所有分析完成，可视化结果保存在 'problem_visualizations' 目录中")

if __name__ == "__main__":
    main() 