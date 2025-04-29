import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import os

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Zen Hei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

def load_data(file_path):
    """加载jsonl文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_rewards(data):
    """分析三种reward的基本统计信息"""
    # 提取三种rewards
    rule_rewards_all = []
    self_rewards_all = []
    majority_rewards_all = []
    
    for entry in data:
        rule_rewards_all.extend(entry['rule_rewards'])
        self_rewards_all.extend(entry['self_reward_rewards'])
        majority_rewards_all.extend(entry['majority_rewards'])
    
    # 基本统计信息
    print("基本统计信息:")
    print("-" * 50)
    
    print("Rule Rewards:")
    rule_counter = Counter(rule_rewards_all)
    for val, count in sorted(rule_counter.items()):
        print(f"  值 {val}: {count}个 ({count/len(rule_rewards_all)*100:.2f}%)")
    
    print("\nSelf Rewards (除以5后):")
    print(f"  最小值: {min(self_rewards_all):.4f}")
    print(f"  最大值: {max(self_rewards_all):.4f}")
    print(f"  平均值: {np.mean(self_rewards_all):.4f}")
    print(f"  中位数: {np.median(self_rewards_all):.4f}")
    print(f"  标准差: {np.std(self_rewards_all):.4f}")
    
    print("\nMajority Rewards:")
    maj_counter = Counter(majority_rewards_all)
    for val, count in sorted(maj_counter.items()):
        print(f"  值 {val}: {count}个 ({count/len(majority_rewards_all)*100:.2f}%)")
    
    return rule_rewards_all, self_rewards_all, majority_rewards_all

def plot_distribution(rule_rewards, self_rewards, majority_rewards):
    """方案1: 绘制三种reward的分布图"""
    # 确保输出目录存在
    output_dir = "reward_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(18, 6))
    
    # Rule Rewards分布
    plt.subplot(1, 3, 1)
    sns.countplot(x=rule_rewards)
    plt.title('Rule Rewards分布')
    plt.xlabel('值')
    plt.ylabel('计数')
    
    # Self Rewards分布
    plt.subplot(1, 3, 2)
    plt.hist(self_rewards, bins=20, alpha=0.7)
    plt.title('Self Rewards分布 (除以5后)')
    plt.xlabel('值')
    plt.ylabel('计数')
    
    # Majority Rewards分布
    plt.subplot(1, 3, 3)
    sns.countplot(x=majority_rewards)
    plt.title('Majority Rewards分布')
    plt.xlabel('值')
    plt.ylabel('计数')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "方案1_reward_distributions.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存reward分布图: {output_path}")

def plot_correlation(data):
    """方案2: 绘制三种reward之间的相关性热图（使用向量相似度）"""
    # 确保输出目录存在
    output_dir = "reward_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 将每个样本的rewards作为向量
    rule_rewards_vectors = []
    self_rewards_vectors = []
    majority_rewards_vectors = []
    
    # 获取每个样本的reward向量
    for entry in data:
        rule_rewards_vectors.append(entry['rule_rewards'])
        self_rewards_vectors.append(entry['self_reward_rewards'])
        majority_rewards_vectors.append(entry['majority_rewards'])
    
    # 确保所有向量长度一致（如果有不一致的情况）
    min_length = min(min(len(v) for v in rule_rewards_vectors), 
                     min(len(v) for v in self_rewards_vectors),
                     min(len(v) for v in majority_rewards_vectors))
    
    # 裁剪向量到最小长度
    rule_rewards_vectors = [v[:min_length] for v in rule_rewards_vectors]
    self_rewards_vectors = [v[:min_length] for v in self_rewards_vectors]
    majority_rewards_vectors = [v[:min_length] for v in majority_rewards_vectors]
    
    # 将reward从0-1范围等比缩放到-1到1范围
    def scale_to_neg1_pos1(reward_vector):
        return [2 * r - 1 for r in reward_vector]  # r'= 2r - 1 将[0,1]映射到[-1,1]
    
    # 应用缩放转换
    rule_rewards_vectors = [scale_to_neg1_pos1(v) for v in rule_rewards_vectors]
    self_rewards_vectors = [scale_to_neg1_pos1(v) for v in self_rewards_vectors]
    majority_rewards_vectors = [scale_to_neg1_pos1(v) for v in majority_rewards_vectors]
    
    # 计算向量之间的余弦相似度
    # 首先计算每组向量与其他两组向量之间的平均余弦相似度
    n_samples = len(rule_rewards_vectors)
    
    # 初始化相似度矩阵
    similarity_matrix = np.zeros((3, 3))
    
    # 计算Rule与Self的相似度
    rule_self_sim = 0
    for i in range(n_samples):
        rule_vec = np.array(rule_rewards_vectors[i])
        self_vec = np.array(self_rewards_vectors[i])
        # 1 - cosine距离 = 余弦相似度
        sim = 1 - cosine(rule_vec, self_vec)
        rule_self_sim += sim
    rule_self_sim /= n_samples
    
    # 计算Rule与Majority的相似度
    rule_maj_sim = 0
    for i in range(n_samples):
        rule_vec = np.array(rule_rewards_vectors[i])
        maj_vec = np.array(majority_rewards_vectors[i])
        sim = 1 - cosine(rule_vec, maj_vec)
        rule_maj_sim += sim
    rule_maj_sim /= n_samples
    
    # 计算Self与Majority的相似度
    self_maj_sim = 0
    for i in range(n_samples):
        self_vec = np.array(self_rewards_vectors[i])
        maj_vec = np.array(majority_rewards_vectors[i])
        sim = 1 - cosine(self_vec, maj_vec)
        self_maj_sim += sim
    self_maj_sim /= n_samples
    
    # 填充相似度矩阵
    similarity_matrix[0, 0] = 1.0  # Rule自身相似度
    similarity_matrix[1, 1] = 1.0  # Self自身相似度
    similarity_matrix[2, 2] = 1.0  # Majority自身相似度
    
    similarity_matrix[0, 1] = rule_self_sim
    similarity_matrix[1, 0] = rule_self_sim
    
    similarity_matrix[0, 2] = rule_maj_sim
    similarity_matrix[2, 0] = rule_maj_sim
    
    similarity_matrix[1, 2] = self_maj_sim
    similarity_matrix[2, 1] = self_maj_sim
    
    # 创建DataFrame用于可视化
    corr_df = pd.DataFrame(
        similarity_matrix, 
        index=['Rule Reward', 'Self Reward', 'Majority Reward'],
        columns=['Rule Reward', 'Self Reward', 'Majority Reward']
    )
    
    # 绘制热图
    plt.figure(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256) 
    sns.heatmap(corr_df, annot=True, cmap=cmap, vmin=-1, vmax=1, center=0, fmt='.4f')
    plt.title('三种Reward向量的余弦相似度 (缩放至[-1,1]范围)')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "方案2_reward_correlation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存相关性热图: {output_path}")
    
    # 为与其他函数兼容，返回基于平均值的DataFrame
    df_data = []
    for entry in data:
        avg_rule = np.mean(entry['rule_rewards'])
        avg_self = np.mean(entry['self_reward_rewards'])
        avg_majority = np.mean(entry['majority_rewards'])
        df_data.append({
            'idx': entry['idx'],
            'avg_rule_reward': avg_rule,
            'avg_self_reward': avg_self, 
            'avg_majority_reward': avg_majority
        })
    
    return pd.DataFrame(df_data)

def plot_scatter_matrix(df):
    """方案3: 绘制散点图矩阵"""
    # 确保输出目录存在
    output_dir = "reward_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    scatter_cols = ['avg_rule_reward', 'avg_self_reward', 'avg_majority_reward']
    pd.plotting.scatter_matrix(df[scatter_cols], alpha=0.5, diagonal='kde')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "方案3_reward_scatter_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存散点图矩阵: {output_path}")

def plot_rewards_by_idx(df, data, sample_size=100):
    """方案4: 绘制按idx排序的reward余弦相似度趋势图"""
    # 确保输出目录存在
    output_dir = "reward_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果数据太多，进行采样
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        indices = sorted(indices)  # 排序以保持idx顺序
        sampled_data = [data[i] for i in indices]
    else:
        sampled_data = data
        
    # 获取所有样本的idx
    idx_list = [entry['idx'] for entry in sampled_data]
    
    # 计算每个样本的reward向量之间的余弦相似度
    rule_self_sim = []
    rule_maj_sim = []
    
    # 将reward从0-1范围等比缩放到-1到1范围
    def scale_to_neg1_pos1(reward_vector):
        return [2 * r - 1 for r in reward_vector]  # r'= 2r - 1 将[0,1]映射到[-1,1]
    
    for entry in sampled_data:
        # 获取三种reward向量
        rule_vec = np.array(scale_to_neg1_pos1(entry['rule_rewards']))
        self_vec = np.array(scale_to_neg1_pos1(entry['self_reward_rewards']))
        maj_vec = np.array(scale_to_neg1_pos1(entry['majority_rewards']))
        
        # 确保向量长度一致（取最小长度）
        min_len = min(len(rule_vec), len(self_vec), len(maj_vec))
        rule_vec = rule_vec[:min_len]
        self_vec = self_vec[:min_len]
        maj_vec = maj_vec[:min_len]
        
        # 计算余弦相似度
        rule_self = 1 - cosine(rule_vec, self_vec)
        rule_maj = 1 - cosine(rule_vec, maj_vec)
        
        rule_self_sim.append(rule_self)
        rule_maj_sim.append(rule_maj)
    
    # 绘制趋势图
    plt.figure(figsize=(15, 6))
    plt.plot(idx_list, rule_self_sim, 'r-', label='Rule-Self相似度')
    plt.plot(idx_list, rule_maj_sim, 'g-', label='Rule-Majority相似度')
    
    plt.title('Rule与其他两种Reward向量之间的余弦相似度趋势 (向量已缩放至[-1,1]范围)')
    plt.xlabel('样本idx')
    plt.ylabel('余弦相似度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "方案4_rewards_similarity_by_idx.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存余弦相似度趋势图: {output_path}")

def plot_heatmap_by_sample(data, sample_size=20):
    """方案5: 绘制样本的reward热图"""
    # 确保输出目录存在
    output_dir = "reward_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sampled_data = [data[i] for i in indices]
    else:
        sampled_data = data
    
    # 创建一个输出图像
    fig, axes = plt.subplots(sample_size, 3, figsize=(15, sample_size*2))
    
    for i, entry in enumerate(sampled_data):
        idx = entry['idx']
        rule = np.array(entry['rule_rewards']).reshape(1, -1)
        self_r = np.array(entry['self_reward_rewards']).reshape(1, -1)
        maj = np.array(entry['majority_rewards']).reshape(1, -1)
        
        # 绘制三种reward的热图
        sns.heatmap(rule, ax=axes[i, 0], cmap='Blues', cbar=False)
        sns.heatmap(self_r, ax=axes[i, 1], cmap='Oranges', cbar=False)
        sns.heatmap(maj, ax=axes[i, 2], cmap='Greens', cbar=False)
        
        axes[i, 0].set_title(f'Rule Reward (idx={idx})' if i == 0 else '')
        axes[i, 1].set_title(f'Self Reward (idx={idx})' if i == 0 else '')
        axes[i, 2].set_title(f'Majority Reward (idx={idx})' if i == 0 else '')
        
        # 移除坐标轴标签
        for j in range(3):
            axes[i, j].set_yticks([])
            axes[i, j].set_xticks([])
            axes[i, j].set_ylabel(f'idx={idx}')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "方案5_reward_heatmaps.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存样本热图: {output_path}")

def main():
    # 确保输出目录存在
    output_dir = "reward_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"图片将保存到目录: {output_dir}")
    
    # 加载数据
    file_path = "processed_data_with_rewards_extracted.jsonl"
    data = load_data(file_path)
    print(f"加载了 {len(data)} 条数据\n")
    
    # 分析rewards
    rule_rewards, self_rewards, majority_rewards = analyze_rewards(data)
    
    # 方案1: 绘制三种reward的分布图
    plot_distribution(rule_rewards, self_rewards, majority_rewards)
    
    # 方案2: 绘制三种reward之间的相关性热图
    df = plot_correlation(data)
    
    # 方案3: 绘制散点图矩阵
    plot_scatter_matrix(df)
    
    # 方案4: 绘制按idx排序的reward余弦相似度趋势图
    plot_rewards_by_idx(df, data)
    
    # 方案5: 绘制样本的reward热图
    plot_heatmap_by_sample(data)

if __name__ == "__main__":
    main()
    print("\n所有可视化图表生成完毕!") 