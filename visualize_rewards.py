import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

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
    plt.savefig('reward_distributions.png', dpi=300)
    plt.close()
    print("已保存reward分布图: reward_distributions.png")

def plot_correlation(data):
    """方案2: 绘制三种reward之间的相关性热图"""
    # 为每个样本计算平均reward
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
    
    df = pd.DataFrame(df_data)
    
    # 计算相关性
    corr = df[['avg_rule_reward', 'avg_self_reward', 'avg_majority_reward']].corr()
    
    # 绘制热图
    plt.figure(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256) 
    sns.heatmap(corr, annot=True, cmap=cmap, vmin=-1, vmax=1, center=0, fmt='.4f')
    plt.title('三种Reward平均值的相关性')
    plt.tight_layout()
    plt.savefig('reward_correlation.png', dpi=300)
    plt.close()
    print("已保存相关性热图: reward_correlation.png")
    
    return df

def plot_scatter_matrix(df):
    """方案3: 绘制散点图矩阵"""
    plt.figure(figsize=(12, 10))
    scatter_cols = ['avg_rule_reward', 'avg_self_reward', 'avg_majority_reward']
    pd.plotting.scatter_matrix(df[scatter_cols], alpha=0.5, diagonal='kde')
    plt.tight_layout()
    plt.savefig('reward_scatter_matrix.png', dpi=300)
    plt.close()
    print("已保存散点图矩阵: reward_scatter_matrix.png")

def plot_rewards_by_idx(df, sample_size=100):
    """方案4: 绘制按idx排序的reward趋势图"""
    # 如果数据太多，进行采样
    if len(df) > sample_size:
        sampled_df = df.sample(sample_size, random_state=42).sort_values('idx')
    else:
        sampled_df = df.sort_values('idx')
    
    plt.figure(figsize=(15, 6))
    plt.plot(sampled_df['idx'], sampled_df['avg_rule_reward'], 'r-', label='Rule Reward')
    plt.plot(sampled_df['idx'], sampled_df['avg_self_reward'], 'g-', label='Self Reward')
    plt.plot(sampled_df['idx'], sampled_df['avg_majority_reward'], 'b-', label='Majority Reward')
    
    plt.title('三种Reward随idx变化趋势')
    plt.xlabel('样本idx')
    plt.ylabel('平均Reward值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rewards_by_idx.png', dpi=300)
    plt.close()
    print("已保存趋势图: rewards_by_idx.png")

def plot_boxplot(data):
    """方案5: 绘制三种reward的箱线图"""
    # 准备数据
    rule_box = []
    self_box = []
    majority_box = []
    
    for entry in data:
        rule_avg = np.mean(entry['rule_rewards'])
        self_avg = np.mean(entry['self_reward_rewards'])
        maj_avg = np.mean(entry['majority_rewards'])
        
        rule_box.append(rule_avg)
        self_box.append(self_avg)
        majority_box.append(maj_avg)
    
    box_data = pd.DataFrame({
        'Rule Reward': rule_box,
        'Self Reward': self_box,
        'Majority Reward': majority_box
    })
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=box_data)
    plt.title('三种Reward的箱线图比较')
    plt.ylabel('平均Reward值')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rewards_boxplot.png', dpi=300)
    plt.close()
    print("已保存箱线图: rewards_boxplot.png")

def plot_stacked_bars(data, sample_size=20):
    """方案6: 为一部分样本绘制堆叠柱状图"""
    # 随机选择样本
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sampled_data = [data[i] for i in indices]
    else:
        sampled_data = data
        
    idx_list = [entry['idx'] for entry in sampled_data]
    rule_mean = [np.mean(entry['rule_rewards']) for entry in sampled_data]
    self_mean = [np.mean(entry['self_reward_rewards']) for entry in sampled_data]
    maj_mean = [np.mean(entry['majority_rewards']) for entry in sampled_data]
    
    plt.figure(figsize=(15, 6))
    bar_width = 0.8
    
    plt.bar(idx_list, rule_mean, bar_width, label='Rule Reward', color='skyblue')
    plt.bar(idx_list, self_mean, bar_width, bottom=rule_mean, label='Self Reward', color='orange')
    plt.bar(idx_list, maj_mean, bar_width, bottom=[r+s for r,s in zip(rule_mean, self_mean)], 
            label='Majority Reward', color='green')
    
    plt.title('样本的三种Reward堆叠柱状图')
    plt.xlabel('样本idx')
    plt.ylabel('平均Reward值')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stacked_bars.png', dpi=300)
    plt.close()
    print("已保存堆叠柱状图: stacked_bars.png")

def plot_heatmap_by_sample(data, sample_size=20):
    """方案7: 绘制样本的reward热图"""
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
    plt.savefig('reward_heatmaps.png', dpi=300)
    plt.close()
    print("已保存样本热图: reward_heatmaps.png")

def main():
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
    
    # 方案4: 绘制按idx排序的reward趋势图
    plot_rewards_by_idx(df)
    
    # 方案5: 绘制三种reward的箱线图
    plot_boxplot(data)
    
    # 方案6: 为一部分样本绘制堆叠柱状图
    plot_stacked_bars(data)
    
    # 方案7: 绘制样本的reward热图
    plot_heatmap_by_sample(data)

if __name__ == "__main__":
    main()
    print("\n所有可视化图表生成完毕!") 