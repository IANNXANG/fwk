import json
import numpy as np
from collections import Counter

# 读取jsonl文件
data = []
with open('processed_data_with_rewards_extracted.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 计算每个样本的reward hit rate
hit_rates = []
for item in data:
    rule_rewards = item['rule_rewards']
    majority_rewards = item['majority_rewards']
    
    # 计算匹配的比例
    matches = sum(1 for r, m in zip(rule_rewards, majority_rewards) if float(r) == float(m))
    hit_rate = matches / len(rule_rewards)
    hit_rates.append((item['idx'], hit_rate))

# 统计分析
hit_rate_values = [rate for _, rate in hit_rates]
avg_hit_rate = np.mean(hit_rate_values)
median_hit_rate = np.median(hit_rate_values)
min_hit_rate = np.min(hit_rate_values)
max_hit_rate = np.max(hit_rate_values)

# 计算分布情况
hit_rate_counts = Counter([round(rate, 2) for _, rate in hit_rates])
sorted_rates = sorted(hit_rate_counts.items())

print(f"Reward Hit Rate分析结果:")
print(f"平均值: {avg_hit_rate:.4f}")
print(f"中位数: {median_hit_rate:.4f}")
print(f"最小值: {min_hit_rate:.4f}")
print(f"最大值: {max_hit_rate:.4f}")
print("\nReward Hit Rate分布:")
for rate, count in sorted_rates:
    print(f"{rate:.2f}: {count}个样本 ({count/len(hit_rates)*100:.2f}%)")

# 计算不同区间的分布
print("\n区间分布:")
intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for start, end in intervals:
    count = sum(1 for _, rate in hit_rates if start <= rate < end)
    print(f"{start} - {end}: {count}个样本 ({count/len(hit_rates)*100:.2f}%)")

# 完全匹配的样本数
perfect_match = sum(1 for _, rate in hit_rates if rate == 1.0)
print(f"\n完全匹配(hit_rate=1.0): {perfect_match}个样本 ({perfect_match/len(hit_rates)*100:.2f}%)")

# 查找一致性较低的样本（hit_rate < 0.6）
low_hit_rates = [(idx, rate) for idx, rate in hit_rates if rate < 0.6]
low_hit_rates.sort(key=lambda x: x[1])  # 按hit_rate从低到高排序

print("\n一致性较低的样本(hit_rate < 0.6):")
print(f"总计: {len(low_hit_rates)}个样本")
print("前10个一致性最低的样本:")
for i, (idx, rate) in enumerate(low_hit_rates[:10]):
    print(f"样本 {idx}：hit_rate = {rate:.4f}")
    
    # 分析不一致的位置
    item = data[idx]
    rule_rewards = item['rule_rewards']
    majority_rewards = item['majority_rewards']
    
    inconsistent_positions = [(i, rule_rewards[i], majority_rewards[i]) 
                              for i in range(len(rule_rewards)) 
                              if float(rule_rewards[i]) != float(majority_rewards[i])]
    
    print(f"  - 不一致位置数: {len(inconsistent_positions)}")
    print(f"  - 样本中规则奖励的1的比例: {sum(1 for r in rule_rewards if float(r) == 1.0)/len(rule_rewards):.2f}")
    print(f"  - 样本中多数投票奖励的1的比例: {sum(1 for m in majority_rewards if float(m) == 1.0)/len(majority_rewards):.2f}")

# 可视化分布 (如果环境支持matplotlib)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist([rate for _, rate in hit_rates], bins=20, edgecolor='black')
    plt.xlabel('Reward Hit Rate')
    plt.ylabel('Sample Count')
    plt.title('Reward Hit Rate Distribution')
    plt.savefig('reward_hit_rate_distribution.png')
    print("\n已保存分布图到 reward_hit_rate_distribution.png")
except ImportError:
    print("\n无法导入matplotlib, 跳过图表生成") 