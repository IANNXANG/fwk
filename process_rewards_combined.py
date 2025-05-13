import json
from collections import Counter

def process_jsonl_file(input_file, output_file):
    """
    处理jsonl文件，提取并处理rewards信息：
    1. 只保留idx和三种rewards
    2. 计算majority_rewards
    3. 将self_reward_rewards归一化（除以5）
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # 计算majority rewards
                preds = data['preds_group_idx']
                cnt = Counter(preds)
                majority = cnt.most_common(1)[0][0]
                majority_rewards = [1 if pred == majority else 0 for pred in preds]
                
                # 提取所需字段并处理
                processed_entry = {
                    'idx': data['idx'],
                    'rule_rewards': data['rule_rewards'],
                    'self_reward_rewards': [val / 5 for val in data['self_reward_rewards']],  # 归一化到0~1
                    'majority_rewards': majority_rewards
                }
                
                processed_data.append(processed_entry)
                
            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in data: {e}")
                continue
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"处理完成! 总共处理了 {len(processed_data)} 条数据")
    print(f"结果已保存到: {output_file}")

def main():
    # 处理两个输入文件
    input_files = [
        "rewards_correction/llama_32_3b_math500_with_self_math_reward_with_rule_reward_maj_eval.jsonl",
        "rewards_correction/qwen25_7b_math500_with_self_math_reward_with_rule_reward_maj_eval.jsonl"
    ]
    
    output_files = [
        "rewards_correction/llama_32_3b_processed_rewards.jsonl",
        "rewards_correction/qwen25_7b_processed_rewards.jsonl"
    ]
    
    for input_file, output_file in zip(input_files, output_files):
        print(f"\n开始处理文件: {input_file}")
        process_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    main() 