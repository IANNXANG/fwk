import json
from collections import Counter

def load_reference_rewards(ref_file):
    """加载参考奖励数据"""
    ref_rewards = {}
    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                idx = data.get('idx')
                if idx is not None:
                    ref_rewards[idx] = {
                        'rule_rewards': data.get('rule_rewards', []),
                        'self_reward_rewards': data.get('self_reward_rewards', [])
                    }
            except json.JSONDecodeError as e:
                print(f"Error processing reference file line: {e}")
    return ref_rewards

def process_rewards_with_matching(input_file, output_file, ref_rewards):
    """处理奖励数据并匹配参考奖励"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                idx = data.get('idx')
                
                # 获取并处理 preds_group_idx
                preds = data.get('preds_group_idx', [])
                if preds:
                    # 统计每个预测值出现的次数
                    cnt = Counter(preds)
                    # 获取出现次数最多的预测值
                    majority = cnt.most_common(1)[0][0]
                    # 生成新的奖励列表
                    new_rewards = [1 if pred == majority else 0 for pred in preds]
                    data['majority_rewards'] = new_rewards
                
                # 匹配参考奖励
                if idx is not None and idx in ref_rewards:
                    data['rule_rewards'] = ref_rewards[idx]['rule_rewards']
                    data['self_reward_rewards'] = ref_rewards[idx]['self_reward_rewards']
                
                # 删除多余字段
                fields_to_keep = ['idx', 'majority_rewards', 'rule_rewards', 'self_reward_rewards']
                data_cleaned = {k: v for k, v in data.items() if k in fields_to_keep}
                
                # 写入处理后的数据
                f_out.write(json.dumps(data_cleaned, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error processing input file line: {e}")
                continue

def main():
    # 加载参考奖励数据
    ref_file = "processed_data_with_rewards_extracted.jsonl"
    print("加载参考奖励数据...")
    ref_rewards = load_reference_rewards(ref_file)
    print(f"已加载 {len(ref_rewards)} 条参考奖励数据")

    # 处理 Qwen 数据
    qwen_input = "qwen25_7b_math500_with_self_math_reward_with_rule_reward_maj_eval.jsonl"
    qwen_output = "qwen25_7b_maj_rewards.jsonl"
    print(f"\n处理 Qwen 数据...")
    process_rewards_with_matching(qwen_input, qwen_output, ref_rewards)
    print(f"Qwen 数据处理完成，已保存到：{qwen_output}")

    # 处理 LLaMA 数据
    llama_input = "llama_32_3b_with_self_math_reward_with_rule_reward_maj_eval.jsonl"
    llama_output = "llama_32_3b_maj_rewards.jsonl"
    print(f"\n处理 LLaMA 数据...")
    process_rewards_with_matching(llama_input, llama_output, ref_rewards)
    print(f"LLaMA 数据处理完成，已保存到：{llama_output}")

if __name__ == "__main__":
    main() 