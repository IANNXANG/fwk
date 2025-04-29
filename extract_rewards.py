import json

def process_data(input_file, output_file):
    """
    处理jsonl文件，只保留idx和三个rewards，并且将self_reward_rewards所有数值除以5
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # 提取所需字段
                processed_entry = {
                    'idx': data['idx'],
                    'rule_rewards': data['rule_rewards'],
                    'self_reward_rewards': [val / 5 for val in data['self_reward_rewards']],  # 除以5
                    'majority_rewards': data['majority_rewards']
                }
                
                processed_data.append(processed_entry)
                
            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
                continue
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"处理完成! 总共处理了 {len(processed_data)} 条数据")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    input_file = "processed_data_with_rewards.jsonl"
    output_file = "processed_data_with_rewards_extracted.jsonl"
    process_data(input_file, output_file) 