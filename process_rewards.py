import json
from collections import Counter

def process_rewards(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                # 获取 preds_group_idx
                preds = data['preds_group_idx']
                
                # 统计每个预测值出现的次数
                cnt = Counter(preds)
                # 获取出现次数最多的预测值
                majority = cnt.most_common(1)[0][0]
                
                # 生成新的奖励列表
                new_rewards = [1 if pred == majority else 0 for pred in preds]
                
                # 更新数据
                data['majority_rewards'] = new_rewards
                
                # 写入处理后的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
                continue

if __name__ == "__main__":
    input_file = "processed_data.jsonl"
    output_file = "processed_data_with_rewards.jsonl"
    process_rewards(input_file, output_file)
    print(f"处理完成！输出文件：{output_file}") 