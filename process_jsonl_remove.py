import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                # 删除 code、problem、preds、self_reward_resps 和 answer 字段
                fields_to_remove = ['code', 'problem', 'preds', 'self_reward_resps', 'answer']
                for field in fields_to_remove:
                    if field in data:
                        del data[field]
                # 写入处理后的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
                continue

if __name__ == "__main__":
    input_file = "llama32_3b_0424_math500_maj_eval_with_self_rewards_with_rule_rewards.jsonl"
    output_file = "processed_data.jsonl"
    process_jsonl(input_file, output_file)
    print(f"处理完成！输出文件：{output_file}") 