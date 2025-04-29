import json

def verify_rewards(input_file):
    total_lines = 0
    matched_lines = 0
    mismatched_indices = []  # 修改：用于收集不匹配的idx
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                # 获取 maj_num 和 majority_rewards
                maj_num = data['maj_num']
                majority_rewards = data['majority_rewards']
                
                # 计算 majority_rewards 中 1 的个数
                ones_count = sum(1 for x in majority_rewards if x == 1)
                
                # 统计总数和匹配数
                total_lines += 1
                if maj_num == ones_count:
                    matched_lines += 1
                else:
                    mismatched_indices.append(data['idx'])  # 修改：记录不匹配的idx
                    print(f"第 {line_num} 行不一致:")
                    print(f"maj_num: {maj_num}")
                    print(f"majority_rewards 中 1 的个数: {ones_count}")
                    print(f"idx: {data['idx']}")  # 修改：显示idx
                    print(f"majority_rewards: {majority_rewards}")
                    print(f"rule_rewards sum: {sum(data['rule_rewards'])}")  # 新增：打印rule_rewards的和
                    print("-" * 50)
                
            except json.JSONDecodeError as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # 计算并打印统计结果
    if total_lines > 0:
        match_ratio = (matched_lines / total_lines) * 100
        print(f"\n统计结果:")
        print(f"总行数: {total_lines}")
        print(f"匹配行数: {matched_lines}")
        print(f"匹配比例: {match_ratio:.2f}%")
        print(f"\n不匹配的列表: {mismatched_indices}")  # 修改：打印不匹配的idx列表
    else:
        print("没有有效数据行")

if __name__ == "__main__":
    input_file = "processed_data_with_rewards.jsonl"
    verify_rewards(input_file)
    print("验证完成！") 