#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from tabulate import tabulate

def load_jsonl(file_path):
    """加载JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_qwen_tokenizer():
    """创建Qwen3分词器"""
    try:
        # 使用Qwen3分词器
        return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
    except Exception as e:
        print(f"加载Qwen分词器失败: {e}")
        print("请确保已安装transformers库: pip install transformers")
        raise

def count_tokens(text, tokenizer):
    """使用Qwen分词器计算文本的token数量"""
    return len(tokenizer.encode(text))

def find_min_token_problems(file_path, top_n=10):
    """找出指定数据集中token数最少的问题"""
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n处理数据集: {dataset_name}")
    
    try:
        # 加载数据
        data = load_jsonl(file_path)
        print(f"成功加载 {len(data)} 条数据")
        
        # 初始化分词器
        tokenizer = create_qwen_tokenizer()
        
        # 存储问题及其token数量
        problems = []
        
        for i, item in enumerate(tqdm(data, desc=f"分析 {dataset_name} 数据集中的问题")):
            # 从数据中提取问题
            text = None
            if 'problem' in item:
                text = item['problem']
                field_name = 'problem'
            elif 'instruction' in item:
                text = item['instruction']
                field_name = 'instruction'
            else:
                continue
            
            if text:
                # 计算token数量
                token_count = count_tokens(text, tokenizer)
                
                # 存储问题信息
                problems.append({
                    'id': item.get('id', i),
                    'field': field_name,
                    'token_count': token_count,
                    'text': text
                })
        
        # 按token数排序
        problems.sort(key=lambda x: x['token_count'])
        
        # 返回token数最少的top_n个问题
        min_token_problems = problems[:top_n]
        
        return dataset_name, min_token_problems
    
    except Exception as e:
        print(f"处理数据集 {file_path} 时出错: {str(e)}")
        return dataset_name, []

def save_results(all_results, output_dir):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存汇总报告
    summary_file = os.path.join(output_dir, "min_token_problems_summary.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("每个数据集中Token数最少的问题汇总\n")
        f.write("=" * 60 + "\n\n")
        
        for dataset_name, problems in all_results.items():
            f.write(f"\n数据集: {dataset_name}\n")
            f.write("-" * 50 + "\n")
            
            if not problems:
                f.write("没有找到有效问题\n")
                continue
            
            for i, problem in enumerate(problems):
                f.write(f"[{i+1}] ID: {problem['id']}, Token数: {problem['token_count']}, 字段: {problem['field']}\n")
                f.write(f"问题内容: {problem['text'][:200]}{'...' if len(problem['text']) > 200 else ''}\n")
                f.write("-" * 40 + "\n")
    
    print(f"汇总报告已保存至: {summary_file}")
    
    # 为每个数据集生成详细的CSV文件
    for dataset_name, problems in all_results.items():
        if not problems:
            continue
        
        # 转换为DataFrame
        df = pd.DataFrame(problems)
        
        # 保存为CSV
        csv_file = os.path.join(output_dir, f"{dataset_name}_min_token_problems.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"数据集 {dataset_name} 的详细结果已保存至: {csv_file}")

def display_results(all_results):
    """在终端显示结果"""
    print("\n\n每个数据集中Token数最少的问题：")
    print("=" * 80)
    
    for dataset_name, problems in all_results.items():
        print(f"\n数据集: {dataset_name}")
        print("-" * 80)
        
        if not problems:
            print("没有找到有效问题")
            continue
        
        # 准备表格数据
        table_data = []
        for i, problem in enumerate(problems):
            # 截断过长的问题文本，适合终端显示
            truncated_text = problem['text'][:50] + ('...' if len(problem['text']) > 50 else '')
            table_data.append([
                i+1,
                problem['id'],
                problem['token_count'],
                problem['field'],
                truncated_text
            ])
        
        # 使用tabulate打印表格
        print(tabulate(
            table_data,
            headers=["序号", "ID", "Token数", "字段", "问题内容(截断)"],
            tablefmt="grid"
        ))

def main():
    parser = argparse.ArgumentParser(description='找出各数据集中token数最少的问题')
    parser.add_argument('--top_n', type=int, default=10, help='每个数据集显示的最少token问题数量')
    parser.add_argument('--output_dir', type=str, default="min_token_problems", help='输出目录')
    args = parser.parse_args()
    
    # 数据集文件列表
    datasets = [
        "problemdata/1.math7500.jsonl",
        "problemdata/3.math500seed.jsonl",
        "problemdata/6.iter0.jsonl",
        "problemdata/7.iter1.jsonl",
        "problemdata/8.iter2.jsonl"
    ]
    
    # 存储所有数据集的结果
    all_results = {}
    
    # 处理每个数据集
    for dataset_file in datasets:
        # 检查文件是否存在
        if not os.path.exists(dataset_file):
            print(f"警告: 文件 {dataset_file} 不存在，跳过")
            continue
        
        # 找出token数最少的问题
        dataset_name, min_token_problems = find_min_token_problems(
            file_path=dataset_file,
            top_n=args.top_n
        )
        
        all_results[dataset_name] = min_token_problems
    
    # 显示结果
    display_results(all_results)
    
    # 保存结果
    save_results(all_results, args.output_dir)

if __name__ == "__main__":
    main() 