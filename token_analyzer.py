#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys

def load_tokens_from_jsonl(file_path):
    """从JSONL文件加载token计数数据"""
    tokens = []
    total_items = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_items += 1
                item = json.loads(line)
                if 'qwen_token_count' in item and item['qwen_token_count'] is not None:
                    tokens.append(item['qwen_token_count'])
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    print(f"文件 {file_path} 共有 {total_items} 条数据，成功读取 {len(tokens)} 个token数")
    return tokens

def analyze_difficulty(tokens, lower_threshold, upper_threshold):
    """根据阈值分析简单、中等和困难题目的数量和比例"""
    easy = sum(1 for t in tokens if t < lower_threshold)
    medium = sum(1 for t in tokens if lower_threshold <= t <= upper_threshold)
    hard = sum(1 for t in tokens if t > upper_threshold)
    
    total = len(tokens)
    if total == 0:
        return {
            'easy': (0, 0),
            'medium': (0, 0),
            'hard': (0, 0)
        }
    
    return {
        'easy': (easy, easy / total * 100),
        'medium': (medium, medium / total * 100),
        'hard': (hard, hard / total * 100)
    }

def print_results(results, lower_threshold, upper_threshold):
    """格式化输出分析结果"""
    print(f"\n难度分析结果 (阈值: 简单 < {lower_threshold}, 中等 {lower_threshold}-{upper_threshold}, 困难 > {upper_threshold}):")
    print(f"简单题: {results['easy'][0]} 题，占比 {results['easy'][1]:.2f}%")
    print(f"中等题: {results['medium'][0]} 题，占比 {results['medium'][1]:.2f}%")
    print(f"困难题: {results['hard'][0]} 题，占比 {results['hard'][1]:.2f}%")
    print(f"总计: {sum([results['easy'][0], results['medium'][0], results['hard'][0]])} 题")

def main():
    # 默认文件路径
    files = [
        "math7507gen_with_responses.jsonl",
        "math3977gen_with_responses.jsonl"
    ]
    
    # 获取用户输入的阈值
    try:
        lower_threshold = int(input("请输入简单题的上限阈值 (默认 150): ") or "150")
        upper_threshold = int(input("请输入中等题的上限阈值 (默认 500): ") or "500")
    except ValueError:
        print("输入无效，使用默认值: 简单 < 150, 中等 150-500, 困难 > 500")
        lower_threshold = 150
        upper_threshold = 500
    
    # 分析每个文件
    for file_path in files:
        print(f"\n正在分析文件: {file_path}")
        tokens = load_tokens_from_jsonl(file_path)
        if tokens:
            results = analyze_difficulty(tokens, lower_threshold, upper_threshold)
            print_results(results, lower_threshold, upper_threshold)
        else:
            print(f"文件 {file_path} 中没有有效的token数据")
    
    # 提示用户输入任意键退出
    input("\n按回车键退出...")

if __name__ == "__main__":
    main() 