#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from openai import OpenAI
import argparse
import random
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import re

# 评分维度定义
DIMENSIONS = [
    {
        "name": "task_validity",
        "description": "任务有效性",
        "prompt": """
You are an expert evaluator of math problems. Please evaluate the following math problem on whether it describes a valid mathematical task that is clear, unambiguous, and solvable.

Math Problem:
{problem}

First, analyze this problem in detail, considering its clarity, structure, and whether it presents a well-defined mathematical task.

After your analysis, provide your rating on a scale from 0 to 5, where:
0 = Completely invalid mathematical task
5 = Perfectly valid, clear mathematical task

Your response should include your detailed analysis followed by your rating.
Make sure to include your final rating in this exact format at the END of your response:

FINAL RATING: [0-5]

Remember to use only whole numbers (integers) from 0 to 5 for your rating, not decimals or fractions.
"""
    },
    {
        "name": "problem_appropriateness",
        "description": "问题适当性",
        "prompt": """
You are an expert evaluator of math problems. Please evaluate the following math problem on whether it is appropriate in terms of complexity, clarity, and educational value.

Math Problem:
{problem}

First, analyze this problem in detail, considering its complexity level, clarity of presentation, and educational value.

After your analysis, provide your rating on a scale from 0 to 5, where:
0 = Completely inappropriate mathematical problem
5 = Perfectly appropriate mathematical problem

Your response should include your detailed analysis followed by your rating.
Make sure to include your final rating in this exact format at the END of your response:

FINAL RATING: [0-5]

Remember to use only whole numbers (integers) from 0 to 5 for your rating, not decimals or fractions.
"""
    },
    {
        "name": "solution_correctness",
        "description": "解决方案正确性",
        "prompt": """
You are an expert evaluator of math problems. Please evaluate the following math problem on whether it can be solved with standard mathematical techniques and has a definitive correct answer.

Math Problem:
{problem}

First, analyze this problem in detail, considering whether it has a clear solution path and whether it can be solved using standard mathematical approaches.

After your analysis, provide your rating on a scale from 0 to 5, where:
0 = No correct solution path exists
5 = Clear, correct solution path exists

Your response should include your detailed analysis followed by your rating.
Make sure to include your final rating in this exact format at the END of your response:

FINAL RATING: [0-5]

Remember to use only whole numbers (integers) from 0 to 5 for your rating, not decimals or fractions.
"""
    },
    {
        "name": "overall_quality",
        "description": "整体质量",
        "prompt": """
You are an expert evaluator of math problems. Please evaluate the overall quality of the following math problem.

Math Problem:
{problem}

First, analyze this problem in detail, considering all aspects including clarity, educational value, appropriateness, and whether it tests meaningful mathematical concepts.

After your analysis, provide your rating on a scale from 0 to 5, where:
0 = Extremely poor quality mathematical problem
5 = Excellent quality mathematical problem

Your response should include your detailed analysis followed by your rating.
Make sure to include your final rating in this exact format at the END of your response:

FINAL RATING: [0-5]

Remember to use only whole numbers (integers) from 0 to 5 for your rating, not decimals or fractions.
"""
    }
]

def load_jsonl(file_path):
    """加载JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path, mode='w'):
    """保存数据到JSONL文件，支持写入模式(w)和追加模式(a)"""
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_rating(text, max_attempts=3):
    """从模型响应中提取分数"""
    # 多种可能的评分模式匹配，优先检查最后的FINAL RATING格式
    patterns = [
        # 标准的FINAL RATING格式(优先匹配)
        r'(?:^|\n)(?:\*+|\#+)?\s*FINAL RATING:\s*(\d+)(?:\s*|\*+|\#+|\n|$)',  # 带有标题格式的评分
        r'(?:^|\n)FINAL RATING:\s*(\d+)(?:\s*|\n|$)',  # 标准格式: FINAL RATING: 5
        r'(?:^|\n)最终[评分评级][:：]\s*(\d+)(?:\s*|\n|$)', # 中文格式: 最终评分: 5
        
        # 带有装饰符号的RATING格式
        r'(?:^|\n)(?:\*+|\#+)?\s*RATING:\s*(\d+)(?:\s*|\*+|\#+|\n|$)',  # Markdown形式: **RATING: 5**
        r'(?:\n|^)---+\s*\n+\s*FINAL RATING:\s*(\d+)',  # 带分隔线的格式
        r'(?:\n|^)---+\s*\n+\s*RATING:\s*(\d+)',        # 带分隔线的格式
        
        # 基本的RATING格式
        r'(?:^|\n)RATING:\s*(\d+)(?:\s*|\n|$)',          # 标准格式: RATING: 5
        r'(?:^|\n)评分[:：]\s*(\d+)(?:\s*|\n|$)',         # 中文格式: 评分: 5
        
        # 常规文本中的评分表达
        r'(\d+)[\s/]5',                # 分数格式: 4/5 或 4 5
        r'(\d+)分',                    # 中文分数: 4分
        r'评级为\s*(\d+)',             # 评级描述: 评级为4
        r'给[这此]个[问题题目数学问题]打(\d+)分', # 打分描述: 给这个问题打4分
        r'[\[\(](\d+)[\]\)]'           # 括号中的分数: [4] 或 (4)
    ]
    
    # 非法小数评分的模式
    decimal_patterns = [
        r'FINAL RATING:\s*(\d+\.\d+)', # 小数格式: FINAL RATING: 4.5
        r'RATING:\s*(\d+\.\d+)',       # 小数格式: RATING: 4.5
        r'评分[:：]\s*(\d+\.\d+)',      # 中文小数: 评分: 4.5
        r'(\d+\.\d+)[\s/]5',           # 小数分数: 4.5/5
        r'(\d+\.\d+)分'                # 中文小数分: 4.5分
    ]
    
    # 首先尝试查找回答末尾的评分 - 这是最可能的位置
    last_paragraphs = text.split('\n\n')[-3:]  # 获取最后三个段落
    last_text = '\n'.join(last_paragraphs)
    
    # 优先从末尾文本中搜索标准格式
    for pattern in patterns[:6]:  # 优先使用前6个更可能出现在末尾的模式
        match = re.search(pattern, last_text)
        if match:
            try:
                rating = int(match.group(1))
                if 0 <= rating <= 5:
                    return rating
            except ValueError:
                continue
    
    # 如果末尾没找到，搜索整个文本
    for attempt in range(max_attempts):
        # 检查是否有小数评分
        for pattern in decimal_patterns:
            match = re.search(pattern, text)
            if match:
                # 找到小数评分，转换为整数
                try:
                    decimal_rating = float(match.group(1))
                    rating = round(decimal_rating)  # 四舍五入到最接近的整数
                    if 0 <= rating <= 5:
                        # 不打印调试信息
                        return rating
                except ValueError:
                    continue
        
        # 尝试所有整数评分模式
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    rating = int(match.group(1))
                    if 0 <= rating <= 5:
                        return rating
                except ValueError:
                    continue
        
        # 尝试查找最后出现的数字 (通常在结论部分)
        if attempt == max_attempts - 1:
            # 分析最后几行寻找数字
            lines = text.split('\n')
            for i in range(len(lines)-1, max(0, len(lines)-10), -1):  # 检查最后10行
                line = lines[i]
                # 查找独立的数字0-5
                isolated_numbers = re.findall(r'\b([0-5])\b', line)
                if isolated_numbers:
                    rating = int(isolated_numbers[0])
                    # 不打印调试信息
                    return rating
        
        # 如果没有模式匹配成功，尝试查找文本中的独立数字
        if attempt == max_attempts - 1:
            # 查找独立的数字
            isolated_numbers = re.findall(r'\b([0-5])\b', text)
            if isolated_numbers:
                # 取第一个出现的0-5之间的数字
                for num in isolated_numbers:
                    rating = int(num)
                    if 0 <= rating <= 5:
                        # 不打印调试信息
                        return rating
        
        # 如果没有成功提取，进行下一次尝试
        if attempt < max_attempts - 1:
            # 不打印中间失败信息
            pass
    
    # 所有尝试都失败后，尝试从文本中推断评分
    # 查找表示积极程度的关键词
    positive_indicators = ['excellent', 'perfect', 'very good', 'high quality', 'clear', 
                          '优秀', '完美', '很好', '高质量', '清晰', 'well-defined', 'unambiguous',
                          'solvable', 'standard', 'straightforward', 'valid']
    negative_indicators = ['poor', 'invalid', 'unclear', 'ambiguous', 'bad', 
                          '差', '无效', '不清晰', '模糊', '糟糕', 'confusing', 'not clear',
                          'not solvable', 'unsolvable', 'ill-defined']
    
    # 计算关键词权重时考虑末尾段落更重要
    last_paragraphs_text = '\n'.join(last_paragraphs)
    positive_score = 0
    negative_score = 0
    
    # 优先分析末尾文本
    for word in positive_indicators:
        if word.lower() in last_paragraphs_text.lower():
            positive_score += 2  # 末尾出现的关键词权重加倍
    
    for word in negative_indicators:
        if word.lower() in last_paragraphs_text.lower():
            negative_score += 2  # 末尾出现的关键词权重加倍
    
    # 再分析整个文本
    for word in positive_indicators:
        if word.lower() in text.lower():
            positive_score += 1
    
    for word in negative_indicators:
        if word.lower() in text.lower():
            negative_score += 1
    
    # 基于关键词比例推断分数
    if positive_score > 0 or negative_score > 0:
        total = positive_score + negative_score
        ratio = positive_score / total if total > 0 else 0
        inferred_rating = int(round(ratio * 5))
        # 不打印调试信息
        return inferred_rating
    
    # 最终失败时，记录问题
    return None

def process_single_item(client, model_name, problem, dimension, max_retries=3, retry_delay=5):
    """处理单个问题的一个维度评分"""
    # 初始提示
    prompt = dimension["prompt"].format(problem=problem)
    
    for attempt in range(max_retries):
        try:
            # 调用API获取评分
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
                top_p=0.8,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            
            # 提取回答内容
            answer_text = response.choices[0].message.content
            
            # 从回答中提取评分
            rating = extract_rating(answer_text)
            
            if rating is not None:
                return {
                    "dimension": dimension["name"],
                    "rating": rating,
                    "explanation": answer_text,
                    "error": None
                }
            else:
                # 如果无法提取评分，创建一个更明确的提示再试一次
                if attempt < max_retries - 1:
                    # 不打印中间尝试信息
                    # 添加更明确的提示
                    prompt = f"""
You are an expert evaluator of math problems. Your task is to rate the following math problem on a scale from 0 to 5.

Math Problem:
{problem}

Please analyze this problem for the following dimension: {dimension['description']}

IMPORTANT INSTRUCTIONS:
1. First provide your detailed analysis of the problem.
2. Then, at the END of your response, provide your final rating.
3. Your rating MUST be a WHOLE NUMBER from 0 to 5 (no decimals or fractions).
4. Use EXACTLY this format for your rating: "FINAL RATING: X" where X is 0, 1, 2, 3, 4, or 5.
5. The "FINAL RATING: X" line MUST be at the very end of your response.

Start with your detailed analysis, and end with your rating in the specified format.
"""
                    time.sleep(retry_delay)
                else:
                    # 所有尝试都失败后，保存错误信息和完整回答供后续分析
                    print(f"\n\n============ 评分提取失败 ============")
                    print(f"问题: {problem[:100]}...")
                    print(f"维度: {dimension['description']}")
                    print(f"完整回答:\n{answer_text}")
                    print(f"============ 评分提取失败 ============\n\n")
                    return {
                        "dimension": dimension["name"],
                        "rating": None,
                        "explanation": answer_text,
                        "error": "无法提取评分"
                    }
                
        except Exception as e:
            if attempt < max_retries - 1:
                # 不打印重试信息
                time.sleep(retry_delay)
            else:
                # 最终失败时才打印错误信息
                print(f"\n\n============ API调用失败 ============")
                print(f"问题: {problem[:100]}...")
                print(f"维度: {dimension['description']}")
                print(f"错误: {e}")
                print(f"============ API调用失败 ============\n\n")
                return {
                    "dimension": dimension["name"],
                    "rating": None,
                    "explanation": None,
                    "error": str(e)
                }

def continuous_worker(task_queue, result_queue, client, model_name, stop_event, max_retries=3, retry_delay=5):
    """持续从任务队列获取任务并处理"""
    while not stop_event.is_set():
        try:
            # 从队列获取任务，如果队列为空，等待1秒后重试
            try:
                item_idx, problem, dimension = task_queue.get(timeout=1)
            except queue.Empty:
                # 如果队列为空且stop_event被设置，则退出
                if stop_event.is_set():
                    break
                continue
            
            # 处理问题评分
            if problem is not None:
                result = process_single_item(client, model_name, problem, dimension, max_retries, retry_delay)
                result_queue.put((item_idx, dimension["name"], result))
            else:
                result_queue.put((item_idx, dimension["name"], None))
            
            # 标记任务完成
            task_queue.task_done()
        except Exception as e:
            # 只打印关键错误
            print(f"\n\n============ 工作线程错误 ============")
            print(f"错误: {e}")
            print(f"============ 工作线程错误 ============\n\n")
            # 出错后短暂休息，避免过快重试
            time.sleep(1)

def process_dataset(data, client, model_name, output_prefix, batch_size=256, max_retries=3, retry_delay=5, sample_size=500):
    """处理数据集，为每个问题的每个维度评分"""
    # 采样数据
    if len(data) > sample_size:
        random.seed(42)  # 设置随机种子以确保可重复性
        sampled_data = random.sample(data, sample_size)
    else:
        sampled_data = data
    
    print(f"处理数据集: {output_prefix}")
    print(f"样本数量: {len(sampled_data)}")
    print(f"评分维度: {len(DIMENSIONS)}")
    print(f"使用批处理大小: {batch_size}")
    
    # 初始化结果字典
    results = {dimension["name"]: [] for dimension in DIMENSIONS}
    
    # 创建输出文件
    results_file = f'{output_prefix}_ratings.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        pass
    
    # 创建任务队列和结果队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # 创建停止事件
    stop_event = threading.Event()
    
    # 为每个问题的每个维度创建任务
    total_tasks = 0
    for i, item in enumerate(sampled_data):
        # 从数据中提取问题
        if 'problem' in item:
            problem = item['problem']
        elif 'instruction' in item:
            problem = item['instruction']
        else:
            print(f"警告: 无法在第{i}条数据中找到问题字段，跳过")
            problem = None
        
        # 为每个维度添加任务
        if problem:
            for dimension in DIMENSIONS:
                task_queue.put((i, problem, dimension))
                total_tasks += 1
    
    # 创建并启动工作线程池
    num_workers = min(batch_size, total_tasks)
    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(
            target=continuous_worker,
            args=(task_queue, result_queue, client, model_name, stop_event, max_retries, retry_delay),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # 设置进度条
    pbar = tqdm(total=total_tasks, desc="评分进度")
    
    # 处理结果队列
    completed = 0
    failed_extractions = 0
    batch_results = []  # 用于批量保存的结果缓存
    batch_size_save = 100  # 每100条数据保存一次
    
    try:
        while completed < total_tasks:
            try:
                # 从结果队列获取结果
                item_idx, dimension_name, result = result_queue.get(timeout=1)
                
                if result:
                    # 保存结果
                    results[dimension_name].append(result)
                    
                    # 收集结果用于批量写入
                    if result["rating"] is not None:
                        result_entry = {
                            "item_id": item_idx,
                            "dimension": dimension_name,
                            "rating": result['rating'],
                            "explanation": result['explanation']
                        }
                        batch_results.append(result_entry)
                    
                    # 每达到batch_size_save条数据进行一次批量写入
                    if len(batch_results) >= batch_size_save:
                        save_jsonl(batch_results, results_file, mode='a')
                        batch_results = []  # 清空缓存
                    
                    # 检查是否提取失败
                    if result['rating'] is None:
                        failed_extractions += 1
                
                # 更新进度条
                pbar.update(1)
                completed += 1
                
                # 标记结果已处理
                result_queue.task_done()
            
            except queue.Empty:
                # 队列暂时为空，检查是否所有任务都已完成
                if task_queue.empty() and completed >= total_tasks:
                    break
                continue
    
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在保存已完成的结果...")
    
    finally:
        # 将剩余的批量结果写入文件
        if batch_results:
            save_jsonl(batch_results, results_file, mode='a')
        
        # 关闭进度条
        pbar.close()
        
        # 设置停止事件，通知所有线程停止
        stop_event.set()
        
        # 等待一段合理的时间让线程自行终止
        time.sleep(2)
        
        if failed_extractions > 0:
            print(f"警告：有 {failed_extractions} 条评分提取失败。")
        
        # 计算各维度的平均分
        stats = {}
        for dimension in DIMENSIONS:
            dim_name = dimension["name"]
            ratings = [r["rating"] for r in results[dim_name] if r["rating"] is not None]
            if ratings:
                stats[dim_name] = {
                    'mean': float(np.mean(ratings)),
                    'median': float(np.median(ratings)),
                    'min': float(np.min(ratings)),
                    'max': float(np.max(ratings)),
                    'count': len(ratings)
                }
            else:
                stats[dim_name] = {
                    'mean': 0,
                    'median': 0,
                    'min': 0,
                    'max': 0,
                    'count': 0
                }
        
        print(f"\n完成处理: {completed}/{total_tasks} 个评分任务")
        print(f"评分结果已保存至: {results_file}")
    
    return {
        'file_path': results_file,
        'stats': stats
    }

def generate_summary_report(all_results, output_dir, report_prefix):
    """生成所有数据集的评分综合报告"""
    if not all_results:
        print("没有可用的评分结果，无法生成报告")
        return
    
    # 准备报告数据
    datasets = []
    dimension_names = []
    dimension_descriptions = {}
    
    # 收集所有维度名称和描述
    for dimension in DIMENSIONS:
        dimension_names.append(dimension["name"])
        dimension_descriptions[dimension["name"]] = dimension["description"]
    
    # 准备CSV数据
    csv_rows = []
    header = ["Dataset"]
    for dim_name in dimension_names:
        header.append(f"{dimension_descriptions[dim_name]} (平均分)")
        header.append(f"{dimension_descriptions[dim_name]} (中位数)")
        header.append(f"{dimension_descriptions[dim_name]} (有效评分数)")
    
    csv_rows.append(header)
    
    # 准备TXT报告内容
    txt_content = f"数学问题评分综合报告\n"
    txt_content += f"{'='*50}\n\n"
    txt_content += f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    txt_content += f"数据集总数: {len(all_results)}\n"
    txt_content += f"评分维度: {len(dimension_names)}\n\n"
    
    # 创建维度汇总数据
    all_ratings_by_dimension = {dim: [] for dim in dimension_names}
    
    # 处理每个数据集的结果
    for dataset_name, result in all_results.items():
        datasets.append(dataset_name)
        stats = result.get('stats', {})
        
        # 为CSV添加一行
        row = [dataset_name]
        for dim_name in dimension_names:
            dim_stats = stats.get(dim_name, {})
            mean = dim_stats.get('mean', 0)
            median = dim_stats.get('median', 0)
            count = dim_stats.get('count', 0)
            
            # 收集维度评分数据
            if mean > 0:  # 只收集有效的评分
                all_ratings_by_dimension[dim_name].append(mean)
            
            row.append(f"{mean:.2f}")
            row.append(f"{median:.2f}")
            row.append(f"{count}")
        
        csv_rows.append(row)
        
        # 为TXT报告添加数据集部分
        txt_content += f"\n数据集: {dataset_name}\n"
        txt_content += f"{'-'*30}\n"
        for dim_name in dimension_names:
            dim_stats = stats.get(dim_name, {})
            txt_content += f"  {dimension_descriptions[dim_name]} ({dim_name}):\n"
            txt_content += f"    平均分: {dim_stats.get('mean', 0):.2f}\n"
            txt_content += f"    中位数: {dim_stats.get('median', 0):.2f}\n"
            txt_content += f"    最低分: {dim_stats.get('min', 0)}\n"
            txt_content += f"    最高分: {dim_stats.get('max', 0)}\n"
            txt_content += f"    有效评分数: {dim_stats.get('count', 0)}\n"
    
    # 添加维度总结
    txt_content += f"\n所有数据集维度评分总结\n"
    txt_content += f"{'-'*30}\n"
    for dim_name in dimension_names:
        ratings = all_ratings_by_dimension[dim_name]
        if ratings:
            avg = sum(ratings) / len(ratings)
            txt_content += f"  {dimension_descriptions[dim_name]} ({dim_name}):\n"
            txt_content += f"    所有数据集平均分: {avg:.2f}\n"
            txt_content += f"    包含的数据集数量: {len(ratings)}\n"
    
    # 保存CSV报告
    csv_file = os.path.join(output_dir, f"{report_prefix}_summary.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    # 保存TXT报告
    txt_file = os.path.join(output_dir, f"{report_prefix}_summary.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    
    print(f"\n综合报告已生成:")
    print(f"  CSV报告: {csv_file}")
    print(f"  TXT报告: {txt_file}")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='对数学问题数据集进行多维度评分')
    parser.add_argument('--files', type=str, nargs='+', default=[
        "problemdata/1.math7500.jsonl",
        "problemdata/3.math500seed.jsonl", 
        "problemdata/6.iter0.jsonl", 
        "problemdata/7.iter1.jsonl", 
        "problemdata/8.iter2.jsonl"
    ], help='要处理的JSONL文件路径列表')
    parser.add_argument('--port', type=int, default=8001, help='vLLM服务端口')
    parser.add_argument('--model', type=str, default="8001vllm", help='模型名称')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    parser.add_argument('--sample_size', type=int, default=500, help='每个数据集的样本数量')
    parser.add_argument('--output_dir', type=str, default="evaluation_results", help='输出目录')
    parser.add_argument('--report_prefix', type=str, default="math_evaluation", help='报告文件前缀')
    
    args = parser.parse_args()
    
    # 创建OpenAI客户端连接到本地vLLM服务
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY"  # vLLM服务可能不需要API密钥
    )
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有数据集
    all_results = {}
    for file_path in args.files:
        # 提取数据集名称
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        output_prefix = os.path.join(args.output_dir, dataset_name)
        
        print(f"\n{'='*60}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue
            
        # 加载数据集
        print(f"加载数据集: {file_path}")
        try:
            data = load_jsonl(file_path)
            print(f"成功加载 {len(data)} 条数据")
            
            # 处理数据集并获取结果
            result = process_dataset(
                data=data, 
                client=client,
                model_name=args.model,
                output_prefix=output_prefix,
                batch_size=args.batch_size,
                sample_size=args.sample_size
            )
            
            # 保存结果
            all_results[dataset_name] = result
        
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时发生错误: {e}")
            continue
    
    # 生成综合报告
    generate_summary_report(all_results, args.output_dir, args.report_prefix)

if __name__ == "__main__":
    main() 