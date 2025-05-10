#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from openai import OpenAI
import tiktoken
import argparse
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# 移除中文字体设置
plt.rcParams['axes.unicode_minus'] = False

def load_jsonl(file_path):
    """加载JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def append_jsonl(item, file_path):
    """追加单条数据到JSONL文件"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def count_tokens(text, encoding_name="cl100k_base"):
    """计算文本的token数量"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def process_single_item(client, model_name, question, max_retries=3, retry_delay=5):
    """处理单个问题并返回结果"""
    for attempt in range(max_retries):
        try:
            # 调用API生成回答，关闭思考模式
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": question}],
                max_tokens=2048,
                temperature=0,
                top_p=0.8,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            
            # 提取回答内容
            answer_text = response.choices[0].message.content
            
            # 统计token数量
            token_count = count_tokens(answer_text)
            
            # 返回结果
            return {
                "question": question,
                "answer": answer_text,
                "token_count": token_count,
                "usage": response.usage.total_tokens if hasattr(response, 'usage') else None,
                "error": None
            }
                
        except Exception as e:
            print(f"调用API时出错 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                return {
                    "question": question,
                    "answer": None,
                    "token_count": None,
                    "error": str(e)
                }

def continuous_worker(task_queue, result_queue, client, model_name, max_retries=3, retry_delay=5):
    """持续从任务队列获取任务并处理"""
    while True:
        try:
            # 从队列获取任务，如果队列为空，等待1秒后重试
            try:
                item_idx, question = task_queue.get(timeout=1)
            except queue.Empty:
                # 检查是否应该退出
                if threading.current_thread().daemon:
                    break
                continue
            
            # 处理问题
            if question is not None:
                result = process_single_item(client, model_name, question, max_retries, retry_delay)
                result_queue.put((item_idx, result))
            else:
                result_queue.put((item_idx, None))
            
            # 标记任务完成
            task_queue.task_done()
        except Exception as e:
            print(f"工作线程出错: {e}")
            # 出错后短暂休息，避免过快重试
            time.sleep(1)

def process_dataset(data, client, model_name, output_prefix, batch_size=256, max_retries=3, retry_delay=5):
    """处理单个数据集，调用API生成回答，并统计token长度"""
    results = [None] * len(data)  # 预分配结果列表
    response_lengths = []
    
    # 采样500条数据
    # if len(data) > 500:
    #     random.seed(42)  # 设置随机种子以确保可重复性
    #     data = random.sample(data, 500)
    
    print(f"处理数据集: {output_prefix}")
    print(f"数据集样本数量: {len(data)}")
    print(f"使用批处理大小: {batch_size}")
    
    # 创建输出文件并初始化
    output_file = f'{output_prefix}_responses.json'
    processed_file = f'{output_prefix}_with_responses.jsonl'
    
    # 初始化输出文件
    with open(processed_file, 'w', encoding='utf-8') as f:
        pass
    
    # 创建任务队列和结果队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # 准备所有任务
    for i, item in enumerate(data):
        # 从数据中提取问题
        if 'problem' in item:
            question = item['problem']
        elif 'instruction' in item:
            question = item['instruction']
        else:
            print(f"警告: 无法在第{i}条数据中找到问题字段，跳过")
            question = None
        
        # 加入任务队列
        task_queue.put((i, question))
    
    # 创建并启动工作线程池
    num_workers = min(batch_size, len(data))
    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(
            target=continuous_worker,
            args=(task_queue, result_queue, client, model_name, max_retries, retry_delay),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # 设置进度条
    pbar = tqdm(total=len(data), desc="处理数据")
    
    # 处理结果队列
    completed = 0
    try:
        while completed < len(data):
            try:
                # 从结果队列获取结果
                item_idx, result = result_queue.get(timeout=1)
                
                if result is not None:
                    # 保存结果
                    results[item_idx] = result
                    
                    if result["token_count"] is not None:
                        response_lengths.append(result["token_count"])
                    else:
                        response_lengths.append(0)
                    
                    # 实时更新数据文件
                    data_item = data[item_idx].copy()
                    data_item['qwen_response'] = result["answer"]
                    data_item['qwen_token_count'] = result["token_count"]
                    if result["error"]:
                        data_item['qwen_error'] = result["error"]
                    append_jsonl(data_item, processed_file)
                
                # 更新进度条
                pbar.update(1)
                completed += 1
                
                # 标记结果已处理
                result_queue.task_done()
            
            except queue.Empty:
                # 队列暂时为空，检查是否所有任务都已完成
                if task_queue.empty() and completed >= len(data):
                    break
                continue
    
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在保存已完成的结果...")
    
    finally:
        # 关闭进度条
        pbar.close()
        
        # 等待所有工作线程完成
        for thread in threads:
            thread.daemon = False  # 允许线程正常退出
        
        # 过滤掉None值
        results = [r for r in results if r is not None]
        
        # 保存最终结果到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'response_lengths': response_lengths,
                'stats': {
                    'mean': float(np.mean(response_lengths)) if response_lengths else 0,
                    'median': float(np.median(response_lengths)) if response_lengths else 0,
                    'min': float(np.min(response_lengths)) if response_lengths else 0,
                    'max': float(np.max(response_lengths)) if response_lengths else 0
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至 {output_file}")
        print(f"完成处理: {completed}/{len(data)} 条数据")
    
    return {
        'results': results,
        'response_lengths': response_lengths,
        'stats': {
            'mean': float(np.mean(response_lengths)) if response_lengths else 0,
            'median': float(np.median(response_lengths)) if response_lengths else 0,
            'min': float(np.min(response_lengths)) if response_lengths else 0,
            'max': float(np.max(response_lengths)) if response_lengths else 0
        }
    }

def plot_distribution(datasets_results, dataset_names):
    """根据保存的中间结果绘制分布图"""
    for i, (result_data, dataset_name) in enumerate(zip(datasets_results, dataset_names)):
        response_lengths = result_data['response_lengths']
        
        if not response_lengths:
            print(f"警告：数据集 {dataset_name} 没有有效的响应长度数据，跳过绘图")
            continue
            
        plt.figure(figsize=(12, 7))
        
        # 创建直方图，固定刻度
        bins = range(0, max(response_lengths) + 50, 50)
        plt.hist(response_lengths, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 设置标题和标签（使用英文）
        plt.xlabel('Response Length (tokens)', fontsize=14)
        plt.ylabel('Sample Count', fontsize=14)
        plt.title(f'Distribution of Response Length for {dataset_name}', fontsize=16)
        plt.grid(axis='y', alpha=0.75)
        
        # 添加平均值和中位数线
        mean_val = result_data['stats']['mean']
        median_val = result_data['stats']['median']
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # 添加图例
        plt.legend(fontsize=12)
        
        # 保存图片
        output_file = f'{dataset_name}_response_length_distribution.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"分布图已保存至 {output_file}")
        plt.close()

def main(args):
    # 数据集文件
    datasets = [
        {
            'file': args.file1,
            'name': "math7507gen"
        },
        {
            'file': args.file2,
            'name': "math3977gen"
        }
    ]
    
    # 确保输出目录存在
    os.makedirs("problemdata", exist_ok=True)
    
    # 创建OpenAI客户端连接到本地vLLM服务
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY"  # vLLM服务可能不需要API密钥
    )
    
    # 处理每个数据集并保存中间结果
    results = []
    dataset_names = []
    
    for dataset in datasets:
        print(f"\n加载数据集: {dataset['file']}")
        data = load_jsonl(dataset['file'])
        
        result = process_dataset(
            data=data, 
            client=client,
            model_name=args.model,
            output_prefix=dataset['name'],
            batch_size=args.batch_size
        )
        
        results.append(result)
        dataset_names.append(dataset['name'])
    
    # 根据中间结果绘图
    print("\n根据保存的结果绘制图表...")
    plot_distribution(results, dataset_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用Qwen3模型处理数据并分析响应长度')
    parser.add_argument('--file1', type=str, default="problemdata/4.math7507gen.jsonl", help='第一个要处理的JSONL文件路径')
    parser.add_argument('--file2', type=str, default="problemdata/5.math3977gen.jsonl", help='第二个要处理的JSONL文件路径')
    parser.add_argument('--port', type=int, default=8001, help='vLLM服务端口')
    parser.add_argument('--model', type=str, default="8001vllm", help='模型名称')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    
    args = parser.parse_args()
    main(args) 