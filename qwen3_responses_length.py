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

def save_jsonl(data, file_path, mode='w'):
    """保存数据到JSONL文件，支持写入模式(w)和追加模式(a)"""
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
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

def continuous_worker(task_queue, result_queue, client, model_name, stop_event, max_retries=3, retry_delay=5):
    """持续从任务队列获取任务并处理"""
    while not stop_event.is_set():
        try:
            # 从队列获取任务，如果队列为空，等待1秒后重试
            try:
                item_idx, question = task_queue.get(timeout=1)
            except queue.Empty:
                # 如果队列为空且stop_event被设置，则退出
                if stop_event.is_set():
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

def process_dataset(data, client, model_name, output_prefix, output_dir, batch_size=256, max_retries=3, retry_delay=5, sample_size=None):
    """处理单个数据集，调用API生成回答，并统计token长度"""
    # 不再抽样，处理全部数据
    if sample_size and len(data) > sample_size:
        random.seed(42)  # 设置随机种子以确保可重复性
        sampled_data = random.sample(data, sample_size)
        print(f"随机采样 {sample_size} 条数据 (总数据量: {len(data)})")
    else:
        sampled_data = data
        print(f"处理全部 {len(data)} 条数据")
    
    results = [None] * len(sampled_data)  # 预分配结果列表
    response_lengths = []
    
    print(f"处理数据集: {output_prefix}")
    print(f"数据集样本数量: {len(sampled_data)}")
    print(f"使用批处理大小: {batch_size}")
    
    # 创建结果文件路径
    processed_file = os.path.join(output_dir, f'{output_prefix}_with_responses.jsonl')
    
    # 初始化输出文件
    with open(processed_file, 'w', encoding='utf-8') as f:
        pass
    
    # 创建任务队列和结果队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # 创建停止事件
    stop_event = threading.Event()
    
    # 准备所有任务
    for i, item in enumerate(sampled_data):
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
    num_workers = min(batch_size, len(sampled_data))
    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(
            target=continuous_worker,
            args=(task_queue, result_queue, client, model_name, stop_event, max_retries, retry_delay),
            daemon=True  # 设为daemon线程，主线程结束时会自动终止
        )
        thread.start()
        threads.append(thread)
    
    # 设置进度条
    pbar = tqdm(total=len(sampled_data), desc="处理数据")
    
    # 批量保存结果
    batch_results = []
    batch_size_save = 100
    
    # 处理结果队列
    completed = 0
    try:
        while completed < len(sampled_data):
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
                    
                    # 准备保存数据
                    data_item = sampled_data[item_idx].copy()
                    data_item['qwen_response'] = result["answer"]
                    data_item['qwen_token_count'] = result["token_count"]
                    if result["error"]:
                        data_item['qwen_error'] = result["error"]
                    
                    # 添加到批量保存列表
                    batch_results.append(data_item)
                    
                    # 每达到batch_size_save条数据进行一次批量写入
                    if len(batch_results) >= batch_size_save:
                        save_jsonl(batch_results, processed_file, mode='a')
                        batch_results = []  # 清空缓存
                
                # 更新进度条
                pbar.update(1)
                completed += 1
                
                # 标记结果已处理
                result_queue.task_done()
            
            except queue.Empty:
                # 队列暂时为空，检查是否所有任务都已完成
                if task_queue.empty() and completed >= len(sampled_data):
                    break
                continue
    
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在保存已完成的结果...")
    
    finally:
        # 保存剩余批量结果
        if batch_results:
            save_jsonl(batch_results, processed_file, mode='a')
        
        # 关闭进度条
        pbar.close()
        
        # 设置停止事件，通知所有线程停止
        stop_event.set()
        
        # 等待一段合理的时间让线程自行终止
        time.sleep(2)
        
        # 计算统计信息
        stats = {
            'mean': float(np.mean(response_lengths)) if response_lengths else 0,
            'median': float(np.median(response_lengths)) if response_lengths else 0,
            'min': float(np.min(response_lengths)) if response_lengths else 0,
            'max': float(np.max(response_lengths)) if response_lengths else 0,
            'count': len(response_lengths)
        }
        
        print(f"\n完成处理: {completed}/{len(sampled_data)} 条数据")
        print(f"数据已保存至: {processed_file}")
        print(f"统计信息:\n  平均长度: {stats['mean']:.2f}\n  中位数长度: {stats['median']:.2f}\n  最短: {stats['min']}\n  最长: {stats['max']}")
    
    return {
        'file_path': processed_file,
        'response_lengths': response_lengths,
        'stats': stats 
    }

def load_response_lengths_from_jsonl(file_path):
    """从JSONL文件加载响应长度数据"""
    response_lengths = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'qwen_token_count' in item and item['qwen_token_count'] is not None:
                    response_lengths.append(item['qwen_token_count'])
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return response_lengths

def plot_distribution(datasets_results, dataset_names, output_dir):
    """根据处理结果绘制简化分布图（不包含平均值和中位数线）"""
    for i, (result_data, dataset_name) in enumerate(zip(datasets_results, dataset_names)):
        # 如果result_data中不包含response_lengths，则从文件中加载
        if 'response_lengths' not in result_data or not result_data['response_lengths']:
            file_path = result_data.get('file_path')
            if file_path and os.path.exists(file_path):
                response_lengths = load_response_lengths_from_jsonl(file_path)
            else:
                print(f"警告：无法找到数据集 {dataset_name} 的响应长度数据")
                continue
        else:
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
        
        # 保存图片
        output_file = os.path.join(output_dir, f'{dataset_name}_response_length_distribution.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"分布图已保存至 {output_file}")
        plt.close()

def save_average_lengths(datasets_results, dataset_names, output_dir):
    """将所有数据集的平均长度保存到txt文件中"""
    output_file = os.path.join(output_dir, 'average_response_lengths.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("数据集回答长度平均值\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result_data, dataset_name in zip(datasets_results, dataset_names):
            stats = result_data.get('stats', {})
            mean_val = stats.get('mean', 0)
            count = stats.get('count', 0)
            f.write(f"{dataset_name} ({count} 样本): {mean_val:.2f} tokens\n")
    
    print(f"平均长度数据已保存至 {output_file}")

def main(args):
    # 数据集文件
    datasets = [
        {
            'file': "problemdata/1.math7500.jsonl",
            'name': "math7500"
        },
        {
            'file': "problemdata/3.math500seed.jsonl",
            'name': "math500seed"
        },
        {
            'file': "problemdata/6.iter0.jsonl",
            'name': "iter0"
        },
        {
            'file': "problemdata/7.iter1.jsonl",
            'name': "iter1"
        },
        {
            'file': "problemdata/8.iter2.jsonl",
            'name': "iter2"
        }
    ]
    
    # 创建输出目录
    output_dir = "response_length_analysis_full"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建OpenAI客户端连接到本地vLLM服务
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY"  # vLLM服务可能不需要API密钥
    )
    
    # 处理每个数据集并保存中间结果
    results = []
    dataset_names = []
    
    # 处理所有数据集
    for dataset in datasets:
        print(f"\n加载数据集: {dataset['file']}")
        try:
            data = load_jsonl(dataset['file'])
            
            result = process_dataset(
                data=data, 
                client=client,
                model_name=args.model,
                output_prefix=dataset['name'],
                output_dir=output_dir,
                batch_size=args.batch_size,
                sample_size=args.sample_size if args.full_data == False else None
            )
            
            results.append(result)
            dataset_names.append(dataset['name'])
        except Exception as e:
            print(f"处理数据集 {dataset['file']} 时出错: {e}")
    
    # 根据中间结果绘图
    if results:
        print("\n根据保存的结果绘制图表...")
        plot_distribution(results, dataset_names, output_dir)
        
        # 保存平均长度数据到txt文件
        save_average_lengths(results, dataset_names, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析多个数学问题数据集的回答长度')
    parser.add_argument('--port', type=int, default=8001, help='vLLM服务端口')
    parser.add_argument('--model', type=str, default="8001vllm", help='模型名称')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    parser.add_argument('--sample_size', type=int, default=500, help='每个数据集的采样数量（仅当--full_data=false时使用）')
    parser.add_argument('--full_data', action='store_true', default=True, help='处理全部数据，不进行采样')
    
    args = parser.parse_args()
    main(args) 