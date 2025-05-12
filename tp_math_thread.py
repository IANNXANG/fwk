import json
import time
import os
import threading
import queue
from openai import OpenAI
from tqdm import tqdm
import argparse

# 设置OpenAI客户端连接到本地vLLM服务
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"  # 注意端口是8001

# 读取math7500.jsonl文件
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 保存结果到新的jsonl文件
def save_results(results, output_file, mode='w'):
    with open(output_file, mode, encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 调用模型处理问题
def process_problem(client, problem, retry=3):
    prompt = f"{problem}"
    
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model="8001vllm",  # 使用脚本中设置的模型名称
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=1.0,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},  # 开启思考模式
                },
            )
            return response.choices[0].message.content, None
        except Exception as e:
            if attempt < retry - 1:
                print(f"\n尝试 {attempt+1}/{retry} 失败: {str(e)[:100]}... 等待5秒后重试")
                time.sleep(5)  # 出错后等待5秒再重试
            else:
                print(f"\n所有 {retry} 次尝试均失败! 错误: {str(e)[:100]}...")
                return None, str(e)
    
    return None, "模型调用失败"

# 工作线程函数
def worker(task_queue, result_queue, client, stop_event, max_retries=3):
    """持续从任务队列获取任务并处理"""
    while not stop_event.is_set():
        try:
            # 从队列获取任务，如果队列为空，等待1秒后重试
            try:
                idx, item, repeat = task_queue.get(timeout=1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue
            
            # 提取问题
            problem = item.get('problem', '')
            problem_id = item.get('id', idx)
            
            # 调用模型处理问题
            answer, error = process_problem(client, problem, max_retries)
            
            # 保存结果
            result = {
                "id": problem_id,
                "problem": problem,
                "answer": answer if answer else "模型调用失败",
                "error": error,
                "round": repeat
            }
            
            # 将结果放入结果队列
            result_queue.put((idx, result))
            
            # 标记任务完成
            task_queue.task_done()
            
        except Exception as e:
            print(f"\n工作线程错误: {e}")
            # 报告错误但不影响其他任务处理
            result = {
                "id": idx if 'problem_id' not in locals() else problem_id,
                "problem": item.get('problem', '') if 'item' in locals() else "未知问题",
                "answer": "处理过程出现错误",
                "error": str(e),
                "round": repeat if 'repeat' in locals() else 0
            }
            try:
                result_queue.put((idx if 'idx' in locals() else -1, result))
            except:
                # 如果无法放入队列，只能跳过
                pass
            time.sleep(1)

def main(input_file, output_dir, n_repeats=1, batch_size=256, max_retries=3):
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    problems = load_jsonl(input_file)
    print(f"加载了 {len(problems)} 个问题")
    
    # 对每次重复进行处理
    for repeat in range(1, n_repeats+1):
        print(f"开始第 {repeat}/{n_repeats} 轮处理")
        
        # 创建任务队列和结果队列
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # 创建停止事件
        stop_event = threading.Event()
        
        # 添加任务到队列
        for idx, item in enumerate(problems):
            task_queue.put((idx, item, repeat))
        
        # 创建并启动工作线程池
        thread_count = min(batch_size, len(problems))
        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(
                target=worker,
                args=(task_queue, result_queue, client, stop_event, max_retries),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # 初始化进度条
        pbar = tqdm(total=len(problems), desc=f"第{repeat}轮处理")
        
        # 初始化结果存储
        results = []
        batch_results = []  # 用于批量保存的结果缓存
        batch_save_size = 10  # 每10条数据保存一次
        completed = 0
        success_count = 0
        failure_count = 0
        
        try:
            while completed < len(problems):
                try:
                    # 从结果队列获取结果
                    idx, result = result_queue.get(timeout=1)
                    
                    # 检查是否成功
                    if result.get("error") is None:
                        success_count += 1
                    else:
                        failure_count += 1
                    
                    # 保存结果
                    results.append(result)
                    batch_results.append(result)
                    
                    # 每达到batch_save_size条数据进行一次批量写入
                    if len(batch_results) >= batch_save_size:
                        save_results(batch_results, f"{output_dir}/math7500_results_round{repeat}_temp.jsonl", mode='a')
                        batch_results = []  # 清空缓存
                    
                    # 更新进度条
                    pbar.update(1)
                    completed += 1
                    
                    # 标记结果已处理
                    result_queue.task_done()
                
                except queue.Empty:
                    # 队列暂时为空，检查是否所有任务都已完成
                    if task_queue.empty() and completed >= len(problems):
                        break
                    continue
            
        except KeyboardInterrupt:
            print("\n接收到中断信号，正在保存已完成的结果...")
        
        finally:
            # 关闭进度条
            pbar.close()
            
            # 设置停止事件，通知所有线程停止
            stop_event.set()
            
            # 等待线程结束
            time.sleep(2)
            
            # 将剩余的批量结果写入文件
            if batch_results:
                save_results(batch_results, f"{output_dir}/math7500_results_round{repeat}_temp.jsonl", mode='a')
            
            # 保存该轮的最终结果
            output_file = f"{output_dir}/math7500_results_round{repeat}.jsonl"
            
            # 按原始索引顺序排序结果
            sorted_results = sorted(results, key=lambda x: x["id"] if isinstance(x["id"], int) else int(x["id"]))
            save_results(sorted_results, output_file)
            
            # 打印统计信息
            print(f"\n第 {repeat} 轮处理完成:")
            print(f"  总计处理: {completed}/{len(problems)} 个问题")
            print(f"  成功: {success_count} 个问题 ({success_count/completed*100:.1f}%)")
            print(f"  失败: {failure_count} 个问题 ({failure_count/completed*100:.1f}%)")
            print(f"  结果已保存到: {output_file}")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='多线程处理数学问题')
    parser.add_argument('--input', type=str, default="problemdata/1.math7500.jsonl", help='输入文件路径')
    parser.add_argument('--output_dir', type=str, default="math7500_results", help='输出目录')
    parser.add_argument('--repeats', type=int, default=1, help='重复次数')
    parser.add_argument('--batch_size', type=int, default=256, help='线程池大小')
    parser.add_argument('--max_retries', type=int, default=3, help='最大重试次数')
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        output_dir=args.output_dir,
        n_repeats=args.repeats,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )