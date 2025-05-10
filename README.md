# Qwen3 数学问题分析工具集

本项目是一套用于分析大语言模型（特别是Qwen3）处理数学问题时的token长度分布和响应特征的工具集。通过这些工具，我们可以深入了解大语言模型在解决不同难度数学问题时的行为模式。

## 项目结构

```
├── problemdata/                    # 数学问题数据集
│   ├── 1.math7500.jsonl            # 7500个数学问题数据
│   ├── 2.math4097.jsonl            # 4097个数学问题数据
│   ├── 3.math500seed.jsonl         # 500个种子数学问题
│   ├── 4.math7507gen.jsonl         # 7507个生成的数学问题
│   ├── 5.math3977gen.jsonl         # 3977个生成的数学问题
│   ├── 6.iter0.jsonl               # 迭代0问题集
│   ├── 7.iter1.jsonl               # 迭代1问题集
│   └── 8.iter2.jsonl               # 迭代2问题集
├── response_length_analysis_full/  # 全量数据分析结果目录
├── response_length_analysis/       # 采样数据分析结果目录
├── qwen3_token_analysis_full/      # 全量token分析结果目录
├── qwen3_token_analysis/           # 采样token分析结果目录
├── min_token_problems/             # 最小token问题分析结果
├── visualizations/                 # 可视化结果目录
├── requirements.txt                # 项目依赖
├── qwen3_responses_length.py       # 主要数据处理脚本
├── length_distributions.py         # 长度分布可视化脚本
├── token_analyzer.py               # Token数据分析命令行工具
├── find_min_token_problems.py      # 查找最小token问题脚本
└── token_counter.py                # Token计数工具
```

## 主要脚本功能

### qwen3_responses_length.py

这是项目的核心脚本，用于处理数学问题数据集并调用Qwen3模型生成回答，然后分析这些回答的token长度分布。

主要功能：
- 从JSONL文件加载数学问题数据
- 调用本地部署的Qwen3模型API生成回答
- 使用批处理（batch_size=64）并行处理请求，提高效率
- 计算每个回答的token长度
- 实时保存处理结果，确保每处理一条数据就保存一次
- 生成token长度的统计数据（平均值、中位数、最大值、最小值）
- 绘制token长度分布直方图

使用方法：
```bash
python qwen3_responses_length.py --port 8001 --model "8001vllm" --batch_size 64 --full_data
```

### length_distributions.py

用于可视化分析响应长度分布的脚本。

主要功能：
- 从处理结果JSONL文件加载响应长度数据
- 绘制响应长度分布直方图，确保横坐标固定而纵坐标自适应
- 生成难度分布饼图，展示简单(<150 tokens)、中等(150-500 tokens)和困难(>500 tokens)题目的比例

使用方法：
```bash
python length_distributions.py
```

### token_analyzer.py

一个命令行工具，用于分析token数据并允许用户输入自定义阈值。

主要功能：
- 从处理结果文件加载token长度数据
- 允许用户输入自定义阈值划分难度等级
- 分析并输出不同难度等级问题的数量和比例

使用方法：
```bash
python token_analyzer.py
```

### find_min_token_problems.py

用于找出各个数据集中token数最少的问题。

主要功能：
- 加载JSONL格式的数学问题数据
- 使用Qwen分词器计算问题的token数量
- 找出token数最少的问题并保存到CSV文件
- 生成汇总报告

使用方法：
```bash
python find_min_token_problems.py --top_n 10 --output_dir "min_token_problems"
```

## 数据处理流程

1. 数据加载：从JSONL文件加载数学问题数据
2. 数据处理：调用Qwen3模型处理数学问题，生成回答
3. Token计数：计算问题和回答的token长度
4. 数据存储：将处理结果保存为JSONL格式文件
5. 数据分析：计算统计信息，如平均长度、中位数等
6. 数据可视化：绘制分布直方图和饼图，展示不同难度问题的分布

## 环境要求

项目依赖以下Python库：
- numpy
- matplotlib
- tqdm
- transformers
- tiktoken
- openai
- pandas
- tabulate

可通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. 确保已安装所有依赖：`pip install -r requirements.txt`
2. 在本地部署vLLM服务，提供Qwen3模型API
3. 运行数据处理脚本：`python qwen3_responses_length.py`
4. 运行可视化分析脚本：`python length_distributions.py`
5. 使用token分析工具：`python token_analyzer.py`

## 研究结果

通过本项目，我们可以获得以下研究成果：
- 了解Qwen3模型在处理不同类型数学问题时的token长度分布
- 分析不同难度数学问题的比例分布
- 识别模型处理效率高的简单问题和处理复杂的困难问题
- 为优化大语言模型处理数学问题的能力提供数据支持 