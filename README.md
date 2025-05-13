# 数学问题数据集分析项目

本项目针对数学问题数据集进行了多维度分析，包括生成数据的质量评估、令牌长度分析、响应长度分布、语言结构分析和数据集相似度等方面。

## 项目交付任务

本项目完成了以下七项核心分析任务：

### 1. 问题长度分析
- **任务描述:** 分析各数据集问题的令牌长度分布
- **交付文件:** `qwen3_token_analysis_full/` 目录
- **相关代码:** 
  - `token_counter.py` - 计算问题文本的令牌长度
  - `token_analyzer.py` - 分析令牌长度分布并生成统计报告

### 2. 数据质量评估
- **任务描述:** 使用大模型对抽样问题进行质量评分和分析
- **交付文件:** 
  - `evaluation_results/math_evaluation_summary.txt` 和 `.csv` - 原始评分结果
  - `evaluation_results/evaluation_analysis_report.txt` - 详细的评分分析报告
  - `evaluation_results/evaluation_analysis_report.csv` - 评分分析数据表格
- **相关代码:** 
  - `math_problem_evaluator.py` - 使用大模型对数学问题质量进行评分
  - `analyze_evaluation_results.py` - 分析评分结果，包括标准差和跨数据集分析

### 3. 模型响应长度分析
- **任务描述:** 分析大模型对问题的响应长度特征
- **交付文件:** `qwen3_token_analysis_full/qwen3_token_count_summary.txt`
- **相关代码:** 
  - `qwen3_responses_length.py` - 分析大模型对问题的响应长度
  - `response_length_pdf_visualize.py` - 以PDF格式可视化响应长度分布

### 4. 语言结构分析
- **任务描述:** 分析第一轮生成数据的动词-名词结构模式
- **交付文件:** `none_verb/` 目录
- **相关代码:** 
  - `noun_verb.py` - 分析数学问题中的动词-名词结构模式并生成旭日图可视化

### 5. 数据集分布可视化
- **任务描述:** 使用UMAP降维技术可视化数据集分布
- **交付文件:** `umap_visualizations/` 目录
- **相关代码:** 
  - `umap_visualizer_english.py` - 使用UMAP降维算法对问题文本进行二维可视化

### 6. 相似度分析
- **任务描述:** 分析生成问题与种子问题的相似度
- **交付文件:** `math7507gen_rouge_similarity_distribution.pdf` 和 `math3977gen_rouge_similarity_distribution.pdf`
- **相关代码:** 
  - `rouge_similarity_analysis.py` - 计算问题间的ROUGE-L相似度
  - `rouge_similarity_visualize.py` - 生成相似度分布的PDF可视化

### 7. 响应长度分布对比
- **任务描述:** 对比不同数据集的响应长度分布
- **交付文件:** 
  - `response_length_analysis_full/response_length_comparison.pdf` - 五个数据集的响应长度均值对比
  - `math7507gen_response_length_distribution.pdf` 和 `math3977gen_response_length_distribution.pdf`
- **相关代码:** 
  - `compare_response_lengths.py` - 生成五个数据集响应长度对比柱状图
  - `response_length_pdf_visualize.py` - 以3:1比例生成响应长度分布PDF

## 项目内容

项目分析了以下几类数据：
- 三轮生成的数据以及第一轮过滤
- Seed Data (种子数据集)
- MATH 7500 train set (训练集)

## 数据文件说明

项目使用的主要数据文件位于`problemdata/`目录中：

| 文件名 | 描述 | 大小 | 关联分析任务 |
|-------|------|-----|------------|
| `1.math7500.jsonl` | MATH 7500训练集，包含数学问题的原始数据 | 6.8MB | 问题长度分析、UMAP分布可视化 |
| `2.math4097.jsonl` | MATH 4097数据集，补充数学问题集 | 3.8MB | 辅助数据分析 |
| `3.math500seed.jsonl` | 500条种子数据，用于生成新问题的参考模板 | 153KB | 相似度分析、语言结构分析 |
| `4.math7507gen.jsonl` | 第一轮生成的7507条数学问题 | 2.0MB | 相似度分析、响应长度分布分析 |
| `5.math3977gen.jsonl` | 第一轮过滤后的3977条问题 | 1.0MB | 相似度分析、响应长度分布分析 |
| `6.iter0.jsonl` | 同`4.math7507gen.jsonl`，第一轮生成数据 | 2.0MB | UMAP分布可视化、语言结构分析 |
| `7.iter1.jsonl` | 第二轮生成的数学问题 | 2.0MB | 问题质量评估、响应长度分析 |
| `8.iter2.jsonl` | 第三轮生成的数学问题 | 2.0MB | 问题质量评估、响应长度分析 |

## 文件目录结构

### 主要目录

- `evaluation_results/` - 数据质量评估结果
- `min_token_problems/` - 最小令牌长度问题收集
- `none_verb/` - 动词名词结构分析结果
- `problemdata/` - 原始问题数据集
- `qwen3_token_analysis/` - 令牌长度分析
- `qwen3_token_analysis_full/` - 完整令牌长度分析
- `response_length_analysis/` - 响应长度分析
- `response_length_analysis_full/` - 完整响应长度分析
- `reward_visualizations/` - 奖励函数可视化
- `umap_visualizations/` - UMAP降维可视化
- `visualizations/` - 其他数据可视化

### 辅助工具文件

- `extract_rewards.py` - 从数据中提取奖励值
- `find_min_token_problems.py` - 寻找最小令牌长度的问题
- `process_jsonl.py` - 处理JSONL格式数据文件
- `process_rewards.py` - 处理和分析奖励数据
- `verify_rewards.py` - 验证奖励函数的有效性
- `visualize_rewards.py` - 可视化奖励分布情况