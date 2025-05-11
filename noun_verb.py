import benepar
import spacy
import json
import pandas as pd
import tqdm
import plotly.express as px
import plotly.io as pio
import argparse
import os
import re

# 加载SpaCy模型
def load_parser():
    nlp = spacy.load('en_core_web_md')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    return nlp

# 检查文本长度是否超过限制
def check_text_length(text, max_length=480):
    # 简单估算，通常一个单词算作1个token
    words = re.findall(r'\w+', text)
    return len(words) <= max_length

# 查找根动词及其直接宾语
def find_root_verb_and_its_dobj(tree_root):
    # 检查当前节点及其子节点是否满足条件
    if tree_root.pos_ == "VERB":
        for child in tree_root.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                return tree_root.lemma_, child.lemma_
        return tree_root.lemma_, None
    # 如果不满足，检查其子节点
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # 如果没有子节点满足条件，返回None
    return None, None

# 分析字符串中的根动词和直接宾语
def find_root_verb_and_its_dobj_in_string(nlp, s):
    # 首先检查文本长度
    if not check_text_length(s):
        return None, None
        
    try:
        doc = nlp(s)
        first_sent = list(doc.sents)[0]
        return find_root_verb_and_its_dobj(first_sent.root)
    except Exception as e:
        return None, None

# 加载并分析JSONL文件
def analyze_jsonl(file_path, nlp):
    print(f"分析文件: {file_path}")
    
    # 加载JSONL数据
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"共加载 {len(data)} 条数据")
    
    # 提取指令
    instructions = []
    for item in data:
        if "instruction" in item:
            instructions.append(item["instruction"])
        else:
            # 如果没有instruction字段，尝试使用problem或其他可能的字段
            if "problem" in item:
                instructions.append(item["problem"])
            elif "question" in item:
                instructions.append(item["question"])
    
    print(f"共提取 {len(instructions)} 条指令")
    
    # 统计变量
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 分析动词和名词
    raw_phrases = []
    for instruction in tqdm.tqdm(instructions):
        try:
            # 检查长度并跳过过长指令
            if not check_text_length(instruction):
                skipped_count += 1
                continue
                
            verb, noun = find_root_verb_and_its_dobj_in_string(nlp, instruction)
            if verb is not None:
                raw_phrases.append({
                    "verb": verb,
                    "noun": noun,
                    "instruction": instruction
                })
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            error_count += 1
            if error_count < 10:  # 只打印前10个错误，避免日志过长
                print(f"处理时出错: {e}")
                print(f"问题指令摘要: {instruction[:100]}...")
    
    # 打印统计信息
    print(f"\n处理统计:")
    print(f"成功处理: {processed_count} 条")
    print(f"跳过过长或无法解析: {skipped_count} 条")
    print(f"处理错误: {error_count} 条")
    
    return raw_phrases

# 生成旭日图
def generate_sunburst(df, output_dir, export_pdf=True):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用全部数据不进行过滤
    df_filtered = df
    
    # 定义更美观的柔和色彩方案
    custom_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', 
                     '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4',
                     '#fdc086', '#ffff99', '#80b1d3', '#fab0e4', '#fdae61']
    
    # 创建旭日图 - HTML版本
    fig_html = px.sunburst(df_filtered, path=['verb', 'noun'], values='count',
                           color_discrete_sequence=custom_colors)
    fig_html.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        font_family="Arial Unicode MS",  # 支持中文的字体
        font_size=36,  # 增大字体大小(默认值通常是12)
    )
    
    # 更新HTML图表文本样式
    fig_html.update_traces(
        textfont=dict(size=30),  # 数据标签字体大小
        insidetextfont=dict(size=36)  # 内部文本字体大小
    )
    
    # 保存HTML格式
    output_html = os.path.join(output_dir, "verb_noun_sunburst.html")
    fig_html.write_html(output_html)
    print(f"旭日图已保存至: {output_html}")
    
    # 如果需要，创建并保存PDF矢量图(使用更大的字体)
    if export_pdf:
        # 为PDF创建单独的图表实例，使用更大的字体
        fig_pdf = px.sunburst(df_filtered, path=['verb', 'noun'], values='count',
                              color_discrete_sequence=custom_colors)
        fig_pdf.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            font_family="Arial Unicode MS",
            font_size=72,  # PDF版本字体大小更大
        )
        
        # 更新PDF图表文本样式
        fig_pdf.update_traces(
            textfont=dict(size=60),  # PDF数据标签字体大小
            insidetextfont=dict(size=72)  # PDF内部文本字体大小
        )
        
        output_pdf = os.path.join(output_dir, "verb_noun_sunburst.pdf")
        # 增大图像尺寸以容纳更大的字体
        fig_pdf.write_image(output_pdf, format="pdf", width=1600, height=1600, scale=3)
        print(f"PDF矢量图已保存至: {output_pdf}")
    
    # 显示HTML图表
    fig_html.show()

# 保存分析结果到文件
def save_results(phrases_df, df, top_verbs, output_dir):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有动词-名词对
    all_pairs_csv = os.path.join(output_dir, "all_verb_noun_pairs.csv")
    phrases_df.to_csv(all_pairs_csv, index=False)
    print(f"所有动词-名词对已保存至: {all_pairs_csv}")
    
    # 保存最常见动词
    top_verbs_csv = os.path.join(output_dir, "top_verbs.csv")
    top_verbs.to_csv(top_verbs_csv, index=False)
    print(f"最常见动词已保存至: {top_verbs_csv}")
    
    # 保存常见动词-名词组合
    top_pairs_csv = os.path.join(output_dir, "top_verb_noun_pairs.csv")
    df.to_csv(top_pairs_csv, index=False)
    print(f"常见动词-名词组合已保存至: {top_pairs_csv}")
    
    # 保存为JSON格式(方便后续可视化)
    result_json = os.path.join(output_dir, "verb_noun_analysis.json")
    
    result_data = {
        "top_verbs": top_verbs.to_dict(orient="records"),
        "top_pairs": df.to_dict(orient="records")
    }
    
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"分析结果JSON已保存至: {result_json}")

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="分析JSONL文件中的动词和名词结构")
    parser.add_argument("--input", type=str, default="problemdata/6.iter0.jsonl", 
                        help="输入的JSONL文件路径")
    parser.add_argument("--output_dir", type=str, default="none_verb", 
                        help="输出图表的目录")
    parser.add_argument("--top_verbs", type=int, default=20, 
                        help="提取的顶级动词数量")
    parser.add_argument("--top_nouns", type=int, default=4, 
                        help="每个动词对应的顶级名词数量")
    parser.add_argument("--export_pdf", action="store_true", default=True,
                        help="是否导出PDF矢量图")
    parser.add_argument("--only_visualize", action="store_true",
                        help="只进行可视化，从保存的JSON读取分析结果")
    args = parser.parse_args()
    
    if args.only_visualize:
        # 仅可视化模式，从保存的结果加载数据
        try:
            result_json = os.path.join(args.output_dir, "verb_noun_analysis.json")
            with open(result_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从JSON恢复DataFrame
            top_pairs_df = pd.DataFrame(data["top_pairs"])
            
            print("从保存的结果加载数据成功，直接进行可视化...")
            generate_sunburst(top_pairs_df, args.output_dir, args.export_pdf)
            return
        except Exception as e:
            print(f"加载保存的结果失败: {e}")
            print("将进行完整分析...")
    
    # 加载解析器
    nlp = load_parser()
    
    # 分析JSONL文件
    raw_phrases = analyze_jsonl(args.input, nlp)
    
    # 转换为DataFrame并处理数据
    phrases_df = pd.DataFrame(raw_phrases)
    phrases_df = phrases_df.dropna()  # 删除含有NaN的行
    
    # 找出最常见的动词
    top_verbs = phrases_df[["verb"]].groupby(["verb"]).size().nlargest(args.top_verbs).reset_index()
    top_verbs.columns = ["verb", "count"]  # 重命名列以便保存
    
    print(f"\n最常见的{args.top_verbs}个动词:")
    print(top_verbs)
    
    # 筛选包含顶级动词的行
    df = phrases_df[phrases_df["verb"].isin(top_verbs["verb"].tolist())]
    
    # 对每个动词，找出最常见的名词
    df = df.groupby(["verb", "noun"]).size().reset_index().rename(columns={0: "count"}).sort_values(by=["count"], ascending=False)
    df = df.groupby("verb").apply(lambda x: x.sort_values("count", ascending=False).head(args.top_nouns)).reset_index(drop=True)
    
    print(f"\n最常见的动词-名词组合:")
    print(df)
    
    # 保存分析结果
    save_results(phrases_df, df, top_verbs, args.output_dir)
    
    # 生成可视化图表
    generate_sunburst(df, args.output_dir, args.export_pdf)

if __name__ == "__main__":
    main() 