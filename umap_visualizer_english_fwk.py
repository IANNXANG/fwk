#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from collections import Counter
import re
import jieba
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import umap
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize

plt.rc('font',family='times new roman')
# Create output directory
output_dir = "umap_visualizations"
os.makedirs(output_dir, exist_ok=True)

def load_data_from_jsonl(file_path, sample_size=float('inf')):
    """Load data from JSONL file and sample if needed"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Ensure sample size doesn't exceed dataset size
    if sample_size == float('inf'):
        sampled_data = data
    else:
        sample_size = min(sample_size, len(data))
        sampled_data = random.sample(data, sample_size)
    
    # Extract problem text, prioritize 'problem' field, use 'instruction' if not available
    problems = []
    for item in sampled_data:
        if 'problem' in item and item['problem']:
            problems.append(item['problem'])
        elif 'instruction' in item and item['instruction']:
            problems.append(item['instruction'])
        else:
            problems.append('')  # Add empty string if neither field exists
    
    return problems

def extract_features_tfidf(problems):
    """Extract features using TF-IDF"""
    # For English text, we don't need jieba segmentation
    # Use TF-IDF to extract features
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(problems)
    return features.toarray()

def extract_features_bm25(problems):
    """Extract features using BM25"""
    # Tokenize documents
    tokenized_problems = [problem.split() for problem in problems]
    
    # Create BM25 object
    bm25 = BM25Okapi(tokenized_problems)
    
    # Calculate BM25 scores for each document against all others
    features = []
    for i, problem in enumerate(tqdm(tokenized_problems, desc="Extracting BM25 features")):
        # Use unique terms from the problem as queries
        unique_terms = list(set(problem))
        
        if not unique_terms:  # Handle empty problems
            features.append(np.zeros(len(tokenized_problems)))
            continue
            
        # Get scores for all documents against this problem's terms
        doc_scores = []
        for term in unique_terms:
            scores = bm25.get_scores([term])
            doc_scores.append(scores)
        
        # Average the scores for each term
        avg_scores = np.mean(doc_scores, axis=0)
        features.append(avg_scores)
    
    # Convert to numpy array and normalize
    features_array = np.array(features)
    normalized_features = normalize(features_array)
    
    return normalized_features

def extract_features_bert(problems, device='cpu'):
    """Extract features using BERT"""
    try:
        # Load pre-trained BERT model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.to(device)
        model.eval()
        
        features = []
        
        # Use tqdm to show progress
        for problem in tqdm(problems, desc="Extracting BERT features"):
            inputs = tokenizer(problem, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use output of [CLS] token as sentence representation
                features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten())
        
        return np.array(features)
    except Exception as e:
        print(f"BERT feature extraction failed: {e}")
        return None

def apply_umap(features, n_neighbors=15, min_dist=0.1, n_components=2):
    """Apply UMAP for dimensionality reduction"""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    return reducer.fit_transform(features)

def plot_umap_results(umap_results, labels, colors, title, output_file):
    """Plot UMAP results and save visualization as PDF"""
    plt.figure(figsize=(12, 10))
    plt.rc('font',family='times new roman')
    # Convert results to DataFrame for seaborn plotting
    df = pd.DataFrame({
        'x': umap_results[:, 0],
        'y': umap_results[:, 1],
        'category': labels,
        'color': colors
    })
    
    # Create scatter plot with seaborn
    sns.scatterplot(data=df, x='x', y='y', hue='category', palette='tab10', s=120, alpha=0.7)
    
    # plt.title(title, fontsize=16)
    plt.xlabel('Dimension 1', fontsize=40)
    plt.ylabel('Dimension 2', fontsize=40)
    plt.xticks(fontsize=31)
    plt.yticks(fontsize=31)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=r'MATH train', markerfacecolor='#009b9e', markersize=20),
        Line2D([0], [0], marker='o', color='w', label=r'$\mathcal{D}_{\text{seed}}$', markerfacecolor='#f8ad64', markersize=20),
        Line2D([0], [0], marker='o', color='w', label=r'$\mathcal{D}_{\text{gen}}^1$', markerfacecolor='#4472c4', markersize=20),
        # Line2D([0], [0], marker='o', color='w', label=r'$D_{\text{seed}}$', markerfacecolor='#f8ad64', markersize=20),
        # Line2D([0], [0], marker='o', color='w', label=r'$D_{\text{gen}}^1$', markerfacecolor='#4472c4', markersize=20),
    ]
    plt.legend(handles=legend_elements, fontsize=34, loc='upper left', frameon=True, handlelength=1.5)
    plt.tight_layout()
    
    # Save image as PDF
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_problem_lengths(all_problems, labels, output_file):
    """Analyze and visualize problem length distributions"""
    # Calculate length of each problem
    problem_lengths = [len(problem) for problem in all_problems]
    
    # Create DataFrame
    df = pd.DataFrame({
        'length': problem_lengths,
        'category': labels
    })
    
    # Visualize distributions by category
    plt.figure(figsize=(14, 8))
    
    # Draw boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='category', y='length')
    plt.title('Problem Length Boxplot by Dataset', fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Problem Length (characters)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Draw violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x='category', y='length')
    plt.title('Problem Length Violin Plot by Dataset', fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Problem Length (characters)', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate average lengths by category
    avg_lengths = df.groupby('category')['length'].mean()
    return avg_lengths

def analyze_top_words(all_problems, labels, output_file, top_n=20):
    """Analyze and visualize top frequent words in problem texts"""
    # Create a dictionary to store problems by category
    category_problems = {}
    unique_labels = list(set(labels))
    
    for label, problem in zip(labels, all_problems):
        if label not in category_problems:
            category_problems[label] = []
        category_problems[label].append(problem)
    
    # Calculate high-frequency words for each category
    category_top_words = {}
    
    for label in unique_labels:
        # Combine all problems in this category
        combined_text = " ".join(category_problems[label])
        # For English text, split by whitespace
        words = combined_text.lower().split()
        # Filter out stop words and punctuation
        filtered_words = [word for word in words if len(word) > 1 and not bool(re.match(r'[^\w\s]', word))]
        # Count word frequencies
        word_counts = Counter(filtered_words)
        # Save top_n frequent words
        category_top_words[label] = word_counts.most_common(top_n)
    
    # Visualize top words for each category
    fig, axes = plt.subplots(len(unique_labels), 1, figsize=(12, 4 * len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        ax = axes[i] if len(unique_labels) > 1 else axes
        
        words = [item[0] for item in category_top_words[label]]
        counts = [item[1] for item in category_top_words[label]]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # Labels start from top
        ax.set_title(f'Top {top_n} Frequent Words for {label}')
        ax.set_xlabel('Word Frequency')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_top_words

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # JSONL file paths with custom labels
    jsonl_files = [
        ("problemdata/1.math7500.jsonl", "MATH train", "#009b9e"),
        ("problemdata/3.math500seed.jsonl", r"$D_{\text{seed}}$", "#f8ad64"),
        ("problemdata/6.iter0.jsonl", r"$D^1_{\text{gen}}$", "#4472c4"),
    ]
    
    # Load all data from each file
    all_problems = []
    labels = []
    colors = []
    
    for file_path, label, color in jsonl_files:
        print(f"Loading file: {file_path}")
        try:
            # Load all data without sample size limit
            problems = load_data_from_jsonl(file_path)
            all_problems.extend(problems)
            # Use the custom label for each file
            labels.extend([label for _ in range(len(problems))])
            colors.extend([color for _ in range(len(problems))])
            print(f"  Successfully loaded {len(problems)} records")
        except Exception as e:
            print(f"  Loading failed: {e}")
    
    if not all_problems:
        print("No data loaded, exiting program")
        return
    
    print(f"Loaded a total of {len(all_problems)} problem records")
    
    # 1. TF-IDF features for UMAP analysis
    # print("Extracting TF-IDF features...")
    # tfidf_features = extract_features_tfidf(all_problems)
    
    # print("Applying UMAP dimensionality reduction...")
    # umap_results_tfidf = apply_umap(tfidf_features)
    
    # print("Plotting UMAP visualization of TF-IDF features...")
    # plot_umap_results(
    #     umap_results_tfidf, 
    #     labels, 
    #     "UMAP Visualization of Problem Texts (TF-IDF)", 
    #     os.path.join(output_dir, "tfidf_umap_visualization.pdf")
    # )
    
    # 2. BM25 features for UMAP analysis
    # print("Extracting BM25 features...")
    # bm25_features = extract_features_bm25(all_problems)
    
    # print("Applying UMAP to BM25 features...")
    # umap_results_bm25 = apply_umap(bm25_features, n_neighbors=20, min_dist=0.2)
    
    # print("Plotting UMAP visualization of BM25 features...")
    # plot_umap_results(
    #     umap_results_bm25, 
    #     labels, 
    #     "UMAP Visualization of Problem Texts (BM25)", 
    #     os.path.join(output_dir, "bm25_umap_visualization.pdf")
    # )
    
    # 3. BERT features for UMAP analysis (if available)
    bert_features = extract_features_bert(all_problems)
    if bert_features is not None:
        print("Applying UMAP to BERT features...")
        # Use different parameters for different visualization effects
        umap_results_bert = apply_umap(bert_features, n_neighbors=30, min_dist=0.3)
        
        print("Plotting UMAP visualization of BERT features...")
        plot_umap_results(
            umap_results_bert, 
            labels, 
            colors,
            "UMAP Visualization of Problem Texts (BERT)", 
            os.path.join(output_dir, "bert_umap_visualization.pdf")
        )
    
    # 4. Analyze problem lengths
    # print("Analyzing problem length distributions...")
    # avg_lengths = analyze_problem_lengths(
    #     all_problems, 
    #     labels, 
    #     os.path.join(output_dir, "problem_length_analysis.pdf")
    # )
    # print("Average problem length by category:")
    # for category, avg_length in avg_lengths.items():
    #     print(f"  {category}: {avg_length:.2f} characters")
    
    # 5. Analyze frequent words
    # print("Analyzing frequent words in problems...")
    # category_top_words = analyze_top_words(
    #     all_problems, 
    #     labels, 
    #     os.path.join(output_dir, "top_words_analysis.pdf")
    # )
    
    print("All analyses complete, visualization results saved in 'umap_visualizations' directory")

if __name__ == "__main__":
    main() 