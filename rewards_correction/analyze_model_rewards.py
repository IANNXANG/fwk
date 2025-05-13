import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cosine
import random

def load_data(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def scale_to_neg1_pos1(reward_vector):
    """Scale rewards from [0,1] to [-1,1] range"""
    return [2 * r - 1 for r in reward_vector]

def calculate_cosine_similarities(data, sample_size=100):
    """Calculate cosine similarities between reward vectors"""
    # Sample data if needed
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        indices = sorted(indices)
        sampled_data = [data[i] for i in indices]
    else:
        sampled_data = data
        
    idx_list = [entry['idx'] for entry in sampled_data]
    rule_self_sim = []
    rule_maj_sim = []
    
    for entry in sampled_data:
        # Get and scale reward vectors
        rule_vec = np.array(scale_to_neg1_pos1(entry['rule_rewards']))
        self_vec = np.array(scale_to_neg1_pos1(entry['self_reward_rewards']))
        maj_vec = np.array(scale_to_neg1_pos1(entry['majority_rewards']))
        
        # Ensure consistent vector lengths
        min_len = min(len(rule_vec), len(self_vec), len(maj_vec))
        rule_vec = rule_vec[:min_len]
        self_vec = self_vec[:min_len]
        maj_vec = maj_vec[:min_len]
        
        # Calculate cosine similarities
        rule_self = 1 - cosine(rule_vec, self_vec)
        rule_maj = 1 - cosine(rule_vec, maj_vec)
        
        rule_self_sim.append(rule_self)
        rule_maj_sim.append(rule_maj)
    
    return idx_list, rule_self_sim, rule_maj_sim

def plot_similarities(idx_list, rule_self_sim, rule_maj_sim, model_name, output_path):
    """Plot cosine similarities distribution"""
    # Calculate average similarities
    avg_rule_self = np.mean(rule_self_sim)
    avg_rule_maj = np.mean(rule_maj_sim)
    
    plt.figure(figsize=(15, 6))
    
    # Create scatter plot
    plt.scatter(idx_list, rule_self_sim, color='red', alpha=0.7, s=50, 
                label='Rule-Self Similarity')
    plt.scatter(idx_list, rule_maj_sim, color='green', alpha=0.7, s=50, 
                label='Rule-Majority Similarity')
    
    # Add average lines
    plt.axhline(y=avg_rule_self, color='red', linestyle='--', alpha=0.8, 
               label=f'Rule-Self Avg: {avg_rule_self:.4f}')
    plt.axhline(y=avg_rule_maj, color='green', linestyle='--', alpha=0.8, 
               label=f'Rule-Majority Avg: {avg_rule_maj:.4f}')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity plot: {output_path}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Input and output paths with correct file paths
    input_files = {
        'LLaMA-32B': 'rewards_correction/llama_32_3b_processed_rewards.jsonl',
        'Qwen-25B': 'rewards_correction/qwen25_7b_processed_rewards.jsonl'
    }
    
    # Process each model's data
    for model_name, input_file in input_files.items():
        print(f"\nProcessing {model_name} data...")
        
        # Load and process data
        data = load_data(input_file)
        idx_list, rule_self_sim, rule_maj_sim = calculate_cosine_similarities(data)
        
        # Create output filename
        output_file = f"rewards_correction/{model_name.lower().replace('-', '_')}_similarities.pdf"
        
        # Plot and save results
        plot_similarities(idx_list, rule_self_sim, rule_maj_sim, 
                        model_name, output_file)

if __name__ == "__main__":
    main()
    print("\nAnalysis completed!") 