#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_similarity_data(file_path):
    """Load similarity data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_similarity_distribution(data, output_filename, dataset_name):
    """Plot the distribution of similarities and save as PDF"""
    # Extract max similarities
    similarities = np.array(data['max_similarities'])
    
    # Set figure size with 3:1 aspect ratio
    plt.figure(figsize=(12, 4))
    
    # Set only label font sizes to 24, keep axis ticks at default
    plt.rcParams.update({'font.size': 12})  # 重置为默认字体大小
    
    # Plot histogram with more bins for finer granularity
    plt.hist(similarities, bins=40, range=(0, 1), color='skyblue', 
             edgecolor='black', alpha=0.7)
    
    # Add labels (without title)
    plt.xlabel('ROUGE-L Overlap with the Most Similar Seed Instruction', fontsize=24)
    plt.ylabel('# Instructs', fontsize=24)
    
    # Set axis limits
    plt.xlim(0, 1)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Tight layout to ensure all elements are visible
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {output_filename}")
    
    # Close the figure to free memory
    plt.close()

def main():
    # Define input files and dataset names
    datasets = [
        {
            'input_file': 'math3977gen_rouge_similarity_data.json',
            'output_file': 'math3977gen_rouge_similarity_distribution.pdf',
            'name': 'Math3977gen'
        },
        {
            'input_file': 'math7507gen_rouge_similarity_data.json',
            'output_file': 'math7507gen_rouge_similarity_distribution.pdf',
            'name': 'Math7507gen'
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        print(f"Processing {dataset['name']}...")
        
        # Load data
        data = load_similarity_data(dataset['input_file'])
        
        # Plot and save
        plot_similarity_distribution(
            data, 
            dataset['output_file'],
            dataset['name']
        )
    
    print("All visualizations completed successfully!")

if __name__ == "__main__":
    main() 