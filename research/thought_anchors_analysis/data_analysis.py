#!/usr/bin/env python3
"""
Thought Anchors Comparative Analysis: Qwen3 vs DeepSeek-R1
=========================================================

This script analyzes thought anchors datasets from both models to understand
how reasoning has evolved from DeepSeek-R1 to Qwen3.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ThoughtAnchorsAnalyzer:
    """Analyzer for thought anchors datasets comparison."""
    
    def __init__(self):
        self.qwen3_data = None
        self.deepseek_data = None
        self.analysis_results = {}
        
    def load_datasets(self):
        """Load both datasets from HuggingFace."""
        print("Loading Qwen3 thought anchors dataset...")
        qwen3_dataset = load_dataset("codelion/Qwen3-0.6B-pts-thought-anchors")
        self.qwen3_data = pd.DataFrame(qwen3_dataset['train'])
        
        print("Loading DeepSeek-R1 thought anchors dataset...")
        deepseek_dataset = load_dataset("codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors")
        self.deepseek_data = pd.DataFrame(deepseek_dataset['train'])
        
        print(f"Qwen3 dataset: {len(self.qwen3_data)} thought anchors")
        print(f"DeepSeek-R1 dataset: {len(self.deepseek_data)} thought anchors")
        
    def basic_statistics(self):
        """Generate basic statistics for both datasets."""
        print("\n=== BASIC STATISTICS ===")
        
        stats = {
            'qwen3': {
                'total_anchors': len(self.qwen3_data),
                'avg_prob_delta': self.qwen3_data['prob_delta'].mean(),
                'std_prob_delta': self.qwen3_data['prob_delta'].std(),
                'positive_ratio': (self.qwen3_data['prob_delta'] > 0).mean(),
                'strong_positive_ratio': (self.qwen3_data['prob_delta'] > 0.5).mean(),
                'strong_negative_ratio': (self.qwen3_data['prob_delta'] < -0.3).mean(),
                'avg_sentence_length': self.qwen3_data['sentence'].str.len().mean(),
                'unique_queries': self.qwen3_data['query'].nunique(),
            },
            'deepseek': {
                'total_anchors': len(self.deepseek_data),
                'avg_prob_delta': self.deepseek_data['prob_delta'].mean(),
                'std_prob_delta': self.deepseek_data['prob_delta'].std(),
                'positive_ratio': (self.deepseek_data['prob_delta'] > 0).mean(),
                'strong_positive_ratio': (self.deepseek_data['prob_delta'] > 0.5).mean(),
                'strong_negative_ratio': (self.deepseek_data['prob_delta'] < -0.3).mean(),
                'avg_sentence_length': self.deepseek_data['sentence'].str.len().mean(),
                'unique_queries': self.deepseek_data['query'].nunique(),
            }
        }
        
        self.analysis_results['basic_stats'] = stats
        
        # Print comparison
        print(f"Qwen3 vs DeepSeek-R1 Comparison:")
        print(f"  Total anchors: {stats['qwen3']['total_anchors']} vs {stats['deepseek']['total_anchors']}")
        print(f"  Avg prob_delta: {stats['qwen3']['avg_prob_delta']:.3f} vs {stats['deepseek']['avg_prob_delta']:.3f}")
        print(f"  Positive ratio: {stats['qwen3']['positive_ratio']:.3f} vs {stats['deepseek']['positive_ratio']:.3f}")
        print(f"  Strong positive ratio: {stats['qwen3']['strong_positive_ratio']:.3f} vs {stats['deepseek']['strong_negative_ratio']:.3f}")
        print(f"  Strong negative ratio: {stats['qwen3']['strong_negative_ratio']:.3f} vs {stats['deepseek']['strong_negative_ratio']:.3f}")
        
        return stats
        
    def failure_mode_analysis(self):
        """Analyze failure modes across both models."""
        print("\n=== FAILURE MODE ANALYSIS ===")
        
        # Count failure modes
        qwen3_failures = self.qwen3_data['failure_mode'].value_counts()
        deepseek_failures = self.deepseek_data['failure_mode'].value_counts()
        
        print("Qwen3 failure modes:")
        print(qwen3_failures)
        
        print("\nDeepSeek-R1 failure modes:")
        print(deepseek_failures)
        
        # Store results
        self.analysis_results['failure_modes'] = {
            'qwen3': qwen3_failures.to_dict(),
            'deepseek': deepseek_failures.to_dict()
        }
        
        return qwen3_failures, deepseek_failures
        
    def reasoning_quality_analysis(self):
        """Analyze reasoning quality patterns."""
        print("\n=== REASONING QUALITY ANALYSIS ===")
        
        # Probability delta distributions
        qwen3_deltas = self.qwen3_data['prob_delta']
        deepseek_deltas = self.deepseek_data['prob_delta']
        
        print(f"Qwen3 prob_delta distribution:")
        print(f"  Mean: {qwen3_deltas.mean():.3f}")
        print(f"  Median: {qwen3_deltas.median():.3f}")
        print(f"  25th percentile: {qwen3_deltas.quantile(0.25):.3f}")
        print(f"  75th percentile: {qwen3_deltas.quantile(0.75):.3f}")
        
        print(f"\nDeepSeek-R1 prob_delta distribution:")
        print(f"  Mean: {deepseek_deltas.mean():.3f}")
        print(f"  Median: {deepseek_deltas.median():.3f}")
        print(f"  25th percentile: {deepseek_deltas.quantile(0.25):.3f}")
        print(f"  75th percentile: {deepseek_deltas.quantile(0.75):.3f}")
        
        # Store results
        self.analysis_results['reasoning_quality'] = {
            'qwen3': {
                'mean': qwen3_deltas.mean(),
                'median': qwen3_deltas.median(),
                'q25': qwen3_deltas.quantile(0.25),
                'q75': qwen3_deltas.quantile(0.75),
                'distribution': qwen3_deltas.tolist()
            },
            'deepseek': {
                'mean': deepseek_deltas.mean(),
                'median': deepseek_deltas.median(),
                'q25': deepseek_deltas.quantile(0.25),
                'q75': deepseek_deltas.quantile(0.75),
                'distribution': deepseek_deltas.tolist()
            }
        }
        
        return qwen3_deltas, deepseek_deltas
        
    def embedding_analysis(self):
        """Analyze sentence embeddings for clustering insights."""
        print("\n=== EMBEDDING ANALYSIS ===")
        
        # Extract embeddings
        qwen3_embeddings = np.array([emb for emb in self.qwen3_data['sentence_embedding'] if len(emb) == 384])
        deepseek_embeddings = np.array([emb for emb in self.deepseek_data['sentence_embedding'] if len(emb) == 384])
        
        print(f"Qwen3 embeddings shape: {qwen3_embeddings.shape}")
        print(f"DeepSeek-R1 embeddings shape: {deepseek_embeddings.shape}")
        
        # Combine embeddings for joint analysis
        all_embeddings = np.vstack([qwen3_embeddings, deepseek_embeddings])
        model_labels = ['Qwen3'] * len(qwen3_embeddings) + ['DeepSeek-R1'] * len(deepseek_embeddings)
        
        # UMAP dimensionality reduction
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(all_embeddings)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(all_embeddings)
        
        # Store results
        self.analysis_results['embeddings'] = {
            'umap_2d': embedding_2d,
            'model_labels': model_labels,
            'clusters': clusters,
            'qwen3_embeddings': qwen3_embeddings,
            'deepseek_embeddings': deepseek_embeddings
        }
        
        print(f"UMAP embedding completed. 2D shape: {embedding_2d.shape}")
        print(f"Clustering completed. Found {len(np.unique(clusters))} clusters")
        
        return embedding_2d, model_labels, clusters
        
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        # Create figure directory
        import os
        os.makedirs('images', exist_ok=True)
        
        # 1. Probability delta distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram comparison
        qwen3_deltas = self.analysis_results['reasoning_quality']['qwen3']['distribution']
        deepseek_deltas = self.analysis_results['reasoning_quality']['deepseek']['distribution']
        
        ax1.hist(qwen3_deltas, bins=50, alpha=0.7, label='Qwen3', color='blue')
        ax1.hist(deepseek_deltas, bins=50, alpha=0.7, label='DeepSeek-R1', color='red')
        ax1.set_xlabel('Probability Delta')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Probability Delta Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        data_for_box = [qwen3_deltas, deepseek_deltas]
        ax2.boxplot(data_for_box, labels=['Qwen3', 'DeepSeek-R1'])
        ax2.set_ylabel('Probability Delta')
        ax2.set_title('Probability Delta Distribution Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/probability_delta_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Failure mode comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        qwen3_failures = self.analysis_results['failure_modes']['qwen3']
        deepseek_failures = self.analysis_results['failure_modes']['deepseek']
        
        # Get all failure modes
        all_modes = set(qwen3_failures.keys()) | set(deepseek_failures.keys())
        all_modes = [mode for mode in all_modes if mode is not None and mode != 'None']
        
        qwen3_counts = [qwen3_failures.get(mode, 0) for mode in all_modes]
        deepseek_counts = [deepseek_failures.get(mode, 0) for mode in all_modes]
        
        x = np.arange(len(all_modes))
        width = 0.35
        
        ax.bar(x - width/2, qwen3_counts, width, label='Qwen3', color='blue', alpha=0.7)
        ax.bar(x + width/2, deepseek_counts, width, label='DeepSeek-R1', color='red', alpha=0.7)
        
        ax.set_xlabel('Failure Mode')
        ax.set_ylabel('Count')
        ax.set_title('Failure Mode Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(all_modes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/failure_mode_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Embedding clusters visualization
        if 'embeddings' in self.analysis_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            embedding_2d = self.analysis_results['embeddings']['umap_2d']
            model_labels = self.analysis_results['embeddings']['model_labels']
            
            # Create scatter plot
            qwen3_mask = np.array(model_labels) == 'Qwen3'
            deepseek_mask = np.array(model_labels) == 'DeepSeek-R1'
            
            ax.scatter(embedding_2d[qwen3_mask, 0], embedding_2d[qwen3_mask, 1], 
                      c='blue', alpha=0.6, label='Qwen3', s=20)
            ax.scatter(embedding_2d[deepseek_mask, 0], embedding_2d[deepseek_mask, 1], 
                      c='red', alpha=0.6, label='DeepSeek-R1', s=20)
            
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.set_title('Reasoning Pattern Clusters (UMAP Visualization)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('images/embedding_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Reasoning quality evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create comparison metrics
        metrics = ['Avg Prob Delta', 'Positive Ratio', 'Strong Positive Ratio', 'Strong Negative Ratio']
        qwen3_values = [
            self.analysis_results['basic_stats']['qwen3']['avg_prob_delta'],
            self.analysis_results['basic_stats']['qwen3']['positive_ratio'],
            self.analysis_results['basic_stats']['qwen3']['strong_positive_ratio'],
            self.analysis_results['basic_stats']['qwen3']['strong_negative_ratio']
        ]
        deepseek_values = [
            self.analysis_results['basic_stats']['deepseek']['avg_prob_delta'],
            self.analysis_results['basic_stats']['deepseek']['positive_ratio'],
            self.analysis_results['basic_stats']['deepseek']['strong_positive_ratio'],
            self.analysis_results['basic_stats']['deepseek']['strong_negative_ratio']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, qwen3_values, width, label='Qwen3', color='blue', alpha=0.7)
        ax.bar(x + width/2, deepseek_values, width, label='DeepSeek-R1', color='red', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Reasoning Quality Evolution: DeepSeek-R1 to Qwen3')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/reasoning_quality_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to images/ directory")
        
    def save_analysis_results(self):
        """Save analysis results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        results_copy = self.analysis_results.copy()
        if 'embeddings' in results_copy:
            results_copy['embeddings']['umap_2d'] = results_copy['embeddings']['umap_2d'].tolist()
            results_copy['embeddings']['clusters'] = results_copy['embeddings']['clusters'].tolist()
            results_copy['embeddings']['qwen3_embeddings'] = results_copy['embeddings']['qwen3_embeddings'].tolist()
            results_copy['embeddings']['deepseek_embeddings'] = results_copy['embeddings']['deepseek_embeddings'].tolist()
        
        with open('data/analysis_results.json', 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print("Analysis results saved to data/analysis_results.json")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive thought anchors analysis...")
        
        # Load data
        self.load_datasets()
        
        # Run analyses
        self.basic_statistics()
        self.failure_mode_analysis()
        self.reasoning_quality_analysis()
        self.embedding_analysis()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_analysis_results()
        
        print("\n✅ Analysis complete! Check the images/ directory for visualizations.")

if __name__ == "__main__":
    analyzer = ThoughtAnchorsAnalyzer()
    analyzer.run_full_analysis()