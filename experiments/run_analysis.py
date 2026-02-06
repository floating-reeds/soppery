"""
Analysis Scripts for Citation Hallucination Research

Experiments:
1. Fabrication vs Misattribution signature comparison
2. Scaling behavior across model sizes
3. Layer-wise score analysis
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_scores(scores_dir: str) -> pd.DataFrame:
    """Load scores from all model files in directory."""
    scores_dir = Path(scores_dir)
    all_dfs = []
    
    for score_file in scores_dir.glob("*_scores.json"):
        with open(score_file, 'r') as f:
            scores = json.load(f)
        df = pd.DataFrame(scores)
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def experiment_fabrication_vs_misattribution(df: pd.DataFrame, output_dir: Path):
    """
    Experiment 1: Compare PFS signatures between fabrication and misattribution.
    
    Hypothesis: Pure fabrication (invented titles) shows higher PFS than 
    misattribution (real papers, wrong claims).
    """
    print("\n" + "="*60)
    print("Experiment 1: Fabrication vs Misattribution Signatures")
    print("="*60)
    
    # Filter to only fabricated and misattributed
    df_fab = df[df['label'] == 'fabricated']
    df_mis = df[df['label'] == 'misattributed']
    
    if len(df_fab) < 5 or len(df_mis) < 5:
        print("Not enough samples for comparison")
        return
    
    # Compare PFS
    pfs_fab = df_fab['pfs_mean'].values
    pfs_mis = df_mis['pfs_mean'].values
    
    t_stat, p_value = stats.ttest_ind(pfs_fab, pfs_mis)
    
    print(f"\nPFS Comparison:")
    print(f"  Fabricated: mean={np.mean(pfs_fab):.4f}, std={np.std(pfs_fab):.4f}, n={len(pfs_fab)}")
    print(f"  Misattributed: mean={np.mean(pfs_mis):.4f}, std={np.std(pfs_mis):.4f}, n={len(pfs_mis)}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {p_value < 0.05}")
    
    # Compare ICS
    ics_fab = df_fab['ics_mean'].values
    ics_mis = df_mis['ics_mean'].values
    
    t_stat_ics, p_value_ics = stats.ttest_ind(ics_fab, ics_mis)
    
    print(f"\nICS Comparison:")
    print(f"  Fabricated: mean={np.mean(ics_fab):.4f}, std={np.std(ics_fab):.4f}")
    print(f"  Misattributed: mean={np.mean(ics_mis):.4f}, std={np.std(ics_mis):.4f}")
    print(f"  t-statistic: {t_stat_ics:.4f}")
    print(f"  p-value: {p_value_ics:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PFS distribution
    axes[0].hist(pfs_fab, alpha=0.5, label='Fabricated', bins=20)
    axes[0].hist(pfs_mis, alpha=0.5, label='Misattributed', bins=20)
    axes[0].set_xlabel('PFS Mean')
    axes[0].set_ylabel('Count')
    axes[0].set_title('PFS Distribution: Fabrication vs Misattribution')
    axes[0].legend()
    
    # ICS distribution
    axes[1].hist(ics_fab, alpha=0.5, label='Fabricated', bins=20)
    axes[1].hist(ics_mis, alpha=0.5, label='Misattributed', bins=20)
    axes[1].set_xlabel('ICS Mean')
    axes[1].set_ylabel('Count')
    axes[1].set_title('ICS Distribution: Fabrication vs Misattribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fabrication_vs_misattribution.png', dpi=150)
    plt.close()
    
    print(f"\nSaved plot to {output_dir / 'fabrication_vs_misattribution.png'}")


def experiment_scaling_behavior(df: pd.DataFrame, output_dir: Path):
    """
    Experiment 2: Compare POS distributions across model sizes.
    
    Hypothesis: Larger models show more orthogonal pathway coordination
    (higher POS for correct citations).
    """
    print("\n" + "="*60)
    print("Experiment 2: Scaling Behavior Analysis")
    print("="*60)
    
    models = df['model_name'].unique()
    print(f"Models in dataset: {models}")
    
    if len(models) < 2:
        print("Need at least 2 models for scaling comparison")
        return
    
    results = []
    for model in models:
        df_model = df[df['model_name'] == model]
        df_real = df_model[df_model['label'] == 'real']
        df_hall = df_model[df_model['label'].isin(['fabricated', 'misattributed'])]
        
        results.append({
            'model': model,
            'pos_real_mean': df_real['pos_mean'].mean() if len(df_real) > 0 else np.nan,
            'pos_hall_mean': df_hall['pos_mean'].mean() if len(df_hall) > 0 else np.nan,
            'pos_gap': (df_real['pos_mean'].mean() - df_hall['pos_mean'].mean()) 
                       if len(df_real) > 0 and len(df_hall) > 0 else np.nan,
            'ics_real_mean': df_real['ics_mean'].mean() if len(df_real) > 0 else np.nan,
            'ics_hall_mean': df_hall['ics_mean'].mean() if len(df_hall) > 0 else np.nan,
            'n_real': len(df_real),
            'n_hall': len(df_hall),
        })
    
    results_df = pd.DataFrame(results)
    print("\nScaling Results:")
    print(results_df.to_string())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = range(len(results))
    width = 0.35
    
    # POS by model
    axes[0].bar([i - width/2 for i in x], results_df['pos_real_mean'], width, label='Real', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], results_df['pos_hall_mean'], width, label='Hallucinated', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('POS Mean')
    axes[0].set_title('Pathway Orthogonality Score by Model')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.split('/')[-1][:15] for m in results_df['model']], rotation=45, ha='right')
    axes[0].legend()
    
    # ICS by model
    axes[1].bar([i - width/2 for i in x], results_df['ics_real_mean'], width, label='Real', alpha=0.8)
    axes[1].bar([i + width/2 for i in x], results_df['ics_hall_mean'], width, label='Hallucinated', alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('ICS Mean')
    axes[1].set_title('Internal Consistency Score by Model')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.split('/')[-1][:15] for m in results_df['model']], rotation=45, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_behavior.png', dpi=150)
    plt.close()
    
    print(f"\nSaved plot to {output_dir / 'scaling_behavior.png'}")


def experiment_layer_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Experiment 3: Analyze where fabrication emerges across layers.
    
    Track scores across layers to find where fabrication "signature" appears.
    """
    print("\n" + "="*60)
    print("Experiment 3: Layer-wise Score Analysis")
    print("="*60)
    
    # Extract per-layer scores
    df_real = df[df['label'] == 'real']
    df_hall = df[df['label'].isin(['fabricated', 'misattributed'])]
    
    score_types = ['ics_scores', 'pos_scores', 'pfs_scores', 'bas_scores']
    available_types = [s for s in score_types if s in df.columns]
    
    if not available_types:
        print("No per-layer scores available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, score_type in enumerate(available_types):
        # Extract layer-wise means
        real_layers = df_real[score_type].apply(
            lambda x: x if isinstance(x, list) else []
        ).tolist()
        hall_layers = df_hall[score_type].apply(
            lambda x: x if isinstance(x, list) else []
        ).tolist()
        
        # Find max layers
        max_layers = max(
            max((len(x) for x in real_layers), default=0),
            max((len(x) for x in hall_layers), default=0)
        )
        
        if max_layers == 0:
            continue
        
        # Compute means per layer
        real_means = []
        hall_means = []
        
        for layer in range(max_layers):
            real_vals = [x[layer] for x in real_layers if len(x) > layer]
            hall_vals = [x[layer] for x in hall_layers if len(x) > layer]
            real_means.append(np.mean(real_vals) if real_vals else np.nan)
            hall_means.append(np.mean(hall_vals) if hall_vals else np.nan)
        
        # Plot
        ax = axes[idx]
        ax.plot(range(max_layers), real_means, 'g-', label='Real', marker='o', markersize=3)
        ax.plot(range(max_layers), hall_means, 'r-', label='Hallucinated', marker='o', markersize=3)
        ax.set_xlabel('Layer')
        ax.set_ylabel(score_type.replace('_scores', '').upper())
        ax.set_title(f'{score_type.replace("_scores", "").upper()} Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_analysis.png', dpi=150)
    plt.close()
    
    print(f"\nSaved plot to {output_dir / 'layer_analysis.png'}")


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive analysis report."""
    
    report = []
    report.append("# Citation Hallucination Analysis Report\n")
    report.append(f"Total citations analyzed: {len(df)}\n")
    
    # Label distribution
    report.append("\n## Label Distribution\n")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        report.append(f"- {label}: {count} ({count/len(df)*100:.1f}%)\n")
    
    # Model distribution
    report.append("\n## Model Distribution\n")
    model_counts = df['model_name'].value_counts()
    for model, count in model_counts.items():
        report.append(f"- {model}: {count}\n")
    
    # Score statistics
    report.append("\n## Score Statistics by Label\n")
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        report.append(f"\n### {label} (n={len(df_label)})\n")
        report.append(f"- ICS mean: {df_label['ics_mean'].mean():.4f} (std={df_label['ics_mean'].std():.4f})\n")
        report.append(f"- POS mean: {df_label['pos_mean'].mean():.4f} (std={df_label['pos_mean'].std():.4f})\n")
        report.append(f"- PFS mean: {df_label['pfs_mean'].mean():.4f} (std={df_label['pfs_mean'].std():.4f})\n")
        report.append(f"- BAS mean: {df_label['bas_mean'].mean():.4f} (std={df_label['bas_mean'].std():.4f})\n")
    
    # Write report
    report_path = output_dir / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report)
    
    print(f"\nSaved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run analysis experiments")
    parser.add_argument("--scores-dir", type=str, required=True,
                       help="Directory containing score JSON files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for analysis results")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_all_scores(args.scores_dir)
    if len(df) == 0:
        print("No scores found!")
        return
    
    print(f"Loaded {len(df)} citation scores")
    
    # Run experiments
    experiment_fabrication_vs_misattribution(df, output_dir)
    experiment_scaling_behavior(df, output_dir)
    experiment_layer_analysis(df, output_dir)
    generate_report(df, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
