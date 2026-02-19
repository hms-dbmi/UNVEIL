"""
Task configuration generator for Unified Agentic SLURM array jobs.
Focuses on biased tasks with AUROC > 0.55 for the unified agent.
"""

import pandas as pd
import json
import sys
from pathlib import Path


def parse_biased_tasks_csv(csv_path='./results/biased_tasks_summary.csv',
                           min_auroc=0.55):
    """
    Parse biased tasks CSV to extract tasks.
    
    Args:
        csv_path: Path to biased tasks summary CSV
        min_auroc: Minimum AUROC threshold (not used if AUROC column not available)
    
    Returns:
        list: Tasks meeting criteria
    """
    if not Path(csv_path).exists():
        print(f"Warning: Biased tasks CSV not found at {csv_path}", file=sys.stderr)
        print("Falling back to all tasks from exposure-cancer CSV", file=sys.stderr)
        return parse_tasks_csv_fallback()
    
    df = pd.read_csv(csv_path)
    
    # Check if baseline_auroc column exists
    if 'baseline_auroc' in df.columns:
        # Filter by AUROC threshold
        df_filtered = df[df['baseline_auroc'] > min_auroc].copy()
    else:
        # Use all biased tasks (they're already filtered for significance)
        print(f"Note: baseline_auroc column not found. Using all biased tasks.", file=sys.stderr)
        df_filtered = df.copy()
    
    tasks = []
    for _, row in df_filtered.iterrows():
        tasks.append({
            'attribute': row['sensitive_attr'],
            'cancer': row['cancer_type'],
            'gene': row['mutation'],
            'foundation_model': row['foundation_model'],
            'baseline_auroc': row.get('baseline_auroc', None)
        })
    
    return tasks


def parse_tasks_csv_fallback(csv_path='./results/tasks_evaluated_by_exposure_cancer.csv'):
    """Fallback: Parse tasks CSV to extract all (attribute, cancer, genes) combinations."""
    df = pd.read_csv(csv_path)
    tasks = []
    foundation_models = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    
    for _, row in df.iterrows():
        genes = [g.strip() for g in row['tasks'].split(',')]
        for gene in genes:
            for fm in foundation_models:
                tasks.append({
                    'attribute': row['exposure'],
                    'cancer': row['cancer'],
                    'gene': gene,
                    'foundation_model': fm,
                    'baseline_auroc': None  # Unknown
                })
    return tasks


def generate_all_configs(min_auroc=0.55):
    """
    Generate all experiment configurations for unified agent.
    
    Args:
        min_auroc: Minimum baseline AUROC threshold
    
    Returns:
        list: All configurations (dicts with keys: attribute, cancer, gene, foundation_model)
    """
    # Get biased tasks with AUROC > threshold
    tasks = parse_biased_tasks_csv(min_auroc=min_auroc)
    
    # For unified agent, we only need one configuration per task
    # (no need for multiple percentiles or adaptive/fixed variants)
    all_configs = []
    for task in tasks:
        all_configs.append({
            'attribute': task['attribute'],
            'cancer': task['cancer'],
            'gene': task['gene'],
            'foundation_model': task['foundation_model'],
            'strategy': 'unified',
            'baseline_auroc': task.get('baseline_auroc', None)
        })
    
    return all_configs


def get_config_by_array_id(array_id, min_auroc=0.55):
    """Get configuration for a specific SLURM array task ID."""
    configs = generate_all_configs(min_auroc=min_auroc)
    if array_id < 0 or array_id >= len(configs):
        raise IndexError(f"Array ID {array_id} out of range [0, {len(configs)-1}]")
    return configs[array_id]


def print_summary(min_auroc=0.55):
    """Print summary of all configurations."""
    configs = generate_all_configs(min_auroc=min_auroc)
    print(f"Total configurations: {len(configs)}")
    print(f"Minimum baseline AUROC: {min_auroc}")
    print(f"Strategy: unified (enhanced multi-signal ensemble)")
    
    # Group by attribute
    from collections import defaultdict
    by_attr = defaultdict(int)
    for cfg in configs:
        by_attr[cfg['attribute']] += 1
    
    print("\nConfigurations by attribute:")
    for key, count in sorted(by_attr.items()):
        print(f"  {key}: {count}")
    
    # Group by cancer type
    by_cancer = defaultdict(int)
    for cfg in configs:
        by_cancer[cfg['cancer']] += 1
    
    print("\nConfigurations by cancer type:")
    for key, count in sorted(by_cancer.items()):
        print(f"  {key}: {count}")
    
    # Group by foundation model
    by_fm = defaultdict(int)
    for cfg in configs:
        by_fm[cfg['foundation_model']] += 1
    
    print("\nConfigurations by foundation model:")
    for key, count in sorted(by_fm.items()):
        print(f"  {key}: {count}")
    
    # Group by attribute-cancer
    by_attr_cancer = defaultdict(int)
    for cfg in configs:
        by_attr_cancer[f"{cfg['attribute']}-{cfg['cancer']}"] += 1
    
    print("\nConfigurations by attribute-cancer:")
    for key, count in sorted(by_attr_cancer.items()):
        print(f"  {key}: {count}")
    
    # Show AUROC distribution if available
    aurocs = [cfg['baseline_auroc'] for cfg in configs if cfg['baseline_auroc'] is not None]
    if aurocs:
        import numpy as np
        print(f"\nBaseline AUROC distribution:")
        print(f"  Mean: {np.mean(aurocs):.3f}")
        print(f"  Std: {np.std(aurocs):.3f}")
        print(f"  Min: {np.min(aurocs):.3f}")
        print(f"  Max: {np.max(aurocs):.3f}")
        print(f"  Median: {np.median(aurocs):.3f}")


def export_task_list(output_file='unified_tasks.csv', min_auroc=0.55):
    """Export task list to CSV for reference."""
    configs = generate_all_configs(min_auroc=min_auroc)
    df = pd.DataFrame(configs)
    df.to_csv(output_file, index=False)
    print(f"Task list exported to: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Called with array ID - output JSON config
        try:
            array_id = int(sys.argv[1])
            # Use default min_auroc=0.55 for biased tasks with decent performance
            config = get_config_by_array_id(array_id, min_auroc=0.55)
            # Remove baseline_auroc from output (not needed for bash script)
            config_output = {k: v for k, v in config.items() if k != 'baseline_auroc'}
            print(json.dumps(config_output))
        except (IndexError, ValueError) as e:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
            sys.exit(1)
    else:
        # No arguments - print summary
        print_summary(min_auroc=0.55)
        print("\n" + "="*80)
        print("To export task list: python task_config_generator_unified.py --export")
        print("="*80)

