import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.utils import DATASETS

ABLATION_EXPERIMENTS = {
    'components': {
        'module': 'experiments.ablation_components',
        'desc': 'Component Ablation (w/o MoE Enhancement, w/o MoE Fusion, w/o ArcFace, w/o LB Loss)',
    },
    'fusion_strategy': {
        'module': 'experiments.ablation_fusion_strategy',
        'desc': 'Fusion Strategy Ablation (Add, Concat, WeightedAvg)',
    },
    'num_experts': {
        'module': 'experiments.ablation_num_experts',
        'desc': 'Expert Number Ablation (1-5 experts)',
    },
    'single_expert': {
        'module': 'experiments.ablation_single_expert',
        'desc': 'Single Expert Ablation (remove one expert at a time)',
    },
    'loss': {
        'module': 'experiments.ablation_loss',
        'desc': 'Loss Function Ablation (label smoothing, LB loss, ArcFace margin)',
    },
    'feature_dim': {
        'module': 'experiments.ablation_feature_dim',
        'desc': 'Feature Dimension Ablation (128, 256, 512)',
    },
    'stages': {
        'module': 'experiments.ablation_stages',
        'desc': 'Stage Selection Ablation ([3,4], [4,5], [3,4,5])',
    },
    'dual_stream': {
        'module': 'experiments.ablation_dual_stream',
        'desc': 'Dual-Stream vs Single-Stream Ablation',
    },
}


def run_experiment(name, dataset, extra_args=None):
    exp = ABLATION_EXPERIMENTS[name]
    cmd = [sys.executable, '-m', exp['module'], '--dataset', dataset]
    if extra_args:
        cmd.extend(extra_args)

    print(f'\n{"="*60}')
    print(f'Running: {exp["desc"]}')
    print(f'Dataset: {dataset}')
    print(f'Command: {" ".join(cmd)}')
    print(f'{"="*60}')

    result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))
    if result.returncode != 0:
        print(f'WARNING: {name} exited with code {result.returncode}')
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run all ablation experiments')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=DATASETS + ['all'],
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=list(ABLATION_EXPERIMENTS.keys()) + ['all'],
                        help='Experiment name or "all" for all experiments')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides default)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides default)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides default)')

    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    experiments = list(ABLATION_EXPERIMENTS.keys()) if args.experiment == 'all' else [args.experiment]

    extra_args = []
    if args.epochs is not None:
        extra_args.extend(['--epochs', str(args.epochs)])
    if args.batch_size is not None:
        extra_args.extend(['--batch-size', str(args.batch_size)])
    if args.lr is not None:
        extra_args.extend(['--lr', str(args.lr)])

    results = {}
    for dataset in datasets:
        print(f'\n{"#"*60}')
        print(f'# Dataset: {dataset}')
        print(f'{"#"*60}')
        for exp_name in experiments:
            rc = run_experiment(exp_name, dataset, extra_args if extra_args else None)
            results[(dataset, exp_name)] = rc

    print(f'\n{"="*60}')
    print('Ablation Experiments Summary')
    print(f'{"="*60}')
    for (dataset, exp_name), rc in results.items():
        status = 'PASS' if rc == 0 else f'FAIL (code {rc})'
        print(f'  [{dataset}] {ABLATION_EXPERIMENTS[exp_name]["desc"]}: {status}')

    all_csvs = []
    for dataset in datasets:
        ablation_dir = os.path.join(os.path.dirname(__file__), '..', 'experiment_results', dataset, 'ablation')
        if os.path.exists(ablation_dir):
            for f in sorted(os.listdir(ablation_dir)):
                if f.endswith('.csv'):
                    all_csvs.append(os.path.join(ablation_dir, f))

    if all_csvs:
        print(f'\nGenerated CSV files:')
        for csv_path in all_csvs:
            print(f'  {csv_path}')


if __name__ == '__main__':
    main()
