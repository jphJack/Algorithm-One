import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.utils import DATASETS

VIS_EXPERIMENTS = {
    'embeddings': {
        'module': 'experiments.vis_embeddings',
        'desc': 'PCA/t-SNE Embedding Visualization',
    },
    'feature_maps': {
        'module': 'experiments.vis_feature_maps',
        'desc': 'Feature Map Visualization',
    },
    'gradcam': {
        'module': 'experiments.vis_gradcam',
        'desc': 'Grad-CAM Heatmap Visualization',
    },
    'expert_weights': {
        'module': 'experiments.vis_expert_weights',
        'desc': 'Expert Weight Distribution Visualization',
    },
    'cross_attention': {
        'module': 'experiments.vis_cross_attention',
        'desc': 'Cross-Attention Map Visualization',
    },
    'training_curves': {
        'module': 'experiments.vis_training_curves',
        'desc': 'Training Curve Visualization',
    },
    'confusion_matrix': {
        'module': 'experiments.vis_confusion_matrix',
        'desc': 'Confusion Matrix Visualization',
    },
    'roc_cmc': {
        'module': 'experiments.vis_roc_cmc',
        'desc': 'ROC/CMC Curve Visualization',
    },
}


def run_experiment(name, dataset, extra_args=None):
    exp = VIS_EXPERIMENTS[name]
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
    parser = argparse.ArgumentParser(description='Run all visualization experiments')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=DATASETS + ['all'],
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=list(VIS_EXPERIMENTS.keys()) + ['all'],
                        help='Experiment name or "all" for all experiments')
    parser.add_argument('--extra-args', type=str, nargs='*', default=[],
                        help='Extra arguments passed to experiment scripts')

    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    experiments = list(VIS_EXPERIMENTS.keys()) if args.experiment == 'all' else [args.experiment]

    results = {}
    for dataset in datasets:
        print(f'\n{"#"*60}')
        print(f'# Dataset: {dataset}')
        print(f'{"#"*60}')
        for exp_name in experiments:
            rc = run_experiment(exp_name, dataset, args.extra_args)
            results[(dataset, exp_name)] = rc

    print(f'\n{"="*60}')
    print('Visualization Experiments Summary')
    print(f'{"="*60}')
    for (dataset, exp_name), rc in results.items():
        status = 'PASS' if rc == 0 else f'FAIL (code {rc})'
        print(f'  [{dataset}] {VIS_EXPERIMENTS[exp_name]["desc"]}: {status}')


if __name__ == '__main__':
    main()
