import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    'HandsData': {
        'data_dir': os.path.join(BASE_DIR, 'HandsData'),
        'num_classes': 290,
        'print_size': (217, 190),
        'vein_size': (180, 180),
    },
    'CASIA': {
        'data_dir': os.path.join(BASE_DIR, 'data2', 'CASIA'),
        'num_classes': 200,
        'print_size': (217, 190),
        'vein_size': (180, 180),
    },
    'QH': {
        'data_dir': os.path.join(BASE_DIR, 'data2', 'QH'),
        'num_classes': 500,
        'print_size': (217, 190),
        'vein_size': (180, 180),
    },
    'TJ': {
        'data_dir': os.path.join(BASE_DIR, 'data2', 'TJ'),
        'num_classes': 600,
        'print_size': (217, 190),
        'vein_size': (180, 180),
    },
}

DEFAULT_DATASET = 'HandsData'

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4

FEATURE_DIM = 256
NUM_EXPERTS = 3


def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]


def get_save_dir(dataset_name):
    return os.path.join(BASE_DIR, 'checkpoints', dataset_name)
