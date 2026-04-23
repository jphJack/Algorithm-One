import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    'HandsData': {
        'data_dir': os.path.join(BASE_DIR, 'HandsData'),
        'print_train_dir': os.path.join(BASE_DIR, 'HandsData', 'print-train'),
        'vein_train_dir': os.path.join(BASE_DIR, 'HandsData', 'vein-train'),
        'print_test_dir': os.path.join(BASE_DIR, 'HandsData', 'print-test'),
        'vein_test_dir': os.path.join(BASE_DIR, 'HandsData', 'vein-test'),
        'num_classes': 290,
        'img_size': (128, 128),
        'in_channels': 3,
    },
    'CASIA': {
        'data_dir': os.path.join(BASE_DIR, 'data2', 'CASIA'),
        'print_train_dir': os.path.join(BASE_DIR, 'data2', 'CASIA', 'print-train'),
        'vein_train_dir': os.path.join(BASE_DIR, 'data2', 'CASIA', 'vein-train'),
        'print_test_dir': os.path.join(BASE_DIR, 'data2', 'CASIA', 'print-test'),
        'vein_test_dir': os.path.join(BASE_DIR, 'data2', 'CASIA', 'vein-test'),
        'num_classes': 200,
        'img_size': (128, 128),
        'in_channels': 3,
    },
    'QH': {
        'data_dir': os.path.join(BASE_DIR, 'data2', 'QH'),
        'print_train_dir': os.path.join(BASE_DIR, 'data2', 'QH', 'print-train'),
        'vein_train_dir': os.path.join(BASE_DIR, 'data2', 'QH', 'vein-train'),
        'print_test_dir': os.path.join(BASE_DIR, 'data2', 'QH', 'print-test'),
        'vein_test_dir': os.path.join(BASE_DIR, 'data2', 'QH', 'vein-test'),
        'num_classes': 500,
        'img_size': (128, 128),
        'in_channels': 3,
    },
    'TJ': {
        'data_dir': os.path.join(BASE_DIR, 'data2', 'TJ'),
        'print_train_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'print-train'),
        'vein_train_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'vein-train'),
        'print_test_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'print-test'),
        'vein_test_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'vein-test'),
        'num_classes': 600,
        'img_size': (128, 128),
        'in_channels': 3,
    },
    'CUMT2': {
        'data_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2'),
        'print_train_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'print_train'),
        'vein_train_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'vein_train'),
        'print_test_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'print_test'),
        'vein_test_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'vein_test'),
        'num_classes': 532,
        'img_size': (128, 128),
        'in_channels': 3,
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
