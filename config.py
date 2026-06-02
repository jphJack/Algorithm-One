import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    'HandsData': {
        # 'data_dir': os.path.join(BASE_DIR, 'HandsData'),
        # 'print_train_dir': os.path.join(BASE_DIR, 'HandsData', 'print-train'),
        # 'vein_train_dir': os.path.join(BASE_DIR, 'HandsData', 'vein-train'),
        # 'print_test_dir': os.path.join(BASE_DIR, 'HandsData', 'print-test'),
        # 'vein_test_dir': os.path.join(BASE_DIR, 'HandsData', 'vein-test'),
        'data_dir': r'D:\code\plam-vein\Graduate-code\code-one\HandsData',
        'print_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\HandsData', 'print-train'),
        'vein_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\HandsData', 'vein-train'),
        'print_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\HandsData', 'print-test'),
        'vein_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\HandsData', 'vein-test'),
        'num_classes': 290,
        'img_size': (160, 160),
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
        'data_dir': r'D:\code\plam-vein\Graduate-code\code-one\data2\QH',
        'print_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\QH', 'print-train'),
        'vein_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\QH', 'vein-train'),
        'print_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\QH', 'print-test'),
        'vein_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\QH', 'vein-test'),
        'num_classes': 500,
        'img_size': (128, 128),
        'in_channels': 3,
    },
    'TJ': {
        # 'data_dir': os.path.join(BASE_DIR, 'data2', 'TJ'),
        # 'print_train_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'print-train'),
        # 'vein_train_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'vein-train'),
        # 'print_test_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'print-test'),
        # 'vein_test_dir': os.path.join(BASE_DIR, 'data2', 'TJ', 'vein-test'),
        'data_dir': r'D:\code\plam-vein\Graduate-code\code-one\data2\TJ',
        'print_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\TJ', 'print-train'),
        'vein_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\TJ', 'vein-train'),
        'print_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\TJ', 'print-test'),
        'vein_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\TJ', 'vein-test'),
        'num_classes': 600,
        'img_size': (128, 128),
        'in_channels': 3,
    },
    'CUMT2': {
        # 'data_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2'),
        # 'print_train_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'print_train'),
        # 'vein_train_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'vein_train'),
        # 'print_test_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'print_test'),
        # 'vein_test_dir': os.path.join(BASE_DIR, 'data2', 'CUMT2', 'vein_test'),
        'data_dir': r'D:\code\plam-vein\Graduate-code\code-one\data2\CUMT2',
        'print_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\CUMT2', 'print_train'),
        'vein_train_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\CUMT2', 'vein_train'),
        'print_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\CUMT2', 'print_test'),
        'vein_test_dir': os.path.join(r'D:\code\plam-vein\Graduate-code\code-one\data2\CUMT2', 'vein_test'),
        'num_classes': 532,
        'img_size': (160, 160),
        'in_channels': 3,
    },
}

DEFAULT_DATASET = 'HandsData'

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4

SEED = 42
DETERMINISTIC = True
CUDNN_BENCHMARK = False

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

FEATURE_DIM = 256
NUM_EXPERTS = 3
OUT_STAGES = [3, 4, 5]
REDUCER_CHANNELS = 64
LOAD_BALANCE_WEIGHT = 0.01

CLASSIFIER_EMBED_DIM = 256
CLASSIFIER_DROPOUT = 0.5
ARC_MARGIN = 0.5
ARC_SCALE = 30.0


def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]


def get_save_dir(dataset_name):
    return os.path.join(BASE_DIR, 'checkpoints6', dataset_name)

#checkpoints5表示的是三层moe
#checkpoints6表示的是cumt2输入分辨率160x160的版本