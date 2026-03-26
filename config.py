import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'HandsData')

PRINT_TRAIN_DIR = os.path.join(DATA_DIR, 'print-train')
PRINT_TEST_DIR = os.path.join(DATA_DIR, 'print-test')
VEIN_TRAIN_DIR = os.path.join(DATA_DIR, 'vein-train')
VEIN_TEST_DIR = os.path.join(DATA_DIR, 'vein-test')

PRINT_IMAGE_SIZE = (217, 190)
VEIN_IMAGE_SIZE = (180, 180)

NUM_CLASSES = 290
NUM_TRAIN_PER_CLASS = 5
NUM_TEST_PER_CLASS = 5

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4

FEATURE_DIM = 256
NUM_EXPERTS = 3

SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
