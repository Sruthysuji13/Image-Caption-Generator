import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, 'data')
IMAGE_DIR       = os.path.join(DATA_DIR, 'images')
CAPTION_FILE    = os.path.join(DATA_DIR, 'captions.txt')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'best_model.pth')

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

MIN_WORD_FREQ = 5

IMAGE_SIZE = 224
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD  = [0.229, 0.224, 0.225]

EMBED_SIZE   = 256
HIDDEN_SIZE  = 512
NUM_LAYERS   = 1
DROPOUT      = 0.5

BATCH_SIZE    = 32
NUM_EPOCHS    = 25
LEARNING_RATE = 3e-4
GRAD_CLIP     = 5.0

MAX_CAPTION_LEN = 35
BEAM_SIZE       = 1