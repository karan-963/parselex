import os

V1_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.dirname(V1_DIR)
SECTION_DIR = os.path.dirname(PHASE2_DIR)

# Path Settings
INPUT_DATA_DIR = os.path.abspath(os.path.join(PHASE2_DIR, "phase1/input_data"))
TOKEN_DATA_DIR = os.path.join(INPUT_DATA_DIR, "token")
TRAIN_DATA_DIR = os.path.join(TOKEN_DATA_DIR, "train")
VAL_DATA_DIR = os.path.join(TOKEN_DATA_DIR, "val")

# Model Settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CLASSES = 7
SPATIAL_DIM = 10
DROPOUT = 0.1
CONTEXT_EMB_DIM = 32

# Training settings
BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
MAX_GRAD_NORM = 1.0

