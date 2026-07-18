import os

# Database Settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "resume-labeling")

# Path Settings
V5_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = os.path.join(V5_DIR, "input_data")
TOKEN_DATA_DIR = os.path.join(INPUT_DATA_DIR, "token")
TRAIN_DATA_DIR = os.path.join(TOKEN_DATA_DIR, "train")
VAL_DATA_DIR = os.path.join(TOKEN_DATA_DIR, "val")

# Line hybrid heading detection (locked production method)
LINE_MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LINE_SPATIAL_DIM = 10
LINE_MAX_SEQ_LEN = 128
LINE_BATCH_SIZE = 16
LINE_EPOCHS = 8
LINE_LR = 2e-5
LINE_BEST_MODEL = "best_model_line_minilm.pt"
LINE_ONNX_NAME = "line_minilm.onnx"
LINE_REPORT_DIR = os.path.join(V5_DIR, "reports_line_hybrid")
SAVED_MODELS_DIR = V5_DIR
MANIFEST_PATH = os.path.join(SAVED_MODELS_DIR, "model_manifest.json")

# Hybrid thresholds (must match line_hybrid_predict.py)
HEURISTIC_SKIP_CONF = 0.72
MODEL_HEADING_PROB = 0.5
