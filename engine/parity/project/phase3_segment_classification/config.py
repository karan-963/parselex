# Project-specific segment class labels
# B-DESC / I-DESC          → DESC
# B-PROJ_NAME / I-PROJ_NAME → PROJ
# B-PROJ_COMPANY / I-PROJ_COMPANY → COMP
# B-SDATE / I-SDATE        → SDATE
# B-EDATE / I-EDATE        → EDATE

LABEL_LIST = ["O", "PROJECT_NAME", "DATE", "DESC"]
LABEL2ID   = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL   = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
NUM_LABELS  = len(LABEL_LIST)
import classifier_config as _cfg

BACKBONE_NAME = _cfg.BACKBONE_NAME
CHECKPOINT_PATH = _cfg.CHECKPOINT_PATH
REPORTS_DIR = _cfg.REPORTS_DIR
SPATIAL_DIM = _cfg.SPATIAL_DIM

# Pipeline configuration exclusions — shared with Phase 2 and Phase 1
EXCLUDED_RESUMES = {
    "Rutvik_Shinde_QA_Resume",
    "Sanjai_Resume1",
    "Seema_Angadi_BE",
    "Shilpy_CV",
    "Sohan_Nayak_ResumeBA",
    "sahithai_resume_2"
}
