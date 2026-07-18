# Education-specific segment class labels
# B-INST   → Institution / University / College name
# B-DEG    → Degree / Programme / Title (e.g. B.Tech, MBA)
# B-SDATE / B-EDATE → Study start/end dates  (mapped to DATE)
# O / B-GPA / B-LOC → Supporting detail lines (mapped to DESCRIPTION)

LABEL_LIST = ["DESCRIPTION", "INSTITUTION", "DEGREE", "DATE"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

import classifier_config as _cfg

BACKBONE_NAME = _cfg.BACKBONE_NAME
CHECKPOINT_PATH = _cfg.CHECKPOINT_PATH
REPORTS_DIR = _cfg.REPORTS_DIR
SPATIAL_DIM = _cfg.SPATIAL_DIM

GLOBAL_EXCLUSIONS = {
    "Rutvik_Shinde_QA_Resume",
    "Sanjai_Resume1",
    "Seema_Angadi_BE",
    "Shilpy_CV",
    "Sohan_Nayak_ResumeBA",
    "sahithai_resume_2",
}

# Backward compat for any stale imports
EXCLUDED_RESUMES = GLOBAL_EXCLUSIONS


def is_education_section_excluded(doc: dict) -> bool:
    if doc.get("trainingMeta", {}).get("split") == "excluded":
        return True
    return "education" in doc.get("excludedSections", [])
