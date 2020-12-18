from pathlib import Path

# Directories
HOME_DIR = Path.home()
FUSION_DATA_DIR = Path("/data/fusion")
if not FUSION_DATA_DIR.is_dir():
    raise Exception("Please modify PROJECT_DATA_DIR in constants to a valid directory")

# Raw data
EMR_DATA_DIR = FUSION_DATA_DIR / "emr_data"
ALL_EMR_RAW_CSV = EMR_DATA_DIR / "All.csv"
DEMOGRAPHICS_RAW_CSV = EMR_DATA_DIR / "Demographics.csv"
ICD_RAW_CSV = EMR_DATA_DIR / "ICD.csv"
INP_MED_RAW_CSV = EMR_DATA_DIR / "INP_MED.csv"
OUT_MED_RAW_CSV = EMR_DATA_DIR / "OUT_MED.csv"
LABS_RAW_CSV = EMR_DATA_DIR / "LABS.csv"
VITALS_CSV = EMR_DATA_DIR / "Vitals.csv"
RAW_EMR_DATA = [
    ALL_EMR_RAW_CSV,
    DEMOGRAPHICS_RAW_CSV,
    ICD_RAW_CSV,
    INP_MED_RAW_CSV,
    OUT_MED_RAW_CSV,
    LABS_RAW_CSV,
    VITALS_CSV
]

# Vision
VISION_DATA_DIR = FUSION_DATA_DIR / "vision_feature"
VISION_FEATURES = VISION_DATA_DIR / "vision.pickle"

# Mappings
MAPPINGS_DIR = FUSION_DATA_DIR / "mappings"
IDX_2_ACC = MAPPINGS_DIR / "idx_2_acc_stanford.pkl"
IDX_2_SPLIT = MAPPINGS_DIR / "idx_2_split_stanford.pkl"
ACC_2_TYPE = MAPPINGS_DIR / "acc_2_type.pkl"
ACC_2_LABEL = MAPPINGS_DIR / "acc_2_label.pkl"

# Parsed data
PARSED_DATA_DIR = FUSION_DATA_DIR / "parsed_data"
ALL_DIR = PARSED_DATA_DIR / "All"
DEMOGRAPHICS_DIR = PARSED_DATA_DIR / "Demographics"
ICD_DIR = PARSED_DATA_DIR / "ICD"
INP_MED_DIR = PARSED_DATA_DIR / "INP_MED"
OUT_MED_DIR = PARSED_DATA_DIR / "OUT_MED"
LABS_DIR = PARSED_DATA_DIR / "LABS"
VITALS_DIR = PARSED_DATA_DIR / "Vitals"
VISION_DIR = PARSED_DATA_DIR / "Vision"
PARSED_EMR_DATA = [
    ALL_DIR,
    DEMOGRAPHICS_DIR,
    ICD_DIR,
    INP_MED_DIR,
    OUT_MED_DIR,
    LABS_DIR,
    VITALS_DIR,
]
PARSED_EMR_DICT = {
    "All": ALL_DIR,
    "Demographics": DEMOGRAPHICS_DIR,
    "ICD": ICD_DIR,
    "INP_MED": INP_MED_DIR,
    "OUT_MED": OUT_MED_DIR,
    "LABS": LABS_DIR,
    "Vitals": VITALS_DIR,
    "Vision": VISION_DIR,
}

# Logging
LOG_DIR = FUSION_DATA_DIR / 'logs'
CKPT_DIR = FUSION_DATA_DIR / 'ckpt'
RESULTS_DIR = FUSION_DATA_DIR / 'results'

# Column names
ACCESSION_COL = "accession"
PROBS_COL = "probs"
LABELS_COL = "labels"