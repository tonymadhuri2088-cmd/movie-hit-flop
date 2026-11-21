"""Single place for every constant that defines the experiment. Changing one of these changes the science, so each is documented."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
LOG_PATH = ROOT / "experiments" / "experiment_log.jsonl"

SEED = 42
HIT_THRESHOLD = 2.0              # Hit = revenue / budget >= 2.0 (unchanged from the original project; 1.5 and 2.5 are only sensitivity checks)
ROI_THRESHOLDS = (1.5, 2.0, 2.5)
MATURITY_DAYS = 120              # a film's box office counts as known 120 days after release (labels and track records)
MIN_BUDGET, MIN_REVENUE = 100_000, 10_000   # plausibility filters (original project)
SPLIT_DATE = "2023-01-01"        # development data before, frozen final test from this date
N_OUTER = 5                      # expanding-window folds on development data
N_INNER = 3
SMOOTH_M = 2                     # track records are shrunk towards the running overall Hit rate as if 2 average films were added
TOP_CAST = 5
PRIOR_NO_HISTORY = 0.4
