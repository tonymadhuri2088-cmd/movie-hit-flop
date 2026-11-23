import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hitflop import config as C  # noqa: E402
from hitflop import data, history


@pytest.fixture(scope="session")
def world():
    merged = data.load_merged()
    infobox = pd.read_csv(C.DATA_DIR / "wiki_infobox_labelled.csv", keep_default_na=False, encoding="utf-8-sig")
    merged = history.attach_infobox(merged, infobox)
    lab, flog = data.label_movies(merged)
    lab = data.build_basic_features(lab)
    dev, test = data.split_dev_test(lab)
    return {"merged": merged, "lab": lab, "dev": dev, "test": test, "folds": data.outer_folds(dev), "flog": flog}
