"""One call that builds the complete modelling table (all features) for all labelled movies, plus the frozen dev/test split."""
import pandas as pd

from . import config as C
from . import data
from . import features as F
from . import history as H


def build_world(threshold=C.HIT_THRESHOLD):
    merged = data.load_merged()
    infobox = pd.read_csv(C.DATA_DIR / "wiki_infobox_labelled.csv", keep_default_na=False, encoding="utf-8-sig")
    merged = H.attach_infobox(merged, infobox)
    lab, flog = data.label_movies(merged, threshold=threshold)
    lab = data.build_basic_features(lab)
    lab = F.add_research_features(lab, merged)
    lab = lab.join(H.add_track_records(lab, merged))
    dev, test = data.split_dev_test(lab)
    return merged, lab, dev, test
