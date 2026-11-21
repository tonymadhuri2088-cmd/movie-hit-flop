"""The new project must reproduce the original notebook's cohort and chronological split exactly."""
import numpy as np


def test_cohort_sizes_match_original_notebook(world):
    assert len(world["merged"]) == 4273
    assert len(world["lab"]) == 854
    assert len(world["dev"]) == 578
    assert len(world["test"]) == 276


def test_hit_rates_match_original_notebook(world):
    assert round(world["dev"].hit.mean(), 3) == 0.471
    assert round(world["test"].hit.mean(), 3) == 0.297


def test_hit_definition_is_roi_ge_2(world):
    lab = world["lab"]
    assert ((lab.revenue / lab.budget >= 2.0).astype(int) == lab.hit).all()


def test_dev_strictly_before_test(world):
    assert world["dev"].release_date.max() < world["test"].release_date.min()
    assert world["dev"].release_date.max() < np.datetime64("2023-01-01")


def test_every_labelled_movie_is_mature(world):
    last = world["merged"].release_date.max()
    assert (world["lab"].release_date <= last - np.timedelta64(120, "D")).all()


def test_expanding_window_folds_never_look_ahead(world):
    dev = world["dev"]
    seen = []
    for tr, va in world["folds"]:
        assert dev.release_date.iloc[tr].max() <= dev.release_date.iloc[va].min()   # train is the past of validation
        assert tr.min() == 0                                                         # expanding window: always starts at the beginning
        seen.extend(va.tolist())
    assert len(seen) == 480 and len(set(seen)) == 480                                # no validation row appears twice
    assert len(world["folds"]) == 5


def test_folds_train_sizes_grow(world):
    sizes = [len(tr) for tr, _ in world["folds"]]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_unique_ids_and_sorted(world):
    lab = world["lab"]
    assert lab.tmdb_id.is_unique and lab.release_date.is_monotonic_increasing
