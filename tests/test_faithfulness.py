"""Faithfulness verification (M5): decision rule, counterfactual math, baseline, Spark path."""
import numpy as np
import pandas as pd
import pytest

from src.als.factors import FactorModel
from src.evidence.extract import user_evidence
from src.verify.faithfulness import (is_faithful, random_removal, rel_drop, summarize, verify,
                                     verify_user)


# ------------------------------------------------------------------ decision rule
@pytest.mark.parametrize("before,after,rank_after,thr,use_rank,expected", [
    (1.0, 0.85, 3, 0.10, True, True),      # 15% drop
    (1.0, 0.95, 3, 0.10, True, False),     # 5% drop, still in top-k
    (1.0, 0.95, 11, 0.10, True, True),     # small drop but fell out of top-10
    (1.0, 0.95, 11, 0.10, False, False),   # rank criterion disabled
    (1.0, 1.20, 1, 0.10, True, False),     # score went up
    (-0.5, -0.6, 2, 0.10, False, True),    # negative scores: relative to |before|
    (1.0, 0.90, 1, 0.10, False, True),     # exactly at threshold counts
])
def test_is_faithful(before, after, rank_after, thr, use_rank, expected):
    assert is_faithful(before, after, rank_after, 10, thr, use_rank) is expected


def test_rel_drop_zero_safe():
    assert np.isfinite(rel_drop(0.0, -1.0))


# ------------------------------------------------------------------ fixtures
def _planted(implicit=True):
    """Items 0-4 share a 'taste' direction with the target item 50; items 5-29 are noise.
    A user who rated items 0-1 and 5-12 should get item 50 *because of* items 0-1."""
    rng = np.random.default_rng(0)
    k, m = 6, 60
    Y = rng.normal(scale=0.05, size=(m, k))
    taste = np.zeros(k)
    taste[0] = 1.0
    Y[0:5] += taste
    Y[50] += taste
    fm = FactorModel(np.arange(1000, 1000 + m), Y,
                     {"rank": k, "regParam": 0.05, "alpha": 2.0, "implicitPrefs": implicit})
    hist = pd.DataFrame({"userId": 7, "movieId": [1000, 1001] + list(range(1005, 1013)),
                         "rating": [5.0, 5.0] + [3.0] * 8})
    return fm, hist


def _expl(cited, movie=1050, user=7):
    return pd.DataFrame({"userId": [user], "movieId": [movie], "cited_items": [cited]})


# ------------------------------------------------------------------ counterfactual math
def test_cited_driver_is_faithful_and_noise_is_not():
    fm, hist = _planted()
    s = fm.solver()
    good = verify_user(hist, _expl([1000, 1001]), fm, s, top_k=3, threshold=0.10, use_rank=False, trials=5, seed=1)
    bad = verify_user(hist, _expl([1005, 1006]), fm, s, top_k=3, threshold=0.10, use_rank=False, trials=0, seed=1)
    g = good[good.condition == "cited"].iloc[0]
    b = bad[bad.condition == "cited"].iloc[0]
    assert g.rel_drop > 0.5 and g.faithful
    assert b.rel_drop < 0.1 and not b.faithful          # noise removal may even raise the score
    # random baseline draws from non-cited history -> mostly noise items -> much smaller drop
    assert good[good.condition == "random"].rel_drop.mean() < g.rel_drop


def test_scores_match_brute_force_fold_in():
    fm, hist = _planted(implicit=False)
    s = fm.solver()
    out = verify_user(hist, _expl([1000, 1007]), fm, s, top_k=10, threshold=0.1, use_rank=True, trials=2, seed=3)
    idx, r, ids = fm.idx(hist.movieId), hist.rating.to_numpy(), hist.movieId.to_numpy()
    cand = np.ones(len(fm.item_ids), bool)
    cand[idx] = False
    t = fm.index[1050]
    full = fm.item_factors @ s.user_vector(idx, r)
    for row in out.itertuples():
        after = fm.item_factors @ s.user_vector_without(idx, r, np.isin(ids, row.cited_item))
        assert np.isclose(row.score_before, full[t]) and np.isclose(row.score_after, after[t])
        assert row.rank_before == int((full[cand] > full[t]).sum()) + 1
        assert row.rank_after == int((after[cand] > after[t]).sum()) + 1   # candidate set fixed
        assert row.n_history == len(hist)


def test_fold_in_counterfactual_agrees_with_evidence_influence():
    """Removing one cited item must reproduce exactly the influence M2 reported for it."""
    fm, hist = _planted()
    s = fm.solver()
    recs = pd.DataFrame({"userId": 7, "movieId": [1050], "rank": [1]})
    ev = user_evidence(hist, recs, fm, s, top_n=3).iloc[0]
    top = ev.evidence[0]
    out = verify_user(hist, _expl([top["movieId"]]), fm, s, 10, 0.1, True, 0, 0).iloc[0]
    assert np.isclose(out.score_before - out.score_after, top["influence"])
    assert top["movieId"] in (1000, 1001)                           # the planted drivers


# ------------------------------------------------------------------ random baseline
def test_random_removal_properties():
    h = np.arange(100, 120)
    cited = {100, 101}
    a = random_removal(h, cited, 2, seed=5, user=1, movie=9, trial=0)
    b = random_removal(h, cited, 2, seed=5, user=1, movie=9, trial=0)
    c = random_removal(h, cited, 2, seed=5, user=1, movie=9, trial=1)
    assert len(a) == 2 and len(set(a.tolist())) == 2 and not set(a.tolist()) & cited
    assert np.array_equal(a, b)                                      # deterministic
    draws = {tuple(sorted(random_removal(h, cited, 2, 5, 1, 9, t).tolist())) for t in range(10)}
    assert len(draws) > 1                                            # trials give different draws
    assert random_removal(np.array([100, 101, 102]), cited, 2, 0, 1, 1, 0) is None   # not enough items


def test_skips_cited_items_not_in_history_and_seen_targets():
    fm, hist = _planted()
    s = fm.solver()
    assert verify_user(hist, _expl([99999]), fm, s, 10, 0.1, True, 2, 0).empty          # nothing to remove
    assert verify_user(hist, _expl([1000], movie=1005), fm, s, 10, 0.1, True, 2, 0).empty  # target already seen


# ------------------------------------------------------------------ summary + Spark path
def test_summary_and_sweep():
    fm, hist = _planted()
    out = verify_user(hist, _expl([1000, 1001]), fm, fm.solver(), 3, 0.1, False, 4, 0)
    summ, sweep = summarize(out, 3, 0.1, False)
    assert summ["n_explanations"] == 1 and summ["n_random_trials"] == 4
    assert summ["faithfulness_score"] == 1.0
    assert summ["paired"]["cited_beats_random_rate"] == 1.0
    cited_curve = sweep[sweep.condition == "cited"].sort_values("threshold")["faithful_score_only"].to_numpy()
    assert np.all(np.diff(cited_curve) <= 0)                         # monotone in the threshold


def test_spark_verify_matches_numpy(spark):
    fm, hist = _planted()
    hist2 = hist.assign(userId=8, rating=hist.rating[::-1].to_numpy())
    allh = pd.concat([hist, hist2])
    ex = pd.DataFrame({"userId": [7, 8], "movieId": [1050, 1050], "cited_items": [[1000, 1001], [1005]]})
    got = verify(spark, spark.createDataFrame(allh), spark.createDataFrame(ex), fm, 10, 0.1, True, 2, 4).toPandas()
    for u in (7, 8):
        exp = verify_user(allh[allh.userId == u], ex[ex.userId == u], fm, fm.solver(), 10, 0.1, True, 2, 4)
        g = got[got.userId == u].sort_values(["condition", "trial"]).reset_index(drop=True)
        e = exp.sort_values(["condition", "trial"]).reset_index(drop=True)
        np.testing.assert_allclose(g.score_after.to_numpy(), e.score_after.to_numpy())
        assert g.faithful.tolist() == e.faithful.tolist()
