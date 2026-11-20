import numpy as np
import pandas as pd

from src.als.factors import FactorModel
from src.evidence.extract import extract, user_evidence


def _model(seed=0, m=40, k=5, implicit=True):
    rng = np.random.default_rng(seed)
    meta = {"rank": k, "regParam": 0.05, "alpha": 2.0, "implicitPrefs": implicit}
    return FactorModel(np.arange(100, 100 + m), rng.normal(size=(m, k)), meta)


def _hist(user, items, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"userId": user, "movieId": items, "rating": rng.integers(1, 6, len(items)).astype(float)})


def test_influence_equals_brute_force_removal():
    fm = _model()
    s = fm.solver()
    hist = _hist(1, np.arange(100, 115))
    recs = pd.DataFrame({"userId": 1, "movieId": [130, 131], "rank": [1, 2]})
    out = user_evidence(hist, recs, fm, s, top_n=4)
    idx, r = fm.idx(hist.movieId), hist.rating.to_numpy()
    for _, row in out.iterrows():
        y = fm.item_factors[fm.index[row.movieId]]
        full = s.user_vector(idx, r) @ y
        assert np.isclose(row.score, full)
        brute = {int(m): full - s.user_vector_without(idx, r, hist.movieId.to_numpy() == m) @ y for m in hist.movieId}
        ev = row.evidence
        assert 0 < len(ev) <= 4
        expected = sorted([(v, k) for k, v in brute.items() if v > 0], reverse=True)[:4]
        assert [e["movieId"] for e in ev] == [k for _, k in expected]
        np.testing.assert_allclose([e["influence"] for e in ev], [v for v, _ in expected], rtol=1e-8)
        assert row.total_pos_influence >= sum(e["influence"] for e in ev) - 1e-12


def test_unknown_history_items_are_ignored():
    fm = _model()
    hist = _hist(1, [100, 101, 102, 99999])        # 99999 not in the model
    recs = pd.DataFrame({"userId": 1, "movieId": [120], "rank": [1]})
    out = user_evidence(hist, recs, fm, fm.solver(), top_n=5)
    assert out.n_history.iloc[0] == 3
    assert all(e["movieId"] != 99999 for e in out.evidence.iloc[0])


def test_spark_extract_matches_numpy(spark):
    fm = _model(implicit=False)
    hist = pd.concat([_hist(1, np.arange(100, 110)), _hist(2, np.arange(105, 120), seed=2)])
    recs = pd.DataFrame({"userId": [1, 1, 2], "movieId": [125, 126, 127], "rank": [1, 2, 1]})
    got = extract(spark, spark.createDataFrame(hist), spark.createDataFrame(recs), fm, 3).toPandas()
    assert len(got) == 3
    for u in (1, 2):
        exp = user_evidence(hist[hist.userId == u], recs[recs.userId == u], fm, fm.solver(), 3)
        g = got[got.userId == u].sort_values("rank")
        np.testing.assert_allclose(g.score.to_numpy(), exp.score.to_numpy())
        assert [[e["movieId"] for e in ev] for ev in g.evidence] == [[e["movieId"] for e in ev] for ev in exp.evidence]
