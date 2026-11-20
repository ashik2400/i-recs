"""Fold-in must reproduce Spark ALS exactly; leave-one-out must equal direct removal."""
import numpy as np
import pytest
from pyspark.ml.recommendation import ALS

from src.als.factors import FoldIn, rank_among


def _toy_ratings(seed=0, n_users=40, n_items=30, density=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        items = np.flatnonzero(rng.random(n_items) < density)
        if len(items) < 2:
            items = np.array([0, 1])
        for i in items:
            rows.append((u, int(i), float(rng.integers(1, 11) / 2)))
    return rows


@pytest.mark.parametrize("implicit,reg,alpha", [(False, 0.1, 1.0), (True, 0.05, 2.0)])
def test_fold_in_reproduces_spark_user_factors(spark, implicit, reg, alpha):
    rows = _toy_ratings()
    df = spark.createDataFrame(rows, ["userId", "movieId", "rating"])
    m = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=4, regParam=reg,
            alpha=alpha, implicitPrefs=implicit, maxIter=8, seed=1).fit(df)
    items = m.itemFactors.toPandas().sort_values("id")
    ids = items["id"].to_numpy()
    Y = np.vstack(items["features"].to_numpy())
    pos = {int(i): j for j, i in enumerate(ids)}
    fi = FoldIn(Y, reg, implicit, alpha)
    users = {r.id: np.array(r.features) for r in m.userFactors.collect()}
    for u in range(10):
        hist = [(i, r) for (uu, i, r) in rows if uu == u]
        idx = np.array([pos[i] for i, _ in hist])
        r = np.array([r for _, r in hist])
        x = fi.user_vector(idx, r)
        # Spark stores float32 factors; agreement to ~1e-4 relative
        np.testing.assert_allclose(x, users[u], rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("implicit", [False, True])
def test_leave_one_out_matches_direct_removal(implicit):
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(50, 6))
    fi = FoldIn(Y, 0.1, implicit, 3.0)
    idx = rng.choice(50, size=12, replace=False)
    r = rng.integers(1, 6, size=12).astype(float)
    loo = fi.leave_one_out(idx, r, chunk=5)  # chunk < n exercises batching
    for j in range(12):
        mask = np.zeros(12, bool)
        mask[j] = True
        np.testing.assert_allclose(loo[j], fi.user_vector_without(idx, r, mask), rtol=1e-9, atol=1e-10)


def test_edge_cases():
    fi = FoldIn(np.eye(3), 0.1)
    assert np.all(fi.user_vector(np.array([], int), np.array([])) == 0)
    assert np.all(fi.leave_one_out(np.array([1]), np.array([4.0])) == 0)
    scores = np.array([0.9, 0.5, 0.7, 0.1])
    cand = np.array([True, True, True, False])
    assert rank_among(scores, 2, cand) == 2
    assert rank_among(scores, 0, cand) == 1
