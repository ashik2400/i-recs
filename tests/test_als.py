import math

import numpy as np
import pandas as pd

from src.als.data import RATING_SCHEMA, clean, time_split
from src.als.metrics import ranking_metrics
from src.als.train import fit_als, top_k_unseen, factors_np, user_vecs


def _ratings(spark):
    rows = []
    for u in range(1, 6):
        for t in range(10):
            rows.append((u, t + u, float(1 + (t % 5)), 1000 + 10 * t + u))
    rows.append((1, 1, 2.0, 999))      # duplicate (1,1), older -> dropped by clean
    return spark.createDataFrame(rows, RATING_SCHEMA)


def test_clean_keeps_latest_duplicate(spark):
    c = clean(_ratings(spark)).filter("userId = 1 AND movieId = 1").collect()
    assert len(c) == 1 and c[0].timestamp == 1001


def test_per_user_split_is_temporal(spark):
    tr, te = time_split(clean(_ratings(spark)), 0.2, "per_user", min_user_train=3)
    trp, tep = tr.toPandas(), te.toPandas()
    assert len(trp) + len(tep) <= 50
    for u, g in tep.groupby("userId"):
        assert g["timestamp"].min() > trp[trp.userId == u]["timestamp"].max()
    assert set(tep.movieId) <= set(trp.movieId)          # no cold-start items in test


def test_global_split_cutoff(spark):
    tr, te = time_split(clean(_ratings(spark)), 0.3, "global", min_user_train=1)
    assert tr.agg({"timestamp": "max"}).first()[0] < te.agg({"timestamp": "min"}).first()[0]


def test_ranking_metrics_hand_computed(spark):
    recs = spark.createDataFrame([(1, 10, 1), (1, 11, 2), (1, 12, 3), (2, 20, 1), (2, 21, 2), (2, 22, 3)],
                                 ["userId", "movieId", "rank"])
    test = spark.createDataFrame([(1, 11, 5.0), (1, 99, 4.0), (2, 22, 4.5), (2, 20, 2.0)],
                                 ["userId", "movieId", "rating"])
    m = ranking_metrics(recs, test, 3, 4.0)
    # user1: hit at rank 2, n_rel 2 ; user2: hit at rank 3, n_rel 1 (20 is not relevant)
    assert math.isclose(m["precision@3"], (1 / 3 + 1 / 3) / 2)
    assert math.isclose(m["recall@3"], (1 / 2 + 1) / 2)
    assert math.isclose(m["mrr@3"], (1 / 2 + 1 / 3) / 2)
    nd1 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    nd2 = (1 / math.log2(4)) / 1
    assert math.isclose(m["ndcg@3"], (nd1 + nd2) / 2)
    assert m["n_eval_users"] == 2


def test_top_k_excludes_seen(spark):
    df = clean(_ratings(spark))
    m = fit_als(df, rank=3, reg=0.1, max_iter=3, seed=1, implicit=True)
    ids, Y = factors_np(m)
    recs = top_k_unseen(spark, df, user_vecs(m), ids, Y, k=4).toPandas()
    seen = df.toPandas()
    for u, g in recs.groupby("userId"):
        assert not set(g.movieId) & set(seen[seen.userId == u].movieId)
        assert list(g["rank"]) == list(range(1, len(g) + 1))
        assert np.all(np.diff(g["score"].to_numpy()) <= 1e-12)
