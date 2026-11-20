import math

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.io import write_pandas
from src.eval.utility import BASE, EXPL, build_features, fit_eval


def _cfg(tmp_path):
    return Config({"dataset": "toy", "output_root": str(tmp_path), "docs_dir": str(tmp_path / "docs"),
                   "als": {"relevance_threshold": 4.0}, "explain": {"max_cited": 2},
                   "utility": {"test_fraction": 0.3, "seed": 1, "repeats": 2}})


def test_build_features(spark, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.out_dir.mkdir(parents=True)
    ev = pd.DataFrame({"userId": [1, 1], "movieId": [10, 11], "rank": [1, 2], "score": [1.0, 0.5],
                       "n_history": [9, 9], "total_pos_influence": [0.5, 0.2],
                       "evidence": [[{"movieId": 1, "influence": 0.3, "rating": 5.0},
                                     {"movieId": 2, "influence": 0.1, "rating": 3.0},
                                     {"movieId": 3, "influence": 0.05, "rating": 1.0}],
                                    [{"movieId": 2, "influence": 0.1, "rating": 3.0}]]})
    fa = pd.DataFrame({"userId": [1, 1, 1], "movieId": [10, 11, 10], "condition": ["cited", "cited", "random"],
                       "rel_drop": [0.4, 0.05, 0.01], "faithful": [True, False, False]})
    train = pd.DataFrame({"userId": [1, 2, 3], "movieId": [10, 10, 1], "rating": [4.0, 4.0, 4.0], "timestamp": [1, 2, 3]})
    test = pd.DataFrame({"userId": [1, 1], "movieId": [10, 11], "rating": [4.5, 2.0], "timestamp": [9, 9]})
    for name, df in [("evidence", ev), ("faithfulness", fa), ("ratings_train", train), ("ratings_test", test)]:
        write_pandas(df, cfg.path(f"{name}.parquet"))
    f = build_features(spark, cfg).toPandas().set_index("movieId")
    assert f.loc[10, "label"] == 1.0 and f.loc[11, "label"] == 0.0        # 2.0 rating is not acceptance
    assert math.isclose(f.loc[10, "top1_share"], 0.3)
    assert math.isclose(f.loc[10, "topn_share"], 0.45)
    assert math.isclose(f.loc[10, "cited_mean_rating"], 4.0)              # only the 2 cited items
    assert f.loc[10, "cited_max_rating"] == 5.0
    assert f.loc[10, "faithful_f"] == 1.0 and f.loc[10, "cited_rel_drop"] == 0.4
    assert math.isclose(f.loc[10, "log_item_pop"], math.log1p(2)) and f.loc[11, "log_item_pop"] == 0.0
    assert len(f) == 2                                                    # random-condition rows not joined


def test_fit_eval_detects_signal(spark):
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, len(BASE + EXPL)))
    y = (X[:, len(BASE)] + 0.3 * rng.normal(size=n) > 0.8).astype(float)   # label driven by first expl feature
    pdf = pd.DataFrame(X, columns=BASE + EXPL).assign(label=y)
    df = spark.createDataFrame(pdf)
    tr, te = df.limit(300), df.subtract(df.limit(300))
    base = fit_eval(tr, te, BASE)["roc_auc"]
    full = fit_eval(tr, te, BASE + EXPL)["roc_auc"]
    assert full > 0.9 and full > base + 0.2
