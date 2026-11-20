"""Stage 1b: tune + train ALS (Spark MLlib), write factors, top-K recs and metrics.

python -m src.als.train [--set als.ranks=[20] --set als.reg_params=[0.1]]
Reads ratings_train/test.parquet. Writes als_model/ (factors + meta.json),
recs.parquet (userId, movieId, score, rank), als_metrics.json, als_tuning.csv.

Tuning uses a time-based validation split carved from the end of each user's
training history (never the test set), then refits on the full training set.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.als.data import time_split
from src.als.factors import save_factors
from src.als.metrics import ranking_metrics
from src.common.config import cli
from src.common.io import read_parquet, write_parquet
from src.common.spark import get_spark

log = logging.getLogger("train")

RECS_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType()), T.StructField("movieId", T.IntegerType()),
    T.StructField("score", T.DoubleType()), T.StructField("rank", T.IntegerType()),
])


def fit_als(train: DataFrame, rank: int, reg: float, max_iter: int, seed: int,
            implicit: bool = False, alpha: float = 1.0) -> ALSModel:
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=rank, regParam=reg,
              maxIter=max_iter, seed=seed, implicitPrefs=implicit, alpha=alpha,
              coldStartStrategy="drop", nonnegative=False)
    return als.fit(train)


def rmse(model: ALSModel, df: DataFrame) -> float:
    pred = model.transform(df)
    return RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction").evaluate(pred)


def top_k_unseen(spark: SparkSession, history: DataFrame, user_vecs: DataFrame,
                 item_ids: np.ndarray, Y: np.ndarray, k: int, users: DataFrame | None = None) -> DataFrame:
    """Top-k items per user by x_u . y_i, excluding items in the user's history.

    Runs per user in parallel (cogroup + applyInPandas) with item factors broadcast.
    user_vecs: userId, features (array).  For a popularity baseline pass Y as an
    (m,1) popularity column and user vectors of [1.0].
    """
    bc = spark.sparkContext.broadcast((item_ids.astype(np.int64), Y.astype(np.float64)))
    if users is not None:
        history = history.join(users, "userId", "left_semi")
        user_vecs = user_vecs.join(users, "userId", "left_semi")

    def score(key, hist: pd.DataFrame, uv: pd.DataFrame) -> pd.DataFrame:
        if uv.empty:
            return pd.DataFrame(columns=["userId", "movieId", "score", "rank"])
        ids, Ym = bc.value
        x = np.asarray(uv["features"].iloc[0], dtype=np.float64)
        s = Ym @ x
        seen = np.isin(ids, hist["movieId"].to_numpy())
        s = np.where(seen, -np.inf, s)
        kk = min(k, int((~seen).sum()))
        top = np.argpartition(-s, kk - 1)[:kk] if kk > 0 else np.array([], dtype=int)
        top = top[np.argsort(-s[top], kind="stable")]
        return pd.DataFrame({"userId": int(key[0]), "movieId": ids[top].astype(np.int32),
                             "score": s[top], "rank": np.arange(1, len(top) + 1, dtype=np.int32)})

    return (history.select("userId", "movieId").groupBy("userId")
            .cogroup(user_vecs.groupBy("userId")).applyInPandas(score, RECS_SCHEMA))


def factors_np(model: ALSModel) -> tuple[np.ndarray, np.ndarray]:
    items = model.itemFactors.toPandas().sort_values("id")
    return items["id"].to_numpy(np.int64), np.vstack(items["features"].to_numpy())


def user_vecs(model: ALSModel) -> DataFrame:
    return model.userFactors.select(F.col("id").alias("userId"), "features")


def run(cfg, spark) -> dict:
    a, s = cfg["als"], cfg["split"]
    k, implicit = a["top_k"], bool(a["implicit"])
    train = read_parquet(spark, cfg.path("ratings_train.parquet")).cache()
    test = read_parquet(spark, cfg.path("ratings_test.parquet")).cache()

    # ---- tuning on a time-based validation split of train (never the test set)
    tr, val = time_split(train, s["val_fraction"], "per_user", min_user_train=3)
    tr, val = tr.cache(), val.cache()
    val_users = val.select("userId").distinct().cache()
    rows = []
    for rank in a["ranks"]:
        for reg in a["reg_params"]:
            for alpha in (a["alphas"] if implicit else [1.0]):
                t0 = time.time()
                m = fit_als(tr, rank, reg, a["max_iter"], a["seed"], implicit, alpha)
                ids, Y = factors_np(m)
                recs = top_k_unseen(spark, tr, user_vecs(m), ids, Y, k, users=val_users)
                rm = ranking_metrics(recs, val, k, a["relevance_threshold"])
                row = {"implicit": implicit, "rank": rank, "regParam": reg, "alpha": alpha,
                       f"val_ndcg@{k}": rm[f"ndcg@{k}"], f"val_precision@{k}": rm[f"precision@{k}"],
                       "seconds": round(time.time() - t0, 1)}
                log.info("tune %s", row)
                rows.append(row)
    tuning = pd.DataFrame(rows)
    tuning.to_csv(cfg.path("als_tuning.csv"), index=False)
    best = tuning.sort_values(f"val_ndcg@{k}", ascending=False).iloc[0]
    rank, reg, alpha = int(best["rank"]), float(best["regParam"]), float(best["alpha"])
    log.info("best by val NDCG@%d: rank=%d regParam=%s alpha=%s", k, rank, reg, alpha)

    # ---- final model on full train
    t0 = time.time()
    model = fit_als(train, rank, reg, a["max_iter"], a["seed"], implicit, alpha)
    model.userFactors.cache().count()
    fit_s = time.time() - t0
    meta = {"rank": rank, "regParam": reg, "alpha": alpha, "implicitPrefs": implicit,
            "maxIter": a["max_iter"], "dataset": cfg["dataset"]}
    save_factors(model, meta, cfg.path("als_model"), write_parquet)
    ids, Y = factors_np(model)
    recs = top_k_unseen(spark, train, user_vecs(model), ids, Y, k).cache()
    write_parquet(recs, cfg.path("recs.parquet"))
    metrics = {"model": "ALS-implicit" if implicit else "ALS-explicit", **meta, "fit_seconds": round(fit_s, 1)}
    metrics.update(ranking_metrics(recs, test, k, a["relevance_threshold"]))

    # ---- RMSE: explicit ALS as rating predictor (regParam picked on validation RMSE)
    rm_rows = [(reg_e, rmse(fit_als(tr, rank, reg_e, a["max_iter"], a["seed"]), val)) for reg_e in a["rmse_reg_params"]]
    reg_e = min(rm_rows, key=lambda t: t[1])[0]
    explicit = fit_als(train, rank, reg_e, a["max_iter"], a["seed"])
    rmse_info = {"model": "ALS-explicit", "rank": rank, "regParam": reg_e,
                 "val_rmse_grid": {str(r): round(v, 4) for r, v in rm_rows}, "test_rmse": rmse(explicit, test)}
    if not implicit:
        metrics["test_rmse"] = rmse(model, test)

    # ---- popularity baseline under the same unseen-item protocol
    pop = train.groupBy("movieId").count().toPandas().sort_values("movieId")
    pop_vecs = train.select("userId").distinct().withColumn("features", F.array(F.lit(1.0)))
    pop_recs = top_k_unseen(spark, train, pop_vecs, pop["movieId"].to_numpy(),
                            np.log1p(pop["count"].to_numpy(np.float64))[:, None], k)
    baseline = {"model": "Popularity", **ranking_metrics(pop_recs, test, k, a["relevance_threshold"])}

    out = {"recommender": metrics, "popularity": baseline, "rating_prediction": rmse_info}
    with open(cfg.path("als_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    for name, v in out.items():
        log.info("%s: %s", name, v)
    return out


if __name__ == "__main__":
    cfg, _ = cli(__doc__)
    spark = get_spark(cfg, "i-recs-train")
    run(cfg, spark)
    spark.stop()
