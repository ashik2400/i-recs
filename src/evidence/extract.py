"""Stage 2: perturbation-based evidence extraction via fold-in.

python -m src.evidence.extract [--set evidence.n_users=500]

For a sample of users and each of their top-K recommendations, the influence of
past interaction j on recommended item i is

    influence_j(i) = score(i | full history) - score(i | history without j)

where both user vectors come from fold-in (item factors fixed; see
src/als/factors.py). ALS is never retrained. All leave-one-out vectors of a
user are one batched solve, and users are processed in parallel by Spark
(cogroup + applyInPandas, item factors broadcast).

Writes evidence.parquet:
  userId, movieId (recommended), rank, score, n_history, total_pos_influence,
  evidence: array<struct<movieId, influence, rating>>  (top-N by influence, influence > 0)
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.als.factors import FactorModel, load_factor_model
from src.common.config import cli
from src.common.io import read_parquet, write_parquet
from src.common.spark import get_spark

log = logging.getLogger("evidence")

EVIDENCE_ITEM = T.StructType([
    T.StructField("movieId", T.IntegerType()),
    T.StructField("influence", T.DoubleType()),
    T.StructField("rating", T.FloatType()),
])
EVIDENCE_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType()),
    T.StructField("movieId", T.IntegerType()),
    T.StructField("rank", T.IntegerType()),
    T.StructField("score", T.DoubleType()),
    T.StructField("n_history", T.IntegerType()),
    T.StructField("total_pos_influence", T.DoubleType()),
    T.StructField("evidence", T.ArrayType(EVIDENCE_ITEM)),
])


def user_evidence(hist: pd.DataFrame, recs: pd.DataFrame, fm: FactorModel, solver, top_n: int) -> pd.DataFrame:
    """Evidence for one user's recommendations (pure numpy; unit-testable)."""
    hist = hist[fm.idx(hist["movieId"]) >= 0]
    h_ids = hist["movieId"].to_numpy(np.int64)
    idx = fm.idx(h_ids)
    r = hist["rating"].to_numpy(np.float64)
    rec_idx = fm.idx(recs["movieId"])
    Yrec = fm.item_factors[rec_idx]                     # (R, k)
    x = solver.user_vector(idx, r)
    s_full = Yrec @ x                                   # (R,)
    X_loo = solver.leave_one_out(idx, r)                # (n, k)
    infl = s_full[None, :] - X_loo @ Yrec.T             # (n, R)
    rows = []
    for c, (_, rec) in enumerate(recs.iterrows()):
        col = infl[:, c]
        pos = col > 0
        order = np.argsort(-col)[:top_n]
        ev = [{"movieId": int(h_ids[j]), "influence": float(col[j]), "rating": float(r[j])}
              for j in order if col[j] > 0]
        rows.append({"userId": int(rec["userId"]), "movieId": int(rec["movieId"]), "rank": int(rec["rank"]),
                     "score": float(s_full[c]), "n_history": int(len(h_ids)),
                     "total_pos_influence": float(col[pos].sum()), "evidence": ev})
    return pd.DataFrame(rows, columns=[f.name for f in EVIDENCE_SCHEMA.fields])


def sample_users(recs: DataFrame, n: int, seed: int) -> DataFrame:
    users = recs.select("userId").distinct()
    total = users.count()
    frac = min(1.0, 1.5 * n / max(total, 1))
    return users.sample(fraction=frac, seed=seed).orderBy(F.rand(seed)).limit(n).cache()


def extract(spark: SparkSession, train: DataFrame, recs: DataFrame, fm: FactorModel, top_n: int) -> DataFrame:
    bc = spark.sparkContext.broadcast((fm, fm.solver()))

    def fn(key, hist: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
        if rec.empty or hist.empty:
            return pd.DataFrame(columns=[f.name for f in EVIDENCE_SCHEMA.fields])
        f, s = bc.value
        return user_evidence(hist, rec.sort_values("rank"), f, s, top_n)

    return (train.select("userId", "movieId", "rating").groupBy("userId")
            .cogroup(recs.select("userId", "movieId", "rank").groupBy("userId"))
            .applyInPandas(fn, EVIDENCE_SCHEMA))


def run(cfg, spark) -> dict:
    e = cfg["evidence"]
    fm = load_factor_model(cfg.path("als_model"))
    train = read_parquet(spark, cfg.path("ratings_train.parquet"))
    recs = read_parquet(spark, cfg.path("recs.parquet")).filter(F.col("rank") <= e["recs_per_user"])
    users = sample_users(recs, e["n_users"], e["seed"])
    t0 = time.time()
    ev = extract(spark, train.join(users, "userId", "left_semi"), recs.join(users, "userId", "left_semi"),
                 fm, e["top_n"]).cache()
    n = ev.count()
    secs = time.time() - t0
    write_parquet(ev, cfg.path("evidence.parquet"))
    stats = ev.select(
        F.count("*").alias("n_recs"),
        F.countDistinct("userId").alias("n_users"),
        F.avg("n_history").alias("mean_history"),
        F.avg(F.size("evidence")).alias("mean_evidence_items"),
        F.avg((F.size("evidence") > 0).cast("double")).alias("frac_with_evidence"),
        F.avg(F.col("evidence")[0]["influence"] / F.col("score")).alias("mean_top1_share_of_score"),
        F.avg(F.col("evidence")[0]["influence"] / F.col("total_pos_influence")).alias("mean_top1_share_of_pos_infl"),
    ).first().asDict()
    stats.update({"seconds": round(secs, 1), "model": fm.meta})
    with open(cfg.path("evidence_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)
    log.info("evidence for %d recs in %.1fs: %s", n, secs, stats)
    return stats


if __name__ == "__main__":
    cfg, _ = cli(__doc__)
    spark = get_spark(cfg, "i-recs-evidence")
    run(cfg, spark)
    spark.stop()
