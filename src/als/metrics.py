"""Ranking metrics computed with Spark DataFrames."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def ranking_metrics(recs: DataFrame, test: DataFrame, k: int, relevance_threshold: float) -> dict:
    """precision@k, recall@k, MRR@k, NDCG@k averaged over users with >=1 relevant test item.

    recs: userId, movieId, rank (1-based).  test: userId, movieId, rating.
    """
    rel = test.filter(F.col("rating") >= relevance_threshold).select("userId", "movieId").distinct()
    n_rel = rel.groupBy("userId").agg(F.count("*").alias("n_rel"))
    topk = recs.filter(F.col("rank") <= k).select("userId", "movieId", "rank")
    hits = topk.join(rel, ["userId", "movieId"], "inner")
    per_hit = hits.groupBy("userId").agg(
        F.count("*").alias("hits"),
        F.min("rank").alias("first_rank"),
        F.sum(1.0 / F.log2(F.col("rank") + 1)).alias("dcg"),
    )
    # users with relevant items and at least one recommendation
    users = n_rel.join(topk.select("userId").distinct(), "userId", "inner")
    u = users.join(per_hit, "userId", "left").fillna({"hits": 0, "dcg": 0.0})
    idcg = F.aggregate(
        F.sequence(F.lit(1), F.least(F.col("n_rel"), F.lit(k))),
        F.lit(0.0),
        lambda acc, i: acc + 1.0 / F.log2(i.cast("double") + 1),
    )
    u = u.select(
        (F.col("hits") / k).alias("p"),
        (F.col("hits") / F.col("n_rel")).alias("r"),
        F.when(F.col("first_rank").isNotNull(), 1.0 / F.col("first_rank")).otherwise(0.0).alias("rr"),
        (F.col("dcg") / idcg).alias("ndcg"),
    )
    row = u.agg(F.avg("p"), F.avg("r"), F.avg("rr"), F.avg("ndcg"), F.count("*")).first()
    return {
        f"precision@{k}": float(row[0] or 0), f"recall@{k}": float(row[1] or 0),
        f"mrr@{k}": float(row[2] or 0), f"ndcg@{k}": float(row[3] or 0), "n_eval_users": int(row[4]),
    }
