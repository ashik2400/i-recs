"""Stage 1a: load + clean + time-based split.

python -m src.als.prepare [--set dataset=ml-1m]
Writes ratings_train.parquet, ratings_test.parquet, movies.parquet.
"""
from __future__ import annotations

import json
import logging

from pyspark.sql import functions as F

from src.als.data import clean, clean_movies, load_raw, time_split
from src.common.config import cli
from src.common.io import write_parquet
from src.common.spark import get_spark

log = logging.getLogger("prepare")


def run(cfg, spark) -> dict:
    ratings, movies = load_raw(spark, cfg["raw_dir"], cfg["dataset"], cfg.get("max_ratings"))
    ratings = clean(ratings).cache()
    s = cfg["split"]
    train, test = time_split(ratings, s["test_fraction"], s["strategy"], s["min_user_train"])
    write_parquet(train, cfg.path("ratings_train.parquet"))
    write_parquet(test, cfg.path("ratings_test.parquet"))
    write_parquet(clean_movies(movies), cfg.path("movies.parquet"))
    stats = {
        "dataset": cfg["dataset"], "split": s["strategy"],
        "ratings_clean": ratings.count(),
        "train": train.count(), "test": test.count(),
        "users_train": train.select("userId").distinct().count(),
        "users_test": test.select("userId").distinct().count(),
        "items_train": train.select("movieId").distinct().count(),
        "test_relevant": test.filter(F.col("rating") >= cfg["als"]["relevance_threshold"]).count(),
    }
    with open(cfg.path("prepare_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log.info("split stats: %s", stats)
    return stats


if __name__ == "__main__":
    cfg, _ = cli(__doc__)
    spark = get_spark(cfg, "i-recs-prepare")
    run(cfg, spark)
    spark.stop()
