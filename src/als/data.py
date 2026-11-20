"""Load MovieLens, clean it, and make a time-based train/test split."""
from __future__ import annotations

import logging
import os
import re

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common.spark import hadoop_native_available

log = logging.getLogger(__name__)

RATING_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType(), False),
    T.StructField("movieId", T.IntegerType(), False),
    T.StructField("rating", T.FloatType(), False),
    T.StructField("timestamp", T.LongType(), False),
])
MOVIE_SCHEMA = T.StructType([
    T.StructField("movieId", T.IntegerType(), False),
    T.StructField("title", T.StringType(), True),
    T.StructField("genres", T.StringType(), True),
])


def _raw_files(raw_dir: str, dataset: str) -> tuple[str, str, str]:
    d = os.path.join(raw_dir, dataset)
    if os.path.exists(os.path.join(d, "ratings.dat")):  # ml-1m / ml-10m format
        return os.path.join(d, "ratings.dat"), os.path.join(d, "movies.dat"), "dat"
    return os.path.join(d, "ratings.csv"), os.path.join(d, "movies.csv"), "csv"


def load_raw(spark: SparkSession, raw_dir: str, dataset: str, max_ratings: int | None = None) -> tuple[DataFrame, DataFrame]:
    rpath, mpath, fmt = _raw_files(raw_dir, dataset)
    if not os.path.exists(rpath):
        raise FileNotFoundError(f"{rpath} missing; run: python -m src.als.download --dataset {dataset}")
    if fmt == "csv" and hadoop_native_available():
        ratings = spark.read.csv(rpath, header=True, schema=RATING_SCHEMA)
        movies = spark.read.csv(mpath, header=True, schema=MOVIE_SCHEMA, quote='"', escape='"')
    else:
        # Without Hadoop native I/O (stock Windows) parse with pandas, then hand to Spark.
        if fmt == "dat":
            rp = pd.read_csv(rpath, sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"])
            mp = pd.read_csv(mpath, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="latin-1")
        else:
            rp = pd.read_csv(rpath, dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
            mp = pd.read_csv(mpath, dtype={"movieId": "int32", "title": str, "genres": str})
        rp = rp.astype({"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
        mp = mp.astype({"movieId": "int32"})
        ratings = spark.createDataFrame(rp, RATING_SCHEMA)
        movies = spark.createDataFrame(mp, MOVIE_SCHEMA)
    if max_ratings:
        ratings = ratings.orderBy("timestamp", "userId", "movieId").limit(int(max_ratings))
    return ratings, movies


def clean(ratings: DataFrame) -> DataFrame:
    """Drop nulls / out-of-range ratings and keep the latest rating per (user, movie)."""
    r = ratings.dropna().filter((F.col("rating") >= 0.5) & (F.col("rating") <= 5.0))
    w = Window.partitionBy("userId", "movieId").orderBy(F.col("timestamp").desc())
    return r.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def clean_movies(movies: DataFrame) -> DataFrame:
    year = F.regexp_extract("title", r"\((\d{4})\)\s*$", 1)
    return movies.withColumn("year", F.when(year != "", year.cast("int"))).dropDuplicates(["movieId"])


def time_split(ratings: DataFrame, test_fraction: float, strategy: str = "per_user",
               min_user_train: int = 5) -> tuple[DataFrame, DataFrame]:
    """Time-based split.

    per_user: each user's latest `test_fraction` of ratings go to test (no user is
              split across time differently from others' clocks, but the model never
              sees a user's future).
    global:   one timestamp cutoff at the (1 - test_fraction) quantile.
    Test is restricted to users with >= min_user_train train ratings and to movies
    that appear in train (ALS cannot score unseen items).
    """
    if strategy == "global":
        cutoff = ratings.approxQuantile("timestamp", [1 - test_fraction], 0.001)[0]
        train = ratings.filter(F.col("timestamp") < cutoff)
        test = ratings.filter(F.col("timestamp") >= cutoff)
    elif strategy == "per_user":
        w = Window.partitionBy("userId").orderBy("timestamp", "movieId")
        cnt = Window.partitionBy("userId")
        r = ratings.withColumn("_pos", F.row_number().over(w)).withColumn("_n", F.count("*").over(cnt))
        is_test = F.col("_pos") > F.floor(F.col("_n") * (1 - test_fraction))
        train = r.filter(~is_test).drop("_pos", "_n")
        test = r.filter(is_test).drop("_pos", "_n")
    else:
        raise ValueError(strategy)
    ok_users = train.groupBy("userId").count().filter(F.col("count") >= min_user_train).select("userId")
    train = train.join(ok_users, "userId", "inner")
    test = (test.join(ok_users, "userId", "inner")
                .join(train.select("movieId").distinct(), "movieId", "inner")
                .select("userId", "movieId", "rating", "timestamp"))
    return train.select("userId", "movieId", "rating", "timestamp"), test


def parse_year(title: str | None) -> int | None:
    m = re.search(r"\((\d{4})\)\s*$", title or "")
    return int(m.group(1)) if m else None
