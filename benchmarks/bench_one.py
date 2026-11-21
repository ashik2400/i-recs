"""One benchmark run: one dataset x one Spark master. Prints a JSON line.

python -m benchmarks.bench_one --dataset ml-1m --master "local[2]" [--max-ratings N]

Times the Spark stages of the pipeline on fresh data:
  split     clean + per-user temporal split (materialised)
  als_fit   ALS training (fixed rank / maxIter, implicit)
  recs      top-K unseen recommendations for the sampled users
  evidence  leave-one-out fold-in attribution for the sampled users' top-K
  verify    faithfulness (cited top-2 removal + random baseline) for the same recs
Loading the raw file into Spark is excluded (on Windows without winutils it goes
through pandas, which is not what we want to measure).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from src.als.data import RATING_SCHEMA, clean, load_raw, time_split
from src.als.factors import FactorModel
from src.als.train import factors_np, fit_als, top_k_unseen, user_vecs
from src.common.config import ROOT, load_config, setup_logging
from src.common.spark import get_spark
from src.evidence.extract import extract
from src.verify.faithfulness import verify

log = logging.getLogger("bench")


def cached_slice(raw_dir: str, dataset: str, max_ratings: int | None) -> str | None:
    """Earliest `max_ratings` ratings of a big CSV, cached once as Parquet (pandas/pyarrow)."""
    if not max_ratings:
        return None
    out = ROOT / "data" / "bench" / f"{dataset}-{max_ratings}.parquet"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(os.path.join(raw_dir, dataset, "ratings.csv"),
                         dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
        df.sort_values(["timestamp", "userId", "movieId"]).head(int(max_ratings)).to_parquet(out, index=False)
    return str(out)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--max-ratings", type=int, default=None)
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    setup_logging("WARNING")
    cfg = load_config(overrides=a.set)
    b = cfg["bench"]
    cores = int(a.master.strip("local[]")) if a.master != "local[*]" else os.cpu_count()
    ppc = int(b.get("partitions_per_core", 2))
    cfg["spark"]["shuffle_partitions"] = max(1, ppc * cores)
    spark = get_spark(cfg, f"bench-{a.dataset}-{a.master}", master=a.master)

    sliced = cached_slice(cfg["raw_dir"], a.dataset, a.max_ratings)
    if sliced:
        raw = spark.createDataFrame(pd.read_parquet(sliced), RATING_SCHEMA)
    else:
        raw, _ = load_raw(spark, cfg["raw_dir"], a.dataset)
    raw = raw.repartition(max(1, ppc * cores)).cache()
    n = raw.count()                                          # load excluded from timing

    def do_split():
        tr, te = time_split(clean(raw), cfg["split"]["test_fraction"], "per_user", cfg["split"]["min_user_train"])
        tr, te = tr.cache(), te.cache()
        tr.count(), te.count()
        return tr, te
    (train, test), t_split = timed(do_split)

    def do_fit():
        m = fit_als(train, int(b["rank"]), 0.05, int(b["max_iter"]), 42, implicit=True, alpha=1.0)
        m.userFactors.count()
        m.itemFactors.count()
        return m
    model, t_fit = timed(do_fit)

    ids, Y = factors_np(model)
    fm = FactorModel(ids, Y.astype(np.float64), {"rank": model.rank, "regParam": 0.05, "alpha": 1.0, "implicitPrefs": True})
    users = train.select("userId").distinct().orderBy(F.rand(1)).limit(int(b["evidence_users"])).cache()
    users.count()
    hist = train.join(users, "userId", "left_semi").cache()
    hist.count()

    recs, t_recs = timed(lambda: top_k_unseen(spark, hist, user_vecs(model), ids, Y, 10, users=users).cache())
    _, t_recs2 = timed(lambda: recs.count())
    ev, t_ev = timed(lambda: extract(spark, hist, recs, fm, 5).cache())
    _, t_ev2 = timed(lambda: ev.count())
    expl = ev.select("userId", "movieId", F.transform(F.slice("evidence", 1, 2), lambda e: e["movieId"]).alias("cited_items"))
    fa, t_v = timed(lambda: verify(spark, hist, expl, fm, 10, 0.1, True, 3, 7).cache())
    n_faith, t_v2 = timed(lambda: fa.count())
    rate = fa.filter("condition = 'cited'").agg(F.avg(F.col("faithful").cast("double"))).first()[0]

    print("BENCH " + json.dumps({
        "dataset": a.dataset, "max_ratings": a.max_ratings, "master": a.master, "cores": cores,
        "n_ratings": n, "n_train": train.count(), "n_users": train.select("userId").distinct().count(),
        "n_items": len(ids), "evidence_users": int(b["evidence_users"]), "n_recs": ev.count(),
        "t_split": round(t_split, 2), "t_als_fit": round(t_fit, 2), "t_recs": round(t_recs + t_recs2, 2),
        "t_evidence": round(t_ev + t_ev2, 2), "t_verify": round(t_v + t_v2, 2),
        "faithfulness_score": rate, "verify_rows": n_faith, "partitions_per_core": ppc,
    }), flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
