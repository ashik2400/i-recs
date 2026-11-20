"""Stage 5: faithfulness verification (the core contribution).

python -m src.verify.faithfulness [--set verify.rel_drop_threshold=0.05]

An explanation claims "these cited past ratings are why you got this movie".
We test the claim counterfactually: delete ALL cited items from the user's
history, recompute the user vector by fold-in (item factors fixed, ALS is NOT
retrained), and re-score / re-rank the recommended movie among the same
candidate set (items the user had not seen).

  rel_drop = (score_before - score_after) / |score_before|
  faithful = rel_drop >= verify.rel_drop_threshold
             OR (verify.use_rank_criterion AND rank_after > als.top_k)

Random-removal baseline: for the same (user, movie), delete the same number of
items drawn uniformly from the user's non-cited history (verify.random_trials
seeded draws). The faithfulness score is reported next to this baseline, with a
paired test and a threshold sweep.

Writes faithfulness.parquet (one row per explanation x condition x trial):
  userId, movieId, condition ('cited'|'random'), trial, cited_item: array<int>
  (the removed items), score_before, score_after, rel_drop, rank_before,
  rank_after, faithful
and faithfulness_summary.json, faithfulness_sweep.csv.
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

log = logging.getLogger("faithfulness")

FAITH_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType()), T.StructField("movieId", T.IntegerType()),
    T.StructField("condition", T.StringType()), T.StructField("trial", T.IntegerType()),
    T.StructField("cited_item", T.ArrayType(T.IntegerType())),
    T.StructField("score_before", T.DoubleType()), T.StructField("score_after", T.DoubleType()),
    T.StructField("rel_drop", T.DoubleType()),
    T.StructField("rank_before", T.IntegerType()), T.StructField("rank_after", T.IntegerType()),
    T.StructField("faithful", T.BooleanType()),
    T.StructField("n_history", T.IntegerType()),
])
COLS = [f.name for f in FAITH_SCHEMA.fields]


def rel_drop(before: float, after: float) -> float:
    return (before - after) / max(abs(before), 1e-12)


def is_faithful(before: float, after: float, rank_after: int, top_k: int, threshold: float, use_rank: bool) -> bool:
    # tiny tolerance so a drop of exactly `threshold` counts despite float rounding
    return rel_drop(before, after) >= threshold - 1e-9 or (use_rank and rank_after > top_k)


def random_removal(history_ids: np.ndarray, cited: set, size: int, seed: int, user: int, movie: int, trial: int):
    """Deterministic draw of `size` non-cited history items (None if not enough)."""
    pool = np.array([m for m in history_ids if int(m) not in cited])
    if size == 0 or len(pool) < size:
        return None
    rng = np.random.default_rng([seed, int(user), int(movie), trial])
    return rng.choice(pool, size=size, replace=False)


def verify_user(hist: pd.DataFrame, expl: pd.DataFrame, fm: FactorModel, solver, top_k: int,
                threshold: float, use_rank: bool, trials: int, seed: int) -> pd.DataFrame:
    """All conditions for one user's explanations (pure numpy; unit-testable)."""
    hist = hist[fm.idx(hist["movieId"]) >= 0]
    h_ids = hist["movieId"].to_numpy(np.int64)
    idx = fm.idx(h_ids)
    r = hist["rating"].to_numpy(np.float64)
    Y = fm.item_factors
    candidates = np.ones(len(Y), dtype=bool)
    candidates[idx] = False                                     # unseen items, fixed for all conditions
    s_full = Y @ solver.user_vector(idx, r)
    cache: dict[tuple, np.ndarray] = {}

    def scores_without(removed: np.ndarray) -> np.ndarray:
        key = tuple(sorted(int(m) for m in removed))
        if key not in cache:
            cache[key] = Y @ solver.user_vector_without(idx, r, np.isin(h_ids, removed))
        return cache[key]

    rows = []
    for e in expl.itertuples(index=False):
        t = fm.index.get(int(e.movieId))
        if t is None or not candidates[t]:
            continue
        cited = np.array([int(c) for c in (e.cited_items if e.cited_items is not None else []) if int(c) in set(h_ids.tolist())],
                         dtype=np.int64)
        if len(cited) == 0:
            continue
        before, rank_before = float(s_full[t]), int((s_full[candidates] > s_full[t]).sum()) + 1
        conds = [("cited", 0, cited)]
        for k in range(trials):
            rnd = random_removal(h_ids, set(cited.tolist()), len(cited), seed, e.userId, e.movieId, k)
            if rnd is not None:
                conds.append(("random", k, rnd))
        for cond, trial, removed in conds:
            s = scores_without(removed)
            after = float(s[t])
            rank_after = int((s[candidates] > s[t]).sum()) + 1
            rows.append({"userId": int(e.userId), "movieId": int(e.movieId), "condition": cond, "trial": trial,
                         "cited_item": [int(m) for m in removed], "score_before": before, "score_after": after,
                         "rel_drop": rel_drop(before, after), "rank_before": rank_before, "rank_after": rank_after,
                         "faithful": is_faithful(before, after, rank_after, top_k, threshold, use_rank),
                         "n_history": int(len(h_ids))})
    return pd.DataFrame(rows, columns=COLS)


def verify(spark: SparkSession, train: DataFrame, expl: DataFrame, fm: FactorModel, top_k: int,
           threshold: float, use_rank: bool, trials: int, seed: int) -> DataFrame:
    bc = spark.sparkContext.broadcast((fm, fm.solver()))

    def fn(key, hist: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
        if hist.empty or ex.empty:
            return pd.DataFrame(columns=COLS)
        f, s = bc.value
        return verify_user(hist, ex, f, s, top_k, threshold, use_rank, trials, seed)

    users = expl.select("userId").distinct()
    return (train.join(users, "userId", "left_semi").select("userId", "movieId", "rating").groupBy("userId")
            .cogroup(expl.select("userId", "movieId", "cited_items").groupBy("userId"))
            .applyInPandas(fn, FAITH_SCHEMA))


def summarize(fa: pd.DataFrame, top_k: int, threshold: float, use_rank: bool,
              recs: pd.DataFrame | None = None, grounding: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    from scipy.stats import wilcoxon

    cited = fa[fa.condition == "cited"]
    rnd = fa[fa.condition == "random"]
    out_of_k = lambda d: float((d.rank_after > top_k).mean())  # noqa: E731
    summary = {
        "n_explanations": int(len(cited)), "n_random_trials": int(len(rnd)),
        "threshold": threshold, "use_rank_criterion": use_rank, "top_k": top_k,
        "faithfulness_score": float(cited.faithful.mean()),
        "random_baseline_score": float(rnd.faithful.mean()),
        "mean_rel_drop": {"cited": float(cited.rel_drop.mean()), "random": float(rnd.rel_drop.mean())},
        "median_rel_drop": {"cited": float(cited.rel_drop.median()), "random": float(rnd.rel_drop.median())},
        "frac_out_of_top_k": {"cited": out_of_k(cited), "random": out_of_k(rnd)},
        "mean_rank_after": {"cited": float(cited.rank_after.mean()), "random": float(rnd.rank_after.mean())},
    }
    # paired: each explanation's cited drop vs the mean drop of its random trials
    paired = cited.set_index(["userId", "movieId"])["rel_drop"].to_frame("cited").join(
        rnd.groupby(["userId", "movieId"])["rel_drop"].mean().to_frame("random"), how="inner")
    if len(paired) > 0:
        diff = paired["cited"] - paired["random"]
        summary["paired"] = {"n": int(len(paired)), "cited_beats_random_rate": float((diff > 0).mean()),
                             "mean_difference": float(diff.mean()),
                             "wilcoxon_p": float(wilcoxon(paired["cited"], paired["random"], alternative="greater").pvalue)
                             if len(paired) > 10 else None}
    bins = [0, 20, 50, 100, 200, 500, 10 ** 9]
    labels = ["<20", "20-49", "50-99", "100-199", "200-499", "500+"]
    by_hist = {}
    for name, d in (("cited", cited), ("random", rnd)):
        g = d.groupby(pd.cut(d.n_history, bins, right=False, labels=labels), observed=True)
        by_hist[name] = {str(b): {"n": int(len(x)), "faithful": float(x.faithful.mean()), "mean_rel_drop": float(x.rel_drop.mean())}
                         for b, x in g}
    summary["by_history_length"] = by_hist
    if recs is not None:
        m = cited.merge(recs[["userId", "movieId", "rank"]], on=["userId", "movieId"])
        summary["rank_consistency_with_als_recs"] = float((m["rank"] == m["rank_before"]).mean()) if len(m) else None
    if grounding is not None:
        g = cited.merge(grounding[["userId", "movieId", "grounded"]], on=["userId", "movieId"])
        summary["faithfulness_grounded_only"] = float(g[g.grounded].faithful.mean()) if g.grounded.any() else None
        summary["grounded_and_faithful_rate"] = float((g.grounded & g.faithful).mean()) if len(g) else None
    sweep = []
    for th in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for name, d in (("cited", cited), ("random", rnd)):
            score_only = (d.rel_drop >= th).mean()
            with_rank = ((d.rel_drop >= th) | (d.rank_after > top_k)).mean()
            sweep.append({"threshold": th, "condition": name, "faithful_score_only": float(score_only),
                          "faithful_score_or_rank": float(with_rank)})
    return summary, pd.DataFrame(sweep)


def run(cfg, spark) -> dict:
    v, k = cfg["verify"], int(cfg["als"]["top_k"])
    fm = load_factor_model(cfg.path("als_model"))
    train = read_parquet(spark, cfg.path("ratings_train.parquet"))
    expl = read_parquet(spark, cfg.path("explanations.parquet"))
    t0 = time.time()
    fa = verify(spark, train, expl, fm, k, float(v["rel_drop_threshold"]), bool(v["use_rank_criterion"]),
                int(v["random_trials"]), int(v["seed"])).cache()
    n = fa.count()
    secs = time.time() - t0
    write_parquet(fa, cfg.path("faithfulness.parquet"))
    recs = read_parquet(spark, cfg.path("recs.parquet")).join(expl.select("userId", "movieId"), ["userId", "movieId"]).toPandas()
    try:
        grounding = read_parquet(spark, cfg.path("grounding.parquet")).select("userId", "movieId", "grounded").toPandas()
    except Exception:
        grounding = None
    summary, sweep = summarize(fa.toPandas(), k, float(v["rel_drop_threshold"]), bool(v["use_rank_criterion"]),
                               recs, grounding)
    summary["seconds"] = round(secs, 1)
    summary["rows"] = n
    with open(cfg.path("faithfulness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    sweep.to_csv(cfg.path("faithfulness_sweep.csv"), index=False)
    log.info("faithfulness: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    cfg, _ = cli(__doc__)
    spark = get_spark(cfg, "i-recs-faithfulness")
    run(cfg, spark)
    spark.stop()
