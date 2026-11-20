"""Stage 6: utility evaluation via an offline proxy (no user survey).

python -m src.eval.utility

Question: do the signals behind our explanations tell us anything about whether
a user will actually accept the recommendation?

  label  = 1 if the recommended movie appears in the user's FUTURE (test-period)
           ratings with rating >= als.relevance_threshold ("accepted"), else 0.
  base   = recommender-only features: ALS score, rank, log item popularity,
           log user history size.
  +expl  = base + explanation features: top-1 influence share of the score,
           top-N evidence share, mean/max rating of cited evidence, relative
           score drop when the cited items are removed (M5), faithful flag.

A Spark MLlib logistic regression is trained on users in a grouped split (no
user in both train and test), repeated over several seeds. We report ROC-AUC
and PR-AUC for base vs base+expl, plus acceptance rate of faithful vs
unfaithful recommendations.

Caveat: this is an observational proxy. A gain in AUC means explanation-derived
signals carry information about acceptance; it does NOT measure the causal
effect of showing an explanation to a person (that needs a user study).
"""
from __future__ import annotations

import json
import logging

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common.config import cli
from src.common.io import read_parquet
from src.common.spark import get_spark

log = logging.getLogger("utility")

BASE = ["als_score", "rank", "log_item_pop", "log_n_history"]
EXPL = ["top1_share", "topn_share", "cited_mean_rating", "cited_max_rating", "cited_rel_drop", "faithful_f"]


def build_features(spark, cfg) -> DataFrame:
    thr = float(cfg["als"]["relevance_threshold"])
    max_cited = int(cfg["explain"]["max_cited"])
    ev = read_parquet(spark, cfg.path("evidence.parquet"))
    fa = (read_parquet(spark, cfg.path("faithfulness.parquet")).filter("condition = 'cited'")
          .select("userId", "movieId", F.col("rel_drop").alias("cited_rel_drop"), "faithful"))
    train = read_parquet(spark, cfg.path("ratings_train.parquet"))
    test = read_parquet(spark, cfg.path("ratings_test.parquet"))
    pop = train.groupBy("movieId").agg(F.log1p(F.count("*")).alias("log_item_pop"))
    accepted = test.filter(F.col("rating") >= thr).select("userId", "movieId").withColumn("label", F.lit(1.0))
    cited = F.slice("evidence", 1, max_cited)
    df = (ev.join(fa, ["userId", "movieId"], "inner")
            .join(pop, "movieId", "left")
            .join(accepted, ["userId", "movieId"], "left")
            .select(
                "userId", "movieId",
                F.coalesce("label", F.lit(0.0)).alias("label"),
                F.col("score").alias("als_score"), F.col("rank").cast("double").alias("rank"),
                F.coalesce("log_item_pop", F.lit(0.0)).alias("log_item_pop"),
                F.log1p("n_history").alias("log_n_history"),
                (F.col("evidence")[0]["influence"] / F.col("score")).alias("top1_share"),
                (F.aggregate("evidence", F.lit(0.0), lambda a, e: a + e["influence"]) / F.col("score")).alias("topn_share"),
                (F.aggregate(cited, F.lit(0.0), lambda a, e: a + e["rating"]) / F.size(cited)).alias("cited_mean_rating"),
                F.array_max(F.transform(cited, lambda e: e["rating"])).cast("double").alias("cited_max_rating"),
                "cited_rel_drop",
                F.col("faithful").cast("double").alias("faithful_f"),
            ).na.fill(0.0))
    return df


def fit_eval(train: DataFrame, test: DataFrame, cols: list[str]) -> dict:
    pipe = Pipeline(stages=[
        VectorAssembler(inputCols=cols, outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
        LogisticRegression(featuresCol="features", labelCol="label", regParam=0.01, maxIter=100),
    ])
    pred = pipe.fit(train).transform(test)
    roc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
    pr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR").evaluate(pred)
    return {"roc_auc": roc, "pr_auc": pr}


def run(cfg, spark) -> dict:
    u = cfg["utility"]
    df = build_features(spark, cfg).cache()
    n, pos = df.count(), int(df.agg(F.sum("label")).first()[0])
    log.info("utility dataset: %d recs, %d accepted (%.3f)", n, pos, pos / max(n, 1))
    feature_sets = {"base": BASE, "base+expl": BASE + EXPL, "expl_only": EXPL}
    runs = {k: [] for k in feature_sets}
    for i in range(int(u.get("repeats", 5))):
        seed = int(u["seed"]) + i
        split = df.withColumn("_h", F.abs(F.hash("userId", F.lit(seed))) % 1000 / 1000.0)
        tr, te = split.filter(F.col("_h") >= u["test_fraction"]), split.filter(F.col("_h") < u["test_fraction"])
        if te.agg(F.sum("label")).first()[0] in (None, 0):
            continue
        for name, cols in feature_sets.items():
            runs[name].append(fit_eval(tr, te, cols))
    results = {name: {m: {"mean": float(np.mean([r[m] for r in rs])), "std": float(np.std([r[m] for r in rs]))}
                      for m in ("roc_auc", "pr_auc")} | {"repeats": len(rs)} for name, rs in runs.items()}
    d_roc = [b["roc_auc"] - a["roc_auc"] for a, b in zip(runs["base"], runs["base+expl"])]
    d_pr = [b["pr_auc"] - a["pr_auc"] for a, b in zip(runs["base"], runs["base+expl"])]
    results["paired_gain_base+expl_vs_base"] = {
        "roc_auc_per_split": [round(x, 4) for x in d_roc], "pr_auc_per_split": [round(x, 4) for x in d_pr],
        "roc_auc_mean": float(np.mean(d_roc)), "splits_improved": int(sum(x > 0 for x in d_roc))}
    acc = {("faithful" if r["faithful_f"] == 1.0 else "not_faithful"): {"n": r["n"], "acceptance_rate": r["rate"]}
           for r in df.groupBy("faithful_f").agg(F.count("*").alias("n"), F.avg("label").alias("rate")).collect()}
    # coefficients of a full-data model for interpretation (standardised features)
    pipe = Pipeline(stages=[VectorAssembler(inputCols=BASE + EXPL, outputCol="raw"),
                            StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
                            LogisticRegression(featuresCol="features", labelCol="label", regParam=0.01, maxIter=100)])
    lr = pipe.fit(df).stages[-1]
    coefs = dict(zip(BASE + EXPL, [round(float(c), 4) for c in lr.coefficients]))
    # acceptance by faithfulness within history-size strata (controls the main confounder)
    strata = (df.withColumn("hist_bucket", F.when(F.col("log_n_history") < np.log1p(50), "<50").otherwise(">=50"))
                .groupBy("hist_bucket", "faithful_f").agg(F.count("*").alias("n"), F.avg("label").alias("rate")).collect())
    acc_strat = {f"{r['hist_bucket']}|{'faithful' if r['faithful_f'] == 1.0 else 'not_faithful'}":
                 {"n": r["n"], "acceptance_rate": r["rate"]} for r in strata}
    out = {"n": n, "positives": pos, "base_rate": pos / max(n, 1), "auc": results,
           "acceptance_by_faithfulness": acc,
           "acceptance_by_faithfulness_and_history": acc_strat, "standardised_coefficients": coefs,
           "label": f"recommended movie rated >= {cfg['als']['relevance_threshold']} in the user's test period"}
    with open(cfg.path("utility.json"), "w") as f:
        json.dump(out, f, indent=2)
    log.info("utility: %s", json.dumps(out))
    return out


if __name__ == "__main__":
    cfg, _ = cli(__doc__)
    spark = get_spark(cfg, "i-recs-utility")
    run(cfg, spark)
    spark.stop()
