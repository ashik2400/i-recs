"""Stage 4: grounding + hallucination checks on explanations.

python -m src.explain.check

Checks every explanation against the evidence its writer was given (top
`explain.max_cited` items) with the rules in src/explain/grounding.py.
Also runs a stress test: known hallucinations are injected into real
explanations and we measure how many the checker catches (recall) and how often
clean text is flagged (false-positive rate).

Writes grounding.parquet:
  userId, movieId, grounded, unsupported_claims: array<string>, cited_present,
  rec_present, n_words, flesch, llm_name
and grounding_summary.json.
"""
from __future__ import annotations

import json
import logging
import random

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common.config import cli
from src.common.io import read_parquet, write_parquet
from src.common.spark import get_spark
from src.explain.grounding import GENRES, TitleIndex, check, display_title

log = logging.getLogger("grounding")

GROUNDING_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType()), T.StructField("movieId", T.IntegerType()),
    T.StructField("grounded", T.BooleanType()), T.StructField("unsupported_claims", T.ArrayType(T.StringType())),
    T.StructField("cited_present", T.BooleanType()), T.StructField("rec_present", T.BooleanType()),
    T.StructField("n_words", T.IntegerType()), T.StructField("flesch", T.DoubleType()),
    T.StructField("llm_name", T.StringType()),
])

INJECTIONS = {
    "other_title": lambda ctx, rng: f" Fans of {display_title(ctx['other_title'])} also enjoy it.",
    "wrong_year": lambda ctx, rng: f" It first came out in {ctx['wrong_year']}.",
    "wrong_genre": lambda ctx, rng: f" It is also a {ctx['wrong_genre'].lower()} story.",
    "outside_reason": lambda ctx, rng: rng.choice([" It won several awards.", " It is very popular right now.",
                                                   " Critics loved it.", " It is from the same director."]),
    "wrong_rating": lambda ctx, rng: f" You gave {ctx['first_title']} {ctx['wrong_rating']}/5.",
}


def _rows(pdf: pd.DataFrame, mv: dict, max_cited: int):
    for r in pdf.itertuples(index=False):
        t, g = mv.get(int(r.movieId), ("", ""))
        rec = {"movieId": int(r.movieId), "title": t, "genres": g}
        ev = [{"movieId": int(x["movieId"]), "title": mv.get(int(x["movieId"]), ("", ""))[0],
               "genres": mv.get(int(x["movieId"]), ("", ""))[1], "rating": float(x["rating"])}
              for x in (list(r.evidence)[:max_cited] if r.evidence is not None else [])]
        yield r, rec, ev


def run(cfg, spark) -> dict:
    max_cited = int(cfg["explain"]["max_cited"])
    ex = read_parquet(spark, cfg.path("explanations.parquet"))
    ev = read_parquet(spark, cfg.path("evidence.parquet")).select("userId", "movieId", "evidence")
    movies = read_parquet(spark, cfg.path("movies.parquet")).toPandas()
    bc = spark.sparkContext.broadcast({int(r.movieId): (r.title, r.genres) for r in movies.itertuples()})
    joined = ex.join(ev, ["userId", "movieId"], "inner")

    def fn(batches):
        mv = bc.value
        index = TitleIndex.build([(k, v[0]) for k, v in mv.items()])
        for pdf in batches:
            out = []
            for r, rec, evid in _rows(pdf, mv, max_cited):
                g = check(r.text, rec, evid, list(r.cited_items), index)
                out.append({"userId": int(r.userId), "movieId": int(r.movieId), "grounded": g.grounded,
                            "unsupported_claims": g.unsupported_claims, "cited_present": g.cited_present,
                            "rec_present": g.rec_present, "n_words": g.n_words, "flesch": float(g.flesch),
                            "llm_name": r.llm_name})
            yield pd.DataFrame(out, columns=[f.name for f in GROUNDING_SCHEMA.fields])

    gr = joined.mapInPandas(fn, GROUNDING_SCHEMA).cache()
    write_parquet(gr, cfg.path("grounding.parquet"))
    by_writer = {r["llm_name"]: {"n": r["n"], "grounded_rate": r["g"], "mean_words": r["w"], "mean_flesch": r["f"]}
                 for r in gr.groupBy("llm_name").agg(F.count("*").alias("n"), F.avg(F.col("grounded").cast("double")).alias("g"),
                                                       F.avg("n_words").alias("w"), F.avg("flesch").alias("f")).collect()}
    claims = (gr.select(F.explode("unsupported_claims").alias("c"))
                .select(F.regexp_replace("c", r"[:'].*$", "").alias("type")).groupBy("type").count()
                .orderBy(F.desc("count")).limit(10).toPandas())
    stress = stress_test(joined.limit(int(cfg["evidence"].get("stress_n", 500))).toPandas(), bc.value, max_cited)
    summary = {"by_writer": by_writer, "top_claim_types": dict(zip(claims["type"], claims["count"].astype(int))),
               "stress_test": stress}
    with open(cfg.path("grounding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info("grounding: %s", json.dumps(summary, default=float))
    return summary


def stress_test(pdf: pd.DataFrame, mv: dict, max_cited: int, seed: int = 0) -> dict:
    """Inject one known hallucination per (explanation, type); report detection recall + clean FP rate."""
    rng = random.Random(seed)
    index = TitleIndex.build([(k, v[0]) for k, v in mv.items()])
    multiword = [t for t in (v[0] for v in mv.values()) if len(t.split()) >= 4]
    caught = {k: 0 for k in INJECTIONS}
    clean_fp, n = 0, 0
    for r, rec, ev in _rows(pdf, mv, max_cited):
        if not ev:
            continue
        n += 1
        cited = list(r.cited_items)
        clean_fp += int(not check(r.text, rec, ev, cited, index).grounded)
        allowed_genres = set((rec["genres"] or "").split("|")).union(*[set(e["genres"].split("|")) for e in ev])
        ctx = {
            "other_title": rng.choice([t for t in multiword if t not in {e["title"] for e in ev} | {rec["title"]}]),
            "wrong_year": rng.choice([y for y in range(1930, 2020) if str(y) not in r.text and
                                      all(str(y) not in e["title"] for e in ev + [rec])]),
            "wrong_genre": rng.choice([g for g in GENRES if g not in allowed_genres and g != "IMAX"]),
            "first_title": display_title(ev[0]["title"]),
            "wrong_rating": rng.choice([x for x in (1, 2, 3) if float(x) not in {e["rating"] for e in ev}] or [0.5]),
        }
        for k, inj in INJECTIONS.items():
            caught[k] += int(not check(r.text + inj(ctx, rng), rec, ev, cited, index).grounded)
    return {"n": n, "clean_false_positive_rate": clean_fp / max(n, 1),
            "detection_recall": {k: v / max(n, 1) for k, v in caught.items()}}


if __name__ == "__main__":
    cfg, _ = cli(__doc__)
    spark = get_spark(cfg, "i-recs-grounding")
    run(cfg, spark)
    spark.stop()
