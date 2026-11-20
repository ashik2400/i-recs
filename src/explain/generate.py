"""Stage 3: evidence-only explanations.

python -m src.explain.generate        (LLM settings from .env; template if none)

For each (user, recommended movie) in evidence.parquet the writer receives ONLY
the top `explain.max_cited` evidence items (title, the user's rating, shared
genres). With a real LLM, the output is grounding-checked; if it fails it is
regenerated (`explain.retries` times) and finally replaced by the deterministic
template. Only the first `explain.max_llm_calls` rows (each sampled user's top
recommendation first) go to the LLM; the rest use the template.

Writes explanations.parquet:
  userId, movieId, text, cited_items: array<int>, llm_name, prompt_version, attempts, fell_back
"""
from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common.config import cli
from src.common.io import read_parquet, write_parquet
from src.common.spark import get_spark
from src.explain.grounding import TitleIndex, check, norm_text, norm_title
from src.explain.llm_client import LLMClient
from src.explain.prompt import PROMPT_VERSION, SYSTEM_PROMPT, evidence_payload, template_explanation, user_prompt

log = logging.getLogger("explain")

EXPLANATION_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType()), T.StructField("movieId", T.IntegerType()),
    T.StructField("text", T.StringType()), T.StructField("cited_items", T.ArrayType(T.IntegerType())),
    T.StructField("llm_name", T.StringType()), T.StructField("prompt_version", T.StringType()),
    T.StructField("attempts", T.IntegerType()), T.StructField("fell_back", T.BooleanType()),
])


def explain_one(rec: dict, evidence: list[dict], max_cited: int, llm: LLMClient | None,
                retries: int, title_index: TitleIndex | None) -> dict:
    """rec/evidence items carry movieId, title, genres (+ rating for evidence)."""
    cited = evidence[:max_cited]
    payload = evidence_payload(rec, cited)
    attempts = 0
    if llm is not None and not llm.is_template and cited:
        for _ in range(1 + retries):
            attempts += 1
            text = llm.complete(SYSTEM_PROMPT, user_prompt(payload))
            if not text:
                break
            nt = norm_text(text)
            cited_ids = [int(c["movieId"]) for c in cited if norm_title(c["title"]) in nt]
            if check(text, rec, cited, cited_ids, title_index).grounded:
                return {"text": text, "cited_items": cited_ids, "llm_name": llm.name,
                        "attempts": attempts, "fell_back": False}
    return {"text": template_explanation(payload), "cited_items": [int(c["movieId"]) for c in cited],
            "llm_name": "template", "attempts": attempts, "fell_back": attempts > 0}


def run(cfg, spark) -> dict:
    e = cfg["explain"]
    llm = LLMClient()
    llm_conf = (llm.provider, llm.model, llm.api_key, llm.base_url)
    log.info("writer: %s (max %d LLM rows)", llm.name, e["max_llm_calls"])

    ev = read_parquet(spark, cfg.path("evidence.parquet"))
    movies = read_parquet(spark, cfg.path("movies.parquet")).toPandas()
    bc_movies = spark.sparkContext.broadcast(
        {int(r.movieId): (r.title, r.genres) for r in movies.itertuples()})

    # flag the rows that go to the real LLM: top-1 rec of each user first
    w = Window.orderBy("rank", "userId", "movieId")
    ev = ev.withColumn("use_llm", F.row_number().over(w) <= int(e["max_llm_calls"]) if not llm.is_template else F.lit(False))
    max_cited, retries, pv = int(e["max_cited"]), int(e["retries"]), PROMPT_VERSION

    def gen(batches):
        mv = bc_movies.value
        client = LLMClient(*llm_conf)
        tindex = TitleIndex.build([(k, v[0]) for k, v in mv.items()]) if not client.is_template else None
        for pdf in batches:
            out = []
            for r in pdf.itertuples(index=False):
                t, g = mv.get(int(r.movieId), ("", ""))
                rec = {"movieId": int(r.movieId), "title": t, "genres": g}
                evid = [{"movieId": int(x["movieId"]), "title": mv.get(int(x["movieId"]), ("", ""))[0],
                         "genres": mv.get(int(x["movieId"]), ("", ""))[1], "rating": float(x["rating"])}
                        for x in (r.evidence if r.evidence is not None else [])]
                res = explain_one(rec, evid, max_cited, client if r.use_llm else None, retries, tindex)
                out.append({"userId": int(r.userId), "movieId": int(r.movieId), "prompt_version": pv, **res})
            yield pd.DataFrame(out, columns=[f.name for f in EXPLANATION_SCHEMA.fields])

    ex = ev.select("userId", "movieId", "evidence", "use_llm").mapInPandas(gen, EXPLANATION_SCHEMA).cache()
    write_parquet(ex, cfg.path("explanations.parquet"))
    summary = {r["llm_name"]: r["count"] for r in ex.groupBy("llm_name").count().collect()}
    summary["fell_back"] = ex.filter("fell_back").count()
    log.info("explanations by writer: %s", summary)
    for r in ex.orderBy("userId", "movieId").limit(3).collect():
        log.info("example u%d m%d: %s", r.userId, r.movieId, r.text)
    return summary


if __name__ == "__main__":
    cfg, _ = cli(__doc__)
    spark = get_spark(cfg, "i-recs-explain")
    run(cfg, spark)
    spark.stop()
