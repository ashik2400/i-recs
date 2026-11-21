"""One-command demo: user -> recommendation -> evidence -> explanation -> grounding -> faithfulness.

python -m scripts.demo                 # a random sampled user
python -m scripts.demo --user 414 --top 3

Runs on the trained model in outputs/<dataset>/ (run the pipeline first). Every
number is computed live by the same functions the Spark stages use (fold-in
evidence, evidence-only explanation, grounding check, counterfactual removal),
so it works for any user, not only the sampled ones. LLM settings come from .env
(template if none).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap

import numpy as np
import pandas as pd

from src.als.factors import load_factor_model
from src.common.config import load_config, setup_logging
from src.common.io import read_pandas
from src.evidence.extract import user_evidence
from src.explain.generate import explain_one
from src.explain.grounding import TitleIndex, check, display_title
from src.explain.llm_client import LLMClient
from src.verify.faithfulness import verify_user

BAR = "─" * 78


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", type=int)
    p.add_argument("--top", type=int, default=3, help="how many recommendations to explain")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):                  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_logging("WARNING")
    cfg = load_config(overrides=a.set)
    k, v, e = int(cfg["als"]["top_k"]), cfg["verify"], cfg["explain"]

    fm = load_factor_model(cfg.path("als_model"))
    solver = fm.solver()
    train = read_pandas(cfg.path("ratings_train.parquet"))
    test = read_pandas(cfg.path("ratings_test.parquet"))
    recs = read_pandas(cfg.path("recs.parquet"))
    movies = read_pandas(cfg.path("movies.parquet")).set_index("movieId")
    mv = lambda m: (movies.title.get(m, "?"), movies.genres.get(m, ""))  # noqa: E731

    user = a.user if a.user is not None else random.Random().choice(sorted(recs.userId.unique()))
    hist = train[train.userId == user]
    if hist.empty:
        raise SystemExit(f"user {user} has no training history")
    urecs = recs[(recs.userId == user) & (recs["rank"] <= a.top)].sort_values("rank")
    llm = LLMClient()
    index = TitleIndex.build([(int(m), t) for m, t in movies.title.items()])

    print(BAR)
    print(f"i-recs demo · dataset {cfg['dataset']} · user {user} · {len(hist)} past ratings · writer: {llm.name}")
    fav = hist.sort_values(["rating", "timestamp"], ascending=False).head(5)
    print("Some of their highest ratings: " + "; ".join(f"{display_title(mv(m)[0])} ({r:g})" for m, r in zip(fav.movieId, fav.rating)))
    future = set(test[(test.userId == user) & (test.rating >= cfg["als"]["relevance_threshold"])].movieId)

    ev = user_evidence(hist, urecs, fm, solver, int(cfg["evidence"]["top_n"]))
    for row in ev.itertuples(index=False):
        title, genres = mv(row.movieId)
        print(BAR)
        hit = "  ✔ the user rated it highly later (test period)" if row.movieId in future else ""
        print(f"#{row.rank}  RECOMMENDATION: {display_title(title)}  [{genres}]  score {row.score:.3f}{hit}")
        print("    EVIDENCE (score lost if this one past rating is removed, fold-in):")
        for x in row.evidence:
            t, _ = mv(x["movieId"])
            print(f"      {x['influence']:+.4f}  ({x['influence'] / row.score:5.1%} of score)  "
                  f"{display_title(t)}  you rated {x['rating']:g}")
        rec = {"movieId": int(row.movieId), "title": title, "genres": genres}
        evid = [{"movieId": int(x["movieId"]), "title": mv(x["movieId"])[0], "genres": mv(x["movieId"])[1],
                 "rating": float(x["rating"])} for x in row.evidence]
        res = explain_one(rec, evid, int(e["max_cited"]), llm, int(e["retries"]), index)
        print(f"    EXPLANATION ({res['llm_name']}):")
        print(textwrap.indent(textwrap.fill(res["text"], 72), "      "))
        g = check(res["text"], rec, evid[:int(e["max_cited"])], res["cited_items"], index)
        print(f"    GROUNDING: {'grounded ✔' if g.grounded else 'NOT grounded ✘ ' + json.dumps(g.unsupported_claims)}"
              f"  ({g.n_words} words, Flesch {g.flesch:.0f})")
        fa = verify_user(hist, pd.DataFrame({"userId": [user], "movieId": [row.movieId], "cited_items": [res["cited_items"]]}),
                         fm, solver, k, float(v["rel_drop_threshold"]), bool(v["use_rank_criterion"]),
                         int(v["random_trials"]), int(v["seed"]))
        c = fa[fa.condition == "cited"].iloc[0]
        rnd = fa[fa.condition == "random"]
        print(f"    FAITHFULNESS: remove the {len(c.cited_item)} cited ratings -> score {c.score_before:.3f} → "
              f"{c.score_after:.3f} ({c.rel_drop:+.1%} drop), rank {c.rank_before} → {c.rank_after}")
        print(f"      random-removal baseline ({len(rnd)} trials): mean drop {rnd.rel_drop.mean():+.1%}, "
              f"mean rank → {rnd.rank_after.mean():.1f}")
        verdict = "FAITHFUL ✔" if c.faithful else "NOT faithful ✘"
        print(f"      verdict: {verdict}  (rule: drop ≥ {float(v['rel_drop_threshold']):.0%}"
              f"{' or leaves top-' + str(k) if v['use_rank_criterion'] else ''})")
    print(BAR)
    try:
        s = json.load(open(cfg.path("faithfulness_summary.json")))
        print(f"Dataset-level: faithfulness {s['faithfulness_score']:.1%} vs random baseline "
              f"{s['random_baseline_score']:.1%} over {s['n_explanations']:,} explanations.")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
