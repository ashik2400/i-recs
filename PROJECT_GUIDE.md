# i-recs: Intelligent Recommendations with Verifiable Explanations

Big Data and Cloud Computing course project (PySpark). Team of 3.

---

## 1. The idea in one minute

Most recommenders (like ALS) are black boxes. LLMs can write a friendly
explanation ("because you liked X"), but nobody checks whether that explanation
is **true**. i-recs closes that gap:

1. Train a scalable **ALS recommender** in PySpark on MovieLens 25M.
2. For each recommendation, find the **evidence**: which of the user's past
   movies actually pushed that recommendation (by removing them one at a time
   and measuring the score change).
3. Give only that evidence to an **LLM** and make it write the explanation, with
   no freedom to add new reasons.
4. **Verify** the explanation: remove the cited movie and check that the
   recommendation really drops, as the explanation implied.
5. Measure whether verified explanations improve **trust or engagement**.
6. Show the whole pipeline **scales** on Spark.

**One-line pitch:** a recommender that doesn't just explain itself, but proves
the explanation is true, at big-data scale.

## 2. Why this is different from existing work

| Work | What it does | What it misses |
|---|---|---|
| XRec (EMNLP 2024) | LLM + CF embeddings give rich explanations | Never checks if the explanation is correct |
| GenRec (Netflix, 2026) | LLM rankers work at production scale | Leaves explanation as future work, no faithfulness check |
| **i-recs** | Evidence-grounded explanations plus a faithfulness test, on Spark | (this is our gap) |

## 3. Problems we address
1. No faithfulness verification of LLM explanations.
2. Explanations judged on fluency, not on whether they improve trust or engagement.
3. No scalable implementation of generating and verifying explanations.

## 4. Pipeline overview

```
MovieLens 25M
   │  (Person A)
   ▼
Clean + time-based split ──► ALS training/tuning ──► recommendations
                                                        │
                                                        ▼  (Person B)
                                    Perturbation attribution ──► evidence records
                                                        │
                                                        ▼  (Person C)
                                  LLM explanation (evidence-only prompt)
                                                        │
                                  Grounding + hallucination checks (Person C)
                                                        │
                                                        ▼  (Person B)
                              Faithfulness verification ──► faithfulness score
                                                        │
                                                        ▼  (Person C)
                                              Utility evaluation
   (Person A) Scalability benchmarks run across the whole pipeline
```

## 5. Roles and ownership

### Person A: Data, Spark and ALS (`src/als/`, `benchmarks/`)
**Phases 1, 2, 8**
- Set up PySpark (local, Databricks Community Edition, EMR or Dataproc)
- Load MovieLens 25M, clean it, do the **time-based** train/test split
- Train ALS (MLlib); tune rank, regParam and iterations with cross-validation
- Baseline metrics: RMSE, precision@k, recall@k, MRR
- Benchmarks: training time, extraction time and explanation latency vs. data size and worker count; speedup plots; optional CSV vs. Parquet and partitioning comparison
- **Delivers to B:** saved ALS model, train/test Parquet, top-K recommendations

### Person B: XAI evidence and faithfulness (`src/evidence/`, `src/verify/`)
**Phases 3, 6 (the core novel contribution)**
- Perturbation attribution: for a (user, recommended item), remove one past interaction at a time, re-score, and record the score drop
- Rank the user's past interactions by influence; keep the top-N as the evidence record
- Faithfulness verification: remove the item the explanation cites, re-score, and check the recommendation drops as claimed
- Compute the faithfulness score over all tested explanations
- Include a **random-removal baseline** to show that cited evidence matters more than random items
- Scalability note: retraining ALS per removal is too slow. Recompute the user's factor vector without that item (fold-in) instead, using the fixed item factors. Agree this with Person A.
- **Delivers to C:** evidence records
- **Receives from C:** explanations with their cited items

### Person C: LLM, quality checks and utility (`src/explain/`, `eval/`)
**Phases 4, 5, 7**
- Prompt design: feed **structured evidence only** and tell the LLM to phrase it, not to invent reasons
- Generate explanations for a sample of recommendations
- Grounding check (rule-based): does the explanation reference the given evidence?
- Hallucination check: does it mention anything not in the evidence (movies, genres, years, reasons)?
- Basic readability pass
- Utility evaluation: a small user survey (with vs. without explanation, ask trust and preference), or an offline proxy classifier estimating acceptance likelihood
- **Delivers to B:** explanations plus the list of cited item IDs

### Shared (all three)
- Report and slides: each writes the section for their own part
- Demo: user → recommendation → evidence → explanation → faithfulness score
- Rotating code review (see `CONTRIBUTING.md`)

## 6. Data contracts (agree on these in week 1)

Work in parallel by agreeing on file formats first. Each stage writes Parquet;
the next stage reads it. Use mock data until the real files exist.

| File | Columns | Written by | Read by |
|---|---|---|---|
| `ratings_train.parquet`, `ratings_test.parquet` | userId, movieId, rating, timestamp | A | A, B |
| `als_model/` | Spark ALSModel | A | B |
| `recs.parquet` | userId, movieId, score, rank | A | B, C |
| `evidence.parquet` | userId, movieId (recommended), evidence: array<struct<movieId, influence>> | B | C |
| `explanations.parquet` | userId, movieId, text, cited_items: array<int>, llm_name, prompt_version | C | B |
| `grounding.parquet` | userId, movieId, grounded: bool, unsupported_claims | C | report |
| `faithfulness.parquet` | userId, movieId, cited_item, score_before, score_after, rank_before, rank_after, faithful: bool | B | report |

Do not change a schema without telling the other two.

## 7. Key definitions

- **Evidence:** the past interactions with the largest influence on a given recommendation.
- **Influence score:** drop in the predicted score for the recommended item when that past interaction is removed.
- **Grounded explanation:** mentions only items and facts that are in its evidence record.
- **Faithful explanation:** if its cited evidence is removed, the recommendation's score drops materially or its rank falls out of the top-K. Define the threshold together, then fix it.
- **Faithfulness score:** fraction of tested explanations that are faithful. Report it next to the random-removal baseline.

## 8. Timeline (9 weeks)

| Weeks | Person A | Person B | Person C |
|---|---|---|---|
| 1-2 | Spark setup, data load, split | Study ALS internals, design evidence schema | Pick LLM, draft prompt templates |
| 2-3 | ALS training and tuning | Prototype attribution on a small sample | Build grounding checks using mock evidence |
| 3-5 | Baseline metrics, freeze the model | Full-scale attribution on Spark | Generate explanations from real evidence |
| 5-6 | Start benchmarks | Faithfulness verification | Utility evaluation |
| 7-8 | Finish benchmarks and plots | Finalize scores, write up | Finalize results, write up |
| 9 | Report, demo, rehearsal (everyone) | | |

## 9. Tech stack
- Apache Spark (PySpark, MLlib)
- Databricks Community Edition / AWS EMR / Google Dataproc
- LLM API (lightweight open-source or hosted)
- Dataset: MovieLens 25M (not stored in git; see section 10)

## 10. Repo layout

```
i-recs/
├── README.md
├── PROJECT_GUIDE.md
├── CONTRIBUTING.md
├── .gitignore
├── requirements.txt
├── notebooks/        # exploration only, one folder per person
├── src/
│   ├── als/          # Person A
│   ├── evidence/     # Person B
│   ├── verify/       # Person B
│   └── explain/      # Person C
├── eval/             # Person C
├── benchmarks/       # Person A
└── docs/             # report drafts, figures
```

**Data:** MovieLens 25M is too big for git. Download it from GroupLens
(https://grouplens.org/datasets/movielens/25m/) into a local `data/` folder,
which is git-ignored.

## 11. Expected outcomes
- A distributed ALS recommender on a Spark cluster
- An evidence-grounded LLM explanation module with a measurable faithfulness score
- A utility evaluation of whether grounded explanations affect engagement
- Scalability results: time, latency and cost vs. data and cluster size

## 12. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Attribution too slow at 25M scale | Fold-in re-scoring; evaluate on a sample of users first |
| LLM adds unsupported claims | Strict prompt, grounding and hallucination checks, regenerate on failure |
| Faithfulness score looks low | Report it honestly with the random baseline; analyze why |
| Free-tier cluster limits | Develop on a 1M or 100K sample, run full data only for benchmarks |
| One person blocked by another | Mock data and fixed data contracts (section 6) |

## 13. Glossary
- **ALS:** Alternating Least Squares, a matrix-factorization collaborative-filtering method
- **XAI:** explainable AI
- **Perturbation attribution:** measuring importance by removing an input and observing the change
- **Faithfulness:** whether an explanation reflects the model's real behavior
- **Hallucination:** content in the explanation not supported by the evidence