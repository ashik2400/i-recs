# i-recs: Intelligent Recommendations with Verifiable Explanations

A scalable PySpark recommender that explains itself and proves the explanation is true.

i-recs trains an ALS collaborative-filtering model on MovieLens 25M, uses
perturbation-based attribution to extract the concrete evidence behind each
recommendation, and conditions an LLM to phrase explanations strictly from that
evidence. A faithfulness check then removes the cited evidence and tests whether
the recommendation changes as the explanation claims, while a grounding check
catches unsupported statements. The full pipeline runs in parallel on Spark.

Course project: Big Data and Cloud Computing.

## Pipeline

```
MovieLens 25M → clean + split → ALS → recommendations
   → evidence extraction (perturbation) → LLM explanation (evidence-only)
   → grounding/hallucination checks → faithfulness verification → utility evaluation
```

## Documentation
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md): full project idea, roles, data contracts, timeline
- [docs/results.md](docs/results.md): all results, tables and figures (generated)

## Repo structure

```
i-recs/
├── configs/default.yaml   # every setting (dataset, ALS grid, sample sizes, thresholds, benchmarks)
├── src/
│   ├── common/            # config, SparkSession, Parquet I/O
│   ├── als/               # download, prepare (clean + split), train (tune + ALS + recs + metrics), fold-in math
│   ├── evidence/          # perturbation attribution via fold-in
│   ├── explain/           # LLM client, evidence-only prompt, generation, grounding checks
│   ├── verify/            # faithfulness verification + random baseline
│   └── eval/              # utility proxy classifier
├── benchmarks/            # scalability runs (bench_one per dataset x cores, run = grid)
├── scripts/               # run_all, demo, make_report
├── tests/                 # pytest on tiny synthetic data
├── docs/                  # results.md + figures (generated)
├── data/   (git-ignored)  # raw MovieLens
└── outputs/ (git-ignored) # stage outputs per dataset
```

## Getting started

### 1. Prerequisites
- Python 3.10-3.12
- Java 8, 11 or 17 (required by Spark; also verified working on Java 19). Check with `java -version`.
- Git

### 2. Clone and create an environment

```bash
git clone https://github.com/<owner>/i-recs.git
cd i-recs

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

Open `.env` and fill in your own values. **Never commit `.env`.**

### 4. Download the data
MovieLens 25M is not stored in git. Download it from
https://grouplens.org/datasets/movielens/25m/, unzip it, and place it under
`data/ml-25m/`. For development, use a smaller version (MovieLens 100K or 1M)
first so Spark runs fast on laptops.

### 5. Check Spark works

```bash
python -c "from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[*]').getOrCreate(); print(s.version); s.stop()"
```

## Running the pipeline

All settings live in `configs/default.yaml`; override any of them with
`--set key=value` (e.g. `--set dataset=ml-1m`). Outputs go to
`outputs/<dataset>/`. Run from the repo root with the venv active.

| Stage | Command | Writes |
|---|---|---|
| Download data | `python -m src.als.download --dataset ml-latest-small` | `data/raw/ml-latest-small/` |
| M1 prepare (clean + time split) | `python -m src.als.prepare` | `ratings_train/test.parquet`, `movies.parquet` |
| M1 train (tune + ALS + metrics) | `python -m src.als.train` | `als_model/`, `recs.parquet`, `als_metrics.json`, `als_tuning.csv` |
| M2 evidence (fold-in attribution) | `python -m src.evidence.extract` | `evidence.parquet`, `evidence_stats.json` |
| M3 explanations (evidence-only) | `python -m src.explain.generate` | `explanations.parquet` |
| M4 grounding + hallucination checks | `python -m src.explain.check` | `grounding.parquet`, `grounding_summary.json` |
| M5 faithfulness verification | `python -m src.verify.faithfulness` | `faithfulness.parquet`, `faithfulness_summary.json`, `faithfulness_sweep.csv` |
| M6 utility (offline proxy classifier) | `python -m src.eval.utility` | `utility.json` |
| M7 scalability benchmarks | `python -m benchmarks.run` | `docs/bench_results.csv` |
| M8 report tables + figures | `python -m scripts.make_report` | `docs/results.md`, `docs/*.png` |
| **Whole pipeline (M1-M6 + report)** | `python -m scripts.run_all` | everything above |
| **Demo** | `python -m scripts.demo --user 12` | prints user → rec → evidence → explanation → grounding → faithfulness |
| Tests | `python -m pytest -q` | |

Quick start on a fresh clone:

```bash
pip install -r requirements.txt
python -m scripts.run_all            # downloads ml-latest-small, runs M1-M6, writes docs/results.md (~10 min)
python -m scripts.demo --user 12
python -m benchmarks.run             # optional, ~30-60 min (needs ml-1m and ml-25m downloaded)
```

Results and figures: [docs/results.md](docs/results.md).

### Using a real LLM for explanations
Set in `.env` (see `.env.example`), then rerun `python -m src.explain.generate`:
- Local, free: install [Ollama](https://ollama.com), `ollama pull llama3.2:3b`, set `LLM_PROVIDER=ollama`.
- Anthropic: `pip install anthropic`, set `LLM_PROVIDER=anthropic`, `LLM_API_KEY=...` (default model `claude-opus-5-5`;
  server-side refusal fallback is enabled).
- Groq (free tier): set `GROQ_API_KEY=...` (provider is picked automatically when `LLM_PROVIDER` is empty;
  default model `openai/gpt-oss-120b`; 429 rate limits are retried). Used for the reported results.
- OpenAI-compatible: `LLM_PROVIDER=openai`, `LLM_MODEL`, `LLM_API_KEY`, optional `LLM_BASE_URL`.

Only the first `explain.max_llm_calls` rows go to the LLM; any output that fails the grounding check is
regenerated once and then replaced by the deterministic template (`llm_name` records which writer was used).

### Key design decisions
- **Fold-in instead of retraining.** Spark ALS's final half-step solves each user's vector in closed form given
  the item factors, so removing past ratings and re-solving that small k×k system with the item factors fixed
  gives the exact counterfactual user vector. `tests/test_foldin.py` checks it reproduces Spark's stored user
  factors (explicit and implicit). Leave-one-out uses rank-one downdates: one batched solve per user.
- **Implicit ALS for top-K** (beats popularity; explicit ALS does not), explicit ALS reported for RMSE.
- **Per-user temporal split**: each user's latest 20% of ratings are test (`split.strategy: global` also available).
- **Faithful** = removing all cited items jointly lowers the score ≥ `verify.rel_drop_threshold` (10%) or pushes
  the item out of the top-K; compared against a seeded random-removal baseline of the same size.
- Stages exchange Parquet files (schemas in PROJECT_GUIDE.md §6) under `outputs/<dataset>/`.

### Windows notes
- Python 3.12 needs `setuptools` (in `requirements.txt`) because `pyspark.ml` still imports `distutils`.
- Spark's own Parquet/CSV I/O on Windows needs Hadoop's `winutils.exe` + `hadoop.dll` in
  `%HADOOP_HOME%\bin`. Without them, `src/common/io.py` automatically falls back to
  pyarrow (parallel per-partition writes, Arrow reads into Spark DataFrames); with them
  it uses `df.write.parquet` / `spark.read.parquet`. All computation is Spark either way.

## Ownership
Single maintainer (originally planned as a team of three; the role split in
PROJECT_GUIDE.md §5 is kept for reference only). Commits use `area: message`.

## Tech stack
Apache Spark (PySpark, MLlib) · Databricks CE / AWS EMR / Google Dataproc ·
LLM API · MovieLens 25M

## References
- XRec: Ma et al., EMNLP 2024
- GenRec: Netflix, 2026
- MovieLens: Harper and Konstan, ACM TiiS 2015