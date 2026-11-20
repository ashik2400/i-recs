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
- [CONTRIBUTION.md](CONTRIBUTION.md): git workflow every teammate must follow

## Repo structure

```
i-recs/
├── README.md
├── PROJECT_GUIDE.md
├── CONTRIBUTING.md
├── .gitignore
├── .env.example
├── requirements.txt
├── notebooks/        # exploration only
├── src/
│   ├── als/          # data prep, ALS training, tuning
│   ├── evidence/     # perturbation attribution
│   ├── verify/       # faithfulness verification
│   └── explain/      # LLM prompting, grounding checks
├── eval/             # utility evaluation
├── benchmarks/       # scalability experiments
└── docs/             # report drafts, figures
```

## Getting started

### 1. Prerequisites
- Python 3.10 or 3.11
- Java 8, 11 or 17 (required by Spark). Check with `java -version`.
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
| Tests | `python -m pytest -q` | |

### Windows notes
- Python 3.12 needs `setuptools` (in `requirements.txt`) because `pyspark.ml` still imports `distutils`.
- Spark's own Parquet/CSV I/O on Windows needs Hadoop's `winutils.exe` + `hadoop.dll` in
  `%HADOOP_HOME%in`. Without them, `src/common/io.py` automatically falls back to
  pyarrow (parallel per-partition writes, Arrow reads into Spark DataFrames); with them
  it uses `df.write.parquet` / `spark.read.parquet`. All computation is Spark either way.

## Team and ownership

| Role | Area | Folders |
|---|---|---|
| Person A | Data, Spark, ALS, benchmarks | `src/als/`, `benchmarks/` |
| Person B | Evidence extraction, faithfulness verification | `src/evidence/`, `src/verify/` |
| Person C | LLM explanations, grounding checks, utility evaluation | `src/explain/`, `eval/` |

Names: _fill in_

## Workflow in short
Never push to `main`. Create a branch, commit, push, open a Pull Request, and get
one review. Full rules are in [CONTRIBUTION.md](CONTRIBUTION.md).

## Tech stack
Apache Spark (PySpark, MLlib) · Databricks CE / AWS EMR / Google Dataproc ·
LLM API · MovieLens 25M

## References
- XRec: Ma et al., EMNLP 2024
- GenRec: Netflix, 2026
- MovieLens: Harper and Konstan, ACM TiiS 2015