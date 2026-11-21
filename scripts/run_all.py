"""Run the whole pipeline (M1-M6) in one Spark session, then build the report.

python -m scripts.run_all [--set dataset=ml-1m] [--skip prepare,train]
"""
from __future__ import annotations

import logging
import time

from src.als import prepare, train
from src.als.download import download
from src.common.config import cli
from src.common.spark import get_spark
from src.eval import utility
from src.evidence import extract
from src.explain import check, generate
from src.verify import faithfulness

STAGES = [("prepare", prepare.run), ("train", train.run), ("evidence", extract.run), ("explain", generate.run),
          ("grounding", check.run), ("faithfulness", faithfulness.run), ("utility", utility.run)]
log = logging.getLogger("run_all")


def main():
    cfg, a = cli(__doc__, lambda p: p.add_argument("--skip", default=""))
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    download(cfg["dataset"], cfg["raw_dir"])
    spark = get_spark(cfg, "i-recs-all")
    for name, fn in STAGES:
        if name in skip:
            log.info("skip %s", name)
            continue
        t0 = time.time()
        log.info("=== %s ===", name)
        fn(cfg, spark)
        log.info("=== %s done in %.1fs ===", name, time.time() - t0)
    spark.stop()
    import sys

    from scripts import make_report
    sys.argv = [sys.argv[0]] + sum((["--set", s] for s in a.set), [])
    make_report.main()


if __name__ == "__main__":
    main()
