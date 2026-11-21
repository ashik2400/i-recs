"""Scalability benchmarks: every dataset x every local core count, one subprocess each.

python -m benchmarks.run [--set bench.masters='["local[1]","local[2]"]'] [--repeat 1]

Writes docs/bench_results.csv (rewritten each run, saved after every finished run).
Plots and tables come from `python -m scripts.make_report`.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys

import pandas as pd

from src.common.config import ROOT, cli

log = logging.getLogger("benchmarks")


def main():
    def extra(p):
        p.add_argument("--repeat", type=int, default=1)
        p.add_argument("--out", default="bench_results.csv")
    cfg, a = cli(__doc__, extra)
    b = cfg["bench"]
    rows = []
    for ds in b["datasets"]:
        mr = (b.get("max_ratings") or {}).get(ds)
        for master in b["masters"]:
            for rep in range(a.repeat):
                cmd = [sys.executable, "-m", "benchmarks.bench_one", "--dataset", ds, "--master", master]
                cmd += sum((["--set", x] for x in a.set if x.startswith("bench.")), [])
                if mr:
                    cmd += ["--max-ratings", str(mr)]
                log.info("running %s %s (rep %d)", ds, master, rep)
                line = None
                for attempt in range(2):                  # Windows worker start-up can time out under load
                    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                    line = next((l for l in p.stdout.splitlines() if l.startswith("BENCH ")), None)
                    if line:
                        break
                    log.warning("attempt %d failed for %s %s, retrying", attempt + 1, ds, master)
                if line is None:
                    log.error("run failed: %s\n%s", " ".join(cmd), p.stderr[-2000:])
                    continue
                row = json.loads(line[6:]) | {"repeat": rep}
                log.info("%s", row)
                rows.append(row)
                pd.DataFrame(rows).to_csv(cfg.docs / a.out, index=False)
    log.info("wrote %s", cfg.docs / a.out)


if __name__ == "__main__":
    main()
