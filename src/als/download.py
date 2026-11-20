"""Download and unzip a MovieLens dataset into data/raw/.

Usage: python -m src.als.download --dataset ml-latest-small
"""
import argparse
import logging
import zipfile
from pathlib import Path

import requests

URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

log = logging.getLogger(__name__)


def download(name: str, raw_dir: str = "data/raw") -> Path:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / name
    if target.exists():
        log.info("%s already present at %s", name, target)
        return target
    zpath = raw / f"{name}.zip"
    log.info("downloading %s", URLS[name])
    with requests.get(URLS[name], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw)
    zpath.unlink()
    log.info("extracted to %s", target)
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(URLS), default="ml-latest-small")
    p.add_argument("--raw-dir", default="data/raw")
    a = p.parse_args()
    download(a.dataset, a.raw_dir)
