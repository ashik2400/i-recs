"""Config loading: YAML file + .env + `--set key.sub=value` overrides."""
from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("py4j", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _set(cfg: dict, dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)


class Config(dict):
    """dict with attribute-free helpers for the paths every stage needs."""

    @property
    def out_dir(self) -> Path:
        return ROOT / self["output_root"] / self["dataset"]

    def path(self, name: str) -> str:
        """Absolute path of a stage artifact, e.g. cfg.path('recs.parquet')."""
        return str(self.out_dir / name)

    @property
    def docs(self) -> Path:
        d = ROOT / self["docs_dir"]
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if os.getenv("SPARK_MASTER"):
        cfg["spark"]["master"] = os.environ["SPARK_MASTER"]
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        _set(cfg, k.strip(), v)
    c = Config(cfg)
    c.out_dir.mkdir(parents=True, exist_ok=True)
    return c


def cli(description: str, extra: callable | None = None) -> tuple[Config, argparse.Namespace]:
    """Standard CLI for every stage: --config and repeated --set overrides."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--log-level", default="INFO")
    if extra:
        extra(p)
    a = p.parse_args()
    setup_logging(a.log_level)
    return load_config(a.config, a.set), a
