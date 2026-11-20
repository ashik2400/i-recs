"""SparkSession factory (local mode by default, works on Windows)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

log = logging.getLogger(__name__)


def hadoop_native_available() -> bool:
    """True if Spark can use Hadoop's local filesystem for Parquet I/O.

    On Linux/macOS this is always true. On Windows Hadoop needs winutils.exe and
    hadoop.dll under %HADOOP_HOME%\\bin; without them Spark can compute but not
    read/write local files, and src.common.io falls back to pyarrow.
    """
    if os.name != "nt":
        return True
    home = os.getenv("HADOOP_HOME")
    if not home:
        return False
    b = Path(home) / "bin"
    return (b / "winutils.exe").exists() and (b / "hadoop.dll").exists()


def get_spark(cfg: dict | None = None, app: str = "i-recs", master: str | None = None) -> SparkSession:
    scfg = (cfg or {}).get("spark", {})
    master = master or scfg.get("master", "local[*]")
    # Workers must use the same interpreter as the driver (the venv).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    if os.name == "nt" and hadoop_native_available():
        hb = str(Path(os.environ["HADOOP_HOME"]) / "bin")
        if hb not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hb + os.pathsep + os.environ.get("PATH", "")
    spark = (
        SparkSession.builder.appName(app)
        .master(master)
        .config("spark.driver.memory", scfg.get("driver_memory", "4g"))
        .config("spark.sql.shuffle.partitions", str(scfg.get("shuffle_partitions", 16)))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(scfg.get("log_level", "ERROR"))
    log.info("Spark %s master=%s native_io=%s", spark.version, master, hadoop_native_available())
    return spark
