"""Parquet I/O between pipeline stages.

With Hadoop native support (Linux/macOS, or Windows with winutils) this is plain
`df.write.parquet` / `spark.read.parquet`. Without it (stock Windows), writes are
done in parallel by each Spark partition with pyarrow (`mapInArrow`), and reads go
through pyarrow into a Spark DataFrame. Either way the on-disk layout is a
directory of part files that both Spark and pandas can read.
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema

from .spark import hadoop_native_available

log = logging.getLogger(__name__)


def write_parquet(df: DataFrame, path: str) -> str:
    path = os.path.abspath(path)
    if hadoop_native_available():
        df.write.mode("overwrite").parquet(path)
        log.info("wrote %s (spark)", path)
        return path
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    arrow_schema = to_arrow_schema(df.schema)
    target = path

    def _write(batches):
        import os as _os
        import uuid as _uuid

        import pyarrow as _pa
        import pyarrow.parquet as _pq

        batches = list(batches)
        if batches:
            t = _pa.Table.from_batches(batches)
            _pq.write_table(t, _os.path.join(target, f"part-{_uuid.uuid4().hex}.parquet"))
        return iter([])

    df.mapInArrow(_write, "_dummy int").collect()
    if not any(f.endswith(".parquet") for f in os.listdir(path)):
        pq.write_table(arrow_schema.empty_table(), os.path.join(path, f"part-{uuid.uuid4().hex}.parquet"))
    log.info("wrote %s (pyarrow fallback)", path)
    return path


def read_arrow(path: str) -> pa.Table:
    return pq.read_table(os.path.abspath(path))


def read_pandas(path: str):
    return read_arrow(path).to_pandas()


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    path = os.path.abspath(path)
    if hadoop_native_available():
        return spark.read.parquet(path)
    t = read_arrow(path)
    schema = from_arrow_schema(t.schema)
    if t.num_rows == 0:
        return spark.createDataFrame([], schema)
    return spark.createDataFrame(t.to_pandas(), schema=schema)


def write_pandas(pdf, path: str) -> str:
    """Write a small pandas frame (driver-side results) in the same layout."""
    path = os.path.abspath(path)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    pq.write_table(pa.Table.from_pandas(pdf, preserve_index=False), os.path.join(path, "part-0.parquet"))
    return path
