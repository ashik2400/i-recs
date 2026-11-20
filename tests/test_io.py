from pyspark.sql import types as T

from src.common.io import read_pandas, read_parquet, write_parquet

EVIDENCE_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType()),
    T.StructField("movieId", T.IntegerType()),
    T.StructField("evidence", T.ArrayType(T.StructType([
        T.StructField("movieId", T.IntegerType()),
        T.StructField("influence", T.DoubleType()),
    ]))),
])


def test_roundtrip_nested(spark, tmp_path):
    rows = [(1, 10, [(5, 0.3), (6, 0.1)]), (2, 11, [])]
    df = spark.createDataFrame(rows, EVIDENCE_SCHEMA).repartition(2)
    p = str(tmp_path / "ev.parquet")
    write_parquet(df, p)
    back = read_parquet(spark, p)
    assert back.count() == 2
    got = {r.userId: [(e.movieId, e.influence) for e in r.evidence] for r in back.collect()}
    assert got == {1: [(5, 0.3), (6, 0.1)], 2: []}
    assert len(read_pandas(p)) == 2


def test_empty_write(spark, tmp_path):
    df = spark.createDataFrame([], EVIDENCE_SCHEMA)
    p = str(tmp_path / "empty.parquet")
    write_parquet(df, p)
    assert read_parquet(spark, p).count() == 0
