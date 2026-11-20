import pytest

from src.common.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    s = get_spark({"spark": {"master": "local[2]", "shuffle_partitions": 2, "driver_memory": "2g"}}, app="i-recs-tests")
    yield s
    s.stop()
