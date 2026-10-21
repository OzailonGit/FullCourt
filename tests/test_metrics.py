import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import ndcg_at_k, recall_at_k  # noqa: E402


def test_recall_at_k():
    grades = {"a": 2, "b": 1, "c": 0}
    assert recall_at_k(["a", "x", "c"], grades) == pytest.approx(0.5)
    assert recall_at_k(["a"], {"c": 0}) is None


def test_ndcg_perfect_and_partial():
    grades = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], grades) == pytest.approx(1.0)
    swapped = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], grades) == pytest.approx(swapped)
    assert ndcg_at_k(["x"], {"x": 0}) is None
