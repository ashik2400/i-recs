"""ALS factor storage and the numpy fold-in math shared by every stage.

Spark MLlib ALS alternates closed-form least-squares solves; its last half-step
updates the user factors given the final item factors Y. For a user with rated
items I (n = |I|) the user vector is therefore x = A^-1 b with

  explicit (ALS-WR):   A = Y_I^T Y_I                      + lambda*n*I,   b = Y_I^T r
  implicit (Hu et al): A = Y^T Y + Y_I^T diag(alpha*r) Y_I + lambda*n*I,   b = Y_I^T (1 + alpha*r)

(Spark scales lambda by the user's number of ratings in both modes.)

"Fold-in" re-solves this with Y fixed and part of the history removed: the exact
effect of deleting a past interaction on the user's vector, without retraining
ALS. Removing interaction j is a rank-one downdate of A and b, so all
leave-one-out vectors come from one batched solve.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FoldIn:
    """Fold-in solver for one trained ALS model (item factors fixed)."""

    Y: np.ndarray                 # (m, k) all item factors
    reg: float
    implicit: bool = False
    alpha: float = 1.0
    YtY: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Y = np.asarray(self.Y, dtype=np.float64)
        self.k = self.Y.shape[1]
        self.YtY = self.Y.T @ self.Y if self.implicit else np.zeros((self.k, self.k))

    # per-interaction contributions to A (weight w on y y^T) and to b (coef c on y)
    def _terms(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=np.float64)
        if self.implicit:
            conf = self.alpha * np.abs(r)
            return conf, 1.0 + conf
        return np.ones_like(r), r

    def normal_eq(self, idx: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Yi = self.Y[idx]
        w, c = self._terms(r)
        A = self.YtY + (Yi * w[:, None]).T @ Yi + self.reg * len(idx) * np.eye(self.k)
        return A, Yi.T @ c

    def user_vector(self, idx: np.ndarray, r: np.ndarray) -> np.ndarray:
        """x_u from the user's history (item row indices idx, ratings r)."""
        if len(idx) == 0:
            return np.zeros(self.k)
        A, b = self.normal_eq(idx, r)
        return np.linalg.solve(A, b)

    def user_vector_without(self, idx: np.ndarray, r: np.ndarray, remove: np.ndarray) -> np.ndarray:
        """x_u after deleting the interactions where boolean mask `remove` is True."""
        keep = ~np.asarray(remove, dtype=bool)
        return self.user_vector(np.asarray(idx)[keep], np.asarray(r)[keep])

    def leave_one_out(self, idx: np.ndarray, r: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """(n, k): user vector with each single interaction removed (exact, batched)."""
        n = len(idx)
        out = np.zeros((n, self.k))
        if n <= 1:
            return out  # empty history -> Spark would have no factor; use zero vector
        A, b = self.normal_eq(idx, r)
        Yi = self.Y[idx]
        w, c = self._terms(r)
        I = np.eye(self.k)
        for s in range(0, n, chunk):
            Yc = Yi[s:s + chunk]
            As = A[None] - w[s:s + chunk, None, None] * np.einsum("ni,nj->nij", Yc, Yc) - self.reg * I[None]
            bs = b[None] - c[s:s + chunk, None] * Yc
            out[s:s + chunk] = np.linalg.solve(As, bs[..., None])[..., 0]
        return out


def rank_among(scores: np.ndarray, target: int, candidates: np.ndarray) -> int:
    """1-based rank of item index `target` among boolean `candidates` (ties favour target)."""
    return int((scores[candidates] > scores[target]).sum()) + 1


# ---------------------------------------------------------------- storage
@dataclass
class FactorModel:
    item_ids: np.ndarray          # (m,) movieIds, sorted
    item_factors: np.ndarray      # (m, k)
    meta: dict

    def __post_init__(self):
        self.index = {int(i): j for j, i in enumerate(self.item_ids)}

    def idx(self, movie_ids) -> np.ndarray:
        """Row indices for movieIds; -1 for unknown items."""
        return np.array([self.index.get(int(m), -1) for m in movie_ids], dtype=np.int64)

    def solver(self) -> FoldIn:
        return FoldIn(self.item_factors, self.meta["regParam"], self.meta["implicitPrefs"], self.meta.get("alpha", 1.0))


def save_factors(model, meta: dict, path: str, spark_write) -> None:
    """Persist a Spark ALSModel's factors (Parquet) + hyper-parameters (JSON)."""
    from pyspark.sql import functions as F

    os.makedirs(path, exist_ok=True)
    spark_write(model.userFactors.select(F.col("id").alias("userId"), "features"), os.path.join(path, "user_factors"))
    spark_write(model.itemFactors.select(F.col("id").alias("movieId"), "features"), os.path.join(path, "item_factors"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_factor_model(path: str) -> FactorModel:
    from src.common.io import read_pandas

    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    items = read_pandas(os.path.join(path, "item_factors")).sort_values("movieId")
    Y = np.vstack(items["features"].to_numpy()).astype(np.float64)
    return FactorModel(items["movieId"].to_numpy(np.int64), Y, meta)
