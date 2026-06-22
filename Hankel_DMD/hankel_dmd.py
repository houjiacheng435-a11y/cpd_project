"""Hankel-DMD 核心计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HankelDMDResult:
    """Hankel-DMD 拟合结果。"""

    eigenvalues: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    rank: int
    X: np.ndarray | None = None
    Y: np.ndarray | None = None
    A_hat: np.ndarray | None = None


def _as_1d_finite_array(x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("x 必须是一维序列")
    if arr.size == 0:
        raise ValueError("x 不能为空")
    if not np.isfinite(arr).all():
        raise ValueError("x 不能包含 NaN 或 inf")
    return arr


def _validate_hankel_shape(N: int, m: int, n: int) -> tuple[int, int]:
    if not isinstance(m, int) or not isinstance(n, int):
        raise TypeError("m 和 n 必须是整数")
    if m <= 0:
        raise ValueError("m 必须为正整数")
    if n < 0:
        raise ValueError("n 必须为非负整数")
    n_cols = n + 1
    required = m + n + 1
    if required > N:
        raise ValueError(f"样本不足：需要至少 m+n+1={required} 个点，但 x 只有 {N} 个点")
    return m, n_cols


def build_hankel_pair(
    x: np.ndarray | list[float] | tuple[float, ...],
    m: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """构造 Hankel-DMD 的 X/Y 矩阵。

    参数
    ----
    x:
        一维 single observable 序列。
    m:
        Hankel 矩阵行数。
    n:
        Hankel 矩阵列数减一，因此实际列数是 ``n + 1``。

    返回
    ----
    tuple[np.ndarray, np.ndarray]
        ``X`` 和 ``Y``，形状均为 ``(m, n + 1)``。
    """

    arr = _as_1d_finite_array(x)
    m, n_cols = _validate_hankel_shape(arr.size, m, n)

    X = np.empty((m, n_cols), dtype=float)
    Y = np.empty((m, n_cols), dtype=float)
    for i in range(m):
        X[i, :] = arr[i : i + n_cols]
        Y[i, :] = arr[i + 1 : i + n_cols + 1]
    return X, Y


def _validate_rank(rank: int, max_rank: int, singular_values: np.ndarray) -> int:
    if not isinstance(rank, int):
        raise TypeError("rank 必须是整数")
    if rank <= 0:
        raise ValueError("rank 必须为正整数")
    if rank > max_rank:
        raise ValueError(f"rank 不能超过 min(X.shape)={max_rank}")
    kept = singular_values[:rank]
    eps = np.finfo(float).eps
    tol = eps * max(singular_values.shape[0], 1) * singular_values[0] if singular_values.size else 0.0
    if np.any(kept <= tol):
        raise ValueError("rank 包含数值上为 0 的奇异值，请降低 rank")
    return rank


def fit_hankel_dmd(
    x: np.ndarray | list[float] | tuple[float, ...],
    m: int,
    n: int,
    rank: int,
    *,
    return_matrices: bool = False,
) -> HankelDMDResult:
    """执行 single-observable Hankel-DMD 核心计算。

    本函数只完成 Hankel 矩阵构造、截断 SVD、低维 DMD 算子、特征值和
    projected modes 计算。不做画图、诊断报告、参数搜索或回测。
    """

    X, Y = build_hankel_pair(x, m=m, n=n)
    W, singular_values, Vh = np.linalg.svd(X, full_matrices=False)
    actual_rank = _validate_rank(rank, min(X.shape), singular_values)

    W_r = W[:, :actual_rank]
    s_r = singular_values[:actual_rank]
    V_r = Vh[:actual_rank, :].conj().T

    A_hat = W_r.conj().T @ Y @ V_r @ np.diag(1.0 / s_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_hat)
    modes = W_r @ eigenvectors

    return HankelDMDResult(
        eigenvalues=eigenvalues,
        modes=modes,
        singular_values=singular_values,
        rank=actual_rank,
        X=X if return_matrices else None,
        Y=Y if return_matrices else None,
        A_hat=A_hat if return_matrices else None,
    )


__all__ = ["HankelDMDResult", "build_hankel_pair", "fit_hankel_dmd"]
