"""WBS-Lepage nonparametric change point detector."""

import numpy as np
from scipy.stats import rankdata
from typing import List, Optional


class WBSLepageDetector:
    def __init__(self, M: int = 10000, alpha: float = 0.05, min_seg_len: int = 10,
                 precompute: bool = True, max_len: int = 10000, nsim: int = 200):
        self.M = M
        self.alpha = alpha
        self.min_seg_len = min_seg_len
        self.max_len = max_len
        self.nsim = nsim
        self._thresholds = None
        if precompute:
            self._precompute_thresholds()

    @staticmethod
    def _lepage_stat(values: np.ndarray, p: int, q: int, k: int) -> float:
        l = q - p + 1
        if l < 2:
            return 0.0
        n1 = k - p + 1
        n2 = q - k
        if n1 < 1 or n2 < 1:
            return 0.0

        segment = values[p:q+1]
        ranks = rankdata(segment, method='average')
        left_ranks = ranks[:n1]

        U = np.sum(left_ranks) - n1 * (n1 + 1) / 2
        EU = n1 * n2 / 2
        VarU = n1 * n2 * (l + 1) / 12

        M_stat = np.sum((left_ranks - (l + 1) / 2) ** 2)
        EM = n1 * (l ** 2 - 1) / 12
        VarM = n1 * n2 * (l + 1) * (l ** 2 - 4) / 180

        if VarU < 1e-8 or VarM < 1e-8:
            return 0.0
        U_norm = (U - EU) / np.sqrt(VarU)
        M_norm = (M_stat - EM) / np.sqrt(VarM)
        return U_norm ** 2 + M_norm ** 2

    @staticmethod
    def _lepage_max_stat(values: np.ndarray, s: int, e: int, min_len: int) -> tuple[float, Optional[int]]:
        best_stat = -np.inf
        best_k = None
        for k in range(s, e):
            if (k - s + 1) < min_len or (e - k) < min_len:
                continue
            stat = WBSLepageDetector._lepage_stat(values, s, e, k)
            if stat > best_stat:
                best_stat = stat
                best_k = k
        return best_stat, best_k

    def _precompute_thresholds(self):
        np.random.seed(123)
        min_valid_len = 2 * self.min_seg_len
        lengths = list(range(min_valid_len, min(self.max_len, 500) + 1, 10))
        if self.max_len > 500:
            lengths += [1000, 2000, 5000, 10000]
            lengths = sorted(set([l for l in lengths if l <= self.max_len]))

        if not lengths:
            self._thresholds = {}
            return

        thresholds = {}
        for L in lengths:
            sim_stats = []
            for _ in range(self.nsim):
                y = np.random.randn(L)
                max_stat = -np.inf
                for _ in range(self.M):
                    seg_len = np.random.randint(min_valid_len, L + 1)
                    s = np.random.randint(0, L - seg_len + 1)
                    e = s + seg_len - 1
                    stat, _ = self._lepage_max_stat(y, s, e, self.min_seg_len)
                    if stat > max_stat:
                        max_stat = stat
                sim_stats.append(max_stat)
            finite_stats = [x for x in sim_stats if np.isfinite(x)]
            if not finite_stats:
                continue
            thresholds[L] = np.quantile(finite_stats, 1 - self.alpha)

        full = {}
        for l in range(min_valid_len, self.max_len + 1):
            if l in thresholds:
                full[l] = thresholds[l]
            else:
                left = max([ll for ll in thresholds if ll <= l], default=None)
                right = min([ll for ll in thresholds if ll >= l], default=None)
                if left is not None and right is not None:
                    interp = thresholds[left] + (thresholds[right] - thresholds[left]) * (l - left) / (right - left)
                    full[l] = interp
                elif left is not None:
                    full[l] = thresholds[left]
                else:
                    full[l] = thresholds[right]
        self._thresholds = full

    def _get_threshold(self, L: int, q: float) -> float:
        if self._thresholds is None:
            self._precompute_thresholds()
        if not self._thresholds:
            return np.inf
        key = min(L, max(self._thresholds.keys()))
        return q * self._thresholds[key]

    def _wbs_rec(self, values: np.ndarray, p: int, q: int, q_sensitivity: float) -> List[int]:
        L = q - p + 1
        if L < 2 * self.min_seg_len:
            return []

        intervals = []
        for _ in range(self.M):
            seg_len = np.random.randint(2 * self.min_seg_len, L + 1)
            s = np.random.randint(p, q - seg_len + 2)
            e = s + seg_len - 1
            intervals.append((s, e))

        best_stat = -np.inf
        best_split = None
        for s, e in intervals:
            stat, split = self._lepage_max_stat(values, s, e, self.min_seg_len)
            if stat > best_stat:
                best_stat = stat
                best_split = split

        threshold = self._get_threshold(L, q_sensitivity)
        if best_stat <= threshold or best_split is None:
            return []

        left = self._wbs_rec(values, p, best_split, q_sensitivity)
        right = self._wbs_rec(values, best_split + 1, q, q_sensitivity)
        return left + [best_split] + right

    def _prune(self, values: np.ndarray, candidates: List[int], q_sensitivity: float) -> List[int]:
        n = len(values)
        cand = sorted(candidates)
        keep = [True] * len(cand)
        for i, cp in enumerate(cand):
            left = cand[i-1] if i > 0 else -1
            right = cand[i+1] if i+1 < len(cand) else n
            s = left + 1
            e = right - 1
            if e - s + 1 < 2 * self.min_seg_len:
                keep[i] = False
                continue
            max_stat = -np.inf
            for _ in range(self.M):
                seg_len = np.random.randint(2 * self.min_seg_len, e - s + 2)
                ss = np.random.randint(s, e - seg_len + 2)
                ee = ss + seg_len - 1
                stat, _ = self._lepage_max_stat(values, ss, ee, self.min_seg_len)
                if stat > max_stat:
                    max_stat = stat
            thresh = self._get_threshold(e - s + 1, q_sensitivity)
            if max_stat <= thresh:
                keep[i] = False
        return [cp for i, cp in enumerate(cand) if keep[i]]

    def detect(self, series: np.ndarray, q: float = 1.0, prune: bool = True) -> List[int]:
        if len(series) < 2 * self.min_seg_len:
            return []
        values = np.asarray(series).astype(float)
        cand = self._wbs_rec(values, 0, len(values) - 1, q)
        if prune and len(cand) > 0:
            cand = self._prune(values, cand, q)
        return sorted(set(cand))
