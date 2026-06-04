"""Change point detection module extracted from CPD.ipynb.

This module exposes the Brk class and generate_breaks() helper.
"""

from __future__ import annotations

import datetime
from collections import deque
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2


class Brk:
    """Change point detection class with multiple methods."""

    def __init__(self, method: str, q: float = 1.0, **kwargs):
        """Initialize detector.

        Args:
            method: Detection method name.
            q: Sensitivity factor. Larger q means less sensitive.
            kwargs: Method-specific parameters.
        """
        self.method = method.lower()
        self.q = q
        self.kwargs = kwargs
        self._valid_methods = [
            'cusum', 'sprt', 'gma',
            'cusum_ls', 'glr', 'brandt_glr', 'shewhart',
            'recursive_param_seg', 'e_detector', 'ssr_cusum', 'adaptive_cusum',
        ]
        if self.method not in self._valid_methods:
            raise ValueError(f"Unsupported method: {method}. Choose from {self._valid_methods}")

    def detect(self, ds: pd.Series) -> List[datetime.datetime]:
        """Detect change points from a datetime-indexed pandas Series."""
        return [event["timestamp"] for event in self.detect_events(ds)]

    def detect_events(self, ds: pd.Series) -> List[Dict[str, Any]]:
        """Detect change points with direction metadata for market-state vectors."""
        if self.method == 'cusum':
            h = self.kwargs.get('h', 0.03)
            nu = self.kwargs.get('nu', 0.00005)
            return self.cusum_events(ds, self.q, h=h, nu=nu)
        elif self.method == 'cusum_ls':
            warmup = self.kwargs.get('warmup', 30)
            drift = self.kwargs.get('drift', 0.003)
            k = self.kwargs.get('k', 5.0)
            return self.cusum_ls_events(ds, self.q, warmup=warmup, drift=drift, k=k)
        elif self.method == 'sprt':
            warmup = self.kwargs.get('warmup', 20)
            drift = self.kwargs.get('drift', 0)
            k = self.kwargs.get('k', 30.0)
            return self.sprt_events(ds, self.q, warmup=warmup, drift=drift, k=k)
        elif self.method == 'gma':
            lam = self.kwargs.get('lam', 0.9)
            k = self.kwargs.get('k', 2.0)
            return self.gma_events(ds, self.q, lam, k)
        elif self.method == 'glr':
            window = self.kwargs.get('window', None)
            init_len = self.kwargs.get('init_len', 20)
            min_right_len = self.kwargs.get('min_right_len', 5)
            return self.glr_events(ds, self.q, window=window, init_len=init_len, min_right_len=min_right_len)
        elif self.method == 'brandt_glr':
            window = self.kwargs.get('window', 10)
            alpha = self.kwargs.get('alpha', 0.0001)
            return self.brandt_glr_events(ds, self.q, window=window, alpha=alpha)
        elif self.method == 'shewhart':
            window = self.kwargs.get('window', 24)
            c_lim = self.kwargs.get('c_lim', 5)
            k = self.kwargs.get('k', 10.0)
            return self.shewhart_events(ds, self.q, window=window, c_lim=c_lim, k=k)
        elif self.method == 'recursive_param_seg':
            M = self.kwargs.get('M', 3)
            split_top_k = self.kwargs.get('split_top_k', 5)
            min_seg_len = self.kwargs.get('min_seg_len', 10)
            return self.recursive_param_seg_events(ds, self.q, M=M, split_top_k=split_top_k, min_seg_len=min_seg_len)
        elif self.method == 'e_detector':
            vol_window = self.kwargs.get('vol_window', 30)
            warmup = self.kwargs.get('warmup', 50)
            lambda_grid = self.kwargs.get('lambda_grid', (0.01, 0.02, 0.05, 0.1))
            clip = self.kwargs.get('clip', 5.0)
            two_sided = self.kwargs.get('two_sided', True)
            return self.e_detector_events(ds, q=self.q, vol_window=vol_window, warmup=warmup,
                                          lambda_grid=lambda_grid, clip=clip, two_sided=two_sided)
        elif self.method == 'ssr_cusum':
            zeta = self.kwargs.get('zeta', 0.25)
            h0 = self.kwargs.get('h0', 10.0)
            return self.ssr_cusum_events(ds, self.q, zeta=zeta, h0=h0)
        elif self.method == 'adaptive_cusum':
            h0 = self.kwargs.get('h0', 10.0)
            vol_window = self.kwargs.get('vol_window', 30)
            warmup = self.kwargs.get('warmup', None)
            rho_mu = self.kwargs.get('rho_mu', 0.25)
            s = self.kwargs.get('s', 1.0)
            eta = self.kwargs.get('eta', 4.0)
            rho_theta_up = self.kwargs.get('rho_theta_up', 1.05)
            rho_theta_down = self.kwargs.get('rho_theta_down', 1 / 1.05)
            return self.adaptive_cusum_events(ds, self.q, h0=h0, vol_window=vol_window, warmup=warmup,
                                              rho_mu=rho_mu, s=s, eta=eta,
                                              rho_theta_up=rho_theta_up, rho_theta_down=rho_theta_down)

    @staticmethod
    def _event(timestamp, direction: int, source: str, score: float = np.nan) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "direction": int(np.sign(direction)) if direction != 0 else 0,
            "direction_source": source,
            "score": float(score) if np.isfinite(score) else np.nan,
        }

    @staticmethod
    def _infer_direction_from_windows(
        ds: pd.Series,
        cp_time,
        lookback: int = 10,
        min_points: int = 2,
    ) -> int:
        if cp_time not in ds.index:
            return 0
        pos = int(ds.index.get_loc(cp_time))
        left = ds.iloc[max(0, pos - lookback):pos].astype(float)
        right = ds.iloc[pos:min(len(ds), pos + lookback)].astype(float)
        if len(left) < min_points or len(right) < min_points:
            return 0
        diff = float(right.mean() - left.mean())
        return 1 if diff > 0 else -1 if diff < 0 else 0

    @staticmethod
    def _direction_from_value(value: float) -> int:
        if not np.isfinite(value) or value == 0:
            return 0
        return 1 if value > 0 else -1

    def cusum(self, ds: pd.Series, q: float, h: float, nu: float) -> List[datetime.datetime]:
        return [event["timestamp"] for event in self.cusum_events(ds, q, h=h, nu=nu)]

    def cusum_events(self, ds: pd.Series, q: float, h: float, nu: float) -> List[Dict[str, Any]]:
        diff = ds.diff().dropna()
        if len(diff) == 0:
            return []

        values = diff.values
        g_pos = 0.0
        g_neg = 0.0
        events = []

        for idx, x in enumerate(values):
            g_pos = max(0.0, g_pos + x - nu)
            g_neg = max(0.0, g_neg - x - nu)
            if g_pos > q * h or g_neg > q * h:
                direction = 1 if g_pos >= g_neg else -1
                events.append(self._event(diff.index[idx], direction, "native_cpd", max(g_pos, g_neg)))
                g_pos = 0.0
                g_neg = 0.0

        return events

    def cusum_ls(self, ds: pd.Series, q: float, warmup: int, drift: float,k: float) -> List[datetime.datetime]:
        return [event["timestamp"] for event in self.cusum_ls_events(ds, q, warmup=warmup, drift=drift, k=k)]

    def cusum_ls_events(self, ds: pd.Series, q: float, warmup: int, drift: float, k: float) -> List[Dict[str, Any]]:
        diff = ds.diff().dropna()
        if len(diff) < warmup + 1:
            return []
        values = diff.values

        mu = np.mean(values[:warmup])
        var = np.var(values[:warmup])

        g_pos = 0.0
        g_neg = 0.0
        events = []
        n = warmup

        for i in range(warmup, len(values)):
            x = values[i]
            sigma = np.sqrt(max(var, 1e-8))
            threshold = q * k * sigma
            z = x - mu

            g_pos = max(0.0, g_pos + z - drift)
            g_neg = min(0.0, g_neg + z + drift)

            if g_pos > threshold or abs(g_neg) > threshold:
                direction = 1 if g_pos >= abs(g_neg) else -1
                events.append(self._event(diff.index[i], direction, "native_cpd", max(g_pos, abs(g_neg))))
                g_pos = g_neg = 0.0
                mu = x
                var = sigma ** 2
                n = 1
                continue

            n += 1
            delta = x - mu
            mu += delta / n
            var = ((n - 1) * var + delta * (x - mu)) / n

        return events

    def sprt(self, ds: pd.Series, q: float, warmup: int, drift: float, k: float) -> List[datetime.datetime]:
        return [event["timestamp"] for event in self.sprt_events(ds, q, warmup=warmup, drift=drift, k=k)]

    def sprt_events(self, ds: pd.Series, q: float, warmup: int, drift: float, k: float) -> List[Dict[str, Any]]:
        diff = ds.diff().dropna()
        if len(diff) < warmup + 1:
            return []

        values = diff.values
        mu = np.mean(values[:warmup])
        var = np.var(values[:warmup])

        if drift is None:
            drift = 0.5 * np.sqrt(var)

        g = 0.0
        events = []
        n = warmup

        for i in range(warmup, len(values)):
            x = values[i]
            sigma = np.sqrt(max(var, 1e-8))
            h = q * k * sigma
            a = -h
            z = x - mu
            g = g + z - drift

            if g > h or g < a:
                events.append(self._event(diff.index[i], 1 if g > h else -1, "native_cpd", abs(g)))
                g = 0.0
                mu = x
                var = sigma ** 2
                n = 1
                continue

            n += 1
            delta = x - mu
            mu += delta / n
            var = ((n - 1) * var + delta * (x - mu)) / n

        return events

    def gma(self, ds: pd.Series, q: float, lam: float, k: float, warmup: int = 20) -> List[datetime.datetime]:
        return [event["timestamp"] for event in self.gma_events(ds, q, lam=lam, k=k, warmup=warmup)]

    def gma_events(self, ds: pd.Series, q: float, lam: float, k: float, warmup: int = 20) -> List[Dict[str, Any]]:
        diff = ds.diff().dropna()
        if len(diff) < warmup + 1:
            return []

        values = diff.values
        mu = np.mean(values[:warmup])
        var = np.var(values[:warmup])

        g = 0.0
        events = []
        n = warmup

        for i in range(warmup, len(values)):
            x = values[i]
            sigma = np.sqrt(max(var, 1e-8))
            threshold = q * k * sigma
            z = x - mu
            g = lam * g + (1 - lam) * z

            if abs(g) > threshold:
                events.append(self._event(diff.index[i], self._direction_from_value(g), "native_cpd", abs(g)))
                g = 0.0
                mu = x
                var = sigma ** 2
                n = 1
                continue

            n += 1
            delta = x - mu
            mu += delta / n
            var = ((n - 1) * var + delta * (x - mu)) / n

        return events

    def glr(self, ds: pd.Series, q: float, window: int = None, init_len: int = 20,min_right_len = 5) -> List[datetime.datetime]:
        return [
            event["timestamp"]
            for event in self.glr_events(ds, q, window=window, init_len=init_len, min_right_len=min_right_len)
        ]

    def glr_events(
        self,
        ds: pd.Series,
        q: float,
        window: int = None,
        init_len: int = 20,
        min_right_len: int = 5,
    ) -> List[Dict[str, Any]]:
        values = ds.values.astype(float)
        ret = values[1:] - values[:-1]
        n = len(ret)
        if n < 2:
            return []

        idx = ds.index
        chi2_09999 = chi2.ppf(0.9999, df=1)
        threshold = q * chi2_09999

        prefix_sum = np.concatenate(([0.0], np.cumsum(ret)))
        prefix_sq_sum = np.concatenate(([0.0], np.cumsum(ret * ret)))

        events = []
        start = 0

        for t in range(1, n):
            max_stat = -np.inf
            best_direction = 0
            k_start = start
            if window is not None:
                k_start = max(start, t - window)

            for k in range(k_start, t):
                L = k - start + 1
                Rlen = t - k
                if L < init_len or Rlen < min_right_len:
                    continue

                left_sum = prefix_sum[k + 1] - prefix_sum[start]
                left_sq_sum = prefix_sq_sum[k + 1] - prefix_sq_sum[start]
                mean_left = left_sum / L
                var_left = (left_sq_sum - left_sum * mean_left) / (L - 1)
                if var_left < 1e-8:
                    continue

                right_sum = prefix_sum[t + 1] - prefix_sum[k + 1]
                mean_right = right_sum / Rlen
                stat = (mean_left - mean_right) ** 2 / (var_left * (1.0 / L + 1.0 / Rlen))
                if stat > max_stat:
                    max_stat = stat
                    best_direction = self._direction_from_value(mean_right - mean_left)

            if max_stat > threshold:
                cp_time = idx[t + 1]
                events.append(self._event(cp_time, best_direction, "inferred_cpd_mean_shift", max_stat))
                start = t + 1

        return events

    def brandt_glr(self, ds: pd.Series, q: float, alpha: float, window: int) -> List[datetime.datetime]:
        return [event["timestamp"] for event in self.brandt_glr_events(ds, q, alpha=alpha, window=window)]

    def brandt_glr_events(self, ds: pd.Series, q: float, alpha: float, window: int) -> List[Dict[str, Any]]:
        L = window
        n_total = len(ds)
        if L < 2 or n_total < L:
            return []

        threshold = chi2.ppf(1 - alpha, df=1) * 10 * q
        events = []

        # 环形缓冲区，用于记录窗口内的值
        buf = [0.0] * L
        pos = 0          # 下次写入位置
        win_count = 0    # 当前窗口内元素个数
        win_sum = 0.0    # 窗口内元素的和

        # global_data 的增量统计量
        global_n = 0
        global_sum = 0.0
        global_sq_sum = 0.0

        for i, y in enumerate(ds.values):
            y = float(y)

            # --- 1. 更新窗口 ---
            if win_count == L:                     # 窗口已满，需要先弹出旧值
                old_val = buf[pos]                 # 即将被移出窗口的值
                # 移入 global_data
                global_sum += old_val
                global_sq_sum += old_val * old_val
                global_n += 1
                # 用新值替换旧值
                win_sum += y - old_val
                buf[pos] = y
                pos = (pos + 1) % L
            else:                                  # 窗口未满，只添加
                buf[pos] = y
                pos = (pos + 1) % L
                win_count += 1
                win_sum += y

            # --- 2. 检测变点 ---
            if win_count == L and global_n >= 5:
                global_mean = global_sum / global_n
                # 样本方差 (ddof=1)
                var = (global_sq_sum - global_sum * global_mean) / (global_n - 1)
                if var == 0:
                    continue

                win_mean = win_sum / L
                g = L * (win_mean - global_mean) ** 2 / var

                if g > threshold:
                    events.append(
                        self._event(
                            ds.index[i],
                            self._direction_from_value(win_mean - global_mean),
                            "inferred_cpd_mean_shift",
                            g,
                        )
                    )

                    # 重置状态，窗口只保留当前点 y
                    global_n = 0
                    global_sum = 0.0
                    global_sq_sum = 0.0

                    win_count = 1
                    win_sum = y
                    buf = [0.0] * L
                    buf[0] = y
                    pos = 1
                    # 跳过本次循环剩余部分
                    continue

        return events

    def shewhart(self, ds: pd.Series, q: float, window: int, c_lim: int,k: float) -> List[datetime.datetime]:
        return [event["timestamp"] for event in self.shewhart_events(ds, q, window=window, c_lim=c_lim, k=k)]

    def shewhart_events(self, ds: pd.Series, q: float, window: int, c_lim: int, k: float) -> List[Dict[str, Any]]:
        n = len(ds)
        if n < window:
            return []

        events = []
        idx = 0
        while idx < n:
            if idx + window > n:
                break
            seg = ds.iloc[idx:idx + window].values.astype(float)
            mu0 = np.mean(seg)
            sigma0 = np.std(seg, ddof=1)
            t = idx + window
            consecutive = 0
            candidate_break = None
            candidate_direction = 0
            candidate_score = np.nan
            while t < n:
                y = ds.iloc[t]
                score = abs(y - mu0) / max(sigma0, 1e-8)
                if abs(y - mu0) > q * k * sigma0:
                    consecutive += 1
                    if consecutive == 1:
                        candidate_break = ds.index[t]
                        candidate_direction = self._direction_from_value(float(y - mu0))
                        candidate_score = score
                    if consecutive >= c_lim:
                        events.append(
                            self._event(
                                candidate_break,
                                candidate_direction,
                                "inferred_cpd_level_shift",
                                candidate_score,
                            )
                        )
                        idx = t - c_lim + 1
                        break
                else:
                    consecutive = 0
                    candidate_break = None
                t += 1
            else:
                break

        return events

    def recursive_param_seg(self, ds: pd.Series, q: float, M: int, split_top_k: int, min_seg_len: int) -> List[datetime.datetime]:
        return [
            event["timestamp"]
            for event in self.recursive_param_seg_events(
                ds, q, M=M, split_top_k=split_top_k, min_seg_len=min_seg_len
            )
        ]

    def recursive_param_seg_events(
        self,
        ds: pd.Series,
        q: float,
        M: int,
        split_top_k: int,
        min_seg_len: int,
    ) -> List[Dict[str, Any]]:
        import heapq

        if len(ds) < min_seg_len * 2:
            return []

        class Hypothesis:
            __slots__ = ('curr_sum', 'curr_sum2', 'curr_n', 'total_rss', 'total_n', 'bic', 'change_points', 'n_seg')

            def __init__(self):
                self.curr_sum = 0.0
                self.curr_sum2 = 0.0
                self.curr_n = 0
                self.total_rss = 0.0
                self.total_n = 0
                self.bic = np.inf
                self.change_points = []
                self.n_seg = 0

            def clone(self):
                h = Hypothesis()
                h.curr_sum = self.curr_sum
                h.curr_sum2 = self.curr_sum2
                h.curr_n = self.curr_n
                h.total_rss = self.total_rss
                h.total_n = self.total_n
                h.bic = self.bic
                h.change_points = self.change_points.copy()
                h.n_seg = self.n_seg
                return h

        def calc_rss(sum_, sum2_, n):
            if n <= 1:
                return 1e-8
            mean = sum_ / n
            rss = sum2_ - 2 * mean * sum_ + n * mean * mean
            return max(rss, 1e-8)

        def calc_bic(rss, n, n_seg):
            if n <= 1:
                return np.inf
            rss = max(rss, 1e-8)
            return n * np.log(rss / n) + q * 2 * n_seg * np.log(n)

        hypotheses = [Hypothesis()]
        values = ds.values
        times = ds.index

        for t_idx, y in enumerate(values):
            candidates = []
            for hyp in hypotheses:
                h = hyp.clone()
                h.curr_sum += y
                h.curr_sum2 += y * y
                h.curr_n += 1
                curr_rss = calc_rss(h.curr_sum, h.curr_sum2, h.curr_n)
                total_rss = h.total_rss + curr_rss
                total_n = h.total_n + h.curr_n
                h.bic = calc_bic(total_rss, total_n, h.n_seg + 1)
                candidates.append(h)

            parents = heapq.nsmallest(split_top_k, candidates, key=lambda h: h.bic)
            for parent in parents:
                if parent.curr_n < min_seg_len:
                    continue
                child = parent.clone()
                curr_rss = calc_rss(child.curr_sum, child.curr_sum2, child.curr_n)
                child.total_rss += curr_rss
                child.total_n += child.curr_n
                child.curr_sum = 0.0
                child.curr_sum2 = 0.0
                child.curr_n = 0
                child.n_seg += 1
                if t_idx + 1 < len(times):
                    child.change_points.append(times[t_idx + 1])
                child.bic = calc_bic(child.total_rss, child.total_n, child.n_seg)
                candidates.append(child)

            candidates.sort(key=lambda h: h.bic)
            unique = []
            seen = set()
            for h in candidates:
                key = (tuple(h.change_points), h.curr_n)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(h)
                if len(unique) >= M:
                    break
            hypotheses = unique

        for h in hypotheses:
            if h.curr_n == 0:
                continue
            curr_rss = calc_rss(h.curr_sum, h.curr_sum2, h.curr_n)
            total_rss = h.total_rss + curr_rss
            total_n = h.total_n + h.curr_n
            h.bic = calc_bic(total_rss, total_n, h.n_seg + 1)

        hypotheses.sort(key=lambda h: h.bic)
        change_points = hypotheses[0].change_points
        return [
            self._event(
                cp,
                self._infer_direction_from_windows(ds, cp, lookback=min_seg_len),
                "inferred_cpd_segment_mean",
                np.nan,
            )
            for cp in change_points
        ]

    def e_detector(self, x, q: float = 0.5, vol_window=10, warmup=20, lambda_grid=(0.01, 0.02, 0.05, 0.1), clip=5.0, two_sided=True):
        return [
            event["timestamp"]
            for event in self.e_detector_events(
                x, q=q, vol_window=vol_window, warmup=warmup,
                lambda_grid=lambda_grid, clip=clip, two_sided=two_sided
            )
        ]

    def e_detector_events(self, x, q: float = 0.5, vol_window=10, warmup=20, lambda_grid=(0.01, 0.02, 0.05, 0.1), clip=5.0, two_sided=True):
        values = x.values
        r = np.diff(values)
        vol = pd.Series(r).rolling(vol_window).std().values
        z = r / (vol + 1e-8)
        z = np.clip(z, -clip, clip)

        lambdas = list(lambda_grid)
        if two_sided:
            lambdas = sorted(set(lambdas + [-l for l in lambdas]))
        K = len(lambdas)

        M = {lam: 1.0 for lam in lambdas}
        threshold = np.exp(q)
        events = []
        start = max(warmup, vol_window)

        for t in range(start, len(z)):
            zt = z[t]
            if np.isnan(zt):
                continue
            total_evidence = 0.0
            best_lam = 0.0
            best_evidence = -np.inf
            for lam in lambdas:
                L = np.exp(lam * zt - 0.5 * lam**2)
                M[lam] = L * max(M[lam], 1.0)
                total_evidence += M[lam]
                if M[lam] > best_evidence:
                    best_evidence = M[lam]
                    best_lam = lam
            total_evidence /= K
            if total_evidence > threshold:
                direction = self._direction_from_value(best_lam)
                if direction == 0:
                    direction = self._direction_from_value(zt)
                events.append(self._event(x.index[t + 1], direction, "native_cpd_e_value", total_evidence))
                M = {lam: 1.0 for lam in lambdas}

        return events

    def ssr_cusum(self, ds: pd.Series, q: float, zeta: float, h0: float) -> List[datetime.datetime]:
        return [event["timestamp"] for event in self.ssr_cusum_events(ds, q, zeta=zeta, h0=h0)]

    def ssr_cusum_events(self, ds: pd.Series, q: float, zeta: float, h0: float) -> List[Dict[str, Any]]:
        if len(ds) < 2:
            return []

        values = ds.values.astype(float)
        seg_len = 0
        abs_vals = []
        Dpos = 0.0
        Dneg = 0.0
        events = []

        for global_i in range(1, len(values)):
            ret = values[global_i] - values[global_i - 1]
            x = ret
            s = 1 if x >= 0 else -1
            ax = abs(x)
            rank = 1
            for v in abs_vals:
                if v <= ax:
                    rank += 1
            abs_vals.append(ax)
            ii = seg_len + 1
            denom = (2 * ii + 1) * (ii + 1)
            xi = s * rank * np.sqrt(6.0 / denom)
            Dpos = max(0.0, Dpos + xi - zeta)
            Dneg = min(0.0, Dneg + xi + zeta)
            h = q * h0
            if Dpos > h or Dneg < -h:
                direction = 1 if Dpos >= abs(Dneg) else -1
                events.append(self._event(ds.index[global_i], direction, "native_cpd", max(Dpos, abs(Dneg))))
                Dpos = 0.0
                Dneg = 0.0
                abs_vals = []
                seg_len = 0
            else:
                seg_len += 1

        return events

    def adaptive_cusum(self, ds: pd.Series, q: float,
                       h0: float = 5.0,
                       vol_window: int = 30,
                       warmup: int = None,
                       rho_mu: float = 0.25,
                       s: float = 1.0,
                       eta: float = 4.0,
                       rho_theta_up: float = 1.05,
                       rho_theta_down: float = 0.95238) -> List[datetime.datetime]:
        return [
            event["timestamp"]
            for event in self.adaptive_cusum_events(
                ds, q, h0=h0, vol_window=vol_window, warmup=warmup,
                rho_mu=rho_mu, s=s, eta=eta,
                rho_theta_up=rho_theta_up, rho_theta_down=rho_theta_down,
            )
        ]

    def adaptive_cusum_events(self, ds: pd.Series, q: float,
                              h0: float = 5.0,
                              vol_window: int = 30,
                              warmup: int = None,
                              rho_mu: float = 0.25,
                              s: float = 1.0,
                              eta: float = 4.0,
                              rho_theta_up: float = 1.05,
                              rho_theta_down: float = 0.95238) -> List[Dict[str, Any]]:
        if len(ds) < 2:
            return []

        if warmup is None:
            warmup = vol_window

        prices = ds.values.astype(float)
        n = len(prices)
        window_vals = deque(maxlen=vol_window)
        sum_win = 0.0
        sum_sq_win = 0.0

        dirs = []
        for m in ['+', '-', '.']:
            for v in ['+', '-', '.']:
                if m == '.' and v == '.':
                    continue
                dirs.append((m, v))

        C = {d: 0.0 for d in dirs}
        tau = {d: 0 for d in dirs}
        N = {d: 0 for d in dirs}
        S = {d: 0.0 for d in dirs}
        Q = {d: 0.0 for d in dirs}
        mu_hat = {}
        theta_hat = {}

        alpha_up = 12.0
        beta_up = 15.0
        alpha_down = 16.3
        beta_down = 15.0

        for d in dirs:
            m_dir, v_dir = d
            if m_dir == '+':
                mu_hat[d] = rho_mu
            elif m_dir == '-':
                mu_hat[d] = -rho_mu
            else:
                mu_hat[d] = 0.0
            if v_dir == '+':
                theta_hat[d] = rho_theta_up
            elif v_dir == '-':
                theta_hat[d] = rho_theta_down
            else:
                theta_hat[d] = 1.0

        events = []
        time_index = ds.index[1:]

        for t_idx in range(1, n):
            ret = prices[t_idx] - prices[t_idx - 1]
            if len(window_vals) < vol_window:
                window_vals.append(ret)
                sum_win += ret
                sum_sq_win += ret * ret
                continue

            if len(window_vals) == vol_window:
                oldest = window_vals[0]
                sum_win -= oldest
                sum_sq_win -= oldest * oldest
            window_vals.append(ret)
            sum_win += ret
            sum_sq_win += ret * ret

            mean_win = sum_win / vol_window
            var_win = (sum_sq_win - vol_window * mean_win * mean_win) / (vol_window - 1)
            std_win = np.sqrt(max(var_win, 1e-8))
            x = ret / std_win
            maxC = 0.0
            best_d = None

            for d in dirs:
                m_dir, v_dir = d
                llr = 0.5 * (x * x - (x - mu_hat[d]) ** 2 / theta_hat[d] - np.log(theta_hat[d]))
                C[d] = max(0.0, C[d] + llr)

                if C[d] > 0:
                    N[d] += 1
                    S[d] += x
                    if m_dir == '+':
                        mu_hat[d] = max(rho_mu, (s + S[d]) / (eta + N[d]))
                    elif m_dir == '-':
                        mu_hat[d] = min(-rho_mu, (-s + S[d]) / (eta + N[d]))
                    Q[d] += (x - mu_hat[d]) ** 2
                    if v_dir == '+':
                        num = beta_up + Q[d] / 2.0
                        den = alpha_up - 1.0 + N[d] / 2.0
                        theta_hat[d] = max(rho_theta_up, num / den)
                    elif v_dir == '-':
                        num = beta_down + Q[d] / 2.0
                        den = alpha_down - 1.0 + N[d] / 2.0
                        theta_hat[d] = min(rho_theta_down, num / den)
                else:
                    tau[d] = t_idx
                    N[d] = 0
                    S[d] = 0.0
                    Q[d] = 0.0
                    if m_dir == '+':
                        mu_hat[d] = rho_mu
                    elif m_dir == '-':
                        mu_hat[d] = -rho_mu
                    else:
                        mu_hat[d] = 0.0
                    if v_dir == '+':
                        theta_hat[d] = rho_theta_up
                    elif v_dir == '-':
                        theta_hat[d] = rho_theta_down
                    else:
                        theta_hat[d] = 1.0

                if C[d] > maxC:
                    maxC = C[d]
                    best_d = d

            if maxC > q * h0:
                m_dir = best_d[0] if best_d is not None else "."
                if m_dir == "+":
                    direction = 1
                elif m_dir == "-":
                    direction = -1
                else:
                    direction = self._direction_from_value(ret)
                events.append(self._event(time_index[t_idx - 1], direction, "native_cpd_adaptive", maxC))
                for d in dirs:
                    C[d] = 0.0
                    tau[d] = t_idx
                    N[d] = 0
                    S[d] = 0.0
                    Q[d] = 0.0
                    m_dir, v_dir = d
                    if m_dir == '+':
                        mu_hat[d] = rho_mu
                    elif m_dir == '-':
                        mu_hat[d] = -rho_mu
                    else:
                        mu_hat[d] = 0.0
                    if v_dir == '+':
                        theta_hat[d] = rho_theta_up
                    elif v_dir == '-':
                        theta_hat[d] = rho_theta_down
                    else:
                        theta_hat[d] = 1.0

        return events

def generate_breaks(ds: pd.Series, brk: Brk) -> List[datetime.datetime]:
    if not isinstance(ds, pd.Series):
        raise TypeError("ds must be a pandas Series")
    if not isinstance(ds.index, pd.DatetimeIndex):
        raise ValueError("Series index must be datetime type")
    return brk.detect(ds)


__all__ = ['Brk', 'generate_breaks']
