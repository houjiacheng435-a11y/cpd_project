"""Change point detection module extracted from CPD.ipynb.

This module exposes the Brk class and generate_breaks() helper.
"""

from __future__ import annotations

import datetime
from collections import deque
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .wbs_lepage import WBSLepageDetector


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
            'wbs_lepage', 'wbs'
        ]
        if self.method not in self._valid_methods:
            raise ValueError(f"Unsupported method: {method}. Choose from {self._valid_methods}")

    def detect(self, ds: pd.Series) -> List[datetime.datetime]:
        """Detect change points from a datetime-indexed pandas Series."""
        if self.method == 'cusum':
            h = self.kwargs.get('h', 0.03)
            nu = self.kwargs.get('nu', 0.00005)
            return self.cusum(ds, self.q, h=h, nu=nu)
        elif self.method == 'cusum_ls':
            warmup = self.kwargs.get('warmup', 30)
            drift = self.kwargs.get('drift', 0.003)
            k = self.kwargs.get('k', 5.0)
            return self.cusum_ls(ds, self.q, warmup=warmup, drift=drift, k=k)
        elif self.method == 'sprt':
            warmup = self.kwargs.get('warmup', 20)
            drift = self.kwargs.get('drift', 0)
            k = self.kwargs.get('k', 30.0)
            return self.sprt(ds, self.q, warmup=warmup, drift=drift, k=k)
        elif self.method == 'gma':
            lam = self.kwargs.get('lam', 0.9)
            k = self.kwargs.get('k', 2.0)
            return self.gma(ds, self.q, lam, k)
        elif self.method == 'glr':
            window = self.kwargs.get('window', None)
            init_len = self.kwargs.get('init_len', 20)
            min_right_len = self.kwargs.get('min_right_len', 5)
            return self.glr(ds, self.q, window=window, init_len=init_len, min_right_len=min_right_len)
        elif self.method == 'brandt_glr':
            window = self.kwargs.get('window', 10)
            alpha = self.kwargs.get('alpha', 0.0001)
            return self.brandt_glr(ds, self.q, window=window, alpha=alpha)
        elif self.method == 'shewhart':
            window = self.kwargs.get('window', 24)
            c_lim = self.kwargs.get('c_lim', 5)
            k = self.kwargs.get('k', 10.0)
            return self.shewhart(ds, self.q, window=window, c_lim=c_lim, k=k)
        elif self.method == 'recursive_param_seg':
            M = self.kwargs.get('M', 3)
            split_top_k = self.kwargs.get('split_top_k', 5)
            min_seg_len = self.kwargs.get('min_seg_len', 10)
            return self.recursive_param_seg(ds, self.q, M=M, split_top_k=split_top_k, min_seg_len=min_seg_len)
        elif self.method == 'e_detector':
            vol_window = self.kwargs.get('vol_window', 30)
            warmup = self.kwargs.get('warmup', 50)
            lambda_grid = self.kwargs.get('lambda_grid', (0.01, 0.02, 0.05, 0.1))
            clip = self.kwargs.get('clip', 5.0)
            two_sided = self.kwargs.get('two_sided', True)
            return self.e_detector(ds, q=self.q, vol_window=vol_window, warmup=warmup,
                                   lambda_grid=lambda_grid, clip=clip, two_sided=two_sided)
        elif self.method == 'ssr_cusum':
            zeta = self.kwargs.get('zeta', 0.25)
            h0 = self.kwargs.get('h0', 10.0)
            return self.ssr_cusum(ds, self.q, zeta=zeta, h0=h0)
        elif self.method == 'adaptive_cusum':
            h0 = self.kwargs.get('h0', 10.0)
            vol_window = self.kwargs.get('vol_window', 30)
            warmup = self.kwargs.get('warmup', None)
            rho_mu = self.kwargs.get('rho_mu', 0.25)
            s = self.kwargs.get('s', 1.0)
            eta = self.kwargs.get('eta', 4.0)
            rho_theta_up = self.kwargs.get('rho_theta_up', 1.05)
            rho_theta_down = self.kwargs.get('rho_theta_down', 1 / 1.05)
            return self.adaptive_cusum(ds, self.q, h0=h0, vol_window=vol_window, warmup=warmup,
                                       rho_mu=rho_mu, s=s, eta=eta,
                                       rho_theta_up=rho_theta_up, rho_theta_down=rho_theta_down)
        elif self.method in ('wbs_lepage', 'wbs'):
            M = self.kwargs.get('M', 10000)
            alpha = self.kwargs.get('alpha', 0.05)
            min_seg_len = self.kwargs.get('min_seg_len', 10)
            prune = self.kwargs.get('prune', True)
            return self.wbs_lepage(ds, self.q, M=M, alpha=alpha,
                                   min_seg_len=min_seg_len, prune=prune)

    def cusum(self, ds: pd.Series, q: float, h: float, nu: float) -> List[datetime.datetime]:
        diff = ds.diff().dropna()
        if len(diff) == 0:
            return []

        values = diff.values
        g_pos = 0.0
        g_neg = 0.0
        breaks = []

        for idx, x in enumerate(values):
            g_pos = max(0.0, g_pos + x - nu)
            g_neg = max(0.0, g_neg - x - nu)
            if g_pos > q * h or g_neg > q * h:
                breaks.append(diff.index[idx])
                g_pos = 0.0
                g_neg = 0.0

        return breaks

    def cusum_ls(self, ds: pd.Series, q: float, warmup: int, drift: float,k: float) -> List[datetime.datetime]:
        diff = ds.diff().dropna()
        if len(diff) < warmup + 1:
            return []
        values = diff.values

        mu = np.mean(values[:warmup])
        var = np.var(values[:warmup])

        g_pos = 0.0
        g_neg = 0.0
        breaks = []
        n = warmup

        for i in range(warmup, len(values)):
            x = values[i]
            sigma = np.sqrt(max(var, 1e-8))
            threshold = q * k * sigma
            z = x - mu

            g_pos = max(0.0, g_pos + z - drift)
            g_neg = min(0.0, g_neg + z + drift)

            if g_pos > threshold or abs(g_neg) > threshold:
                breaks.append(diff.index[i])
                g_pos = g_neg = 0.0
                mu = x
                var = sigma ** 2
                n = 1
                continue

            n += 1
            delta = x - mu
            mu += delta / n
            var = ((n - 1) * var + delta * (x - mu)) / n

        return breaks

    def sprt(self, ds: pd.Series, q: float, warmup: int, drift: float, k: float) -> List[datetime.datetime]:
        diff = ds.diff().dropna()
        if len(diff) < warmup + 1:
            return []

        values = diff.values
        mu = np.mean(values[:warmup])
        var = np.var(values[:warmup])

        if drift is None:
            drift = 0.5 * np.sqrt(var)

        g = 0.0
        breaks = []
        n = warmup

        for i in range(warmup, len(values)):
            x = values[i]
            sigma = np.sqrt(max(var, 1e-8))
            h = q * k * sigma
            a = -h
            z = x - mu
            g = g + z - drift

            if g > h or g < a:
                breaks.append(diff.index[i])
                g = 0.0
                mu = x
                var = sigma ** 2
                n = 1
                continue

            n += 1
            delta = x - mu
            mu += delta / n
            var = ((n - 1) * var + delta * (x - mu)) / n

        return breaks

    def gma(self, ds: pd.Series, q: float, lam: float, k: float, warmup: int = 20) -> List[datetime.datetime]:
        diff = ds.diff().dropna()
        if len(diff) < warmup + 1:
            return []

        values = diff.values
        mu = np.mean(values[:warmup])
        var = np.var(values[:warmup])

        g = 0.0
        breaks = []
        n = warmup

        for i in range(warmup, len(values)):
            x = values[i]
            sigma = np.sqrt(max(var, 1e-8))
            threshold = q * k * sigma
            z = x - mu
            g = lam * g + (1 - lam) * z

            if abs(g) > threshold:
                breaks.append(diff.index[i])
                g = 0.0
                mu = x
                var = sigma ** 2
                n = 1
                continue

            n += 1
            delta = x - mu
            mu += delta / n
            var = ((n - 1) * var + delta * (x - mu)) / n

        return breaks

    def glr(self, ds: pd.Series, q: float, window: int = None, init_len: int = 20,min_right_len = 5) -> List[datetime.datetime]:
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

        breaks = []
        start = 0

        for t in range(1, n):
            max_stat = -np.inf
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

            if max_stat > threshold:
                cp_time = idx[t + 1]
                breaks.append(cp_time)
                start = t + 1

        return breaks

    def brandt_glr(self, ds: pd.Series, q: float, alpha: float, window: int) -> List[datetime.datetime]:
        L = window
        n_total = len(ds)
        if L < 2 or n_total < L:
            return []

        threshold = chi2.ppf(1 - alpha, df=1) * 10 * q
        breaks = []

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
                    breaks.append(ds.index[i])

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

        return breaks

    def shewhart(self, ds: pd.Series, q: float, window: int, c_lim: int,k: float) -> List[datetime.datetime]:
        n = len(ds)
        if n < window:
            return []

        breaks = []
        idx = 0
        while idx < n:
            if idx + window > n:
                break
            seg = ds.iloc[idx:idx + window].values.astype(float)
            mu0 = np.mean(seg)
            sigma0 = np.std(seg, ddof=1)
            t = idx + window
            consecutive = 0
            while t < n:
                y = ds.iloc[t]
                if abs(y - mu0) > q * k * sigma0:
                    consecutive += 1
                    if consecutive >= c_lim:
                        breaks.append(ds.index[t])
                        idx = t - c_lim + 1
                        break
                else:
                    consecutive = 0
                t += 1
            else:
                break

        return breaks

    def recursive_param_seg(self, ds: pd.Series, q: float, M: int, split_top_k: int, min_seg_len: int) -> List[datetime.datetime]:
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
        return [times[-1] for _ in change_points]

    def e_detector(self, x, q: float = 0.5, vol_window=10, warmup=20, lambda_grid=(0.01, 0.02, 0.05, 0.1), clip=5.0, two_sided=True):
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
        cp = []
        start = max(warmup, vol_window)

        for t in range(start, len(z)):
            zt = z[t]
            if np.isnan(zt):
                continue
            total_evidence = 0.0
            for lam in lambdas:
                L = np.exp(lam * zt - 0.5 * lam**2)
                M[lam] = L * max(M[lam], 1.0)
                total_evidence += M[lam]
            total_evidence /= K
            if total_evidence > threshold:
                cp.append(x.index[t + 1])
                M = {lam: 1.0 for lam in lambdas}

        return cp

    def ssr_cusum(self, ds: pd.Series, q: float, zeta: float, h0: float) -> List[datetime.datetime]:
        if len(ds) < 2:
            return []

        values = ds.values.astype(float)
        seg_len = 0
        abs_vals = []
        Dpos = 0.0
        Dneg = 0.0
        breaks = []

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
                breaks.append(ds.index[global_i])
                Dpos = 0.0
                Dneg = 0.0
                abs_vals = []
                seg_len = 0
            else:
                seg_len += 1

        return breaks

    def adaptive_cusum(self, ds: pd.Series, q: float,
                       h0: float = 5.0,
                       vol_window: int = 30,
                       warmup: int = None,
                       rho_mu: float = 0.25,
                       s: float = 1.0,
                       eta: float = 4.0,
                       rho_theta_up: float = 1.05,
                       rho_theta_down: float = 0.95238) -> List[datetime.datetime]:
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

        breaks = []
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

            if maxC > q * h0:
                breaks.append(time_index[t_idx - 1])
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

        return breaks

    def wbs_lepage(self, ds: pd.Series, q: float = 1.0, M: int = 10000,
                   alpha: float = 0.05, min_seg_len: int = 10, prune: bool = True) -> List[datetime.datetime]:
        ret = ds.diff().dropna()
        if len(ret) < 2 * min_seg_len:
            return []

        detector = WBSLepageDetector(M=M, alpha=alpha, min_seg_len=min_seg_len,
                                     precompute=True, max_len=len(ret), nsim=200)
        indices = detector.detect(series=ret.values, q=q, prune=prune)
        return [ds.index[-1] for i in indices if i + 1 < len(ds)]

def generate_breaks(ds: pd.Series, brk: Brk) -> List[datetime.datetime]:
    if not isinstance(ds, pd.Series):
        raise TypeError("ds must be a pandas Series")
    if not isinstance(ds.index, pd.DatetimeIndex):
        raise ValueError("Series index must be datetime type")
    return brk.detect(ds)


__all__ = ['Brk', 'generate_breaks']
