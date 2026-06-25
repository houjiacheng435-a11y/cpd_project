import datetime
import bisect
from collections import deque
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd


TYPE_DT = Union[datetime.datetime, datetime.date]


class FuncEvtCPD:
    """
    Event sampling methods for an incremental series.

    Each active method shares this shape:
        func(dif, vol, *, b2b=1, **kwargs) -> list[datetime]

    `dif` is the incremental series. 
    `vol` is an external threshold sequence or scalar that directly controls event frequency. 
    Methods compute their own statistic and trigger when statistic > vol[i].
    """

    @staticmethod
    def _prep(dif: pd.Series, vol: pd.Series | float) -> tuple[pd.Series, np.ndarray]:
        if not isinstance(dif, pd.Series):
            raise TypeError("dif must be a pandas Series")
        dif = dif.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if isinstance(vol, pd.Series):
            threshold = vol.reindex(dif.index).astype(float)
            threshold = threshold.ffill().bfill()
            vol_arr = threshold.to_numpy(dtype=float)
        else:
            vol_arr = np.full(len(dif), float(vol), dtype=float)
        if len(vol_arr) != len(dif):
            raise ValueError("vol must be scalar or aligned to dif.index")
        vol_arr = np.where(np.isfinite(vol_arr), vol_arr, np.inf)
        vol_arr = np.maximum(vol_arr, 0.0)
        return dif, vol_arr

    @staticmethod
    def _times(dif: pd.Series, positions: list[int]) -> list[TYPE_DT]:
        return [dif.index[p] for p in positions if 0 <= p < len(dif)]

    @classmethod
    def swt(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, window: int = 20) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        positions: list[int] = []
        i = int(window)
        while i < len(values):
            consecutive = 0
            while i < len(values):
                mu0 = (prefix[i] - prefix[i - window]) / window
                if abs(values[i] - mu0) > vol_arr[i]:
                    consecutive += 1
                    if consecutive >= b2b:
                        positions.append(i)
                        i += 1
                        break
                else:
                    consecutive = 0
                i += 1
            else:
                break
        return cls._times(dif, positions)

    @classmethod
    def cum(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, drift: float = 0.0) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        positions: list[int] = []
        i = 0
        while i < len(values):
            cum_p, cum_n = 0.0, 0.0
            consecutive = 0
            while i < len(values):
                x = values[i]
                cum_p = max(0.0, cum_p + x - drift)
                cum_n = max(0.0, cum_n - x - drift)
                if max(cum_p, cum_n) > vol_arr[i]:
                    consecutive += 1
                    if consecutive >= b2b:
                        positions.append(i)
                        i += 1
                        break
                else:
                    consecutive = 0
                i += 1
            else:
                break
        return cls._times(dif, positions)

    @classmethod
    def cusum_ls(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, warmup: int = 30, drift: float = 0.0) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        positions: list[int] = []
        i = int(warmup)
        while i < len(values):
            mu = float(np.mean(values[max(0, i - warmup):i]))
            var = float(np.var(values[max(0, i - warmup):i]))
            n = max(1, min(warmup, i))
            g_pos = 0.0
            g_neg = 0.0
            consecutive = 0
            while i < len(values):
                x = values[i]
                z = x - mu
                g_pos = max(0.0, g_pos + z - drift)
                g_neg = min(0.0, g_neg + z + drift)
                if max(g_pos, abs(g_neg)) > vol_arr[i]:
                    consecutive += 1
                    if consecutive >= b2b:
                        positions.append(i)
                        i += 1
                        break
                else:
                    consecutive = 0
                n += 1
                delta = x - mu
                mu += delta / n
                var = ((n - 1) * var + delta * (x - mu)) / n
                i += 1
            else:
                break
        return cls._times(dif, positions)

    @classmethod
    def sprt(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, warmup: int = 20, drift: float = 0.0) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        positions: list[int] = []
        i = int(warmup)
        while i < len(values):
            mu = float(np.mean(values[max(0, i - warmup):i]))
            var = float(np.var(values[max(0, i - warmup):i]))
            n = max(1, min(warmup, i))
            g = 0.0
            consecutive = 0
            while i < len(values):
                x = values[i]
                z = x - mu
                g = g + z - drift
                if abs(g) > vol_arr[i]:
                    consecutive += 1
                    if consecutive >= b2b:
                        positions.append(i)
                        i += 1
                        break
                else:
                    consecutive = 0
                n += 1
                delta = x - mu
                mu += delta / n
                var = ((n - 1) * var + delta * (x - mu)) / n
                i += 1
            else:
                break
        return cls._times(dif, positions)

    @classmethod
    def gma(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, warmup: int = 20, lam: float = 0.9) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        positions: list[int] = []
        i = int(warmup)
        while i < len(values):
            mu = float(np.mean(values[max(0, i - warmup):i]))
            var = float(np.var(values[max(0, i - warmup):i]))
            n = max(1, min(warmup, i))
            g = 0.0
            consecutive = 0
            while i < len(values):
                x = values[i]
                g = lam * g + (1.0 - lam) * (x - mu)
                if abs(g) > vol_arr[i]:
                    consecutive += 1
                    if consecutive >= b2b:
                        positions.append(i)
                        i += 1
                        break
                else:
                    consecutive = 0
                n += 1
                delta = x - mu
                mu += delta / n
                var = ((n - 1) * var + delta * (x - mu)) / n
                i += 1
            else:
                break
        return cls._times(dif, positions)

    @classmethod
    def glr(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, window: int = 60, init_len: int = 20, min_right_len: int = 5) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        n = len(values)
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        prefix2 = np.concatenate(([0.0], np.cumsum(values * values)))
        positions: list[int] = []
        start = 0
        consecutive = 0
        for t in range(1, n):
            max_stat = 0.0
            k_start = max(start, t - window) if window is not None else start
            for k in range(k_start, t):
                left_n = k - start + 1
                right_n = t - k
                if left_n < init_len or right_n < min_right_len:
                    continue
                left_sum = prefix[k + 1] - prefix[start]
                left_sum2 = prefix2[k + 1] - prefix2[start]
                left_mean = left_sum / left_n
                left_var = (left_sum2 - left_sum * left_mean) / max(left_n - 1, 1)
                if left_var <= 1e-12:
                    continue
                right_sum = prefix[t + 1] - prefix[k + 1]
                right_mean = right_sum / right_n
                stat = (left_mean - right_mean) ** 2 / (left_var * (1.0 / left_n + 1.0 / right_n))
                if stat > max_stat:
                    max_stat = stat
            if max_stat > vol_arr[t]:
                consecutive += 1
                if consecutive >= b2b:
                    positions.append(t)
                    start = t + 1
                    consecutive = 0
            else:
                consecutive = 0
        return cls._times(dif, positions)

    @classmethod
    def brandt_glr(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, window: int = 20, min_global: int = 5) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        positions: list[int] = []
        buf = [0.0] * window
        pos = 0
        win_count = 0
        win_sum = 0.0
        global_n = 0
        global_sum = 0.0
        global_sum2 = 0.0
        consecutive = 0
        for i, y in enumerate(values):
            if win_count == window:
                old = buf[pos]
                global_sum += old
                global_sum2 += old * old
                global_n += 1
                win_sum += y - old
                buf[pos] = y
                pos = (pos + 1) % window
            else:
                buf[pos] = y
                pos = (pos + 1) % window
                win_count += 1
                win_sum += y
            if win_count == window and global_n >= min_global:
                global_mean = global_sum / global_n
                var = (global_sum2 - global_sum * global_mean) / max(global_n - 1, 1)
                if var <= 1e-12:
                    continue
                win_mean = win_sum / window
                stat = window * (win_mean - global_mean) ** 2 / var
                if stat > vol_arr[i]:
                    consecutive += 1
                    if consecutive >= b2b:
                        positions.append(i)
                        global_n = 0
                        global_sum = 0.0
                        global_sum2 = 0.0
                        win_count = 1
                        win_sum = y
                        buf = [0.0] * window
                        buf[0] = y
                        pos = 1
                        consecutive = 0
                else:
                    consecutive = 0
        return cls._times(dif, positions)

    @classmethod
    def e_detector(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, std_window: int = 30, warmup: int = 50, lambda_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1), clip: float = 5.0, two_sided: bool = True) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        std = pd.Series(values).rolling(std_window, min_periods=max(2, std_window // 2)).std().to_numpy()
        z = values / (std + 1e-12)
        z = np.clip(z, -clip, clip)
        lambdas = list(lambda_grid)
        if two_sided:
            lambdas = sorted(set(lambdas + [-x for x in lambdas]))
        capital = {lam: 1.0 for lam in lambdas}
        positions: list[int] = []
        consecutive = 0
        for i in range(max(std_window, warmup), len(values)):
            if not np.isfinite(z[i]):
                continue
            total = 0.0
            for lam in lambdas:
                inc = np.exp(lam * z[i] - 0.5 * lam**2)
                capital[lam] = inc * max(capital[lam], 1.0)
                total += capital[lam]
            stat = total / len(lambdas)
            if stat > vol_arr[i]:
                consecutive += 1
                if consecutive >= b2b:
                    positions.append(i)
                    capital = {lam: 1.0 for lam in lambdas}
                    consecutive = 0
            else:
                consecutive = 0
        return cls._times(dif, positions)

    @classmethod
    def ssr_cusum(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, zeta: float = 0.25) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)
        positions: list[int] = []
        abs_vals: list[float] = []
        seg_len = 0
        d_pos = 0.0
        d_neg = 0.0
        consecutive = 0
        for i, x in enumerate(values):
            sign = 1 if x >= 0 else -1
            ax = abs(x)
            rank = bisect.bisect_right(abs_vals, ax) + 1
            bisect.insort(abs_vals, ax)
            ii = seg_len + 1
            denom = (2 * ii + 1) * (ii + 1)
            xi = sign * rank * np.sqrt(6.0 / denom)
            d_pos = max(0.0, d_pos + xi - zeta)
            d_neg = min(0.0, d_neg + xi + zeta)
            if max(d_pos, abs(d_neg)) > vol_arr[i]:
                consecutive += 1
                if consecutive >= b2b:
                    positions.append(i)
                    abs_vals = []
                    seg_len = 0
                    d_pos = 0.0
                    d_neg = 0.0
                    consecutive = 0
            else:
                seg_len += 1
                consecutive = 0
        return cls._times(dif, positions)

    @classmethod
    def adaptive_cusum(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, std_window: int = 30, rho_mu: float = 0.25, s: float = 1.0, eta: float = 4.0, rho_theta_up: float = 1.05, rho_theta_down: float = 1 / 1.05) -> list[TYPE_DT]:
        dif, vol_arr = cls._prep(dif, vol)
        values = dif.to_numpy(dtype=float)

        dirs = [(m, v) for m in ["+", "-", "."] for v in ["+", "-", "."] if not (m == "." and v == ".")]
        c = {d: 0.0 for d in dirs}
        n_obs = {d: 0 for d in dirs}
        s_obs = {d: 0.0 for d in dirs}
        q_obs = {d: 0.0 for d in dirs}
        mu_hat: dict[tuple[str, str], float] = {}
        theta_hat: dict[tuple[str, str], float] = {}
        for d in dirs:
            m_dir, v_dir = d
            mu_hat[d] = rho_mu if m_dir == "+" else -rho_mu if m_dir == "-" else 0.0
            theta_hat[d] = rho_theta_up if v_dir == "+" else rho_theta_down if v_dir == "-" else 1.0

        alpha_up, beta_up = 12.0, 15.0
        alpha_down, beta_down = 16.3, 15.0
        window_vals: deque[float] = deque(maxlen=std_window)
        sum_win = 0.0
        sum2_win = 0.0
        consecutive = 0
        positions: list[int] = []

        for i, x_raw in enumerate(values):
            if len(window_vals) == std_window:
                old = window_vals[0]
                sum_win -= old
                sum2_win -= old * old
            window_vals.append(x_raw)
            sum_win += x_raw
            sum2_win += x_raw * x_raw
            if len(window_vals) < std_window:
                continue
            mean_win = sum_win / std_window
            var_win = (sum2_win - std_window * mean_win * mean_win) / max(std_window - 1, 1)
            x = x_raw / np.sqrt(max(var_win, 1e-12))
            max_c = 0.0
            for d in dirs:
                m_dir, v_dir = d
                llr = 0.5 * (x * x - (x - mu_hat[d]) ** 2 / theta_hat[d] - np.log(theta_hat[d]))
                c[d] = max(0.0, c[d] + llr)
                if c[d] > 0:
                    n_obs[d] += 1
                    s_obs[d] += x
                    if m_dir == "+":
                        mu_hat[d] = max(rho_mu, (s + s_obs[d]) / (eta + n_obs[d]))
                    elif m_dir == "-":
                        mu_hat[d] = min(-rho_mu, (-s + s_obs[d]) / (eta + n_obs[d]))
                    q_obs[d] += (x - mu_hat[d]) ** 2
                    if v_dir == "+":
                        theta_hat[d] = max(rho_theta_up, (beta_up + q_obs[d] / 2.0) / (alpha_up - 1.0 + n_obs[d] / 2.0))
                    elif v_dir == "-":
                        theta_hat[d] = min(rho_theta_down, (beta_down + q_obs[d] / 2.0) / (alpha_down - 1.0 + n_obs[d] / 2.0))
                else:
                    n_obs[d] = 0
                    s_obs[d] = 0.0
                    q_obs[d] = 0.0
                    if m_dir == "+":
                        mu_hat[d] = rho_mu
                    elif m_dir == "-":
                        mu_hat[d] = -rho_mu
                    else:
                        mu_hat[d] = 0.0
                    if v_dir == "+":
                        theta_hat[d] = rho_theta_up
                    elif v_dir == "-":
                        theta_hat[d] = rho_theta_down
                    else:
                        theta_hat[d] = 1.0
                if c[d] > max_c:
                    max_c = c[d]
            if max_c > vol_arr[i]:
                consecutive += 1
                if consecutive >= b2b:
                    positions.append(i)
                    for d in dirs:
                        c[d] = 0.0
                        n_obs[d] = 0
                        s_obs[d] = 0.0
                        q_obs[d] = 0.0
                    consecutive = 0
            else:
                consecutive = 0
        return cls._times(dif, positions)

    @classmethod
    def methods(cls) -> Dict[str, Callable]:
        return {
            "swt": cls.swt,
            "cum": cls.cum,
            "cusum_ls": cls.cusum_ls,
            "sprt": cls.sprt,
            "gma": cls.gma,
            "glr": cls.glr,
            "brandt_glr": cls.brandt_glr,
            "e_detector": cls.e_detector,
            "ssr_cusum": cls.ssr_cusum,
            "adaptive_cusum": cls.adaptive_cusum,
        }

    @classmethod
    def detect(cls, method: str, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, **kwargs) -> list[TYPE_DT]:
        try:
            func = cls.methods()[method]
        except KeyError as exc:
            raise ValueError(f"Unsupported method: {method}") from exc
        return func(dif, vol, b2b=b2b, **kwargs)
