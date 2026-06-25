import numpy as np
import pandas as pd
import datetime
from typing import Union
TYPE_DT = Union[datetime.datetime, datetime.date]


class FuncEvt:
    """
    Breakpoint sampling from incremental series (dif = diff of log-price).

    All methods share the signature:
        func(dif, vol, *, b2b) -> list[datetime]
    where:
        dif: pd.Series  - incremental series (returns / log-price diffs)
        vol: pd.Series | float - controls sampling frequency (higher = fewer events)
        b2b: int - number of consecutive events

    Active methods:
        swt(dif, vol, *, b2b) - Shewhart
        cum(dif, vol, *, b2b) - CUSUM
    """

    @classmethod
    def swt(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1, window: int = 20) -> list[TYPE_DT]:
        positions = []
        vol = vol if isinstance(vol, pd.Series) else pd.Series(vol, index=dif.index)
        val = np.concatenate(([0.0], np.cumsum(dif)))  # csum[k] == sum(vals[:k])
        i = window  # need a full trailing window before the first observation
        while i < len(dif):
            consecutive = 0
            while i < len(dif):
                mu0 = (val[i] - val[i - window]) / window
                if abs(dif[i] - mu0) > vol[i]:
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
        return [dif.index[p] for p in positions]

    @classmethod
    def cum(cls, dif: pd.Series, vol: pd.Series | float, *, b2b: int = 1) -> list[TYPE_DT]:
        positions = []
        vol = vol if isinstance(vol, pd.Series) else pd.Series(vol, index=dif.index)
        i = 0
        while i < len(dif):
            cum_p, cum_n = 0.0, 0.0
            consecutive = 0
            while i < len(dif):
                cum_p = max(0.0, cum_p + dif.iat[i])
                cum_n = max(0.0, cum_n - dif.iat[i])
                if max(cum_p, cum_n) > vol.iat[i]:
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
        return [dif.index[p] for p in positions]
