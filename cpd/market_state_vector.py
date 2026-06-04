"""Online market state vector construction based on CPD + Automatic 1-2-3 logic.

This module uses the MSV-specific CPD detector to obtain online Direction
events, then follows an Automatic One-Two-Three style state machine:

    Direction -> Excep -> Status -> MinMax process -> Trend structure -> State vector

The implementation is strictly online.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .brk_detector_for_msv import Brk
from .utils import (
    _argmax,
    _argmin,
    _ensure_ohlc_frame,
    load_data,
    plot_price_state_background,
)

# ---------------------------------------------------------------------------
# Structural state
# ---------------------------------------------------------------------------


@dataclass
class Extremum:
    """One confirmed relevant extremum in the online MinMax process."""

    kind: str  # "high" or "low"
    idx: int
    timestamp: pd.Timestamp
    value: float


@dataclass
class _TrendState:
    name: str = "no_trend"
    code: float = 0.0
    candidate: str = "no_candidate"
    point2: float = np.nan
    point3: float = np.nan
    point2_idx: Optional[int] = None
    point3_idx: Optional[int] = None
    candidate_point2: float = np.nan
    candidate_point3: float = np.nan
    candidate_point2_idx: Optional[int] = None
    candidate_point3_idx: Optional[int] = None


@dataclass
class _StructureState:
    last_max_idx: Optional[int] = None
    last_min_idx: Optional[int] = None
    temp_max_idx: Optional[int] = None
    temp_min_idx: Optional[int] = None

    prev_direction: int = 0
    prev_status: int = 0
    excep: int = 1

    extrema: List[Extremum] = field(default_factory=list)
    trend: _TrendState = field(default_factory=_TrendState)
    invalid_up_point3_idx: int = -1
    invalid_down_point3_idx: int = -1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MarketStateVector:
    """Build an online market state vector from OHLC data and CPD events."""

    def __init__(
        self,
        detector_method: str = "cusum",
        detector_q: float = 1.0,
        detector_kwargs: Optional[Dict[str, Any]] = None,
        cpd_confirm_lag: int = 0,
        reset_structure_on_cpd: bool = False,
    ) -> None:
        self.detector_method = detector_method
        self.detector_q = detector_q
        self.detector_kwargs = detector_kwargs or {}
        self.cpd_confirm_lag = max(0, int(cpd_confirm_lag))
        self.reset_structure_on_cpd = bool(reset_structure_on_cpd)
        self.reset()

    # -------------------------- CPD / Direction --------------------------

    def _new_detector(self) -> Brk:
        return Brk(method=self.detector_method, q=self.detector_q, **self.detector_kwargs)

    def reset(self) -> None:
        """Reset online state so subsequent update calls start a fresh stream."""
        self._detector = self._new_detector()
        self._timestamps: List[pd.Timestamp] = []
        self._open_values: List[float] = []
        self._high_values: List[float] = []
        self._low_values: List[float] = []
        self._close_values: List[float] = []
        self._idx_to_pos: Dict[pd.Timestamp, int] = {}
        self._confirmed_cp_set: set[int] = set()
        self._confirmed_cp_positions: List[int] = []
        self._confirmed_cp_direction_by_pos: Dict[int, int] = {}
        self._confirmed_cp_source_by_pos: Dict[int, str] = {}
        self._confirmed_cp_score_by_pos: Dict[int, float] = {}
        self._last_confirm_bar: Optional[int] = None
        self._structure = _StructureState()
        self._rows: List[Dict[str, Any]] = []

    def _update_cpd_events(
        self,
        detector: Brk,
        close_prefix: pd.Series,
        idx_to_pos: Dict[pd.Timestamp, int],
        i: int,
        confirmed_set: set[int],
        confirmed_positions: List[int],
    ) -> List[Dict[str, Any]]:
        """Run the detector on the current prefix and return newly confirmed CPD events."""
        cpd_events = detector.detect_events(close_prefix)
        new_events: List[Dict[str, Any]] = []
        for event in cpd_events or []:
            ts = pd.Timestamp(event["timestamp"])
            if ts not in idx_to_pos:
                continue
            pos = idx_to_pos[ts]
            if pos > i or pos in confirmed_set:
                continue
            if i - pos < self.cpd_confirm_lag:
                continue
            confirmed_set.add(pos)
            confirmed_positions.append(pos)
            event_out = {
                "position": int(pos),
                "timestamp": ts,
                "direction": int(event.get("direction", 0)),
                "direction_source": str(event.get("direction_source", "unknown")),
                "score": float(event.get("score", np.nan)),
            }
            self._confirmed_cp_direction_by_pos[pos] = int(event_out["direction"])
            self._confirmed_cp_source_by_pos[pos] = event_out["direction_source"]
            self._confirmed_cp_score_by_pos[pos] = event_out["score"]
            new_events.append(event_out)
        confirmed_positions.sort()
        return new_events

    # -------------------------- Exception / Status -----------------------

    def _initialise_minmax_if_needed(
        self,
        i: int,
        direction: int,
        highs: np.ndarray,
        lows: np.ndarray,
        state: _StructureState,
    ) -> None:
        """Initialise last fixed extremum after Direction first becomes nonzero."""
        if direction > 0 and state.last_min_idx is None:
            state.last_min_idx = _argmin(lows, 0, i)
            start = (state.last_min_idx or 0) + 1
            state.temp_max_idx = _argmax(highs, start, i) or i
        elif direction < 0 and state.last_max_idx is None:
            state.last_max_idx = _argmax(highs, 0, i)
            start = (state.last_max_idx or 0) + 1
            state.temp_min_idx = _argmin(lows, start, i) or i

    def _update_excep_and_status(
        self,
        i: int,
        direction: int,
        highs: np.ndarray,
        lows: np.ndarray,
        state: _StructureState,
    ) -> Tuple[int, int]:
        """Update persistent Excep and compute Status.

        This follows the article's idea more closely than a one-bar condition:
        once an exception starts, it remains active until either Direction
        reverses or price recovers beyond the relevant opposite fixed extremum.
        """
        if direction == 0:
            return state.excep, 0

        prev_direction = state.prev_direction
        lastmax = state.last_max_idx
        lastmin = state.last_min_idx

        if state.excep == -1:
            direction_reversed = prev_direction != 0 and prev_direction * direction == -1
            recovered_up = prev_direction == 1 and lastmax is not None and highs[i] >= highs[lastmax]
            recovered_down = prev_direction == -1 and lastmin is not None and lows[i] <= lows[lastmin]
            if direction_reversed or recovered_up or recovered_down:
                state.excep = 1
        else:
            # The article checks exceptional situations only when Direction has
            # not already changed. This avoids double-counting normal reversals.
            same_direction = prev_direction == 0 or prev_direction == direction
            if same_direction:
                breaks_last_low = direction > 0 and lastmin is not None and lows[i] <= lows[lastmin]
                breaks_last_high = direction < 0 and lastmax is not None and highs[i] >= highs[lastmax]
                if breaks_last_low or breaks_last_high:
                    state.excep = -1

        status = int(direction * state.excep)
        return state.excep, status

    # -------------------------- MinMax process ---------------------------

    def _append_extremum(
        self,
        kind: str,
        idx: int,
        value: float,
        timestamp: pd.Timestamp,
        state: _StructureState,
    ) -> None:
        """Append a confirmed extremum while keeping a clean alternating sequence."""
        ext = Extremum(kind=kind, idx=int(idx), value=float(value), timestamp=pd.Timestamp(timestamp))
        if not state.extrema:
            state.extrema.append(ext)
            return

        last = state.extrema[-1]
        if last.kind != kind:
            # Avoid exact duplicate bars of the same kind after noisy switches.
            if last.idx != ext.idx:
                state.extrema.append(ext)
            return

        # If two same-type extrema arrive consecutively, keep the more extreme one.
        if kind == "high" and ext.value >= last.value:
            state.extrema[-1] = ext
        elif kind == "low" and ext.value <= last.value:
            state.extrema[-1] = ext

    def _update_minmax(
        self,
        i: int,
        ts: pd.Timestamp,
        status: int,
        highs: np.ndarray,
        lows: np.ndarray,
        state: _StructureState,
    ) -> bool:
        """Update temporary extrema and fix extrema on Status switches.

        Returns True when a new relevant extremum is confirmed.
        """
        confirmed = False
        prev_status = state.prev_status

        # First nonzero status: start searching in the current direction.
        if prev_status == 0:
            if status == 1:
                if state.temp_max_idx is None:
                    start = (state.last_min_idx or -1) + 1
                    state.temp_max_idx = _argmax(highs, start, i) or i
                if highs[i] >= highs[state.temp_max_idx]:
                    state.temp_max_idx = i
            elif status == -1:
                if state.temp_min_idx is None:
                    start = (state.last_max_idx or -1) + 1
                    state.temp_min_idx = _argmin(lows, start, i) or i
                if lows[i] <= lows[state.temp_min_idx]:
                    state.temp_min_idx = i
            state.prev_status = status
            return False

        # While the previous status was +1, keep moving the temporary maximum.
        if prev_status == 1:
            if state.temp_max_idx is None or highs[i] >= highs[state.temp_max_idx]:
                state.temp_max_idx = i
            if status == -1:
                # Status changed: fix the temporary maximum forever.
                state.last_max_idx = state.temp_max_idx
                self._append_extremum("high", state.last_max_idx, highs[state.last_max_idx], ts, state)
                confirmed = True
                start = state.last_max_idx + 1
                state.temp_min_idx = _argmin(lows, start, i) or i

        # While the previous status was -1, keep moving the temporary minimum.
        if prev_status == -1:
            if state.temp_min_idx is None or lows[i] <= lows[state.temp_min_idx]:
                state.temp_min_idx = i
            if status == 1:
                # Status changed: fix the temporary minimum forever.
                state.last_min_idx = state.temp_min_idx
                self._append_extremum("low", state.last_min_idx, lows[state.last_min_idx], ts, state)
                confirmed = True
                start = state.last_min_idx + 1
                state.temp_max_idx = _argmax(highs, start, i) or i

        # After possible switch, update the new temporary extremum for this bar.
        if status == 1:
            if state.temp_max_idx is None or highs[i] >= highs[state.temp_max_idx]:
                state.temp_max_idx = i
        elif status == -1:
            if state.temp_min_idx is None or lows[i] <= lows[state.temp_min_idx]:
                state.temp_min_idx = i

        state.prev_status = status
        return confirmed

    # -------------------------- Trend structure --------------------------

    @staticmethod
    def _recent_extrema(extrema: List[Extremum], kind: str, n: int = 2) -> List[Extremum]:
        items = [e for e in extrema if e.kind == kind]
        return items[-n:]

    def _update_trend_structure(
        self,
        i: int,
        close_i: float,
        high_i: float,
        low_i: float,
        state: _StructureState,
    ) -> _TrendState:
        """Derive CurrentTrend from 1-2-3 candidates plus point2 breakout.

        A confirmed trend needs two steps:
        1. a structural candidate from confirmed extrema;
        2. a later breakout through point2 while point3 remains unbroken.

        The confirmed trend persists until point3 is broken. A pullback inside
        the trend is represented by current_phase, not by downgrading the trend.
        """
        prev = state.trend

        up_candidate: Optional[Tuple[Extremum, Extremum]] = None
        down_candidate: Optional[Tuple[Extremum, Extremum]] = None
        for a, b, c in zip(state.extrema, state.extrema[1:], state.extrema[2:]):
            if (
                a.kind == "low"
                and b.kind == "high"
                and c.kind == "low"
                and c.value >= a.value
                and c.idx > state.invalid_up_point3_idx
            ):
                # Automatic 1-2-3 up candidate: low1 -> high1(point2) -> higher/equal low2(point3).
                up_candidate = (b, c)
            elif (
                a.kind == "high"
                and b.kind == "low"
                and c.kind == "high"
                and c.value <= a.value
                and c.idx > state.invalid_down_point3_idx
            ):
                # Automatic 1-2-3 down candidate: high1 -> low1(point2) -> lower/equal high2(point3).
                down_candidate = (b, c)

        confirmed_trend_invalidated = False
        if prev.name == "uptrend" and np.isfinite(prev.point3) and low_i <= prev.point3:
            if prev.point3_idx is not None:
                state.invalid_up_point3_idx = max(state.invalid_up_point3_idx, prev.point3_idx)
            prev = _TrendState(name="no_trend", code=0.0)
            confirmed_trend_invalidated = True
        elif prev.name == "downtrend" and np.isfinite(prev.point3) and high_i >= prev.point3:
            if prev.point3_idx is not None:
                state.invalid_down_point3_idx = max(state.invalid_down_point3_idx, prev.point3_idx)
            prev = _TrendState(name="no_trend", code=0.0)
            confirmed_trend_invalidated = True

        candidates: List[_TrendState] = []
        if up_candidate is not None:
            point2, point3 = up_candidate
            stop_alive = i <= point3.idx or low_i > point3.value
            if not stop_alive:
                state.invalid_up_point3_idx = max(state.invalid_up_point3_idx, point3.idx)
            elif high_i >= point2.value and i > point3.idx and not confirmed_trend_invalidated:
                candidates.append(
                    _TrendState(
                        name="uptrend",
                        code=1.0,
                        candidate="uptrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )
            else:
                candidates.append(
                    _TrendState(
                        name="uptrend_question",
                        code=0.5,
                        candidate="uptrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )

        if down_candidate is not None:
            point2, point3 = down_candidate
            stop_alive = i <= point3.idx or high_i < point3.value
            if not stop_alive:
                state.invalid_down_point3_idx = max(state.invalid_down_point3_idx, point3.idx)
            elif low_i <= point2.value and i > point3.idx and not confirmed_trend_invalidated:
                candidates.append(
                    _TrendState(
                        name="downtrend",
                        code=-1.0,
                        candidate="downtrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )
            else:
                candidates.append(
                    _TrendState(
                        name="downtrend_question",
                        code=-0.5,
                        candidate="downtrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )

        latest_candidate = max(
            candidates,
            key=lambda t: -1 if t.candidate_point3_idx is None else t.candidate_point3_idx,
            default=None,
        )
        triggered = [t for t in candidates if t.name in {"uptrend", "downtrend"}]
        latest_triggered = max(
            triggered,
            key=lambda t: -1 if t.point3_idx is None else t.point3_idx,
            default=None,
        )

        if (
            prev.name in {"uptrend", "downtrend"}
            and latest_triggered is not None
            and latest_triggered.name == prev.name
            and (
                prev.point3_idx is None
                or (
                    latest_triggered.point3_idx is not None
                    and latest_triggered.point3_idx > prev.point3_idx
                )
            )
        ):
            trend = latest_triggered
        elif prev.name in {"uptrend", "downtrend"}:
            trend = _TrendState(
                name=prev.name,
                code=prev.code,
                candidate=latest_candidate.candidate if latest_candidate is not None else "no_candidate",
                point2=prev.point2,
                point3=prev.point3,
                point2_idx=prev.point2_idx,
                point3_idx=prev.point3_idx,
                candidate_point2=latest_candidate.candidate_point2 if latest_candidate is not None else np.nan,
                candidate_point3=latest_candidate.candidate_point3 if latest_candidate is not None else np.nan,
                candidate_point2_idx=latest_candidate.candidate_point2_idx if latest_candidate is not None else None,
                candidate_point3_idx=latest_candidate.candidate_point3_idx if latest_candidate is not None else None,
            )
        elif latest_triggered is not None:
            trend = latest_triggered
        elif latest_candidate is not None:
            trend = latest_candidate
        else:
            trend = _TrendState(name="no_trend", code=0.0)

        state.trend = trend
        return trend

    @staticmethod
    def _derive_current_phase(trend: _TrendState, status: int, direction: int) -> str:
        """Separate structural trend from the current movement/correction leg."""
        if trend.name in {"uptrend", "uptrend_question"}:
            leg_sign = status if status != 0 else direction
            if leg_sign > 0:
                return "up_movement"
            if leg_sign < 0:
                return "up_correction"
            return "up_phase_unknown"
        if trend.name in {"downtrend", "downtrend_question"}:
            leg_sign = status if status != 0 else direction
            if leg_sign < 0:
                return "down_movement"
            if leg_sign > 0:
                return "down_correction"
            return "down_phase_unknown"
        return "no_phase"

    @staticmethod
    def _trend_quality_features(
        i: int,
        trend: _TrendState,
        highs: np.ndarray,
        lows: np.ndarray,
        extrema: List[Extremum],
    ) -> Tuple[float, float, float]:
        """Return trend dynamic, movement life-time, and bars since point3.

        The dynamic follows the 1-2-3 trend indicator definition:
        |P2new - P3| / |P2 - P3|. Online, P2new is represented by the best
        direction-aligned price reached since P3. Life-time is counted in
        trend movements and is at least 2 for an active confirmed trend.
        """
        if trend.name not in {"uptrend", "downtrend"}:
            return np.nan, np.nan, np.nan
        if (
            trend.point2_idx is None
            or trend.point3_idx is None
            or trend.point3_idx > i
            or not np.isfinite(trend.point2)
            or not np.isfinite(trend.point3)
        ):
            return np.nan, np.nan, np.nan

        denom = abs(float(trend.point2) - float(trend.point3))
        if denom <= 0 or not np.isfinite(denom):
            return np.nan, np.nan, np.nan

        start = max(0, int(trend.point3_idx))
        if trend.name == "uptrend":
            p2_new = float(np.nanmax(highs[start : i + 1]))
            dynamic = abs(p2_new - float(trend.point3)) / denom
            continuation_extrema = [
                e for e in extrema if e.kind == "high" and e.idx > int(trend.point3_idx)
            ]
        else:
            p2_new = float(np.nanmin(lows[start : i + 1]))
            dynamic = abs(p2_new - float(trend.point3)) / denom
            continuation_extrema = [
                e for e in extrema if e.kind == "low" and e.idx > int(trend.point3_idx)
            ]

        lifetime_movements = max(2, 1 + len(continuation_extrema))
        lifetime_bars_since_point3 = i - int(trend.point3_idx)
        return float(dynamic), float(lifetime_movements), float(lifetime_bars_since_point3)

    # -------------------------- Online update / replay --------------------

    def update(
        self,
        bar: pd.Series | Dict[str, Any],
        timestamp: Optional[pd.Timestamp | str] = None,
    ) -> pd.Series:
        """Update the detector with one OHLC bar and return the current state row.

        The returned Series uses only data seen up to this call. Events that are
        confirmed now are reported on the current row; historical rows are not
        rewritten with future confirmation information.
        """
        if isinstance(bar, pd.Series):
            ts = pd.Timestamp(timestamp if timestamp is not None else bar.name)
            raw = bar.to_dict()
        else:
            if timestamp is None:
                raise ValueError("timestamp is required when bar is not a pandas Series")
            ts = pd.Timestamp(timestamp)
            raw = dict(bar)
        if pd.isna(ts):
            raise ValueError("timestamp must be a valid datetime")

        ohlc = _ensure_ohlc_frame(pd.DataFrame([raw], index=pd.DatetimeIndex([ts])))
        row = ohlc.iloc[0]

        i = len(self._timestamps)
        self._timestamps.append(pd.Timestamp(ohlc.index[0]))
        self._open_values.append(float(row["Open"]))
        self._high_values.append(float(row["High"]))
        self._low_values.append(float(row["Low"]))
        self._close_values.append(float(row["Close"]))
        self._idx_to_pos[pd.Timestamp(ohlc.index[0])] = i

        open_arr = np.asarray(self._open_values, dtype=float)
        high_arr = np.asarray(self._high_values, dtype=float)
        low_arr = np.asarray(self._low_values, dtype=float)
        close_arr = np.asarray(self._close_values, dtype=float)
        close_prefix = pd.Series(close_arr, index=pd.DatetimeIndex(self._timestamps))

        new_cp_events = self._update_cpd_events(
            detector=self._detector,
            close_prefix=close_prefix,
            idx_to_pos=self._idx_to_pos,
            i=i,
            confirmed_set=self._confirmed_cp_set,
            confirmed_positions=self._confirmed_cp_positions,
        )
        cpd_confirm_event = bool(new_cp_events)
        if cpd_confirm_event:
            self._last_confirm_bar = i

        if cpd_confirm_event and self.reset_structure_on_cpd:
            self._structure = _StructureState()

        structure = self._structure
        regime_start = self._confirmed_cp_positions[-1] if self._confirmed_cp_positions else 0
        cpd_regime_direction = self._confirmed_cp_direction_by_pos.get(regime_start, 0)
        cpd_direction_source = self._confirmed_cp_source_by_pos.get(regime_start, None)
        cpd_direction_score = self._confirmed_cp_score_by_pos.get(regime_start, np.nan)
        direction = cpd_regime_direction

        self._initialise_minmax_if_needed(i, direction, high_arr, low_arr, structure)
        excep, status = self._update_excep_and_status(i, direction, high_arr, low_arr, structure)
        status_switch = i > 0 and status != structure.prev_status
        new_extremum_confirmed = self._update_minmax(i, pd.Timestamp(ohlc.index[0]), status, high_arr, low_arr, structure)
        trend = self._update_trend_structure(i, close_arr[i], high_arr[i], low_arr[i], structure)
        current_phase = self._derive_current_phase(trend, status, direction)
        trend_dynamic, trend_lifetime_movements, trend_lifetime_bars_since_point3 = (
            self._trend_quality_features(i, trend, high_arr, low_arr, structure.extrema)
        )
        confirmed_extremum = structure.extrema[-1] if new_extremum_confirmed and structure.extrema else None

        last_max = high_arr[structure.last_max_idx] if structure.last_max_idx is not None else np.nan
        last_min = low_arr[structure.last_min_idx] if structure.last_min_idx is not None else np.nan
        temp_max = high_arr[structure.temp_max_idx] if structure.temp_max_idx is not None else np.nan
        temp_min = low_arr[structure.temp_min_idx] if structure.temp_min_idx is not None else np.nan

        denom = last_max - last_min if np.isfinite(last_max) and np.isfinite(last_min) else np.nan
        position = (close_arr[i] - last_min) / denom if np.isfinite(denom) and abs(denom) > 0 else np.nan

        prev_row = self._rows[-1] if self._rows else None
        delta_position = np.nan
        if prev_row is not None and np.isfinite(position) and np.isfinite(prev_row.get("position", np.nan)):
            delta_position = float(position - prev_row["position"])

        distance_to_high = close_arr[i] - last_max if np.isfinite(last_max) else np.nan
        distance_to_low = close_arr[i] - last_min if np.isfinite(last_min) else np.nan
        distance_to_point2 = close_arr[i] - trend.point2 if np.isfinite(trend.point2) else np.nan
        distance_to_point3 = close_arr[i] - trend.point3 if np.isfinite(trend.point3) else np.nan
        close_i = close_arr[i]
        dist_point2_pct = distance_to_point2 / close_i if np.isfinite(distance_to_point2) and close_i != 0 else np.nan
        dist_point3_pct = distance_to_point3 / close_i if np.isfinite(distance_to_point3) and close_i != 0 else np.nan

        if regime_start < i:
            returns = np.diff(close_arr[regime_start : i + 1])
            regime_volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        else:
            regime_volatility = 0.0

        time_since_last_cpd = i - regime_start if self._confirmed_cp_positions else i + 1
        time_since_last_cpd_confirm = i - self._last_confirm_bar if self._last_confirm_bar is not None else i + 1
        trend_age = 0
        if prev_row is not None and prev_row["current_trend"] == trend.name:
            trend_age = int(prev_row["trend_age"] + 1)

        structure.prev_direction = direction

        latest_event = new_cp_events[-1] if new_cp_events else None
        latest_confirmed_pos = latest_event["position"] if latest_event is not None else np.nan
        latest_confirmed_ts = (
            self._timestamps[int(latest_confirmed_pos)] if latest_event is not None else pd.NaT
        )
        state_row: Dict[str, Any] = {
            "timestamp": pd.Timestamp(ohlc.index[0]),
            "Open": float(open_arr[i]),
            "High": float(high_arr[i]),
            "Low": float(low_arr[i]),
            "Close": float(close_arr[i]),
            "detector_method": self.detector_method,
            "detector_q": float(self.detector_q),
            "cpd_direction_source": cpd_direction_source,
            "cpd_direction_score": float(cpd_direction_score) if np.isfinite(cpd_direction_score) else np.nan,
            "cpd_confirm_event": cpd_confirm_event,
            "cpd_confirmed_position": int(latest_confirmed_pos) if latest_event is not None else np.nan,
            "cpd_confirmed_timestamp": latest_confirmed_ts,
            "cpd_confirmed_direction": int(latest_event["direction"]) if latest_event is not None else np.nan,
            "cpd_confirmed_direction_source": latest_event["direction_source"] if latest_event is not None else None,
            "cpd_confirmed_score": (
                float(latest_event["score"])
                if latest_event is not None and np.isfinite(latest_event["score"])
                else np.nan
            ),
            "direction": int(direction),
            "excep": int(excep),
            "status": int(status),
            "status_switch": bool(status_switch),
            "new_extremum_confirmed": bool(new_extremum_confirmed),
            "confirmed_extremum_kind": confirmed_extremum.kind if confirmed_extremum is not None else None,
            "confirmed_extremum_idx": confirmed_extremum.idx if confirmed_extremum is not None else None,
            "confirmed_extremum_value": (
                float(confirmed_extremum.value) if confirmed_extremum is not None else np.nan
            ),
            "confirmed_extremum_confirmed_at": pd.Timestamp(ohlc.index[0])
            if confirmed_extremum is not None
            else pd.NaT,
            "current_trend": trend.name,
            "current_trend_code": float(trend.code),
            "current_phase": current_phase,
            "trend_candidate": trend.candidate,
            "trend_age": int(trend_age),
            "trend_dynamic": float(trend_dynamic) if np.isfinite(trend_dynamic) else np.nan,
            "trend_lifetime_movements": (
                float(trend_lifetime_movements)
                if np.isfinite(trend_lifetime_movements)
                else np.nan
            ),
            "trend_lifetime_bars_since_point3": (
                float(trend_lifetime_bars_since_point3)
                if np.isfinite(trend_lifetime_bars_since_point3)
                else np.nan
            ),
            "position": float(position) if np.isfinite(position) else np.nan,
            "delta_position": float(delta_position) if np.isfinite(delta_position) else np.nan,
            "distance_to_high": float(distance_to_high) if np.isfinite(distance_to_high) else np.nan,
            "distance_to_low": float(distance_to_low) if np.isfinite(distance_to_low) else np.nan,
            "distance_to_point2": float(distance_to_point2) if np.isfinite(distance_to_point2) else np.nan,
            "distance_to_point3": float(distance_to_point3) if np.isfinite(distance_to_point3) else np.nan,
            "dist_point2_pct": float(dist_point2_pct) if np.isfinite(dist_point2_pct) else np.nan,
            "dist_point3_pct": float(dist_point3_pct) if np.isfinite(dist_point3_pct) else np.nan,
            "time_since_last_cpd": int(time_since_last_cpd),
            "time_since_last_cpd_confirm": int(time_since_last_cpd_confirm),
            "regime_volatility": float(regime_volatility),
            "point2": float(trend.point2) if np.isfinite(trend.point2) else np.nan,
            "point3": float(trend.point3) if np.isfinite(trend.point3) else np.nan,
            "point2_idx": trend.point2_idx,
            "point3_idx": trend.point3_idx,
            "candidate_point2": (
                float(trend.candidate_point2) if np.isfinite(trend.candidate_point2) else np.nan
            ),
            "candidate_point3": (
                float(trend.candidate_point3) if np.isfinite(trend.candidate_point3) else np.nan
            ),
            "candidate_point2_idx": trend.candidate_point2_idx,
            "candidate_point3_idx": trend.candidate_point3_idx,
            "last_max": float(last_max) if np.isfinite(last_max) else np.nan,
            "last_min": float(last_min) if np.isfinite(last_min) else np.nan,
            "temp_max": float(temp_max) if np.isfinite(temp_max) else np.nan,
            "temp_min": float(temp_min) if np.isfinite(temp_min) else np.nan,
            "num_confirmed_extrema": int(len(structure.extrema)),
            "regime_start_idx": int(regime_start),
        }

        self._rows.append(state_row)
        return pd.Series(state_row, name=pd.Timestamp(ohlc.index[0]))

    def build_market_state_vector(
        self,
        data: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Replay OHLC history through update(bar) and return one state row per bar."""
        if data is None:
            if symbol is None:
                raise ValueError("Either data or symbol must be provided")
            data = load_data(
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
            )
        ohlc = _ensure_ohlc_frame(data)
        if ohlc.empty:
            raise ValueError("Cannot build market state vector from empty OHLC data")

        self.reset()
        rows = [self.update(row, timestamp=ts).to_dict() for ts, row in ohlc.iterrows()]
        state_df = pd.DataFrame(rows).set_index("timestamp")
        state_df.index = pd.to_datetime(state_df.index)
        return state_df

    # -------------------------- Evaluation -------------------------------

    def evaluate_state_vector(
        self,
        state_df: pd.DataFrame,
        horizons: Sequence[int] = (5,),
        position_bins: Optional[Sequence[float]] = None,
        price_is_log: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate persistence and future-return separation."""
        if not isinstance(state_df, pd.DataFrame):
            raise TypeError("state_df must be a pandas DataFrame")
        if "Close" not in state_df.columns:
            raise ValueError("state_df must include a Close column")

        df = state_df.sort_index().copy()
        horizons = tuple(int(h) for h in horizons)

        for h in horizons:
            if price_is_log:
                df[f"future_return_{h}"] = df["Close"].shift(-h) - df["Close"]
            else:
                df[f"future_return_{h}"] = df["Close"].shift(-h) / df["Close"] - 1.0

        trend_change = df["current_trend"].ne(df["current_trend"].shift()).fillna(False)
        phase_change = df["current_phase"].ne(df["current_phase"].shift()).fillna(False)
        status_change = df["status"].ne(df["status"].shift()).fillna(False)
        trend_group = trend_change.cumsum()
        phase_group = phase_change.cumsum()

        duration_table = (
            df.groupby(trend_group)
            .agg(
                current_trend=("current_trend", "first"),
                start=("current_trend", lambda s: s.index[0]),
                end=("current_trend", lambda s: s.index[-1]),
                bars=("current_trend", "size"),
            )
            .reset_index(drop=True)
        )
        duration_summary = duration_table.groupby("current_trend")["bars"].agg(
            ["count", "mean", "median", "min", "max"]
        )
        phase_duration_table = (
            df.groupby(phase_group)
            .agg(
                current_phase=("current_phase", "first"),
                start=("current_phase", lambda s: s.index[0]),
                end=("current_phase", lambda s: s.index[-1]),
                bars=("current_phase", "size"),
            )
            .reset_index(drop=True)
        )
        phase_duration_summary = phase_duration_table.groupby("current_phase")["bars"].agg(
            ["count", "mean", "median", "min", "max"]
        )

        transition_matrix = pd.crosstab(
            df["current_trend"].shift(),
            df["current_trend"],
            normalize="index",
        ).fillna(0.0)
        phase_transition_matrix = pd.crosstab(
            df["current_phase"].shift(),
            df["current_phase"],
            normalize="index",
        ).fillna(0.0)

        if position_bins is None:
            position_bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
        position_bucket = pd.cut(df["position"], bins=position_bins, include_lowest=True)

        position_future_returns: Dict[int, pd.DataFrame] = {}
        trend_future_returns: Dict[int, pd.DataFrame] = {}
        phase_future_returns: Dict[int, pd.DataFrame] = {}
        for h in horizons:
            col = f"future_return_{h}"
            position_future_returns[h] = df.groupby(position_bucket, observed=False)[col].agg(
                ["count", "mean", "std", "median", "min", "max"]
            )
            trend_future_returns[h] = df.groupby("current_trend")[col].agg(
                ["count", "mean", "std", "median", "min", "max"]
            )
            phase_future_returns[h] = df.groupby("current_phase")[col].agg(
                ["count", "mean", "std", "median", "min", "max"]
            )

        quality_columns = [
            "trend_dynamic",
            "trend_lifetime_movements",
            "trend_lifetime_bars_since_point3",
        ]
        if set(quality_columns).issubset(df.columns):
            active_quality = df[df["current_trend"].isin(["uptrend", "downtrend"])][
                quality_columns + ["current_trend"]
            ].copy()
            trend_quality_summary = active_quality.groupby("current_trend")[quality_columns].agg(
                ["count", "mean", "median", "min", "max"]
            )

            terminal_quality = (
                df[df["current_trend"].isin(["uptrend", "downtrend"])]
                .assign(_trend_group=trend_group)
                .groupby("_trend_group")
                .agg(
                    current_trend=("current_trend", "first"),
                    start=("current_trend", lambda s: s.index[0]),
                    end=("current_trend", lambda s: s.index[-1]),
                    bars=("current_trend", "size"),
                    terminal_dynamic=("trend_dynamic", "last"),
                    terminal_lifetime_movements=("trend_lifetime_movements", "last"),
                    terminal_lifetime_bars_since_point3=(
                        "trend_lifetime_bars_since_point3",
                        "last",
                    ),
                )
                .reset_index(drop=True)
            )
        else:
            trend_quality_summary = pd.DataFrame()
            terminal_quality = pd.DataFrame()

        point23_required = {"point2_idx", "point3_idx", "current_trend"}
        if point23_required.issubset(df.columns):
            point23 = df.dropna(subset=["point2_idx", "point3_idx"])[
                ["current_trend", "point2_idx", "point3_idx"]
            ].copy()
            point23["point2_idx"] = point23["point2_idx"].astype(int)
            point23["point3_idx"] = point23["point3_idx"].astype(int)
            point23 = point23.drop_duplicates(["point2_idx", "point3_idx"], keep="last")
            point23 = point23[point23["point3_idx"] >= point23["point2_idx"]].copy()
            point23["point2_timestamp"] = df.index[point23["point2_idx"].to_numpy()]
            point23["point3_timestamp"] = df.index[point23["point3_idx"].to_numpy()]
            point23["point2_point3_interval_bars"] = point23["point3_idx"] - point23["point2_idx"]
            point23_interval_table = point23.reset_index(drop=True)
            point23_interval_summary = point23_interval_table.groupby("current_trend")[
                "point2_point3_interval_bars"
            ].agg(["count", "mean", "std", "median", "min", "max"])
        else:
            point23_interval_table = pd.DataFrame()
            point23_interval_summary = pd.DataFrame()

        return {
            "state_duration_table": duration_table,
            "state_duration_summary": duration_summary,
            "phase_duration_table": phase_duration_table,
            "phase_duration_summary": phase_duration_summary,
            "switch_counts": {
                "current_trend_switch_count": int(max(trend_change.sum() - 1, 0)),
                "current_phase_switch_count": int(max(phase_change.sum() - 1, 0)),
                "status_switch_count": int(max(status_change.sum() - 1, 0)),
                "confirmed_extrema_count": int(df.get("new_extremum_confirmed", pd.Series(False, index=df.index)).sum()),
            },
            "transition_matrix": transition_matrix,
            "phase_transition_matrix": phase_transition_matrix,
            "position_future_returns": position_future_returns,
            "trend_future_returns": trend_future_returns,
            "phase_future_returns": phase_future_returns,
            "trend_quality_summary": trend_quality_summary,
            "trend_quality_terminal": terminal_quality,
            "point23_interval_table": point23_interval_table,
            "point23_interval_summary": point23_interval_summary,
        }


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def build_market_state_vector(
    data: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    detector_method: str = "cusum",
    detector_q: float = 1.0,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    cpd_confirm_lag: int = 0,
    reset_structure_on_cpd: bool = False,
) -> pd.DataFrame:
    builder = MarketStateVector(
        detector_method=detector_method,
        detector_q=detector_q,
        detector_kwargs=detector_kwargs,
        cpd_confirm_lag=cpd_confirm_lag,
        reset_structure_on_cpd=reset_structure_on_cpd,
    )
    return builder.build_market_state_vector(
        data=data,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
    )


def get_core_state_vector(state_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact core market state vector from a full state_df."""
    if not isinstance(state_df, pd.DataFrame):
        raise TypeError("state_df must be a pandas DataFrame")

    columns = [
        "current_trend",
        "current_phase",
        "position",
        "regime_volatility",
    ]
    core_df = state_df[columns].copy()

    close = state_df["Close"].replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        core_df["dist_point2_pct"] = state_df["distance_to_point2"] / close
        core_df["dist_point3_pct"] = state_df["distance_to_point3"] / close
    core_df["dist_point2_pct"] = core_df["dist_point2_pct"].replace([np.inf, -np.inf], np.nan)
    core_df["dist_point3_pct"] = core_df["dist_point3_pct"].replace([np.inf, -np.inf], np.nan)

    return core_df[
        [
            "current_trend",
            "current_phase",
            "position",
            "regime_volatility",
            "dist_point2_pct",
            "dist_point3_pct",
        ]
    ]


def evaluate_state_vector(
    state_df: pd.DataFrame,
    horizons: Sequence[int] = (5,),
    position_bins: Optional[Sequence[float]] = None,
    price_is_log: bool = False,
) -> Dict[str, Any]:
    return MarketStateVector().evaluate_state_vector(
        state_df=state_df,
        horizons=horizons,
        position_bins=position_bins,
        price_is_log=price_is_log,
    )


__all__ = [
    "Extremum",
    "MarketStateVector",
    "load_data",
    "build_market_state_vector",
    "get_core_state_vector",
    "evaluate_state_vector",
    "plot_price_state_background",
]
