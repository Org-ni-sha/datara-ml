"""B4/E3 evaluation: next-hour error, PET error on simulated quota cycles, and Global vs Local.

Everything is scored with **whole devices held out**: the devices are split into the same 5
grouped folds as ``train_v0`` (``device_folds``, seed 0). For each fold the Global Tweedie model is
tuned (``train_v0.tune``, inner device folds on the training devices only) and fitted on the other
folds, then every held-out device is replayed hour by hour. The naive baselines need no training
and are scored on the same rows and cycles.

1. **Next hour** (the training rows of ``features.build_training_set``: observed hours with at
   least 6 observed hours in the previous 24): MAE / RMSE in MB, R² (bytes and log1p), and
   predicted ÷ actual total, for the Global model and three baselines:

   * ``last_hour``: the last hour's bytes (missing → 24 h average → 0)
   * ``last_24h_avg``: the average of the observed hours among the last 24
   * ``same_hour_yesterday``: hour ``t − 24h`` (missing → 24 h average → 0)

2. **PET error** on simulated quota cycles (ROADMAP B4). A cycle starts at every training row's
   hour ``s`` of a held-out device (the hours at which the app would show a forecast) with each
   quota ``Q``. The projected PET comes from ``features.project()``, the app's hour-by-hour rule.
   The *actual* PET is when the measured cumulative usage from ``s`` crossed ``Q``, interpolated
   inside the crossing hour exactly like the projection. Rules for the actual PET
   (:func:`actual_crossing`):

   * the start hour ``s`` is observed (it is a training row);
   * usage is accumulated hour by hour. An *observed* hour adds its NSM bytes. An unobserved hour
     inside long usable NSM windows adds its share of a total that NSM did measure (the window's
     bytes spread over its span, :func:`known_hourly`): the cumulative is exact again at the end of
     every such window, only the placement inside it is unknown;
   * the cycle is scored only when the crossing falls in an **observed** hour. If it falls in an
     unobserved hour (``crossing_unobserved``), or an hour on the way has no NSM total at all
     (bad status, overlapping windows, coverage hole: ``unknown_gap``), or the device's data or the
     30-day horizon ends first (``never_crossed``), the cycle is not scored. Counts per reason
     are reported.

   Error = projected − actual, in hours: **positive = late** (the PET came after the quota was
   really reached, the bad direction for a warning). A projection that does not reach ``Q`` in 30
   days is scored as the horizon end (``start + 720 h``), so it counts as late, and is also counted
   in ``beyond_horizon``. Methods projected the same way: the Global model, ``last_hour``
   (repeated), ``last_24h_avg`` (repeated: a constant consumption rate) and
   ``same_hour_yesterday`` (yesterday's profile repeated, through the projection's own lags).

3. **E3, Global vs Local**: the app's local layer (``LocalLayer.kt``: the usage ratio per 6-hour
   block, EWMA α = 0.05, block ratio after 12 hours, ready after 24 learned hours, clamped
   0.2–5×; the switch after 168 learned hours and a recent local MAE ≥ 5% below the global one)
   is replayed on every held-out device in time order, one hourly run per clock hour from the
   device's first to its last hour. Reported: next-hour and PET error for Global only, for what the
   user would have seen (``local_switched``), and for the local layer in shadow wherever it was
   ready (``local_shadow``, on those rows/cycles only, compared with Global on the same subset).

4. **Forecast alerts as classification** (the app's D8 ``forecast_24h`` / ``forecast_3h``,
   ``AlertRules.kt``: an alert when the PET is within 24 h / 3 h). At every start hour and quota,
   each method's projection is a yes/no prediction ("the quota will be reached within W hours"),
   and the measured usage gives the actual answer (:func:`alert_label`). Unlike the PET error,
   negatives count too: a start whose quota was *not* reached within W hours is a true negative
   or a false alarm. Reported per horizon and quota: the confusion matrix (TP, FP, FN, TN),
   precision, recall, F1, accuracy and specificity, for the Global model and the baselines.
   Labels follow the same measurement rules as the actual PET: usage inside long NSM windows is
   exact only at the end of the window, so a start is left out (``unknown``) when the window
   holding the crossing straddles the W-hour boundary, or the data has a hole or ends first.

5. **Feature importance** (gain) of the final model, fitted on every device with the same tuning
   as ``train_v0`` (so the same model as the bundled one for the same data).

Outputs go to ``artifacts/eval/``: ``report.json``, ``report.md`` (tables for Chapter 4) and PNG
figures. Devices appear only as anonymous labels (D01, D02, … in order of first hour).

Usage::

    python -m datara_ml.evaluate --apps data/app_usage_records.csv --usage data/usage_records.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from datara_ml import train_v0 as tv
from datara_ml.features import (
    FEATURES, HOUR_MS, MIN_OBSERVED_LAST_24H, PH_OFFSET_MS, PROJECTION_HORIZON_HOURS, History, HourStats,
    build_training_set, device_histories, feature_vector, observed_last_24h, pooled_network, project,
)
from datara_ml.nsm_hourly import Window, _hour_bytes

QUOTAS_MB = (100, 500, 1000, 3000)
NEXT_HOUR_METHODS = ("global", "last_hour", "last_24h_avg", "same_hour_yesterday")
PET_METHODS = NEXT_HOUR_METHODS
LABELS = {
    "global": "Global Tweedie model",
    "last_hour": "Last hour",
    "last_24h_avg": "Average of last 24 h",
    "same_hour_yesterday": "Same hour yesterday",
    "local_switched": "Local layer (as switched)",
    "local_shadow": "Local layer (shadow, when ready)",
    "global_on_ready": "Global, same rows/cycles",
}
WITHIN_HOURS = (3, 24)
# The forecast alerts (AlertRules.kt FORECAST_24H_MS / FORECAST_3H_MS): "PET within W hours"
ALERT_HORIZONS = (24, 3)


# ---------------------------------------------------------------------------------------------
# Measured cumulative usage and the actual PET
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _AnyLengthWindow(Window):
    """A window whose total counts at any length (for the cumulative, not for features)."""

    @property
    def usable(self) -> bool:
        from datara_ml.nsm_hourly import USABLE_STATUSES
        return self.status in USABLE_STATUSES and self.end_ms > self.start_ms


def known_hourly(windows: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Per device, hour -> bytes for every hour whose NSM total is known: the contract's rule
    (``nsm_hourly._hour_bytes``: no overlap, covered to within 60 s, usable statuses) but with
    windows of **any** length, their bytes spread evenly. Observed hours get the same value as in
    ``nsm_hourly``; hours inside long windows get an estimate whose sum over the window is exact.
    """
    out: dict[str, dict[int, float]] = {}
    for device, dw in windows.groupby("device_id", sort=True):
        by_hour: dict[int, list[Window]] = {}
        for r in dw.itertuples(index=False):
            w = _AnyLengthWindow(int(r.start_ms), int(r.end_ms), float(r.mobile_bytes), r.status)
            if w.end_ms <= w.start_ms:
                if w.mobile_bytes > 0:
                    by_hour.setdefault((w.start_ms // HOUR_MS) * HOUR_MS, []).append(w)
                continue
            h = (w.start_ms // HOUR_MS) * HOUR_MS
            while h < w.end_ms:
                by_hour.setdefault(h, []).append(w)
                h += HOUR_MS
        hours = {h: _hour_bytes(h, ws) for h, ws in by_hour.items()}
        out[device] = {h: b for h, b in hours.items() if not math.isnan(b)}
    return out


@dataclass(frozen=True)
class Crossing:
    status: str  # crossed | crossing_unobserved | unknown_gap | never_crossed
    reach_ms: int | None = None


def actual_crossing(observed: History, known: dict[int, float], start_ms: int, quota_bytes: float,
                    last_hour_ms: int, horizon_hours: int = PROJECTION_HORIZON_HOURS) -> Crossing:
    """When measured usage from ``start_ms`` reached ``quota_bytes`` (see the module docstring).

    ``observed``: the device's observed hours (``features.device_histories``); ``known``: its
    :func:`known_hourly`; ``last_hour_ms``: its last hour with data.
    """
    cumulative = 0.0
    for k in range(horizon_hours):
        h = start_ms + k * HOUR_MS
        if h > last_hour_ms:
            break
        if h in observed:
            b = observed[h].mobile_bytes
            if b > 0 and cumulative + b >= quota_bytes:
                return Crossing("crossed", h + math.floor((quota_bytes - cumulative) / b * HOUR_MS))
        elif h in known:
            b = known[h]
            if b > 0 and cumulative + b >= quota_bytes:
                return Crossing("crossing_unobserved")
        else:
            return Crossing("unknown_gap")
        cumulative += b
    return Crossing("never_crossed")


def alert_label(observed: History, known: dict[int, float], start_ms: int, quota_bytes: float,
                last_hour_ms: int, hours: int) -> bool | None:
    """Did measured usage from ``start_ms`` reach ``quota_bytes`` within ``hours`` hours? None when
    the data cannot tell.

    Observed hours are exact. A run of unobserved hours inside long NSM windows (``known``) is exact
    only at its end: if the cumulative reaches the quota inside a run, the answer is known when the
    whole run lies before the boundary (True) or after it (False), and unknown when it straddles it.
    A run that ends below the quota after the boundary proves False (usage only grows). A hole in the
    data, or the data ending, before the answer is known gives None.
    """
    cumulative = 0.0
    k = 0
    while True:
        if k >= hours:
            return False
        h = start_ms + k * HOUR_MS
        if h > last_hour_ms:
            return None
        if h in observed:
            cumulative += observed[h].mobile_bytes
            if cumulative >= quota_bytes and observed[h].mobile_bytes > 0:
                return True
            k += 1
            continue
        if h not in known:
            return None
        while start_ms + k * HOUR_MS in known and start_ms + k * HOUR_MS not in observed \
                and start_ms + k * HOUR_MS <= last_hour_ms:
            cumulative += known[start_ms + k * HOUR_MS]
            k += 1
        last = k - 1
        if cumulative >= quota_bytes:
            return True if last < hours else None
        # Below the quota at the end of the run: nothing was reached within it


# ---------------------------------------------------------------------------------------------
# Projection: one trajectory serves every quota
# ---------------------------------------------------------------------------------------------

def trajectory(history: History, start_ms: int, max_quota_bytes: float,
               predict: Callable[[np.ndarray], float], offset_ms: int = PH_OFFSET_MS,
               horizon_hours: int = PROJECTION_HORIZON_HOURS) -> list[float]:
    """The hourly predictions ``features.project()`` makes from ``start_ms`` until it reaches
    ``max_quota_bytes`` (or the 30-day horizon). The predictions do not depend on the quota, so
    :func:`reach_from_trajectory` gives the same PET as ``project()`` for every smaller quota."""
    seen: list[float] = []

    def recording(x: np.ndarray) -> float:
        b = max(0.0, float(predict(x)))
        seen.append(b)
        return b

    project(history, start_ms, max_quota_bytes, recording, offset_ms, horizon_hours)
    return seen


def batch_trajectories(jobs: list[tuple[History, int, float]], predict_batch: Callable[[np.ndarray], np.ndarray],
                       offset_ms: int = PH_OFFSET_MS,
                       horizon_hours: int = PROJECTION_HORIZON_HOURS) -> list[list[float]]:
    """:func:`trajectory` for many ``(history, start_ms, max_quota_bytes)`` jobs at once: the same
    steps as ``features.project()`` (checked by a test), with one model call per hour step for all
    projections still running. Used to tune by PET error in reasonable time."""
    n = len(jobs)
    hists = [dict(h) for h, _, _ in jobs]
    pooled = [pooled_network(h, s) for h, s, _ in jobs]
    cumulative = [0.0] * n
    out: list[list[float]] = [[] for _ in range(n)]
    active = [i for i in range(n)]
    for k in range(horizon_hours):
        if not active:
            break
        x = np.stack([feature_vector(hists[i], jobs[i][1] + k * HOUR_MS, offset_ms, None if k == 0 else pooled[i])
                      for i in active])
        preds = np.asarray(predict_batch(x), float).reshape(-1)
        still = []
        for i, p in zip(active, preds):
            b = max(0.0, float(p))
            out[i].append(b)
            remaining = jobs[i][2]
            if k == 0 and remaining <= 0:
                continue
            if b > 0 and cumulative[i] + b >= remaining:
                continue
            cumulative[i] += b
            hists[i][jobs[i][1] + k * HOUR_MS] = HourStats(mobile_bytes=b)
            still.append(i)
        active = still
    return out


def reach_from_trajectory(bs: list[float], start_ms: int, quota_bytes: float,
                          horizon_hours: int = PROJECTION_HORIZON_HOURS) -> int | None:
    """``project()``'s step 3 on recorded predictions; None = beyond the horizon."""
    if quota_bytes <= 0:
        return start_ms
    cumulative = 0.0
    for k, b in enumerate(bs[:horizon_hours]):
        if b > 0 and cumulative + b >= quota_bytes:
            return start_ms + k * HOUR_MS + math.floor((quota_bytes - cumulative) / b * HOUR_MS)
        cumulative += b
    return None


# ---------------------------------------------------------------------------------------------
# Baselines on feature rows (bytes)
# ---------------------------------------------------------------------------------------------

def _expm1_first(*values: float) -> float:
    for v in values:
        if not math.isnan(v):
            return math.expm1(v)
    return 0.0


def baseline_bytes(name: str, x: np.ndarray) -> float:
    """One baseline's next-hour bytes from a feature row (contract order)."""
    x = [float(v) for v in x]
    if name == "last_hour":
        return _expm1_first(x[4], x[6])
    if name == "last_24h_avg":
        return _expm1_first(x[6])
    if name == "same_hour_yesterday":
        return _expm1_first(x[7], x[6])
    raise ValueError(name)


def repeated_first(fn: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    """A predictor that repeats its first prediction for every later hour (a constant rate)."""
    state: list[float] = []

    def predict(x: np.ndarray) -> float:
        if not state:
            state.append(fn(x))
        return state[0]

    return predict


def pet_predictor(name: str) -> Callable[[np.ndarray], float]:
    """Baselines as projected: last hour and 24 h average repeated; yesterday's profile repeated
    (``same_hour_yesterday`` reads hour t − 24h, which the projection fills with its own
    predictions after the first day)."""
    if name in ("last_hour", "last_24h_avg"):
        return repeated_first(lambda x: baseline_bytes(name, x))
    return lambda x: baseline_bytes(name, x)


# ---------------------------------------------------------------------------------------------
# The app's local layer (LocalLayer.kt, contract v4), ported line for line
# ---------------------------------------------------------------------------------------------

BLOCKS = 4
ALPHA = 0.05
MIN_BLOCK_HOURS = 12
MIN_HOURS_TO_PREDICT = 24
MIN_FACTOR, MAX_FACTOR = 0.2, 5.0
MAE_ALPHA = 1.0 / 168
MIN_HOURS_FOR_LOCAL = 7 * 24
ENTER_MARGIN = 0.05
GLOBAL, LOCAL = "GLOBAL", "LOCAL"


def block(local_hour: int) -> int:
    return local_hour // 6


def ratio(actual: float, predicted: float) -> float:
    return 1.0 if predicted <= 0.0 else min(MAX_FACTOR, max(MIN_FACTOR, actual / predicted))


def _ewma(mean: float, value: float, count: int) -> float:
    return value if count == 0 else mean + ALPHA * (value - mean)


def _mae_ewma(mae: float | None, error: float) -> float:
    return error if mae is None else mae + MAE_ALPHA * (error - mae)


@dataclass
class LocalParams:
    overall_actual: float = 0.0
    overall_predicted: float = 0.0
    overall_hours: int = 0
    block_actual: list[float] = field(default_factory=lambda: [0.0] * BLOCKS)
    block_predicted: list[float] = field(default_factory=lambda: [0.0] * BLOCKS)
    block_hours: list[int] = field(default_factory=lambda: [0] * BLOCKS)
    pending_hour: int | None = None
    pending_local_hour: int | None = None
    pending_global_bytes: float | None = None
    pending_local_bytes: float | None = None
    last_learned_hour: int | None = None

    @property
    def is_ready(self) -> bool:
        return self.overall_hours >= MIN_HOURS_TO_PREDICT

    def factor(self, local_hour: int) -> float:
        b = block(local_hour)
        if self.block_hours[b] >= MIN_BLOCK_HOURS:
            return ratio(self.block_actual[b], self.block_predicted[b])
        return ratio(self.overall_actual, self.overall_predicted)


@dataclass
class LayerScores:
    hours_seen: int = 0
    global_recent_mae: float | None = None
    local_recent_mae: float | None = None


def learn(params: LocalParams, scores: LayerScores, actual_bytes: float | None) -> tuple[LocalParams, LayerScores]:
    """``LocalLayer.learn``: learn from the pending prediction once its hour has been measured."""
    hour = params.pending_hour
    if hour is None:
        return params, scores
    global_ = params.pending_global_bytes
    cleared = copy.deepcopy(params)
    cleared.pending_hour = cleared.pending_local_hour = None
    cleared.pending_global_bytes = cleared.pending_local_bytes = None
    cleared.last_learned_hour = hour
    if actual_bytes is None or global_ is None:
        return cleared, scores
    b = block(params.pending_local_hour or 0)
    cleared.block_actual[b] = _ewma(params.block_actual[b], actual_bytes, params.block_hours[b])
    cleared.block_predicted[b] = _ewma(params.block_predicted[b], global_, params.block_hours[b])
    cleared.block_hours[b] = params.block_hours[b] + 1
    cleared.overall_actual = _ewma(params.overall_actual, actual_bytes, params.overall_hours)
    cleared.overall_predicted = _ewma(params.overall_predicted, global_, params.overall_hours)
    cleared.overall_hours = params.overall_hours + 1
    local_error = abs(actual_bytes - params.pending_local_bytes) if params.pending_local_bytes is not None else None
    new_scores = LayerScores(
        hours_seen=scores.hours_seen + 1,
        global_recent_mae=_mae_ewma(scores.global_recent_mae, abs(actual_bytes - global_)),
        local_recent_mae=_mae_ewma(scores.local_recent_mae, local_error) if local_error is not None
        else scores.local_recent_mae,
    )
    return cleared, new_scores


def decide(current: str, scores: LayerScores, local_ready: bool) -> str:
    """``ModelSwitch.decide``: reversible, with a 5% margin to enter LOCAL."""
    g, loc = scores.global_recent_mae, scores.local_recent_mae
    if not local_ready or scores.hours_seen < MIN_HOURS_FOR_LOCAL or g is None or loc is None:
        return GLOBAL
    if current == LOCAL:
        return GLOBAL if loc >= g else LOCAL
    return LOCAL if loc < g * (1 - ENTER_MARGIN) else GLOBAL


def local_hour_of(hour_ms: int, offset_ms: int = PH_OFFSET_MS) -> int:
    return int(((hour_ms + offset_ms) // HOUR_MS) % 24)


@dataclass
class HourRun:
    """One hourly run of the app that made a prediction."""
    hour_ms: int
    global_bytes: float
    local_bytes: float | None
    active_model: str
    params: LocalParams  # the local layer used for this hour's projection
    hours_seen: int


def replay_device(history: History, first_hour_ms: int, last_hour_ms: int,
                  global_predict: Callable[[np.ndarray], float],
                  offset_ms: int = PH_OFFSET_MS) -> list[HourRun]:
    """Replay ``ForecastEngine.compute()`` once per clock hour: learn from the previous hour's
    prediction, predict this hour when at least 6 of the last 24 hours are observed, then decide
    the switch. Returns the runs that predicted."""
    params, scores, active = LocalParams(), LayerScores(), GLOBAL
    runs: list[HourRun] = []
    for hour in range(first_hour_ms, last_hour_ms + HOUR_MS, HOUR_MS):
        if params.pending_hour is not None and params.pending_hour < hour:
            s = history.get(params.pending_hour)
            params, scores = learn(params, scores, s.mobile_bytes if s is not None else None)
        if observed_last_24h(history, hour) < MIN_OBSERVED_LAST_24H:
            continue
        x = feature_vector(history, hour, offset_ms)
        g = max(0.0, float(global_predict(x)))
        ready = params.is_ready
        lh = local_hour_of(hour, offset_ms)
        loc = g * params.factor(lh) if ready else None
        if params.pending_hour != hour:
            params.pending_hour, params.pending_local_hour = hour, lh
            params.pending_global_bytes, params.pending_local_bytes = g, loc
        active = decide(active, scores, ready)
        runs.append(HourRun(hour, g, loc, active, copy.deepcopy(params), scores.hours_seen))
    return runs


def local_predictor(global_predict: Callable[[np.ndarray], float], params: LocalParams,
                    start_ms: int, offset_ms: int = PH_OFFSET_MS) -> Callable[[np.ndarray], float]:
    """The app's local projection: global × factor(local hour of the projected hour). ``project``
    calls the predictor once per hour in order, so the k-th call is hour ``start + k``."""
    k = [0]

    def predict(x: np.ndarray) -> float:
        t = start_ms + k[0] * HOUR_MS
        k[0] += 1
        return max(0.0, float(global_predict(x))) * params.factor(local_hour_of(t, offset_ms))

    return predict


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------

def next_hour_metrics(y_bytes: np.ndarray, p_bytes: np.ndarray) -> dict:
    y, p = np.asarray(y_bytes, float), np.maximum(0, np.asarray(p_bytes, float))
    if len(y) == 0:
        return {"n": 0}
    y_mb, p_mb = y / 1e6, p / 1e6
    ss = float(np.sum((y_mb - y_mb.mean()) ** 2))
    yl, pl = np.log1p(y), np.log1p(p)
    ssl = float(np.sum((yl - yl.mean()) ** 2))
    return {
        "n": int(len(y)),
        "mae_mb": float(np.mean(np.abs(y_mb - p_mb))),
        "rmse_mb": float(math.sqrt(np.mean((y_mb - p_mb) ** 2))),
        "r2": 1 - float(np.sum((y_mb - p_mb) ** 2)) / ss if ss > 0 else float("nan"),
        "r2_log": 1 - float(np.sum((yl - pl) ** 2)) / ssl if ssl > 0 else float("nan"),
        "pred_over_actual": float(p.sum() / y.sum()) if y.sum() > 0 else float("nan"),
    }


def pet_metrics(errors_h: np.ndarray, devices: np.ndarray | None = None,
                beyond: np.ndarray | None = None) -> dict:
    e = np.asarray(errors_h, float)
    if len(e) == 0:
        return {"n": 0}
    a = np.abs(e)
    out = {
        "n": int(len(e)),
        "mae_h": float(a.mean()),
        "median_abs_h": float(np.median(a)),
        "rmse_h": float(math.sqrt(np.mean(e ** 2))),
        **{f"within_{w}h": float(np.mean(a <= w)) for w in WITHIN_HOURS},
        "bias_h": float(e.mean()),
        "median_signed_h": float(np.median(e)),
        "share_late": float(np.mean(e > 0)),
        "share_early": float(np.mean(e < 0)),
    }
    if beyond is not None:
        out["beyond_horizon"] = int(np.sum(beyond))
    if devices is not None:
        per = pd.Series(a).groupby(np.asarray(devices)).mean()
        out["devices"] = int(len(per))
        out["device_mean_mae_h"] = float(per.mean())
    return out


def alert_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Confusion matrix and the usual classification scores; a score is NaN when undefined
    (e.g. precision with no alert at all)."""
    a, p = np.asarray(actual, bool), np.asarray(predicted, bool)
    if len(a) == 0:
        return {"n": 0}
    tp, fp = int(np.sum(a & p)), int(np.sum(~a & p))
    fn, tn = int(np.sum(a & ~p)), int(np.sum(~a & ~p))

    def div(x, y):
        return x / y if y else float("nan")

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if not (math.isnan(precision) or math.isnan(recall)) and precision + recall > 0 else float("nan"))
    return {"n": int(len(a)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(a), "specificity": div(tn, tn + fp),
            "positive_share": float(a.mean())}


def predicted_within(bs: list[float], start_ms: int, quota_bytes: float, hours: int) -> bool:
    """The app's alert test on a projection: the PET is at most ``hours`` after the forecast hour."""
    reach = reach_from_trajectory(bs, start_ms, quota_bytes, horizon_hours=hours)
    return reach is not None and reach - start_ms <= hours * HOUR_MS


# ---------------------------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------------------------

def booster_predictor(model) -> Callable[[np.ndarray], float]:
    booster = model.model.get_booster()
    return lambda x: float(booster.inplace_predict(np.asarray(x, np.float32).reshape(1, -1))[0])


def booster_batch_predictor(model) -> Callable[[np.ndarray], np.ndarray]:
    booster = model.model.get_booster()
    return lambda x: booster.inplace_predict(np.asarray(x, np.float32)).astype(np.float64)


def device_labels(hourly: pd.DataFrame) -> dict[str, str]:
    """Anonymous labels D01, D02, … in order of each device's first hour (then id)."""
    first = hourly.groupby("device_id")["hour_start_ms"].min().reset_index()
    first = first.sort_values(["hour_start_ms", "device_id"])
    return {d: f"D{i + 1:02d}" for i, d in enumerate(first["device_id"])}


@dataclass
class PetContext:
    """What PET scoring needs per device, with the actual crossings cached (they do not depend on
    the model). Used by :func:`evaluate` and by ``train_v0.tune`` to select hyperparameters by PET
    error."""
    histories: dict[str, dict[int, HourStats]]
    known: dict[str, dict[int, float]]
    last_hour: dict[str, int]
    starts: dict[str, list[int]]
    quotas: list[float]
    offset_ms: int = PH_OFFSET_MS
    _cache: dict = field(default_factory=dict)

    def crossings(self, device: str, start_ms: int) -> dict[float, Crossing]:
        key = (device, start_ms)
        if key not in self._cache:
            self._cache[key] = {q: actual_crossing(self.histories[device], self.known.get(device, {}), start_ms, q,
                                                   self.last_hour[device]) for q in self.quotas}
        return self._cache[key]

    def alert_labels(self, device: str, start_ms: int) -> dict[tuple[float, int], bool | None]:
        """:func:`alert_label` for every quota and alert horizon (cached; model-independent)."""
        key = ("alert", device, start_ms)
        if key not in self._cache:
            self._cache[key] = {(q, w): alert_label(self.histories[device], self.known.get(device, {}), start_ms, q,
                                                    self.last_hour[device], w)
                                for q in self.quotas for w in ALERT_HORIZONS}
        return self._cache[key]

    def errors(self, predict_batch: Callable[[np.ndarray], np.ndarray], devices, stride: int = 1) -> np.ndarray:
        """Signed PET errors (hours, + = late) of a model projected from every ``stride``-th start
        hour of ``devices``, over every quota whose crossing is scored (as in :func:`evaluate`).
        ``predict_batch`` maps an ``[n, 12]`` float32 array to n byte predictions; the projections
        run in lockstep (:func:`batch_trajectories`) so the model is called once per hour step."""
        jobs, targets = [], []
        for device in devices:
            hist = self.histories.get(device)
            if hist is None:
                continue
            for start in self.starts.get(device, [])[::stride]:
                scored = [(q, c.reach_ms) for q, c in self.crossings(device, start).items() if c.status == "crossed"]
                if scored:
                    jobs.append((hist, start, max(q for q, _ in scored)))
                    targets.append(scored)
        out = []
        for (hist, start, _), scored, bs in zip(jobs, targets, batch_trajectories(jobs, predict_batch, self.offset_ms)):
            for q, actual in scored:
                reach = reach_from_trajectory(bs, start, q)
                if reach is None:
                    reach = start + PROJECTION_HORIZON_HOURS * HOUR_MS
                out.append((reach - actual) / HOUR_MS)
        return np.asarray(out, float)


def pet_context(hourly: pd.DataFrame, known: dict[str, dict[int, float]], quotas_mb=QUOTAS_MB,
                offset_ms: int = PH_OFFSET_MS) -> PetContext:
    """Start hours are the training rows (observed, >= 6 observed of the previous 24); the data of a
    device ends at its last hour with a known NSM total."""
    histories = device_histories(hourly)
    span_max = hourly.groupby("device_id")["hour_start_ms"].max()
    last_hour, starts = {}, {}
    for device, hist in histories.items():
        k = known.get(device, {})
        last_hour[device] = max(k) if k else int(span_max[device])
        starts[device] = [h for h in sorted(hist) if observed_last_24h(hist, h) >= MIN_OBSERVED_LAST_24H]
    return PetContext(histories, known, last_hour, starts, [q * 1e6 for q in quotas_mb], offset_ms)


def evaluate(hourly: pd.DataFrame, known: dict[str, dict[int, float]], quotas_mb=QUOTAS_MB,
             fixed_params: dict | None = None, folds: int = tv.FOLDS,
             offset_ms: int = PH_OFFSET_MS, progress: Callable[[str], None] = lambda s: None) -> dict:
    """Run B4/E3 on an hourly table (``features.build_hourly_table``) and :func:`known_hourly`.

    ``fixed_params``: skip tuning and use these Tweedie params in every fold (quick runs, tests).
    Returns the report dict (anonymous labels only) plus ``_rows``/``_cycles`` frames for figures.
    """
    train = build_training_set(hourly, offset_ms)
    train[FEATURES] = train[FEATURES].astype(np.float32)
    histories = device_histories(hourly)
    labels = device_labels(hourly)
    span = hourly.groupby("device_id")["hour_start_ms"].agg(["min", "max"])
    quotas = [q * 1e6 for q in quotas_mb]
    max_q = max(quotas)
    devices = train["device_id"].to_numpy()
    ctx = pet_context(hourly, known, quotas_mb, offset_ms)

    row_records, cycle_records, skipped, device_rows, params_per_fold = [], [], [], [], []
    alert_records, alert_unknown = [], []
    for fold, held in enumerate(tv.device_folds(devices, k=min(folds, len(set(devices))))):
        te = np.isin(devices, held)
        params = fixed_params if fixed_params is not None else tv.tune(train[~te], "tweedie", pet=ctx)
        params_per_fold.append(params)
        x, y = tv._xy(train[~te])
        model = tv.TweedieModel(**params).fit(x, y)
        g_pred = booster_predictor(model)
        progress(f"fold {fold + 1}: {len(held)} devices held out, params {params}")

        for device in sorted(held):
            hist = histories[device]
            runs = replay_device(hist, int(span.loc[device, "min"]), int(span.loc[device, "max"]), g_pred, offset_ms)
            switched_hours = sum(r.active_model == LOCAL for r in runs)
            device_rows.append({
                "device": labels[device], "fold": fold + 1, "rows": len(runs),
                "hours_learned": runs[-1].hours_seen if runs else 0,
                "local_ready_hours": sum(r.local_bytes is not None for r in runs),
                "local_active_hours": switched_hours,
            })
            for r in runs:
                if r.hour_ms not in hist:
                    continue  # the app predicted, but the hour was never measured: not scorable
                xs = feature_vector(hist, r.hour_ms, offset_ms)
                actual = hist[r.hour_ms].mobile_bytes
                shown_local = r.active_model == LOCAL and r.local_bytes is not None
                rec = {"device": labels[device], "hour_ms": r.hour_ms, "actual": actual,
                       "global": r.global_bytes, "local_shadow": r.local_bytes,
                       "local_switched": r.local_bytes if shown_local else r.global_bytes,
                       "local_active": shown_local}
                for name in NEXT_HOUR_METHODS[1:]:
                    rec[name] = baseline_bytes(name, xs)
                row_records.append(rec)

                # Forecast alerts: every method's projection over the next 24 h as a yes/no prediction
                labels_now = ctx.alert_labels(device, r.hour_ms)
                for (q, w), lab in labels_now.items():
                    if lab is None:
                        alert_unknown.append({"quota_mb": q / 1e6, "horizon_h": w})
                if any(lab is not None for lab in labels_now.values()):
                    horizon = max(ALERT_HORIZONS)
                    short = {"global": trajectory(hist, r.hour_ms, max_q, g_pred, offset_ms, horizon)}
                    for name in PET_METHODS[1:]:
                        short[name] = trajectory(hist, r.hour_ms, max_q, pet_predictor(name), offset_ms, horizon)
                    for (q, w), lab in labels_now.items():
                        if lab is None:
                            continue
                        rec = {"device": labels[device], "start_ms": r.hour_ms, "quota_mb": q / 1e6,
                               "horizon_h": w, "actual": lab}
                        for name, bs in short.items():
                            rec[name] = predicted_within(bs, r.hour_ms, q, w)
                        alert_records.append(rec)

                crossings = ctx.crossings(device, r.hour_ms)
                for q, c in crossings.items():
                    if c.status != "crossed":
                        skipped.append({"quota_mb": q / 1e6, "reason": c.status})
                scored = [q for q, c in crossings.items() if c.status == "crossed"]
                if not scored:
                    continue
                trajs = {"global": trajectory(hist, r.hour_ms, max_q, g_pred, offset_ms)}
                for name in PET_METHODS[1:]:
                    trajs[name] = trajectory(hist, r.hour_ms, max_q, pet_predictor(name), offset_ms)
                if r.local_bytes is not None:
                    trajs["local_shadow"] = trajectory(
                        hist, r.hour_ms, max_q, local_predictor(g_pred, r.params, r.hour_ms, offset_ms), offset_ms)
                for q in scored:
                    actual_ms = crossings[q].reach_ms
                    rec = {"device": labels[device], "start_ms": r.hour_ms, "quota_mb": q / 1e6,
                           "actual_hours_ahead": (actual_ms - r.hour_ms) / HOUR_MS,
                           "local_active": r.active_model == LOCAL and r.local_bytes is not None}
                    for name, bs in trajs.items():
                        reach = reach_from_trajectory(bs, r.hour_ms, q)
                        rec[f"{name}_beyond"] = reach is None
                        if reach is None:
                            reach = r.hour_ms + PROJECTION_HORIZON_HOURS * HOUR_MS
                        rec[name] = (reach - actual_ms) / HOUR_MS
                    if "local_shadow" not in trajs:
                        rec["local_shadow"], rec["local_shadow_beyond"] = math.nan, False
                    rec["local_switched"] = rec["local_shadow"] if rec["local_active"] else rec["global"]
                    rec["local_switched_beyond"] = (rec["local_shadow_beyond"] if rec["local_active"]
                                                    else rec["global_beyond"])
                    cycle_records.append(rec)

    rows = pd.DataFrame(row_records)
    cycles = pd.DataFrame(cycle_records)
    skipped_df = pd.DataFrame(skipped, columns=["quota_mb", "reason"])
    report = _summarize(train, rows, cycles, skipped_df, device_rows, quotas_mb, params_per_fold, folds)
    alerts = pd.DataFrame(alert_records, columns=["device", "start_ms", "quota_mb", "horizon_h", "actual", *PET_METHODS])
    report["alerts"] = alert_summary(alerts, pd.DataFrame(alert_unknown, columns=["quota_mb", "horizon_h"]), quotas_mb)
    report["_rows"], report["_cycles"], report["_alerts"] = rows, cycles, alerts
    return report


def alert_summary(alerts: pd.DataFrame, unknown: pd.DataFrame, quotas_mb) -> dict:
    """Per alert horizon and quota (and all quotas): :func:`alert_metrics` per method, plus how many
    start hours were labelled and how many the data could not label."""
    out = {}
    for w in ALERT_HORIZONS:
        by_q = {}
        for q in [*quotas_mb, "all"]:
            a = alerts[alerts["horizon_h"] == w]
            u = unknown[unknown["horizon_h"] == w]
            if q != "all":
                a, u = a[a["quota_mb"] == q], u[u["quota_mb"] == q]
            by_q[str(q)] = {"labelled": int(len(a)), "unknown": int(len(u)),
                            **{m: alert_metrics(a["actual"].to_numpy(bool), a[m].to_numpy(bool)) for m in PET_METHODS}}
        out[f"{w}h"] = by_q
    return out


def _summarize(train, rows, cycles, skipped, device_rows, quotas_mb, params_per_fold, folds) -> dict:
    y = rows["actual"].to_numpy(float)
    next_hour = {m: next_hour_metrics(y, rows[m].to_numpy(float)) for m in NEXT_HOUR_METHODS}
    ready = rows["local_shadow"].notna()
    e3_next = {
        "global": next_hour["global"],
        "local_switched": next_hour_metrics(y, rows["local_switched"].to_numpy(float)),
        "global_on_ready": next_hour_metrics(y[ready], rows.loc[ready, "global"].to_numpy(float)),
        "local_shadow": next_hour_metrics(y[ready], rows.loc[ready, "local_shadow"].to_numpy(float)),
    }

    def pet_table(methods, subset=None):
        out = {}
        for q in [*quotas_mb, "all"]:
            c = cycles if q == "all" else cycles[cycles["quota_mb"] == q] if len(cycles) else cycles
            if subset is not None and len(c):
                c = c[subset(c)]
            out[str(q)] = {m: pet_metrics(c[m].to_numpy(float), c["device"].to_numpy(),
                                          c[f"{m}_beyond"].to_numpy(bool)) if len(c) else {"n": 0}
                           for m in methods}
        return out

    pet = pet_table(PET_METHODS)
    e3_pet = pet_table(["global", "local_switched"])
    e3_pet_ready = pet_table(["global", "local_shadow"], subset=lambda c: c["local_shadow"].notna())

    counts = {}
    for q in quotas_mb:
        s = skipped[skipped["quota_mb"] == q]["reason"].value_counts().to_dict() if len(skipped) else {}
        scored = int((cycles["quota_mb"] == q).sum()) if len(cycles) else 0
        counts[str(q)] = {"start_hours": int(len(rows)), "scored": scored,
                          **{k: int(s.get(k, 0)) for k in ("crossing_unobserved", "unknown_gap", "never_crossed")}}

    devices_df = pd.DataFrame(device_rows).sort_values("device") if device_rows else pd.DataFrame()
    return {
        "rows": int(len(train)),
        "devices": int(train["device_id"].nunique()),
        "hours_from": int(train["hour_start_ms"].min()) if len(train) else None,
        "hours_to": int(train["hour_start_ms"].max()) if len(train) else None,
        "cv": f"{folds}-fold grouped by device (train_v0.device_folds, seed 0); Tweedie tuned inside "
              f"each training fold ({tv.INNER_FOLDS} inner device folds)",
        "params_per_fold": params_per_fold,
        "next_hour": next_hour,
        "pet": pet,
        "pet_cycle_counts": counts,
        "e3": {
            "next_hour": e3_next,
            "pet": e3_pet,
            "pet_when_local_ready": e3_pet_ready,
            "devices_reaching_168_learned_hours": int((devices_df["hours_learned"] >= MIN_HOURS_FOR_LOCAL).sum())
            if len(devices_df) else 0,
            "devices_local_ready": int((devices_df["local_ready_hours"] > 0).sum()) if len(devices_df) else 0,
            "devices_switched_to_local": int((devices_df["local_active_hours"] > 0).sum()) if len(devices_df) else 0,
            "rows_local_active": int(rows["local_active"].sum()) if len(rows) else 0,
        },
        "per_device": devices_df.to_dict(orient="records"),
        "rules": {
            "start_hours": "every training row of a held-out device (observed hour, >= 6 observed of the last 24)",
            "quotas_mb": list(quotas_mb),
            "actual_pet": "measured cumulative from the start hour; long NSM windows add their spread total; "
                          "scored only if the crossing falls in an observed hour",
            "error_sign": "projected - actual in hours; positive = late",
            "beyond_horizon": "a projection that never reaches the quota in 720 h is scored as start + 720 h",
            "alerts": "every start hour and quota; predicted = the projection reaches the quota within W h "
                      "(AlertRules: 24 h, 3 h); actual = measured usage reached it within W h; starts the data "
                      "cannot label (a long NSM window across the boundary, a hole, or the data ending) are left out",
        },
    }


def sensitivity(fixed_report: dict, params: dict) -> dict:
    """The Global rows of a report made with ``params`` fixed in every fold."""
    return {
        "params": params,
        "next_hour": fixed_report["next_hour"]["global"],
        "pet": {q: by["global"] for q, by in fixed_report["pet"].items()},
    }


def final_model_importance(hourly: pd.DataFrame, fixed_params: dict | None = None,
                           known: dict[str, dict[int, float]] | None = None,
                           offset_ms: int = PH_OFFSET_MS) -> dict:
    """Gain importance of the model ``train_v0`` would ship: tuned (by PET error when ``known`` is
    given, as ``train_v0`` does) and fitted on every device."""
    train = build_training_set(hourly, offset_ms)
    train[FEATURES] = train[FEATURES].astype(np.float32)
    pet = pet_context(hourly, known, offset_ms=offset_ms) if known is not None else None
    params = fixed_params if fixed_params is not None else tv.tune(train, "tweedie", pet=pet)
    x, y = tv._xy(train)
    model = tv.TweedieModel(**params).fit(x, y)
    gain = model.model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    imp = {FEATURES[int(k[1:])]: v for k, v in gain.items()}
    ordered = sorted(((f, imp.get(f, 0.0)) for f in FEATURES), key=lambda kv: -kv[1])
    return {"params": params, "gain": {f: round(v, 4) for f, v in ordered},
            "gain_share": {f: round(v / total, 4) for f, v in ordered}}


# ---------------------------------------------------------------------------------------------
# Writing the outputs
# ---------------------------------------------------------------------------------------------

def _f(v, digits=1, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.{digits}f}"


def markdown(report: dict) -> str:
    L = []
    L.append("# B4/E3 evaluation\n")
    L.append(f"{report['rows']} hours from {report['devices']} devices; every number is on devices held out "
             f"of training ({report['cv']}).\n")

    L.append("## Next-hour usage error\n")
    L.append("| Method | MAE (MB) | RMSE (MB) | R² | R² (log) | Predicted ÷ actual total |")
    L.append("|---|---|---|---|---|---|")
    for m in NEXT_HOUR_METHODS:
        r = report["next_hour"][m]
        L.append(f"| {LABELS[m]} | {_f(r.get('mae_mb'))} | {_f(r.get('rmse_mb'))} | {_f(r.get('r2'), 2)} | "
                 f"{_f(r.get('r2_log'), 2)} | {_f(r.get('pred_over_actual'), 2)} |")
    L.append(f"\n{report['next_hour']['global'].get('n', 0)} held-out hours.\n")

    L.append("## PET error on simulated quota cycles\n")
    L.append("Error = projected − actual PET in hours; positive = late. Cycles start at every forecast hour "
             "of a held-out device; see “Rules” below.\n")
    L.append("| Quota | Method | Cycles | Devices | MAE (h) | Median abs (h) | Within ±3 h | Within ±24 h | "
             "Bias (h) | Late | Beyond 30 d |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for q, by in report["pet"].items():
        for m in PET_METHODS:
            r = by[m]
            if not r.get("n"):
                L.append(f"| {_quota(q)} | {LABELS[m]} | 0 | – | – | – | – | – | – | – | – |")
                continue
            L.append(f"| {_quota(q)} | {LABELS[m]} | {r['n']} | {r.get('devices', '–')} | {_f(r['mae_h'])} | "
                     f"{_f(r['median_abs_h'])} | {_f(r['within_3h'], pct=True)} | {_f(r['within_24h'], pct=True)} | "
                     f"{_f(r['bias_h'], 1)} | {_f(r['share_late'], pct=True)} | {r.get('beyond_horizon', 0)} |")
    L.append("\nDevice-averaged MAE (each device weighs the same, since heavy users give more cycles):\n")
    L.append("| Quota | " + " | ".join(LABELS[m] for m in PET_METHODS) + " |")
    L.append("|---|" + "---|" * len(PET_METHODS))
    for q, by in report["pet"].items():
        L.append(f"| {_quota(q)} | " + " | ".join(_f(by[m].get("device_mean_mae_h")) for m in PET_METHODS) + " |")

    sens = report.get("sensitivity_final_params")
    if sens:
        L.append("\n### Sensitivity: the final model's hyperparameters in every fold\n")
        L.append(f"Nested CV lets each fold pick its own hyperparameters. The shipped model uses "
                 f"`{json.dumps(sens['params'])}` (tuned on every device); the same held-out evaluation with "
                 f"those fixed in every fold gives next-hour MAE {_f(sens['next_hour'].get('mae_mb'))} MB and:\n")
        L.append("| Quota | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) | Beyond 30 d |")
        L.append("|---|---|---|---|---|---|---|")
        for q, r in sens["pet"].items():
            if r.get("n"):
                L.append(f"| {_quota(q)} | {r['n']} | {_f(r['mae_h'])} | {_f(r['median_abs_h'])} | "
                         f"{_f(r['within_24h'], pct=True)} | {_f(r['bias_h'])} | {r.get('beyond_horizon', 0)} |")
    L.append("\n### Which cycles could be scored\n")
    L.append("| Quota | Start hours | Scored | Crossing in an unobserved hour | Unknown hours before crossing | "
             "Never crossed in the data |")
    L.append("|---|---|---|---|---|---|")
    for q, c in report["pet_cycle_counts"].items():
        L.append(f"| {_quota(q)} | {c['start_hours']} | {c['scored']} | {c['crossing_unobserved']} | "
                 f"{c['unknown_gap']} | {c['never_crossed']} |")

    e3 = report["e3"]
    L.append("\n## E3: Global vs Local\n")
    L.append(f"Devices that learned ≥ 168 hours (the switch's minimum): **{e3['devices_reaching_168_learned_hours']}** "
             f"of {report['devices']}; local layer ready (≥ 24 learned hours) on {e3['devices_local_ready']}; "
             f"switched to Local on **{e3['devices_switched_to_local']}** ({e3['rows_local_active']} hours shown Local).\n")
    L.append("| Next hour | Hours | MAE (MB) | RMSE (MB) | Predicted ÷ actual total |")
    L.append("|---|---|---|---|---|")
    for m in ("global", "local_switched", "global_on_ready", "local_shadow"):
        r = e3["next_hour"][m]
        L.append(f"| {LABELS[m]} | {r.get('n', 0)} | {_f(r.get('mae_mb'))} | {_f(r.get('rmse_mb'))} | "
                 f"{_f(r.get('pred_over_actual'), 2)} |")
    L.append("\n| PET, all quotas | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) |")
    L.append("|---|---|---|---|---|---|")
    for key, m in (("pet", "global"), ("pet", "local_switched"),
                   ("pet_when_local_ready", "global"), ("pet_when_local_ready", "local_shadow")):
        r = e3[key]["all"][m]
        name = LABELS["global_on_ready"] if key == "pet_when_local_ready" and m == "global" else LABELS[m]
        L.append(f"| {name} | {r.get('n', 0)} | {_f(r.get('mae_h'))} | {_f(r.get('median_abs_h'))} | "
                 f"{_f(r.get('within_24h'), pct=True)} | {_f(r.get('bias_h'))} |")

    alerts = report.get("alerts")
    if alerts:
        L.append("\n## Forecast alerts as classification\n")
        L.append("The app warns when the PET is within 24 h or within 3 h (D8). At every forecast hour and quota "
                 "of a held-out device, each method's projection is a yes/no prediction, scored against whether "
                 "measured usage really reached the quota within that time. Positive = reached within the "
                 "horizon. Precision: when it warned, was it right? Recall: of the times the quota was about to "
                 "be reached, how often did it warn?\n")
        for hz, by in alerts.items():
            g = by["all"]
            L.append(f"### Within {hz} (all quotas: {g['labelled']} labelled starts, {g['unknown']} not labelable)\n")
            gm = g["global"]
            if gm.get("n"):
                L.append("Confusion matrix, Global model:\n")
                L.append("| | Actually reached | Not reached |")
                L.append("|---|---|---|")
                L.append(f"| **Alert** | {gm['tp']} (TP) | {gm['fp']} (FP) |")
                L.append(f"| **No alert** | {gm['fn']} (FN) | {gm['tn']} (TN) |\n")
            L.append("| Quota | Method | Starts | Positives | Precision | Recall | F1 | Accuracy | Specificity |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for q, row in by.items():
                for m in PET_METHODS:
                    r = row[m]
                    if not r.get("n"):
                        L.append(f"| {_quota(q)} | {LABELS[m]} | 0 | – | – | – | – | – | – |")
                        continue
                    L.append(f"| {_quota(q)} | {LABELS[m]} | {r['n']} | {_f(r['positive_share'], pct=True)} | "
                             f"{_f(r['precision'], 2)} | {_f(r['recall'], 2)} | {_f(r['f1'], 2)} | "
                             f"{_f(r['accuracy'], 2)} | {_f(r['specificity'], 2)} |")
            L.append("")

    L.append("\n## Feature importance (gain, final model)\n")
    L.append("| Feature | Gain share |")
    L.append("|---|---|")
    for f_, s in report["importance"]["gain_share"].items():
        L.append(f"| `{f_}` | {_f(s, pct=True)} |")

    L.append("\n## Per device (anonymous)\n")
    L.append("Counts are the app's hourly runs that made a forecast (including hours never measured afterwards); "
             "“learned” counts the measured hours the local layer learned from.\n")
    L.append("| Device | Fold | Forecast runs | Hours learned | Runs with local ready | Runs shown Local |")
    L.append("|---|---|---|---|---|---|")
    for d in report["per_device"]:
        L.append(f"| {d['device']} | {d['fold']} | {d['rows']} | {d['hours_learned']} | {d['local_ready_hours']} | "
                 f"{d['local_active_hours']} |")

    L.append("\n## Rules\n")
    for k, v in report["rules"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    return "\n".join(L)


def _quota(q: str) -> str:
    if q == "all":
        return "All"
    v = float(q)
    return f"{v / 1000:g} GB" if v >= 1000 else f"{v:g} MB"


# Reference categorical palette (dataviz skill, light mode), fixed order per method
COLORS = {"global": "#2a78d6", "last_hour": "#eb6834", "last_24h_avg": "#1baf7a",
          "same_hour_yesterday": "#eda100", "local_switched": "#6250d6", "local_shadow": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(report: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    def save(fig, name, layout=True):
        if layout:
            fig.tight_layout()
        fig.savefig(out / name, dpi=200, facecolor="white")
        plt.close(fig)
        written.append(name)

    # 1. Next-hour MAE and RMSE
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key, title in ((axes[0], "mae_mb", "Next-hour MAE"), (axes[1], "rmse_mb", "Next-hour RMSE")):
        vals = [report["next_hour"][m].get(key, 0) for m in NEXT_HOUR_METHODS]
        bars = ax.bar(range(len(vals)), vals, color=[COLORS[m] for m in NEXT_HOUR_METHODS], width=0.6)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK2)
        ax.set_xticks(range(len(vals)), [LABELS[m].replace(" ", "\n", 1) for m in NEXT_HOUR_METHODS], fontsize=8)
        _style(ax, title, "MB")
    save(fig, "next_hour_error.png")

    # 2. PET MAE by quota, grouped bars
    quotas = [q for q in report["pet"] if q != "all"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / len(PET_METHODS)
    for i, m in enumerate(PET_METHODS):
        vals = [report["pet"][q][m].get("mae_h", np.nan) for q in quotas]
        ax.bar(np.arange(len(quotas)) + (i - (len(PET_METHODS) - 1) / 2) * w, vals, w * 0.92,
               color=COLORS[m], label=LABELS[m])
    ns = [report["pet"][q]["global"].get("n", 0) for q in quotas]
    ax.set_xticks(range(len(quotas)), [f"{_quota(q)}\n(n = {n})" for q, n in zip(quotas, ns)], fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax, "PET error by quota size (held-out devices)", "MAE (hours)")
    save(fig, "pet_mae_by_quota.png")

    # 3. Signed PET error distribution, all quotas
    cycles = report["_cycles"]
    if len(cycles):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        data = [cycles[m].to_numpy(float) for m in PET_METHODS]
        bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 1.5})
        for patch, m in zip(bp["boxes"], PET_METHODS):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.75)
            patch.set_edgecolor(COLORS[m])
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_ylim(-72, 72)  # "last hour" repeated reaches hundreds of hours; keep the others readable
        ax.set_xticks(range(1, len(PET_METHODS) + 1), [LABELS[m].replace(" ", "\n", 1) for m in PET_METHODS], fontsize=8)
        _style(ax, "Signed PET error, all quotas (above 0 = late; box = middle 50%, axis cut at ±72 h)", "hours")
        save(fig, "pet_signed_error.png")

        # 4. Projected vs actual hours ahead, Global
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        actual = cycles["actual_hours_ahead"].to_numpy(float)
        proj = actual + cycles["global"].to_numpy(float)
        ax.scatter(actual, proj, s=10, color=COLORS["global"], alpha=0.5, linewidths=0)
        lim = max(1.0, float(np.nanmax([actual.max(), np.nanpercentile(proj, 99)])))
        ax.plot([0, lim], [0, lim], color=INK2, linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("actual hours until the quota was reached", color=INK2)
        _style(ax, "Global model: projected vs actual PET", "projected hours")
        save(fig, "pet_projected_vs_actual.png")

    # 5. Forecast alerts: Global confusion matrices, and precision / recall / F1 per method
    alerts = report.get("alerts")
    if alerts and any(alerts[h]["all"]["global"].get("n") for h in alerts):
        hz = list(alerts)
        fig, axes = plt.subplots(1, len(hz), figsize=(3.4 * len(hz), 3.2))
        axes = np.atleast_1d(axes)
        for ax, h in zip(axes, hz):
            r = alerts[h]["all"]["global"]
            m = np.array([[r.get("tp", 0), r.get("fp", 0)], [r.get("fn", 0), r.get("tn", 0)]], float)
            ax.imshow(m, cmap="Blues", vmin=0, vmax=max(1.0, m.max()))
            for (i, j), v in np.ndenumerate(m):
                ax.text(j, i, f"{int(v)}\n{['TP', 'FP', 'FN', 'TN'][i * 2 + j]}", ha="center", va="center",
                        fontsize=9, color="white" if v > m.max() / 2 else INK)
            ax.set_xticks([0, 1], ["reached", "not reached"], fontsize=8)
            ax.set_yticks([0, 1], ["alert", "no alert"], fontsize=8)
            ax.set_title(f"Within {h}, Global (all quotas)", loc="left", fontsize=10, color=INK)
            for sp in ax.spines.values():
                sp.set_visible(False)
        save(fig, "alert_confusion.png")

        fig, axes = plt.subplots(1, len(hz), figsize=(4.2 * len(hz), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        keys = ("precision", "recall", "f1")
        for ax, h in zip(axes, hz):
            w = 0.8 / len(PET_METHODS)
            for i, m in enumerate(PET_METHODS):
                vals = [alerts[h]["all"][m].get(k, np.nan) for k in keys]
                ax.bar(np.arange(len(keys)) + (i - (len(PET_METHODS) - 1) / 2) * w, vals, w * 0.92,
                       color=COLORS[m], label=LABELS[m])
            ax.set_xticks(range(len(keys)), ["Precision", "Recall", "F1"], fontsize=9)
            ax.set_ylim(0, 1)
            _style(ax, f"Alerts within {h} (all quotas)", "score")
        # One legend under both panels, so it never covers a bar
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=len(PET_METHODS), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        save(fig, "alert_scores.png", layout=False)

    # 6. Feature importance
    gain = report["importance"]["gain_share"]
    names = list(gain)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.barh(names, [gain[n] * 100 for n in names], color=COLORS["global"], height=0.6)
    ax.tick_params(axis="y", labelsize=8)
    _style(ax, "Feature importance of the final model (gain)", "")
    ax.set_xlabel("share of total gain (%)", color=INK2)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    save(fig, "feature_importance.png")
    return written


def write_outputs(report: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    (out / "report.json").write_text(json.dumps(public, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    return ["report.json", "report.md", *figures(report, out)]


def load(apps_path: str, usage_path: str) -> tuple[pd.DataFrame, dict[str, dict[int, float]]]:
    """The hourly table exactly as ``features.main`` builds it, and :func:`known_hourly`."""
    from datara_ml.features import build_hourly_table, hourly_network
    from datara_ml.nsm_hourly import nsm_hourly, windows_from_app_rows

    apps = pd.read_csv(apps_path)
    usage = pd.read_csv(usage_path, usecols=["device_id", "timestamp", "network_type", "signal_strength"])
    hourly = build_hourly_table(nsm_hourly(apps), hourly_network(usage))
    return hourly, known_hourly(windows_from_app_rows(apps))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="B4/E3 evaluation with whole devices held out.")
    p.add_argument("--apps", required=True, help="CSV export of app_usage_records (NSM windows)")
    p.add_argument("--usage", required=True, help="CSV export of usage_records (30-second rows)")
    p.add_argument("--train", help="optional: data/train.csv from datara_ml.features, checked to match")
    p.add_argument("--out", default="artifacts/eval", help="output folder")
    p.add_argument("--quotas-mb", default=",".join(str(q) for q in QUOTAS_MB))
    p.add_argument("--no-sensitivity", action="store_true",
                   help="skip re-scoring with the final model's hyperparameters in every fold")
    p.add_argument("--fixed-params", help='JSON Tweedie params to skip tuning, e.g. '
                   '\'{"tweedie_variance_power": 1.9, "max_depth": 2, "n_estimators": 150}\'')
    args = p.parse_args(argv)

    hourly, known = load(args.apps, args.usage)
    if args.train:
        given = pd.read_csv(args.train, usecols=["device_id", "hour_start_ms"])
        rebuilt = build_training_set(hourly)[["device_id", "hour_start_ms"]]
        if len(given) != len(rebuilt) or not (given.to_numpy() == rebuilt.to_numpy()).all():
            print(f"error: {args.train} ({len(given)} rows) does not match the training set rebuilt from "
                  f"--apps/--usage ({len(rebuilt)} rows); rerun datara_ml.features first", file=sys.stderr)
            return 2
    fixed = json.loads(args.fixed_params) if args.fixed_params else None
    quotas = tuple(int(q) for q in args.quotas_mb.split(","))
    report = evaluate(hourly, known, quotas, fixed, progress=print)
    report["importance"] = final_model_importance(hourly, fixed, known)
    if fixed is None and not args.no_sensitivity:
        # The model that ships is tuned on every device; nested CV mixes the folds' choices. Score
        # the shipped hyperparameters in every fold too, since the PET can depend on them.
        print(f"sensitivity: final model's params {report['importance']['params']} in every fold")
        report["sensitivity_final_params"] = sensitivity(
            evaluate(hourly, known, quotas, report["importance"]["params"]), report["importance"]["params"])
    written = write_outputs(report, Path(args.out))

    nh = report["next_hour"]
    for m in NEXT_HOUR_METHODS:
        r = nh[m]
        print(f"next hour {m:20s} MAE {r['mae_mb']:6.1f} MB  RMSE {r['rmse_mb']:6.1f}  R2 {r['r2']:.2f}  "
              f"ratio {r['pred_over_actual']:.2f}")
    for q, by in report["pet"].items():
        line = "  ".join(f"{m} {by[m]['mae_h']:.1f}h" for m in PET_METHODS if by[m].get("n"))
        print(f"PET {_quota(q):6s} n={by['global'].get('n', 0):4d}  {line}")
    for hz, by in report["alerts"].items():
        g = by["all"]["global"]
        if g.get("n"):
            print(f"alerts within {hz}: n={g['n']} TP {g['tp']} FP {g['fp']} FN {g['fn']} TN {g['tn']}  "
                  f"precision {g['precision']:.2f} recall {g['recall']:.2f} F1 {g['f1']:.2f}")
    e3 = report["e3"]
    print(f"E3: {e3['devices_reaching_168_learned_hours']} devices reached 168 learned hours, "
          f"{e3['devices_switched_to_local']} switched to Local")
    print(f"wrote {', '.join(written)} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
