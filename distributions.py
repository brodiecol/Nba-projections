"""
Ranges and over/under probabilities for the projections.

A projection is an AVERAGE. Real results scatter around it, and the scatter grows with the projection. We model
the variance of a stat as   variance = a x mean + b x mean^2   (a count-like part plus an extra-spread part),
with a, b learned in the backtest (backtest.py writes data/dispersion.json). Until you run the backtest the app
uses the rough starting values below, which are my estimates, not fitted numbers.

Combos (PRA etc.) add the variances of their parts, plus the way the parts move together (correlations).
"""
import json
import math
import os

import numpy as np

STATS = ["PTS", "REB", "AST", "3PM"]
COMBOS = {"PRA": ["PTS", "REB", "AST"], "PR": ["PTS", "REB"], "PA": ["PTS", "AST"], "RA": ["REB", "AST"]}

DEFAULT_DISPERSION = {"PTS": (1.0, 0.07), "REB": (1.0, 0.045), "AST": (1.0, 0.04), "3PM": (0.9, 0.0)}   # (a, b)
DEFAULT_CORR = {("PTS", "REB"): 0.05, ("PTS", "AST"): 0.25, ("REB", "AST"): 0.0,
                ("PTS", "3PM"): 0.5, ("REB", "3PM"): -0.05, ("AST", "3PM"): 0.05}

PARAMS_PATH = os.path.join("data", "dispersion.json")


def load_params(path=PARAMS_PATH):
    """Returns (dispersion, corr, fitted). fitted=False means the rough defaults are in use."""
    try:
        with open(path) as f:
            raw = json.load(f)
        disp = {s: tuple(raw["dispersion"][s]) for s in STATS}
        corr = {tuple(k.split("|")): v for k, v in raw["corr"].items()}
        return disp, corr, True
    except (OSError, ValueError, KeyError):
        return DEFAULT_DISPERSION, DEFAULT_CORR, False


def save_params(dispersion, corr, path=PARAMS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"dispersion": {s: list(v) for s, v in dispersion.items()},
                   "corr": {f"{a}|{b}": float(v) for (a, b), v in corr.items()}}, f)


def variance(stat, mean, disp):
    a, b = disp[stat]
    return max(a * mean + b * mean * mean, 0.05)


def _corr(a, b, corr):
    return corr.get((a, b), corr.get((b, a), 0.0))


def combo_variance(parts, means, disp, corr):
    v = sum(variance(p, means[p], disp) for p in parts)
    for i, p in enumerate(parts):
        for q in parts[i + 1:]:
            v += 2 * _corr(p, q, corr) * math.sqrt(variance(p, means[p], disp) * variance(q, means[q], disp))
    return max(v, 0.05)


def mean_var(stat, means, disp, corr):
    """(mean, variance) for a stat or a combo name, from the four projected means {'PTS':..,'REB':..,'AST':..,'3PM':..}."""
    if stat in COMBOS:
        parts = COMBOS[stat]
        return sum(means[p] for p in parts), combo_variance(parts, means, disp, corr)
    return means[stat], variance(stat, means[stat], disp)


def pmf(mean, var, kmax=None):
    """Probability of each whole-number outcome 0, 1, 2, ... (negative binomial if the stat is over-dispersed,
    otherwise a rounded normal)."""
    if mean < 1e-3:
        return np.array([1.0])                       # nobody scores with zero expected minutes
    sd = math.sqrt(var)
    kmax = kmax or int(min(400, mean + 9 * sd + 12))
    k = np.arange(kmax + 1)
    if var > mean * 1.02:
        r = mean * mean / (var - mean)
        p = r / (r + mean)
        logp = (np.array([math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1) for x in k])
                + r * math.log(p) + k * math.log(1 - p))
        out = np.exp(logp)
    else:
        cdf = np.array([0.5 * (1 + math.erf((x + 0.5 - mean) / (sd * math.sqrt(2)))) for x in k])
        out = np.diff(np.concatenate([[0.0], cdf]))
    s = out.sum()
    return out / s if s > 0 else out


def prob_over(mean, var, line):
    """P(result > line). A line of 24.5 means 25 or more; a line of 24 means 25 or more too (a push isn't an over)."""
    p = pmf(mean, var)
    k = int(math.floor(line)) + 1
    return float(p[k:].sum()) if k < len(p) else 0.0


def prob_under(mean, var, line):
    p = pmf(mean, var)
    k = int(math.ceil(line)) - 1                       # results strictly below the line
    return float(p[: k + 1].sum()) if k >= 0 else 0.0


def interval(mean, var, lo=0.10, hi=0.90):
    """Whole-number range that should hold the result ~80% of the time (lo..hi quantiles)."""
    cdf = np.cumsum(pmf(mean, var))
    a = int(np.searchsorted(cdf, lo))
    b = int(np.searchsorted(cdf, hi))
    return a, b


# ------------------------------------------------------------------ fitting (used by the backtest)
def fit_dispersion(pred, actual):
    """Least-squares fit of (actual - pred)^2 ~ a*pred + b*pred^2 with a, b >= 0. Returns (a, b)."""
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    y = (actual - pred) ** 2
    X = np.column_stack([pred, pred ** 2])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if b < 0:                                           # refit with only the count-like part
        a, b = float((pred * y).sum() / (pred * pred).sum()), 0.0
    if a < 0:                                           # refit with only the extra-spread part
        a, b = 0.0, float((pred ** 2 * y).sum() / (pred ** 4).sum())
    return max(a, 0.0), max(b, 0.0)


def fit_corr(resid):
    """resid: {stat: array of (actual - pred)} for the four stats. Returns {(a, b): correlation}."""
    out = {}
    for i, a in enumerate(STATS):
        for b in STATS[i + 1:]:
            c = np.corrcoef(resid[a], resid[b])[0, 1]
            out[(a, b)] = float(0.0 if np.isnan(c) else c)
    return out
