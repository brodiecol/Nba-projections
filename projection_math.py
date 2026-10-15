"""The projection formulas (same math as the spreadsheet), kept separate so they are easy to test."""
import math

import numpy as np
import pandas as pd

STATS = ["PTS", "REB", "AST", "3PM"]
OFFENSE_STATS = ("PTS", "AST", "3PM")          # stats moved by the team 'offense adjustment' slider


def compute(df, teams, league_pace, home_adj, away_adj, b2b_adj):
    """
    df needs columns: Team, Opp, H/A, B2B, Default min, Your min (blank = use default),
                      Role adj % (5 = +5%), rate_PTS, rate_REB, rate_AST, rate_3PM
    optional columns: Pace adj % (whole game), Offense adj % (this player's team; moves PTS, AST, 3PM),
                      dvp_PTS, dvp_REB, dvp_AST, dvp_3PM (defense-vs-position multipliers, 1.0 = no effect),
                      tm_PTS, tm_REB, tm_AST, tm_3PM (teammate-out multipliers, 1.0 = no effect)
    teams: {abbr: {"pace": x, "PTS": opp factor, "REB": ..., "AST": ..., "3PM": ...}}
    Adjustments are fractions (0.01 = +1%).
    Returns a copy of df with Min, factors, the four projections and PRA, PR, PA, RA.
    """
    out = df.copy()
    out["Min"] = out["Your min"].where(out["Your min"].notna(), out["Default min"]).astype(float)
    for c in ("Pace adj %", "Offense adj %"):
        if c not in out:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)

    pace_team = pd.to_numeric(out["Team"].map(lambda t: teams.get(t, {}).get("pace")), errors="coerce")
    pace_opp = pd.to_numeric(out["Opp"].map(lambda t: teams.get(t, {}).get("pace")), errors="coerce")
    out["Pace factor"] = ((pace_team + pace_opp - league_pace) / pace_team).fillna(1.0)   # unknown team -> neutral

    venue = np.where(out["H/A"] == "H", home_adj, away_adj)
    rest = np.where(out["B2B"] == "Y", b2b_adj, 0.0)
    out["Venue/rest factor"] = (1 + venue) * (1 + rest)

    role = 1 + out["Role adj %"].fillna(0.0) / 100.0
    pace_adj = 1 + out["Pace adj %"] / 100.0
    off_adj = 1 + out["Offense adj %"] / 100.0
    base = out["Min"] * out["Pace factor"] * out["Venue/rest factor"] * role * pace_adj
    for s in STATS:
        out[f"oppf_{s}"] = out["Opp"].map(lambda t, s=s: teams.get(t, {}).get(s, 1.0)).astype(float)
        dvp = out[f"dvp_{s}"].fillna(1.0) if f"dvp_{s}" in out else 1.0          # defense-vs-position (optional)
        tm = out[f"tm_{s}"].fillna(1.0) if f"tm_{s}" in out else 1.0              # teammate-out effect (optional)
        out[s] = base * out[f"rate_{s}"] * out[f"oppf_{s}"] * dvp * tm * (off_adj if s in OFFENSE_STATS else 1.0)

    out["PRA"] = out["PTS"] + out["REB"] + out["AST"]
    out["PR"] = out["PTS"] + out["REB"]
    out["PA"] = out["PTS"] + out["AST"]
    out["RA"] = out["REB"] + out["AST"]
    return out


def bench_projection(team_row, minutes, league_rates):
    """
    Projection for the 'other players' bucket (everyone not listed). Uses the league-average per-minute
    rates and the same pace / venue / opponent / adjustment factors as the team's listed players.
    team_row: any one calc() row from that team (they share the same game factors).
    """
    base = (minutes * team_row["Pace factor"] * team_row["Venue/rest factor"]
            * (1 + team_row["Pace adj %"] / 100.0))
    out = {}
    for s in STATS:
        mult = (1 + team_row["Offense adj %"] / 100.0) if s in OFFENSE_STATS else 1.0
        out[s] = float(base * league_rates[s] * team_row[f"oppf_{s}"] * mult)
    return out


def balance_minutes(defaults, fixed, caps=None, total=240.0, default_cap=48.0):
    """
    Make a team's minutes add up to `total`.
      defaults: {name: default minutes} for everyone on the team (including the 'others' bucket)
      fixed:    {name: minutes the user set}; these stay exactly as set
    Everyone else is scaled up or down in proportion to their default minutes (respecting each cap).
    Returns ({name: minutes}, status) where status is
      'ok'        total reached
      'all_fixed' every player is locked and the total isn't `total`
      'over'      locked players alone already exceed `total`
      'capped'    couldn't reach `total` because players hit their caps
    """
    caps = caps or {}
    cap = lambda k: caps.get(k, default_cap)
    fixed = {k: min(max(float(v), 0.0), cap(k)) for k, v in fixed.items() if k in defaults}
    mins = dict(fixed)
    free = [k for k in defaults if k not in fixed]
    if not free:
        return mins, ("ok" if abs(sum(mins.values()) - total) < 0.05 else "all_fixed")
    remaining = total - sum(fixed.values())
    if remaining <= 0:
        for k in free:
            mins[k] = 0.0
        return mins, ("ok" if abs(remaining) < 0.05 else "over")
    status, active = "ok", list(free)
    while active:
        w = sum(defaults[k] for k in active)
        share = {k: (defaults[k] / w if w > 0 else 1.0 / len(active)) for k in active}
        over = [k for k in active if remaining * share[k] > cap(k) + 1e-9]
        if not over:
            for k in active:
                mins[k] = remaining * share[k]
            return mins, status
        status = "capped"
        for k in over:
            mins[k] = cap(k)
            remaining -= cap(k)
            active.remove(k)
        if remaining <= 0:
            for k in active:
                mins[k] = 0.0
            return mins, status
    return mins, "capped"


def team_score_topdown(team, opp, is_home, b2b, teams, league_pace, league_ortg,
                       home_adj, away_adj, b2b_adj, pace_adj_pct=0.0, off_adj_pct=0.0):
    """
    Team score from team ratings alone (no players):
        possessions  = this team's pace + opponent's pace - league pace      (the game's expected pace)
        efficiency   = this team's ORTG + opponent's DRTG - league ORTG      (points per 100 possessions)
        score        = possessions x efficiency / 100, then home/away and back-to-back adjustments
    Returns None if the ratings aren't available (e.g. an older data file).
    """
    t, o = teams.get(team, {}), teams.get(opp, {})
    if league_ortg is None or "ortg" not in t or "drtg" not in o or "pace" not in t or "pace" not in o:
        return None
    poss = (t["pace"] + o["pace"] - league_pace) * (1 + pace_adj_pct / 100.0)
    eff = t["ortg"] + o["drtg"] - league_ortg
    venue = (1 + (home_adj if is_home else away_adj)) * (1 + (b2b_adj if b2b else 0.0))
    return float(poss * eff / 100.0 * venue * (1 + off_adj_pct / 100.0))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def expected_excess(margin, sd=12.0, free=10.0):
    """E[max(0, |M| - free)] where the final margin M ~ Normal(margin, sd): how far past `free` points we expect it to go."""
    def one_side(mu):                                  # E[(X - free)+] for X ~ Normal(mu, sd)
        d = (mu - free) / sd
        return (mu - free) * _norm_cdf(d) + sd * _norm_pdf(d)
    return one_side(margin) + one_side(-margin)


def blowout_factor(margin, c, sd=12.0, floor=0.6):
    """Share of their usual minutes the starters keep, given the expected margin. c comes from blowout_coefficient()."""
    return max(floor, 1.0 - c * expected_excess(margin, sd))


def minutes_bucket(m):
    return "lt20" if m < 20 else ("20to30" if m < 30 else "30plus")


def apply_calibration(df, cal):
    """
    Multiply each projected stat by a small correction learned in the backtest, by minutes bucket.
    cal = {"PTS": {"lt20": 1.01, "20to30": 0.99, "30plus": 1.0}, ...}. Recomputes the combo columns.
    """
    out = df.copy()
    bucket = out["Min"].map(minutes_bucket)
    for s in STATS:
        factors = cal.get(s, {})
        out[s] = out[s] * bucket.map(lambda b, f=factors: f.get(b, 1.0)).astype(float)
    out["PRA"] = out["PTS"] + out["REB"] + out["AST"]
    out["PR"] = out["PTS"] + out["REB"]
    out["PA"] = out["PTS"] + out["AST"]
    out["RA"] = out["REB"] + out["AST"]
    return out
