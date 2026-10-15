"""
Sportsbook lines: parse what you paste, then compare every line with the model's probabilities.

Paste one line per row, in any of these forms (commas, tabs or | between the parts; odds are optional):
    Jayson Tatum, PTS, 27.5, -115, -105
    Jayson Tatum  Points  27.5
    Nikola Jokic | Pts+Reb+Ast | 48.5 | 1.87 | 1.95
Odds can be American (-110, +120) or decimal (1.91). With no odds, -110 on both sides is assumed.

The model's chance of going over/under comes from the projection and the spread of results
(distributions.py: fitted by backtest.py once you've run it, rough defaults before that).
Expected value (EV) counts a push (landing exactly on a whole-number line) as a refunded stake.
"""
import difflib
import re
import unicodedata

import pandas as pd

import distributions

STAT_ALIASES = {
    "PTS": ["pts", "points", "point", "pt"],
    "REB": ["reb", "rebs", "rebounds", "rebound"],
    "AST": ["ast", "asts", "assists", "assist"],
    "3PM": ["3pm", "3pt", "3pts", "3ptm", "3p", "3s", "fg3m", "threes", "three", "3-pointers", "3pointers",
            "3 pointers made", "three pointers made", "3-pointers made", "3pointers made", "threes made"],
    "PRA": ["pra", "pts+reb+ast", "pts+rebs+asts", "points+rebounds+assists", "p+r+a", "pts+rebs+ast"],
    "PR": ["pr", "pts+reb", "pts+rebs", "points+rebounds", "p+r"],
    "PA": ["pa", "pts+ast", "pts+asts", "points+assists", "p+a"],
    "RA": ["ra", "reb+ast", "rebs+asts", "rebounds+assists", "r+a"],
}
_STAT_LOOKUP = {a: s for s, al in STAT_ALIASES.items() for a in al + [s.lower()]}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_stat(text):
    t = re.sub(r"\s*\+\s*", "+", str(text).strip().lower())
    t = re.sub(r"\s+", " ", t)
    return _STAT_LOOKUP.get(t)


def norm_name(text):
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    t = re.sub(r"[.'`’\-]", " ", t.lower())
    parts = [p for p in re.sub(r"[^a-z0-9 ]", " ", t).split() if p not in _SUFFIXES]
    return " ".join(parts)


def match_player(name, names):
    """Best match for a pasted name among the projected players; None if nothing is close enough."""
    index = {}
    for n in names:
        index.setdefault(norm_name(n), n)
    key = norm_name(name)
    if key in index:
        return index[key]
    close = difflib.get_close_matches(key, list(index), n=1, cutoff=0.86)
    return index[close[0]] if close else None


def to_decimal(token):
    """American (-110, +120) or decimal (1.91) odds -> decimal odds, or None."""
    t = str(token).strip().replace("−", "-").replace("–", "-")
    if not t:
        return None
    try:
        if t[0] in "+-" or ("." not in t and t.lstrip("+-").isdigit() and abs(float(t)) >= 100):
            v = float(t)
            if abs(v) < 100:
                return None
            return 1 + (100 / abs(v) if v < 0 else v / 100)
        v = float(t)
        return v if 1.01 <= v <= 50 else None
    except ValueError:
        return None


_STAT_WORDS = "|".join(sorted((re.escape(a) for a in _STAT_LOOKUP), key=len, reverse=True))
_SPACE_FORM = re.compile(rf"^(?P<player>.+?)\s+(?P<stat>{_STAT_WORDS})\s+(?P<line>\d+(?:\.\d+)?)(?P<rest>.*)$", re.I)


def parse_lines(text, names):
    """
    Returns (lines DataFrame, problems). Columns: Player, Stat, Line, OverDec, UnderDec, Input.
    problems = [(original line, reason)].
    """
    rows, problems = [], []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.lower().startswith(("player", "name")):
            continue
        parts = [p.strip() for p in re.split(r"[,\t|;]+", line) if p.strip()]
        rest = []
        if len(parts) >= 3:
            player, stat_txt, line_txt, rest = parts[0], parts[1], parts[2], parts[3:]
        else:
            m = _SPACE_FORM.match(line)
            if not m:
                problems.append((raw, "couldn't find player, stat and line"))
                continue
            player, stat_txt, line_txt = m.group("player"), m.group("stat"), m.group("line")
            rest = m.group("rest").split()
        stat = norm_stat(stat_txt)
        if stat is None:
            problems.append((raw, f"unknown stat '{stat_txt}'"))
            continue
        try:
            value = float(line_txt)
        except ValueError:
            problems.append((raw, f"'{line_txt}' isn't a number"))
            continue
        who = match_player(player, names)
        if who is None:
            problems.append((raw, f"no projected player matches '{player}' (is he playing today?)"))
            continue
        over = to_decimal(rest[0]) if len(rest) >= 1 else None
        under = to_decimal(rest[1]) if len(rest) >= 2 else None
        over = over if over is not None else 1 + 100 / 110            # no odds given: assume -110
        under = under if under is not None else 1 + 100 / 110
        rows.append({"Player": who, "Stat": stat, "Line": value, "OverDec": over, "UnderDec": under, "Input": player})
    return pd.DataFrame(rows, columns=["Player", "Stat", "Line", "OverDec", "UnderDec", "Input"]), problems


def evaluate(lines, calc, disp, corr):
    """Add the model's numbers to each line and rank by expected value. calc needs Player, Team, Game, Min, PTS.."""
    if lines.empty:
        return pd.DataFrame()
    by_player = calc.drop_duplicates("Player").set_index("Player")
    out = []
    for r in lines.itertuples(index=False):
        p = by_player.loc[r.Player]
        means = {k: float(p[k]) for k in distributions.STATS}
        mean, var = distributions.mean_var(r.Stat, means, disp, corr)
        po, pu = distributions.prob_over(mean, var, r.Line), distributions.prob_under(mean, var, r.Line)
        ev_over = po * (r.OverDec - 1) - pu
        ev_under = pu * (r.UnderDec - 1) - po
        side, p_side, dec, ev = ("Over", po, r.OverDec, ev_over) if ev_over >= ev_under else ("Under", pu, r.UnderDec, ev_under)
        lo, hi = distributions.interval(mean, var)
        out.append({"Player": r.Player, "Team": p["Team"], "Game": p["Game"], "Min": float(p["Min"]), "Stat": r.Stat,
                    "Line": r.Line, "Projection": mean, "80% range": f"{lo}–{hi}", "P(over)": po, "P(under)": pu,
                    "Best side": side, "Model P": p_side, "Odds (dec)": dec, "Breakeven": 1 / dec,
                    "Edge (pts)": 100 * (p_side - 1 / dec), "EV %": 100 * ev})
    return pd.DataFrame(out).sort_values("EV %", ascending=False).reset_index(drop=True)
