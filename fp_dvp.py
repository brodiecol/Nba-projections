"""
Defense vs position from a FantasyPros table that YOU copy in (personal use).

Why a file and not an automatic download: the FantasyPros page builds its table with JavaScript (a script
can't read it from the raw page), and their terms limit copying their content to a single personal copy.
So: copy the table for each position from
    https://www.fantasypros.com/nba/defense-vs-position.php?year=2025
into the template, save it as  data/fantasypros_dvp_2025.csv  (2026 once that season's table is up), and keep
that file on your own computer (don't upload it to GitHub).

File format: one row per team per position, with these columns (extra columns are ignored):
    Position, Team, PTS, REB, AST, 3PM
    Position = PG, SG, SF, PF or C.  Team = abbreviation or name.  Stats = what that defense allows per game
    to that position.
"""
import os

import pandas as pd

# The app uses the table for the season that is current on the date. Add a line each year.
SEASON_STARTS = {2025: "2025-10-21", 2026: "2026-10-20"}      # year -> opening night

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
STATS = ["PTS", "REB", "AST", "3PM"]
STAT_ALIASES = {
    "PTS": ["pts", "points", "pt", "pts/g"],
    "REB": ["reb", "rebounds", "trb", "reb/g"],
    "AST": ["ast", "assists", "ast/g"],
    "3PM": ["3pm", "3pt", "3ptm", "3p", "3pm/g", "fg3m", "threes", "3s"],
}

_TEAMS = {
    "ATL": ["atlanta", "hawks"], "BOS": ["boston", "celtics"], "BKN": ["brooklyn", "nets", "bk", "brk"],
    "CHA": ["charlotte", "hornets", "cha", "cho"], "CHI": ["chicago", "bulls"], "CLE": ["cleveland", "cavaliers", "cavs"],
    "DAL": ["dallas", "mavericks", "mavs"], "DEN": ["denver", "nuggets"], "DET": ["detroit", "pistons"],
    "GSW": ["golden state", "warriors", "gs"], "HOU": ["houston", "rockets"], "IND": ["indiana", "pacers"],
    "LAC": ["la clippers", "clippers", "los angeles clippers"], "LAL": ["la lakers", "lakers", "los angeles lakers"],
    "MEM": ["memphis", "grizzlies"], "MIA": ["miami", "heat"], "MIL": ["milwaukee", "bucks"],
    "MIN": ["minnesota", "timberwolves", "wolves"], "NOP": ["new orleans", "pelicans", "no", "nor", "nola"],
    "NYK": ["new york", "knicks", "ny"], "OKC": ["oklahoma city", "thunder", "okl"], "ORL": ["orlando", "magic"],
    "PHI": ["philadelphia", "76ers", "sixers"], "PHX": ["phoenix", "suns", "pho"], "POR": ["portland", "trail blazers", "blazers"],
    "SAC": ["sacramento", "kings"], "SAS": ["san antonio", "spurs", "sa"], "TOR": ["toronto", "raptors"],
    "UTA": ["utah", "jazz", "utah jazz"], "WAS": ["washington", "wizards", "wsh"],
}
_LOOKUP = {}
for abbr, names in _TEAMS.items():
    _LOOKUP[abbr.lower()] = abbr
    for n in names:
        _LOOKUP[n] = abbr


def team_abbr(text):
    """'Boston Celtics' / 'BOS' / 'GS' -> NBA abbreviation, or None."""
    t = str(text).strip().lower()
    if t in _LOOKUP:
        return _LOOKUP[t]
    for key, abbr in _LOOKUP.items():                # e.g. "boston celtics" contains "celtics"
        if len(key) > 3 and key in t:
            return abbr
    return None


def dvp_year(today):
    """Which FantasyPros season to use on this date (switches automatically on opening night)."""
    t = pd.Timestamp(today)
    for y in sorted(SEASON_STARTS, reverse=True):
        if t >= pd.Timestamp(SEASON_STARTS[y]):
            return y
    return min(SEASON_STARTS)


def find_file(today, folder="data"):
    """Returns (path, year_used, note). Falls back to the previous year's file if the new one isn't there yet."""
    year = dvp_year(today)
    for y in (year, year - 1):
        path = os.path.join(folder, f"fantasypros_dvp_{y}.csv")
        if os.path.exists(path):
            note = "" if y == year else f"No {year} file yet, using {y}."
            return path, y, note
    return None, year, f"No file found. Save your table as {os.path.join(folder, f'fantasypros_dvp_{year}.csv')}."


def _first_number(series):
    return pd.to_numeric(series.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def load_table(path):
    """Read the CSV/Excel file. Returns (skews, notes): skews = {team: {position: {stat: multiplier}}}."""
    df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    notes = []
    cols = {str(c).strip().lower(): c for c in df.columns}
    pos_col = next((cols[k] for k in ("position", "pos") if k in cols), None)
    team_col = next((cols[k] for k in ("team", "opp", "opponent", "defense", "name") if k in cols), None)
    if pos_col is None or team_col is None:
        raise ValueError("Need a 'Position' column and a 'Team' column.")
    stat_col = {}
    for s, aliases in STAT_ALIASES.items():
        c = next((cols[a] for a in [s.lower()] + aliases if a in cols), None)
        if c is None:
            raise ValueError(f"Couldn't find a {s} column (looked for: {', '.join([s] + aliases)}).")
        stat_col[s] = c

    t = pd.DataFrame({"team": df[team_col].map(team_abbr), "pos": df[pos_col].astype(str).str.strip().str.upper()})
    for s in STATS:
        t[s] = _first_number(df[stat_col[s]])
    bad = int(t["team"].isna().sum())
    if bad:
        notes.append(f"{bad} row(s) had a team name I couldn't match and were skipped.")
    t = t.dropna(subset=["team"] + STATS)
    t = t[t["pos"].isin(POSITIONS)]
    found = sorted(t["pos"].unique(), key=POSITIONS.index)
    if len(found) < len(POSITIONS):
        notes.append(f"Only found positions: {', '.join(found) or 'none'}. All five (PG SG SF PF C) are needed.")
        return {}, notes
    t = t.groupby(["team", "pos"], as_index=False)[STATS].mean()

    skews = {}
    league = t.groupby("pos")[STATS].mean()                       # league average allowed, per position
    for team, g in t.groupby("team"):
        if set(g["pos"]) != set(POSITIONS):
            notes.append(f"{team} is missing a position and was skipped.")
            continue
        g = g.set_index("pos")
        for s in STATS:
            # position-specific allowed vs league, divided by the team's overall level, so only the
            # *position skew* is left (the overall defense is already handled by the team opponent factor)
            team_rel = g[s].sum() / league[s].sum()
            for p in POSITIONS:
                skews.setdefault(team, {}).setdefault(p, {})[s] = float((g.loc[p, s] / league.loc[p, s]) / team_rel)
    if len(skews) < 28:
        notes.append(f"Only {len(skews)} teams loaded (expected 30).")
    return skews, notes


def load_position_overrides(path=os.path.join("data", "positions.csv")):
    """Optional: your own position labels. CSV with columns Player, Position (e.g. PG, or PG/SG for two)."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "player" not in cols or "position" not in cols:
        return {}
    out = {}
    for name, pos in zip(df[cols["player"]], df[cols["position"]]):
        ps = [p.strip().upper() for p in str(pos).replace(",", "/").split("/")]
        ps = [p for p in ps if p in POSITIONS]
        if ps:
            out[str(name).strip()] = ps
    return out


_NBA_TO_FIVE = {"G": ["PG", "SG"], "G-F": ["SG", "SF"], "F-G": ["SF", "SG"], "F": ["SF", "PF"],
                "F-C": ["PF", "C"], "C-F": ["C", "PF"], "C": ["C"]}


def positions_for(pos_full, override=None):
    """The NBA only labels G / F / C. Map that onto FantasyPros' five positions (or use your override)."""
    if override:
        return override
    return _NBA_TO_FIVE.get(str(pos_full).strip().upper(), [])


def player_skew(skews, opp, positions, stat):
    """Average the multiplier over the player's position(s); 1.0 if unknown."""
    vals = [skews[opp][p][stat] for p in positions if opp in skews and p in skews[opp]]
    return sum(vals) / len(vals) if vals else 1.0
