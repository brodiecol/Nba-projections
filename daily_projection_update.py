"""
Daily NBA projection updater
============================
Pulls fresh NBA data, works out per-minute rates, default minutes, team pace and
opponent factors, and fills in the blue cells of NBA_Projections_Template.xlsx.
All the projection formulas live in the spreadsheet; this script only loads data.

HOW TO RUN (Google Colab)
  1. New notebook at colab.research.google.com
  2. Cell 1:   !pip install nba_api
  3. Cell 2:   paste this whole file and run it. When asked, upload
               NBA_Projections_Template.xlsx. A dated .xlsx downloads when it finishes.

NOTE: stats.nba.com sometimes blocks cloud servers such as Colab. If you see
timeout errors, run the same file on your own computer instead
(pip install nba_api pandas openpyxl, then: python daily_projection_update.py).
"""
import datetime as dt
import os
import pickle
import time
from copy import copy

import numpy as np
import pandas as pd

# =====================================================================
# SETTINGS YOU CAN TUNE (these are my starting values, not tested on data)
# =====================================================================
TEMPLATE_FILE = "NBA_Projections_Template.xlsx"
PREVIOUS_FILE = None          # e.g. "NBA_Projections_2026-10-27.xlsx" to carry over your minute/role overrides
STATS = ["PTS", "REB", "AST", "FG3M"]

RATE_HALF_LIFE_GAMES = 25     # a game this many games ago counts half as much in the per-minute rate
PRIOR_MINUTES = 150           # pulls rates toward league average; bigger = more caution with small samples
PRIOR_TSA = 300               # volume x efficiency: shooting efficiency is pulled toward league average as if this many shot attempts
PRIOR_3PA = 200               # ...and 3-point % as if this many 3-point attempts
MINUTES_WINDOW_GAMES = 15     # default minutes look at this many recent games...
MINUTES_HALF_LIFE_GAMES = 5   # ...weighted so recent games count more
MIN_GAMES_REQUIRED = 3        # skip players with fewer total games in the last two seasons (rookies appear after this many)
MIN_DEFAULT_MINUTES = 10      # skip players whose default minutes are below this
TEAM_BLEND_GAMES = 15         # team factors: weight on this season = GP / (GP + this); rest is last season

FIRST_ROW = 5                 # first data row on the Projections tab
N_COLS = 27                   # columns A..AA


# =====================================================================
# 1. FETCHING DATA (needs internet)
# =====================================================================
CACHE_DIR = os.path.join("data", "cache")


def _cached(key, max_age_hours, fetch):
    """
    Saves what we download so we ask the NBA's site less often (it blocks people who ask too much).
    Fresh saved copy -> use it. Otherwise download; if the download fails, fall back to an older saved copy.
    A result of None is never saved.
    """
    path = os.path.join(CACHE_DIR, key + ".pkl")
    age = (time.time() - os.path.getmtime(path)) / 3600 if os.path.exists(path) else None
    if age is not None and age <= max_age_hours:
        with open(path, "rb") as f:
            return pickle.load(f)
    try:
        val = fetch()
    except Exception as e:  # noqa: BLE001
        if age is None:
            raise
        print(f"  (couldn't download {key} ({type(e).__name__}); using a saved copy that is {age:.0f} hours old)")
        with open(path, "rb") as f:
            return pickle.load(f)
    if val is not None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(val, f)
    return val


def _season_is_over(season):
    """True once a season is finished, so its numbers can be saved for good."""
    return season != season_strings(pd.Timestamp.now())[0]


def _call(endpoint, tries=3, **kwargs):
    """Call an nba_api endpoint with retries; returns the first table as a DataFrame."""
    err = None
    for i in range(tries):
        try:
            return endpoint(timeout=60, **kwargs).get_data_frames()[0]
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (i + 1))
    raise err


def season_strings(today):
    start = today.year if today.month >= 10 else today.year - 1
    fmt = lambda y: f"{y}-{str(y + 1)[-2:]}"
    return fmt(start), fmt(start - 1)


SCHEDULE_URLS = ["https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json",
                 "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"]
GAMES_FILE = os.path.join("data", "games_today.csv")      # games you typed in (Date, Away, Home); see the app's sidebar


def today_et():
    """Today's date in US Eastern time (roughly), which is how NBA game days are dated."""
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).date()


def _games_from_file(day, path=None):
    """Games you typed in yourself, as [(home_team_id, away_team_id)], or None if there are none for that day."""
    path = path or GAMES_FILE
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "away" not in cols or "home" not in cols:
        return None
    if "date" in cols:
        df = df[pd.to_datetime(df[cols["date"]]).dt.strftime("%Y-%m-%d") == pd.Timestamp(day).strftime("%Y-%m-%d")]
    from nba_api.stats.static import teams as static_teams

    abbr2id = {t["abbreviation"]: t["id"] for t in static_teams.get_teams()}
    pairs = []
    for a, h in zip(df[cols["away"]], df[cols["home"]]):
        a, h = str(a).strip().upper(), str(h).strip().upper()
        if a in abbr2id and h in abbr2id:
            pairs.append((abbr2id[h], abbr2id[a]))
    return pairs or None


def _games_from_schedule(day):
    """Games on a date from the NBA's published season schedule file (a plain download, no stats-site call)."""
    import requests

    def go():
        last = None
        for url in SCHEDULE_URLS:
            try:
                r = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = e
        raise last

    data = _cached("schedule_league", 12, go)
    target = pd.Timestamp(day).strftime("%Y-%m-%d")
    pairs = []
    for gd in data["leagueSchedule"]["gameDates"]:
        for g in gd.get("games", []):
            when = str(g.get("gameDateEst") or g.get("gameDateTimeEst") or "")[:10]
            if not when:
                when = pd.to_datetime(gd.get("gameDate")).strftime("%Y-%m-%d")
            if when == target:
                pairs.append((int(g["homeTeam"]["teamId"]), int(g["awayTeam"]["teamId"])))
    return pairs


def fetch_today(date=None):
    """
    Returns (date, [(home_team_id, away_team_id), ...]). Tries, in order: games you typed in, the live scoreboard
    (today only), the NBA's season schedule file, and the stats site. A date like "2026-04-08" looks up that day.
    """
    target = pd.Timestamp(date) if date is not None else pd.Timestamp(today_et())
    errors = []

    try:
        pairs = _games_from_file(target)
        if pairs:
            print(f"  Using the {len(pairs)} game(s) you entered ({GAMES_FILE}).")
            return target, pairs
    except Exception as e:  # noqa: BLE001
        errors.append(f"typed-in games: {type(e).__name__}: {e}")

    if date is None:
        try:
            from nba_api.live.nba.endpoints import scoreboard

            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()
            if games:
                return pd.Timestamp(board.score_board_date), [
                    (g["homeTeam"]["teamId"], g["awayTeam"]["teamId"]) for g in games]
        except Exception as e:  # noqa: BLE001
            errors.append(f"live scoreboard: {type(e).__name__}")
            print(f"  (live scoreboard unavailable: {type(e).__name__}; trying the NBA schedule file)")

    schedule_pairs, schedule_ok = [], False
    try:
        schedule_pairs, schedule_ok = _games_from_schedule(target), True
        if schedule_pairs:
            return target, schedule_pairs
    except Exception as e:  # noqa: BLE001
        errors.append(f"schedule file: {type(e).__name__}: {e}")
        print(f"  (schedule file unavailable: {type(e).__name__}; trying the stats site)")

    try:
        from nba_api.stats.endpoints import scoreboardv2

        df = _call(scoreboardv2.ScoreboardV2, game_date=target.strftime("%m/%d/%Y"))
        pairs = list(zip(df["HOME_TEAM_ID"].astype(int), df["VISITOR_TEAM_ID"].astype(int))) if not df.empty else []
        return target, pairs
    except Exception as e:  # noqa: BLE001
        errors.append(f"stats site: {type(e).__name__}")
        if schedule_ok:                       # the schedule loaded fine and has no games that day
            return target, []
        raise RuntimeError("couldn't load the list of games from any source (" + "; ".join(errors) + "). "
                           "In the app, use 'Games not loading?' in the sidebar to type in today's games, "
                           "or test with a past date: py refresh_data.py 2026-04-08") from e


def fetch_logs(season):
    """Every player's game log for the season. Finished seasons are saved for good; the current one for 6 hours."""
    def go():
        from nba_api.stats.endpoints import leaguegamelog

        return _call(leaguegamelog.LeagueGameLog, season=season,
                     player_or_team_abbreviation="P", season_type_all_star="Regular Season")

    return _cached(f"logs_{season}", 24 * 400 if _season_is_over(season) else 6, go)


def fetch_team_table(season):
    """One row per team: PACE and opponent per-100-possession stats. None if the season has no games yet."""
    def go():
        from nba_api.stats.endpoints import leaguedashteamstats

        adv = _call(leaguedashteamstats.LeagueDashTeamStats, season=season,
                    measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
                    season_type_all_star="Regular Season")
        opp = _call(leaguedashteamstats.LeagueDashTeamStats, season=season,
                    measure_type_detailed_defense="Opponent", per_mode_detailed="Per100Possessions",
                    season_type_all_star="Regular Season")
        if adv.empty or opp.empty:
            return None
        df = adv[["TEAM_ID", "GP", "PACE"]].merge(
            opp[["TEAM_ID", "OPP_PTS", "OPP_REB", "OPP_AST", "OPP_FG3M"]], on="TEAM_ID")
        return df if df["GP"].max() > 0 else None

    try:
        return _cached(f"teamstats_{season}", 24 * 400 if _season_is_over(season) else 6, go)
    except Exception as e:  # noqa: BLE001
        print(f"  (no team stats for {season}: {type(e).__name__}: {e})")
        return None


def fetch_current_team_map(season):
    """player_id -> current team_id (handles offseason trades). Returns {} if unavailable."""
    try:
        df = fetch_player_index(season)
        df = df[df["TEAM_ID"].fillna(0) > 0]
        return dict(zip(df["PERSON_ID"].astype(int), df["TEAM_ID"].astype(int)))
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't load current rosters, using each player's last team: {e})")
        return {}


def fetch_player_index(season):
    """Player list with positions (used by the backtest's defense-vs-position test)."""
    def go():
        from nba_api.stats.endpoints import playerindex

        return _call(playerindex.PlayerIndex, season=season)

    return _cached(f"playerindex_{season}", 24, go)


# =====================================================================
# 2. CALCULATIONS (pure pandas, no internet)
# =====================================================================
def prepare_logs(logs, before=None):
    """Clean the game logs. before=<date> drops that day and later, so we only use games already played."""
    logs = logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    for c in ["MIN"] + STATS:
        logs[c] = pd.to_numeric(logs[c], errors="coerce")
    logs = logs.dropna(subset=["MIN"] + STATS)
    if before is not None:
        logs = logs[logs["GAME_DATE"] < before]
    logs = logs[logs["MIN"] > 0].sort_values(["PLAYER_ID", "GAME_DATE"])
    logs["games_ago"] = logs.groupby("PLAYER_ID").cumcount(ascending=False)  # 0 = most recent game
    return logs


def player_table(logs):
    """Per player: recency-weighted per-minute rates (shrunk to league avg) and default minutes."""
    league = {s: logs[s].sum() / logs["MIN"].sum() for s in STATS}
    w = 0.5 ** (logs["games_ago"] / RATE_HALF_LIFE_GAMES)
    work = pd.DataFrame({"PLAYER_ID": logs["PLAYER_ID"], "w_min": w * logs["MIN"]})
    for s in STATS:
        work[f"w_{s}"] = w * logs[s]
    agg = work.groupby("PLAYER_ID").sum()
    info = logs.groupby("PLAYER_ID").agg(name=("PLAYER_NAME", "last"),
                                         last_team_id=("TEAM_ID", "last"),
                                         games=("MIN", "size"))
    out = info.join(agg)
    for s in STATS:
        out[f"rate_{s}"] = (out[f"w_{s}"] + PRIOR_MINUTES * league[s]) / (out["w_min"] + PRIOR_MINUTES)

    recent = logs[logs["games_ago"] < MINUTES_WINDOW_GAMES].copy()
    recent["w2"] = 0.5 ** (recent["games_ago"] / MINUTES_HALF_LIFE_GAMES)
    recent["wm"] = recent["w2"] * recent["MIN"]
    g = recent.groupby("PLAYER_ID")
    out["default_min"] = g["wm"].sum() / g["w2"].sum()
    try:
        out = out.join(_volume_efficiency_rates(logs))
    except Exception:  # noqa: BLE001  (columns missing in older data: just skip this extra)
        pass
    return out


def _volume_efficiency_rates(logs):
    """
    Alternative scoring rates: shot VOLUME per minute (stable) x shooting EFFICIENCY (noisy, so pulled hard
    toward the league average). rate_PTS_ve = true-shooting attempts per minute x points per attempt;
    rate_FG3M_ve = 3-point attempts per minute x 3-point %.
    """
    x = logs.copy()
    for c in ("FGA", "FTA", "FG3A"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["FGA", "FTA", "FG3A"])
    x["TSA"] = x["FGA"] + 0.44 * x["FTA"]
    w = 0.5 ** (x["games_ago"] / RATE_HALF_LIFE_GAMES)
    work = pd.DataFrame({"PLAYER_ID": x["PLAYER_ID"], "w_min": w * x["MIN"], "w_TSA": w * x["TSA"],
                         "w_3PA": w * x["FG3A"], "w_PTS": w * x["PTS"], "w_3PM": w * x["FG3M"]})
    a = work.groupby("PLAYER_ID").sum()
    lg_tsa, lg_3pa = x["TSA"].sum() / x["MIN"].sum(), x["FG3A"].sum() / x["MIN"].sum()
    lg_pts_per_tsa, lg_3p = x["PTS"].sum() / x["TSA"].sum(), x["FG3M"].sum() / x["FG3A"].sum()
    tsa_rate = (a["w_TSA"] + PRIOR_MINUTES * lg_tsa) / (a["w_min"] + PRIOR_MINUTES)
    eff = (a["w_PTS"] + PRIOR_TSA * lg_pts_per_tsa) / (a["w_TSA"] + PRIOR_TSA)
    three_rate = (a["w_3PA"] + PRIOR_MINUTES * lg_3pa) / (a["w_min"] + PRIOR_MINUTES)
    pct = (a["w_3PM"] + PRIOR_3PA * lg_3p) / (a["w_3PA"] + PRIOR_3PA)
    return pd.DataFrame({"rate_PTS_ve": tsa_rate * eff, "rate_FG3M_ve": three_rate * pct})


def team_table(cur, prev, season_games=TEAM_BLEND_GAMES):
    """Blend this season's and last season's team numbers into pace + opponent factors."""
    cols = ["PACE", "OPP_PTS", "OPP_REB", "OPP_AST", "OPP_FG3M"]

    def norm(df):
        df = df.set_index("TEAM_ID").copy()
        for c in cols[1:]:
            df[c] = df[c] / df[c].mean()          # 1.00 = league average
        return df

    if prev is None and cur is None:
        raise RuntimeError("No team stats available for either season.")
    if cur is None:
        return norm(prev)[cols]
    if prev is None:
        return norm(cur)[cols]
    c, p = norm(cur), norm(prev)
    weight = (c["GP"] / (c["GP"] + season_games)).reindex(p.index).fillna(0)
    out = pd.DataFrame(index=p.index)
    for col in cols:
        out[col] = weight * c[col].reindex(p.index).fillna(p[col]) + (1 - weight) * p[col]
    return out


def build_rows(players, games, today, team_played_yesterday, current_team, id2abbr):
    players = players[(players["games"] >= MIN_GAMES_REQUIRED) &
                      (players["default_min"] >= MIN_DEFAULT_MINUTES)].copy()
    players["team_id"] = [int(current_team.get(pid, tid)) for pid, tid in zip(players.index, players["last_team_id"])]
    rows = []
    for home_id, away_id in games:
        for team_id, opp_id, ha in ((home_id, away_id, "H"), (away_id, home_id, "A")):
            sub = players[players["team_id"] == team_id].sort_values("default_min", ascending=False)
            for pid, p in sub.iterrows():
                rows.append({
                    "pid": int(pid), "name": p["name"], "team": id2abbr.get(team_id, str(team_id)),
                    "opp": id2abbr.get(opp_id, str(opp_id)), "ha": ha,
                    "b2b": "Y" if team_id in team_played_yesterday else "N",
                    "default_min": round(float(p["default_min"]), 1),
                    "rates": [round(float(p[f"rate_{s}"]), 4) for s in STATS],
                })
    return rows


# ---- team ratings and defense-vs-position (built from the same game logs) ----
EXTRA_COLS = ["FGA", "FTA", "OREB", "TOV"]


def enrich_logs(raw):
    """Clean one season's game logs and add: poss (estimated possessions), home flag, opp_id (opposing team id)."""
    x = raw.copy()
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"])
    for c in ["MIN"] + STATS + EXTRA_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["MIN"] + STATS + EXTRA_COLS)
    x = x[x["MIN"] > 0].copy()
    x["poss"] = x["FGA"] + 0.44 * x["FTA"] - x["OREB"] + x["TOV"]
    x["home"] = x["MATCHUP"].str.contains("vs.", regex=False)             # "BOS vs. NYK" = home, "BOS @ NYK" = away
    abbr2id = dict(zip(x["TEAM_ABBREVIATION"], x["TEAM_ID"]))
    x["opp_id"] = x["MATCHUP"].str.split().str[-1].map(abbr2id)
    return x


def team_ratings(logs):
    """Offensive / defensive rating (points scored / allowed per 100 possessions) per team from enriched logs."""
    if logs is None or logs.empty:
        return None
    g = logs.groupby(["GAME_ID", "TEAM_ID"]).agg(poss=("poss", "sum"), pts=("PTS", "sum")).reset_index()
    m = g.merge(g, on="GAME_ID", suffixes=("", "_o"))
    m = m[m["TEAM_ID"] != m["TEAM_ID_o"]]
    out = m.groupby("TEAM_ID").agg(GP=("GAME_ID", "size"), pts=("pts", "sum"), poss=("poss", "sum"),
                                   pts_o=("pts_o", "sum"), poss_o=("poss_o", "sum"))
    out["ORTG"] = 100.0 * out["pts"] / out["poss"]
    out["DRTG"] = 100.0 * out["pts_o"] / out["poss_o"]
    return out.reset_index()[["TEAM_ID", "GP", "ORTG", "DRTG"]]


def blend_ratings(cur, prev, season_games=TEAM_BLEND_GAMES):
    """Blend this season's and last season's ratings (weight on this season = GP / (GP + season_games))."""
    if cur is None and prev is None:
        return pd.DataFrame(columns=["ORTG", "DRTG"])
    if prev is None:
        return cur.set_index("TEAM_ID")[["ORTG", "DRTG"]]
    p = prev.set_index("TEAM_ID")
    if cur is None:
        return p[["ORTG", "DRTG"]]
    c = cur.set_index("TEAM_ID")
    w = (c["GP"] / (c["GP"] + season_games)).reindex(p.index).fillna(0)
    out = pd.DataFrame(index=p.index)
    for col in ("ORTG", "DRTG"):
        out[col] = w * c[col].reindex(p.index).fillna(p[col]) + (1 - w) * p[col]
    return out


def pos_group(p):
    """NBA positions are G, F, C (or mixes like G-F). Returns the first letter, or None."""
    if isinstance(p, str) and p.strip() and p.strip()[0] in "GFC":
        return p.strip()[0]
    return None


def dvp_skew_table(L, half_life_games=40, prior_minutes=800):
    """
    Defense vs position from the logs. For each (opposing team, position G/F/C): does that defense give up more
    or less than its usual to that position, relative to how it treats everyone? ~1.00 = no effect, 1.10 = allows
    10% more. Shrunk toward 1.00 when there is little data. L needs columns pos, opp_id, GAME_DATE, MIN + STATS.
    """
    x = L[L["pos"].notna() & L["opp_id"].notna()].copy()
    if x.empty:
        return None
    games_ago = x.groupby("opp_id")["GAME_DATE"].rank(method="dense", ascending=False) - 1
    w = 0.5 ** (games_ago / half_life_games)
    x["wm"] = w * x["MIN"]
    for s in STATS:
        x[f"w_{s}"] = w * x[s]
    cols = ["wm"] + [f"w_{s}" for s in STATS]
    by_pt = x.groupby(["opp_id", "pos"])[cols].sum()
    by_t = x.groupby("opp_id")[cols].sum()
    by_p = x.groupby("pos")[cols].sum()
    tot = x[cols].sum()
    opp_idx = by_pt.index.get_level_values("opp_id")
    pos_idx = by_pt.index.get_level_values("pos")
    M = by_pt["wm"].to_numpy()
    out = pd.DataFrame(index=by_pt.index)
    for s in STATS:
        lr_all = tot[f"w_{s}"] / tot["wm"]
        team_rel = ((by_t[f"w_{s}"] / by_t["wm"]) / lr_all).reindex(opp_idx).to_numpy()
        lr_pos = (by_p[f"w_{s}"] / by_p["wm"]).reindex(pos_idx).to_numpy()
        pos_rel = (by_pt[f"w_{s}"].to_numpy() / M) / lr_pos
        out[f"skew_{s}"] = ((pos_rel / team_rel) * M + prior_minutes) / (M + prior_minutes)
    return out.reset_index()


def teammate_effects(logs, team_pids, min_without=3, prior_minutes=300.0, prior_games=3.0):
    """
    What happens to a player when a specific teammate does NOT play? Learned from past games.
      logs:      enriched logs (several seasons ok) with PLAYER_ID, PLAYER_NAME, TEAM_ID, GAME_ID, GAME_DATE, MIN + STATS
      team_pids: {team_id: [player ids in today's rotation]}
    Returns {player name: {teammate name: {"min": extra minutes, "PTS": x, "REB": x, "AST": x, "3PM": x}}}
    where the stat numbers are per-minute-rate multipliers (1.05 = +5%). Everything is shrunk toward "no effect",
    and a pair is only kept if the player appeared in at least `min_without` games without that teammate while he
    was on the team (so trades and offseason gaps aren't mistaken for injuries).
    """
    names = dict(zip(logs["PLAYER_ID"], logs["PLAYER_NAME"]))
    out = {}
    cols = ["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN"] + STATS
    for tid, pids in team_pids.items():
        tl = logs.loc[logs["TEAM_ID"] == tid, cols]
        pids = [p for p in pids if p in set(tl["PLAYER_ID"])]
        if len(pids) < 2:
            continue
        order = tl.groupby("GAME_ID")["GAME_DATE"].min().sort_values(kind="stable").index
        mins = tl.pivot_table(index="GAME_ID", columns="PLAYER_ID", values="MIN", aggfunc="sum").reindex(order)
        stats = {s: tl.pivot_table(index="GAME_ID", columns="PLAYER_ID", values=s, aggfunc="sum").reindex(order)
                 for s in STATS}
        played = mins.notna() & (mins > 0)
        n_games = len(order)
        pos = np.arange(n_games)
        tenure = {}
        for b in pids:
            idx = pos[played[b].to_numpy()]
            tenure[b] = (idx.min(), idx.max()) if len(idx) else (0, -1)
        for a in pids:
            pa = played[a].to_numpy()
            ma = mins[a].fillna(0).to_numpy()
            for b in pids:
                if a == b:
                    continue
                lo, hi = tenure[b]
                in_ten = (pos >= lo) & (pos <= hi)
                pb = played[b].to_numpy()
                w_with, w_wo = pa & pb, pa & ~pb & in_ten
                n_wo, n_w = int(w_wo.sum()), int(w_with.sum())
                if n_wo < min_without or n_w < 5:
                    continue
                boost = n_wo / (n_wo + prior_games) * (ma[w_wo].mean() - ma[w_with].mean())
                entry = {"min": round(float(boost), 2)}
                for s in STATS:
                    sa = stats[s][a].fillna(0).to_numpy()
                    rate_with = sa[w_with].sum() / ma[w_with].sum()
                    rate_wo = (sa[w_wo].sum() + prior_minutes * rate_with) / (ma[w_wo].sum() + prior_minutes)
                    entry["3PM" if s == "FG3M" else s] = round(float(rate_wo / rate_with), 4) if rate_with > 0 else 1.0
                out.setdefault(names[a], {})[names[b]] = entry
    return out


def blowout_coefficient(logs):
    """
    How many minutes do the five busiest players on a team lose in lopsided games?
    Fits  minutes_ratio = base - c x max(0, |final margin| - 10)  on past games. Returns c (e.g. 0.015 means a
    25-point game costs starters about 22% of their usual minutes), or None if it can't be estimated.
    """
    g = logs.groupby(["GAME_ID", "TEAM_ID"]).agg(pts=("PTS", "sum")).reset_index()
    m = g.merge(g, on="GAME_ID", suffixes=("", "_o"))
    m = m[m["TEAM_ID"] != m["TEAM_ID_o"]].copy()
    m["margin"] = m["pts"] - m["pts_o"]
    avg = logs.groupby(["TEAM_ID", "PLAYER_ID"])["MIN"].mean().rename("avg_min").reset_index()
    top = avg.sort_values(["TEAM_ID", "avg_min"], ascending=[True, False]).groupby("TEAM_ID").head(5)
    x = logs[["GAME_ID", "TEAM_ID", "PLAYER_ID", "MIN"]].merge(top, on=["TEAM_ID", "PLAYER_ID"])
    x = x.merge(m[["GAME_ID", "TEAM_ID", "margin"]], on=["GAME_ID", "TEAM_ID"])
    if len(x) < 500:
        return None
    x["ratio"] = x["MIN"] / x["avg_min"]
    x["excess"] = (x["margin"].abs() - 10).clip(lower=0)
    base = x.loc[x["excess"] == 0, "ratio"].mean()
    denom = float((x["excess"] ** 2).sum())
    if denom <= 0 or pd.isna(base):
        return None
    c = float((x["excess"] * (base - x["ratio"])).sum() / denom)
    return float(min(max(c, 0.0), 0.05))


# =====================================================================
# 3. WRITING THE SPREADSHEET
# =====================================================================
def row_formulas(r):
    T = "Teams!$A$2:$A$200"
    look = lambda col: f'=IFERROR(INDEX(Teams!${col}$2:${col}$200,MATCH($C{r},{T},0)),1)'
    return {
        8: f'=IF(G{r}="",F{r},G{r})',
        14: f'=IFERROR((INDEX(Teams!$B$2:$B$200,MATCH(B{r},{T},0))+INDEX(Teams!$B$2:$B$200,MATCH(C{r},{T},0))-Settings!$B$3)/INDEX(Teams!$B$2:$B$200,MATCH(B{r},{T},0)),1)',
        15: f'=(1+IF(D{r}="H",Settings!$B$4,Settings!$B$5))*(1+IF(E{r}="Y",Settings!$B$6,0))',
        16: look("C"), 17: look("D"), 18: look("E"), 19: look("F"),
        20: f"=$H{r}*I{r}*$N{r}*$O{r}*(1+$M{r})*P{r}",
        21: f"=$H{r}*J{r}*$N{r}*$O{r}*(1+$M{r})*Q{r}",
        22: f"=$H{r}*K{r}*$N{r}*$O{r}*(1+$M{r})*R{r}",
        23: f"=$H{r}*L{r}*$N{r}*$O{r}*(1+$M{r})*S{r}",
        24: f"=T{r}+U{r}+V{r}", 25: f"=T{r}+U{r}", 26: f"=T{r}+V{r}", 27: f"=U{r}+V{r}",
    }


def read_overrides(path):
    """Carry over YOUR MIN (col G) and ROLE ADJ % (col M) from an earlier output file, by player name."""
    from openpyxl import load_workbook

    ws = load_workbook(path)["Projections"]
    found = {}
    for r in range(FIRST_ROW, ws.max_row + 1):
        name, mins, role = ws.cell(r, 1).value, ws.cell(r, 7).value, ws.cell(r, 13).value
        if name and (mins is not None or role not in (None, 0)):
            found[name] = (mins, role)
    return found


def write_workbook(template, out_path, rows, teams, league_pace, id2abbr, stamp, overrides=None):
    from openpyxl import load_workbook

    overrides = overrides or {}
    wb = load_workbook(template)
    pj, tm, st = wb["Projections"], wb["Teams"], wb["Settings"]

    # remember the template's cell styles (so new rows look like the example row), then clear old rows
    pj_style = {c: copy(pj.cell(FIRST_ROW, c)._style) for c in range(1, N_COLS + 1)}
    tm_style = {c: copy(tm.cell(2, c)._style) for c in range(1, 7)}
    for r in range(FIRST_ROW, pj.max_row + 1):
        for c in range(1, N_COLS + 1):
            pj.cell(r, c).value = None
    for r in range(2, tm.max_row + 1):
        for c in range(1, 7):
            tm.cell(r, c).value = None

    # Teams tab
    for i, (tid, t) in enumerate(teams.iterrows(), start=2):
        vals = [id2abbr.get(int(tid), str(tid)), round(float(t["PACE"]), 1)] + \
               [round(float(t[c]), 3) for c in ("OPP_PTS", "OPP_REB", "OPP_AST", "OPP_FG3M")]
        for c, v in enumerate(vals, start=1):
            cell = tm.cell(i, c, v)
            cell._style = copy(tm_style[c])
    tm["H1"] = (f"Updated {stamp}. Pace from stats.nba.com; opponent factor = what that defense allows per 100 "
                "possessions / league average, blended with last season early in the year.")
    st["B3"] = round(float(league_pace), 1)

    # Projections tab
    for i, row in enumerate(rows):
        r = FIRST_ROW + i
        mins, role = overrides.get(row["name"], (None, 0))
        vals = {1: row["name"], 2: row["team"], 3: row["opp"], 4: row["ha"], 5: row["b2b"],
                6: row["default_min"], 7: mins,
                9: row["rates"][0], 10: row["rates"][1], 11: row["rates"][2], 12: row["rates"][3],
                13: role if role is not None else 0}
        vals.update(row_formulas(r))
        for c in range(1, N_COLS + 1):
            cell = pj.cell(r, c)
            cell._style = copy(pj_style[c])
            cell.value = vals.get(c)
    pj["A2"] = ("Yellow cells are yours to edit. Blue text = data loaded by the daily script. Black = formulas "
                "(don't type over them). Green columns = projections.")
    pj["A3"] = f"Data refreshed: {stamp}"
    if "Read Me" in wb.sheetnames:
        wb["Read Me"]["A3"] = ("1. Projections tab: one row per player in today's games, loaded by the daily script. "
                               "Set a player's YOUR MIN to 0 if they are out.")
    wb.save(out_path)


# =====================================================================
# 4. PUTTING IT TOGETHER
# =====================================================================
def main():
    from nba_api.stats.static import teams as static_teams

    id2abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

    if not os.path.exists(TEMPLATE_FILE):
        try:
            from google.colab import files  # type: ignore
            print(f"Please upload {TEMPLATE_FILE}")
            files.upload()
        except ImportError:
            raise FileNotFoundError(f"Put {TEMPLATE_FILE} in the same folder as this script.")

    print("Checking today's games...")
    today, games = fetch_today()
    if not games:
        print(f"No games found on the NBA scoreboard for {today.date()}. Nothing to project.")
        return
    print(f"  {len(games)} games on {today.date()}")

    cur_season, prev_season = season_strings(today)
    print(f"Pulling player game logs for {prev_season} and {cur_season}...")
    log_parts = [fetch_logs(prev_season)]
    try:
        log_parts.append(fetch_logs(cur_season))
    except Exception as e:  # noqa: BLE001
        print(f"  (no {cur_season} logs yet: {e})")
    logs = prepare_logs(pd.concat([p for p in log_parts if p is not None and not p.empty], ignore_index=True),
                        before=today)
    team_played_yesterday = set(logs.loc[logs["GAME_DATE"] == today - pd.Timedelta(days=1), "TEAM_ID"].astype(int))

    print("Pulling team pace and opponent numbers...")
    teams = team_table(fetch_team_table(cur_season), fetch_team_table(prev_season))
    league_pace = teams["PACE"].mean()

    print("Pulling current rosters...")
    current_team = fetch_current_team_map(cur_season)

    print("Calculating projections inputs...")
    players = player_table(logs)
    rows = build_rows(players, games, today, team_played_yesterday, current_team, id2abbr)
    if not rows:
        print("No players matched today's teams. Check the filters at the top of the script.")
        return

    overrides = read_overrides(PREVIOUS_FILE) if PREVIOUS_FILE and os.path.exists(PREVIOUS_FILE) else {}
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    out_path = f"NBA_Projections_{today.date()}.xlsx"
    write_workbook(TEMPLATE_FILE, out_path, rows, teams, league_pace, id2abbr, stamp, overrides)
    print(f"Done: {len(rows)} players, saved as {out_path}")

    try:
        from google.colab import files  # type: ignore
        files.download(out_path)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
