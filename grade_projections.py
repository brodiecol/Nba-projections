"""
Accuracy log: grade your saved projections against what really happened.

The app saves each day's projections into data/history/ (automatically the first time it loads a day, and again
whenever you click "Save my projections" in the Accuracy tab). After the games are finished, run:

    py grade_projections.py

It looks up the real box scores, compares them with every saved day that hasn't been graded yet, and adds the
results to data/accuracy_log.csv (players) and data/accuracy_games.csv (team scores). The app's Accuracy tab
reads those files. Saved "model" projections are the untouched model; "mine" are the ones with your edits.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

import daily_projection_update as d

HISTORY_DIR = os.path.join("data", "history")
LOG_PATH = os.path.join("data", "accuracy_log.csv")
GAMES_LOG_PATH = os.path.join("data", "accuracy_games.csv")
STATS = ["PTS", "REB", "AST", "3PM"]
COMBOS = {"PRA": ["PTS", "REB", "AST"], "PR": ["PTS", "REB"], "PA": ["PTS", "AST"], "RA": ["REB", "AST"]}
ALL_STATS = STATS + list(COMBOS)


def _actuals(day_logs):
    """One row per player who played: Player, Team, act_min and the four stats (+ combos)."""
    a = pd.DataFrame({"Player": day_logs["PLAYER_NAME"], "Team": day_logs["TEAM_ABBREVIATION"],
                      "act_min": pd.to_numeric(day_logs["MIN"], errors="coerce"),
                      "act_PTS": pd.to_numeric(day_logs["PTS"], errors="coerce"),
                      "act_REB": pd.to_numeric(day_logs["REB"], errors="coerce"),
                      "act_AST": pd.to_numeric(day_logs["AST"], errors="coerce"),
                      "act_3PM": pd.to_numeric(day_logs["FG3M"], errors="coerce")})
    for c, parts in COMBOS.items():
        a[f"act_{c}"] = sum(a[f"act_{p}"] for p in parts)
    return a


def grade_day(proj, games, day_logs, date, version):
    """
    proj:  saved player projections (Player, Team, Game, Min, PTS, REB, AST, 3PM, PRA, PR, PA, RA)
    games: saved team score projections (Team, Game, Projected, Players, Ratings)
    day_logs: the real player game logs for that date (NBA LeagueGameLog rows)
    Returns (player_rows, game_rows).
    """
    act = _actuals(day_logs[day_logs["MIN"].astype(float) > 0])
    m = proj.merge(act, on=["Player", "Team"], how="left")
    m["dnp"] = m["act_min"].isna()
    out = pd.DataFrame({"date": str(pd.Timestamp(date).date()), "version": version, "Player": m["Player"],
                        "Team": m["Team"], "Game": m["Game"], "dnp": m["dnp"], "proj_min": m["Min"], "act_min": m["act_min"]})
    for s in ALL_STATS:
        out[f"proj_{s}"] = m[s]
        out[f"act_{s}"] = m[f"act_{s}"]
    team_pts = day_logs.assign(PTS=pd.to_numeric(day_logs["PTS"], errors="coerce")).groupby("TEAM_ABBREVIATION")["PTS"].sum()
    g = games.copy()
    g["actual"] = g["Team"].map(team_pts)
    g.insert(0, "version", version)
    g.insert(0, "date", str(pd.Timestamp(date).date()))
    return out, g


def summarize(log, games_log=None):
    """Accuracy summary tables from the log files. Returns (player_summary, team_summary)."""
    if log is None or log.empty:
        return pd.DataFrame(), pd.DataFrame()
    played = log[~log["dnp"].astype(bool)]
    rows = []
    for v, grp in played.groupby("version"):
        row = {"version": v, "player-games": len(grp), "days": grp["date"].nunique(),
               "DNP rate %": round(100 * log[log["version"] == v]["dnp"].astype(bool).mean(), 1),
               "Minutes MAE": round((grp["proj_min"] - grp["act_min"]).abs().mean(), 2)}
        for s in ALL_STATS:
            e = grp[f"proj_{s}"] - grp[f"act_{s}"]
            row[f"{s} MAE"] = round(e.abs().mean(), 2)
        for s in STATS:
            row[f"{s} bias"] = round((grp[f"proj_{s}"] - grp[f"act_{s}"]).mean(), 2)
        rows.append(row)
    team = pd.DataFrame()
    if games_log is not None and not games_log.empty:
        gl = games_log.dropna(subset=["actual"])
        trows = []
        for v, grp in gl.groupby("version"):
            r = {"version": v, "team-games": len(grp), "Blended MAE": round((grp["Projected"] - grp["actual"]).abs().mean(), 2),
                 "Blended bias": round((grp["Projected"] - grp["actual"]).mean(), 2)}
            for c in ("Players", "Ratings"):
                if c in grp and grp[c].notna().any():
                    r[f"{c} MAE"] = round((grp[c] - grp["actual"]).abs().mean(), 2)
            trows.append(r)
        team = pd.DataFrame(trows)
    return pd.DataFrame(rows), team


def _parse(path):
    m = re.search(r"proj_(\d{4}-\d{2}-\d{2})_(\w+)\.csv$", os.path.basename(path))
    return (m.group(1), m.group(2)) if m else (None, None)


def main():
    files = sorted(glob.glob(os.path.join(HISTORY_DIR, "proj_*_*.csv")))
    if not files:
        print(f"No saved projections found in {HISTORY_DIR}. Open the app on a game day first.")
        return
    log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
    glog = pd.read_csv(GAMES_LOG_PATH) if os.path.exists(GAMES_LOG_PATH) else pd.DataFrame()
    done = set(zip(log["date"], log["version"])) if not log.empty else set()
    today_et = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5)).date()
    cache, new_p, new_g = {}, [], []
    for path in files:
        date, version = _parse(path)
        if date is None or (date, version) in done:
            continue
        if pd.Timestamp(date).date() >= today_et:
            print(f"  {date} ({version}): games not finished yet, skipping.")
            continue
        season = d.season_strings(pd.Timestamp(date))[0]
        if season not in cache:
            print(f"Pulling {season} box scores...")
            cache[season] = d.fetch_logs(season)
        lg = cache[season]
        day = lg[pd.to_datetime(lg["GAME_DATE"]) == pd.Timestamp(date)]
        if day.empty:
            print(f"  {date} ({version}): no box scores found yet.")
            continue
        proj = pd.read_csv(path)
        gpath = path.replace("proj_", "games_")
        games = pd.read_csv(gpath) if os.path.exists(gpath) else pd.DataFrame(columns=["Team", "Game", "Projected"])
        p, g = grade_day(proj, games, day, date, version)
        new_p.append(p)
        new_g.append(g)
        print(f"  graded {date} ({version}): {len(p)} players, {int(p['dnp'].sum())} did not play")
    if new_p:
        log = pd.concat([log] + new_p, ignore_index=True)
        log.to_csv(LOG_PATH, index=False)
        glog = pd.concat([glog] + new_g, ignore_index=True)
        glog.to_csv(GAMES_LOG_PATH, index=False)
    if log.empty:
        print("Nothing graded yet.")
        return
    ps, ts = summarize(log, glog)
    pd.set_option("display.width", 200)
    print("\n=== Player accuracy (average miss per player-game; players who played) ===")
    print(ps.T.to_string(header=False))
    if not ts.empty:
        print("\n=== Team score accuracy ===")
        print(ts.to_string(index=False))
    print(f"\nSaved {LOG_PATH} and {GAMES_LOG_PATH}")


if __name__ == "__main__":
    main()
