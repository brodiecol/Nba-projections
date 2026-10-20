"""
Pulls fresh NBA data and saves it to data/latest.json, which the app reads.
Run it directly:   py refresh_data.py   (or: py refresh_data.py 2026-04-08 to test with a past game day)
The app's "Refresh data" button runs the same thing.
Needs: nba_api, pandas, openpyxl, and daily_projection_update.py in the same folder.
"""
import json
import os
import re

import pandas as pd

import daily_projection_update as d

SNAPSHOT_PATH = os.path.join("data", "latest.json")
OUT_NAMES = ["PTS", "REB", "AST", "3PM"]           # order matches d.STATS (PTS, REB, AST, FG3M)
STAT_TO_OUT = dict(zip(d.STATS, OUT_NAMES))


class NothingToProject(Exception):
    pass


def _team_abbr_map():
    from nba_api.stats.static import teams as static_teams
    return {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}


def _positions(seasons):
    """player_id -> position text (e.g. 'G-F') from the NBA player index (best effort)."""
    pos = {}
    for s in seasons:
        try:
            idx = d.fetch_player_index(s)
            for pid, p in zip(idx["PERSON_ID"], idx["POSITION"]):
                pos.setdefault(int(pid), p if isinstance(p, str) else "")
        except Exception as e:  # noqa: BLE001
            print(f"  (couldn't load positions for {s}: {e})")
    return pos


def _step(label, fn, *args, **kwargs):
    """Run one download step; if it fails, say which one (the NBA site sometimes returns an empty reply)."""
    try:
        return fn(*args, **kwargs)
    except NothingToProject:
        raise
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed while {label}: {type(e).__name__}: {e}") from e


def build_snapshot(date=None):
    id2abbr = _team_abbr_map()

    today, games = _step("loading today's games", d.fetch_today, date)
    nba_games = [(h, a) for h, a in games if h in id2abbr and a in id2abbr]
    if len(nba_games) < len(games):
        print(f"  Skipping {len(games) - len(nba_games)} game(s) that include a non-NBA team "
              "(e.g. preseason games against international teams).")
    games, used = [], set()
    for h, a in nba_games:                       # a team can only play once a day: keep the first game per team
        if h in used or a in used:
            print(f"  Skipping a repeated game involving {id2abbr[h]} / {id2abbr[a]}.")
            continue
        used.update((h, a))
        games.append((h, a))
    if not games:
        raise NothingToProject(f"No NBA-vs-NBA games found for {today.date()}. If you expected games, type them in "
                               "with 'Games not loading?' in the app's sidebar, or test with a past date: "
                               "py refresh_data.py 2026-04-08")

    cur_season, prev_season = d.season_strings(today)
    prev_raw = _step(f"downloading {prev_season} game logs", d.fetch_logs, prev_season)
    cur_raw = None
    try:
        cur_raw = d.fetch_logs(cur_season)
    except Exception as e:  # noqa: BLE001
        print(f"  (no {cur_season} logs yet: {e})")
    parts = [prev_raw] + ([cur_raw] if cur_raw is not None and not cur_raw.empty else [])

    # player rates and minutes (only games before the projection day)
    logs = d.prepare_logs(pd.concat(parts, ignore_index=True), before=today)
    played_yesterday = set(logs.loc[logs["GAME_DATE"] == today - pd.Timedelta(days=1), "TEAM_ID"].astype(int))
    teams = _step("building team pace and defense numbers", d.team_table,
                  d.fetch_team_table(cur_season), d.fetch_team_table(prev_season))
    current_team = d.fetch_current_team_map(cur_season)
    players = d.player_table(logs)
    rows = d.build_rows(players, games, today, played_yesterday, current_team, id2abbr)
    if not rows:
        raise NothingToProject("No players matched today's teams (check the filters at the top of daily_projection_update.py).")

    # team offensive / defensive ratings, blended with last season early in the year
    # (if anything here fails, the refresh still succeeds and the app just uses player projections for team scores)
    ratings = pd.DataFrame(columns=["ORTG", "DRTG"])
    prev_e, cur_e = None, None
    try:
        prev_e = d.enrich_logs(prev_raw)
        prev_e = prev_e[prev_e["GAME_DATE"] < today]
        if cur_raw is not None and not cur_raw.empty:
            cur_e = d.enrich_logs(cur_raw)
            cur_e = cur_e[cur_e["GAME_DATE"] < today]
            cur_e = cur_e if not cur_e.empty else None
        ratings = d.blend_ratings(d.team_ratings(cur_e), d.team_ratings(prev_e if not prev_e.empty else None))
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't build team ratings: {e})")

    # positions + defense vs position (from the NBA logs); also fail-soft
    pos_full = _positions([cur_season, prev_season])
    dvp = {}
    allx = None
    try:
        allx = pd.concat([x for x in (prev_e, cur_e) if x is not None and not x.empty], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't combine game logs: {e})")
    try:
        if pos_full and allx is not None:
            allx = allx.copy()
            allx["pos"] = allx["PLAYER_ID"].map({k: d.pos_group(v) for k, v in pos_full.items()})
            skew = d.dvp_skew_table(allx)
            if skew is not None:
                for r in skew.itertuples(index=False):
                    abbr = id2abbr.get(int(r.opp_id))
                    if abbr:
                        dvp.setdefault(abbr, {})[r.pos] = {STAT_TO_OUT[s]: round(float(getattr(r, f"skew_{s}")), 4)
                                                          for s in d.STATS}
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't build defense-vs-position: {e})")
        dvp = {}

    # what happens to each player when a specific teammate sits, and how much starters lose in blowouts
    tm_effects, blowout_c = {}, None
    try:
        if allx is not None:
            abbr2id = {v: k for k, v in id2abbr.items()}
            team_pids = {}
            for r in rows:
                team_pids.setdefault(abbr2id[r["team"]], []).append(r["pid"])
            tm_effects = d.teammate_effects(allx, team_pids)
            blowout_c = d.blowout_coefficient(allx)
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't build teammate-out / blowout effects: {e})")
        tm_effects, blowout_c = {}, None

    # each player's last 15 games (for the app's Player tab)
    recent = {}
    try:
        pids = {r["pid"] for r in rows}
        last = logs[logs["PLAYER_ID"].isin(pids) & (logs["games_ago"] < 15)].sort_values(["PLAYER_ID", "GAME_DATE"])
        for pid, grp in last.groupby("PLAYER_ID"):
            recent[int(pid)] = [[str(rw.GAME_DATE.date()), str(rw.MATCHUP).split(" ", 1)[1].replace("vs.", "vs"),
                                 round(float(rw.MIN), 1), round(float(rw.PTS), 1), round(float(rw.REB), 1),
                                 round(float(rw.AST), 1), round(float(rw.FG3M), 1)] for rw in grp.itertuples()]
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't build recent-game tables: {e})")

    # how each player has done against today's opponent team over the last two seasons
    h2h = {}
    try:
        if allx is not None:
            grp = allx.groupby(["PLAYER_ID", "opp_id"]).agg(n=("MIN", "size"), MIN=("MIN", "mean"), PTS=("PTS", "mean"),
                                                            REB=("REB", "mean"), AST=("AST", "mean"), FG3M=("FG3M", "mean"))
            abbr2id_ = {v: k for k, v in id2abbr.items()}
            for r in rows:
                key = (r["pid"], abbr2id_.get(r["opp"]))
                if key in grp.index:
                    g_ = grp.loc[key]
                    h2h[r["pid"]] = {"n": int(g_["n"]), "MIN": round(float(g_["MIN"]), 1), "PTS": round(float(g_["PTS"]), 1),
                                     "REB": round(float(g_["REB"]), 1), "AST": round(float(g_["AST"]), 1),
                                     "3PM": round(float(g_["FG3M"]), 1)}
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't build head-to-head tables: {e})")
        h2h = {}

    snap_players = []
    for r in rows:
        p = {"name": r["name"], "team": r["team"], "opp": r["opp"], "ha": r["ha"], "b2b": r["b2b"],
             "default_min": r["default_min"], "pos_full": pos_full.get(r["pid"], ""),
             "pos": d.pos_group(pos_full.get(r["pid"], ""))}
        for name, rate in zip(OUT_NAMES, r["rates"]):
            p[f"rate_{name}"] = rate
        for col, name in (("rate_PTS_ve", "PTS"), ("rate_FG3M_ve", "3PM")):
            if col in players.columns and pd.notna(players.loc[r["pid"], col]):
                p[f"rate_{name}_ve"] = round(float(players.loc[r["pid"], col]), 4)
        if r["pid"] in recent:
            p["recent"] = recent[r["pid"]]
        if r["pid"] in h2h:
            p["h2h"] = h2h[r["pid"]]
        snap_players.append(p)

    snap_teams = {}
    for tid, t in teams.iterrows():
        abbr = id2abbr.get(int(tid), str(tid))
        entry = {"pace": round(float(t["PACE"]), 2), "PTS": round(float(t["OPP_PTS"]), 4),
                 "REB": round(float(t["OPP_REB"]), 4), "AST": round(float(t["OPP_AST"]), 4),
                 "3PM": round(float(t["OPP_FG3M"]), 4)}
        if int(tid) in ratings.index:
            entry["ortg"] = round(float(ratings.loc[int(tid), "ORTG"]), 2)
            entry["drtg"] = round(float(ratings.loc[int(tid), "DRTG"]), 2)
        snap_teams[abbr] = entry

    league_ortg = round(float(ratings["ORTG"].mean()), 2) if len(ratings) else None
    return {"date": str(today.date()), "stamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "league_pace": round(float(teams["PACE"].mean()), 2), "league_ortg": league_ortg,
            "teams": snap_teams, "players": snap_players, "dvp": dvp,
            "teammate_effects": tm_effects, "blowout_c": None if blowout_c is None else round(blowout_c, 5)}


def write_manual_games(text, date=None, path=None):
    """
    Save games you type in (one per line, 'AWAY @ HOME', abbreviations or team names) so the data pull can use them
    when the NBA's game lists won't load. Returns the number of games saved; raises ValueError listing bad lines.
    """
    import fp_dvp

    path = path or d.GAMES_FILE
    day = pd.Timestamp(date) if date else pd.Timestamp(d.today_et())
    rows, bad = [], []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s*@\s*|\s+at\s+", line, flags=re.I)
        a = fp_dvp.team_abbr(parts[0]) if len(parts) == 2 else None
        h = fp_dvp.team_abbr(parts[1]) if len(parts) == 2 else None
        if a and h and a != h:
            rows.append({"Date": str(day.date()), "Away": a, "Home": h})
        else:
            bad.append(line)
    if bad:
        raise ValueError("Couldn't read: " + "; ".join(bad) + " (use the form AWAY @ HOME, e.g. BOS @ CLE)")
    if not rows:
        raise ValueError("No games entered.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def clear_manual_games(path=None):
    path = path or d.GAMES_FILE
    if os.path.exists(path):
        os.remove(path)


def refresh(path=SNAPSHOT_PATH, date=None):
    snap = build_snapshot(date)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snap, f)
    return snap


if __name__ == "__main__":
    import sys

    # optional: py refresh_data.py 2026-04-08   (use a past game day to test while the season is off)
    try:
        s = refresh(date=sys.argv[1] if len(sys.argv) > 1 else None)
        print(f"Saved {len(s['players'])} players for {s['date']} to {SNAPSHOT_PATH}")
    except NothingToProject as e:
        print(e)
