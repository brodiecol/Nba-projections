"""
NBA Player Projections: Streamlit app
Run with:   streamlit run app.py
Reads data/latest.json (made by refresh_data.py or the app's Refresh button).

Tabs
  Games        game-by-game view: team score projections, minutes chart + sliders that add to 240, pace/offense
               sliders, saved scenarios you can load and compare, and a who-guards-whom matchup tracker
               (team score blends the player projections with team ORTG / DRTG / pace / home-away)
  All players  every player playing today in one table with ranges, over/under calculator, and a player detail
               view (last 15 games, minutes trend, how each adjustment changed the projection)
  Lines        paste sportsbook lines, see the model's chance of over/under and expected value, ranked
  Accuracy     saves your projections, shows how they scored against real results (run grade_projections.py)
"""
import copy
import glob
import io
import json
import math
import os
import types
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

import distributions
import fp_dvp
import lines as lines_mod
from projection_math import (OFFENSE_STATS, STATS, apply_calibration, balance_minutes, bench_projection, blowout_factor,
                             compute, team_score_topdown)

try:
    import grade_projections as GP
except Exception:  # noqa: BLE001
    GP = None

HOSTED = os.getcwd().startswith("/mount/src")            # running on Streamlit Community Cloud
APP_VERSION = "v11"
SNAPSHOT_PATH = os.path.join("data", "latest.json")
STATE_PATH = os.path.join("data", "overrides.json")      # your edits for the day
CAL_PATH = os.path.join("data", "calibration.json")      # made by backtest.py
HISTORY_DIR = os.path.join("data", "history")            # saved projections, graded by grade_projections.py
SCENARIOS_PATH = os.path.join("data", "scenarios.json")   # named sets of edits
LINES_DIR = "data"                                        # lines_<date>.txt: sportsbook lines you pasted
TEAM_MINUTES = 240.0
PLAYER_CAP = 48.0

st.set_page_config(page_title="NBA Projections", layout="wide")


# ---------- loading / saving ----------
def find_snapshot_path():
    """data/latest.json, or latest.json in the main folder, or the newest latest*.json (e.g. 'latest (1).json')."""
    for p in (SNAPSHOT_PATH, "latest.json"):
        if os.path.exists(p):
            return p
    found = glob.glob(os.path.join("data", "latest*.json")) + glob.glob("latest*.json")
    return max(found, key=os.path.getmtime) if found else None


def load_snapshot():
    if st.session_state.get("uploaded_snap"):               # a file you loaded through the sidebar this session
        return st.session_state["uploaded_snap"]
    path = find_snapshot_path()
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def load_state(date):
    """Your saved edits, only if they were saved for this same game date."""
    if HOSTED:                      # online copy: edits live in each visitor's own browser session only
        return {}, {}
    try:
        with open(STATE_PATH) as f:
            saved = json.load(f)
        if saved.get("date") == date:
            return saved.get("overrides", {}), saved.get("game_adj", {})
    except (OSError, ValueError):
        pass
    return {}, {}


def save_state():
    ov = {k: v for k, v in st.session_state["ov"].items() if v.get("min") is not None or v.get("role", 0) != 0}
    gov = {g: a for g, a in st.session_state["gov"].items()
           if a.get("pace", 0) != 0 or any(x != 0 for x in a.get("off", {}).values()) or a.get("total", 0) > 0
           or a.get("spread", 0) != 0 or bool(a.get("matchups"))}
    st.session_state["ov"], st.session_state["gov"] = ov, gov
    if HOSTED:
        return
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"date": st.session_state["date"], "overrides": ov, "game_adj": gov}, f)


# ---------- callbacks (run when you touch a widget) ----------
def on_min(name, key):
    e = st.session_state["ov"].setdefault(name, {"min": None, "role": 0.0})
    e["min"] = float(st.session_state[key])
    save_state()


def on_role(name, key):
    e = st.session_state["ov"].setdefault(name, {"min": None, "role": 0.0})
    e["role"] = float(st.session_state[key])
    save_state()


def on_pace(game, key):
    st.session_state["gov"].setdefault(game, {"pace": 0.0, "off": {}})["pace"] = float(st.session_state[key])
    save_state()


def on_off(game, team, key):
    st.session_state["gov"].setdefault(game, {"pace": 0.0, "off": {}})["off"][team] = float(st.session_state[key])
    save_state()


def on_total(game, key):
    st.session_state["gov"].setdefault(game, {"pace": 0.0, "off": {}})["total"] = float(st.session_state[key])
    save_state()


def on_spread(game, key):
    st.session_state["gov"].setdefault(game, {"pace": 0.0, "off": {}})["spread"] = float(st.session_state[key])
    save_state()


def save_history(calc_df, team_rows, date, version):
    """Save a day's projections so grade_projections.py can score them against the real box scores."""
    if HOSTED:
        return
    os.makedirs(HISTORY_DIR, exist_ok=True)
    cols = ["Player", "Team", "Opp", "Game", "Default min", "Min"] + STATS + ["PRA", "PR", "PA", "RA"]
    calc_df[cols].round(3).to_csv(os.path.join(HISTORY_DIR, f"proj_{date}_{version}.csv"), index=False)
    pd.DataFrame(team_rows).round(2).to_csv(os.path.join(HISTORY_DIR, f"games_{date}_{version}.csv"), index=False)


def on_matchup(game, off_player, def_key, star_key):
    g = st.session_state["gov"].setdefault(game, {"pace": 0.0, "off": {}})
    g.setdefault("matchups", {})[off_player] = {"def": st.session_state[def_key], "star": bool(st.session_state[star_key])}
    save_state()


def reset_game(game, names):
    for n in names:
        st.session_state["ov"].pop(n, None)
    st.session_state["gov"].pop(game, None)
    save_state()


def reset_all():
    st.session_state["ov"], st.session_state["gov"] = {}, {}
    save_state()


def load_scenarios(date):
    if HOSTED:
        return st.session_state.setdefault("scn_store", {})
    try:
        with open(SCENARIOS_PATH) as f:
            saved = json.load(f)
        return saved.get("scenarios", {}) if saved.get("date") == date else {}
    except (OSError, ValueError):
        return {}


def write_scenarios(date, scns):
    if HOSTED:
        st.session_state["scn_store"] = scns
        return
    os.makedirs(os.path.dirname(SCENARIOS_PATH), exist_ok=True)
    with open(SCENARIOS_PATH, "w") as f:
        json.dump({"date": date, "scenarios": scns}, f)


def save_scenario():
    name = st.session_state.get("scn_name", "").strip()
    if not name:
        return
    scns = load_scenarios(st.session_state["date"])
    scns[name] = {"overrides": copy.deepcopy(st.session_state["ov"]), "game_adj": copy.deepcopy(st.session_state["gov"])}
    write_scenarios(st.session_state["date"], scns)
    st.session_state["scn_name"] = ""


def load_scenario(name):
    scn = load_scenarios(st.session_state["date"]).get(name)
    if scn:
        st.session_state["ov"] = copy.deepcopy(scn["overrides"])
        st.session_state["gov"] = copy.deepcopy(scn["game_adj"])
        save_state()


def delete_scenario(name):
    scns = load_scenarios(st.session_state["date"])
    if scns.pop(name, None) is not None:
        write_scenarios(st.session_state["date"], scns)


def save_lines():
    if HOSTED:
        return
    os.makedirs(LINES_DIR, exist_ok=True)
    with open(os.path.join(LINES_DIR, f"lines_{st.session_state['date']}.txt"), "w") as f:
        f.write(st.session_state.get("lines_text", ""))


def minutes_chart(df):
    """Column chart of projected minutes, one column per player, biggest first."""
    try:
        st.bar_chart(df, sort=False)
    except TypeError:                       # older Streamlit without the sort option (columns will be alphabetical)
        st.bar_chart(df)


def clean_players(plist):
    """
    Safety net for odd data: a team can only play one game a day. If the same team shows up in more than one
    game (or a player is listed twice), keep the first game and drop the later ones. Returns (players, dropped_games).
    """
    game_of = lambda p: f"{p['opp']} @ {p['team']}" if p["ha"] == "H" else f"{p['team']} @ {p['opp']}"
    used, keep_games, dropped = set(), set(), []
    for g in dict.fromkeys(game_of(p) for p in plist):
        a, h = g.split(" @ ")
        if a in used or h in used:
            dropped.append(g)
        else:
            used.update((a, h))
            keep_games.add(g)
    out, seen = [], set()
    for p in plist:
        if game_of(p) in keep_games and (p["team"], p["name"]) not in seen:
            seen.add((p["team"], p["name"]))
            out.append(p)
    return out, dropped


# ---------- page ----------
st.title("NBA Player Projections")
st.caption(f"App {APP_VERSION}" + ("" if HOSTED else f" · running from {os.getcwd()}")
           + " · statistical estimates for information only, with no guarantee of results.")
with st.expander("Where to find things"):
    st.markdown(
        "**Games tab**: pick a game. Each team column has the minutes total, the teammate-out adjustments, a minutes "
        "chart (one column per player) and the minutes sliders. **Scenarios** (save, load, compare) is the expander "
        "near the top, above the game picker.\n\n"
        "**All players tab**: the table, the over/under calculator, and **Player detail** at the bottom (pick a player "
        "from the dropdown: last 15 games, minutes trend, projection with range, and how each adjustment changed it).\n\n"
        "**Lines tab**: paste sportsbook lines to rank edges. **Accuracy tab**: saved projections and grading.")
with st.sidebar:
    with st.expander("Load a data file (latest.json)"):
        st.caption("No data showing? Pick the latest.json your computer made (py refresh_data.py). It is used for "
                   "this browser session only; to keep it for everyone, upload it to GitHub's data folder.")
        up_file = st.file_uploader("latest.json", type=["json"], key="snap_upload")
if up_file is not None and st.session_state.get("snap_upload_done") != (up_file.name, up_file.size):
    try:
        loaded = json.loads(up_file.getvalue().decode("utf-8"))
        if "players" in loaded and "teams" in loaded:
            st.session_state["uploaded_snap"] = loaded
            st.session_state["snap_upload_done"] = (up_file.name, up_file.size)
        else:
            st.sidebar.error("That file doesn't look like a latest.json from this app.")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Couldn't read that file: {e}")
snap = load_snapshot()

with st.sidebar:
    st.header("Data")
    if HOSTED:
        st.caption("Online version: the data comes from the file data/latest.json in your GitHub repo, which the "
                   "daily GitHub job (or you) updates. The Refresh button only works when the app runs on your own computer.")
    elif st.button("Refresh data"):
        with st.spinner("Pulling the latest NBA data (can take a minute)..."):
            try:
                import refresh_data
                refresh_data.refresh()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't refresh: {e}")
                st.caption("'Expecting value' or a timeout means the NBA's site sent back an empty reply, which usually "
                           "means it is temporarily limiting requests from this connection. Wait 15-30 minutes and "
                           "try again (the app now saves what it downloads, so retries ask for less). If it keeps "
                           "failing, try another network such as a phone hotspot, with any VPN off.")
    if not HOSTED:
        with st.expander("Games not loading? Enter today's games"):
            st.caption("If the NBA's game list won't load, type today's games here, one per line as AWAY @ HOME "
                       "(for example BOS @ CLE), then press the button. Everything else is still pulled as usual.")
            manual_text = st.text_area("Today's games", key="manual_games", height=140)
            if st.button("Use these games and refresh", key="manual_go"):
                with st.spinner("Pulling data for the games you entered..."):
                    try:
                        import refresh_data
                        refresh_data.write_manual_games(manual_text)
                        refresh_data.refresh()
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Couldn't refresh: {e}")
    if snap:
        st.caption(f"Games on {snap['date']} · data pulled {snap['stamp']}")

if snap:
    et_today = (datetime.now(timezone.utc) - timedelta(hours=5)).date().isoformat()   # roughly US Eastern
    if snap["date"] < et_today:
        st.warning(f"These projections are for games on {snap['date']}, not today's. "
                   "Click Refresh data in the sidebar (or run refresh_data.py) to load today's games.")

if not snap:
    if HOSTED:
        st.info("No data file found. Add `data/latest.json` to your GitHub repo: run `py refresh_data.py` on your "
                "computer and upload the file it makes inside a folder named `data`, or run the **Refresh NBA data** "
                "job from the repo's Actions tab. For a quick look, you can also load the file through "
                "**Load a data file** in the sidebar.")
        try:
            st.caption("What the app can see right now: main folder = " + ", ".join(sorted(os.listdir("."))[:40])
                       + (" | data folder = " + ", ".join(sorted(os.listdir("data"))) if os.path.isdir("data")
                          else " | there is no data folder"))
        except OSError:
            pass
    else:
        st.info("No data yet. Click **Refresh data** in the sidebar, or run `python refresh_data.py`.")
    st.stop()

# new data -> load whatever edits were saved for this date
if st.session_state.get("snap_stamp") != snap["stamp"]:
    st.session_state["snap_stamp"] = snap["stamp"]
    st.session_state["date"] = snap["date"]
    st.session_state["ov"], st.session_state["gov"] = load_state(snap["date"])
    st.session_state.pop("sel_game", None)
ov, gov = st.session_state["ov"], st.session_state["gov"]
players, dropped_games = clean_players(snap["players"])

with st.sidebar:
    st.header("Minutes")
    balance_on = st.checkbox("Keep each team at 240 minutes", value=True, key="balance_on",
                             help="When you move a player's minutes, everyone you haven't touched on that team "
                                  "is scaled up or down automatically so the team still adds to 240.")
    st.header("Adjustments (starting guesses)")
    home = st.number_input("Home (%)", value=1.0, step=0.5) / 100
    away = st.number_input("Away (%)", value=-1.0, step=0.5) / 100
    b2b = st.number_input("Back-to-back (%)", value=-2.0, step=0.5) / 100
    st.header("Team score model")
    w_players = st.slider("Weight on player projections (%)", 0, 100, 50, 5,
                          help="Team score = this % x (sum of player projections) + the rest x (ratings-based score "
                               "from pace, ORTG, DRTG, home/away). 100 = players only, 0 = ratings only.") / 100
    st.header("Defense vs position")
    dvp_mode = st.selectbox("Source", ["NBA data (automatic)", "FantasyPros file", "Off"], key="dvp_mode",
                            help="Adjusts each player's stats for how the opponent defends his position.")
    dvp_strength = st.slider("Strength (%)", 0, 100, 50, 5,
                             help="How much of the defense-vs-position effect to apply. 0 = none, 100 = all of it.") / 100
    fp_skews, pos_over = {}, {}
    if dvp_mode == "FantasyPros file":
        fp_path, fp_year, fp_note = fp_dvp.find_file(snap["date"])
        if fp_path:
            try:
                fp_skews, fp_notes = fp_dvp.load_table(fp_path)
                pos_over = fp_dvp.load_position_overrides()
                st.caption(f"Using {os.path.basename(fp_path)} (season {fp_year}). {fp_note} {' '.join(fp_notes)}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't read {os.path.basename(fp_path)}: {e}")
        else:
            st.info(fp_note)
        if not fp_skews:
            st.caption("No FantasyPros adjustment is being applied.")
    st.header("Projection model")
    rates_method = st.selectbox("Scoring rates", ["Per-minute rates", "Volume × efficiency"], key="rates_method",
                                help="Volume × efficiency projects PTS and 3PM as shot attempts × a shrunk shooting "
                                     "percentage, which regresses hot or cold shooting harder. Compare them in the backtest.")
    use_tm = st.checkbox("Teammate-out effects", value=True, key="use_tm",
                         help="Set a teammate's minutes to 0 and the players who historically gain when he sits get "
                              "extra minutes and a usage bump (learned from past games, heavily shrunk).")
    use_blowout = st.checkbox("Blowout risk (starters' minutes)", value=True, key="use_blowout",
                              help="Lopsided expected margins trim starters' minutes (learned from past games).")
    try:
        with open(CAL_PATH) as f_:
            cal = json.load(f_)
    except (OSError, ValueError):
        cal = {}
    use_cal = st.checkbox("Apply backtest calibration", value=bool(cal), key="use_cal", disabled=not cal,
                          help="Small per-stat corrections learned in backtest.py (data/calibration.json).")
    if not cal:
        st.caption("Calibration needs a backtest run (py backtest.py).")
    w_vegas = st.slider("Weight on Vegas lines (%)", 0, 100, 50, 5,
                        help="Only used for games where you type a Vegas total and spread in the Games tab.") / 100
    st.header("All players tab")
    all_games = list(dict.fromkeys(
        (f"{p['opp']} @ {p['team']}" if p["ha"] == "H" else f"{p['team']} @ {p['opp']}") for p in players))
    pick_games = st.multiselect("Game", all_games, default=all_games)
    search = st.text_input("Search player")
    st.button("Reset ALL my edits", on_click=reset_all)

game_of = lambda p: f"{p['opp']} @ {p['team']}" if p["ha"] == "H" else f"{p['team']} @ {p['opp']}"
bench_key = lambda team: f"__others__{team}"


# ---------- defense vs position multipliers ----------
def dvp_factors(p):
    """1.00 = no effect. Moves that fraction (Strength slider) of the way to the full defense-vs-position effect."""
    out = {}
    for s_ in STATS:
        if dvp_mode == "NBA data (automatic)":
            sk = snap.get("dvp", {}).get(p["opp"], {}).get(p.get("pos") or "", {}).get(s_, 1.0)
        elif dvp_mode == "FantasyPros file" and fp_skews:
            positions = fp_dvp.positions_for(p.get("pos_full", ""), pos_over.get(p["name"]))
            sk = fp_dvp.player_skew(fp_skews, p["opp"], positions, s_)
        else:
            sk = 1.0
        out[f"dvp_{s_}"] = 1.0 + dvp_strength * (sk - 1.0)
    return out


def rate_of(p, s_):
    if rates_method == "Volume × efficiency" and f"rate_{s_}_ve" in p:
        return p[f"rate_{s_}_ve"]
    return p[f"rate_{s_}"]


def run_pipeline(ov, gov):
    """
    Everything from "who plays how many minutes" to "what each team scores", for ONE set of edits (ov = minutes/role
    edits, gov = per-game edits). The live page uses your current edits; the Scenarios tab runs it on saved ones.
    """
    # ---------- game context: expected margin (ratings and/or Vegas), blowout risk ----------
    by_team = {}
    for p in players:
        by_team.setdefault(p["team"], []).append(p)
    team_first = {t: pl[0] for t, pl in by_team.items()}
    blowout_c = snap.get("blowout_c")
    tm_fx = snap.get("teammate_effects", {}) if use_tm else {}


    def ratings_only(team):
        """Team score from pace + ORTG/DRTG + home/away alone (no players); None if ratings aren't in the data file."""
        p = team_first[team]
        g = gov.get(game_of(p), {})
        return team_score_topdown(team, p["opp"], p["ha"] == "H", p["b2b"] == "Y", snap["teams"], snap["league_pace"],
                                  snap.get("league_ortg"), home, away, b2b, g.get("pace", 0.0), g.get("off", {}).get(team, 0.0))


    def expected_margin(game):
        """Expected home margin from the ratings model, blended with the Vegas spread when you've entered one."""
        a, h = game.split(" @ ")
        g = gov.get(game, {})
        ra = ratings_only(a) if a in by_team else None
        rh = ratings_only(h) if h in by_team else None
        model = None if ra is None or rh is None else rh - ra
        vegas = -g["spread"] if g.get("spread", 0) != 0 else None          # a spread alone is enough
        if vegas is None:
            return 0.0 if model is None else model
        return vegas if model is None else w_vegas * vegas + (1 - w_vegas) * model


    # ---------- work out minutes for every team (blowout + teammate-out, then auto-balanced to 240) ----------
    defaults_by_team, mins_used, status, tm_factor, notes_by_game, tm_notes = {}, {}, {}, {}, {}, {}
    for team, plist in by_team.items():
        game = game_of(team_first[team])
        d0 = {p["name"]: float(p["default_min"]) for p in plist}
        notes = notes_by_game.setdefault(game, [])

        margin = expected_margin(game)
        fb = blowout_factor(abs(margin), blowout_c) if (use_blowout and blowout_c) else 1.0
        if fb < 0.999:
            top = sorted(d0, key=d0.get, reverse=True)[:5]
            s_top = sum(d0[n] for n in top)
            rest = sum(d0.values()) - s_top
            scale_rest = 1 + s_top * (1 - fb) / rest if rest > 0 else 1.0
            d0 = {n: v * (fb if n in top else scale_rest) for n, v in d0.items()}
            if team == game.split(" @ ")[1]:
                notes.append(f"Blowout risk: expected margin {abs(margin):.1f} points, so each team's top-5 minutes players "
                             f"keep about {fb * 100:.0f}% of their usual minutes.")

        out_names = {n for n in d0 if ov.get(n, {}).get("min") == 0}
        tm_factor[team] = {}
        if tm_fx and out_names:
            for n in d0:
                effs = tm_fx.get(n, {})
                hit = {b: effs[b] for b in out_names if b in effs and b != n and n not in out_names}
                if hit:
                    d0[n] = max(0.0, d0[n] + sum(e["min"] for e in hit.values()))
                    tm_factor[team][n] = {s_: min(1.25, max(0.85, math.prod(e[s_] for e in hit.values()))) for s_ in STATS}
                    tm_notes.setdefault(team, []).append(
                        f"{n}: {sum(e['min'] for e in hit.values()):+.1f} min and stat bump with {', '.join(hit)} out.")

        defaults = dict(d0)
        defaults[bench_key(team)] = max(0.0, TEAM_MINUTES - sum(d0.values()))      # everyone not listed
        fixed = {k: ov[k]["min"] for k in defaults if k in ov and ov[k].get("min") is not None}
        caps = {k: PLAYER_CAP for k in defaults}
        caps[bench_key(team)] = TEAM_MINUTES
        if balance_on:
            mins, stat = balance_minutes(defaults, fixed, caps, TEAM_MINUTES)
        else:
            mins, stat = {k: fixed.get(k, v) for k, v in defaults.items()}, "off"
        defaults_by_team[team], mins_used[team], status[team] = defaults, mins, stat

    # ---------- projections for every player ----------
    def build(extra_pace, extra_off):
        """Player projections and team scores. extra_* are the game-environment nudges from Vegas lines (in %)."""
        rows = []
        for p in players:
            game = game_of(p)
            g = gov.get(game, {})
            rows.append({"Player": p["name"], "Game": game, "Team": p["team"], "Opp": p["opp"], "H/A": p["ha"],
                         "B2B": p["b2b"], "Default min": p["default_min"],
                         "Your min": mins_used[p["team"]][p["name"]],
                         "Role adj %": ov.get(p["name"], {}).get("role", 0.0),
                         "Pace adj %": g.get("pace", 0.0) + extra_pace.get(game, 0.0),
                         "Offense adj %": g.get("off", {}).get(p["team"], 0.0) + extra_off.get(p["team"], 0.0),
                         **dvp_factors(p),
                         **{f"tm_{s_}": tm_factor[p["team"]].get(p["name"], {}).get(s_, 1.0) for s_ in STATS}})
        inputs = pd.DataFrame(rows)
        rates = pd.DataFrame([{f"rate_{s_}": rate_of(p, s_) for s_ in STATS} for p in players])
        pre_ = compute(inputs.join(rates), snap["teams"], snap["league_pace"], home, away, b2b)
        calc_ = apply_calibration(pre_, cal) if (use_cal and cal) else pre_.copy()
        for s_ in STATS:                                     # how much calibration moved each stat (1.0 = not at all)
            calc_[f"cal_{s_}"] = (calc_[s_] / pre_[s_]).where(pre_[s_] > 0, 1.0)
        lr = {s_: float(rates[f"rate_{s_}"].mean()) for s_ in STATS}      # used for the unlisted-players bucket
        bench, t_players, t_ratings, t_blend = {}, {}, {}, {}
        for team in by_team:
            trows = calc_[calc_["Team"] == team]
            first = trows.iloc[0]
            bench[team] = bench_projection(first, mins_used[team][bench_key(team)], lr)
            t_players[team] = float(trows["PTS"].sum() + bench[team]["PTS"])
            gv = gov.get(game_of(team_first[team]), {})          # ratings model uses only YOUR sliders, not Vegas
            t_ratings[team] = team_score_topdown(
                team, first["Opp"], first["H/A"] == "H", first["B2B"] == "Y", snap["teams"], snap["league_pace"],
                snap.get("league_ortg"), home, away, b2b, gv.get("pace", 0.0), gv.get("off", {}).get(team, 0.0))
            r_ = t_ratings[team]
            t_blend[team] = t_players[team] if r_ is None else w_players * t_players[team] + (1 - w_players) * r_
        return types.SimpleNamespace(pre=pre_, calc=calc_, league_rates=lr, bench=bench, players=t_players,
                                     ratings=t_ratings, blend=t_blend)

    B = build({}, {})
    all_game_names = list(dict.fromkeys(B.calc["Game"]))
    games = [g for g in all_game_names if all(t in by_team for t in g.split(" @ "))]       # skip games vs non-NBA teams
    game_teams = {g: tuple(g.split(" @ ")) for g in games}          # (away, home)

    # Vegas lines you typed in (a spread, a total, or both): blend the line-implied scores with the model's, then
    # nudge the game's pace and each team's offense so the PLAYER projections move with it
    team_pts_model, team_pts, team_vegas = dict(B.blend), dict(B.blend), {}
    extra_pace, extra_off = {}, {}
    for g_ in games:
        gg = gov.get(g_, {})
        has_total, has_spread = gg.get("total", 0) > 0, gg.get("spread", 0) != 0
        if not (has_total or has_spread):
            continue
        a_, h_ = game_teams[g_]
        model_total, model_margin = B.blend[h_] + B.blend[a_], B.blend[h_] - B.blend[a_]
        total_v = gg["total"] if has_total else model_total
        margin_v = -gg["spread"] if has_spread else model_margin          # home margin; spread -4.5 means home by 4.5
        team_vegas[h_], team_vegas[a_] = (total_v + margin_v) / 2, (total_v - margin_v) / 2
        for t_ in (a_, h_):
            team_pts[t_] = (1 - w_vegas) * B.blend[t_] + w_vegas * team_vegas[t_]
        pace_mult = min(1.15, max(0.85, (team_pts[h_] + team_pts[a_]) / model_total))
        extra_pace[g_] = (pace_mult - 1) * 100
        for t_ in (a_, h_):
            extra_off[t_] = (min(1.10, max(0.90, team_pts[t_] / (B.blend[t_] * pace_mult))) - 1) * 100
        notes_by_game.setdefault(g_, []).append(
            f"Vegas lines applied ({w_vegas * 100:.0f}% weight): game pace {extra_pace[g_]:+.1f}%, {a_} offense "
            f"{extra_off[a_]:+.1f}%, {h_} offense {extra_off[h_]:+.1f}%. These flow into every player's projection.")
    if extra_pace:
        B = build(extra_pace, extra_off)
    pre, calc, league_rates, bench_proj = B.pre, B.calc, B.league_rates, B.bench
    team_pts_players, team_pts_ratings = B.players, B.ratings

    return types.SimpleNamespace(
        by_team=by_team, team_first=team_first, defaults_by_team=defaults_by_team, mins_used=mins_used, status=status,
        tm_factor=tm_factor, notes_by_game=notes_by_game, tm_notes=tm_notes, calc=calc, pre=pre, league_rates=league_rates,
        team_pts=team_pts, team_pts_players=team_pts_players, team_pts_ratings=team_pts_ratings, bench_proj=bench_proj,
        team_pts_model=team_pts_model, team_vegas=team_vegas, all_game_names=all_game_names, games=games,
        game_teams=game_teams)


R = run_pipeline(ov, gov)
by_team, team_first, defaults_by_team, mins_used, status = R.by_team, R.team_first, R.defaults_by_team, R.mins_used, R.status
tm_factor, notes_by_game, calc, pre, league_rates = R.tm_factor, R.notes_by_game, R.calc, R.pre, R.league_rates
tm_notes_by_team = R.tm_notes
team_pts, team_pts_players, team_pts_ratings, bench_proj = R.team_pts, R.team_pts_players, R.team_pts_ratings, R.bench_proj
team_pts_model, team_vegas, all_game_names, games, game_teams = R.team_pts_model, R.team_vegas, R.all_game_names, R.games, R.game_teams
if not games:
    st.warning("No complete NBA games in this data file (preseason games against international teams are skipped). "
               "Try Refresh data again on a regular game day.")
    st.stop()
if dropped_games:
    st.warning("Some games in the data file were ignored because a team appeared in more than one game "
               f"(a team can only play once a day): {', '.join(dropped_games)}. Run Refresh data again.")
if len(games) < len(all_game_names):
    st.caption(f"Skipped {len(all_game_names) - len(games)} game(s) that include a team with no player data "
               "(e.g. preseason games against international teams).")


STATUS_NOTE = {
    "off": "Auto-balance is off (sidebar), so the total can drift from 240.",
    "all_fixed": "Every player is locked at the minutes you set, so nothing can auto-adjust. Lower or raise one player.",
    "over": "The minutes you set already exceed 240; the rest are at 0.",
    "capped": "Couldn't reach 240 because players hit the 48-minute cap.",
}

# ---------- saved projections (graded later by grade_projections.py) ----------
team_rows = []
for g_ in games:
    for t_ in game_teams[g_]:
        r_ = team_pts_ratings.get(t_)
        team_rows.append({"Team": t_, "Game": g_, "Projected": team_pts[t_], "Players": team_pts_players[t_],
                          "Ratings": float("nan") if r_ is None else r_})
no_edits = not ov and not any(a.get("pace", 0) or a.get("total", 0) or a.get("spread", 0) or any(a.get("off", {}).values())
                              for a in gov.values())
if no_edits and st.session_state.get("model_saved") != snap["stamp"]:
    try:
        save_history(calc, team_rows, snap["date"], "model")
        st.session_state["model_saved"] = snap["stamp"]
    except OSError:
        pass

tab_games, tab_all, tab_lines, tab_acc = st.tabs(["Games", "All players", "Lines", "Accuracy"])

# =====================================================================
# GAMES TAB
# =====================================================================
with tab_games:
    st.caption("Pick a game, then drag **Minutes** sliders. Players you move stay where you put them and everyone "
               "else on the team adjusts so the team keeps adding to 240. **Role %** nudges one player on all four "
               "stats. To mark a player OUT, set his minutes to 0: teammates who historically gain when he sits get "
               "extra minutes and usage. The team score blends the player projections with the ratings model "
               "(and Vegas lines if you enter them).")
    with st.expander("All games at a glance (projected scores)"):
        summary = pd.DataFrame([{"Game": g, "Away score": team_pts[game_teams[g][0]],
                                 "Home score": team_pts[game_teams[g][1]],
                                 "Total": team_pts[game_teams[g][0]] + team_pts[game_teams[g][1]],
                                 "Home margin": team_pts[game_teams[g][1]] - team_pts[game_teams[g][0]]} for g in games])
        st.dataframe(summary, hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f")
                                    for c in ["Away score", "Home score", "Total", "Home margin"]})

    with st.expander("Scenarios: save, load and compare your edits", expanded=False):
        saved_scns = load_scenarios(snap["date"])
        st.caption("A scenario is a saved set of your edits (minutes, roles, pace/offense sliders, Vegas lines) for today's slate. "
                   "Save one, change things, and compare. Scenarios are kept for the day's data only.")
        n1, n2 = st.columns([3, 1])
        n1.text_input("Name for the current edits", key="scn_name", placeholder="e.g. Tatum out")
        n2.button("Save current edits", on_click=save_scenario, key="scn_save")
        if saved_scns:
            if st.session_state.get("scn_pick") not in saved_scns:
                st.session_state["scn_pick"] = next(iter(saved_scns))
            p1, p2, p3 = st.columns([3, 1, 1])
            pick = p1.selectbox("Saved scenarios", list(saved_scns), key="scn_pick")
            p2.button("Load", on_click=load_scenario, args=(pick,), key="scn_load")
            p3.button("Delete", on_click=delete_scenario, args=(pick,), key="scn_delete")
        else:
            st.info("No saved scenarios yet.")

        st.markdown("**Compare two scenarios**")
        choices = ["Current edits", "No edits (model only)"] + list(saved_scns)
        for k_, default_ in (("scn_a", "Current edits"), ("scn_b", choices[1])):         # e.g. a scenario was just deleted
            if st.session_state.get(k_) not in choices:
                st.session_state[k_] = default_
        c1_, c2_ = st.columns(2)
        sel_a = c1_.selectbox("A", choices, key="scn_a")
        sel_b = c2_.selectbox("B", choices, key="scn_b")

        def scenario_result(choice):
            if choice == "Current edits":
                return R
            if choice == "No edits (model only)":
                return run_pipeline({}, {})
            scn = saved_scns[choice]
            return run_pipeline(scn["overrides"], scn["game_adj"])

        RA, RB = scenario_result(sel_a), scenario_result(sel_b)
        team_cmp = pd.DataFrame([{"Game": g_, "Team": t_, "A": RA.team_pts[t_], "B": RB.team_pts[t_],
                                  "B - A": RB.team_pts[t_] - RA.team_pts[t_]}
                                 for g_ in RA.games for t_ in RA.game_teams[g_]])
        st.dataframe(team_cmp, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.1f")
                                                               for c in ["A", "B", "B - A"]})
        keep = ["Player", "Game", "Min", "PTS", "REB", "AST", "3PM", "PRA"]
        pa_, pb_ = RA.calc[keep].set_index("Player"), RB.calc[keep].set_index("Player")
        cmp_ = pd.DataFrame({"Game": pa_["Game"]})
        for c_ in ["Min", "PTS", "REB", "AST", "3PM", "PRA"]:
            cmp_[f"{c_} A"], cmp_[f"{c_} B"], cmp_[f"{c_} Δ"] = pa_[c_], pb_[c_], pb_[c_] - pa_[c_]
        cmp_ = cmp_.reset_index()
        show_all = st.checkbox("Show players with no change too", value=False, key="scn_show_all")
        if not show_all:
            cmp_ = cmp_[(cmp_[[f"{c_} Δ" for c_ in ["Min", "PTS", "REB", "AST", "3PM"]]].abs() > 0.05).any(axis=1)]
        cmp_ = cmp_.reindex(cmp_["PRA Δ"].abs().sort_values(ascending=False).index)
        st.dataframe(cmp_, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.1f")
                                                           for c in cmp_.columns if c not in ("Player", "Game")})
        st.download_button("Download comparison as CSV", cmp_.round(2).to_csv(index=False),
                           file_name=f"scenario_compare_{snap['date']}.csv", mime="text/csv")

    game = st.selectbox("Game", games, key="sel_game")
    away_t, home_t = game_teams[game]
    names_in_game = [n for t in (away_t, home_t) for n in defaults_by_team[t]]

    st.subheader(f"{away_t} @ {home_t}")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{away_t} (away)", f"{team_pts[away_t]:.1f}")
    m2.metric(f"{home_t} (home)", f"{team_pts[home_t]:.1f}")
    m3.metric("Total", f"{team_pts[away_t] + team_pts[home_t]:.1f}",
              f"{home_t} {team_pts[home_t] - team_pts[away_t]:+.1f}", delta_color="off")
    for c_, t_ in zip((m1, m2), (away_t, home_t)):
        r_ = team_pts_ratings[t_]
        c_.caption(f"Players {team_pts_players[t_]:.1f} · Ratings " + ("n/a (refresh data)" if r_ is None else f"{r_:.1f}")
                   + (f" · Vegas {team_vegas[t_]:.1f}" if t_ in team_vegas else ""))

    gk0 = gov.get(game, {})
    vk_t, vk_s = f"vtot|{game}", f"vspr|{game}"
    st.session_state[vk_t] = float(gk0.get("total", 0.0))
    st.session_state[vk_s] = float(gk0.get("spread", 0.0))
    v1, v2 = st.columns(2)
    v1.number_input("Vegas total (0 = not entered)", 0.0, 300.0, step=0.5, key=vk_t,
                    on_change=on_total, args=(game, vk_t))
    v2.number_input(f"Vegas spread for {home_t} (-4.5 = {home_t} favored by 4.5; 0 = not entered)", -40.0, 40.0, step=0.5, key=vk_s,
                    on_change=on_spread, args=(game, vk_s))
    st.caption("Enter a spread, a total, or both. Team scores blend your line with the model (weight in the sidebar), "
               "and every player's projection moves with it: a higher total means more possessions for both teams, "
               "and a big spread means the starters sit more.")
    for line in notes_by_game.get(game, []):
        st.caption(line)

    for t_ in (away_t, home_t):
        r_ = team_pts_ratings[t_]
        if not 85 <= team_pts[t_] <= 145:
            st.warning(f"{t_}'s projected score ({team_pts[t_]:.1f}) is outside the normal NBA range. "
                       "Check the minutes and the data (try Refresh data).")
        elif r_ is not None and abs(team_pts_players[t_] - r_) > 0.10 * r_:
            st.warning(f"{t_}: the player-based score ({team_pts_players[t_]:.1f}) and the ratings-based score "
                       f"({r_:.1f}) differ by more than 10%, so the blend is less reliable than usual.")

    gk = gov.get(game, {})
    pace_key = f"pace|{game}"
    st.session_state[pace_key] = float(gk.get("pace", 0.0))
    st.slider("Game pace adjustment (%): more or fewer possessions for both teams", -15.0, 15.0, step=0.5,
              key=pace_key, on_change=on_pace, args=(game, pace_key))

    for col, team in zip(st.columns(2), (away_t, home_t)):
        with col:
            st.markdown(f"### {team} ({'away' if team == away_t else 'home'})")
            off_key = f"off|{game}|{team}"
            st.session_state[off_key] = float(gk.get("off", {}).get(team, 0.0))
            st.slider(f"{team} offense adjustment (%): moves PTS, AST, 3PM", -15.0, 15.0, step=0.5,
                      key=off_key, on_change=on_off, args=(game, team, off_key))

            total = sum(mins_used[team].values())
            if abs(total - TEAM_MINUTES) < 0.05:
                st.success(f"Minutes: {total:.1f} / 240 ✓")
            else:
                st.warning(f"Minutes: {total:.1f} / 240 ({total - TEAM_MINUTES:+.1f}). "
                           f"{STATUS_NOTE.get(status[team], '')}")

            tn = tm_notes_by_team.get(team, [])                    # who gains when a teammate is out, under this team
            if tn:
                st.markdown("**Adjustments with teammates out**\n\n" + "\n".join(f"- {line}" for line in tn))

            ranked = sorted([k for k in mins_used[team] if k != bench_key(team)], key=lambda k: -mins_used[team][k])
            minutes_chart(pd.DataFrame({"Projected minutes": [mins_used[team][k] for k in ranked] + [mins_used[team][bench_key(team)]]},
                                       index=ranked + ["Other players"]))

            h1, h2, h3 = st.columns([2.4, 3, 1.3])
            h1.caption("Player · projection")
            h2.caption("Minutes")
            h3.caption("Role %")

            tcalc = calc[(calc["Game"] == game) & (calc["Team"] == team)].sort_values(["Min", "Default min"], ascending=False)
            for _, r in tcalc.iterrows():
                name = r["Player"]
                c1, c2, c3 = st.columns([2.4, 3, 1.3])
                flag = " ✎" if name in ov else (" ↔" if abs(r["Min"] - r["Default min"]) > 0.5 else "")
                c1.markdown(f"**{name}**{flag}")
                c1.caption(f"PTS {r['PTS']:.1f} · REB {r['REB']:.1f} · AST {r['AST']:.1f} · 3PM {r['3PM']:.1f} "
                           f"· PRA {r['PRA']:.1f}")
                mk, rk = f"min|{game}|{team}|{name}", f"role|{game}|{team}|{name}"
                st.session_state[mk] = float(min(max(mins_used[team][name], 0.0), PLAYER_CAP))
                st.session_state[rk] = float(ov.get(name, {}).get("role", 0.0))
                c2.slider("Minutes", 0.0, PLAYER_CAP, step=0.5, key=mk, label_visibility="collapsed",
                          on_change=on_min, args=(name, mk))
                c3.number_input("Role %", -50.0, 100.0, step=1.0, key=rk, label_visibility="collapsed",
                                on_change=on_role, args=(name, rk))

            # everyone not listed (deep bench), so the team really adds to 240
            bk = bench_key(team)
            b = bench_proj[team]
            c1, c2, c3 = st.columns([2.4, 3, 1.3])
            c1.markdown(f"**Other players (not listed)**{' ✎' if bk in ov else ''}")
            c1.caption(f"PTS {b['PTS']:.1f} · REB {b['REB']:.1f} · AST {b['AST']:.1f} · 3PM {b['3PM']:.1f} "
                       "(league-average rates)")
            mk = f"min|{game}|{team}|{bk}"
            st.session_state[mk] = float(min(max(mins_used[team][bk], 0.0), TEAM_MINUTES))
            c2.slider("Minutes", 0.0, TEAM_MINUTES, step=0.5, key=mk, label_visibility="collapsed",
                      on_change=on_min, args=(bk, mk))

    st.button("Reset this game's edits", on_click=reset_game, args=(game, names_in_game))

    # ---------------------------------------------------------------------
    # MATCHUP TRACKER: who guards whom
    # ---------------------------------------------------------------------
    st.divider()
    st.subheader("Matchup tracker: who guards whom")
    st.caption("Pick which opposing player guards each starter (starters = the five with the most projected minutes). "
               "The defaults pair players by position. ★ marks a key matchup. This is a tracker for your own reads: it "
               "doesn't change the projections. The star rating comes from how the opponent has defended these "
               "players' positions (NBA defense-vs-position data).")
    pos_by_name = {p["name"]: p for p in players}
    pos_label = lambda n: pos_by_name.get(n, {}).get("pos_full") or "?"
    slot = lambda n: {"G": 0, "F": 1, "C": 2}.get(pos_label(n)[:1], 1)

    def starters(team):
        names = [k for k in mins_used[team] if k != bench_key(team) and mins_used[team][k] > 0]
        return sorted(names, key=lambda k: -mins_used[team][k])[:5]

    def default_pairs(off_t, def_t):
        offs = sorted(starters(off_t), key=lambda n: (slot(n), -mins_used[off_t][n]))
        defs = sorted(starters(def_t), key=lambda n: (slot(n), -mins_used[def_t][n]))
        return dict(zip(offs, defs))

    def matchup_stars(off_t, def_t):
        vals = [snap.get("dvp", {}).get(def_t, {}).get(pos_label(n)[:1], {}).get("PTS") for n in starters(off_t)]
        vals = [v for v in vals if v]
        if not vals:
            return "rating n/a (needs defense-vs-position data)"
        m = sum(vals) / len(vals)
        n_ = 1 if m < 0.94 else 2 if m < 0.98 else 3 if m < 1.02 else 4 if m < 1.06 else 5
        return "★" * n_ + "☆" * (5 - n_)

    saved_mu = gov.get(game, {}).get("matchups", {})
    cproj = calc.set_index("Player")
    for col, off_t, def_t in zip(st.columns(2), (away_t, home_t), (home_t, away_t)):
        with col:
            st.markdown(f"**{off_t} OFFENSE**  {matchup_stars(off_t, def_t)}")
            st.caption(f"{def_t} defender")
            def_opts = [k for k in sorted(mins_used[def_t], key=lambda k: -mins_used[def_t][k])
                        if k != bench_key(def_t) and mins_used[def_t][k] > 0]
            defaults_ = default_pairs(off_t, def_t)
            used = []
            for n in starters(off_t):
                saved = saved_mu.get(n, {})
                choice = saved.get("def") if saved.get("def") in def_opts else defaults_.get(n, def_opts[0] if def_opts else None)
                if choice is None:
                    continue
                dk, sk = f"mu_def|{game}|{off_t}|{n}", f"mu_star|{game}|{off_t}|{n}"
                st.session_state[dk] = choice
                st.session_state[sk] = bool(saved.get("star", False))
                used.append(choice)
                r_ = cproj.loc[n]
                m1, m2, m3 = st.columns([3, 4, 1])
                m1.markdown(f"**{n}** {pos_label(n)}")
                m1.caption(f"{r_['PTS']:.1f} pts · {r_['REB']:.1f} reb · {r_['AST']:.1f} ast · {r_['Min']:.0f} min")
                m2.selectbox("Defender", def_opts, key=dk, label_visibility="collapsed",
                             format_func=lambda x: f"{x} ({pos_label(x)}) · {mins_used[def_t][x]:.0f} min",
                             on_change=on_matchup, args=(game, n, dk, sk))
                m3.checkbox("★", key=sk, on_change=on_matchup, args=(game, n, dk, sk))
            dupes = sorted({u for u in used if used.count(u) > 1})
            if dupes:
                st.caption("Heads up: " + ", ".join(dupes) + " is guarding more than one starter.")

    with st.expander("View H2H matchup data (each starter against this opponent)"):
        h_rows = []
        for off_t, def_t in ((away_t, home_t), (home_t, away_t)):
            for n in starters(off_t):
                p_ = pos_by_name[n]
                h_, rec_ = p_.get("h2h"), p_.get("recent") or []
                recent_pts = sum(x[3] for x in rec_) / len(rec_) if rec_ else None
                h_rows.append({"Player": n, "Team": off_t, "Opponent": def_t,
                               "Games vs opp": h_["n"] if h_ else 0,
                               "MIN": h_["MIN"] if h_ else None, "PTS": h_["PTS"] if h_ else None,
                               "REB": h_["REB"] if h_ else None, "AST": h_["AST"] if h_ else None,
                               "3PM": h_["3PM"] if h_ else None, "Last-15 PTS": recent_pts})
        if not any(x["Games vs opp"] for x in h_rows):
            st.info("No head-to-head data in this file yet. Click Refresh data to load it.")
        else:
            st.dataframe(pd.DataFrame(h_rows), hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format="%.1f")
                                        for c in ["MIN", "PTS", "REB", "AST", "3PM", "Last-15 PTS"]})
            st.caption("Averages against the opposing TEAM over the last two seasons (box scores don't record who guarded "
                       "whom, so player-vs-player history isn't available). Small samples swing a lot.")

# =====================================================================
# ALL PLAYERS TAB (read-only table; reflects whatever you did in Games)
# =====================================================================
calc["Edited"] = ["✎" if n in ov else ("↔" if abs(m - d_) > 0.5 else "")
                  for n, m, d_ in zip(calc["Player"], calc["Min"], calc["Default min"])]
calc["DvP (PTS)"] = calc["dvp_PTS"]
view = calc[calc["Game"].isin(pick_games)]
if search:
    view = view[view["Player"].str.contains(search, case=False, na=False)]
cols = ["Player", "Game", "Edited", "Min"] + STATS + ["PRA", "PR", "PA", "RA", "DvP (PTS)"]
disp, corr, spread_fitted = distributions.load_params()
RANGE_STATS = ["PTS", "REB", "AST", "3PM", "PRA"]


def ranges_for(df):
    out = {s_: [] for s_ in RANGE_STATS}
    for _, r_ in df.iterrows():
        means = {k: float(r_[k]) for k in STATS}
        for s_ in RANGE_STATS:
            m_, v_ = distributions.mean_var(s_, means, disp, corr)
            lo, hi = distributions.interval(m_, v_)
            out[s_].append(f"{lo}–{hi}")
    return out


with tab_all:
    show_ranges = st.checkbox("Show 80% ranges", value=False, key="show_ranges",
                              help="The result should land inside the range about 80% of the time.")
    table = view[cols].copy()
    if show_ranges:
        rg = ranges_for(view)
        for s_ in RANGE_STATS:
            table[f"{s_} range"] = rg[s_]
    st.dataframe(table, hide_index=True,
                 column_config={**{c: st.column_config.NumberColumn(format="%.1f") for c in ["Min"] + STATS + ["PRA", "PR", "PA", "RA"]},
                                "DvP (PTS)": st.column_config.NumberColumn(format="%.2f")})
    d1, d2 = st.columns(2)
    d1.download_button("Download as CSV", table.round(2).to_csv(index=False),
                       file_name=f"projections_{snap['date']}.csv", mime="text/csv")
    try:
        buf = io.BytesIO()
        table.round(2).to_excel(buf, index=False)
        d2.download_button("Download as Excel", buf.getvalue(), file_name=f"projections_{snap['date']}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:  # noqa: BLE001  (needs openpyxl; CSV above always works)
        d2.caption("Excel download needs the openpyxl package (pip install openpyxl).")
    st.caption("Click a column header to sort. ✎ = you set this player's minutes or role; ↔ = minutes adjusted "
               "automatically (blowout risk, teammate out, or keeping the team at 240). Make changes in the Games tab.")

    with st.expander("Over/under calculator"):
        if st.session_state.get("ou_player") not in set(calc["Player"]):
            st.session_state.pop("ou_player", None)
        pl = st.selectbox("Player", list(calc["Player"]), key="ou_player")
        stat = st.selectbox("Stat", STATS + ["PRA", "PR", "PA", "RA"], key="ou_stat")
        r_ = calc[calc["Player"] == pl].iloc[0]
        means = {k: float(r_[k]) for k in STATS}
        m_, v_ = distributions.mean_var(stat, means, disp, corr)
        line = st.number_input("Line", min_value=0.0, max_value=200.0, value=float(math.floor(m_) + 0.5), step=0.5,
                               key=f"ou_line|{pl}|{stat}")
        po, pu = distributions.prob_over(m_, v_, line), distributions.prob_under(m_, v_, line)
        lo, hi = distributions.interval(m_, v_)
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Projection", f"{m_:.1f}")
        o2.metric("80% range", f"{lo}–{hi}")
        o3.metric("Over", f"{po * 100:.1f}%", f"fair odds {1 / po:.2f}" if po > 0.001 else None, delta_color="off")
        o4.metric("Under", f"{pu * 100:.1f}%", f"fair odds {1 / pu:.2f}" if pu > 0.001 else None, delta_color="off")
        st.caption("Spread of results: " + ("fitted from your backtest." if spread_fitted else
                   "rough starting values. Run `py backtest.py` to fit it from real games.")
                   + " Fair odds are decimal odds with no margin for the sportsbook.")

    st.divider()
    st.subheader("Player detail")
    if st.session_state.get("pl_player") not in set(calc["Player"]):
        st.session_state.pop("pl_player", None)
    pl = st.selectbox("Choose a player", list(calc["Player"]), key="pl_player")
    r_ = calc[calc["Player"] == pl].iloc[0]
    prow = next(p for p in players if p["name"] == pl)
    st.markdown(f"### {pl} · {r_['Team']} {'vs' if r_['H/A'] == 'H' else '@'} {r_['Opp']}")
    means = {k: float(r_[k]) for k in STATS}
    cols_ = st.columns(6)
    cols_[0].metric("Minutes", f"{r_['Min']:.1f}", f"default {r_['Default min']:.1f}", delta_color="off")
    for c_, s_ in zip(cols_[1:], STATS + ["PRA"]):
        m_, v_ = distributions.mean_var(s_, means, disp, corr)
        lo, hi = distributions.interval(m_, v_)
        c_.metric(s_, f"{m_:.1f}", f"80%: {lo}–{hi}", delta_color="off")
    for line in tm_notes_by_team.get(r_["Team"], []):
        if line.startswith(pl + ":"):
            st.caption(line)

    st.subheader("How the projection is built")
    steps = [("Minutes used", lambda s_: r_["Min"]),
             ("Rate per minute", lambda s_: r_[f"rate_{s_}"]),
             ("× Pace factor", lambda s_: r_["Pace factor"]),
             ("× Home/away and rest", lambda s_: r_["Venue/rest factor"]),
             ("× Role adjustment", lambda s_: 1 + r_["Role adj %"] / 100),
             ("× Game pace adjustment", lambda s_: 1 + r_["Pace adj %"] / 100),
             ("× Opponent defense", lambda s_: r_[f"oppf_{s_}"]),
             ("× Defense vs position", lambda s_: r_[f"dvp_{s_}"]),
             ("× Teammate-out effect", lambda s_: r_[f"tm_{s_}"]),
             ("× Team offense adjustment", lambda s_: (1 + r_["Offense adj %"] / 100) if s_ in OFFENSE_STATS else 1.0),
             ("× Calibration", lambda s_: r_[f"cal_{s_}"]),
             ("= Projection", lambda s_: r_[s_])]
    breakdown = pd.DataFrame({s_: [f"{fn(s_):.3f}" if lab != "= Projection" else f"{fn(s_):.2f}" for lab, fn in steps]
                              for s_ in STATS}, index=[lab for lab, _ in steps])
    st.dataframe(breakdown)
    st.caption("Each row multiplies the one above it. 1.000 means that factor isn't changing anything for this player.")

    st.subheader("Last 15 games")
    rec = prow.get("recent")
    if not rec:
        st.info("No recent-game data in this file. Click Refresh data to load it.")
    else:
        rdf = pd.DataFrame(rec, columns=["Date", "Game", "MIN", "PTS", "REB", "AST", "3PM"])
        rdf["PRA"] = rdf["PTS"] + rdf["REB"] + rdf["AST"]
        avg = pd.DataFrame({"Last 5": rdf.tail(5)[["MIN", "PTS", "REB", "AST", "3PM", "PRA"]].mean(),
                            "Last 10": rdf.tail(10)[["MIN", "PTS", "REB", "AST", "3PM", "PRA"]].mean(),
                            "Last 15": rdf[["MIN", "PTS", "REB", "AST", "3PM", "PRA"]].mean(),
                            "Projection": pd.Series({"MIN": r_["Min"], "PTS": r_["PTS"], "REB": r_["REB"], "AST": r_["AST"],
                                                     "3PM": r_["3PM"], "PRA": r_["PRA"]})}).T.round(1)
        st.dataframe(avg)
        st.dataframe(rdf.iloc[::-1], hide_index=True)
        chart = rdf.set_index("Date")[["MIN"]].assign(**{"Projected minutes": float(r_["Min"])})
        st.line_chart(chart)
        st.line_chart(rdf.set_index("Date")[["PTS", "REB", "AST"]])

# =====================================================================
# LINES TAB (sportsbook lines -> model probability and expected value)
# =====================================================================
with tab_lines:
    st.caption("Paste lines, one per row: `Jayson Tatum, PTS, 27.5, -115, -105` (player, stat, line, then optional over "
               "and under odds, American like -110 or decimal like 1.91). Stats: PTS REB AST 3PM PRA PR PA RA. "
               "With no odds, -110 on both sides is assumed. Pushes on whole-number lines count as refunds.")
    lines_file = os.path.join(LINES_DIR, f"lines_{snap['date']}.txt")
    up = st.file_uploader("Or load lines from a CSV/text file", type=["csv", "txt"], key="lines_upload")
    if up is not None and st.session_state.get("lines_upload_done") != (up.name, up.size):
        st.session_state["lines_text"] = up.getvalue().decode("utf-8", errors="ignore")
        st.session_state["lines_upload_done"] = (up.name, up.size)
        save_lines()
    if "lines_text" not in st.session_state and HOSTED:
        st.session_state["lines_text"] = ""
    if "lines_text" not in st.session_state:
        try:
            with open(lines_file) as f_:
                st.session_state["lines_text"] = f_.read()
        except OSError:
            st.session_state["lines_text"] = ""
    st.text_area("Lines", key="lines_text", height=180, on_change=save_lines)

    parsed, problems = lines_mod.parse_lines(st.session_state["lines_text"], list(calc["Player"]))
    if problems:
        with st.expander(f"{len(problems)} line(s) couldn't be used"):
            for raw_, why_ in problems:
                st.caption(f"{raw_}  ->  {why_}")
    if parsed.empty:
        st.info("No lines yet. Paste some above.")
    else:
        ev = lines_mod.evaluate(parsed, calc, disp, corr)
        ev["Note"] = ["minutes edited" if p_ in ov else ("low minutes" if m_ < 15 else "")
                      for p_, m_ in zip(ev["Player"], ev["Min"])]
        f1, f2 = st.columns(2)
        min_ev = f1.number_input("Only show EV % above", value=0.0, step=1.0, key="min_ev")
        sides = f2.selectbox("Side", ["Both", "Over", "Under"], key="side_pick")
        shown = ev[ev["EV %"] >= min_ev]
        if sides != "Both":
            shown = shown[shown["Best side"] == sides]
        shown = shown.assign(**{"P(over)": 100 * shown["P(over)"], "P(under)": 100 * shown["P(under)"],
                                "Model P": 100 * shown["Model P"], "Breakeven": 100 * shown["Breakeven"]})
        show_cols = ["Player", "Game", "Stat", "Line", "Projection", "80% range", "P(over)", "P(under)", "Best side",
                     "Model P", "Odds (dec)", "Breakeven", "Edge (pts)", "EV %", "Min", "Note"]
        st.dataframe(shown[show_cols], hide_index=True, column_config={
            **{c: st.column_config.NumberColumn(format="%.1f") for c in ["Projection", "P(over)", "P(under)", "Model P",
                                                                          "Breakeven", "Edge (pts)", "EV %", "Min"]},
            "Odds (dec)": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Download as CSV", shown[show_cols].round(3).to_csv(index=False),
                           file_name=f"edges_{snap['date']}.csv", mime="text/csv")
        st.caption("Model P is the model's chance for the best side. Edge = Model P minus the break-even chance implied by the odds. "
                   "EV % is the expected profit per 100 staked. These are estimates: the spread of results is "
                   + ("fitted from your backtest." if spread_fitted else "from rough starting values (run backtest.py)."))

# =====================================================================
# ACCURACY TAB
# =====================================================================
with tab_acc:
  if HOSTED:
    st.info("Accuracy tracking runs on your own computer (it saves projections to files and grades them against real "
            "box scores). On the shared online copy it is switched off.")
  else:
    st.caption("The app saves the untouched model's projections each time you load a new day's data, and your "
               "edited projections whenever you click the button below. After the games finish, run "
               "`py grade_projections.py` in your proj folder and reload this page to see how they scored.")
    if st.button("Save my projections (with my edits)", key="save_mine"):
        try:
            save_history(calc, team_rows, snap["date"], "mine")
            st.success(f"Saved your projections for {snap['date']}.")
        except OSError as e:
            st.error(f"Couldn't save: {e}")
    saved = sorted(f for f in os.listdir(HISTORY_DIR) if f.startswith("proj_")) if os.path.isdir(HISTORY_DIR) else []
    st.caption(f"Saved so far: {len(saved)} file(s)" + (f", latest {saved[-1]}" if saved else ""))
    log_path, glog_path = os.path.join("data", "accuracy_log.csv"), os.path.join("data", "accuracy_games.csv")
    if GP is None:
        st.info("grade_projections.py isn't in this folder, so the accuracy summary can't be shown.")
    elif os.path.exists(log_path):
        log = pd.read_csv(log_path)
        glog = pd.read_csv(glog_path) if os.path.exists(glog_path) else pd.DataFrame()
        ps, ts = GP.summarize(log, glog)
        st.subheader("Players: average miss per player-game")
        st.dataframe(ps.set_index("version").T)
        if not ts.empty:
            st.subheader("Team scores")
            st.dataframe(ts, hide_index=True)
        played = log[~log["dnp"].astype(bool)].copy()
        if not played.empty:
            played["PRA miss"] = (played["proj_PRA"] - played["act_PRA"]).abs()
            st.line_chart(played.groupby(["date", "version"])["PRA miss"].mean().unstack())
        e1, e2 = st.columns(2)
        e1.download_button("Download accuracy log (players)", log.to_csv(index=False), file_name="accuracy_log.csv", mime="text/csv")
        if not glog.empty:
            e2.download_button("Download accuracy log (team scores)", glog.to_csv(index=False), file_name="accuracy_games.csv", mime="text/csv")
    else:
        st.info("Nothing graded yet. After today's games are over, run `py grade_projections.py`.")
