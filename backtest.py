"""
Backtest: how good are the projections, and do the extra adjustments actually help?
==================================================================================
For every game day in a past stretch of the season, this script pretends it is the
morning of that day: it uses ONLY games played before that day, builds projections, and
then compares them with what the players really did.

It scores several versions of the model side by side:

  AVG20        simple baseline: the player's average over his last 20 games (ignores minutes, opponent...)
  RATE_MIN     per-minute rate x default minutes (no adjustments)
  FULL         RATE_MIN + pace + home/away + back-to-back + opponent factors  (what the app does today)
  FULL_DVP     FULL + defense-vs-position
  FULL_VE      FULL but scoring built as shot volume x shrunk efficiency (PTS and 3PM only)

and the same three again with ACTUAL minutes plugged in ("_AM"). That second set shows how good the
model is when YOU supply the right minutes, separating minutes errors from everything else.

HOW TO RUN (in the proj folder):   py backtest.py
Optional dates:                    py backtest.py 2026-01-15 2026-04-08
It needs internet for ~4 data pulls, then takes a few minutes. It writes backtest_summary.csv
and backtest_rows.csv next to the script. Copy the printed results back to Claude.
"""
import sys

import numpy as np
import pandas as pd

import daily_projection_update as d
import distributions
import json
import os

# ---------------------------------------------------------------- settings
SEASON = "2025-26"             # the finished season to test on
START_DATE = "2026-01-15"      # first day to score (earlier days just feed the model history)
END_DATE = "2026-04-08"        # last day to score

HOME_ADJ, AWAY_ADJ, B2B_ADJ = 0.01, -0.01, -0.02    # same starting guesses the app uses
BASELINE_GAMES = 20            # games in the simple-average baseline
DVP_HALF_LIFE_GAMES = 40       # defense-vs-position: older games count less
DVP_PRIOR_MINUTES = 800        # defense-vs-position: higher = more caution (pulls toward "no effect")

RAW = list(d.STATS)            # PTS, REB, AST, FG3M (names used by the NBA data)
NAMES = ["PTS", "REB", "AST", "3PM"]
EXTRA = ["FGA", "FTA", "OREB", "TOV"]
COMBOS = {"PRA": ["PTS", "REB", "AST"], "PR": ["PTS", "REB"], "PA": ["PTS", "AST"], "RA": ["REB", "AST"]}
VARIANTS = ["AVG20", "RATE_MIN", "FULL", "FULL_DVP", "FULL_VE", "RATE_MIN_AM", "FULL_AM", "FULL_DVP_AM", "FULL_VE_AM"]


# ---------------------------------------------------------------- helpers
def clean(raw):
    x = raw.copy()
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"])
    for c in ["MIN"] + RAW + EXTRA:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["MIN"] + RAW + EXTRA)
    x = x[x["MIN"] > 0].copy()
    x["poss"] = x["FGA"] + 0.44 * x["FTA"] - x["OREB"] + x["TOV"]      # estimated possessions
    x["home"] = x["MATCHUP"].str.contains("vs.", regex=False)           # "BOS vs. NYK" = home, "BOS @ NYK" = away
    x["opp_abbr"] = x["MATCHUP"].str.split().str[-1]
    return x


def pos_group(p):
    """NBA positions are G, F, C (or mixes like G-F). We use the first letter."""
    if isinstance(p, str) and p.strip() and p.strip()[0] in "GFC":
        return p.strip()[0]
    return None


def team_stats(logs):
    """Team pace and 'what each defense allows per 100 possessions', built from the player logs."""
    if logs is None or logs.empty:
        return None
    agg = {"MIN": "sum", "poss": "sum"}
    agg.update({s: "sum" for s in RAW})
    g = logs.groupby(["GAME_ID", "TEAM_ID"]).agg(agg).reset_index()
    m = g.merge(g, on="GAME_ID", suffixes=("", "_o"))
    m = m[m["TEAM_ID"] != m["TEAM_ID_o"]].copy()
    m["pace"] = 0.5 * (m["poss"] + m["poss_o"]) * 240.0 / m["MIN"]
    out = m.groupby("TEAM_ID").agg(GP=("GAME_ID", "size"), PACE=("pace", "mean"),
                                   oposs=("poss_o", "sum"), **{f"o_{s}": (f"{s}_o", "sum") for s in RAW})
    for s in RAW:
        out[f"OPP_{s}"] = 100.0 * out[f"o_{s}"] / out["oposs"]
    return out.reset_index()[["TEAM_ID", "GP", "PACE"] + [f"OPP_{s}" for s in RAW]]


def dvp_skew_table(L):
    """
    Defense vs position. For each (opposing team, position) it asks: does this defense give up more or
    less than its usual to that position, compared with how it treats everyone else? Result ~1.00 = no
    effect, 1.10 = allows 10% more. Pulled toward 1.00 when there isn't much data.
    """
    x = L[L["pos"].notna() & L["opp_id"].notna()].copy()
    if x.empty:
        return None
    games_ago = x.groupby("opp_id")["GAME_DATE"].rank(method="dense", ascending=False) - 1
    w = 0.5 ** (games_ago / DVP_HALF_LIFE_GAMES)
    x["wm"] = w * x["MIN"]
    for s in RAW:
        x[f"w_{s}"] = w * x[s]
    cols = ["wm"] + [f"w_{s}" for s in RAW]
    by_pt = x.groupby(["opp_id", "pos"])[cols].sum()
    by_t = x.groupby("opp_id")[cols].sum()
    by_p = x.groupby("pos")[cols].sum()
    tot = x[cols].sum()
    opp_idx = by_pt.index.get_level_values("opp_id")
    pos_idx = by_pt.index.get_level_values("pos")
    M = by_pt["wm"].to_numpy()
    out = pd.DataFrame(index=by_pt.index)
    for s in RAW:
        lr_all = tot[f"w_{s}"] / tot["wm"]
        team_rel = ((by_t[f"w_{s}"] / by_t["wm"]) / lr_all).reindex(opp_idx).to_numpy()
        lr_pos = (by_p[f"w_{s}"] / by_p["wm"]).reindex(pos_idx).to_numpy()
        pos_rel = (by_pt[f"w_{s}"].to_numpy() / M) / lr_pos
        ratio = pos_rel / team_rel
        out[f"skew_{s}"] = (ratio * M + DVP_PRIOR_MINUTES) / (M + DVP_PRIOR_MINUTES)
    return out.reset_index()


# ---------------------------------------------------------------- the backtest
def run_backtest(prev_raw, cur_raw, positions=None, start=START_DATE, end=END_DATE):
    prev, cur = clean(prev_raw), clean(cur_raw)
    prev["tag"], cur["tag"] = "prev", "cur"
    allx = pd.concat([prev, cur], ignore_index=True)
    abbr2id = dict(zip(allx["TEAM_ABBREVIATION"], allx["TEAM_ID"]))
    allx["opp_id"] = allx["opp_abbr"].map(abbr2id)
    pos_map = {int(k): pos_group(v) for k, v in (positions or {}).items()}
    allx["pos"] = allx["PLAYER_ID"].map(pos_map)
    prev, cur = allx[allx["tag"] == "prev"], allx[allx["tag"] == "cur"]
    prev_team = team_stats(prev)

    days = [t for t in sorted(pd.to_datetime(cur["GAME_DATE"].unique()))
            if pd.Timestamp(start) <= t <= pd.Timestamp(end)]
    print(f"Scoring {len(days)} game days...")
    out = []
    for i, D in enumerate(days, start=1):
        if i % 10 == 0:
            print(f"  day {i}/{len(days)}")
        L = d.prepare_logs(allx, before=D)                       # only games before D
        players = d.player_table(L)
        df = cur[cur["GAME_DATE"] == D].merge(players, left_on="PLAYER_ID", right_index=True,
                                              how="inner", suffixes=("", "_p"))
        df = df[(df["games"] >= d.MIN_GAMES_REQUIRED) & (df["default_min"] >= d.MIN_DEFAULT_MINUTES)]
        if df.empty:
            continue
        teams = d.team_table(team_stats(cur[cur["GAME_DATE"] < D]), prev_team)
        lp = teams["PACE"].mean()
        tp, op = df["TEAM_ID"].map(teams["PACE"]), df["opp_id"].map(teams["PACE"])
        pf = ((tp + op - lp) / tp).fillna(1.0)
        yday = set(allx.loc[allx["GAME_DATE"] == D - pd.Timedelta(days=1), "TEAM_ID"])
        b2b = df["TEAM_ID"].isin(yday)
        venue = (1 + np.where(df["home"], HOME_ADJ, AWAY_ADJ)) * (1 + np.where(b2b, B2B_ADJ, 0.0))

        skews = dvp_skew_table(L)
        sk = df[["opp_id", "pos"]].merge(skews, on=["opp_id", "pos"], how="left") if skews is not None else None

        res = pd.DataFrame(index=df.index)
        res["date"], res["player"], res["team"] = D, df["PLAYER_NAME"], df["TEAM_ABBREVIATION"]
        res["home"], res["b2b"], res["pos"] = df["home"], b2b, df["pos"]
        res["default_min"], res["actual_min"] = df["default_min"], df["MIN"]
        avg = L[L["games_ago"] < BASELINE_GAMES].groupby("PLAYER_ID")[RAW].mean()
        for s, n in zip(RAW, NAMES):
            res[f"act__{n}"] = df[s]
            res[f"AVG20__{n}"] = df["PLAYER_ID"].map(avg[s])
            oppf = df["opp_id"].map(teams[f"OPP_{s}"]).fillna(1.0)
            skew = sk[f"skew_{s}"].fillna(1.0).to_numpy() if sk is not None else 1.0
            for tag, mins in (("", df["default_min"]), ("_AM", df["MIN"])):
                base = mins * df[f"rate_{s}"]
                full = base * pf * venue * oppf
                res[f"RATE_MIN{tag}__{n}"] = base
                res[f"FULL{tag}__{n}"] = full
                res[f"FULL_DVP{tag}__{n}"] = full * skew
                ve = df[f"rate_{s}_ve"] if f"rate_{s}_ve" in df else df[f"rate_{s}"]     # volume x efficiency rates
                res[f"FULL_VE{tag}__{n}"] = mins * ve * pf * venue * oppf
        out.append(res)

    res = pd.concat(out, ignore_index=True).dropna(subset=["AVG20__PTS"])
    for v in VARIANTS + ["act"]:
        for c, parts in COMBOS.items():
            res[f"{v}__{c}"] = sum(res[f"{v}__{p}"] for p in parts)
    return res


def evaluate(res):
    stats = NAMES + list(COMBOS)
    rows = []
    for v in VARIANTS:
        for st in stats:
            e = res[f"{v}__{st}"] - res[f"act__{st}"]
            rows.append((v, st, e.abs().mean(), float(np.sqrt((e ** 2).mean())), e.mean()))
    summary = pd.DataFrame(rows, columns=["variant", "stat", "MAE", "RMSE", "bias"])

    mae = summary.pivot(index="variant", columns="stat", values="MAE").loc[VARIANTS, stats]
    pd.set_option("display.width", 200)
    print(f"\n=== Average miss per player-game (MAE, lower is better). {len(res)} player-games scored ===")
    print(mae.round(2).to_string())
    print("\n=== % better (+) or worse (-) than the AVG20 baseline ===")
    print((100 * (1 - mae.div(mae.loc["AVG20"], axis=1))).round(1).drop(index="AVG20").to_string())

    def gain(a, b):
        return (100 * (1 - mae.loc[a, stats] / mae.loc[b, stats])).round(1)
    print("\n=== What each piece adds (% change in MAE; + = helps) ===")
    print(pd.DataFrame({"adjustments (FULL vs RATE_MIN)": gain("FULL", "RATE_MIN"),
                        "defense-vs-position (FULL_DVP vs FULL)": gain("FULL_DVP", "FULL"),
                        "same, actual minutes (FULL_AM vs RATE_MIN_AM)": gain("FULL_AM", "RATE_MIN_AM"),
                        "DvP, actual minutes (FULL_DVP_AM vs FULL_AM)": gain("FULL_DVP_AM", "FULL_AM")}).T.to_string())

    # what do the data say about home/away and back-to-back effects? (actual minutes, no adjustments)
    act, pred = res["act__PRA"], res["RATE_MIN_AM__PRA"]
    overall = act.sum() / pred.sum()
    print("\n=== Real-world effects found in the data (compare with the app's +1% / -1% / -2% guesses) ===")
    for label, mask in (("Home", res["home"]), ("Away", ~res["home"]),
                        ("Back-to-back", res["b2b"]), ("Rested", ~res["b2b"])):
        if mask.sum() > 30:
            print(f"  {label:13s} {100 * ((act[mask].sum() / pred[mask].sum()) / overall - 1):+.1f}%   (n={int(mask.sum())})")
    print("\n=== Volume x efficiency scoring (FULL_VE vs FULL, % change in MAE; + = helps) ===")
    print(pd.DataFrame({"default minutes": gain("FULL_VE", "FULL"), "actual minutes": gain("FULL_VE_AM", "FULL_AM")}).T.to_string())
    fit_spread_and_calibration(res)
    print("\nOverall minutes error: projected default minutes are off by "
          f"{(res['default_min'] - res['actual_min']).abs().mean():.1f} min on average.")
    return summary


def fit_spread_and_calibration(res, folder="data"):
    """
    Learns (1) how spread out results are around a projection and how the stats move together (used for the app's
    ranges and over/under probabilities) -> data/dispersion.json, and (2) a small per-stat correction by minutes
    bucket (used by the app's calibration switch) -> data/calibration.json.
    """
    os.makedirs(folder, exist_ok=True)
    disp, resid = {}, {}
    for n in NAMES:
        pred, act = res[f"FULL__{n}"].to_numpy(), res[f"act__{n}"].to_numpy()
        disp[n] = distributions.fit_dispersion(pred, act)
        resid[n] = act - pred
    corr = distributions.fit_corr(resid)
    distributions.save_params(disp, corr, os.path.join(folder, "dispersion.json"))
    print("\n=== Spread of results (variance = a x projection + b x projection^2) ===")
    for n in NAMES:
        a, b = disp[n]
        print(f"  {n:4s} a={a:.2f} b={b:.3f}   (a typical 20-point scorer: std about {np.sqrt(a*20 + b*400):.1f})" if n == "PTS"
              else f"  {n:4s} a={a:.2f} b={b:.3f}")
    print("  correlations between stats:", {f"{a}-{b}": round(v, 2) for (a, b), v in corr.items()})

    cal = {}
    bucket = res["actual_min"].map(lambda m: "lt20" if m < 20 else ("20to30" if m < 30 else "30plus"))
    print("\n=== Calibration: actual / projected with actual minutes, by minutes bucket (1.00 = unbiased) ===")
    for n in NAMES:
        cal[n] = {}
        for b in ("lt20", "20to30", "30plus"):
            m = bucket == b
            if m.sum() < 50:
                continue
            ratio = res.loc[m, f"act__{n}"].sum() / res.loc[m, f"FULL_AM__{n}"].sum()
            w = m.sum() / (m.sum() + 300.0)                      # only move part of the way when the sample is small
            cal[n][b] = round(float(1 + w * (ratio - 1)), 4)
        print(f"  {n:4s} {cal[n]}")
    with open(os.path.join(folder, "calibration.json"), "w") as f:
        json.dump(cal, f)
    print("  Saved data/dispersion.json and data/calibration.json (the app picks them up automatically).")


def load_positions(seasons):
    """player_id -> position, from the NBA player index (best effort)."""
    pos = {}
    for s in seasons:
        try:
            idx = d.fetch_player_index(s)
            for pid, p in zip(idx["PERSON_ID"], idx["POSITION"]):
                pos.setdefault(int(pid), p)
        except Exception as e:  # noqa: BLE001
            print(f"  (couldn't load positions for {s}: {e})")
    return pos


def main():
    start = sys.argv[1] if len(sys.argv) > 2 else START_DATE
    end = sys.argv[2] if len(sys.argv) > 2 else END_DATE
    y = int(SEASON[:4])
    prev_season = f"{y - 1}-{str(y)[-2:]}"
    print(f"Pulling game logs for {prev_season} and {SEASON}...")
    prev_raw, cur_raw = d.fetch_logs(prev_season), d.fetch_logs(SEASON)
    print("Pulling player positions...")
    positions = load_positions([SEASON, prev_season])
    if not positions:
        print("  No positions available: the defense-vs-position test will show no effect.")
    res = run_backtest(prev_raw, cur_raw, positions, start, end)
    summary = evaluate(res)
    res.to_csv("backtest_rows.csv", index=False)
    summary.to_csv("backtest_summary.csv", index=False)
    print("\nSaved backtest_summary.csv and backtest_rows.csv")


if __name__ == "__main__":
    main()
