# NBA Player Projections

A Streamlit app that projects NBA player stats (points, rebounds, assists, threes and combos) and team scores for
each game day, with editable minutes, teammate-out effects, blowout risk, over/under probabilities and more.

* Online: deployed on Streamlit Community Cloud (the app reads `data/latest.json`).
* Locally: see `DEPLOY_GUIDE.txt`, or run `py refresh_data.py` then `py -m streamlit run app.py`.

Data comes from the NBA's public stats via the `nba_api` package. Projections are statistical estimates for
information only, with no guarantee of results. Non-commercial use.
