---
title: BrainsVSBrawn
emoji: 🥊
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# UFC Matchup Predictor

Pick two fighters and a weight division, and a model (logistic regression,
random forest, or XGBoost) trained on historical UFC Elo-style features
predicts a win probability for each corner, plus a "tale of the tape"
stat comparison and any past meetings between the two fighters.

Gradio UI in `frontend/gradio.py`, calling a FastAPI backend (`api/main.py`)
- both started together by `app.py` inside this Space.
