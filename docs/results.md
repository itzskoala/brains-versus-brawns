# UFC Elo — Results & Decisions

This is the project's living record of experiments, findings, recommendations, and important technical decisions.

Do not use this file as a task list. Tasks belong in `tasks.md`.

---

# Project Findings

Record important discoveries about the dataset, architecture, models, and system here.

## Dataset

### Findings

* Tables: `fighter.csv` (4,581 fighters), `event.csv` (1,259 events, 1993-11-12 → 2026-08-08), `fight.csv` (11,441 fights), `round.csv` (25,131 round rows), `fighter_bonus.csv` (2,409 bonus records), `master.csv` (11,441 rows — a pre-joined, denormalized fight+fighter+round-totals table).
* Keys are clean: `fight_id`/`fighter_id`/`event_id` are unique per row in their source tables, zero exact duplicate rows anywhere, and `winner_id` is always either `r_id`, `b_id`, or null — no orphaned foreign keys between tables.
* `result_status`: win 11,238 / no_contest 118 / draw 85. `winner_id` is null for exactly the 203 non-`win` rows — consistent.
* `method` has 11 categories; decisions (unanimous+split+majority) are ~43% of fights, KO/TKO ~32%, submission ~21%.
* Volume grows over time: <100 fights/year before 2000, ramps through the 2000s, stabilizes at ~500-600 fights/year from 2014 onward (2026 partial-year to date).
* Round-by-round data (`round.csv`) is incomplete for older fights: 330 of 11,441 fights (2.9%) have no round rows at all, and `r_ctrl`/`b_ctrl` (control time) is null for 2,658 of 25,131 round rows.
* `fighter.csv` career-stat columns (`slpm`, `str_acc`, `sapm`, `str_def`, `td_avg`, `td_acc`, `td_def`, `sub_avg`) use `0` for both "genuinely zero" and "no UFC.com record available" — 858 fighters (~19%) have `slpm == 0`, which is a missingness marker, not a true rate, and must not be imputed/used as a literal 0 in features.
* `height` is a free-text string (`5' 10"`), not numeric — needs parsing before use.
* `reach_inches` is missing for 1,992 fighters (43%) and `stance` for 897 (20%) — both meaningfully missing-not-at-random (older/lower-profile fighters).
* 8 pairs of fighters share an identical `fighter_name` but have distinct `fighter_id`, different `dob`/`weight_lbs` (e.g. two "Bruno Silva", two "Michael McDonald") — real namesakes, not a dedup bug. **Never join/dedupe on `fighter_name`; always use `fighter_id`.**
* 2,239 of 4,124 fighters (54%) have fought from both the red (`r_id`) and blue (`b_id`) corner across their career — corner is not a fixed attribute of a fighter, it's assigned per fight.

### Data Quality Issues

* **Red-corner leakage in historical data (critical, affects modeling).** Red corner (`r_id`) wins 67.6% of decided fights overall — far from the 50/50 a coin-flip-corner-assignment would produce. Breaking this out by year shows why: red wins **~100% of fights from 1993–2009**, then drops sharply to a stable **55–60% from 2010 onward**. The pre-2010 near-100% rate means `r_id` was effectively backfilled as "the winner" for early UFCStats records rather than reflecting a real pre-fight corner assignment — using `r_id`/`b_id` as a naive feature would leak the outcome for any fight before ~2010. Post-2010, red still wins ~57% (and 68–77% in title fights specifically), which looks like a real convention (red corner = higher-ranked/betting-favorite fighter) rather than leakage, but it's still a strong prior baked into the label that any model comparison must account for.
* **Naive baseline is not 50%.** Because of the above, "always predict red" scores ~67.6% accuracy over the full dataset and ~57% even restricted to 2010+. Any classifier must be benchmarked against this corner-prior baseline, not a coin flip, or its offline metrics will look better than they are.
* `weight_lbs` has extreme outliers (up to 770 lbs) — these are legitimate historical UFC 1-era fighters (e.g. Emmanuel Yarborough), not data errors, but they will skew scaling/normalization if not treated (e.g. winsorized or excluded from early modern-era modeling).
* 330 fights (2.9%) have no round-level detail in `round.csv` — any round-level feature will need to handle these as missing rather than assuming coverage.
* `event.csv.location` missing for 476 of 1,259 events (38%); `bonuses` in `master.csv` is null for 9,090 of 11,441 rows (most fights simply have no bonus, consistent with `fighter_bonus.csv` covering only 2,351 fights).

### Important Assumptions

* `fighter_id` is the correct join key for fighter identity everywhere; fighter name is not unique and must not be used for matching or deduplication.
* Career stat columns in `fighter.csv` (slpm, str_acc, etc.) appear to be **as-of-latest-scrape** aggregates, not point-in-time-of-fight snapshots — using them directly as pre-fight features for older fights would leak future career performance into the past. This needs to be confirmed before use in features/Elo, and until confirmed, treat these columns as unsuitable for time-aware modeling without per-fight historical reconstruction from `round.csv`/`fight.csv`.
* Given the red-corner artifact above, any model trained across the full date range should either (a) restrict training/eval to 2010+ fights where corner reflects a real signal, or (b) explicitly model/neutralize corner as a confound (e.g. symmetrize features by swapping red/blue at training time) rather than treating `r_id`/`b_id` as arbitrary labels.

---

# Experiments

Record meaningful experiments here.

## Experiment Template

### Experiment — [Name]

**Date:**

**Goal:**

**Model:**

**Features:**

**Configuration:**

**Dataset:**

**Evaluation:**

**Results:**

```text
Metric: Value
Metric: Value
Metric: Value
```

**Observations:**

*

**Conclusion:**

*

**Next Step:**

*

---

# Model Results

Track important model results here.

| Model    | Features | Metric 1 | Metric 2 | Metric 3 | Notes |
| -------- | -------- | -------: | -------: | -------: | ----- |
| Baseline | —        |        — |        — |        — | —     |

---

# Elo Results

Record meaningful findings from the Elo system.

### Configuration

```text
Initial Rating:
K Factor:
Other Parameters:
```

### Findings

*

### Experiments

*

---

# Feature Findings

Record useful discoveries about feature engineering.

### Useful Features

*

### Features That Did Not Help

*

### Features Removed

*

### Notes

*

---

# Error Analysis

Record meaningful patterns in model errors.

### Observed Errors

*

### Patterns

*

### Potential Explanations

*

### Follow-Up Experiments

*

---

# Recommendations

Record recommendations made during development.

## Recommendation — [Topic]

**Recommendation:**

**Reason:**

**Alternatives considered:**

**Decision:**

**Status:**

* [ ] Pending
* [ ] Accepted
* [ ] Rejected

---

# Technical Decisions

Record decisions that affect the architecture or implementation.

## Decision — [Topic]

**Decision:**

**Why:**

**Alternatives:**

**Consequences:**

---

# Lessons Learned

Record useful lessons discovered during development.

*

---

# Limitations

Document known limitations of the data, Elo system, models, evaluation, API, or deployment.

*

---

# Final Results

Complete this section when the major project work is finished.

## Final Model

**Model:**

**Features:**

**Configuration:**

## Evaluation

| Metric      | Result |
| ----------- | -----: |
| Accuracy    |      — |
| ROC-AUC     |      — |
| Log Loss    |      — |
| Brier Score |      — |

## Final Findings

*

## Known Limitations

*

## Future Work

*
