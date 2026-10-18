# Feature Reference

Every feature `src/features/engineering.py`'s `engineer_fold_features` produces, grouped by the
function that builds it (same grouping the code itself uses). All of them are leakage-safe by
construction: every "prior_*"/history feature is computed from a fighter's fights **strictly
before** the current one (`_prior_sum_by_date`, with explicit same-date-tie handling), and the
`fit_imputer`/`fit_scaler` steps that run on top of these are always fit on the current
walk-forward fold's training rows only (see `src/features/imputation.py`, `src/features/scaling.py`).

Columns are built per corner (`r_<name>`/`b_<name>`) and, for most groups, also as a
`<name>_diff` (`r_<name> - b_<name>`). `src/models/logistic_regression.py`'s `_symmetrize` further
re-expresses many of these as `self_<base>`/`opp_<base>` pairs (`PAIRED_BASES`) for the
"combined" representation that is the actual production `FEATURE_COLUMNS` - see that file's
module docstring for why (plain diffs lose information a tree/linear model can otherwise use).
**Not every column documented here is in `FEATURE_COLUMNS`** - some exist and are computed on
every fold but are not yet fed to the trained model (noted per-feature below); promoting one
requires an ablation per the existing convention in `docs/results.md`, not just adding it to
the list.

## Outcome handling (read this first)

As of the "complete fight data" change, `src/data/cleaning.py` no longer drops draw/no-contest
fights - every fight (win, draw, no-contest, including DQ and overturned results) flows through
`engineer_fold_features` so a fighter's **history features are complete**. `_add_outcome_flags`
(the shared helper every history function below uses) derives `is_winner`/`is_loser`/`is_draw`/
`is_no_contest` from `result_status` (not `winner_id` alone, which is null for both a draw and a
no-contest). A DQ counts as an ordinary win/loss for the two fighters involved (`result_status`
is `"win"` for a DQ, same as any decided fight) - no special-casing needed there.

**Hard constraint, separate from the above:** the model itself must only ever predict Red or Blue
- never a draw, no-contest, DQ, or overturned result. `is_predictable_outcome` (also in
`engineering.py`) is the filter every `run_backtest`/`nested_validation`/`fit_final_model` call
applies to its fold's rows *after* `engineer_fold_features` (so history stays complete) but
*before* fitting/scoring (so these outcomes are never a training or evaluation example). See
`tests/test_train_predict_evaluate.py::test_run_backtest_never_trains_or_tests_on_a_non_predictable_outcome`.

## `add_matchup_features` - static, no history/leakage risk at all

| Feature | Formula |
|---|---|
| `r_age` / `b_age` | days between `dob` and `event_date`, ÷365.25; clipped to NaN outside [16, 60] (implausible age -> bad DOB, not a real value) |
| `height_diff`, `reach_diff`, `weight_diff`, `age_diff` | `r_<x> - b_<x>` |
| `reach_diff_x_<division>` | `reach_diff` where `weight_class == division`, else 0 - one column per of the 8 `MAIN_DIVISIONS`, so a model can learn a division-specific reach effect instead of one averaged across all of them |

## `add_fighter_history_features` - career win/loss/draw/no-contest record

| Feature | Formula |
|---|---|
| `prior_fights` | count of all prior fights (any outcome) |
| `prior_wins` / `prior_losses` / `prior_draws` / `prior_no_contests` | raw counts from `_add_outcome_flags` |
| `prior_win_rate` | `prior_wins / (prior_wins + prior_losses)` - draws/no-contests excluded from the denominator (a draw is not a loss); NaN if 0 decided fights |
| `prior_draw_rate`, `prior_no_contest_rate` | `prior_draws / prior_fights`, `prior_no_contests / prior_fights` - **not yet in `FEATURE_COLUMNS`** (computed, available for a future ablation) |
| `days_since_last_fight` | days since the fighter's most recent strictly-earlier date |
| `experience_diff` | `r_prior_fights - b_prior_fights` |
| `prior_win_rate_diff`, `prior_draw_rate_diff`, `prior_no_contest_rate_diff` | `r_<x> - b_<x>` (the latter two not yet in `FEATURE_COLUMNS`, same as their non-diff versions) |

## `add_career_stat_features` - career-to-date volume/style, per minute

Built from `FIGHT_STAT_COLUMNS` (21 raw per-fight stats: knockdowns, sig/total strikes
landed+attempted overall and by target/position, takedowns, sub attempts, reversals, control
seconds). A draw/no-contest/DQ fight's stats **do** count toward these totals (only the
win/loss/draw bookkeeping above treats them specially) - this is what "preserve strikes and
other action recorded before a fight ended" means in practice. Fights with no round-by-round
detail (`rounds_fought == 0`, 330 of 11,441) are excluded from both the numerator and the
duration denominator (a missingness marker, not a genuine zero-output fight).

| Feature | Formula |
|---|---|
| `<stat>_per_min` (one per raw stat, 21 total) | prior cumulative `<stat>` ÷ prior cumulative fight-duration minutes |
| `career_sig_str_acc`, `career_td_acc` | landed / attempted, from prior cumulative totals |
| `head_strike_share`, `body_strike_share`, `leg_strike_share`, `distance_strike_share`, `clinch_strike_share`, `ground_strike_share` | that target/position's prior landed strikes ÷ prior total landed strikes |
| every one of the above, `_diff` version | `r_<x> - b_<x>` |

## `add_finish_tendency_features` - how a fighter's fights tend to end

Every `METHOD_LABEL_MAP` category (`ko_tko`, `submission`, the three decision subtypes, `dq`,
`overturned`, `could_not_continue`, `tko_doctor_stoppage`, `other`) gets its own rate - none are
merged into a generic bucket. A draw/no-contest's `method_label` (how the fight ended, independent
of who "won") counts toward `prior_<method>_rate`; it cannot count toward `prior_win_<method>_rate`
since nobody won it (`is_winner` is 0 for both corners on a draw/NC).

| Feature | Formula |
|---|---|
| `prior_<method>_rate` (11 columns) | count of prior fights ending in that method ÷ `prior_fights` |
| `prior_win_<method>_rate` (11 columns) | count of prior **wins** by that method ÷ `prior_wins`; NaN if 0 prior wins |
| `avg_finish_round_prior` | mean `finish_round` over prior fights that were NOT a decision (`DECISION_LABELS`) |
| every one of the above, `_diff` version | `r_<x> - b_<x>` |

## `add_recent_form_features` - streaks and short-term form

A draw/no-contest is neither a win nor a loss for streak purposes: it breaks whichever streak was
running (win or loss) without starting the other kind - see `_outcome` in the function body. Blocks
are grouped by this 3-way outcome, not by `is_winner` alone, specifically so this holds.

| Feature | Formula |
|---|---|
| `win_streak` / `loss_streak` | length of the run of consecutive wins/losses entering this fight; a draw/no-contest resets both to 0 |
| `form_last5_win_rate` | rolling mean of `is_winner` (0 for a loss AND a draw/no-contest - a simplification, not a 0.5 credit for a draw) over the last 5 fights, shifted by one so it never includes the current fight |
| every one of the above, `_diff` version | `r_<x> - b_<x>` |

## `add_stance_matchup_features` - static, no leakage risk

| Feature | Formula |
|---|---|
| `stance_mismatch` | 1 if one fighter is Orthodox and the other Southpaw, else 0 - symmetric (not a `_diff`; `_symmetrize` leaves it unnegated for the blue-perspective row) |

## `add_early_pace_features` - needs `fight_rounds.csv`

| Feature | Formula |
|---|---|
| `r1_avg_sig_landed_prior` | mean round-1 significant strikes landed over prior fights that have round-level detail; `_diff` version too |

## Display-only stats (API/UI, never model inputs)

`src/features/snapshot.py`'s `build_fighter_opponent_stats`/`strength_of_schedule` compute a few
more numbers shown in the Tale of the Tape but never passed to any trained model:

| Feature | Formula |
|---|---|
| `sapm` | opponent's significant strikes landed against this fighter, per minute |
| `striking_defense`, `takedown_defense` | `1 - (opponent's successful strikes or takedowns against this fighter / their attempts)` |
| `avg_fight_time_min` | mean fight duration across this fighter's detailed-round fights |
| `strength_of_schedule` | mean of each past opponent's own `prior_win_rate` **as of the date they were faced** (not their eventual career record - leakage-checked) |

## Not yet promoted to `FEATURE_COLUMNS`

`prior_draw_rate`, `prior_no_contest_rate`, and their `_diff` versions are computed on every fold
but not fed to any trained model yet. Promoting them follows the same ablation process already used
for `career_volume_stats`/`career_style_stats`/`stance_mismatch` (`docs/results.md`, 2026-10-06) -
not a default inclusion.
