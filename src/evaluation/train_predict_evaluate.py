from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import logistic_regression, random_forest, xgboost_model

MODELS = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
    "xgboost_model": xgboost_model,
}


def run_all(fights, rounds=None, freq="Y", min_train_size=500):
    """Run every model in MODELS' own run_backtest(..., return_predictions=True)
    on the same fights/rounds/freq/min_train_size, tag each model's rows with
    a "model" column, and concatenate. Returns (predictions_df, metrics_df)."""
    all_predictions = []
    all_metrics = []

    for model_name, module in MODELS.items():
        metrics, predictions = module.run_backtest(
            fights, rounds=rounds, freq=freq, min_train_size=min_train_size, return_predictions=True)
        metrics = metrics.copy()
        predictions = predictions.copy()
        metrics.insert(0, "model", model_name)
        predictions.insert(0, "model", model_name)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    return predictions_df, metrics_df


def summarize_metrics(metrics_df):
    """One row per model: mean of every per-fold numeric metric, plus how
    many folds beat the mean-baseline - a groupfby-mean of metrics_df itself,
    so this can never disagree with the saved per-fold CSV."""
    numeric_cols = ["accuracy", "log_loss", "roc_auc", "mse", "target_variance"]
    summary = metrics_df.groupby("model")[numeric_cols].mean()
    summary["n_folds"] = metrics_df.groupby("model").size()
    summary["folds_beating_baseline"] = metrics_df.groupby("model")["beats_mean_baseline"].sum()
    return summary.reset_index()


def save_outputs(predictions_df, metrics_df, out_dir="docs"):
    """Save per-fold predictions, per-fold metrics, and the metrics summary
    as CSVs under out_dir. Each fight_id appears twice per fold per model -
    once per corner perspective, see _symmetrize - y_true/y_pred/y_prob are
    each that perspective's own values."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(out_dir / "winner_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "winner_metrics_by_fold.csv", index=False)
    summarize_metrics(metrics_df).to_csv(out_dir / "winner_metrics_summary.csv", index=False)


def save_charts(metrics_df, out_dir="docs"):
    """Save a per-fold accuracy line chart and a mean-metric comparison bar
    chart across models, as PNGs under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in metrics_df.groupby("model"):
        ax.plot(group["period"], group["accuracy"], marker="o", label=model_name)
    ax.set_xlabel("Walk-forward fold (year)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Winner-prediction accuracy by walk-forward fold")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "winner_accuracy_by_fold.png")
    plt.close(fig)

    summary = summarize_metrics(metrics_df).set_index("model")
    metrics_to_plot = ["accuracy", "roc_auc", "log_loss"]
    fig, ax = plt.subplots(figsize=(8, 5))
    summary[metrics_to_plot].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean across folds")
    ax.set_title("Winner-prediction model comparison")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "winner_model_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    from src.data.cleaning import clean_fights
    from src.data.ingestion import build_master_equivalent, load_fighters, load_fights

    print("[train_predict_evaluate] loading data/fights.csv + data/fighter.csv")
    master = build_master_equivalent(load_fights(), load_fighters())
    fights = clean_fights(master)
    print(f"[train_predict_evaluate] {len(fights)} fights after cleaning; running all models")

    predictions_df, metrics_df = run_all(fights)
    summary = summarize_metrics(metrics_df)

    print("\n" + summary.to_string(index=False))

    save_outputs(predictions_df, metrics_df)
    save_charts(metrics_df)
    print("\n[train_predict_evaluate] saved docs/winner_predictions.csv, "
          "docs/winner_metrics_by_fold.csv, docs/winner_metrics_summary.csv, "
          "docs/winner_accuracy_by_fold.png, docs/winner_model_comparison.png")
