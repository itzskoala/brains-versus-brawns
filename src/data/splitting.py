def split_fights(df, val_size=0.15, test_size=0.15):
    """Split fights chronologically: oldest fights for training, most recent
    for test. Order matters here, not randomness - the model will only ever
    predict fights it hasn't seen yet, so evaluation has to respect that.
    Returns (train, val, test)."""
    df = df.sort_values("event_date")

    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))

    train = df.iloc[:val_start]
    val = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]

    return train, val, test


def walk_forward_splits(df, freq="Y", min_train_size=500):
    """Generate expanding-window folds ordered by event_date: fights are
    bucketed into calendar periods (freq: "Y" yearly, "Q" quarterly), and
    for each period the training set is every fight strictly before it while
    the test set is the fights within it. The training window only ever
    grows forward in time, so no fold trains on a fight that happened after
    one it's being evaluated against - unlike a per-fighter split, where two
    fighters' career-relative cutoffs can land on different calendar dates
    and let a "train" fight occur after a "test" fight.

    Periods with less than min_train_size accumulated training fights are
    skipped, since the earliest UFC years have too few events per year to
    train or evaluate on meaningfully. Yields (period, train, test) tuples
    in chronological order."""
    df = df.sort_values("event_date").copy()
    df["_period"] = df["event_date"].dt.to_period(freq)

    for period in sorted(df["_period"].unique()):
        train = df[df["_period"] < period]
        test = df[df["_period"] == period]
        if len(train) < min_train_size or len(test) == 0:
            continue
        yield period, train.drop(columns="_period"), test.drop(columns="_period")
