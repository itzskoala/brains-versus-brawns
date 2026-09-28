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
