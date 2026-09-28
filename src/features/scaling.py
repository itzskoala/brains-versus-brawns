from sklearn.preprocessing import StandardScaler


def fit_scaler(df, columns):
    """Fit a StandardScaler on the given columns. Call this on the training
    split only - fitting on validation/test data leaks information into it."""
    scaler = StandardScaler()
    scaler.fit(df[columns])
    return scaler


def apply_scaler(df, columns, scaler):
    """Apply an already-fit scaler to the given columns. Returns a new
    DataFrame. Use the same scaler (fit on train) for validation, test, and
    serving so all data is scaled the same way."""
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df
