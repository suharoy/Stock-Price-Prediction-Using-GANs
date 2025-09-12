import argparse, os, yaml, pandas as pd

def mar_normalize(df: pd.DataFrame, cols, w: int) -> pd.DataFrame:
    mu = df[cols].rolling(window=w, min_periods=w).mean()
    sd = df[cols].rolling(window=w, min_periods=w).std().replace(0, 1e-8)
    z  = (df[cols] - mu) / sd
    z.columns = [f"{c}_z" for c in cols]
    return pd.concat([df, z], axis=1).dropna()

def read_with_date_index(path: str) -> pd.DataFrame:
    """
    Handle files where 'date' may appear on row 3 (after 'Price'/'Ticker' rows),
    or as an unnamed index column.
    """
    # peek first few lines
    with open(path, "r", encoding="utf-8") as f:
        lines = [next(f) for _ in range(5)]
    # find the line that contains 'date' as the first field
    header_row = None
    for i, L in enumerate(lines):
        first = L.strip().split(",")[0].strip().lower()
        if first == "date":
            header_row = i
            break

    if header_row is not None:
        df = pd.read_csv(path, header=header_row)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        # fallback: first column is the date index
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

    return df.sort_index()

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    raw_dir, out_dir = cfg["paths"]["raw"], cfg["paths"]["interim"]
    os.makedirs(out_dir, exist_ok=True)

    cols = cfg["features"]  # [high, low, open, close, volume, turnover, ma5]
    for sym in cfg["symbols"]:
        path = os.path.join(raw_dir, f"{sym}.csv")
        df = read_with_date_index(path)
        zdf = mar_normalize(df, cols, cfg["mar_window"])
        out = os.path.join(out_dir, f"{sym}_mar.csv")
        zdf.to_csv(out)
        print(f"saved: {out} -> {zdf.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)