import argparse, os, yaml, pandas as pd, yfinance as yf
from tqdm import tqdm

def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} in [{start}, {end}]")
    df = df.rename(columns=str.lower)
    df = df[["open","high","low","close","volume"]].copy()
    df.index.name = "date"
    df["ma5"] = df["close"].rolling(5).mean()
    vol_mean20 = df["volume"].rolling(20).mean().replace(0, 1e-8)
    df["turnover"] = (df["volume"] / vol_mean20).clip(lower=0)
    return df.dropna()

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    raw_dir = cfg["paths"]["raw"]
    os.makedirs(raw_dir, exist_ok=True)
    for sym in tqdm(cfg["symbols"], desc="Ingest"):
        df = fetch(sym, cfg["start"], cfg["end"])
        out = os.path.join(raw_dir, f"{sym}.csv")
        df.to_csv(out)
        print(f"saved: {out} -> {df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)