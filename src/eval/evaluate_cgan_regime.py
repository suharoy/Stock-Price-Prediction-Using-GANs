import os, json, yaml, argparse, math, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset

# cGAN generator 
from src.models.cgan import GeneratorLSTMCond

# -------------------- metrics --------------------
def mae(y_pred, y_true):  return float(np.mean(np.abs(y_pred - y_true)))
def rmse(y_pred, y_true): return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
def mape(y_pred, y_true, eps=1e-8):
    d = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / d)))

# -------------------- io helpers --------------------
def load_npz(split, pdir):
    d = np.load(os.path.join(pdir, f"{split}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)          # [N,T,F]
    y = torch.tensor(d["y"][:, 0, 0], dtype=torch.float32) # [N]
    return X, y

def chrono_slices(n, tr, val):
    n_tr = int(n * tr); n_va = n_tr + int(n * val)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def target_dates(idx: pd.DatetimeIndex, slc: slice, T: int, H: int):
    sub = idx[slc]
    return sub[T: len(sub) - H + 1]

# -------------------- regime condition --------------------
def regime3_onehot(dates: pd.DatetimeIndex, close_series: pd.Series, w: int = 20) -> np.ndarray:
    """
    Build 3 volatility regimes on the FULL close series (as a pandas Series with a DateTimeIndex),
    then align them to 'dates' and return a one-hot array [len(dates), 3].
    """
    if not isinstance(close_series, pd.Series):
        # if an ndarray slipped in, wrap it with the dates index of the full series (not target dates!)
        raise TypeError("close_series must be a pandas Series with a DateTime index.")

    # daily returns + rolling std
    ret = close_series.pct_change().fillna(0.0)
    vol = ret.rolling(w, min_periods=w).std().bfill()  # bfill() avoids the future-warning

    # thresholds from the entire history
    q1, q2 = vol.quantile([0.33, 0.66])

    # classify: 0 (low), 1 (mid), 2 (high); keep DateTime index by wrapping in a Series
    reg_arr = np.digitize(vol.to_numpy(), bins=[q1, q2])          # ndarray of ints in {0,1,2}
    reg_ser = pd.Series(reg_arr, index=vol.index, dtype="Int64")  # pandas NA-aware integer

    # align to target dates (returns pandas Series indexed by 'dates')
    reg_aligned = reg_ser.reindex(dates)

    # one-hot encode, skipping NA
    C = 3
    out = np.zeros((len(dates), C), dtype=np.float32)
    for i, r in enumerate(reg_aligned):
        if pd.isna(r): 
            continue
        rr = int(r)
        if 0 <= rr < C:
            out[i, rr] = 1.0
    return out

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True, help="runs/cgan_regime_msft/ckpt.pt")
    ap.add_argument("--config", required=True, help="configs/dataset.yaml")
    ap.add_argument("--split",  choices=["train", "val", "test"], default="test")
    args = ap.parse_args()

    ds = yaml.safe_load(open(args.config))
    p  = ds["paths"]

    meta = json.load(open(os.path.join(p["processed"], "meta.json")))
    T, H, F   = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol     = meta.get("symbol", ds["symbols"][0])
    close_idx  = int(meta["close_index"])
    mar_window = int(ds["mar_window"])

    # arrays (z-space)
    X, y_true_z = load_npz(args.split, p["processed"])

    # load market series (USD) and build target dates + regimes
    df = pd.read_csv(os.path.join(p["interim"], f"{symbol}_mar.csv"),
                     parse_dates=["date"]).set_index("date")
    close_full = df["close"].astype(float)      # pandas Series with DateTimeIndex
    n = len(df)
    sl_tr, sl_va, sl_te = chrono_slices(n, ds["splits"]["train"], ds["splits"]["val"])
    dates = target_dates(df.index, {"train": sl_tr, "val": sl_va, "test": sl_te}[args.split], T, H)

    # condition vectors (C=3)
    cond_np = regime3_onehot(dates, close_full, w=20)
    C = cond_np.shape[1]

    # rebuild Generator
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt.get("cfg", {}).get("model", {})
    hidden  = int(model_cfg.get("hidden", 128))
    layers  = int(model_cfg.get("layers", 1))
    dropout = float(model_cfg.get("dropout", 0.2))

    G = GeneratorLSTMCond(F, C, hidden=hidden, layers=layers, dropout=dropout)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # predict in z-space
    cond = torch.tensor(cond_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y_true_z, cond), batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _, cb in loader:
            preds.append(G(xb, cb)[:, close_idx].cpu().numpy())
    y_pred_z = np.concatenate(preds)

    # denormalize to USD via MAR
    mu = close_full.rolling(mar_window, min_periods=mar_window).mean()
    sd = close_full.rolling(mar_window, min_periods=mar_window).std().replace(0, 1e-8)
    mu_t = mu.reindex(dates).to_numpy()
    sd_t = sd.reindex(dates).to_numpy()

    y_true = y_true_z.numpy() * sd_t + mu_t
    y_pred = y_pred_z * sd_t + mu_t

    print(f"== cGAN-Regime | {args.split.upper()} (USD) | Symbol={symbol} ==")
    print(f"MAE  : {mae(y_pred, y_true):.4f}")
    print(f"RMSE : {rmse(y_pred, y_true):.4f}")
    print(f"MAPE : {mape(y_pred, y_true):.4f}")
    print(f"(T={T} H={H} F={F} | MAR window={mar_window} | C=3 regimes)")

if __name__ == "__main__":
    main()