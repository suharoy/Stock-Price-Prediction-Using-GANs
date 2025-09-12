import os
import json
import yaml
import argparse
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- import your baseline model ---
from src.models.lstm_reg import LSTMRegressor


# ----------------- metrics (USD) -----------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


# ----------------- helpers -----------------
def load_npz(name: str, processed_dir: str):
    """Load a split from data/processed/*.npz and return torch tensors."""
    d = np.load(os.path.join(processed_dir, f"{name}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)           # [N,T,F]
    y = torch.tensor(d["y"][:, 0, 0], dtype=torch.float32)  # [N] (only first horizon)
    return X, y

def chrono_slices(n: int, train_frac: float, val_frac: float):
    n_tr = int(n * train_frac)
    n_va = n_tr + int(n * val_frac)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def gen_target_dates(index: pd.DatetimeIndex, slc: slice, T: int, H: int):
    """
    Recreate the target dates produced by window.py:
      for i in range(T, len(sub) - H + 1): target date is sub[i]
    """
    sub = index[slc]
    start = T
    end = len(sub) - H + 1
    if end <= start:
        return pd.DatetimeIndex([])
    return sub[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True, help="Path to checkpoint, e.g. runs/lstm_msft/ckpt.pt")
    ap.add_argument("--config", required=True, help="Path to configs/dataset.yaml")
    ap.add_argument("--split",  default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    # ---- load dataset config and meta ----
    ds_cfg = yaml.safe_load(open(args.config))
    p = ds_cfg["paths"]
    meta = json.load(open(os.path.join(p["processed"], "meta.json")))
    T, H, F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol = meta.get("symbol", ds_cfg["symbols"][0])
    close_idx = int(meta["close_index"])
    mar_w = int(ds_cfg["mar_window"])

    # ---- load split arrays (z-space) ----
    X, y_true_z = load_npz(args.split, p["processed"])

    # ---- rebuild model from checkpoint config ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt["cfg"]["model"]  # expects keys: hidden, layers, dropout
    model = LSTMRegressor(F, **model_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- predict in z-space ----
    loader = DataLoader(TensorDataset(X, y_true_z), batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb).cpu().numpy())
    y_pred_z = np.concatenate(preds, axis=0)  # [N]

    # ---- denormalize to USD using MAR rolling μ,σ on raw close ----
    # read interim to recompute μ,σ (no leakage; same window as training)
    df_mar = pd.read_csv(os.path.join(p["interim"], f"{symbol}_mar.csv"),
                         parse_dates=["date"]).set_index("date")
    # raw close in dollars
    close_usd = df_mar["close"].astype(float)

    # rolling mean/std (uses past mar_w days due to min_periods=mar_w)
    mu = close_usd.rolling(mar_w, min_periods=mar_w).mean()
    sd = close_usd.rolling(mar_w, min_periods=mar_w).std().replace(0, 1e-8)

    # reproduce the same chronological split used in window.py
    n_all = len(df_mar)
    slc_train, slc_val, slc_test = chrono_slices(n_all, ds_cfg["splits"]["train"], ds_cfg["splits"]["val"])
    slc_map = {"train": slc_train, "val": slc_val, "test": slc_test}
    target_dates = gen_target_dates(df_mar.index, slc_map[args.split], T, H)

    if len(target_dates) != len(y_pred_z):
        raise RuntimeError(f"Alignment mismatch: {len(target_dates)} dates vs {len(y_pred_z)} predictions.")

    mu_t = mu.reindex(target_dates).to_numpy()
    sd_t = sd.reindex(target_dates).to_numpy()

    y_pred_usd = y_pred_z * sd_t + mu_t
    y_true_usd = y_true_z.numpy() * sd_t + mu_t

    # ---- metrics in USD ----
    mae_usd  = mae(y_true_usd, y_pred_usd)
    rmse_usd = rmse(y_true_usd, y_pred_usd)
    mape_usd = mape(y_true_usd, y_pred_usd)

    # ---- report ----
    print(f"== {args.split.upper()} (USD) ==")
    print(f"MAE  : {mae_usd:.4f}")
    print(f"RMSE : {rmse_usd:.4f}")
    print(f"MAPE : {mape_usd:.4f}")
    print(f"(Symbol={symbol} | T={T} H={H} F={F} | MAR window={mar_w})")


if __name__ == "__main__":
    main()