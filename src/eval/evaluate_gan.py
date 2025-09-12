import os
import json
import yaml
import argparse
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.models.gans_classical import GeneratorLSTM
except Exception:
    from src.models.gan_classical import GeneratorLSTM


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
    d = np.load(os.path.join(processed_dir, f"{name}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)           # [N,T,F]
    y = torch.tensor(d["y"][:, 0, 0], dtype=torch.float32)  # [N] (first horizon)
    return X, y

def chrono_slices(n: int, train_frac: float, val_frac: float):
    n_tr = int(n * train_frac)
    n_va = n_tr + int(n * val_frac)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def gen_target_dates(index: pd.DatetimeIndex, slc: slice, T: int, H: int):
    sub = index[slc]
    start = T
    end = len(sub) - H + 1
    if end <= start:
        return pd.DatetimeIndex([])
    return sub[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True, help="GAN checkpoint path (e.g., runs/gan2019_msft/ckpt.pt)")
    ap.add_argument("--config", required=True, help="configs/dataset.yaml for MAR & paths")
    ap.add_argument("--split",  default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    # ---- dataset config + meta ----
    ds_cfg = yaml.safe_load(open(args.config))
    p = ds_cfg["paths"]
    meta = json.load(open(os.path.join(p["processed"], "meta.json")))
    T, H, F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol = meta.get("symbol", ds_cfg["symbols"][0])
    close_idx = int(meta["close_index"])
    mar_w = int(ds_cfg["mar_window"])

    # ---- load arrays (z-space) ----
    X, y_true_z = load_npz(args.split, p["processed"])

    # ---- rebuild Generator from ckpt ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # try to read model hyperparams; fall back to defaults used in training
    model_cfg = {}
    try:
        # some train scripts store model params here
        model_cfg = ckpt["cfg"].get("model", {})
    except Exception:
        pass
    hidden = int(model_cfg.get("hidden", 128))
    layers = int(model_cfg.get("layers", 1))
    dropout = float(model_cfg.get("dropout", 0.2))

    G = GeneratorLSTM(F, hidden=hidden, layers=layers, dropout=dropout)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # ---- predict (z-space) ----
    loader = DataLoader(TensorDataset(X, y_true_z), batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(G(xb)[:, close_idx].cpu().numpy())
    y_pred_z = np.concatenate(preds, axis=0)  # [N]

    # ---- denormalize to USD using MAR ----
    df_mar = pd.read_csv(os.path.join(p["interim"], f"{symbol}_mar.csv"),
                         parse_dates=["date"]).set_index("date")
    close_usd = df_mar["close"].astype(float)
    mu = close_usd.rolling(mar_w, min_periods=mar_w).mean()
    sd = close_usd.rolling(mar_w, min_periods=mar_w).std().replace(0, 1e-8)

    n_all = len(df_mar)
    slc_train, slc_val, slc_test = chrono_slices(n_all, ds_cfg["splits"]["train"], ds_cfg["splits"]["val"])
    slc_map = {"train": slc_train, "val": slc_val, "test": slc_test}
    dates = gen_target_dates(df_mar.index, slc_map[args.split], T, H)

    if len(dates) != len(y_pred_z):
        raise RuntimeError(f"Alignment mismatch: {len(dates)} dates vs {len(y_pred_z)} predictions.")

    mu_t = mu.reindex(dates).to_numpy()
    sd_t = sd.reindex(dates).to_numpy()
    y_pred_usd = y_pred_z * sd_t + mu_t
    y_true_usd = y_true_z.numpy() * sd_t + mu_t

    # ---- metrics in USD ----
    mae_usd  = mae(y_true_usd, y_pred_usd)
    rmse_usd = rmse(y_true_usd, y_pred_usd)
    mape_usd = mape(y_true_usd, y_pred_usd)

    # ---- report ----
    print(f"== {args.split.upper()} (USD) | Symbol={symbol} ==")
    print(f"MAE  : {mae_usd:.4f}")
    print(f"RMSE : {rmse_usd:.4f}")
    print(f"MAPE : {mape_usd:.4f}")
    print(f"(T={T} H={H} F={F} | MAR window={mar_w} | G: hidden={hidden}, layers={layers}, dropout={dropout})")


if __name__ == "__main__":
    main()