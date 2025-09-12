import os, json, yaml, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.cgan import GeneratorLSTMCond, DiscriminatorMLPCond  # uses your existing cGAN models

# ---------- utils ----------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def load_npz(name, processed_dir):
    d = np.load(os.path.join(processed_dir, f"{name}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)           # [N,T,F]
    y = torch.tensor(d["y"][:,0,0], dtype=torch.float32)    # [N] (next-day close_z)
    return X, y

def chrono_slices(n: int, tr: float, va: float):
    n_tr = int(n*tr); n_va = n_tr + int(n*va)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def target_dates(index, slc, T, H):
    sub = index[slc]
    return sub[T: len(sub)-H+1]

def regime3_onehot(dates, close_series, w=20):
    """
    Build 3 volatility regimes on the full close series, then align to dates.
    Returns one-hot matrix shape [len(dates), 3].
    """
    # daily returns + rolling std (vol)
    ret = close_series.pct_change().fillna(0.0)
    vol = ret.rolling(w, min_periods=w).std().bfill()   # avoid future warning

    # thresholds on the full history
    q1, q2 = vol.quantile([0.33, 0.66])

    # digitize -> numpy array; wrap into a Series to keep the datetime index
    reg_arr = np.digitize(vol.to_numpy(), bins=[q1, q2])      # values in {0,1,2}
    reg_ser = pd.Series(reg_arr, index=vol.index)             # <- indexed series

    # align to the target dates
    reg_aligned = reg_ser.reindex(dates).to_numpy()

    # one-hot encode
    C = 3
    out = np.zeros((len(dates), C), dtype=np.float32)
    for i, r in enumerate(reg_aligned):
        if 0 <= r < C and not np.isnan(r):
            out[i, int(r)] = 1.0
    return out

def denorm_usd(y_z, dates, interim_dir, symbol, mar_w):
    df = pd.read_csv(os.path.join(interim_dir, f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    close = df["close"].astype(float)
    mu = close.rolling(mar_w, min_periods=mar_w).mean()
    sd = close.rolling(mar_w, min_periods=mar_w).std().replace(0, 1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    return y_z * sd_t + mu_t

# ---------- training ----------
def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))

    meta = json.load(open(os.path.join(cfg["data"]["processed"], "meta.json")))
    T, H, F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol = meta["symbol"]; close_idx = int(meta["close_index"])

    # data arrays
    Xtr, ytr = load_npz("train", cfg["data"]["processed"])
    Xva, yva = load_npz("val",   cfg["data"]["processed"])
    Xte, yte = load_npz("test",  cfg["data"]["processed"])

    # build conditions via regimes
    ds_cfg = yaml.safe_load(open("configs/dataset.yaml"))  # for mar_window & splits
    df_all = pd.read_csv(os.path.join(cfg["data"]["interim"], f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    close_full = df_all["close"].astype(float)
    n = len(df_all)
    sl_tr, sl_va, sl_te = chrono_slices(n, ds_cfg["splits"]["train"], ds_cfg["splits"]["val"])
    dates_tr = target_dates(df_all.index, sl_tr, T, H)
    dates_va = target_dates(df_all.index, sl_va, T, H)
    dates_te = target_dates(df_all.index, sl_te, T, H)

    ctr = torch.tensor(regime3_onehot(dates_tr, close_full), dtype=torch.float32)
    cva = torch.tensor(regime3_onehot(dates_va, close_full), dtype=torch.float32)
    cte = torch.tensor(regime3_onehot(dates_te, close_full), dtype=torch.float32)
    C = ctr.shape[1]

    tr_loader = DataLoader(TensorDataset(Xtr, ytr, ctr), batch_size=int(cfg["train"]["batch_size"]), shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva, cva), batch_size=512)
    te_loader = DataLoader(TensorDataset(Xte, yte, cte), batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = GeneratorLSTMCond(F, C, hidden=128, layers=1, dropout=0.2).to(device)
    D = DiscriminatorMLPCond(T, F, C, hidden=(72,100,10), dropout=0.2).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=float(cfg["train"]["lr_g"]))
    optD = torch.optim.Adam(D.parameters(), lr=float(cfg["train"]["lr_d"]))
    bce, mse = nn.BCELoss(), nn.MSELoss()
    lam_mse, lam_gan = float(cfg["train"]["lambda_mse"]), float(cfg["train"]["lambda_gan"])
    grad_clip, d_steps = float(cfg["train"]["grad_clip"]), int(cfg["train"]["d_steps"])

    best_va = float("inf")
    os.makedirs(cfg["logging"]["outdir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["logging"]["outdir"], "ckpt.pt")

    for epoch in range(1, int(cfg["train"]["epochs"])+1):
        G.train(); D.train()
        for X, yclose, c in tr_loader:
            X, yclose, c = X.to(device), yclose.to(device), c.to(device)

            # ---- Train D ----
            for _ in range(d_steps):
                optD.zero_grad()
                real_next = torch.zeros(X.size(0), F, device=device)
                real_next[:, close_idx] = yclose
                fake_next = G(X, c).detach()
                r = D(X, real_next, c)
                f = D(X, fake_next, c)
                lossD = bce(r, torch.ones_like(r)) + bce(f, torch.zeros_like(f))
                lossD.backward(); optD.step()

            # ---- Train G ----
            optG.zero_grad()
            gen = G(X, c)
            adv = D(X, gen, c)
            lossG = lam_gan * bce(adv, torch.ones_like(adv)) + lam_mse * mse(gen[:, close_idx], yclose)
            lossG.backward()
            if grad_clip > 0: nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            optG.step()

        # val RMSE (z)
        G.eval()
        with torch.no_grad():
            preds = []
            for X, y, c in va_loader:
                X, c = X.to(device), c.to(device)
                preds.append(G(X, c)[:, close_idx].cpu().numpy())
        y_pred_z = np.concatenate(preds)
        val_rmse_z = rmse(yva.numpy(), y_pred_z)
        print(f"epoch {epoch:02d} | val_RMSE_z={val_rmse_z:.4f}")

        if val_rmse_z < best_va:
            best_va = val_rmse_z
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "meta": meta, "cfg": cfg}, ckpt_path)

    print("saved best:", ckpt_path)

    # quick USD summary on test
    preds = []
    with torch.no_grad():
        for X, y, c in te_loader:
            X, c = X.to(device), c.to(device)
            preds.append(G(X, c)[:, close_idx].cpu().numpy())
    y_pred_z = np.concatenate(preds); y_true_z = yte.numpy()
    y_pred_usd = denorm_usd(y_pred_z, dates_te, cfg["data"]["interim"], symbol, ds_cfg["mar_window"])
    y_true_usd = denorm_usd(y_true_z, dates_te, cfg["data"]["interim"], symbol, ds_cfg["mar_window"])
    print("== TEST (USD) after cGAN-regime ==")
    print("MAE :", f"{mae(y_true_usd, y_pred_usd):.4f}")
    print("RMSE:", f"{rmse(y_true_usd, y_pred_usd):.4f}")
    print("MAPE:", f"{mape(y_true_usd, y_pred_usd):.4f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # configs/cgan_regime.yaml
    args = ap.parse_args()
    main(args.config)