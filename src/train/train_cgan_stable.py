import os, json, yaml, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.cgan import GeneratorLSTMCond, DiscriminatorMLPCond

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def load_npz(name, processed_dir):
    d = np.load(os.path.join(processed_dir, f"{name}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"][:,0,0], dtype=torch.float32)
    return X, y

def chrono_slices(n: int, tr: float, va: float):
    n_tr = int(n*tr); n_va = n_tr + int(n*va)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def target_dates(index, slc, T, H):
    sub = index[slc]
    return sub[T: len(sub)-H+1]

def dow_onehot(dates):
    out = np.zeros((len(dates), 5), dtype=np.float32)
    for i, dt in enumerate(dates):
        w = dt.weekday()
        if 0 <= w <= 4: out[i, w] = 1.0
    return out

def denorm_usd(y_z, dates, interim_dir, symbol, mar_w):
    df = pd.read_csv(os.path.join(interim_dir, f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    close = df["close"].astype(float)
    mu = close.rolling(mar_w, min_periods=mar_w).mean()
    sd = close.rolling(mar_w, min_periods=mar_w).std().replace(0, 1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    return y_z * sd_t + mu_t

# ----- stabilization tricks -----
def smooth(label, eps=0.1):
    return label * (1 - eps) + 0.5 * eps

def add_noise(x, sigma=0.05):
    return x + sigma * torch.randn_like(x)

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))

    meta = json.load(open(os.path.join(cfg["data"]["processed"], "meta.json")))
    T, H, F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol = meta["symbol"]; close_idx = int(meta["close_index"])

    Xtr, ytr = load_npz("train", cfg["data"]["processed"])
    Xva, yva = load_npz("val",   cfg["data"]["processed"])
    Xte, yte = load_npz("test",  cfg["data"]["processed"])

    ds_cfg = yaml.safe_load(open("configs/dataset.yaml"))
    df_all = pd.read_csv(os.path.join(cfg["data"]["interim"], f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    n = len(df_all)
    sl_tr, sl_va, sl_te = chrono_slices(n, ds_cfg["splits"]["train"], ds_cfg["splits"]["val"])
    dates_tr = target_dates(df_all.index, sl_tr, T, H)
    dates_va = target_dates(df_all.index, sl_va, T, H)
    dates_te = target_dates(df_all.index, sl_te, T, H)

    ctr = torch.tensor(dow_onehot(dates_tr), dtype=torch.float32)
    cva = torch.tensor(dow_onehot(dates_va), dtype=torch.float32)
    cte = torch.tensor(dow_onehot(dates_te), dtype=torch.float32)
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

            # ---- Train D (with noise + label smoothing) ----
            for _ in range(d_steps):
                optD.zero_grad()
                real_next = torch.zeros(X.size(0), F, device=device)
                real_next[:, close_idx] = yclose
                fake_next = G(X, c).detach()

                r = D(add_noise(X), add_noise(real_next), c)
                f = D(add_noise(X), add_noise(fake_next), c)
                real_labels = smooth(torch.ones_like(r), 0.1)
                fake_labels = smooth(torch.zeros_like(f), 0.1)
                lossD = bce(r, real_labels) + bce(f, fake_labels)
                lossD.backward(); optD.step()

            # ---- Train G (with smoothed real labels) ----
            optG.zero_grad()
            gen = G(X, c)
            adv = D(X, gen, c)
            real_labels = smooth(torch.ones_like(adv), 0.1)
            lossG = lam_gan * bce(adv, real_labels) + lam_mse * mse(gen[:, close_idx], yclose)
            lossG.backward()
            if grad_clip > 0: nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            optG.step()

        # val (z)
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

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # configs/cgan_stable.yaml
    args = ap.parse_args()
    main(args.config)