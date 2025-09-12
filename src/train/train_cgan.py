import os, json, yaml, math, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.cgan import GeneratorLSTMCond, DiscriminatorMLPCond

# ---------- metrics ----------
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_npz(name, processed_dir):
    d = np.load(os.path.join(processed_dir, f"{name}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)           # [N,T,F]
    y = torch.tensor(d["y"][:,0,0], dtype=torch.float32)    # [N] (next-day close_z)
    return X, y

# ----- dates & conditions -----
def chrono_slices(n: int, tr: float, va: float):
    n_tr = int(n*tr); n_va = n_tr + int(n*va)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def target_dates(index, slc, T, H):
    sub = index[slc]
    return sub[T: len(sub)-H+1]

def dow_onehot(dates):
    # Monday=0 ... Friday=4 ; weekend rows shouldn't appear for market data
    d = np.array([dt.weekday() for dt in dates])
    C = 5
    out = np.zeros((len(d), C), dtype=np.float32)
    # map 0..4 into one-hot; ignore weekends if present
    for i, w in enumerate(d):
        if 0 <= w <= 4:
            out[i, w] = 1.0
    return out

def denorm_usd(y_z, dates, interim_dir, symbol, mar_w):
    import pandas as pd
    df = pd.read_csv(os.path.join(interim_dir, f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    close = df["close"].astype(float)
    mu = close.rolling(mar_w, min_periods=mar_w).mean()
    sd = close.rolling(mar_w, min_periods=mar_w).std().replace(0, 1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    return y_z * sd_t + mu_t

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))

    # meta
    meta = json.load(open(os.path.join(cfg["data"]["processed"], "meta.json")))
    T, H, F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol = meta["symbol"]; close_idx = int(meta["close_index"])

    # arrays
    Xtr, ytr = load_npz("train", cfg["data"]["processed"])
    Xva, yva = load_npz("val",   cfg["data"]["processed"])
    Xte, yte = load_npz("test",  cfg["data"]["processed"])

    # build condition vectors by reconstructing target dates per split
    import pandas as pd
    ds_cfg = yaml.safe_load(open("configs/dataset.yaml"))  # to get MAR window & splits
    df_all = pd.read_csv(os.path.join(cfg["data"]["interim"], f"{symbol}_mar.csv"),
                         parse_dates=["date"]).set_index("date")
    n = len(df_all)
    sl_tr, sl_va, sl_te = chrono_slices(n, ds_cfg["splits"]["train"], ds_cfg["splits"]["val"])
    dates_tr = target_dates(df_all.index, sl_tr, T, H)
    dates_va = target_dates(df_all.index, sl_va, T, H)
    dates_te = target_dates(df_all.index, sl_te, T, H)

    if not (len(dates_tr)==len(Xtr) and len(dates_va)==len(Xva) and len(dates_te)==len(Xte)):
        raise RuntimeError("Dates/arrays length mismatch; check windowing + splits.")

    C = int(cfg["train"]["cond_dim"])
    if cfg["train"]["cond_type"] == "dow":
        ctr = torch.tensor(dow_onehot(dates_tr), dtype=torch.float32)
        cva = torch.tensor(dow_onehot(dates_va), dtype=torch.float32)
        cte = torch.tensor(dow_onehot(dates_te), dtype=torch.float32)
    else:
        raise ValueError("Unsupported cond_type; start with 'dow'.")

    # loaders with condition
    tr_loader = DataLoader(TensorDataset(Xtr, ytr, ctr), batch_size=int(cfg["train"]["batch_size"]), shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva, cva), batch_size=512)
    te_loader = DataLoader(TensorDataset(Xte, yte, cte), batch_size=512)

    # models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = GeneratorLSTMCond(F, C, hidden=128, layers=1, dropout=0.2).to(device)
    D = DiscriminatorMLPCond(T, F, C, hidden=(72,100,10), dropout=0.2).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=float(cfg["train"]["lr_g"]))
    optD = torch.optim.Adam(D.parameters(), lr=float(cfg["train"]["lr_d"]))
    bce, mse = nn.BCELoss(), nn.MSELoss()
    lam_mse, lam_gan = float(cfg["train"]["lambda_mse"]), float(cfg["train"]["lambda_gan"])
    grad_clip = float(cfg["train"]["grad_clip"])

    best_va = float("inf")
    os.makedirs(cfg["logging"]["outdir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["logging"]["outdir"], "ckpt.pt")

    for epoch in range(1, int(cfg["train"]["epochs"])+1):
        G.train(); D.train()
        for X, yclose, c in tr_loader:
            X, yclose, c = X.to(device), yclose.to(device), c.to(device)

            # --- Train D ---
            for _ in range(int(cfg["train"]["d_steps"])):
                optD.zero_grad()
                real_next = torch.zeros(X.size(0), F, device=device)
                real_next[:, close_idx] = yclose
                fake_next = G(X, c).detach()
                r = D(X, real_next, c)
                f = D(X, fake_next, c)
                lossD = bce(r, torch.ones_like(r)) + bce(f, torch.zeros_like(f))
                lossD.backward(); optD.step()

            # --- Train G ---
            optG.zero_grad()
            gen = G(X, c)
            adv = D(X, gen, c)
            lossG = lam_gan * bce(adv, torch.ones_like(adv)) + lam_mse * mse(gen[:, close_idx], yclose)
            lossG.backward()
            if grad_clip > 0: nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            optG.step()

        # quick val (z-space)
        G.eval()
        with torch.no_grad():
            preds = []
            for X, y, c in va_loader:
                X, c = X.to(device), c.to(device)
                preds.append(G(X, c)[:, close_idx].cpu().numpy())
            y_pred_z = np.concatenate(preds)
            y_true_z = yva.numpy()
            val_rmse_z = rmse(y_true_z, y_pred_z)

        print(f"epoch {epoch:02d} | val_RMSE_z={val_rmse_z:.4f}")

        if val_rmse_z < best_va:
            best_va = val_rmse_z
            torch.save({"G": G.state_dict(),
                        "D": D.state_dict(),
                        "meta": meta,
                        "cfg": cfg}, ckpt_path)

    print("saved best:", ckpt_path)

    # ---- Test USD summary right after training (optional) ----
    preds = []
    with torch.no_grad():
        for X, y, c in te_loader:
            X, c = X.to(device), c.to(device)
            preds.append(G(X, c)[:, close_idx].cpu().numpy())
    y_pred_z = np.concatenate(preds)
    y_true_z = yte.numpy()

    dates_te = dates_te  # already built above
    y_pred_usd = denorm_usd(y_pred_z, dates_te, cfg["data"]["interim"], symbol, mar_w=int(ds_cfg["mar_window"]))
    y_true_usd = denorm_usd(y_true_z, dates_te, cfg["data"]["interim"], symbol, mar_w=int(ds_cfg["mar_window"]))

    print("== TEST (USD) after cGAN training ==")
    print("MAE :", f"{mae(y_true_usd, y_pred_usd):.4f}")
    print("RMSE:", f"{rmse(y_true_usd, y_pred_usd):.4f}")
    print("MAPE:", f"{mape(y_true_usd, y_pred_usd):.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # configs/cgan.yaml
    args = ap.parse_args()
    main(args.config)