import os, json, yaml, math, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.gans_classical import GeneratorLSTM, DiscriminatorMLP

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

def denorm_usd(y_z, dates, interim_dir, symbol, mar_w):
    """Convert z-scores back to USD using MAR rolling mean/std from interim CSV."""
    import pandas as pd
    df = pd.read_csv(os.path.join(interim_dir, f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    close = df["close"].astype(float)
    mu = close.rolling(mar_w, min_periods=mar_w).mean()
    sd = close.rolling(mar_w, min_periods=mar_w).std().replace(0, 1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    return y_z * sd_t + mu_t

def target_dates_for_split(interim_dir, symbol, T, H, splits, which):
    import pandas as pd
    df = pd.read_csv(os.path.join(interim_dir, f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    n = len(df); tr = int(n*splits["train"]); va = tr + int(n*splits["val"])
    slc = {"train": slice(0,tr), "val": slice(tr,va), "test": slice(va,n)}[which]
    sub = df.index[slc]
    return sub[T: len(sub)-H+1]

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))

    meta = json.load(open(os.path.join(cfg["data"]["processed"], "meta.json")))
    T, H, F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol = meta["symbol"]; close_idx = int(meta["close_index"])

    Xtr, ytr = load_npz("train", cfg["data"]["processed"])
    Xva, yva = load_npz("val",   cfg["data"]["processed"])
    Xte, yte = load_npz("test",  cfg["data"]["processed"])

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg["train"]["batch_size"], shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=512)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = GeneratorLSTM(F, hidden=128, layers=1, dropout=0.2).to(device)
    D = DiscriminatorMLP(T, F, hidden=(72,100,10), dropout=0.2).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=float(cfg["train"]["lr_g"]))
    optD = torch.optim.Adam(D.parameters(), lr=float(cfg["train"]["lr_d"]))
    bce, mse = nn.BCELoss(), nn.MSELoss()
    lam_mse, lam_gan = float(cfg["train"]["lambda_mse"]), float(cfg["train"]["lambda_gan"])
    grad_clip = float(cfg["train"]["grad_clip"])

    # training loop
    best_va = float("inf")
    os.makedirs(cfg["logging"]["outdir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["logging"]["outdir"], "ckpt.pt")

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        G.train(); D.train()
        for X, yclose in tr_loader:
            X = X.to(device); yclose = yclose.to(device)

            # ----- Train D -----
            for _ in range(int(cfg["train"]["d_steps"])):
                optD.zero_grad()
                # real next-day vector: only close dimension filled; others zeros
                real_next = torch.zeros(X.size(0), F, device=device)
                real_next[:, close_idx] = yclose
                # fake
                fake_next = G(X).detach()
                # scores
                r = D(X, real_next)
                f = D(X, fake_next)
                lossD = bce(r, torch.ones_like(r)) + bce(f, torch.zeros_like(f))
                lossD.backward()
                optD.step()

            # ----- Train G -----
            optG.zero_grad()
            gen = G(X)
            adv = D(X, gen)
            # adversarial + supervised (close MSE)
            lossG = lam_gan * bce(adv, torch.ones_like(adv)) + lam_mse * mse(gen[:, close_idx], yclose)
            lossG.backward()
            if grad_clip > 0: nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            optG.step()

        # ---- quick val RMSE in z-space ----
        G.eval()
        with torch.no_grad():
            preds = []
            for X, y in va_loader:
                X = X.to(device)
                preds.append(G(X)[:, close_idx].cpu().numpy())
            y_pred_z = np.concatenate(preds)
            y_true_z = np.load(os.path.join(cfg["data"]["processed"], "val.npz"))["y"][:,0,0]
            val_rmse_z = rmse(y_true_z, y_pred_z)

        print(f"epoch {epoch:02d} | val_RMSE_z={val_rmse_z:.4f}")

        if val_rmse_z < best_va:
            best_va = val_rmse_z
            torch.save({"G": G.state_dict(),
                        "D": D.state_dict(),
                        "meta": meta,
                        "cfg": cfg}, ckpt_path)

    print("saved best:", ckpt_path)

    # --------- OPTIONAL: USD metrics on test split right after training ----------
    # get predictions on test in z-space
    G.eval()
    preds = []
    with torch.no_grad():
        for X, _ in te_loader:
            X = X.to(device)
            preds.append(G(X)[:, close_idx].cpu().numpy())
    y_pred_z = np.concatenate(preds)
    y_true_z = np.load(os.path.join(cfg["data"]["processed"], "test.npz"))["y"][:,0,0]

    # dates for test targets + denorm
    dates = target_dates_for_split(cfg["data"]["interim"], symbol, T, H, splits=yaml.safe_load(open("configs/dataset.yaml"))["splits"], which="test")
    y_pred_usd = denorm_usd(y_pred_z, dates, cfg["data"]["interim"], symbol, mar_w=yaml.safe_load(open("configs/dataset.yaml"))["mar_window"])
    y_true_usd = denorm_usd(y_true_z, dates, cfg["data"]["interim"], symbol, mar_w=yaml.safe_load(open("configs/dataset.yaml"))["mar_window"])

    print("== TEST (USD) after training ==")
    print("MAE :", f"{mae(y_true_usd, y_pred_usd):.4f}")
    print("RMSE:", f"{rmse(y_true_usd, y_pred_usd):.4f}")
    print("MAPE:", f"{mape(y_true_usd, y_pred_usd):.4f}")
    

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # configs/gan_2019.yaml
    args = ap.parse_args()
    main(args.config)