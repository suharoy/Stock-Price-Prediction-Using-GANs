import os, json, yaml, math, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.lstm_reg import LSTMRegressor

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_split(name, processed_dir):
    d = np.load(os.path.join(processed_dir, f"{name}.npz"))
    # X:[N,T,F], y:[N,1,1] -> use y[:,0,0] as next-day close_z
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"][:,0,0], dtype=torch.float32)
    return X, y

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))

    meta = json.load(open(os.path.join(cfg["data"]["processed"], "meta.json")))
    T, F = meta["T"], meta["F"]
    close_index = meta["close_index"]  # for info

    Xtr, ytr = load_split("train", cfg["data"]["processed"])
    Xva, yva = load_split("val",   cfg["data"]["processed"])
    Xte, yte = load_split("test",  cfg["data"]["processed"])

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=512, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(F, **cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = nn.MSELoss()

    best_va = float("inf")
    os.makedirs(cfg["logging"]["outdir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["logging"]["outdir"], "ckpt.pt")

    def evaluate(loader):
        model.eval()
        loss, n = 0.0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                l = loss_fn(pred, y)
                loss += l.item() * len(y)
                n += len(y)
        rmse = math.sqrt(loss / n)
        return rmse

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

        va_rmse = evaluate(va_loader)
        te_rmse = evaluate(te_loader)
        print(f"epoch {epoch:02d} | val_RMSE_z={va_rmse:.4f} | test_RMSE_z={te_rmse:.4f}")

        if va_rmse < best_va:
            best_va = va_rmse
            torch.save({"model": model.state_dict(), "meta": meta, "cfg": cfg}, ckpt_path)

    print("saved best:", ckpt_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)