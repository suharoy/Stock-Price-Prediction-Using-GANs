import os, json, yaml, argparse, math, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.cgan import GeneratorLSTMCond

def mae(y,t):  return float(np.mean(np.abs(y - t)))
def rmse(y,t): return float(np.sqrt(np.mean((y - t) ** 2)))
def mape(y,t,eps=1e-8):
    d = np.maximum(np.abs(t), eps)
    return float(np.mean(np.abs((y - t) / d)))

def load_npz(split, pdir):
    d = np.load(os.path.join(pdir, f"{split}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"][:,0,0], dtype=torch.float32)
    return X, y

def chrono_slices(n,tr,val):
    n_tr = int(n*tr); n_va = n_tr + int(n*val)
    return slice(0,n_tr), slice(n_tr,n_va), slice(n_va,n)

def target_dates(idx, slc, T, H):
    sub = idx[slc]; return sub[T: len(sub)-H+1]

def dow_onehot(dates):
    out = np.zeros((len(dates),5), dtype=np.float32)
    for i,dt in enumerate(dates):
        w = dt.weekday()
        if 0<=w<=4: out[i,w]=1.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True)      # runs/cgan_stable_msft/ckpt.pt
    ap.add_argument("--config", required=True)      # configs/dataset.yaml
    ap.add_argument("--split",  choices=["train","val","test"], default="test")
    args = ap.parse_args()

    ds = yaml.safe_load(open(args.config))
    p  = ds["paths"]
    meta = json.load(open(os.path.join(p["processed"], "meta.json")))
    T,H,F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol  = meta["symbol"]; close_idx = int(meta["close_index"])

    X, y_true_z = load_npz(args.split, p["processed"])

    df = pd.read_csv(os.path.join(p["interim"], f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    n = len(df)
    sl_tr, sl_va, sl_te = chrono_slices(n, ds["splits"]["train"], ds["splits"]["val"])
    sl_map = {"train": sl_tr, "val": sl_va, "test": sl_te}
    dates = target_dates(df.index, sl_map[args.split], T, H)
    cond_np = dow_onehot(dates); C = cond_np.shape[1]

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfgm = ckpt.get("cfg",{}).get("model", {})
    G = GeneratorLSTMCond(F, C,
                          hidden=int(cfgm.get("hidden",128)),
                          layers=int(cfgm.get("layers",1)),
                          dropout=float(cfgm.get("dropout",0.2)))
    G.load_state_dict(ckpt["G"]); G.eval()

    cond = torch.tensor(cond_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y_true_z, cond), batch_size=512, shuffle=False)
    preds=[]
    with torch.no_grad():
        for xb,_,cb in loader:
            preds.append(G(xb, cb)[:, close_idx].cpu().numpy())
    y_pred_z = np.concatenate(preds)

    close = df["close"].astype(float)
    mu = close.rolling(ds["mar_window"], min_periods=ds["mar_window"]).mean()
    sd = close.rolling(ds["mar_window"], min_periods=ds["mar_window"]).std().replace(0,1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    y_true = y_true_z.numpy()*sd_t + mu_t
    y_pred = y_pred_z*sd_t + mu_t

    print(f"== cGAN-Stable | {args.split.upper()} (USD) | Symbol={symbol} ==")
    print(f"MAE  : {mae(y_pred, y_true):.4f}")
    print(f"RMSE : {rmse(y_pred, y_true):.4f}")
    print(f"MAPE : {mape(y_pred, y_true):.4f}")
    print(f"(T={T} H={H} F={F} | MAR={ds['mar_window']} | cond=DOW, C=5)")

if __name__ == "__main__":
    main()