import os, json, yaml, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_reg import LSTMRegressor
from src.models.cgan import GeneratorLSTMCond

# ---------- metrics ----------
def mae(y,t):  return float(np.mean(np.abs(y - t)))
def rmse(y,t): return float(np.sqrt(np.mean((y - t) ** 2)))
def mape(y,t,eps=1e-8):
    d = np.maximum(np.abs(t), eps)
    return float(np.mean(np.abs((y - t) / d)))

# ---------- helpers ----------
def load_npz(split, pdir):
    d = np.load(os.path.join(pdir, f"{split}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = d["y"][:,0,0]
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

def parse_list(arg):
    if arg is None or arg.strip()=="":
        return []
    return [s.strip() for s in arg.split(",") if s.strip()]

def parse_weights(n, weights_arg):
    if not weights_arg:
        return np.ones(n, dtype=np.float32) / n
    w = np.array([float(x) for x in weights_arg.split(",")], dtype=np.float32)
    assert len(w)==n, f"weights length {len(w)} must equal number of models {n}"
    s = float(np.sum(w))
    if s == 0:  # avoid divide by zero
        w = np.ones(n, dtype=np.float32)
        s = float(np.sum(w))
    return w / s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset.yaml")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    # comma-separated lists
    ap.add_argument("--lstm_ckpts", default="", help="comma-separated LSTM ckpt paths")
    ap.add_argument("--cgan_ckpts", default="", help="comma-separated cGAN (DOW) ckpt paths")
    ap.add_argument("--weights", default="", help="optional comma-separated weights (same length as total models)")
    ap.add_argument("--out_csv", default="", help="optional CSV path to append results")
    args = ap.parse_args()

    ds = yaml.safe_load(open(args.dataset))
    p  = ds["paths"]
    meta = json.load(open(os.path.join(p["processed"], "meta.json")))
    T,H,F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol  = meta["symbol"]; close_idx = int(meta["close_index"])

    # data
    X, y_true_z = load_npz(args.split, p["processed"])

    # dates + DOW condition (for cGANs)
    df = pd.read_csv(os.path.join(p["interim"], f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    n = len(df)
    sl_tr, sl_va, sl_te = chrono_slices(n, ds["splits"]["train"], ds["splits"]["val"])
    sl_map = {"train": sl_tr, "val": sl_va, "test": sl_te}
    dates = target_dates(df.index, sl_map[args.split], T, H)
    cond = torch.tensor(dow_onehot(dates), dtype=torch.float32)

    # parse ckpt lists and weights
    lstm_paths = parse_list(args.lstm_ckpts)
    cgan_paths = parse_list(args.cgan_ckpts)
    all_paths  = lstm_paths + cgan_paths
    assert len(all_paths) > 0, "Provide at least one checkpoint via --lstm_ckpts or --cgan_ckpts"
    weights = parse_weights(len(all_paths), args.weights)

    # load models
    models_lstm = []
    for pth in lstm_paths:
        ckpt = torch.load(pth, map_location="cpu")
        mdl = LSTMRegressor(F, **ckpt["cfg"]["model"])
        mdl.load_state_dict(ckpt["model"]); mdl.eval()
        models_lstm.append(mdl)

    models_cgan = []
    for pth in cgan_paths:
        ckpt = torch.load(pth, map_location="cpu")
        # assume DOW cGAN (cond_dim=5)
        mdl = GeneratorLSTMCond(F, 5)
        mdl.load_state_dict(ckpt["G"]); mdl.eval()
        models_cgan.append(mdl)

    # predictions (z-space)
    preds_each = []
    with torch.no_grad():
        # LSTM preds
        for mdl in models_lstm:
            chunks=[]
            for i in range(0, len(X), 512):
                chunks.append(mdl(X[i:i+512]).cpu().numpy())
            preds_each.append(np.concatenate(chunks))
        # cGAN preds (need cond)
        for mdl in models_cgan:
            chunks=[]
            for i in range(0, len(X), 512):
                xb = X[i:i+512]; cb = cond[i:i+512]
                chunks.append(mdl(xb, cb)[:, close_idx].cpu().numpy())
            preds_each.append(np.concatenate(chunks))

    preds_each = np.stack(preds_each, axis=0)   # [M, N]
    y_pred_z = np.average(preds_each, axis=0, weights=weights)  # [N]

    # denormalize to USD
    close = df["close"].astype(float)
    mu = close.rolling(ds["mar_window"], min_periods=ds["mar_window"]).mean()
    sd = close.rolling(ds["mar_window"], min_periods=ds["mar_window"]).std().replace(0,1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    y_true = y_true_z*sd_t + mu_t
    y_pred = y_pred_z*sd_t + mu_t

    # metrics
    m_mae  = mae(y_pred, y_true)
    m_rmse = rmse(y_pred, y_true)
    m_mape = mape(y_pred, y_true)

    # report
    label = []
    if lstm_paths: label.append(f"LSTMx{len(lstm_paths)}")
    if cgan_paths: label.append(f"cGANx{len(cgan_paths)}")
    label = " + ".join(label)
    print(f"== Ensemble++ ({label}) | {args.split.upper()} (USD) ==")
    print(f"MAE  : {m_mae:.4f}")
    print(f"RMSE : {m_rmse:.4f}")
    print(f"MAPE : {m_mape:.4f}")
    print(f"Weights: {weights.tolist()}")

    # optional CSV append
    if args.out_csv:
        import csv, datetime as dt
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        header = ["timestamp","split","label","mae_usd","rmse_usd","mape_pct","paths","weights"]
        row = [
            dt.datetime.utcnow().isoformat(),
            args.split, label, f"{m_mae:.6f}", f"{m_rmse:.6f}", f"{m_mape:.6f}",
            "|".join(all_paths), "|".join([str(w) for w in weights.tolist()])
        ]
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header: w.writerow(header)
            w.writerow(row)
        print(f"[saved] appended to {args.out_csv}")
        
if __name__ == "__main__":
    main()