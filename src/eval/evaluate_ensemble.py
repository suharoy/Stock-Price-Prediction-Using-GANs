import os, json, yaml, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.lstm_reg import LSTMRegressor
from src.models.cgan import GeneratorLSTMCond

def mae(y,t):  return float(np.mean(np.abs(y - t)))
def rmse(y,t): return float(np.sqrt(np.mean((y - t) ** 2)))
def mape(y,t,eps=1e-8):
    d = np.maximum(np.abs(t), eps)
    return float(np.mean(np.abs((y - t) / d)))

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset.yaml")
    ap.add_argument("--lstm_ckpt", required=True)
    ap.add_argument("--cgan_ckpt",  required=True)      # day-of-week cGAN ckpt
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    args = ap.parse_args()

    ds = yaml.safe_load(open(args.dataset))
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
    cond = torch.tensor(dow_onehot(dates), dtype=torch.float32)

    # LSTM
    ckpt_lstm = torch.load(args.lstm_ckpt, map_location="cpu")
    lstm = LSTMRegressor(F, **ckpt_lstm["cfg"]["model"]); lstm.load_state_dict(ckpt_lstm["model"]); lstm.eval()

    # cGAN (DOW)
    ckpt_cgan = torch.load(args.cgan_ckpt, map_location="cpu")
    cgan = GeneratorLSTMCond(F, 5); cgan.load_state_dict(ckpt_cgan["G"]); cgan.eval()

    preds_lstm, preds_cgan = [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = X[i:i+512]; cb = cond[i:i+512]
            preds_lstm.append(lstm(xb).cpu().numpy())
            preds_cgan.append(cgan(xb, cb)[:, close_idx].cpu().numpy())
    y_pred_lstm = np.concatenate(preds_lstm)
    y_pred_cgan = np.concatenate(preds_cgan)
    y_pred_z = 0.5 * y_pred_lstm + 0.5 * y_pred_cgan

    mu = df["close"].rolling(ds["mar_window"], min_periods=ds["mar_window"]).mean()
    sd = df["close"].rolling(ds["mar_window"], min_periods=ds["mar_window"]).std().replace(0,1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    y_true = y_true_z*sd_t + mu_t
    y_pred = y_pred_z*sd_t + mu_t

    print(f"== Ensemble (LSTM + cGAN DOW) | {args.split.upper()} (USD) ==")
    print(f"MAE  : {mae(y_pred, y_true):.4f}")
    print(f"RMSE : {rmse(y_pred, y_true):.4f}")
    print(f"MAPE : {mape(y_pred, y_true):.4f}")

if __name__ == "__main__":
    main()