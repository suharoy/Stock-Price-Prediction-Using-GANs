import os, json, yaml, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_reg import LSTMRegressor
from src.models.cgan import GeneratorLSTMCond

def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def chrono_slices(n: int, tr: float, va: float):
    n_tr = int(n * tr); n_va = n_tr + int(n * va)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def gen_target_dates(index: pd.DatetimeIndex, slc: slice, T: int, H: int):
    sub = index[slc]; start = T; end = len(sub) - H + 1
    if end <= start: return pd.DatetimeIndex([])
    return sub[start:end]

def dow_onehot(dates):
    out = np.zeros((len(dates), 5), dtype=np.float32)
    for i, dt in enumerate(dates):
        w = dt.weekday()
        if 0 <= w <= 4: out[i, w] = 1.0
    return out

def main(ds_cfg_path, lstm_ckpt, cgan_ckpt, split):
    ds_cfg = yaml.safe_load(open(ds_cfg_path))
    p = ds_cfg["paths"]
    meta = json.load(open(os.path.join(p["processed"], "meta.json")))
    T, H, F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol = meta["symbol"]; close_idx = int(meta["close_index"])

    d = np.load(os.path.join(p["processed"], f"{split}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)
    y_true_z = d["y"][:,0,0]

    df = pd.read_csv(os.path.join(p["interim"], f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    sl_tr, sl_va, sl_te = chrono_slices(len(df), ds_cfg["splits"]["train"], ds_cfg["splits"]["val"])
    sl_map = {"train": sl_tr, "val": sl_va, "test": sl_te}
    dates = gen_target_dates(df.index, sl_map[split], T, H)
    cond = torch.tensor(dow_onehot(dates), dtype=torch.float32)

    # LSTM
    ckpt_lstm = torch.load(lstm_ckpt, map_location="cpu")
    lstm = LSTMRegressor(F, **ckpt_lstm["cfg"]["model"]); lstm.load_state_dict(ckpt_lstm["model"]); lstm.eval()

    # cGAN
    ckpt_cgan = torch.load(cgan_ckpt, map_location="cpu")
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

    # denorm
    close = df["close"].astype(float)
    mu = close.rolling(ds_cfg["mar_window"], min_periods=ds_cfg["mar_window"]).mean()
    sd = close.rolling(ds_cfg["mar_window"], min_periods=ds_cfg["mar_window"]).std().replace(0,1e-8)
    mu_t = mu.reindex(dates).to_numpy(); sd_t = sd.reindex(dates).to_numpy()
    y_true_usd = y_true_z * sd_t + mu_t
    y_pred_usd = y_pred_z * sd_t + mu_t

    print("== Ensemble (LSTM + cGAN) |", split.upper(), "==")
    print("MAE :", f"{mae(y_true_usd, y_pred_usd):.4f}")
    print("RMSE:", f"{rmse(y_true_usd, y_pred_usd):.4f}")
    print("MAPE:", f"{mape(y_true_usd, y_pred_usd):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset.yaml")
    ap.add_argument("--lstm_ckpt", required=True)
    ap.add_argument("--cgan_ckpt", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    main(args.dataset, args.lstm_ckpt, args.cgan_ckpt, args.split)