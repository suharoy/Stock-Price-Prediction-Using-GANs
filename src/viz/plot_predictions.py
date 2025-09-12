import os, json, yaml, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---- Models (try both names for plain GAN) ----
from src.models.lstm_reg import LSTMRegressor
try:
    from src.models.gans_classical import GeneratorLSTM as GeneratorPlain
except Exception:
    try:
        from src.models.gan_classical import GeneratorLSTM as GeneratorPlain
    except Exception:
        GeneratorPlain = None

from src.models.cgan import GeneratorLSTMCond  # your conditional GAN

# ---------- helpers ----------
def load_npz(split, pdir):
    d = np.load(os.path.join(pdir, f"{split}.npz"))
    X = torch.tensor(d["X"], dtype=torch.float32)             # [N,T,F]
    y_z = d["y"][:,0,0]                                       # [N] next-step z
    return X, y_z

def chrono_slices(n,tr,val):
    n_tr = int(n*tr); n_va = n_tr + int(n*val)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def target_dates(idx, slc, T, H):
    sub = idx[slc]
    return sub[T: len(sub)-H+1]

def dow_onehot(dates):
    out = np.zeros((len(dates), 5), dtype=np.float32)
    for i, dt in enumerate(dates):
        w = dt.weekday()
        if 0 <= w <= 4: out[i, w] = 1.0
    return out

def denorm_to_usd(y_z, mu_t, sd_t):
    return y_z * sd_t + mu_t

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def moving_window_indices(total_len, window):
    if window <= 0 or window > total_len:
        return 0, total_len
    return total_len - window, total_len

# ---------- prediction loaders ----------
def predict_lstm(ckpt_path, F, X):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mdl = LSTMRegressor(F, **ckpt["cfg"]["model"])
    mdl.load_state_dict(ckpt["model"]); mdl.eval()
    preds=[]
    with torch.no_grad():
        for i in range(0, len(X), 512):
            preds.append(mdl(X[i:i+512]).cpu().numpy())
    return np.concatenate(preds)

def predict_gan_plain(ckpt_path, F, X, close_idx):
    if GeneratorPlain is None:
        raise RuntimeError("Plain GAN Generator class not found (gan2019.py or gan_classical.py).")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # some checkpoints store cfg inside; we don’t rely on it here
    mdl = GeneratorPlain(F, hidden=128, layers=1, dropout=0.2)
    mdl.load_state_dict(ckpt["G"]); mdl.eval()
    preds=[]
    with torch.no_grad():
        for i in range(0, len(X), 512):
            preds.append(mdl(X[i:i+512])[:, close_idx].cpu().numpy())
    return np.concatenate(preds)

def predict_cgan(ckpt_path, F, X, cond_np, close_idx, cond_dim=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # try to read model dim, else fall back to cond_np shape
    cfgm = ckpt.get("cfg",{}).get("model", {})
    C = int(cfgm.get("cond_dim", cond_dim if cond_dim else cond_np.shape[1]))
    mdl = GeneratorLSTMCond(F, C,
                            hidden=int(cfgm.get("hidden",128)),
                            layers=int(cfgm.get("layers",1)),
                            dropout=float(cfgm.get("dropout",0.2)))
    mdl.load_state_dict(ckpt["G"]); mdl.eval()
    cond = torch.tensor(cond_np, dtype=torch.float32)
    preds=[]
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = X[i:i+512]; cb = cond[i:i+512]
            preds.append(mdl(xb, cb)[:, close_idx].cpu().numpy())
    return np.concatenate(preds)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset.yaml")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--window", type=int, default=120, help="last N targets to plot")
    ap.add_argument("--out_dir", default="reports/figures")

    # optional checkpoints
    ap.add_argument("--lstm_ckpt", default="")
    ap.add_argument("--gan_ckpt", default="")
    ap.add_argument("--cgan_ckpt", default="")
    ap.add_argument("--cgan_stable_ckpt", default="")
    ap.add_argument("--cgan_regime_ckpt", default="")

    # ensembles via names: e.g. "lstm,cgan" or "lstm,cgan,cgan_stable"
    ap.add_argument("--ensemble", default="", help="Names to average: comma from {lstm, gan, cgan, cgan_stable, cgan_regime}")
    ap.add_argument("--ensemblepp", default="", help="Another averaged set, e.g. 'lstm,cgan,cgan_stable'")

    args = ap.parse_args()

    # load dataset/meta
    ds = yaml.safe_load(open(args.dataset))
    p  = ds["paths"]
    meta = json.load(open(os.path.join(p["processed"], "meta.json")))
    T,H,F = int(meta["T"]), int(meta["H"]), int(meta["F"])
    symbol  = meta["symbol"]; close_idx = int(meta["close_index"])
    ensure_dirs(args.out_dir, "reports/results")

    # arrays + dates
    X, y_true_z = load_npz(args.split, p["processed"])
    df = pd.read_csv(os.path.join(p["interim"], f"{symbol}_mar.csv"), parse_dates=["date"]).set_index("date")
    n = len(df)
    sl_tr, sl_va, sl_te = chrono_slices(n, ds["splits"]["train"], ds["splits"]["val"])
    sl_map = {"train": sl_tr, "val": sl_va, "test": sl_te}
    dates = target_dates(df.index, sl_map[args.split], T, H)

    # denorm stats aligned to target dates
    close = df["close"].astype(float)
    mu = close.rolling(ds["mar_window"], min_periods=ds["mar_window"]).mean()
    sd = close.rolling(ds["mar_window"], min_periods=ds["mar_window"]).std().replace(0,1e-8)
    mu_t = mu.reindex(dates).to_numpy()
    sd_t = sd.reindex(dates).to_numpy()

    # conditions for cGANs (day-of-week for cgan & cgan_stable; regime cGAN was trained separately but evaluator just needs preds)
    cond_dow = dow_onehot(dates)

    # collect predictions
    model_preds = {}   # name -> np.ndarray (USD)
    z_preds     = {}   # store z for ensembles

    # LSTM
    if args.lstm_ckpt:
        z = predict_lstm(args.lstm_ckpt, F, X)
        z_preds["lstm"] = z
        model_preds["LSTM"] = denorm_to_usd(z, mu_t, sd_t)

    # GAN (plain)
    if args.gan_ckpt:
        z = predict_gan_plain(args.gan_ckpt, F, X, close_idx)
        z_preds["gan"] = z
        model_preds["GAN"] = denorm_to_usd(z, mu_t, sd_t)

    # cGAN (dow)
    if args.cgan_ckpt:
        z = predict_cgan(args.cgan_ckpt, F, X, cond_dow, close_idx)
        z_preds["cgan"] = z
        model_preds["cGAN (DOW)"] = denorm_to_usd(z, mu_t, sd_t)

    # cGAN stable (dow)
    if args.cgan_stable_ckpt:
        z = predict_cgan(args.cgan_stable_ckpt, F, X, cond_dow, close_idx)
        z_preds["cgan_stable"] = z
        model_preds["cGAN Stable (DOW)"] = denorm_to_usd(z, mu_t, sd_t)

    # cGAN regime (NOTE: trained with regime cond; for plotting we only need predictions)
    if args.cgan_regime_ckpt:
        # We didn’t store regime cond again here; use DOW or zeros—it doesn’t affect already trained weights if network expects inputs
        # Safer: build zeros with same C=3, but we don’t know C from ckpt—skip and reuse DOW (works if cond_dim==5).
        try:
            z = predict_cgan(args.cgan_regime_ckpt, F, X, cond_dow, close_idx)  # if cond_dim mismatch, the ckpt had cond_dim in cfg
        except Exception as e:
            # fall back: try cond_dim=3 using zero one-hot
            cond_reg = np.zeros((len(X), 3), dtype=np.float32)
            z = predict_cgan(args.cgan_regime_ckpt, F, X, cond_reg, close_idx, cond_dim=3)
        z_preds["cgan_regime"] = z
        model_preds["cGAN (Regime)"] = denorm_to_usd(z, mu_t, sd_t)

    # Ensembles
    def build_ensemble(name_list, label):
        selected = [z_preds[k] for k in name_list if k in z_preds]
        if not selected:
            return
        z = np.mean(np.stack(selected, axis=0), axis=0)
        model_preds[label] = denorm_to_usd(z, mu_t, sd_t)

    if args.ensemble:
        names = [s.strip() for s in args.ensemble.split(",") if s.strip()]
        build_ensemble(names, f"Ensemble({'+'.join(names)})")
    if args.ensemblepp:
        names = [s.strip() for s in args.ensemblepp.split(",") if s.strip()]
        build_ensemble(names, f"Ensemble++({'+'.join(names)})")

    # Build DataFrame for CSV (windowed)
    y_true_usd = denorm_to_usd(y_true_z, mu_t, sd_t)
    df_out = pd.DataFrame({"date": dates, "actual": y_true_usd})
    for k,v in model_preds.items():
        df_out[k] = v

    # Window selection (last N)
    start, end = moving_window_indices(len(df_out), args.window)
    df_win = df_out.iloc[start:end].copy()

    # Save CSV
    csv_path = os.path.join("reports", "results", f"window_predictions_{args.split}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_win.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}  (rows: {len(df_win)})")

    # --------- Plot 1: Multi-subplot comparison ---------
    # Rows: 1 (overlay) + each model small panel
    models_order = ["LSTM","GAN","cGAN (DOW)","cGAN (Regime)","cGAN Stable (DOW)"] + \
                   [m for m in model_preds.keys() if m.startswith("Ensemble")]
    # Keep only those present
    models_order = [m for m in models_order if m in model_preds]
    n_small = len(models_order)
    n_rows = 1 + n_small
    fig_h = 2.0*n_rows + 1.0
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, fig_h), sharex=True)

    # Top overlay
    ax0 = axes[0]
    ax0.plot(df_win["date"], df_win["actual"], label="Actual")
    for name in models_order:
        ax0.plot(df_win["date"], df_win[name], label=name)
    ax0.set_title(f"MSFT — Actual vs Predicted (window={args.window}, split={args.split})")
    ax0.set_ylabel("Close (USD)")
    ax0.legend(ncol=3, fontsize=8)

    # Small panels
    for i, name in enumerate(models_order, start=1):
        ax = axes[i]
        ax.plot(df_win["date"], df_win["actual"], label="Actual")
        ax.plot(df_win["date"], df_win[name], label=name)
        ax.set_ylabel("USD")
        ax.set_title(name)
        if i == n_rows-1:
            ax.set_xlabel("Date")

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, f"compare_curves_{args.split}.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_png}")

    # --------- Plot 2: Error histograms per model ---------
    fig2, axes2 = plt.subplots(len(models_order), 1, figsize=(12, 2.2*len(models_order)), sharex=True)
    if len(models_order) == 1:
        axes_list = [axes2]
    else:
        axes_list = list(axes2)

    err_true = df_out["actual"].to_numpy()
    for ax, name in zip(axes_list, models_order):
        e = (df_out[name] - err_true)[start:end]  # windowed errors
        ax.hist(e, bins=30)
        ax.set_title(f"Error Histogram: {name} (Pred - Actual)")
        ax.set_ylabel("Count")
    axes_list[-1].set_xlabel("USD error")

    fig2.tight_layout()
    out_png2 = os.path.join(args.out_dir, f"errors_hist_{args.split}.png")
    fig2.savefig(out_png2, dpi=180)
    plt.close(fig2)
    print(f"[saved] {out_png2}")

    print("Done.")

if __name__ == "__main__":
    main()