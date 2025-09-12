import argparse, json, os, yaml, numpy as np, pandas as pd

def make_windows(arr: np.ndarray, T: int, H: int, close_idx: int):
    X, y = [], []
    for i in range(T, len(arr) - H + 1):
        X.append(arr[i-T:i, :])                 # past T days
        y.append(arr[i:i+H, close_idx])         # next-day close_z
    X = np.stack(X)
    y = np.array(y).reshape(-1, H, 1)
    return X, y

def chrono_splits(n: int, tr: float, va: float):
    n_tr = int(n * tr); n_va = n_tr + int(n * va)
    return slice(0, n_tr), slice(n_tr, n_va), slice(n_va, n)

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    p = cfg["paths"]
    feats = [f"{c}_z" for c in cfg["features"]]   # normalized features
    sym = cfg["symbols"][0]                      # first symbol only

    df = pd.read_csv(os.path.join(p["interim"], f"{sym}_mar.csv"),
                     parse_dates=["date"]).set_index("date")
    df = df[feats].copy()

    # index of close_z
    try:
        close_idx = feats.index("close_z")
    except ValueError:
        raise RuntimeError("close_z not found in features. Check configs/dataset.yaml")

    # chronological split
    tr_s, va_s, te_s = chrono_splits(len(df), cfg["splits"]["train"], cfg["splits"]["val"])
    tr_df, va_df, te_df = df.iloc[tr_s], df.iloc[va_s], df.iloc[te_s]

    # build windows
    T, H = cfg["window_T"], cfg["horizon"]
    Xtr, ytr = make_windows(tr_df.values, T, H, close_idx)
    Xva, yva = make_windows(va_df.values, T, H, close_idx)
    Xte, yte = make_windows(te_df.values, T, H, close_idx)

    os.makedirs(p["processed"], exist_ok=True)
    np.savez_compressed(os.path.join(p["processed"], "train.npz"), X=Xtr, y=ytr)
    np.savez_compressed(os.path.join(p["processed"], "val.npz"),   X=Xva, y=yva)
    np.savez_compressed(os.path.join(p["processed"], "test.npz"),  X=Xte, y=yte)
    meta = dict(symbol=sym, T=T, H=H, F=len(feats), features=feats, close_index=close_idx)
    json.dump(meta, open(os.path.join(p["processed"], "meta.json"), "w"), indent=2)
    print("saved:", p["processed"],
          "| shapes:", Xtr.shape, Xva.shape, Xte.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)