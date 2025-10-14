from pathlib import Path
import numpy as np

OUTDIR = Path("data_processed")
OUTDIR.mkdir(exist_ok=True, parents=True)

def minmax_norm(x, eps=1e-6):
    x = x.copy()
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + eps)

def make_coords(H, W, tile=16, stride=8):
    coords = [(i, j) for i in range(0, H - tile + 1, stride)
                      for j in range(0, W - tile + 1, stride)]
    train = [(i, j) for (i, j) in coords if ((i//stride)+(j//stride)) % 2 == 0]
    val   = [(i, j) for (i, j) in coords if ((i//stride)+(j//stride)) % 2 == 1]
    return train, val

def majority_label(tile_gt):
    m = tile_gt > 0
    if not m.any(): 
        return 0
    vals, cnts = np.unique(tile_gt[m], return_counts=True)
    return int(vals[np.argmax(cnts)])

def extract_tiles(X, Y, coords, tile=16):
    X_tiles, y_labs = [], []
    for (i, j) in coords:
        xt = X[i:i+tile, j:j+tile, :]
        yt = Y[i:i+tile, j:j+tile]
        m = yt > 0
        if not m.any():
            continue
        vals, cnts = np.unique(yt[m], return_counts=True)
        X_tiles.append(xt)
        y_labs.append(int(vals[np.argmax(cnts)]))
    return np.asarray(X_tiles), np.asarray(y_labs)

def main():
    print("Demo only — replace with actual Indian Pines data if available.")
    H, W, C = 145, 145, 200
    X = np.random.rand(H, W, C).astype(np.float32)
    Y = np.zeros((H, W), dtype=np.int32)
    Y[30:60, 40:70] = 11
    Y[80:120, 20:60] = 14

    X = minmax_norm(X)
    tr_coords, va_coords = make_coords(H, W, 16, 8)
    Xtr, Ytr = extract_tiles(X, Y, tr_coords)
    Xva, Yva = extract_tiles(X, Y, va_coords)

    np.save(OUTDIR / "X_train_preview.npy", Xtr[:32])
    np.save(OUTDIR / "y_train_preview.npy", Ytr[:32])
    np.save(OUTDIR / "X_val_preview.npy",   Xva[:32])
    np.save(OUTDIR / "y_val_preview.npy",   Yva[:32])

    print("✅ Demo preprocessing complete. Files saved in", OUTDIR)

if __name__ == "__main__":
    main()
