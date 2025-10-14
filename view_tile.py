import sys, numpy as np
import matplotlib
# Use non-interactive backend so it works on servers without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_tile(path):
    tile = np.load(path)
    if tile.ndim != 3:
        raise ValueError(f"Expected 3D tile (H, W, Bands), got {tile.shape}")
    print(f"Loaded {path}  shape={tile.shape}")
    return tile

def to_rgb(tile, n_components=3):
    H, W, C = tile.shape
    flat = tile.reshape(-1, C)
    rgb_flat = PCA(n_components=n_components).fit_transform(flat)
    rgb = rgb_flat.reshape(H, W, n_components)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_tile.py <tile.npy>")
        sys.exit(1)
    path = sys.argv[1]
    tile = load_tile(path)
    rgb = to_rgb(tile)
    # Save instead of show (headless)
    out = path.replace(".npy", "_rgb.png")
    plt.figure(figsize=(4,4))
    plt.imshow(rgb); plt.axis("off"); plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"✅ Saved {out}")

if __name__ == "__main__":
    main()
