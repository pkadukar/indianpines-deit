# Indian Pines — DeiT baseline (tile-level classification)

This repo contains a minimal pipeline + demo inference for classifying tiles from the **Indian Pines** hyperspectral scene using a **DeiT (facebook/deit-base-distilled-patch16-224)** backbone with a 13-class head.

## What’s included
- `demo_inference.py`: loads a trained checkpoint and runs a forward pass.
- `CHECKPOINT_PATH.txt`: path to the trained checkpoint on Sol.
- `.gitignore`: excludes large training artifacts.

## How to run the demo
```bash
conda activate furi310
python demo_inference.py
