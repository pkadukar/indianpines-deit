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
```bash
conda activate furi310
python demo_inference.py

2) fix the `classification_report` crash (so future runs don’t die when a class is missing):
```python
from sklearn.metrics import classification_report
labels_used = sorted(set(targs) | set(preds))
id2 = getattr(model.config, "id2label", {})
names = [id2.get(i, str(i)) for i in labels_used] if id2 else None
print(classification_report(targs, preds, labels=labels_used,
                            target_names=names, digits=3))
```bash
conda activate furi310
python demo_inference.py
```bash
git add README.md
git commit -m "README: add quick inference section"
git push
git add README.md
git commit -m "README: add quick inference section"
git push
