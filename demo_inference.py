import os, torch, numpy as np
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoConfig

# Where the trained checkpoint lives (default reads CHECKPOINT_PATH.txt)
with open("CHECKPOINT_PATH.txt") as f:
    ckpt = f.read().strip()

print("Loading checkpoint from:", ckpt)
cfg = AutoConfig.from_pretrained(ckpt)
print("Backbone:", cfg._name_or_path)
print("Num labels:", cfg.num_labels)
print("id2label:", getattr(cfg, "id2label", {}))

model = AutoModelForImageClassification.from_pretrained(ckpt).eval()
proc  = AutoImageProcessor.from_pretrained(ckpt)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# quick dummy run
pv = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    logits = model(pixel_values=pv).logits
pred = int(torch.argmax(logits, dim=-1)[0])
print("Logits shape:", tuple(logits.shape))
print("Pred class id:", pred)
