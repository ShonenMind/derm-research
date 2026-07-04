"""
Local CPU smoke test for the DINOv2-Large section of notebook.ipynb.

This machine has no CUDA-enabled torch install and no GPU-accelerated
training path, so a full fine-tune (all 17,476 images, unfrozen backbone,
10 epochs, as configured in the notebook) would take far too long here.

Instead this script:
  1. Takes a small, class-capped stratified subsample of the real
     train/val/test CSVs produced by prepare_data.py.
  2. Runs the real DINOv2-Large backbone (facebook/dinov2-large) once per
     image to extract CLS-token embeddings (frozen backbone).
  3. Trains the *same* classification head architecture used in the
     notebook's DINOv2Classifier on those cached embeddings (linear probe).
  4. Evaluates with the same metrics as the notebook's run_full_evaluation:
     balanced accuracy, macro F1, macro AUROC, per-class sensitivity,
     confusion matrix.

This is a smoke test to prove the data pipeline + model + metrics code
all work end to end and to produce real (if noisy, small-sample) numbers.
The full benchmark run — all data, unfrozen backbone, 10 epochs — is what
Section 5 of notebook.ipynb runs on Colab with a GPU.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from transformers import AutoImageProcessor, Dinov2Model

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES = len(CLASS_NAMES)
SEED = 42
MODEL_ID = "facebook/dinov2-large"

DATA_DIR = Path("data")
OUT_DIR = Path("local_smoke_test/dinov2_large_linear_probe")
CACHE_DIR = OUT_DIR / "embedding_cache"

CAPS = {"train": 60, "val": 15, "test": 15}
BATCH_SIZE = 8
HEAD_EPOCHS = 60
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 0.01


def subsample(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(cap, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, processor, backbone, split: str) -> tuple[np.ndarray, np.ndarray]:
    cache_path = CACHE_DIR / f"{split}.npz"
    if cache_path.is_file():
        cached = np.load(cache_path)
        if len(cached["labels"]) == len(df):
            print(f"  [{split}] loaded {len(df)} cached embeddings from {cache_path}")
            return cached["embeddings"], cached["labels"]

    embeddings, labels = [], []
    n = len(df)
    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
        inputs = processor(images=images, return_tensors="pt")
        out = backbone(pixel_values=inputs["pixel_values"])
        cls = out.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls)
        labels.extend(batch["label"].tolist())
        done = min(start + BATCH_SIZE, n)
        elapsed = time.time() - t0
        print(f"  [{split}] {done}/{n} embedded  ({elapsed:.0f}s elapsed)", flush=True)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, labels=labels)
    return embeddings, labels


class Head(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading full train/val/test CSVs produced by prepare_data.py ...")
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    val_full = pd.read_csv(DATA_DIR / "val.csv")
    test_full = pd.read_csv(DATA_DIR / "test.csv")

    train_df = subsample(train_full, CAPS["train"], SEED)
    val_df = subsample(val_full, CAPS["val"], SEED)
    test_df = subsample(test_full, CAPS["test"], SEED)
    print(f"Subsample sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    print(f"\nLoading {MODEL_ID} (frozen backbone, CPU) ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    backbone = Dinov2Model.from_pretrained(MODEL_ID)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    hidden_size = backbone.config.hidden_size

    print("\nExtracting CLS embeddings (one real forward pass per image) ...")
    train_emb, train_y = extract_embeddings(train_df, processor, backbone, "train")
    val_emb, val_y = extract_embeddings(val_df, processor, backbone, "val")
    test_emb, test_y = extract_embeddings(test_df, processor, backbone, "test")

    train_emb_t = torch.tensor(train_emb, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.long)
    val_emb_t = torch.tensor(val_emb, dtype=torch.float32)
    val_y_t = torch.tensor(val_y, dtype=torch.long)
    test_emb_t = torch.tensor(test_emb, dtype=torch.float32)
    test_y_t = torch.tensor(test_y, dtype=torch.long)

    class_weights = compute_class_weights(train_y, N_CLASSES)
    rounded_weights = [round(w, 3) for w in class_weights.tolist()]
    print(f"\nClass weights: {dict(zip(CLASS_NAMES, rounded_weights))}")

    head = Head(hidden_size, N_CLASSES)
    optimizer = optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\nTraining linear-probe head for {HEAD_EPOCHS} epochs on cached embeddings ...")
    best_val_bal_acc = -1.0
    best_state = None
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(train_emb_t)
        loss = criterion(logits, train_y_t)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_emb_t)
            val_preds = val_logits.argmax(dim=1).numpy()
        val_bal_acc = balanced_accuracy_score(val_y_t.numpy(), val_preds)

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  train_loss={loss.item():.4f}  val_bal_acc={val_bal_acc:.4f}")

    print(f"\nBest val balanced accuracy: {best_val_bal_acc:.4f}")
    head.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "head_best.pth")

    head.eval()
    with torch.no_grad():
        test_logits = head(test_emb_t)
        test_probs = torch.softmax(test_logits, dim=1).numpy()
        test_preds = test_probs.argmax(axis=1)
    y_true = test_y_t.numpy()

    bal_acc = balanced_accuracy_score(y_true, test_preds)
    macro_f1 = f1_score(y_true, test_preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, test_probs, multi_class="ovr", average="macro")
    except Exception:
        auroc = float("nan")

    print(f"\n{'-'*50}")
    print("TEST RESULTS -- DINOv2-Large (local linear-probe smoke test)")
    print(f"{'-'*50}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Macro AUROC       : {auroc:.4f}")

    report_dict = classification_report(
        y_true, test_preds, labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES, digits=4, zero_division=0, output_dict=True,
    )
    print()
    print(classification_report(
        y_true, test_preds, labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    ))

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(OUT_DIR / "dinov2_large_report.csv")

    cm = confusion_matrix(y_true, test_preds, labels=list(range(N_CLASSES)))
    np.savetxt(OUT_DIR / "dinov2_large_confusion.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")

    results = {
        "model": "DINOv2-Large (local linear-probe smoke test)",
        "note": (
            f"Frozen backbone, cached CLS embeddings, {CAPS['train']}/class train "
            f"cap, {HEAD_EPOCHS}-epoch linear head only. Not the full benchmark run."
        ),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "balanced_acc": round(float(bal_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_auroc": round(float(auroc), 4),
        "best_val_bal_acc": round(float(best_val_bal_acc), 4),
    }
    for cls in CLASS_NAMES:
        if cls in report_dict:
            results[f"sens_{cls}"] = round(report_dict[cls]["recall"], 4)

    with open(OUT_DIR / "dinov2_large_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results, report, and confusion matrix to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
