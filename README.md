# derm-research

Benchmarking 6 open-source AI models (DINOv2-Large, ConvNeXt V2-Large, MedGemma-4B, Qwen2.5-VL-7B, LLaVA-Med-7B, Qwen3-VL-8B) on classifying dermoscopic skin lesion images into 8 diagnostic categories, using a BCN20000-style dataset.

Authors: Rohan Rattan, Nawoda Wijesooriya, Arjun Damerla — Dept. of Dermatology, UC San Diego

## Repository layout

```
notebook.ipynb      Main Colab notebook — all 6 models, shared training/eval utilities
scripts/
  prepare_data.py         Connects metadata.csv <-> derm_images/, builds train/val/test.csv
  dinov2_smoke_test.py    Local CPU smoke test for the DINOv2 pipeline (see below)
README.md
```

**Not in this repo** (too large for git — see "Data setup" below): `metadata.csv`, `derm_images/` (~18.9k images, ~1.2GB), `licenses/`, `attribution.txt`. These live on Google Drive instead.

## Data setup (required before running the notebook)

The raw data — `metadata.csv`, `derm_images/`, `licenses/`, `attribution.txt` — is not pushed to GitHub due to size. Get a copy from Nawoda and upload it to Google Drive at:

```
MyDrive/BCN20000/
├── metadata.csv
├── derm_images/
│   ├── ISIC_0053453.jpg
│   └── ... (~18,946 images)
├── licenses/
│   └── CC-BY-NC.txt
└── attribution.txt
```

If you use a different Drive folder name, edit `DATA_SOURCE_ROOT` in **Cell 1.5** of the notebook to match.

Data is CC-BY-NC 4.0 licensed, attributed to Hospital Clínic de Barcelona — see `attribution.txt` / `licenses/`. Non-commercial research use only.

### The problem this data setup solves

`metadata.csv` has an `isic_id` per row but no column pointing at the actual image file, and not every row has a usable diagnosis. **Cell 2.0** in the notebook (via `scripts/prepare_data.py`) fixes both:

1. **Connects metadata to images** — builds `image_path` = `derm_images/<isic_id>.jpg` for every row (verified 1:1 — no orphaned rows or images).
2. **Drops unlabeled/unusable rows** — of 18,946 total rows, 1,156 have no diagnosis at all and 314 are labeled "Scar" (not one of the 8 target classes). Both groups are dropped, leaving **17,476 labeled images**.
3. **Maps diagnoses to the 8 target classes** (`diagnosis_3`, falling back to `diagnosis_2` for vascular lesions which have no `diagnosis_3`):

   | Class | Source diagnosis | Count |
   |---|---|---|
   | MEL  | Melanoma, NOS + Melanoma metastasis | 4,636 |
   | NV   | Nevus | 5,647 |
   | BCC  | Basal cell carcinoma | 3,676 |
   | AK   | Solar or actinic keratosis | 1,088 |
   | BKL  | Seborrheic keratosis + Solar lentigo | 1,551 |
   | DF   | Dermatofibroma | 168 |
   | VASC | diagnosis_2 == "Benign soft tissue proliferations - Vascular" | 151 |
   | SCC  | Squamous cell carcinoma, NOS | 559 |

4. **Builds a stratified 70/15/15 train/val/test split** (fixed seed = 42): **train 12,233 / val 2,621 / test 2,622**.

Re-running Cell 2.0 regenerates `train.csv`/`val.csv`/`test.csv` from scratch each time (deterministic, a few seconds) — no need to hand-edit those files.

## Running in Colab

1. Open `notebook.ipynb` from GitHub in Colab (badge at the top of the notebook, or File → Open notebook → GitHub tab).
2. Runtime → Change runtime type → GPU.
3. Run **Section 1** top to bottom (Cells 1.1–1.5):
   - Cell 1.2 mounts Drive (for data + checkpoints) and clones this repo into `/content/derm-research` (for code — `scripts/`).
   - Cell 1.5 sets `DATA_SOURCE_ROOT` (your Drive data folder), `CHECKPOINT_DIR` (Drive, persists across sessions), and `CLASS_NAMES`. `VERIFY_IMAGE_FILES = True` checks every image exists on first run — set to `False` on later runs since per-file checks over a mounted Drive folder are slow.
4. Run **Cell 2.0** (connects data, builds splits) then **Cells 2.1–2.3** (Dataset class, class-imbalance plot, class weights).
5. Run **Section 3** and **Section 4** (shared training/eval utilities — no changes needed).
6. Jump to whichever model section you're working on (Sections 5–10) and run it top to bottom. Each is self-contained.
7. **Section 11** builds the final cross-model comparison table/charts once multiple models have saved results to `CHECKPOINT_DIR`.

Per-model time estimates (T4 vs A100) and the full "before you start" checklist are in the notebook's own intro cell (Cell index 1, "Section 0 — How to use this notebook”) — note its checklist still mentions a separate `bcn20000_data_preparation.py` script; that's superseded by Cell 2.0 now (not yet cleaned up, see Known issues below).

## Status of each model section

| Section | Model | Status |
|---|---|---|
| 5 | DINOv2-Large | Data pipeline validated locally (see smoke test below). Full fine-tune not yet run on Colab GPU. |
| 6 | ConvNeXt V2-Large | Scaffolded, not yet run. |
| 7 | MedGemma-4B | Scaffolded (QLoRA), not yet run. |
| 8 | Qwen2.5-VL-7B | Scaffolded (QLoRA), not yet run. |
| 9 | LLaVA-Med-7B | Scaffolded (QLoRA), not yet run. |
| 10 | Qwen3-VL-8B | Scaffolded (QLoRA), not yet run. |

## Local smoke test (no GPU needed)

`scripts/dinov2_smoke_test.py` validates the data + DINOv2 pipeline end-to-end without needing a GPU or the full dataset — useful for checking changes locally before spending Colab GPU time. It runs the real `facebook/dinov2-large` backbone (frozen, CPU) on a small class-balanced subsample, caches the embeddings, then trains the same classification head architecture used in Cell 5.2 as a linear probe.

```
python scripts/prepare_data.py --data-root . --out-dir data
python scripts/dinov2_smoke_test.py
```

Latest run (480 train / 120 val / 120 test images, frozen backbone, 60-epoch head only — **not** the full benchmark, which fine-tunes the whole backbone on all 17,476 images):

| Metric | Score |
|---|---|
| Balanced Accuracy | 0.533 |
| Macro F1 | 0.540 |
| Macro AUROC | 0.849 |

Best classes: VASC, DF (0.73 sensitivity — visually distinctive). Hardest: BKL (0.40 — commonly confused with NV). Full results/confusion matrix saved to `local_smoke_test/dinov2_large_linear_probe/` (gitignored, local only).

This machine has an RTX 3060 but CPU-only PyTorch installed, so the smoke test is intentionally small-scale — it exists to prove the pipeline works, not to benchmark. The real run happens in Section 5 on Colab with a GPU and the full dataset.

## Known issues / follow-ups

- **Section 2.4 ("Image Preprocessing Script")** is a standalone CLI script (argparse + `__main__` guard) pasted into a notebook cell. It isn't wired into the data pipeline and will error if run as-is in a notebook — leave it alone unless you're refactoring it into a real cell.
- The intro cell's "Before you start" checklist references the old `bcn20000_data_preparation.py` workflow, which Cell 2.0 now replaces — worth a cleanup pass.
- `data/` (generated CSVs) and `checkpoints/` (model checkpoints/results) are gitignored — they're regenerated/produced by running the notebook, not meant to be committed.
