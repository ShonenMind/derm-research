"""
Connects metadata.csv to the image files in derm_images/ and builds the
train / val / test CSVs consumed by BCN20000Dataset in notebook.ipynb.

Problem 1 — metadata and images are disconnected:
    metadata.csv has an `isic_id` column but no path to the actual jpg.
    We add an `image_path` column pointing at derm_images/<isic_id>.jpg.

Problem 2 — not every image has a usable diagnosis:
    diagnosis_1/2/3 are missing for 1,156 images (no diagnosis at all), and
    314 images are labeled "Scar" (diagnosis_2: Fibro-histiocytic), which is
    not one of the 8 target diagnostic classes. Both groups are dropped.

Label mapping (diagnosis_3, falling back to diagnosis_2 for vascular
lesions which have no diagnosis_3) -> the 8 BCN20000-style classes already
defined in the notebook's global config (CLASS_NAMES):

    MEL  <- Melanoma, NOS | Melanoma metastasis
    NV   <- Nevus
    BCC  <- Basal cell carcinoma
    AK   <- Solar or actinic keratosis
    BKL  <- Seborrheic keratosis | Solar lentigo
    DF   <- Dermatofibroma
    VASC <- diagnosis_2 == "Benign soft tissue proliferations - Vascular"
    SCC  <- Squamous cell carcinoma, NOS

Usage:
    python scripts/prepare_data.py --data-root . --out-dir data

`--data-root` points at the folder that contains metadata.csv and
derm_images/ directly (not the git repo -- on Colab this is a Google Drive
folder, since the images are too large to keep in the GitHub repo).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

DIAGNOSIS3_TO_CLASS = {
    "Melanoma, NOS": "MEL",
    "Melanoma metastasis": "MEL",
    "Nevus": "NV",
    "Basal cell carcinoma": "BCC",
    "Solar or actinic keratosis": "AK",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    "Dermatofibroma": "DF",
    "Squamous cell carcinoma, NOS": "SCC",
    # "Scar" is intentionally omitted -> dropped (not one of the 8 classes)
}
VASCULAR_DIAGNOSIS2 = "Benign soft tissue proliferations - Vascular"


def map_label(row) -> str | None:
    d3 = row["diagnosis_3"]
    if isinstance(d3, str) and d3 in DIAGNOSIS3_TO_CLASS:
        return DIAGNOSIS3_TO_CLASS[d3]
    if pd.isna(d3) and row["diagnosis_2"] == VASCULAR_DIAGNOSIS2:
        return "VASC"
    return None  # no diagnosis, or a diagnosis outside our 8-class taxonomy


def build_connected_dataframe(data_source_root: Path, verify_files: bool = True) -> pd.DataFrame:
    df = pd.read_csv(data_source_root / "metadata.csv")
    n_total = len(df)

    # --- Problem 1: connect metadata rows to actual image files ---
    df["image_path"] = df["isic_id"].apply(
        lambda isic_id: str(data_source_root / "derm_images" / f"{isic_id}.jpg")
    )
    if verify_files:
        # Checking is_file() on every row is fast on local/cloned disk, but can
        # be slow over a mounted Google Drive folder with ~19k small files.
        # Set verify_files=False there once you've confirmed the upload is complete.
        exists = df["image_path"].apply(lambda p: Path(p).is_file())
        n_missing_file = (~exists).sum()
        df = df[exists].copy()
    else:
        n_missing_file = 0

    # --- Problem 2: drop rows without a usable diagnosis ---
    df["class_name"] = df.apply(map_label, axis=1)
    n_unlabeled = df["class_name"].isna().sum()
    df = df.dropna(subset=["class_name"]).copy()

    df["label"] = df["class_name"].map({c: i for i, c in enumerate(CLASS_NAMES)})
    df["anatom_site_general"] = df["anatom_site_1"]

    print(f"Total metadata rows      : {n_total:,}")
    print(f"Dropped (image missing)  : {n_missing_file:,}")
    print(f"Dropped (no usable label): {n_unlabeled:,}")
    print(f"Usable labeled rows      : {len(df):,}")
    print()
    print("Class distribution:")
    print(df["class_name"].value_counts().reindex(CLASS_NAMES))

    return df


def split_and_save(df: pd.DataFrame, out_dir: Path, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "isic_id", "image_path", "label", "class_name",
        "diagnosis_1", "diagnosis_2", "diagnosis_3",
        "anatom_site_general", "age_approx", "sex", "melanocytic", "lesion_id",
    ]
    df = df[keep_cols]

    # 70/15/15 stratified split, fixed seed for reproducibility across models
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed,
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print()
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"Saved to {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verify", action="store_true",
                         help="Skip per-file existence check (faster on Google Drive)")
    args = parser.parse_args()

    connected = build_connected_dataframe(Path(args.data_root), verify_files=not args.skip_verify)
    split_and_save(connected, Path(args.out_dir), seed=args.seed)
