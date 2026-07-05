"""
Detect and crop black borders from dermoscopy images.

Some dermoscopic images have black vignette borders from the dermoscope
aperture. This script identifies images where the lesion area is
significantly brighter than the border (intensity ratio > 1.1) and crops
them to the bounding box of the non-black region.

Usage:
    python scripts/crop_borders.py \
        --csv-path metadata.csv \
        --data-dir derm_images/ \
        --dest-dir derm_images_cropped/

The CSV must have an `isic_id` column. Images are read from
<data-dir>/<isic_id>.jpg and written (if cropped) to
<dest-dir>/<isic_id>.jpg. Images that don't need cropping are copied
unchanged.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def obtain_bounding_box(threshed: np.ndarray) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 100 and h >= 100:
            return x, y, w, h
    return None


def intensity_ratio(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    cropped = gray[y:y+h, x:x+w]
    crop_intensity = cropped.mean()
    full_intensity = gray.mean()
    if full_intensity == 0:
        return 1.0
    return crop_intensity / full_intensity


def crop_if_needed(src_path: Path, dst_path: Path, threshold: float = 1.1) -> bool:
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, 0)

    bbox = obtain_bounding_box(thresh)
    if bbox is None:
        cv2.imwrite(str(dst_path), img)
        return False

    x, y, w, h = bbox
    ratio = intensity_ratio(gray, x, y, w, h)

    if ratio > threshold:
        img = img[y:y+h, x:x+w]
        cv2.imwrite(str(dst_path), img)
        return True
    else:
        cv2.imwrite(str(dst_path), img)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to metadata CSV (must have isic_id column)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing source images")
    parser.add_argument("--dest-dir", type=str, required=True,
                        help="Directory to write (possibly cropped) images")
    parser.add_argument("--threshold", type=float, default=1.1,
                        help="Intensity ratio above which to crop (default: 1.1)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    cropped_count = 0

    for isic_id in tqdm(df["isic_id"], desc="Processing"):
        src = data_dir / f"{isic_id}.jpg"
        dst = dest_dir / f"{isic_id}.jpg"
        if src.is_file():
            if crop_if_needed(src, dst, threshold=args.threshold):
                cropped_count += 1

    print(f"Done. {cropped_count}/{len(df)} images were cropped.")


if __name__ == "__main__":
    main()
