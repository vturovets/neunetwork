# calc_norm.py
import sys, math
from pathlib import Path
from PIL import Image
import numpy as np

def iter_images(root):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts:
            yield p

def main(root):
    n_pixels_total = 0
    s1 = 0.0      # sum of pixel values
    s2 = 0.0      # sum of squared pixel values
    n_files = 0

    for img_path in iter_images(root):
        try:
            img = Image.open(img_path).convert("L").resize((28, 28), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
            s1 += float(arr.sum())
            s2 += float((arr * arr).sum())
            n_pixels_total += arr.size
            n_files += 1
        except Exception as e:
            print(f"[warn] skip {img_path}: {e}")

    if n_pixels_total == 0:
        print("No images found.")
        return

    mean = s1 / n_pixels_total
    var = (s2 / n_pixels_total) - (mean * mean)
    std = math.sqrt(max(var, 0.0))

    print(f"Files used: {n_files}")
    print(f"Mean (grayscale, [0,1]): {mean:.6f}")
    print(f"Std  (grayscale, [0,1]): {std:.6f}")
    print("\nMNIST defaults for reference: mean=0.1307, std=0.3081")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calc_norm.py <path_to_images_root>")
        sys.exit(1)
    main(sys.argv[1])
