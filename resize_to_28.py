
# (same as previous cell; abbreviated header)
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List
from PIL import Image, ImageOps, ImageStat

def parse_args():
    p = argparse.ArgumentParser(description="Batch-resize images to 28x28.")
    p.add_argument("--src", required=True, help="Source folder or file")
    p.add_argument("--dst", required=True, help="Destination folder")
    p.add_argument("--mode", default="pad", choices=["pad", "fit", "stretch"])
    p.add_argument("--bg", default="white", choices=["white", "black"])
    p.add_argument("--grayscale", default="true", choices=["true","false"])
    p.add_argument("--invert", default="false", choices=["true","false"])
    p.add_argument("--auto_invert", default="false", choices=["true","false"])
    p.add_argument("--exts", default="png,jpg,jpeg,bmp,ppm,pgm")
    return p.parse_args()

def booly(s: str) -> bool:
    return s.lower() in ("1","true","yes","y")

def collect_images(src: Path, allowed_exts: set) -> List[Path]:
    if src.is_file():
        return [src]
    out = []
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower().lstrip(".") in allowed_exts:
            out.append(p)
    return sorted(out)

def resize_pad(img, bg: str):
    from PIL import Image
    target = (28, 28)
    img = img.copy()
    if img.mode not in ("L","RGB","RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    scale = min(target[0]/w, target[1]/h)
    new_w, new_h = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas_color = 255 if bg == "white" else 0
    mode = "L" if img.mode == "L" else "RGB"
    canvas = Image.new(mode, target, color=(canvas_color if mode=="L" else (canvas_color,)*3))
    off = ((target[0]-new_w)//2, (target[1]-new_h)//2)
    canvas.paste(img, off)
    return canvas.convert("L")

def resize_fit(img):
    from PIL import Image
    img = img.copy()
    if img.mode not in ("L","RGB","RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side)//2
    top = (h - side)//2
    img = img.crop((left, top, left+side, top+side))
    img = img.resize((28, 28), Image.LANCZOS)
    return img.convert("L")

def resize_stretch(img):
    from PIL import Image
    return img.convert("L").resize((28,28), Image.LANCZOS)

def maybe_grayscale(img, flag: bool):
    return img.convert("L") if flag and img.mode != "L" else img

def maybe_invert(img, invert: bool, auto_invert: bool):
    if auto_invert:
        m = ImageStat.Stat(img.convert("L")).mean[0] / 255.0
        if m > 0.5:  # bright background -> invert
            return ImageOps.invert(img.convert("L"))
        return img
    if invert:
        return ImageOps.invert(img.convert("L"))
    return img

def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    exts = set([e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()])

    files = collect_images(src, exts)
    if not files:
        print("No matching images found.", file=sys.stderr)
        sys.exit(2)

    for fp in files:
        try:
            img = Image.open(fp)
            img = maybe_grayscale(img, booly(args.grayscale))
            img = maybe_invert(img, booly(args.invert), booly(args.auto_invert))
            if args.mode == "pad":
                out = resize_pad(img, args.bg)
            elif args.mode == "fit":
                out = resize_fit(img)
            else:
                out = resize_stretch(img)
            out = out.convert("L")
            out_path = dst / (fp.stem + "_28x28.png")
            out.save(out_path, format="PNG")
            print(f"OK  {fp.name} -> {out_path.name}")
        except Exception as e:
            print(f"ERR {fp.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
