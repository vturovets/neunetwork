from pathlib import Path
from torchvision import datasets, transforms, utils

out = Path("my_imgs"); out.mkdir(exist_ok=True)
ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
# save first 20 test samples as PNGs with label in filename
for i in range(20):
    img, lbl = ds[i]
    utils.save_image(img, out / f"{i:03d}_label{lbl}.png")
print("Saved to", out.resolve())