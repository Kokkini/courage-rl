from PIL import Image
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("--img-dir")
args = args.parse_args()

files = os.listdir(args.img_dir)
files = sorted(files)
imgs = []
for file in files:
    if file[-3:] != "jpg":
        continue
    path = os.path.join(args.img_dir, file)
    im = Image.open(path)
    imgs.append(im)
path = os.path.join(args.img_dir, "danger_score.gif")
imgs[0].save(path, save_all=True, append_images=imgs[1:])
