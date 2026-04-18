import os
import argparse
from glob import glob
from multiprocessing import Pool

import numpy as np
from metric import metric as module_metric
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Image Inpainting")
parser.add_argument("--real_dir", required=True, type=str)
parser.add_argument("--fake_dir", required=True, type=str)
parser.add_argument("--metric", type=str, nargs="+")
args = parser.parse_args()


def read_img(name_pair):
    rname, fname = name_pair
    rimg = Image.open(rname).convert("RGB")
    fimg = Image.open(fname).convert("RGB")

    # 关键：把 real resize 到 fake 的尺寸（用 bicubic）
    if rimg.size != fimg.size:
        rimg = rimg.resize(fimg.size, Image.BICUBIC)

    return np.array(rimg), np.array(fimg)

def main(num_worker=8):
    # 1) 收集 fake（你这里建议只放 comp_only 里的 *_comp.png）
    fake_names = sorted(glob(os.path.join(args.fake_dir, "*.png")))

    # 2) 按 fake 文件名去 real_dir 找同名 GT（支持 .png/.jpg）
    pairs = []
    missing = 0

    for fname in fake_names:
        base = os.path.splitext(os.path.basename(fname))[0]  # e.g. "113_comp"
        # 如果你的 fake 名字是 113_comp.png，就去掉 "_comp"
        if base.endswith("_comp"):
            stem = base[:-5]  # 去掉 "_comp" -> "113"
        else:
            stem = base  # 如果你 fake 没有 _comp 后缀，就直接用原名

        # 依次尝试在 real_dir 里找这些后缀
        candidates = [
            os.path.join(args.real_dir, stem + ".png"),
            os.path.join(args.real_dir, stem + ".jpg"),
            os.path.join(args.real_dir, stem + ".jpeg"),
        ]

        rname = None
        for c in candidates:
            if os.path.exists(c):
                rname = c
                break

        if rname is None:
            missing += 1
            # 只打印前几十个，避免刷屏（你也可以删掉这个限制）
            if missing <= 30:
                print(f"[WARN] GT not found for {os.path.basename(fname)} (tried: {stem}.png/.jpg/.jpeg)")
            continue

        pairs.append((rname, fname))

    print(f"fake images: {len(fake_names)}, paired: {len(pairs)}, missing_gt: {missing}")

    if len(pairs) == 0:
        raise RuntimeError("No valid (GT, fake) pairs found. Check file names and directories.")

    # 3) 读取图片（多进程）
    real_images = []
    fake_images = []

    pool = Pool(num_worker)
    for rimg, fimg in tqdm(
        pool.imap_unordered(read_img, pairs),
        total=len(pairs),
        desc="loading images"
    ):
        real_images.append(rimg)
        fake_images.append(fimg)

    # 4) 计算指标
    metrics = {met: getattr(module_metric, met) for met in args.metric}
    evaluation_scores = {key: 0 for key in metrics.keys()}
    for key, val in metrics.items():
        evaluation_scores[key] = val(real_images, fake_images, num_worker=num_worker)

    print(" ".join(["{}: {:6f},".format(key, val) for key, val in evaluation_scores.items()]))


if __name__ == "__main__":
    main()