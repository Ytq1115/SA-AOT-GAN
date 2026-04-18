# import importlib
# import os
# from glob import glob
#
# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import ToTensor
# from utils.option import args
#
#
# def postprocess(image):
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)
#
#
# def main_worker(args, use_gpu=True):
#     # device = torch.device("cuda") if use_gpu else torch.device("cpu")
#
#     # Model and version
#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).cuda()
#     model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
#     model.eval()
#
#     # prepare dataset
#     image_paths = []
#     for ext in [".jpg", ".png"]:
#         image_paths.extend(glob(os.path.join(args.dir_image, "*" + ext)))
#     image_paths.sort()
#     #mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))
#     mask_paths = sorted(glob(os.path.join(args.dir_mask, args.mask_type, "*.png")))
#     os.makedirs(args.outputs, exist_ok=True)
#
#     # iteration through datasets
#     for ipath, mpath in zip(image_paths, mask_paths):
#         img_pil = Image.open(ipath).convert("RGB")
#         img_pil = img_pil.resize((args.image_size, args.image_size), Image.BICUBIC)
#         image = ToTensor()(img_pil)
#         image = (image * 2.0 - 1.0).unsqueeze(0)
#         mask_pil = Image.open(mpath).convert("L")
#         mask_pil = mask_pil.resize((args.image_size, args.image_size), Image.NEAREST)
#         mask = ToTensor()(mask_pil)
#         mask = (mask > 0.5).float()
#         mask = mask.unsqueeze(0)
#
#         image, mask = image.cuda(), mask.cuda()
#         image_masked = image * (1 - mask.float()) + mask
#
#         with torch.no_grad():
#             pred_img, _ = model(image_masked, mask, return_edge=True)
#
#         comp_imgs = (1 - mask) * image + mask * pred_img
#         image_name = os.path.basename(ipath).split(".")[0]
#         postprocess(image_masked[0]).save(os.path.join(args.outputs, f"{image_name}_masked.png"))
#         postprocess(pred_img[0]).save(os.path.join(args.outputs, f"{image_name}_pred.png"))
#         postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f"{image_name}_comp.png"))
#         print(f"saving to {os.path.join(args.outputs, image_name)}")
#
#
# if __name__ == "__main__":
#     main_worker(args)



# import importlib
# import os
# from glob import glob
#
# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import ToTensor
# from utils.option import args
#
#
# def postprocess(image):
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)
#
#
# def load_mask_and_ratio(mpath, image_size):
#     """
#     不强制二值化mask：
#       mask_tensor: [1,1,H,W] float in [0,1]
#       ratio: float = mask.mean()
#     """
#     mask_pil = Image.open(mpath).convert("L")
#     mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)
#     mask = ToTensor()(mask_pil)            # [1,H,W] in [0,1]
#     ratio = float(mask.mean().item())      # 平均遮挡强度/洞比例
#     mask = mask.unsqueeze(0)               # [1,1,H,W]
#     return mask, ratio
#
#
# def forward_compatible(model, image_masked, mask):
#     """
#     ✅ 兼容两种 Generator forward：
#     1) 创新版：model(image_masked, mask, return_edge=True) -> (pred_img, edge_logits)
#     2) 原版：model(image_masked, mask) -> pred_img
#
#     返回 pred_img（统一输出）
#     """
#     try:
#         out = model(image_masked, mask, return_edge=True)
#         # 常见情况：tuple/list
#         if isinstance(out, (tuple, list)):
#             return out[0]
#         return out
#     except TypeError:
#         # 原版不接受 return_edge 参数
#         return model(image_masked, mask)
#
#
# def main_worker(args, use_gpu=True):
#     device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
#
#     # Model
#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).to(device)
#     model.load_state_dict(torch.load(args.pre_train, map_location=device))
#     model.eval()
#
#     # prepare images
#     image_paths = []
#     for ext in [".jpg", ".png"]:
#         image_paths.extend(glob(os.path.join(args.dir_image, "*" + ext)))
#     image_paths.sort()
#
#     # prepare masks
#     mask_paths_all = sorted(glob(os.path.join(args.dir_mask, args.mask_type, "*.png")))
#
#     # ====== 按参数筛选 mask ======
#     ratio_min = float(getattr(args, "mask_ratio_min", 0.0))
#     ratio_max = float(getattr(args, "mask_ratio_max", 1.0))
#
#     selected_mask_paths = []
#     selected_ratios = []
#
#     for mpath in mask_paths_all:
#         _, r = load_mask_and_ratio(mpath, args.image_size)
#         if (r >= ratio_min) and (r <= ratio_max):
#             selected_mask_paths.append(mpath)
#             selected_ratios.append(r)
#
#     if len(selected_mask_paths) == 0:
#         raise RuntimeError(
#             f"No masks found in ratio range [{ratio_min}, {ratio_max}]. "
#             f"Total masks={len(mask_paths_all)}. "
#             f"Try widening the range or check mask folder: {os.path.join(args.dir_mask, args.mask_type)}"
#         )
#
#     os.makedirs(args.outputs, exist_ok=True)
#
#     # 打印筛选结果（方便汇报）
#     rmin_found = min(selected_ratios)
#     rmax_found = max(selected_ratios)
#     print(
#         f"[Mask Filter] Using {len(selected_mask_paths)}/{len(mask_paths_all)} masks "
#         f"in ratio range [{ratio_min}, {ratio_max}] "
#         f"(found range [{rmin_found:.3f}, {rmax_found:.3f}])"
#     )
#
#     # ====== 测试：每张图从筛选后的 mask 列表中选一个（可复现） ======
#     for ipath in image_paths:
#         img_pil = Image.open(ipath).convert("RGB")
#         img_pil = img_pil.resize((args.image_size, args.image_size), Image.BICUBIC)
#         image = ToTensor()(img_pil)
#         image = (image * 2.0 - 1.0).unsqueeze(0).to(device)
#
#         image_name = os.path.basename(ipath).split(".")[0]
#
#         # 可复现选mask
#         idx = abs(hash(image_name)) % len(selected_mask_paths)
#         mpath = selected_mask_paths[idx]
#
#         mask, ratio = load_mask_and_ratio(mpath, args.image_size)
#         mask = mask.to(device)
#
#         image_masked = image * (1 - mask.float()) + mask
#
#         with torch.no_grad():
#             pred_img = forward_compatible(model, image_masked, mask)
#
#         comp_imgs = (1 - mask) * image + mask * pred_img
#
#         # 保存
#         postprocess(image_masked[0]).save(os.path.join(args.outputs, f"{image_name}_masked_r{ratio:.3f}.png"))
#         postprocess(pred_img[0]).save(os.path.join(args.outputs, f"{image_name}_pred_r{ratio:.3f}.png"))
#         postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f"{image_name}_comp_r{ratio:.3f}.png"))
#
#         print(f"saving {image_name}: ratio={ratio:.3f}, mask={os.path.basename(mpath)}")
#
#
# if __name__ == "__main__":
#     main_worker(args)

import importlib
import os
import json
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from utils.option import args


def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def load_mask_and_ratio(mpath, image_size):
    """
    不强制二值化mask：
      mask_tensor: [1,1,H,W] float in [0,1]
      ratio: float = mask.mean()
    """
    mask_pil = Image.open(mpath).convert("L")
    mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)
    mask = ToTensor()(mask_pil)            # [1,H,W] in [0,1]
    ratio = float(mask.mean().item())      # 平均遮挡强度/洞比例
    mask = mask.unsqueeze(0)               # [1,1,H,W]
    return mask, ratio


def forward_compatible(model, image_masked, mask):
    """
    兼容两种 Generator forward：
    1) 创新版：model(image_masked, mask, return_edge=True) -> (pred_img, edge_logits)
    2) 原版：model(image_masked, mask) -> pred_img
    返回 pred_img
    """
    try:
        out = model(image_masked, mask, return_edge=True)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
    except TypeError:
        return model(image_masked, mask)


def build_or_load_assignment(image_paths, selected_mask_paths, assignment_path, seed=2026):
    """
    构建或读取固定的 image->mask 分配表，保证：
    1. 新模型和baseline使用完全一样的 mask 分配
    2. 多次运行稳定复现
    """
    if os.path.exists(assignment_path):
        with open(assignment_path, "r", encoding="utf-8") as f:
            assignment = json.load(f)
        print(f"[Assignment] Loaded existing assignment from: {assignment_path}")
        return assignment

    # 固定随机种子，对筛选后的 mask 做一次可复现打乱
    rng = random.Random(seed)
    shuffled_masks = selected_mask_paths.copy()
    rng.shuffle(shuffled_masks)

    assignment = {}

    # 采用“按顺序循环分配”的方式，稳定且公平
    for i, ipath in enumerate(image_paths):
        image_name = os.path.basename(ipath).split(".")[0]
        mpath = shuffled_masks[i % len(shuffled_masks)]
        assignment[image_name] = mpath

    os.makedirs(os.path.dirname(assignment_path), exist_ok=True)
    with open(assignment_path, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=2, ensure_ascii=False)

    print(f"[Assignment] Saved new assignment to: {assignment_path}")
    return assignment


def main_worker(args, use_gpu=True):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    # ---------------- Model ----------------
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args).to(device)
    model.load_state_dict(torch.load(args.pre_train, map_location=device))
    model.eval()

    # ---------------- Images ----------------
    image_paths = []
    for ext in [".jpg", ".png"]:
        image_paths.extend(glob(os.path.join(args.dir_image, "*" + ext)))
    image_paths.sort()

    # ---------------- Masks ----------------
    mask_paths_all = sorted(glob(os.path.join(args.dir_mask, args.mask_type, "*.png")))

    ratio_min = float(getattr(args, "mask_ratio_min", 0.0))
    ratio_max = float(getattr(args, "mask_ratio_max", 1.0))
    mask_seed = int(getattr(args, "mask_seed", 2026))

    selected_mask_paths = []
    selected_ratios = []

    for mpath in mask_paths_all:
        _, r = load_mask_and_ratio(mpath, args.image_size)
        if (r >= ratio_min) and (r <= ratio_max):
            selected_mask_paths.append(mpath)
            selected_ratios.append(r)

    if len(selected_mask_paths) == 0:
        raise RuntimeError(
            f"No masks found in ratio range [{ratio_min}, {ratio_max}]. "
            f"Total masks={len(mask_paths_all)}. "
            f"Try widening the range or check mask folder: {os.path.join(args.dir_mask, args.mask_type)}"
        )

    os.makedirs(args.outputs, exist_ok=True)

    rmin_found = min(selected_ratios)
    rmax_found = max(selected_ratios)
    print(
        f"[Mask Filter] Using {len(selected_mask_paths)}/{len(mask_paths_all)} masks "
        f"in ratio range [{ratio_min}, {ratio_max}] "
        f"(found range [{rmin_found:.3f}, {rmax_found:.3f}])"
    )

    # ---------------- Fair & Stable Assignment ----------------
    # 强烈建议：新模型和 baseline 都指定同一个 assignment 文件路径
    assignment_path = getattr(args, "assignment_path", None)
    if assignment_path is None:
        assignment_path = os.path.join(
            args.outputs,
            f"mask_assignment_{ratio_min:.2f}_{ratio_max:.2f}_seed{mask_seed}.json"
        )

    assignment = build_or_load_assignment(
        image_paths=image_paths,
        selected_mask_paths=selected_mask_paths,
        assignment_path=assignment_path,
        seed=mask_seed
    )

    # ---------------- Test ----------------
    for ipath in image_paths:
        img_pil = Image.open(ipath).convert("RGB")
        img_pil = img_pil.resize((args.image_size, args.image_size), Image.BICUBIC)
        image = ToTensor()(img_pil)
        image = (image * 2.0 - 1.0).unsqueeze(0).to(device)

        image_name = os.path.basename(ipath).split(".")[0]

        if image_name not in assignment:
            raise RuntimeError(f"Image {image_name} not found in assignment file.")

        mpath = assignment[image_name]
        mask, ratio = load_mask_and_ratio(mpath, args.image_size)
        mask = mask.to(device)

        image_masked = image * (1 - mask.float()) + mask

        with torch.no_grad():
            pred_img = forward_compatible(model, image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img

        postprocess(image_masked[0]).save(
            os.path.join(args.outputs, f"{image_name}_masked_r{ratio:.3f}.png")
        )
        postprocess(pred_img[0]).save(
            os.path.join(args.outputs, f"{image_name}_pred_r{ratio:.3f}.png")
        )
        postprocess(comp_imgs[0]).save(
            os.path.join(args.outputs, f"{image_name}_comp.png")
        )

        print(f"saving {image_name}: ratio={ratio:.3f}, mask={os.path.basename(mpath)}")


if __name__ == "__main__":
    main_worker(args)