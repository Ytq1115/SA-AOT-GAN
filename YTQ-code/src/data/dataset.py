# import os
# from glob import glob
#
# import numpy as np
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F
# from PIL import Image
# from torch.utils.data import Dataset
#
#
# class InpaintingData(Dataset):
#     def __init__(self, args):
#         super(Dataset, self).__init__()
#         self.w = self.h = args.image_size
#         self.mask_type = args.mask_type
#
#         # image and mask
#         self.image_path = []
#         for ext in ["*.jpg", "*.png"]:
#             self.image_path.extend(glob(os.path.join(args.dir_image, args.data_train, ext)))
#         self.mask_path = glob(os.path.join(args.dir_mask, args.mask_type, "*.png"))
#
#         # augmentation (we will sync geometry aug in __getitem__)
#         self.color_jitter = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
#
#         # optional: keep rotation, set to 0 if you don't want it
#         self.max_rotate = 45  # degrees
#
#     def __len__(self):
#         return len(self.image_path)
#
#     def __getitem__(self, index):
#         # load image
#         image = Image.open(self.image_path[index]).convert("RGB")
#         filename = os.path.basename(self.image_path[index])
#
#         if self.mask_type == "pconv":
#             index = np.random.randint(0, len(self.mask_path))
#             mask = Image.open(self.mask_path[index])
#             mask = mask.convert("L")
#         else:
#             mask = np.zeros((self.h, self.w)).astype(np.uint8)
#             mask[self.h // 4 : self.h // 4 * 3, self.w // 4 : self.w // 4 * 3] = 1
#             mask = Image.fromarray(mask).convert("L")
#
#         # augment
#         # ---------------------------
#         # Sync geometry augmentation
#         # ---------------------------
#
#         # 1) Make mask same spatial size as image BEFORE sampling crop params
#         #    (pconv masks might not match image size; use NEAREST to avoid soft edges)
#         if mask.size != image.size:
#             mask = mask.resize(image.size, resample=Image.NEAREST)
#
#         # 2) Sample ONE set of RandomResizedCrop params from image size
#         i, j, h, w = transforms.RandomResizedCrop.get_params(
#             image, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
#         )
#
#         # 3) Apply the SAME resized crop to both
#         #    image: bilinear; mask: nearest
#         image = F.resized_crop(image, i, j, h, w, (self.h, self.w), interpolation=transforms.InterpolationMode.BILINEAR)
#         mask = F.resized_crop(mask, i, j, h, w, (self.h, self.w), interpolation=transforms.InterpolationMode.NEAREST)
#
#         # 4) Apply SAME horizontal flip to both
#         if np.random.rand() < 0.5:
#             image = F.hflip(image)
#             mask = F.hflip(mask)
#
#         # 5) (Optional) Apply SAME rotation to both
#         #    If you don't want rotation, set self.max_rotate = 0 in __init__
#         if getattr(self, "max_rotate", 0) and self.max_rotate > 0:
#             angle = float(np.random.uniform(-self.max_rotate, self.max_rotate))
#             image = F.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
#             mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=0)
#
#         # 6) Color jitter ONLY for image
#         image = self.color_jitter(image)
#
#         # 7) To tensor
#         image = F.to_tensor(image) * 2.0 - 1.0
#         mask = F.to_tensor(mask)
#
#         # ---------------------------
#         # Binarize mask (IMPORTANT)
#         # ---------------------------
#         mask = (mask > 0.5).float()
#
#         return image, mask, filename
#
#
# if __name__ == "__main__":
#     from attrdict import AttrDict
#
#     args = {
#         "dir_image": "../../../dataset",
#         "data_train": "places2",
#         "dir_mask": "../../../dataset",
#         "mask_type": "pconv",
#         "image_size": 512,
#     }
#     args = AttrDict(args)
#
#     data = InpaintingData(args)
#     print(len(data), len(data.mask_path))
#     img, mask, filename = data[0]
#     print(img.size(), mask.size(), filename)

import os
from glob import glob

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type

        # image and mask
        self.image_path = []
        for ext in ["*.jpg", "*.png"]:
            self.image_path.extend(glob(os.path.join(args.dir_image, args.data_train, ext)))
        self.mask_path = glob(os.path.join(args.dir_mask, args.mask_type, "*.png"))

        # augmentation
        self.img_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),
            ]
        )
        self.mask_trans = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert("RGB")
        filename = os.path.basename(self.image_path[index])

        if self.mask_type == "pconv":
            index = np.random.randint(0, len(self.mask_path))
            mask = Image.open(self.mask_path[index])
            mask = mask.convert("L")
        else:
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[self.h // 4 : self.h // 4 * 3, self.w // 4 : self.w // 4 * 3] = 1
            mask = Image.fromarray(mask).convert("L")

        # augment
        image = self.img_trans(image) * 2.0 - 1.0
        mask = F.to_tensor(self.mask_trans(mask))

        return image, mask, filename


if __name__ == "__main__":
    from attrdict import AttrDict

    args = {
        "dir_image": "../../../dataset",
        "data_train": "places2",
        "dir_mask": "../../../dataset",
        "mask_type": "pconv",
        "image_size": 512,
    }
    args = AttrDict(args)

    data = InpaintingData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)