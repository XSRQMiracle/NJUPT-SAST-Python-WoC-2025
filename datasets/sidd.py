import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SIDDDataset(Dataset):
    def __init__(self, data_dir, patch_size=128, augment=True, cache_in_memory=False):
        super().__init__()
        self.patch_size = patch_size
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        self.image_cache = {}

        # 收集所有配对
        self.pairs = []
        scene_dirs = sorted(glob.glob(os.path.join(data_dir, '*')))

        for scene_dir in scene_dirs:
            if not os.path.isdir(scene_dir):
                continue
            gt_files = sorted(glob.glob(os.path.join(scene_dir, '*_GT_SRGB_*.PNG')))
            for gt_path in gt_files:
                noisy_path = gt_path.replace('_GT_SRGB_', '_NOISY_SRGB_')
                if os.path.exists(noisy_path):
                    self.pairs.append((noisy_path, gt_path))

        print(f"找到 {len(self.pairs)} 对 NOISY-GT 图像配对")

    def _load_image(self, path):
        if self.cache_in_memory and path in self.image_cache:
            return self.image_cache[path]

        img = Image.open(path).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0

        if self.cache_in_memory:
            self.image_cache[path] = img
        return img

    def _resize(self, noisy, gt):
        ps = self.patch_size
        noisy = np.array(Image.fromarray((noisy * 255).astype(np.uint8)).resize(
            (ps, ps), Image.BICUBIC), dtype=np.float32) / 255.0
        gt = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize(
            (ps, ps), Image.BICUBIC), dtype=np.float32) / 255.0
        return noisy, gt

    def _augment(self, noisy, gt):
        # 随机水平翻转
        if random.random() > 0.5:
            noisy = np.flip(noisy, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
        # 随机垂直翻转
        if random.random() > 0.5:
            noisy = np.flip(noisy, axis=0).copy()
            gt = np.flip(gt, axis=0).copy()
        # 随机旋转 90°
        k = random.randint(0, 3)
        if k > 0:
            noisy = np.rot90(noisy, k).copy()
            gt = np.rot90(gt, k).copy()
        return noisy, gt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            noisy_path, gt_path = self.pairs[idx]
            noisy = self._load_image(noisy_path)
            gt = self._load_image(gt_path)
        except (OSError, IOError, SyntaxError) as e:
            print(f"\n⚠ 跳过损坏图片: {self.pairs[idx][0]}, 错误: {e}")
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))

        noisy, gt = self._resize(noisy, gt)

        if self.augment:
            noisy, gt = self._augment(noisy, gt)

        # 转为tensor
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float()
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()

        return noisy, gt
