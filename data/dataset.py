from torch.utils.data import Dataset

import rasterio
import numpy as np
import glob
import os

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        assert len(self.img_files) == len(self.mask_files), "Mismatched images/masks!"
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        
        with rasterio.open(self.img_files[idx]) as img_file:
            img = img_file.read().astype(np.float32) / 255.0  # (C, H, W)

        
        with rasterio.open(self.mask_files[idx]) as mask_file:
            mask = (mask_file.read(1) > 0).astype(np.uint8)  # (H, W)

        img = np.transpose(img, (1, 2, 0))  #  (C, H, W) to (H, W, C)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # Ensure (1, H, W)

        return img, mask

