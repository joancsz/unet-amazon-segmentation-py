import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0, 0, 0, 0), std=(1, 1, 1, 1)),
        ToTensorV2(),
    ])