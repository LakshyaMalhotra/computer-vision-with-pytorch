import os
import time

import torch
import cv2 as cv
import albumentations as A
from torch.utils.data import Dataset, DataLoader


class CollateDataset(Dataset):
    def __init__(self, data_dir, aug=None):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)
        self.aug = aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        f = self.data[index]
        image = cv.imread(os.path.join(self.data_dir, f))[:, :, 0]
        label = f.split("@")[0] == "x"

        if self.aug:
            image = self.aug(image=image)["image"]

        image = torch.as_tensor(image, dtype=torch.float)
        label = torch.as_tensor(label, dtype=torch.long)

        return image, label


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    # `collate=False` always worked better and `num_workers=0` always worked
    # better in the dataloaders than `num_workers>0`
    collate = False
    data_dir = "/home/lakshya/side_projects/computer_vision_with_pytorch/visualizing_cnns/data/all"
    transforms = A.Compose([A.Resize(20, 20)])
    dataset = CollateDataset(data_dir, transforms)
    dl = DataLoader(
        dataset,
        batch_size=5,
        num_workers=0,
        collate_fn=collate_fn if collate else None,
    )
    start = time.time()
    for batches in dl:
        if collate:
            print(batches[0][0].shape, batches[1])
        else:
            print(batches[0].shape, batches[1])

    print(f"Task finished in {(time.time() - start):.3f} seconds.")
