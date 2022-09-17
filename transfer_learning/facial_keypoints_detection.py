import os
from copy import deepcopy

import timm
import torch
import cv2 as cv
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from albumentations import Normalize
from torch.utils.data import Dataset, DataLoader
from sklearn import cluster
from torchinfo import summary
from mpl_toolkits.mplot3d import Axes3D


class FacesData(Dataset):
    def __init__(self, data_dir, df):
        super().__init__()
        self.data_dir = data_dir
        self.df = df
        self.normalize = Normalize(p=1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        img_path = os.path.join(self.data_dir, self.df.iloc[ix, 0])
        img = self._read_image(img_path)
        kp = deepcopy(self.df.iloc[ix, 1:].tolist())
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        kp_y = (np.array(kp[12]) / img.shape[0]).tolist()
        processed_kp = kp_x + kp_y
        img = self._preprocess_image(img)
        return img, processed_kp

    def load_img(self, ix):
        img_path = os.path.join(self.data_dir, self.df.iloc[ix, 0])
        img = self._read_image(img_path)
        img = cv.resize(img, (224, 224))
        return img

    def _read_image(self, path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0
        return img

    def _preprocess_image(self, image):
        image = cv.resize(image, (224, 224))
        # bring channels first
        image = torch.as_tensor(image).permute(2, 0, 1)
        image = self.normalize(image).float()
        return image


if __name__ == "__main__":
    root_dir = "data/P1_Facial_Keypoints/data"
    meta_data = pd.read_csv(
        os.path.join(root_dir, "training_frames_keypoints.csv")
    )
    print(meta_data.head())
