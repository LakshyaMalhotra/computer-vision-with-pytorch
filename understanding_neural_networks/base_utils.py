import os
import random
import stat
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseUtils:
    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def seed_everything(seed: int = 0):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_metadata(self, file_name: str) -> pd.DataFrame:
        """Load image meta data from a file to a pandas dataframe."""
        df = pd.read_csv(os.path.join(self.data_path, file_name))

        return df

    def split_data(
        self,
        file_name: str,
        target_name: str,
        test_size: Optional[Union[int, float]] = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into train and test splits."""
        df = self.load_metadata(self.data_path, file_name)
        y = df[target_name]
        train_df, valid_df = train_test_split(
            df, test_size=test_size, stratify=y
        )

        return train_df, valid_df


# model
class CNNModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.LazyLinear(out_features=256)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = F.relu(x)
        x = self.maxpool2(self.conv2(x))
        x = F.relu(x)
        x = self.maxpool3(self.conv3(x))
        x = F.relu(x)
        x = self.maxpool4(self.conv4(x))
        x = F.relu(x)
        x = self.maxpool5(self.conv5(x))
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
