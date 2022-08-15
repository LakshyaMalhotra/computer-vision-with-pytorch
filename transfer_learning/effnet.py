import os
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.folder import (
    DatasetFolder,
    default_loader,
    IMG_EXTENSIONS,
)
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data_path(root_path: str):
    train_data_dir = os.path.join(root_path, "train")
    valid_data_dir = os.path.join(root_path, "valid")

    return train_data_dir, valid_data_dir


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]

    # class CustomImageFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(sample)

        # sample = torch.as_tensor(sample, dtype=torch.float32)
        # target = torch.as_tensor(target, dtype=torch.long)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        if self.transform is not None:
            try:
                imgs = self.transform(imgs)
            except Exception:
                imgs = self.transform(imgs)["image"]

        if self.target_transform is not None:
            try:
                imgs = self.target_transform(imgs)
            except Exception:
                imgs = self.target_transform(imgs)["image"]

        return imgs, targets


def get_dataloaders(
    batch_size=64,
    data_dir="/home/lakshya/side_projects/image_classification/multiclass_classification/dog_images",
):
    # define train, valid and test directories
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    # normalize the images with imagenet dataset
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.RandomCrop(height=224, width=224),
            # A.ShiftScaleRotate(
            #     shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.
            # ),
            A.RGBShift(
                r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent=(0.15, 0.3),
                rotate=(15, 30),
                shear=(5, 12),
                cval=0,
                mode=0,
                keep_ratio=True,
                p=0.4,
            ),
            A.Blur(blur_limit=(3, 7), p=0.3),
            A.CLAHE(clip_limit=(4, 8), tile_grid_size=(8, 8), p=0.3),
            A.Equalize(mode="cv", by_channels=True, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.GaussNoise(p=0.3),
            normalize,
            # ToTensorV2(),
        ]
    )
    # # transforms for validation and test set
    # transform = A.Compose(
    #     [A.Resize(256, 256), A.CenterCrop(224, 224), normalize, ToTensorV2()]
    # )
    transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=224, width=224),
            normalize,
            # ToTensorV2(),
        ]
    )

    # define train, valid and test datasets
    train_data = datasets.ImageFolder(
        train_dir, transform=Transforms(train_transform)
    )
    valid_data = datasets.ImageFolder(
        valid_dir, transform=Transforms(transform)
    )
    # train_data = CustomImageFolder(
    #     train_dir, transform=Transforms(train_transform)
    # )
    # valid_data = CustomImageFolder(valid_dir, transform=Transforms(transform))

    # dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        # collate_fn=train_data.collate_fn,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        # collate_fn=valid_data.collate_fn,
    )

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
    }

    return loaders


def get_model(num_classes=133):
    model = timm.create_model(
        "efficientnet_b2", pretrained=True, num_classes=num_classes
    )
    for params in model.parameters():
        params.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        OrderedDict(
            [
                (
                    "fc1",
                    nn.Linear(
                        in_features=in_features, out_features=num_classes
                    ),
                ),
                ("activation", nn.LogSoftmax(dim=1)),
            ]
        )
    )
    nn.init.kaiming_normal_(model.classifier.fc1.weight)
    summary(model, input_size=(4, 3, 224, 224), device="cpu")

    return model


@torch.no_grad()
def accuracy(predictions, targets):
    _, idx_max = predictions.max(-1)
    is_correct = idx_max == targets
    return is_correct


def train(
    n_epochs, dataloaders, model, optimizer, criterion, device, scheduler=None
):
    start = time.time()

    # monitor training and validation losses
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    # iterate through all the epochs
    for epoch in range(1, n_epochs + 1):
        train_epoch_losses, train_epoch_accuracies = [], []
        valid_epoch_losses, valid_epoch_accuracies = [], []
        epoch_start = time.time()

        # put the model to training mode
        model.train()
        for (images, targets) in dataloaders["train"]:
            images = images.to(device)
            targets = targets.to(device)

            # clear any accumulated gradients
            optimizer.zero_grad()

            # feed forward
            predictions = model(images)

            # calculate the loss
            loss = criterion(predictions, targets)

            # backprop
            loss.backward()

            # optimizer step
            optimizer.step()

            # store batch loss and accuracy
            train_epoch_losses.append(loss.item())

            batch_accuracy = accuracy(predictions, targets)
            train_epoch_accuracies.extend(batch_accuracy.cpu().numpy())
        train_epoch_loss = np.mean(train_epoch_losses)
        train_epoch_accuracy = np.mean(train_epoch_accuracies) * 100

        # evaluate
        model.eval()
        for (images, targets) in dataloaders["valid"]:
            images = images.to(device)
            targets = targets.to(device)

            # compute predictions with no backprop
            with torch.no_grad():
                predictions = model(images)

                # calculate the loss
                loss = criterion(predictions, targets)

            # store batch loss and accuracy
            valid_epoch_losses.append(loss.item())

            batch_accuracy = accuracy(predictions, targets)
            valid_epoch_accuracies.extend(batch_accuracy.cpu().numpy())

        valid_epoch_loss = np.mean(valid_epoch_losses)
        valid_epoch_accuracy = np.mean(valid_epoch_accuracies) * 100

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)

        valid_losses.append(valid_epoch_loss)
        valid_accuracies.append(valid_epoch_accuracy)

        if scheduler is not None:
            scheduler.step(valid_epoch_loss)

        print(
            f"Epoch {epoch:02d} / {n_epochs} \tTime taken:{(time.time() - epoch_start):.2f} seconds"
        )
        print(
            f"Training loss: {train_epoch_loss:5f} \tValidation loss: {valid_epoch_loss:5f}"
        )
        print(
            f"Training accuracy: {train_epoch_accuracy:.3f} \tValidation accuracy: {valid_epoch_accuracy:.3f}"
        )
    print(f"Training finished in {(time.time()-start):.2f} seconds.")

    return train_losses, valid_losses, train_accuracies, valid_accuracies


def plot_metrics(
    train_losses, train_accuracies, valid_losses, valid_accuracies
):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 8), dpi=150)

    x = range(1, len(train_losses) + 1)
    ax[0].plot(x, train_losses, label="Train")
    ax[0].plot(x, valid_losses, label="Valid")
    ax[0].legend(frameon=False, fontsize=8)
    # ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax[0].tick_params(axis="both", which="both", labelsize=8)
    ax[0].set_title("Loss", fontsize=10)

    ax[1].plot(x, train_accuracies, label="Train")
    ax[1].plot(x, valid_accuracies, label="Valid")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].xaxis.set_major_locator(plt.MultipleLocator(2))
    ax[1].tick_params(axis="x", which="both", labelsize=7)
    ax[1].set_xlabel("Epoch", fontsize=9)
    ax[1].set_title("Accuracy", fontsize=10)

    fig.tight_layout()
    plt.show()


def main(n_epochs=10, batch_size=64):
    # get dataloaders
    dataloaders = get_dataloaders(batch_size=batch_size)
    print(f"Number of training batches: {len(dataloaders['train'])}")
    print(f"Number of validation batches: {len(dataloaders['valid'])}")
    images, labels = next(iter(dataloaders["train"]))
    print(f"Images shape: {images[0].shape}")
    print(f"Labels shape: {labels[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # get model
    model = get_model(num_classes=133)
    model = model.to(device)

    # define loss function
    criterion = nn.NLLLoss()

    # define optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4
    )

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True, factor=0.6, patience=4
    )

    # train and evaluate
    train_losses, valid_losses, train_accuracies, valid_accuracies = train(
        n_epochs, dataloaders, model, optimizer, criterion, device, scheduler
    )

    # plot the results
    plot_metrics(
        train_losses, train_accuracies, valid_losses, valid_accuracies
    )


if __name__ == "__main__":
    main(n_epochs=20, batch_size=64)
