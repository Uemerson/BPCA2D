"""
Entrypoint application for training a CNN
on the ImageNet64 dataset.
"""

import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class ImageNet64Dataset(Dataset):
    """Custom Dataset for loading the ImageNet64 dataset from a pickle file."""

    def __init__(self, file_path: str, transform=None):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.images = data["data"]  # shape: [N, 12288]
        self.labels = np.array(data["labels"])
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return the image and label for a given index."""
        img_flat = self.images[idx]

        # reshape para [C, H, W]
        img = img_flat.reshape(3, 64, 64).astype(np.float32) / 255.0

        img = torch.tensor(img)  # tensor [3, 64, 64]
        label = int(self.labels[idx]) - 1

        if self.transform:
            img = self.transform(img)

        return img, label


class BPCA2D(nn.Module):
    """
    A custom pooling layer that applies PCA to local patches of the input feature maps.
    """

    def __init__(self, kernel_size=3, stride=3, q=1):
        """Initialize the BPCA2D layer."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.q = q

    def forward(self, x):
        """
        Define the forward pass of the BPCA2D layer.
        """

        B, C, H, W = x.shape
        K = self.kernel_size
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        L = H_out * W_out

        unfold = F.unfold(
            x, kernel_size=K, stride=self.stride
        )  # [B, C*K*K, L]
        patches = unfold.view(B, C, K * K, L).permute(
            0, 1, 3, 2
        )  # [B, C, L, K*K]
        x_flat = patches.reshape(B, C * L, K * K)  # [B, C*L, K*K]

        x_mean = x_flat.mean(dim=1, keepdim=True)
        x_centered = x_flat - x_mean  # [B, C*L, K*K]

        _, _, Vh = torch.linalg.svd(
            x_centered, full_matrices=False
        )  # [B, K*K, K*K]
        V = Vh[:, :1, :].transpose(-1, -2)  # [B, K*K, 1]

        projected = x_centered @ V  # [B, C*L, 1]
        return projected.view(B, C, H_out, W_out)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)  # skip connection


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32),
            BPCA2D(2, 2),
        )

        # Block 2: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            BPCA2D(2, 2),
        )

        # Block 3: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            BPCA2D(2, 2),
        )

        # Block 4: 8x8 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            BPCA2D(2, 2),
        )

        # Classifier: 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1000),
        )

    def save_feature_maps(self, x, save_path="feature_maps.png"):
        feature_maps = x[0].detach().cpu()
        n_maps = feature_maps.shape[0]

        cols = 16
        rows = max(1, n_maps // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = np.array(axes).reshape(rows, cols)

        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i < n_maps:
                ax.imshow(feature_maps[i], cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def forward(self, x, save_feature_maps=False):
        if save_feature_maps:
            self.save_feature_maps(x, "feature_maps_input.png")

        x = self.block1(x)
        if save_feature_maps:
            self.save_feature_maps(x, "feature_maps_block1.png")

        x = self.block2(x)
        if save_feature_maps:
            self.save_feature_maps(x, "feature_maps_block2.png")

        x = self.block3(x)
        if save_feature_maps:
            self.save_feature_maps(x, "feature_maps_block3.png")

        x = self.block4(x)
        if save_feature_maps:
            self.save_feature_maps(x, "feature_maps_block4.png")

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def set_seed(seed=42):
    """
    Set the random seed for reproducibility
    across various libraries and environments.

    42 is a number from "The Hitchhiker's Guide to the Galaxy"
    often used as a default seed.
    """
    random.seed(seed)  # Python
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    # For absolute reproducibility (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)

    file_path = "./data/Imagenet64_train_part1/train_data_batch_1"

    dataset = ImageNet64Dataset(file_path)

    print("Dataset size:", len(dataset))

    indices = list(range(len(dataset)))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=dataset.labels,  # garante proporção igual de cada classe
        random_state=42,
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_classes = np.unique(dataset.labels[train_dataset.indices] - 1)
    test_classes = np.unique(dataset.labels[test_dataset.indices] - 1)

    print("Train classes:", train_classes)
    print("Test classes:", test_classes)

    batch_size = 128
    mini_batches = 50

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    # labels = dataset.labels
    # indices = np.where(labels == 3)[0][:5]

    # plt.figure(figsize=(10, 2))

    # for i, idx in enumerate(indices):
    #     img, label = dataset[idx]

    #     # converter para [H, W, C]
    #     img = img.permute(1, 2, 0).numpy()

    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(img)
    #     plt.title(f"Label {label}")
    #     plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    print()

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        start_time = time.time()

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, save_feature_maps=(i == 0 and epoch == 0))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:  # print every 100 mini-batches
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}"
                )
                running_loss = 0.0

        end_time = time.time()
        print(
            f"Epoch {epoch + 1} training time: {end_time - start_time:.2f} seconds"
        )
        print()

    net.eval()  # set the model to evaluation mode

    # prepare to count predictions for each class
    correct_pred = {i: 0 for i in range(1000)}
    total_pred = {i: 0 for i in range(1000)}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[label.item()] += 1
                total_pred[label.item()] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:4d} is {accuracy:.1f} %")
