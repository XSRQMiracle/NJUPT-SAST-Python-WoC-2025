import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CIFAR10CDataset(Dataset):
    def __init__(self, data_dir, corruption_type, severity, transform=None):
        self.transform = transform
        data = np.load(os.path.join(data_dir, f'{corruption_type}.npy'))
        labels = np.load(os.path.join(data_dir, 'labels.npy'))

        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = data[start:end]
        self.labels = labels[start:end].astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
