import os
import os.path
import torch
import torchvision
import scipy.io
import numpy as np
import torch.utils.data as data
import random
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    return pil_loader(path)

def default_loader(path):
    return pil_loader(path)

class LIVEChallengeFolder(data.Dataset):
    def __init__(self, root, loader, index, transforms=None):
        self.root = root
        self.loader = loader

        self.imgpath = scipy.io.loadmat(os.path.join(self.root, 'Data', 'AllImages_release.mat'))
        self.imgpath = self.imgpath['AllImages_release']
        self.imgpath = self.imgpath[7:1169]

        self.mos = scipy.io.loadmat(os.path.join(self.root, 'Data', 'AllMOS_release.mat'))
        self.labels = self.mos['AllMOS_release'].astype(np.float32)
        self.labels = self.labels[0][7:1169]

        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(self.root, 'Images', self.imgpath[item][0][0]), self.labels[item]))
        self.samples = sample

        self.transform = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
