import torch
import os
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from PIL import Image
import numpy as np
import h5py
from torchvision.transforms.functional import to_tensor
import cv2 as cv
import re

def gray_loader(path):
    return Image.open(path).convert('RGB')

def CropPatches(image, patch_size=224, stride=80):
    w, h = image.size
    patches = ()
    for i in range(0, h-stride, stride):
        for j in range(0, w-stride, stride):
            patch = to_tensor(image.crop((j, i, j+patch_size, i+patch_size)))
            patches = patches + (patch,)
    return patches

class IQADataset(Dataset):
    def __init__(self, dataset, config, index, status):
        self.loader = gray_loader
        im_dir = config[dataset]['im_dir']
        self.patch_size = config['patch_size']
        self.stride = config['stride']

        datainfo = config[dataset]['datainfo']
        Info = h5py.File(datainfo, 'r')
        ref_ids = Info['ref_ids'][0, :]

        test_ratio = config['test_ratio']
        train_ratio = config['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, test_index = [], []

        for i in range(len(ref_ids)):
            if (ref_ids[i] in trainindex):
                train_index.append(i)
            elif (ref_ids[i] in testindex):
                test_index.append(i)

        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)


        self.mos = Info['subjective_scores'][0, self.index]
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes() \
                        [::2].decode() for i in self.index]

        self.patches = ()
        self.label = []

        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(self.im_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            patches = CropPatches(im, self.patch_size, self.stride)

            if status == 'train':
                self.patches = self.patches + patches
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
            else:
                self.patches = self.patches + (torch.stack(patches), )
                self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], (torch.Tensor([self.label[idx]]))
