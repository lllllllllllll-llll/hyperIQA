import os
import os.path
import torch
import torchvision
import scipy.io
import numpy as np
import torch.utils.data as data
import random
from argparse import ArgumentParser
from PIL import Image
import yaml
from scipy import stats
import torch.nn as nn

from network import hyperIQA
from Datasets.LIVE import IQADataset

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    return pil_loader(path)

def default_loader(path):
    return pil_loader(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dataset", type=str, default="LIVE")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--save_path", type=dir, default="./savemodel/model.pth")
    args = parser.parse_args()

    seed = random.randint(10000000, 99999999)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("seed:", seed)

    save_model = "./savemodel/model.pth"

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

    index = []
    num = 0
    if args.dataset == "LIVE":
        print("dataset: LIVE")
        index = list(range(1, 30))
        num = len(range(1, 30))
        random.shuffle(index)
    elif args.dataset == "LIVEC":
        print("dataset: LIVEC")
        index = list(range(0, 1162))
        num = len(range(0, 1162))
        random.shuffle(index)

    train_index = index[0:round(0.8 * num)]
    test_index = index[round(0.8 * num):num]

    print('train_index', train_index)
    print('test_index', test_index)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])

    if args.dataset == 'LIVEC':
        from Datasets.LIVEC import LIVEChallengeFolder
        liveroot = config[args.dataset]['dataroot']
        trainset = LIVEChallengeFolder(root=liveroot,
                                       loader=default_loader,
                                       index=train_index,
                                       transforms=train_transforms)
        testset = LIVEChallengeFolder(root=liveroot,
                                      loader=default_loader,
                                      index=test_index,
                                      transforms=test_transforms)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)
    elif args.dataset == 'LIVE':
        train_dataset = IQADataset("LIVE", config, index, "train")
        test_dataset = IQADataset("LIVE", config, index, "test")

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)

    model = hyperIQA().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9, last_epoch=-1)

    best_SROCC = -1
    for epoch in range(args.epochs):
        # train
        model.train()
        LOSS = 0
        for i, (patches, label) in enumerate(train_loader):
            patches = patches.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            LOSS = LOSS + loss.item()
        train_loss = LOSS / (i+1)

        #test
        y_pred = []
        y_val = []
        model.eval()
        L = 0
        with torch.no_grad():
            for i, (patches, label) in enumerate(test_loader):
                y_val.append(label.item())
                patches = patches.to(device)
                label = label.to(device)
                outputs = model(patches)
                score = outputs.mean()
                y_pred.append(score.item())
                loss = criterion(score, label[0])
                L = L + loss.item()
        val_loss = L / (i + 1)

        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]

        print("Epoch {} Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f}".format(epoch, val_loss, val_SROCC, val_PLCC))

        if val_SROCC > best_SROCC and epoch > 100:
            print("Update Epoch {} best test SROCC".format(epoch))
            print("Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f}".format(val_loss,val_SROCC,val_PLCC))
            torch.save(model.state_dict(), save_model)
            best_SROCC = val_SROCC

    # final test
    model.load_state_dict(torch.load(save_model))
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_test = []
        L = 0
        for i, (patches, label) in enumerate(test_loader):
            y_val.append(label.item())
            patches = patches.to(device)
            label = label.to(device)
            outputs = model(patches)
            score = outputs.mean()
            y_pred.append(score.item())
            loss = criterion(score, label[0])
            L = L + loss.item()
    test_loss = L / (i + 1)
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]

    print("Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f}".format(test_loss,SROCC,PLCC))
