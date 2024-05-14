# Copyright (c) 2024 ✨Challyfilio✨
# 2024/2/21
# 提取原型特征

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
from resnet_gap import resnet50
from tqdm import tqdm


def create_dataloader(data_dir):
    image_size = 224
    trans = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
    dataset = datasets.ImageFolder(os.path.join(data_dir), transforms.Compose(trans))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    return dataloader


def load_net():
    # 当GPU可用时
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = resnet50()
    # 设置requires_grad为False，表示不需要对模型的参数进行梯度更新
    for param in net.parameters():
        param.requires_grad = False
    net = net.to(device)

    return net, device


if __name__ == "__main__":
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    model, device = load_net()
    feat_list = []
    feat_list.append(torch.zeros(1, 1024))
    for class_name in tqdm(CLASSES):
        data_path = '/workspace/pycharm_project/mmrotate/2023df/post_classification/cls_images/' + class_name
        dataloader = create_dataloader(data_path)
        images, _ = next(iter(dataloader))
        features = model(images.to(device))
        features = torch.mean(features, dim=0, keepdim=True)
        feat_list.append(features.cpu())
    print(len(feat_list))
    result = torch.cat(feat_list, dim=0)
    print(result.shape)
    torch.save(result, 'prototype.pt')
    t = torch.load('prototype.pt')
    print(t)
    exit()
