#!/usr/bin/env python
# encoding: utf-8

import os
import matplotlib as mpl
mpl.use('Agg')

import torch
from torch.autograd import Variable
from util.DataAugmentation import FaceIdPoseDataset
from util.SaveFeature import SaveFeature
from torch.utils.data import DataLoader
from torchvision import transforms


def Extract_Feature(Model, args):
    save_dir = '{}/{}/Feature'.format(args.output, args.snapshot)

    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    if args.cuda:
        Model.cuda()

    Model.eval()
    count = 0

    transformed_dataset = FaceIdPoseDataset(args.test_csv_file, transform=transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(224),
         transforms.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.Test_Batch, shuffle=False)

    for i, batch_data in enumerate(dataloader):
        batch_image = torch.FloatTensor(batch_data[0].float())
        minibatch_size = len(batch_image)

        if args.cuda:
            batch_image = batch_image.cuda()
        with torch.no_grad():
            batch_image = Variable(batch_image)

        batchImageName = batch_data[1]
        features = Model(batch_image, ExtractMode=True)

        features = (features.data).cpu().numpy()
        SaveFeature(features, batchImageName, save_dir, args)
        ID = batch_data[2]
        count += minibatch_size
        print("Finish Processing {} images...".format(count))