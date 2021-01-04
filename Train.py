#/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from util.log_learning import log_learning
from util.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop
from util.exp_lr_scheduler import adjust_learning_rate
from util.Validation import Validation_Process
from util.checkpoint import save_checkpoint
import torch.backends.cudnn as CUDNN

writer = SummaryWriter()

def Train(Model, args):
    #Define num of classes
    Nd = args.Nd
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2

    #Define gpu mode
    if args.cuda:
        Model.cuda()
    #choose your optimizer
    optimizer = optim.Adam(Model.parameters(), lr=args.lr, betas=(beta1_Adam, beta2_Adam))
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        optimizer.load_state_dict(checkpoint['optimizer'])

    Model.train()

    loss_criterion = nn.CrossEntropyLoss().cuda()

    steps = 0

    CUDNN.benchmark = True

    for epoch in range(args.start_epoch, args.epochs+1):

        # Every args.lr_step, changes learning rate by multipling args.lr_decay
        if args.step_learning:
            adjust_learning_rate(optimizer, epoch, args)

        # Load augmented data
        transformed_dataset = FaceIdPoseDataset(args.train_csv_file, args.data_place,
                                        transform = transforms.Compose([Resize((256,256)), RandomCrop((224,224))]))
        dataloader = DataLoader(transformed_dataset, batch_size = args.Train_Batch, shuffle=True)

        for i, batch_data in enumerate(dataloader):

            # backward() function accumulates gradients, however we don't want to mix up gradients between minibatches
            Model.zero_grad()
            batch_image = torch.FloatTensor(batch_data[0].float())
            batch_id_label = batch_data[2]

            if args.cuda:
                batch_image, batch_id_label = batch_image.cuda(), batch_id_label.cuda()

            batch_image, batch_id_label = Variable(batch_image), Variable(batch_id_label)

            steps += 1

            Prediction = Model(batch_image)
            Loss = loss_criterion(Prediction[:, :Nd], batch_id_label)
            Loss.backward()
            optimizer.step()
            log_learning(epoch, steps, 'VGG16_Model', args.lr, Loss.data[0], args)
            writer.add_scalar('Train/Train_Loss', Loss, steps)

        Validation_Process(Model, epoch, writer, args)

        if epoch % args.save_freq == 0:
            if not os.path.isdir(args.snapshot_dir): os.makedirs(args.snapshot_dir)
            save_path = os.path.join(args.snapshot_dir, 'epoch{}.pt'.format(epoch))
            torch.save(Model.state_dict(), save_path)
            save_checkpoint({
                'epoch': epoch + 1,
                'Model': Model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_dir=os.path.join(args.snapshot_dir, 'epoch{}'.format(epoch)))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()