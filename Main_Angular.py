#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tensorboardX import SummaryWriter
from util.log_learning import log_learning
from util.DataAugmentation import FaceIdPoseDataset
from util.exp_lr_scheduler import adjust_learning_rate
from util.Validation import Validation_Process
from util.checkpoint import save_checkpoint
import torch.backends.cudnn as CUDNN
from model.VGG16_Model import VGG16
from Extract_Feature import Extract_Feature
from vggface import vgg_face_dag as vggface


def Train(Model, args):

    Nd = args.Nd
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2

    if args.cuda:
        Model.cuda()

    optimizer = optim.Adam(Model.parameters(), lr=args.lr, betas=(beta1_Adam, beta2_Adam))
    #optimizer = optim.SGD(Model.parameters(), lr=args.lr)
    Model.train()
    steps = 0
    CUDNN.benchmark = True

    for epoch in range(args.start_epoch, args.epochs+1):

        if args.step_learning:
            adjust_learning_rate(optimizer, epoch, args)

        transformed_dataset = FaceIdPoseDataset(args.train_csv_file, transform=transforms.Compose(
            [transforms.Resize(256),
             transforms.RandomCrop(224),
             transforms.ToTensor()]))
        dataloader = DataLoader(transformed_dataset, batch_size=args.Train_Batch, shuffle=True)

        for i, batch_data in enumerate(dataloader):
            Model.zero_grad()
            batch_image = torch.FloatTensor(batch_data[0].float())
            batch_id_label = batch_data[2]
            if args.cuda:
                batch_image, batch_id_label = batch_image.cuda(), batch_id_label.cuda()
            batch_image, batch_id_label = Variable(batch_image), Variable(batch_id_label)

            steps += 1

            Prediction = Model(batch_image)
            Loss = Model.ID_Loss(Prediction, batch_id_label)

            Loss.backward()
            optimizer.step()
            log_learning(epoch, steps, 'VGG16_Model', args.lr, Loss.data, args)
            writer.add_scalar('Train/Train_Loss', Loss, steps)
            # Validation_Process(Model, epoch, writer, args)
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


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning & saving parameterss
    parser.add_argument('-train', action='store_true', default=False,
                        help='Generate pose modified image from given image')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-step-learning', action='store_true', default=False, help='enable lr step learning')
    parser.add_argument('-lr-decay', type=float, default=0.1, help='initial decay learning rate [default: 0.1]')
    parser.add_argument('-lr-step', type=int, default=35, help='Set Step to change lr by multiply lr-decay thru every lr-step epoch [default: 35]')
    parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 1000]')
    parser.add_argument('-Train-Batch', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-Val-Batch', type=int, default=32, help='batch size for training [default: 4]')
    parser.add_argument('-Test-Batch', type=int, default=32, help='batch size for training [default: 64]')
    parser.add_argument('-snapshot-dir', type=str, default='snapshot', help='where to save the snapshot while training')
    parser.add_argument('-save-freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('-start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    # data souce
    parser.add_argument('-data-place', type=str, default=None, help='prepared data path to run program')
    parser.add_argument('-output', type=str, default='Output', help='Output path for features')
    parser.add_argument('-train-csv-file', type=str, default=None, help='csv file to load image for training')
    parser.add_argument('-val-csv-file', type=str, default=None, help='csv file to load image for validation')
    parser.add_argument('-test-csv-file', type=str, default=None, help='csv file to load image for test')
    parser.add_argument('-Nd', type=int, default=10, help='initial Number of ID [default: 188]')
    parser.add_argument('-Channel', type=int, default=3, help='initial Number of Channel [default: 3 (RGB Three Channel)]')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    parser.add_argument('-test', action='store_true', default=None, help='Generate pose modified image from given image')
    parser.add_argument('-resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-Angle-Loss', action='store_true', default=False, help='Use Angle Loss')
    parser.add_argument('-pretrain', action='store_true', default=False)
    parser.add_argument('-vggface', action='store_true', default=False)

    args = parser.parse_args()
    writer = SummaryWriter()

    # update args and print
    if args.train:
        args.snapshot_dir = os.path.join(args.snapshot_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) #拼接路徑
        os.makedirs(args.snapshot_dir)

    if args.test:
        if args.snapshot is None:
            print(">>> Sorry, please set snapshot path while extracting features")
            exit()
        else:
            print('\n>>> Loading model from [%s]...' % args.snapshot)
                      
            if args.vggface:
              Model=vggface('/content/DSS_Loss_Function/vgg_face_dag.pth')
            else :
              checkpoint = torch.load('{}_checkpoint.pth.tar'.format(args.snapshot)) 
              Model = VGG16(args) 
              Model.load_state_dict(checkpoint['Model'])

            Extract_Feature(Model, args)

    elif args.train:
        print("Parameters:")
        for attr, value in sorted(args.__dict__.items()):
            text = "\t{}={}\n".format(attr.upper(), value)
            print(text)
            with open('{}/Parameters.txt'.format(args.snapshot_dir), 'a') as f:
                f.write(text)

        if args.train_csv_file is None or args.val_csv_file is None:
            print(">>> Sorry, please set csv-file for your training/validation data")
            exit()

        else:
            Model = VGG16(args)
            print(Model)
            Train(Model, args)


