from util.DataAugmentation import FaceIdPoseDataset#, Resize, RandomCrop
import torch
import os
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from util.SaveFeature import SaveFeature
from torch.utils.data import DataLoader

def Test_Process(Model, args):

    save_dir = '{}/{}/Feature'.format(args.output, args.snapshot)

    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    if args.cuda:
        Model.cuda()

    loss_criterion = nn.CrossEntropyLoss().cuda()
    Model.eval()
    Nd = args.Nd
    count = 0
    print("Start Testing...")

    test_dataset = FaceIdPoseDataset(args.test_csv_file,
                                            transform=transforms.Compose([transforms.Resize(256),
                                                                          transforms.CenterCrop(224),
                                                                          transforms.ToTensor()]))
    dataloader = DataLoader(test_dataset, batch_size=args.Test_Batch, shuffle=False)

    ID_Real_Precision = []
    Losses = []

    print(len(dataloader))


    for i, batch_data in enumerate(dataloader):  # all data in dataloader???
        batch_image = torch.FloatTensor(batch_data[0].float())
        minibatch_size = len(batch_image)
        batch_id_label = batch_data[2]


        if args.cuda:
            batch_image, batch_id_label = \
                batch_image.cuda(), batch_id_label.cuda()
        with torch.no_grad():
            batch_image, batch_id_label = Variable(batch_image), Variable(batch_id_label)

        batchImageName = batch_data[1]
        Prediction = Model(batch_image)


        Loss = loss_criterion(Prediction[:, :Nd], batch_id_label)
        _, id_real_ans = torch.max(Prediction[:, :Nd], 1)
        id_real_precision = (id_real_ans == batch_id_label).type(torch.FloatTensor).sum() / Prediction.size()[0]
        ID_Real_Precision.append(id_real_precision.data.cpu().numpy())
        Losses.append(Loss.cpu().data)
        count += minibatch_size
        print("Finish Processing {} images...".format(count))

    ID_Real_Precisions = sum(ID_Real_Precision)/len(ID_Real_Precision)

    print(">>> Accuracy: {} %".format(ID_Real_Precisions*100))







