from util.DataAugmentation import FaceIdPoseDataset#, Resize, RandomCrop
import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

def Validation_Process(Model, epoch, writer, args):

    # loss_criterion = nn.CrossEntropyLoss().cuda()
    # loss_criterion_Angular = AngleLoss().cuda()
    Model.eval()
    Nd = args.Nd
    AngleLoss = args.Angle_Loss
    print("Start Validating...")

    validation_dataset = FaceIdPoseDataset(args.val_csv_file,
                                            transform=transforms.Compose([transforms.Resize(256),
                                                                          transforms.CenterCrop(224),
                                                                          transforms.ToTensor()]))
    dataloader = DataLoader(validation_dataset, batch_size=args.Val_Batch, shuffle=False)

    ID_Real_Precision = []
    Losses = []

    for i, batch_data in enumerate(dataloader):  # all data in dataloader???
        batch_image = torch.FloatTensor(batch_data[0].float())
        batch_id_label = batch_data[2]


        if args.cuda:
            batch_image, batch_id_label = \
                batch_image.cuda(), batch_id_label.cuda()
        with torch.no_grad():
            batch_image, batch_id_label = Variable(batch_image), Variable(batch_id_label)

        Prediction = Model(batch_image)
        Loss = Model.ID_Loss(Prediction, batch_id_label)

        if AngleLoss:
            _, id_real_ans = torch.max(Prediction[0][:, :Nd], 1)
            id_real_precision = (id_real_ans == batch_id_label).type(torch.FloatTensor).sum() / Prediction[0].size()[0]
        else:
            _, id_real_ans = torch.max(Prediction[:, :Nd], 1)
            id_real_precision = (id_real_ans == batch_id_label).type(torch.FloatTensor).sum() / Prediction.size()[0]

        ID_Real_Precision.append(id_real_precision.data.cpu().numpy())
        Losses.append(Loss.cpu().data)

    ID_Real_Precisions = sum(ID_Real_Precision)/len(ID_Real_Precision)
    ID_Loss = sum(Losses)/len(Losses)
    print(">>> epoch: '{}'".format(epoch))
    print(">>> ID_Precision: '{}'".format(ID_Real_Precisions))
    print(">>> Validation_Loss: '{}'".format(ID_Loss))


    writer.add_scalar('Validation/ID_Precision', ID_Real_Precisions, epoch)
    writer.add_scalar('Validation/Validation_Loss', ID_Loss, epoch)


