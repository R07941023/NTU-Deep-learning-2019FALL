import os, sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from model import ConvNet, Fully, TA_2018_Classifier, resnet152_1d, vgg19_1d, densenet161_1d, densenet201_1d
from data import get_dataloader

if __name__ == "__main__":
    # download the model

    data_set = 'hw3_testing'
    output_path = './'
    output_n = 7
    imgae_size = 48
    model_n = 233
    dim = 1
    batch_size = 7000
    folder, model_type = sys.argv[1], 'densenet201_1d'  # data_path, model_type, output = sys.argv[1], sys.argv[2], sys.argv[3]

    # Get data loaders of training set and validation set
    test_set, test_loader = get_dataloader(folder, data_set, imgae_size, batch_size, '', '')

    if model_type == 'conv':
        model = ConvNet(output_n, imgae_size)
    elif model_type == 'fully':
        model = Fully(output_n, imgae_size)
    elif model_type == 'resnet152_1d':
        model = resnet152_1d()
    elif model_type == 'densenet161_1d':
        model = densenet161_1d()
    elif model_type == 'densenet201_1d':
        model = densenet201_1d(output_n, imgae_size, dim)
    elif model_type == 'vgg19_1d':
        model = vgg19_1d()
    elif model_type == 'TA_2018_Classifier':
        model = TA_2018_Classifier( output_n, imgae_size )
    print('Model = ', model_type)

    #######################################################################
    # Modifiy this part to load your trained model
    # TODO
    model.load_state_dict(torch.load(output_path+str(model_n)+'_model.pth')['model_state_dict'])
    # model.load_state_dict(torch.load(pretrain)['model_state_dict'])
    #######################################################################


    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    model.eval()
    # testing
    with torch.no_grad():
        for batch_idx, (x,name) in enumerate(test_loader):
            if use_cuda:
                x = x.cuda()
            out = model(x)
            _, pred_label = torch.max(out, 1)

    sum = 0
    with open(sys.argv[2], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(pred_label.shape[0]):
            value = int(pred_label[i])
            writer.writerow([i, value])
            sum += value
    print('value = ', sum)