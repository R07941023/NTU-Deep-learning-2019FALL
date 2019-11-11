import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from model import ConvNet, Fully, TA_2018_Classifier, resnet152_1d, vgg19_1d, densenet161_1d, densenet201_1d
from data import get_dataloader
from torchvision import models
import numpy as np
import os

if __name__ == "__main__":

    # Specifiy data folder path and model type(fully/conv)
    pretrain = ''
    data_set = 'hw3'
    output_path = './'
    output_n = 7
    val_proportion = 0.001
    dim = 1
    l2_norm = 1e-4
    imgae_size = 48
    num_epoch = 200000
    batch_size = 500
    optimizer_type = ['Adam', 0.0001]  # ['SGD', 0.01, 0.9], ['Adam', 0.0001]
    folder, model_type = sys.argv[1], 'densenet201_1d'  # sys.argv[1], sys.argv[2]  TA_2018_Classifier, resnet152_1d
    folder_csv = sys.argv[2]

    # Get data loaders of training set and validation set
    train_set, val_set, train_loader, val_loader = get_dataloader( folder, data_set, imgae_size, batch_size, val_proportion, folder_csv )

    # test_x = torch.utils.data.PackedSequencer(test_loader)
    # print(test_x)
    # exit()
    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet(output_n, dim, imgae_size)
    elif model_type == 'fully':
        model = Fully(output_n, imgae_size)
    elif model_type == 'densenet161_1d':
        model = densenet161_1d()
    elif model_type == 'vgg19_1d':
        model = vgg19_1d()
    elif model_type == 'resnet152_1d':
        model = resnet152_1d()
    elif model_type == 'densenet201_1d':
        model = densenet201_1d(output_n, imgae_size, dim)
    elif model_type == 'TA_2018_Classifier':
        model = TA_2018_Classifier( output_n, imgae_size )
    print('Model = ', model_type)

    # pretrain
    if pretrain:
        model.load_state_dict(torch.load(pretrain)['model_state_dict'])

    # Set the type of gradient optimizer and the model it update
    if optimizer_type[0] == 'Adam':
        optimizer = optim.Adam( model.parameters(), lr=optimizer_type[1], weight_decay=l2_norm)
    elif optimizer_type[0] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer_type[1], momentum=optimizer_type[2])


    # Choose loss function
    loss = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    # pretrain
    if pretrain:
        optimizer.load_state_dict(torch.load(pretrain)['optimizer_state_dict'])
    best_acc = 0.0
    epoch_i = 1
    if pretrain:
        epoch_i = torch.load(pretrain)['epoch']
    record_type = 'w'
    if pretrain:
        record_type = 'a'
    with open(output_path + 'record.txt', record_type) as f:
        for epoch in range(epoch_i, num_epoch ):
            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            model.train()
            for i, data in enumerate( train_loader ):
                optimizer.zero_grad()
                # gray to rgb
                # x = x.repeat( 1, 3, 1, 1 )

                train_pred = model( data[0].cuda() )
                batch_loss = loss( train_pred, data[1].cuda() )
                batch_loss.backward()
                optimizer.step()

                train_acc += np.sum( np.argmax( train_pred.cpu().data.numpy(), axis=1 ) == data[1].numpy() )
                train_loss += batch_loss.item()

                progress = ('#' * int( float( i ) / len( train_loader ) * 40 )).ljust( 40 )
                print( '[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, (time.time() - epoch_start_time), progress), end='\r', flush=True )


            model.eval()
            for i, data in enumerate( val_loader ):
                val_pred = model( data[0].cuda() )
                batch_loss = loss( val_pred, data[1].cuda() )

                val_acc += np.sum( np.argmax( val_pred.cpu().data.numpy(), axis=1 ) == data[1].numpy() )
                val_loss += batch_loss.item()

                progress = ('#' * int( float( i ) / len( val_loader ) * 40 )).ljust( 40 )
                print( '[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, (time.time() - epoch_start_time), progress), end='\r', flush=True )
            train_acc = train_acc / train_set.__len__()
            train_loss = train_loss/train_set.__len__()*1000
            val_acc = val_acc / val_set.__len__()
            val_loss = val_loss/val_set.__len__()*1000
            print( '[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (epoch, num_epoch, time.time() - epoch_start_time, train_acc, train_loss, val_acc, val_loss) )

            f.write(str(epoch) + ' ' + str(train_loss) + ' ' + str(val_loss) + ' ' + str(train_acc) + ' ' + str(val_acc) + '\n')
            if (val_acc > best_acc):
                with open(output_path + 'acc.txt', 'w') as f1:
                    f1.write(str(i) + ' ' + str(epoch) + ' ' + str(val_acc) + '\n')
                torch.save( {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': val_loss, 'acc': val_acc}, output_path+str(epoch)+'_model.pth' )
                best_acc = val_acc
                print( 'Model Saved!' )

    # # Run any number of epochs you want
    # ep = 10
    # for epoch in range(ep):
    #     print('Epoch:', epoch)
    #     ##############
    #     ## Training ##
    #     ##############
    #
    #     # Record the information of correct prediction and loss
    #     correct_cnt, total_loss, total_cnt = 0, 0, 0
    #
    #     # Load batch data from dataloader
    #     for batch, (x, label) in enumerate(train_loader,1):
    #
    #         # gray to rgb
    #         # x = x.repeat( 1, 3, 1, 1 )
    #
    #         # Set the gradients to zero (left by previous iteration)
    #         optimizer.zero_grad()
    #         # Put input tensor to GPU if it's available
    #         if use_cuda:
    #             x, label = x.cuda(), label.cuda()
    #         # Forward input tensor through your model
    #         out = model(x)
    #         # Calculate loss
    #         loss = criterion(out, label)
    #         # Compute gradient of each model parameters base on calculated loss
    #         loss.backward()
    #         # Update model parameters using optimizer and gradients
    #         optimizer.step()
    #
    #         # Calculate the training loss and accuracy of each iteration
    #         total_loss += loss.item()
    #         _, pred_label = torch.max(out, 1)
    #         total_cnt += x.size(0)
    #         correct_cnt += (pred_label == label).sum().item()
    #
    #         # Show the training information
    #         if batch % 500 == 0 or batch == len(train_loader):
    #             acc = correct_cnt / total_cnt
    #             ave_loss = total_loss / batch
    #             print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
    #                 batch, ave_loss, acc))
    #
    #     ################
    #     ## Validation ##
    #     ################
    #     model.eval()
    #     vali_y = model( vali_x )
    #     print(vali_y)
    #     exit()
    #     model.train()
    #
    # # Save trained model
    # torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())
    #
    # # Plot Learning Curve
    # # TODO