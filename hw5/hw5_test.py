import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import mini_LSTM, bi_LSTM, Fully
from data import get_dataloader
from torchvision import models
import numpy as np
import os
import csv
import pandas as pd
# del
# import torchsummary
# import matplotlib.pyplot as plt

def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

if __name__ == "__main__":
    pretrain = './good.pth'  # ./model/bi_LSTM/case2/36_model.pth
    data_set = 'hw5_training_em'  # hw5_training / hw5_testing
    output_path = './'  # model output: ./model/AE_rgb/
    optimizer_type = ['Adam', 1e-4]  # ['SGD', 0.01, 0.9], ['Adam', 0.0001]
    l2_norm = 5e-4
    num_epoch = 2000000
    batch_size = 200
    train_proportion = 0.9
    seed = ''

    # LSTM
    num_layers = 5
    hidden_dim = 200  # 200  220:0.783

    # embed
    embed_dim = 300
    seq_len = 35
    wndw_size = 3
    word_cnt = 1
    word_iter = 50
    work_worker = 8

    folder_X, folder_Y, model_type = '', '', 'bi_LSTM'  # sys.argv[1], sys.argv[2]  TA_2018_Classifier, resnet152_1d
    testing_csv_path = sys.argv[1]
    folder_csv = sys.argv[2]
    # Get data loaders of training set and validation set
    # train_set, val_set, train_loader, val_loader, vectors = get_dataloader( folder_X, folder_Y, data_set, train_proportion, batch_size, seed, embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len)
    test_loader, vectors = get_dataloader( folder_X, folder_Y, 'hw5_testing_token', train_proportion, batch_size, seed, embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len, testing_csv_path)
    # Specify the type of model
    if model_type == 'mini_LSTM':
        model = mini_LSTM(vectors, embed_dim, hidden_dim, num_layers)
    elif model_type == 'bi_LSTM':
        model = bi_LSTM(vectors, embed_dim, hidden_dim, num_layers)
    elif model_type == 'Fully':
        model = Fully(vectors)
    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    print('Model = ', model_type)
    # model parameter
    # torchsummary.summary(model, (dim, imgae_size, imgae_size))

    criterion = nn.BCELoss()

    with open(output_path + 'model_setting.txt', 'w') as f1:
        f1.write('model = ' + str(model_type) + '\n')
        f1.write('optimalize = ' + str(optimizer_type) + '\n')
        f1.write('l2_norm = ' + str(l2_norm) + '\n')
        f1.write('batch_size = ' + str(batch_size) + '\n')
        f1.write('train_proportion = ' + str(train_proportion) + '\n')
        f1.write('seed = ' + str(seed) + '\n')
        f1.write('num_layers = ' + str(num_layers) + '\n')
        f1.write('hidden_dim = ' + str(hidden_dim) + '\n')
        f1.write('embed_dim = ' + str(embed_dim) + '\n')
        f1.write('seq_len = ' + str(seq_len) + '\n')
        f1.write('wndw_size = ' + str(wndw_size) + '\n')
        f1.write('word_cnt = ' + str(word_cnt) + '\n')
        f1.write('word_iter = ' + str(word_iter) + '\n')
        f1.write('work_worker = ' + str(work_worker) + '\n')
    # pretrain
    if pretrain:
        model.load_state_dict(torch.load(pretrain)['model_state_dict'])
        model.eval()
        for i, data in enumerate(test_loader):
            outputs = model(data[0].cuda())
            outputs = outputs.cpu().squeeze()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0

        # CSV
        sum = 0
        with open(folder_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'label'])
            for i in range(outputs.shape[0]):
                value = int(outputs[i])
                writer.writerow([i, value])
                sum += value
        print('value = ', sum)

    else:
        # Set the type of gradient optimizer and the model it update
        if optimizer_type[0] == 'Adam':
            optimizer = optim.Adam( model.parameters(), lr=optimizer_type[1], weight_decay=l2_norm)
        elif optimizer_type[0] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=optimizer_type[1], momentum=optimizer_type[2])

        best_acc = 0.0
        epoch_i = 1
        record_type = 'w'
        if pretrain:
            record_type = 'a'
        with open(output_path + 'record.txt', record_type) as f:
            for epoch in range(epoch_i, num_epoch ):
                epoch_start_time = time.time()
                train_loss, train_acc, val_acc, val_loss = 0.0, 0.0, 0.0, 0.0
                model.train()
                for i, data in enumerate( train_loader ):
                    optimizer.zero_grad()
                    # check_tensor_img(noise_data)
                    outputs = model(data[0].cuda())
                    outputs = outputs.squeeze()
                    batch_loss = criterion(outputs, data[1].float().cuda())
                    batch_loss.backward()
                    optimizer.step()
                    train_loss += batch_loss.item()
                    train_acc += evaluation(outputs, data[1].float().cuda())
                    progress = ('#' * int( float( i ) / len( train_loader ) * 40 )).ljust( 40 )
                    print( '[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, (time.time() - epoch_start_time), progress), end='\r', flush=True )

                model.eval()
                for i, data in enumerate(val_loader):
                    outputs = model(data[0].cuda())
                    outputs = outputs.squeeze()
                    batch_loss = criterion(outputs, data[1].float().cuda())
                    val_loss += batch_loss.item()
                    val_acc += evaluation(outputs, data[1].float().cuda())

                    progress = ('#' * int(float(i) / len(val_loader) * 40)).ljust(40)
                print('[%03d/%03d] %2.2f sec(s) | %s |' % (
                epoch + 1, num_epoch, (time.time() - epoch_start_time), progress), end='\r', flush=True)
                train_acc = train_acc / train_set.__len__()
                train_loss = train_loss / train_set.__len__() * 1000
                val_acc = val_acc / val_set.__len__()
                val_loss = val_loss / val_set.__len__() * 1000
                print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (
                epoch, num_epoch, time.time() - epoch_start_time, train_acc, train_loss, val_acc, val_loss))

                f.write(str(epoch) + ' ' + str(train_loss) + ' ' + str(val_loss) + ' ' + str(train_acc) + ' ' + str(
                    val_acc) + '\n')
                if (val_acc > best_acc):
                    with open(output_path + 'acc.txt', 'w') as f1:
                        f1.write(str(i) + ' ' + str(epoch) + ' ' + str(val_acc) + '\n')
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(), 'loss': val_loss, 'acc': val_acc},
                               output_path + str(epoch) + '_model.pth')
                    best_acc = val_acc
                    print('Model Saved!')

                    for i, data in enumerate(test_loader):
                        outputs = model(data[0].cuda())
                        outputs = outputs.cpu().squeeze()
                        outputs[outputs >= 0.5] = 1
                        outputs[outputs < 0.5] = 0
                    # CSV
                    sum = 0
                    with open(folder_csv, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['id', 'label'])
                        for i in range(outputs.shape[0]):
                            value = int(outputs[i])
                            writer.writerow([i, value])
                            sum += value
                    print('value = ', sum)





