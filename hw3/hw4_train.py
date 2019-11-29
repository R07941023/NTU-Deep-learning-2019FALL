import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from model import mini_AE, VAE, Autoencoder, ConvNet, Fully, TA_2018_Classifier, resnet152_1d, vgg19_1d, densenet161_1d, densenet201_1d
from data import get_dataloader, image_tensor_noise, check_tensor_img
from torchvision import models
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
import csv
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
import cv2

# del
# import torchsummary
# import matplotlib.pyplot as plt

def re_catelog(data, index):
    zero_index = ['']
    for i in range(len(index)):
        temp = np.argwhere(data == data[index[i]]).reshape(-1)
        # print(temp.shape)
        if zero_index[0] == '':
            zero_index = temp
        else:
            zero_index = np.hstack((zero_index, temp))
    zero_index.sort()
    data[zero_index] = 0
    data[data != 0] = 1
    return data

def VAE_loss(x, encoder, var, decoder):
    KLD = -0.5*torch.sum(1+var-encoder.pow(2)-var.exp())
    # BCE = nn.functional.binary_cross_entropy(decoder, x, size_average=False)
    BCE = nn.functional.mse_loss(decoder, x, size_average=False)
    return BCE+KLD

def view_data(o_data, data, n):
    o_data = o_data[n, :, :, :].detach().numpy()
    o_data = (np.transpose(o_data, (0, 2, 3, 1)) + 1) * 255 / 2
    o_data = o_data.astype(np.uint8)
    data = data[n, :, :, :].cpu().detach().numpy()
    data = (np.transpose(data, (0, 2, 3, 1)) + 1) * 255 / 2
    data = data.astype(np.uint8)
    loss = 0
    for i in range(data.shape[0]):
        cv2.imshow('origin' + str(n[i]), o_data[i])
        cv2.imshow('reconstruction' + str(n[i]), data[i])
        temp_loss = np.sum(np.abs(o_data[i]-data[i]))*2/255-1
        loss += temp_loss
    loss = loss/data.shape[0]
    print('loss = ', loss)
    cv2.waitKey(0)

# # test
# result = pd.read_csv('good1.csv').values
# result = result[:, 1]
# label = np.load('./trainY.npy')
# acc = 1 - (np.sum(np.abs(result - label)) / label.shape)[0]
# print('acc = ', acc)
# exit()


if __name__ == "__main__":
    # model=100, n=10, iter=100, acc = 0.79
    # ./model/VAE_hg_fp_rotate_gray_l2-e-4_lre-3/7835_model.pth
    # ./model/model1_gray_noise/537_model.pth
    pretrain = './100_model.pth'  # ./model/VAE_hg_fp_rotate_l2-e-4_lre-3/50_model.pth'
    data_set = 'hw4'
    output_path = './model/mini_AE_rgb/'  # model output: ./model/AE_rgb/
    first_decom = ''  # PCA/TSNE
    second_decom = 'PCA'  # PCA/TSNE
    cluster = 'KMeans'  # KMeans/GMM/BGM/Spectral
    denoise = ''
    n = [256, 256]
    class_ns = [6]  # [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    neighbors = [10]  # [3, 4, 5, 6]
    gammas = [100]  # [0.01, 0.1, 1, 10]
    l2_norm = 1e-4
    num_epoch = 20000
    batch_size = 3000
    optimizer_type = ['Adam', 1e-3]  # ['SGD', 0.01, 0.9], ['Adam', 0.0001]

    # hyperparameter
    output_n = 2
    val_proportion = 0.001
    dim = 3
    imgae_size = 32
    folder, model_type = sys.argv[1], 'mini_AE'  # sys.argv[1], sys.argv[2]  TA_2018_Classifier, resnet152_1d
    folder_csv = sys.argv[2]
    print(pretrain , first_decom, second_decom, cluster, model_type)

    # Get data loaders of training set and validation set
    train_set, val_set, train_loader, val_loader = get_dataloader( folder, data_set, imgae_size, batch_size, val_proportion, '' )
    # Specify the type of model
    if model_type == 'mini_AE':
        model = mini_AE()
    elif model_type == 'VAE':
        model = VAE(dim, imgae_size, pretrain)
    elif model_type == 'AE':
        model = Autoencoder(dim, imgae_size)
    elif model_type == 'conv':
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
    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    print('Model = ', model_type)
    # model parameter
    # torchsummary.summary(model, (dim, imgae_size, imgae_size))

    AE_loss = nn.L1Loss()  # nn.MSELoss()

    # pretrain
    if pretrain:
        model.load_state_dict(torch.load(pretrain)['model_state_dict'])
    else:
        # Set the type of gradient optimizer and the model it update
        if optimizer_type[0] == 'Adam':
            optimizer = optim.Adam( model.parameters(), lr=optimizer_type[1], weight_decay=l2_norm)
        elif optimizer_type[0] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=optimizer_type[1], momentum=optimizer_type[2])

        # pretrain
        if pretrain:
            optimizer.load_state_dict(torch.load(pretrain)['optimizer_state_dict'])
        best_loss = 1e10
        epoch_i = 1
        if pretrain:
            epoch_i = torch.load(pretrain)['epoch']
        record_type = 'w'
        if pretrain:
            record_type = 'a'
        with open(output_path + 'record.txt', record_type) as f:
            for epoch in range(epoch_i, num_epoch ):
                epoch_start_time = time.time()
                train_loss = 0.0
                model.train()
                for i, data in enumerate( train_loader ):
                    optimizer.zero_grad()
                    if denoise:
                        noise_data = image_tensor_noise(data)
                    else:
                        noise_data = data.clone()
                    # check_tensor_img(noise_data)
                    if model_type == 'VAE':
                        latent, var, train_pred = model( noise_data.cuda() )
                        batch_loss = VAE_loss(data.cuda(), latent, var, train_pred)
                    else:
                        latent, reconstruct = model(noise_data.cuda())
                        batch_loss = AE_loss(reconstruct, data.cuda())

                    batch_loss.backward()

                    optimizer.step()
                    train_loss += batch_loss.item()

                    progress = ('#' * int( float( i ) / len( train_loader ) * 40 )).ljust( 40 )
                    print( '[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, (time.time() - epoch_start_time), progress), end='\r', flush=True )


                train_loss = train_loss/train_set.__len__()
                print( '[%03d/%03d] %2.2f sec(s) Loss: %3.6f' % (epoch, num_epoch, time.time() - epoch_start_time, train_loss) )

                f.write(str(epoch) + ' ' + str(train_loss) + ' ''\n')
                if (train_loss < best_loss):
                    with open(output_path + 'acc.txt', 'w') as f1:
                        f1.write(str(i) + ' ' + str(epoch) + ' ' + str(train_loss) + '\n')
                    torch.save( {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss}, output_path+str(epoch)+'_model.pth' )
                    best_loss = train_loss
                    print( 'Model Saved!' )

    oupout = sys.argv[2]
    with open(oupout + 'acc.txt', 'w') as f7:
        f7.writelines(['time', 'acc'])


    for neighbor in neighbors:
        for gamma in gammas:
            model.cpu()
            for class_n in class_ns:
                print('class_n / neighbor / gammma = ', class_n, neighbor, gamma)
                # folder_csv = oupout+str(class_n)+'_.csv'
                latents = []
                for x in val_loader:
                    if model_type == 'VAE':
                        latent, var, reconstruct = model(x)
                    else:
                        latent, reconstruct = model(x)
                    # view_data(x, reconstruct, [0, 10, 20, 30])
                    latents.append(latent.cpu().detach().numpy())
                    latents = np.concatenate(latents, axis=0)
                    latents = latents.reshape([9000, -1])
                    latents_mean = np.mean(latents, axis=0)
                    latents_std = np.std(latents, axis=0)
                    latents = (latents - latents_mean) / latents_std
                print('AutoEncoder output = ', latents.shape[1])

                # first composition
                if first_decom == 'PCA':
                    latents = PCA(n_components=n[0], whiten=True).fit_transform(latents)
                elif first_decom == 'TSNE':
                    tsne = TSNE(n_components=3, perplexity=51)
                    X_tsne = tsne.fit_transform(latents)
                    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                    latents = (X_tsne - x_min) / (x_max - x_min)
                else:
                    pass
                print('first composition = ', latents.shape[1])

                # second composition
                if second_decom == 'PCA':
                    latents = PCA(n_components=n[1], whiten=True).fit_transform(latents)
                elif second_decom == 'TSNE':
                    tsne = TSNE(n_components=3, perplexity=51)
                    X_tsne = tsne.fit_transform(latents)
                    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                    latents = (X_tsne - x_min) / (x_max - x_min)
                else:
                    pass
                print('second composition = ', latents.shape[1])


                if cluster == 'KMeans':
                    result = KMeans(n_clusters=class_n, n_init=neighbor, max_iter=gamma).fit(latents).labels_
                elif cluster == 'BGM':
                    bgm = BGM(n_components=class_n).fit(latents)
                    result = bgm.predict(latents)
                elif cluster == 'Spectral':
                    s = SpectralClustering(n_clusters=class_n, n_neighbors=neighbor, gamma=gamma)  # n_neighbors=neighbor, gamma=gamma
                    result = s.fit_predict(latents)
                    # proba = s.predict_proba(latents)
                    # print(proba)s
                elif cluster == 'GMM':
                    gmm = GMM(n_components=class_n)
                    result = gmm.fit_predict(latents)
                if class_n != 2:
                    # re-catelog
                    result = re_catelog(result, [0, 1, 2, 3, 4])

                # CSV
                sum = 0
                with open(folder_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['id', 'label'])
                    for i in range(result.shape[0]):
                        value = int(result[i])
                        writer.writerow([i, value])
                        sum += value
                print('value = ', sum)

                # # cheat
                # label_cheat = np.load('./trainY.npy')
                # acc = 1 - (np.sum(np.abs(result - label_cheat)) / label_cheat.shape)[0]
                # print('acc = ', acc)
                # label_cheat = pd.read_excel('./label.xlsx').values
                # feature = np.insert(label_cheat[:, 0], 0, 0)
                # target = np.insert(label_cheat[:, 1], 0, 0)
                # zero_index = np.argwhere(target == 0).flatten()
                # pred_y = result[zero_index]
                # acc = (zero_index.shape[0]-np.sum(pred_y))/zero_index.shape[0]
                # print('acc = ', acc)

                # plt
                # plt.figure()
                # for i in range(latents.shape[0]):
                #     plt.text(latents[i, 0], latents[i, 1], str(label_cheat[i]), color=plt.cm.Set1(int(label_cheat[i])))
                # plt.xticks([])
                # plt.yticks([])
                # plt.show()

                # with open(oupout + 'acc.txt', 'a') as f7:
                #     f7.writelines([str(class_n), str(acc)+'\n'])
                # print('')







