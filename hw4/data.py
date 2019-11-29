import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
# from sklearn.model_selection import StratifiedKFold
import sys
import glob, os
from os import listdir
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator



def check_tensor_img(feature):
    feature = feature.numpy().copy()
    feature = (np.transpose(feature, (0, 2, 3, 1))+1)*255/2
    feature = feature.astype(np.uint8)
    cv2.imshow('1230', feature[0])
    cv2.waitKey(1000)
    wait()

def image_tensor_noise(feature):
    feature = feature.numpy().copy()
    feature = (np.transpose(feature, (0, 2, 3, 1)) + 1) * 255 / 2
    feature = feature.astype(np.uint8)
    new_feature = []
    if feature.shape[3] == 3:  # rgb
        for img in range(len(feature)):
            noise_img = cv2.randn(feature[img].copy(), (0, 0, 0), (255, 255, 255))+feature[img]
            new_feature.append(noise_img)
        new_feature = np.array(new_feature)
    elif feature.shape[3] == 1:  # gray
        for img in range(len(feature)):
            noise_img = cv2.randn(feature[img][:, :, 0].copy(), 0, 255) + feature[img][:, :, 0]
            new_feature.append(noise_img)
        new_feature = np.array(new_feature)
        new_feature = new_feature.reshape(new_feature.shape[0], new_feature.shape[1], new_feature.shape[2], 1)
    new_feature = np.transpose(new_feature, (0, 3, 1, 2)) / 255. * 2 - 1
    new_feature = torch.Tensor(new_feature)
    return new_feature

def image_generation_colormap(feature, label):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])

        gan = cv2.applyColorMap(feature[img].copy(), cv2.COLORMAP_JET)

        new_feature.append(gan)
        new_label.append(label[img])

        # cv2.imshow('img', gan)
        # cv2.waitKey(0)
    return new_feature, new_label

def image_generation_historgram(feature, label):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])

        ycrcb = cv2.cvtColor(feature[img], cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        gan = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, feature[img])
        # cv2.imshow('img', gan)
        # cv2.waitKey(0)

        new_feature.append(gan)
        new_label.append(label[img])

    return new_feature, new_label

def image_generation_threshold(feature, label):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])

        gray = cv2.cvtColor(feature[img], cv2.COLOR_BGR2GRAY)
        gan = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        gan = cv2.cvtColor(gan, cv2.COLOR_GRAY2BGR)

        new_feature.append(gan)
        new_label.append(label[img])

        # cv2.imshow('img', gan)
        # cv2.waitKey(0)
    return new_feature, new_label

def image_generation_GaussianBlur(feature, label, kernel):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])

        gan = cv2.GaussianBlur(feature[img].copy(), kernael, 0)

        new_feature.append(gan)
        new_label.append(label[img])

        # cv2.imshow('img', gan)
        # cv2.waitKey(0)
    return new_feature, new_label

def image_generation_keypoint(feature, label):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])

        gan = feature[img].copy()
        gray = cv2.cvtColor(feature[img], cv2.COLOR_BGR2GRAY)
        keypoint = cv2.goodFeaturesToTrack(gray, 40, 0.06, 10)
        if not keypoint is None:
            for i in keypoint:
                cv2.circle(gan, tuple(i[0]), 3, (0, 0, 255), 0)
        new_feature.append(gan)
        new_label.append(label[img])

        # cv2.imshow('img', gan)
        # cv2.waitKey(0)
    return new_feature, new_label

def image_generation_noise(feature, label):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])

        gan = cv2.randn(feature[img].copy(), (0,0,0), (255,255,255))
        gan = feature[img]+gan

        new_feature.append(gan)
        new_label.append(label[img])

        # cv2.imshow('img', gan)
        # cv2.waitKey(0)
    return new_feature, new_label

def image_generation_Affine(feature, label):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])
        pts1 = np.float32([[5, 5], [40, 5], [5, 40]])
        pts2 = np.float32([[0,0], [48, 0], [10, 40]])
        rows, cols = feature[img].shape[:2]
        M = cv2.getAffineTransform(pts1, pts2)
        gan = cv2.warpAffine(feature[img].copy(), M, (rows, cols))
        # gan = feature[img]+gan

        new_feature.append(gan)
        new_label.append(label[img])

        # cv2.imshow('img', gan)
        # cv2.waitKey(0)
    return new_feature, new_label

def image_generation_rotate(feature, rotate):
    new_feature = []
    for img in range(len(feature)):
        new_feature.append(feature[img])

        (h, w) = feature[img].shape[:2]
        center = (w / 2, h / 2)
        for i in rotate:
            M = cv2.getRotationMatrix2D(center, i, 1.0)
            gan = cv2.warpAffine(feature[img].copy(), M, (h, w))
            new_feature.append(gan)
            # print(gan.shape)
            # cv2.imwrite(str(i)+'_.jpg', gan)
        # exit()
        # cv2.imshow('img', gan)
        # cv2.waitKey(0)
    return new_feature

def image_generation_Flip(feature, type):
    new_feature = []
    for img in range(len(feature)):
        new_feature.append(feature[img])

        for i in type:
            gan = cv2.flip(feature[img], i)
            new_feature.append(gan)

    return new_feature

def image_generation_canny(feature, label):
    new_feature = []
    new_label = []
    for img in range(len(feature)):
        new_feature.append(feature[img])
        new_label.append(label[img])

        gan = cv2.Canny(feature[img].copy(), 100, 200)
        gan = cv2.cvtColor(gan, cv2.COLOR_GRAY2BGR)

        new_feature.append(gan)
        new_label.append(label[img])

        # cv2.imshow('img', gan)
        # cv2.waitKey(200)
    return new_feature, new_label


def image_gray(feature):
    new_feature = []
    for img in range(len(feature)):
        gray = cv2.cvtColor(feature[img], cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(gray.shape[0], gray.shape[1], 1)
        new_feature.append(gray)

    return new_feature

def get_dataloader(folder, dataset, image_scale, batch_size, val_proportion, folder_csv):
    # Data preprocessing
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

    if image_scale:
        trans = transforms.Compose([transforms.Grayscale(), transforms.Resize(image_scale), transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(image_scale-4), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    else:
        trans = transforms.Compose( [transforms.Grayscale(), transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(image_scale-4), transforms.ToTensor(), transforms.Normalize( (0.5,), (1.0,) )] )

    if dataset == 'hw4':
        data = np.load(folder)
        # image processing #
        train_set = []
        val_set = []
        print('Original...')
        for i in range(data.shape[0]):
            tmp = data[i]
            #
            # ycrcb = cv2.cvtColor(tmp, cv2.COLOR_BGR2YCR_CB)
            # channels = cv2.split(ycrcb)
            # cv2.equalizeHist(channels[0], channels[0])
            # cv2.merge(channels, ycrcb)
            # tmp = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, tmp)

            # tmp = cv2.resize(tmp, (image_scale, image_scale), interpolation=cv2.INTER_CUBIC)
            train_set.append(tmp)
            val_set.append(tmp)
        print('data number = ', len(train_set))

        # print('Flip...')
        # train_set = image_generation_Flip(train_set, [1])  # 1, 0, -1
        # print('data number = ', len(train_set))
        #
        # print('rotate...')
        # train_set = image_generation_rotate(train_set, [90, -90])  # 0/45/-45/90/-90/180
        # print('data number = ', len(train_set))

        # print('gray...')
        # train_set = image_gray(train_set)
        # val_set = image_gray(val_set)
        # print('data number = ', len(train_set))

        train_set = np.array(train_set)
        val_set = np.array(val_set)
        train_set = np.transpose(train_set, (0, 3, 1, 2)) / 255. *2 -1
        val_set = np.transpose(val_set, (0, 3, 1, 2)) / 255. * 2 - 1
        train_set = torch.Tensor(train_set)
        val_set = torch.Tensor(val_set)
        print('Data size = ', train_set.shape)
    elif dataset == 'hw4_gray':
        data = np.load(folder)
        # image processing #
        train_set = []
        for i in range(data.shape[0]):
            tmp = cv2.resize(data[i], (image_scale, image_scale), interpolation=cv2.INTER_CUBIC)
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            train_set.append(tmp)
        train_set = np.array(train_set)
        train_set = train_set.reshape(train_set.shape[0], train_set.shape[1], train_set.shape[2], 1)
        train_set = np.transpose(train_set, (0, 3, 1, 2)) / 255. *2 -1
        val_set = train_set.copy()
        train_set = torch.Tensor(train_set)
        val_set = torch.Tensor(val_set)
        print('Data size = ', train_set.shape)
    elif dataset == 'hw4_canny':
        data = np.load(folder)
        # image processing #
        train_set = []
        for i in range(data.shape[0]):
            tmp = data[i].copy()
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            tmp = cv2.Canny(tmp, 100, 200)
            tmp = cv2.resize(tmp, (image_scale, image_scale), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite('./read_img/img_canny/'+str(i)+'.jpg', tmp)
            train_set.append(tmp)
        train_set = np.array(train_set)
        train_set = train_set.reshape(train_set.shape[0], train_set.shape[1], train_set.shape[2], 1)
        train_set = np.transpose(train_set, (0, 3, 1, 2)) / 255. *2 -1
        val_set = train_set.copy()
        train_set = torch.Tensor(train_set)
        val_set = torch.Tensor(val_set)
        print('Data size = ', train_set.shape)


    elif dataset == 'cv_hw2_p1':
        def hw_rule(x):
            data = []
            normailize = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            x = np.array(x)
            for i in range(x.shape[0]):
                data.append(normailize(x[i]).numpy())
            data = np.array(data)
            data = torch.tensor(data)
            return data
        # hw rule
        kind = 10
        x_train = []
        x_label = []
        val_data = []
        val_label = []
        # training date
        for i in range(kind):
            print('class', str(i), '...')
            x_data = '../../CV/hw2/hw2-4_data/problem1/train/class_' + str(i) + '/'
            imgs = listdir(x_data)
            imgs = sorted(imgs)
            label = i
            drop_label = []
            for img in range(len(imgs)):  # len(imgs)
                if not img in drop_label:
                    tmp = cv2.imread(x_data + imgs[img])
                    tmp = cv2.resize(tmp, (image_scale, image_scale), interpolation=cv2.INTER_CUBIC)
                    # preprocessing
                    ycrcb = cv2.cvtColor(tmp, cv2.COLOR_BGR2YCR_CB)
                    channels = cv2.split(ycrcb)
                    cv2.equalizeHist(channels[0], channels[0])
                    cv2.merge(channels, ycrcb)
                    tmp = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, tmp)
                    tmp = cv2.GaussianBlur(tmp, (3, 3), 0)
                    # load the data
                    x_train.append(tmp)
                    x_label.append(label)

            print('training data number = ', len(x_train))
        # validation date
        for i in range(kind):
            print('class', str(i), '...')
            x_data = '../../CV/hw2/hw2-4_data/problem1/valid/class_' + str(i) + '/'
            imgs = listdir(x_data)
            imgs = sorted(imgs)
            label = i
            drop_label = []
            for img in range(len(imgs)):  # len(imgs)
                if not img in drop_label:
                    tmp = cv2.imread(x_data + imgs[img])
                    tmp = cv2.resize(tmp, (image_scale, image_scale), interpolation=cv2.INTER_CUBIC)
                    # preprocessing
                    ycrcb = cv2.cvtColor(tmp, cv2.COLOR_BGR2YCR_CB)
                    channels = cv2.split(ycrcb)
                    cv2.equalizeHist(channels[0], channels[0])
                    cv2.merge(channels, ycrcb)
                    tmp = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, tmp)
                    tmp = cv2.GaussianBlur(tmp, (3, 3), 0)
                    # load the data
                    val_data.append(tmp)
                    val_label.append(label)
            print('validation data number = ', len(val_data))

        # hw rule
        x_train = hw_rule(x_train)
        val_data = hw_rule(val_data)

        # list to np
        val_label = np.array(val_label, dtype=float)
        val_label = np.array(val_label, dtype=float)

        # np to tensor
        x_label = torch.LongTensor(x_label)
        val_label = torch.LongTensor(val_label)

        # dataset
        train_set = TensorDataset(x_train, x_label)
        val_set = TensorDataset(val_data, val_label)


    elif dataset == 'hw3':
        # face_file = 'face.xml'
        # face_web = 'https://raw.githubusercontent.com/a1996850622/FaceDetection/master/Source/haarcascade_frontalface_default.xml'
        # urlretrieve(face_web, face_file)
        # face_cascade = cv2.CascadeClassifier(face_file)
        label = pd.read_csv( folder_csv ).values[:, 1]
        x_data = folder+'/'
        imgs = listdir(x_data)
        # imgs = imgs.sort()
        imgs = sorted(imgs)
        x_train = []
        x_label = []
        val_data = []
        val_label = []
        # tmp = cv2.bilateralFilter(tmp, 9, 75, 75)
        print('original...')
        drop_label = [6, 177, 774, 1039, 2018, 3658, 3968, 3991, 4002, 5574, 5839, 5855, 6923, 6940, 7070, 7134, 7415, 7497, 7902, 8601, 8923, 10762, 10807, 11143, 11468, 12230, 12353, 12874, 13593, 14769, 16101, 16800, 17043, 17389, 17875, 18065, 18087, 18221, 18584, 18991, 19582, 20568, 23043, 24111, 24938, 24978, 25340, 25457, 25797, 26054, 26466, 26542, 26546, 27100, 27424, 27586, 28585, 28601]  # ~20000
        for img in range(len(imgs)):  # len(imgs)
            if not img in drop_label:
                tmp = cv2.imread(x_data + imgs[img])
                # preprocessing
                ycrcb = cv2.cvtColor(tmp, cv2.COLOR_BGR2YCR_CB)
                channels = cv2.split(ycrcb)
                cv2.equalizeHist(channels[0], channels[0])
                cv2.merge(channels, ycrcb)
                tmp = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, tmp)
                tmp = cv2.GaussianBlur(tmp, (3, 3), 0)


                if (img % (1 / val_proportion) == 0):
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                    val_label.append(label[img])
                    val_data.append(tmp)
                else:
                    # origin
                    x_train.append(tmp)
                    x_label.append(label[img])
        print('data number = ', len(x_train))

        # print('canny...')
        # x_train, x_label = image_generation_canny(x_train, x_label)  # 1, 0, -1
        # print('data number = ', len(x_train))

        # print('keypoint...')
        # x_train, x_label = image_generation_keypoint(x_train, x_label)  # for keypoint
        # print('data number = ', len(x_train))

        # print('Affine...')
        # x_train, x_label = image_generation_Affine(x_train, x_label)
        # print('data number = ', len(x_train))

        # print('Flip...')
        # x_train, x_label = image_generation_Flip(x_train, x_label, [1])  # 1, 0, -1
        # print('data number = ', len(x_train))
        #
        # print('rotate...')
        # x_train, x_label = image_generation_rotate(x_train, x_label, [45, -45, 90, -90])  # 0/45/-45/90/-90/180
        # print('data number = ', len(x_train))

        # print('noise...')
        # x_train, x_label = image_generation_noise(x_train, x_label)
        # print('data number = ', len(x_train))

        # print('GaussianBlur...')
        # x_train, x_label = image_generation_GaussianBlur(x_train, x_label, (3, 3))
        # print('data number = ', len(x_train))

        # print('colormap...')
        # x_train, x_label = image_generation_colormap(x_train, x_label)
        # print('data number = ', len(x_train))

        # print('threshold...')
        # x_train, x_label = image_generation_threshold(x_train, x_label)
        # print('data number = ', len(x_train))

        print('gray...')
        x_train, x_label = image_gray(x_train, x_label)
        print('data number = ', len(x_train))

        # list to np
        x_train = np.array(x_train, dtype=float) / 255.0
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        x_label = np.array( x_label, dtype=float )
        # datagen.fit(x_train)
        # generator = datagen.flow(x_train, x_label)
        # print(x_train.shape)
        # print('generator...')
        # x_train = np.concatenate((x_train, generator[0][0], generator[1][0]))
        # print('data number = ', x_train.shape[0])
        # x_label = np.concatenate((x_label, generator[0][1], generator[1][1]))
        # splits = list(StratifiedKFold(n_splits=5, shuffle=True).split(x_train, x_label))
        # print(splits[:3])
        # exit()

        val_data = np.array( val_data, dtype=float ) / 255.0
        val_data = val_data.reshape(val_data.shape[0], 1, val_data.shape[1], val_data.shape[2])
        val_label = np.array( val_label, dtype=float )
        # np to tensor
        x_train = torch.FloatTensor( x_train )
        val_data = torch.FloatTensor( val_data )
        x_label = torch.LongTensor( x_label )
        val_label = torch.LongTensor( val_label )
        # dataset
        train_set = TensorDataset( x_train, x_label )
        val_set = TensorDataset( val_data, val_label )

    elif dataset == 'hw3_testing':
        x_data = folder + '/'
        imgs = listdir(x_data)
        imgs = sorted(imgs)
        x_train = []
        x_zero = []
        for img in range(len(imgs)):  # len(imgs)
            tmp = cv2.imread(x_data + imgs[img])

            # preprocessing
            ycrcb = cv2.cvtColor(tmp, cv2.COLOR_BGR2YCR_CB)
            channels = cv2.split(ycrcb)
            cv2.equalizeHist(channels[0], channels[0])
            cv2.merge(channels, ycrcb)
            tmp = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, tmp)
            tmp = cv2.GaussianBlur(tmp, (3, 3), 0)

            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])

            x_train.append(tmp)
            x_zero.append(img)
        # list to np
        x_train = np.array(x_train, dtype=float) / 255.0
        x_zero = np.array(x_zero)
        # np to tensor
        x_train = torch.FloatTensor(x_train)
        x_zero = torch.FloatTensor(x_zero)
        # dataset
        train_set = TensorDataset(x_train, x_zero)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
        print('==>>> total trainning batch number: {}'.format(len(train_loader)))
        return train_set, train_loader

    else:
        print('Custom data...')
        train_path, test_path = os.path.join(folder,'train'), os.path.join(folder,'valid')
        # print(train_path)
        #

        # Get dataset using pytorch functions
        train_set = ImageFolder(root='./data/train/00000', transform=trans)
        # test_set =  ImageFolder(test_path,  transform=trans)
        exit()

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    val_loader  = torch.utils.data.DataLoader(dataset=val_set,  batch_size=9000, shuffle=False, pin_memory=True, drop_last=False)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total validation batch number: {}'.format(len(val_loader)))
    return train_set, val_set, train_loader, val_loader