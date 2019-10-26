import pandas as pd
import numpy as np
import csv
import sys

# best
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_selection import SelectFromModel

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets import mnist, cifar10

# ???
import pickle
import warnings
import math
from itertools import combinations, permutations




class tool():

    def cross_validation(x, y ,validation_n, seed):
        if seed:
            np.random.seed(seed)
        times = int(len(y)*validation_n/100)
        mask = np.random.choice(len(x), len(x), replace=False)[:times]
        train_x = x.copy()
        train_y = y.copy()
        vali_x = np.zeros((mask.shape[0], x.shape[1]))
        vali_y = np.zeros((mask.shape[0], 1))

        train_x = np.delete(train_x, mask, axis=0)
        train_y = np.delete(train_y, mask, axis=0)
        for i in range(len(mask)):
            vali_x[i] = x[mask[i]]
            vali_y[i] = y[mask[i]]
        return train_x, train_y, vali_x, vali_y

    def normalize(index, data):

        x_all = data
        mean = np.mean(x_all, axis=0)
        std = np.std(x_all, axis=0)

        mean_vec = np.zeros(x_all.shape[1])
        std_vec = np.ones(x_all.shape[1])
        mean_vec[index] = mean[index]
        std_vec[index] = std[index]

        x_all_nor = (x_all - mean_vec) / std_vec

        return x_all_nor

    def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

class data_loder():

    def TA_hw2(feature, data_train_X, data_train_Y, data_test_X, nor_index):
        # 讀檔如果像這樣把路徑寫死交到github上去會馬上死去喔
        # 還不知道怎寫請參考上面的連結
        x_train = pd.read_csv(data_train_X)
        x_test = pd.read_csv(data_test_X)

        x_train = tool.normalize(nor_index, x_train.values)[:, feature]
        x_test = tool.normalize(nor_index, x_test.values)[:, feature]

        y_train = pd.read_csv(data_train_Y, header=None)
        y_train = y_train.values
        y_train = y_train.reshape(-1)

        return x_train, y_train, x_test

    def hw2_test(feature, data_test_X, nor_index):
        x_test = pd.read_csv(data_test_X)
        x_test = tool.normalize(nor_index, x_test.values)[:, feature]
        return x_test


    def hw2(train_X_path, train_Y_path, number, model, feature, drops, train_proportion, fix_random, seed):
        # download the data

        filepath_train_X = train_X_path
        filepath_train_Y = train_Y_path
        X_df = pd.read_csv( filepath_train_X, encoding='utf-8' )
        Y_df = pd.read_csv( filepath_train_Y, encoding='utf-8' )


        all_df = pd.concat([Y_df, X_df], axis=1)

        # drop the bad feature (optional)
        df = all_df.copy()
        if drops[0]:
            for i in drops:
                df = df.drop(i, axis=1)


        # # special process (optional)
        df = df.values
        df = np.delete(df, -1, axis=0)

        # assign the data > training / testing
        x = df[:, 1:]
        if feature[0] != 'all':
            x = x[:, tuple(feature)]
        y = df[:, 0]

        if fix_random:
            np.random.seed(seed)
        msk = np.random.rand(len(df)) < train_proportion
        x_train = x[msk]
        x_test = x[~msk]
        y_train = y[msk]
        y_test = y[~msk]


        # normalized
        minmax_scale = preprocessing.MinMaxScaler()
        x_train = minmax_scale.fit_transform(x_train)
        x_test = minmax_scale.fit_transform(x_test)

        # data number
        if not number or number > len(x_train):
            number = len(x_train)
        # print('Get ', number, ' data.')
        x_train = x_train[0:number]
        y_train = y_train[0:number]
        label_train = x_test.reshape(-1)
        label_test = y_test.reshape(-1)

        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        # y_train = np_utils.to_categorical(y_train, 2)
        # y_test = np_utils.to_categorical(y_test, 2)


        if model == 'CNN':
            x_train = x_train.reshape(number, x_train.shape[1], 1, 1)  # 3: channel
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
        elif model == 'RNN':
            x_train = x_train.reshape(number, x_train.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        return (x_train, y_train), (x_test, y_test), (label_train, label_test)

    def hw2_predict(test_path, number, model, feature, drops):
        # download the data

        filepath_test_X = test_path
        all_df = pd.read_csv( filepath_test_X )


        # drop the bad feature (optional)
        df = all_df.copy()
        if drops[0]:
            for i in drops:
                df = df.drop(i, axis=1)


        # assign the data > training / testing
        x_train = df.values
        if feature[0] != 'all':
            x_train = x_train[:, tuple(feature)]


        # normalized
        minmax_scale = preprocessing.MinMaxScaler()
        x_train = minmax_scale.fit_transform(x_train)


        return x_train

class linear():
    def __init__(self, feature, type_decom, n_decom):  # Run it once
        train_X_path = 'X_train'
        train_Y_path = 'Y_train'
        seed = 700


        test_path = sys.argv[5]  # 'testing_data.csv'
        self.test_output = sys.argv[6]  # 'sample_submission.csv'
        self.save_w_path = 'map_w_0004'
        self.save_b_path = 'map_b_0004'
        self.save_SVM_path = 'SVM.pickle'
        self.save_boosting_path = 'boosting_0001.pickle'

        self.regular = 0.00001
        self.lr = 0.05
        self.show_result = 10
        self.plt_count = [10, 8260, '']

        # DNN
        self.model_name = 'DNN_keras.h5'
        self.n_epochs = 11
        self.units = 500
        self.batch_size = 30
        self.hidden_layer_number = 10

        # hw2
        train_proportion = 0.90  # 0.9 = 90%

        # load data
        nor_index = [0, 1, 3, 4, 5]  # normailize
        self.test_x =data_loder.hw2_test(feature, test_path, nor_index)
        # print(self.test.shape)
        # self.x, self.y, self.test_x = data_loder.TA_hw2(feature, train_X_path, train_Y_path, test_path, nor_index)

        # validation
        # self.train_x, self.vali_x, self.train_y, self.vali_y = train_test_split(self.x, self.y, train_size=train_proportion, random_state=0)


    def DNN(self):
        self.train_y = np_utils.to_categorical(self.train_y, 2)
        self.vali_y = np_utils.to_categorical(self.vali_y, 2)
        self.y = np_utils.to_categorical(self.y, 2)

        ## model structure
        # dense = fully connect layer, untis = output, kernel_initializer = random type
        # activation = softplus / softsign / sigmoid / tanh / hard_sigmoid / linear
        # loss = SGD / PMSprop / Adagrad / Adadelta / Adam / Adamax / Nadam / categorical_crossentropy
        # Dropout = Dropout the neural (%)
        model = Sequential()
        model.add(Dense(input_dim=self.train_x.shape[1], units=self.units, kernel_initializer='normal', activation='relu'))
        for i in range(self.hidden_layer_number):
            model.add(Dense(units=self.units, kernel_initializer='normal', activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))
        model.add(Dense(units=self.train_y.shape[1], kernel_initializer='normal', activation='softmax'))
        # adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # check the model structure
        print(model.summary())

        ## training
        # validation_split:Verification data is the last percentage of the training data
        # verbose: 0 (not show), 1 (detail), 2 (brevity)
        train_history = model.fit(self.x, self.y, batch_size=self.batch_size, epochs=self.n_epochs, validation_split=0.1, verbose=2)
        # tool.show_train_history(train_history, 'acc', 'val_acc')
        # tool.show_train_history(train_history, 'loss', 'val_loss')


        ## testing
        result = model.evaluate(self.train_x, self.train_y)  # loss / accuracy
        vali_result = model.evaluate(self.vali_x, self.vali_y)  # loss / accuracy
        loss = 1-result[1]
        vali_loss = 1-vali_result[1]
        print('\n acc:', result[1], vali_result[1])


        # Result
        # prediction = model.predict_classes(self.vali_x)  # [5, 6, 2, 4, ...]
        # print(prediction.shape)
        # crosstab = pd.crosstab(self.label_test, prediction, rownames=['label'], colnames=['predict'])
        # print(crosstab)
        # df = pd.DataFrame({'label': self.label_test, 'predict': prediction})
        # print(df[(df.label == 5) & (df.predict == 3)])

        ## save model
        model.save(self.model_name)
        self.DNN_testing()
        del model
        return loss, vali_loss, ''

    def DNN_testing(self):
        model = load_model(self.model_name)
        # result = model.evaluate(self.x, self.y)  # loss / accuracy
        # print('\n Train Acc:', result)

        pre_y = model.predict_classes(self.test_x)

        n = 0
        with open(self.test_output, 'w', newline='') as csvfile:
            csv_f = csv.writer(csvfile)
            csv_f.writerow(['id', 'label'])
            for i in range(pre_y.shape[0]):
                value = int(pre_y[i])
                # pre_y.append(value)
                csv_f.writerow([str(n+1), value])
                n += 1
        print('pre_y_size = ', pre_y.shape, 'One number = ', pre_y.sum())


    def TA_generative(self):

        def sigmoid(z):
            res = 1 / (1.0 + np.exp(-z))
            return np.clip(res, 1e-6, 1 - 1e-6)

        cnt1 = 0
        cnt2 = 0
        dim = self.train_x.shape[1]
        x_train = self.x
        y_train = self.y

        mu1 = np.zeros((dim,))
        mu2 = np.zeros((dim,))

        for i in range(x_train.shape[0]):
            if y_train[i] == 1:
                cnt1 += 1
                mu1 += x_train[i]
            else:
                cnt2 += 1
                mu2 += x_train[i]
        mu1 /= cnt1
        mu2 /= cnt2

        sigma1 = np.zeros((dim, dim))
        sigma2 = np.zeros((dim, dim))
        for i in range(x_train.shape[0]):
            if y_train[i] == 1:
                sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
            else:
                sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
        sigma1 /= cnt1
        sigma2 /= cnt2

        share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
        sigma_inverse = np.linalg.inv(share_sigma)
        N1, N2 = cnt1, cnt2

        w = np.dot((mu1 - mu2), sigma_inverse)
        b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse),
                                                                                mu2) + np.log(float(N1) / N2)


        # record the loss

        pre_y = sigmoid(np.dot(w, self.train_x.T) + b)
        pre_y[pre_y < 0.5] = 0
        pre_y[pre_y >= 0.5] = 1
        loss = np.abs(pre_y - self.train_y).sum() / self.train_y.shape[0]
        acc = 1 - loss

        vali_y_pred = sigmoid(np.dot(w, self.vali_x.T) + b)
        vali_y_pred[vali_y_pred < 0.5] = 0
        vali_y_pred[vali_y_pred >= 0.5] = 1
        vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
        vali_acc = 1-vali_loss

        print('acc = ', acc, vali_acc, vali_y_pred.sum())
        np.save(self.save_w_path, w)
        np.save(self.save_b_path, b)
        self.prediction_testing()

        return loss, vali_loss, ''



    def TA_logistic(self):

        def sigmoid(z):
            res = 1 / (1.0 + np.exp(-z))
            return np.clip(res, 1e-6, 1 - 1e-6)

        x_train = self.train_x
        y_train = self.train_y

        b = 0.0
        w = np.zeros(x_train.shape[1])
        lr = self.lr
        epoch = self.plt_count[1]
        b_lr = 0
        w_lr = np.ones(x_train.shape[1])
        acc_record = []
        loss_record = []
        vali_acc_record = []
        vali_loss_record = []
        count_record = []

        for count in range(epoch+1):

            ### record

            # training (real loss)
            pre_y = sigmoid(np.dot(self.train_x, w) + b)
            pre_y[pre_y < 0.5] = 0
            pre_y[pre_y >= 0.5] = 1
            loss = np.abs(pre_y - self.train_y).sum() / self.train_y.shape[0]
            loss_record.append(loss)
            acc_record.append(1-loss)

            # validation (real loss)
            vali_y_pred = sigmoid(np.dot(self.vali_x, w) + b)
            vali_y_pred[vali_y_pred < 0.5] = 0
            vali_y_pred[vali_y_pred >= 0.5] = 1
            vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
            vali_loss_record.append(vali_loss)
            vali_acc_record.append(1-vali_loss)

            ### start
            z = np.dot(x_train, w) + b
            pred = sigmoid(z)
            loss = y_train - pred
            # if count > 5:
            #     loss = -1 * (y_train * np.log(pred + 1e-100) + (1 - y_train) * np.log(1 - pred + 1e-100))


            b_grad = -1 * np.sum(loss)
            w_grad = -1 * np.dot(loss, x_train)

            b_lr += b_grad ** 2
            w_lr += w_grad ** 2

            b = b - lr / np.sqrt(b_lr) * b_grad
            w = w - lr / np.sqrt(w_lr) * w_grad

            count_record.append(count)

            if count % self.show_result == 0:
                print('acc(count/train/validation) = ', count, acc_record[-1], vali_acc_record[-1], vali_y_pred.sum())
                if (count > 0 and self.plt_count[2] == 'stop' and vali_loss > vali_loss_record[count - self.show_result]) or count == self.plt_count[1]:

                    pre_x = np.linspace(0, self.train_x.shape[0], num=len(self.train_x))
                    pre_vali_x = np.linspace(self.train_x.shape[0], self.vali_x.shape[0]+self.train_x.shape[0], num=len(self.vali_x))
                    pre_x = pre_x.reshape(pre_x.shape[0], 1)
                    pre_vali_x = pre_vali_x.reshape(pre_vali_x.shape[0], 1)
                    plt.plot(pre_x, pre_y, 'r', label='training prediction')
                    plt.plot(pre_vali_x, vali_y_pred, 'r', label='validation prediction')
                    plt.scatter(pre_x, self.train_y, label='training')
                    plt.scatter(pre_vali_x, self.vali_y, label='validation')

                    # plt.scatter(self.vali_x, self.vali_y, label='validation')
                    plt.legend(loc='upper right')
                    title = ['Step = ', str(count)]
                    plt.title(''.join(title))
                    plt.xlabel('PM2.5')
                    plt.ylabel('value')
                    plt.show()

                    # learning rate
                    plt.scatter(count_record[self.plt_count[0]:], vali_loss_record[self.plt_count[0]:], label='validation')
                    plt.scatter(count_record[self.plt_count[0]:], loss_record[self.plt_count[0]:], label='training')
                    plt.legend(loc='upper right')
                    title = ['Step = ', str(count)]
                    plt.title(''.join(title))
                    plt.xlabel('epchs')
                    plt.ylabel('acc')
                    plt.show()
                    break

        np.save(self.save_w_path, w)
        np.save(self.save_b_path, b)
        self.prediction_testing()

        # y_pred = sigmoid(np.dot(self.train_x, w) + b)
        # y_pred[y_pred < 0.5] = 0
        # y_pred[y_pred >= 0.5] = 1
        # loss = np.abs(y_pred - self.train_y).sum() / self.train_y.shape[0]
        # vali_y_pred = sigmoid(np.dot(self.vali_x, w) + b)
        # vali_y_pred[vali_y_pred < 0.5] = 0
        # vali_y_pred[vali_y_pred >= 0.5] = 1
        # vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
        # print(1-loss, 1-vali_loss)
        # return loss, vali_loss, ''

    def simple(self):
        # validation
        vali_one_term = self.vali_x.copy()
        vali_two_term = self.vali_x ** 1
        vali_three_term = np.exp(self.vali_x)
        vali_four_term = np.cos(self.vali_x)
        vali_total_term = (vali_one_term, vali_two_term)
        vali_X = np.concatenate(vali_total_term, axis=1)

        # training
        count = 0
        one_term = self.train_x.copy()
        two_term = self.train_x ** 1
        three_term = np.exp(self.train_x)
        four_term = np.cos(self.train_x)
        total_term = (one_term, two_term)
        X = np.concatenate(total_term, axis=1)
        w = np.random.rand(X.shape[1], 1)
        b = np.zeros([1, 1])
        vali_loss_record = []
        loss_record = []
        count_record = []

        while count < self.plt_count[1]+1:
            count_record.append(count)

            if self.loss == 'rmse':
                # validation
                vali_y_pred = np.dot(vali_X, w) + b
                vali_loss = np.power(vali_y_pred - self.vali_y, 2)
                vali_loss = vali_loss.sum() / vali_loss.shape[0]
                vali_loss_record.append(vali_loss)

                # forward
                y_pred = np.dot(X, w) + b
                loss = np.power(y_pred - self.train_y, 2)  # + self.regular * sum(w)
                loss = loss.sum() / loss.shape[0]
                loss_record.append(loss)
            if self.loss == 'cross_entropy':
                # validation
                z = np.dot(vali_X, w) + b
                sigmoid = 1 / (np.exp(-z) + 1)
                sigmoid[sigmoid < 0.5] = 0
                sigmoid[sigmoid >= 0.5] = 1
                zero_n = sigmoid.sum()
                vali_loss = np.abs(sigmoid-self.vali_y).sum()/self.vali_y.shape[0]
                vali_y_pred = sigmoid
                vali_loss_record.append(vali_loss)


                z = np.dot(X, w) + b
                sigmoid = 1 / (np.exp(-z) + 1)
                loss = -(np.multiply(self.train_y, np.log(sigmoid)) + np.multiply(1 - self.train_y, np.log(1 - sigmoid)))
                loss = loss.sum() / loss.shape[0]
                loss_record.append(loss)
                # sigmoid[sigmoid < 0.5] = 0
                # sigmoid[sigmoid >= 0.5] = 1
                y_pred = sigmoid

                # loss = np.power(y_pred - sigmoid, 2)  # + self.regular * sum(w)
                # loss = loss.sum() / loss.shape[0]

            # backward
            dloss = 1
            dy_pred = dloss * (y_pred - self.train_y)
            dw = np.dot(X.T, dy_pred)
            db = dy_pred.sum()
            w -= self.lr * dw
            b -= self.lr * db

            if count % self.show_result == 0:
                # fitting
                # print('count = ', count, ' - loss (training, validation) = ', loss, vali_loss )
                if (count > 0 and self.plt_count[2] == 'stop' and vali_loss > vali_loss_record[count-self.show_result]) or count == self.plt_count[1]:

                    # pre_x = np.linspace(0, self.train_x.shape[0], num=len(self.train_x))
                    # pre_vali_x = np.linspace(self.train_x.shape[0], self.vali_x.shape[0]+self.train_x.shape[0], num=len(self.vali_x))
                    # pre_x = pre_x.reshape(pre_x.shape[0], 1)
                    # pre_y = y_pred
                    # pre_vali_x = pre_vali_x.reshape(pre_vali_x.shape[0], 1)
                    # plt.plot(pre_x, pre_y, 'r', label='training prediction')
                    # plt.plot(pre_vali_x, vali_y_pred, 'r', label='validation prediction')
                    # plt.scatter(pre_x, self.train_y, label='training')
                    # plt.scatter(pre_vali_x, self.vali_y, label='validation')
                    #
                    # # plt.scatter(self.vali_x, self.vali_y, label='validation')
                    # plt.legend(loc='upper right')
                    # title = ['Step = ', str(count)]
                    # plt.title(''.join(title))
                    # plt.xlabel('PM2.5')
                    # plt.ylabel('value')
                    # plt.show()
                    #
                    # # learning rate
                    # plt.scatter(count_record[self.plt_count[0]:], vali_loss_record[self.plt_count[0]:], label='validation')
                    # plt.scatter(count_record[self.plt_count[0]:], loss_record[self.plt_count[0]:], label='training')
                    # plt.legend(loc='upper right')
                    # title = ['Step = ', str(count)]
                    # plt.title(''.join(title))
                    # plt.xlabel('epchs')
                    # plt.ylabel('loss')
                    # plt.show()
                    break
            count += 1

        # testing
        # np.save(self.save_w_path, w)
        # np.save(self.save_b_path, b[0][0])
        # self.prediction_testing()
        return loss, vali_loss, zero_n


    def gradient_boosting(self):

        boosting = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, max_depth=8, min_samples_split=50, min_samples_leaf=7, subsample=0.8, max_features='sqrt', random_state=10)
        # boosting = GradientBoostingClassifier()
        boosting.fit(self.x, self.y)
        pre_y = boosting.predict(self.train_x)
        vali_y_pred = boosting.predict(self.vali_x)

        loss = np.abs(pre_y - self.train_y).sum() / self.train_y.shape[0]
        acc = 1 - loss
        vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
        vali_acc = 1 - vali_loss
        print('acc = ', acc, vali_acc, vali_y_pred.sum())
        # save model
        with open(self.save_boosting_path, 'wb') as f:
            pickle.dump(boosting, f)
        self.gradient_boosting_testing()
        return loss, vali_loss, vali_y_pred.sum()

    def gradient_boosting_testing(self):

        # load model
        with open(self.save_boosting_path, 'rb') as f:
            clf = pickle.load(f)
            test_pre = clf.predict(self.test_x)

        # pre_y = clf.predict(self.train_x)
        # loss = np.abs(pre_y - self.train_y).sum() / self.train_y.shape[0]
        # acc = 1-loss
        # vali_y_pred = clf.predict(self.vali_x)
        # vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
        # vali_acc = 1-vali_loss

        # print('acc = ', acc, vali_acc, vali_y_pred.sum())

        # testing
        n = 0
        with open(self.test_output, 'w', newline='') as csvfile:
            csv_f = csv.writer(csvfile)
            csv_f.writerow(['id', 'label'])
            for i in range(test_pre.shape[0]):
                csv_f.writerow([str(n + 1), test_pre[i]])
                n += 1
        print('One number = ', test_pre.sum())


    def SVM(self):
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn', lineno=193)
        warnings.filterwarnings('ignore', category=DataConversionWarning)

        # normal case
        clf = svm.SVC()
        clf.fit(self.x, self.y)
        # save model
        with open(self.save_SVM_path, 'wb') as f:
            pickle.dump(clf, f)
        self.SVM_testing()
        print(classification_report(self.vali_y, clf.predict(self.vali_x)))

        pre_y = clf.predict(self.train_x)
        loss = np.abs(pre_y - self.train_y).sum() / self.train_y.shape[0]
        acc = 1-loss
        vali_y_pred = clf.predict(self.vali_x)
        vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
        vali_acc = 1 - vali_loss
        print('acc = ', acc, vali_acc, vali_y_pred.sum())

        # # force case
        # temp_vali_error = 2
        # print('Using SVM model...')
        # for i in range(1, 11):
        #     c = i / 10
        #     clf = svm.SVC(C=c, kernel='rbf', gamma=20, decision_function_shape='ovr')
        #     clf.fit(self.train_x, self.train_y.ravel())
        #     pre_y = clf.predict(self.train_x)
        #     loss = np.abs(pre_y - self.train_y).sum() / self.train_y.shape[0]
        #     vali_y_pred = clf.predict(self.vali_x)
        #     vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
        #     vali_acc = 1 - vali_loss
        #     print(loss, vali_loss, c, vali_acc)
        #     if vali_loss > temp_vali_error:
        #         vali_loss = temp_vali_error
        #         loss = temp_loss
        #         break
        #     else:
        #         pass



        return loss, vali_loss, ''

    def SVM_testing(self):

        # load model
        with open(self.save_SVM_path, 'rb') as f:
            clf = pickle.load(f)
            test_pre = clf.predict(self.test_x)

        pre_y = clf.predict(self.train_x)
        loss = np.abs(pre_y - self.train_y).sum() / self.train_y.shape[0]
        acc = 1-loss
        vali_y_pred = clf.predict(self.vali_x)
        vali_loss = np.abs(vali_y_pred - self.vali_y).sum() / self.vali_y.shape[0]
        vali_acc = 1-vali_loss

        print('acc = ', acc, vali_acc, vali_y_pred.sum())


        # testing
        n = 0
        with open(self.test_output, 'w', newline='') as csvfile:
            csv_f = csv.writer(csvfile)
            csv_f.writerow(['id', 'label'])
            for i in range(test_pre.shape[0]):
                csv_f.writerow([str(n + 1), test_pre[i]])
                n += 1
        print('One number = ', test_pre.sum())

    def prediction_testing(self):

        def sigmoid(z):
            res = 1 / (1.0 + np.exp(-z))
            return np.clip(res, 1e-6, 1 - 1e-6)

        w, b = np.load(self.save_w_path+'.npy'), np.load(self.save_b_path+'.npy')
        pre_y = sigmoid(np.dot(self.test_x, w) + b)
        pre_y[pre_y < 0.5] = 0
        pre_y[pre_y >= 0.5] = 1
        X = self.test_x
        n = 0
        with open(self.test_output, 'w', newline='') as csvfile:
            csv_f = csv.writer(csvfile)
            csv_f.writerow(['id', 'label'])
            for i in range(X.shape[0]):
                value = int(pre_y[i])
                # pre_y.append(value)
                csv_f.writerow([str(n+1), value])
                n += 1
        print('pre_y_size = ', pre_y.shape, 'One number = ', pre_y.sum())

    def unsenble(self):
        candidate = ['./result/0002.csv', './result/0003.csv', './result/0004.csv']
        kaggle = [0.85380, 0.85823, 0.85417]  # [0.86941, 0.85380, 0.85823, 0.85417, 0.84324]
        # kaggle = [1, 1, 1, 1]
        model = []
        n = 1
        for cand in candidate:
            with open(cand, newline='') as csvfile:
                rows = csv.reader(csvfile)
                count = 0
                for row in rows:
                    if not row[1] == 'label':
                        if n == 1:
                            model.append(float(row[1])*kaggle[n-1]/sum(kaggle))
                        else:
                            model[count] += float(row[1])*kaggle[n-1]/sum(kaggle)
                            count += 1

            n += 1
        model = np.array(model)

        n = 0
        with open('./result/unsenble.csv', 'w', newline='') as csvfile:
            csv_f = csv.writer(csvfile)
            csv_f.writerow(['id', 'label'])
            for i in range(model.shape[0]):
                if model[i] >= 0.5:
                    model[i] = 1
                elif model[i] < 0.5:
                    model[i] = 0
                value = int(model[i])
                # pre_y.append(value)
                csv_f.writerow([str(n+1), value])
                n += 1
        print('pre_y_size = ', model.shape, 'One number = ', model.sum())



if __name__ == '__main__':


    def check_good_feature(feature, set):
        ret = True
        for i in feature:
            if not i in set:
                return False
        return ret
    def check_bad_feature(feature, set):
        ret = True
        for i in feature:
            if i in set:
                return False
        return ret


    # decomposition: PCA, Tree
    # a = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 50, 52, 53, 56, 58, 60, 61, 62, 65, 66, 67, 69, 73, 75, 82, 85, 98, 102, 105]
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62,
         63, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95,
         96, 98, 100, 102, 103, 105]

    # a = np.linspace(0, 105, 106).astype(np.uint8)
    # linear(a, '', '').gradient_boosting_testing()
    # linear(a, '', '').gradient_boosting()
    linear(a, '', '').prediction_testing()
    # linear(a, '', '').TA_generative()
    # linear(a, '', '').TA_logistic()
    # linear(a, '', '').SVM()

    # for i in range(1, a.shape[0]+1):
    #     print(i)
    #     linear(a, 'PCA', i).TA_generative()

    exit()





