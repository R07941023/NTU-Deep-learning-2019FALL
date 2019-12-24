import pandas as pd
import numpy as np
import csv
import math
import sys

from matplotlib import pyplot as plt

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

class data_loder():

    def TA_PM25_training(data, term, case1_data):

        def TA_valid(x, y,case1_data):
            if y <= 0 or y > 100:
                return False
            for i in range(x.shape[1]):
                # PM2.5
                if x[0, i] <= 0 or x[0, i] > 100:
                    return False
                # other feature
                for j in range(x.shape[0]):
                    if j in case1_data:
                        if x[j, i] <= 0:
                            print(x[j, i])
                            return False
                    else:
                        if x[j, i] <= 0 or x[j, i] > 100:
                            return False
            return True

        # 把有些數字後面的奇怪符號刪除
        data = pd.read_csv( data, encoding='utf-8' )
        for col in list( data.columns[2:] ):
            data[col] = data[col].astype( str ).map( lambda x: x.rstrip( 'x*#A' ) )
        data = data.values

        # 刪除欄位名稱及日期
        data = np.delete( data, [0, 1], 1 )

        # 特殊值補0
        data[data == 'NR'] = 0
        data[data == ''] = 0
        data[data == 'nan'] = 0
        data = data.astype( np.float )

        # 整理data
        N = data.shape[0] // 18
        temp = data[:18, :]

        # Shape 會變成 (x, 18) x = 取多少hours
        for i in range( 1, N ):
            temp = np.hstack( (temp, data[i * 18: i * 18 + 18, :]) )
        data = temp[tuple(term), :]

        # 用前面9筆資料預測下一筆PM2.5 所以需要-9
        x = []
        y = []
        total_length = data.shape[1] - 9
        for i in range( total_length ):
            x_tmp = data[:, i:i + 9]
            y_tmp = data[0,i+9]
            if TA_valid( x_tmp, y_tmp , case1_data):
                x.append( x_tmp.reshape( -1, ) )
                y.append( y_tmp )
        # x 會是一個(n, 18, 9)的陣列， y 則是(n, 1)
        x = np.array( x )
        y = np.array( y ).reshape(len(y), 1)
        return x, y  # feature target

    def PM25_training(path, term, validation_n, bad_term, map_value):
        # data loader
        train = pd.read_csv(path, encoding='utf-8')
        narratives = 2  # The first two columns are narratives
        feature_n = 18  # reshape - period = 18
        hours_n = 24
        # drop data
        train = train.drop('日期', axis=1)
        train = train.drop('測項', axis=1)
        # preprocess the data
        # train = train.replace('NR', 0.0)
        # pd to np
        train = train.values
        # get the feature
        train = train.reshape(int(train.shape[0]*train.shape[1]/(feature_n*hours_n)), feature_n, hours_n)
        train_term = np.zeros((train.shape[0], len(term), hours_n), dtype=object)
        for i in range(train.shape[0]):
            n = 0
            for j in range(train.shape[1]):
                if j in term:
                    # print(term.index(j), n)
                    train_term[i][term.index(j)] = train[i][j]
                    n += 1
        # train_feature = train_feature.T
        # check the data value
        drop_day = []
        for j in range(train_term.shape[1]):
            dict_good_term = {bad_term[0]: 0, bad_term[1]: 0}
            good_day = True
            for i in range(train_term.shape[0]):
                for k in range(train_term.shape[2]):
                    if train_term[i][j][k] != train_term[i][j][k]:  # nan != nan
                        dict_good_term[bad_term[1]] += 1
                        good_day = False
                        train_term[i][j][k] = bad_term[1]
                        if train_term[i][j][k] in map_value:
                            good_day = True
                            train_term[i][j][k] = 0
                    elif train_term[i][j][k] == bad_term[0]:
                        dict_good_term[bad_term[0]] += 1
                        good_day = False
                        if train_term[i][j][k] in map_value:
                            good_day = True
                            train_term[i][j][k] = 0
                    elif not train_term[i][j][k][-1].isdigit():
                        train_term[i][j][k] = float(train_term[i][j][k][:-1])
                    else:
                        train_term[i][j][k] = float(train_term[i][j][k])
                    if train_term[i][j][k] > 200 or train_term[i][j][k] <= 0:
                        drop_day.append(i)
                    if good_day == False:
                        drop_day.append(i)
            # calculate the good term value
            print('')
            for index in dict_good_term:
                print('The term ', term[j], ' : ', index, '=', dict_good_term[index])
                # if not index in map_value and dict_good_term[index] != 0:
                # 	good_day = False
            print('')

        # drop_day
        drop_day = list(set(drop_day))
        print('drop_day = ', drop_day)
        train_term = np.delete(train_term, drop_day, axis=0)

        # get target and feature
        print('train_term = ', train_term.shape)
        print('')

        feature = []
        target = []
        for i in range(train_term.shape[0]):
            for j in range(train_term.shape[2]-9):
                feature.append(train_term[i, :, j:j+9].reshape(-1))
                target.append(train_term[i][0][j+9])
        feature = np.array(feature).astype('float64')
        target = np.array(target).reshape(len(target), 1).astype('float64')
        return feature, target

    def PM25_testing(path, term, validation_n, bad_term, map_value):
        # data loader
        train = pd.read_csv(path, encoding='utf-8')
        narratives = 2  # The first two columns are narratives
        feature_n = 18  # reshape - period = 18
        hours_n = 9
        # drop data
        train = train.drop('id', axis=1)
        train = train.drop('測項', axis=1)
        # preprocess the data
        # train = train.replace('NR', 0.0)
        # pd to np
        train = train.values
        # get the feature
        train = train.reshape(int(train.shape[0]*train.shape[1]/(feature_n*hours_n)), feature_n, hours_n)
        train_term = np.zeros((train.shape[0], len(term), hours_n), dtype=object)
        for i in range(train.shape[0]):
            n = 0
            for j in range(train.shape[1]):
                if j in term:
                    # print(term.index(j), n)
                    train_term[i][term.index(j)] = train[i][j]
                    n += 1
        # train_feature = train_feature.T
        # check the data value
        drop_day = []
        for j in range(train_term.shape[1]):
            dict_good_term = {bad_term[0]: 0, bad_term[1]: 0}
            good_day = True
            for i in range(train_term.shape[0]):
                for k in range(train_term.shape[2]):
                    if train_term[i][j][k] != train_term[i][j][k]:  # nan != nan
                        dict_good_term[bad_term[1]] += 1
                        good_day = False
                        train_term[i][j][k] = bad_term[1]
                        if train_term[i][j][k] in map_value:
                            good_day = True
                            train_term[i][j][k] = 0
                    elif train_term[i][j][k] == bad_term[0]:
                        dict_good_term[bad_term[0]] += 1
                        good_day = False
                        if train_term[i][j][k] in map_value:
                            good_day = True
                            train_term[i][j][k] = 0
                    elif not train_term[i][j][k][-1].isdigit():
                        train_term[i][j][k] = float(train_term[i][j][k][:-1])
                    else:
                        train_term[i][j][k] = float(train_term[i][j][k])
                    if train_term[i][j][k] > 200 or train_term[i][j][k] <= 0 :
                        # pass
                        train_term[i][j][k] = 0

        # get target and feature
        print('')

        feature = []
        for i in range(train_term.shape[0]):
            feature.append(train_term[i, :, :].reshape(-1))
        feature = np.array(feature).astype('float64')
        return feature

class linear():
    def __init__(self):  # Run it once
        train_path1 = 'year1-data.csv'
        train_path2 = 'year2-data.csv'
        test_path1 = sys.argv[1]  # 'testing_data.csv'
        self.test_output = sys.argv[2]  # 'sample_submission.csv'
        self.save_w_path = 'map_w'
        self.save_b_path = 'map_b'
        seed = 700


        self.validation_n = 5
        # term = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]  # target + feature
        # term = [9]  # target + feature
        term = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]  # target + feature
        case1_data = []  # [14, 15]  # fit the the data > 200, ex:'WD_HR', 'WIND_DIR'
        self.regular = 0.00001
        self.lr = 1e-4
        self.show_result = 5
        self.plt_count = [2, 30000]

        # TA_adam
        self.batch_size = 64
        self.lam = 0.001
        self.epsilon = 1e-8

        self.x1, self.y1 = data_loder.TA_PM25_training(train_path1, term, case1_data)
        self.x2, self.y2 = data_loder.TA_PM25_training( train_path2, term, case1_data )
        self.x = np.concatenate( (self.x1, self.x2) )
        self.y = np.concatenate( (self.y1, self.y2) )
        self.train_x, self.train_y, self.vali_x, self.vali_y = tool.cross_validation( self.x, self.y, self.validation_n, seed )

        # Ny fumction
        bad_term = ['NR', 'empty']
        map_value = [bad_term[0], bad_term[1]]  # 'empty', 'NR'
        # self.x1, self.y1 = data_loder.PM25_training(train_path1, term, self.validation_n, bad_term, map_value)
        # self.x2, self.y2 = data_loder.PM25_training(train_path2, term, self.validation_n, bad_term, map_value)
        # self.x = np.concatenate((self.x1, self.x2))
        # self.y = np.concatenate((self.y1, self.y2))
        # self.train_x, self.train_y, self.vali_x, self.vali_y = tool.cross_validation(self.x, self.y, self.validation_n, seed)
        # testing
        self.test_x = data_loder.PM25_testing( test_path1, term, self.validation_n, bad_term, map_value )
        print('train_x, train_y, vali_x, vali_y, testing_x = ', self.train_x.shape, self.train_y.shape, self.vali_x.shape, self.vali_y.shape, self.test_x.shape)


    def TA_adam(self):
        # 打亂data順序
        one_term = self.train_x
        two_term = np.power(self.train_x, 1)
        three_term = np.power(self.train_x, 3)
        four_term = np.exp(self.train_x)
        total = (one_term, two_term)
        # calculate the w
        self.train_x = np.concatenate(total, axis=1)

        one_term = self.vali_x
        two_term = np.power(self.vali_x, 1)
        three_term = np.power(self.vali_x, 3)
        four_term = np.exp(self.vali_x)
        total = (one_term, two_term)
        # calculate the w
        self.vali_x = np.concatenate(total, axis=1)

        index = np.arange( self.train_x.shape[0] )
        np.random.shuffle( index )
        x = self.train_x[index]
        y = self.train_y[index]

        lr = self.lr
        batch_size = self.batch_size

        # 訓練參數以及初始化
        lam = self.lam
        epsilon = self.epsilon

        beta_1 = np.full( x[0].shape, 0.9 ).reshape( -1, 1 )
        beta_2 = np.full( x[0].shape, 0.99 ).reshape( -1, 1 )
        w = np.full( x[0].shape, 0.1 ).reshape( -1, 1 )
        bias = 0.1
        m_t = np.full( x[0].shape, 0 ).reshape( -1, 1 )
        v_t = np.full( x[0].shape, 0 ).reshape( -1, 1 )
        m_t_b = 0.0
        v_t_b = 0.0
        t = 0
        count = 1
        train_loss_record = []
        vali_loss_record = []
        count_record = []

        while count < self.plt_count[1] + 1:
            count_record.append( count )
            for b in range( int( x.shape[0] / batch_size ) ):
                t += 1
                x_batch = x[b * batch_size:(b + 1) * batch_size]
                y_batch = y[b * batch_size:(b + 1) * batch_size].reshape( -1, 1 )
                loss = y_batch - np.dot( x_batch, w ) - bias

                # 計算gradient
                g_t = np.dot( x_batch.transpose(), loss ) * (-2) + 2 * lam * np.sum( w )
                g_t_b = loss.sum( axis=0 ) * (2)
                m_t = beta_1 * m_t + (1 - beta_1) * g_t
                v_t = beta_2 * v_t + (1 - beta_2) * np.multiply( g_t, g_t )
                m_cap = m_t / (1 - (beta_1 ** t))
                v_cap = v_t / (1 - (beta_2 ** t))
                m_t_b = 0.9 * m_t_b + (1 - 0.9) * g_t_b
                v_t_b = 0.99 * v_t_b + (1 - 0.99) * (g_t_b * g_t_b)
                m_cap_b = m_t_b / (1 - (0.9 ** t))
                v_cap_b = v_t_b / (1 - (0.99 ** t))
                w_0 = np.copy( w )

                # 更新weight, bias
                w -= ((lr * m_cap) / (np.sqrt( v_cap ) + epsilon)).reshape( -1, 1 )
                bias -= (lr * m_cap_b) / (math.sqrt( v_cap_b ) + epsilon)

            train_loss = np.power( (np.dot(self.train_x, w) + bias) - self.train_y, 2 )  # + self.regular * sum(w)
            train_loss = train_loss.sum() / train_loss.shape[0]
            train_loss_record.append( train_loss )
            vali_loss = np.power( (np.dot( self.vali_x, w ) + bias) - self.vali_y, 2 )  # + self.regular * sum(w)
            vali_loss = vali_loss.sum() / vali_loss.shape[0]
            vali_loss_record.append( vali_loss )

            pre_y = (np.dot( self.train_x, w ) + bias)
            vali_y_pred = (np.dot( self.vali_x, w ) + bias)
            if count % self.show_result == 0:
                # fitting
                print('count = ', count, ' - loss (training, validation) = ', train_loss, vali_loss )
                # print( (np.dot( self.train_x, w ) + bias)[0:3] )
            if count == self. plt_count[1]:

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
                plt.savefig('fitting.png')
                plt.show()

                # learning rate
                plt.scatter(count_record[self.plt_count[0]:], vali_loss_record[self.plt_count[0]:], label='validation')
                plt.scatter(count_record[self.plt_count[0]:], train_loss_record[self.plt_count[0]:], label='training')
                plt.legend(loc='upper right')
                title = ['Step = ', str(count)]
                plt.title(''.join(title))
                plt.xlabel('epchs')
                plt.ylabel('loss')
                plt.savefig('learning_rate.png')
                plt.show()

            count += 1

        print(w.shape, bias)
        # testing
        np.save(self.save_w_path, w)
        np.save(self.save_b_path, bias[0])
        self.prediction_testing()

        return w, bias

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

        while count < self.plt_count[1]+100:
            count_record.append(count)

            # validation
            vali_y_pred = np.dot(vali_X, w) + b
            vali_loss = np.power(vali_y_pred - self.vali_y, 2)
            vali_loss = vali_loss.sum()/vali_loss.shape[0]
            vali_loss_record.append(vali_loss)

            # forward
            y_pred = np.dot(X, w) + b
            loss = np.power(y_pred - self.train_y, 2)  # + self.regular * sum(w)
            loss = loss.sum()/loss.shape[0]
            loss_record.append(loss)

            # backward
            dloss = 1
            dy_pred = dloss * (y_pred - self.train_y)
            dw = np.dot(X.T, dy_pred)
            db = dy_pred.sum()
            w -= self.lr * dw
            b -= self.lr * db

            if count % self.show_result == 0:
                # fitting
                print('count = ', count, ' - loss (training, validation) = ', loss, vali_loss )
            if count == self.plt_count[1]:
                pre_x = np.linspace(0, self.train_x.shape[0], num=len(self.train_x))
                pre_vali_x = np.linspace(self.train_x.shape[0], self.vali_x.shape[0]+self.train_x.shape[0], num=len(self.vali_x))
                pre_x = pre_x.reshape(pre_x.shape[0], 1)
                pre_y = y_pred
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
                plt.ylabel('loss')
                plt.show()
            count += 1

        # testing
        np.save(self.save_w_path, w)
        np.save(self.save_b_path, b[0][0])
        self.prediction_testing()

    def analytical_solution(self):
        # plt the role data

        pre_x = np.linspace(0, self.train_x.shape[0], num=len(self.train_x))
        plt.scatter(pre_x, self.train_y)

        # determine the term
        one_term = np.ones(len(self.train_x)).reshape(len(self.train_x), 1)
        two_term = self.train_x.copy()
        three_term = np.power(self.train_x, 2)

        # four_term = np.exp(self.x)
        total = (one_term, two_term)

        # calculate the w
        X = np.concatenate(total, axis=1).T
        w = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), self.train_y)

        # plt the predicted value
        pre_y = []
        for i in range(pre_x.shape[0]):
            value = w[0][0]
            for j in range(1, len(w)):
                value += X.T[i][j-1] * w[j][0]
            pre_y.append(value)
        pre_y = np.array(pre_y).reshape(len(pre_y), 1)
        loss = np.power(pre_y - self.train_y, 2)
        loss = sum(loss)/loss.shape[0]
        print('loss = ', loss)

        plt.plot(pre_x, pre_y, 'r')
        # plt.show()

        # testing
        np.save(self.save_w_path, w[1:])
        np.save(self.save_b_path, w[0][0])
        self.prediction_testing()




    def prediction_testing(self):
        w, b = np.load(self.save_w_path+'.npy'), np.load(self.save_b_path+'.npy')

        one_term = self.test_x.copy()
        two_term = self.test_x ** 1
        three_term = np.exp(self.test_x)
        four_term = np.cos(self.test_x)
        total_term = (one_term, two_term)
        X = np.concatenate(total_term, axis=1)
        pre_y = []
        n = 0
        with open(self.test_output, 'w', newline='') as csvfile:
            csv_f = csv.writer(csvfile)
            csv_f.writerow(['id', 'value'])
            for i in range(X.shape[0]):
                value = b.copy()
                for j in range(len(w)):
                    value += X[i][j] * w[j][0]
                pre_y.append(value)
                csv_f.writerow(['id_'+str(n), value])
                n += 1
        pre_y = np.array(pre_y).reshape(len(pre_y), 1)
        print('pre_y_size = ', pre_y.shape)




if __name__ == '__main__':
    # linear().simple()
    linear().TA_adam()
    # linear().analytical_solution()

