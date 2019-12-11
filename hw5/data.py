import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
import sys
import glob, os, re
from os import listdir
import numpy as np
import pandas as pd
import time
import unicodedata
import string
import spacy
from gensim.models import Word2Vec

def tokenize(data):
    nlp = spacy.load("en_core_web_sm")
    del_word = ['#', '@user', 'URL', '!', '"', '.', 'via', '@', '/', '(', ')', ':', '-', 'MEGA', 'mega', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    del_backall = ['qerqrqer']
    for i in range(data.shape[0]):
        # print('role data....')
        # print(data[i])
        # data[i] = ''.join(re.split('!', data[i]))
        data[i] = data[i].encode('ascii', 'ignore').decode('ascii')
        for j in del_word:
            data[i] = data[i].replace(j, ' ')
        for k in del_backall:
            index = data[i].find(k)
            if index != -1:
                data[i] = data[i][:index]
        # print('')
        # print('fundamental data....')
        # print(data[i])
        # spacy token
        tokens = []
        for token in nlp(data[i]):
            token = token.lemma_  # lemmatization (text normalize)
            token = str(token).lower()  # transform spacy to string
            lexeme = nlp.vocab[token]  # remove stop words, check if stopwords
            # add new
            if lexeme.is_stop == False and not token.isdigit() and token not in ['//', 'v.', '1/2', '......', '.....',
                                                                                 '....', '--', '1/3', 'him.why', '.lol',
                                                                                 '.....', '️', '️-', ':', '.@user', '=',
                                                                                 '-pron-', ' ', '  ', '   ', '#', '!',
                                                                                 '?', '...', '..', '.', '"', '/', '@',
                                                                                 "'", '’', '%', '&', ';', '-', '(', ')',
                                                                                 ',', '+']:
                if len(token) > 1 and token.find(' ') == -1:
                    tokens.append(token)
        data[i] = tokens
        # print('')
        # print('spacy....')
        # print(data[i])
        # exit()
        # token = []
        # if nlp.vocab[data[i]].is_stop == False and not token.isdigit()
        # lexeme = nlp.vocab[token]
        # print('')
        if i % 1000 == 0:
            print('Processing the ', str(i), 'data')
    return data

def data_split(X, Y, train_proportion, seed):
    if seed != '':
        np.random.seed(seed)
    random_index = np.random.choice(X.shape[0], X.shape[0], replace=False)
    # arrange
    X, Y = X[random_index], Y[random_index]
    n = int(X.shape[0]*train_proportion)
    train_X, train_Y = X[:n], Y[:n]
    vali_X, vali_Y = X[n:], Y[n:]
    train_X = torch.LongTensor(train_X)  # LongTensor
    train_Y = torch.LongTensor(train_Y)
    vali_X = torch.LongTensor(vali_X)
    vali_Y = torch.LongTensor(vali_Y)
    return train_X, train_Y, vali_X, vali_Y

def add_embedding(word, vectors, word2index, index2word, embed_dim):
    new_vector = torch.empty(1, embed_dim)
    torch.nn.init.uniform_(new_vector)
    word2index[word] = len(word2index)
    index2word.append(word)
    vectors = torch.cat([vectors, new_vector], 0)
    return word2index, index2word, vectors


def pad_to_len(arr, seq_len, seq_pad):
    if len(arr) < seq_len:
        arr.extend([seq_pad] * (seq_len - len(arr)))
        return arr
    elif len(arr) > seq_len:
        return arr[:seq_len]
    else:
        return arr


def get_indices(data, word2index, seq_len, test=False):
    all_indices = []
    # Use tokenized data
    for i, sentence in enumerate(data):
        print('=== sentence count #{}'.format(i + 1), end='\r')
        sentence_indices = []
        for word in sentence:
            if word in word2index:  # 'uhh'
                sentence_indices.append(word2index[word])  # 13646
            else:
                sentence_indices.append(word2index['<UNK>'])
        # pad all sentence to fixed length
        sentence_indices = pad_to_len(sentence_indices, seq_len, word2index["<PAD>"])
        all_indices.append(sentence_indices)
    return np.array(all_indices)

def word_embedding(data, data_total, embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len, test=False):
    # Get Word2vec word embedding
    word2index = {}
    index2word = []
    vectors = []
    if test == False:
        print("=== Compute embedding...")
        embed = Word2Vec(data_total, size=embed_dim, window=wndw_size, min_count=word_cnt, iter=word_iter, workers=work_worker)
        embed.save('embed')
    elif test == True:
        print("=== Load embedding...")
        embed = Word2Vec.load('embed')
    for i, word in enumerate(embed.wv.vocab):  # word to vector
        print('=== get words #{}'.format(i + 1), end='\r')
        word2index[word] = len(word2index)  # 'sha':13615 'uhh':13646
        index2word.append(word)  # ['sha', ..., ''uhh']
        vectors.append(embed[word])  # [vector1, ..., vector2]
    vectors = torch.tensor(vectors)
    # Add special tokens
    word2index, index2word, vectors = add_embedding("<UNK>", vectors, word2index, index2word, embed_dim)
    word2index, index2word, vectors = add_embedding("<PAD>", vectors, word2index, index2word, embed_dim)
    data = get_indices(data, word2index, seq_len)
    # print("=== total words: {}".format(len(vectors)))
    return data, vectors

def sortedDictValues1(di):
    return [(k,di[k]) for k in sorted(di.keys())]

def analysis_token(data):
    analysis = {}
    print('data size = ', data.shape[0])

    for i in range(data.shape[0]):
        if len(data[i]) not in analysis:
            analysis[len(data[i])] = 1
        else:
            analysis[len(data[i])] += 1
    analysis = sortedDictValues1(analysis)
    print('segment length = ')
    for i in analysis:
        print(i[0], i[1])
    N_set = []
    for i in data:
        for j in i:
            if j not in N_set:
                N_set.append(j)
    print('total voca = ', len(N_set))

def one_N_encoding(data, X_total):
    N_set = []
    for i in X_total:
        for j in i:
            if j not in N_set:
                N_set.append(j)
    X = np.zeros((data.shape[0], len(N_set)+1))
    for i in range(data.shape[0]):
        for j in data[i]:
            try:
                index = N_set.index(j)
                X[i][index] += 1
            except:
                X[i][-1] += 1
    return X, len(N_set)+1

def get_dataloader(folder_X, folder_Y, dataset, train_proportion, batch_size, seed, embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len,testing_csv_path):

    if dataset == 'hw5_training_token':  # data number
        # token
        data = pd.read_csv(folder_X).values
        data_test = pd.read_csv(sys.argv[3]).values
        label = pd.read_csv(folder_Y).values
        X = tokenize(data[:, 1])
        X_test = tokenize(data_test[:, 1])
        Y = label[:, 1]
        np.save('X_token', X)
        np.save('Y', Y)
        np.save('X_test_token', X_test)
        X_total = np.hstack((X, X_test))
        # embeding
        X, vectors = word_embedding(X, X_total, embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len)
        np.save('X_em', X)
        np.save('vector_em', vectors)
        vectors = torch.Tensor(vectors).float()
        # data split
        train_X, train_Y, vali_X, vali_Y = data_split(X, Y, train_proportion, seed)
        train_set = TensorDataset(train_X, train_Y)
        val_set = TensorDataset(vali_X, vali_Y)


    elif dataset == 'hw5_training_em':  # data number
        X = np.load('./X_token.npy', allow_pickle=True)
        # check
        # analysis_token(X)
        X_test = np.load('./X_test_token.npy', allow_pickle=True)
        X_total = np.hstack((X, X_test))
        Y = np.load('./Y.npy', allow_pickle=True)
        # embeding
        X, vectors = word_embedding(X, X_total, embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len)
        np.save('X_em', X)
        np.save('vector_em', vectors)
        vectors = torch.Tensor(vectors).float()
        # data split
        train_X, train_Y, vali_X, vali_Y = data_split(X, Y, train_proportion, seed)
        train_set = TensorDataset(train_X, train_Y)
        val_set = TensorDataset(vali_X, vali_Y)

    elif dataset == 'hw5_training':
        X = np.load('./X_em.npy', allow_pickle=True)
        Y = np.load('./Y.npy', allow_pickle=True)
        vectors = np.load('./vector_em.npy', allow_pickle=True)
        vectors = torch.Tensor(vectors).float()
        # data split
        train_X, train_Y, vali_X, vali_Y = data_split(X, Y, train_proportion, seed)
        train_set = TensorDataset(train_X, train_Y)
        val_set = TensorDataset(vali_X, vali_Y)

    elif dataset == 'hw5_training_BOW':
        X = np.load('./X_token.npy', allow_pickle=True)
        Y = np.load('./Y.npy', allow_pickle=True)
        X_test = np.load('./X_test_token.npy', allow_pickle=True)
        X_total = np.hstack((X, X_test))
        # one of N encoding
        X, vectors = one_N_encoding(X, X_total)
        train_X, train_Y, vali_X, vali_Y = data_split(X, Y, train_proportion, seed)
        train_set = TensorDataset(train_X, train_Y)
        val_set = TensorDataset(vali_X, vali_Y)

    elif dataset == 'hw5_testing_token':
        data = pd.read_csv(testing_csv_path).values
        vectors = np.load('./vector_em.npy', allow_pickle=True)
        vectors = torch.Tensor(vectors).float()
        print('Testing tokenize...')
        X = tokenize(data[:, 1])
        # check
        # analysis_token(X)
        np.save('X_test_token', X)
        Y = data[:, 0].astype(np.float)
        # embeding
        X, _ = word_embedding(X, '', embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len, test=True)
        np.save('X_test_em', X)
        X, Y = torch.LongTensor(X), torch.LongTensor(Y)
        test_set = TensorDataset(X, Y)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=860, shuffle=False, pin_memory=True, drop_last=False)
        print('==>>> total testing batch number: {}'.format(len(test_loader)))
        return test_loader, vectors

    elif dataset == 'hw5_testing':
        data = pd.read_csv(testing_csv_path).values
        X = np.load('./X_test_em.npy', allow_pickle=True)
        Y = data[:, 0].astype(np.float)
        # embeding
        X, Y = torch.LongTensor(X), torch.LongTensor(Y)
        test_set = TensorDataset(X, Y)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=860, shuffle=False, pin_memory=True, drop_last=False)
        print('==>>> total testing batch number: {}'.format(len(test_loader)))
        return test_loader

    elif dataset == 'hw5_testing_report':
        report_X = np.array([['today', 'is', 'hot', 'but', 'i', 'am', 'happy'], ['i', 'am', 'happy', 'but', 'today', 'is', 'hot']])
        Y = np.zeros((2,))
        # embeding
        X, _ = word_embedding(report_X, '', embed_dim, wndw_size, word_cnt, word_iter, work_worker, seq_len, test=True)
        X = X.astype(np.float)
        X, Y = torch.LongTensor(X), torch.LongTensor(Y)
        test_set = TensorDataset(X, Y)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=860, shuffle=False, pin_memory=True,
                                                  drop_last=False)
        print('==>>> total testing batch number: {}'.format(len(test_loader)))
        return test_loader

    elif dataset == 'hw5_testing_BOW':
        data = pd.read_csv(testing_csv_path).values
        Y = data[:, 0].astype(np.float)
        X = np.load('./X_token.npy', allow_pickle=True)
        X_test = np.load('./X_test_token.npy', allow_pickle=True)
        X_total = np.hstack((X, X_test))
        # one of N encoding
        X, vectors = one_N_encoding(X_test, X_total)
        X = X.astype(np.float)
        X, Y = torch.Tensor(X), torch.Tensor(Y)
        test_set = TensorDataset(X, Y)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=860, shuffle=False, pin_memory=True,
                                                  drop_last=False)
        print('==>>> total testing batch number: {}'.format(len(test_loader)))
        return test_loader

    elif dataset == 'hw5_testing_BOW_report':
        report_X = np.array([['today', 'is', 'hot', 'but', 'i', 'am', 'happy'], ['i', 'am', 'happy', 'but', 'today', 'is', 'hot']])
        Y = np.zeros((2,))
        X = np.load('./X_token.npy', allow_pickle=True)
        X_test = np.load('./X_test_token.npy', allow_pickle=True)
        X_total = np.hstack((X, X_test))
        # one of N encoding
        X, vectors = one_N_encoding(report_X, X_total)
        X = X.astype(np.float)
        X, Y = torch.Tensor(X), torch.Tensor(Y)
        test_set = TensorDataset(X, Y)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=860, shuffle=False, pin_memory=True,
                                                  drop_last=False)
        print('==>>> total testing batch number: {}'.format(len(test_loader)))
        return test_loader


    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    val_loader  = torch.utils.data.DataLoader(dataset=val_set,  batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total validation batch number: {}'.format(len(val_loader)))
    return train_set, val_set, train_loader, val_loader, vectors