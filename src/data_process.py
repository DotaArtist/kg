#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from bert_pre_train import BertPreTrain


class DataProcess(object):
    def __init__(self, _show_token=False, mode='remote'):
        self.bert_batch_size = 32
        self.batch_size = 32
        self.data_path = None
        self.show_token = _show_token
        self.data = None
        self.bert_model = BertPreTrain(mode=mode)
        self.data_x = None
        self.data_y = None

    def load_data(self, file_list, is_shuffle=True):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:

            sen_1_list = []
            sen_2_list = []
            label_list = []

            data_tmp = pd.DataFrame()
            with open(i, encoding='utf-8', mode='r') as _f:
                for line in _f.readlines():
                    sen_1, sen_2, label = line.strip().split('\t')
                    sen_1_list.append(sen_1)
                    sen_2_list.append(sen_2)
                    label_list.append(label)

            data_tmp['sentence_1'] = pd.Series(sen_1_list)
            data_tmp['sentence_2'] = pd.Series(sen_2_list)
            data_tmp['label'] = pd.Series(label_list)

            data = pd.concat([data, data_tmp])

        if is_shuffle:
            data = shuffle(data)
        self.data = data

    def load_predict_data(self, file_list):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:

            sen_1_list = []
            sen_2_list = []
            idx_list = []

            data_tmp = pd.DataFrame()
            with open(i, encoding='utf-8', mode='r') as _f:
                for line in _f.readlines():
                    idx, sen_1, sen_2 = line.strip().split('\t')
                    idx_list.append(idx)
                    sen_1_list.append(sen_1)
                    sen_2_list.append(sen_2)

            data_tmp['sentence_1'] = pd.Series(sen_1_list)
            data_tmp['sentence_2'] = pd.Series(sen_2_list)
            data_tmp['idx'] = pd.Series(idx_list)

            data = pd.concat([data, data_tmp])

        self.data = data

    def get_feature(self, mode='train'):
        data_x = []
        data_y = []

        _sentence_pair_list = []

        for index, row in tqdm(self.data.iterrows()):
            if mode == 'predict':
                data_y.append([0, 1])
            elif int(row['label']) == 1:
                data_y.append([0, 1])
            elif int(row['label']) == 0:
                data_y.append([1, 0])
            else:
                print('error')
                continue

            _sentence_pair = " ||| ".join([str(row['sentence_1']), str(row['sentence_2'])])
            _sentence_pair_list.append(_sentence_pair)

            if len(_sentence_pair_list) == 32:
                data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))
                _sentence_pair_list = []

        if len(_sentence_pair_list) > 0:
            data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        self.data_x = data_x
        self.data_y = data_y

        print("data_x shape:", data_x.shape)
        print("data_y shape:", data_y.shape)

    def next_batch(self):
        counter = 0
        batch_x = []
        batch_y = []
        for (_x, _y) in zip(self.data_x, self.data_y):
            if counter == 0:
                batch_x = []
                batch_y = []

            batch_x.append(_x)
            batch_y.append(_y)
            counter += 1

            if counter == self.batch_size:
                counter = 0
                yield np.array(batch_x), np.array(batch_y)
        yield np.array(batch_x), np.array(batch_y)


if __name__ == '__main__':
    data_list = [  # '../data/ca/task3_train_100.txt',
        '../data/ca/task3_dev_100.txt',
    ]

    a = DataProcess(_show_token=False, mode='remote')
    a.load_predict_data(file_list=data_list)

    a.get_feature(mode='predict')

    for x, y in a.next_batch():
        print(x.dtype, y.shape)
