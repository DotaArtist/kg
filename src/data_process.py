#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from bert_pre_train import BertPreTrain


class DataProcess(object):
    def __init__(self, _show_token=False):
        self.bert_batch_size = 8
        self.batch_size = 32
        self.data_path = None
        self.show_token = _show_token
        self.data = None
        self.bert_model = BertPreTrain(mode='remote')
        self.data_x = None
        self.data_y = None

    def load_data(self, file_list, is_shuffle=True):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:
            data_tmp = pd.read_csv(open(i, encoding='utf-8'), header=0, sep='\t', engine='c', error_bad_lines=False)
            data_tmp.columns = ["sentence_1", "sentence_2", "label"]

            data = pd.concat([data, data_tmp])

        if is_shuffle:
            data = shuffle(data)
        self.data = data

    def get_feature(self, _data_path, mode='online'):
        if mode == 'online':
            data_x = []
            data_y = []

            _counter = 1
            _sentence_pair_list = []
            _data_y_list = []
            for index, row in tqdm(self.data.iterrows()):

                if _counter % 32 == 0:
                    data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))
                    data_y.extend(_data_y_list)

                    _sentence_pair_list = []
                    _data_y_list = []
                    _counter = 1
                else:
                    try:
                        if int(row['label']) == 1:
                            _data_y_list.append([0, 1])
                        elif int(row['label']) == 0:
                            _data_y_list.append([1, 0])

                        _sentence_pair = " ||| ".join([str(row['sentence_1']), str(row['sentence_2'])])
                        _sentence_pair_list.append(_sentence_pair)
                        _counter += 1
                    except ValueError:
                        pass

            data_x.extend(list(self.bert_model.get_output(_sentence_pair_list, _show_tokens=False)))
            data_y.extend(_data_y_list)

            data_x = np.array(data_x)
            data_y = np.array(data_y)

        #     print("start save npy...")
        #     if not os.path.isdir('../data/{0}'.format(str(_data_path))):
        #         os.mkdir('../data/{0}'.format(str(_data_path)))
        #
        #     np.save('../data/{0}/data_x.npy'.format(str(_data_path)), np.array(data_x, dtype=np.float16))
        #     np.save('../data/{0}/data_y.npy'.format(str(_data_path)), np.array(data_y, dtype=np.float16))
        #
        else:
            print("start load npy...")
            data_x = np.load('../data/{0}/data_x.npy'.format(str(_data_path)))
            data_y = np.load('../data/{0}/data_y.npy'.format(str(_data_path)))

            data_x = np.array(data_x, dtype=np.float32)
            data_y = np.array(data_y, dtype=np.float32)

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
    data_list = ['../data/ca/task3_train_100.txt',
                 ]

    a = DataProcess(_show_token=False)
    a.load_data(file_list=data_list)

    a.get_feature(mode='offline', _data_path='100')

    for x, y in a.next_batch():
        print(x.dtype, y.shape)
