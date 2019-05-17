#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

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
        self.bert_model = BertPreTrain()
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

    def get_feature(self):
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

        self.data_x = data_x
        self.data_y = data_y

        print("data_x shape:", np.array(data_x).shape)
        print("data_y shape:", np.array(data_y).shape)

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
    data_list = ['../data/ca/task3_train_train.txt',
                 ]

    a = DataProcess(_show_token=False)
    a.load_data(file_list=data_list)

    a.get_feature()

    for x, y in a.next_batch():
        print(x.shape, y.shape)
