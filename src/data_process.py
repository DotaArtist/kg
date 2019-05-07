#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from ber_pre_train import BertPreTrain


class DataProcess(object):
    def __init__(self):
        self.batch_size = 32
        self.data_path = None
        self.data = None
        self.bert_model = BertPreTrain()
        self.data_x = None
        self.data_y = None

    def load_data(self, file_list):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:
            data_tmp = pd.read_excel(i, header=0)
            data_tmp.columns = ["pid", "label", "content"]

            data = pd.concat([data, data_tmp])

        data = shuffle(data)
        self.data = data

    def get_feature(self):
        data_x = []
        data_y = []
        for index, row in tqdm(self.data.iterrows()):
            try:
                data_x.append(list(self.bert_model.get_output([str(row['content'])])[0]))
                if int(row['label']) == 1:
                    data_y.append([0, 1])
                elif int(row['label']) == 0:
                    data_y.append([1, 0])

            except TypeError:
                print(row['content'])

            except UnicodeEncodeError:
                _ = row['content'].encode('utf-8').strip()
                print(_)
                data_x.append(list(self.bert_model.get_output([_])[0]))
                if int(row['label']) == 1:
                    data_y.append([0, 1])
                elif int(row['label']) == 0:
                    data_y.append([1, 0])

        self.data_x = data_x
        self.data_y = data_y

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
    data_list = ['../../data_v2/标注_买手聊天_训练.xlsx',
                 '../../data_v2/标注_补充.xlsx',
                 '../../data_v2/标注_商品描述_短句训练正.xlsx',
                 '../../data_v2/标注_商品描述_短句训练负_5w.xlsx',
                 '../../data_v2/标注_商品描述_短句训练负_10w.xlsx',
                 '../../data_v2/04_message_train.xlsx',
                 ]

    a = DataProcess()
    a.load_data(file_list=data_list)
    a.get_feature()

    for _x, _y in a.next_batch():
        print(_x.shape, _y.shape)
