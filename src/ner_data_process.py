#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from bert_pre_train import BertPreTrain

MAX_LEN_SENTENCE = 200


def get_ner_label(sentence, target):
    _label = [[1, 0, 0, 0] for _ in range(MAX_LEN_SENTENCE)]
    try:
        _start = sentence.index(target)
        _end = _start + len(target) - 1

        _label[_start] = [0, 1, 0, 0]
        _label[_end] = [0, 0, 0, 1]
        _label = [[0, 0, 1, 0] if _start < _index < _end else _ for _index, _ in enumerate(_label)]

        return _label
    except ValueError:
        return [[1, 0, 0, 0] for _ in range(MAX_LEN_SENTENCE)]


class DataProcess(object):
    def __init__(self, _show_token=False, feature_mode='remote'):
        self.bert_batch_size = 32
        self.batch_size = 32
        self.data_path = None
        self.show_token = _show_token
        self.data = None
        self.bert_model = BertPreTrain(mode=feature_mode)
        self.data_x = None
        self.data_y = None

    def load_data(self, file_list, is_shuffle=True):
        self.data_path = file_list
        data = pd.DataFrame()
        for i in file_list:

            idx_list = []
            sent_list = []
            type_list = []
            ner_list = []

            data_tmp = pd.DataFrame()
            with open(i, encoding='utf-8', mode='r') as _f:
                for line in _f.readlines():
                    try:
                        idx, sent, ty, ner = line.strip().strip("\"").split('\",\"')
                    except ValueError:
                        idx, sent, ty = line.strip().strip("\"").split('\",\"')
                        ner = 'NONE'

                    idx_list.append(idx)
                    sent_list.append(sent)
                    type_list.append(ty)
                    ner_list.append(ner)

            data_tmp['idx'] = pd.Series(idx_list)
            data_tmp['sentence'] = pd.Series(sent_list)
            data_tmp['type'] = pd.Series(type_list)
            data_tmp['ner'] = pd.Series(ner_list)

            data = pd.concat([data, data_tmp])

        if is_shuffle:
            data = shuffle(data)
        self.data = data

    def get_feature(self):
        data_x = []
        data_y = []

        _sentence_pair_list = []

        for index, row in tqdm(self.data.iterrows()):

            label = get_ner_label(row['sentence'], row['ner'])
            data_y.append(label)

            _sentence_pair = " ||| ".join([row['sentence'], row['type']])
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
    data_list = ['../data/fn/event_type_entity_extract_train_100.csv',
                 ]

    a = DataProcess(_show_token=False, feature_mode='remote')
    a.load_data(file_list=data_list, is_shuffle=False)

    a.get_feature()

    for x, y in a.next_batch():
        print(x.shape, y.shape)
