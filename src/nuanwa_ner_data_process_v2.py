#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""电子病历抽取"""

__author__ = 'yp'

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from bert_pre_train import BertPreTrain

MAX_LEN_SENTENCE = 100


ner_label_map = {
    'other': 0,
    'disease': 1, 'disease_i': 2,
    'symptom': 3, 'symptom_i': 4,
    'drug': 5, 'drug_i': 6,
    'diagnosis': 7, 'diagnosis_i': 8,
    'duration': 9, 'duration_i': 10,
    'start_time': 11, 'start_time_i': 12,
    'end_time': 13, 'end_time_i': 14
}


def get_disease_from_tag(sentence, tag, target=[1, 2]):
    sentence = list(sentence)
    out = []
    counter = 0
    for word, index in zip(sentence, tag):
        if counter == 1 and index not in target:
            counter = 0
            out.append(',')

        if index in target:
            counter = 1
            out.append(word)
    print(''.join(out))


def get_ner_label(sentence, target):
    if len(sentence) > MAX_LEN_SENTENCE:
        sentence = sentence[:MAX_LEN_SENTENCE]
    _label = [0 for _ in range(MAX_LEN_SENTENCE)]

    target = target.split(':')[1]
    if target == 'null':
        return _label

    target_list = target.split('&&')
    for _target in target_list:
        _key, _value = _target.split('@@')

        for m in re.finditer(_value, sentence):
            _label[m.start()] = ner_label_map[_key]
            if len(_value) > 1:
                _label[m.start() + 1: m.start() + len(_value)] = [ner_label_map[_key + '_i'] for _ in range(len(_value) - 1)]

    return _label


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

            sent_list = []
            ner_list = []

            data_tmp = pd.DataFrame()
            with open(i, encoding='utf-8', mode='r') as _f:
                for line in _f.readlines():
                    try:
                        sent, ner = line.strip().strip("\n").split('\t')
                    except ValueError:
                        sent = line.strip().strip("\n").split('\t')
                        ner = 'NONE'

                    sent_list.append(sent)
                    ner_list.append(ner)

            data_tmp['sentence'] = pd.Series(sent_list)
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

            _sentence_pair = row['sentence']
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

    def get_one_sentence_feature(self, sentence):
        data_x = []
        data_y = []
        data_x.extend(list(self.bert_model.get_output([sentence], _show_tokens=False)))
        data_y.append([0 for _ in range(MAX_LEN_SENTENCE)])
        return np.array(data_x), np.array(data_y, dtype=np.int64)


def test():
    a = '晨起痰多黄白，咳嗽，无发热'
    b = [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0]
    get_disease_from_tag(sentence=a, tag=b)


if __name__ == '__main__':
    test()
    # data_list = [
    #     '../data/fn/event_type_entity_extract_train_100.csv',
        # '../data/medical_record/normal_train/100.txt'
    # ]
    #
    # a = DataProcess(_show_token=False, feature_mode='remote')
    # a.load_data(file_list=data_list, is_shuffle=False)
    #
    # a.get_feature()
    #
    # for x, y in a.next_batch():
    #     print(x.shape, y.shape)
    #     print(y)
    #
    # x, y = a.get_one_sentence_feature('今天天气很好')
    # print(x.shape, y.shape)
    # print(y)
