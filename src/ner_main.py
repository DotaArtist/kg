#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import numpy as np
import pandas as pd
import tensorflow as tf
from ner_data_process import DataProcess
from tensorflow.contrib.crf import viterbi_decode
from ner_model_1 import Model1

FEATURE_MODE = 'local'
TRAIN_MODE = 'train'

train_data_list = ['../data/fn/event_type_entity_extract_train_100.csv']
test_data_list = ['../data/fn/event_type_entity_extract_train_100.csv']

model = Model1(learning_rate=0.0001)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if TRAIN_MODE == 'train':

    with tf.Session(config=config) as sess:
        sess.run(init)

        train_data_process = DataProcess(feature_mode=FEATURE_MODE)
        train_data_process.load_data(file_list=train_data_list)
        train_data_process.get_feature()

        test_data_process = DataProcess(feature_mode=FEATURE_MODE)
        test_data_process.load_data(file_list=test_data_list, is_shuffle=False)
        test_data_process.get_feature()

        step = 0
        epoch = 20

        for i in range(epoch):
            for batch_x, batch_y in train_data_process.next_batch():
                model.is_training = True
                _seq_len = np.array([len(_) for _ in batch_x])
                _logits, _loss, _opt, transition_params = sess.run([model.logits,
                                                                    model.loss_val,
                                                                    model.train_op,
                                                                    model.transition_params
                                                                    ],
                                                                   feed_dict={model.input_x: batch_x,
                                                                              model.input_y: batch_y,
                                                                              model.sequence_lengths: _seq_len,
                                                                              model.keep_prob: 0.8})

                step += 1

                for logit, seq_len in zip(_logits, _seq_len):
                    viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                    print(viterbi_seq)

                if step % 1000 == 0:
                    print("===step:{0} ===loss:{1}".format(step, _loss))
