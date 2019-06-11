#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from nuanwa_ner_data_process import DataProcess
from tensorflow.contrib.crf import viterbi_decode
from ner_model_1 import Model1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURE_MODE = 'remote'
TRAIN_MODE = 'train'

train_data_list = ['../data/medical_record/train_3w.txt']
test_data_list = ['../data/medical_record/test_5k.txt']

model = Model1(learning_rate=0.0001, sequence_length_val=100)

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

            sum_counter = 0
            right_counter = 0

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

                for logit, seq_len, _y_label in zip(_logits, _seq_len, batch_y):
                    viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                    if viterbi_seq == list(_y_label):
                        right_counter += 1
                    sum_counter += 1

                if step % 500 == 0:
                    print("step:{0} ===loss:{1}".format(step, _loss))

            save_path = saver.save(sess, "../model/%s/model_epoch_%s" % (str(i), str(i)))

            # test
            y_predict_list = []
            y_label_list = []

            sum_counter = 0
            right_counter = 0
            for batch_x, batch_y in test_data_process.next_batch():
                model.is_training = False
                _seq_len = np.array([len(_) for _ in batch_x])
                _logits, transition_params = sess.run([model.logits,
                                                       model.transition_params], feed_dict=
                                                      {model.input_x: batch_x,
                                                       model.input_y: batch_y,
                                                       model.sequence_lengths: _seq_len,
                                                       model.keep_prob: 0.8})
                for logit, seq_len, _y_label in zip(_logits, _seq_len, batch_y):
                    viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                    if viterbi_seq == list(_y_label):
                        right_counter += 1
                    sum_counter += 1

            print("======acc rate: {}".format(str(right_counter / sum_counter)))

if TRAIN_MODE == 'demo':
    predict_data_list = ['../data/medical_record/train_3w.txt']

    predict_data_process = DataProcess(feature_mode=FEATURE_MODE)
    predict_data_process.load_data(file_list=predict_data_list)
    predict_data_process.get_feature()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../model/19/model_epoch_19")

        y_predict_list = []
        for batch_x, batch_y in predict_data_process.next_batch():
            model.is_training = False
            _seq_len = np.array([len(_) for _ in batch_x])
            _logits, transition_params = sess.run([model.logits,
                                                   model.transition_params],
                                                  feed_dict={model.input_x: batch_x,
                                                             model.sequence_lengths: _seq_len,
                                                             model.keep_prob: 1.0})

            for logit, seq_len in zip(_logits, _seq_len):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                y_predict_list.append(viterbi_seq)

        _out_file = predict_data_process.data
        _out_file['y_pred'] = pd.Series(y_predict_list)
        _out_file.to_csv('./final_predict.tsv', sep='\t')
