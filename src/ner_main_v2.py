#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from nuanwa_ner_data_process_v2 import *
from tensorflow.contrib.crf import viterbi_decode
from ner_model_3 import Model3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURE_MODE = 'local'
TRAIN_MODE = 'demo'

# train_data_list = ['../data/medical_record/train_3w.txt']
# test_data_list = ['../data/medical_record/test_5k.txt']

train_data_list = ['../data/medical_record/normal_train/100.txt']
test_data_list = ['../data/medical_record/normal_train/100.txt']

model = Model3(learning_rate=0.0001, sequence_length_val=100, num_tags=15)

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
                _logits, _loss, _opt, transition_params, decode_tags = sess.run([model.logits,
                                                                                 model.loss_val,
                                                                                 model.train_op,
                                                                                 model.transition_params,
                                                                                 model.decode_tags
                                                                                 ],
                                                                                feed_dict={model.input_x: batch_x,
                                                                                           model.input_y: batch_y,
                                                                                           model.sequence_lengths: _seq_len,
                                                                                           model.keep_prob: 0.8})

                step += 1

                if step % 10 == 0:
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
                _, decode_tags = sess.run([model.logits, model.decode_tags], feed_dict={model.input_x: batch_x,
                                                                                        model.input_y: batch_y,
                                                                                        model.sequence_lengths: _seq_len,
                                                                                        model.keep_prob: 1.0})
                for _y_label, _decode_tags in zip(batch_y, decode_tags):
                    if list(_decode_tags) == list(_y_label):
                        right_counter += 1
                    sum_counter += 1

            print("epoch: {}======acc rate: {}".format(str(i), str(right_counter / sum_counter)))

if TRAIN_MODE == 'demo':
    predict_data_list = ['../data/medical_record/train_3w.txt']

    predict_data_process = DataProcess(feature_mode=FEATURE_MODE)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../model/19/model_epoch_30")

        sentence_str = ""

        while sentence_str != 'q':
            sentence_str = input("input:")
            batch_x, batch_y = predict_data_process.get_one_sentence_feature(sentence_str)
            model.is_training = False
            _seq_len = np.array([len(_) for _ in batch_x])
            _logits, _loss, transition_params = sess.run([model.logits,
                                                          model.loss_val,
                                                          model.transition_params],
                                                         feed_dict={model.input_x: batch_x,
                                                                    model.sequence_lengths: _seq_len,
                                                                    model.keep_prob: 1.0})

            for logit, seq_len in zip(_logits, _seq_len):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[1, 2])
                print('disease: {}'.format(out))
