#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import numpy as np
import pandas as pd
import tensorflow as tf
from data_process import DataProcess
from sklearn.metrics import classification_report
from model_3 import Model3

FEATURE_MODE = 'remote'
TRAIN_MODE = 'train'

train_data_list = ['../data/ca/task3_train_5w.txt']
test_data_list = ['../data/ca/task3_train_test.txt']

model = Model3(learning_rate=0.0001)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if TRAIN_MODE == 'train':

    with tf.Session(config=config) as sess:
        sess.run(init)

        train_data_process = DataProcess(mode=FEATURE_MODE)
        train_data_process.load_data(file_list=train_data_list)
        train_data_process.get_feature()

        test_data_process = DataProcess(mode=FEATURE_MODE)
        test_data_process.load_data(file_list=test_data_list, is_shuffle=False)
        test_data_process.get_feature()

        step = 0
        epoch = 20

        for i in range(epoch):
            for batch_x, batch_y in train_data_process.next_batch():
                model.is_training = True
                _seq_len = np.array([len(_) for _ in batch_x])
                _loss, _opt = sess.run([model.loss_val, model.train_op],
                                       feed_dict={model.input_x: batch_x,
                                                  model.input_y: batch_y,
                                                  model.sequence_lengths: _seq_len,
                                                  model.keep_prob: 0.8})

                step += 1
                if step % 1000 == 0:
                    print("===step:{0} ===loss:{1}".format(step, _loss))

            save_path = saver.save(sess, "../model/%s/model_epoch_%s" % (str(i), str(i)))

            y_predict_list = []
            y_hat_list = []
            y_list = []
            for batch_x, batch_y in test_data_process.next_batch():
                model.is_training = False
                _seq_len = np.array([len(_) for _ in batch_x])
                _y_pred, _y_hat = sess.run([model.y_predict_val, model.logits], feed_dict={
                    model.input_x: batch_x,
                    model.input_y: batch_y,
                    model.keep_prob: 1.0,
                    model.sequence_lengths: _seq_len
                })
                y_predict_list.extend(list(_y_pred[0]))
                y_list.extend(list(batch_y))
                y_hat_list.extend(list(_y_hat))

            y_label_list = [0 if _i[0] == 0 else 1 for _i in y_list]

            _out_file = test_data_process.data
            _out_file['y_hat'] = pd.Series(y_hat_list)

            print("====epoch: {0}".format(i))
            print(classification_report(y_true=y_label_list, y_pred=y_predict_list))
            _out_file.to_csv('./test_predict_{0}.tsv'.format(str(i)), sep='\t')


elif TRAIN_MODE == 'predict':
    predict_data_list = ['../data/ca/task3_train_test.txt']

    predict_data_process = DataProcess(mode=FEATURE_MODE)
    predict_data_process.load_data(file_list=predict_data_list)
    predict_data_process.get_feature()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../model/model_epoch_1")

        for batch_x, batch_y in predict_data_process.next_batch():
            model.is_training = False
            _y_pred = sess.run([model.y_predict_val], feed_dict={model.input_x: batch_x,
                                                                 model.input_y: batch_y,
                                                                 model.sequence_lengths: batch_x.shape[1],
                                                                 model.keep_prob: 1.0})
