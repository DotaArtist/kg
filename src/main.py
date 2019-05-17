#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import tensorflow as tf
from data_process import DataProcess
from sklearn.metrics import classification_report
from mode_1 import Model1

train_data_list = ['../data/ca/task3_train_train.txt']

test_data_list = ['../data/ca/task3_train_test.txt']

bert_size = 768
class_num = 2
learning_rate = 0.0001

model = Model1(learning_rate=0.0001)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_data_process = DataProcess()
    train_data_process.load_data(file_list=train_data_list)
    train_data_process.get_feature()

    test_data_process = DataProcess()
    test_data_process.load_data(file_list=test_data_list)
    test_data_process.get_feature()

    step = 0
    epoch = 5

    for i in range(epoch):

        for batch_x, batch_y in train_data_process.next_batch():
            _y_hat, _y_pred, _loss, _opt, _acc = sess.run([model.logits,
                                                           model.y_predict_val,
                                                           model.loss_val,
                                                           model.train_op,
                                                           model.accuracy_val,
                                                           ],
                                                          feed_dict={model.input_x: batch_x,
                                                                     model.input_y: batch_y,
                                                                     model.keep_prob: 0.9})

            step += 1
            if step % 100 == 0:
                print("===step:{0} ===loss:{1}".format(step, _loss))

        y_predict_list = []
        y_list = []
        for batch_x, batch_y in test_data_process.next_batch():
            _y_pred = sess.run([model.predict()], feed_dict={
                model.input_x: batch_x,
                model.input_y: batch_y,
                model.keep_prob: 1.0,
            })
            y_predict_list.extend(list(_y_pred[0]))
            y_list.extend(list(batch_y))

        y_label_list = [0 if i[0] == 0 else 1 for i in y_list]
        print("====epoch: {0}".format(epoch))
        print(classification_report(y_true=y_label_list, y_pred=y_predict_list))
