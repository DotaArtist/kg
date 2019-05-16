#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import tensorflow as tf
from data_process import DataProcess
from sklearn.metrics import classification_report
from mode_1 import Model1


train_data_list = ['../data/ca/task3_train_1k.txt',
                   ]

test_data_list = ['../data/ca/task3_train_1k.txt']

bert_size = 768
class_num = 2
learning_rate = 0.0001

input_x = tf.placeholder(tf.float32, shape=[None, bert_size], name='input_x')
input_y = tf.placeholder(tf.int64, shape=[None, class_num], name='input_y')
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
# is_training = tf.placeholder('bool', [])

model = Model1(learning_rate=0.0001)

y_hat = model.inference(input_x)
loss = model.loss(input_x, input_y)
train_op = model.optimize()
y_predict = model.predict(input_x)

accuracy = tf.reduce_mean(tf.cast(tf.equal(y_predict, input_y), tf.float32), name="accuracy")

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
            _, _loss = sess.run([train_op, loss], feed_dict={
                input_x: batch_x,
                input_y: batch_y,
                keep_prob: 0.9,
            })

            step += 1
            print("===step:{0} ===loss:{1}".format(step, _loss))

        y_predict_list = []
        y_list = []
        for batch_x, batch_y in test_data_process.next_batch():
            _y_pred = sess.run([y_predict], feed_dict={
                input_x: batch_x,
                input_y: batch_y,
                keep_prob: 1.0,
            })
            y_predict_list.extend(list(_y_pred[0]))
            y_list.extend(list(batch_y))

        y_label_list = [0 if i[0] == 0 else 1 for i in y_list]
        print("====epoch: {0}".format(epoch))
        print(classification_report(y_true=y_label_list, y_pred=y_predict_list))
