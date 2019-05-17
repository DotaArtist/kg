#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import tensorflow as tf
from data_process import DataProcess
from sklearn.metrics import classification_report
from model_2 import Model2

MODE = 'train'

train_data_list = ['../data/ca/task3_train_train.txt']
test_data_list = ['../data/ca/task3_train_test.txt']

model = Model2(learning_rate=0.0001)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

if MODE == 'train':

    with tf.Session() as sess:
        sess.run(init)

        train_data_process = DataProcess()
        train_data_process.load_data(file_list=train_data_list)
        train_data_process.get_feature()

        test_data_process = DataProcess()
        test_data_process.load_data(file_list=test_data_list)
        test_data_process.get_feature()

        step = 0
        epoch = 20

        for i in range(epoch):
            for batch_x, batch_y in train_data_process.next_batch():
                model.is_training = True
                _loss, _opt = sess.run([model.loss_val, model.train_op],
                                       feed_dict={model.input_x: batch_x,
                                                  model.input_y: batch_y,
                                                  model.keep_prob: 0.8})

                step += 1
                if step % 10 == 0:
                    print("===step:{0} ===loss:{1}".format(step, _loss))

            save_path = saver.save(sess, "../model/%s/model_epoch_%s" % (str(i), str(i)))

            y_predict_list = []
            y_list = []
            for batch_x, batch_y in test_data_process.next_batch():
                model.is_training = False
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


elif MODE == 'predict':
    predict_data_list = ['../data/ca/task3_train_test.txt']

    predict_data_process = DataProcess()
    predict_data_process.load_data(file_list=predict_data_list)
    predict_data_process.get_feature()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../model/model_epoch_1")

        for batch_x, batch_y in predict_data_process.next_batch():
            model.is_training = False
            _y_pred = sess.run([model.y_predict_val], feed_dict={model.input_x: batch_x,
                                                                 model.input_y: batch_y,
                                                                 model.keep_prob: 1.0})
