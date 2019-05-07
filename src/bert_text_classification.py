#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import tensorflow as tf
from data_process import DataProcess
from sklearn.metrics import classification_report


train_data_list = ['../../data_v2/标注_买手聊天_训练.xlsx',
                   '../../data_v2/标注_补充.xlsx',
                   '../../data_v2/标注_商品描述_短句训练正.xlsx',
                   '../../data_v2/标注_商品描述_短句训练负_5w.xlsx',
                   '../../data_v2/标注_商品描述_短句训练负_10w.xlsx',
                   '../../data_v2/04_message_train.xlsx',
                   ]
test_data_list = ['../../data_v2/04_message_test.xlsx']

bert_size = 768
class_num = 2
learning_rate = 0.0001

input_x = tf.placeholder(tf.float32, shape=[None, bert_size], name='input_x')
input_y = tf.placeholder(tf.int64, shape=[None, class_num], name='input_y')
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
# is_training = tf.placeholder('bool', [])

with tf.variable_scope('fc_1', reuse=tf.AUTO_REUSE):
    weights = tf.get_variable(shape=[bert_size, class_num], initializer=tf.random_normal_initializer(), name="w", trainable=True)
    biases = tf.get_variable(shape=[class_num], initializer=tf.random_normal_initializer(), name="b", trainable=True)
    fc_1_output = tf.nn.xw_plus_b(input_x, weights, biases)
    fc_1_drop_out = tf.nn.dropout(fc_1_output, keep_prob)

y_hat = tf.nn.softmax(fc_1_drop_out)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=input_y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

y_predict = tf.argmax(y_hat, 1, name="y_pred")
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

        builder = tf.saved_model.builder.SavedModelBuilder("".join(["./bert_model/", str(i)]))
        model_input = tf.saved_model.utils.build_tensor_info(input_x)
        model_output = tf.saved_model.utils.build_tensor_info(y_predict)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'model_input': model_input,
                },
                outputs={'model_output': model_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
        )

        builder.save()
        print('Done exporting!')
