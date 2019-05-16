# coding=utf-8
"""fc"""
import tensorflow as tf


class Model1(object):
    def __init__(self, is_training=True, num_classes=2, learning_rate=0.0001, bert_size=768, keep_prob=0.9):
        self.is_training = is_training
        self.num_classes = num_classes
        self.bert_size = bert_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate

        self.y_hat = None
        self.loss = None
        self.learning_rate = None
        self.train_op = None
        self.y_predict = None

    def inference(self, input_x):
        with tf.variable_scope('fc_1', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(shape=[self.bert_size, self.num_classes],
                                      initializer=tf.random_normal_initializer(), name="w",
                                      trainable=self.is_training)
            biases = tf.get_variable(shape=[self.num_classes],
                                     initializer=tf.random_normal_initializer(), name="b",
                                     trainable=self.is_training)
            fc_1_output = tf.nn.xw_plus_b(input_x, weights, biases)
            fc_1_drop_out = tf.nn.dropout(fc_1_output, self.keep_prob)
            y_hat = tf.nn.softmax(fc_1_drop_out)
            self.y_hat = y_hat
            return self.y_hat

    def predict(self, input_x):
        y_hat = self.inference(input_x)
        y_predict = tf.argmax(y_hat, 1, name="y_pred")
        self.y_predict = y_predict
        return self.y_predict

    def loss(self, input_x, input_y):
        y_hat = self.inference(input_x)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=input_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([cross_entropy_mean] + regularization_losses)
        self.loss = loss
        return self.loss

    def optimize(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.train_op = train_op
        return self.train_op
