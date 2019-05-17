# coding=utf-8
"""model_1"""
import tensorflow as tf


class Model1(object):
    def __init__(self, is_training=True, num_classes=2, learning_rate=0.0001, bert_size=768, keep_prob=0.9):
        self.is_training = is_training
        self.num_classes = num_classes
        self.bert_size = bert_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate

        self.input_x = tf.placeholder(tf.float32, shape=[None, bert_size], name='input_x')
        self.input_y = tf.placeholder(tf.int64, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.logits = self.inference()
        self.y_predict_val = self.predict()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.accuracy_val = self.accuracy()

    def inference(self):
        with tf.variable_scope('fc_1', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(shape=[self.bert_size, self.num_classes],
                                      initializer=tf.random_normal_initializer(), name="w",
                                      trainable=self.is_training)
            biases = tf.get_variable(shape=[self.num_classes],
                                     initializer=tf.random_normal_initializer(), name="b",
                                     trainable=self.is_training)
            fc_1_output = tf.nn.xw_plus_b(self.input_x, weights, biases)
            fc_1_drop_out = tf.nn.dropout(fc_1_output, self.keep_prob)
            logits = tf.nn.softmax(fc_1_drop_out)
            return logits

    def predict(self):
        y_predict = tf.argmax(self.logits, axis=1, name="y_pred")
        return y_predict

    def loss(self, l2_lambda=0.0001):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = cross_entropy_mean + regularization_losses
        return loss

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
        return train_op

    def accuracy(self):
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.y_predict_val, tf.argmax(self.input_y, axis=1)), tf.float32), name="accuracy")
        return accuracy
