# coding=utf-8
"""ner_model_1"""
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class Model1(object):
    def __init__(self, is_training=True, num_tags=4, learning_rate=0.0001,
                 bert_size=768, sequence_length_val=200,
                 keep_prob=0.9, fc_hidden_num=200,
                 bilstm_hidden_num=100):
        self.sequence_length_val = sequence_length_val
        self.is_training = is_training
        self.num_tags = num_tags
        self.bert_size = bert_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.bilstm_hidden_num = bilstm_hidden_num
        self.fc_hidden_num = fc_hidden_num

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.sequence_length_val, self.bert_size], name='input_x')
        self.input_y = tf.placeholder(tf.int64, shape=[None, self.sequence_length_val, self.num_tags], name='input_y')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.logits = self.inference()
        self.y_predict_val = self.predict()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.accuracy_val = self.accuracy()

    def inference(self):
        with tf.variable_scope('bilstm_layer', reuse=tf.AUTO_REUSE):
            cell_fw = LSTMCell(self.bilstm_hidden_num)
            cell_bw = LSTMCell(self.bilstm_hidden_num)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.input_x,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            bilstm_layer_output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            bilstm_layer_output = tf.nn.dropout(bilstm_layer_output, self.keep_prob)

        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(shape=[self.bilstm_hidden_num * 2, self.num_tags],
                                      initializer=tf.random_normal_initializer(), name="w",
                                      trainable=self.is_training)
            biases = tf.get_variable(shape=[self.num_tags],
                                     initializer=tf.random_normal_initializer(), name="b",
                                     trainable=self.is_training)
            fc_output = tf.nn.xw_plus_b(bilstm_layer_output, weights, biases)
            fc_output = tf.nn.relu(fc_output)
            logits = tf.nn.softmax(fc_output)

        with tf.variable_scope('crf_layer', reuse=tf.AUTO_REUSE):
            pass
        return logits

    def predict(self):
        y_predict = tf.argmax(self.logits, axis=1, name="y_pred")
        return y_predict

    def loss(self,):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        return cross_entropy_mean

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
        return train_op

    def accuracy(self):
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.y_predict_val, tf.argmax(self.input_y, axis=1)), tf.float32), name="accuracy")
        return accuracy
