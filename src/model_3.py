# coding=utf-8
"""model_3"""
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


def attention(inputs, attention_size, name="att", train_type=True):
    """
    Attention mechanism layer.

    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :param name:name of op
    :param train_type:for test, non random uninitialized
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    with tf.name_scope(name):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)

        sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

        # Attention mechanism
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name='att_w', trainable=train_type)

        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='att_b', trainable=train_type)

        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='att_u', trainable=train_type)

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w_omega) + tf.reshape(b_omega, [1, -1]), name='att_v')
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]), name='att_vu')
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length], name='att_exps')
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1], name='att_alphas')

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1, name='att_output')
        return output, w_omega


class Model3(object):
    def __init__(self, is_training=True, num_classes=2, learning_rate=0.0001,
                 bert_size=768, sequence_length_val=150,
                 keep_prob=0.9, fc_hidden_num=200,
                 bilstm_hidden_num=100, attention_size=50):
        self.sequence_length_val = sequence_length_val
        self.is_training = is_training
        self.num_classes = num_classes
        self.bert_size = bert_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.bilstm_hidden_num = bilstm_hidden_num
        self.fc_hidden_num = fc_hidden_num
        self.attention_size = attention_size

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.sequence_length_val, self.bert_size], name='input_x')
        self.input_y = tf.placeholder(tf.int64, shape=[None, self.num_classes], name='input_y')
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

        with tf.variable_scope('attention_layer', reuse=tf.AUTO_REUSE):
            attention_output, _attention_weights = attention(bilstm_layer_output, self.attention_size, "att",
                                                             train_type=self.is_training)

        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(shape=[self.bilstm_hidden_num*2, self.num_classes],
                                      initializer=tf.random_normal_initializer(), name="w",
                                      trainable=self.is_training)
            biases = tf.get_variable(shape=[self.num_classes],
                                     initializer=tf.random_normal_initializer(), name="b",
                                     trainable=self.is_training)
            fc_output = tf.nn.xw_plus_b(attention_output, weights, biases)
            fc_output = tf.nn.relu(fc_output)
            logits = tf.nn.softmax(fc_output)
            return logits

    def predict(self):
        y_predict = tf.argmax(self.logits, axis=1, name="y_pred")
        return y_predict

    def loss(self, l2_lambda=0.0001):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # regularization_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
        # if 'bias' not in v.name]) * l2_lambda
        # loss = cross_entropy_mean + regularization_losses
        return cross_entropy_mean

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
        return train_op

    def accuracy(self):
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.y_predict_val, tf.argmax(self.input_y, axis=1)), tf.float32), name="accuracy")
        return accuracy
