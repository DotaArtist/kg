#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import os
import json
import tensorflow as tf
from nuanwa_ner_data_process_v2 import *
from tensorflow.contrib.crf import viterbi_decode
from ner_model_1 import Model1
from flask import abort
from flask import Flask
from flask import request
from mylogparse import LogParse

a = LogParse()
a.set_profile(path="./log", filename="test")

app = Flask(__name__)
app.logger.addHandler(a.handler)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURE_MODE = 'local'

model = Model1(learning_rate=0.0001, sequence_length_val=100, num_tags=15)

init = tf.global_variables_initializer()
# saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver()
saver.restore(sess, "../model_v2/30/model_epoch_30")

predict_data_process = DataProcess(feature_mode=FEATURE_MODE)


@a.exception
@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        if "sentence" in request.json.keys():
            disease_out_list = []
            symp_out_list = []
            drug_out_list = []
            diagnosis_out_list = []
            duration_out_list = []

            sentence_str = request.json["sentence"]

            batch_x, batch_y = predict_data_process.get_one_sentence_feature(sentence_str)
            model.is_training = False
            _seq_len = np.array([len(_) for _ in batch_x])
            _logits, _loss, transition_params = sess.run([model.logits,
                                                          model.loss_val,
                                                          model.transition_params],
                                                         feed_dict={model.input_x: batch_x,
                                                                    model.input_y: batch_y,
                                                                    model.sequence_lengths: _seq_len,
                                                                    model.keep_prob: 1.0})

            for logit, seq_len in zip(_logits, _seq_len):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                disease_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[1, 2])
                symp_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[3, 4])
                drug_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[5, 6])
                diagnosis_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[7, 8])
                duration_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[9, 10])

                disease_out_list.append(disease_out)
                symp_out_list.append(symp_out)
                drug_out_list.append(drug_out)
                diagnosis_out_list.append(diagnosis_out)
                duration_out_list.append(duration_out)

            ner_dict = dict()
            ner_dict['dise'] = disease_out_list
            ner_dict['symp'] = symp_out_list
            ner_dict['drug'] = drug_out_list
            ner_dict['diag'] = diagnosis_out_list
            ner_dict['dura'] = duration_out_list

            return json.dumps(ner_dict)

        else:
            abort(404)

    elif request.method == 'GET':
        if request.args.get('sentence'):
            disease_out_list = []
            symp_out_list = []
            drug_out_list = []
            diagnosis_out_list = []
            duration_out_list = []

            sentence_str = request.args.get('sentence')

            batch_x, batch_y = predict_data_process.get_one_sentence_feature(sentence_str)
            model.is_training = False
            _seq_len = np.array([len(_) for _ in batch_x])
            _logits, _loss, transition_params = sess.run([model.logits,
                                                          model.loss_val,
                                                          model.transition_params],
                                                         feed_dict={model.input_x: batch_x,
                                                                    model.input_y: batch_y,
                                                                    model.sequence_lengths: _seq_len,
                                                                    model.keep_prob: 1.0})

            for logit, seq_len in zip(_logits, _seq_len):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                disease_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[1, 2])
                symp_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[3, 4])
                drug_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[5, 6])
                diagnosis_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[7, 8])
                duration_out = get_disease_from_tag(sentence=sentence_str, tag=viterbi_seq, target=[9, 10])

                disease_out_list.append(disease_out)
                symp_out_list.append(symp_out)
                drug_out_list.append(drug_out)
                diagnosis_out_list.append(diagnosis_out)
                duration_out_list.append(duration_out)

            ner_dict = dict()
            ner_dict['dise'] = disease_out_list
            ner_dict['symp'] = symp_out_list
            ner_dict['drug'] = drug_out_list
            ner_dict['diag'] = diagnosis_out_list
            ner_dict['dura'] = duration_out_list

            return json.dumps(ner_dict)

        else:
            abort(404)

    else:
        abort(404)


if __name__ == "__main__":
    app.run('0.0.0.0', port=1001)
