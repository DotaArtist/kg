#!/usr/bin/env bash

# bert pre-train
bert-serving-start -pooling_strategy CLS_TOKEN -model_dir ../model/chinese_L-12_H-768_A-12/ -num_worker=1
# show token
bert-serving-start -show_tokens_to_client $true -port 5555 -port_out 5556 -pooling_strategy CLS_TOKEN -model_dir ./chinese_L-12_H-768_A-12/ -num_worker=1

## Strategy	Description
NONE	no pooling at all, useful when you want to use word embedding instead of sentence embedding. This will results in a [max_seq_len, 768] encode matrix for a sequence.
REDUCE_MEAN	take the average of the hidden state of encoding layer on the time axis
REDUCE_MAX	take the maximum of the hidden state of encoding layer on the time axis
REDUCE_MEAN_MAX	do REDUCE_MEAN and REDUCE_MAX separately and then concat them together on the last axis, resulting in 1536-dim sentence encodes
CLS_TOKEN or FIRST_TOKEN	get the hidden state corresponding to [CLS], i.e. the first token
SEP_TOKEN or LAST_TOKEN	get the hidden state corresponding to [SEP], i.e. the last token


# bert server
docker build -t bert-as-service -f ./Dockerfile .
NUM_WORKER=1
PATH_MODEL=/data/yp/chinese_L-12_H-768_A-12/
docker run --runtime nvidia -dit -p 5555:5555 -p 5556:5556 -v $PATH_MODEL:/model -t bert-as-service $NUM_WORKER

# ca
bert-serving-start -show_tokens_to_client $true -port 5555 -port_out 5556 -pooling_strategy CLS_TOKEN -max_seq_len 250 -model_dir ./chinese_L-12_H-768_A-12/ -num_worker=1

#
bert-serving-start -port 5555 -port_out 5556 -pooling_strategy NONE -max_seq_len 100 -model_dir ./chinese_L-12_H-768_A-12/ -num_worker=1