# coding=utf-8

train_data_path = '../data/ca/task3_dev.txt'

max_len = 0

with open(train_data_path, encoding='utf-8', mode='r') as f1:
    for line in f1.readlines():
        sentence_1, sentence_2, label = line.strip().split('\t')
        _sentence_len = max(len(sentence_1), len(sentence_2))

        if _sentence_len > max_len:
            max_len = _sentence_len

print(max_len)
