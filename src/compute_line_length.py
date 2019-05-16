# coding=utf-8

train_data_path = '../data/ca/task3_train.txt'

max_len = 0

with open(train_data_path, encoding='utf-8', mode='r') as f1:
    for line in f1.readlines():
        if len(line) > max_len:
            max_len = len(line)

print(max_len)
