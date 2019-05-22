# coding=utf-8

train_data_path = '../data/fn/event_type_entity_extract_eval.csv'

max_len = 0

with open(train_data_path, encoding='utf-8', mode='r') as f1:
    for line in f1.readlines():
        idx, sent, ty = line.strip().strip("\"").split('\",\"')
        _sentence_len = len(sent) + len(ty)

        if _sentence_len > max_len:
            max_len = _sentence_len
            print(idx)

print(max_len)
