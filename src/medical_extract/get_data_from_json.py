# coding=utf-8
"""解析标注数据"""
import json
import codecs

path = 'C:/Users/YP_TR/Desktop/数据标注/已标注数据/已标注_unlabel_ner_02_11.json'
out_path = 'C:/Users/YP_TR/Desktop/数据标注/标准训练数据/11.txt'

data = json.load(codecs.open(path, mode='r', encoding='utf-8-sig'))
example_items = data['rasa_nlu_data']['common_examples']


with open(out_path, mode='w', encoding='utf-8') as f1:
    for _example in example_items:
        _text = _example['text'].strip()
        _entity_list = _example['entities']

        _ner_list = []

        for _entity in _entity_list:
            _ner_list.append('@@'.join([_entity['entity'], _entity['value']]))

        _ner_label = '&&'.join(_ner_list)

        if _ner_label == '':
            _ner_label = 'null'
        f1.writelines('{}\tner_label:{}\n'.format(_text, _ner_label))
