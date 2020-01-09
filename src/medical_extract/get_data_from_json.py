# coding=utf-8
"""解析标注数据"""
import os
import json
import codecs

path = 'C:/Users/YP_TR/Desktop/回收数据_v2/已标注_unlabel_ner_02_19.json'
out_path = 'C:/Users/YP_TR/Desktop/数据标注/标准训练数据/19.txt'


def process_json(json_path, out_path):
    data = json.load(codecs.open(json_path, mode='r', encoding='utf-8-sig'))
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


def main():
    path = 'C:/Users/YP_TR/Desktop/回收数据_v2/'
    out = 'C:/Users/YP_TR/Desktop/回收训练数据/'

    file_list = os.listdir(path)
    file_list = [i for i in file_list if i[-4:] == 'json']

    for i in file_list:
        process_json(os.path.join(path, i), os.path.join(out, i[:-4] + 'txt'))


if __name__ == '__main__':
    main()
