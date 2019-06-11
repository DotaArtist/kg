# coding=utf-8
"""病历拆解分句转json"""

import json
import src.medical_extract.config as config
from src.medical_extract.extract_medical_record import *

symptom_extractor = load_config(config.pattern_symptom)
drug_extractor = load_config(config.pattern_drug)
disease_extractor = load_config(config.pattern_disease)


def get_ner(_sent, _tar, _label):
    _start = _sent.index(_tar)
    _end = _start + len(_tar) - 1
    return {'start': _start, 'end': _end, 'value': _tar, 'entity': _label}


def process_data(_path):
    out_json = {
        "rasa_nlu_data":
            {
                "common_examples": []
            }
    }
    with open(_path, mode='r', encoding='utf-8') as f1:
        for line in f1.readlines():
            _example = dict()
            _example['text'] = line
            _example['intent'] = "ner"
            _example['entities'] = []

            _dis = extract_disease(_disease=line, _disease_extractor=disease_extractor)
            for _ in _dis:
                try:
                    _example['entities'].append(get_ner(_sent=line, _tar=_, _label='disease'))
                except:
                    print(_)

            _sym = extract_symptom(_symptom=line, _symptom_extractor=symptom_extractor)
            for _ in _sym:
                if _ not in ''.join(_dis):
                    try:
                        _example['entities'].append(get_ner(_sent=line, _tar=_, _label='symptom'))
                    except:
                        print(_)

            _drug = extract_drug(line, drug_extractor)
            for _ in _drug:
                try:
                    _example['entities'].append(get_ner(_sent=line, _tar=_, _label='drug'))
                except:
                    print(_)

            _time = extract_time(line)
            for _ in _time:
                try:
                    _example['entities'].append(get_ner(_sent=line, _tar=_, _label='duration'))
                    _example['entities'].append(get_ner(_sent=line, _tar=_, _label='start_time'))
                    _example['entities'].append(get_ner(_sent=line, _tar=_, _label='end_time'))
                except:
                    print(_)
            out_json['rasa_nlu_data']['common_examples'].append(_example)
        json.dump(out_json, open(_path+'.json', mode='w', encoding='utf-8'))


# split -l 500 -d -a 2 nuanwa_ner_test.txt unlabel_ner_test_
if __name__ == '__main__':
    import os

    _path = 'C:/Users/YP_TR/Desktop/数据标注/data_v2/05/'
    file_list = os.listdir(_path)

    for i in file_list:
        process_data(_path + i)
