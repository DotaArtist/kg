# coding=utf-8

import re
import pandas as pd
import flashtext
from src.medical_extract.ac_automation import AcAutomation
import src.medical_extract.config as config


def load_origin_data():
    _origin_data = pd.read_csv(config.data_path, sep='\t', header=0, index_col=0, error_bad_lines=False).fillna('')
    return _origin_data


def load_config(_config_path):
    _config = pd.read_csv(_config_path, sep=',', index_col=0)
    _keys = _config.iloc[:, 0].values.tolist()

    _extractor = flashtext.KeywordProcessor()
    # _extractor = AcAutomation()
    for _key in _keys:
        _extractor.add_keyword(_key)
    return _extractor


def load_disease_config(_config_path, _c_name_path):
    _config = pd.read_csv(_config_path, sep=',', index_col=0)
    _config = _config[_config['编码类型'] == '疾病ICD10编码']
    _disease = _config.iloc[:, 0].values.tolist()

    _c_name = pd.read_csv(_c_name_path, sep=',', index_col=0).fillna('')
    _disease_cname = _c_name['疾病别名'].values.tolist()

    _extractor = flashtext.KeywordProcessor()
    # _extractor = AcAutomation()
    for _key in _disease:
        _extractor.add_keyword(_key)

    for _cna in _disease_cname:
        _extractor.add_keyword(_cna)

    return _extractor


def chinese_to_number(_str):
    _str_list = []
    for i in list(_str):
        __ = config.chinese_number_dict.get(i, 0)
        if __ == 0:
            __ = i
        _str_list.append(str(__))
    return ''.join(_str_list)


def number_to_chinese(_str):
    _str_list = []
    for i in list(_str):
        __ = config.number_chinese_dict.get(i, 0)
        if __ == 0:
            __ = i
        _str_list.append(str(__))
    return ''.join(_str_list)


def delete_same(func):
    if hasattr(func, '__call__'):
        def wrapper(*args, **kw):
            _list = func(*args, **kw)
            _set = set(_list)
            try:
                _set.remove('')
            except KeyError:
                pass
            return list(_set)

        return wrapper

    elif isinstance(func, list):
        _list = []
        for _ in func:
            if len(_) > 0:
                _list.extend(_)

        _set = set(_list)

        try:
            _set.remove('')
        except KeyError:
            pass
        return number_to_chinese(','.join(list(_set)))


def delete_all_number(func):
    def wrapper(*args, **kw):
        _output = []
        _list = func(*args, **kw).split(',')

        for _words in _list:
            _counter = 0
            for j in _words:
                if j in config.chinese_number_dict:
                    _counter += 1
            if _counter != len(_words):
                _output.append(_words)
        return _output

    return wrapper


def delete_short(func):
    def wrapper(*args, **kw):
        _output = []
        _list = func(*args, **kw)

        for _words in _list:
            if len(_words) > 2:
                _output.append(_words)
        return _output

    return wrapper


def delete_deny(func):
    def wrapper(*args, **kw):
        _list = func(*args, **kw)
        _list_no_deny = list()

        if '_disease' in kw.keys():
            _sentence = kw.get('_disease')
        elif '_symptom' in kw.keys():
            _sentence = kw.get('_symptom')
        else:
            _sentence = ''

        for _ in _list:
            try:
                _location = _sentence.index(_)
                if _sentence[_location - 1] != '无':
                    _list_no_deny.append(_)
            except ValueError:
                print(_sentence, _)
        return _list_no_deny

    return wrapper


@delete_same
@delete_deny
def extract_symptom(_symptom, _symptom_extractor):
    return _symptom_extractor.extract_keywords(number_to_chinese(_symptom))


@delete_same
def extract_drug(_drug, _drug_extractor):
    return _drug_extractor.extract_keywords(number_to_chinese(_drug))


@delete_same
def extract_operation(_operation, _operation_extractor):
    return _operation_extractor.extract_keywords(number_to_chinese(_operation))


@delete_same
@delete_deny
def extract_disease(_disease, _disease_extractor):
    return _disease_extractor.extract_keywords(number_to_chinese(_disease))


@delete_same
def extract_time(_time):
    output = []
    for _str in re.split('[，。,;；]', _time):
        _str = chinese_to_number(_str)
        new_pattern = '([0-9]{1,4}[年\\-\\/][0-9]{1,2}[月\\-\\/][0-9]{1,2}[日\\-\\/]{0,1})|([0-9-数多]{1,4}[\u4e00-\u9fa5]*?[时日天年周月分][余钟前]{0,1})'
        _output = re.findall(new_pattern, _str)

        _output = [_[1] if _[0] == '' else _[0] for _ in _output]
        output.extend(_output)
    return output


def process_record(_json):
    _pattern = '[\u4e00-\u9fa5,，]+'
    _output = re.findall(_pattern, number_to_chinese(_json))
    return ','.join(_output)


def main():
    # load extractor
    symptom_extractor = load_config(config.pattern_symptom)
    drug_extractor = load_config(config.pattern_drug)
    disease_extractor = load_disease_config(config.pattern_disease, config.pattern_disease_cname)

    # load data
    origin_data = load_origin_data()

    # process json
    origin_data['json'] = origin_data.apply(
        lambda row: delete_same([
            process_record(row['total_record_info'])
        ]), axis=1)

    # extract info
    origin_data['drug'] = origin_data.apply(
        lambda row: delete_same([
            extract_drug(row['diagnosis_treatment'], drug_extractor),
            extract_drug(row['medication_recommendations'], drug_extractor),
            extract_drug(row['json'], drug_extractor),
        ]), axis=1)

    origin_data['disease'] = origin_data.apply(
        lambda row: delete_same([
            # extract_disease(row['diagnosis_treatment'], disease_extractor),
            extract_disease(row['history_present_illness'], disease_extractor),
        ]), axis=1)

    origin_data['symptom'] = origin_data.apply(
        lambda row: delete_same([
            extract_symptom(row['cheif_complaint'], symptom_extractor),
            extract_symptom(row['history_present_illness'], symptom_extractor),
        ]), axis=1)

    origin_data['time'] = origin_data.apply(
        lambda row: delete_same([
            # extract_time(row['diagnosis_treatment']),
            extract_time(row['cheif_complaint']),
        ]), axis=1)

    # save
    origin_data.to_excel(config.output_data_path, encoding='utf-8')
    origin_data[['history_present_illness', 'disease']].to_csv(config.train_data_path, sep='\t', )
    # origin_data[['cheif_complaint',
    #              'diagnosis_treatment',
    #              'history_present_illness',
    #              'past_disease_history',
    #              'personal_history'
    #              ]].to_csv('', sep='\t', )
    # origin_data = origin_data[(origin_data['cheif_complaint'] != '')
    #                           & (origin_data['symptom'] != '')
    #                           & (origin_data['time'] != '')]
    # origin_data[['cheif_complaint', 'symptom', 'time']].to_csv(
    #     config.output_data_path, encoding='utf-8', sep='\t', index=None, header=None)

    # with open(config.output_data_path, mode='r', encoding='utf-8') as fo:
    #     with open('need_to_label.tsv', mode='w', encoding='utf-8') as fn:
    #         with open('no_need_to_label.tsv', mode='w', encoding='utf-8') as fnn:
    #             for line in fo.readlines():
    #                 content, target, label = line.strip().split('\t')
    #                 target_list = target.split(',')
    #                 if len(target_list) > 1:
    #                     for _target in target_list:
    #                         fn.writelines('{}\t{}\t{}\n'.format(number_to_chinese(content), _target, label))
    #                 else:
    #                     fnn.writelines('{}\t{}\t{}\n'.format(number_to_chinese(content), target, label))


def test():
    str_a = "尿液分析+沉渣定量 妇科常规检查Qd×1天 会阴冲洗(含扩阴器)1Qd×1天 阴道分泌物常规 细菌性阴道病检查 HPV核酸分型检测 妇科TCT 硝呋太尔片[0.2g*20片]0.4g口服Tid×6天 经带宁胶囊[0.3g*36粒]4粒口服Tid×5天 聚维酮碘溶液[20g:200ml]20g外用Qd×1天 甲硝唑阴道凝胶[5g/支]（尼美欣）5g阴道给药Qd×4天 随诊"

    symptom_extractor = load_config(config.pattern_symptom)
    drug_extractor = load_config(config.pattern_drug)
    disease_extractor = load_disease_config(_config_path=config.pattern_disease, _c_name_path=config.pattern_disease_cname)

    print(extract_symptom(_symptom=str_a, _symptom_extractor=symptom_extractor))
    print(extract_drug(str_a, drug_extractor))
    print(extract_disease(_disease=str_a, _disease_extractor=disease_extractor))

    @delete_same
    def f():
        return ['石膏', '甲钴胺', '石膏']

    # print(f())


def get_train_data():
    disease_extractor = load_config(config.pattern_disease)

    with open(config.data_path, mode='r', encoding='utf-8') as f1:
        with open(config.train_data_path, mode='w', encoding='utf-8') as fo:
            counter = 0
            for line in f1.readlines()[1:]:
                try:
                    content = line.strip().split('\t')[3]
                    for _content in re.split('[。，]', content):
                        if _content.strip() == '':
                            continue

                        if '无' in _content:
                            fo.writelines('{}\t{}\t{}\n'.format(str(counter), _content, ''))
                        else:
                            fo.writelines('{}\t{}\t{}\n'.format(str(counter), _content,
                                                                ','.join(extract_disease(_content, disease_extractor))))
                        counter += 1

                        if counter % 1000 == 0:
                            print(counter)
                except IndexError:
                    print(len(line.strip().split('\t')))


if __name__ == "__main__":
    # main()
    test()
    # get_train_data()
