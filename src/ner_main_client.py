# coding=utf-8
import re
import requests
import pandas as pd


def get_disease(sentence_1):
    """返回疾病列表"""
    data = dict()
    data["sentence"] = sentence_1.replace(' ', '')
    data["Content-Type"] = "application/json"
    r = requests.post("http://192.168.236.14:1086", json=data)

    dise_out = r.json()['dise'][0]
    symp_out = r.json()['symp'][0]
    drug_out = r.json()['drug'][0]

    dise_output = []
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    for i in dise_out:
        res = re.findall(re_words, i)
        dise_output.extend([_ for _ in res if len(_) > 1])

    symp_output = []
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    for i in symp_out:
        res = re.findall(re_words, i)
        symp_output.extend([_ for _ in res if len(_) > 1])

    drug_output = []
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    for i in drug_out:
        res = re.findall(re_words, i)
        drug_output.extend([_ for _ in res if len(_) > 1])

    out_dict = dict()
    out_dict['dise'] = dise_output
    out_dict['symp'] = symp_output
    out_dict['drug'] = drug_output

    return out_dict


def segment_content(content):
    _content_list = re.split("。", content)

    if len(_content_list) == 1:
        pass
    else:
        _content_list = [i + '。' for i in _content_list if i != ""]
    return _content_list


def segment_sentence(sentence):
    _sentence_list = sentence.split('，')

    def merge_sentence(_sentence_list):
        if len(_sentence_list) == 1:
            return _sentence_list
        else:
            for _idx, _sentence in enumerate(_sentence_list):
                if len(_sentence) < 15:
                    if _idx == 0:
                        _sentence_list[1] = _sentence + '，' + _sentence_list[1]
                        _sentence_list = _sentence_list[1:]
                        return merge_sentence(_sentence_list)
                    else:
                        _sentence_list[_idx-1] = _sentence_list[_idx-1] + '，' + _sentence
                        _sentence_list = _sentence_list[:_idx] + _sentence_list[_idx+1:]
                        return merge_sentence(_sentence_list)
            return _sentence_list

    _sentence_list = merge_sentence(_sentence_list)
    return _sentence_list


def main():
    """电子病历抽取"""
    path = '../data/medical_record/电子病历数据_7493.txt'
    with open('final_file_7493.txt', mode='w', encoding='utf-8') as f0:
        with open(path, mode='r', encoding='utf-8') as f1:

            for line in f1.readlines():
                line = line.strip().split('\t')

                disease_list = []
                symp_list = []
                drug_list = []

                short_sentence_list = []

                for content in line[1:]:
                    if content != 'null':
                        content_list = segment_content(content)

                        for sentence in content_list:
                            sentence_list = segment_sentence(sentence)

                            short_sentence_list.extend(sentence_list)
                # print(short_sentence_list)

                for short_sentence in short_sentence_list:
                    if short_sentence.replace(' ', '') != '':
                        model_out = get_disease(short_sentence)
                        disease = model_out['dise']
                        drug = model_out['drug']
                        symp = model_out['symp']

                        disease_list.extend(disease)
                        symp_list.extend(symp)
                        drug_list.extend(drug)

                print(list(set(disease_list)), list(set(symp_list)), list(set(drug_list)))
                f0.writelines('{}\t{}\t{}\t{}\n'.format(line[0],
                                                        str(list(set(disease_list))),
                                                        str(list(set(symp_list))),
                                                        str(list(set(drug_list))),
                                                        ))


def rule_filter(word, word_list):
    if len(word) > 1:
        return True
    else:
        if word not in word_list:
            return False
        else:
            return True


def merge_medical():
    """多病历合并"""
    _c_name = pd.read_csv('../data/medical_record/config/disease_cname.csv', sep=',', index_col=0).fillna('')
    _disease_cname = _c_name['疾病别名'].values.tolist()

    _config = pd.read_csv('../data/medical_record/config/idc_disease.csv', sep=',', index_col=0)
    _keys = _config.iloc[:, 0].values.tolist()
    _keys.extend(_disease_cname)

    disease_total_list = list(set(_keys))

    path = './抽取结果.txt'
    data = pd.read_csv(path, sep='\t', encoding='utf-8', header=None, index_col=None, engine='python')
    data.columns = ['medical_num', 'disease', 'sym', 'drug']

    data_new = pd.DataFrame()

    name_list = []
    disease_list = []
    symp_list = []
    drug_list = []

    counter = 0

    for name, group in data.groupby('medical_num'):
        counter += 1
        print(counter)
        name_list.append(name)
        tmp_dise, tmp_symp, tmp_drug = [], [], []

        [tmp_dise.extend(i) for i in group.disease.apply(lambda x: eval(x)).tolist()]
        [tmp_symp.extend(i) for i in group.sym.apply(lambda x: eval(x)).tolist()]
        [tmp_drug.extend(i) for i in group.drug.apply(lambda x: eval(x)).tolist()]

        tmp_dise = list(set(tmp_dise))
        tmp_dise = [i for i in tmp_dise if rule_filter(i, disease_total_list)]

        disease_list.append(str(list(set(tmp_dise))).replace('\'\', ', '').replace('\'\'', ''))
        symp_list.append(str(list(set(tmp_symp))).replace('\'\', ', '').replace('\'\'', ''))
        drug_list.append(str(list(set(tmp_drug))).replace('\'\', ', '').replace('\'\'', ''))

    data_new['medical_num'] = pd.Series([i for i in name_list if i != ''])

    data_new['disease'] = pd.Series([i for i in disease_list if i != ''])
    data_new['sym'] = pd.Series([i for i in symp_list if i != ''])
    data_new['drug'] = pd.Series([i for i in drug_list if i != ''])

    data_new.to_csv('./抽取结果合并.txt', sep='\t', index=None, header=None)


if __name__ == '__main__':
    # str_a = '患者于入院1个月前于劳累时发作胸痛，疼痛位于胸骨后，手掌大小面积，为压迫样疼痛，无出汗，疼痛无放散，疼痛发作约3－5分钟可自行好转，未在意。上述症状反复发作，多于活动时发生，发作时含服硝酸甘油后1－2分钟可好转，未住院系统治疗。此次患者于入院1周前无明显诱因再发胸痛，疼痛部位性质同前，疼痛较剧烈，伴出汗，向后背部放散，发作约10分钟可逐渐好转，上述症状反复发作，今为求系统诊治就诊于我院门诊，心电图示心肌缺血，门诊以“冠心病，心绞痛”为诊断收入院治疗。病来无发热，无咳嗽，无咳痰，无咯血，无腹痛腹泻，无恶心呕吐，无头痛，无晕厥抽搐，体重无明显减轻，饮食睡眠尚可，二便正常。'
    # b = segment_content(str_a)
    # out = []
    # out_1 = []
    # for _ in b:
    #     c = segment_sentence(_)
    #     for d in c:
    #         if d != '':
    #             print('原句：', d)
    #             f = get_disease(d)
    #             print('预抽取疾病：', f['dise'])
    #             print('预抽取症状：', f['symp'])
    #             out.extend(f['dise'])
    #             out_1.extend(f['symp'])
    # print(list(set(out)))
    # print(list(set(out_1)))

    # a = get_disease('予查MRI示皮下囊肿，患者口述，具体报告未见，未予药物治疗，一周后消肿如常。')
    # print(a)
    # print(segment_sentence('反复胸骨后隐痛不适伴中上腹不适2周，无反酸，无嗳气，有烧心，大便正常'))
    # main()

    merge_medical()
