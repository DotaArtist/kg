import pkuseg
import re
import flashtext
from itertools import chain
import pandas as pd
"""
通过flashtest做的疾病、药品、诊疗、材料的关键信息抽取

"""

class PatternExtractor(object):
    """
    功能：
    extract --> 抽取一串文本中 非标准名 映射的 标准名
    replace --> 将一串文本中的 非标准名 替换成 标准名
    """

    def __init__(self):
        self.ac = flashtext.KeywordProcessor()
        # 字典格式 --> "非标准名":["标准名1", "标准名2", ...]
        self.project = {}

    def add_pattern(self, filepath, col):
        """
        :param filepath: 必须是同一个的格式的txt文件，以\t分隔的两列，第一列是标准名，第二列是非标准名,不要header.
        :return: None
        """
        file_list = pd.read_csv(filepath)[col]
        for file in file_list:
            self.ac.add_keyword(file)

        # for k, v in self.project.items():
        #     self.ac.add_keyword(k, "#*#".join(v))  #如果有一个非标准名对应多个标准名，多个标准名用"#*#"连接

    def extract(self, text):
        """
        抽取一串文本中 非标准名 映射的 标准名
        :param text: 文本
        :return: list: 提取到的标准词的列表
        """
        # text = " ".join(list(text))
        res = self.ac.extract_keywords(text)
        # results = [it.split("#*#") for it in res]
        # results = list(set(chain.from_iterable(results)))
        return res

    def replace(self, text):
        """
        将一串文本中的 非标准名 替换成 标准名
        :param text: 文本
        :return: 替换后的字符串
        """
        text = " ".join(list(text))
        res = self.ac.replace_keywords(text)
        result = "".join(res.split(" "))
        return result


def seg_words(sentence):
    seg = pkuseg.pkuseg(model_name='medicine')  # 程序会自动下载所对应的细领域模型
    # text = seg.cut('我爱北京天安门')              # 进行分词
    # print(text)

    sentence = ['患者入院前4天（05月05日）劳累后出现发热', '自服感冒药后热退',
                '今日（05月09日02：00）无明显诱因再次出现发热',
                '头痛发作前无黑曚、闪光等视觉先兆', '病前无腹泻、咳嗽、咳痰等不适']

    result = []
    for sen in sentence:
        res = seg.cut(sen)
        result.append(res)
    print(result)


def use_extractor(origin):
    pp = PatternExtractor()
    pp.add_pattern(pattern_path, "symptom")
    drug_pp = PatternExtractor()
    drug_pp.add_pattern(pattern_drug, "drug")
    check_pp = PatternExtractor()
    check_pp.add_pattern(pattern_check, "check")
    disease_pp = PatternExtractor()
    disease_pp.add_pattern(pattern_disease, "disease_name")
    symptom_lists = []
    drug_lists = []
    check_lists = []
    disease_lists = []
    if origin == '[]':
        return symptom_lists, drug_lists, check_lists, disease_lists
    for per_origin in origin:
        symptom = pp.extract(per_origin)
        drug = drug_pp.extract(per_origin)
        check = check_pp.extract(per_origin)
        disease = disease_pp.extract(per_origin)
        symptom_lists.append(symptom)
        drug_lists.append(drug)
        check_lists.append(check)
        disease_lists.append(disease)
    return symptom_lists, drug_lists, check_lists, disease_lists


def sen_sele(pattern, seg_sen):
    flag = re.search(pattern, seg_sen)
    res_list = list()
    if flag:
        res_list = extract_keywd(seg_sen, flag, pattern)
    return res_list

def extract_keywd(seg_sen ,flag, pattern):
    pos_loc = list(flag.regs[0])
    cur_sen = seg_sen[pos_loc[0]: pos_loc[1]]
    # cur_sen = re.sub(pattern.replace('.*','') , '', cur_sen)
    return cur_sen

if __name__ == '__main__':
    data_path = "./result_data/history_present.csv"
    pattern_path = "./pattern/all_data/symptomAll_combine.csv"
    pattern_drug = "./pattern/all_data/drug_combine.csv"
    pattern_check = "./pattern/all_data/check_combine.csv"
    pattern_disease = "./pattern/all_data/disease_name_combine.csv"


    #
    data = pd.read_csv(data_path)
    data = data[~data['origin'].isin(["-1"])]
    origin = data['origin']
    time = data['time']
    pattern_time = r"(20\d{2}([\.\-/|年月\s]{1,3}\d{1,2}){2}日?(\s?\d{2}:\d{2}(:\d{2})?)?)|" \
                   r"(\d{1,2}\s?(分钟|小时|天)前)|.*月*(前|后)|.*年*月|.*年|.*日|.*时"
    for data in time:
        res_list = list()
        data_list = data.split(",")
        for i in data_list:
            cur_sen = sen_sele(pattern_time,i)
            symptom, drug, check, disease = use_extractor(i)

    symptom_lists, drug_lists, check_lists, disease_lists = use_extractor(origin)
    data['symptom'] = symptom_lists
    data['drug'] = drug_lists
    data['check'] = check_lists
    data['disease'] = disease_lists
    data.to_csv("./result_data/process_symptom2.csv")
