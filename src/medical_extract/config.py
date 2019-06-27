# coding=utf-8

data_path = '../../data/medical_record/origin_data.tsv'
output_data_path = '../../data/medical_record/final_extract_output.xlsx'
train_data_path = '../../data/medical_record/train.txt'
ner_data_path = '../../data/medical_record/ner_label.txt'

ner_origin_json_data = '../../data/medical_record/shuf_ner_label_100.txt'
ner_origin_json = '../../data/medical_record/nuan_wa_ner.json'

pattern_symptom = "../../data/medical_record/config/symptom_combine_all6.csv"
pattern_drug = "../../data/medical_record/config/drug_combine.csv"
pattern_check = "../../data/medical_record/config/check_combine.csv"
# pattern_disease = "../../data/medical_record/config/disease_combine_all.csv"
pattern_disease = "../../data/medical_record/config/idc_disease.csv"
pattern_disease_cname = "../../data/medical_record/config/disease_cname.csv"
pattern_operation = "../../data/medical_record/config/operation.csv"

health_insurance_tag = "../../data/medical_record/config/health_insurance_tag_deliver.csv"
tag_disease_detail = "../../data/medical_record/config/众安保险-诊断主编码-核赔(诊断)标签条目.csv"

chinese_number_dict = {
    '一': 1,
    '二': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9,
    '两': 2,
    '零': 0
}

number_chinese_dict = {
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九',
    '0': '零'
}
