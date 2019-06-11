import pandas as pd
import numpy as np
import re

def cal_null_por():
    """
    计算空值率
    :return:
    """
    data = "mdp_medical_record.csv"
    columns = "columns.csv"
    data = pd.read_csv(data, header = None)
    columns = pd.read_csv(columns).columns
    
    null_count = data.apply(lambda col: col.isnull().sum())
    null_por = (null_count / len(data)) * 100
    
    print(null_por)
    df = pd.DataFrame([null_por.values], columns = columns)
    df.to_csv("statistic.csv")



def extract_keywd(seg_sen ,flag, pattern):
    pos_loc = list(flag.regs[0])
    cur_sen = seg_sen[pos_loc[0]: pos_loc[1]]
    # cur_sen = re.sub(pattern.replace('.*','') , '', cur_sen)
    return cur_sen


def sen_sele(pattern, seg_sen, res_list):
    flag = re.search(pattern, seg_sen)
    if flag:
        res_list.append(extract_keywd(seg_sen, flag, pattern))
    return res_list



def ext_per_his(text):
    """
    个人史抽取的方法，对于生长地，既往史等进行提取
    :param text:
    :return:
    """
    born_lists, grow_lists, deny_lists, past_lists = list(), list(), list(), list()
    for per_text in text:
        born_list, grow_list, deny_list, past_list = list(), list(), list(), list()
        if per_text != -1:
            tmp_list = re.split(r'[,|，|\d|"或"|；|。|;]', per_text)
            print("seg")
        # except:
        #     print("error extract: ", per_name)
        else:
            born_lists.append([])
            grow_lists.append([])
            deny_lists.append([])
            past_lists.append([])
            continue
        for seg_sen in tmp_list:
            seg_sen = seg_sen.replace(' ', '')
            pattern_born = r"出生于.*|生于.*"
            pattern_grow = r"生长于.*|生、长于.*"
            pattern_deny = r"否认.*|无.*"
            pattern_past = r"既往.*"
            born_list = sen_sele(pattern_born, seg_sen, born_list)
            grow_list = sen_sele(pattern_grow, seg_sen, grow_list)
            deny_list = sen_sele(pattern_deny, seg_sen, deny_list)
            past_list = sen_sele(pattern_past, seg_sen, past_list)
        born_lists.append(born_list)
        grow_lists.append(grow_list)
        deny_lists.append(deny_list)
        past_lists.append(past_list)
        print("finish")
    df = pd.DataFrame({"origin":text,"born":born_lists, "grow":grow_lists,
                       "deny": deny_lists, "past": past_lists})
    df.to_csv("personal_history.csv",encoding= 'utf-8')



def ext_his_pre_ill(text):
    """
    针对于现病史抽取的函数，主要利用了时间正则表达式
    :param text:
    :return:
    """
    time_lists, deny_lists = list(), list()
    for per_text in text:
        time_list, deny_list = list(), list()
        if per_text != -1:
            tmp_list = re.split(r'[,|，|"或"|；|。|;]', per_text)
            print("seg")
        # except:
        #     print("error extract: ", per_name)
        else:
            time_lists.append([])
            deny_lists.append([])
            continue
        for seg_sen in tmp_list:
            seg_sen = seg_sen.replace(' ', '')
            pattern_deny = r"否认.*|无.*"
            pattern_time = r"(20\d{2}([\.\-/|年月\s]{1,3}\d{1,2}){2}日?(\s?\d{2}:\d{2}(:\d{2})?)?)|" \
                           r"(\d{1,2}\s?(分钟|小时|天)前).*|.*月*(前|后).*|.*年*月.*|.*年.*|.*日.*|.*时.*"
            #.*月*前|.*年*前|.*月|.*年|.*日|.*时|.*分|.*秒|\d{4}-\d{2}-\d{2}

            deny_list = sen_sele(pattern_deny, seg_sen, deny_list)
            time_list = sen_sele(pattern_time, seg_sen, time_list)
        time_lists.append(time_list)
        deny_lists.append(deny_list)
    df = pd.DataFrame({"origin": text,
                       "deny": deny_lists, "time": time_lists})
    df.to_csv("history_present.csv", encoding='utf-8')

    print("finish")




def extract_info(data_path):
    """
    用于抽取数据的函数
    cheif_complaint: 主诉
    history_present_illness: 现病史
    total_record_info :
    personal_history :
    past_disease_history :
    """
    data = pd.read_csv(data_path)
    data = data.fillna(-1)
    id = data.iloc[:, 0]
    chief_com = data.iloc[:,15]
    his_pre_illness = data.iloc[:,19] #
    total_record_info = data.iloc[:,33]
    personal_history = data.iloc[:,21]
    past_disease_history = data.iloc[:,20]
    # ext_per_his(personal_history) #提取个人史
    ext_his_pre_ill(his_pre_illness) #提取现病史

    pass

if __name__ == '__main__':
    data_path = "mdp_medical_record.csv"
    # cal_null_por()
    extract_info(data_path)
