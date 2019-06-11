# coding=utf-8
"""得到病历分局拆解"""

import src.medical_extract.config as config


with open(config.data_path, mode='r', encoding='utf-8') as fo:
    with open(config.ner_data_path, mode='w', encoding='utf-8') as fn:
        for line in fo.readlines()[1:]:
            content = line.strip('\n').split('\t')
            content = content[1:6]
            for _content in content:
                _content_list = _content.split('。')
                if len(_content_list) == 1:
                    fn.writelines('{}\n'.format(_content_list[0]))
                for sentence in _content.split('。'):
                    if sentence != "":
                        fn.writelines('{}。\n'.format(_content_list[0]))
