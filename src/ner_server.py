# coding=utf-8

import json
from flask import Flask
from flask import request
from mylogparse import LogParse
from src.ner_main_client import get_disease
from src.ner_main_client import segment_content
from src.ner_main_client import segment_sentence

a = LogParse()
a.set_profile(path="./log", filename="test")

app = Flask(__name__)
app.logger.addHandler(a.handler)


@a.exception
@app.route('/ner_extractor/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        if "content" in request.json.keys():
            content_str = request.json["content"]
            sentence_list = segment_content(content_str)

            dise_out = []
            symp_out = []
            drug_out = []
            diag_out = []
            for _ in sentence_list:
                short_sentence_list = segment_sentence(_)
                for short_sentence in short_sentence_list:
                    if short_sentence != '':
                        print('原句：', short_sentence)
                        f = get_disease(short_sentence)
                        print('预抽取疾病：', f['dise'])
                        print('预抽取症状：', f['symp'])
                        dise_out.extend(f['dise'])
                        symp_out.extend(f['symp'])
                        drug_out.extend(f['drug'])
                        diag_out.extend(f['diag'])

            return json.dumps({'disease': dise_out,
                               'sympton': symp_out,
                               'drug': drug_out,
                               'operation': diag_out})


if __name__ == "__main__":
    app.run('0.0.0.0', port=8888)
