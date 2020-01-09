# coding=utf-8

import json
import requests
from flask import Flask
from flask import request
from mylogparse import LogParse

a = LogParse()
a.set_profile(path="./log", filename="log")

app = Flask(__name__)
app.logger.addHandler(a.handler)


@a.exception
@app.route('/medical_record/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        if "content" in request.json.keys():
            content_str = request.json["content"]
            data = dict()
            data["data"] = {"content": content_str}

            a = requests.post("http://192.168.15.71:8800/rule_ner/", json=data['data'])
            b = requests.post("http://192.168.15.71:8888/ner/", json=data['data']).json()

            dise_out = a['disease']
            symp_out = b['sympton']
            drug_out = a['drug']
            b['operation'].extend(a['operation'])
            diag_out = b['operation']

            return json.dumps({'disease': list(set(dise_out)),
                               'sympton': list(set(symp_out)),
                               'drug': list(set(drug_out)),
                               'operation': list(set(diag_out))})


if __name__ == "__main__":
    app.run('0.0.0.0', port=1000)
