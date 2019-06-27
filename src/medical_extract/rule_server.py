# coding=utf-8

import json
from flask import Flask
from flask import request
from src.mylogparse import LogParse
from src.medical_extract.extract_medical_record import *

a = LogParse()
a.set_profile(path="./log", filename="test")

app = Flask(__name__)
app.logger.addHandler(a.handler)

symptom_extractor = load_config(config.pattern_symptom)
drug_extractor = load_config(config.pattern_drug)
# operation_extractor = load_config(config.pattern_operation)
disease_extractor = load_disease_config(_config_path=config.pattern_disease, _c_name_path=config.pattern_disease_cname)


@a.exception
@app.route('/rule_ner/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        if "content" in request.json.keys():
            content_str = request.json["content"]

            symp_out = extract_symptom(_symptom=content_str, _symptom_extractor=symptom_extractor)
            drug_out = extract_drug(_drug=content_str, _drug_extractor=drug_extractor)
            dise_out = extract_disease(_disease=content_str, _disease_extractor=disease_extractor)
            # diag_out = extract_operation(_operation=content_str, _operation_extractor=operation_extractor)
            diag_out = []

            return json.dumps({'disease': list(set(dise_out)),
                               'sympton': list(set(symp_out)),
                               'drug': list(set(drug_out)),
                               'operation': list(set(diag_out))})


if __name__ == "__main__":
    app.run('0.0.0.0', port=8800)
