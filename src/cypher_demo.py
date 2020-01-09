# coding=utf-8

from py2neo import Graph
import pandas as pd


class Neo4jObj(object):
    def __init__(self):
        self.host = "http://10.253.113.198:7474",
        self.username = "neo4j",
        self.password = "123456",
        self.graph = None

    def start(self):
        self.graph = Graph(
            "http://10.253.113.198:7474",
            username="neo4j",
            password="123456"
        )

    def run(self, _query):
        return self.graph.run(_query)

    def get_drug(self, _drug):
        _data = self.run("""match (d:diseaseName) -[r:disease_drug]-> (s:drug)
        where s.name =~ ".*{}.*"
        return d.icd,d.name,s.name;""".format(_drug)).data()
        return pd.DataFrame(_data)

    def get_disease(self, _disease):
        _data = self.run("""match (d:diseaseName) -[r:disease_drug]-> (s:drug)
        where d.name =~ ".*{}.*"
        return d.icd,d.name,s.name;""".format(_disease)).data()
        return pd.DataFrame(_data)


if __name__ == '__main__':
    db = Neo4jObj()
    db.start()

    data_total = pd.DataFrame(
        db.run('match (d:diseaseName) -[r:disease_symptom]-> (b:symptom) return d.name,b.name').data()
    )
    data_total.to_csv('./disease_symptom_relation.csv', sep='\t')
    # import pandas as pd
    # disease_list = pd.read_excel("C:/Users/YP_TR/Desktop/nuanwa/二度关系/disease_split.xlsx")['disease'].values.tolist()
    # drug_list = pd.read_excel("C:/Users/YP_TR/Desktop/nuanwa/二度关系/drug.xlsx")['药品'].tolist()
    #
    # data_total = pd.DataFrame()
    #
    # for _disease in disease_list:
    #     _data = db.get_disease(_disease=_disease)
    #
    #     if len(_data) > 0:
    #         data_total = pd.concat([data_total, _data])
    #     else:
    #         print(_disease)
    #
    # for _drug in drug_list:
    #     _data = db.get_drug(_drug=_drug)
    #
    #     if len(_data) > 0:
    #         data_total = pd.concat([data_total, _data])
    #     else:
    #         print(_drug)
    #
    # data_total.to_csv('./disease_drug_relation.tsv', sep='\t')
