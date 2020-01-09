# coding=utf-8
import pandas as pd
from tqdm import tqdm
from py2neo import Node, Graph, Relationship


class DataToNeo4j(object):
    """将excel中数据存入neo4j"""

    def __init__(self):
        link = Graph("http://localhost:7474", username="neo4j", password="123456")
        self.graph = link

        self.invoice_name = 'entityKey'
        self.invoice_value = 'entityValue'
        self.graph.delete_all()

    def create_node(self, node_list_key, node_list_value):
        """add node"""
        for name in tqdm(node_list_key):
            name_node = Node(self.invoice_name, name=name)
            try:
                self.graph.create(name_node)
                print('===add: {}'.format(name))
            except AttributeError:
                print(name)
        for name in tqdm(node_list_value):
            value_node = Node(self.invoice_value, name=name)
            try:
                self.graph.create(value_node)
                print('===add: {}'.format(name))
            except AttributeError:
                print(name)

    def create_relation(self, df_data):
        """add relation"""

        for index, row in tqdm(df_data.iterrows()):
            try:
                rel = Relationship(
                    self.graph.find_one(label=self.invoice_name, property_key='name', property_value=str(row['entity_a'])),
                    str(row['relation']),
                    self.graph.find_one(label=self.invoice_value, property_key='name', property_value=str(row['entity_b']))
                )
                self.graph.create(rel)
                print('===add relation: {}'.format(str(row['relation'])))
            except AttributeError as e:
                print(e, row)


if __name__ == '__main__':
    path = 'C:/Users/YP_TR/Desktop/nuanwa资料/图谱数据/CmeKG/CMeKG.xlsx'
    io = pd.io.excel.ExcelFile(path)
    kg_tuple = pd.read_excel(io, sheetname=u'三元组')
    kg_disease = pd.read_excel(io, sheetname=u'疾病')
    kg_drug = pd.read_excel(io, sheetname=u'药品')
    kg_symptom = pd.read_excel(io, sheetname=u'症状')
    kg_diag = pd.read_excel(io, sheetname=u'诊疗')
    kg_entity = pd.read_excel(io, sheetname=u'实体')
    kg_relation = pd.read_excel(io, sheetname=u'关系')

    kd_database = DataToNeo4j()

    node_list_a = list(set(kg_tuple['entity_a'].tolist()))
    node_list_b = list(set(kg_tuple['entity_b'].tolist()))
    node_list_a = [str(i) for i in node_list_a]
    node_list_b = [str(i) for i in node_list_b]
    kd_database.create_node(node_list_key=node_list_a, node_list_value=node_list_b)

    kd_database.create_relation(kg_tuple)
