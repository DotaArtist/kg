# coding=utf-8

from py2neo import Graph
import pandas as pd


test_graph = Graph(
    "http://10.253.113.198:7474",
    username="neo4j",
    password="123456"
)

out = test_graph.run("""match (d:diseaseName) -[r:disease_drug]-> (s:drug)
where s.name =~ ".*氨氯地平.*" and d.name =~ ".*高血压.*"
return d.icd,d.name,s.name;""").data()
print(pd.DataFrame(out))
