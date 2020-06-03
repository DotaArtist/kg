#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导入数据"""

__author__ = 'yp'

from py2neo import Graph

# 连接neo4j数据库
graph = Graph("http://127.0.0.1:7687", username="neo4j", password="123")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_bodypart.csv' AS line
CREATE (n:bodypart{id:line.id,name:line.name,node_type:line.node_type});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_check.csv' AS line
CREATE (n:check{id:line.id,name:line.name,node_type:line.node_type});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_department.csv' AS line
CREATE (n:department{id:line.id,name:line.name,node_type:line.node_type});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_disease.csv' AS line
CREATE (n:disease{id:line.id,name:line.name,node_type:line.node_type,icd:line.icd});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_diseasename.csv' AS line
merge(n:disease{id:line.id})
on create set n.id=line.id,n.name=line.name,n.icd=line.icd,n.node_type=line.node_type
on match set n.node_type=line.node_type;""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_drug.csv' AS line
CREATE (n:drug{id:line.id,name:line.name,node_type:line.node_type,code:line.code});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_material.csv' AS line
CREATE (n:material{id:line.id,name:line.name,node_type:line.node_type,code:line.code});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_symptom.csv' AS line
CREATE (n:symptom{id:line.id,name:line.name,node_type:line.node_type});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///node_treatment.csv' AS line
CREATE (n:treatment{id:line.id,name:line.name,node_type:line.node_type,code:line.code});""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_check_bodypart.csv' AS line
MATCH (b:bodypart) where b.id = line.to WITH b,line
MATCH (c:check) where c.id = line.from
MERGE (c)-[:check_bodypart]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_department_bodypart.csv' AS line
MATCH (b:bodypart) where b.id = line.to WITH b,line
MATCH (c:department) where c.id = line.from
MERGE (c)-[:department_bodypart]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_disease_bodypart.csv' AS line
MATCH (b:bodypart) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:disease_bodypart]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_disease_check.csv' AS line
MATCH (b:check) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:disease_check]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_disease_department.csv' AS line
MATCH (b:department) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:disease_department]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_disease_drug.csv' AS line
MATCH (b:drug) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:disease_drug]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_disease_material.csv' AS line
MATCH (b:material) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:disease_material]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_disease_symptom.csv' AS line
MATCH (b:symptom) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:disease_symptom]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_disease_treatment.csv' AS line
MATCH (b:treatment) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:disease_treatment]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_symptom_bodypart.csv' AS line
MATCH (b:bodypart) where b.id = line.to WITH b,line
MATCH (c:symptom) where c.id = line.from
MERGE (c)-[:symptom_bodypart]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_has_entry_word.csv' AS line
MATCH (b) where b.id = line.to WITH b,line
MATCH (c) where c.id = line.from
MERGE (c)-[:similar_from_word]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_has_entry_word_duima.csv' AS line
MATCH (b:disease) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:similar_from_duima]->(b)""")

graph.run("""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relation_is_a.csv' AS line
MATCH (b:disease) where b.id = line.to WITH b,line
MATCH (c:disease) where c.id = line.from
MERGE (c)-[:similar_is_a]->(b)""")
