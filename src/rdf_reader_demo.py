# coding=utf-8

from rdflib import Graph
from rdflib import URIRef

g = Graph()
g.parse("../data/rdf/dsc.nlp-bigdatalab.org.ttl", format="ttl")
g.bind('ex', URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#'))

qres = g.query("""
SELECT ?y
WHERE
  { ?y ex:type "疾病" .
  FILTER regex(?y, "r", "肺")
  }
""")

for row in qres:
    print(row)
