# coding=utf-8

from rdflib import Graph
from rdflib import URIRef

g = Graph()
g.parse("../data/rdf/dsc.nlp-bigdatalab.org.ttl", format="ttl")
# http://openkg.cn/dataset/symptom-in-chinese

g.bind('owl', URIRef('http://www.w3.org/2002/07/owl#'))
g.bind('rdf', URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#'))
g.bind('rdfs', URIRef('http://www.w3.org/2000/01/rdf-schema#'))
g.bind('skos', URIRef('http://www.w3.org/2004/02/skos/core#'))
g.bind('xsd', URIRef('http://www.w3.org/2001/XMLSchema#'))

# qres = g.query("""SELECT ?x WHERE { ?x rdfs:subClassOf* owl:症状. }""")
qres = g.query("""SELECT ?Class  (COUNT(?friend) AS ?count){
    ?x rdfs:Class rdf:type ? type.
}""")

for row in qres:
    print(row)
