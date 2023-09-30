"""
   脚本描述：构造虚拟的制图方法本体
"""
# the virtual ontology for dimension map method which has enumerate value
import copy

from src.main.util import OntologyUtil

MapMethodValues = ['No Method', 'Area Method', 'Quality Base Method', 'Point Symbol Method', 'Line Symbol Method', 'Choloplethic Method', 'Others']

Ontologies = {}
Ancestors = {}
Direct_Hyponyms = {}
Direct_Ancestors = {}
Neighborhood = {}
Peer = {}
Information_Content = {}
Ontology_Root = "MapMethodRoot"

root_hyponyms = []
root_neighborhood = []
root_ancestors = []
root_direct_ancestor = []

for tmp_value in MapMethodValues:
    Ontologies[tmp_value] = []
    Ancestors[tmp_value] = [Ontology_Root]
    Direct_Ancestors[tmp_value] = [Ontology_Root]
    Neighborhood[tmp_value] = [Ontology_Root]
    Peer[tmp_value] = copy.deepcopy(MapMethodValues)
    Peer[tmp_value].remove(tmp_value)
    root_hyponyms.append(tmp_value)
    root_neighborhood.append(tmp_value)
Ontologies[Ontology_Root] = root_hyponyms
Ancestors[Ontology_Root] = root_ancestors
Direct_Ancestors[Ontology_Root] = root_direct_ancestor
Neighborhood[Ontology_Root] = root_neighborhood
Peer[Ontology_Root] = []
Direct_Hyponyms = Ontologies              # 深度为2的本体，所有下位概念和直接下位概念完全相同

Ontology_Values = list(Ontologies.keys())

# 计算每个概念的信息量
max_nodes = len(Ancestors)
max_leaves_num = len(OntologyUtil.get_all_leaves_set(Ontologies))
for tmp_concept in Ontologies.keys():
    tmp_concept_depth = OntologyUtil.get_concept_max_depth(tmp_concept, Direct_Ancestors, Ontology_Root)
    tmp_max_depth = OntologyUtil.get_max_depth(Direct_Ancestors, Ontologies, Ontology_Root)
    tmp_hypernyms_num = len(Ancestors[tmp_concept])
    tmp_concept_leaves_num = OntologyUtil.get_concept_leaves_num(tmp_concept, Ontologies)
    tmp_concept_information_content = OntologyUtil.get_information_content(tmp_concept_depth, tmp_max_depth,
                                                                           tmp_hypernyms_num, max_nodes,
                                                                           tmp_concept_leaves_num, max_leaves_num)
    Information_Content[tmp_concept] = tmp_concept_information_content

# print(Information_Content)
# print(Ontologies)
# print(Direct_Hyponyms)



