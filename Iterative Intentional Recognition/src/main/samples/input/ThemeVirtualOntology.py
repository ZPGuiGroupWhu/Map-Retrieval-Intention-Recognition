"""
   脚本描述：构造虚拟的主题本体
"""
# the virtual ontology for dimension theme which has enumerate value
from src.main.util import OntologyUtil
import copy

ThemeValues = ['Agriculture', 'Biodiversity', 'Climate', 'Disaster', 'Ecosystem', 'Energy', 'Geology', 'Health', 'Water', 'Weather']
Ontologies = {}
Ancestors = {}
Direct_Hyponyms = {}
Direct_Ancestors = {}
Neighborhood = {}
Peer = {}
Information_Content = {}
Ontology_Root = "ThemeRoot"

root_hyponyms = []
root_neighborhood = []
root_ancestors = []
root_direct_ancestor = []

for tmp_concept in ThemeValues:
    Ontologies[tmp_concept] = []
    Ancestors[tmp_concept] = [Ontology_Root]
    Direct_Ancestors[tmp_concept] = [Ontology_Root]
    Neighborhood[tmp_concept] = [Ontology_Root]
    Peer[tmp_concept] = copy.deepcopy(ThemeValues)
    Peer[tmp_concept].remove(tmp_concept)
    root_hyponyms.append(tmp_concept)
    root_neighborhood.append(tmp_concept)
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


