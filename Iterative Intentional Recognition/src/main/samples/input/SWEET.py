"""
   脚本描述：读取SWEET本体概念提取结果
"""

from src.main.util.FileUtil import load_json
import os.path

__dir__ = os.path.dirname(os.path.abspath(__file__))
file_path_prefix = os.path.abspath(os.path.join(__dir__, "../../../../resources/ontologies/sweet"))

# 只考虑上下位关系
# Ontologies = load_json(os.path.join(file_path_prefix, "all_hyponyms_dimension_divided.json"))
# Ancestor = load_json(os.path.join(file_path_prefix, "all_ancestors_dimension_divided.json"))
# Neighborhood = load_json(os.path.join(file_path_prefix, "neighbors_dimension_divided.json"))
# direct_Ancestor = load_json(os.path.join(file_path_prefix, "direct_ancestors_dimension_divided.json"))

# OntologyRoot = load_json(os.path.join(file_path_prefix, "dimensions_root_concept.json"))

Ontologies_path = os.path.join(file_path_prefix, "all_hyponyms.json")
Ancestors_path = os.path.join(file_path_prefix, "all_ancestors.json")
Neighborhood_path = os.path.join(file_path_prefix, "neighbors.json")
Direct_Hyponyms_path = os.path.join(file_path_prefix, "direct_hyponyms.json")
Direct_Ancestors_path = os.path.join(file_path_prefix, "direct_ancestors.json")
Peer_path = os.path.join(file_path_prefix, "peers.json")
Information_Content_path = os.path.join(file_path_prefix, "concept_information_content_yuan2013.json")

Ontologies = load_json(Ontologies_path)  # 包含SWEET本体内某概念C的所有下位概念
Ancestors = load_json(Ancestors_path)  # 包含SWEET本体内某概念C的所有上位概念
Neighborhood = load_json(Neighborhood_path)  # 包含SWEET本体内某概念C的上一级和下一级概念
Direct_Hyponyms = load_json(Direct_Hyponyms_path)  # 包含SWEET本体内某概念C的下一级概念
Direct_Ancestors = load_json(Direct_Ancestors_path)  # 包含SWEET本体内某概念C的上一级概念
Peer = load_json(Peer_path)  # 包含SWEET本体内某概念C的同级概念（不包括它本身）
Information_Content = load_json(Information_Content_path)  # 包含SWEET本体内某概念C的信息量

# 为SWEET本体添加顶级Thing概念
Ontology_Root = "Thing"
top_concepts = list(filter(lambda x: len(Ancestors[x]) == 0, list(Ancestors.keys())))

for tmp_top_concept in top_concepts:
    Direct_Ancestors[tmp_top_concept] = [Ontology_Root]
    Neighborhood[tmp_top_concept].append(Ontology_Root)
for tmp_concept in Ancestors:
    Ancestors[tmp_concept].append(Ontology_Root)

all_concepts = list(Ontologies.keys())
Ontologies[Ontology_Root] = all_concepts
Ancestors[Ontology_Root] = []
Neighborhood[Ontology_Root] = top_concepts
Direct_Hyponyms[Ontology_Root] = []
Direct_Ancestors[Ontology_Root] = []
Information_Content[Ontology_Root] = 0.0
for tmp_top_concept in top_concepts:
    Direct_Hyponyms[Ontology_Root].append(tmp_top_concept)

# TODO: 提取的SWEET本体中（Ontologies）缺少部分概念
# print('None' in Direct_Ancestors)
# print('None' in Ontologies)
# print(Ontologies['http://sweetontology.net/matrElement/TransitionMetal'])
# print(Ontologies['http://sweetontology.net/matrRocklgneous/VolcanicRock'])
# print(Ontologies['http://sweetontology.net/matrCompound/IronOxide'])
# print(len(Ontologies))

# 通过Direct_Ancestor和Direct_Hyponyms获取同级概念（不包括它本身）
# from src.main.util.FileUtil import save_as_json
# Peer = {}
# for concept in Direct_Ancestors:
#     direct_ancestors = Direct_Ancestors[concept]
#     # 处理根节点的情况
#     if len(direct_ancestors) == 0:
#         Peer[concept] = []
#     else:
#         peer_concepts = []
#         for tmp_direct_ancestor in direct_ancestors:
#             peer_concepts = list(set(peer_concepts).union(set(Direct_Hyponyms[tmp_direct_ancestor])))
#         peer_concepts.remove(concept)
#         # 同级概念（不包括它本身）
#         Peer[concept] = peer_concepts
# print(Peer["Thing"])
# save_as_json(Peer, "../../../../resources/ontologies/sweet/peers.json")
