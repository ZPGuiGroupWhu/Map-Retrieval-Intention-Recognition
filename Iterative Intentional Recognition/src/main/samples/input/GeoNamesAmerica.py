"""
   脚本描述：读取GeoNames本体概念的提取结果（美洲地区的上下位概念）
"""
from src.main.util.FileUtil import load_json
import os.path

__dir__ = os.path.dirname(os.path.abspath(__file__))
file_path_prefix = os.path.abspath(os.path.join(__dir__, "../../../../resources/ontologies/geonames_america"))

# 只考虑空间上下位从属关系
Ontologies_path = os.path.join(file_path_prefix, "america_all_hyponyms.json")
Ancestors_path = os.path.join(file_path_prefix, "america_all_ancestors.json")
Direct_Hyponyms_path = os.path.join(file_path_prefix, "america_direct_hyponyms.json")
Direct_Ancestors_path = os.path.join(file_path_prefix, "america_direct_ancestors.json")
Neighborhood_path = os.path.join(file_path_prefix, "america_neighbors.json")
Peer_path = os.path.join(file_path_prefix, "america_peers.json")
Information_Content_path = os.path.join(file_path_prefix, "america_information_content.json")

Ontologies = load_json(Ontologies_path)              # 空间实体区域内的所有级别的所有空间实体，最小级别为一级行政区（省级）
Ancestors = load_json(Ancestors_path)                # 包含某个空间实体E的所有上级空间实体，最上级为美洲（America）
Direct_Hyponyms = load_json(Direct_Hyponyms_path)    # 包含某个空间实体E的下一级空间实体
Direct_Ancestors = load_json(Direct_Ancestors_path)  # 包含某个空间实体E的上一级空间实体，最上级为美洲（America）
Neighborhood = load_json(Neighborhood_path)          # 包含某个空间实体E的上一级实体和E包含的下一级空间实体
Peer = load_json(Peer_path)                          # 包含某个空间实体E的同级概念（不包括它本身）
Information_Content = load_json(Information_Content_path)     # 包含某个空间实体E的信息量
Ontology_Root = "America"


# print(Ontologies["Cuba"])
# print(len(Ontologies["United States"]))

# 通过Neighborhood和Direct_Ancestors关系计算直接下位概念Direct_Hyponyms,并保存为json文件
# from src.main.util.FileUtil import save_as_json
#
# Direct_Hyponyms = {}
# for concept in Direct_Ancestors:
#     Direct_Hyponyms[concept] = list(set(Neighborhood[concept]) - set(Direct_Ancestors[concept]))
# print(Direct_Hyponyms["North America"])
# save_as_json(Direct_Hyponyms, "../../../../resources/ontologies/geonames_america/america_direct_hyponyms.json")

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
# print(Peer["Canada"])
# save_as_json(Peer, "../../../../resources/ontologies/geonames_america/america_peers.json")
