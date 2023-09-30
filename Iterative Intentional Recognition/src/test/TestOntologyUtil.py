"""
   脚本描述：测试查询某个概念的所有上位、所有下位、直接上位、直接下位概念、同级概念
"""
from src.main.samples.generation import ConceptIdTransform
from src.main.samples.input import SWEET, GeoNamesAmerica, ThemeVirtualOntology, MapMethodVirtualOntology
from src.main.util.FileUtil import save_as_json


# 功能：查询某个概念的所有上位、所有下位、直接上位、直接下位概念、同级概念
# 输入：
#     ontology: 本体，枚举值{Spatial, MapContent, MapMethod, Theme}, 必选
#     concept: 概念, 概念全称，必选
#     relation：关系, 枚举值{all_ancestors, all_hyponyms, direct_ancestors, direct_hyponyms, all_relation}，
# 输出：
#     打印控制台
def test_get_concepts_by_target_concept_and_relation(dim, concept, relation="all_relation"):
    # 如果待查询概念是数字哈希的表示形式，先将其转化成完整形式
    if concept.isdigit():
        concept = ConceptIdTransform.id_to_concept(concept)

    ontology = None
    if dim == "MapContent":
        ontology = SWEET
    elif dim == "Spatial":
        ontology = GeoNamesAmerica
    elif dim == "Theme":
        ontology = ThemeVirtualOntology
    elif dim == "MapMethod":
        ontology = MapMethodVirtualOntology
    else:
        print("维度出错")
        return

    # 获取同级概念
    direct_ancestors = ontology.Direct_Ancestors[concept]
    peer_concepts = []
    for tmp_direct_ancestor in direct_ancestors:
        peer_concepts = list(set(peer_concepts).union(set(ontology.Direct_Hyponyms[tmp_direct_ancestor])))
    peer_concepts.remove(concept)

    if relation == "all_ancestors":
        # print(f"All ancestors of {concept} are {ontology.Ancestors[concept]}")
        result = ontology.Ancestors[concept]

    elif relation == "all_hyponyms":
        # print(f"All hyponyms of {concept} are {ontology.Ontologies[concept]}")
        result = ontology.Ontologies[concept]

    elif relation == "direct_ancestors":
        # print(f"The direct ancestors of {concept} are {ontology.Direct_Ancestors[concept]}")
        result = ontology.Direct_Ancestors[concept]

    elif relation == "direct_hyponyms":
        # print(f"The direct hyponyms of {concept} are {ontology.Direct_Hyponyms[concept]}")
        result = ontology.Direct_Hyponyms[concept]

    elif relation == "peer":  # 同级概念
        # print(f"The peer concepts of {concept} are {peer_concepts}")
        result = peer_concepts
    else:
        result = {
            "all ancestors": ontology.Ancestors[concept],
            "all hyponyms": ontology.Ontologies[concept],
            "The direct ancestors": ontology.Direct_Ancestors[concept],
            "The direct hyponyms": ontology.Direct_Hyponyms[concept],
            "The peer concepts": peer_concepts
        }
        # print(f"All ancestors of {concept} are {ontology.Ancestors[concept]}")
        # print(f"All hyponyms of {concept} are {ontology.Ontologies[concept]}")
        # print(f"The direct ancestors of {concept} are {ontology.Direct_Ancestors[concept]}")
        # print(f"The direct hyponyms of {concept} are {ontology.Direct_Hyponyms[concept]}")
        # print(f"The peer concepts of {concept} are {peer_concepts}")
    return result


if __name__ == "__main__":
    # 测试get_concepts_by_target_concept_and_relation的可行性
    # test_concept_1 = "http://sweetontology.net/propTemperature/Temperature"
    # test_concept_2 = "United States"
    # test_concept_3 = "Point Symbol Method"
    # test_concept_4 = "Water"
    # test_get_concepts_by_target_concept_and_relation("C", test_concept_1)
    # test_get_concepts_by_target_concept_and_relation("S", test_concept_2)
    # test_get_concepts_by_target_concept_and_relation("M", test_concept_3)
    # test_get_concepts_by_target_concept_and_relation("T", test_concept_4)

    # # 测试SWEET本体中目标概念与根节点的关系
    # root_concepts = [
    #     "http://sweetontology.net/matr/Substance",
    #     "http://sweetontology.net/prop/ThermodynamicProperty",
    #     "http://sweetontology.net/matrBiomass/LivingEntity",
    #     "http://sweetontology.net/phenSystem/Oscillation",
    #     "http://sweetontology.net/propQuantity/PhysicalQuantity",
    #     "http://sweetontology.net/phenAtmo/MeteorologicalPhenomena", ]
    #
    # for root_concept in root_concepts:
    #     print(f"{root_concept}: {len(SWEET.Ontologies[root_concept])}")
    #
    # target_concepts = [
    #                   # "http://sweetontology.net/matrElement/TransitionMetal",
    #                   "http://sweetontology.net/matrRockIgneous/ExtrusiveRock",
    #                   "http://sweetontology.net/propTemperature/Temperature",
    #                   "http://sweetontology.net/matrRockIgneous/VolcanicRock",
    #                   # "http://sweetontology.net/phenWave/Wave",
    #                   # "http://sweetontology.net/phenAtmoWind/Wind",
    #                   "http://sweetontology.net/matrRock/Rock",
    #                   "http://sweetontology.net/matrAnimal/Animal"
    # ]
    # contain_target_root_concepts = {}
    # for tmp_root_concept in root_concepts:
    #     contain_target_root_concepts[tmp_root_concept] = []
    #     for tmp_target_concept in target_concepts:
    #         if tmp_target_concept in SWEET.Ontologies[tmp_root_concept]:
    #             contain_target_root_concepts[tmp_root_concept].append(tmp_target_concept)
    # print(contain_target_root_concepts)

    # 为样本总库的生成提供目标概念
    # selected_concepts_for_database = {}
    # # "http://sweetontology.net/matrElement/TransitionMetal","http://sweetontology.net/phenAtmoWind/Wind","http://sweetontology.net/phenWave/Wave", "North America", "http://sweetontology.net/phenWave/Wave",
    # intention_concepts = {
    #     "MapContent": [
    #           "http://sweetontology.net/matrRockIgneous/ExtrusiveRock",
    #           "http://sweetontology.net/propTemperature/Temperature",
    #           "http://sweetontology.net/matrRockIgneous/VolcanicRock",
    #           "http://sweetontology.net/matrAnimal/Animal", "http://sweetontology.net/matrRock/Rock"],
    #     "Spatial": ["United States", "Brazil", "Florida", "Colorado"],
    # }
    # for tmp_dim in intention_concepts:
    #     selected_concepts_for_database[tmp_dim] = []
    #     for tmp_concept in intention_concepts[tmp_dim]:
    #         tmp_selected_concepts = []
    #         all_hyponyms = test_get_concepts_by_target_concept_and_relation(tmp_dim, tmp_concept, "all_hyponyms")
    #         all_ancestors = test_get_concepts_by_target_concept_and_relation(tmp_dim, tmp_concept, "all_ancestors")
    #         peer = test_get_concepts_by_target_concept_and_relation(tmp_dim, tmp_concept, "peer")
    #         tmp_selected_concepts = list(set(all_hyponyms + all_ancestors + peer))
    #         print(f"{tmp_concept}: {len(tmp_selected_concepts)}")
    #         selected_concepts_for_database[tmp_dim] += tmp_selected_concepts
    #     selected_concepts_for_database[tmp_dim] = list(set(selected_concepts_for_database[tmp_dim]))
    # selected_concepts_for_database["Theme"] = ["Geology", "Biodiversity", "Climate", "Water", "Disaster"]
    # selected_concepts_for_database["MapMethod"] = ["Area Method", "Quality Base Method", "Point Symbol Method",
    #                                        "Choloplethic Method", "Line Symbol Method"]
    # for tmp_dim in selected_concepts_for_database:
    #     print(f"{tmp_dim}: {len(selected_concepts_for_database[tmp_dim])}")
    # save_as_json(selected_concepts_for_database, "../../resources/samples/target_dim_concepts.json")
    print("Aye")
