"""
   脚本描述：算法内部的评价指标（置信度）
   只需要算法识别意图 extracted_intention
"""

from src.main.util.RetrievalUtil import get_intention_covered_samples, get_sub_intention_covered_samples


# param:
#      samples:             当前反馈样本集合 {"positive":[], "negative":[], ...}
#      extracted_intention: 算法识别的意图 [{"Spacial": "North America", "Theme": "Water", ...}, {...}]
#      ontologies:          各个维度的下位概念 {"Spatial": [], "Theme": [], ...}
# 计算意图相容性（在当前反馈集合中的准确率）
def get_intention_compatibility(samples, extracted_intention, ontologies):
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]
    all_samples = positive_samples + negative_samples
    extracted_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, all_samples, ontologies)
    extracted_intention_retrieved_positive_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, positive_samples, ontologies)
    # 避免分母为0
    if len(extracted_intention_retrieved_all_samples_index) == 0:
        result = 0
    else:
        result = len(extracted_intention_retrieved_positive_samples_index) / len(
                    extracted_intention_retrieved_all_samples_index)

    return result


# 计算意图完备性（在当前反馈集合中的召回率）
def get_intention_completeness(samples, extracted_intention, ontologies):
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]

    extracted_intention_retrieved_positive_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, positive_samples, ontologies)
    result = len(extracted_intention_retrieved_positive_samples_index) / len(positive_samples)
    return result


# 计算意图的置信度，计算公式为意图相容性与完备性的调和平均数
def get_intention_conf(samples, extracted_intention, ontologies):
    compatibility = get_intention_compatibility(samples, extracted_intention, ontologies)
    completeness = get_intention_completeness(samples, extracted_intention, ontologies)
    # 避免分母为0
    if compatibility + completeness == 0:
        conf = 0
    else:
        conf = 2 * compatibility * completeness / (compatibility + completeness)
    return conf


# param:
#      samples:             当前反馈样本集合 {"positive":[], "negative":[], ...}
#      sub_intention:       子意图 {"Spacial": "North America", "Theme": "Water", ...}
#      ontologies:          各个维度的下位概念 {"Spatial": [], "Theme": [], ...}
# 计算子意图相容性（在当前反馈集合中的准确率）
def get_sub_intention_compatibility(samples, sub_intention, ontologies):
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]
    all_samples = positive_samples + negative_samples
    sub_intention_retrieved_all_samples_index, _ = \
        get_sub_intention_covered_samples(sub_intention, all_samples, ontologies)
    sub_intention_retrieved_positive_samples_index, _ = \
        get_sub_intention_covered_samples(sub_intention, positive_samples, ontologies)
    # 避免分母为0
    if len(sub_intention_retrieved_all_samples_index) == 0:
        result = 0
    else:
        result = len(sub_intention_retrieved_positive_samples_index) / len(
            sub_intention_retrieved_all_samples_index)
    return result


# 计算子意图完备性（在当前反馈集合中的召回率）
def get_sub_intention_completeness(samples, sub_intention, ontologies):
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]

    sub_intention_retrieved_positive_samples_index, _ = \
        get_sub_intention_covered_samples(sub_intention, positive_samples, ontologies)
    result = len(sub_intention_retrieved_positive_samples_index) / len(positive_samples)
    return result


# 计算子意图的置信度，计算公式为子意图相容性与完备性的调和平均数
def get_sub_intention_conf(samples, sub_intention, ontologies):
    compatibility = get_sub_intention_compatibility(samples, sub_intention, ontologies)
    completeness = get_sub_intention_completeness(samples, sub_intention, ontologies)
    # 若相容性与完备性都为0，则置信度为0
    if (compatibility + completeness) == 0:
        conf = 0
    else:
        conf = 2 * compatibility * completeness / (compatibility + completeness)
    return conf


if __name__ == "__main__":
    # 测试置信度计算的准确性
    # from src.main.samples.input import Sample
    # from src.main.samples.input.Data import Data
    # scene = "35"
    # sample_version = "scenes_v4_7"
    # sample_path = "../../../resources/samples/" + sample_version + "/Scene" + scene + "/samples_F0p2_L0p2.json"
    # samples, real_intention = Sample.load_sample_from_file(sample_path)
    # print(get_intention_compatibility(samples, real_intention, Data.Ontologies))
    # print(get_intention_completeness(samples, real_intention, Data.Ontologies))
    # print(get_intention_conf(samples, real_intention, Data.Ontologies))
    #
    # print(get_sub_intention_compatibility(samples, real_intention[1], Data.Ontologies))
    # print(get_sub_intention_completeness(samples, real_intention[1], Data.Ontologies))
    # print(get_sub_intention_conf(samples, real_intention[1], Data.Ontologies))
    pass
