"""
   脚本描述：依据上轮意图识别结果及置信度，生成下轮具有引导用户表达意图作用的样本（guide_feedback)
   对比方法：直接反馈（direct_feedback）和相似度反馈（sim_feedback)

"""
import copy
import random
import time
from collections import defaultdict

from src.main.util import OntologyUtil, RetrievalUtil, FileUtil
from src.main.samples.input.Data import Data

distance_cache = FileUtil.load_json("../../../resources/samples/all_dimension_concept_distance.json")
similarity_cache = FileUtil.load_json("../../../resources/samples/all_dimension_concept_similarity.json")


# 直接反馈：直接将意图检索到的图层反馈给用户进行下一轮标注
# param:
#      intention: 上轮识别到的意图(list)
#      all_samples: 样本总库(list)
#      feedback_num: 反馈数量
# output:
#      feedback_samples: 下轮迭代的候选样本集合（list）
#      retrieval_time_use: 从数据库中检索出目标样本的耗时
#      cal_sim_time_use: 意图与样本相似度计算耗时
def direct_feedback(intention, all_samples, feedback_num=None):
    samples_id = {}

    time0 = time.time()
    for i, sub_intention in enumerate(intention):
        samples_id[i] = set(RetrievalUtil.get_sub_intention_covered_samples(sub_intention, all_samples, Data.Ontologies)[0])
    time1 = time.time()
    retrieval_time_use = time1 - time0

    if feedback_num is None:
        # 如果不限制反馈数目, 则返回所有符合条件的样本
        result_id = set([x for y in samples_id.values() for x in y])
        # random.shuffle(result_id)
        result = [all_samples[tmp_id] for tmp_id in result_id]
    else:
        # 如果限制反馈数目，则要求各反馈子意图覆盖的样本数量大致相同（数量均衡）
        # 若某子意图候选样本数目不足，则选择该子意图的所有候选样本，剩余的待选数量平均分到剩余的子意图中
        k = len(intention)
        result_id = set()
        # 每个子意图应选的样本数量
        sub_intention_cover_samples_num = [int(feedback_num / k)] * k
        sub_intention_cover_samples_num[k - 1] += int(feedback_num % k)
        # 每个子意图实际选择的样本数量
        selected_num = copy.deepcopy(sub_intention_cover_samples_num)
        # 每个子意图剩余待选的样本数量
        rest_selected_samples_num = 0
        for i, sub_intention in enumerate(intention):
            # 剩余的待选数量平均分到剩余的候选子意图中
            for j in range(i, k):
                sub_intention_cover_samples_num[j] += int(rest_selected_samples_num / (k - i))
            sub_intention_cover_samples_num[k - 1] += int(rest_selected_samples_num % (k - i))
            # 为该子意图选择候选样本
            selected_num[i] = min(len(samples_id[i]), sub_intention_cover_samples_num[i])
            tmp_result_id = set(random.sample(list(samples_id[i]), selected_num[i]))
            rest_selected_samples_num = sub_intention_cover_samples_num[i] - selected_num[i]
            result_id |= tmp_result_id
            # 对下一个子意图覆盖的样本集合进行去重处理
            if i < k - 1:
                samples_id[i + 1] -= set(tmp_result_id)
        result = [all_samples[tmp_id] for tmp_id in result_id]
    return result, retrieval_time_use, 0


# 引导反馈：通过相似性指标评价样本对目标意图的代表性，距离指标评价样本对模糊意图的可区分性，选择高价值样本组成候选样本集合供用户进行下轮标注
# param:
#      intention: 上轮识别到的意图(list)
#      intention_conf: 意图的置信度(int)
#      sub_intention_conf: 子意图的置信度(list)
#      conflict_samples: 问题样本，每轮反馈中与该轮识别意图冲突的样本(list)
#      all_samples: 样本总库(list)
#      feedback_num: 反馈数量
# output:
#      feedback_samples: 下轮迭代的候选样本集合（list）
#      retrieval_time_use: 从数据库中检索出目标样本的耗时
#      cal_sim_time_use: 意图与样本相似度计算耗时
def guide_feedback(intention, intention_conf, sub_intention_conf, conflict_samples, all_samples, feedback_num=500):
    # 依据引导样本集合选择策略生成下轮候选的待标注样本
    feedback_samples = []

    # 需要选择的问题样本数量
    selected_conflict_samples_num = int(feedback_num * (1 - intention_conf))

    # 依据样本加入的时间顺序，在问题样本集合中从后往前选择样本，若问题样本过少，则将剩余的待选数量归到需要选择的意图样本数量中
    conflict_samples_reverse = copy.deepcopy(conflict_samples)
    conflict_samples_reverse.reverse()
    selected_conflict_samples = conflict_samples_reverse[:min(len(conflict_samples), selected_conflict_samples_num)]
    feedback_samples += selected_conflict_samples

    # 需要选择的意图样本数量（包括代表性与多样性）
    selected_intention_samples_num = feedback_num - len(selected_conflict_samples)

    # 需要选择的子意图样本数量（包括代表性与多样性）
    selected_sub_intention_samples_num = int(selected_intention_samples_num / len(intention))

    retrieval_time_use = 0
    cal_sim_time_use = 0
    # 通过样本代表性与多样性评价样本价值
    for i, sub_intention in enumerate(intention):
        # 每个子意图的代表性样本数量
        selected_sub_intention_representative_num = int(sub_intention_conf[i] * selected_sub_intention_samples_num)

        # 意图覆盖的样本
        time0 = time.time()
        sub_intention_cover_samples_id, sub_intention_cover_samples = \
            RetrievalUtil.get_sub_intention_covered_samples(sub_intention, all_samples, Data.Ontologies)
        time1 = time.time()

        # 在意图覆盖的样本中，依据样本与意图的相似性计算样本的代表性
        sim_result, _ = cal_samples_representative(sub_intention, sub_intention_cover_samples)
        time2 = time.time()

        # 相似性从高到低选择该子意图对应的代表性样本, 若代表性样本不够，则将剩余的待选数量累加到需要选择的多样性样本数量中
        selected_sub_intention_representative_num = min(len(sim_result), selected_sub_intention_representative_num)
        selected_sub_intention_representative_samples = \
            [tmp_record["sample"] for tmp_record in sim_result[:selected_sub_intention_representative_num]]

        # 每个子意图的多样性样本数量
        selected_sub_intention_informative_num = \
            selected_sub_intention_samples_num - len(selected_sub_intention_representative_samples)

        # 通过意图覆盖的样本补集，找到非意图覆盖样本，计算样本的多样性（信息量）。
        time3 = time.time()
        candidate_informative_samples_id = set(range(0, len(all_samples))) - set(sub_intention_cover_samples_id)
        candidate_informative_samples = [all_samples[tmp_id] for tmp_id in candidate_informative_samples_id]
        time4 = time.time()

        # 在非意图覆盖的样本集合中，依据样本与意图的距离计算样本的多样性（信息量）
        dis_result, _ = cal_samples_informative(sub_intention, candidate_informative_samples)
        time5 = time.time()

        # 距离从高到低选择该子意图对应的多样性样本
        selected_sub_intention_informative_samples =  \
            [tmp_record["sample"] for tmp_record in dis_result[:selected_sub_intention_informative_num]]

        retrieval_time_use += (time1 - time0) + (time4 - time3)
        cal_sim_time_use += (time2 - time1) + (time5 - time4)

        # 每个子意图选择的代表性与多样性样本
        selected_sub_intention_samples = \
            selected_sub_intention_representative_samples + selected_sub_intention_informative_samples

        feedback_samples += selected_sub_intention_samples

    # 引导反馈样本集合去重
    feedback_samples_str_set = set([str(s) for s in feedback_samples])
    feedback_samples = [eval(s) for s in feedback_samples_str_set]

    return feedback_samples, retrieval_time_use, cal_sim_time_use


# 相似度反馈：计算样本与意图的相似度(样本与子意图的相似度取最大值）,依据相似度大小排列样本顺序进行反馈
# param:
#      intention: 上轮识别到的意图(list)
#      all_samples: 样本总库(list)
#      feedback_num: 反馈数量
# output:
#      feedback_samples: 下轮迭代的候选样本集合（list）
#      retrieval_time_use: 从数据库中检索出目标样本的耗时
#      cal_sim_time_use: 意图与样本相似度计算耗时
def sim_feedback(intention, all_samples, feedback_num=None):
    if not feedback_num:
        candidate_samples_num = len(all_samples)
    else:
        candidate_samples_num = feedback_num
    similarity = []

    time0 = time.time()
    for tmp_sample in all_samples:
        # 计算样本与每个子意图的相似度
        sub_intention_sample_sim = 0
        for i, sub_intention in enumerate(intention):
            tmp_sub_intention_sample_sim = cal_sim_of_sub_intention_and_sample(sub_intention, tmp_sample)
            # 取样本与子意图的相似度最大值
            sub_intention_sample_sim = max(tmp_sub_intention_sample_sim, sub_intention_sample_sim)
        similarity.append({"sim": sub_intention_sample_sim, "sample": tmp_sample})
    similarity = sorted(similarity, key=lambda x: x["sim"], reverse=True)
    time1 = time.time()
    cal_sim_time_use = time1 - time0
    feedback_samples = [tmp_record["sample"] for tmp_record in similarity[:candidate_samples_num]]
    return feedback_samples, 0, cal_sim_time_use


# 在意图覆盖的样本中，依据样本与意图的相似性计算样本的代表性
# param：
#      sub_intention: 子意图(dict)
#      candidate_samples: 意图覆盖样本(list)
# output:
#      result: 相似性从高到低的样本信息[{"sim": 1.0, "sample": {"Spatial": ...}}, {"sim": 1.0, ...}, ...]
#      hash_map_result: similarity值分布的哈希表, 与result不同之处是返回一个dict，键(key)是相似度，值(value)是相应的样本数组，
#                       {1.0: [{"Spatial": ...},], 0.98: [{"Spatial": ...},]}
def cal_samples_representative(sub_intention, candidate_samples):
    result = []
    # 计算意图覆盖样本与意图的相似度
    for tmp_sample in candidate_samples:
        similarity = cal_sim_of_sub_intention_and_sample(sub_intention, tmp_sample)
        result.append({"sim": similarity, "sample": tmp_sample})
    result = sorted(result, key=lambda x: x["sim"], reverse=True)
    # 同时返回一个similarity值分布的哈希表
    hash_map_result = defaultdict(list)
    for tmp_dict in result:
        similarity = tmp_dict["sim"]
        hash_map_result[similarity].append(tmp_dict["sample"])
    return result, hash_map_result


# 计算子意图与样本之间的相似度，计算方式为各维度相似度均值
# param:
#      sub_intention: 子意图(dict)
#      sample: 单个样本(dict)
# output:
#      result: 子意图A与样本B之间的相似度
def cal_sim_of_sub_intention_and_sample(sub_intention, sample):
    result = 0
    # 各个维度权重，等权
    dimensions_weight = {'Spatial': 0.25,
                         'Theme': 0.25,
                         'MapMethod': 0.25,
                         'MapContent': 0.25}
    for tmp_dim in Data.dimensions:
        sample_values = sample[tmp_dim]
        sub_intention_value = sub_intention[tmp_dim]
        dimension_sim = 0
        # 样本维度多值情况，取维度平均相似度
        for tmp_sample_value in sample_values:
            # 不直接计算，从缓存中读取
            key = str(sorted([sub_intention_value, tmp_sample_value]))
            if key in similarity_cache[tmp_dim]:
                tmp_sim = similarity_cache[tmp_dim][key]
            else:
                _, tmp_sim = OntologyUtil.get_similarity_Lin(sub_intention_value, tmp_sample_value,
                                                             Data.direct_Ancestor[tmp_dim],
                                                             Data.Ontology_Root[tmp_dim],
                                                             Data.concept_information_content[tmp_dim])
            dimension_sim += tmp_sim
        dimension_sim /= len(sample_values)
        result += dimension_sim * dimensions_weight[tmp_dim]
    return result


# 在非意图覆盖的样本集合中，依据样本与意图的距离计算样本的多样性（信息量）
# param：
#      sub_intention: 子意图(dict)
#      candidate_samples: 意图覆盖样本(list)
# output:
#      result: 距离从近到远的样本信息[{"dis": 1.0, "sample": {"Spatial": ...}}, {"dis": 1.0, ...}, ...]
#      hash_map_result: distance值分布的哈希表, 与result不同之处是返回一个dict，键(key)是距离，值(value)是相应的样本数组，
#                       {1.0: [{"Spatial": ...},], 2.0: [{"Spatial": ...},]}
def cal_samples_informative(sub_intention, candidate_samples):
    result = []
    for tmp_sample in candidate_samples:
        distance = cal_dis_of_sub_intention_and_sample(sub_intention, tmp_sample)
        result.append({"dis": distance, "sample": tmp_sample})
    result = sorted(result, key=lambda x: x["dis"])
    # 由于用distance计算概念间的距离有很多的相同值，同时返回一个distance值分布的哈希表
    hash_map_result = defaultdict(list)
    for tmp_dict in result:
        distance = tmp_dict["dis"]
        hash_map_result[distance].append(tmp_dict["sample"])
    return result, hash_map_result


# 计算子意图与样本之间的距离, 计算方式为各个维度的距离和
# param:
#      sub_intention: 子意图(dict)
#      sample: 单个样本(dict)
# output:
#      result: 子意图A与样本B之间的距离
def cal_dis_of_sub_intention_and_sample(sub_intention, sample):
    result = 0
    # 各个维度权重，等权
    dimensions_weight = {'Spatial': 1,
                         'Theme': 1,
                         'MapMethod': 1,
                         'MapContent': 1}
    for tmp_dim in Data.dimensions:
        sample_values = sample[tmp_dim]
        sub_intention_value = sub_intention[tmp_dim]
        dimension_distance = 999
        # 样本维度多值情况，取维度最小值
        for tmp_sample_value in sample_values:
            # 不直接计算，从缓存中读取
            key = str(sorted([sub_intention_value, tmp_sample_value]))
            if key in distance_cache[tmp_dim]:
                tmp_dis = distance_cache[tmp_dim][key]
            else:
                tmp_dis = OntologyUtil.get_value_distance1(sub_intention_value, tmp_sample_value,
                                                           Data.direct_Ancestor[tmp_dim], Data.Ontology_Root[tmp_dim])

            if tmp_dis < dimension_distance:
                dimension_distance = tmp_dis
        # dimension_distance /= len(sample_values)
        result += dimension_distance * dimensions_weight[tmp_dim]
    return result


# 计算本体概念的距离,生成缓存文件并保存，提高计算的效率（已生成）
# 本体概念距离：两个概念的距离为其最短路径上的节点数目减1
# 采用方案一（对应get_value_distance1函数）：先寻找最近公共祖先，再计算两个概念到最近公共祖先的距离（即可确认最短路径）。
# 方案二（对应get_value_distance2函数）：采用dijkstra算法解决，效率低下。舍弃
def get_values_distance_cache():
    from src.main.util.FileUtil import save_as_json
    from src.main.samples.generation import ConceptIdTransform

    # 每个维度的概念距离缓存
    distance_cache = {}
    # # 获取地理要素维度约束样本库的所有概念
    # map_content_restriction_concepts = [
    #     "http://sweetontology.net/matr/Substance",
    #     "http://sweetontology.net/prop/ThermodynamicProperty",
    #     "http://sweetontology.net/matrBiomass/LivingEntity",
    #     "http://sweetontology.net/phenSystem/Oscillation",
    #     # "http://sweetontology.net/propQuantity/PhysicalQuantity",
    #     "http://sweetontology.net/phenAtmo/MeteorologicalPhenomena",
    # ]
    # all_map_content_restriction_values_set = []
    # for tmp_concept in map_content_restriction_concepts:
    #     all_map_content_restriction_values_set.append(tmp_concept)
    #     all_map_content_restriction_values_set += Data.Ontologies['MapContent'][tmp_concept]
    # # 加上地理要素维度的根节点
    # all_map_content_restriction_values_set.append("Thing")
    # all_map_content_restriction_values_set = list(set(all_map_content_restriction_values_set))
    # print(len(all_map_content_restriction_values_set))

    for tmp_dim in Data.dimensions:

        # if tmp_dim == "MapContent":
        #     dim_values = all_map_content_restriction_values_set
        # else:
        #     dim_values = Data.Ontology_values[tmp_dim]

        dim_values = Data.Ontology_values[tmp_dim]
        # 计算两两概念间的距离
        dim_dis_cache = {}
        for value1 in dim_values:
            for value2 in dim_values:
                value1_id = ConceptIdTransform.concept_to_id(value1)
                value2_id = ConceptIdTransform.concept_to_id(value2)
                sort_key = str(sorted([value1_id, value2_id]))
                # sort_key = str(sorted([value1, value2]))
                if sort_key not in dim_dis_cache:
                    distance = OntologyUtil.get_value_distance1(value1, value2, Data.direct_Ancestor[tmp_dim],
                                                                Data.Ontology_Root[tmp_dim])
                    dim_dis_cache[sort_key] = distance
        distance_cache[tmp_dim] = dim_dis_cache

    save_as_json(distance_cache, '../../../resources/samples/all_dimension_concept_distance_id.json')
    print("finish")


# 计算本体概念的相似度,生成缓存文件并保存，提高计算的效率（已生成）
# 概念相似度采用Lin算法
def get_values_similarity_cache():
    from src.main.util.FileUtil import save_as_json
    from src.main.samples.generation import ConceptIdTransform

    similarity_cache = {}  # 各个维度的相似度缓存

    # # 获取地理要素维度约束样本库的所有概念
    # map_content_restriction_concepts = [
    #     "http://sweetontology.net/matr/Substance",
    #     "http://sweetontology.net/prop/ThermodynamicProperty",
    #     "http://sweetontology.net/matrBiomass/LivingEntity",
    #     "http://sweetontology.net/phenSystem/Oscillation",
    #     # "http://sweetontology.net/propQuantity/PhysicalQuantity",
    #     "http://sweetontology.net/phenAtmo/MeteorologicalPhenomena",
    # ]
    # all_map_content_restriction_values_set = []
    # for tmp_concept in map_content_restriction_concepts:
    #     all_map_content_restriction_values_set.append(tmp_concept)
    #     all_map_content_restriction_values_set += Data.Ontologies['MapContent'][tmp_concept]
    # # 加上地理要素维度的根节点
    # all_map_content_restriction_values_set.append("Thing")
    # all_map_content_restriction_values_set = list(set(all_map_content_restriction_values_set))
    # print(len(all_map_content_restriction_values_set))

    for tmp_dim in Data.dimensions:
        # if tmp_dim == "MapContent":
        #     dim_values = all_map_content_restriction_values_set
        # else:
        #     dim_values = Data.Ontology_values[tmp_dim]
        dim_values = Data.Ontology_values[tmp_dim]

        # 计算各个维度的相似度
        dim_sim_cache = {}
        for value1 in dim_values:
            for value2 in dim_values:
                value1_id = ConceptIdTransform.concept_to_id(value1)
                value2_id = ConceptIdTransform.concept_to_id(value2)
                sort_key = str(sorted([value1_id, value2_id]))
                # sort_key = str(sorted([value1, value2]))
                if sort_key not in dim_sim_cache:
                    _, similarity = OntologyUtil.get_similarity_Lin(value1, value2, Data.direct_Ancestor[tmp_dim],
                                                                 Data.Ontology_Root[tmp_dim],
                                                                 Data.concept_information_content[tmp_dim])
                    dim_sim_cache[sort_key] = similarity
        similarity_cache[tmp_dim] = dim_sim_cache

    save_as_json(similarity_cache, '../../../resources/samples/all_dimension_concept_similarity_id.json')
    print("finish")


if __name__ == "__main__":
    # from src.main.util.FileUtil import load_json, save_as_json
    # sample_database = load_json("../../../resources/samples/all_samples.json")
    # sub_intention_a = {"Spatial": "Florida", "MapMethod": "Point Symbol Method", "Theme": "Geology",
    #                  "MapContent": "http://sweetontology.net/matrRockIgneous/VolcanicRock"}
    # sub_intention_b = {"Spatial": "America", "MapMethod": "MapMethodRoot", "Theme": "Biodiversity",
    #                    "MapContent": "Thing"}
    # sample = {"Spatial": ["Florida"], "MapMethod": ["Point Symbol Method", "Line Symbol Method"], "Theme": ["Geology"],
    #                  "MapContent": ["http://sweetontology.net/propTemperature/Temperature"], "ID": 2}
    #
    # sim = cal_sim_of_sub_intention_and_sample(sub_intention_a, sample)
    # dis = cal_dis_of_sub_intention_and_sample(sub_intention_a, sample)
    # print(sim)
    # print(dis)
    #
    # time0 = time.time()
    # samples, time_use_1, time_use2 = direct_feedback([sub_intention_b], sample_database)
    # time1 = time.time()
    # samples, time_use_1, time_use2 = guide_feedback([sub_intention_b], 0.7, [0.5, 0.5, 0.8], [sample] * 10, sample_database)
    # time2 = time.time()
    # samples, time_use_1, time_use2 = sim_feedback([sub_intention_b], sample_database)
    # time3 = time.time()
    #
    # print(time1 - time0)
    # print(time2 - time1)
    # print(time3 - time2)

    # get_values_distance_cache()
    # get_values_similarity_cache()
    pass
