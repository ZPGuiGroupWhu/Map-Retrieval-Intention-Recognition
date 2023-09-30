"""
   脚本描述：算法的评价指标，Jaccard、 BMASS、 Precision、Recall、F1
   需要有真实意图 real_intention与算法识别意图 extracted_intention
"""

import copy
import itertools
from itertools import permutations
from src.main.util.RetrievalUtil import get_intention_covered_samples
# the Jaccard Distance
from src.main.util import OntologyUtil


# 两个数组求交集（数组中可能有重复元素），不是两个集合求交集（多次反馈中可能存在相同的样本，此时不应只计数1次）
# 采用相同排序，双指针的算法
# param
#      num1, num2: 两个list数组
#      res:  输出交集数组
def interaction(nums1, nums2):
    nums1.sort()
    nums2.sort()
    res = []
    p1 = 0
    p2 = 0
    while p1 < len(nums1) and p2 < len(nums2):
        if nums1[p1] == nums2[p2]:
            res.append(nums1[p1])
            p1 += 1
            p2 += 1
        elif nums1[p1] < nums2[p2]:
            p1 += 1
        else:
            p2 += 1
    return res


# 两个数组求并集（数组中可能有重复元素），不是两个集合求并集
# 采用相同排序，双指针的算法
# param
#      num1, num2: 两个list数组
#      res:  输出交集数组
def union(nums1, nums2):
    nums1.sort()
    nums2.sort()
    res = []
    p1 = 0
    p2 = 0
    while p1 < len(nums1) and p2 < len(nums2):
        if nums1[p1] == nums2[p2]:
            res.append(nums1[p1])
            p1 += 1
            p2 += 1
        elif nums1[p1] < nums2[p2]:
            res.append(nums1[p1])
            p1 += 1
        else:
            res.append(nums2[p2])
            p2 += 1
    if p1 < len(nums1):
        res += nums1[p1:]
    if p2 < len(nums2):
        res += nums2[p2:]
    return res


# Attention：在样本总库中使用Jaccard、Precision、Recall评价
def get_jaccard_index(samples, real_intention, extracted_intention, ontologies):
    # retrieve the samples
    real_intention_retrieved_samples_index, _ = get_intention_covered_samples(real_intention, samples, ontologies)

    extracted_intention_retrieved_samples_index, _ = get_intention_covered_samples(extracted_intention, samples,
                                                                                   ontologies)

    tmp_value1 = len(interaction(real_intention_retrieved_samples_index, extracted_intention_retrieved_samples_index))
    tmp_value2 = len(union(real_intention_retrieved_samples_index, extracted_intention_retrieved_samples_index))

    # tmp_value1 = len(real_intention_retrieved_samples_index.intersection(
    #     extracted_intention_retrieved_samples_index))
    # tmp_value2 = len(real_intention_retrieved_samples_index.union(
    #     extracted_intention_retrieved_samples_index))

    return tmp_value1 / tmp_value2


# 计算两个（关系，取值）形式的子意图的相似度
def similarity_of_sub_intentions(sub_intention1, sub_intention2, direct_ancestors, ontology_root,
                                 concept_information_content_yuan2013):
    if sub_intention2 is None or sub_intention1 is None:
        return 0
    considered_dimension_num = 0
    sum_similarity = 0
    for tmp_dim in sub_intention1:
        sub_intention1_tmp_dim_value = sub_intention1[tmp_dim]
        sub_intention2_tmp_dim_value = sub_intention2[tmp_dim]
        considered_dimension_num += 1
        if sub_intention1_tmp_dim_value == sub_intention2_tmp_dim_value == ontology_root[tmp_dim]:
            tmp_similarity = 1
        else:
            _, tmp_similarity = OntologyUtil.get_similarity_Lin(sub_intention1_tmp_dim_value,
                                                                sub_intention2_tmp_dim_value,
                                                                direct_ancestors[tmp_dim], ontology_root[tmp_dim],
                                                                concept_information_content_yuan2013[tmp_dim])
        sum_similarity += tmp_similarity
    if considered_dimension_num == 0:
        return 0
    return sum_similarity / considered_dimension_num


# 计算补空意图后最佳映射平均相似度BMASS
#   1.对于子意图数量较少的意图A，选择意图A的任一子意图补全意图A，使两个意图A、B的子意图数量相等（需要遍历所有情况）
#   2.得到两个意图间所有的一对一映射，定义所有具有映射关系的子意图的语义相似度之和取平均作为两个意图的相似度
#   3.将最大相似度作为两个意图的最终相似度
def get_intention_similarity(intention_a, intention_b, direct_ancestors, ontology_root,
                             concept_information_content_yuan2013):
    intention_a_copy = copy.deepcopy(intention_a)
    intention_b_copy = copy.deepcopy(intention_b)
    # 以None代表空子意图
    min_length_intention = intention_a_copy if len(intention_a_copy) <= len(intention_b_copy) else intention_b_copy
    max_length_intention = intention_a_copy if len(intention_a_copy) > len(intention_b_copy) else intention_b_copy

    # 在长度较短的意图中遍历所有的子意图，补充较短的意图直至两意图数量相等
    possible_min_length_intention_list = []
    selected_sub_intention_num = len(max_length_intention) - len(min_length_intention)

    all_candidate_sub_intention_combinations = list(
        itertools.product(min_length_intention, repeat=selected_sub_intention_num))
    all_candidate_sub_intention_permutations = [list(comb) for comb in all_candidate_sub_intention_combinations]
    for candidate_sub_intention_permutation in all_candidate_sub_intention_permutations:
        possible_min_length_intention_list.append(
            copy.deepcopy(min_length_intention) + candidate_sub_intention_permutation)

    # 得到所有映射, 固定一个意图的子意图顺序不变，求另一个意图所有的子意图排列情况
    # 固定min_length_intention, 排列max_length_intention，排列通过索引实现
    max_similarity = 0
    for min_length_intention in possible_min_length_intention_list:
        maps = permutations(range(len(max_length_intention)), len(min_length_intention))
        for tmp_max_length_intention_index in maps:
            # 对于每一种映射，求取其相似度
            tmp_similarity = 0
            for i in range(len(tmp_max_length_intention_index)):
                tmp_min_length_sub_intention_index = i
                tmp_max_length_sub_intention_index = tmp_max_length_intention_index[i]
                tmp_min_length_sub_intention = min_length_intention[tmp_min_length_sub_intention_index]
                tmp_max_length_sub_intention = max_length_intention[tmp_max_length_sub_intention_index]
                tmp_sub_intention_similarity = similarity_of_sub_intentions(tmp_min_length_sub_intention,
                                                                            tmp_max_length_sub_intention,
                                                                            direct_ancestors, ontology_root,
                                                                            concept_information_content_yuan2013)
                tmp_similarity += tmp_sub_intention_similarity
            tmp_similarity /= len(max_length_intention)  # 平均相似度
            if max_similarity < tmp_similarity:
                max_similarity = tmp_similarity
    return max_similarity  # 以最佳映射作为最终相似度结果


# 在样本总集中计算准确率
def get_precision(samples, real_intention, extracted_intention, ontologies):
    # positive_samples = test_samples["positive"]
    # negative_samples = test_samples["negative"]
    # all_samples = positive_samples + negative_samples
    real_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(real_intention, samples, ontologies)
    extracted_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, samples, ontologies)
    true_positive_num = \
        len(interaction(real_intention_retrieved_all_samples_index, extracted_intention_retrieved_all_samples_index))
    result = 0 if true_positive_num == 0 else true_positive_num / len(extracted_intention_retrieved_all_samples_index)
    return result


# 在当前反馈样本中计算召回率，而非在样本总集中计算召回率
def get_recall(samples, real_intention, extracted_intention, ontologies):
    # positive_samples = test_samples["positive"]
    # negative_samples = test_samples["negative"]
    # all_samples = positive_samples + negative_samples
    real_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(real_intention, samples, ontologies)
    extracted_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, samples, ontologies)
    true_positive_num = \
        len(interaction(real_intention_retrieved_all_samples_index, extracted_intention_retrieved_all_samples_index))
    result = true_positive_num / len(real_intention_retrieved_all_samples_index)
    return result


def get_F1_score(test_samples, real_intention, extracted_intention, ontologies):
    precision = get_precision(test_samples, real_intention, extracted_intention, ontologies)
    recall = get_recall(test_samples, real_intention, extracted_intention, ontologies)
    # 避免分母为0
    if recall + precision == 0:
        result = 0
    else:
        result = 2 * precision * recall / (recall + precision)
    return result


# 在当前反馈中计算Jaccard，而非在样本总集中计算Jaccard，要求samples未扩展
def get_jaccard_index_old_version(samples, real_intention, extracted_intention, ontologies):
    # retrieve the samples
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]

    real_intention_retrieved_positive_samples_index, _ = \
        get_intention_covered_samples(real_intention, positive_samples, ontologies)
    real_intention_retrieved_negative_samples_index, _ = \
        get_intention_covered_samples(real_intention, negative_samples, ontologies)

    extracted_intention_retrieved_positive_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, positive_samples, ontologies)
    extracted_intention_retrieved_negative_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, negative_samples, ontologies)

    tmp_value1_positive_samples = len(real_intention_retrieved_positive_samples_index.intersection(
        extracted_intention_retrieved_positive_samples_index))
    tmp_value2_positive_samples = len(real_intention_retrieved_positive_samples_index.union(
        extracted_intention_retrieved_positive_samples_index))
    tmp_value1_negative_samples = len(real_intention_retrieved_negative_samples_index.intersection(
        extracted_intention_retrieved_negative_samples_index))
    tmp_value2_negative_samples = len(real_intention_retrieved_negative_samples_index.union(
        extracted_intention_retrieved_negative_samples_index))
    tmp_value1 = tmp_value1_positive_samples + tmp_value1_negative_samples
    tmp_value2 = tmp_value2_positive_samples + tmp_value2_negative_samples
    return tmp_value1 / tmp_value2


# 由于子意图数量对该计算方法的影响过大（意图数量对相似性的影响大于维度取值），导致意图数量不等时，BMASS普遍偏低，且在维度取值不同时没有区分性，故废弃
# 计算补空意图后最佳映射平均相似度BMASS
#   1.将子意图数量较少的意图添加空子意图使两个意图子意图数量相等
#   2.得到两个意图间所有的一对一映射，定义所有具有映射关系的子意图的语义相似度之和取平均作为两个意图的相似度
#   3.将最大相似度作为两个意图的最终相似度
def get_intention_similarity_old_version(intention_a, intention_b, direct_ancestors, ontology_root,
                                         concept_information_content_yuan2013):
    # if
    intention_a_copy = copy.deepcopy(intention_a)
    intention_b_copy = copy.deepcopy(intention_b)
    # 以None代表空子意图
    min_length_intention = intention_a_copy if len(intention_a_copy) <= len(intention_b_copy) else intention_b_copy
    max_length_intention = intention_a_copy if len(intention_a_copy) > len(intention_b_copy) else intention_b_copy

    # 得到所有映射, 固定一个意图的子意图顺序不变，求另一个意图所有的子意图排列情况
    # 固定min_length_intention, 排列max_length_intention，排列通过索引实现
    maps = permutations(range(len(max_length_intention)), len(min_length_intention))
    max_similarity = 0
    for tmp_max_length_intention_index in maps:
        # 对于每一种映射，求取其相似度
        tmp_similarity = 0
        for i in range(len(tmp_max_length_intention_index)):
            tmp_min_length_sub_intention_index = i
            tmp_max_length_sub_intention_index = tmp_max_length_intention_index[i]
            tmp_min_length_sub_intention = min_length_intention[tmp_min_length_sub_intention_index]
            tmp_max_length_sub_intention = max_length_intention[tmp_max_length_sub_intention_index]
            tmp_sub_intention_similarity = similarity_of_sub_intentions(tmp_min_length_sub_intention,
                                                                        tmp_max_length_sub_intention,
                                                                        direct_ancestors, ontology_root,
                                                                        concept_information_content_yuan2013)
            tmp_similarity += tmp_sub_intention_similarity
        tmp_similarity /= len(max_length_intention)  # 平均相似度
        if max_similarity < tmp_similarity:
            max_similarity = tmp_similarity
    return max_similarity  # 以最佳映射作为最终相似度结果


# 在当前反馈样本中计算准确率，而非在样本总集中计算准确率
def get_precision_old_version(test_samples, real_intention, extracted_intention, ontologies):
    positive_samples = test_samples["positive"]
    negative_samples = test_samples["negative"]
    all_samples = positive_samples + negative_samples
    real_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(real_intention, all_samples, ontologies)
    extracted_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, all_samples, ontologies)
    true_positive_num = \
        len(real_intention_retrieved_all_samples_index & extracted_intention_retrieved_all_samples_index)
    result = 0 if true_positive_num == 0 else true_positive_num / len(extracted_intention_retrieved_all_samples_index)
    return result


# 在当前反馈样本中计算召回率，而非在样本总集中计算召回率
def get_recall_old_version(test_samples, real_intention, extracted_intention, ontologies):
    positive_samples = test_samples["positive"]
    negative_samples = test_samples["negative"]
    all_samples = positive_samples + negative_samples
    real_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(real_intention, all_samples, ontologies)
    extracted_intention_retrieved_all_samples_index, _ = \
        get_intention_covered_samples(extracted_intention, all_samples, ontologies)
    true_positive_num = \
        len(real_intention_retrieved_all_samples_index & extracted_intention_retrieved_all_samples_index)
    result = true_positive_num / len(real_intention_retrieved_all_samples_index)
    return result


if __name__ == "__main__":
    # # 测试get_intention_similarity准确性
    # from src.main.samples.input.Data import Data
    #
    # # intention_a = [{'MapMethod': 'Line Symbol Method', 'Spatial': 'United States', 'Theme': 'Water', 'MapContent': 'http://sweetontology.net/phenWave/Wave'},
    # #                {'Spatial': 'North America', 'MapMethod': 'Point Symbol Method', 'Theme': 'Geology', 'MapContent': 'http://sweetontology.net/matrRockIgneous/VolcanicRock'}]
    # # intention_b = [{'MapContent': 'http://sweetontology.net/repr/Representation', 'Spatial': 'America', 'Theme': 'Water', 'MapMethod': 'MapMethodRoot'},
    # #                {'MapMethod': 'Line Symbol Method', 'Spatial': 'United States', 'Theme': 'Water', 'MapContent': 'http://sweetontology.net/phenWave/Wave'},
    # #                {'Spatial': 'North America', 'MapMethod': 'Point Symbol Method', 'Theme': 'Geology', 'MapContent': 'http://sweetontology.net/matrRockIgneous/VolcanicRock'},
    # #                {'Spatial': 'South America', 'MapMethod': 'Point Symbol Method', 'Theme': 'Geology', 'MapContent': 'http://sweetontology.net/matrRockIgneous/VolcanicRock'}]
    # # result = get_intention_similarity(intention_a, intention_b, Data.direct_Ancestor, Data.Ontology_Root,
    # #                          Data.concept_information_content)
    # # print(result)
    # intention_a = [
    #     {"MapMethod": "Area Method", "MapContent": "http://sweetontology.net/matrAnimal/Animal", "Spatial": "Brazil",
    #      "Theme": "ThemeRoot"}]
    # intention_b = [
    #     {"MapMethod": "Area Method", "MapContent": "http://sweetontology.net/matrAnimal/Animal", "Spatial": "Brazil",
    #      "Theme": "ThemeRoot"}]
    # # from src.main.util.FileUtil import load_json
    # # all_samples = load_json("../../../resources/samples/all_samples.json")
    # # intention_a = [{"Theme": "Biodiversity", "Spatial": "America", "MapMethod": "MapMethodRoot", "MapContent": "Thing"}]
    # # intention_b = [{"Theme": "Biodiversity", "Spatial": "America", "MapMethod": "MapMethodRoot", "MapContent": "Thing"}]
    # # result = get_jaccard_index(all_samples, intention_a, intention_b, Data.Ontologies)
    # result = get_intention_similarity(intention_a, intention_b, Data.direct_Ancestor, Data.Ontology_Root,
    #                                   Data.concept_information_content)
    # print(result)
    pass
