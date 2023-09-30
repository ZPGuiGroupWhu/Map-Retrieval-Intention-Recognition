# 基于频繁项集和最小描述长度的意图识别算法
# 算法分为两步：首先使用频繁项集挖掘算法提取候选子意图，然后使用贪心算法识别最优意图
import copy
import time
from math import log2
from src.main.samples.input.Data import Data

sub_intention_encoding_length = None

"""
   第一步，使用维度约束的Apriori挖掘候选子意图
"""


# param：
#   all_relevance_concepts：正样本中所有标签的所有相关概念，包括标签本身及它们的所有上位概念
#   该概念集合由Data.py在导入相关反馈样本的同时生成，其格式为{dim1: [concept11, concept12], dim2: [concpet21, ]}
# 初始化可能的频繁1项集合
# result = [("Spatial", "North America"), ("Theme", "Agriculture")}
def get_init_sub_intentions(all_relevance_concepts):
    result = []
    for tmp_dim in all_relevance_concepts:
        tmp_dim_all_relevance_concepts = all_relevance_concepts[tmp_dim]
        for tmp_concept in tmp_dim_all_relevance_concepts:
            tmp_item = (tmp_dim, tmp_concept)
            result.append(frozenset({tmp_item}))
    result.sort()
    return result


# 获取频繁维度分量集合（也即候选子意图）覆盖的样本集合，通过sample_type指定样本集合的正负, 频繁项集中调用
# 参数：
#   sub_intention: [(dim_name, concept), ...]
#   all_relevance_concepts_covered_samples：正样本相关概念覆盖的正负样本id集合，该变量由Data.py在导入样本的时候生成
def get_sub_intention_covered_samples(sub_intention, all_relevance_concepts_covered_samples, sample_type):
    result = set()
    first_value = True
    for tmp_dim_value in sub_intention:
        tmp_dim_name, tmp_concept = tmp_dim_value
        tmp_relation_value_tuple_covered_samples_index = \
            all_relevance_concepts_covered_samples[tmp_dim_name][tmp_concept][sample_type]
        if first_value:
            result |= tmp_relation_value_tuple_covered_samples_index
            first_value = False
        else:
            result &= tmp_relation_value_tuple_covered_samples_index
    return result


# 获取候选子意图覆盖的样本集合，通过sample_type指定样本集合的正负， 贪心搜索最优意图中调用
#   该方法与get_sub_intention_covered_samples的不同点在于子意图的格式不同
# sub_intention: {dim_name: concept, ...}
def get_sub_intention_covered_samples2(sub_intention, all_relevance_concepts_covered_samples, sample_type):
    result = set()
    first_value = True
    for tmp_dim_name in sub_intention:
        tmp_concept = sub_intention[tmp_dim_name]
        tmp_relation_value_tuple_covered_samples_index = \
            all_relevance_concepts_covered_samples[tmp_dim_name][tmp_concept][sample_type]
        if first_value:
            result |= tmp_relation_value_tuple_covered_samples_index
            first_value = False
        else:
            result &= tmp_relation_value_tuple_covered_samples_index
    return result


# intention: 意图[[(dim_name, concept), ...], ...]
def get_intention_covered_samples(intention, all_relevance_concepts_covered_samples, sample_type):
    result = set()
    for tmp_sub_intention in intention:
        tmp_sub_intention_covered_samples = \
            get_sub_intention_covered_samples(tmp_sub_intention, all_relevance_concepts_covered_samples,
                                              sample_type)
        result |= tmp_sub_intention_covered_samples
    return result


# 检查sub_intention是否合法，即是否每个维度仅包含一个值
def is_legal_sub_intention(sub_intention, dimensions):
    sub_intention_dims = [tmp_dim_value[0] for tmp_dim_value in sub_intention]
    for tmp_dim in dimensions:
        tmp_dim_count = sub_intention_dims.count(tmp_dim)
        if tmp_dim_count > 1:
            return False
    return True


# 检查intention是否合法，即是否每个维度仅包含一个值, 未被使用
def is_legal_intention(intention, dimensions):
    intention_dims = [[tmp_dim_value[0] for tmp_dim_value in sub_intention] for sub_intention in intention]
    for tmp_sub_intention_dims in intention_dims:
        for tmp_dim in dimensions:
            tmp_dim_count = tmp_sub_intention_dims.count(tmp_dim)
            if tmp_dim_count > 1:
                return False
    return True


# 扫描项集，去除支持度小于阈值的项集
def scan_sub_intentions(sub_intentions_k, min_support, all_relevance_concepts_covered_samples, samples,
                        dimensions):
    # 计算sub_intention_k中每一个子意图覆盖的正样本数量，返回满足最小支持度且各维度仅含有一个取值的子意图
    # 并返回支持度信息
    result = []
    result_hash_map = {}
    relevance_samples_num = len(samples["positive"])
    for tmp_sub_intention in sub_intentions_k:
        if not is_legal_sub_intention(tmp_sub_intention, dimensions):
            continue
        tmp_sub_intention_covered_relevance_samples = \
            get_sub_intention_covered_samples(tmp_sub_intention, all_relevance_concepts_covered_samples,
                                              "positive")
        tmp_sub_intention_covered_relevance_samples_num = len(tmp_sub_intention_covered_relevance_samples)
        tmp_sub_intention_support = float(tmp_sub_intention_covered_relevance_samples_num) / relevance_samples_num
        if tmp_sub_intention_support > min_support:
            result.append(tmp_sub_intention)
            result_hash_map[tmp_sub_intention] = tmp_sub_intention_covered_relevance_samples_num
    return result, result_hash_map


# 由频繁k项集扩展得到频繁k+1项集
def extend_sub_intentions(sub_intention_k, k):
    result = []
    sub_intention_k_num = len(sub_intention_k)
    for i in range(sub_intention_k_num):
        for j in range(i + 1, sub_intention_k_num):
            L1 = list(sub_intention_k[i])[: k - 2]
            L2 = list(sub_intention_k[j])[: k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                result.append(sub_intention_k[i] | sub_intention_k[j])
    return result


# 将频繁维度分量转换为候选子意图的形式
def transform_sub_intention(sub_intention, ontology_root_concepts):
    result = {}
    for tmp_dim_value in sub_intention:
        tmp_dim_name, tmp_relation_value_tuple = tmp_dim_value
        result[tmp_dim_name] = tmp_relation_value_tuple
    for tmp_dim in ontology_root_concepts:
        if tmp_dim not in result:
            result[tmp_dim] = ontology_root_concepts[tmp_dim]
    return result


# 基于正样本集合获得候选子意图（正向）
# 参数：
#   samples：反馈样本集合，格式为{"relevance": [sample1, ...], "irrelevance": [sample2, ...]}
#       sample1 = {"dim1": [concept1, concept2], dim2: [concept3, ...], ...}
#   min_support：候选子意图的最小支持度阈值，取值为(0, 1]
#   k_max：候选子意图中包含的最大维度分量数量，由于当前形式化表达方式规定子意图中各维度至多出现一次，因此该值等于维度数量
def get_all_candidate_sub_intentions(data, min_support, k_max=4):
    # data = Data(samples)
    all_relevance_concepts = data.all_relevance_concepts
    samples = data.docs
    dimensions = data.dimensions
    all_relevance_concepts_covered_samples = data.all_relevance_concepts_retrieved_docs
    init_sub_intentions = get_init_sub_intentions(all_relevance_concepts)
    L1, L1_hash_map = scan_sub_intentions(init_sub_intentions, min_support,
                                          all_relevance_concepts_covered_samples, samples, dimensions)
    L = [L1]
    L_hash_map = [L1_hash_map]
    if k_max == 1:
        return L1
    k = 2
    while len(L[k - 2]) > 0:
        Ck = extend_sub_intentions(L[k - 2], k)
        Lk, Lk_hash_map = scan_sub_intentions(Ck, min_support, all_relevance_concepts_covered_samples,
                                              samples, dimensions)
        L.append(Lk)
        L_hash_map.append(Lk_hash_map)
        k += 1
        if k_max is not None:
            if k > k_max:
                break
    candidate_sub_intentions = [x for y in L for x in y]
    candidate_sub_intentions = [transform_sub_intention(x, data.Ontology_Root) for x in candidate_sub_intentions]

    # frequent_items = [{frequent_item: count} for y in L_hash_map for frequent_item, count in y.items()]
    # 返回每轮的频繁项集frequent_item, 以及支持该项集的记录数量L_hash_map
    return candidate_sub_intentions, L_hash_map


"""
   第二步，贪心搜索最优（编码长度）最短的子意图组合
"""


# 判断子意图A是否覆盖子意图B，若子意图A在各个维度上的概念都是子意图B对应维度的等价概念或是祖先概念，则子意图A覆盖子意图B，否则，不覆盖
# sub_intention_a(sub_intention_b) = {'MapContent': 'http://sweetontology.net/matrRockIgneous/VolcanicRock',
#                                     'Theme': 'Geology', 'Spatial': 'North America', 'MapMethod': 'MapMethodRoot'}
def is_sub_intention_cover(sub_intention_a, sub_intention_b, ontologies):
    for tmp_dim in sub_intention_a:
        tmp_dim_value_a = sub_intention_a[tmp_dim]
        tmp_dim_value_b = sub_intention_b[tmp_dim]
        if tmp_dim_value_a == tmp_dim_value_b or tmp_dim_value_b in ontologies[tmp_dim][tmp_dim_value_a]:
            continue
        else:
            return False
    return True


# 判断意图A是否覆盖子意图B，标准：任意子意图A覆盖子意图B, 则意图A是否覆盖子意图B，否则，则不覆盖
# intention_a = [{'MapContent': '', 'Theme': '', 'Spatial': '', 'MapMethod': ''}],
# sub_intention_b = {'MapContent': '', 'Theme': '', 'Spatial': '', 'MapMethod': ''}
def is_intention_cover_sub_intention(intention_a, sub_intention_b, ontologies):
    for tmp_sub_intention in intention_a:
        if is_sub_intention_cover(tmp_sub_intention, sub_intention_b, ontologies):
            return True
    return False


# 样本平均编码长度
def get_average_encoding_length(positive_num, negative_num):
    if positive_num == 0 or negative_num == 0:
        return 0
    p = positive_num / (positive_num + negative_num)
    return -(p * log2(p) + (1 - p) * log2(1 - p))


# the average minimum encoding length by Shannon's Noiseless Channel Coding Theorem
def get_data_encoding_length_by_amcl(positive_num, negative_num):
    if positive_num == 0 or negative_num == 0:
        return 0
    tmp_average_encoding_length = get_average_encoding_length(positive_num, negative_num)
    result = tmp_average_encoding_length * (positive_num + negative_num)
    return result


# Rissanen's universal code for integers
# This code makes no priori assumption about the maximum number num
def rissanen(num):
    k0 = 2.865064
    result = log2(k0)
    tmp_value = num
    while tmp_value >= 1:
        tmp_value = log2(tmp_value)
        result += tmp_value
    return result


# values in every dimension encoding separately using uniform code
# rissanen(4) means the encoding length of the dimension number of every sub intention
# sub_intention_encoding_length = sum([log2(x) for x in dimension_value_nums]) + rissanen(4)
def get_sub_intention_encoding_length(dimensions, ontology_values):
    global sub_intention_encoding_length
    if sub_intention_encoding_length is None:
        # 初始化无意图，sub_intention_num = 1，避免取对数时出错
        sub_intention_num = 1
        for tmp_dim in dimensions:
            #     print(len(Data.Ontologies[tmp_dim]))
            # 只考虑从属关系
            tmp_dim_relation_num = 1
            tmp_dim_values_num = len(ontology_values[tmp_dim])
            sub_intention_num *= (tmp_dim_values_num * tmp_dim_relation_num)
        sub_intention_encoding_length = log2(sub_intention_num)
    return sub_intention_encoding_length


# 将对应的编码长度计算分离出来，用于在过滤掉不合格子规则后调用重新计算编码长度。其中不合格子规则是覆盖的正样本占正样本集合的比例低于阈值的子规则。
# input params:
# intention_with_stat_info = [[sub_intention,
#     sub_intention_covered_positive_samples_index,
#     sub_intention_covered_negative_samples_index,
#     sub_intention_covered_positive_negative_num,
#     average_encoding_length]]
# positive_samples_num: 正样本数量
# negative_samples_num: 负样本数量
# per_positive_sample_times: 正样本的复制次数
# per_negative_sample_times: 负样本的复制次数
# without sub intention order constraint
# result = (
#   tmp_remain_uncovered_positive_samples_index = [],
#   tmp_remain_uncovered_negative_samples_index = [],
#   new_rules_list = [rule1, rule2, ...],
#       rule = [tmp_merged_rule,
#               tmp_merged_rule_covered_positive_samples_index,
#               tmp_merged_rule_covered_negative_samples_index,
#               tmp_merged_rule_covered_positive_negative_num,
#               average_encoding_length]
#   encoding_length = [total_encoding_length, intention_encoding_length, sample_encoding_length]
# )
def get_intention_encoding_length_without_order_constraint(intention_with_stat_info,
                                                           positive_samples_num, negative_samples_num,
                                                           per_positive_sample_times, per_negative_sample_times,
                                                           dimensions, ontology_values):
    intention_with_stat_info.sort(key=lambda x: x[-1], reverse=False)
    # get every rule covered samples id and calculate total encoding length
    tmp_intention_encoding_length = 0
    tmp_sample_encoding_length = 0

    tmp_covered_positive_samples_index = set()
    tmp_covered_negative_samples_index = set()
    new_intention_with_stat_info = {}
    tmp_index = 0

    for tmp_sub_intention_with_stat_info in intention_with_stat_info:
        tmp_sub_intention_covered_positive_samples_index = tmp_sub_intention_with_stat_info[1]
        tmp_sub_intention_covered_negative_samples_index = tmp_sub_intention_with_stat_info[2]

        # 开始分配覆盖的正负反馈样本，子意图对应的样本平均编码长度越小，则分配越多的正样本与负样本
        tmp_sub_intention_covered_samples_num = \
            len(tmp_sub_intention_covered_positive_samples_index) * per_positive_sample_times \
            + len(tmp_sub_intention_covered_negative_samples_index) * per_negative_sample_times
        tmp_sub_intention_real_covered_positive_samples_index = \
            tmp_sub_intention_covered_positive_samples_index - tmp_covered_positive_samples_index
        tmp_sub_intention_real_covered_negative_samples_index = \
            tmp_sub_intention_covered_negative_samples_index - tmp_covered_negative_samples_index

        # 考虑样本增强系数
        tmp_sub_intention_real_covered_positive_samples_num = \
            len(tmp_sub_intention_real_covered_positive_samples_index) * per_positive_sample_times
        tmp_sub_intention_real_covered_negative_samples_num = \
            len(tmp_sub_intention_real_covered_negative_samples_index) * per_negative_sample_times
        tmp_sub_intention_real_covered_samples_num = \
            tmp_sub_intention_real_covered_positive_samples_num + tmp_sub_intention_real_covered_negative_samples_num
        # 如果一个子意图覆盖的正样本数量小于负样本数量，则说明此子意图描述的不是意图而是非意图，应该去除它
        if tmp_sub_intention_real_covered_positive_samples_num < tmp_sub_intention_real_covered_negative_samples_num:
            continue
        # 如果某个子意图不再覆盖正样本，则说明它原来覆盖的样本已经被其他子意图覆盖了，也应该去除它
        if tmp_sub_intention_real_covered_positive_samples_num == 0:
            continue
        # add the sample encoding length covered by sub intention
        # 将给定当前子意图后，当前子意图覆盖的正样本的编码长度加到给定意图后样本的编码长度上
        # 给定子意图后，样本的编码长度分为两部分，一部分是子意图覆盖的正负样本的平均编码长度乘以子意图覆盖的样本数量，另一部分是当前子意图覆盖的正样本的编码长度
        tmp_sample_encoding_length += tmp_sub_intention_real_covered_samples_num * tmp_sub_intention_with_stat_info[4] \
                                      + log2(tmp_sub_intention_real_covered_positive_samples_num)
        # add sub intention encoding length
        # 加上子意图本身的编码长度，包括子意图本身的编码长度
        tmp_intention_encoding_length += get_sub_intention_encoding_length(dimensions, ontology_values)

        # 更新样本覆盖情况
        tmp_covered_positive_samples_index |= tmp_sub_intention_covered_positive_samples_index
        tmp_covered_negative_samples_index |= tmp_sub_intention_covered_negative_samples_index
        # 将当前子意图添加到意图中
        # new_intention_with_stat_info.append(tmp_sub_intention_with_stat_info)
        new_intention_with_stat_info[tmp_index] = tmp_sub_intention_with_stat_info
        tmp_index += 1
    # 计算剩余样本的编码长度
    tmp_remain_uncovered_positive_samples_index = set(range(positive_samples_num)) - tmp_covered_positive_samples_index
    tmp_remain_uncovered_negative_samples_index = set(range(negative_samples_num)) - tmp_covered_negative_samples_index
    tmp_remain_uncovered_positive_samples_num = \
        len(tmp_remain_uncovered_positive_samples_index) * per_positive_sample_times
    tmp_remain_uncovered_negative_samples_num = \
        len(tmp_remain_uncovered_negative_samples_index) * per_negative_sample_times
    # add the remain samples encoding length
    tmp_sample_encoding_length += get_data_encoding_length_by_amcl(tmp_remain_uncovered_positive_samples_num,
                                                                   tmp_remain_uncovered_negative_samples_num)
    # 意图编码长度还需要加上子意图数量的编码长度
    tmp_intention_encoding_length += rissanen(len(new_intention_with_stat_info) + 1)
    tmp_total_encoding_length = tmp_sample_encoding_length + tmp_intention_encoding_length
    return (
        tmp_remain_uncovered_positive_samples_index,
        tmp_remain_uncovered_negative_samples_index,
        new_intention_with_stat_info,
        [tmp_total_encoding_length, tmp_intention_encoding_length, tmp_sample_encoding_length],
    )


# 尝试将子意图加入到已有意图中，计算加入后的样本编码长度
# without sub intention order constraint
# input params:
# data : instance of Data2 class
# sub_intention_to_add : {dim1_name: dim1_relation_value_tuple, ...}
# current_intention_with_stat_info: {0: rule0, 1: rule1, ...}
# #       rule = [tmp_merged_rule,
# #               tmp_merged_rule_covered_positive_samples_index,
# #               tmp_merged_rule_covered_negative_samples_index,
# #               tmp_merged_rule_covered_positive_negative_num,
# #               average_encoding_length]
# positive_samples:
# result = (
#   tmp_remain_uncovered_positive_samples_index = [],
#   tmp_remain_uncovered_negative_samples_index = [],
#   new_rules_list = {0: rule0, 1: rule1, ...}
#       rule = [tmp_merged_rule,
#               tmp_merged_rule_covered_positive_samples_index,
#               tmp_merged_rule_covered_negative_samples_index,
#               tmp_merged_rule_covered_positive_negative_num,
#               average_encoding_length],
#   encoding_length = [total_encoding_length, intention_encoding_length, sample_encoding_length]
# )
def get_sub_intention_to_add_statistics_without_order_constraint(data, sub_intention_to_add,
                                                                 current_intention_with_stat_info,
                                                                 positive_samples_num, negative_samples_num,
                                                                 per_positive_sample_times, per_negative_sample_times):
    # global time_use_retrieve_docs
    all_relevance_concepts_retrieved_samples = data.all_relevance_concepts_retrieved_docs

    dimensions = data.dimensions
    ontologies = data.Ontologies
    ontology_values = data.Ontology_values

    # retrieve the positive samples and negative samples and get retrieved ids
    # time01 = time.time()
    # 将子意图覆盖的样本的计算转换为各维度分量覆盖样本集合的交集计算
    sub_intention_to_add_covered_positive_samples_index = \
        get_sub_intention_covered_samples2(sub_intention_to_add,
                                           all_relevance_concepts_retrieved_samples, "positive")

    sub_intention_to_add_covered_negative_samples_index = \
        get_sub_intention_covered_samples2(sub_intention_to_add,
                                           all_relevance_concepts_retrieved_samples, "negative")
    # time02 = time.time()
    # time_use_retrieve_docs += time02 - time01

    sub_intention_to_add_covered_positive_samples_num = \
        len(sub_intention_to_add_covered_positive_samples_index) * per_positive_sample_times
    sub_intention_to_add_covered_negative_samples_num = \
        len(sub_intention_to_add_covered_negative_samples_index) * per_negative_sample_times
    sub_intention_to_add_covered_positive_negative_num = [sub_intention_to_add_covered_positive_samples_num,
                                                          sub_intention_to_add_covered_negative_samples_num]
    # get the average minimum encoding length based on Shannon encoding method
    average_encoding_length = get_average_encoding_length(sub_intention_to_add_covered_positive_samples_num,
                                                          sub_intention_to_add_covered_negative_samples_num)
    # calculate the encoding length
    # sort rules by their average encoding length and then get every rules covered samples
    # if some rules cover no sample, then it will be filtered
    sorted_sub_intentions = [[sub_intention_to_add,
                              sub_intention_to_add_covered_positive_samples_index,
                              sub_intention_to_add_covered_negative_samples_index,
                              sub_intention_to_add_covered_positive_negative_num,
                              average_encoding_length]]
    for tmp_sub_intention_id in current_intention_with_stat_info:
        tmp_sub_intention_with_stat_info = current_intention_with_stat_info[tmp_sub_intention_id]
        tmp_sub_intention_content = tmp_sub_intention_with_stat_info[0]

        # TODO 这里是否还需要过滤掉被需要添加的子意图覆盖的已有子意图？
        if is_sub_intention_cover(sub_intention_to_add, tmp_sub_intention_content, ontologies):
            continue
        sorted_sub_intentions.append(copy.deepcopy(tmp_sub_intention_with_stat_info))
    return get_intention_encoding_length_without_order_constraint(sorted_sub_intentions,
                                                                  positive_samples_num, negative_samples_num,
                                                                  per_positive_sample_times, per_negative_sample_times,
                                                                  dimensions, ontology_values)


# 计算样本增强系数
def get_sample_augment_index(positive_samples_num, negative_samples_num, dimensions, ontology_values):
    tmp_sub_intention_encoding_length = get_sub_intention_encoding_length(dimensions, ontology_values)
    tmp_average_encoding_length = get_average_encoding_length(positive_samples_num, negative_samples_num)
    result = \
        (rissanen(positive_samples_num + 1) +
         positive_samples_num * tmp_sub_intention_encoding_length) / \
        ((positive_samples_num + negative_samples_num) * tmp_average_encoding_length) + 1
    return result


# IntentionWithStatInfo2在intention_with_stat_info之外又包装了一层，用于记录意图未覆盖的正负样本ID集合及编码长度
# intention_with_stat_info = {0: sub_intention_with_stat_info0, 1: sub_intention_with_stat_info1, ...}
#       sub_intention_with_stat_info = [sub_intention,
#               sub_intention_covered_positive_samples_index,
#               sub_intention_covered_negative_samples_index,
#               sub_intention_covered_positive_negative_num,
#               average_encoding_length]
class IntentionWithStatInfo2:
    def __init__(self, uncovered_positive_samples_index, uncovered_negative_samples_index,
                 intention_with_stat_info, encoding_length):
        self.uncovered_positive_samples_index = uncovered_positive_samples_index
        self.uncovered_negative_samples_index = uncovered_negative_samples_index
        self.intention_with_stat_info = intention_with_stat_info
        self.total_encoding_length = encoding_length[0]
        self.method_log = []
        self.intention_encoding_length = encoding_length[1]
        self.sample_encoding_length = encoding_length[2]

    def get_pure_intention(self):
        if len(self.intention_with_stat_info) == 0:
            return [copy.deepcopy(Data.Ontology_Root)]
        result = []
        for tmp_sub_intention_id in self.intention_with_stat_info:
            tmp_sub_intention_with_stat_info = self.intention_with_stat_info[tmp_sub_intention_id]
            tmp_pure_sub_intention = tmp_sub_intention_with_stat_info[0]
            result.append(tmp_pure_sub_intention)
        return result


# 基于随机合并与集束搜索的意图提取
# params
#   samples 用于提取意图的正负样本，各维度取值已经被扩展
#   data_encoding_method 数据编码方式
#   random_merge_number 每次随机合并的次数
#   beam_width  集束宽度，每次保留的个数
def get_intention_by_method6(data, config, candidate_sub_intentions):
    # time00 = time.time()
    # 初始化
    samples = data.docs
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]
    positive_samples_num = len(positive_samples)
    negative_samples_num = len(negative_samples)
    ontologies = data.Ontologies
    ontology_values = data.Ontology_values
    ontology_root = data.Ontology_Root
    dimensions = data.dimensions

    # 是否进行样本增强
    if config.adjust_sample_num:
        per_positive_sample_times = get_sample_augment_index(positive_samples_num, negative_samples_num,
                                                             dimensions, ontology_values)
    else:
        per_positive_sample_times = 1
    per_negative_sample_times = per_positive_sample_times
    uncovered_negative_samples_index = set(range(positive_samples_num))
    uncovered_positive_samples_index = set(range(negative_samples_num))
    init_intention_encoding_length = rissanen(1)
    init_sample_encoding_length = get_data_encoding_length_by_amcl(positive_samples_num * per_positive_sample_times,
                                                                   negative_samples_num * per_negative_sample_times)
    init_total_encoding_length = init_intention_encoding_length + init_sample_encoding_length
    init_encoding_length = [init_total_encoding_length, init_intention_encoding_length, init_sample_encoding_length]

    beam_width = config.beam_width

    # time01 = time.time()
    # time_use_sample_enhancement += time01 - time00

    # 初始化意图列表，添加集束宽度个空规则
    #   rules == intention， rule == sub_intention
    intention_list = []
    # 将本体的根节点当做无意图
    init_intention = {
        "Spatial": ontology_root['Spatial'],
        "Theme": ontology_root['Theme'],
        "MapMethod": ontology_root['MapMethod'],
        "MapContent": ontology_root["MapContent"]}
    iteration_log = [[{
        "iteration": 0,
        "covered_positive_sample_rates": [1.0],
        "total_encoding_length": init_total_encoding_length,
        "intention_encoding_length": init_intention_encoding_length,
        "sample_encoding_length": init_sample_encoding_length,
        "encoding_length_compression_rates": 1.0,
        "intention_result": init_intention
    }]]
    iteration_number = 1
    # 初始化临时意图列表
    tmp_intention_with_stat_info2_list = []
    # 用于记录剩余候选子意图，每个意图对应一个剩余子意图集合
    tmp_remain_sub_intention_list = []
    for i in range(beam_width):
        tmp_intention_with_stat_info2 = IntentionWithStatInfo2(copy.deepcopy(uncovered_positive_samples_index),
                                                               copy.deepcopy(uncovered_negative_samples_index), {},
                                                               init_encoding_length)
        tmp_intention_with_stat_info2_list.append(tmp_intention_with_stat_info2)
        tmp_intention_remain_sub_intention = copy.deepcopy(candidate_sub_intentions)
        tmp_remain_sub_intention_list.append(tmp_intention_remain_sub_intention)
    while True:
        # print([x.rules for x in tmp_rules_list])

        tmp_iteration_log = []

        new_tmp_intention_with_stat_info2_list = []
        # 开始遍历临时意图列表，尝试为每个意图添加剩余的最适合他们的候选子意图，需要为每个意图记录一下剩余的候选子意图

        for i, tmp_intention_with_stat_info2 in enumerate(tmp_intention_with_stat_info2_list):
            tmp_intention_with_stat_info = tmp_intention_with_stat_info2.intention_with_stat_info
            # 遍历当前意图的候选子意图集合，找到最好的那个
            tmp_intention_remain_sub_intentions = tmp_remain_sub_intention_list[i]
            tmp_added_candidate_sub_intention_num = 0  # 因为拥有比现有意图更小的总编码长度而被添加的候选子意图数量
            for tmp_sub_intention_to_add in tmp_intention_remain_sub_intentions:
                # 计算样本总编码长度
                tmp_sub_intention_to_add_stat = \
                    get_sub_intention_to_add_statistics_without_order_constraint(data, tmp_sub_intention_to_add,
                                                                                 tmp_intention_with_stat_info,
                                                                                 positive_samples_num,
                                                                                 negative_samples_num,
                                                                                 per_positive_sample_times,
                                                                                 per_negative_sample_times)
                tmp_remain_uncovered_positive_samples_index = tmp_sub_intention_to_add_stat[0]
                tmp_remain_uncovered_negative_samples_index = tmp_sub_intention_to_add_stat[1]
                new_intention_with_stat_info = tmp_sub_intention_to_add_stat[2]
                tmp_encoding_lengths = tmp_sub_intention_to_add_stat[3]
                tmp_total_encoding_length = tmp_encoding_lengths[0]
                # tmp_intention_encoding_length = tmp_encoding_lengths[1]
                # tmp_sample_encoding_length = tmp_encoding_lengths[2]

                # 如果加入当前候选子意图后编码长度变小
                if tmp_total_encoding_length < tmp_intention_with_stat_info2.total_encoding_length:
                    tmp_added_candidate_sub_intention_num += 1
                    # 得到添加后的新意图
                    new_tmp_intention = IntentionWithStatInfo2(tmp_remain_uncovered_positive_samples_index,
                                                               tmp_remain_uncovered_negative_samples_index,
                                                               new_intention_with_stat_info, tmp_encoding_lengths)
                    # 将合并得到的意图添加到临时意图列表
                    new_tmp_intention_with_stat_info2_list.append(new_tmp_intention)
                    # for tmp_rule_id in rules:
                    #     print(rules[tmp_rule_id])
                    # print(min_encoding_length, init_min_encoding_length)

            # 如果没有新的子意图被添加，则说明无法找到使总编码长度更小的意图了，停止针对当前意图的迭代搜索，将其加入最终意图列表
            # 但是在此时加入的话，最终意图列表intention_list中的意图数量可能大于beam_width，因为每条搜索路径的最终意图都被加入进来了
            if tmp_added_candidate_sub_intention_num == 0:
                intention_list.append(tmp_intention_with_stat_info2)

        # 根据临时意图列表判断是否继续，如果临时意图列表为空，表明此次搜索没有找到更好的意图，退出
        if len(new_tmp_intention_with_stat_info2_list) == 0:
            break
        # 否则保留beam_width个意图识别结果，并更新每个意图结果对应的剩余候选子意图集合
        else:
            new_tmp_intention_with_stat_info2_list.sort(key=lambda x: x.total_encoding_length)
            # todo: 总编码长度相等时，通过意图覆盖来取本体概念中最下层的意图（最具体的意图）
            tmp_intention_with_stat_info2_list = \
                new_tmp_intention_with_stat_info2_list[:min(len(new_tmp_intention_with_stat_info2_list), beam_width)]
            # record iteration log
            if config.record_iteration_progress:
                for intention_index, tmp_intention_with_stat_info2 in enumerate(tmp_intention_with_stat_info2_list):
                    tmp_intention_result = tmp_intention_with_stat_info2.get_pure_intention()
                    tmp_covered_positive_sample_rates = []
                    tmp_intention_with_stat_info = tmp_intention_with_stat_info2.intention_with_stat_info
                    for tmp_sub_intention_id in tmp_intention_with_stat_info:
                        tmp_sub_intention_with_stat_info = tmp_intention_with_stat_info[tmp_sub_intention_id]
                        tmp_sub_intention_covered_positive_samples_num = len(tmp_sub_intention_with_stat_info[1])
                        tmp_sub_intention_covered_positive_sample_rate = \
                            tmp_sub_intention_covered_positive_samples_num / positive_samples_num
                        tmp_covered_positive_sample_rates.append(tmp_sub_intention_covered_positive_sample_rate)
                    total_encoding_length = tmp_intention_with_stat_info2.total_encoding_length
                    encoding_length_compression_rates = total_encoding_length / init_total_encoding_length
                    tmp_intention_log = {
                        "iteration": iteration_number,
                        "covered_positive_sample_rates": tmp_covered_positive_sample_rates,
                        "total_encoding_length": total_encoding_length,
                        "intention_encoding_length": tmp_intention_with_stat_info2.intention_encoding_length,
                        "sample_encoding_length": tmp_intention_with_stat_info2.sample_encoding_length,
                        "encoding_length_compression_rates": encoding_length_compression_rates,
                        "intention_result": tmp_intention_result
                    }
                    tmp_iteration_log.append(tmp_intention_log)
                iteration_log.append(tmp_iteration_log)
                iteration_number += 1

            # 更新每个意图对应的剩余候选子意图集合
            new_tmp_remain_sub_intention_list = []
            if beam_width == 1:
                tmp_intention = tmp_intention_with_stat_info2_list[0].get_pure_intention()
                tmp_remain_sub_intentions = tmp_remain_sub_intention_list[0]
                new_tmp_remain_sub_intentions = []
                for tmp_remain_sub_intention in tmp_remain_sub_intentions:
                    if not is_intention_cover_sub_intention(tmp_intention, tmp_remain_sub_intention, ontologies):
                        new_tmp_remain_sub_intentions.append(tmp_remain_sub_intention)
                new_tmp_remain_sub_intention_list.append(new_tmp_remain_sub_intentions)
            else:
                for i, tmp_intention_with_stat_info2 in enumerate(tmp_intention_with_stat_info2_list):
                    tmp_intention = tmp_intention_with_stat_info2.get_pure_intention()
                    new_tmp_remain_sub_intentions = []
                    for tmp_remain_sub_intention in candidate_sub_intentions:
                        if not is_intention_cover_sub_intention(tmp_intention, tmp_remain_sub_intention, ontologies):
                            new_tmp_remain_sub_intentions.append(tmp_remain_sub_intention)
                    new_tmp_remain_sub_intention_list.append(new_tmp_remain_sub_intentions)
            tmp_remain_sub_intention_list = new_tmp_remain_sub_intention_list

    # 找出最好的那个意图
    intention_list.sort(key=lambda x: x.total_encoding_length)
    best_intention = intention_list[0]
    # time07 = time.time()
    # time_all = time07 - time00

    return (best_intention.get_pure_intention(), best_intention.total_encoding_length,
            init_total_encoding_length, iteration_log)


# 根据data和设置得到意图
def get_intentions_by_greedy_search(samples, config):
    # expended_samples = Sample.add_samples_relation_information(samples)
    time0 = time.time()
    data = Data(samples)
    time1 = time.time()
    # 首先得到候选子意图
    min_support = config.rule_covered_positive_sample_rate_threshold
    k_max = len(Data.dimensions)
    candidate_sub_intentions, _ = get_all_candidate_sub_intentions(data, min_support, k_max)
    time2 = time.time()
    # 然后开始进行意图识别
    # 初始化意图为空，然后尝试加入候选子意图，尝试时计算编码长度，若确定要加入某个子意图，则需要去掉候选子意图中被当前子意图覆盖的部分
    # 需要记录每个子意图覆盖的正负样本编号
    # 需要记录每次加入子意图后的编码长度
    # 需要计算意图长度与给定意图后的样本编码长度
    intention, total_encoding_length, init_total_encoding_length, iteration_log = \
        get_intention_by_method6(data, config, candidate_sub_intentions)
    time3 = time.time()
    # print("candidate_num", len(candidate_sub_intentions))
    method_log = {
        # 添加时间记录
        "time_use": {
            "time_use_expend_sample_and_create_hash": (time1 - time0),
            "time_use_candidate": (time2 - time1),
            "time_use_greedy_search": (time3 - time2)
        },
        "iteration_log": iteration_log,
        "candidate_num": len(candidate_sub_intentions)
    }
    return intention, total_encoding_length, init_total_encoding_length, method_log


if __name__ == "__main__":
    # from src.main.samples.input import DimensionValues Ontologies = {'Spatial':
    # DimensionValues.SpatialValue.Ontologies, 'Theme': DimensionValues.ThemeValues.Ontologies, 'MapMethod':
    # DimensionValues.MapMethodValues.Ontologies, 'MapContent': DimensionValues.MapContentValues.Ontologies}
    # test_intention_a = [
    # {'MapContent': 'http://sweetontology.net/matrRockIgneous/VolcanicRock', 'Theme': 'Water',
    # 'Spatial': 'America', 'MapMethod': 'MapMethodRoot'} ,
    # {'MapContent': 'http://sweetontology.net/matrRockIgneous/VolcanicRock', 'Theme': 'Geology',
    # 'Spatial': 'North America', 'MapMethod': 'MapMethodRoot'}]
    # test_sub_intention_b = {'MapContent': 'http://sweetontology.net/matrRockIgneous/VolcanicRock', 'Theme': 'Geology',
    # 'Spatial': 'America', 'MapMethod':
    # 'Area Method'} print(is_intention_cover_sub_intention(test_intention_a, test_sub_intention_b, Ontologies))
    pass
