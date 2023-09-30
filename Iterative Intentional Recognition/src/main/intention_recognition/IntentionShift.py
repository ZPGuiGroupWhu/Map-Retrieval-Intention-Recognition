"""
   脚本描述：意图偏移判断准则（压缩率、意图覆盖率、子意图覆盖率）
   需要上轮意图识别结果、上轮以及本轮的正负反馈样本
"""
from src.main.intention_recognition.Apriori_MDL import *


# 判断指标，若意图对正样本覆盖率不减少且对负样本覆盖率不增加，则认为意图未发生变化，否则，意图发生改变
# 意图对正样本的覆盖率等于意图覆盖的正样本占本轮反馈所有正样本的比例
# 意图对负样本的覆盖率等于意图覆盖的负样本占本轮反馈所有负样本的比例
# param:
#      data: 初始化后的样本数据（对样本进行重新组织），来自samples.input.Data中Data类的封装
#      last_intention: 上轮意图识别结果（list）
#      last_intention_cover_rate: 上轮意图对上轮样本的覆盖情况 {positive_samples_rate: 0.8, negative_samples_rate: 0.2}
def shift_judgement_by_intention_cover_rate(data, last_intention, last_intention_cover_rate):
    positive_samples_support, negative_samples_support = cal_intention_cover_sample_rate(data, last_intention)

    if positive_samples_support >= last_intention_cover_rate["positive_samples_rate"] \
            and negative_samples_support <= last_intention_cover_rate["negative_samples_rate"]:
        return False
    else:
        return True


# 判断指标，若对于所有子意图而言，子意图正样本覆盖率不减少且负样本覆盖率不增加，则认为意图未发生变化，否则，意图发生改变
# 子意图对正样本的覆盖率等于子意图覆盖的正样本占本轮反馈所有正样本的比例
# 子意图对负样本的覆盖率等于子意图覆盖的负样本占本轮反馈所有负样本的比例
# param:
#      data: 初始化后的样本数据（对样本进行重新组织），来自samples.input.Data中Data类的封装
#      last_intention: 上轮意图识别结果（list）
#      last_sub_intention_cover_rate: 上轮意图对上轮样本的覆盖情况 [{"positive_samples_rate": 0.8, "negative_samples_rate": 0.2}]
def shift_judgement_by_sub_intention_cover_rate(data, last_intention, last_sub_intention_cover_rate):
    is_intention_shift = False
    for i, sub_intention in enumerate(last_intention):
        sub_intention_positive_samples_support, sub_intention_negative_samples_support = \
            cal_intention_cover_sample_rate(data, [sub_intention])
        if sub_intention_positive_samples_support < last_sub_intention_cover_rate[i]["positive_samples_rate"] or \
                sub_intention_negative_samples_support > last_sub_intention_cover_rate[i]["negative_samples_rate"]:
            is_intention_shift = True
            return is_intention_shift

    return is_intention_shift


# 计算意图对正负样本的覆盖率
# param:
#      data: 初始化后的样本数据（对样本进行重新组织），来自samples.input.Data中Data类的封装
#      intention: 意图识别结果（list）
def cal_intention_cover_sample_rate(data, intention):
    samples = data.docs
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]
    covered_positive_samples_index = set()
    covered_negative_samples_index = set()
    for sub_intention in intention:
        covered_positive_samples_index |= \
            get_sub_intention_covered_samples2(sub_intention, data.all_relevance_concepts_retrieved_docs,
                                               "positive")
        covered_negative_samples_index |= \
            get_sub_intention_covered_samples2(sub_intention, data.all_relevance_concepts_retrieved_docs,
                                               "negative")
    intention_cover_positive_samples_rate = len(covered_positive_samples_index) / len(positive_samples)
    intention_cover_negative_samples_rate = len(covered_negative_samples_index) / len(negative_samples)

    return intention_cover_positive_samples_rate, intention_cover_negative_samples_rate


# 判断指标，若意图编码的压缩率没有增加，则认为意图没有改变，否则意图发生变化
# param:
#      data: 初始化后的样本数据（对样本进行重新组织），来自sample.input.Data中Data类的封装
#      last_intention: 上轮意图识别结果（list）
#      last_compress_rate: 上轮意图对上轮反馈的编码压缩率（float)
#      config: Aprior_MDL意图识别算法的参数,来自samples.input.Config中Config类的封装
def shift_judgement_by_compression_rate(data, last_intention, last_compress_rate, config):
    total_encoding_length, init_total_encoding_length = cal_compression_rate_by_intention(data, last_intention, config)
    # 当前意图覆盖的编码长度
    new_compress_rate = total_encoding_length / init_total_encoding_length

    if new_compress_rate <= last_compress_rate:
        return False
    else:
        return True


# 依据意图计算意图编码压缩情况
# param:
#      data: 初始化后的样本数据（对样本进行重新组织），来自sample.input.Data中Data类的封装
#      intention: 意图识别结果（list）
#      config: Aprior_MDL意图识别算法的参数,来自samples.input.Config中Config类的封装
def cal_compression_rate_by_intention(data, intention, config):
    samples = data.docs
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]
    positive_samples_num = len(positive_samples)
    negative_samples_num = len(negative_samples)
    ontologies = data.Ontologies
    ontology_values = data.Ontology_values
    ontology_root = data.Ontology_Root
    dimensions = data.dimensions

    # 是否进行样本增强, todo：不同的复制次数会导致压缩率不一样？
    if config.adjust_sample_num:
        per_positive_sample_times = get_sample_augment_index(positive_samples_num, negative_samples_num,
                                                             dimensions, ontology_values)
    else:
        per_positive_sample_times = 1

    per_negative_sample_times = per_positive_sample_times

    init_intention_encoding_length = rissanen(1)
    init_sample_encoding_length = get_data_encoding_length_by_amcl(
        positive_samples_num * per_positive_sample_times,
        negative_samples_num * per_negative_sample_times)
    init_total_encoding_length = init_intention_encoding_length + init_sample_encoding_length

    sorted_sub_intentions = []
    # 获取每一个子意图的样本覆盖情况
    for sub_intention in intention:
        sub_intention_cover_positive_samples_index = \
            get_sub_intention_covered_samples2(sub_intention, data.all_relevance_concepts_retrieved_docs,
                                               "positive")
        sub_intention_cover_negative_samples_index = \
            get_sub_intention_covered_samples2(sub_intention, data.all_relevance_concepts_retrieved_docs,
                                               "negative")
        sub_intention_cover_positive_samples_num = \
            len(sub_intention_cover_positive_samples_index) * per_positive_sample_times
        sub_intention_cover_negative_samples_num = \
            len(sub_intention_cover_negative_samples_index) * per_negative_sample_times
        sub_intention_covered_positive_negative_num = [sub_intention_cover_positive_samples_num,
                                                       sub_intention_cover_negative_samples_num]
        # get the average minimum encoding length based on Shannon encoding method
        average_encoding_length = get_average_encoding_length(sub_intention_cover_positive_samples_num,
                                                              sub_intention_cover_negative_samples_num)
        # calculate the encoding length
        # sort rules by their average encoding length and then get every rules covered samples
        # if some rules cover no sample, then it will be filtered
        sorted_sub_intentions.append([sub_intention,
                                      sub_intention_cover_positive_samples_index,
                                      sub_intention_cover_negative_samples_index,
                                      sub_intention_covered_positive_negative_num,
                                      average_encoding_length])

    # 意图编码长度
    intention_encoding_length = 0
    # 样本编码长度
    sample_encoding_length = 0

    tmp_covered_positive_samples_index = set()
    tmp_covered_negative_samples_index = set()

    # 根据每个子意图覆盖样本的平均编码长度，分配子意图覆盖的样本（解决重复覆盖的问题）
    # 子意图对应的样本平均编码长度越小，则分配越多的正样本与负样本
    sorted_sub_intentions.sort(key=lambda x: x[-1], reverse=False)

    for tmp_sub_intention_with_stat_info in sorted_sub_intentions:
        intention_encoding_length += get_sub_intention_encoding_length(dimensions, ontology_values)

        # 更新子意图对样本的覆盖情况
        sub_intention_real_covered_positive_samples_index = \
            tmp_sub_intention_with_stat_info[1] - tmp_covered_positive_samples_index
        sub_intention_real_covered_negative_samples_index = \
            tmp_sub_intention_with_stat_info[2] - tmp_covered_negative_samples_index

        # 考虑样本增强系数
        sub_intention_real_covered_positive_samples_num = \
            len(sub_intention_real_covered_positive_samples_index) * per_positive_sample_times
        sub_intention_real_covered_negative_samples_num = \
            len(sub_intention_real_covered_negative_samples_index) * per_negative_sample_times
        sub_intention_real_covered_samples_num = \
            sub_intention_real_covered_positive_samples_num + sub_intention_real_covered_negative_samples_num

        # 更新子意图对应的编码长度
        # get the average minimum encoding length based on Shannon encoding method
        real_average_encoding_length = get_average_encoding_length(
            sub_intention_real_covered_positive_samples_num,
            sub_intention_real_covered_negative_samples_num)

        # 将给定当前子意图后，当前子意图覆盖的正样本的编码长度加到给定意图后样本的编码长度上
        # 给定子意图后，样本的编码长度分为两部分，一部分是子意图覆盖的正负样本的平均编码长度乘以子意图覆盖的样本数量，另一部分是当前子意图覆盖的正样本的编码长度
        # 如果子意图覆盖的真实样本数目为0，则Log2部分直接取0，避免算法报错
        # 可能出现这种情况的原因为，上轮识别到了错误意图，本轮反馈进行了纠正，导致该子意图在本轮反馈中没有覆盖任一正样本
        if sub_intention_real_covered_positive_samples_num == 0:
            sample_encoding_length += sub_intention_real_covered_samples_num * real_average_encoding_length
        else:
            sample_encoding_length += sub_intention_real_covered_samples_num * real_average_encoding_length \
                                  + log2(sub_intention_real_covered_positive_samples_num)

        # 更新样本覆盖情况
        tmp_covered_positive_samples_index |= sub_intention_real_covered_positive_samples_index
        tmp_covered_negative_samples_index |= sub_intention_real_covered_negative_samples_index

    # 计算剩余样本的编码长度
    remain_uncovered_positive_samples_index = set(range(positive_samples_num)) - tmp_covered_positive_samples_index
    remain_uncovered_negative_samples_index = set(range(negative_samples_num)) - tmp_covered_negative_samples_index
    remain_uncovered_positive_samples_num = \
        len(remain_uncovered_positive_samples_index) * per_positive_sample_times
    remain_uncovered_negative_samples_num = \
        len(remain_uncovered_negative_samples_index) * per_negative_sample_times

    # 样本编码中还需加上剩余样本的编码长度
    # add the remain samples encoding length
    sample_encoding_length += get_data_encoding_length_by_amcl(remain_uncovered_positive_samples_num,
                                                               remain_uncovered_negative_samples_num)
    # 意图编码长度还需要加上子意图数量的编码长度
    intention_encoding_length += rissanen(len(sorted_sub_intentions) + 1)
    # 总编码长度
    total_encoding_length = intention_encoding_length + sample_encoding_length

    return total_encoding_length, init_total_encoding_length


# 寻找本轮反馈中与意图冲突的正负样本（问题样本）
# 意图冲突的正样本：本轮反馈中，未被意图覆盖的正样本集合
# 意图冲突的负样本：本轮反馈中，被意图覆盖的负样本集合
# param：
#      data: 初始化后的样本数据（对样本进行重新组织），来自samples.input.Data中Data类的封装
#      intention: 意图识别结果（list）
def get_conflict_samples_index(data, intention):
    samples = data.docs
    positive_samples = samples["positive"]
    negative_samples = samples["negative"]
    covered_positive_samples_index = set()
    covered_negative_samples_index = set()
    for sub_intention in intention:
        covered_positive_samples_index |= \
            get_sub_intention_covered_samples2(sub_intention, data.all_relevance_concepts_retrieved_docs,
                                               "positive")
        covered_negative_samples_index |= \
            get_sub_intention_covered_samples2(sub_intention, data.all_relevance_concepts_retrieved_docs,
                                               "negative")
    conflict_positive_samples_index = set(range(len(positive_samples))) - covered_positive_samples_index
    conflict_negative_samples_index = covered_negative_samples_index

    return conflict_positive_samples_index, conflict_negative_samples_index