# coding:utf-8
# Document the details of the iterative process
import time
import random
from math import log2, gamma
from Map_RIR.src.main.intention_recognition import FrequentItemsets, MDL_RM, Config
import copy

from Map_RIR.src.main.intention_recognition.MDL_RM_NE import rules_to_intention
from Map_RIR.src.main.samples.input import Sample
from Map_RIR.src.main.samples.input.Data import Data
from Map_RIR.src.main.util import FileUtil, RetrievalUtil
time_use_sample_enhancement = 0
time_use_merge = 0
time_use_calculate_merge_statistic = 0
time_use_update_rule = 0
time_use_retrieve_docs = 0
time_use_get_max_similarity_value_pair = 0
time_use_get_LCA = 0
time_use_get_similarity_Lin = 0
time_others = 0

def init_for_intention(samples):
    origin_positive_samples = samples["relevance"]
    origin_negative_samples = samples["irrelevance"]

    positive_samples = copy.deepcopy(origin_positive_samples)
    negative_samples = copy.deepcopy(origin_negative_samples)
    init_intention_encoding_length = MDL_RM.rissanen(1)
    init_sample_encoding_length = MDL_RM.get_data_encoding_length(len(positive_samples),
                                                                      len(negative_samples),
                                                                      data_encoding_method)

    init_total_encoding_length = init_intention_encoding_length + init_sample_encoding_length
    init_encoding_length = [init_total_encoding_length, init_intention_encoding_length,
                            init_sample_encoding_length]
    uncovered_negative_samples_id = set(range(len(negative_samples)))
    uncovered_positive_samples_id = set(range(len(positive_samples)))
    return positive_samples, negative_samples, uncovered_positive_samples_id, uncovered_negative_samples_id, init_encoding_length


def get_rules_encoding_length_without_order_constraint(data, rules, positive_samples, negative_samples,
                                                       per_positive_sample_times, per_negative_sample_times):
    #     global time_use_retrieve_docs
    #
    #
    #     time01 = time.time()
    #
    #
    #     time02 = time.time()
    #     time_use_retrieve_docs += time02 - time01
    #
    rules.sort(key=lambda x: x[-1], reverse=False)
    # get every rule covered samples id and calculate total encoding length
    tmp_intention_encoding_length = 0
    tmp_sample_encoding_length = 0

    tmp_covered_positive_samples_id = set()
    tmp_covered_negative_samples_id = set()
    new_rules_list = []

    for tmp_rule in rules:
        #print("didi")
        tmp_rule_covered_positive_samples_id = tmp_rule[1]
        tmp_rule_covered_negative_samples_id = tmp_rule[2]
        tmp_rule_covered_samples_num = \
            len(tmp_rule_covered_positive_samples_id) * per_positive_sample_times \
            + len(tmp_rule_covered_negative_samples_id) * per_negative_sample_times
        tmp_rule_real_covered_positive_samples_id = \
            tmp_rule_covered_positive_samples_id - tmp_covered_positive_samples_id
        tmp_rule_real_covered_negative_samples_id = \
            tmp_rule_covered_negative_samples_id - tmp_covered_negative_samples_id

        tmp_rule_real_covered_positive_samples_num = \
            len(tmp_rule_real_covered_positive_samples_id) * per_positive_sample_times
        tmp_rule_real_covered_negative_samples_num = \
            len(tmp_rule_real_covered_negative_samples_id) * per_negative_sample_times
        tmp_rule_real_covered_samples_num = \
            tmp_rule_real_covered_positive_samples_num + tmp_rule_real_covered_negative_samples_num
        # 对多个意图进行筛选
        if tmp_rule_real_covered_positive_samples_num < tmp_rule_real_covered_negative_samples_num:
            continue
        if tmp_rule_real_covered_positive_samples_num == 0:
            continue
        # add the sample encoding length covered by sub intention
        tmp_sample_encoding_length += tmp_rule_real_covered_samples_num * tmp_rule[4]
        # add sub intention encoding length
        tmp_intention_encoding_length += log2(
            tmp_rule_covered_samples_num) + MDL_RM.get_sub_intention_encoding_length()

        # print("tmp_rule[0].positivee_sub_Intention", tmp_rule[0].positive_sub_Intention)
        # print("tmp_rule[0].negative_sub_Intention", tmp_rule[0].negative_sub_Intention)
        # print("tmp_rule[0].positive_sample", tmp_rule[1])
        # print("tmp_rule[0].negative_sample", tmp_rule[2])

        if tmp_rule[0].negative_sub_Intention:
            tmp_intention_encoding_length += len(tmp_rule[0].negative_sub_Intention) * \
                                             (MDL_RM.get_sub_negative_encoding_length() + log2(
                                                 4))+MDL_RM.rissanen(len(tmp_rule[0].negative_sub_Intention)+1)

        tmp_covered_positive_samples_id |= tmp_rule_covered_positive_samples_id
        tmp_covered_negative_samples_id |= tmp_rule_covered_negative_samples_id
        new_rules_list.append(tmp_rule[0])
    tmp_remain_uncovered_positive_samples_id = set(range(len(positive_samples))) - tmp_covered_positive_samples_id
    tmp_remain_uncovered_negative_samples_id = set(range(len(negative_samples))) - tmp_covered_negative_samples_id
    tmp_remain_uncovered_positive_samples_num = \
        len(tmp_remain_uncovered_positive_samples_id) * per_positive_sample_times
    tmp_remain_uncovered_negative_samples_num = \
        len(tmp_remain_uncovered_negative_samples_id) * per_negative_sample_times
    # add the remain samples encoding length
    tmp_sample_encoding_length += MDL_RM.get_data_encoding_length_by_amcl(tmp_remain_uncovered_positive_samples_num,
                                                                              tmp_remain_uncovered_negative_samples_num)

    tmp_intention_encoding_length += MDL_RM.rissanen(len(new_rules_list) + 1)
    tmp_total_encoding_length = tmp_sample_encoding_length + tmp_intention_encoding_length
    return (
        tmp_remain_uncovered_positive_samples_id,
        tmp_remain_uncovered_negative_samples_id,
        new_rules_list,
        [tmp_total_encoding_length, tmp_intention_encoding_length, tmp_sample_encoding_length],
    )


# def is_intention_cover(intent_a, intent_b, ancestors, ontology_root):
#     dims_result = []
#     for tmp_dim in intent_a:
#         intent_a_value = intent_a[tmp_dim]
#         intent_b_value = intent_b[tmp_dim]
#         if intent_a_value == intent_b_value:
#             dims_result.append("e")
#         elif intent_a_value == ontology_root[tmp_dim] or \
#                 (ancestors[tmp_dim] is not None and intent_a_value in ancestors[tmp_dim][intent_b_value]):
#             dims_result.append("a")
#     if len(dims_result) >= len(intent_a.keys()):
#         return True
#     else:
#         return False
# tmp_dim == tmp_negative_intention[0]:
#     tmp_value = tmp_negative_intention[1]
# 判断一个负向意图是否包含另一个意图
def is_negative_cover(intent_a, intent_b, ancestors, ontology_root):
    dims_result = []
    is_cover = False
    num = len(intent_a)

    for sub_intent_a in intent_a:
        for sub_intent_b in intent_b:
            if intent_a[0] == intent_b[0]:
                intent_a_value = sub_intent_a[1]
                intent_b_value = sub_intent_b[1]
                print("key: ", sub_intent_a[0])
                print("value: ", sub_intent_a[1])
                if ancestors[intent_b[0]] is not None and intent_a_value in ancestors[intent_b[0]][intent_b_value]:
                    num -= 1
                    break
    if num == 0:
        is_cover = True
    return is_cover


def init_intention(rules_num, candidate_sub_intentions_list):
    intentions = []
    first_sub_intent = random.choice(candidate_sub_intentions_list)
    intentions.append(first_sub_intent)
    class_num = 1
    n = 1
    while class_num < rules_num and n < 30:
        n += 1
        temp_sub_intent = random.choice(candidate_sub_intentions_list)
        for sub_intent in intentions:
            if not MDL_RM.is_intention_cover(sub_intent.positive_sub_Intention,
                                                 temp_sub_intent.positive_sub_Intention, Data.Ancestor,
                                                 Data.Ontology_Root) \
                    and not MDL_RM.is_intention_cover(temp_sub_intent.positive_sub_Intention,
                                                          sub_intent.positive_sub_Intention, Data.Ancestor,
                                                          Data.Ontology_Root):
                intentions.append(temp_sub_intent)
                class_num += 1

    return intentions


def get_intention(samples, candidate_sub_intentions_list, random_merge_number,
                  beam_width, rule_covered_positive_sample_rate_threshold):
    global time_use_sample_enhancement, time_use_init, time_use_calculate_merge_statistic, \
        time_use_update_rule, similarity_Lin_cache

    time00 = time.time()

    # 初始化
    data, positive_samples, negative_samples, per_positive_sample_times, \
    per_negative_sample_times, uncovered_positive_samples_id, \
    uncovered_negative_samples_id, encoding_length = MDL_RM.init_for_intention_extraction(samples, "amcl")
    min_encoding_length = encoding_length[0]
    init_min_encoding_length = copy.deepcopy(min_encoding_length)
    time01 = time.time()
    time_use_sample_enhancement += time01 - time00

    # 初始化意图列表，添加集束宽度个空规则
    #   rules == intention， rule == sub_intention
    rules_list = []
    #random_merge_number = 200
    iteration_number = 0
    iteration_number_update = 1
    # 初始化临时意图列表

    tmp_rules = Rules(copy.deepcopy(uncovered_positive_samples_id), copy.deepcopy(uncovered_negative_samples_id),
                      [], encoding_length)

    data = Data(samples)
    rules_to_intention([])[0]
    posi=rules_to_intention([])[0]
    init_intent = [SubIntention(rules_to_intention([])[0],[])]
    process_log = [{
        "iteration": 0,
        "total_encoding_length": encoding_length[0],
        "intention_encoding_length": encoding_length[1],
        "sample_encoding_length": encoding_length[2],
        "encoding_length_compression_rates": 1.0,
        "intention_result": init_intent
    }]
    while True:
        iteration_number +=1
        random_merge_number = random_merge_number - 1
        if random_merge_number == 0:
            #print("最短：", tmp_rules.total_encoding_length, tmp_rules.intention_encoding_length,
            #      tmp_rules.sample_encoding_length)
            time07 = time.time()
            #print("time_use_calculate_merge_statistic",time_use_calculate_merge_statistic)
            return tmp_rules,time07-time00,process_log
        rules_num = beam_width
        time02 = time.time()
        # new_rules = random.sample(candidate_sub_intentions_list, beam_width)
        new_rules = init_intention(rules_num, candidate_sub_intentions_list)
        # rules_num = random.randint(1, beam_width)
        #
        #
        # new_rules = random.sample(candidate_sub_intentions_list, rules_num)

        time03 = time.time()
        #print("rules_num", len(new_rules))
        sorted_rules = []
        time04 = time.time()
        for tmp_rule in new_rules:
            tmp_rule_covered_positive_samples_id = \
                RetrievalUtil.retrieve_docs_based_on_terms_covered_samples2(tmp_rule.positive_sub_Intention,
                                                                            tmp_rule.negative_sub_Intention,
                                                                            data.all_relevance_concepts_retrieved_docs,
                                                                            "positive")
            tmp_rule_covered_negative_samples_id = \
                RetrievalUtil.retrieve_docs_based_on_terms_covered_samples2(tmp_rule.positive_sub_Intention,
                                                                            tmp_rule.negative_sub_Intention,
                                                                            data.all_relevance_concepts_retrieved_docs,
                                                                            "negative")

            tmp_rule_covered_positive_negative_num = [len(tmp_rule_covered_positive_samples_id),
                                                      len(tmp_rule_covered_negative_samples_id)]
            # get the average minimum encoding length based on Shannon encoding method
            average_encoding_length = MDL_RM.get_average_encoding_length(
                len(tmp_rule_covered_positive_samples_id) * per_positive_sample_times,
                len(tmp_rule_covered_negative_samples_id) * per_negative_sample_times)
            sorted_rule = [tmp_rule,
                           tmp_rule_covered_positive_samples_id,
                           tmp_rule_covered_negative_samples_id,
                           tmp_rule_covered_positive_negative_num,
                           average_encoding_length]
            sorted_rules.append(sorted_rule)
        time05 = time.time()
        tmp_merged_rule_statistics = get_rules_encoding_length_without_order_constraint \
            (data, sorted_rules, positive_samples, negative_samples, per_positive_sample_times,
             per_negative_sample_times)

        tmp_encoding_length = tmp_merged_rule_statistics[-1]
        tmp_total_encoding_length = tmp_encoding_length[0]
        #print(tmp_encoding_length)
        # remain_uncovered_positive_samples_id
        new_rules_uncovered_positive_samples_id = tmp_merged_rule_statistics[-4]
        # remain_uncovered_negative_samples_id
        new_rules_uncovered_negative_samples_id = tmp_merged_rule_statistics[-3]
        new_rules = tmp_merged_rule_statistics[-2]#########
        time06 = time.time()
        time_use_calculate_merge_statistic += time06-time05
        isRenew = False
        isRenew2 = False

        if tmp_total_encoding_length < tmp_rules.total_encoding_length:
            isRenew = True
            new_rules1 = Rules(new_rules_uncovered_positive_samples_id,
                               new_rules_uncovered_negative_samples_id,
                               new_rules, tmp_encoding_length)
            tmp_rules = new_rules1
        elif tmp_total_encoding_length == tmp_rules.total_encoding_length:
            for temp_sub_intent in tmp_rules.intention:
                for new_sub_intent in new_rules:
                    # 判断包含关系
                    if MDL_RM.is_intention_cover(temp_sub_intent.positive_sub_Intention,
                                                     new_sub_intent.positive_sub_Intention, Data.Ancestor,
                                                     Data.Ontology_Root):
                        if temp_sub_intent.positive_sub_Intention != new_sub_intent.positive_sub_Intention:
                            isRenew2 = True
                        temp_sub_intent.positive_sub_Intention = new_sub_intent.positive_sub_Intention

                    if (new_sub_intent.negative_sub_Intention is not None) and (new_sub_intent.negative_sub_Intention is not None) \
                            and new_sub_intent.negative_sub_Intention != new_sub_intent.negative_sub_Intention:

                        if is_negative_cover(temp_sub_intent.negative_sub_Intention,
                                                         new_sub_intent.negative_sub_Intention, Data.Ancestor,
                                                         Data.Ontology_Root):
                            if temp_sub_intent.negative_sub_Intention != new_sub_intent.negative_sub_Intention:
                                isRenew2 = True
                            temp_sub_intent.negative_sub_Intention = new_sub_intent.negative_sub_Intention
        # 输出迭代过程参数
        if Config.TAG_RECORD_PROCESS and isRenew:
            # for rules_index, tmp_rules in enumerate(tmp_rules_list):
            #     intention_result = rules_to_intention(tmp_rules.rules)
            #     covered_positive_sample_rates = []
            #     for tmp_rule_id in tmp_rules.rules:
            #         tmp_rule = tmp_rules.rules[tmp_rule_id]
            #         tmp_rule_covered_positive_sample_rate = len(tmp_rule[1]) / len(positive_samples)
            #         covered_positive_sample_rates.append(tmp_rule_covered_positive_sample_rate)
            tmp_rules_log = {
                "iteration": iteration_number_update,
                "total_encoding_length": tmp_rules.total_encoding_length,
                "intention_encoding_length": tmp_rules.intention_encoding_length,
                "sample_encoding_length": tmp_rules.sample_encoding_length,
                "encoding_length_compression_rates": tmp_rules.total_encoding_length / init_min_encoding_length,
                "intention_result": tmp_rules.intention
            }
            if isRenew2:
                process_log.pop()

            process_log.append(tmp_rules_log)
            iteration_number_update += 1


# positive_sub_Intention:{'MapContent': 'Thing', 'Theme': 'ThemeRoot', 'MapMethod': 'MapMethodRoot', 'Spatial': 'North America'}
# negative_sub_Intention：[('Spatial', 'United States')]
class SubIntention:
    def __init__(self, positive_sub_Intention, negative_sub_Intention):
        self.positive_sub_Intention = positive_sub_Intention
        self.negative_sub_Intention = negative_sub_Intention


class Rules:
    def __init__(self, uncovered_positive_samples_id, uncovered_negative_samples_id, intention, encoding_length):
        self.uncovered_positive_samples_id = uncovered_positive_samples_id
        self.uncovered_negative_samples_id = uncovered_negative_samples_id
        self.intention = intention
        self.total_encoding_length = encoding_length[0]
        self.method_log = []
        self.intention_encoding_length = encoding_length[1]
        self.sample_encoding_length = encoding_length[2]


# 返回值：SubIntention的list
def get_all_candidate_sub_intentions(samples, min_support):
    candidate_items = {
        'relevance': FrequentItemsets.get_all_relevance_candidate_sub_intentions(samples, min_support),
        'irrelevance': FrequentItemsets.get_all_irrelevance_candidate_sub_intentions(samples,
                                                                                     min_support)}
    data = Data(samples)
    candidate_sub_intentions_list = []
    for sub_relevance_intention in candidate_items['relevance']:
        # tmp_sub_intention = SubIntention(sub_relevance_intention, {})
        # candidate_sub_intentions_list.append(tmp_sub_intention)

        # tmp_item = (tmp_dim, tmp_concept)
        negative_labels_list = []
        # 标签有四个维度
        # negative_labels = {
        #     'Spatial': [],
        #     'Theme': [],
        #     'MapMethod': [],
        #     'MapContent': []
        # }
        for tmp_negative_intention in candidate_items['irrelevance']:
            paste_dim = RetrievalUtil.is_negative_paste(sub_relevance_intention, tmp_negative_intention, data.Ancestor,
                                                        data.Ontology_Root)
            if paste_dim == 0:
                continue
            else:
                tmp_item = (paste_dim, tmp_negative_intention[paste_dim])
                if tmp_item not in negative_labels_list:
                    negative_labels_list.append(tmp_item)
        # print(negative_labels_list)
        lables_list = [[]]
        for i in range(len(negative_labels_list)):  # 定长
            for j in range(len(lables_list)):  # 变长
                # 如果包含具有包含关系的项就排除
                is_add = True
                for lable in lables_list[j]:
                    if RetrievalUtil.is_inclusive(lable, negative_labels_list[i], data.Ancestor, data.Ontology_Root):
                        is_add = False
                if is_add:
                    lables_list.append(lables_list[j] + [negative_labels_list[i]])
        for labels in lables_list:
            # print(sub_relevance_intention)
            # print(labels)
            candidate_sub_intentions_list.append(SubIntention(sub_relevance_intention, labels))
    #print("num of candidate_sub_intentions_list:", len(candidate_sub_intentions_list))
    return candidate_sub_intentions_list


def get_all_candidate_sub_intentions_P(samples, min_support):
    candidate_items = {
        'relevance': FrequentItemsets.get_all_relevance_candidate_sub_intentions(samples, min_support),
        'irrelevance':[]}
    data = Data(samples)
    candidate_sub_intentions_list = []
    for sub_relevance_intention in candidate_items['relevance']:
        # tmp_sub_intention = SubIntention(sub_relevance_intention, {})
        # candidate_sub_intentions_list.append(tmp_sub_intention)

        # tmp_item = (tmp_dim, tmp_concept)
        negative_labels_list = []
        # 标签有四个维度
        # negative_labels = {
        #     'Spatial': [],
        #     'Theme': [],
        #     'MapMethod': [],
        #     'MapContent': []
        # }
        for tmp_negative_intention in candidate_items['irrelevance']:
            paste_dim = RetrievalUtil.is_negative_paste(sub_relevance_intention, tmp_negative_intention, data.Ancestor,
                                                        data.Ontology_Root)
            if paste_dim == 0:
                continue
            else:
                tmp_item = (paste_dim, tmp_negative_intention[paste_dim])
                if tmp_item not in negative_labels_list:
                    negative_labels_list.append(tmp_item)
        # print(negative_labels_list)
        lables_list = [[]]
        for i in range(len(negative_labels_list)):  # 定长
            for j in range(len(lables_list)):  # 变长
                # 如果包含具有包含关系的项就排除
                is_add = True
                for lable in lables_list[j]:
                    if RetrievalUtil.is_inclusive(lable, negative_labels_list[i], data.Ancestor, data.Ontology_Root):
                        is_add = False
                if is_add:
                    lables_list.append(lables_list[j] + [negative_labels_list[i]])
        for labels in lables_list:
            # print(sub_relevance_intention)
            # print(labels)
            candidate_sub_intentions_list.append(SubIntention(sub_relevance_intention, labels))
    #print("num of candidate_sub_intentions_list:", len(candidate_sub_intentions_list))
    return candidate_sub_intentions_list


# print(candidate_sub_intentions)

def run_IM(samples, min_support, random_merge_number):
    time1 = time.time()
    candidate_sub_intentions_list = get_all_candidate_sub_intentions(samples, min_support)
    time2 = time.time()
    # print("get_all_candidate_time",time2-time1)
    result, time_all, process_log= get_intention(samples, candidate_sub_intentions_list, random_merge_number, 2, 0.3)
    if not result.intention:
        positive_intent = {'MapContent': 'Thing', 'Spatial': 'America', 'Theme': 'ThemeRoot', 'MapMethod': 'MapMethodRoot'}
        result.intention.append(SubIntention(positive_intent,[]))
    time3 = time.time()
    print("get_all_candidate_sub_intentions", time2 - time1)
    print("get_intention", time3 - time2)
    return result, time_all,process_log

def run_IM_P(samples, min_support, random_merge_number):
    time1 = time.time()
    candidate_sub_intentions_list = get_all_candidate_sub_intentions_P(samples, min_support)
    time2 = time.time()
    # print("get_all_candidate_time",time2-time1)
    result, time_all, process_log = get_intention(samples, candidate_sub_intentions_list, random_merge_number, 2, 0.3)
    if not result.intention:
        positive_intent = {'MapContent': 'Thing', 'Spatial': 'America', 'Theme': 'ThemeRoot', 'MapMethod': 'MapMethodRoot'}
        result.intention.append(SubIntention(positive_intent,[]))
    time3 = time.time()
    # print("get_intention", time3 - time2)
    return result, time_all




if __name__ == "__main__":
    scene = "581"
    sample_version = "scenes_negative"
    test_sample_path = "./../../../resources/samples/" + sample_version + "/Scene" + scene + "/final_samples.json"
    samples = FileUtil.load_json(test_sample_path)  # 加载样本文件
    samples = Sample.transform_sample(samples)  # 转换样本文件
    min_support = 0.3
    # candidate_sub_intentions_list = FrequentItemsets.get_all_candidate_sub_intentions(samples, min_support)
    # candidate_sub_intentions_list = get_all_candidate_sub_intentions(samples, min_support)
    # # print(candidate_sub_intentions_list)
    data_encoding_method = "amcl"
    time1 = time.time()
    # result, time = get_intention(samples, candidate_sub_intentions_list, 5000, 2, 0.3)
    result, time_IM, process_log = run_IM(samples, min_support, 2000)
    time2 = time.time()
    print("timeall",process_log)
    method_result1 = MDL_RM \
        .get_intention_by_method6(samples, data_encoding_method, 50, 1, 0.3)
    print(result.total_encoding_length)
    print("time:",time_IM)
    print("sample positive:", result.uncovered_positive_samples_id)
    print("sample negative:", result.uncovered_negative_samples_id)
    print("sample negative num:", len(result.uncovered_negative_samples_id))
    print(process_log)
    for subIntention in result.intention:
        print("P：", subIntention.positive_sub_Intention)
        print("N：", subIntention.negative_sub_Intention)
    print(method_result1)
