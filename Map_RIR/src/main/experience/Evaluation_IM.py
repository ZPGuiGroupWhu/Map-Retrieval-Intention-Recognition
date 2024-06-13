from __future__ import division
# the evaluation index of the methods
from Map_RIR.src.main.samples.input.Data import Data
from Map_RIR.src.main.util.FileUtil import load_json
from Map_RIR.src.main.util.RetrievalUtil import retrieve_docs_by_complete_intention
import copy
from itertools import permutations

# the Jaccard Distance
from Map_RIR.src.main.samples.input import OntologyUtil, Sample


def get_jaccard_index(samples, real_intention, extracted_intention, ontologies, ontology_root):
    # retrieve the samples
    positive_samples = samples["relevance"]
    negative_samples = samples["irrelevance"]
    real_intention_retrieved_positive_samples_index = \
        retrieve_docs_by_complete_intention(real_intention, positive_samples, ontologies, ontology_root)
    real_intention_retrieved_negative_samples_index = \
        retrieve_docs_by_complete_intention(real_intention, negative_samples, ontologies, ontology_root)

    extracted_intention_retrieved_positive_samples_index = \
        retrieve_docs_by_complete_intention(extracted_intention, positive_samples, ontologies, ontology_root)
    extracted_intention_retrieved_negative_samples_index = \
        retrieve_docs_by_complete_intention(extracted_intention, negative_samples, ontologies, ontology_root)

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


# 这里的实现与MDL_RM不同，包括：只针对子意图；不考虑取值全为根节点的维度
def similarity_of_sub_intentions(sub_intention1, sub_intention2, direct_ancestors, ontology_root,
                                 concept_information_content_yuan2013):
    if sub_intention2 is None or sub_intention1 is None:
        return 0
    considered_dimension_num = 0
    sum_similarity = 0
    # 正向意图的相似度
    for tmp_dim in sub_intention1["positive"]:
        sub_intention1_tmp_dim_value = sub_intention1["positive"][tmp_dim]
        sub_intention2_tmp_dim_value = sub_intention2["positive"][tmp_dim]
        considered_dimension_num += 1
        _, tmp_similarity = OntologyUtil.get_similarity_Lin(sub_intention1_tmp_dim_value,
                                                            sub_intention2_tmp_dim_value,
                                                            direct_ancestors[tmp_dim], ontology_root[tmp_dim],
                                                            concept_information_content_yuan2013[tmp_dim])
        sum_similarity += tmp_similarity

    # 负向意图的相似度
    min_length_intention = sub_intention1["negative"] if len(sub_intention1["negative"]) <= len(
        sub_intention2["negative"]) else sub_intention2["negative"]
    max_length_intention = sub_intention1["negative"] if len(sub_intention1["negative"]) > len(
        sub_intention2["negative"]) else sub_intention2["negative"]

    # 得到所有映射, 固定一个子意图的负意图顺序不变，求另一个子意图所有的负意图排列情况
    # 固定min_length_intention, 排列max_length_intention，排列通过索引实现
    if (len(min_length_intention) == 0):
        considered_dimension_num += len(max_length_intention) * 0.5
    else:
        maps = permutations(range(len(max_length_intention)), len(min_length_intention))
        if len(max_length_intention) > 0:
            max_similarity = 0
            negative_considered_dimension_num = 0
            for tmp_max_length_intention_index in maps:
                similarity1 = 0
                tmp_considered_dimension_num = 0

                for i in range(len(tmp_max_length_intention_index)):
                    tmp_min_length_sub_intention_index = i
                    tmp_max_length_sub_intention_index = tmp_max_length_intention_index[i]
                    tmp_min_length_sub_intention = min_length_intention[tmp_min_length_sub_intention_index]
                    tmp_max_length_sub_intention = max_length_intention[tmp_max_length_sub_intention_index]
                    if tmp_min_length_sub_intention is None or tmp_max_length_sub_intention is None:
                        tmp_considered_dimension_num += max(len(tmp_min_length_sub_intention),
                                                            len(tmp_max_length_sub_intention))
                        continue
                    # 判断是否在相同维度
                    if tmp_min_length_sub_intention[0] == tmp_max_length_sub_intention[0]:
                        tmp_considered_dimension_num = tmp_considered_dimension_num + 1
                        # print("dim", tmp_min_length_sub_intention[0])
                        # print("tmp_min_length_sub_intention[1]",tmp_min_length_sub_intention[1])
                        # print("ttmp_max_length_sub_intention[1]", tmp_max_length_sub_intention[1])
                        _, tmp_similarity = OntologyUtil.get_similarity_Lin(tmp_min_length_sub_intention[1],
                                                                            tmp_max_length_sub_intention[1],
                                                                            direct_ancestors[
                                                                                tmp_min_length_sub_intention[0]],
                                                                            ontology_root[
                                                                                tmp_min_length_sub_intention[0]],
                                                                            concept_information_content_yuan2013[
                                                                                tmp_min_length_sub_intention[0]])

                        similarity1 += tmp_similarity
                # 平均相似度
                if max_similarity < similarity1:
                    max_similarity = similarity1
                    negative_considered_dimension_num = tmp_considered_dimension_num

            sum_similarity += max_similarity * 0.5
            # print("sum_similarity",max_similarity*0.5)

            considered_dimension_num += negative_considered_dimension_num * 0.5
            # print("considered_dimension_num", negative_considered_dimension_num*0.5)
    if considered_dimension_num == 0:
        return 0

    return sum_similarity / considered_dimension_num


# 计算补空意图后最佳映射平均相似度BMASS
#   1.将子意图数量较少的意图添加空子意图使两个意图子意图数量相等
#   2.得到两个意图间所有的一对一映射，定义所有具有映射关系的子意图的语义相似度之和取平均作为两个意图的相似度
#   3.将最大相似度作为两个意图的最终相似度
def get_intention_similarity(intention_a, intention_b, direct_ancestors, ontology_root,
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
            # print("tmp_sub_intention_similarity ", tmp_sub_intention_similarity)

            tmp_similarity += tmp_sub_intention_similarity
        tmp_similarity /= len(max_length_intention)  # 平均相似度
        if max_similarity < tmp_similarity:
            max_similarity = tmp_similarity
    return max_similarity  # 以最佳映射作为最终相似度结果


def get_precision(test_samples, real_intention, extracted_intention, ontologies, ontology_root):
    positive_samples = test_samples["relevance"]
    negative_samples = test_samples["irrelevance"]
    all_samples = positive_samples + negative_samples
    real_intention_retrieved_all_samples_index = \
        retrieve_docs_by_complete_intention(real_intention, all_samples, ontologies, ontology_root)
    extracted_intention_retrieved_all_samples_index = \
        retrieve_docs_by_complete_intention(extracted_intention, all_samples, ontologies, ontology_root)
    true_positive_num = \
        len(real_intention_retrieved_all_samples_index & extracted_intention_retrieved_all_samples_index)
    result = 0 if true_positive_num == 0 else true_positive_num / len(extracted_intention_retrieved_all_samples_index)
    return result


def get_recall(test_samples, real_intention, extracted_intention, ontologies, ontology_root):
    positive_samples = test_samples["relevance"]
    negative_samples = test_samples["irrelevance"]
    all_samples = positive_samples + negative_samples
    real_intention_retrieved_all_samples_index = \
        retrieve_docs_by_complete_intention(real_intention, all_samples, ontologies, ontology_root)
    extracted_intention_retrieved_all_samples_index = \
        retrieve_docs_by_complete_intention(extracted_intention, all_samples, ontologies, ontology_root)
    true_positive_num = \
        len(real_intention_retrieved_all_samples_index & extracted_intention_retrieved_all_samples_index)
    if len(real_intention_retrieved_all_samples_index) == 0:
        print(real_intention)
    else:
        result = true_positive_num / len(real_intention_retrieved_all_samples_index)

    return result


def get_F1_score(test_samples, real_intention, extracted_intention, ontologies, ontology_root):
    precision = get_precision(test_samples, real_intention, extracted_intention, ontologies, ontology_root)
    recall = get_recall(test_samples, real_intention, extracted_intention, ontologies, ontology_root)
    result = 2 * precision * recall / (recall + precision)
    return result


def get_real_intent():
    # real_intents = {}
    # json_intent = load_json("../../../../MDL_RM/resources/samples/Intention7.30.json")
    # # real_intents = copy.deepcopy(json_intent)
    # for key in json_intent:
    #     real_intents[key] = {}
    #     real_intents[key]["positive"] = []
    #     dim_trans = {"S": "Spatial", "C": "MapContent", "M": "MapMethod", "T": "Theme"}
    #     for sub_positive_intention in json_intent[key]["positive"]:
    #         real_intents[key]["positive"].append(
    #             {"Spatial": sub_positive_intention["S"] if "S" in sub_positive_intention else sub_positive_intention[
    #                 "Spatial"],
    #              "MapContent": sub_positive_intention["C"] if "C" in sub_positive_intention else sub_positive_intention[
    #                  "MapContent"],
    #              "MapMethod": sub_positive_intention["M"] if "M" in sub_positive_intention else sub_positive_intention[
    #                  "MapMethod"],
    #              "Theme": sub_positive_intention["T"] if "T" in sub_positive_intention else sub_positive_intention[
    #                  "Theme"]})
    #     real_intents[key]["negative"] = []
    #     if json_intent[key]["negative"] is not None:
    #         for negative_intent in json_intent[key]["negative"]:
    #             real_intents[key]["negative"].append(
    #                 {(dim_trans[list(negative_intent.keys())[0]], list(negative_intent.values())[0])})
    #     print(real_intents[key])
    real_intents = {}
    json_intent = load_json("../../../../MDL_RM/resources/samples/Intention7.30.json")
    # real_intents = copy.deepcopy(json_intent)
    for key in json_intent:
        real_intents[key] = []
        dim_trans = {"S": "Spatial", "C": "MapContent", "M": "MapMethod", "T": "Theme"}
        for sub_intention in json_intent[key]:
            real_sub_positive = {"Spatial": sub_intention["positive"]["S"] if "S" in sub_intention["positive"] else
            sub_intention["positive"]["Spatial"],
                                 "MapContent": sub_intention["positive"]["C"] if "C" in sub_intention["positive"] else
                                 sub_intention["positive"][
                                     "MapContent"],
                                 "MapMethod": sub_intention["positive"]["M"] if "M" in sub_intention["positive"] else
                                 sub_intention["positive"][
                                     "MapMethod"],
                                 "Theme": sub_intention["positive"]["T"] if "T" in sub_intention["positive"] else
                                 sub_intention["positive"]["Theme"]}
            real_sub_intent = {"positive": real_sub_positive,
                               "negative": []}
            if sub_intention["negative"] is not None:
                for negative_intent in sub_intention["negative"]:
                    real_sub_intent["negative"].append(
                        (dim_trans[list(negative_intent.keys())[0]], list(negative_intent.values())[0]))
            real_intents[key].append(real_sub_intent)
        # print(real_intents[key])
    return real_intents


# 无负向意图转换成有负向意图格式
def trans_intent(intent):
    new_intent = []
    for tmp_intent in intent:
        sub_intent = {"positive": tmp_intent, "negative": []}
        new_intent.append(sub_intent)
    return new_intent


if __name__ == "__main__":
    real_intentions = get_real_intent()
    tmp_sample_path = "D:\\xxj\\porject\\Map-Retrieval-Intention-Recognition-master\\" \
                      "Map-Retrieval-Intention-Recognition-master\\MDL_RM\\resources\samples" \
                      "\\scenes_negative\\Scene501\\noise_samples_S0p3_L0p4.json"
    tmp_sample_name = "Scene501"
    docs, real_intention = Sample.load_real_intents(real_intentions, tmp_sample_path, tmp_sample_name)
    data = Data(docs, real_intention)
    ancestors = Data.Ancestor
    ontologies = Data.Ontologies
    ontology_root = Data.Ontology_Root
    direct_ancestors = Data.direct_Ancestor
    information_content = data.concept_information_content
    sub_intention1 = {
          "positive": {
            "MapContent": "Thing",
            "Spatial": "America",
            "Theme": "ThemeRoot",
            "MapMethod": "MapMethodRoot"
          },
          "negative": []
        }
    sub_intention2 = {"positive": {'Spatial': 'United States', 'Theme': 'ThemeRoot', 'MapMethod': 'MapMethodRoot',
                                   'MapContent': 'Thing'}, "negative": [('Theme', 'ThemeRoot')]}
    similarity_of_sub_intentions(sub_intention1, sub_intention2, direct_ancestors, ontology_root,
                                 information_content)
    print(similarity_of_sub_intentions(sub_intention1, sub_intention2, direct_ancestors, ontology_root,
                                 information_content))
    intention1 = [{
        "positive": {
            "MapContent": "Thing",
            "Spatial": "America",
            "Theme": "ThemeRoot",
            "MapMethod": "MapMethodRoot"
        },
        "negative": []
    },{"positive": {'Spatial': 'United States', 'Theme': 'ThemeRoot', 'MapMethod': 'MapMethodRoot',
                                   'MapContent': 'Thing'}, "negative": [('Theme', 'ThemeRoot')]}]
    intention2 = [{"positive": {'Spatial': 'United States', 'Theme': 'ThemeRoot', 'MapMethod': 'MapMethodRoot',
                                   'MapContent': 'Thing'}, "negative": [('Theme', 'ThemeRoot')]},{
        "positive": {
            "MapContent": "Thing",
            "Spatial": "America",
            "Theme": "ThemeRoot",
            "MapMethod": "MapMethodRoot"
        },
        "negative": []
    }]
    print(get_intention_similarity(intention1, intention2, direct_ancestors, ontology_root,
                                       information_content))


