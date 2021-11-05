from math import log2, gamma
import time

from MDL_RM.src.main.experience import EvaluationIndex
from MDL_RM.src.main.intention_recognition import Config, MDL_RM
from MDL_RM.src.main.samples.input import Sample, Data


def _test_get_encoding_length_by_ppc():
    value1 = -log2(gamma(10 + 1) * gamma(8 + 1) * gamma(7) / gamma(10 + 8 + 7))
    value2 = -log2(gamma(13 + 1) * gamma(1 + 1) * gamma(7) / gamma(13 + 1 + 7))
    value3 = -log2(gamma(20 + 1) * gamma(7) / gamma(20 + 7))
    value4 = -log2(gamma(4 + 1) * gamma(4 + 1) * gamma(7) / gamma(4 + 4 + 7))
    value5 = -log2(gamma(41 + 1) * gamma(7) / gamma(41 + 7))
    print(value1 + value2 + value3 + value4 + value5)


def _test_get_intention_by_method1():
    scene = "11"
    test_sample_path = "./../../../resources/samples/scenes_v4_5/Scene" + scene + "/noise_samples_S0p1_L1.json"
    Sample.load_sample(test_sample_path)
    Data.init(Sample)
    # intention = Sample.real_intention
    test_samples = Data.docs
    data_encoding_method = "amcl"
    for i in range(10):
        method_result = MDL_RM.get_intention_by_method1(test_samples, data_encoding_method, 0.3)
        predict_intention = MDL_RM.result_to_intention(method_result)
        print(predict_intention, method_result[1])


def _test_get_intention_by_method6():
    MDL_RM.init_time_use()
    scene = "362"
    sample_version = "scenes_v4_5"
    test_sample_path = "./../../../resources/samples/" + sample_version + "/Scene" + scene + "/noise_samples_L1.json"
    Sample.load_sample(test_sample_path)
    Data.init(Sample)
    real_intention = Sample.real_intention
    test_ontologies = Data.Ontologies
    test_ontology_root = Data.Ontology_Root
    test_direct_ancestors = Data.direct_Ancestor
    test_information_content = Sample.concept_information_content
    test_samples = Data.docs
    data_encoding_method = "amcl"
    Config.adjust_sample_num = True
    Config.TAG_RECORD_MERGE_PROCESS = True
    time01 = time.time()
    method_result = None
    predict_intention = None
    best_num = 0
    for i in range(1):
        method_result = MDL_RM.get_intention_by_method6(test_samples, data_encoding_method, 200, 1, 0.3)
        predict_intention = MDL_RM.result_to_intention(method_result)
        for sub_intention in predict_intention:
            print(sub_intention)
        print(method_result[1])
        if method_result[1] == 181.0278169704525:
            best_num += 1
    print("best_num", best_num)
    time02 = time.time()
    print("time_use", time02 - time01)
    print("time_use_init", MDL_RM.time_use_sample_enhancement)
    print("time_use_merge", MDL_RM.time_use_merge)
    print("\ttime_get_max_similarity_value_pair", MDL_RM.time_use_calculate_merge_statistic)
    print("\t\ttime_get_similarity_Lin", MDL_RM.time_use_get_similarity_Lin)
    print("\ttime_get_LCA", MDL_RM.time_use_get_LCA)
    print("time_use_calculate_merge_statistic", MDL_RM.time_use_calculate_merge_statistic)
    print("time_update_rule", MDL_RM.time_use_update_rule)
    print("time_retrieve_docs", MDL_RM.time_use_retrieve_docs)

    jaccard_score = EvaluationIndex.get_jaccard_index(test_samples, real_intention,
                                                      predict_intention, test_ontologies, test_ontology_root)
    intention_similarity = EvaluationIndex.get_intention_similarity(real_intention, predict_intention,
                                                                    test_direct_ancestors, test_ontology_root,
                                                                    test_information_content)
    print(jaccard_score)
    print(intention_similarity)

    method_log = method_result[-1]
    time_use_log = method_log["time_use"]
    merge_process_log = method_log["merge_process"]
    print("time_use", time_use_log)
    print("merge_process")
    for i, tmp_iteration_log in enumerate(merge_process_log):
        print("\t iteration", i)
        for tmp_rules_log in tmp_iteration_log:
            print("\t\t", tmp_rules_log)


def _test_init_for_intention_extraction():
    scene = "362"
    sample_version = "scenes_v4_5"
    test_sample_path = "./../../../resources/samples/" + sample_version + "/Scene" + scene + "/noise_samples_L1.json"
    Sample.load_sample(test_sample_path)
    Data.init(Sample)
    test_samples = Data.docs
    test_origin_positive_samples = test_samples["relevance"]
    test_origin_negative_samples = test_samples["irrelevance"]
    print(f"before balancing: \n"
          f"\torigin_positive_samples_num: {len(test_origin_positive_samples)}\n"
          f"\torigin_negative_samples_num: {len(test_origin_negative_samples)}\n\n")
    positive_samples, negative_samples, positive_samples_id_num_dict, \
    negative_samples_id_num_dict, uncovered_positive_samples_id, \
    uncovered_negative_samples_id, min_encoding_length = MDL_RM.init_for_intention_extraction(test_samples, "amcl")
    print(f"after balancing: \n"
          f"\torigin_positive_samples_num: {len(positive_samples)}\n"
          f"\torigin_negative_samples_num: {len(negative_samples)}\n"
          f"\tpositive_samples_id_num_dict: {positive_samples_id_num_dict}\n"
          f"\tnegative_samples_id_num_dict: {negative_samples_id_num_dict}\n")


if __name__ == "__main__":
    time0 = time.time()
    # _test_get_intention_by_method1()
    _test_get_intention_by_method6()
    # _test_init_for_intention_extraction()
    # print(MDL_RM.get_sub_intention_encoding_length())
    time1 = time.time()
    print("total time use", time1 - time0)
    print("Aye")
