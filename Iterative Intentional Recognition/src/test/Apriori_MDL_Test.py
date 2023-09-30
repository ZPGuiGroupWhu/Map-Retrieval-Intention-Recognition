import time

from src.main.intention_recognition import Config, Apriori_MDL, EvaluationIndex
from src.main.samples.input import Sample
from src.main.samples.input.Data import Data


def _test_get_all_candidate_sub_intentions():
    scene = "17"
    sample_version = "scenes_v4_7"
    sample_path = "../../resources/samples/" + sample_version + "/Scene" + scene + "/samples_F0p1_L0p2.json"
    samples, real_intention = Sample.load_sample_from_file(sample_path)
    config = Config.Config()
    method_result = Apriori_MDL.get_intentions_by_greedy_search(samples, config)

    # # print(samples)
    # time0 = time.time()
    # # expended_samples = Sample.add_samples_relation_information(samples)
    # # print(expended_samples)
    # data = Data2(samples)
    # config = Config2.Config()
    # config.beam_width = 1
    # min_support = 0.3
    # k_max = len(Data2.Dimensions)
    #
    # candidate_sub_intentions = Apriori_MDL.get_all_candidate_sub_intentions(data, min_support, k_max)
    # print(time.time() - time0)
    # print(len(candidate_sub_intentions))
    # # print(candidate_sub_intentions)
    #
    # predict_intention, total_coding_length, init_coding_length, method_log = \
    #     Apriori_MDL.get_intention_by_method6(data, config, candidate_sub_intentions)
    # print(predict_intention)
    # print(total_coding_length)
    # print(init_coding_length)
    # time1 = time.time()
    # print(time1 - time0)

    predict_intention = method_result[0]
    method_log = method_result[-1]
    tmp_encoding_length_compression_rates = method_result[1] / method_result[2]
    jaccard_score = EvaluationIndex.get_jaccard_index(samples, real_intention,
                                                      predict_intention, Data.Ontologies)
    intention_similarity = EvaluationIndex.get_intention_similarity(real_intention, predict_intention,
                                                                    Data.direct_Ancestor,
                                                                    Data.Ontology_Root,
                                                                    Data.concept_information_content)
    precision = EvaluationIndex.get_precision(samples, real_intention, predict_intention,
                                              Data.Ontologies)
    recall = EvaluationIndex.get_recall(samples, real_intention, predict_intention,
                                        Data.Ontologies)
    print("predict_intention", predict_intention)
    print("tmp_encoding_length_compression_rates", tmp_encoding_length_compression_rates)
    print("method_log", method_log)
    print("jaccard_score", jaccard_score)
    print("intention_similarity", intention_similarity)
    print("precision", precision)
    print("recall", recall)


if __name__ == "__main__":
    _test_get_all_candidate_sub_intentions()
    # get_intentions_by_branch_and_bound_test()
    print("Aye")


