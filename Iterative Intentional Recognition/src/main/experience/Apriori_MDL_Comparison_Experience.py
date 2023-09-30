"""
   脚本描述：单轮意图识别算法（前置实验）
   评价指标：
   ① Jaccard系数， F1值， BMASS（外部评价）
   ② 置信度（内部评价）
   ③ 耗时
"""

import os
import random
import time
import multiprocessing

from src.main import Version
from src.main.samples.input import Sample
from src.main.samples.input.Data import Data
from src.main.intention_recognition import Apriori_MDL,  Config
from src.main.intention_recognition import EvaluationIndex, IntentionConf
from src.main.util.FileUtil import save_as_json, load_json

sample_version = Version.__sample_version__
output_path_prefix = os.path.join("../../../result/Apriori_MDL", "scenes_" +
                                  sample_version + "_" + Version.__intention_version__)
if not os.path.exists(output_path_prefix):
    os.mkdir(output_path_prefix)
auto_save_threshold = 100

sample_database = load_json("../../../resources/samples/all_samples.json")


# 基于频繁项集和最小描述长度的单次意图识别算法 Apriori_MDL
def experience_get_specific_samples_result(part_name, sample_paths):

    output_dir = os.path.join(output_path_prefix, "experience_comparison_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tmp_round_num = 0  # 为了自动保存而设置的变量
    # set parameters
    feedback_noise_rates = [0, 0.1, 0.2, 0.3, 0.4]
    label_noise_rates = [0, 0.2, 0.4, 0.6, 0.8]
    methods = ["Apriori_MDL"]
    total_run_num = len(sample_paths) * len(feedback_noise_rates) * len(label_noise_rates) * len(methods)

    # config
    run_time = 5
    # for MDL_RM
    config = Config.Config()
    config.rule_covered_positive_sample_rate_threshold = 0.4
    config.adjust_sample_num = True
    config.record_iteration_progress = False

    output_path = os.path.join(output_dir,
                               "result_jaccard_similarity_time_use_avg" + str(run_time) + "_" + part_name + ".json")
    result = []
    finished_record_keys = []
    if os.path.exists(output_path):
        result = load_json(output_path)
        finished_record_keys = [(x["scene"], x["feedback_noise_rate"], x["label_noise_rate"], x["method"])
                                for x in result]

    tmp_run_num = 0
    for tmp_sample_name in sample_paths:
        print("##### ", tmp_sample_name, " #####")
        tmp_sample_path_prefix = sample_paths[tmp_sample_name]

        for tmp_feedback_noise_rate in feedback_noise_rates:
            tmp_feedback_noise_rate_str = "_F" + str(tmp_feedback_noise_rate).replace(".", "p")
            for tmp_label_noise_rate in label_noise_rates:
                tmp_label_noise_rate_str = "_L" + str(tmp_label_noise_rate).replace(".", "p")
                tmp_sample_path = os.path.join(tmp_sample_path_prefix, "samples" + tmp_feedback_noise_rate_str
                                               + tmp_label_noise_rate_str + ".json")

                # load sample data
                samples, real_intention = Sample.load_sample_from_file(tmp_sample_path)
                for tmp_method in methods:
                    tmp_run_num += 1
                    tmp_record_key = (tmp_sample_name, tmp_feedback_noise_rate, tmp_label_noise_rate, tmp_method)

                    # if tmp_record_key in finished_record_keys:
                    #     continue

                    print(
                        f"running: {part_name} - {tmp_run_num}/{total_run_num} - {tmp_sample_name} - "
                        f"{tmp_feedback_noise_rate} - {tmp_label_noise_rate} - {tmp_method}")
                    all_jaccard_index = []
                    all_intention_similarity = []
                    all_precision = []
                    all_recall = []
                    all_time_use = []
                    all_rules = []
                    all_method_log = []
                    all_encoding_length_compression_rates = []

                    all_intention_compatibility = []
                    all_intention_completeness = []
                    all_intention_conf = []
                    for i in range(run_time):
                        time01 = time.time()
                        intention_result = None
                        method_log = None
                        tmp_encoding_length_compression_rates = None
                        if tmp_method == "Apriori_MDL":
                            method_result = Apriori_MDL.get_intentions_by_greedy_search(samples, config)
                            intention_result = method_result[0]
                            method_log = method_result[-1]
                            tmp_encoding_length_compression_rates = method_result[1] / method_result[2]
                        # elif tmp_method == "RuleGO":
                        #     method_result = RuleGO_Gruca2017_2.RuleGo(samples, config)
                        #     intention_result = method_result[0]
                        #     method_log = method_result[-1]

                        time02 = time.time()
                        all_time_use.append(time02 - time01)
                        jaccard_index = EvaluationIndex.get_jaccard_index(sample_database, real_intention,
                                                                          intention_result,
                                                                          Data.Ontologies)
                        best_map_average_semantic_similarity = \
                            EvaluationIndex.get_intention_similarity(intention_result, real_intention,
                                                                     Data.direct_Ancestor,
                                                                     Data.Ontology_Root,
                                                                     Data.concept_information_content)
                        precision = EvaluationIndex.get_precision(sample_database, real_intention,
                                                                  intention_result,
                                                                  Data.Ontologies)
                        recall = EvaluationIndex.get_recall(sample_database, real_intention,
                                                            intention_result,
                                                            Data.Ontologies)

                        compatibility = IntentionConf.get_intention_compatibility(samples, intention_result, Data.Ontologies)
                        completeness = IntentionConf.get_intention_completeness(samples, intention_result, Data.Ontologies)
                        conf = IntentionConf.get_intention_conf(samples, intention_result, Data.Ontologies)

                        all_intention_similarity.append(best_map_average_semantic_similarity)
                        all_jaccard_index.append(jaccard_index)
                        all_precision.append(precision)
                        all_recall.append(recall)
                        all_rules.append(intention_result)
                        all_method_log.append(method_log)
                        all_encoding_length_compression_rates.append(tmp_encoding_length_compression_rates)

                        all_intention_compatibility.append(compatibility)
                        all_intention_completeness.append(completeness)
                        all_intention_conf.append(conf)

                    tmp_result = {"scene": tmp_sample_name,
                                  "feedback_noise_rate": tmp_feedback_noise_rate,
                                  "label_noise_rate": tmp_label_noise_rate,
                                  "method": tmp_method,
                                  "time_use": all_time_use,
                                  "jaccard_index": all_jaccard_index,
                                  "intention_similarity": all_intention_similarity,
                                  "precision": all_precision,
                                  "recall": all_recall,
                                  "extracted_rules_json": all_rules,
                                  "method_log": all_method_log,
                                  "encoding_length_compression_rates": all_encoding_length_compression_rates,
                                  "param_config": config.to_json(tmp_method),

                                  "intention_compatibility": all_intention_compatibility,
                                  "intention_completeness": all_intention_completeness,
                                  "intention_conf": all_intention_conf
                                  }
                    result.append(tmp_result)
                    finished_record_keys.append(tmp_record_key)
                    tmp_round_num += 1
                    if tmp_round_num == auto_save_threshold:
                        save_as_json(result, output_path)
                        tmp_round_num = 0
    save_as_json(result, output_path)


# take scene, sample level noise rate, label level noise rate method as variable
# and record the time use, jaccard score， intention_similarity，and rules(in json_str).
def experience_get_all_samples_result():
    # load sample paths
    # sample_paths = get_sample_paths()
    samples_dir = os.path.join("../../../resources/samples", "scenes_" + sample_version)
    sample_names = os.listdir(samples_dir)
    sample_paths = {}
    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    random.shuffle(sample_names)
    print(len(sample_names), sample_names)
    part_num = 12  # 12 process
    split_size = len(sample_names) // part_num
    rest_size = len(sample_names) % part_num
    rest_count = 0
    all_samples_parts = []
    for i in range(part_num):
        tmp_sample_part = {}

        if rest_count < rest_size:
            new_split_size = split_size + 1
            rest_count += 1
        else:
            new_split_size = split_size

        for j in range(new_split_size):
            tmp_sample_name = sample_names.pop(0)
            tmp_sample_part[tmp_sample_name] = sample_paths[tmp_sample_name]
        all_samples_parts.append(tmp_sample_part)

    # # for test
    # experience_get_specific_samples_result("PART_8", all_samples_parts[8])

    for i, tmp_sample_part in enumerate(all_samples_parts):
        part_name = "PART_" + str(i)
        tmp_p = multiprocessing.Process(target=experience_get_specific_samples_result,
                                        args=(part_name, tmp_sample_part))
        tmp_p.start()


if __name__ == "__main__":
    experience_get_all_samples_result()
    print("Aye")
