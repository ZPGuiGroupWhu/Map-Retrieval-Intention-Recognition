# coding:utf-8
import csv
import os
import time
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Map_RIR.src.main import Intention
from Map_RIR.src.main.samples.input import Sample
from Map_RIR.src.main.samples.input.Data import Data
from Map_RIR.src.main.intention_recognition import Config, IM_Details
from Map_RIR.src.main.experience import Evaluation_IM
from Map_RIR.src.main.util import FileUtil
from Map_RIR.src.main.util.FileUtil import save_as_json, load_json

sample_version = Intention.__sample_version__
output_path_prefix = os.path.join("/Map_RIR\\result", "scenes_" +
                                  sample_version + "_" + Intention.__version__)
if not os.path.exists(output_path_prefix):
    os.mkdir(output_path_prefix)
auto_save_threshold = 100


def experience_get_specific_samples_result(part_name, sample_paths):
    output_dir = os.path.join(output_path_prefix, "experience_feasibility_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tmp_round_num = 0  # 为了自动保存而设置的变量
    # set parameters
    feedback_noise_rates = [0, 0.1, 0.2, 0.3]
    label_noise_rates = [0, 0.2, 0.4, 0.6, 0.8, 1]
    methods = ["IM"]
    total_run_num = len(sample_paths) * len(feedback_noise_rates) * len(label_noise_rates) * len(methods)

    # config
    run_time = 1
    # for MDL_RM
    tmp_threshold = Config.rule_covered_positive_sample_rate_threshold
    Config.adjust_sample_num = True
    Config.TAG_RECORD_PROCESS = True

    output_path = os.path.join(output_dir,
                               "result_jaccard_similarity_time_use_avg" + str(run_time) + "_" + part_name + ".json")
    result = []
    finished_record_keys = []
    if os.path.exists(output_path):
        result = load_json(output_path)
        finished_record_keys = [(x["scene"], x["feedback_noise_rate"], x["label_noise_rate"], x["method"])
                                for x in result]

    tmp_run_num = 0
    real_intentions = Sample.get_real_intent()
    for tmp_sample_name in sample_paths:
        print("##### ", tmp_sample_name, " #####")
        tmp_sample_path_prefix = sample_paths[tmp_sample_name]

        for tmp_feedback_noise_rate in feedback_noise_rates:
            tmp_feedback_noise_rate_str = "_S" + str(tmp_feedback_noise_rate).replace(".", "p")
            for tmp_label_noise_rate in label_noise_rates:
                tmp_label_noise_rate_str = "_L" + str(tmp_label_noise_rate).replace(".", "p")

                if tmp_feedback_noise_rate == 0:
                    if tmp_label_noise_rate == 0:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix, "final_samples.json")
                    else:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_label_noise_rate_str + ".json")
                else:
                    if tmp_label_noise_rate == 0:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_feedback_noise_rate_str + ".json")
                    else:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_feedback_noise_rate_str
                                                       + tmp_label_noise_rate_str + ".json")
                # load sample data
                # docs, real_intention = Sample.load_sample_from_file(tmp_sample_path)

                docs, real_intention = Sample.load_real_intents(real_intentions, tmp_sample_path,
                                                                tmp_sample_name)

                data = Data(docs, real_intention)

                samples = data.docs
                positive_samples = samples["relevance"]
                negative_samples = samples["irrelevance"]
                ancestors = Data.Ancestor
                ontologies = Data.Ontologies
                ontology_root = Data.Ontology_Root
                direct_ancestors = Data.direct_Ancestor
                information_content = data.concept_information_content
                terms = data.all_relevance_concepts
                terms_covered_samples = data.all_relevance_concepts_retrieved_docs
                for tmp_method in methods:
                    tmp_run_num += 1
                    tmp_record_key = (
                        tmp_sample_name, tmp_feedback_noise_rate, tmp_label_noise_rate, tmp_method)
                    if tmp_record_key in finished_record_keys:
                        continue

                    print(
                        f"running: {part_name} - {tmp_run_num}/{total_run_num} - {tmp_sample_name} - "
                        f"{tmp_feedback_noise_rate} - {tmp_label_noise_rate} - {tmp_method}")
                    all_jaccard_index = []
                    all_intention_similarity = []
                    all_precision = []
                    all_recall = []
                    all_time_use = []
                    all_rules = []
                    all_processes_log = []
                    all_encoding_length_compression_rates = []
                    for i in range(run_time):
                        time01 = time.time()
                        intention_result = None

                        if tmp_method == "IM":
                            tmp_result, usetime,process_log = IM_Details.run_IM(samples, 0.3, 2000)
                            time02 = time.time()
                            intention_result = []
                            # 转换成字典格式
                            for sub_intent in tmp_result.intention:
                                real_sub_intent = {"positive": sub_intent.positive_sub_Intention,
                                                   "negative": sub_intent.negative_sub_Intention}
                                intention_result.append(real_sub_intent)

                        time02 = time.time()
                        all_time_use.append(time02 - time01)
                        jaccard_index = Evaluation_IM.get_jaccard_index(samples, real_intention,
                                                                        intention_result,
                                                                        Data.Ontologies, Data.Ontology_Root)
                        best_map_average_semantic_similarity = \
                            Evaluation_IM.get_intention_similarity(intention_result, real_intention,
                                                                   direct_ancestors, ontology_root,
                                                                   information_content)

                        precision = Evaluation_IM.get_precision(samples, real_intention,
                                                                intention_result,
                                                                Data.Ontologies, Data.Ontology_Root)
                        recall = Evaluation_IM.get_recall(samples, real_intention,
                                                          intention_result,
                                                          Data.Ontologies, Data.Ontology_Root)



                        for tmp_iteration_log in process_log:

                            tmp_rules_intention_result = []
                            for sub_intent in tmp_iteration_log["intention_result"]:
                                real_sub_intent = {"positive": sub_intent.positive_sub_Intention,
                                                   "negative": sub_intent.negative_sub_Intention}
                                tmp_rules_intention_result.append(real_sub_intent)
                            tmp_iteration_log["intention_result"] = tmp_rules_intention_result
                            tmp_rules_BMASS = \
                                Evaluation_IM.get_intention_similarity(tmp_rules_intention_result, real_intention,
                                                                         direct_ancestors, ontology_root,
                                                                         information_content)
                            tmp_rules_jaccard_index = Evaluation_IM.get_jaccard_index(samples, real_intention,
                                                                                        tmp_rules_intention_result,
                                                                                        Data.Ontologies,
                                                                                        Data.Ontology_Root)
                            tmp_rules_precision = Evaluation_IM.get_precision(samples, real_intention,
                                                                                tmp_rules_intention_result,
                                                                                Data.Ontologies, Data.Ontology_Root)
                            tmp_rules_recall = Evaluation_IM.get_recall(samples, real_intention,
                                                                          tmp_rules_intention_result,
                                                                          Data.Ontologies, Data.Ontology_Root)
                            tmp_iteration_log["intention_similarity"] = tmp_rules_BMASS
                            tmp_iteration_log["jaccard_index"] = tmp_rules_jaccard_index
                            tmp_iteration_log["precision"] = tmp_rules_precision
                            tmp_iteration_log["recall"] = tmp_rules_recall

                        # tmp_encoding_length_compression_rates = method_result[1] / method_result[2]

                        all_intention_similarity.append(best_map_average_semantic_similarity)
                        all_jaccard_index.append(jaccard_index)
                        all_precision.append(precision)
                        all_recall.append(recall)
                        all_rules.append(intention_result)
                        all_processes_log.append(process_log)
                        # all_encoding_length_compression_rates.append(tmp_encoding_length_compression_rates)

                    tmp_result = {"scene": tmp_sample_name, "feedback_noise_rate": tmp_feedback_noise_rate,
                                  "label_noise_rate": tmp_label_noise_rate,
                                  "method": tmp_method,
                                  "time_use": all_time_use,
                                  "rule_covered_positive_sample_rate_threshold": tmp_threshold,
                                  "jaccard_index": all_jaccard_index,
                                  "intention_similarity": all_intention_similarity,
                                  "precision": all_precision,
                                  "recall": all_recall,
                                  "extracted_rules_json": all_rules,
                                  "merge_processes_log": all_processes_log,
                                  "encoding_length_compression_rates": all_encoding_length_compression_rates}
                    result.append(tmp_result)
                    finished_record_keys.append(tmp_record_key)
                    tmp_round_num += 1
                    if tmp_round_num == auto_save_threshold:
                        save_as_json(result, output_path)
                        tmp_round_num = 0
    save_as_json(result, output_path)

def export_MDL_feasibility_experience_result_for_iteration():
    sample_version = Intention.__sample_version__
    code_version = Intention.__version__
    result_dirname = "scenes_" + sample_version + "_" + code_version
    scenes = ["单意图单维度", "单意图多维度", "多意图单维度", "含负向意图", "多意图多维度"]
    result_path_prefix = os.path.join(
        "/Map_RIR\\result",
        result_dirname)

    # load result
    data_file_path = os.path.join(result_path_prefix, "experience_feasibility_result",
                                  "result_jaccard_similarity_time_use_avg5.json")
    all_data = load_json(data_file_path)
    iteration_nums = [0] * 20
    iteration_BMASS_encoding_length_rates_increase_num = [0] * 20
    output_dir = os.path.join(result_path_prefix, "experience_feasibility_result")
    for tmp_data in all_data:
        tmp_merge_process_log = tmp_data["merge_processes_log"]
        for tmp_round_log in tmp_merge_process_log:
            tmp_iteration_num = len(tmp_round_log) - 1
            iteration_nums[tmp_iteration_num] += 1
            # get_first_and_last_iteration
            first_iteration_rule_log = tmp_round_log[0]
            last_iteration_rule_log = tmp_round_log[-1]
            first_iteration_BMASS = first_iteration_rule_log["intention_similarity"]
            first_iteration_intention_encoding_length = first_iteration_rule_log["intention_encoding_length"]
            first_iteration_sample_encoding_length = first_iteration_rule_log["sample_encoding_length"]
            last_iteration_BMASS = last_iteration_rule_log["intention_similarity"]
            last_iteration_intention_encoding_length = last_iteration_rule_log["intention_encoding_length"]
            last_iteration_sample_encoding_length = last_iteration_rule_log["sample_encoding_length"]
            first_iteration_rate = first_iteration_BMASS / (first_iteration_intention_encoding_length
                                                            + first_iteration_sample_encoding_length)
            last_iteration_rate = last_iteration_BMASS / (last_iteration_intention_encoding_length
                                                          + last_iteration_sample_encoding_length)
            if first_iteration_rate < last_iteration_rate:
                iteration_BMASS_encoding_length_rates_increase_num[tmp_iteration_num] += 1
    for i in range(len(iteration_nums)):
        if iteration_nums[i] != 0:
            iteration_BMASS_encoding_length_rates_increase_num[i] /= iteration_nums[i]
    num_values = list(range(20))
    iteration_nums_result = \
        np.transpose([num_values, iteration_nums, iteration_BMASS_encoding_length_rates_increase_num]).tolist()
    iteration_nums_result_output_path = os.path.join(output_dir, "iteration_nums.csv")
    FileUtil.save_as_csv(iteration_nums_result, iteration_nums_result_output_path)

    specific_iteration_num = 10
    indexes = ["intention_similarity", "jaccard_index", "encoding_length_compression_rates",
               "intention_encoding_length", "sample_encoding_length"]

    for tmp_iteration_num in range(1, specific_iteration_num):
        result = []
        for i in range(tmp_iteration_num + 1):
            result.append([0.0] * 5)
        frequency = 0
        for tmp_data in all_data:
            tmp_merge_process_log = tmp_data["merge_processes_log"]
            for tmp_round_log in tmp_merge_process_log:
                if len(tmp_round_log) == tmp_iteration_num + 1:
                    frequency += 1
                    for i, tmp_iteration_log in enumerate(tmp_round_log):
                        tmp_rules_log = tmp_iteration_log
                        for j, tmp_index in enumerate(indexes):
                            result[i][j] += tmp_rules_log[tmp_index]
        print(frequency)
        for tmp_row_index in range(len(result)):
            for tmp_column_index in range(len(result[tmp_row_index])):
                result[tmp_row_index][tmp_column_index] /= frequency
        output_path = \
            os.path.join(output_dir, f"indexed_average_with_iteration_num_{tmp_iteration_num}.csv")
        FileUtil.save_as_csv(result, output_path)




def export_MDL_feasibility_experience_result_for_BMASS():
    sample_version = Intention.__sample_version__
    code_version = Intention.__version__
    result_dirname = "scenes_" + sample_version + "_" + code_version
    scenes = ["单意图单维度", "单意图多维度", "多意图单维度", "含负向意图", "多意图多维度"]
    result_path_prefix = os.path.join(
        "/Map_RIR\\result",
        result_dirname)

    # load result
    data_file_path = os.path.join(result_path_prefix, "experience_feasibility_result",
                                  "result_jaccard_similarity_time_use_avg5.json")
    all_data = load_json(data_file_path)

    output_dir = os.path.join(result_path_prefix, "experience_feasibility_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # export iteration num
    max_iteration_nums = 5
    iteration_BMASS_encoding_length_rates_increase_num = [0] * 20
    iteration_BMASS_sample_encoding_lengths = []
    iteration_BMASS_intent_encoding_lengths =[]
    iteration_jaccard_sample_encoding_lengths = {}
    result_for_BMASS=[]
    BMASS=[]

    eval_all_encoding_lengths=[0] * 20
    eval_intent_encoding_lengths = [0] * 20
    # each samples
    for tmp_data in all_data:
        tmp_iter=0
        tmp_process_log = tmp_data["merge_processes_log"]
        if (tmp_data["feedback_noise_rate"]>0) or (tmp_data["label_noise_rate"]>0):
            continue
        # each iteration
        for tmp_round_log in tmp_process_log[0]:
            tmp_iter+=1
            if tmp_iter>=max_iteration_nums:
                break
            tmp_iteration_num = len(tmp_round_log) - 1
            tmp_BMASS = tmp_round_log["intention_similarity"]
            tmp_jaccard = tmp_round_log["jaccard_index"]
            if tmp_BMASS not in BMASS:
                BMASS.append(tmp_BMASS)
                iteration_BMASS_intent_encoding_lengths.append([tmp_round_log["intention_encoding_length"]])
                iteration_BMASS_sample_encoding_lengths.append([tmp_round_log["sample_encoding_length"]])
            for index in range(len(BMASS)):
                if BMASS[index]==tmp_BMASS:
                    iteration_BMASS_intent_encoding_lengths[index].append(tmp_round_log["intention_encoding_length"])
                    iteration_BMASS_sample_encoding_lengths[index].append(tmp_round_log["sample_encoding_length"])


            # iteration_BMASS_sample_encoding_lengths[tmp_BMASS].append(tmp_round_log["sample_encoding_length"])
            # iteration_jaccard_sample_encoding_lengths[tmp_jaccard].append(tmp_round_log["sample_encoding_length"])
            # # get_first_and_last_iteration
            # first_iteration_rule_log = tmp_round_log[0][0]
            # last_iteration_rule_log = tmp_round_log[-1][0]
            # max_iteration_nums.append(last_iteration_rule_log["iteration"])
            # first_iteration_BMASS = first_iteration_rule_log["intention_similarity"]
            # first_iteration_intention_encoding_length = first_iteration_rule_log["intention_encoding_length"]
            # first_iteration_sample_encoding_length = first_iteration_rule_log["sample_encoding_length"]
            # last_iteration_BMASS = last_iteration_rule_log["intention_similarity"]
            # last_iteration_intention_encoding_length = last_iteration_rule_log["intention_encoding_length"]
            # last_iteration_sample_encoding_length = last_iteration_rule_log["sample_encoding_length"]
            # first_iteration_rate = first_iteration_BMASS / (first_iteration_intention_encoding_length
            #                                                 + first_iteration_sample_encoding_length)
            # last_iteration_rate = last_iteration_BMASS / (last_iteration_intention_encoding_length
            #                                               + last_iteration_sample_encoding_length)
            # if first_iteration_rate < last_iteration_rate:
            #     iteration_BMASS_encoding_length_rates_increase_num[tmp_iteration_num] += 1

    # BMASS = iteration_BMASS_sample_encoding_lengths.keys().tolist()
    for index in range(len(BMASS)):
        tmp_result={}
        tmp_result["BMASS"]=BMASS[index]
        tmp_result["intention_encoding_length"] = 0
        tmp_result["sample_encoding_length"] = 0
        for i in iteration_BMASS_intent_encoding_lengths[index]:
            tmp_result["intention_encoding_length"]+=i
        for s in iteration_BMASS_sample_encoding_lengths[index]:
            tmp_result["sample_encoding_length"]+=s
        tmp_result["intention_encoding_length"]/=len(iteration_BMASS_intent_encoding_lengths[index])
        tmp_result["sample_encoding_length"]/=len(iteration_BMASS_sample_encoding_lengths[index])
        result_for_BMASS.append(tmp_result)
    header=["BMASS","intention_encoding_length","sample_encoding_length"]
    print(BMASS)
    print(result_for_BMASS)






    iteration_nums_result_output_path = os.path.join(output_dir, "BMASS.csv")
    # FileUtil.save_as_csv(max_iteration_nums)
    with open(iteration_nums_result_output_path,'a',newline='',encoding='utf-8') as csv_file:
        writer=csv.DictWriter(csv_file,fieldnames=header)
        writer.writeheader()
        writer.writerows(result_for_BMASS)
    # data = pd.read_csv(iteration_nums_result_output_path)
    # print(data.info())
    #
    # BMASS = data.values["BMASS"]  # 获取 0列的所有数据
    # intent = data.values["intention_encoding_length"]
    # sample = data.values[:, 145]
    # average2 = data.values[:, 111]
    # std_12 = data.values[:, 112]
    # # average2 = data.values[:, 67]
    # # std_12 = data.values[:, 68]
    # print(sample_num)
    #
    # # for tmp_sample_num in sample_num:
    # #     mbss[tmp_sample_num]=data.
    #
    # print(average2)
    #
    # # 绘图
    # plt.plot(sample_num, average, 'b-', label='mean_1')
    # plt.fill_between(sample_num, average - std_1, average + std_1, color='b', alpha=0.2)
    # plt.plot(sample_num, average2, 'r-', label='mean_1')
    # plt.fill_between(sample_num, average2 - std_12, average2 + std_12, color='r', alpha=0.2)
    # plt.show()
    # # export the scene and noise information of the records with iteration num 0
    # iteration_nums = list(range(7))
    # samples_with_specific_iteration_num_record_num = []
    # for i in range(len(feedback_noise_rateS) * 5):  # 5 scenes
    #     samples_with_specific_iteration_num_record_num.append(
    #         [0] * (len(label_noise_rateS) * len(iteration_nums)))
    # for tmp_specific_iteration_num in iteration_nums:
    #     for tmp_data in all_data:
    #         tmp_scene_name = tmp_data["scene"]
    #         tmp_scene = get_scene_v4_4(tmp_scene_name)
    #         tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
    #         tmp_label_noise_rate = tmp_data["label_noise_rate"]
    #         tmp_merge_process_log = tmp_data["merge_processes_log"]
    #         for tmp_round_log in tmp_merge_process_log:
    #             tmp_iteration_num = len(tmp_round_log) - 1
    #             if tmp_iteration_num == tmp_specific_iteration_num:
    #                 tmp_feedback_noise_rate_index = feedback_noise_rateS_REVERSE.index(
    #                     tmp_feedback_noise_rate)
    #                 tmp_label_noise_rate_index = label_noise_rateS.index(tmp_label_noise_rate)
    #                 tmp_row_index = \
    #                     scenes.index(tmp_scene) * len(feedback_noise_rateS) + tmp_feedback_noise_rate_index
    #                 tmp_column_index = \
    #                     tmp_specific_iteration_num * len(label_noise_rateS) + tmp_label_noise_rate_index
    #                 samples_with_specific_iteration_num_record_num[tmp_row_index][tmp_column_index] += 1
    # samples_with_specific_iteration_num_record_num_output_path = \
    #     os.path.join(output_dir, f"samples_record_num_with_different_iteration_num.csv")
    # FileUtil.save_as_csv(samples_with_specific_iteration_num_record_num,
    #                      samples_with_specific_iteration_num_record_num_output_path)

# take scene, sample level noise rate, label level noise rate method as variable
# and record the time use, jaccard score， intention_similarity，and rules(in json_str).
def experience_get_all_samples_result():
    Config.TAG_RECORD_MERGE_PROCESS = True
    # load sample paths
    samples_dir = os.path.join("./../../../resources/samples", "scenes_" + sample_version, "scenes_v5")
    sample_names = os.listdir(samples_dir)
    sample_paths = {}
    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    # 过滤“无意图”样本集
    sample_names = list(filter(lambda x: not (11 <= int(x[5:]) <= 13 or 361 <= int(x[5:]) <= 387), sample_names))
    print(len(sample_names), sample_names)
    # return

    # split all samples name to several parts, and calculate every part with a thread
    part_num = 1  # 12 process
    split_size = len(sample_names) // part_num
    all_samples_parts = []
    for i in range(part_num - 1):
        tmp_sample_part = {}
        for j in range(split_size):
            tmp_sample_name = sample_names.pop(0)
            tmp_sample_part[tmp_sample_name] = sample_paths[tmp_sample_name]
        all_samples_parts.append(tmp_sample_part)
    last_sample_part = {}
    for tmp_sample_name in sample_names:
        last_sample_part[tmp_sample_name] = sample_paths[tmp_sample_name]
    all_samples_parts.append(last_sample_part)
    for i, tmp_sample_part in enumerate(all_samples_parts):
        part_name = "PART_" + str(i)
        tmp_p = multiprocessing.Process(target=experience_get_specific_samples_result,
                                        args=(part_name, tmp_sample_part))
        tmp_p.start()


def experience_get_all_samples_result_MDL():
    Config.TAG_RECORD_MERGE_PROCESS = True
    # load sample paths
    samples_dir = os.path.join("./../../../resources/samples", "scenes_" + sample_version, "scenes_v5")
    sample_names = os.listdir(samples_dir)
    sample_paths = {}
    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    # 过滤“无意图”样本集
    # sample_names = list(filter(lambda x: not (11 <= int(x[5:]) <= 13 or 361 <= int(x[5:]) <= 387), sample_names))
    sample_names = list(filter(lambda x: (0 <= int(x[5:]) <= 600), sample_names))
    print(len(sample_names), sample_names)
    # return
    part_num = 12  # 12 process
    split_size = len(sample_names) // part_num
    all_samples_parts = []
    for i in range(part_num - 1):
        tmp_sample_part = {}
        for j in range(split_size):
            tmp_sample_name = sample_names.pop(0)
            tmp_sample_part[tmp_sample_name] = sample_paths[tmp_sample_name]
        all_samples_parts.append(tmp_sample_part)
    last_sample_part = {}
    for tmp_sample_name in sample_names:
        last_sample_part[tmp_sample_name] = sample_paths[tmp_sample_name]
    all_samples_parts.append(last_sample_part)
    for i, tmp_sample_part in enumerate(all_samples_parts):
        part_name = "PART_" + str(i)
        tmp_p = multiprocessing.Process(target=experience_get_specific_samples_result,
                                        args=(part_name, tmp_sample_part))
        tmp_p.start()


def experience_get_all_samples_result_single_MDL():
    Config.TAG_RECORD_MERGE_PROCESS = True
    # load sample paths
    samples_dir = os.path.join("./../../../resources/samples", "scenes_" + sample_version, "scenes_v5")
    sample_names = os.listdir(samples_dir)
    sample_paths = {}
    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    # 过滤“无意图”样本集
    #sample_names = list(filter(lambda x: not (11 <= int(x[5:]) <= 13 or 361 <= int(x[5:]) <= 387), sample_names))
    sample_names = list(filter(lambda x: (540 <= int(x[5:]) <= 600 ), sample_names))
    print(len(sample_names), sample_names)
    # return
    experience_get_specific_samples_result_single_MDL(sample_names, sample_paths)


def experience_get_specific_samples_result_single_MDL(sample_names, sample_paths):
    output_dir = os.path.join(output_path_prefix, "experience_feasibility_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tmp_round_num = 0  # 为了自动保存而设置的变量
    # set parameters
    feedback_noise_rates = [0, 0.1, 0.2, 0.3]
    label_noise_rates = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # feedback_noise_rates = [0]
    # label_noise_rates = [0]
    methods = ["IM"]
    total_run_num = len(sample_names) * len(feedback_noise_rates) * len(label_noise_rates) * len(methods)

    # config
    run_time = 1
    # for MDL_RM
    tmp_threshold = Config.rule_covered_positive_sample_rate_threshold
    Config.adjust_sample_num = True
    Config.TAG_RECORD_PROCESS = True

    output_path = os.path.join(output_dir,
                               "result_jaccard_similarity_time_use_avg" + str(run_time) + "_single" + ".json")
    result = []
    finished_record_keys = []
    if os.path.exists(output_path):
        result = load_json(output_path)
        finished_record_keys = [(x["scene"], x["feedback_noise_rate"], x["label_noise_rate"], x["method"])
                                for x in result]

    tmp_run_num = 0
    real_intentions = Sample.get_real_intent()
    for tmp_sample_name in sample_names:
        print("##### ", tmp_sample_name, " #####")
        tmp_sample_path_prefix = sample_paths[tmp_sample_name]

        for tmp_feedback_noise_rate in feedback_noise_rates:
            tmp_feedback_noise_rate_str = "_S" + str(tmp_feedback_noise_rate).replace(".", "p")
            for tmp_label_noise_rate in label_noise_rates:
                tmp_label_noise_rate_str = "_L" + str(tmp_label_noise_rate).replace(".", "p")

                if tmp_feedback_noise_rate == 0:
                    if tmp_label_noise_rate == 0:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix, "final_samples.json")
                    else:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_label_noise_rate_str + ".json")
                else:
                    if tmp_label_noise_rate == 0:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_feedback_noise_rate_str + ".json")
                    else:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_feedback_noise_rate_str
                                                       + tmp_label_noise_rate_str + ".json")
                # load sample data
                # docs, real_intention = Sample.load_sample_from_file(tmp_sample_path)

                docs, real_intention = Sample.load_real_intents(real_intentions, tmp_sample_path,
                                                                tmp_sample_name)

                data = Data(docs, real_intention)

                samples = data.docs
                positive_samples = samples["relevance"]
                negative_samples = samples["irrelevance"]
                ancestors = Data.Ancestor
                ontologies = Data.Ontologies
                ontology_root = Data.Ontology_Root
                direct_ancestors = Data.direct_Ancestor
                information_content = data.concept_information_content
                terms = data.all_relevance_concepts
                terms_covered_samples = data.all_relevance_concepts_retrieved_docs
                for tmp_method in methods:
                    tmp_run_num += 1
                    tmp_record_key = (
                        tmp_sample_name, tmp_feedback_noise_rate, tmp_label_noise_rate, tmp_method)
                    if tmp_record_key in finished_record_keys:
                        continue

                    print(
                        f"running: {tmp_run_num}/{total_run_num} - {tmp_sample_name} - "
                        f"{tmp_feedback_noise_rate} - {tmp_label_noise_rate} - {tmp_method}")
                    all_jaccard_index = []
                    all_intention_similarity = []
                    all_precision = []
                    all_recall = []
                    all_time_use = []
                    all_rules = []
                    all_processes_log = []
                    all_encoding_length_compression_rates = []
                    for i in range(run_time):
                        time01 = time.time()
                        intention_result = None

                        if tmp_method == "IM":
                            tmp_result, usetime, process_log = IM_Details.run_IM(samples, 0.3, 2000)
                            time02 = time.time()
                            intention_result = []
                            # 转换成字典格式
                            for sub_intent in tmp_result.intention:
                                real_sub_intent = {"positive": sub_intent.positive_sub_Intention,
                                                   "negative": sub_intent.negative_sub_Intention}
                                intention_result.append(real_sub_intent)

                        time02 = time.time()
                        all_time_use.append(time02 - time01)
                        jaccard_index = Evaluation_IM.get_jaccard_index(samples, real_intention,
                                                                        intention_result,
                                                                        Data.Ontologies, Data.Ontology_Root)
                        best_map_average_semantic_similarity = \
                            Evaluation_IM.get_intention_similarity(intention_result, real_intention,
                                                                   direct_ancestors, ontology_root,
                                                                   information_content)

                        precision = Evaluation_IM.get_precision(samples, real_intention,
                                                                intention_result,
                                                                Data.Ontologies, Data.Ontology_Root)
                        recall = Evaluation_IM.get_recall(samples, real_intention,
                                                          intention_result,
                                                          Data.Ontologies, Data.Ontology_Root)

                        for tmp_iteration_log in process_log:

                            tmp_rules_intention_result = []
                            for sub_intent in tmp_iteration_log["intention_result"]:
                                real_sub_intent = {"positive": sub_intent.positive_sub_Intention,
                                                   "negative": sub_intent.negative_sub_Intention}
                                tmp_rules_intention_result.append(real_sub_intent)
                            tmp_iteration_log["intention_result"] = tmp_rules_intention_result
                            tmp_rules_BMASS = \
                                Evaluation_IM.get_intention_similarity(tmp_rules_intention_result, real_intention,
                                                                       direct_ancestors, ontology_root,
                                                                       information_content)
                            tmp_rules_jaccard_index = Evaluation_IM.get_jaccard_index(samples, real_intention,
                                                                                      tmp_rules_intention_result,
                                                                                      Data.Ontologies,
                                                                                      Data.Ontology_Root)
                            tmp_rules_precision = Evaluation_IM.get_precision(samples, real_intention,
                                                                              tmp_rules_intention_result,
                                                                              Data.Ontologies, Data.Ontology_Root)
                            tmp_rules_recall = Evaluation_IM.get_recall(samples, real_intention,
                                                                        tmp_rules_intention_result,
                                                                        Data.Ontologies, Data.Ontology_Root)
                            tmp_iteration_log["intention_similarity"] = tmp_rules_BMASS
                            tmp_iteration_log["jaccard_index"] = tmp_rules_jaccard_index
                            tmp_iteration_log["precision"] = tmp_rules_precision
                            tmp_iteration_log["recall"] = tmp_rules_recall

                        # tmp_encoding_length_compression_rates = method_result[1] / method_result[2]

                        all_intention_similarity.append(best_map_average_semantic_similarity)
                        all_jaccard_index.append(jaccard_index)
                        all_precision.append(precision)
                        all_recall.append(recall)
                        all_rules.append(intention_result)
                        all_processes_log.append(process_log)
                        # all_encoding_length_compression_rates.append(tmp_encoding_length_compression_rates)

                    tmp_result = {"scene": tmp_sample_name, "feedback_noise_rate": tmp_feedback_noise_rate,
                                  "label_noise_rate": tmp_label_noise_rate,
                                  "method": tmp_method,
                                  "time_use": all_time_use,
                                  "rule_covered_positive_sample_rate_threshold": tmp_threshold,
                                  "jaccard_index": all_jaccard_index,
                                  "intention_similarity": all_intention_similarity,
                                  "precision": all_precision,
                                  "recall": all_recall,
                                  "extracted_rules_json": all_rules,
                                  "merge_processes_log": all_processes_log,
                                  "encoding_length_compression_rates": all_encoding_length_compression_rates}
                    result.append(tmp_result)
                    finished_record_keys.append(tmp_record_key)
                    tmp_round_num += 1
                    if tmp_round_num == auto_save_threshold:
                        save_as_json(result, output_path)
                        tmp_round_num = 0
    save_as_json(result, output_path)


if __name__ == "__main__":
    # experience_get_all_samples_result()
    # experience_get_all_samples_result_MDL()
    # experience_get_all_samples_result_single_MDL()
    # export_MDL_feasibility_experience_result_for_BMASS()
    export_MDL_feasibility_experience_result_for_iteration()
    #experience_get_all_samples_result_single_Apriori()
    print("Aye")
