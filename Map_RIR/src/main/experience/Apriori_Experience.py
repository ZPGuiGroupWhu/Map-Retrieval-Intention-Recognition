# coding:utf-8
import os
import time
import copy
import multiprocessing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Map_RIR.src.main import Intention
from Map_RIR.src.main.experience.Result_Analysis import get_scene_v4_4

from Map_RIR.src.main.samples.input import Sample
from Map_RIR.src.main.samples.input.Data import Data
from Map_RIR.src.main.intention_recognition import Config, Run_Map_RIR, Filter_without_FIM
from Map_RIR.src.main.experience import Evaluation_IM, Result_Analysis
from Map_RIR.src.main.util.FileUtil import save_as_json, load_json
from Map_RIR.src.main.util import FileUtil

sample_version = Intention.__sample_version__
file_path_prefix = "../../../resources/samples"
scenes_path = os.path.join(file_path_prefix, sample_version)
output_path_prefix = os.path.join(
    "/Map_RIR\\result", "scenes_" +
                        sample_version + "_" + Intention.__version__)
if not os.path.exists(output_path_prefix):
    os.mkdir(output_path_prefix)

auto_save_threshold = 100


def experience_get_specific_samples_result(part_name, sample_paths):
    output_dir = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tmp_round_num = 0  # 为了自动保存而设置的变量
    # set parameters
    feedback_noise_rates = [0, 0.1, 0.2, 0.3]
    label_noise_rates = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # feedback_noise_rates = [0]
    # label_noise_rates = [0]
    methods = ["IM", "IM_without_Apriori"]
    # random_merge_numbers = [2000]
    random_merge_numbers = [10]
    rule_covered_positive_sample_rate_thresholds = [0.1, 0.2, 0.3]
    # rule_covered_positive_sample_rate_thresholds = [0.05, 0.3]
    # random_merge_numbers = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    # random_merge_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # random_merge_numbers = [0.5, 1, 5, 10, 15, 20, 25, 30, 35, 40]
    # random_merge_numbers = [50,100,200,400,600,800,1000,1500,2000,2500]
    # rule_covered_positive_sample_rate_thresholds = [0.3]
    total_run_num = len(sample_paths) * len(feedback_noise_rates) * len(
        label_noise_rates) * len(methods) * (len(random_merge_numbers)
                                             + len(rule_covered_positive_sample_rate_thresholds) - 1)
    #  若要运行两个参数所有组合以确定两者最佳取值，需将total run num改为下面注释的部分
    # total_run_num = len(sample_paths) * len(feedback_noise_rates) * len(
    #     label_noise_rates) * len(methods) * len(
    #     random_merge_numbers) * len(rule_covered_positive_sample_rate_thresholds)

    # config
    run_time = 1
    # for MDL_RM
    Config.adjust_sample_num = True

    output_path = os.path.join(output_dir,
                               "result_jaccard_similarity_time_use_avg" + str(run_time) + "_" + part_name + ".1.json")

    finished_result_path = os.path.join(output_path_prefix, "experience_parameter_sensibility_result",
                                        "result_jaccard_similarity_time_use_avg" + str(
                                            run_time) + "_" + part_name + ".json")
    result = []
    finished_record_keys = []
    if os.path.exists(finished_result_path):
        result = load_json(finished_result_path)
        finished_record_keys = [(copy.copy(x["scene"]),
                                 copy.copy(x["feedback_noise_rate"]),
                                 copy.copy(x["label_noise_rate"]),
                                 copy.copy(x["method"]),
                                 copy.copy(x["random_merge_number"]),
                                 copy.copy(x["rule_covered_positive_sample_rate_threshold"]))
                                for x in result]
        print("####", len(finished_record_keys))
    real_intentions = Sample.get_real_intent()
    tmp_run_num = 0
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
                    for tmp_random_merge_number in random_merge_numbers:
                        for tmp_threshold in rule_covered_positive_sample_rate_thresholds:
                            # 探究随机合并次数时，将正样本覆盖占比阈值固定为0.3
                            # 探究正样本覆盖占比阈值时，将随机合并次数固定为50
                            # 若要运行两个参数所有组合以确定两者最佳取值，注释此if语句即可
                            if not ((tmp_random_merge_number == 10 and
                                     tmp_threshold in rule_covered_positive_sample_rate_thresholds)
                                    or
                                    (tmp_random_merge_number in random_merge_numbers and tmp_threshold == 0.3)):
                                continue
                            tmp_run_num += 1
                            tmp_record_key = (
                                tmp_sample_name, tmp_feedback_noise_rate,
                                tmp_label_noise_rate, tmp_method, tmp_random_merge_number, tmp_threshold)
                            if tmp_record_key in finished_record_keys:
                                print(
                                    f"running: {part_name} - {tmp_run_num}/{total_run_num} - {tmp_sample_name} - "
                                    f"{tmp_feedback_noise_rate} - {tmp_label_noise_rate} - "
                                    f"{tmp_method} - {tmp_random_merge_number} - {tmp_threshold} - skip")
                                finished_record_keys.remove(tmp_record_key)
                                continue

                            print(
                                f"running: {part_name} - {tmp_run_num}/{total_run_num} - {tmp_sample_name} - "
                                f"{tmp_feedback_noise_rate} - {tmp_label_noise_rate} - "
                                f"{tmp_method} - {tmp_random_merge_number} - {tmp_threshold}")
                            all_jaccard_index = []
                            all_intention_similarity = []
                            all_precision = []
                            all_recall = []
                            all_time_use = []
                            all_rules = []
                            all_encoding_length_compression_rates = []
                            for i in range(run_time):
                                time01 = time.time()
                                intention_result = None
                                method_result = None
                                if tmp_method == "IM":
                                    # method_result = Run_MDL_RM.get_intention_by_MDL_RM_r(samples,
                                    #                                                      tmp_random_merge_number,
                                    #                                                      tmp_threshold)
                                    # intention_result = Run_MDL_RM.result_to_intention(method_result)
                                    tmp_result, usetime = Run_Map_RIR.run_IM(samples, tmp_threshold,
                                                                             tmp_random_merge_number)
                                    time02 = time.time()
                                    intention_result = []
                                    # 转换成字典格式
                                    for sub_intent in tmp_result.intention:
                                        real_sub_intent = {"positive": sub_intent.positive_sub_Intention,
                                                           "negative": sub_intent.negative_sub_Intention}
                                        intention_result.append(real_sub_intent)
                                if tmp_method == "IM_without_Apriori":
                                    # method_result = Run_MDL_RM.get_intention_by_MDL_RM_r(samples,
                                    #                                                      tmp_random_merge_number,
                                    #                                                      tmp_threshold)
                                    # intention_result = Run_MDL_RM.result_to_intention(method_result)
                                    tmp_result, usetime = Filter_without_FIM.run_IM_without_Apriori_method2(samples,
                                                                                                            tmp_threshold,
                                                                                                            tmp_random_merge_number)
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

                                # tmp_encoding_length_compression_rates = method_result[1] / method_result[2]

                                all_intention_similarity.append(best_map_average_semantic_similarity)
                                all_jaccard_index.append(jaccard_index)
                                all_precision.append(precision)
                                all_recall.append(recall)
                                all_rules.append(intention_result)
                                # all_encoding_length_compression_rates.append(tmp_encoding_length_compression_rates)

                            tmp_result = {"scene": tmp_sample_name,
                                          "feedback_noise_rate": tmp_feedback_noise_rate,
                                          "label_noise_rate": tmp_label_noise_rate,
                                          "method": tmp_method,
                                          "random_merge_number": tmp_random_merge_number,
                                          "rule_covered_positive_sample_rate_threshold": tmp_threshold,
                                          "time_use": all_time_use,
                                          "jaccard_index": all_jaccard_index,
                                          "intention_similarity": all_intention_similarity,
                                          "precision": all_precision,
                                          "recall": all_recall,
                                          "extracted_rules_json": all_rules,
                                          "encoding_length_compression_rates": all_encoding_length_compression_rates}
                            result.append(tmp_result)
                            tmp_round_num += 1
                            if tmp_round_num == auto_save_threshold:
                                save_as_json(result, output_path)
                                tmp_round_num = 0
    save_as_json(result, output_path)


# take scene, sample level noise rate, label level noise rate, use sub intention order constraint or not,
#   method as variable and record the time use, jaccard index and rules(in json_str).
def experience_get_all_samples_result():
    # Run_MDL_RM.TAG_RECORD_MERGE_PROCESS = False
    # get sample paths
    # samples_dir = os.path.join("../../../resources/samples", "scenes_negative")
    samples_dir = os.path.join("../../../resources/samples/scenes_v5", "scenes_v5")
    sample_names = os.listdir(samples_dir)
    sample_paths = {}
    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    sample_names = list(filter(lambda x: (404 <= int(x[5:]) <= 600), sample_names))
    sample_names.sort()
    print(len(sample_names), sample_names)

    # split all samples name to several parts, and calculate every part with a thread
    part_num = 1  # 12 process
    split_size = len(sample_names) // part_num + 1
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


def get_data_frame_of_comparison_result(all_data):
    # {(tmp_scene, tmp_method, tmp_sub_intention_ordered):{"scenes": set(), "time_use":[[]], "jaccard_index": [[]]}}
    data_frames_data = {}
    for tmp_data in all_data:
        tmp_scene_name = tmp_data["scene"]
        tmp_sub_scene_number_str = tmp_data["scene"][6:]
        tmp_scene = Result_Analysis.get_scene_v4_4(tmp_scene_name)

        # tmp_method = str(tmp_data["rule_covered_positive_sample_rate_threshold"])
        tmp_method = tmp_data["method"]
        tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
        tmp_label_noise_rate = tmp_data["label_noise_rate"]
        tmp_df_row_index = Result_Analysis.feedback_noise_rateS_REVERSE.index(tmp_feedback_noise_rate)
        tmp_df_column_index = Result_Analysis.label_noise_rateS.index(tmp_label_noise_rate)
        tmp_df_key = (tmp_scene, tmp_method)
        if tmp_df_key in data_frames_data:
            pass
        else:
            tmp_df_template = []
            for tmp_i in range(len(Result_Analysis.feedback_noise_rateS)):
                tmp_row = []
                for tmp_j in range(len(Result_Analysis.label_noise_rateS)):
                    tmp_row.append(0)
                tmp_df_template.append(tmp_row)

            tmp_df_scenes = set()  # 11, 12, 13, 21, 22, 23, ..., 351, 352, 353
            tmp_df_scenes_list = []
            tmp_df = {"scenes": tmp_df_scenes, "scenes_list": tmp_df_scenes_list}
            for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
                if tmp_index in tmp_data:
                    tmp_df_of_tmp_index = copy.deepcopy(tmp_df_template)
                    tmp_df[tmp_index] = tmp_df_of_tmp_index

            data_frames_data[tmp_df_key] = tmp_df

        tmp_df = data_frames_data[tmp_df_key]
        tmp_df["scenes"].add(tmp_sub_scene_number_str)
        tmp_df["scenes_list"].append(
            [tmp_sub_scene_number_str, tmp_method, tmp_feedback_noise_rate, tmp_label_noise_rate])

        for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
            if tmp_index in tmp_data:
                tmp_index_value = tmp_data[tmp_index]
                tmp_df_of_tmp_index = tmp_df[tmp_index]
                tmp_df_of_tmp_index[tmp_df_row_index][tmp_df_column_index] += tmp_index_value

    data_frames = {}
    df_average_by_scene = {}  # df average by main scenes like 无意图,单意图单维度,单意图多维度,多意图单维度,多意图多维度
    scene_set = set()  #
    for tmp_df_key in data_frames_data:
        tmp_df = data_frames_data[tmp_df_key]
        tmp_df_scenes = tmp_df["scenes"]
        tmp_df_scenes_list = tmp_df["scenes_list"]
        tmp_df_result = {}
        tmp_df_of_indexes_average = {}
        for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
            if tmp_index in tmp_df:
                tmp_df_of_tmp_index = tmp_df[tmp_index]
                tmp_df_of_tmp_index_average = np.array(tmp_df_of_tmp_index) / len(tmp_df_scenes)

                print(f"# {tmp_df_key}, {tmp_index}, {len(tmp_df_scenes)}, {len(tmp_df_scenes_list)}")
                print(tmp_df_of_tmp_index_average)
                tmp_df_of_indexes_average[tmp_index] = copy.deepcopy(tmp_df_of_tmp_index_average)
                tmp_df_of_tmp_index_average_result = pd.DataFrame(tmp_df_of_tmp_index_average,
                                                                  index=Result_Analysis.feedback_noise_rateS_REVERSE_STR,
                                                                  columns=Result_Analysis.label_noise_rateS_STR)
                tmp_df_result[tmp_index] = tmp_df_of_tmp_index_average_result

        data_frames[tmp_df_key] = tmp_df_result

        # calculate average for every method
        tmp_scene, tmp_method = tmp_df_key
        scene_set.add(tmp_scene)
        tmp_key = tmp_method
        if tmp_key in df_average_by_scene:
            tmp_method_average_df = df_average_by_scene[tmp_key]
            for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
                if tmp_index in tmp_method_average_df:
                    tmp_method_average_df_tmp_index = tmp_method_average_df[tmp_index]
                    tmp_method_average_df_tmp_index += tmp_df_of_indexes_average[tmp_index]
        else:
            tmp_method_average_df = {}
            for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
                if tmp_index in tmp_df_of_indexes_average:
                    tmp_method_average_df[tmp_index] = copy.deepcopy(tmp_df_of_indexes_average[tmp_index])
            df_average_by_scene[tmp_key] = tmp_method_average_df

    # add average df
    for tmp_key in df_average_by_scene:
        tmp_method = tmp_key
        tmp_method_average_df = df_average_by_scene[tmp_key]
        tmp_df_result = {}
        for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
            if tmp_index in tmp_method_average_df:
                tmp_method_average_df[tmp_index] /= len(scene_set)
                tmp_method_average_df_of_tmp_index_result = pd.DataFrame(tmp_method_average_df[tmp_index],
                                                                         index=Result_Analysis.feedback_noise_rateS_REVERSE_STR,
                                                                         columns=Result_Analysis.label_noise_rateS_STR)
                tmp_df_result[tmp_index] = tmp_method_average_df_of_tmp_index_result
        data_frames[("Average", tmp_method)] = tmp_df_result

    # add_difference_for_comparison_result

    # minuend_method = "0.3"
    minuend_method = "IM"
    different_data_frames = {}
    for tmp_df_key in data_frames:
        tmp_scene, tmp_method = tmp_df_key
        if tmp_method == minuend_method:
            continue

        other_tmp_df_key = (tmp_scene, minuend_method)
        tmp_data_frame = data_frames[tmp_df_key]
        other_tmp_data_frame = data_frames[other_tmp_df_key]

        different_df = {}
        for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
            if tmp_index in tmp_data_frame:
                tmp_df_of_tmp_index = tmp_data_frame[tmp_index]
                other_tmp_df_of_tmp_index = other_tmp_data_frame[tmp_index]
                different_of_tmp_index = other_tmp_df_of_tmp_index - tmp_df_of_tmp_index
                different_df[tmp_index] = different_of_tmp_index
        tmp_different_key = (tmp_scene, tmp_method + "_different")
        different_data_frames[tmp_different_key] = different_df
    for tmp_df_key in different_data_frames:
        data_frames[tmp_df_key] = different_data_frames[tmp_df_key]
    return data_frames


def export_heatmap_data_of_comparison_experience(data_frames):
    # evaluate_result_dir_path = os.path.join(analysis_result_dir_path, "experience_comparison_result")
    evaluate_result_dir_path = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result")
    if not os.path.exists(evaluate_result_dir_path):
        os.mkdir(evaluate_result_dir_path)
    scenes = ["单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度", "含负向意图", "Average"]
    # scenes = ["单意图单维度", "单意图多维度", "多意图单维度",  "Average"]
    # COMPARISON_DIFFERENT_METHODS = ["0.1_different", "0.2_different"]
    COMPARISON_DIFFERENT_METHODS = ["IM_without_Apriori_different"]
    index_to_plot = ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]
    # COMPARISON_METHODS = ["0.1","0.2","0.3"]
    COMPARISON_METHODS = ["IM", "IM_without_Apriori"]
    for tmp_var in index_to_plot:
        tmp_all_values_dict = {}
        for tmp_scene in scenes:
            tmp_row_index = scenes.index(tmp_scene)
            for tmp_method in COMPARISON_METHODS + COMPARISON_DIFFERENT_METHODS:
                if tmp_method.endswith("different"):
                    tmp_method_index = COMPARISON_DIFFERENT_METHODS.index(tmp_method)
                    tmp_column_index = tmp_method_index + len(COMPARISON_METHODS)
                else:
                    tmp_column_index = COMPARISON_METHODS.index(tmp_method)
                tmp_df_key = (tmp_scene, tmp_method)
                if tmp_var not in data_frames[tmp_df_key]:
                    continue
                tmp_df = data_frames[tmp_df_key][tmp_var]
                tmp_all_values_dict[(tmp_row_index, tmp_column_index)] = tmp_df

        tmp_values_output_path = os.path.join(evaluate_result_dir_path, tmp_var + "_heatmap_values.csv")
        tmp_index_values = []
        for i in range(len(scenes) * len(Result_Analysis.feedback_noise_rateS)):
            tmp_index_values.append([None] * ((len(COMPARISON_METHODS) + len(COMPARISON_DIFFERENT_METHODS))
                                              * len(Result_Analysis.label_noise_rateS)))
        for tmp_key in tmp_all_values_dict:
            tmp_row_index, tmp_column_index = tmp_key
            tmp_df = tmp_all_values_dict[tmp_key]
            for sub_row_index in range(len(Result_Analysis.feedback_noise_rateS)):
                for sub_column_index in range(len(Result_Analysis.label_noise_rateS)):
                    tmp_df_value = tmp_df.iat[sub_row_index, sub_column_index]
                    tmp_value_row_index = tmp_row_index * len(
                        Result_Analysis.feedback_noise_rateS) + sub_row_index
                    tmp_value_column_index = tmp_column_index * len(
                        Result_Analysis.label_noise_rateS) + sub_column_index
                    tmp_index_values[tmp_value_row_index][tmp_value_column_index] = tmp_df_value
        Result_Analysis.FileUtil.save_as_csv(tmp_index_values, tmp_values_output_path)

        if Result_Analysis.EXPORT_FIGURES:
            Result_Analysis.plt.rcParams['font.sans-serif'] = ['Times New Roman']
            tmp_fig, tmp_axes = Result_Analysis.plt.subplots(figsize=(10, 10), dpi=250, nrows=4,
                                                                    ncols=len(COMPARISON_METHODS) * 2 - 1)
            for tmp_scene in scenes:
                tmp_row_index = scenes.index(tmp_scene)
                for tmp_method in COMPARISON_METHODS + COMPARISON_DIFFERENT_METHODS:

                    if tmp_method.endswith("different"):
                        tmp_method_index = COMPARISON_DIFFERENT_METHODS.index(tmp_method)
                        tmp_column_index = tmp_method_index + len(COMPARISON_METHODS)
                    else:
                        tmp_column_index = COMPARISON_METHODS.index(tmp_method)
                    tmp_ax = tmp_axes[tmp_row_index][tmp_column_index]

                    tmp_df_key = (tmp_scene, tmp_method)
                    # if has not data,
                    if tmp_df_key not in data_frames:
                        if tmp_scene != "Average":
                            tmp_ax.set_xticklabels([])  # 设置x轴图例为空值
                        if tmp_method != COMPARISON_METHODS[0] or not tmp_method.endswith("different"):
                            tmp_ax.set_yticklabels([])  # 设置y轴图例为空值
                        continue
                    if tmp_var not in data_frames[tmp_df_key]:
                        continue
                    tmp_df = data_frames[tmp_df_key][tmp_var]

                    tmp_all_values_dict[(tmp_row_index, tmp_column_index)] = tmp_df

                    if tmp_method.endswith("different"):
                        vmax = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var + " different"]["vmax"]
                        vmin = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var + " different"]["vmin"]
                    else:
                        vmax = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var]["vmax"]
                        vmin = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var]["vmin"]
                    Result_Analysis.sns.heatmap(tmp_df, linewidths=0.05, ax=tmp_ax, vmax=vmax, vmin=vmin,
                                                       cmap="coolwarm", cbar=False)
                    # tmp_ax.set_xlabel('标签噪声比例')
                    # tmp_ax.set_ylabel('反馈噪声比例')
                    tmp_ax.set_xlabel('Label noise ratio')
                    tmp_ax.set_ylabel('Sample noise ratio')
                    tmp_ax.set_xticklabels(labels=[0, 0.2, 0.4, 0.6, 0.8, 1], rotation=90, fontsize=12)  # 将字体进行旋转
                    tmp_ax.set_yticklabels(labels=[0.3, 0.2, 0.1, 0], rotation=360, fontsize=12)
                    if tmp_scene != "Average":
                        tmp_ax.set_xlabel('')
                        tmp_ax.set_xticklabels([])  # 设置x轴图例为空值
                        tmp_ax.xaxis.set_ticks([])
                    if tmp_method != COMPARISON_METHODS[0]:
                        tmp_ax.set_ylabel('')
                        tmp_ax.set_yticklabels([])  # 设置y轴图例为空值
                        tmp_ax.yaxis.set_ticks([])
            tmp_output_path = os.path.join(evaluate_result_dir_path, tmp_var + "_heatmap.png")
            # plt.show()
            Result_Analysis.plt.savefig(tmp_output_path, dpi=1000)
            Result_Analysis.plt.close()


def export_heatmap_data_of_comparison_experience(data_frames):
    # evaluate_result_dir_path = os.path.join(analysis_result_dir_path, "experience_comparison_result")
    evaluate_result_dir_path = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result")
    if not os.path.exists(evaluate_result_dir_path):
        os.mkdir(evaluate_result_dir_path)
    scenes = ["单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度", "含负向意图", "Average"]
    # COMPARISON_DIFFERENT_METHODS = ["0.1_different","0.2_different" ]
    COMPARISON_DIFFERENT_METHODS = ["IM_without_Apriori_different"]
    index_to_plot = ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]
    # COMPARISON_METHODS = ["0.1","0.2","0.3" ]
    COMPARISON_METHODS = ["IM", "IM_without_Apriori"]
    for tmp_var in index_to_plot:
        tmp_all_values_dict = {}
        for tmp_scene in scenes:
            tmp_row_index = scenes.index(tmp_scene)
            for tmp_method in COMPARISON_METHODS + COMPARISON_DIFFERENT_METHODS:
                if tmp_method.endswith("different"):
                    tmp_method_index = COMPARISON_DIFFERENT_METHODS.index(tmp_method)
                    tmp_column_index = tmp_method_index + len(COMPARISON_METHODS)
                else:
                    tmp_column_index = COMPARISON_METHODS.index(tmp_method)
                tmp_df_key = (tmp_scene, tmp_method)
                if tmp_var not in data_frames[tmp_df_key]:
                    continue
                tmp_df = data_frames[tmp_df_key][tmp_var]
                tmp_all_values_dict[(tmp_row_index, tmp_column_index)] = tmp_df

        tmp_values_output_path = os.path.join(evaluate_result_dir_path, tmp_var + "_heatmap_values.csv")
        tmp_index_values = []
        for i in range(len(scenes) * len(Result_Analysis.feedback_noise_rateS)):
            tmp_index_values.append([None] * ((len(COMPARISON_METHODS) + len(COMPARISON_DIFFERENT_METHODS))
                                              * len(Result_Analysis.label_noise_rateS)))
        for tmp_key in tmp_all_values_dict:
            tmp_row_index, tmp_column_index = tmp_key
            tmp_df = tmp_all_values_dict[tmp_key]
            for sub_row_index in range(len(Result_Analysis.feedback_noise_rateS)):
                for sub_column_index in range(len(Result_Analysis.label_noise_rateS)):
                    tmp_df_value = tmp_df.iat[sub_row_index, sub_column_index]
                    tmp_value_row_index = tmp_row_index * len(
                        Result_Analysis.feedback_noise_rateS) + sub_row_index
                    tmp_value_column_index = tmp_column_index * len(
                        Result_Analysis.label_noise_rateS) + sub_column_index
                    tmp_index_values[tmp_value_row_index][tmp_value_column_index] = tmp_df_value
        Result_Analysis.FileUtil.save_as_csv(tmp_index_values, tmp_values_output_path)

        if Result_Analysis.EXPORT_FIGURES:
            Result_Analysis.plt.rcParams['font.sans-serif'] = ['Times New Roman']
            tmp_fig, tmp_axes = Result_Analysis.plt.subplots(figsize=(10, 10), dpi=250, nrows=6,
                                                                    ncols=len(COMPARISON_METHODS) * 2 - 1)
            for tmp_scene in scenes:
                tmp_row_index = scenes.index(tmp_scene)
                for tmp_method in COMPARISON_METHODS + COMPARISON_DIFFERENT_METHODS:

                    if tmp_method.endswith("different"):
                        tmp_method_index = COMPARISON_DIFFERENT_METHODS.index(tmp_method)
                        tmp_column_index = tmp_method_index + len(COMPARISON_METHODS)
                    else:
                        tmp_column_index = COMPARISON_METHODS.index(tmp_method)
                    tmp_ax = tmp_axes[tmp_row_index][tmp_column_index]

                    tmp_df_key = (tmp_scene, tmp_method)
                    # if has not data,
                    if tmp_df_key not in data_frames:
                        if tmp_scene != "Average":
                            tmp_ax.set_xticklabels([])  # 设置x轴图例为空值
                        if tmp_method != COMPARISON_METHODS[0] or not tmp_method.endswith("different"):
                            tmp_ax.set_yticklabels([])  # 设置y轴图例为空值
                        continue
                    if tmp_var not in data_frames[tmp_df_key]:
                        continue
                    tmp_df = data_frames[tmp_df_key][tmp_var]

                    tmp_all_values_dict[(tmp_row_index, tmp_column_index)] = tmp_df

                    if tmp_method.endswith("different"):
                        vmax = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var + " different"]["vmax"]
                        vmin = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var + " different"]["vmin"]
                    else:
                        vmax = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var]["vmax"]
                        vmin = Result_Analysis.LEGEND_NAMES[tmp_var][tmp_var]["vmin"]
                    Result_Analysis.sns.heatmap(tmp_df, linewidths=0.05, ax=tmp_ax, vmax=vmax, vmin=vmin,
                                                       cmap="coolwarm", cbar=False)
                    # tmp_ax.set_xlabel('标签噪声比例')
                    # tmp_ax.set_ylabel('反馈噪声比例')
                    tmp_ax.set_xlabel('Label noise ratio')
                    tmp_ax.set_ylabel('Sample noise ratio')
                    tmp_ax.set_xticklabels(labels=[0, 0.2, 0.4, 0.6, 0.8, 1], rotation=90, fontsize=12)  # 将字体进行旋转
                    tmp_ax.set_yticklabels(labels=[0.3, 0.2, 0.1, 0], rotation=360, fontsize=12)
                    if tmp_scene != "Average":
                        tmp_ax.set_xlabel('')
                        tmp_ax.set_xticklabels([])  # 设置x轴图例为空值
                        tmp_ax.xaxis.set_ticks([])
                    if tmp_method != COMPARISON_METHODS[0]:
                        tmp_ax.set_ylabel('')
                        tmp_ax.set_yticklabels([])  # 设置y轴图例为空值
                        tmp_ax.yaxis.set_ticks([])
            tmp_output_path = os.path.join(evaluate_result_dir_path, tmp_var + "_heatmap.png")
            # plt.show()
            Result_Analysis.plt.savefig(tmp_output_path, dpi=1000)
            Result_Analysis.plt.close()


def get_data_frame_of_noise_result(all_data):
    # {(tmp_scene, tmp_method, tmp_sub_intention_ordered):{"scenes": set(), "time_use":[[]], "jaccard_index": [[]]}}
    evaluate_result_dir_path = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result")
    if not os.path.exists(evaluate_result_dir_path):
        os.mkdir(evaluate_result_dir_path)
    data_frames_data = {}
    for tmp_data in all_data:
        tmp_scene_name = tmp_data["scene"]
        tmp_sub_scene_number_str = tmp_data["scene"][6:]
        tmp_scene = Result_Analysis.get_scene_v4_4(tmp_scene_name)

        # tmp_method = tmp_data["rule_covered_positive_sample_rate_threshold"]
        tmp_method = tmp_data["method"]
        tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
        tmp_label_noise_rate = tmp_data["label_noise_rate"]
        # # tmp_df_row_index = MDL_RM_Result_Analysis.feedback_noise_rateS_REVERSE.index(tmp_feedback_noise_rate)
        # # tmp_df_column_index = MDL_RM_Result_Analysis.label_noise_rateS.index(tmp_label_noise_rate)

        # tmp_df_key = (tmp_feedback_noise_rate, tmp_method)
        tmp_df_key = (tmp_label_noise_rate, tmp_method)
        if tmp_df_key in data_frames_data:
            pass
        else:
            # tmp_df_template = []
            # for tmp_i in range(len(MDL_RM_Result_Analysis.feedback_noise_rateS)):
            #     tmp_row = []
            #     for tmp_j in range(len(MDL_RM_Result_Analysis.label_noise_rateS)):
            #         tmp_row.append(0)
            #     tmp_df_template.append(tmp_row)

            tmp_df_scenes = set()  # 11, 12, 13, 21, 22, 23, ..., 351, 352, 353
            tmp_df_scenes_list = []
            tmp_df = {"scenes": tmp_df_scenes, "scenes_list": tmp_df_scenes_list}
            for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
                if tmp_index in tmp_data:
                    tmp_df_of_tmp_index = []
                    tmp_df[tmp_index] = tmp_df_of_tmp_index

            data_frames_data[tmp_df_key] = tmp_df

        # data_frames_data[(tmp_feedback_noise_rate, tmp_method)]={tmp_index:[]}

        tmp_df = data_frames_data[tmp_df_key]
        tmp_df["scenes"].add(tmp_sub_scene_number_str)
        # tmp_df["scenes_list"].append(
        #     [tmp_sub_scene_number_str, tmp_method, tmp_feedback_noise_rate, tmp_label_noise_rate])

        for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
            if tmp_index in tmp_data:
                tmp_index_value = tmp_data[tmp_index]
                tmp_df_of_tmp_index = tmp_df[tmp_index]
                tmp_df_of_tmp_index.append(tmp_index_value)

    # "time_use", "jaccard_index", "intention_similarity" with diff noise and diff method
    for tmp_index in ["time_use", "jaccard_index", "intention_similarity"]:
        data = []
        # for tmp_noise in MDL_RM_Result_Analysis.feedback_noise_rateS:
        for tmp_noise in Result_Analysis.label_noise_rateS:
            # for tmp_method in [0.1, 0.2, 0.3]:
            for tmp_method in ["IM", "IM_without_Apriori"]:
                tmp_key = (tmp_noise, tmp_method)
                tmp_data_values = data_frames_data[tmp_key][tmp_index]
                data.append(tmp_data_values)
        values_output_path = os.path.join(evaluate_result_dir_path, f"{tmp_index}_label_noise_values.csv")
        tmp_values = np.transpose(data[:]).tolist()
        FileUtil.save_as_csv(tmp_values, values_output_path)
    # all index in diff method
    data = []
    for tmp_index in ["time_use", "jaccard_index", "intention_similarity"]:
        for tmp_method in ["IM", "IM_without_Apriori"]:
            # for tmp_method in [0.1, 0.2, 0.3]:
            tmp_data_values = []
            # for tmp_noise in MDL_RM_Result_Analysis.feedback_noise_rateS:
            for tmp_noise in Result_Analysis.label_noise_rateS:
                tmp_key = (tmp_noise, tmp_method)
                for tmp_data_frames in data_frames_data[tmp_key][tmp_index]:
                    tmp_data_values.append(tmp_data_frames)
            data.append(tmp_data_values)
    values_output_path = os.path.join(evaluate_result_dir_path, "all_index_values.csv")
    tmp_values = np.transpose(data[:]).tolist()
    # for i in range(len(data)):
    #     print(len(data[i]))
    FileUtil.save_as_csv(tmp_values, values_output_path)

    # #noise-free的情况
    # for tmp_index in ["time_use", "jaccard_index", "intention_similarity"]:
    #     data = []
    #     tmp_noise = 0
    #     for tmp_method in [0.1, 0.2, 0.3]:
    #         tmp_key = (tmp_noise, tmp_method)
    #         tmp_data_values = data_frames_data[tmp_key][tmp_index]
    #         data.append(tmp_data_values)
    #
    #     values_output_path = os.path.join(evaluate_result_dir_path, f"{tmp_index}_noise_values.csv")
    #     tmp_values = np.transpose(data[:]).tolist()
    #     FileUtil.save_as_csv(tmp_values, values_output_path)
    # data = []
    # for tmp_index in ["time_use", "jaccard_index", "intention_similarity"]:
    #     for tmp_method in [0.1, 0.2, 0.3]:
    #         tmp_data_values = []
    #         tmp_noise = 0
    #         tmp_key = (tmp_noise, tmp_method)
    #         for tmp_data_frames in data_frames_data[tmp_key][tmp_index]:
    #             tmp_data_values.append(tmp_data_frames)
    #         data.append(tmp_data_values)
    # values_output_path = os.path.join(evaluate_result_dir_path, "all_index_values.csv")
    # tmp_values = np.transpose(data[:]).tolist()
    # # for i in range(len(data)):
    # #     print(len(data[i]))
    # FileUtil.save_as_csv(tmp_values, values_output_path)


def export_boxplot_different_scene(all_data):
    EXPORT_FIGURES = True
    output_dir = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result")
    methods = ["IM", "IM_without_Apriori"]
    indexes = ["jaccard_index", "intention_similarity", "time_use"]
    scene_names = ["单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度", "含负向意图"]
    data_cache = {}
    for tmp_index in indexes:
        tmp_index_data_cache = {}
        for tmp_scene in scene_names:
            for tmp_method in methods:
                tmp_key = (tmp_scene, tmp_method)
                tmp_index_data_cache[tmp_key] = []
        data_cache[tmp_index] = tmp_index_data_cache
    for tmp_data in all_data:
        tmp_method = tmp_data["method"]
        tmp_scene_name = tmp_data["scene"]
        tmp_scene = get_scene_v4_4(tmp_scene_name)
        # tmp_scene = get_scene_v4_3(tmp_scene_name)
        if tmp_scene is None:
            print(tmp_scene_name)
            continue
        if tmp_method not in methods:
            continue
        for tmp_index in indexes:
            tmp_index_values = tmp_data[tmp_index]
            tmp_key = (tmp_scene, tmp_method)
            # all_scene_key = ("Average", tmp_method)
            data_cache[tmp_index][tmp_key].append(tmp_index_values)
            # data_cache[tmp_index][all_scene_key] += tmp_index_values
    # 绘制图像
    for tmp_index in indexes:
        data = []
        labels = []
        for tmp_scene in scene_names:
            for tmp_method in methods:
                tmp_key = (tmp_scene, tmp_method)
                tmp_data_values = data_cache[tmp_index][tmp_key]
                data.append(tmp_data_values)
                labels.append(tmp_method)
        values_output_path = os.path.join(output_dir, f"{tmp_index}_boxplot_values.csv")
        tmp_values = np.transpose(data[:]).tolist()
        # for i in range(len(data)):
        #     print(len(data[i]))
        FileUtil.save_as_csv(tmp_values, values_output_path)

        if EXPORT_FIGURES:
            fig, ax = plt.subplots(figsize=(10, 3), dpi=250)
            # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
            plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 显示中文
            plt.rcParams['axes.unicode_minus'] = False
            # colors = ['pink', 'lightblue'] * 6
            # colors = ['blue', 'orange'] * 6
            colors = ['#6098f9', '#fb6f66'] * 5
            positions = [0, 0.6, 1.5, 2.1, 3, 3.6, 4.5, 5.1, 6.0, 6.6]
            bplot = ax.boxplot(data,
                               positions=positions,
                               # 用positions参数设置各箱线图的位置
                               patch_artist=True,
                               sym='',
                               vert=True,
                               notch=None,
                               medianprops={'color': 'red'})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            # ax.set_xticklabels(labels)  # 设置x轴刻度标
            ylabels = {"jaccard_index": "Jaccard index", "time_use": "time_use/s", "intention_similarity": "BMASS",
                       "precision": "precision", "recall": "recall"
                       }
            plt.ylabel(ylabels[tmp_index], size=22)
            plt.tick_params(labelsize=18)  # 刻度字体大小13
            # plt.show()
            output_path = os.path.join(output_dir, f"{tmp_index}_boxplot.png")
            plt.savefig(output_path)
            plt.close()


def export_comparison_result_heatmap():
    result_paths = [os.path.join(output_path_prefix, "experience_Apriori_feasibility_result",
                                 "result_jaccard_similarity_time_use_avg1_PART_0.1.json")]
    result = []
    for tmp_path in result_paths:
        tmp_result = load_json(tmp_path)
        for tmp_data in tmp_result:
            for tmp_index in ["time_use", "jaccard_index", "min_encoding_length", "time_use_sample_enhancement",
                              "time_use_merge", "time_use_others", "precision", "intention_similarity", "recall"]:
                if tmp_index in tmp_data:
                    tmp_index_json = tmp_data.pop(tmp_index)
                    tmp_data[tmp_index] = np.mean(tmp_index_json)
                    # tmp_data[tmp_index + "_std"] = np.std(np.array(tmp_index_json), ddof=1)  # std
            tmp_extracted_rules = tmp_data["extracted_rules_json"]
            tmp_extracted_rules_num = [len(x)
                                       if x != [{"Spatial": "America", "Theme": "ThemeRoot",
                                                 "MapMethod": "MapMethodRoot", "MapContent": "Thing"}]
                                       else 0 for x in tmp_extracted_rules]
            tmp_data["sub_intention_num"] = sum(tmp_extracted_rules_num) / len(tmp_extracted_rules_num)

        result += tmp_result
    test_all_data = result
    get_data_frame_of_noise_result(test_all_data)
    # test_data_frames = get_data_frame_of_comparison_result(test_all_data)
    # # #
    # export_heatmap_data_of_comparison_experience(test_data_frames)
    # export_boxplot_different_scene(test_all_data)


def export_boxplot_different_method(all_data):
    evaluate_result_dir_path = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result")
    if not os.path.exists(evaluate_result_dir_path):
        os.mkdir(evaluate_result_dir_path)
    data_frames_data = {}
    for tmp_data in all_data:
        tmp_scene_name = tmp_data["scene"]
        tmp_sub_scene_number_str = tmp_data["scene"][6:]
        tmp_scene = Result_Analysis.get_scene_v4_4(tmp_scene_name)

        tmp_support = str(tmp_data["rule_covered_positive_sample_rate_threshold"])
        tmp_method = tmp_data["method"]
        tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
        tmp_label_noise_rate = tmp_data["label_noise_rate"]
        # distinct without_Apriori method
        # if tmp_method == "IM":
        #     tmp_df_key = (tmp_label_noise_rate, tmp_support)
        # else:
        #     tmp_df_key = (tmp_label_noise_rate, tmp_support+"without")
        if tmp_method == "IM":
            tmp_df_key = (tmp_feedback_noise_rate, tmp_support)
        else:
            tmp_df_key = (tmp_feedback_noise_rate, tmp_support+"without")

        if tmp_df_key in data_frames_data:
            pass
        else:
            # tmp_df_template = []
            # for tmp_i in range(len(MDL_RM_Result_Analysis.feedback_noise_rateS)):
            #     tmp_row = []
            #     for tmp_j in range(len(MDL_RM_Result_Analysis.label_noise_rateS)):
            #         tmp_row.append(0)
            #     tmp_df_template.append(tmp_row)

            tmp_df_scenes = set()  # 11, 12, 13, 21, 22, 23, ..., 351, 352, 353
            tmp_df_scenes_list = []
            tmp_df = {"scenes": tmp_df_scenes, "scenes_list": tmp_df_scenes_list}
            for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
                if tmp_index in tmp_data:
                    tmp_df_of_tmp_index = []
                    tmp_df[tmp_index] = tmp_df_of_tmp_index
            data_frames_data[tmp_df_key] = tmp_df

        tmp_df = data_frames_data[tmp_df_key]
        tmp_df["scenes"].add(tmp_sub_scene_number_str)
        # tmp_df["scenes_list"].append(
        #     [tmp_sub_scene_number_str, tmp_method, tmp_feedback_noise_rate, tmp_label_noise_rate])

        for tmp_index in ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]:
            if tmp_index in tmp_data:
                tmp_index_value = tmp_data[tmp_index]
                tmp_df_of_tmp_index = tmp_df[tmp_index]
                tmp_df_of_tmp_index.append(tmp_index_value)

    # "time_use", "jaccard_index", "intention_similarity" with diff noise and diff method
    for tmp_index in ["time_use", "jaccard_index", "intention_similarity"]:
        data = []
        for tmp_noise in Result_Analysis.feedback_noise_rateS:
        # for tmp_noise in MDL_RM_Result_Analysis.label_noise_rateS:
            # for tmp_method in [0.1, 0.2, 0.3]:
            for tmp_method1 in ["0.1", "0.1without","0.2", "0.2without","0.3","0.3without"]:
                tmp_key = (tmp_noise, tmp_method1)
                tmp_data_values = data_frames_data[tmp_key][tmp_index]
                data.append(tmp_data_values)
        values_output_path = os.path.join(evaluate_result_dir_path, f"{tmp_index}_method_sample_noise_values.csv")
        tmp_values = np.transpose(data[:]).tolist()
        FileUtil.save_as_csv(tmp_values, values_output_path)
    # all index in diff method
    data = []
    for tmp_index in ["time_use", "jaccard_index", "intention_similarity"]:
        for tmp_method in ["0.1", "0.1without","0.2", "0.2without","0.3","0.3without"]:
            # for tmp_method in [0.1, 0.2, 0.3]:
            tmp_data_values = []
            for tmp_noise in Result_Analysis.feedback_noise_rateS:
            # for tmp_noise in MDL_RM_Result_Analysis.label_noise_rateS:
                tmp_key = (tmp_noise, tmp_method)
                for tmp_data_frames in data_frames_data[tmp_key][tmp_index]:
                    tmp_data_values.append(tmp_data_frames)
            data.append(tmp_data_values)
    values_output_path = os.path.join(evaluate_result_dir_path, "all_method_index_values.csv")
    tmp_values = np.transpose(data[:]).tolist()
    # for i in range(len(data)):
    #     print(len(data[i]))
    FileUtil.save_as_csv(tmp_values, values_output_path)


def export_comparison_result_support_and_filter():
    result_paths_support = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result",
                                        "result_jaccard_similarity_time_use_avg1_PART_0.1_q.json")
    result_paths_filter = os.path.join(output_path_prefix, "experience_Apriori_feasibility_result",
                                       "result_jaccard_similarity_time_use_avg1_PART_0.1_filter.json")
    # result_paths = [os.path.join(output_path_prefix, "experience_Apriori_feasibility_result",
    #                              "result_jaccard_similarity_time_use_avg1_PART_0.1_filter.json"),
    #                 os.path.join(output_path_prefix, "experience_Apriori_feasibility_result",
    #                              "result_jaccard_similarity_time_use_avg1_PART_0.1_auto.json")]
    result = []

    tmp_result = load_json(result_paths_support)
    print("result size", len(tmp_result))
    for tmp_data in tmp_result:
        for tmp_index in ["time_use", "jaccard_index", "min_encoding_length", "time_use_sample_enhancement",
                          "time_use_merge", "time_use_others", "precision", "intention_similarity", "recall"]:
            if tmp_index in tmp_data:
                tmp_index_json = tmp_data.pop(tmp_index)
                tmp_data[tmp_index] = np.mean(tmp_index_json)
                # tmp_data[tmp_index + "_std"] = np.std(np.array(tmp_index_json), ddof=1)  # std
        tmp_extracted_rules = tmp_data["extracted_rules_json"]
        tmp_extracted_rules_num = [len(x)
                                   if x != [{"Spatial": "America", "Theme": "ThemeRoot",
                                             "MapMethod": "MapMethodRoot", "MapContent": "Thing"}]
                                   else 0 for x in tmp_extracted_rules]
        tmp_data["sub_intention_num"] = sum(tmp_extracted_rules_num) / len(tmp_extracted_rules_num)
    result += tmp_result
    print("result size", len(result))
    def is_03_filter(tmp_data1):
        is_add = True
        tmp_sub_scene_number = int(tmp_data1["scene"][5:])
        tmp_scene = None
        # if 501 <= tmp_sub_scene_number <= 593:
        #     sample_names.append(sample_name)
        if 380 <= tmp_sub_scene_number <= 387:
            is_add = False
        # if tmp_data1["rule_covered_positive_sample_rate_threshold"] == 0.3:
        #     is_add = False
        return is_add

    # in result_paths_support already record support:0.3,method:IM
    result = list(filter(is_03_filter, result))
    print("result size",len(result))
    # tmp_result = load_json(result_paths_filter)
    # for tmp_data in tmp_result:
    #
    #     for tmp_index in ["time_use", "jaccard_index", "min_encoding_length", "time_use_sample_enhancement",
    #                       "time_use_merge", "time_use_others", "precision", "intention_similarity", "recall"]:
    #         if tmp_index in tmp_data:
    #             tmp_index_json = tmp_data.pop(tmp_index)
    #             tmp_data[tmp_index] = np.mean(tmp_index_json)
    #             # tmp_data[tmp_index + "_std"] = np.std(np.array(tmp_index_json), ddof=1)  # std
    #     tmp_extracted_rules = tmp_data["extracted_rules_json"]
    #     tmp_extracted_rules_num = [len(x)
    #                                if x != [{"Spatial": "America", "Theme": "ThemeRoot",
    #                                          "MapMethod": "MapMethodRoot", "MapContent": "Thing"}]
    #                                else 0 for x in tmp_extracted_rules]
    #     tmp_data["sub_intention_num"] = sum(tmp_extracted_rules_num) / len(tmp_extracted_rules_num)
    # result += tmp_result

    test_all_data = result
    # get_data_frame_of_noise_result(test_all_data)
    # test_data_frames = get_data_frame_of_comparison_result(test_all_data)
    # # #
    # export_heatmap_data_of_comparison_experience(test_data_frames)
    export_boxplot_different_method(test_all_data)


if __name__ == "__main__":
    experience_get_all_samples_result()
    # export_comparison_result_heatmap()
    # export_comparison_result_support_and_filter()

    print("Aye")
