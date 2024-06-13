# coding=UTF8
# compare with
#   DTHF：Kinnunen N . Decision tree learning with hierarchical features.  2018.
#   RuleGO：Gruca A ,  Sikora M . Data- and expert-driven rule induction and filtering framework for functional
#       interpretation and description of gene sets[J]. Journal of Biomedical Semantics, 2017, 8(1).


import os
import time
import multiprocessing
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from Map_RIR.src.main import Intention
from Map_RIR.src.main.experience.Result_Analysis import code_version, calculate_transformed_data, \
    get_data_frame_of_comparison_result, LEGEND_NAMES, feedback_noise_rateS_REVERSE_STR, label_noise_rateS_STR, \
    get_scene_v4_4
from Map_RIR.src.main.samples.input import Sample
from Map_RIR.src.main.samples.input.Data import Data
from Map_RIR.src.main.intention_recognition import Config, DTHF_Kinnunen2018, RuleGO_Gruca2017, Run_Map_RIR
from Map_RIR.src.main.experience import Evaluation_IM
from Map_RIR.src.main.util import FileUtil
from Map_RIR.src.main.util.FileUtil import save_as_json, load_json
import numpy as np

sample_version = Intention.__sample_version__
output_path_prefix = os.path.join("/Map_RIR\\result", "scenes_" +
                                  "v4_5" + "_" + Intention.__version__)
if not os.path.exists(output_path_prefix):
    os.mkdir(output_path_prefix)
auto_save_threshold = 100
EXPORT_FIGURES = True

def experience_get_specific_samples_result(part_name, sample_paths):
    # import time, datetime
    output_dir = os.path.join(output_path_prefix, "experience_negative_comparison_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tmp_round_num = 0  # 为了自动保存而设置的变量
    # set parameters
    feedback_noise_rates = [0, 0.1, 0.2, 0.3]
    label_noise_rates = [0, 0.2, 0.4, 0.6, 0.8, 1]
    methods = ["IM_MDL", "IM_MDL_P"]
    total_run_num = len(sample_paths) * len(feedback_noise_rates) * len(label_noise_rates) * len(methods)

    # config
    run_time = 1
    # for IM_MDL
    tmp_threshold = Config.rule_covered_positive_sample_rate_threshold
    Config.adjust_sample_num =True
    Config.TAG_RECORD_MERGE_PROCESS = False



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
        # print("##### ", tmp_sample_name, " #####")
        tmp_sample_path_prefix = sample_paths[tmp_sample_name]

        for tmp_sample_level_noise_rate in feedback_noise_rates:
            tmp_sample_level_noise_rate_str = "_S" + str(tmp_sample_level_noise_rate).replace(".", "p")
            for tmp_label_level_noise_rate in label_noise_rates:
                tmp_label_level_noise_rate_str = "_L" + str(tmp_label_level_noise_rate).replace(".", "p")

                if tmp_sample_level_noise_rate == 0:
                    if tmp_label_level_noise_rate == 0:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix, "final_samples.json")
                    else:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_label_level_noise_rate_str+".json")
                else:
                    if tmp_label_level_noise_rate == 0:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_sample_level_noise_rate_str+".json")
                    else:
                        tmp_sample_path = os.path.join(tmp_sample_path_prefix,
                                                       "noise_samples" + tmp_sample_level_noise_rate_str
                                                       + tmp_label_level_noise_rate_str+".json")
                # load sample data
                # docs, real_intention = Sample.load_sample_from_file(tmp_sample_path)

                docs, real_intention = Sample.load_real_intents(real_intentions, tmp_sample_path, tmp_sample_name)

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
                        tmp_sample_name, tmp_sample_level_noise_rate, tmp_label_level_noise_rate, tmp_method)
                    if tmp_record_key in finished_record_keys:
                        continue

                    print(
                        f"running: {part_name} - {tmp_run_num}/{total_run_num} - {tmp_sample_name} - "
                        f"{tmp_sample_level_noise_rate} - {tmp_label_level_noise_rate} - {tmp_method}")
                    all_jaccard_index = []
                    all_intention_similarity = []
                    all_precision = []
                    all_recall = []
                    all_time_use = []
                    all_rules = []
                    for i in range(run_time):
                        time01 = time.time()
                        intention_result = []
                        if tmp_method == "IM_MDL":
                            tmp_result, usetime = ItemsMerger.run_IM(samples, 0.3, 15)
                            time02 = time.time()
                            intention_result = []
                            # 转换成字典格式
                            for sub_intent in tmp_result.intention:
                                real_sub_intent = {"positive": sub_intent.positive_sub_Intention,
                                                   "negative": sub_intent.negative_sub_Intention}
                                intention_result.append(real_sub_intent)

                        elif tmp_method == "IM_MDL_P":
                            tmp_result, usetime = ItemsMerger.run_IM_P(samples, 0.3, 15)
                            time02 = time.time()
                            intention_result = []
                            # 转换成字典格式
                            for sub_intent in tmp_result.intention:
                                real_sub_intent = {"positive": sub_intent.positive_sub_Intention,
                                                   "negative": sub_intent.negative_sub_Intention}
                                intention_result.append(real_sub_intent)

                        all_time_use.append(time02 - time01)
                        jaccard_index = Evaluation_IM.get_jaccard_index(samples, real_intention,
                                                                          intention_result,
                                                                          Data.Ontologies, Data.Ontology_Root)
                        best_map_average_semantic_similarity = \
                            Evaluation_IM.get_intention_similarity(intention_result, real_intention,
                                                                     direct_ancestors, ontology_root,
                                                                     information_content)
                        # if tmp_method == "MDL_RM_r":
                        #     print("time", all_time_use)
                        #     print("intention_result",intention_result)
                        #     print("real_intention", real_intention)
                        #     print("best_map_average_semantic_similarity", best_map_average_semantic_similarity)
                        precision = Evaluation_IM.get_precision(samples, real_intention,
                                                                  intention_result,
                                                                  Data.Ontologies, Data.Ontology_Root)
                        recall = Evaluation_IM.get_recall(samples, real_intention,
                                                            intention_result,
                                                            Data.Ontologies, Data.Ontology_Root)
                        all_intention_similarity.append(best_map_average_semantic_similarity)
                        all_jaccard_index.append(jaccard_index)
                        all_precision.append(precision)
                        all_recall.append(recall)
                        all_rules.append(intention_result)

                    tmp_result = {"scene": tmp_sample_name, "feedback_noise_rate": tmp_sample_level_noise_rate,
                                  "label_noise_rate": tmp_label_level_noise_rate,
                                  "method": tmp_method,
                                  "rule_covered_positive_sample_rate_threshold": tmp_threshold,
                                  "time_use": all_time_use,
                                  "jaccard_index": all_jaccard_index,
                                  "intention_similarity": all_intention_similarity,
                                  "precision": all_precision,
                                  "recall": all_recall,
                                  "extracted_rules_json": all_rules}
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
    # samples_dir = os.path.join("./../../../resources/samples", "scenes_" + "negative")
    samples_dir = os.path.join("./../../../resources/samples", "scenes_" + "v4_5")
    all_sample_names = os.listdir(samples_dir)
    sample_names = []
    sample_paths = {}
    for sample_name in all_sample_names:
        tmp_sub_scene_number = int(sample_name[5:])
        tmp_scene = None
        # if 501 <= tmp_sub_scene_number <= 593:
        #     sample_names.append(sample_name)
        if 0 <= tmp_sub_scene_number <= 408:
            sample_names.append(sample_name)
    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    print(len(sample_names), sample_names)
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


def add_difference_for_comparison_result(data_frames):
    minuend_method = "IM_MDL"
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


def export_heatmap_data_of_comparison_experience(data_frames):
    #evaluate_result_dir_path = os.path.join(analysis_result_dir_path, "experience_comparison_result")
    evaluate_result_dir_path = os.path.join(output_path_prefix, "experience_negative_comparison_result")
    if not os.path.exists(evaluate_result_dir_path):
        os.mkdir(evaluate_result_dir_path)
    scenes = ["负-单意图单维度", "负-单意图多维度", "负-多意图单维度", "负-多意图多维度", "Average"]
    COMPARISON_METHODS = ["IM_MDL", "IM_MDL_P"]
    COMPARISON_DIFFERENT_METHODS = ["IM_MDL_P_different"]
    inex_to_plot = ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]
    feedback_noise_rateS = [0, 0.1, 0.2, 0.3]
    feedback_noise_rateS_REVERSE = [0.3, 0.2, 0.1, 0]
    feedback_noise_rateS_REVERSE_STR = [str(x) for x in feedback_noise_rateS_REVERSE]
    label_noise_rateS = [0, 0.2, 0.4, 0.6, 0.8, 1]
    label_noise_rateS_STR = [str(x) for x in label_noise_rateS]






    for tmp_var in inex_to_plot:
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
        for i in range(len(scenes) * len(feedback_noise_rateS)):
            tmp_index_values.append([None] * ((len(COMPARISON_METHODS) + len(COMPARISON_DIFFERENT_METHODS))
                                              * len(label_noise_rateS)))
        for tmp_key in tmp_all_values_dict:
            tmp_row_index, tmp_column_index = tmp_key
            tmp_df = tmp_all_values_dict[tmp_key]
            for sub_row_index in range(len(feedback_noise_rateS)):
                for sub_column_index in range(len(label_noise_rateS)):
                    tmp_df_value = tmp_df.iat[sub_row_index, sub_column_index]
                    tmp_value_row_index = tmp_row_index * len(feedback_noise_rateS) + sub_row_index
                    tmp_value_column_index = tmp_column_index * len(label_noise_rateS) + sub_column_index
                    tmp_index_values[tmp_value_row_index][tmp_value_column_index] = tmp_df_value
        FileUtil.save_as_csv(tmp_index_values, tmp_values_output_path)

        if EXPORT_FIGURES:
            tmp_fig, tmp_axes = plt.subplots(figsize=(10, 10), dpi=250, nrows=5, ncols=len(COMPARISON_METHODS) * 2 - 1)
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
                        vmax = LEGEND_NAMES[tmp_var][tmp_var + " different"]["vmax"]
                        vmin = LEGEND_NAMES[tmp_var][tmp_var + " different"]["vmin"]
                    else:
                        vmax = LEGEND_NAMES[tmp_var][tmp_var]["vmax"]
                        vmin = LEGEND_NAMES[tmp_var][tmp_var]["vmin"]
                    sns.heatmap(tmp_df, linewidths=0.05, ax=tmp_ax, vmax=vmax, vmin=vmin, cmap="coolwarm", cbar=False)
                    tmp_ax.set_xlabel('标签噪声比例')
                    tmp_ax.set_ylabel('反馈噪声比例')
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
            plt.savefig(tmp_output_path)
            plt.close()


def draw_legends():
    legends_dir_path = os.path.join(output_path_prefix, "experience_negative_comparison_result", "legends")
    if not os.path.exists(legends_dir_path):
        os.mkdir(legends_dir_path)
    for tmp_var in LEGEND_NAMES:
        tmp_var_legend_names = LEGEND_NAMES[tmp_var]
        for tmp_legend_name in tmp_var_legend_names:
            vmax = tmp_var_legend_names[tmp_legend_name]["vmax"]
            vmin = tmp_var_legend_names[tmp_legend_name]["vmin"]
            major_locator = tmp_var_legend_names[tmp_legend_name]["major_locator"]
            tmp_output_path = os.path.join(legends_dir_path, tmp_legend_name)
            # 绘制耗时图例
            tmp_fig, tmp_ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
            tmp_data = [([vmax] * 3 + [vmin] * 3)] * 4
            tmp_df = pd.DataFrame(tmp_data, index=feedback_noise_rateS_REVERSE_STR,
                                  columns=label_noise_rateS_STR)
            h = sns.heatmap(tmp_df, linewidths=0.05, ax=tmp_ax, vmax=vmax, vmin=vmin, cmap="coolwarm", cbar=False)
            cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
            cb.ax.tick_params(labelsize=28)  # 设置colorbar刻度字体大小。
            cb.ax.yaxis.set_major_locator(MultipleLocator(major_locator))
            # cb.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            tmp_ax.tick_params(axis='y', labelsize=20)  # y轴
            tmp_ax.tick_params(axis='x', labelsize=20)  # y轴
            plt.savefig(tmp_output_path)
            plt.close()


def export_comparison_result_heatmap():
    result_paths = [os.path.join(output_path_prefix, "experience_negative_comparison_result",
                                 "result_jaccard_similarity_time_use_avg1_PART_0.json")]
    test_all_data = []
    for tmp_path in result_paths:
        tmp_result = load_json(tmp_path)
        calculate_transformed_data(tmp_result)
        test_all_data += tmp_result
    # test_all_data = get_all_data()
    test_data_frames = get_data_frame_of_comparison_result(test_all_data)
    add_difference_for_comparison_result(test_data_frames)
    export_heatmap_data_of_comparison_experience(test_data_frames)
    if EXPORT_FIGURES:
        draw_legends()



def experience_get_samples_result():
    # load sample paths
    # sample_paths = get_sample_paths()
    samples_dir = os.path.join("./../../../resources/samples", "scenes_" + sample_version)

    all_sample_names = os.listdir(samples_dir)
    sample_names = []
    sample_paths = {}
    for sample_name in all_sample_names:
        tmp_sub_scene_number = int(sample_name[5:])
        tmp_scene = None
        if 501 <= tmp_sub_scene_number <= 593:
            sample_names.append(sample_name)

    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    print(len(sample_names), sample_names)
    part_num = 1  # 12 process
    split_size = len(sample_names) // part_num + 1
    all_samples_parts = []
    for i in range(len(sample_names)):
        tmp_sample_part = {}
        for j in range(1):
            tmp_sample_name = sample_names.pop(0)
            tmp_sample_part[tmp_sample_name] = sample_paths[tmp_sample_name]
        all_samples_parts.append(tmp_sample_part)
    last_sample_part = {}
    for tmp_sample_name in sample_names:
        last_sample_part[tmp_sample_name] = sample_paths[tmp_sample_name]
    all_samples_parts.append(last_sample_part)
    experience_get_specific_samples_result("PART_36", all_samples_parts[0])

def export_effectiveness_result_boxplot():
    all_data_path = os.path.join(output_path_prefix, "experience_negative_comparison_result",
                                 "result_jaccard_similarity_time_use_avg1_PART_0.json")
    all_data = load_json(all_data_path)
    output_dir = os.path.join(output_path_prefix, "experience_effectiveness_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    methods = ["IM_MDL", "IM_MDL_P"]
    indexes = ["jaccard_index", "intention_similarity", "time_use",
               "precision", "recall"]
    scene_names = ["负-单意图单维度", "负-单意图多维度", "负-多意图单维度", "负-多意图多维度", "Average"]
    scene_names = ["单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度", "Average"]
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
        if tmp_scene is None or tmp_scene == "无意图":
            print(tmp_scene_name)
            continue
        if tmp_method not in methods:
            continue
        for tmp_index in indexes:
            tmp_index_values = tmp_data[tmp_index]
            tmp_key = (tmp_scene, tmp_method)
            all_scene_key = ("Average", tmp_method)
            data_cache[tmp_index][tmp_key] += tmp_index_values
            data_cache[tmp_index][all_scene_key] += tmp_index_values
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
        FileUtil.save_as_csv(tmp_values, values_output_path)

        if EXPORT_FIGURES:
            fig, ax = plt.subplots(figsize=(10, 3), dpi=250)

            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
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

            ylabels = {"jaccard_index": "Jaccard index", "time_use": "time_use/s", "intention_similarity": "BMASS",
                       "precision": "precision", "recall": "recall"
                       }
            plt.ylabel(ylabels[tmp_index], size=22)
            plt.tick_params(labelsize=18)
            # plt.show()
            output_path = os.path.join(output_dir, f"{tmp_index}_boxplot.png")
            plt.savefig(output_path)
            plt.close()

if __name__ == "__main__":
    experience_get_all_samples_result()
    #export_comparison_result_heatmap()
    #experience_get_samples_result()
    export_effectiveness_result_boxplot()
    print("Aye")
