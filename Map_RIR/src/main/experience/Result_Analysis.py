# 实验结果分析
from matplotlib.ticker import MultipleLocator

from Map_RIR.src.main import Intention
from Map_RIR.src.main.util import FileUtil
from Map_RIR.src.main.util.FileUtil import load_json, save_as_json, save_as_csv

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import copy

plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文不乱码，微软雅黑

sample_version = Intention.__sample_version__
code_version = Intention.__version__
result_dirname = "scenes_" + sample_version + "_" + code_version
result_path_prefix = os.path.join("../../../result", result_dirname)
sample_path_prefix = os.path.join("./../../../resources/samples", "scenes_" + sample_version)
analysis_result_dir_path = os.path.join(result_path_prefix, "experience_analysis_result")
if not os.path.exists(analysis_result_dir_path):
    os.mkdir(analysis_result_dir_path)

# params
feedback_noise_rateS = [0, 0.1, 0.2, 0.3]
feedback_noise_rateS_REVERSE = [0.3, 0.2, 0.1, 0]
feedback_noise_rateS_REVERSE_STR = [str(x) for x in feedback_noise_rateS_REVERSE]
label_noise_rateS = [0, 0.2, 0.4, 0.6, 0.8, 1]
label_noise_rateS_STR = [str(x) for x in label_noise_rateS]
COMPARISON_METHODS = ["MDL_RM_r", "RuleGO", "DTHF"]
LEGEND_NAMES = {
    "time_use": {
        "time_use different": {"vmax": 0.2, "vmin": -0.2, "major_locator": 0.1},
        "time_use": {"vmax": 0.8, "vmin": 0, "major_locator": 0.2},
        "time_use_only_MDL_RM_method2": {"vmax": 0.2, "vmin": 0, "major_locator": 0.05},
    },
    "jaccard_index": {
        "jaccard_index different": {"vmax": 0.2, "vmin": -0.2, "major_locator": 0.1},
        "jaccard_index": {"vmax": 1, "vmin": 0, "major_locator": 0.2}
    },
    "intention_similarity": {
        "intention_similarity different": {"vmax": 0.5, "vmin": -0.5, "major_locator": 0.25},
        "intention_similarity": {"vmax": 1, "vmin": 0, "major_locator": 0.2},
    },
    "precision": {
        "precision different": {"vmax": 0.5, "vmin": -0.5, "major_locator": 0.25},
        "precision": {"vmax": 1, "vmin": 0, "major_locator": 0.2},
    },
    "recall": {
        "recall different": {"vmax": 0.5, "vmin": -0.5, "major_locator": 0.25},
        "recall": {"vmax": 1, "vmin": 0, "major_locator": 0.2}
    }
}
EXPORT_FIGURES = True


# 计算解析结果，例如计算平均值
def calculate_transformed_data(transformed_data):
    for tmp_data in transformed_data:
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


# 获取所有可用的数据
def get_all_data():
    result_paths = [os.path.join(result_path_prefix, "experience_comparison_result",
                                 "result_jaccard_similarity_time_use_avg5")]
    result = []
    for tmp_path in result_paths:
        tmp_result = load_json(tmp_path)
        calculate_transformed_data(tmp_result)
        result += tmp_result
    return result


# 根据4_4以上版本的样本集
def get_scene_v4_4(scene_name):
    tmp_sub_scene_number = int(scene_name[5:])
    tmp_scene = None
    if 11 <= tmp_sub_scene_number <= 13 or 361 <= tmp_sub_scene_number <= 387:
        tmp_scene = "无意图"
    elif 21 <= tmp_sub_scene_number <= 53 or 388 <= tmp_sub_scene_number <= 405:
        tmp_scene = "单意图单维度"
    elif 61 <= tmp_sub_scene_number <= 133 or 151 <= tmp_sub_scene_number <= 163:
        tmp_scene = "单意图多维度"
    elif 171 <= tmp_sub_scene_number <= 263:
        tmp_scene = "多意图单维度"
    elif 271 <= tmp_sub_scene_number <= 353 or 406 <= tmp_sub_scene_number <= 408:
        tmp_scene = "多意图多维度"
    elif 501 <= tmp_sub_scene_number <= 599:
        tmp_scene = "含负向意图"
    # elif 501 <= tmp_sub_scene_number <= 509:
    #     tmp_scene = "负-单意图单维度"
    # elif 511 <= tmp_sub_scene_number <= 539:
    #     tmp_scene = "负-单意图多维度"
    # elif 541 <= tmp_sub_scene_number <= 569:
    #     tmp_scene = "负-多意图单维度"
    # elif 571 <= tmp_sub_scene_number <= 599:
    #     tmp_scene = "负-多意图多维度"
    return tmp_scene


def draw_legends():
    legends_dir_path = os.path.join(result_path_prefix, "experience_comparison_result", "legends")
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
            plt.savefig(tmp_output_path, dpi=1000)
            plt.close()


# 合并不同进程的结果
def merge_result(result_to_merge_dirs):
    for tmp_dir in result_to_merge_dirs:
        #all_parts_result_dir = os.path.join(analysis_result_dir_path, tmp_dir)
        all_parts_result_dir = os.path.join(result_path_prefix, tmp_dir)
        output_result_path = analysis_result_dir_path
        if not os.path.exists(output_result_path):
            os.mkdir(output_result_path)
        tmp_all_samples_result = []
        tmp_sub_part_file_names = os.listdir(all_parts_result_dir)
        for tmp_sub_part_file_name in tmp_sub_part_file_names:
            if "PART" not in tmp_sub_part_file_name:
                continue
            print(tmp_sub_part_file_name)
            tmp_sub_part_path = os.path.join(all_parts_result_dir, tmp_sub_part_file_name)
            tmp_sub_part_json = load_json(tmp_sub_part_path)

            for tmp_data in tmp_sub_part_json:
                if "methods" in tmp_data:
                    tmp_data["method"] = tmp_data.pop("methods").replace("MRIEMR", "MDL_RM")
                if "sample_level_noise_rates" in tmp_data:
                    tmp_data["feedback_noise_rate"] = tmp_data.pop("sample_level_noise_rates")
                if "label_level_noise_rates" in tmp_data:
                    tmp_data["label_noise_rate"] = tmp_data.pop("label_level_noise_rates")
                if "max_merge_count" in tmp_data:
                    tmp_data["random_merge_number"] = tmp_data.pop("max_merge_count")
                tmp_all_samples_result.append(tmp_data)

            # tmp_all_samples_result += tmp_sub_part_json
        all_result_path = os.path.join(all_parts_result_dir, "result_jaccard_similarity_time_use_avg5.json")
        save_as_json(tmp_all_samples_result, all_result_path)


def get_data_frame_of_comparison_result(all_data):
    # {(tmp_scene, tmp_method, tmp_sub_intention_ordered):{"scenes": set(), "time_use":[[]], "jaccard_index": [[]]}}
    data_frames_data = {}
    for tmp_data in all_data:
        tmp_scene_name = tmp_data["scene"]
        tmp_sub_scene_number_str = tmp_data["scene"][6:]
        tmp_scene = get_scene_v4_4(tmp_scene_name)

        tmp_method = tmp_data["method"]
        tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
        tmp_label_noise_rate = tmp_data["label_noise_rate"]
        tmp_df_row_index = feedback_noise_rateS_REVERSE.index(tmp_feedback_noise_rate)
        tmp_df_column_index = label_noise_rateS.index(tmp_label_noise_rate)
        tmp_df_key = (tmp_scene, tmp_method)
        if tmp_df_key in data_frames_data:
            pass
        else:
            tmp_df_template = []
            for tmp_i in range(len(feedback_noise_rateS)):
                tmp_row = []
                for tmp_j in range(len(label_noise_rateS)):
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
                                                                  index=feedback_noise_rateS_REVERSE_STR,
                                                                  columns=label_noise_rateS_STR)
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
                                                                         index=feedback_noise_rateS_REVERSE_STR,
                                                                         columns=label_noise_rateS_STR)
                tmp_df_result[tmp_index] = tmp_method_average_df_of_tmp_index_result
        data_frames[("Average", tmp_method)] = tmp_df_result
    return data_frames


def add_difference_for_comparison_result(data_frames):
    # minuend_method = "0.3"
    minuend_method = "MDL_RM_r"
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


# 输出用于绘制对比实验（实验2）结果热力图的数据
# 若开启将EXPORT_FIGURES设置为True，则使用matplot输出热力图到结果目录
def export_heatmap_data_of_comparison_experience(data_frames):
    #evaluate_result_dir_path = os.path.join(analysis_result_dir_path, "experience_comparison_result")
    evaluate_result_dir_path = os.path.join(result_path_prefix, "experience_comparison_result")
    if not os.path.exists(evaluate_result_dir_path):
        os.mkdir(evaluate_result_dir_path)
    scenes = ["单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度", "Average"]
    COMPARISON_DIFFERENT_METHODS = ["RuleGO_different", "DTHF_different", ]
    index_to_plot = ["time_use", "jaccard_index", "intention_similarity", "precision", "recall"]
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
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
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
            tmp_output_path = os.path.join(evaluate_result_dir_path, tmp_var + "_heatmap.pdf")
            # plt.show()
            plt.savefig(tmp_output_path, dpi=1000)
            plt.close()


# 输出用于绘制热力图的对比实验结果，也可通过设置EXPORT_FIGURES=True直接绘制热力图
def export_comparison_result_heatmap():
    test_all_data = get_all_data()
    test_data_frames = get_data_frame_of_comparison_result(test_all_data)
    add_difference_for_comparison_result(test_data_frames)
    export_heatmap_data_of_comparison_experience(test_data_frames)
    if EXPORT_FIGURES:
        draw_legends()


# 数组都是按从小到大排过序的
# 计算中位数
def count_median(data):
    if len(data) % 2 == 0:
        mid = float((data[int(len(data) / 2)] + data[int(len(data) / 2) - 1])) / 2
    else:
        mid = data[len(data) / 2]
    return mid


# 计算上下四分位数
def count_quartiles(data):
    q1_index = int(1 + (float(len(data)) - 1) * 1 / 4)
    q3_index = int(1 + (float(len(data)) - 1) * 3 / 4)
    q1, q3 = data[q1_index], data[q3_index]

    return q1, q3


# x from 1 - 100
def count_x_tiles(data, x):
    t1_index = int((float(len(data))) * x / 100.0)
    t3_index = int((float(len(data))) * ((100 - x) / 100.0)) - 1
    t1, t3 = data[t1_index], data[t3_index]

    return t1, t3


# 计算上下边缘
def count_margin(q1, q3):
    q4 = q1 - 1.5 * (q3 - q1)
    q5 = q3 + 1.5 * (q3 - q1)
    return q4, q5


# 获取上下异常值
def get_exception_values(data):
    # 排序
    data = sorted(data)
    q1, q3 = count_quartiles(data)
    q4, q5 = count_margin(q1, q3)
    # print(q5, q3, q1, q4)
    exception_values1 = list(filter(lambda x: x < q4, data))
    exception_values2 = list(filter(lambda x: x > q5, data))
    return exception_values1, exception_values2


# 输出用于绘制对比实验（实验2）结果箱线图的数据
# 如果设置EXPORT_FIGURES设置为True，则时候matlibplt绘制图像到结果目录
def export_comparison_result_boxplot():
    # 整理结果
    all_data_path = os.path.join(result_path_prefix, "experience_comparison_result",
                                 "result_jaccard_similarity_time_use_avg5")
    all_data = load_json(all_data_path)
    output_dir = os.path.join(result_path_prefix, "experience_comparison_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    methods = ["MDL_RM_r", "RuleGO", "DTHF"]
    indexes = ["jaccard_index", "intention_similarity", "time_use", "precision", "recall"]

    target_scene_names = ["无意图", "单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度"]
    target_scene_title = "所有场景"
    target_feedback_noise_rates = [0, 0.1, 0.2, 0.3]
    target_label_noise_rates = [0, 0.2, 0.4, 0.6, 0.8, 1]
    target_feedback_noise_rate_title = "All"
    target_label_noise_rate_title = "All"
    data_cache = {}
    for tmp_index in indexes:
        tmp_index_data_cache = {}
        for tmp_method in methods:
            tmp_index_data_cache[tmp_method] = []
        data_cache[tmp_index] = tmp_index_data_cache
    for tmp_data in all_data:
        tmp_method = tmp_data["method"]
        tmp_scene_name = tmp_data["scene"]
        tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
        tmp_label_noise_rate = tmp_data["label_noise_rate"]
        if not (
                get_scene_v4_4(tmp_scene_name) in target_scene_names and
                # tmp_scene_name in target_scene_names and
                tmp_feedback_noise_rate in target_feedback_noise_rates and
                tmp_label_noise_rate in target_label_noise_rates):
            continue
        for tmp_index in indexes:
            tmp_index_values = tmp_data[tmp_index]
            data_cache[tmp_index][tmp_method] += tmp_index_values
    # 为每个index绘制箱线图
    for tmp_index in data_cache:
        tmp_index_data_cache = data_cache[tmp_index]
        labels = list(tmp_index_data_cache.keys())
        values = [tmp_index_data_cache[x] for x in labels]
        values_output_path = os.path.join(output_dir,
                                          f"{tmp_index}_{target_scene_title}_{target_feedback_noise_rate_title}_"
                                          f"{target_label_noise_rate_title}_boxplot_values")
        tmp_values = [labels]
        tmp_values += np.transpose(values).tolist()
        FileUtil.save_as_csv(tmp_values, values_output_path)

        if EXPORT_FIGURES:
            plt.figure(figsize=(8, 4), dpi=150)  # 设置画布的尺寸
            # title = f"{tmp_index}\nscene={target_scene_title}," \
            #         f"feedback_noise_rate={target_feedback_noise_rate_title}," \
            #         f"label_noise_rate={target_label_noise_rate_title}"
            # plt.title(title, fontsize=20)  # 标题，并设定字号大小
            plt.tick_params(labelsize=18)  # 刻度字体大小13
            #colors = ['#f56c00', '#595959', '#10a37f']
            colors = ['#f9cdb8', '#fddebb', '#f6e0ab']
            f = plt.boxplot(values, labels=labels, patch_artist=True, sym='')  # grid=False：代表不显示背景中的网格线
            for patch, color in zip(f['boxes'], colors):
                patch.set_facecolor(color)
            #plt.xticks(ticks=[1, 2, 3], labels=["Our", "RuleGO", "DTHF"])
            plt.xticks([1, 2, 3],["Our", "RuleGO", "DTHF"])
            ylabels = {"jaccard_index": "Jaccard", "time_use": "time/s", "intention_similarity": "BMASS",
                       "precision": "查准率", "recall": "查全率"}
            plt.ylabel(ylabels[tmp_index], size=18)
            output_path = os.path.join(output_dir,
                                       f"{tmp_index}_{target_scene_title}_{target_feedback_noise_rate_title}_"
                                       f"{target_label_noise_rate_title}_boxplot.png")
            plt.show()  # 显示图像
            #plt.savefig(output_path)
            plt.close()


# 绘制箱线图形式的RuleGO调参结果
def plot_RuleGO_tuning_hyperparameters_result_boxplot():
    all_data_path = os.path.join(result_path_prefix, "experience_RuleGO_tuning_hyperparameters_result",
                                 "result_jaccard_similarity_time_use_avg5.json")
    all_data = load_json(all_data_path)
    output_dir = os.path.join(analysis_result_dir_path, "experience_RuleGO_tuning_hyperparameters_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # keys: scene, feedback_noise_rates, label_noise_rates, method, min_support,
    # 添加min_support信息
    min_support_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    i = 4
    target_scene_names = ["无意图", "单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度"][i:i + 1]
    # target_scene_title = "所有场景"
    target_scene_title = target_scene_names[0]
    target_feedback_noise_rates = [0, 0.1, 0.2, 0.3]
    target_label_noise_rates = [0, 0.2, 0.4, 0.6, 0.8, 1]
    target_feedback_noise_rate_title = "All"
    target_label_noise_rate_title = "All"
    print(len(all_data))
    for i, tmp_data in enumerate(all_data):
        tmp_min_support = min_support_values[i % 10]
        tmp_data["min_support"] = tmp_min_support
        tmp_data["use_similarity_criteria"] = False
    all_data_save_path = os.path.join(result_path_prefix, "experience_RuleGO_tuning_hyperparameters_result",
                                      "result_jaccard_similarity_time_use_avg50_with_min_support.json")
    save_as_json(all_data, all_data_save_path)

    # 整理数据
    indexes = ["jaccard_index", "intention_similarity", "time_use"]
    data_cache = {}
    for tmp_index in indexes:
        tmp_index_data_cache = {}
        for tmp_min_support in min_support_values:
            tmp_index_data_cache[tmp_min_support] = []
        data_cache[tmp_index] = tmp_index_data_cache
    for tmp_data in all_data:
        tmp_min_support = tmp_data["min_support"]
        tmp_scene_name = tmp_data["scene"]
        tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
        tmp_label_noise_rate = tmp_data["label_noise_rate"]
        if not (
                get_scene_v4_4(tmp_scene_name) in target_scene_names and
                # tmp_scene_name in target_scene_names and
                tmp_feedback_noise_rate in target_feedback_noise_rates and
                tmp_label_noise_rate in target_label_noise_rates):
            continue
        for tmp_index in indexes:
            tmp_index_values = tmp_data[tmp_index]
            data_cache[tmp_index][tmp_min_support] += tmp_index_values

    # 为每个index绘制箱线图
    for tmp_index in data_cache:
        tmp_index_data_cache = data_cache[tmp_index]

        for tmp_method in tmp_index_data_cache:
            tmp_index_data_cache[tmp_method] = sorted(tmp_index_data_cache[tmp_method])
            q1, q3 = count_quartiles(tmp_index_data_cache[tmp_method])
            q4, q5 = count_margin(q1, q3)
            tmp_index_evs1, tmp_index_evs2 = get_exception_values(tmp_index_data_cache[tmp_method])

            print(tmp_index, len(tmp_index_data_cache[tmp_method]))
            print("\t", q4, q1, q3, q5)
            print("\t", tmp_method, len(tmp_index_evs1), len(tmp_index_evs2))
            print("\taverage", np.mean(tmp_index_data_cache[tmp_method]))
        print("")

        plt.figure(figsize=(15, 5))  # 设置画布的尺寸
        title = f"{tmp_index}\nscene={target_scene_title}," \
                f"feedback_noise_rate={target_feedback_noise_rate_title}," \
                f"label_noise_rate={target_label_noise_rate_title}"
        plt.title(title, fontsize=20)  # 标题，并设定字号大小
        labels = list(tmp_index_data_cache.keys())
        values = [tmp_index_data_cache[x] for x in labels]
        print([len(x) for x in values])
        plt.boxplot(values, labels=labels)  # grid=False：代表不显示背景中的网格线
        # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
        output_path = os.path.join(output_dir,
                                   f"{tmp_index}_{target_scene_title}_"
                                   f"{target_feedback_noise_rate_title}_"
                                   f"{target_label_noise_rate_title}.png")
        # plt.show()  # 显示图像
        plt.savefig(output_path)
        plt.close()


# Output statistics before and after sample augmentation Each column of the csv file is, in order, the number of
# positive/negative samples, the mean, the standard deviation, the maximum value, the minimum value, the upper bound,
# the upper bound of 80% of the data, the upper quartile, the median, the lower quartile, the lower bound of 80% of
# the data, and the lower bound.
def export_sample_adjusting_experience_result():
    adjust_type = "sample_num"
    adjust_values = [False, True]
    # indexes = ["jaccard_index", "intention_similarity", "encoding_length_compression_rates", "precision", "recall"]
    indexes = ["jaccard_index", "intention_similarity",  "precision", "recall"]
    scene_names = ["无意图","单意图单维度", "单意图多维度", "多意图单维度", "多意图多维度", "含负向意图", "所有有意图场景"]
    parameter_values = [5, 10, 20, 30, 40, 50, 75, 100, 125, 150]
    data_file_path = os.path.join(result_path_prefix,
                                  f"experience_sample_adjusting_result/result_jaccard_similarity_time_use_avg5.json")
    all_data = load_json(data_file_path)
    data_cache = {}
    for tmp_parameter_value in parameter_values:
        for tmp_adjust in adjust_values:
            for tmp_index in indexes:
                for tmp_scene_title in scene_names:
                    tmp_key = (tmp_index, tmp_scene_title, tmp_parameter_value, tmp_adjust)
                    data_cache[tmp_key] = []
                tmp_key_all_scene = (tmp_index, "所有有意图场景", tmp_parameter_value, tmp_adjust)
                data_cache[tmp_key_all_scene] = []
    for tmp_data in all_data:
        tmp_scene = tmp_data["scene"]
        tmp_scene_title = get_scene_v4_4(tmp_scene)
        if tmp_data["method"] != "IM_MDL":
            continue
        tmp_adjust = tmp_data["adjust"]
        tmp_parameter_value = tmp_data["parameter_value"]
        for tmp_index in indexes:
            tmp_key = (tmp_index, tmp_scene_title, tmp_parameter_value, tmp_adjust)
            tmp_key_all_scene = (tmp_index, "所有有意图场景", tmp_parameter_value, tmp_adjust)
            tmp_index_values = tmp_data[tmp_index]
            data_cache[tmp_key] += tmp_index_values
            if tmp_scene_title != "无意图":
                data_cache[tmp_key_all_scene] += tmp_index_values
    # for tmp_key in data_cache.keys():
    #     print(tmp_key)
    output_dir = os.path.join(analysis_result_dir_path, "sample_adjusting_experience")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for tmp_index in indexes:
        tmp_index_values = []
        for tmp_parameter_value in parameter_values:
            tmp_row_values = [tmp_parameter_value]
            tmp_scene_title ="所有有意图场景"
            for tmp_adjust in adjust_values:
                tmp_key = (tmp_index, tmp_scene_title, tmp_parameter_value, tmp_adjust)
                tmp_values = data_cache[tmp_key]
                if len(tmp_values) == 0:
                    print(tmp_key)
                tmp_index_avg = np.mean(tmp_values)
                tmp_index_std = np.std(tmp_values, ddof=1)
                tmp_index_max = np.max(tmp_values)
                tmp_index_min = np.min(tmp_values)
                tmp_values.sort()
                tmp_index_medium = count_median(tmp_values)
                tmp_index_q1, tmp_index_q3 = count_quartiles(tmp_values)
                tmp_index_t1, tmp_index_t3 = count_x_tiles(tmp_values, 10)  # 80% 在这两个数之内
                tmp_index_up_margin, tmp_index_down_margin = count_margin(tmp_index_q1, tmp_index_q3)
                tmp_row_values.append(tmp_index_avg)
                tmp_row_values.append(tmp_index_std)
                tmp_row_values.append(tmp_index_max)
                tmp_row_values.append(tmp_index_min)
                tmp_row_values.append(tmp_index_up_margin)
                tmp_row_values.append(tmp_index_t3)
                tmp_row_values.append(tmp_index_q3)
                tmp_row_values.append(tmp_index_medium)
                tmp_row_values.append(tmp_index_q1)
                tmp_row_values.append(tmp_index_t1)
                tmp_row_values.append(tmp_index_down_margin)
            tmp_index_values.append(tmp_row_values)
        print(tmp_index_values)
        output_path = os.path.join(output_dir, f"result_{tmp_index}_{adjust_type}.csv")
        save_as_csv(tmp_index_values, output_path)


# Itera_index sensitivity analysis
def export_parameter_sensibility_experience_result():
    # load result
    output_dir = os.path.join(result_path_prefix, "experience_parameter_sensibility_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    indexes = ["intention_similarity", "jaccard_index",  "time_use"]

    itera_index = [0.5, 1, 5, 10, 15, 20, 25, 30, 35, 40]
    data_file_path = os.path.join(result_path_prefix, "experience_parameter_sensibility_result",
                                  "result_jaccard_similarity_time_use_avg5.json")
    all_data = load_json(data_file_path)

    scenes = ["单意图单维度", "单意图多维度", "多意图单维度",  "多意图多维度", "含负向意图"]
    partition_params = [
        [scenes[0:5], [0, 0.1,0.2, 0.3], "all_", 5 * 30 * 4 * 6],

    ]

    for tmp_partition_param in partition_params:
        tmp_scnens, tmp_feedback_noise_rates, tmp_output_name_prefix, tmp_values_num_pass = tmp_partition_param
        print("\n", tmp_output_name_prefix)
        result_values = {}
        result_data_count = {}
        for tmp_index in indexes:
            tmp_index_result_values = {}
            tmp_index_result_data_count = {}
            for tmp_random_merge_number in itera_index:

                tmp_index_result_values[tmp_random_merge_number] = []
                tmp_index_result_data_count[tmp_random_merge_number] = 0
            result_values[tmp_index] = tmp_index_result_values
            result_data_count[tmp_index] = tmp_index_result_data_count
        scene_names = set()
        num = 0
        for tmp_data in all_data:
            tmp_random_merge_number = tmp_data["random_merge_number"]

            tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
            tmp_rule_covered_positive_sample_rate_threshold = tmp_data["rule_covered_positive_sample_rate_threshold"]
            tmp_scene_name = tmp_data["scene"]
            scene_names.add(tmp_scene_name)
            tmp_scene = get_scene_v4_4(tmp_scene_name)
            if tmp_rule_covered_positive_sample_rate_threshold != 0.3:
                continue
            if tmp_scene in tmp_scnens and tmp_feedback_noise_rate in tmp_feedback_noise_rates:
                num += 1
                for tmp_index in indexes:
                    tmp_data_tmp_index_values = tmp_data[tmp_index]
                    if tmp_random_merge_number in result_values[tmp_index]:
                        result_values[tmp_index][tmp_random_merge_number] += tmp_data_tmp_index_values
                    else:
                        result_values[tmp_index][tmp_random_merge_number] = copy.copy(tmp_data_tmp_index_values)
                    result_data_count[tmp_index][tmp_random_merge_number] += 1

        for tmp_index in result_values.keys():
            tmp_index_result_values = result_values[tmp_index]
            tmp_output_path = os.path.join(output_dir,
                                           f"{tmp_output_name_prefix}{tmp_index}_values_rate_threshold_0p3.csv")
            tmp_result = []
            for tmp_random_merge_number in itera_index:
                tmp_values = tmp_index_result_values[tmp_random_merge_number]
                print(tmp_random_merge_number, len(tmp_values), tmp_values_num_pass)
                tmp_index_avg = np.mean(tmp_values)
                tmp_index_std = np.std(tmp_values, ddof=1)
                tmp_result.append([tmp_random_merge_number, tmp_index_avg, tmp_index_std])
            FileUtil.save_as_csv(tmp_result, tmp_output_path)


# Support sensitivity analysis
def export_parameter_sensibility_experience_result2():
    dirnames = ["experience_parameter_sensibility_result"]
    indexes = ["intention_similarity", "jaccard_index", "encoding_length_compression_rates", "time_use"]
    # rule_covered_positive_sample_rate_threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rule_covered_positive_sample_rate_threshold = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_cache = {}
    result_keys = set()
    for tmp_index in indexes:
        result_cache[tmp_index] = {}

    data_file_path = os.path.join(result_path_prefix, "experience_parameter_sensibility_result",
                                  "result_jaccard_similarity_time_use_avg5.json")
    all_data = load_json(data_file_path)

    for tmp_data in all_data:
        tmp_sample_name = tmp_data["scene"]
        tmp_feedback_noise_rate = tmp_data["feedback_noise_rate"]
        tmp_label_noise_rate = tmp_data["label_noise_rate"]
        tmp_random_merge_number = tmp_data["random_merge_number"]
        tmp_rate_threshold = tmp_data["rule_covered_positive_sample_rate_threshold"]
        if tmp_random_merge_number == 10:
            tmp_key = (tmp_sample_name, tmp_feedback_noise_rate, tmp_label_noise_rate,
                       tmp_random_merge_number, tmp_rate_threshold)
            if tmp_key in result_keys:
                continue
            else:
                result_keys.add(tmp_key)
            for tmp_index in indexes:
                tmp_data_tmp_index_values = tmp_data[tmp_index]
                if tmp_rate_threshold in result_cache[tmp_index]:
                    result_cache[tmp_index][tmp_rate_threshold] += tmp_data_tmp_index_values
                else:
                    result_cache[tmp_index][tmp_rate_threshold] = copy.copy(tmp_data_tmp_index_values)
    print(len(result_keys), "pass" if len(result_keys) == 5 * 30 * 4 * 6 * 11 else "wrong")
    output_dir = os.path.join(analysis_result_dir_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for tmp_index in indexes:
        tmp_output_path = os.path.join(output_dir, f"{tmp_index}_avg_random_merge_number_50.csv")
        tmp_result = []
        for i, tmp_rate_threshold in enumerate(rule_covered_positive_sample_rate_threshold):
            tmp_values = result_cache[tmp_index][tmp_rate_threshold]
            print(len(tmp_values))
            tmp_avg = np.mean(tmp_values)
            tmp_std = np.std(tmp_values, ddof=1)
            tmp_row = [tmp_rate_threshold, tmp_avg, tmp_std]
            tmp_result.append(tmp_row)
        save_as_csv(tmp_result, tmp_output_path)
    finished_keys_path = os.path.join(output_dir, "processed_keys_random_merge_number_50_with_part5.csv")
    result_keys = [list(x) for x in list(result_keys)]
    save_as_csv(result_keys, finished_keys_path)


if __name__ == "__main__":
    # Merge the results of each process into a json file for easy analysis
    # result_to_merge_dirs = [
    #     "experience_comparison_result",
    #     "experience_feasibility_result",
    #     "experience_effectiveness_result",
    #     "experience_sample_adjusting_result",
    #     "experience_parameter_sensibility_result"
    # ]
    # result_to_merge_dirs = [
    #          "experience_feasibility_result"]
    # merge_result(result_to_merge_dirs)

    # Output comparative experimental results and heatmap
    export_comparison_result_heatmap()
    #
    # # Output the results of comparative experiments for plotting box plots
    # export_comparison_result_boxplot()

    #
    # sample augmentation
    #export_sample_adjusting_experience_result()
    #
    #
    # Itera_index sensitivity analysis
    # export_parameter_sensibility_experience_result()

    # Support sensitivity analysis
    # export_parameter_sensibility_experience_result2()
    #draw_legends()
    print("以上")
