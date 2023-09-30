"""
    脚本描述：结果数据整合与处理（便于画图）
    包含三个实验的统计数据处理
"""

import os
from collections import defaultdict

from numpy import mean

from src.main import Version
from src.main.util.FileUtil import load_json, save_as_json, save_as_csv, save_as_excel

sample_version = Version.__sample_version__
code_version = Version.__intention_version__
result_dirname = "scenes_" + sample_version + "_" + code_version

result_path_prefix = os.path.join("../../../result/Iterative_Feedback", result_dirname)


# 合并不同进程的结果
def merge_result(result_to_merge_dirs):
    for tmp_dir in result_to_merge_dirs:
        all_parts_result_dir = os.path.join(result_path_prefix, tmp_dir)
        output_result_path = result_path_prefix
        if not os.path.exists(output_result_path):
            os.mkdir(output_result_path)
        tmp_all_samples_result = []
        tmp_sub_part_file_names = os.listdir(all_parts_result_dir)
        for tmp_sub_part_file_name in tmp_sub_part_file_names:
            # if "PART" not in tmp_sub_part_file_name:
            #     continue
            print(tmp_sub_part_file_name)
            tmp_sub_part_path = os.path.join(all_parts_result_dir, tmp_sub_part_file_name)
            tmp_sub_part_json = load_json(tmp_sub_part_path)

            for tmp_data in tmp_sub_part_json:
                tmp_all_samples_result.append(tmp_data)

        all_result_path = os.path.join(all_parts_result_dir, "all_result.json")
        save_as_json(tmp_all_samples_result, all_result_path)


# 针对意图偏移判断准则有效性实验的数据处理
def intention_shift_detector_analysis(filename):
    output_result_path = os.path.join(result_path_prefix, "intention_shift_detector_analysis")
    if not os.path.exists(output_result_path):
        os.mkdir(output_result_path)
    all_scenes_result_dir = os.path.join(result_path_prefix, filename)
    scene_file_names = os.listdir(all_scenes_result_dir)

    result = defaultdict(list)
    for tmp_scene_file_name in scene_file_names:
        print(tmp_scene_file_name)
        tmp_sub_part_path = os.path.join(all_scenes_result_dir, tmp_scene_file_name)
        scene_data = load_json(tmp_sub_part_path)

        for tmp_scene_data in scene_data:
            tmp_scene_data["sort_key"] = tmp_scene_data["intention_similarity"][-1]
        scene_data = sorted(scene_data, key=lambda x: x["sort_key"], reverse=True)

        tar_data = scene_data[:5]

        # 约定detector1: 压缩率，detector2: 意图，detector3: 子意图
        tmp_scene = ""
        tmp_detect_num = 0

        tmp_detector1_success = 0
        tmp_detector2_success = 0
        tmp_detector3_success = 0
        tmp_detector1_2_success = 0
        tmp_detector1_3_success = 0
        tmp_detector2_3_success = 0
        tmp_detector1_2_3_success = 0

        tmp_detector1_time_use = 0
        tmp_detector2_time_use = 0
        tmp_detector3_time_use = 0
        for tmp_scene_data in tar_data:
            tmp_scene = tmp_scene_data["scene"]
            tmp_detect_num += tmp_scene_data["iterative_num"] - 1
            tmp_detector1_success += tmp_scene_data["is_success_by_compression_detector"].count(True)
            tmp_detector2_success += tmp_scene_data["is_success_by_intention_detector"].count(True)
            tmp_detector3_success += tmp_scene_data["is_success_by_sub_intention_detector"].count(True)
            tmp_detector1_2_success += [x or y for x, y in zip(tmp_scene_data["is_success_by_compression_detector"], tmp_scene_data["is_success_by_intention_detector"])].count(True)
            tmp_detector1_3_success += [x or y for x, y in zip(tmp_scene_data["is_success_by_compression_detector"], tmp_scene_data["is_success_by_sub_intention_detector"])].count(True)
            tmp_detector2_3_success += [x or y for x, y in zip(tmp_scene_data["is_success_by_intention_detector"], tmp_scene_data["is_success_by_sub_intention_detector"])].count(True)
            tmp_detector1_2_3_success += [x or y or z for x, y, z in zip(tmp_scene_data["is_success_by_compression_detector"], tmp_scene_data["is_success_by_intention_detector"], tmp_scene_data["is_success_by_sub_intention_detector"])].count(True)

            tmp_detector1_time_use += mean(tmp_scene_data["time_use_by_compression_detector"])
            tmp_detector2_time_use += mean(tmp_scene_data["time_use_by_intention_detector"])
            tmp_detector3_time_use += mean(tmp_scene_data["time_use_by_sub_intention_detector"])

        result["scene"].append(int(tmp_scene[5:].split("_")[0]))
        result["detect_num"].append(tmp_detect_num)
        result["compression_success"].append(tmp_detector1_success)
        result["compression_time_use"].append(tmp_detector1_time_use)
        result["intention_success"].append(tmp_detector2_success)
        result["intention_time_use"].append(tmp_detector2_time_use)
        result["sub_intention_success"].append(tmp_detector3_success)
        result["sub_intention_time_use"].append(tmp_detector3_time_use)

        result["1+2"].append(tmp_detector1_2_success)
        result["1+3"].append(tmp_detector1_3_success)
        result["2+3"].append(tmp_detector2_3_success)
        result["1+2+3"].append(tmp_detector1_2_3_success)



    result2 = {}
    result2["scene"] = [result["scene"][i] for i in range(0, len(result["scene"]), 4)]
    result2["detect_num"] = [sum(result["detect_num"][i: i+4]) for i in range(0, len(result["detect_num"]), 4)]
    result2["compression_success"] = [sum(result["compression_success"][i: i + 4]) for i in range(0, len(result["compression_success"]), 4)]
    result2["compression_time_use"] = [mean(result["compression_time_use"][i: i + 4]) for i in range(0, len(result["compression_time_use"]), 4)]
    result2["intention_success"] = [sum(result["intention_success"][i: i + 4]) for i in range(0, len(result["intention_success"]), 4)]
    result2["intention_time_use"] = [mean(result["intention_time_use"][i: i + 4]) for i in range(0, len(result["intention_time_use"]), 4)]
    result2["sub_intention_success"] = [sum(result["sub_intention_success"][i: i + 4]) for i in range(0, len(result["sub_intention_success"]), 4)]
    result2["sub_intention_time_use"] = [mean(result["sub_intention_time_use"][i: i + 4]) for i in range(0, len(result["sub_intention_time_use"]), 4)]

    result2["1+2"] = [sum(result["1+2"][i: i + 4]) for i in range(0, len(result["1+2"]), 4)]
    result2["1+3"] = [sum(result["1+3"][i: i + 4]) for i in range(0, len(result["1+3"]), 4)]
    result2["2+3"] = [sum(result["2+3"][i: i + 4]) for i in range(0, len(result["2+3"]), 4)]
    result2["1+2+3"] = [sum(result["1+2+3"][i: i + 4]) for i in range(0, len(result["1+2+3"]), 4)]

    save_as_excel(result2, os.path.join(output_result_path, "heatmap_for_intention_shift.xlsx"))


# 针对引导式反馈有效性实验的数据处理
def guide_feedback_comparison_analysis(filename):
    output_result_path = os.path.join(result_path_prefix, "guide_feedback_analysis")
    if not os.path.exists(output_result_path):
        os.mkdir(output_result_path)
    all_scenes_result_dir = os.path.join(result_path_prefix, filename)
    scene_file_names = os.listdir(all_scenes_result_dir)

    feedback_type = ["Guide Feedback", "Direct Feedback", "Sim Feedback"]
    scene_type = ["单意单维", "单意多维", "多意单维", "多意多维"]
    result = defaultdict(list)
    result2 = defaultdict(list)
    result3 = defaultdict(int)
    tmp = defaultdict(int)

    for i in scene_type:
        for j in feedback_type:
            for k in range(3):
                tmp[f"{i}_{j}_{k}"] = 0

    for tmp_scene_file_name in scene_file_names:
        print(tmp_scene_file_name)
        tmp_sub_part_path = os.path.join(all_scenes_result_dir, tmp_scene_file_name)
        scene_data = load_json(tmp_sub_part_path)

        for tmp_scene_data in scene_data:
            tmp_scene_data["sort_key"] = tmp_scene_data["intention_similarity"][-1]
        scene_data = sorted(scene_data, key=lambda x: x["sort_key"], reverse=True)

        # 最小的迭代次数
        min_iterative_num = min(scene_data, key=lambda x:x["iterative_num"])["iterative_num"]
        for tmp_method in ["Guide Feedback", "Direct Feedback", "Sim Feedback"]:
            tmp_method_scene_data = [r for r in scene_data if r["method"] == tmp_method]
            tar_data = tmp_method_scene_data[:5]
            # 获取各个统计量信息
            for tmp_scene_data in tar_data:
                result["scene"].append(tmp_scene_data["scene"])
                result["method"].append(tmp_scene_data["method"])
                for i in range(10):
                    # 如果不存在该迭代轮次，则取最后一次迭代的结果
                    if i > tmp_scene_data["iterative_num"] - 1:
                        iterative_round = tmp_scene_data["iterative_num"] - 1
                    else:
                        iterative_round = i
                    result[f"{i + 1}_BMASS"].append(tmp_scene_data["intention_similarity"][iterative_round])
                    result[f"{i + 1}_jaccard"].append(tmp_scene_data["jaccard_index"][iterative_round])
                    result[f"{i + 1}_F1"].append(tmp_scene_data["F1"][iterative_round])

                    if iterative_round >= len(tmp_scene_data["select_guide_samples_time_use"]):
                        result[f"{i + 1}_candidate_feedback_time_use"].append(0)
                        result[f"{i + 1}_retrieval_time_use"].append(0)
                        result[f"{i + 1}_cal_sim_time_use"].append(0)
                    else:
                        result[f"{i + 1}_candidate_feedback_time_use"].append(tmp_scene_data["select_guide_samples_time_use"][iterative_round])
                        result[f"{i + 1}_retrieval_time_use"].append(tmp_scene_data["retrieval_time_use"][iterative_round])
                        result[f"{i + 1}_cal_sim_time_use"].append(tmp_scene_data["cal_sim_time_use"][iterative_round])

                    result[f"{i + 1}_positive_num"].append(tmp_scene_data["positive_num"][iterative_round])
                    result[f"{i + 1}_negative_num"].append(tmp_scene_data["negative_num"][iterative_round])

                result["interruption_reason"].append(tmp_scene_data["interruption_reason"])
                result["iterative_num"].append(tmp_scene_data["iterative_num"])

            # 获取正负反馈样本数目
            for tmp_scene_data in tar_data:
                result2[f"{tmp_method}_positive_num"] += tmp_scene_data["positive_num"][1:min_iterative_num]
                result2[f"{tmp_method}_negative_num"] += tmp_scene_data["negative_num"][1:min_iterative_num]

            # 获取各个维度的正负样本数目
            for tmp_scene_data in tar_data:
                scene_int = int(tmp_scene_data["scene"].split("_")[0][5:])
                if scene_int <= 6:
                    tmp_scene_type = scene_type[0]
                elif 6 < scene_int <= 12:
                    tmp_scene_type = scene_type[1]
                elif 12 < scene_int <= 18:
                    tmp_scene_type = scene_type[2]
                elif 18 < scene_int <= 24:
                    tmp_scene_type = scene_type[3]

                result2[f"{tmp_scene_type}_{tmp_method}_positive_num"] += tmp_scene_data["positive_num"][1:min_iterative_num]
                result2[f"{tmp_scene_type}_{tmp_method}_negative_num"] += tmp_scene_data["negative_num"][1:min_iterative_num]

            # 获取中止退出程序的统计值
            for tmp_scene_data in tar_data:
                scene_int = int(tmp_scene_data["scene"].split("_")[0][5:])
                if scene_int <= 6:
                    tmp_scene_type = scene_type[0]
                elif 6 < scene_int <= 12:
                    tmp_scene_type = scene_type[1]
                elif 12 < scene_int <= 18:
                    tmp_scene_type = scene_type[2]
                elif 18 < scene_int <= 24:
                    tmp_scene_type = scene_type[3]
                tmp[f"{tmp_scene_type}_{tmp_method}_{tmp_scene_data['interruption_reason']}"] += 1

    max_len = max(len(v) for v in result2.values())
    for k, v in result2.items():
        diff_len = max_len - len(v)
        v.extend([None] * diff_len)

    result3["scene"] = tmp.keys()
    result3["value"] = tmp.values()

    # 输出结果文件
    # save_as_excel(result, os.path.join(output_result_path, "guide_feedback_comparison_analysis.xlsx"))
    save_as_excel(result2, os.path.join(output_result_path, "guide_feedback_comparison_analysis2.xlsx"))
    # save_as_excel(result3, os.path.join(output_result_path, "guide_feedback_comparison_analysis3.xlsx"))


# 针对历史反馈数据有效性实验的数据处理
def history_information_effectiveness_analysis(filename):
    output_result_path = os.path.join(result_path_prefix, "history_information_effectiveness_analysis")
    if not os.path.exists(output_result_path):
        os.mkdir(output_result_path)
    all_scenes_result_dir = os.path.join(result_path_prefix, filename)
    scene_file_names = os.listdir(all_scenes_result_dir)

    feedback_type = ["All History", "Last History", "No History"]
    scene_type = ["单意单维", "单意多维", "多意单维", "多意多维"]
    result = defaultdict(list)
    result2 = defaultdict(list)
    result3 = defaultdict(list)
    result4 = defaultdict(list)

    # result4 初始化
    for index in ["Bmass", "Jaccard", "F1"]:
        for prefix in ["init", "final", "dif"]:
            for i in range(4):
                result4[f"{prefix}_{index}_{i}"] = [None] * 25

    for tmp_scene_file_name in scene_file_names:
        print(tmp_scene_file_name)
        tmp_sub_part_path = os.path.join(all_scenes_result_dir, tmp_scene_file_name)
        scene_data = load_json(tmp_sub_part_path)

        for tmp_scene_data in scene_data:
            tmp_scene_data["sort_key"] = tmp_scene_data["intention_similarity"][-1]
        scene_data = sorted(scene_data, key=lambda x: x["sort_key"], reverse=True)

        # 最小的迭代次数
        min_iterative_num = min(scene_data, key=lambda x: x["iterative_num"])["iterative_num"]
        for tmp_method in feedback_type :
            tmp_method_scene_data = [r for r in scene_data if r["method"] == tmp_method]
            tar_data = tmp_method_scene_data[:5]
            # 获取各个统计量信息
            for tmp_scene_data in tar_data:
                result["scene"].append(tmp_scene_data["scene"])
                result["method"].append(tmp_scene_data["method"])
                for i in range(10):
                    # 如果不存在该迭代轮次，则取最后一次迭代的结果
                    if i > tmp_scene_data["iterative_num"] - 1:
                        iterative_round = tmp_scene_data["iterative_num"] - 1
                    else:
                        iterative_round = i
                    result[f"{i + 1}_BMASS"].append(tmp_scene_data["intention_similarity"][iterative_round])
                    result[f"{i + 1}_jaccard"].append(tmp_scene_data["jaccard_index"][iterative_round])
                    result[f"{i + 1}_F1"].append(tmp_scene_data["F1"][iterative_round])

                    if iterative_round >= len(tmp_scene_data["select_guide_samples_time_use"]):
                        result[f"{i + 1}_generate_sub_intention_time_use"].append(0)
                        result[f"{i + 1}_greedy_search_time_use"].append(0)
                    else:
                        result[f"{i + 1}_generate_sub_intention_time_use"].append(
                            tmp_scene_data["generate_sub_intention_time_use"][iterative_round])
                        result[f"{i + 1}_greedy_search_time_use"].append(
                            tmp_scene_data["greedy_search_time_use"][iterative_round])

                    result[f"{i + 1}_positive_num"].append(tmp_scene_data["positive_num"][iterative_round])
                    result[f"{i + 1}_negative_num"].append(tmp_scene_data["negative_num"][iterative_round])

                result["iterative_num"].append(tmp_scene_data["iterative_num"])

            # 获取总的BMASS、Jaccard、F1及耗时指标
            for tmp_scene_data in tar_data:
                result2[f"{tmp_method}_BMASS"].append(max(tmp_scene_data["intention_similarity"]))
                result2[f"{tmp_method}_Jaccard"].append(max(tmp_scene_data["jaccard_index"]))
                result2[f"{tmp_method}_F1"].append(max(tmp_scene_data["F1"]))
                result2[f"{tmp_method}_time_use"].append(min(tmp_scene_data["generate_sub_intention_time_use"]) +  min(tmp_scene_data["greedy_search_time_use"]))

            # 获取各个维度的BMASS、Jaccard、F1及耗时指标

            for tmp_scene_data in tar_data:
                scene_int = int(tmp_scene_data["scene"].split("_")[0][5:])
                if scene_int <= 6:
                    tmp_scene_type = scene_type[0]
                elif 6 < scene_int <= 12:
                    tmp_scene_type = scene_type[1]
                elif 12 < scene_int <= 18:
                    tmp_scene_type = scene_type[2]
                elif 18 < scene_int <= 24:
                    tmp_scene_type = scene_type[3]

                result3[f"{tmp_scene_type}_{tmp_method}_BMASS"].append(max(tmp_scene_data["intention_similarity"]))
                result3[f"{tmp_scene_type}_{tmp_method}_Jaccard"].append(max(tmp_scene_data["jaccard_index"]))
                result3[f"{tmp_scene_type}_{tmp_method}_F1"].append(max(tmp_scene_data["F1"]))
                result3[f"{tmp_scene_type}_{tmp_method}_time_use"].append(min(tmp_scene_data["generate_sub_intention_time_use"]) + min(
                    tmp_scene_data["greedy_search_time_use"]))

            # 获取BMASS、Jaccard、F1的差值信息
            if tmp_method_scene_data[0]["method"] == 'All History':
                # 映射信息
                tmp_scene_str = tar_data[0]["scene"].split('_')
                scene_int = int(tmp_scene_str[0][5:])
                noise = tmp_scene_str[1] + tmp_scene_str[2]

                col = {'F0p2L0p8': 0, 'F0p3L0p6': 1, 'F0p4L0p4': 2, 'F0p4L0p8': 3}
                tmp_init_Bmass = 0
                tmp_final_Bmass = 0
                tmp_init_Jaccard = 0
                tmp_final_Jaccard = 0
                tmp_init_F1 = 0
                tmp_final_F1 = 0
                tmp_init_time = 0
                tmp_final_time = 0
                for tmp_scene_data in tar_data:
                    if tmp_init_Bmass < tmp_scene_data["intention_similarity"][0]:
                        tmp_init_Bmass = tmp_scene_data["intention_similarity"][0]
                    if tmp_final_Bmass < max(tmp_scene_data["intention_similarity"]):
                        tmp_final_Bmass = max(tmp_scene_data["intention_similarity"])
                    if tmp_init_Jaccard < tmp_scene_data["jaccard_index"][0]:
                        tmp_init_Jaccard = tmp_scene_data["jaccard_index"][0]
                    if tmp_final_Jaccard < max(tmp_scene_data["jaccard_index"]):
                        tmp_final_Jaccard = max(tmp_scene_data["jaccard_index"])
                    if tmp_init_F1 < tmp_scene_data["F1"][0]:
                        tmp_init_F1 = tmp_scene_data["F1"][0]
                    if tmp_final_F1 < max(tmp_scene_data["F1"]):
                        tmp_final_F1 = max(tmp_scene_data["F1"])
                result4[f"init_Bmass_{col[noise]}"][scene_int] = tmp_init_Bmass
                result4[f"final_Bmass_{col[noise]}"][scene_int] = tmp_final_Bmass
                result4[f"dif_Bmass_{col[noise]}"][scene_int] = tmp_final_Bmass - tmp_init_Bmass
                result4[f"init_Jaccard_{col[noise]}"][scene_int] = tmp_init_Jaccard
                result4[f"final_Jaccard_{col[noise]}"][scene_int] = tmp_final_Jaccard
                result4[f"dif_Jaccard_{col[noise]}"][scene_int] = tmp_final_Jaccard - tmp_init_Jaccard
                result4[f"init_F1_{col[noise]}"][scene_int] = tmp_init_F1
                result4[f"final_F1_{col[noise]}"][scene_int] = tmp_final_F1
                result4[f"dif_F1_{col[noise]}"][scene_int] = tmp_final_F1 - tmp_init_F1

    # 输出结果文件
    # save_as_excel(result, os.path.join(output_result_path, "history_information_effectiveness_analysis.xlsx"))
    # save_as_excel(result2, os.path.join(output_result_path, "history_information_effectiveness_analysis2.xlsx"))
    # save_as_excel(result3, os.path.join(output_result_path, "history_information_effectiveness_analysis3.xlsx"))
    save_as_excel(result4, os.path.join(output_result_path, "history_information_effectiveness_analysis4.xlsx"))


if __name__ == "__main__":
    # 合并各进程的结果到一个json文件，便于分析
    result_to_merge_dirs = [
        "intention_shift_detector_comparison_result",
        "history_information_effectiveness_result",
        "guide_feedback_comparison_result"
    ]
    # merge_result(result_to_merge_dirs)

    # intention_shift_detector_analysis(result_to_merge_dirs[0])
    # history_information_effectiveness_analysis(result_to_merge_dirs[1])
    # guide_feedback_comparison_analysis(result_to_merge_dirs[2])


