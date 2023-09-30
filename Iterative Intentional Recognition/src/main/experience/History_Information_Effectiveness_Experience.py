"""
   脚本描述：比较是否使用历史信息的引导迭代反馈的效果
           使用所有设计的典型场景以及所有设计的噪声；反馈模拟时约定反馈噪声Feedback_noise比上次反馈低0.1, 标签噪声label_noise比上次反馈低0.2
   评价指标：
   ① Jaccard系数， F1值， BMASS
   ② 耗时
"""
import copy
import os
import random
import time
import multiprocessing

from src.main import Version
from src.main.intention_recognition \
    import Apriori_MDL, FeedbackSimulation, GuideFeedback, EvaluationIndex, IntentionConf, Config, IntentionShift
from src.main.samples.input import Sample
from src.main.util.FileUtil import save_as_json, load_json

sample_version = Version.__sample_version__
output_path_prefix = os.path.join("../../../result/Iterative_Feedback", "scenes_" +
                                  sample_version + "_" + Version.__intention_version__)
if not os.path.exists(output_path_prefix):
    os.mkdir(output_path_prefix)

sample_database = load_json("../../../resources/samples/all_samples.json")


def experience_get_specific_samples_result(part_name, sample_paths):
    output_dir = os.path.join(output_path_prefix, "history_information_effectiveness_result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    methods = ["All History", "Last History", "No History"]
    # config
    run_time = 8
    # for MDL_RM
    config = Config.Config()
    config.rule_covered_positive_sample_rate_threshold = 0.4
    config.adjust_sample_num = True
    config.record_iteration_progress = False

    total_run_num = len(sample_paths) * len(methods) * run_time

    tmp_run_num = 0
    for tmp_sample_name, tmp_sample_path_prefix in sample_paths.items():
        # print("##### ", tmp_sample_name, " #####")
        # 输出结果路径
        result = []
        output_path = os.path.join(output_dir, tmp_sample_name + ".json")

        for tmp_method in methods:
            for i in range(run_time):
                tmp_run_num += 1
                # 打印执行进度
                print(f"running: {part_name} - {tmp_run_num}/{total_run_num} - {tmp_sample_name} - {tmp_method}")

                iterative_num = 0  # 记录迭代次数
                # 样本与标签噪声
                noise_rate_str = tmp_sample_name.split("_")[1:]
                feedback_noise_rate = float(noise_rate_str[0][1:].replace("p", "."))
                label_noise_rate = float(noise_rate_str[1][1:].replace("p", "."))

                # 记录迭代过程中的信息
                # 反馈信息
                all_feedback_samples = []
                all_positive_num = []
                all_negative_num = []
                all_feedback_noise_rate = []
                all_label_noise_rate = []
                all_frequent_items = []
                all_conflict_samples = []
                all_candidate_sub_intention_num = []  # 候选子意图的数量

                # 意图识别准确度信息
                all_extracted_intention = []
                all_jaccard_index = []
                all_F1 = []
                all_intention_similarity = []
                all_encoding_length_compression_rates = []

                # 意图置信度，用于构建引导反馈样本与循环终止条件
                all_intention_compatibility = []
                all_intention_completeness = []
                all_intention_conf = []
                all_sub_intention_compatibility = []
                all_sub_intention_completeness = []
                all_sub_intention_conf = []

                # 意图覆盖率
                all_intention_cover_sample_rates = []
                all_sub_intention_cover_sample_rates = []

                # 记录每个部分的耗时信息
                all_generate_sub_intention_time_use = []
                all_greedy_search_time_use = []
                all_evaluation_time_use = []
                all_select_guide_samples_time_use = []

                # todo: 目前只使用了历史样本，需要使用历史项集（历史信息利用有误）
                while iterative_num < 10:
                    # 加载本轮反馈样本
                    tmp_sample_path = os.path.join(tmp_sample_path_prefix, "samples_I" + str(iterative_num) + ".json")
                    samples, real_intention = Sample.load_sample_from_file(tmp_sample_path)

                    # 正负反馈样本过少时，跳出循环
                    if len(samples["positive"]) == 0 or len(samples["negative"]) == 0:
                        break

                    # print(f"第{i}次重复实验，第{iterative_num}轮迭代")

                    # 初始化样本数据
                    if tmp_method == "All History":
                        # 结合所有的历史反馈样本
                        combine_history_samples = {"positive": copy.deepcopy(samples["positive"]),
                                                   "negative": copy.deepcopy(samples["negative"])}
                        if iterative_num > 0:
                            combine_history_samples["positive"] += [s for iterative_samples in all_feedback_samples for
                                                                    s in
                                                                    iterative_samples["positive"]]
                            combine_history_samples["negative"] += [s for iterative_samples in all_feedback_samples for
                                                                    s in
                                                                    iterative_samples["negative"]]
                        data = Apriori_MDL.Data(combine_history_samples)

                    elif tmp_method == "Last History":
                        # 仅考虑上轮反馈样本
                        combine_history_samples = {"positive": copy.deepcopy(samples["positive"]),
                                                   "negative": copy.deepcopy(samples["negative"])}
                        if iterative_num > 0:
                            combine_history_samples["positive"] += [s for s in all_feedback_samples[iterative_num - 1]["positive"]]
                            combine_history_samples["negative"] += [s for s in all_feedback_samples[iterative_num - 1]["negative"]]
                        data = Apriori_MDL.Data(combine_history_samples)

                    elif tmp_method == "No History":
                        data = Apriori_MDL.Data(samples)

                    min_support = config.rule_covered_positive_sample_rate_threshold
                    k_max = len(Apriori_MDL.Data.dimensions)

                    # 进行意图识别
                    time0 = time.time()
                    # 首先得到候选子意图
                    candidate_sub_intentions, frequent_items = Apriori_MDL.get_all_candidate_sub_intentions(data,
                                                                                                            min_support,
                                                                                                            k_max)
                    time1 = time.time()
                    # 然后开始进行意图识别
                    # 初始化意图为空，然后尝试加入候选子意图，尝试时计算编码长度，若确定要加入某个子意图，则需要去掉候选子意图中被当前子意图覆盖的部分
                    # 需要记录每个子意图覆盖的正负样本编号
                    # 需要记录每次加入子意图后的编码长度
                    # 需要计算意图长度与给定意图后的样本编码长度
                    extracted_intention, total_encoding_length, init_total_encoding_length, iteration_log = \
                        Apriori_MDL.get_intention_by_method6(data, config, candidate_sub_intentions)
                    time2 = time.time()
                    encoding_length_compression_rates = total_encoding_length / init_total_encoding_length

                    # 计算意图及子意图对正负样本的覆盖率
                    cover_positive_sample_rate, cover_negative_sample_rate = IntentionShift.cal_intention_cover_sample_rate(
                        data,
                        extracted_intention)
                    intention_cover_sample_rates = {"positive_samples_rate": cover_positive_sample_rate,
                                                    "negative_samples_rate": cover_negative_sample_rate}
                    sub_intention_cover_sample_rates = []
                    for sub_intention in extracted_intention:
                        sub_intention_cover_positive_sample_rate, sub_intention_cover_negative_sample_rate = \
                            IntentionShift.cal_intention_cover_sample_rate(data, [sub_intention])
                        sub_intention_cover_sample_rates.append(
                            {
                                "positive_samples_rate": sub_intention_cover_positive_sample_rate,
                                "negative_samples_rate": sub_intention_cover_negative_sample_rate
                            }
                        )

                    # 寻找本轮反馈中与意图冲突的正负样本（问题样本）
                    conflict_positive_samples_index, conflict_negative_samples_index = \
                        IntentionShift.get_conflict_samples_index(data, extracted_intention)
                    conflict_samples = [data.docs["positive"][index] for index in conflict_positive_samples_index] + \
                                       [data.docs["negative"][index] for index in conflict_negative_samples_index]

                    # jaccard系数
                    jaccard_index = EvaluationIndex.get_jaccard_index(sample_database, real_intention,
                                                                      extracted_intention,
                                                                      Apriori_MDL.Data.Ontologies)
                    # BMASS指标
                    best_map_average_semantic_similarity = \
                        EvaluationIndex.get_intention_similarity(extracted_intention, real_intention,
                                                                 Apriori_MDL.Data.direct_Ancestor,
                                                                 Apriori_MDL.Data.Ontology_Root,
                                                                 Apriori_MDL.Data.concept_information_content)
                    # F1值
                    F1 = EvaluationIndex.get_F1_score(sample_database, real_intention,
                                                      extracted_intention,
                                                      Apriori_MDL.Data.Ontologies)

                    # 意图及子意图的置信度
                    cpa = IntentionConf.get_intention_compatibility(samples, extracted_intention,
                                                                    Apriori_MDL.Data.Ontologies)
                    cpl = IntentionConf.get_intention_completeness(samples, extracted_intention,
                                                                   Apriori_MDL.Data.Ontologies)
                    conf = IntentionConf.get_intention_conf(samples, extracted_intention, Apriori_MDL.Data.Ontologies)

                    sub_cpa = []
                    sub_cpl = []
                    sub_conf = []
                    for sub_intention in extracted_intention:
                        sub_cpa.append(
                            IntentionConf.get_sub_intention_compatibility(samples, sub_intention,
                                                                          Apriori_MDL.Data.Ontologies))
                        sub_cpl.append(
                            IntentionConf.get_sub_intention_completeness(samples, sub_intention,
                                                                         Apriori_MDL.Data.Ontologies))
                        sub_conf.append(
                            IntentionConf.get_sub_intention_conf(samples, sub_intention, Apriori_MDL.Data.Ontologies))

                    time3 = time.time()
                    # todo: 频繁项集及其支持度没有利用
                    all_feedback_samples.append(samples)
                    all_positive_num.append(len(samples["positive"]))
                    all_negative_num.append(len(samples["negative"]))
                    all_feedback_noise_rate.append(feedback_noise_rate)
                    all_label_noise_rate.append(label_noise_rate)
                    all_frequent_items.append(frequent_items)
                    all_conflict_samples.append(conflict_samples)

                    all_extracted_intention.append(extracted_intention)

                    all_jaccard_index.append(jaccard_index)
                    all_intention_similarity.append(best_map_average_semantic_similarity)
                    all_F1.append(F1)
                    all_encoding_length_compression_rates.append(encoding_length_compression_rates)

                    all_intention_compatibility.append(cpa)
                    all_intention_completeness.append(cpl)
                    all_intention_conf.append(conf)
                    all_sub_intention_compatibility.append(sub_cpa)
                    all_sub_intention_completeness.append(sub_cpl)
                    all_sub_intention_conf.append(sub_conf)

                    all_intention_cover_sample_rates.append(intention_cover_sample_rates)
                    all_sub_intention_cover_sample_rates.append(sub_intention_cover_sample_rates)

                    all_generate_sub_intention_time_use.append(time1 - time0)
                    all_greedy_search_time_use.append(time2 - time1)
                    all_evaluation_time_use.append(time3 - time2)

                    all_candidate_sub_intention_num.append(len(candidate_sub_intentions))

                    # 若本轮置信度大于0.85，且与上次置信度差值小于0.15（变化不大），则终止迭代
                    if conf >= 0.85 and iterative_num - 1 >= 0 and (
                            conf - all_intention_conf[iterative_num - 1]) < 0.15:
                        break

                    iterative_num += 1
                    # 生成引导反馈样本， 这里考虑之前所有的问题样本
                    guide_feedback_samples,_,_ = GuideFeedback.guide_feedback(extracted_intention, conf, sub_conf,
                                                                          [x for y in all_conflict_samples for x in y],
                                                                          sample_database, 500)
                    time4 = time.time()
                    all_select_guide_samples_time_use.append(time4 - time3)

                    # 用户反馈模拟
                    current_feedback_noise = feedback_noise_rate - 0.1 if feedback_noise_rate >= 0.1 else 0
                    current_label_noise = label_noise_rate - 0.2 if label_noise_rate >= 0.2 else 0
                    current_noise = {"feedback_noise": current_feedback_noise, "label_noise": current_label_noise}
                    iterative_samples_output_path = \
                        [os.path.join(tmp_sample_path_prefix, "samples_I" + str(iterative_num) + ".json"),
                         os.path.join(tmp_sample_path_prefix, "samples_I" + str(iterative_num) + ".xlsx")]
                    _, feedback_noise_rate, label_noise_rate = \
                        FeedbackSimulation.feedback_simulator(guide_feedback_samples, real_intention,
                                                              iterative_samples_output_path,
                                                              {}, current_noise, True)

                # 输出迭代结果
                result.append({
                    "scene": tmp_sample_name,
                    "method": tmp_method,
                    "positive_num": all_positive_num,
                    "negative_num": all_negative_num,
                    "iterative_num": len(all_feedback_noise_rate),
                    "feedback_noise_rate": all_feedback_noise_rate,
                    "label_noise_rate": all_label_noise_rate,
                    "encoding_length_compression_rates": all_encoding_length_compression_rates,
                    "candidate_sub_intention_num": all_candidate_sub_intention_num,

                    "extracted_intention": all_extracted_intention,
                    "jaccard_index": all_jaccard_index,
                    "F1": all_F1,
                    "intention_similarity": all_intention_similarity,

                    "intention_compatibility": all_intention_compatibility,
                    "intention_completeness": all_intention_completeness,
                    "intention_conf": all_intention_conf,
                    "sub_intention_compatibility": all_sub_intention_compatibility,
                    "sub_intention_completeness": all_sub_intention_completeness,
                    "sub_intention_conf": all_intention_conf,

                    "intention_cover_sample_rates": all_intention_cover_sample_rates,
                    "sub_intention_cover_sample_rates": all_sub_intention_cover_sample_rates,

                    "generate_sub_intention_time_use": all_generate_sub_intention_time_use,
                    "greedy_search_time_use": all_greedy_search_time_use,
                    "evaluation_time_use": all_evaluation_time_use,
                    "select_guide_samples_time_use": all_select_guide_samples_time_use

                })

        # 保存该场景下不同算法多次重复计算得到的结果
        save_as_json(result, output_path)

    pass


# and record the time use, jaccard score， intention_similarity，and rules(in json_str).
def experience_get_all_samples_result():
    samples_dir = os.path.join("../../../resources/samples", "scenes_" + sample_version)
    sample_names = os.listdir(samples_dir)
    sample_paths = {}
    for tmp_sample_name in sample_names:
        sample_paths[tmp_sample_name] = os.path.join(samples_dir, tmp_sample_name)
    sample_names = list(sample_paths.keys())
    print(len(sample_names), sample_names)

    # 找到未完成的场景继续执行
    # rest_sample_paths = {}
    # complete_sample_names = os.listdir("../../../result/Iterative_Feedback/scenes_v4_10_Intention_20230403/history_information_effectiveness_result")
    #
    # tmp_name_list = []
    # for tmp_name_str in complete_sample_names:
    #     tmp_name_list.append(tmp_name_str.split(".")[0])
    #
    # for tmp_name in sample_names:
    #     if tmp_name not in tmp_name_list:
    #         rest_sample_paths[tmp_name] = sample_paths[tmp_name]
    #
    # sample_paths = copy.deepcopy(rest_sample_paths)
    # sample_names = list(sample_paths.keys())

    random.shuffle(sample_names)

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

    # for i, tmp_sample_part in enumerate(all_samples_parts):
    #     experience_get_specific_samples_result(f"part{i}", tmp_sample_part)

    for i, tmp_sample_part in enumerate(all_samples_parts):
        part_name = "PART_" + str(i)
        tmp_p = multiprocessing.Process(target=experience_get_specific_samples_result,
                                        args=(part_name, tmp_sample_part))
        tmp_p.start()


if __name__ == "__main__":
    experience_get_all_samples_result()
    print("Aye")
