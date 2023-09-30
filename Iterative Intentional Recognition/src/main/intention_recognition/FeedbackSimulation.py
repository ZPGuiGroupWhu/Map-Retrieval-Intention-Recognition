"""
   脚本描述：用户反馈模拟器，根据用户检索的目标意图从引导反馈样本集合中随机选择正负样本作为下轮反馈的输入
   基本假设：随着迭代次数的增加，用户反馈和标签噪声比例逐渐减少（理想的用户反馈）
"""
import copy
import random
import numpy as np
from pandas import DataFrame

from src.main.util import RetrievalUtil
from src.main.samples.input.Data import Data


# 正样本集合： 每个子意图对应的真正正样本，反馈噪声，每个子意图对应的标签噪声
# 负样本集合： 真正负样本，反馈噪声
# 选择流程：
# ① 从每个子意图覆盖的候选正样本中，等量选择真正正样本和对应标签噪声，在所有候选的正样本中去除已选的样本
# ② 从所有的候选负样本中选择正样本集合中的反馈噪声，在所有候选负样本中去除已选的正样本集合反馈噪声
# ③ 从所有的候选负样本中选择真正负样本，在所有候选负样本中去除已选的真正负样本
# ④ 从所有的候选正样本中选择负样本集合中的反馈噪声，在所有候选正样本中去除已选的负样本集合反馈噪声
# ⑤ 更新反馈与标签噪声比例
# param：
#    samples: 输入的引导反馈样本集合
#    intention: 预定义的用户真实负样本
#    last_noise: 上次反馈噪声
#    current_noise（可选）：当前噪声比例
#    is_specified_noise（可选）： 是否指定反馈噪声比例
from src.main.util.FileUtil import save_as_json


def feedback_simulator(samples, intention, output_path, last_noise, current_noise=None, is_specified_noise=False):
    selected_positive_negative_samples_num = 50  # 给定反馈正负样本数量
    positive_samples = []  # 选择的正样本
    negative_samples = []  # 选择的负样本

    # 确定反馈噪声比例
    if is_specified_noise:
        current_feedback_noise = current_noise["feedback_noise"]
        current_label_noise = current_noise["label_noise"]
    else:
        last_feedback_noise = last_noise["feedback_noise"]
        last_label_noise = last_noise["label_noise"]
        current_feedback_noise = random.choice(np.arange(0, last_feedback_noise + 0.00001, 0.1))
        current_label_noise = random.choice(np.arange(0, last_label_noise + 0.00001, 0.1))

    # 意图数量
    k = len(intention)
    # 所有候选正样本索引
    all_candidate_positive_samples_index, _ = RetrievalUtil.get_intention_covered_samples(intention, samples,
                                                                                       Data.Ontologies)
    # 所有候选负样本索引
    all_candidate_negative_samples_index = set([s["ID"] for s in samples]) - set(all_candidate_positive_samples_index)

    # 每个子意图的候选正样本集合
    candidate_positive_samples_index = []

    # 每个子意图的候选真实正样本集合和候选标签噪声集合
    candidate_true_positive_samples_index = []
    candidate_label_noise_positive_samples_index = []

    for i, sub_intention in enumerate(intention):
        # 找到每个子意图的候选正样本集合
        candidate_positive_samples_index.append(set(RetrievalUtil.get_sub_intention_covered_samples(
            sub_intention, RetrievalUtil.get_samples_by_ID_list(samples, all_candidate_positive_samples_index),
            Data.Ontologies)[0]))
        # 找到每个子意图的候选真实正样本集合和候选标签噪声集合
        single_label_samples_index, multiple_label_samples_index = \
            RetrievalUtil.divide_into_single_and_multiple_value_samples(
                RetrievalUtil.get_samples_by_ID_list(samples, candidate_positive_samples_index[i]))
        candidate_true_positive_samples_index.append(single_label_samples_index)
        candidate_label_noise_positive_samples_index.append(multiple_label_samples_index)

    # 需要选择的反馈噪声数量
    selected_feedback_noise_samples_num = int(selected_positive_negative_samples_num * current_feedback_noise)
    # 需要选择的真正负样本数量
    selected_true_negative_samples_num = \
        int(selected_positive_negative_samples_num - selected_feedback_noise_samples_num)
    # 每个子意图需要选择的正样本数量 （解决了正样本数目不能被意图数量整除的情况）
    quotient = (selected_positive_negative_samples_num - selected_feedback_noise_samples_num) // k
    remainder = (selected_positive_negative_samples_num - selected_feedback_noise_samples_num) % k
    sub_intention_cover_positive_samples_num = [quotient] * k
    sub_intention_cover_positive_samples_num[k - 1] += remainder

    # 每个子意图需要选择的标签噪声数量
    sub_intention_cover_label_noise_positive_samples_num = []
    # 每个子意图需要选择的真实正样本数量
    sub_intention_cover_true_positive_samples_num = []
    for i in range(k):
        sub_intention_cover_label_noise_positive_samples_num.append(
            int(sub_intention_cover_positive_samples_num[i] * current_label_noise))
        sub_intention_cover_true_positive_samples_num.append(
            int(sub_intention_cover_positive_samples_num[i] - sub_intention_cover_label_noise_positive_samples_num[i]))

    # 为每个子意图选择等量的真实正样本和标签噪声
    for i, sub_intention in enumerate(intention):

        # 从候选真正正样本集合中随机选择
        selected_true_positive_samples_index = random.sample(list(candidate_true_positive_samples_index[i]),
                                                             min(len(candidate_true_positive_samples_index[i]),
                                                                 sub_intention_cover_true_positive_samples_num[i]))
        # 从候选标签噪声集合中随机选择
        selected_label_noise_positive_samples_index = \
            random.sample(list(candidate_label_noise_positive_samples_index[i]),
                          min(len(candidate_label_noise_positive_samples_index[i]),
                              sub_intention_cover_label_noise_positive_samples_num[i]))
        note_and_add_sample(samples, "true positive sample", selected_true_positive_samples_index, positive_samples)
        note_and_add_sample(samples, "label noise", selected_label_noise_positive_samples_index, positive_samples)

        # 在候选正样本集合及下一个子意图覆盖的单（多）标签样本中去除已选元素
        all_candidate_positive_samples_index = set(all_candidate_positive_samples_index) - \
                                               set(selected_true_positive_samples_index) -\
                                               set(selected_label_noise_positive_samples_index)
        if i < k - 1:
            candidate_true_positive_samples_index[i + 1] -= set(selected_true_positive_samples_index)
            candidate_label_noise_positive_samples_index[i + 1] -= set(selected_label_noise_positive_samples_index)

    # 随机挑选反馈噪声添加进正样本中
    selected_feedback_noise_index_in_positive = \
        random.sample(list(all_candidate_negative_samples_index),
                      min(len(all_candidate_negative_samples_index), selected_feedback_noise_samples_num))
    note_and_add_sample(samples, "feedback noise in positive", selected_feedback_noise_index_in_positive,
                        positive_samples)
    # 在候选正样本集合中去除已选元素
    all_candidate_negative_samples_index -= set(selected_feedback_noise_index_in_positive)

    # 随机挑选真实负样本
    selected_true_negative_samples_index = \
        random.sample(list(all_candidate_negative_samples_index),
                      min(len(all_candidate_negative_samples_index), selected_true_negative_samples_num))
    note_and_add_sample(samples, "true negative sample", selected_true_negative_samples_index, negative_samples)
    # 在候选负样本集合中去除已选元素
    all_candidate_negative_samples_index -= set(selected_true_negative_samples_index)

    # 随机挑选反馈噪声添加进负样本中
    selected_feedback_noise_index_in_negative = \
        random.sample(list(all_candidate_positive_samples_index),
                      min(len(all_candidate_positive_samples_index), selected_feedback_noise_samples_num))
    note_and_add_sample(samples, "feedback noise in negative", selected_feedback_noise_index_in_negative,
                        negative_samples)
    # 在候选正样本集合中去除已选元素
    all_candidate_positive_samples_index -= set(selected_feedback_noise_index_in_negative)

    # 更新标签噪声与样本噪声比例
    true_positive_samples = [s for s in positive_samples if s["note"] == "true positive sample"]
    label_noise = [s for s in positive_samples if s["note"] == "label noise"]
    true_negative_samples = [s for s in negative_samples if s["note"] == "true negative sample"]
    feedback_noise_in_positive = [s for s in positive_samples if s["note"] == "feedback noise in positive"]
    feedback_noise_in_negative = [s for s in negative_samples if s["note"] == "feedback noise in negative"]
    new_current_label_noise = len(label_noise) / (len(positive_samples) + 0.0000001)
    new_current_feedback_noise = (len(feedback_noise_in_positive) + len(feedback_noise_in_negative)) / (len(positive_samples) + len(negative_samples))

    # print("正样本：")
    # print(f"selected_true_positive_num: {len(true_positive_samples)}")
    # print(f"selected_label_noise_num: {len(label_noise)}")
    # print(f"selected_positive_feedback_noise_num:{len(feedback_noise_in_positive)}")
    # print("负样本：")
    # print(f"selected_true_negative_num: {len(true_negative_samples)}")
    # print(f"selected_negative_feedback_noise_num: {len(feedback_noise_in_negative)}")
    # print(f"new_current_label_noise: {new_current_label_noise}")
    # print(f"new_current_feedback_noise: {new_current_feedback_noise}")

    # 输出
    if output_path:
        result = {"intention": intention, "positive_samples": positive_samples,
                  "negative_samples": negative_samples}
        export_samples(result, output_path)
    return positive_samples + negative_samples, new_current_feedback_noise, new_current_label_noise


# 为样本添加注释类型名称，并通过ID将样本加入到反馈集合中
# param:
#      samples: 样本总集合 [{dim1: value1, dim2: value2, ...}, ...]
#      note_label: 注释类型名称(str)
#      selected_samples_index: 被选择的样本索引 [13, 15, ...]
#      feedback_set: 输出的反馈集合[{dim1: value1, dim2: value2, ...}, ...]
def note_and_add_sample(samples, note_label, selected_samples_index, feedback_set):
    for i in selected_samples_index:
        tmp_sample = RetrievalUtil.get_sample_by_ID(samples, i)
        tmp_sample["note"] = note_label
        feedback_set.append(tmp_sample)


# params:
#   final_samples = {"intention": [{"Spatial": "North America"}, ...],
#                    "positive_samples": [{"Spatial": "North America"}, ...],
#                    "negative_samples": [{"Spatial": "North America"}, ...]
#   final_samples_paths：最终样本保存路径list
def export_samples(final_samples, final_samples_paths):
    result = final_samples
    intention = final_samples["intention"]
    for final_samples_path in final_samples_paths:
        if final_samples_path.endswith('json'):
            # remove 'note'
            result_copy = copy.deepcopy(result)
            all_samples_copy = result_copy["positive_samples"] + result_copy["negative_samples"]
            for tmp_sample in all_samples_copy:
                if 'note' in tmp_sample:
                    tmp_sample.pop("note")
            save_as_json(result_copy, final_samples_path)

        elif final_samples_path.endswith('xlsx') or final_samples_path.endswith(".xls"):
            result_copy = copy.deepcopy(result)
            all_samples_copy = result_copy["positive_samples"] + result_copy["negative_samples"]
            for tmp_sample in all_samples_copy:
                if 'MapContent' in tmp_sample:
                    tmp_sample['MapContent'] = [x.replace("http://sweetontology.net", "") for x in
                                                tmp_sample['MapContent']]

            # 制作DataFrame
            df_data = []
            dims = ['note', 'ID']
            dims += list(intention[0].keys())

            # 输出各个子意图
            for sub_intention in intention:
                df_sub_intention = []
                for tmp_dim in dims:
                    if tmp_dim == 'note':
                        df_sub_intention.append('sub_intention')
                    elif tmp_dim == 'ID':
                        df_sub_intention.append(-1)
                    else:
                        df_sub_intention.append(sub_intention[tmp_dim])
                df_data.append(df_sub_intention)
            df_data.append([""] * len(dims))

            for tmp_sample in all_samples_copy:
                df_tmp_sample = []
                for tmp_dim in dims:
                    df_tmp_sample.append(tmp_sample[tmp_dim])
                df_data.append(df_tmp_sample)
            df = DataFrame(df_data, columns=dims)
            df.to_excel(final_samples_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    # 测试用户反馈模拟器
    # from src.main.samples.input import Sample
    # from src.main.samples.input.Data import Data
    #
    # scene = "32"
    # sample_version = "scenes_v4_9"
    # sample_path = "../../../resources/samples/" + sample_version + "/Scene" + scene + "/samples_F0p2_L0p2.json"
    # samples, real_intention = Sample.load_sample_from_file(sample_path)
    # all_samples = samples["positive"] + samples["negative"]
    # feedback_simulator(all_samples, real_intention, ['./tmp.json', './tmp.xlsx'],
    #                    {"feedback_noise": 0.2, "label_noise": 0.2},
    #                    {"feedback_noise": 0.2, "label_noise": 0.2}, True)
    pass
