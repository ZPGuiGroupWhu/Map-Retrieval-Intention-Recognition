"""
   脚本描述：2023.04.03 合成数据生成代码（sample_version = v4.10)
   用于生成迭代反馈实验中的样本总集以及初始反馈不同噪声比例的正负样本

   与2023.03.15版本的差异：
   ① 每个样本中添加ID字段，方便在从样本总库中溯源
   ② 生成的样本总库中，包含维度单标签样本和多标签样本（比例为1：1），样本的目标概念与非目标概念数量比例为4：1
   ③ 所有样本及噪声都直接从样本总库中选取，不再进行标签添加与样本交换操作来产生标签及样本噪声
   ④ 所有场景中的负样本都是从非正集合中随机挑选，不再用约束意图生成（更符合初次反馈的场景）

"""

from src.main import Version
from src.main.util import RetrievalUtil
from src.main.samples.input import DimensionValues
from src.main.util.FileUtil import save_as_json, load_csv, load_json
import os.path
import random
import copy
import math
from pandas import DataFrame

__dir__ = os.path.dirname(os.path.abspath(__file__))
sample_version = Version.__sample_version__
all_samples_path = "../../../../resources/samples/all_samples.json"
single_label_samples_path = "../../../../resources/samples/single_label_samples.json"
multiple_label_samples_path = "../../../../resources/samples/multiple_label_samples.json"
intention_path = "../../../../resources/samples/Intention_v1.6.json"
intention_csv_path = "../../../../resources/samples/各场景具体意图v1.6.csv"
all_output_path_prefix = os.path.join("../../../../resources/samples", "scenes_" + sample_version)
if not os.path.exists(all_output_path_prefix):
    os.mkdir(all_output_path_prefix)

# S:Spatial, T:Theme, M:MapMethod, C:MapContent
intention_dimensions = ["Spatial", "MapContent", "MapMethod", "Theme"]

ontology_root = {"Spatial": DimensionValues.SpatialValue.Ontology_Root,
                 "Theme": DimensionValues.ThemeValues.Ontology_Root,
                 "MapMethod": DimensionValues.MapMethodValues.Ontology_Root,
                 "MapContent": DimensionValues.MapContentValues.Ontology_Root}
ontologies = {"Spatial": DimensionValues.SpatialValue.Ontologies,
              "Theme": DimensionValues.ThemeValues.Ontologies,
              "MapMethod": DimensionValues.MapMethodValues.Ontologies,
              "MapContent": DimensionValues.MapContentValues.Ontologies}
direct_ancestors = {"Spatial": DimensionValues.SpatialValue.Direct_Ancestors,
                    "Theme": DimensionValues.ThemeValues.Direct_Ancestors,
                    "MapMethod": DimensionValues.MapMethodValues.Direct_Ancestors,
                    "MapContent": DimensionValues.MapContentValues.Direct_Ancestors}
concept_information_content = {
    "Spatial": DimensionValues.SpatialValue.Information_Content,
    "Theme": DimensionValues.ThemeValues.Information_Content,
    "MapMethod": DimensionValues.MapMethodValues.Information_Content,
    "MapContent": DimensionValues.MapContentValues.Information_Content}


# 生成concept_id文件
def generate_concept_id_file():
    output_path = os.path.join(os.path.abspath(os.path.join(__dir__, "../../../../")),
                               "resources/samples/concept_id_dict.json")

    all_values = []
    for tmp_dim in ontologies:
        tmp_dim_values = ontologies[tmp_dim].keys()
        print(tmp_dim, len(tmp_dim_values))
        all_values += tmp_dim_values

    all_values = list(set(all_values))  # 去重
    concept_id_dict = {}
    for i, tmp_value in enumerate(all_values):
        concept_id_dict[tmp_value] = i
    save_as_json(concept_id_dict, output_path)


# 样本ID与样本在总库中的索引一一对应，提高检索效率
# 按照特定的比例（目标概念库：所有概念库）随机组合取值生成总样本集并以json形式保存到指定路径
# params
#   max_label_num：样本上每个维度上最大标签数量
#   all_samples_num：生成的总样本数量
#   output_path：根据具体意图的值生成的所有样本保存的位置
# max_label_num = {'S': 2, 'T': 2, 'M': 2, 'C': 2}
# intention = [{'S': 'Ohio', 'T': 'Water', 'M': 'Area Method', 'C': 'Temperature'}, {...}]
def generate_all_samples(max_label_num, all_samples_num, output_path):
    # 如果目录下存在all_samples.json文件，则不再重新生成
    if os.path.exists(output_path):
        return

    # 使用部分SWEET本体概念生成样本
    use_filtered_concepts = True
    # 按照4:1 的比例（目标概念库:所有念库）生成合成样本
    target_samples_ratio = 4 / 5
    target_samples_num = all_samples_num * target_samples_ratio
    rest_samples_num = all_samples_num - target_samples_num
    # 生成的所有样本
    all_samples = []
    # 单标签样本
    single_label_samples = []
    # 多标签样本
    multiple_label_samples = []
    # 每个维度上所有概念集合
    per_dimension_sample_values = {}
    # 获取目标概念集合
    target_per_dimension_sample_values = load_json("../../../../resources/samples/target_dim_concepts.json")
    # target_per_dimension_sample_values = ConceptIdTransform.sample_concept_to_id(target_per_dimension_sample_values)

    # 获取地理要素维度约束样本库的所有概念
    map_content_concepts_restriction = [
        "http://sweetontology.net/matr/Substance",
        "http://sweetontology.net/prop/ThermodynamicProperty",
        "http://sweetontology.net/matrBiomass/LivingEntity",
        "http://sweetontology.net/phenSystem/Oscillation",
        # "http://sweetontology.net/propQuantity/PhysicalQuantity",
        "http://sweetontology.net/phenAtmo/MeteorologicalPhenomena",
    ]
    map_content_values_set = []
    for tmp_concept in map_content_concepts_restriction:
        map_content_values_set.append(tmp_concept)
        map_content_values_set += ontologies['MapContent'][tmp_concept]
    # 加上地理要素维度的根节点
    map_content_values_set.append("Thing")
    map_content_values_set = list(set(map_content_values_set))
    print(len(map_content_values_set))

    for tmp_dim in intention_dimensions:
        tmp_dim_sample_values = None
        # 地图内容维度取值约束，只取部分SWEET本体，其余维度不做约束
        if tmp_dim == 'MapContent':
            if not use_filtered_concepts:
                tmp_dim_sample_values = list(ontologies[tmp_dim].keys())
            else:
                tmp_dim_sample_values = map_content_values_set
        else:
            tmp_dim_sample_values = list(ontologies[tmp_dim].keys())
        #  TODO：如果空间范围取值也有约束，则添加相应的elif分支
        per_dimension_sample_values[tmp_dim] = tmp_dim_sample_values
        # 若使用约束概念，则目标概念也应在约束概念范围内
        target_per_dimension_sample_values[tmp_dim] = \
            list(set(target_per_dimension_sample_values[tmp_dim]) & set(per_dimension_sample_values[tmp_dim]))

    count = 0  # 生成的样本数量
    # 按照4:1 的比例（目标概念:所有概念）生成合成样本
    for tmp_type in ["target_samples", "rest_samples"]:
        samples_num = 0
        tmp_count = 0
        if tmp_type == "target_samples":
            samples_num = target_samples_num
            tmp_per_dimension_sample_values = target_per_dimension_sample_values
        elif tmp_type == "rest_samples":
            samples_num = rest_samples_num
            tmp_per_dimension_sample_values = per_dimension_sample_values

        # 单标签样本数量：多标签样本数量 = 1:1
        single_label_samples_ratio = 0.5
        single_label_samples_num = samples_num * single_label_samples_ratio
        multiple_label_samples_num = samples_num - single_label_samples_num

        while tmp_count < samples_num:
            # 为每个样本添加一个ID属性（便于迭代反馈中，可追溯）
            tmp_sample = {"ID": count}
            # 生成单标签样本
            if tmp_count < single_label_samples_num:
                for tmp_dim in intention_dimensions:
                    tmp_sample[tmp_dim] = [random.choice(tmp_per_dimension_sample_values[tmp_dim])]
                # 如果所有维度都是根节点，则丢弃该样本，需重新生成
                all_none = True
                for tmp_dim in intention_dimensions:
                    if tmp_sample[tmp_dim] != [ontology_root[tmp_dim]]:
                        all_none = False
                        break
                if all_none:
                    continue
                else:
                    single_label_samples.append(tmp_sample)
                    tmp_count += 1
                    count += 1
            # 生成多标签样本
            else:
                for tmp_dim in intention_dimensions:
                    tmp_dim_max_label_num = max_label_num[tmp_dim]
                    tmp_dim_label_num = random.randint(1, tmp_dim_max_label_num)
                    tmp_dim_sample_values = tmp_per_dimension_sample_values[tmp_dim]
                    tmp_sample_tmp_dim_sample_values = random.sample(tmp_dim_sample_values, tmp_dim_label_num)

                    # if label is the root concept, then only reserve the root concept and take it as 'no label'
                    if ontology_root[tmp_dim] in tmp_sample_tmp_dim_sample_values:
                        tmp_sample_tmp_dim_sample_values = [ontology_root[tmp_dim]]
                    else:
                        tmp_sample_tmp_dim_sample_values = list(set(tmp_sample_tmp_dim_sample_values))
                    tmp_sample[tmp_dim] = tmp_sample_tmp_dim_sample_values
                # 如果所有维度都是根节点，则丢弃该样本，需重新生成
                all_none = True
                for tmp_dim in intention_dimensions:
                    if tmp_sample[tmp_dim] != [ontology_root[tmp_dim]]:
                        all_none = False
                        break
                if all_none:
                    continue
                # 如果所有维度都只有一个取值，不满足多标签样本条件，丢弃
                all_single_label = True
                for tmp_dim in intention_dimensions:
                    if len(tmp_sample[tmp_dim]) > 1:
                        all_single_label = False
                        break
                if all_single_label:
                    continue
                else:
                    multiple_label_samples.append(tmp_sample)
                    tmp_count += 1
                    count += 1

    all_samples = single_label_samples + multiple_label_samples
    # 样本ID与样本在总库中的索引一一对应，提高检索效率
    all_samples = sorted(all_samples, key=lambda x: x["ID"])
    save_as_json(single_label_samples, single_label_samples_path)
    save_as_json(multiple_label_samples, multiple_label_samples_path)
    save_as_json(all_samples, output_path)
    return None


# 将预定义意图文件从csv格式转为json
def get_intention_file_from_csv():
    content_concept = {"TransitionMetal": "http://sweetontology.net/matrElement/TransitionMetal",
                       "ExtrusiveRock": "http://sweetontology.net/matrRockIgneous/ExtrusiveRock",
                       "Temperature": "http://sweetontology.net/propTemperature/Temperature",
                       "VolcanicRock": "http://sweetontology.net/matrRockIgneous/VolcanicRock",
                       "Wave": "http://sweetontology.net/phenWave/Wave",
                       "Wind": "http://sweetontology.net/phenAtmoWind/Wind",
                       "Rock": "http://sweetontology.net/matrRock/Rock",
                       "Animal": "http://sweetontology.net/matrAnimal/Animal",
                       "None": "None"}
    intention_csv = load_csv(intention_csv_path)
    tmp_intention = []
    tmp_intention_name = None
    intentions = {}
    for tmp_line in intention_csv:
        if len(tmp_line[0]) > 0:
            # 如果已经开始下一个意图，则将意图放入意图哈希表
            intentions[copy.deepcopy(tmp_intention_name)] = copy.deepcopy(tmp_intention)
            tmp_intention_name = tmp_line[0].replace("场景", "Scene")
            tmp_intention = [{"Theme": tmp_line[1], "MapContent": content_concept[tmp_line[2]],
                              "Spatial": tmp_line[3], "MapMethod": tmp_line[4]}]
        else:
            # 当前意图还有其他的子意图，先把当前子意图放入临时意图
            tmp_intention.append(
                {"Theme": tmp_line[1], "MapContent": content_concept[tmp_line[2]],
                 "Spatial": tmp_line[3], "MapMethod": tmp_line[4]})
    intentions[copy.deepcopy(tmp_intention_name)] = copy.deepcopy(tmp_intention)
    del intentions[None]

    # 将intention中的无意图None,转换成根节点
    for tmp_intention in intentions.values():
        for sub_intention in tmp_intention:
            for tmp_dim in intention_dimensions:
                if sub_intention[tmp_dim] == "None":
                    sub_intention[tmp_dim] = ontology_root[tmp_dim]
    save_as_json(intentions, intention_path)


# 依据意图生成不同噪声比例的样本集合,作为初次反馈
def generate_samples_for_first_feedback():
    single_label_samples = load_json(single_label_samples_path)
    multiple_label_samples = load_json(multiple_label_samples_path)
    intentions = load_json(intention_path)

    noise_configs = [
        # {"feedback_noise": 0, "label_noise": 0},
        # {"feedback_noise": 0.1, "label_noise": 0},
        # {"feedback_noise": 0.2, "label_noise": 0},
        # {"feedback_noise": 0.3, "label_noise": 0},
        # {"feedback_noise": 0.4, "label_noise": 0},
        # {"feedback_noise": 0.5, "label_noise": 0},
        # {"feedback_noise": 0, "label_noise": 0.2},
        # {"feedback_noise": 0, "label_noise": 0.4},
        # {"feedback_noise": 0, "label_noise": 0.6},
        # {"feedback_noise": 0, "label_noise": 0.8},
        # {"feedback_noise": 0, "label_noise": 1},
        # {"feedback_noise": 0.1, "label_noise": 0.2},
        # {"feedback_noise": 0.1, "label_noise": 0.4},
        # {"feedback_noise": 0.1, "label_noise": 0.6},
        # {"feedback_noise": 0.1, "label_noise": 0.8},
        # {"feedback_noise": 0.1, "label_noise": 1, },
        # {"feedback_noise": 0.2, "label_noise": 0.2, },
        # {"feedback_noise": 0.2, "label_noise": 0.4, },
        # {"feedback_noise": 0.2, "label_noise": 0.6, },
        {"feedback_noise": 0.2, "label_noise": 0.8, },
        # {"feedback_noise": 0.2, "label_noise": 1, },
        # {"feedback_noise": 0.3, "label_noise": 0.2, },
        # {"feedback_noise": 0.3, "label_noise": 0.4, },
        {"feedback_noise": 0.3, "label_noise": 0.6, },
        # {"feedback_noise": 0.3, "label_noise": 0.8, },
        # {"feedback_noise": 0.3, "label_noise": 1, },
        # {"feedback_noise": 0.4, "label_noise": 0.2, },
        {"feedback_noise": 0.4, "label_noise": 0.4, },
        # {"feedback_noise": 0.4, "label_noise": 0.6, },
        {"feedback_noise": 0.4, "label_noise": 0.8, },
        # {"feedback_noise": 0.4, "label_noise": 1, "max_label_num": max_label_num},
        # {"feedback_noise": 0.5, "label_noise": 0.2, "max_label_num": max_label_num},
        # {"feedback_noise": 0.5, "label_noise": 0.4, "max_label_num": max_label_num},
        # {"feedback_noise": 0.5, "label_noise": 0.6, "max_label_num": max_label_num},
        # {"feedback_noise": 0.5, "label_noise": 0.8, "max_label_num": max_label_num},
        # {"feedback_noise": 0.5, "label_noise": 1, "max_label_num": max_label_num},
    ]

    # 每个意图场景生成样本
    for scene_name in intentions.keys():

        tmp_intention = intentions[scene_name]

        # 样本总集
        all_samples = single_label_samples + multiple_label_samples

        # 所有候选正样本索引
        all_candidate_positive_samples_index, _ = RetrievalUtil.get_intention_covered_samples(tmp_intention,
                                                                                              all_samples,
                                                                                              ontologies)
        # 所有候选负样本索引
        all_candidate_negative_samples_index = set(range(len(all_samples))) - set(all_candidate_positive_samples_index)

        # 每个子意图的候选真正正样本集合
        candidate_true_positive_samples_index = []
        # 每个子意图的候选标签噪声集合
        candidate_label_noise_positive_samples_index = []
        for i, sub_intention in enumerate(tmp_intention):
            # 每个子意图的候选真正正样本集合
            candidate_true_positive_samples_index.append(
                set(RetrievalUtil.get_sub_intention_covered_samples(sub_intention, single_label_samples, ontologies)[0]))
            # 每个子意图的候选标签噪声集合
            candidate_label_noise_positive_samples_index.append(
                set(RetrievalUtil.get_sub_intention_covered_samples(sub_intention, multiple_label_samples, ontologies)[0]))

            print(f"the {i} th intention:")
            print(f"candidate true positive samples num: {len(candidate_true_positive_samples_index[i])}")
            print(f"candidate label noise num: {len(candidate_label_noise_positive_samples_index[i])}")

        # 依据意图选择样本，生成不同噪声比例的样本集合
        for tmp_noise_config in noise_configs:
            export_scene_name = scene_name
            feedback_noise_rate = tmp_noise_config["feedback_noise"]
            feedback_noise_rate_str = str(feedback_noise_rate).replace(".", "p")
            label_noise_rate = tmp_noise_config["label_noise"]
            label_noise_rate_str = str(label_noise_rate).replace(".", "p")
            export_scene_name += "_F" + feedback_noise_rate_str
            export_scene_name += "_L" + label_noise_rate_str

            tmp_scene_path = os.path.join(all_output_path_prefix, export_scene_name)
            if not os.path.exists(tmp_scene_path):
                os.mkdir(tmp_scene_path)
            print(
                f"generating {scene_name}, with feedback_noise_rate: {feedback_noise_rate} label_noise_rate: {label_noise_rate}")

            positive_samples, negative_samples = \
                select_samples_by_intention(tmp_intention, all_samples, tmp_noise_config,
                                            all_candidate_positive_samples_index, all_candidate_negative_samples_index,
                                            candidate_true_positive_samples_index,
                                            candidate_label_noise_positive_samples_index)
            # 输出xlsx和json文件
            tmp_export_paths = get_export_paths_by_noise_config(tmp_scene_path, tmp_noise_config)
            result = {"intention": tmp_intention, "positive_samples": positive_samples,
                      "negative_samples": negative_samples}
            export_samples(result, tmp_export_paths)


# 初次反馈样本的输出路径
# param:
#      dirname：场景文件夹路径
#      noise_config: {"feedback noise": 0, "label noise": 0} 样本及标签噪声比例
def get_export_paths_by_noise_config(dirname, noise_config):
    export_file_name = "samples"

    feedback_noise_rate = noise_config["feedback_noise"]
    feedback_noise_rate_str = str(feedback_noise_rate).replace(".", "p")
    label_noise_rate = noise_config["label_noise"]
    label_noise_rate_str = str(label_noise_rate).replace(".", "p")

    # export_file_name += "_F" + feedback_noise_rate_str
    # export_file_name += "_L" + label_noise_rate_str
    export_paths = [os.path.join(dirname, export_file_name + "_I0.json"),
                    os.path.join(dirname, export_file_name + "_I0.xlsx")]
    return export_paths


# 依据意图选择样本，生成不同噪声比例的样本集合
# 正样本集合： 每个子意图对应的真正正样本，反馈噪声，每个子意图对应的标签噪声
# 负样本集合： 真正负样本，反馈噪声
# 选择流程：
# ① 从每个子意图覆盖的候选正样本中，等量选择真正正样本和对应标签噪声，在所有候选的正样本中去除已选的样本
# ② 从所有的候选负样本中选择正样本集合中的反馈噪声，在所有候选负样本中去除已选的正样本集合反馈噪声
# ③ 从所有的候选负样本中选择真正负样本，在所有候选负样本中去除已选的真正负样本
# ④ 从所有的候选正样本中选择负样本集合中的反馈噪声，在所有候选正样本中去除已选的负样本集合反馈噪声
# param:
#      intention: 预定义意图（list)
#      all_samples: 样本总库(list)
#      noise_config: {"feedback noise": 0, "label noise": 0} 样本及标签噪声比例
#      all_candidate_positive_samples_index, all_candidate_negative_samples_index 候选正（负）样本ID
#      candidate_true_positive_samples_index, candidate_label_noise_positive_samples_index  候选真正正样本及标签噪声ID
def select_samples_by_intention(intention, all_samples, noise_config, all_candidate_positive_samples_index,
                                all_candidate_negative_samples_index, candidate_true_positive_samples_index,
                                candidate_label_noise_positive_samples_index):
    selected_positive_negative_samples_num = 50  # 给定反馈正负样本数量

    tmp_all_candidate_positive_samples_index = copy.deepcopy(all_candidate_positive_samples_index)
    tmp_all_candidate_negative_samples_index = copy.deepcopy(all_candidate_negative_samples_index)

    positive_samples = []  # 选择的正样本
    negative_samples = []  # 选择的负样本

    # 确定反馈噪声比例
    current_feedback_noise = noise_config["feedback_noise"]
    current_label_noise = noise_config["label_noise"]

    print(f"feedback noise: {current_feedback_noise}, label noise: {current_label_noise}")

    # 子意图数量
    k = len(intention)

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

    # 每个子意图的候选真正正样本集合
    tmp_candidate_true_positive_samples_index = copy.deepcopy(candidate_true_positive_samples_index)
    # 每个子意图的候选标签噪声集合
    tmp_candidate_label_noise_positive_samples_index = copy.deepcopy(candidate_label_noise_positive_samples_index)

    # 为每个子意图选择等量的真实正样本和标签噪声
    for i, sub_intention in enumerate(intention):
        # 从候选真正正样本集合中随机选择
        selected_true_positive_samples_index = random.sample(list(tmp_candidate_true_positive_samples_index[i]),
                                                             min(len(tmp_candidate_true_positive_samples_index[i]),
                                                                 sub_intention_cover_true_positive_samples_num[i]))
        # 从候选标签噪声集合中随机选择
        selected_label_noise_positive_samples_index = \
            random.sample(list(tmp_candidate_label_noise_positive_samples_index[i]),
                          min(len(tmp_candidate_label_noise_positive_samples_index[i]),
                              sub_intention_cover_label_noise_positive_samples_num[i]))
        note_and_add_sample(all_samples, "true positive sample", selected_true_positive_samples_index, positive_samples)
        note_and_add_sample(all_samples, "label noise", selected_label_noise_positive_samples_index, positive_samples)

        # 在候选正样本集合及下一个子意图覆盖的单（多）标签样本中去除已选元素
        tmp_all_candidate_positive_samples_index = set(tmp_all_candidate_positive_samples_index) - \
                                                   set(selected_true_positive_samples_index) - \
                                                   set(selected_label_noise_positive_samples_index)
        if i < k - 1:
            tmp_candidate_true_positive_samples_index[i + 1] -= set(selected_true_positive_samples_index)
            tmp_candidate_label_noise_positive_samples_index[i + 1] -= set(selected_label_noise_positive_samples_index)

    # 随机挑选反馈噪声添加进正样本中
    selected_feedback_noise_index_in_positive = \
        random.sample(list(tmp_all_candidate_negative_samples_index),
                      min(len(tmp_all_candidate_negative_samples_index), selected_feedback_noise_samples_num))
    note_and_add_sample(all_samples, "feedback noise in positive", selected_feedback_noise_index_in_positive,
                        positive_samples)
    # 在候选负样本集合中去除已选元素
    tmp_all_candidate_negative_samples_index -= set(selected_feedback_noise_index_in_positive)

    # 随机挑选真实负样本
    selected_true_negative_samples_index = \
        random.sample(list(tmp_all_candidate_negative_samples_index),
                      min(len(tmp_all_candidate_negative_samples_index), selected_true_negative_samples_num))
    note_and_add_sample(all_samples, "true negative sample", selected_true_negative_samples_index, negative_samples)
    # 在候选负样本集合中去除已选元素
    tmp_all_candidate_negative_samples_index -= set(selected_true_negative_samples_index)

    # 随机挑选反馈噪声添加进负样本中
    selected_feedback_noise_index_in_negative = \
        random.sample(list(tmp_all_candidate_positive_samples_index),
                      min(len(tmp_all_candidate_positive_samples_index), selected_feedback_noise_samples_num))
    note_and_add_sample(all_samples, "feedback noise in negative", selected_feedback_noise_index_in_negative,
                        negative_samples)
    # 在候选正样本集合中去除已选元素
    tmp_all_candidate_positive_samples_index -= set(selected_feedback_noise_index_in_negative)

    print(f"positive_samples_num: {len(positive_samples)}")
    print(f"negative_samples_num: {len(negative_samples)}")

    # sorted(positive_samples, key=lambda x: x["ID"])
    # sorted(negative_samples, key=lambda x: x["ID"])
    return positive_samples, negative_samples


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
    # 生成concept_id文件
    # generate_concept_id_file()

    # # 生成总样本集，设置为多标签
    # max_label_num = {'Spatial': 2, 'Theme': 2, 'MapMethod': 2, 'MapContent': 2}
    # # 数量为五十万
    # all_samples_num = 500_000
    # generate_all_samples(max_label_num, all_samples_num,
    #                      all_samples_path)

    # 将预定义意图文件从csv格式转为json
    # 提醒:将xlsx另存为csv utf-8格式后，还需要手动将空白的单元格替换为None
    # get_intention_file_from_csv()

    # 为每个意图场景挑选不同噪声比例的初次反馈样本
    generate_samples_for_first_feedback()

    print("Aye")
