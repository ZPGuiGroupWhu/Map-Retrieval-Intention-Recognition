import copy
import json
import pandas as pd
import sys
import os
import random
from itertools import combinations
import math

import matplotlib.pyplot as plt
from collections import Counter

from FileUtil import load_json, save_as_json

# 堆栈递归深度
sys.setrecursionlimit(1400)

# <editor-folder desc = "Tree, AllIntents">

#制图方法6    主题10
AllIntent = { }
AllIntent["T"] = ["Agriculture","Biodiversity","Climate","Disaster","Ecosystem",
         "Energy","Geology","Health","Water","Weather"]
AllIntent["M"] = ["Raw Image","Area Method","Quality Base Method","Point Symbol Method",
           "Line Symbol Method","Choloplethic Method","Others"]
AllIntent["C"] = load_json("../SweetData/all_concepts.json")
AllIntent["S"] = load_json("../SweetData/Lall_concepts.json")

# 空树
def GenerateTree(dim):
    tree={}
    for value in AllIntent[dim]:
        tree[value] = []
    return tree

# 所有树状结构
Tree = {}
Tree["C"] = {}
#所有上位概念，不包括自己
Tree["C"]["all_ancestors"] = load_json("../SweetData/all_ancestors.json")
#所有下位概念，不包括自己
Tree["C"]["all_hyponyms"] = load_json("../SweetData/all_hyponyms.json")
#相等概念，不包括自己
Tree["C"]["equivalent_relations"] = load_json("../SweetData/equivalent_relations.json")
# 每一个节点对应的根节点
Tree["C"]["all_roots"] = load_json("../SweetData/all_roots.json")

Tree["S"] = {}
#所有上位概念，不包括自己
Tree["S"]["all_ancestors"] = load_json("../SweetData/Lall_ancestors.json")
#所有下位概念，不包括自己
Tree["S"]["all_hyponyms"] = load_json("../SweetData/Lall_hyponyms.json")
#相等概念，不包括自己
Tree["S"]["equivalent_relations"] = GenerateTree("S")
# 每一个节点对应的根节点
Tree["S"]["all_roots"] = load_json("../SweetData/Lall_roots.json")

Tree["T"] = {}
#所有上位概念，不包括自己
Tree["T"]["all_ancestors"] = GenerateTree("T")
#所有下位概念，不包括自己
Tree["T"]["all_hyponyms"] = GenerateTree("T")
#相等概念
Tree["T"]["equivalent_relations"] = GenerateTree("T")
# 每一个节点对应的根节点
Tree["T"]["all_roots"] = GenerateTree("T")

Tree["M"] = {}
#所有上位概念，不包括自己
Tree["M"]["all_ancestors"] = GenerateTree("M")
#所有下位概念，不包括自己
Tree["M"]["all_hyponyms"] = GenerateTree("M")
#相等概念，不包括自己
Tree["M"]["equivalent_relations"] = GenerateTree("M")
# 每一个节点对应的根节点，不包括自己
Tree["M"]["all_roots"] = GenerateTree("M")

# </editor-folder>

# 无意图
def NoIntention(p,num,repeat_index):

    Samples = {}
    Samples['intentions'] = [{"C":[],"T":[],"M":[],"S":[]}]
    Samples['positive_samples'] = []
    Samples['negative_samples'] = []

    workbook = pd.read_csv('PreProgressedSamples.csv', encoding='utf-8')
    sample_num = workbook.shape[0]

    # 正样本
    sample_id = random.sample(range(0,sample_num),num)
    for i in sample_id:
        onesample = {"C": [], "M": [], "T": [], "S": []}  # 修改读取到的样本格式
        Cs = str(workbook["Content"][i]).split(",")
        if Cs != ['nan']: onesample["C"] = Cs
        Ts = str(workbook["Topic"][i]).split(",")
        if Ts != ['nan']: onesample["T"] = Ts
        Ms = str(workbook["Mapping"][i]).split(",")
        if Ms != ['nan']: onesample["M"] = Ms
        Ss = str(workbook["Space"][i]).split(",")
        if Ss != ['nan']: onesample["S"] = Ss

        Samples['positive_samples'].append(onesample)

    Path = "SampleSelector\\S1\\" + str(1) + "\\num" + str(num) + "\\p" + str(p) + "_" + str(repeat_index) + ".json"

    if not os.path.exists(os.path.dirname(Path)):
        os.makedirs(os.path.dirname(Path))
    save_as_json(Samples, str(Path))

    return Samples

# 单个样本的意图满足率
def Calculate_p(sample,intention):

    Pos_tag = 0
    Pos_num = 0
    for i_dim, i_values in intention.items():
        Pos_num += len(i_values)
        for i_c in i_values:
            Pos_group = []
            # 概念
            Pos_group += [i_c]
            # 下位概念
            Pos_group += Tree[i_dim]["all_hyponyms"][i_c]
            # 相等概念
            Pos_group += Tree[i_dim]["equivalent_relations"][i_c]
            # 上位概念
            Pos_group += Tree[i_dim]["all_ancestors"][i_c]

            for s_dim, s_values in sample.items():
                for s_c in s_values:
                    if s_c in Pos_group:
                        Pos_tag += 1
                        break
                else:
                    continue
                break # 内层被break，外层break

    if Pos_num !=0 :
        p = Pos_tag / Pos_num
    else:
        p=1
    return p

def pick_random_samples(original_list, num_list):
    if num_list <= len(original_list):
        result = random.sample(original_list, num_list)  # 从原始列表中随机挑选指定数量的值
    else:
        result = original_list
        while len(result) < num_list:
            result.append(random.choice(original_list))  # 随机重复部分元素直到达到指定数量
    return result

# 单意图
def OneIntGenerate(intention,p,num,Path=''):
    intSamples = {}
    # 与多意图统一格式，key命名为intentions
    intSamples["intentions"] = intention
    intSamples['positive_samples'] = []
    intSamples['negative_samples'] = []
    oneIntention = intention[0]

    # 与意图相关的标签列表
    oneIntGroup = {}
    ancestors_group = {}
    for dim, values in oneIntention.items():
        oneIntGroup[dim] = {}
        ancestors_group[dim] = {}
        for v in values:
            if dim == "S" and v == "Global":
                # 单独处理 Global
                oneIntGroup[dim][v] = []
                # 概念
                oneIntGroup[dim][v] += [v]

                # 单独处理 Global
                ancestors_group[dim][v] = []

            else:
                oneIntGroup[dim][v] = []
                # 概念
                oneIntGroup[dim][v] += [v]
                # 下位概念
                oneIntGroup[dim][v] += Tree[dim]["all_hyponyms"][v]
                # 相等概念
                oneIntGroup[dim][v] += Tree[dim]["equivalent_relations"][v]
                # # 上位概念
                ancestors_group[dim][v] = []
                ancestors_group[dim][v] += Tree[dim]["all_ancestors"][v]

    # 读取样本库
    workbook = pd.read_csv('PreProgressedSamples.csv', encoding='utf-8')
    all_sample_num = workbook.shape[0]

    # 筛选样本
    pos_samples = []
    neg_samples = []
    pos_anc_samples = []

    for i in range(all_sample_num):
        onesample = {"C": [], "M": [], "T": [],"S":[]}  # 修改读取到的样本格式
        Cs = str(workbook["Content"][i]).split(",")
        if Cs != ['nan']: onesample["C"] = Cs
        Ts = str(workbook["Topic"][i]).split(",")
        if Ts != ['nan']: onesample["T"] = Ts
        Ms = str(workbook["Mapping"][i]).split(",")
        if Ms != ['nan']: onesample["M"] = Ms
        Ss = str(workbook["Space"][i]).split(",")
        if Ss != ['nan']: onesample["S"] = Ss

        # 判断样本是否为正样本
        pos_flag_list = []  # 正样本标签
        # 0 无关标签，1 与正样本标签相关的标签，-1 负样本标签，在同一个根节点下，但是不被意图标签覆盖，在意图识别时有干扰
        for dim, values in oneIntGroup.items():
            # 维度值域不为空，该维度有意图
            if values:
                for value, value_list in values.items():
                    # 每一个意图标签占位0
                    pos_flag_list.append(0)
                    # 遍历样本当前对应维度的所有取值
                    for sample_value in onesample[dim]:
                        # 说明当前意图标签被样本说明
                        if sample_value in value_list:
                            pos_flag_list[-1] = 1
                            break
                        elif sample_value in ancestors_group[dim][value]:
                            # 该标签为上位意图标签
                            pos_flag_list[-1] = 2
                        # 说明这个样本标签是干扰标签
                        elif Tree[dim]["all_roots"][value] == Tree[dim]["all_roots"][sample_value] != []:
                            pos_flag_list[-1] = -1

        # 正样本 全1
        if all(elem == 1 for elem in pos_flag_list):
            pos_samples.append(onesample)
        # 上位正样本 只有1 和 2
        elif all(elem in [1,2] for elem in pos_flag_list):
            pos_anc_samples.append(onesample)
        # 负样本 不全为0
        elif any(elem != 0 for elem  in pos_flag_list):
            neg_samples.append(onesample)
        # 无关样本 全0

    if len(neg_samples)>0 and len(pos_samples) >0:
        # 挑选有代表性指定数量的样本
        # 挑选指定数量的样本
        neg_num = math.ceil(num * p)
        pos_num = num - neg_num
        repeat_flag = 1
        max_repeat = 100
        count = 0
        while repeat_flag and count < max_repeat:
            count = count + 1
            intSamples['negative_samples'] = pick_random_samples(neg_samples, neg_num)
            # 每组0.2比例的上位概念噪声
            pos_anc_num = math.ceil(pos_num * 0.2)
            intSamples['positive_samples'] = []
            if pos_anc_num < len(pos_anc_samples):
                intSamples['positive_samples'] += pick_random_samples(pos_anc_samples, pos_anc_num)
                intSamples['positive_samples'] += pick_random_samples(pos_samples, (pos_num - pos_anc_num))
            else:
                intSamples['positive_samples'] += pos_anc_samples
                intSamples['positive_samples'] += pick_random_samples(pos_samples, (pos_num - len(pos_anc_samples)))


            repeat_flag=0 # 不评价代表性，随机选样本
            # # 评价正样本代表性
            # # 每组中有直接的意图标签
            # int_flag = []
            # for dim, int_values in intSamples['intentions'][0].items():
            #     for val in int_values:
            #         int_flag.append(0)
            #         for one_pos_sample in intSamples['positive_samples']:
            #             if val in one_pos_sample[dim]:
            #                 int_flag[-1] = 1
            #                 break
            #
            # # 全1全命中
            # if all(x == 1 for x in int_flag):
            #     repeat_flag = 0

    print("noise rate:",p,"pos_num:",len(pos_samples),"pos_anc_num:",len(pos_anc_samples),"neg_num:",len(neg_samples))

    if Path:
        if not os.path.exists(os.path.dirname(Path)):
            os.makedirs(os.path.dirname(Path))
        save_as_json(intSamples, Path)

    return intSamples

# 单意图场景p：噪声率，
def OneIntScenes(p,num,repeat_index):

    #<editor-folder desc="s2">
    S2_path = []
    S2_scene = []

    S2_path.append("SampleSelector\\S2\\" + str(1) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S2_scene.append([{"C": ["http://sweetontology.net/realmHydroBody/BodyOfWater"],
                      "T": [],
                      "M": [],
                      "S": []}
                     ])

    S2_path.append("SampleSelector\\S2\\" + str(2) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S2_scene.append([{"C": [],
                      "T": [],
                      "M": ["Area Method","Line Symbol Method"],
                      "S": []}
                     ])

    S2_path.append("SampleSelector\\S2\\" + str(3) + "\\num" + str(num) + "\\p" + str(p) + "_" + str(repeat_index) + ".json")
    S2_scene.append([{"C": ["http://sweetontology.net/humanKnowledgeDomain/Science","http://sweetontology.net/matrSediment/Soil"],
                      "T": [],
                      "M": [],
                      "S": []}
                     ])

    S2_path.append("SampleSelector\\S2\\" + str(4) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S2_scene.append([{"C": [],
                      "T": ["Agriculture","Water"],
                      "M": [],
                      "S": []}
                     ])

    S2_path.append("SampleSelector\\S2\\" + str(5) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S2_scene.append([{"C": [],
                      "T": [],
                      "M": [],
                      "S": ["United Kingdom"]}
                     ])

    #</editor-folder>

    #<editor-folder desc="s3"
    S3_path = []
    S3_scene = []

    S3_path.append("SampleSelector\\S3\\" + str(1) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S3_scene.append([{"C": ["http://sweetontology.net/stateRoleGeographic/Boundary"],
                      "T": ["Geology"],
                      "M": ["Area Method"],
                      "S": []}
                     ])

    S3_path.append("SampleSelector\\S3\\" + str(2) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S3_scene.append(
        [{"C": ["http://sweetontology.net/realmLandCoastal/Shoreline", "http://sweetontology.net/phenSystem/Change"],
          "T": ["Geology"],
          "M": ["Line Symbol Method"],
          "S": ["United States"]}
         ])

    S3_path.append("SampleSelector\\S3\\" + str(3) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S3_scene.append([{"C": ["http://sweetontology.net/phenAtmoWind/Wind"],
                      "T": [],
                      "M": ["Choloplethic Method"],
                      "S": ["Global"]}
                     ])

    S3_path.append("SampleSelector\\S3\\" + str(4) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S3_scene.append([{"C": ["http://sweetontology.net/humanKnowledgeDomain/Science","http://sweetontology.net/matrSediment/Soil"],
                      "T": [],
                      "M": ["Choloplethic Method"],
                      "S": []}
                     ])

    S3_path.append("SampleSelector\\S3\\" + str(5) + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json")
    S3_scene.append([{"C": ["http://sweetontology.net/realmHydroBody/BodyOfWater"],
                      "T": [],
                      "M": ["Area Method"],
                      "S": ["Europe"]}
                     ])

    #</editor-folder>

    for i in range(len(S2_path)):
        print("S2",i+1,'*************************')
        OneIntGenerate(S2_scene[i], p,num,S2_path[i])

    for i in range(len(S3_path)):
        print("S3",i+1,'*************************')
        OneIntGenerate(S3_scene[i], p,num, S3_path[i])

    return

# 根据单意图合并为多意图
def ComBinSamp(BaseScenePath,OutPath, p,num,repeat_index):

    # 指定文件夹路径
    # 单意图
    folder_path = BaseScenePath

    # 指定读取的文件夹名称
    folders = os.listdir(folder_path)

    # 循环读取每两个不同的文件夹中的json文件
    for i in range(len(folders)-1):
        for j in range(i + 1, len(folders)):
            intSamples = {}
            intSamples["intentions"] = []
            intSamples['positive_samples'] = []
            intSamples['negative_samples'] = []
            folder_a_path = os.path.join(folder_path, folders[i])
            folder_b_path = os.path.join(folder_path, folders[j])

            # a的正样本和b的正样本组合成positive_sample
            files_a = folder_a_path + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json"
            files_b = folder_b_path + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json"

            sample_a = load_json(files_a)
            sample_b = load_json(files_b)

            intSamples["intentions"] = sample_a["intentions"] + sample_b["intentions"]

            # 各取a b 有代表性的一半，组成新的样本库
            # 挑选指定数量的样本
            neg_num = math.ceil(num * p)
            pos_num = num-neg_num
            pos_ab_num = math.ceil(pos_num / 2)
            neg_ab_num = math.ceil(neg_num / 2)

            pos_half_sample_a = []
            pos_half_sample_b = []

            repeat_flag = 1
            max_repeat_times = 100
            count = 0
            while repeat_flag and count <max_repeat_times:
                count = count + 1
                pos_half_sample_a = pick_random_samples(sample_a['positive_samples'], pos_ab_num)

                repeat_flag = 0 # 不评价代表性
                # # 评价正样本代表性
                # # 每组中有直接的意图标签
                # int_flag = []
                # for dim, int_values in sample_a["intentions"][0].items():
                #     for val in int_values:
                #         int_flag.append(0)
                #         for one_pos_sample in pos_half_sample_a:
                #             if val in one_pos_sample[dim]:
                #                 int_flag[-1] = 1
                #                 break
                #
                # # 全1全命中
                # if all(x == 1 for x in int_flag):
                #     repeat_flag = 0

            # if count == 100:
            #     print("aaaaaaa")
            #     print(files_a,"  ",files_b)

            repeat_flag = 1
            count = 0
            while repeat_flag and count <max_repeat_times:
                count = count + 1
                pos_half_sample_b = pick_random_samples(sample_b['positive_samples'], (pos_num - pos_ab_num))

            repeat_flag = 0 # 不评价代表性
            #     # 评价正样本代表性
            #     # 每组中有直接的意图标签
            #     int_flag = []
            #     for dim, int_values in sample_b["intentions"][0].items():
            #         for val in int_values:
            #             int_flag.append(0)
            #             for one_pos_sample in pos_half_sample_b:
            #                 if val in one_pos_sample[dim]:
            #                     int_flag[-1] = 1
            #                     break
            #
            #     # 全1全命中
            #     if all(x == 1 for x in int_flag):
            #         repeat_flag = 0
            #
            # if count == 100:
            #     print("bbbbbbbb")
            #     print(files_a,"  ",files_b)

            # 组合正样本
            intSamples['positive_samples'] = pos_half_sample_a + pos_half_sample_b
            # 随机挑选负样本
            if neg_ab_num>0:
                intSamples['negative_samples'] = \
                    pick_random_samples(sample_a['negative_samples'],neg_ab_num) \
                    + pick_random_samples(sample_b['negative_samples'], (neg_num - neg_ab_num))

            # 将数据写入新的文件
            output_folder_path = os.path.join(OutPath,"{}_{}".format(folders[i],folders[j]))
            output_file_path = output_folder_path + "\\num" + str(num) + "\\p" + str(p) + "_" +str(repeat_index) + ".json"

            # 新建文件夹
            if not os.path.exists(os.path.dirname(output_file_path)):
                os.makedirs(os.path.dirname(output_file_path))

            save_as_json(intSamples, output_file_path)

    return

if __name__ == "__main__":

    # P = [[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]] # 意图满足率
    P = [0,0.2,0.4,0.6,0.8] # 噪声率
    Num = [10,20,40,60,80,100] # 样本数量

    repeat_index = 3

    BaseScenePath = "SampleSelector\\S3"
    OutPath = "SampleSelector\\S5"
    for p in P:
        for num in Num:
            # 单意图场景
            # OneIntScenes(p,num,repeat_index)
            # 多意图场景
            ComBinSamp(BaseScenePath, OutPath, p, num, repeat_index)

    # # 无意图场景
    # for num in Num:
    #     NoIntention(0, num, repeat_index)

    print('well done')
