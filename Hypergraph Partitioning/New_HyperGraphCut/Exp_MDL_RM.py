import math
import numpy as np

from scipy.optimize import linear_sum_assignment
import pandas as pd

from FileUtil import load_json
from GenerateSweet import SWEET
from GenerateGeoNames import GeoNames
from MDL_RM.src.main.intention_recognition import Config,Run_MDL_RM

import os
import csv


# "all_ancestors", "all_hyponyms", "equivalent_relations", "all_roots"
#<editor-fold desc = "Tree,AllIntent">
DimList = ["C","M","S","T"]

# 所有概率 制图方法6    主题10
AllIntent = {}
AllIntent["T"] = ["Agriculture","Biodiversity","Climate","Disaster","Ecosystem",
         "Energy","Geology","Health","Water","Weather","nan"]
AllIntent["M"] = ["Raw Image","Area Method","Quality Base Method","Point Symbol Method",
           "Line Symbol Method","Choloplethic Method","Others"]
AllIntent["C"] = SWEET.all_concepts
AllIntent["S"] = GeoNames.all_concepts

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
Tree["C"]["all_ancestors"] = SWEET.all_ancestors
#所有下位概念，不包括自己
Tree["C"]["all_hyponyms"] = SWEET.all_hyponyms
#相等概念，不包括自己
Tree["C"]["equivalent_relations"] = SWEET.equivalent_relations
# 每一个节点对应的根节点
Tree["C"]["all_roots"] = SWEET.all_roots

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

Tree["S"] = {}
#所有上位概念，不包括自己
Tree["S"]["all_ancestors"] = GeoNames.all_ancestors
#所有下位概念，不包括自己
Tree["S"]["all_hyponyms"] = GeoNames.all_hyponyms
#相等概念，不包括自己
Tree["S"]["equivalent_relations"] = GenerateTree("S")
# 每一个节点对应的根节点
Tree["S"]["all_roots"] = GeoNames.all_roots

# 不包括自己
def RelateConcept(content,dim = ''):
    concepts = []
    if dim:
        concepts = Tree[dim]["all_ancestors"] + Tree[dim]["all_hyponyms"]+ Tree[dim]["equivalent_relations"]
    else:
        for dim in DimList:
            if content in AllIntent[dim]:
                concepts = Tree[dim]["all_ancestors"][content] + Tree[dim]["all_hyponyms"][content]+ Tree[dim]["equivalent_relations"][content]
                break

    return concepts

#</editor-fold>

# <editor-folder desc = "Calculate">

# 计算信息熵
def CalIC(content, dim):
    IC = 0
    if content == 'root':
        IC = 0
    else:
        N = len(AllIntent[dim])
        n = len(Tree[dim]["all_hyponyms"][content]) + 1

        if N != 0:
            IC = - math.log(n / N, math.e)

    return IC


# 计算某维度的距离
def CalICDis(content1, content2, dim):
    dis = 0
    maxIC = 0
    if content1 != 'root' and content2 != 'root':
        if content2 in Tree[dim]["equivalent_relations"][content1]:
            dis = 0
        else:
            comm_ancestors = set(Tree[dim]["all_ancestors"][content1] + [content1]).intersection(
                set(Tree[dim]["all_ancestors"][content2] + [content2]))
            if comm_ancestors:
                maxIC = max(list(map(lambda i: CalIC(i, dim), comm_ancestors)))
            else:
                maxIC = 0
    else:
        maxIC = 0

    dis = CalIC(content1, dim) + CalIC(content2, dim) - 2 * maxIC

    return dis


# 意图间的距离,与CalRelI_PreIDis的区别是对[]的处理不一样
def CalIntDis(intention1, intention2):
    # 补全长度不一致
    dimlen = {}
    for dim in DimList:
        dimlen[dim] = max(len(intention1[dim]), len(intention2[dim]))


    # 维度距离
    # 全部上位共同节点，最大信息量共同节点
    disList = []
    for dim in DimList:

        if intention1[dim] == [] or intention2[dim] == []:
            dimdis = 0
            for content in intention1[dim]:
                dimdis += CalIC(content, dim)
            for content in intention2[dim]:
                dimdis += CalIC(content, dim)

        else:
            temp_intention1 = intention1[dim] + ['root'] * (dimlen[dim] - len(intention1[dim]))
            temp_intention2 = intention2[dim] + ['root'] * (dimlen[dim] - len(intention2[dim]))
            dimdis = float('inf')
            if temp_intention1 == temp_intention2:  # 考虑相等概念
                dimdis = 0
            else:
                # 单维度相似度距离

                # 全局最优解
                # 距离矩阵
                size = dimlen[dim]
                disMatrix = np.full((size, size), float('inf'))
                for i in range(size):
                    for j in range(size):
                        disMatrix[i][j] = CalICDis(temp_intention1[i], temp_intention2[j], dim)

                row_ind, col_ind = linear_sum_assignment(disMatrix)
                dimdis = disMatrix[row_ind, col_ind].sum()

        disList.append(dimdis)

    # 三维度二范数round(
    return np.linalg.norm(disList)


def CalRelI_PreIDis(RelI, PreI):
    # 当预测意图能覆盖真意图，则距离为0

    # 补全长度不一致
    dimlen = {}
    for dim in DimList:
        dimlen[dim] = max(len(RelI[dim]), len(PreI[dim]))

    # 维度距离
    # 全部上位共同节点，最大信息量共同节点
    disList = []
    for dim in DimList:
        dimdis = 0
        if RelI[dim] == []:
            dimdis = 0
        elif PreI[dim] == []:
            for content in RelI[dim]:
                dimdis += CalIC(content, dim)
        elif set(RelI[dim]).issubset(set(PreI[dim])):  # 包含关系
            dimdis = 0
        else:
            temp_RelI = RelI[dim] + ['root'] * (dimlen[dim] - len(RelI[dim]))
            temp_PreI = PreI[dim] + ['root'] * (dimlen[dim] - len(PreI[dim]))
            dimdis = 0
            # 单维度相似度距离
            # 全局最优解
            # 距离矩阵
            size = dimlen[dim]
            disMatrix = np.full((size, size), float('inf'))
            for i in range(size):
                for j in range(size):
                    disMatrix[i][j] = CalICDis(temp_RelI[i], temp_PreI[j], dim)

            row_ind, col_ind = linear_sum_assignment(disMatrix)
            dimdis = disMatrix[row_ind, col_ind].sum()
            # 维度内每个标签的距离之和

        disList.append(dimdis)

    # 四维度二范数
    # return  (np.linalg.norm(disList)) / (len(DimList))
    return np.mean(disList)


# 超边覆盖的节点距离
def CalHeatKernelW(group, dim, miu=1):
    sumdis = 0
    for id_i, i in enumerate(group):
        for id_j, j in enumerate(group):
            if id_i != id_j:
                dis = CalICDis(i, j, dim)
                sumdis += math.exp(-dis * dis / miu * 0.1)

    n = len(group)
    w = sumdis / (n * (n - 1))

    return w

# </editor-folder>

# <editor-folder desc = "Method">

def ParTrans(samples,clientPar):

    par = {}
    Config.TAG_RECORD_MERGE_PROCESS = True  # 若要记录每次迭代的情况，需要将此参数设置为True

    if clientPar:

        if ("random_merge_number" in clientPar) & ("rule_covered_positive_sample_rate_threshold" in clientPar):
            method_result = Run_MDL_RM.get_intention_by_MDL_RM_r \
                (samples, clientPar["random_merge_number"],
                 clientPar["rule_covered_positive_sample_rate_threshold"])  # 调用意图识别方法
            par["mergeNum"] = clientPar["random_merge_number"]
            par["filtrationCoefficient"] = clientPar["rule_covered_positive_sample_rate_threshold"]
        elif "random_merge_number" in clientPar:
            method_result = Run_MDL_RM.get_intention_by_MDL_RM_r \
                (samples, clientPar["random_merge_number"],
                 0.3)  # 调用意图识别方法
            par["mergeNum"] = clientPar["random_merge_number"]
            par["filtrationCoefficient"] = 0.3

        elif "rule_covered_positive_sample_rate_threshold" in clientPar:
            method_result = Run_MDL_RM.get_intention_by_MDL_RM_r \
                (samples, 50,
                 clientPar["rule_covered_positive_sample_rate_threshold"])  # 调用意图识别方法
            par["mergeNum"] = 50
            par["filtrationCoefficient"] = clientPar["rule_covered_positive_sample_rate_threshold"]

    # 默认参数
    else:
        method_result = Run_MDL_RM.get_intention_by_MDL_RM_r \
            (samples, 50, 0.3)  # 调用意图识别方法

        par["mergeNum"] = 50
        par["filtrationCoefficient"] = 0.3

    intention, min_encoding_length, init_min_encoding_length, method_log = method_result  # 方法运行结果包含四项内容

    best_PintV = []
    best_PredInts = []
    for key,values in intention.items():
        best_PredInts.append({})
        # 没有置信度
        best_PredInts[-1]['Conf'] = -1

        if values[0]['MapContent'] == 'Thing':
            best_PredInts[-1]['C'] = []
        else: best_PredInts[-1]['C'] = [values[0]['MapContent']]

        if  values[0]['MapMethod'] == 'MapMethodRoot':
            best_PredInts[-1]['M'] = []
        else:best_PredInts[-1]['M'] = [values[0]['MapMethod']]

        if  values[0]['Theme'] == 'ThemeRoot':
            best_PredInts[-1]['T'] = []
        else:best_PredInts[-1]['T'] = [values[0]['Theme']]

        best_PredInts[-1]['S'] = [values[0]['Spatial']]

        best_PintV += list(values[1])

    best_PintV = list(set(best_PintV))


    par["encodingLength"] = []
    for process in method_log['merge_process']:
        for one in process:
            par["encodingLength"].append(one['total_encoding_length'])

    par["initMinEncodingLength"] = init_min_encoding_length
    par["minEncodingLength"] = min_encoding_length


    return best_PredInts,best_PintV,par

def recognizeIntention(FeedbackSamples):
    clientPar = {
        "random_merge_number": 50,
        "rule_covered_positive_sample_rate_threshold": 0.3
    }

    # 获取样本
    samples = {}
    samples["relevance"] = []
    samples["irrelevance"] = [{"Theme": [], "MapMethod": [], "MapContent": [], "Spatial": []}]

    for smp in FeedbackSamples:
        samples["relevance"].append({})
        samples["relevance"][-1]["Theme"] = smp["T"]
        samples["relevance"][-1]["MapMethod"] = smp["M"]
        samples["relevance"][-1]["MapContent"] = smp["C"]
        samples["relevance"][-1]["Spatial"] = smp["S"]

    try:
        best_PredInts, best_PintV, par = ParTrans(samples, clientPar)
    except Exception as e:
        best_PredInts = [{"C":[],"M":[],"T":[],"S":[]}]
        best_PintV = []
        par = {}
        print(e)

    return best_PredInts,best_PintV,par

# </editor-folder>

# <editor-folder desc="Measure">

def CalSU(cluster):
    # 计算意图的生成的取值集合
    # 每个意图概念的下位概念数量
    L_I_Hyponyms_num = 1
    L_S_Centainty = 1
    for e in cluster['E']:
        I_num = 1+ len(Tree[e['dim']]['all_hyponyms'][e['value']])
        I_c = [e['value']] + Tree[e['dim']]['all_hyponyms'][e['value']]
        # 不确定标签的组合
        L_I_Hyponyms_num = L_I_Hyponyms_num *I_num

        Centain_c = []
        for sample in cluster['V']:
            # 确定性标签
            for v in sample[e['dim']]:
                # 下位概念找交集
                if v in I_c:
                    Centain_c += [v] + Tree[e['dim']]['all_hyponyms'][v]
                    Centain_c = list(set(Centain_c))
                elif v in Tree[e['dim']]['all_ancestors'][e['value']]:
                    Centain_c = I_c
                    break

        intersection = list(set(I_c).intersection(Centain_c))
        L_Centainty_num = len(intersection)
        L_S_Centainty = L_S_Centainty * L_Centainty_num

    SU = (L_I_Hyponyms_num - L_S_Centainty) / L_I_Hyponyms_num

    return SU

# 意图应该覆盖的样本集合
def CalJaccard(TintV, PintV):
    jaccard = 0
    if TintV or PintV:
        # 任意一个集合不为空，union不为0
        intersection = len(list(set(TintV).intersection(set(PintV))))
        union = len(list(set(TintV).union(set(PintV))))
        jaccard = float(intersection) / float(union)
    else:
        # 同为空集
        jaccard = 1

    return jaccard

def CalRec_Pre(TrueInts, PredInts, mu):
    # 空意图，补齐长度
    a = len(TrueInts)
    b = len(PredInts)

    simMatrix = np.full((a, b), float('inf'))
    for i, tint in enumerate(TrueInts):
        for j, pint in enumerate(PredInts):
            dis = CalRelI_PreIDis(tint, pint)
            simMatrix[i][j] = math.exp(-dis * dis / mu)

    # 在底部添加全为0的行，使得输入矩阵变为方阵
    if b > a:
        simMatrix = np.vstack((simMatrix, np.zeros(((b - a), simMatrix.shape[1]))))
    elif b < a:
        simMatrix = np.hstack((simMatrix, np.zeros((simMatrix.shape[0], (a - b)))))

    # 求解最大权匹配问题
    row_ind, col_ind = linear_sum_assignment(simMatrix, maximize=True)

    TP = simMatrix[row_ind, col_ind].sum()

    if len(PredInts) == 0:
        P_TP_FP = 1
    else:
        P_TP_FP = len(PredInts)

    if len(TrueInts) == 0:
        T_TP_FP = 1
    else:
        T_TP_FP = len(TrueInts)

    # 正确意图占预测意图的比例
    Precision = TP / P_TP_FP
    # 正确意图占真实意图的比例
    Recall = TP / T_TP_FP

    return Precision, Recall

# </editor-folder>

def MDL_RM_RunScene(SamplesPath,mu = 10):

    # <editor-folder desc="load data">
    OriSamples = load_json(SamplesPath)

    TrueInts = OriSamples['intentions']
    Samples = OriSamples["positive_samples"] + OriSamples["negative_samples"]

    TintVid = list(range(len(OriSamples["positive_samples"])))
    TnoiseVid = list(range(len(OriSamples["positive_samples"]),len(Samples)))

    # </editor-folder>

    # <editor-folder desc="Methods">

    best_PredInts, best_PintV, par = recognizeIntention(Samples)

    #</editor-folder>

    # <editor-folder desc = "Measure">

    jac = CalJaccard(TintVid, best_PintV)
    Precision, Recall = CalRec_Pre(TrueInts, best_PredInts,mu)
    LIP = 1-Precision
    MLMSSM = Recall

    # </editor-folder>

    # writer_process.write_rows([["fin_id:", fin_id]])

    return jac,LIP,MLMSSM

class CSVWriter:
    __instance = None  # 保存单例实例的变量
    __file = None      # 文件对象
    __writer = None    # CSV writer 对象

    def __new__(cls, file_path):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def open_file(self, file_path):
        self.close_file()  # 关闭已打开的文件（如果有）
        self.__file = open(file_path, 'a', newline='')
        self.__writer = csv.writer(self.__file)

    def write_rows(self, rows_data):
        if self.__writer is None:
            raise ValueError("文件未打开，请先调用 open_file 方法打开文件。")
        self.__writer.writerows(rows_data)

    def close_file(self):
        if self.__file is not None:
            self.__file.close()
            self.__file = None
            self.__writer = None

# # 使用：
# writer_process = CSVWriter('results\process.csv')  # 获取全局的 CSV 文件写入指针实例
#
# # 打开文件并写入数据行
# writer_process.open_file('results\process.csv')

if __name__ == '__main__':

    # 参数
    P = [0, 0.2, 0.4, 0.6, 0.8]  # 噪声率
    Num = [20]  # 样本数量 10, 20, 40, 60, 80, 100
    repeat_index = 2 # 第一次抽样的数据

    # 由语义距离换算为相似度的系数，越大，则削弱权重差异的能力越强v
    mu = 10

    RootPath = 'Samples\\SampleSelector'
    Scene_List = ['S2','S3','S4','S5']

    for Scene in Scene_List:

        SceneFolder = os.path.join(RootPath, Scene)

        # 读取场景文件夹名称
        folders = os.listdir(SceneFolder)
        # 读取场景文件夹中的不同样本集合
        # 创建一个空的 DataFrame，指定列名
        Score_df = pd.DataFrame(
            columns=['Scene', 'Sindex', 'Num', 'Noise_Rate', 'Jaccard', 'LIP', 'MLMSSM'])
        df_id = 0
        for i in range(0, len(folders)):
            Scene_index_Folder = os.path.join(SceneFolder, folders[i])
            print("Folder:", Scene, folders[i])
            # 读取不同参数的样本
            for num in Num:
                for p in P:
                    SamplesPath = Scene_index_Folder + "\\num" + str(num) + "\\p" + str(p) + "_" + str(
                        repeat_index) + ".json"
                    jac, LIP, MLMSSM= MDL_RM_RunScene(SamplesPath,mu)
                    Score_df.loc[df_id] = [Scene, folders[i], num, p, jac, LIP, MLMSSM]
                    df_id = df_id + 1

        out_file_path = 'results\\20240403_MDL_RM_num20_Rp2.csv'
        # 打开 CSV 文件并创建 csv.writer 对象
        if not os.path.exists(os.path.dirname(out_file_path)):
            os.makedirs(os.path.dirname(out_file_path))

        Score_df.to_csv(out_file_path, mode='a', header=True, index=False)

    print("good, good, very good!")