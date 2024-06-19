import math
from functools import reduce
import numpy as np
from scipy.linalg import sqrtm
from sklearn import metrics
from sklearn.cluster import KMeans
from collections import Counter
from itertools import product
import random
from scipy.optimize import linear_sum_assignment
from scipy.linalg import fractional_matrix_power

import matplotlib.pyplot as plt
import pandas as pd

from FileUtil import load_json
from GenerateSweet import SWEET
from GenerateGeoNames import GeoNames


import os
import csv
import time

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

    # # 四维度二范数
    # return (np.linalg.norm(disList)) / (len(DimList))
    # 均值
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

# <editor-folder desc = "HyperGraph">

def generate_freq_supports(data_set, item_set, min_support):
    freq_set = set()  # 保存频繁项集元素
    item_count = {}  # 保存元素频次，用于计算支持度
    supports = {}  # 保存支持度

    # 如果项集中元素在数据集中则计数
    for record in data_set:
        for item in item_set:
            if item.issubset(record):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1

    data_len = float(len(data_set))

    # 计算项集支持度
    for item in item_count:
        if (item_count[item] / data_len) >= min_support:
            freq_set.add(item)
            supports[item] = item_count[item] / data_len

    return freq_set, supports



# 简单版超边生成
def SampleEdgesSet(Samples, min_support=0.3):
    # AncesU = {}
    # for dim in DimList:
    #     AncesU[dim] = []
    #
    # for v in Samples:
    #     for dim in AncesU.keys():
    #         for val in v[dim]:
    #             AncesU[dim] += Tree[dim]["all_ancestors"][val] + [val]
    #
    #         AncesU[dim] = list(set(AncesU[dim]))
    #
    # AncesN = {}  # 上位节点覆盖的节点数统计
    # for key, values in AncesU.items():
    #     AncesN[key] = {}
    #     for val in values:
    #         hyps = Tree[key]["all_hyponyms"][val] + [val]
    #         AncesN[key][val] = 0
    #         for v in Samples:
    #             for cspt in v[key]:
    #                 if cspt in hyps:
    #                     AncesN[key][val] += 1
    #                     break
    #
    #                     # AncesN[key] = dict(Counter(AncesU[key]))
    #
    # AncesNn = {}  # 删除为1的节点，至少2点形成一条超边
    # for dim, ancesdic in AncesN.items():
    #     AncesNn[dim] = {key: val for key, val in AncesN[dim].items() if val != 1}
    #
    # # 去掉覆盖重复的上位节点
    # edges_dict = {}
    # for dim, dimedict in AncesNn.items():
    #     edges_dict[dim] = []
    #     # 覆盖重复的节点
    #     Num_dict = {}
    #     for edge, num in dimedict.items():
    #         try:
    #             Num_dict[num]
    #         except:
    #             Num_dict[num] = [edge]
    #             continue
    #         Num_dict[num].append(edge)
    #
    #     # 相同覆盖的超边，保留下位节点
    #     for num, edges in Num_dict.items():
    #         delcpt = []
    #         if len(edges) > 1:
    #             for i in edges:
    #                 for j in edges:
    #                     if i != j and i in Tree[dim]["all_ancestors"][j]:
    #                         delcpt.append(i)
    #
    #             delcpt = list(set(delcpt))
    #
    #         if delcpt:
    #             for cpt in delcpt:
    #                 edges.remove(cpt)
    #         delcpt.clear()
    #
    #         edges_dict[dim] += edges

    # 扩展LCA上位概念的属性集合
    data = []
    for v in Samples:
        data.append([])
        for dim, values in v.items():
            if dim != 'id':
                data[-1] += values
                # 不使用语义扩展
                # for content in values:
                #     contentAnces = Tree[dim]["all_ancestors"][content]
                #     data[-1] += list(set(contentAnces) & set(edges_dict[dim]))
        data[-1] = list(set(data[-1]))

    # 候选项1项集
    c1 = set()
    for items in data:
        for item in items:
            item_set = frozenset([item])
            c1.add(item_set)

    # 频繁项1项集及其支持度
    l1, support1 = generate_freq_supports(data, c1, min_support)

    # groups是一组频繁项集
    # onegroup内部是相互有关系的combinations
    groups = []

    # 相互有关联的在一个onegroup
    for id, onecombinations in enumerate(l1):
        # 组合中不可有任意一个不可合并
        combinations_list = list(onecombinations)
        new = []

        for content in combinations_list:
            for dim in DimList:
                if content in AllIntent[dim]:
                    new += [content] + Tree[dim]["all_roots"][content]
                    break

        msg = 1
        for onegroup in groups:
            for groupcombinations in onegroup:
                old = []
                for content in groupcombinations:
                    for dim in DimList:
                        if content in AllIntent[dim]:
                            old += [content] + Tree[dim]["all_roots"][content]
                            break

                if set(new) & set(old):
                    onegroup.append(combinations_list)
                    msg = 0
                    break
            else:
                continue
            break

        if msg:
            groups.append([])
            groups[-1].append(combinations_list)

    # onegroup内部处理，不相关联的并列项
    edges = []
    for onegroup in groups:
        tempedges = []
        flaglist = [0] * len(onegroup)
        for i in range(len(onegroup)):  # 以i为每个合并的开头
            tid = len(tempedges)
            tempedges.append(onegroup[i].copy())
            eid = tid + 1

            msgdlt = 1  # 不能与其它下位项集合并，且已经与其余项集合并过，则删除
            for j in range(i + 1, len(onegroup)):  # 遍历j能否放入合并
                conceptslist = []
                for content in onegroup[j]:
                    # conceptslist += RelateConcept(content)
                    conceptslist += [content]

                msgnew = 1  # 不能与当前i开头已合并过其它项集的组合继续合并，但能与i合并
                for jj in range(tid, eid):  # 放入以i开头向下的新list
                    msg1 = 1
                    for content in tempedges[jj]:
                        if content in conceptslist:
                            msg1 = 0
                            break
                    if msg1:
                        tempedges[jj] += onegroup[j].copy()
                        tempedges[jj] = list(set(tempedges[jj]))
                        flaglist[j] = 1
                        msgdlt = 0
                        msgnew = 0

                if msgnew:
                    msg2 = 1  # 与i是否可合并为新的
                    for content in onegroup[i]:
                        if content in conceptslist:
                            msg2 = 0
                            break
                    if msg2:
                        tempedges.append(onegroup[i].copy())
                        tempedges[-1] += onegroup[j].copy()
                        tempedges[-1] = list(set(tempedges[-1]))
                        eid += 1
                        flaglist[j] = 1
                        msgdlt = 0
            # 未与其它后面的概念合并过
            if msgdlt == 1 and flaglist[i] == 1:
                tempedges.pop()
            else:
                flaglist[i] = 1

        # 去掉合并后，重复的项
        tempedges2 = []
        for x in tempedges:
            msg = 1
            x.sort()
            for y in tempedges:
                y.sort()
                if set(y) > set(x):  # y包含x
                    msg = 0
                    break
            if msg and x not in tempedges2:
                tempedges2.append(x)

        edges.append(tempedges2)

    # 所有onegroup之间笛卡尔积
    def combi(seq, depth):
        if not seq:
            yield []
        else:
            for element in seq[0]:
                for rest in combi(seq[1:], depth + 1):
                    yield [element] + rest

    # 去重,去列表
    temp_combi = list(combi(edges, 0))
    combi = []

    for i in temp_combi:
        temp = []
        for contentgroup in i:
            temp += contentgroup

        temp = list(set(temp))
        if temp not in combi:
            combi.append(temp)

    # 形式化表达
    cmtcombi = []
    for i in combi:
        k = {}
        for dim in DimList:
            k[dim] = []

        for content in i:
            for dim in DimList:
                if content in AllIntent[dim]:
                    if content not in k[dim]:
                        k[dim].append(content)
        if k not in cmtcombi:
            cmtcombi.append(k)

    return [cmtcombi]


# 图切代价函数
def CalargminC(hg, y_pred):
    k = len(set(y_pred))
    argminC = 0
    volS = [0] * k
    # 簇体积
    for vIndex, v in enumerate(hg.V):
        volS[y_pred[vIndex]] += hg.Dv[vIndex][vIndex]

    # 边界体积
    edge_cluster = np.zeros((len(hg.E), k))
    for eIndex, e in enumerate(hg.E):
        for vIndex, v in enumerate(hg.V):
            if hg.H[vIndex][eIndex] != 0:
                edge_cluster[eIndex][y_pred[vIndex]] += 1

    voleS = [0] * k
    for i in range(k):
        for eid in range(edge_cluster.shape[0]):
            one_edge_cut = 0
            for cid in range(edge_cluster.shape[1]):
                if i != cid:
                    one_edge_cut += edge_cluster[eid][i] * edge_cluster[eid][cid]
            one_edge_cut = hg.W[eid][eid] * one_edge_cut / hg.De[eid][eid]
            voleS[i] += one_edge_cut

    n = k  # 真实的类簇数目
    for i in range(k):
        if volS[i] == 0 and voleS[i] == 0:
            n = n - 1
        elif volS[i] != 0:
            argminC += voleS[i] / volS[i]

    if k > 1:
        argminC = argminC / (k - 1)
    else:
        argminC = 0

    return argminC


class HyperGraph(object):
    def __init__(self, Samples, wIndex=1):

        # 超图的权重系数
        self.wIndex = wIndex

        # 一个样本=一个顶点，顶点 v = [ M 制图方法，T 地理主题，C 地图内容 ]
        self.V = self.InitVertexs(Samples)
        # 初始化超边 e = [ 属性（M、T、C），值value，权重 w ]
        self.E = self.InitEdges()
        # 点边关联矩阵 H = long(V) × long(E)
        self.H, self.De, self.Dv, self.W = self.CreateH()

        return

    def InitVertexs(self, Samples):
        try:
            if Samples[0]['id']:
                return Samples
        except:
            for smpIdex, smpValue in enumerate(Samples):
                Samples[smpIdex]['id'] = smpIdex
        return Samples

    # 相同覆盖的超边，保留的子超边
    def InitEdges(self):

        E = []

        for dim in DimList:
            E.append({'dim': dim, 'value': '', 'w': 0})

        return E

    def CreateH(self, reset_flag=0):

        H = np.zeros([len(self.V), len(self.E)])  # v * e
        W = np.zeros([len(self.E), len(self.E)])  # e * e
        De = np.zeros([len(self.E), len(self.E)])  # e * e
        for eIndex, e in enumerate(self.E):
            groupdis = []
            temp = []
            if e['value'] in AllIntent[e['dim']]:
                # 超边为Global，不能覆盖下位节点
                if e['dim'] == "S" and e['value'] == "Global":
                    temp = [e['value']]
                else:
                    # 不使用本体进行语义关系的判断
                    temp = [e['value']]
                           # + Tree[e['dim']]["all_hyponyms"][e['value']] \
                           # + Tree[e['dim']]["all_ancestors"][e['value']]
            for vIndex, v in enumerate(self.V):
                dis = []
                disIndex = {}
                for id, value in enumerate(v[e['dim']]):
                    if value in temp:
                        s = CalICDis(value, e['value'], e['dim'])
                        dis.append(s)
                        disIndex[s] = id
                if dis:
                    # 最短距离作为权重
                    mindis = min(dis)
                    # H[vIndex][eIndex] = math.exp(-mindis*mindis / self.wIndex)
                    H[vIndex][eIndex] = 1
                    De[eIndex][eIndex] += 1
                    groupdis.append(mindis)
                    # id = disIndex[mindis]
                    # group.append(v[e['dim']][id])

            # 根据接触度计算权重
            if groupdis:

                ContactDgr = 0
                for i in groupdis:
                    ContactDgr += math.exp(-i * i / self.wIndex)
                w = ContactDgr / De[eIndex][eIndex]

                # w = CalHeatKernelW(group,e['dim'],self.wIndex)
            else:
                w = 0

            self.E[eIndex]['w'] = w
            W[eIndex][eIndex] = w
            # group.clear()

        Dv = np.zeros([len(self.V), len(self.V)])  # v * v
        del_v_index = []
        for i in range(len(self.V)):
            x = 0
            for j in range(len(self.E)):
                if H[i][j] != 0:
                    x += W[j][j]
            # 节点不在任何超边上，修改H、Dv、V
            if x == 0 and reset_flag == 1: del_v_index.append(i)

            Dv[i][i] = x

        # 未被超边覆盖的节点，不在超图上
        if del_v_index and reset_flag == 1:
            self.V = [i for j, i in enumerate(self.V) if j not in del_v_index]
            Dv = np.delete(Dv, del_v_index, axis=0)
            Dv = np.delete(Dv, del_v_index, axis=1)
            H = np.delete(H, del_v_index, axis=0)

        # 删除没有覆盖任何节点的超边
        # 找到全为 0 的列的索引
        del_e = np.all(De == 0, axis=0)
        if any(del_e) and reset_flag == 1:
            del_e_index = []
            for id,bool in enumerate(del_e):
                if bool:
                    del_e_index.append(id)

            self.E = [i for j, i in enumerate(self.E) if j not in del_e_index]
            De = np.delete(De, del_e_index, axis=0)
            De = np.delete(De, del_e_index, axis=1)

            H = np.delete(H, del_e_index, axis=1)

            W = np.delete(W, del_e_index, axis=0)
            W = np.delete(W, del_e_index, axis=1)

        return H, De, Dv, W

    def ResetEdges(self, edges):

        self.E = edges
        self.ResetAttr()

        return

    def ResetAttr(self):
        self.H, self.De, self.Dv, self.W = self.CreateH(reset_flag=1)
        return


# </editor-folder>

# <editor-folder, desc="Clustering"
# matrix：拉普拉斯矩阵，k：前k小特征向量
def topvec(matrix, k):  # 输出非0前k小的特征值对应的特征向量
    e_vals, e_vecs = np.linalg.eig(matrix)  # 拉普拉斯矩阵分解
    sorted_indices = np.argsort(e_vals)
    # print("sorted:",sorted_indices)
    # print("sorted:",sorted_indices[:k])
    return e_vals[sorted_indices[1:k]], e_vecs[:, sorted_indices[1:k]]


# Path：样本路径，SC：Spectral Clustering，k1：画图
def MyKmeans(hg, gamma=1):
    # , k1 = 2, flagD = False, flagP = ''

    kmax = min(5, len(hg.V))
    ok_group = {}

    if len(hg.V) > 2:
        I = np.eye(len(hg.V))
        # 节点的度不能为0！！！！构图前要检查
        L = I - fractional_matrix_power(hg.Dv, -0.5) @ hg.H @ hg.W @ np.linalg.inv(
            hg.De) @ hg.H.T @ fractional_matrix_power(hg.Dv, -0.5)
    else:
        # 无意图,样本不够
        NoInt = [{}]
        for dim in DimList:
            NoInt[0][dim] = []
        ok_group[0] = \
            {'avgconf': 0,
             'argminC': 2,
             'PredInts': NoInt,
             'PintV': []}
        return ok_group

    CC_list = []
    ks = []
    for index, k in enumerate(range(2, kmax)):
        # 找到前k小的特征向量
        # print("laplacian matrix n-min eigenvector, k1 = ",k1)
        val, vec = topvec(L, k)
        vec_real = vec.real  # 返回的数值为负数，其虚部都为0，直接取实数部分
        y_pred = KMeans(n_clusters=k).fit_predict(vec_real)

        argminC = CalargminC(hg, y_pred)
        score = argminC

        # 可分割，找最佳分割
        if score < gamma:
            argminC = 0-argminC
            CC_list.append(score)
            ks.append(k)
        # 不可分割，单意图场景,结束循环
        elif k == 2 and argminC >= gamma:
            argminC = 0-argminC
            one_k = 1
            one_y_pred = [0] * len(hg.V)

            clusters, PredInts, PintV, PnoiseV = GetIntention(hg, one_k, one_y_pred,argminC)

            # 若CC系数相同，保留平均置信度最高的意图
            avgconf = 0
            if len(PredInts) > 0:
                for intention in PredInts:
                    avgconf += intention['Conf']
                avgconf = avgconf / len(PredInts)

            ok_group[one_k] = {'avgconf': avgconf, 'argminC': argminC, 'PredInts': PredInts, 'PintV': PintV}
            return ok_group

    # 找到曲线最弯曲（拐点）处的簇数，作为最优的k值
    # 计算斜率和二阶差分
    optimal_k = 2
    if len(CC_list) > 2:
        dydx = np.diff(CC_list) / np.diff(ks)
        # d2ydx2 = np.diff(dydx) / np.diff(ks[:-1])
        # 查找变化率的峰值
        max_diff_idx = np.argmax(dydx)
        optimal_k = ks[max_diff_idx]

    elif len(CC_list) == 1:
        optimal_k = ks[0]
    elif len(CC_list) == 2:
        dydx = np.diff(CC_list) / np.diff(ks)
        # 斜率，变化小取k的大值，变化大取k的小值
        if dydx[0] < 0.3:
            optimal_k = ks[1]
        else:
            optimal_k = ks[0]

    # 再次执行k-means算法得到最终的聚类结果
    val, vec = topvec(L, optimal_k)
    vec_real = vec.real  # 返回的数值为负数，其虚部都为0，直接取实数部分
    y_pred = KMeans(n_clusters=optimal_k).fit_predict(vec_real)
    argminC = CalargminC(hg, y_pred)

    clusters, PredInts, PintV, PnoiseV = GetIntention(hg, optimal_k, y_pred,argminC)

    # 若CC系数相同，保留平均置信度最高的意图
    avgconf = 0
    for intention in PredInts:
        avgconf += intention['Conf']
    avgconf = avgconf / len(PredInts)

    ok_group[optimal_k] = {'avgconf': avgconf, 'argminC': argminC, 'PredInts': PredInts, 'PintV': PintV}

    return ok_group

# </editor-folder>

# <editor-folder desc="GetIntention,Measure">
# hg：超图,k: 聚类数目（子意图个数），y_pred：聚类结果label，minCovR：最小覆盖度，意图提取
def GetIntention(hg, k, y_pred,argminC, intention_edge_support=0.7):
    # 类簇
    clusters = []
    # 意图
    PredInts = []
    # 正样本集合
    PintV = []
    # 噪声点集合
    PnoiseV = []

    # 初始化聚类簇
    for i in range(k):
        clusters.append({})
        clusters[i]['V'] = []
        clusters[i]['S'] = []
        clusters[i]['Conf'] = 0

    # 聚类簇顶点分组
    for vIndex, v in enumerate(hg.V):
        if y_pred[vIndex] >= 0:
            clusters[y_pred[vIndex]]['S'].append(vIndex)  # 簇中的顶点id
            clusters[y_pred[vIndex]]['V'].append(v)


    for cluster in clusters:
        # TopInt初始化
        cluster['TopInt'] = {}  # 覆盖样本的最佳组合
        for dim in DimList:
            cluster['TopInt'][dim] = []
        cluster['E'] = []  # 聚类簇中的超边
        cluster['Eid'] = []  # 聚类簇中的超边id

        # 类簇不足两个点
        if len(cluster['S']) < 2:
            continue

        Volume = 0
        for i in range(len(hg.E)):
            x = 0
            for v in cluster['S']:
                if hg.H[v][i] != 0:
                    x += 1

            support = float(x) / float(len(cluster['S']))
            # 大于最小支持度的超边即为意图超边，保留
            if support > intention_edge_support:
                cluster['E'].append(hg.E[i])
                cluster['Eid'].append(i)
                Volume += hg.E[i]['w'] * x

        # 判断为噪声簇还是意图簇
        # 意图覆盖的样本id
        new_arr = hg.H[:, cluster['Eid']]  # 取意图超边所在的列
        # 提取出每行全为 1 的行的 ID，被超边覆盖的样本id（正样本）
        pos_id = list(np.where((new_arr != 0).all(axis=1))[0])

        if len(pos_id) >= 2:

            PintV += pos_id
            maxVolume = len(cluster['E']) * len(cluster['S'])

            # 簇置信度（超图的最大体积/实际体积）
            if maxVolume != 0:
                SS = Volume / maxVolume
                SU = CalSU(cluster)
                if argminC < 0:  # 单意图场景
                    cluster['Conf'] = 0.5 * (1 - SU + SS) * (argminC + gamma) / (gamma - 1)
                else:
                    cluster['Conf'] = 0.5 * (1 - SU + SS) * (gamma - argminC) / gamma

            # TopInt初始化
            for eIdx, e in enumerate(cluster['E']):
                cluster['TopInt'][e['dim']].append(e['value'])

            if cluster['TopInt'] not in PredInts:
                # 意图覆盖的概念
                cluster['TopInt']['Conf'] = cluster['Conf']
                PredInts.append(cluster['TopInt'])

        # # 提取出每行有 0 的行的 ID, 不被所有超边覆盖的样本id（负样本）
        # PnoiseV += list(np.where((new_arr == 0).any(axis=1))[0])

    PintV = list(set(PintV))
    V_id = np.arange(hg.H.shape[0])
    PnoiseV = [x for x in V_id if x not in PintV]

    return clusters, PredInts, PintV, PnoiseV

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

def RunScene(SamplesPath,min_support = 0.3,mu = 10,gamma = 0.8):

    # <editor-folder desc="load data">
    OriSamples = load_json(SamplesPath)

    TrueInts = OriSamples['intentions']
    Samples = OriSamples["positive_samples"] + OriSamples["negative_samples"]

    TintVid = list(range(len(OriSamples["positive_samples"])))
    TnoiseVid = list(range(len(OriSamples["positive_samples"]),len(Samples)))

    # </editor-folder>

    # <editor-folder desc="Methods">

    start_time = time.time()
    # edges_set = EdgesSet(Samples=Samples,min_support=min_support)
    # 简单版，基于频繁一项集生成
    edges_set = SampleEdgesSet(Samples=Samples,min_support=min_support)

    l = len(edges_set)
    end_time = time.time()
    # Edges_time = end_time - start_time
    # print("Edges_time ",Edges_time)

    # 实例化超图
    hg = HyperGraph(Samples=Samples,wIndex = mu)
    # 每种构图的意图提取结果
    ok_group = {}

    Graph_time = 0
    Cluster_time = 0
    for i in range(l-1,-1,-1):
        for j,oneset in enumerate(edges_set[i]):
            # 一组超边
            edges = []
            for dim,values in oneset.items():
                for v in values:
                    edges.append({'dim': dim, 'value': v, 'w': 0})

            start_time = time.time()
            hg = HyperGraph(Samples=Samples, wIndex=mu)
            hg.ResetEdges(edges)
            end_time = time.time()
            Graph_time += end_time - start_time

            start_time = time.time()
            ok_group[(i, j)] = MyKmeans(hg, gamma=gamma)
            end_time = time.time()
            Cluster_time += end_time - start_time

    # print("Graph_time ", Graph_time)
    # print("Cluster_time ", Cluster_time)

    #</editor-folder>

    # <editor-folder desc = "Intention,Measure">

    # CC最小的为最终意图
    best_argminC = 1.1 # 为1的纯纯单意图
    best_avgconf = 0
    best_PintV = []
    best_PredInts = []

    # 统计argminC值小于0.5和大于0.5的元素个数
    # 初始化小于0.5和大于0.5的计数器
    count_less_than_gamma = 0
    count_greater_than_gamma = 0

    # 遍历字典中的值
    for value in ok_group.values():
        for inner_value in value.values():
            argminC_value = inner_value['argminC']
            if argminC_value >= 0:
                count_less_than_gamma += 1
            elif argminC_value < 0:
                count_greater_than_gamma += 1

    for edges_id, k_groups in ok_group.items():
        for k, values in k_groups.items():
            # 多 意图场景
            if count_less_than_gamma >= count_greater_than_gamma:
                if values['argminC'] < best_argminC and values['argminC'] > 0:
                    best_argminC = values['argminC']
                    best_avgconf = values['avgconf']
                    best_PintV = values['PintV']
                    best_PredInts = values['PredInts']
            else:  # 单 意图场景
                if values['argminC'] < best_argminC and values['argminC'] < 0:
                    best_argminC = values['argminC']
                    best_avgconf = values['avgconf']
                    best_PintV = values['PintV']
                    best_PredInts = values['PredInts']

    jac = CalJaccard(TintVid, best_PintV)
    Precision, Recall = CalRec_Pre(TrueInts, best_PredInts, mu)
    LIP = 1-Precision
    MLMSSM = Recall

    # </editor-folder>

    return jac,LIP,MLMSSM,best_argminC,best_avgconf

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
    gamma = 0.7  # 大于该值为单意图场景，不判断是否可切割
    min_support = 0.3
    # GetIntention处是0.7

    RootPath = 'Samples\\SampleSelector'
    Scene_List = ['S2','S3','S4','S5']

    for Scene in Scene_List:

        SceneFolder = os.path.join(RootPath, Scene)

        # 读取场景文件夹名称
        folders = os.listdir(SceneFolder)
        # 读取场景文件夹中的不同样本集合
        # 创建一个空的 DataFrame，指定列名
        Score_df = pd.DataFrame(
            columns=['Scene', 'Sindex', 'Num', 'Noise_Rate', 'Jaccard', 'LIP', 'MLMSSM', 'best_argminC',
                     'best_avgConf'])
        df_id = 0
        for i in range(0, len(folders)):
            Scene_index_Folder = os.path.join(SceneFolder, folders[i])
            print("Folder:", Scene, folders[i])
            # 读取不同参数的样本
            for num in Num:
                for p in P:
                    SamplesPath = Scene_index_Folder + "\\num" + str(num) + "\\p" + str(p) + "_" + str(
                        repeat_index) + ".json"
                    jac, LIP,MLMSSM, best_argminC, best_avgconf = RunScene(SamplesPath, min_support, mu,
                                                                                  gamma)
                    Score_df.loc[df_id] = [Scene, folders[i], num, p, jac, LIP,MLMSSM, best_argminC,
                                           best_avgconf]
                    df_id = df_id + 1

        out_file_path = 'results\\20240403_NoOntology_gamma07_num20_Rp2.csv'
        # 打开 CSV 文件并创建 csv.writer 对象
        if not os.path.exists(os.path.dirname(out_file_path)):
            os.makedirs(os.path.dirname(out_file_path))

        Score_df.to_csv(out_file_path, mode='a', header=True, index=False)

    print("good, good, very good!")