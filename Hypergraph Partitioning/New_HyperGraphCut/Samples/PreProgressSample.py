from FileUtil import load_json, save_as_json
import pandas as pd

import difflib

#所有上位概念，不包括自己
TreeC = load_json("../SweetData/all_ancestors.json")
AllIntentC = load_json("../SweetData/all_concepts.json")
TreeS = load_json("../SweetData/Lall_ancestors.json")

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

# 样本预处理，去除父子关系标签，只保留子标签
def PreProcessingSmpC(sampleC):
    # 删除上位节点
    delcpt = []
    for indexi,i in enumerate(sampleC):
        sampleC[indexi] = i.rstrip("\r\n")
        for indexj,j in enumerate(sampleC):
            sampleC[indexj] =j.rstrip("\r\n")
            if i != j and i not in TreeC[j]:
                continue
            elif i != j:
                delcpt.append(i)
    delcpt = list(set(delcpt))

    if delcpt:
        for cpt in delcpt:
            sampleC.remove(cpt)
    delcpt.clear()

    return sampleC

def PreProcessingSmpS(sampleS):
    # 删除上位节点
    delcpt = []
    for i in sampleS:
        for j in sampleS:
            if i != j and i not in j and j not in i:
                continue
            elif i != j:
                # 空间范围维度，原始数据储存了路径
                if i in j:
                    delcpt.append(i)
                if j in i:
                    delcpt.append(j)
    delcpt = list(set(delcpt))

    if delcpt:
        for cpt in delcpt:
            sampleS.remove(cpt)
    delcpt.clear()

    #删除路径中的上位结点
    for i in range(len(sampleS)):
        index = sampleS[i].rfind("/")
        sampleS[i] = sampleS[i][index + 1:]


    return sampleS


if __name__=="__main__":

    workbook = pd.read_csv('Org_Samples.csv', encoding='utf-8')
    sample_num = workbook.shape[0]
    workbook = workbook.drop(["Discard"], axis = 1)
    workbook = workbook.drop(["State"], axis = 1)
    df = pd.DataFrame(columns=['Content', 'Space', 'Topic', 'Mapping'])

    for i in range(sample_num):
        Cs = str(workbook["IRIs"][i]).split(",")
        for index,content in enumerate(Cs):
            if content not in AllIntentC:
                if content == "http://sweetontology.net/stateRealm/UnderWater":
                    Cs[index] = "http://sweetontology.net/stateRealm/Underwater"
                if content == "http://sweetontology.net/matrWater/WasteWater":
                    Cs[index] = "http://sweetontology.net/matrWater/Wastewater"
                if content == "http://sweetontology.net/propCount/Population\n":
                    Cs[index] = "http://sweetontology.net/propCount/Population"
                if content == "http://sweetontology.net/propPressure/AtmosphericPressure\n":
                    Cs[index] = "http://sweetontology.net/propPressure/AtmosphericPressure"
                if content == "http://sweetontology.net/matrWater/RainWater":
                    Cs[index] = "http://sweetontology.net/matrWater/Rainwater"
                if content == "http://sweetontology.net/phenEnergy/SolarPowerRiver":
                    Cs[index] = "http://sweetontology.net/phenEnergy/SolarPower"
                if content == "http://sweetontology.net/matrWater/GroundWater":
                    Cs[index] = "http://sweetontology.net/matrWater/Groundwater"

        Cs = PreProcessingSmpC(Cs)

        Ss = str(workbook["Space"][i]).split(",")
        Ss = PreProcessingSmpS(Ss)

        if Cs:
            df.loc[i, "Content"] = ','.join(Cs)
        if Ss:
            df.loc[i, "Space"] = ','.join(Ss)

        df.loc[i,"Topic"] = workbook["Topic"][i]
        df.loc[i,"Mapping"] = workbook["Style"][i]

    # 验证所有概念都在SWEET中
    flag = 1
    for i in range(sample_num):
        Cs = str(df["Content"][i]).split(",")
        Ss = str(df["Space"][i]).split(",")
        for content in Cs:
            if content not in AllIntentC:
                print(content)
                flag = 0
        for space in Ss:
            s_detail = space.split("/")[-1]
            if s_detail not in TreeS.keys():
                print(s_detail)
                flag = 0

    if flag:
        df.to_csv("PreProgressedSamples.csv",index=False,header=True)


# 有问题的标签
# TrueTag = [
#     "http://sweetontology.net/stateRealm/Underwater",
#     "http://sweetontology.net/matrWater/Wastewater",
#     "http://sweetontology.net/propCount/Population",
#     "http://sweetontology.net/propPressure/AtmosphericPressure",
#     "http://sweetontology.net/matrWater/Rainwater",
#     "http://sweetontology.net/matrWater/Groundwater",
#     "http://sweetontology.net/phenEnergy/SolarPower"
# ]

# [
# 'http://sweetontology.net/stateRealm/UnderWater',
# 'http://sweetontology.net/matrWater/WasteWater',
# 'http://sweetontology.net/propCount/Population\n',
# 'http://sweetontology.net/propPressure/AtmosphericPressure\n',
# 'http://sweetontology.net/matrWater/RainWater',
# 'http://sweetontology.net/phenEnergy/SolarPowerRiver',
# 'http://sweetontology.net/matrWater/GroundWater'
# ]

