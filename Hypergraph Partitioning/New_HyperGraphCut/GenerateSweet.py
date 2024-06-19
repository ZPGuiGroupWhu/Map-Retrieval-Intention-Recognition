import  os
import sys  # 导入sys模块

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from FileUtil import load_json, save_as_json


class SWEET(object):
    ####   类属性

    # 直接上位概念
    direct_ancestors = load_json("SweetData/direct_ancestors.json")
    # 直接下位概念
    direct_hyponyms = load_json("SweetData/direct_hyponyms.json")

    # 所有上位概念，不包括自己
    all_ancestors = load_json("SweetData/all_ancestors.json")
    # 所有下位概念，不包括自己
    all_hyponyms = load_json("SweetData/all_hyponyms.json")
    # 相等概念，不包括自己
    equivalent_relations = load_json("SweetData/equivalent_relations.json")

    # 所有概念
    all_concepts = load_json("SweetData/all_concepts.json")
    all_roots = load_json("SweetData/all_roots.json")
    all_leaves = load_json("SweetData/all_leaves.json")

    ###类方法

    # content:string,one_SWEET_content
    def __init__(self,content):
        self.content = content
        return

    # 上位+下位+相等+自己
    def relate_concept(self):
        temp = []
        temp.append(self.content)
        temp += self.all_ancestors[self.content]
        temp += self.all_hyponyms[self.content]
        temp += self.equivalent_relations[self.content]
        return temp


temp_series = []
temp_leaf = []
temp_root = []

# 对称相等关系
def equivalent(Path):
    data = load_json(Path)

    for key, value in data.items():
        if value:
            for content in value:
                if content in data.keys():
                    if key not in data[content]:
                        data[content].append(key)

    save_as_json(data, Path)

    return

# 递归下位概念
def downdowm(content):
    global temp_series,temp_leaf
    if SWEET.direct_hyponyms[content]:
        for i in SWEET.direct_hyponyms[content]:
            if i not in temp_series:
                temp_series.append(i)
            downdowm(i)
    else:
        temp_leaf.append(content)
    return

# 递归上位概念
def upup(content):
    global temp_series,temp_root
    if SWEET.direct_ancestors[content]:
        for i in SWEET.direct_ancestors[content]:
            if i not in temp_series:
                temp_series.append(i)
            upup(i)
    else:
        temp_root.append(content)
    return


def leaf_root(leavesPath,rootsPath):

    all_roots = {}
    all_leaves = {}

    # global temp_root,temp_leaf,temp_series
    # temp_series.clear()
    # temp_leaf.clear()
    # temp_root.clear()
    #
    #
    # # 递归
    # for cont in SWEET.all_concepts:
    #     downdowm(cont)
    #     all_leaves[cont] = temp_leaf
    #     upup(cont)
    #     all_roots[cont] = temp_root
    #     temp_series.clear()
    #     temp_leaf.clear()
    #     temp_root.clear()
    #


    # 上位节点为[]，即为根节点
    root_content = []
    for key,value in SWEET.all_ancestors.items():
        if value:
            continue
        else:
            root_content.append(key)

    for cont in SWEET.all_concepts:
        all_roots[cont] = []
        for root in root_content:
            if cont in SWEET.all_hyponyms[root]:
                all_roots[cont].append(root)


    save_as_json(all_leaves, leavesPath)
    save_as_json(all_roots, rootsPath)

    return


if __name__ == '__main__':

    # leavesPath = "all_leaves.json"
    # rootsPath = "all_roots.json"
    #
    # leaf_root(leavesPath, rootsPath)

    print("sweet")





