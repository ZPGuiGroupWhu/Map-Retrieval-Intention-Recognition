import os
import sys  # 导入sys模块

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from FileUtil import load_json, save_as_json


# 空间维度的本体库
class GeoNames(object):
    ####   类属性

    # 直接上位概念
    direct_ancestors = load_json("SweetData/Ldirect_Ancestor.json")

    # 所有上位概念，不包括自己
    all_ancestors = load_json("SweetData/Lall_ancestors.json")
    # 所有下位概念，不包括自己
    all_hyponyms = load_json("SweetData/Lall_hyponyms.json")

    # 所有概念
    all_concepts = load_json("SweetData/Lall_concepts.json")
    all_roots = load_json("SweetData/Lall_roots.json")

    ###类方法

    # content:string,one_SWEET_content
    def __init__(self, content):
        self.content = content
        return

    # 上位+下位+相等+自己
    def relate_concept(self):
        temp = []
        temp.append(self.content)
        temp += self.all_ancestors[self.content]
        temp += self.all_hyponyms[self.content]
        return temp


if __name__ == '__main__':
    # leavesPath = "all_leaves.json"
    # rootsPath = "all_roots.json"
    #
    # leaf_root(leavesPath, rootsPath)

    print("sweet")

