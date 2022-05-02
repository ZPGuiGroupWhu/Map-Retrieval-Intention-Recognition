from MDL_RM.src.main.intention_recognition import SemanticApriori
from MDL_RM.src.main.samples.input import Sample
from MDL_RM.src.main.util import FileUtil


def get_all_candidate_sub_intentions_test():
    scene = "223"
    sample_version = "scenes_v4_5"
    test_sample_path = "./../../../resources/samples/" + sample_version + "/Scene" + scene + "/final_samples.json"
    samples = FileUtil.load_json(test_sample_path)  # 加载样本文件
    samples = Sample.transform_sample(samples)  # 转换样本文件
    min_support = 0.3
    result = SemanticApriori.get_all_candidate_sub_intentions(samples, min_support)
    print(result)
    print(len(result))


if __name__ == "__main__":
    get_all_candidate_sub_intentions_test()
