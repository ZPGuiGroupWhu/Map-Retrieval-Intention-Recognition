"""
   脚本描述：读取实验用的的样本，并做转换
   具体而言：
      ①将维度S,C,M,T名称 转化成 Spatial,MapContent,MapMethod,Theme
      ②将维度取值None转化成本体的根节点
"""
# 试验用样本
from src.main.samples.input import DimensionValues
from src.main.util.FileUtil import load_json

Ontology_Root = {'Spatial': DimensionValues.SpatialValue.Ontology_Root,
                 'Theme': DimensionValues.ThemeValues.Ontology_Root,
                 'MapMethod': DimensionValues.MapMethodValues.Ontology_Root,
                 'MapContent': DimensionValues.MapContentValues.Ontology_Root}

Ontologies = {'Spatial': DimensionValues.SpatialValue.Ontologies,
              'Theme': DimensionValues.ThemeValues.Ontologies,
              'MapMethod': DimensionValues.MapMethodValues.Ontologies,
              'MapContent': DimensionValues.MapContentValues.Ontologies}
Ancestor = {'Spatial': DimensionValues.SpatialValue.Ancestors,
            'Theme': DimensionValues.ThemeValues.Ancestors,
            'MapMethod': DimensionValues.MapMethodValues.Ancestors,
            'MapContent': DimensionValues.MapContentValues.Ancestors}
Neighborhood = {'Spatial': DimensionValues.SpatialValue.Neighborhood,
                'Theme': DimensionValues.ThemeValues.Neighborhood,
                'MapMethod': DimensionValues.MapMethodValues.Neighborhood,
                'MapContent': DimensionValues.MapContentValues.Neighborhood}
direct_Ancestor = {'Spatial': DimensionValues.SpatialValue.Direct_Ancestors,
                   'Theme': DimensionValues.ThemeValues.Direct_Ancestors,
                   'MapMethod': DimensionValues.MapMethodValues.Direct_Ancestors,
                   'MapContent': DimensionValues.MapContentValues.Direct_Ancestors}
concept_information_content = {'Spatial': DimensionValues.SpatialValue.Information_Content,
                               'Theme': DimensionValues.ThemeValues.Information_Content,
                               'MapMethod': DimensionValues.MapMethodValues.Information_Content,
                               'MapContent': DimensionValues.MapContentValues.Information_Content}

Dimension_Name = ["Spatial", "MapContent", "MapMethod", "Theme"]
def transform_sample(relevance_feedback_samples):
    # transform S,C,M,T to Spatial,MapContent,MapMethod,Theme
    positive_samples = []
    negative_samples = []
    positive_sample_key = "positive_samples" if "positive_samples" in relevance_feedback_samples else "relevance"
    negative_sample_key = "negative_samples" if "negative_samples" in relevance_feedback_samples else "irrelevance"
    for tmp_sample in relevance_feedback_samples[positive_sample_key]:
        tmp_sample_copy = {
                           "ID": tmp_sample["ID"],
                           "Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        positive_samples.append(tmp_sample_copy)
    for tmp_sample in relevance_feedback_samples[negative_sample_key]:
        tmp_sample_copy = {
                           "ID": tmp_sample["ID"],
                           "Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        negative_samples.append(tmp_sample_copy)

    # change None to root concept,空间范围维度使用America表示无意图
    for tmp_sample in positive_samples:
        for tmp_dim in Dimension_Name:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]
    for tmp_sample in negative_samples:
        for tmp_dim in Dimension_Name:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]
    result = {"positive": positive_samples, "negative": negative_samples}
    return result


def load_sample_from_file(sample_path):
    tmp_samples = load_json(sample_path)
    tmp_intention = tmp_samples["intention"]
    tmp_intention_copy = []
    for sub_intention in tmp_intention:
        tmp_intention_copy.append({"Spatial": sub_intention["S"] if "S" in sub_intention else sub_intention["Spatial"],
                                   "MapContent": sub_intention["C"] if "C" in sub_intention else sub_intention[
                                       "MapContent"],
                                   "MapMethod": sub_intention["M"] if "M" in sub_intention else sub_intention[
                                       "MapMethod"],
                                   "Theme": sub_intention["T"] if "T" in sub_intention else sub_intention["Theme"]})
    real_intention = tmp_intention_copy

    # change None to root concept
    for sub_intention in real_intention:
        for tmp_dim in Dimension_Name:
            if sub_intention[tmp_dim] == "None":
                sub_intention[tmp_dim] = Ontology_Root[tmp_dim]
    docs = transform_sample(tmp_samples)
    return docs, real_intention


if __name__ == "__main__":
    test_sample_path = "../../../../resources/samples/scenes_v4_7/Scene1/final_samples.json"
    tmp_docs, tmp_real_intention = load_sample_from_file(test_sample_path)
    print(tmp_real_intention)
    print(tmp_docs)