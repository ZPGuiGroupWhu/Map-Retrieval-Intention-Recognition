# 试验用样本

from MDL_RM.src.main.util.FileUtil import load_json
from MDL_RM.src.main.util import RetrievalUtil
from MDL_RM.src.main.samples.input import DimensionValues

Ontologies = {'Spatial': DimensionValues.SpatialValue.Ontologies,
              'Theme': DimensionValues.ThemeValues.Ontologies,
              'MapMethod': DimensionValues.MapMethodValues.Ontologies,
              'MapContent': DimensionValues.MapContentValues.Ontologies_All_Dimensions}
Ancestor = {'Spatial': DimensionValues.SpatialValue.Ancestors,
            'Theme': DimensionValues.ThemeValues.Ancestors,
            'MapMethod': DimensionValues.MapMethodValues.Ancestors,
            'MapContent': DimensionValues.MapContentValues.Ancestor_All_Dimensions}
Neighborhood = {'Spatial': DimensionValues.SpatialValue.Neighborhood,
                'Theme': DimensionValues.ThemeValues.Neighborhood,
                'MapMethod': DimensionValues.MapMethodValues.Neighborhood,
                'MapContent': DimensionValues.MapContentValues.Neighborhood_All_Dimensions}
direct_Ancestor = {'Spatial': DimensionValues.SpatialValue.direct_Ancestor,
                   'Theme': DimensionValues.ThemeValues.direct_Ancestor,
                   'MapMethod': DimensionValues.MapMethodValues.direct_Ancestor,
                   'MapContent': DimensionValues.MapContentValues.direct_Ancestor_All_Dimensions}
concept_information_content = {'Spatial': DimensionValues.SpatialValue.Information_Content,
                               'Theme': DimensionValues.ThemeValues.Information_Content,
                               'MapMethod': DimensionValues.MapMethodValues.Information_Content,
                               'MapContent': DimensionValues.MapContentValues.Information_Content}
Ontology_Root = {'Spatial': DimensionValues.SpatialValue.Ontology_Root,
                 'Theme': DimensionValues.ThemeValues.Ontology_Root,
                 'MapMethod': DimensionValues.MapMethodValues.Ontology_Root,
                 'MapContent': DimensionValues.MapContentValues.Ontology_Root}
docs = None
real_intention_key = None
real_intention = None


def load_sample(sample_path):
    global docs, real_intention_key, real_intention
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

    # transform S,C,M,T to Spatial,MapContent,MapMethod,Theme
    positive_samples = []
    negative_samples = []
    for tmp_sample in tmp_samples["positive_samples"]:
        tmp_sample_copy = {"Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        positive_samples.append(tmp_sample_copy)
    for tmp_sample in tmp_samples["negative_samples"]:
        tmp_sample_copy = {"Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        negative_samples.append(tmp_sample_copy)

    # change None to root concept
    for sub_intention in real_intention:
        for tmp_dim in sub_intention:
            if sub_intention[tmp_dim] == "None":
                sub_intention[tmp_dim] = Ontology_Root[tmp_dim]
    for tmp_sample in positive_samples:
        for tmp_dim in tmp_sample:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]
    for tmp_sample in negative_samples:
        for tmp_dim in tmp_sample:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]

    real_intention_key = RetrievalUtil.get_intent_key(real_intention)
    docs = {"relevance": positive_samples, "irrelevance": negative_samples}
    print("path loaded:", sample_path)


if __name__ == "__main__":
    test_scene = "103"
    test_sample_path = "../../../../result/file/samples/scenes_v1/Scene" + \
                       test_scene[0:(2 if len(test_scene) == 3 else 1)] + "/Scene" + test_scene + "/final_samples.json"
    load_sample(test_sample_path)
    print(real_intention)
    print(real_intention_key)
    print(docs)