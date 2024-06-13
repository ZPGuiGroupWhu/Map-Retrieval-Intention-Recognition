# 试验用样本
from numpy.ma import copy

from Map_RIR.src.main.samples.input import DimensionValues
from Map_RIR.src.main.util.FileUtil import load_json

Ontology_Root = {'Spatial': DimensionValues.SpatialValue.Ontology_Root,
                 'Theme': DimensionValues.ThemeValues.Ontology_Root,
                 'MapMethod': DimensionValues.MapMethodValues.Ontology_Root,
                 'MapContent': DimensionValues.MapContentValues.Ontology_Root}


def transform_sample(relevance_feedback_samples):
    # transform S,C,M,T to Spatial,MapContent,MapMethod,Theme
    positive_samples = []
    negative_samples = []
    positive_sample_key = "positive_samples" if "positive_samples" in relevance_feedback_samples else "relevance"
    negative_sample_key = "negative_samples" if "negative_samples" in relevance_feedback_samples else "irrelevance"
    for tmp_sample in relevance_feedback_samples[positive_sample_key]:
        tmp_sample_copy = {"Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        positive_samples.append(tmp_sample_copy)
    for tmp_sample in relevance_feedback_samples[negative_sample_key]:
        tmp_sample_copy = {"Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        negative_samples.append(tmp_sample_copy)

    # change None to root concept
    for tmp_sample in positive_samples:
        for tmp_dim in tmp_sample:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]
    for tmp_sample in negative_samples:
        for tmp_dim in tmp_sample:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]
    result = {"relevance": positive_samples, "irrelevance": negative_samples}
    return result


def transform_sample_reverse(relevance_feedback_samples):
    # transform S,C,M,T to Spatial,MapContent,MapMethod,Theme
    positive_samples = []
    negative_samples = []
    positive_sample_key = "positive_samples" if "positive_samples" in relevance_feedback_samples else "relevance"
    negative_sample_key = "negative_samples" if "negative_samples" in relevance_feedback_samples else "irrelevance"
    for tmp_sample in relevance_feedback_samples[positive_sample_key]:
        tmp_sample_copy = {"Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        positive_samples.append(tmp_sample_copy)
    for tmp_sample in relevance_feedback_samples[negative_sample_key]:
        tmp_sample_copy = {"Spatial": tmp_sample["S"] if "S" in tmp_sample else tmp_sample["Spatial"],
                           "MapContent": tmp_sample["C"] if "C" in tmp_sample else tmp_sample["MapContent"],
                           "MapMethod": tmp_sample["M"] if "M" in tmp_sample else tmp_sample["MapMethod"],
                           "Theme": tmp_sample["T"] if "T" in tmp_sample else tmp_sample["Theme"]}
        negative_samples.append(tmp_sample_copy)

    # change None to root concept
    for tmp_sample in positive_samples:
        for tmp_dim in tmp_sample:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]
    for tmp_sample in negative_samples:
        for tmp_dim in tmp_sample:
            if tmp_sample[tmp_dim] == ["None"]:
                tmp_sample[tmp_dim] = [Ontology_Root[tmp_dim]]
    result = {"relevance": negative_samples, "irrelevance": positive_samples}
    return result


def load_sample_from_file(sample_path):
    tmp_samples = load_json(sample_path)
    tmp_intention = tmp_samples["intention"]["positive"]
    #tmp_intention = tmp_samples["intention"]
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
        for tmp_dim in sub_intention:
            if sub_intention[tmp_dim] == "None":
                sub_intention[tmp_dim] = Ontology_Root[tmp_dim]
    docs = transform_sample(tmp_samples)
    return docs, real_intention


def get_real_intent():
    real_intents = {}
    json_intent = load_json("../../../../RFSCE_MRIR/resources/samples/Intention7.30.json")
    #real_intents = copy.deepcopy(json_intent)
    for key in json_intent:
        real_intents[key] = []
        dim_trans = {"S": "Spatial", "C": "MapContent", "M": "MapMethod", "T": "Theme"}
        for sub_intention in json_intent[key]:
            real_sub_positive = {"Spatial": sub_intention["positive"]["S"] if "S" in sub_intention["positive"] else sub_intention["positive"]["Spatial"],
                 "MapContent": sub_intention["positive"]["C"] if "C" in sub_intention["positive"] else sub_intention["positive"][
                     "MapContent"],
                 "MapMethod": sub_intention["positive"]["M"] if "M" in sub_intention["positive"] else sub_intention["positive"][
                     "MapMethod"],
                 "Theme": sub_intention["positive"]["T"] if "T" in sub_intention["positive"] else sub_intention["positive"]["Theme"]}
            real_sub_intent = {"positive": real_sub_positive,
                               "negative": []}
            if sub_intention["negative"] is not None:
                for negative_intent in sub_intention["negative"]:
                    real_sub_intent["negative"].append((dim_trans[list(negative_intent.keys())[0]], list(negative_intent.values())[0]))
            real_intents[key].append(real_sub_intent)
        #print(real_intents[key])

    return real_intents

def load_real_intents(real_intents, sample_path, scene_name):
    tmp_samples = load_json(sample_path)
    # change None to root concept
    real_intention = real_intents[scene_name]
    for sub_intention in real_intention:
        for tmp_dim in sub_intention["positive"]:
            if sub_intention["positive"][tmp_dim] == "None":
                sub_intention["positive"][tmp_dim] = Ontology_Root[tmp_dim]
    docs = transform_sample(tmp_samples)
    return docs, real_intention





def load_sample_from_file_reverse(sample_path):
    global docs, real_intention_key, real_intention
    tmp_samples = load_json(sample_path)
    tmp_intention = tmp_samples["intention"]["positive"]
    #tmp_intention = tmp_samples["intention"]
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
        for tmp_dim in sub_intention:
            if sub_intention[tmp_dim] == "None":
                sub_intention[tmp_dim] = Ontology_Root[tmp_dim]
    docs = transform_sample_reverse(tmp_samples)
    return docs, real_intention

