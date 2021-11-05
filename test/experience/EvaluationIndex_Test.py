from main.experience import EvaluationIndex
from main.intention_recognition import Config, MDL_RM
from main.samples.input import Sample, Data


def test_get_intention_similarity():
    test_intention_b = [{"Spatial": "America", "MapContent": "Thing",
                         "MapMethod": "MapMethodRoot", "Theme": "ThemeRoot"}]
    test_intention_c = [{"Spatial": "America", "MapContent": "Thing",
                         "MapMethod": "MapMethodRoot", "Theme": "ThemeRoot"}]
    print(EvaluationIndex.get_intention_similarity(test_intention_c, test_intention_b, Sample.direct_Ancestor,
                                                   Sample.Ontology_Root, Sample.concept_information_content))


def test_get_evaluation_index():
    scene = "151"
    sample_version = "scenes_v4_5"
    test_sample_path = "./../../resources/samples/" + sample_version \
                       + "/Scene" + scene + "/noise_samples_S0p3_L0p8.json"
    Sample.load_sample(test_sample_path)
    Data.init(Sample)
    real_intention = Sample.real_intention
    test_ontologies = Data.Ontologies
    test_ontology_root = Data.Ontology_Root
    test_direct_ancestors = Data.direct_Ancestor
    test_information_content = Sample.concept_information_content
    test_samples = Data.docs
    data_encoding_method = "amcl"
    method_result = MDL_RM.get_intention_by_method6(test_samples, data_encoding_method, 50, 10, 0.3)
    predict_intention = MDL_RM.result_to_intention(method_result)
    jaccard_score = EvaluationIndex.get_jaccard_index(test_samples, real_intention,
                                                      predict_intention, test_ontologies, test_ontology_root)
    intention_similarity = EvaluationIndex.get_intention_similarity(real_intention, predict_intention,
                                                                    test_direct_ancestors, test_ontology_root,
                                                                    test_information_content)
    precision = EvaluationIndex.get_precision(test_samples, real_intention, predict_intention,
                                              test_ontologies, test_ontology_root)
    recall = EvaluationIndex.get_recall(test_samples, real_intention, predict_intention,
                                        test_ontologies, test_ontology_root)
    print("jaccard_score", jaccard_score)
    print("intention_similarity", intention_similarity)
    print("precision", precision)
    print("recall", recall)


if __name__ == "__main__":
    test_get_intention_similarity()
    test_get_evaluation_index()
    print("Aye")
