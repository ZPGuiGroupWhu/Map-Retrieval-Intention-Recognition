"""
   脚本描述：反馈数据初始化，为频繁项集挖掘做准备
   ① 分维度统计所有正样本中出现的概念及其所有祖先概念 self.all_relevance_concepts
   ② 分维度统计所有正样本中出现的概念及其所有祖先概念覆盖的正负样本集合  self.all_relevance_concepts_retrieved_docs
"""

from src.main.samples.input import DimensionValues
from src.main.util import RetrievalUtil


class Data:
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
    Ontology_Root = {'Spatial': DimensionValues.SpatialValue.Ontology_Root,
                     'Theme': DimensionValues.ThemeValues.Ontology_Root,
                     'MapMethod': DimensionValues.MapMethodValues.Ontology_Root,
                     'MapContent': DimensionValues.MapContentValues.Ontology_Root}

    Ontology_values = {'Spatial': DimensionValues.SpatialValue.Ontologies.keys(),
                       'Theme': DimensionValues.ThemeValues.Ontologies.keys(),
                       'MapMethod': DimensionValues.MapMethodValues.Ontologies.keys(),
                       'MapContent': DimensionValues.MapContentValues.Ontologies.keys()}
    direct_Hyponyms = {'Spatial': DimensionValues.SpatialValue.Direct_Hyponyms,
                       'Theme': DimensionValues.ThemeValues.Direct_Hyponyms,
                       'MapMethod': DimensionValues.MapMethodValues.Direct_Hyponyms,
                       'MapContent': DimensionValues.MapContentValues.Direct_Hyponyms
                       }
    dimensions = list(Ontology_Root.keys())  # 涉及到的意图维度

    def __init__(self, relevance_feedback_samples, real_intention=None):
        self.real_intention = real_intention
        self.docs = relevance_feedback_samples
        self.real_intention_key = None
        if real_intention is not None:
            self.real_intention_key = RetrievalUtil.get_intent_key(real_intention)

        # self.dimensions = dimensions  # 涉及到的意图维度

        self.all_relevance_concepts = {}  # 所有正样本中出现的概念及其所有祖先概念
        for dim in self.dimensions:
            self.all_relevance_concepts[dim] = self.get_all_relevance_concepts(dim)

        self.all_relevance_concepts_retrieved_docs = {}  # 所有正样本中出现的概念及其所有祖先概念能够检索到的文档
        relevance_docs = self.docs["positive"]
        irrelevance_docs = self.docs["negative"]

        for tmp_dim in self.all_relevance_concepts.keys():
            tmp_dim_all_relevance_concepts = self.all_relevance_concepts[tmp_dim]
            tmp_dim_all_relevance_concepts_retrieved_docs = {}
            for tmp_concept in tmp_dim_all_relevance_concepts:
                tmp_concept_retrieved_relevance_docs = \
                    self.get_concept_retrieved_docs(tmp_dim, tmp_concept, relevance_docs, Data.Ontology_Root[tmp_dim])
                tmp_concept_retrieved_irrelevance_docs = self.get_concept_retrieved_docs(tmp_dim, tmp_concept,
                                                                                         irrelevance_docs,
                                                                                         Data.Ontology_Root[tmp_dim])
                tmp_concept_retrieved_docs = {"positive": tmp_concept_retrieved_relevance_docs,
                                              "negative": tmp_concept_retrieved_irrelevance_docs}
                tmp_dim_all_relevance_concepts_retrieved_docs[tmp_concept] = tmp_concept_retrieved_docs
            tmp_dim_all_relevance_concepts_retrieved_docs[Data.Ontology_Root[tmp_dim]] = \
                {"positive": set(range(len(relevance_docs))),
                 "negative": set(range(len(irrelevance_docs)))}
            self.all_relevance_concepts_retrieved_docs[tmp_dim] = tmp_dim_all_relevance_concepts_retrieved_docs

    # 获得某个维度在相关文档(正样本集合）中出现的概念
    def get_doc_concepts(self, dim_name):
        relevance_docs = self.docs["positive"]
        result = set()
        for doc in relevance_docs:
            if dim_name in doc:
                for tmp_concept in doc[dim_name]:
                    result.add(tmp_concept)
        result = list(result)
        return result

    # 获取某个维度所有相关概念，包括在相关文档（正样本集合）中出现的概念及这些概念的所有祖先概念（包括维度的根节点）
    # result = [concept1, concept2, ...]
    def get_all_relevance_concepts(self, dim_name):
        result = self.get_doc_concepts(dim_name)
        # 如果当前维度取值为枚举类型，则不存在上位概念，此维度的相关概念只包括相关文档中出现的概念（判断永远不成立）
        if Data.Ancestor[dim_name] is None:
            return result

        # 获取文档中出现的概念的上位概念
        hypernyms = []
        for concept in result:
            # print(dim_name, concept,)
            hypernyms += Data.Ancestor[dim_name][concept]

        result = result + hypernyms
        result = list(set(result))
        return result

    # 获取某个维度的某个概念的所有相关文档的索引,(获取概念覆盖的正（负）样本对应的ID)
    # result = {doc1_index, doc2_index, ...}
    def get_concept_retrieved_docs(self, param_dim, param_concept, param_docs, ontology_root_concept):
        tmp_concept_all_relevance_concepts = [param_concept]
        # 取值为枚举类型的维度的取值没有下位概念
        if Data.Ontologies[param_dim] is not None:
            tmp_concept_all_relevance_concepts += Data.Ontologies[param_dim][param_concept]
        tmp_concept_retrieved_tmp_docs = set()
        for i, tmp_doc in enumerate(param_docs):
            if param_dim not in tmp_doc.keys():
                continue
            else:
                tmp_dim_doc_values = tmp_doc[param_dim]
                tmp_can_be_retrieved = False
                if param_concept == ontology_root_concept:
                    tmp_can_be_retrieved = True
                else:
                    for tmp_dim_doc_value in tmp_dim_doc_values:
                        if tmp_dim_doc_value in tmp_concept_all_relevance_concepts:
                            tmp_can_be_retrieved = True
                            break
                if tmp_can_be_retrieved:
                    tmp_concept_retrieved_tmp_docs.add(i)
        # if param_concept == 'None':
        #     print(param_dim, param_concept, tmp_concept_all_relevance_concepts, tmp_concept_retrieved_tmp_docs)
        return tmp_concept_retrieved_tmp_docs
