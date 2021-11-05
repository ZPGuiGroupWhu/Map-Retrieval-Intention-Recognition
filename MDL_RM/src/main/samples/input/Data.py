
Ontologies = {}
Ancestor = {}
Neighborhood = {}
direct_Ancestor = {}
Ontology_Root = {}  # 各个维度的根节点
docs = {}
real_intention_key = None
dimensions = []
all_relevance_concepts = None  # 所有正样本中出现的概念及其所有祖先概念
all_relevance_concepts_retrieved_docs = None  # 所有正样本中出现的概念及其所有祖先概念能够检索到的文档


# 获得某个维度在相关文档中出现的概念
def get_doc_concepts(dim_name):
    relevance_docs = docs["relevance"]
    result = set()
    for doc in relevance_docs:
        if dim_name in doc:
            for tmp_concept in doc[dim_name]:
                result.add(tmp_concept)
    result = list(result)
    return result


# 获取某个维度所有相关概念，包括在相关文档中出现的概念及这些概念的所有祖先概念
# result = [concept1, concept2, ...]
def get_all_relevance_concepts(dim_name):
    result = get_doc_concepts(dim_name)
    # 如果当前维度取值为枚举类型，则不存在上位概念，此维度的相关概念只包括相关文档中出现的概念
    if Ancestor[dim_name] is None:
        return result

    # 获取文档中出现的概念的上位概念
    hypernyms = []
    for concept in result:
        # print(dim_name, concept,)
        hypernyms += Ancestor[dim_name][concept]

    result = result + hypernyms
    result = list(set(result))
    return result


# 获取某个维度的某个概念的所有相关文档的索引
# result = {doc1_index, doc2_index, ...}
def get_concept_retrieved_docs(param_dim, param_concept, param_docs, ontology_root_concept):
    tmp_concept_all_relevance_concepts = [param_concept]
    # 取值为枚举类型的维度的取值没有下位概念
    if Ontologies[param_dim] is not None:
        tmp_concept_all_relevance_concepts += Ontologies[param_dim][param_concept]
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


def preprocess(new_docs):
    global Ontologies, Ancestor, Neighborhood, direct_Ancestor, docs, real_intention_key, dimensions, \
        all_relevance_concepts, all_relevance_concepts_retrieved_docs, Ontology_Root
    if dimensions is None:
        return
    docs = new_docs

    all_relevance_concepts = None  # 所有正样本中出现的概念及其所有祖先概念
    all_relevance_concepts_retrieved_docs = None  # 所有正样本中出现的概念及其所有祖先概念能够检索到的文档

    all_relevance_concepts = {}
    for dim in dimensions:
        all_relevance_concepts[dim] = get_all_relevance_concepts(dim)

    all_relevance_concepts_retrieved_docs = {}
    relevance_docs = docs["relevance"]
    irrelevance_docs = docs["irrelevance"]

    for tmp_dim in all_relevance_concepts.keys():
        tmp_dim_all_relevance_concepts = all_relevance_concepts[tmp_dim]
        tmp_dim_all_relevance_concepts_retrieved_docs = {}
        for tmp_concept in tmp_dim_all_relevance_concepts:
            tmp_concept_retrieved_relevance_docs = get_concept_retrieved_docs(tmp_dim, tmp_concept, relevance_docs,
                                                                              Ontology_Root[tmp_dim])
            tmp_concept_retrieved_irrelevance_docs = get_concept_retrieved_docs(tmp_dim, tmp_concept,
                                                                                irrelevance_docs,
                                                                                Ontology_Root[tmp_dim])
            tmp_concept_retrieved_docs = {"relevance": tmp_concept_retrieved_relevance_docs,
                                          "irrelevance": tmp_concept_retrieved_irrelevance_docs}
            tmp_dim_all_relevance_concepts_retrieved_docs[tmp_concept] = tmp_concept_retrieved_docs
        tmp_dim_all_relevance_concepts_retrieved_docs[Ontology_Root[tmp_dim]] = \
            {"relevance": set(range(len(relevance_docs))),
             "irrelevance": set(range(len(irrelevance_docs)))}
        all_relevance_concepts_retrieved_docs[tmp_dim] = tmp_dim_all_relevance_concepts_retrieved_docs


def init(sample):
    global Ontologies, Ancestor, Neighborhood, direct_Ancestor, docs, real_intention_key, dimensions, \
        all_relevance_concepts, all_relevance_concepts_retrieved_docs, Ontology_Root

    Ontologies = sample.Ontologies
    Ancestor = sample.Ancestor
    Neighborhood = sample.Neighborhood
    direct_Ancestor = sample.direct_Ancestor
    Ontology_Root = sample.Ontology_Root  # 各个维度的根节点
    real_intention_key = sample.real_intention_key
    dimensions = list(sample.real_intention[0].keys())

    preprocess(sample.docs)
    print("data inited")

