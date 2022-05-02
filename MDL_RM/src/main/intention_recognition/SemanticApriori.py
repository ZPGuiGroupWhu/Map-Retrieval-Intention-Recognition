# 使用维度约束的Apriori挖掘候选子意图


# param：
#   all_relevance_concepts：正样本中所有标签的所有相关概念，包括标签本身及它们的所有上位概念
#   该概念集合由Data.py在导入相关反馈样本的同时生成，其格式为{dim1: [concept11, concept12], dim2: [concpet21, ]}
from MDL_RM.src.main.samples.input.Data import Data


def get_init_sub_intentions(all_relevance_concepts):
    result = []
    for tmp_dim in all_relevance_concepts:
        tmp_dim_all_relevance_concepts = all_relevance_concepts[tmp_dim]
        for tmp_concept in tmp_dim_all_relevance_concepts:
            tmp_item = (tmp_dim, tmp_concept)
            result.append(frozenset({tmp_item}))
    result.sort()
    return result


# 获取频繁维度分量集合（也即候选子意图）覆盖的样本集合，通过sample_type指定样本集合的正负
# 参数：
#   sub_intention: [(dim_name, concept), ...]
#   all_relevance_concepts_covered_samples：正样本相关概念覆盖的正负样本id集合，该变量由Data.py在导入样本的时候生成
def get_sub_intention_covered_samples(sub_intention, all_relevance_concepts_covered_samples, sample_type):
    result = set()
    first_value = True
    for tmp_dim_value in sub_intention:
        tmp_dim_name, tmp_concept = tmp_dim_value
        tmp_relation_value_tuple_covered_samples_id = \
            all_relevance_concepts_covered_samples[tmp_dim_name][tmp_concept][sample_type]
        if first_value:
            result |= tmp_relation_value_tuple_covered_samples_id
            first_value = False
        else:
            result &= tmp_relation_value_tuple_covered_samples_id
    return result


# 获取候选子意图覆盖的样本集合，通过sample_type指定样本集合的正负
#   该方法与get_sub_intention_covered_samples的不同点在于子意图的格式不同
# sub_intention: {dim_name: concept, ...}
def get_sub_intention_covered_samples2(sub_intention, all_relevance_concepts_covered_samples, sample_type):
    result = set()
    first_value = True
    for tmp_dim_name in sub_intention:
        tmp_concept = sub_intention[tmp_dim_name]
        tmp_relation_value_tuple_covered_samples_id = \
            all_relevance_concepts_covered_samples[tmp_dim_name][tmp_concept][sample_type]
        if first_value:
            result |= tmp_relation_value_tuple_covered_samples_id
            first_value = False
        else:
            result &= tmp_relation_value_tuple_covered_samples_id
    return result


# intention: 意图[[(dim_name, concept), ...], ...]
def get_intention_covered_samples(intention, all_relevance_concepts_covered_samples, sample_type):
    result = set()
    for tmp_sub_intention in intention:
        tmp_sub_intention_covered_samples = \
            get_sub_intention_covered_samples(tmp_sub_intention, all_relevance_concepts_covered_samples,
                                              sample_type)
        result |= tmp_sub_intention_covered_samples
    return result


# 检查sub_intention是否合法，即是否每个维度仅包含一个值
def is_legal_sub_intention(sub_intention, dimensions):
    sub_intention_dims = [tmp_dim_value[0] for tmp_dim_value in sub_intention]
    for tmp_dim in dimensions:
        tmp_dim_count = sub_intention_dims.count(tmp_dim)
        if tmp_dim_count > 1:
            return False
    return True


# 检查intention是否合法，即是否每个维度仅包含一个值
def is_legal_intention(intention, dimensions):
    intention_dims = [[tmp_dim_value[0] for tmp_dim_value in sub_intention] for sub_intention in intention]
    for tmp_sub_intention_dims in intention_dims:
        for tmp_dim in dimensions:
            tmp_dim_count = tmp_sub_intention_dims.count(tmp_dim)
            if tmp_dim_count > 1:
                return False
    return True


# 扫描项集，去除支持度小于阈值的项集
def scan_sub_intentions(sub_intentions_k, min_support, all_relevance_concepts_covered_samples, samples,
                        dimensions):
    # 计算sub_intention_k中每一个子意图覆盖的正样本数量，返回满足最小支持度且各维度仅含有一个取值的子意图
    # 并返回支持度信息
    result = []
    relevance_samples_num = len(samples["relevance"])
    for tmp_sub_intention in sub_intentions_k:
        if not is_legal_sub_intention(tmp_sub_intention, dimensions):
            continue
        tmp_sub_intention_covered_relevance_samples = \
            get_sub_intention_covered_samples(tmp_sub_intention, all_relevance_concepts_covered_samples,
                                              "relevance")
        tmp_sub_intention_covered_relevance_samples_num = len(tmp_sub_intention_covered_relevance_samples)
        tmp_sub_intention_support = float(tmp_sub_intention_covered_relevance_samples_num) / relevance_samples_num
        if tmp_sub_intention_support > min_support:
            result.append(tmp_sub_intention)
    return result


# 由频繁k项集扩展得到频繁k+1项集
def extend_sub_intentions(sub_intention_k, k):
    result = []
    sub_intention_k_num = len(sub_intention_k)
    for i in range(sub_intention_k_num):
        for j in range(i + 1, sub_intention_k_num):
            L1 = list(sub_intention_k[i])[: k - 2]
            L2 = list(sub_intention_k[j])[: k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                result.append(sub_intention_k[i] | sub_intention_k[j])
    return result


# 将频繁维度分量转换为候选子意图的形式
def transform_sub_intention(sub_intention, ontology_root_concepts):
    result = {}
    for tmp_dim_value in sub_intention:
        tmp_dim_name, tmp_relation_value_tuple = tmp_dim_value
        result[tmp_dim_name] = tmp_relation_value_tuple
    for tmp_dim in ontology_root_concepts:
        if tmp_dim not in result:
            result[tmp_dim] = ontology_root_concepts[tmp_dim]
    return result


# 基于正样本集合获得候选子意图（正向）
# 参数：
#   samples：反馈样本集合，格式为{"relevance": [sample1, ...], "irrelevance": [sample2, ...]}
#       sample1 = {"dim1": [concept1, concept2], dim2: [concept3, ...], ...}
#   min_support：候选子意图的最小支持度阈值，取值为(0, 1]
#   k_max：候选子意图中包含的最大维度分量数量，由于当前形式化表达方式规定子意图中各维度至多出现一次，因此该值等于维度数量
def get_all_candidate_sub_intentions(samples, min_support, k_max=4):
    data = Data(samples)
    all_relevance_concepts = data.all_relevance_concepts
    samples = data.docs
    dimensions = data.dimensions
    all_relevance_concepts_covered_samples = data.all_relevance_concepts_retrieved_docs
    init_sub_intentions = get_init_sub_intentions(all_relevance_concepts)
    L1 = scan_sub_intentions(init_sub_intentions, min_support,
                             all_relevance_concepts_covered_samples, samples, dimensions)
    L = [L1]
    if k_max == 1:
        return L1
    k = 2
    while len(L[k - 2]) > 0:
        Ck = extend_sub_intentions(L[k - 2], k)
        Lk = scan_sub_intentions(Ck, min_support, all_relevance_concepts_covered_samples,
                                 samples, dimensions)
        L.append(Lk)
        k += 1
        if k_max is not None:
            if k > k_max:
                break
    candidate_sub_intentions = [x for y in L for x in y]
    candidate_sub_intentions = [transform_sub_intention(x, data.Ontology_Root) for x in candidate_sub_intentions]
    return candidate_sub_intentions

