"""
    脚本描述：提供意图与样本关系判断接口
"""
import copy

Dimension_Name = ["Spatial", "MapContent", "MapMethod", "Theme"]


# 判断某个维度的取值是否覆盖某个样本，规则：若样本某维度存在语义上等价或从属于某子意图对应维度分量取值的标签，则认为样本在此维度满足该子意图
def is_value_cover_sample(dim, value, sample, ontologies):
    if value == "None":
        return True

    tmp_dim_sample_values = sample[dim]
    # 样本某维度从属某个意图对应维度分量取值的标签
    for tmp_sample_value in tmp_dim_sample_values:
        # 判断样本某维度是否等价某个意图对应维度分量取值的标签
        if tmp_sample_value == value:
            return True
        # 判断样本某维度是否从属某个意图对应维度分量取值的标签
        if tmp_sample_value in ontologies[dim][value]:
            return True
    return False


# 获取某个维度dim取值覆盖的样本id
# input params:
#   dim: 维度
#   relation_value_tuple: 关系取值二元组
#   samples: 未扩展的原始样本
def get_value_covered_samples(dim, value, samples, ontologies):
    value_covered_samples_index = set()  # 记录维度覆盖的样本ID
    for tmp_sample in samples:
        if is_value_cover_sample(dim, value, tmp_sample, ontologies):
            value_covered_samples_index.add(tmp_sample["ID"])
    return value_covered_samples_index


# 判断某个子意图是否覆盖某个样本
def is_sub_intention_cover_sample(sub_intention, sample, ontologies):
    for tmp_dim in Dimension_Name:
        tmp_dim_relation_value_tuple = sub_intention[tmp_dim]
        if not is_value_cover_sample(tmp_dim, tmp_dim_relation_value_tuple,
                                     sample, ontologies):
            return False
    return True


# 得到子意图覆盖的所有样本
def get_sub_intention_covered_samples(sub_intention, samples, ontologies):
    result_id = []          # 记录意图覆盖的样本ID
    result = []             # 记录意图覆盖的样本值
    for tmp_sample in samples:
        if is_sub_intention_cover_sample(sub_intention, tmp_sample, ontologies):
            result_id.append(tmp_sample["ID"])
            result.append(tmp_sample)
    return result_id, result


# 判断某个意图是否覆盖某个样本
def is_intention_cover_sample(intention, sample, ontologies):
    for tmp_sub_intention in intention:
        if is_sub_intention_cover_sample(tmp_sub_intention, sample, ontologies):
            return True
    return False


# 获得意图覆盖的所有样本
def get_intention_covered_samples(intention, samples, ontologies):
    result_index = []         # 记录意图覆盖的样本ID
    result = []               # 记录意图覆盖的样本值
    for tmp_sample in samples:
        if is_intention_cover_sample(intention, tmp_sample, ontologies):
            result_index.append(tmp_sample["ID"])
            result.append(tmp_sample)
    return result_index, result


# 判断该样本是否与意图等价（是否是意图的等价样本），意图等价样本定义：样本的每个维度都存在意图的相应维度的概念
def is_sub_intention_equal_sample(sub_intention, sample):
    for tmp_dim in Dimension_Name:
        sample_dim_values = sample[tmp_dim]
        sub_intention_dim_value = sub_intention[tmp_dim]
        is_dim_equal = False
        for tmp_sample_dim_value in sample_dim_values:
            if tmp_sample_dim_value == sub_intention_dim_value:
                is_dim_equal = True
                break
        if not is_dim_equal:
            return False
    return True


# 获得子意图等价样本，意图等价样本定义：样本的每个维度都存在意图的相应维度的概念
def get_sub_intention_equal_samples(sub_intention, samples):
    result_index = set()  # 记录子意图等价的样本ID
    result = []  # 记录子意图等价的样本值
    for tmp_sample in samples:
        if is_sub_intention_equal_sample(sub_intention, tmp_sample):
            result_index.add(tmp_sample["ID"])
            result.append(tmp_sample)
    return result_index, result


# 判断一个样本是维度单标签样本（标准：若样本的各个维度都只有一个取值，则是单标签样本，否则则是多标签样本）
def is_single_value_in_dim(sample):
    for tmp_dim in Dimension_Name:
        if len(sample[tmp_dim]) != 1:
            return False
    return True


# 将样本集合分为单标签样本和多标签样本索引（可视为反馈噪声）
def divide_into_single_and_multiple_value_samples(samples):
    single_value_samples_index = set()
    multiple_value_samples_index = set()
    for tmp_sample in samples:
        if is_single_value_in_dim(tmp_sample):
            single_value_samples_index.add(tmp_sample["ID"])
        else:
            multiple_value_samples_index.add(tmp_sample["ID"])
    return single_value_samples_index, multiple_value_samples_index


# 根据ID列表从样本集合中找到样本, 与get_sample_by_ID的区别是这里的ID是一个列表，ID= [](或者set())
# 不能在样本总库all_samples中使用，效率非常低下
def get_samples_by_ID_list(samples, ID_list):
    result = []
    for ID in ID_list:
        result.append(get_sample_by_ID(samples, ID))
    return result


# 根据ID从样本集合中找到样本, 与get_samples_by_ID_list区别是这里的ID是一个字符串（str）
# 不能在样本总库all_samples中使用，效率非常低下
def get_sample_by_ID(samples, ID):
    for tmp_sample in samples:
        if tmp_sample["ID"] == ID:
            return copy.deepcopy(tmp_sample)


# 获取某个维度dim下的某个relation_value_tuple覆盖的样本id， 弃用！！！
# 对于不同类型的关系需要使用不同类型的方法判断概念是否覆盖文档
# input params:
#   dim: 维度
#   relation_value_tuple: 关系取值二元组
#   samples: 样本，必须已经在Sample.py进行扩展，将每个维度的取值替换为（关系,取值）二元组的形式
# 如果是(subclass_of, A)，则需要利用此概念A的所有下位概念A_hyponyms，只要此样本中包含（subclass_of, B），其中B属于A_hyponyms或等于A
# 如果是(spatial_equals, A)，则需要标签中包含（spatial_equals, B）,且A=B
# 如果是(nearby, A)，则需要利用A的所有邻近概念A_nearby_features，要求样本中包含(spatial_equals, B), B属于A_nearby_features
# 如果是(outer_direction_relation_of, A)，则要求样本中包含(spatial_equals, B)，A属于B_outer_direction_relation_of_features
# 如果是(inner_direction_relation_of, A)，则要求样本中包含(spatial_equals, B)，A属于B_inner_direction_relation_of_features
# 如果是(part_of, A)，则要求样本中包含(spatial_equals, B)，A属于B_parent_of_features
def get_relation_value_tuple_covered_expanded_samples(dim, relation_value_tuple, expanded_samples,
                                                      ontology_all_hyponyms, ontology_with_concept_relation,
                                                      ontology_relation_type):
    if dim == "Spatial" and relation_value_tuple == 'None':
        return set(range(len(expanded_samples)))
    relation_value_tuple_covered_samples = set()
    relation_name, value = relation_value_tuple
    for i, tmp_sample in enumerate(expanded_samples):
        tmp_sample_tmp_dim_relation_value_tuples = tmp_sample[dim]
        # tmp_sample_tmp_dim_spatial_equals_values = [x[1] for x in filter(lambda x: x[0] == "spatial_equals",
        #                                                                  tmp_sample_tmp_dim_relation_value_tuples)]
        if relation_name == "subclass_of":
            value_all_hyponyms = ontology_all_hyponyms[dim][value]
            tmp_sample_tmp_dim_subclassof_values = [x[1] for x in filter(lambda x: x[0] == "subclass_of",
                                                                         tmp_sample_tmp_dim_relation_value_tuples)]
            if value in tmp_sample_tmp_dim_subclassof_values or len(
                    set(value_all_hyponyms) & set(tmp_sample_tmp_dim_subclassof_values)) != 0:
                relation_value_tuple_covered_samples.add(i)
        elif relation_name in ontology_relation_type["Spatial"]:
            tmp_sample_spatial_equals_value = \
                list(filter(lambda x: x[0] == "spatial_equals", tmp_sample_tmp_dim_relation_value_tuples))[0][1]
            tmp_sample_value_spatial_relation_features = \
                ontology_with_concept_relation[dim][relation_name][tmp_sample_spatial_equals_value]
            if value in tmp_sample_value_spatial_relation_features:
                relation_value_tuple_covered_samples.add(i)
    return relation_value_tuple_covered_samples


# 2023.02.15从以往代码中找到retrieve_docs 和 get_intent_key函数并加入，确保GenerateSample.py能正常运行，但无法保证其正确性
# 查询文档
# 输出的是能查询到的文档的索引
# 这里的intent需要改成sub intent
def retrieve_docs(intent, docs, ontologies, ontology_root):
    intent_dim_values = {}
    for tmp_dim in intent:
        tmp_dim_value = intent[tmp_dim]
        tmp_dim_values = [tmp_dim_value]
        if ontologies[tmp_dim] is not None:
            tmp_dim_values += ontologies[tmp_dim][tmp_dim_value]
        tmp_dim_values = list(set(tmp_dim_values))
        intent_dim_values[tmp_dim] = tmp_dim_values
        # print(tmp_dim, "None ", concept_None_id in tmp_dim_values)
        # print("\t", tmp_dim_valu      es)

    retrieved_docs = set()

    for i in range(len(docs)):
        # print(i / len(docs))
        tmp_doc = docs[i]
        can_retrieve_tmp_doc = True
        for tmp_dim in intent.keys():
            tmp_dim_value = intent[tmp_dim]
            if tmp_dim_value == ontology_root[tmp_dim]:
                continue
            else:
                tmp_dim_values = intent_dim_values[tmp_dim]

                # if len(set(tmp_dim_values) & set(tmp_doc[tmp_dim])) == 0:
                #     can_retrieve_tmp_doc = False
                #     break

                tmp_dim_can_retrieve = False
                for tmp_label in tmp_doc[tmp_dim]:
                    if tmp_label in tmp_dim_values:
                        tmp_dim_can_retrieve = True
                        break
                if not tmp_dim_can_retrieve:
                    can_retrieve_tmp_doc = False
                    break
        if can_retrieve_tmp_doc:
            retrieved_docs.add(i)
    #
    #
    #     if term == "None" or term in tmp_doc[dim]:
    #         term_retrieved_docs.add(i)
    #
    # for dim in intent:
    #     dim_retrieved_docs = set()
    #     term = intent[dim]
    #     # for term in intent[dim]:
    #     term_retrieved_docs = set()
    #     for i in range(len(docs)):
    #         # print(i / len(docs))
    #         tmp_doc = docs[i]
    #         if term == "None" or term in tmp_doc[dim]:
    #             term_retrieved_docs.add(i)
    #         else:
    #             if ontologies[dim] is not None:
    #                 tmp_hyponyms = ontologies[dim][term]
    #                 for tmp_label in tmp_doc[dim]:
    #                     if tmp_label in tmp_hyponyms:
    #                         term_retrieved_docs.add(i)
    #                         break
    #
    #         # if term in tmp_doc[dim] or term == 'None':
    #         #     term_retrieved_docs.add(i)
    #         # elif ontologies[dim] is not None:     # 如果某个维度是None，则说明这个维度的值是枚举类型的
    #         #     hyponyms = ontologies[dim][term]
    #         #     for hyponym in hyponyms:
    #         #         if hyponym in tmp_doc[dim]:
    #         #             term_retrieved_docs.add(i)
    #         #             break
    #     if len(dim_retrieved_docs) == 0:
    #         dim_retrieved_docs = term_retrieved_docs
    #     else:
    #         dim_retrieved_docs &= term_retrieved_docs
    #
    #     if len(retrieved_docs) == 0:
    #         retrieved_docs = dim_retrieved_docs
    #     else:
    #         retrieved_docs &= dim_retrieved_docs
    return retrieved_docs


# 得到意图的键形式
# 这里的intent是全意图
def get_intent_key(intent):
    intent = [list(x.items()) for x in intent]
    for sub_intent in intent:
        sub_intent.sort()
    # print(intent)
    intent.sort()
    intent = str(intent)
    # intent = [tuple(x) for x in intent]
    # intent = tuple(intent)
    return intent
