from Map_RIR.src.main.samples.input import Data


def get_term_document_count(x, dim, docs, ancestor):
    num = 0
    for tmp_doc in docs:
        found_x = False
        for tmp_term in tmp_doc[dim]:
            # print(dim, tmp_term)
            # 若维度dim的取值是枚举类型，则不存在上位概念
            tmp_hypernyms = []
            if ancestor[dim] is not None:
                tmp_hypernyms = ancestor[dim][tmp_term]

            if tmp_term == x or x in tmp_hypernyms:
                found_x = True
                break
        if found_x:
            num += 1
    return num


# 查询文档
# 输出的是能查询到的文档的索引
def retrieve_docs(sub_intent, docs, ontologies, ontology_root):
    sub_intent_dim_values = {}

    for tmp_dim in sub_intent:
        tmp_dim_value = sub_intent[tmp_dim]
        tmp_dim_values = [tmp_dim_value]
        if ontologies[tmp_dim] is not None:
            tmp_dim_values += ontologies[tmp_dim][tmp_dim_value]
        tmp_dim_values = list(set(tmp_dim_values))
        sub_intent_dim_values[tmp_dim] = tmp_dim_values
        # print(tmp_dim, "None ", concept_None_id in tmp_dim_values)
        # print("\t", tmp_dim_values)
    retrieved_docs = set()
    for i in range(len(docs)):
        # print(i / len(docs))
        tmp_doc = docs[i]
        can_retrieve_tmp_doc = True
        for tmp_dim in sub_intent.keys():
            tmp_dim_value = sub_intent[tmp_dim]
            if tmp_dim_value == ontology_root[tmp_dim]:
                continue
            else:
                tmp_dim_values = sub_intent_dim_values[tmp_dim]

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
    return retrieved_docs


# 查询文档
# 输出的是能查询到的文档的索引
# 排除sub_intent的负向意图
def retrieve_docs_negative(sub_intent, docs, ontologies, ontology_root):
    sub_intent_dim_values = {}
    negative_dim_values = {}
    positive_intent = sub_intent["positive"]
    negative_list = sub_intent["negative"]
    for tmp_dim in positive_intent:
        tmp_dim_value = positive_intent[tmp_dim]
        tmp_dim_values = [tmp_dim_value]
        if ontologies[tmp_dim] is not None:
            tmp_dim_values += ontologies[tmp_dim][tmp_dim_value]
        tmp_dim_values = list(set(tmp_dim_values))
        sub_intent_dim_values[tmp_dim] = tmp_dim_values
        negative_dim_values[tmp_dim] = []
        # print(tmp_dim, "None ", concept_None_id in tmp_dim_values)
        # print("\t", tmp_dim_values)
    for negative_intent in negative_list:
        tmp_dim_value = negative_intent[1]
        tmp_dim_values = [tmp_dim_value]
        tmp_dim = negative_intent[0]
        if ontologies[tmp_dim] is not None:
            tmp_dim_values += ontologies[tmp_dim][tmp_dim_value]
        tmp_dim_values = list(set(tmp_dim_values))
        negative_dim_values[tmp_dim] += tmp_dim_values
    retrieved_docs = set()
    for i in range(len(docs)):
        # print(i / len(docs))
        tmp_doc = docs[i]
        can_retrieve_tmp_doc = True
        for tmp_dim in positive_intent.keys():
            tmp_dim_value = positive_intent[tmp_dim]
            tmp_dim_sift_values = negative_dim_values[tmp_dim]
            if tmp_dim_value == ontology_root[tmp_dim]:
                tmp_dim_can_retrieve = False
                for tmp_label in tmp_doc[tmp_dim]:
                    if tmp_label not in tmp_dim_sift_values:
                        tmp_dim_can_retrieve = True
                        break
                if not tmp_dim_can_retrieve:
                    can_retrieve_tmp_doc = False
                    break
            else:
                tmp_dim_values = sub_intent_dim_values[tmp_dim]
                tmp_dim_can_retrieve = False
                for tmp_label in tmp_doc[tmp_dim]:
                    if tmp_label in tmp_dim_values and tmp_label not in tmp_dim_sift_values:
                        tmp_dim_can_retrieve = True
                        break
                if not tmp_dim_can_retrieve:
                    can_retrieve_tmp_doc = False
                    break
        if can_retrieve_tmp_doc:
            retrieved_docs.add(i)
    return retrieved_docs


def is_negative_paste(sub_intention, sub_negative_intention, ancestors, ontology_root):
    dims_result = []
    paste_dim = None
    for tmp_dim in sub_intention:
        intent_a_value = sub_intention[tmp_dim]
        intent_b_value = sub_negative_intention[tmp_dim]
        if intent_a_value == intent_b_value:
            dims_result.append("e")
        elif intent_b_value == ontology_root[tmp_dim] or \
                (ancestors[tmp_dim] is not None and intent_b_value in ancestors[tmp_dim][intent_a_value]):
            dims_result.append("a")
        else:
            paste_dim = tmp_dim

    if len(dims_result) == (len(sub_intention.keys()) - 1) and \
            (ancestors[paste_dim] is not None and sub_intention[paste_dim] in ancestors[paste_dim][
                sub_negative_intention[paste_dim]]):
        return paste_dim
    else:
        return 0


def retrieve_docs_based_on_terms_covered_samples1(data, sub_intention, terms_covered_samples, sample_category,
                                                  negative_rules):
    result = set()
    negative_covered_samples = set()
    first_term = True
    first_term_negative = True
    if negative_rules != {}:
        for tmp_negative_intention in negative_rules:
            paste_dim = is_negative_paste(sub_intention, tmp_negative_intention, data.Ancestor, data.Ontology_Root)

            if paste_dim == 0:
                continue
            else:
                for tmp_dim in sub_intention:
                    if tmp_dim == paste_dim:
                        tmp_value = tmp_negative_intention[tmp_dim]
                    else:
                        tmp_value = sub_intention[tmp_dim]
                    tmp_value_covered_samples = terms_covered_samples[tmp_dim][tmp_value]
                    tmp_value_covered_specific_samples = tmp_value_covered_samples["irrelevance"]
                    if first_term_negative:  # 某个概念可能本来覆盖的负样本就是0，因此不能通过result是否为空来判断
                        negative_covered_samples = negative_covered_samples.union(tmp_value_covered_specific_samples)
                        first_term_negative = False
                    else:
                        negative_covered_samples = negative_covered_samples.intersection(
                            tmp_value_covered_specific_samples)

    for tmp_dim in sub_intention:
        tmp_value = sub_intention[tmp_dim]
        tmp_value_covered_specific_samples1 = None
        tmp_value_covered_samples1 = terms_covered_samples[tmp_dim][tmp_value]
        if sample_category == "positive":
            tmp_value_covered_specific_samples1 = tmp_value_covered_samples1["relevance"]
        elif sample_category == "negative":
            tmp_value_covered_specific_samples1 = tmp_value_covered_samples1["irrelevance"]
            a = len(tmp_value_covered_specific_samples1)
            tmp_value_covered_specific_samples1 = tmp_value_covered_specific_samples1 - negative_covered_samples
            # print("cah", a - len(tmp_value_covered_specific_samples1))
        if first_term:  # 某个概念可能本来覆盖的负样本就是0，因此不能通过result是否为空来判断
            result = result.union(tmp_value_covered_specific_samples1)
            first_term = False
        else:
            result = result.intersection(tmp_value_covered_specific_samples1)

    return result


def retrieve_docs_based_on_terms_covered_samples2 \
                (positive_sub_Intention, negative_sub_Intention, terms_covered_samples, sample_category):
    result = set()
    negative_covered_negative_samples = set()
    negative_covered_positive_samples = set()
    first_term = True
    all_negative_covered_negative_samples = set()
    all_negative_covered_positive_samples = set()
    # print("len(negative_sub_Intention)",negative_sub_Intention)
    if negative_sub_Intention:
        for tmp_negative_intention in negative_sub_Intention:
            first_term_negative = True
            for tmp_dim in positive_sub_Intention:
                if tmp_dim == tmp_negative_intention[0]:
                    tmp_value = tmp_negative_intention[1]
                else:
                    tmp_value = positive_sub_Intention[tmp_dim]
                tmp_value_covered_samples = terms_covered_samples[tmp_dim][tmp_value]
                tmp_value_covered_negative_samples = tmp_value_covered_samples["irrelevance"]
                tmp_value_covered_positive_samples = tmp_value_covered_samples["relevance"]
                if first_term_negative:  # 某个概念可能本来覆盖的负样本就是0，因此不能通过result是否为空来判断
                    negative_covered_negative_samples = negative_covered_negative_samples.union(
                        tmp_value_covered_negative_samples)
                    negative_covered_positive_samples = negative_covered_positive_samples.union(
                        tmp_value_covered_positive_samples)
                    first_term_negative = False

                else:
                    negative_covered_negative_samples = negative_covered_negative_samples.intersection(
                        tmp_value_covered_negative_samples)
                    negative_covered_positive_samples = negative_covered_positive_samples.intersection(
                        tmp_value_covered_positive_samples)
            all_negative_covered_negative_samples = all_negative_covered_negative_samples.union(
                negative_covered_negative_samples)
            all_negative_covered_positive_samples = all_negative_covered_positive_samples.union(
                negative_covered_positive_samples)

    for tmp_dim in positive_sub_Intention:
        tmp_value = positive_sub_Intention[tmp_dim]
        tmp_value_covered_specific_samples1 = None
        tmp_value_covered_samples1 = terms_covered_samples[tmp_dim][tmp_value]
        if sample_category == "positive":
            tmp_value_covered_specific_samples1 = tmp_value_covered_samples1[
                                                      "relevance"] - all_negative_covered_positive_samples
        elif sample_category == "negative":
            tmp_value_covered_specific_samples1 = tmp_value_covered_samples1["irrelevance"]
            a = len(tmp_value_covered_specific_samples1)
            # print("negative_covered_samples",all_negative_covered_samples)
            tmp_value_covered_specific_samples1 = tmp_value_covered_specific_samples1 - all_negative_covered_negative_samples
            # print("cah", a - len(tmp_value_covered_specific_samples1))
        if first_term:  # 某个概念可能本来覆盖的负样本就是0，因此不能通过result是否为空来判断
            result = result.union(tmp_value_covered_specific_samples1)
            first_term = False
        else:
            result = result.intersection(tmp_value_covered_specific_samples1)

    return result


def retrieve_docs_based_on_terms_covered_samples(sub_intention, terms_covered_samples, sample_category):
    result = set()
    first_term = True
    for tmp_dim in sub_intention:
        tmp_value = sub_intention[tmp_dim]
        tmp_value_covered_specific_samples = None
        tmp_value_covered_samples = terms_covered_samples[tmp_dim][tmp_value]
        if sample_category == "positive":
            tmp_value_covered_specific_samples = tmp_value_covered_samples["relevance"]
        elif sample_category == "negative":
            tmp_value_covered_specific_samples = tmp_value_covered_samples["irrelevance"]
        if first_term:  # 某个概念可能本来覆盖的负样本就是0，因此不能通过result是否为空来判断
            result = result.union(tmp_value_covered_specific_samples)
            first_term = False
        else:
            result = result.intersection(tmp_value_covered_specific_samples)
    return result


def retrieve_docs_by_complete_intention(intention, docs, ontologies, ontology_root):
    result = set()
    for sub_intention in intention:
        if "negative" not in sub_intention:
            result = result.union(retrieve_docs(sub_intention, docs, ontologies, ontology_root))
        else:
            result = result.union(retrieve_docs_negative(sub_intention, docs, ontologies, ontology_root))
    return result


def retrieve_docs_by_complete_intention_based_on_terms_covered_samples(intention,
                                                                       terms_covered_samples, sample_category):
    result = set()
    for sub_intention in intention:
        result = result.union(retrieve_docs_based_on_terms_covered_samples(sub_intention,
                                                                           terms_covered_samples, sample_category))
    return result


# 判断是否是互相包含的关系
# 格式：('Spatial', 'United States')
def is_inclusive(concept1, concept2, ancestors, ontology_root):
    is_concept_inclusive = False
    if concept1[0] is not None and concept1[0] == concept2[0]:
        # if concept1[1] in ancestors[concept1[0]][concept2[1]] or concept2[1] in ancestors[concept1[0]][concept1[1]]:
        if concept1[1] in ancestors[concept1[0]][concept2[1]]:
            is_concept_inclusive = True
    return is_concept_inclusive


# 得到意图的键形式
# 这里的intent是全意图
def get_intent_key(intent):
    #intent = [list(x.items()) for x in intent]
    positive_intent = [list(x["positive"].items()) for x in intent]
    for sub_intent in intent:
        sub_intent.sort()
    #intent.sort()
    positive_intent.sort()
    intent = str(intent)
    print(intent)
    return intent


if __name__ == "__main__":
    test_intents = [[{'d1': 'C27', 'd2': None}, {'d1': 'C26', 'd2': None}],
                    [{'d2': None, 'd1': 'C27'}, {'d1': 'C26', 'd2': None}],
                    [{'d1': 'C26', 'd2': None}, {'d1': 'C27', 'd2': None}],
                    [{'d1': 'C26', 'd2': None}, {'d2': None, 'd1': 'C27'}]]
    for test_intent in test_intents:
        print(get_intent_key(test_intent))
        print("[[('d1', 'C26'), ('d2', None)], [('d1', 'C27'), ('d2', None)]]" == get_intent_key(test_intent))
    print("Aye")
