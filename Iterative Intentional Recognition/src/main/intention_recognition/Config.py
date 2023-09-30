"""
   脚本描述：
   Apriori_MDL 算法参数设置，
   RuleGO 算法参数设置（暂时用不到）
"""


class Config:

    def __init__(self):
        # for Apriori_MDL
        self.adjust_sample_num = True  # 是否调整正负样本数量
        self.rule_covered_positive_sample_rate_threshold = 0.4
        self.record_iteration_progress = False
        self.beam_width = 1

        # for RuleGO
        self.RuleGO_statistical_significance_level = 0.05  # 规则的统计显著性阈值
        self.RuleGO_min_support = 40  # 最小支持度，调优得到40
        self.RuleGO_max_term_num_in_rule = 4  # 一个规则包含的词项的最大值
        self.RuleGO_rule_similarity_threshold = 0.5  # 意图的相似度阈值
        self.RuleGO_use_similarity_criteria = False  # 是否基于意图相似度保证意图多样性

    def to_json(self, method):
        result = {}
        if method.startswith("Apriori_MDL"):
            result = {
                "adjust_sample_num": self.adjust_sample_num,
                "rule_covered_positive_sample_rate_threshold": self.rule_covered_positive_sample_rate_threshold,
                "record_iteration_progress": self.record_iteration_progress,
                "beam_width": self.beam_width
            }
        elif method == "RuleGO":
            result = {
                "statistical_significance_level": self.RuleGO_statistical_significance_level,
                "min_support": self.RuleGO_min_support,
                "max_term_num_in_rule": self.RuleGO_max_term_num_in_rule,
                "rule_similarity_threshold": self.RuleGO_rule_similarity_threshold,
                "use_similarity_criteria": self.RuleGO_use_similarity_criteria
            }
        return result
