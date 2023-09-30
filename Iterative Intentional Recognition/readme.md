### 基于引导式迭代反馈的地图检索意图更新算法
#### 项目介绍
  Python3在单轮基于最小描述长度准则（(Minimum Description Length Principle, MDL)的意图识别算法基础上，实现基于引导式迭代反馈的地图检索意图更新算法 

#### 代码目录及文件介绍
```
├─resources                                           // 包含本体与反馈数据
│  ├─ontologies                                       
│  │  ├─geonames_america                              // 空间维度的Geonames本体
│  │  │      all_adm1.csv                             // Geonames所有地名
│  │  │      all_america_country.csv                  // 美洲的所有国家（以下json皆统计至美洲一级（省级）行政区）
│  │  │      america_all_ancestors.json               // 所有上位概念
│  │  │      america_all_hyponyms.json                // 所有下位概念
│  │  │      america_direct_ancestors.json            // 直接上位概念
│  │  │      america_direct_hyponyms.json             // 直接下位概念
│  │  │      america_information_content.json         // 地名概念信息量
│  │  │      america_neighbors.json                   // 直接上位、下位概念
│  │  │      america_peers.json                       // 有共同直接祖先的概念（同级概念）
│  │  │      concept_leaves_num.json                  // 每个概念的叶子节点数量
│  │  │      concept_max_depth.json                   // 每个概念的最大深度
│  │  │      ontology_statistics.json                 // 统计信息
│  │  │
│  │  └─sweet                                         // 地理要素维度的SWEET本体 
│  │          all_ancestors.json                      // 所有上位概念
│  │          all_hyponyms.json                       // 所有下位概念
│  │          concept_information_content_yuan2013.json // 概念信息量
│  │          direct_ancestors.json                   // 直接上位概念
│  │          direct_hyponyms.json                    // 直接下位概念
│  │          neighbors.json                          // 直接上位、下位概念
│  │          peers.json                              // 有共同直接祖先的概念（同级概念）
│  │
│  └─samples                                          // 反馈数据
│          ├─scenes_v4_10                             // 24种初始反馈样本数据
│          all_dimension_concept_distance.json        // 两个概念间的最短距离（以概念全称的方式存储）
│          all_dimension_concept_distance_id.json     // 两个概念间的最短距离（以概念ID的方式存储）
│          all_dimension_concept_similarity.json      // 两个概念间的相似度（以概念全称的方式存储）
│          all_dimension_concept_similarity_id.json   // 两个概念间的相似度（以概念ID的方式存储）
│          all_samples.json                           // 生成的合成样本集合
│          concept_id_dict.json                       // 概念全称到概念ID的映射（用于加速检索）
│          Intention_v1.6.json
│          multiple_label_samples.json                // 生成的多标签样本（样本的至少一个维度有多个取值）
│          single_label_samples.json                  // 生成的单标签样本（样本的每个维度只有一个取值）
│          target_dim_concepts.json                   // 预定义意图下每个维度的目标概念
│          各场景具体意图v1.6.csv                       // 预定义24种意图识别场景
│          各场景具体意图v1.6.xlsx
│
├─result                                                     // 实验结果
│  └─Iterative_Feedback
│      └─scenes_v4_10_Intention_20230403
│          ├─guide_feedback_analysis                         // 引导反馈有效性的分析数据（用于统计绘图）
│          │
│          ├─guide_feedback_comparison_result                // 引导反馈有效性实验
│          │                                                 // 比较引导反馈、直接反馈、与相似度反馈在准确程度、时间消耗、用户意图可表达的范围方面的效果
│          ├─history_information_effectiveness_analysis      // 历史信息有效性的分析数据（用于统计绘图）
│          │
│          ├─history_information_effectiveness_result        // 历史信息有效实验
│          │                                                 // 比较使用所有历史反馈数据、仅使用上轮反馈数据、不使用历史反馈数据在意图识别准确性与时间消耗方面的效果
│          ├─intention_shift_detector_analysis_result        // 偏移检测可靠性的分析数据（用于统计绘图）
│          │                                                 // 偏移检测可靠性实验
│          └─intention_shift_detector_comparison_result      // 比较3种偏移检测方案在成功率与耗时情况
│
└─src                                                        // 核心代码
    ├─main
    │  │  Version.py                                         // 版本编号
    │  │
    │  ├─experience
    │  │      Apriori_MDL_Comparison_Experience.py                  // 单轮意图识别算法（前置实验）
    │  │      Guide_Feedback_Comparison_Experience.py               // 引导反馈有效性实验（与直接反馈、相似度反馈对比）
    │  │      History_Information_Effectiveness_Experience.py       // 历史反馈数据有效性实验（是否使用历史信息）
    │  │      Intention_Shift_Detector_Comparison_Experience.py     // 意图偏移判断准则有效性实验（三个准则）
    │  │      Iterative_Result_Analysis.py                          // 结果数据整合和处理（便于画图）
    │  │
    │  ├─intention_recognition
    │  │  │  Apriori_MDL.py                                // MDL核心方法（频繁项集挖掘 + 最小描述长度寻找最优意图 + 意图偏移判断准则）
    │  │  │  Config.py                                     // MDL参数配置
    │  │  │  EvaluationIndex.py                            // 意图识别结果外部评价（Jaccard、BMASS、Precision、Recall、F1计算)
    │  │  │  FeedbackSimulation.py                         // 用户反馈模拟器，模拟用户的迭代反馈行为
    │  │  │  GuideFeedback.py                              // 构建引导式反馈样本集合（同时实现了直接反馈和相似度反馈作为对比方法）
    │  │  │  IntentionConf.py                              // 意图识别结果内部评价（意图及子意图的置信度计算）
    │  │  └─ IntentionShift.py                             // 意图偏移判断准则（压缩率、意图覆盖率、子意图覆盖率）
    │  │
    │  ├─samples                                           
    │  │  ├─generation                                     // 生成合成数据
    │  │  │  │  ConceptIdTransform.py                      // 本体概念映射
    │  │  │  └─ GenerateSamples2.py                        // 合成数据生成器
    │  │  │
    │  │  └─input
    │  │      │  Data.py                                   // 初始化反馈数据
    │  │      │  DimensionValues.py                        // 各维度本体
    │  │      │  GeoNamesAmerica.py                        // 读取GeoNames本体
    │  │      │  MapMethodVirtualOntology.py               // 生成MapMethod虚拟本体
    │  │      │  Sample.py                                 // 读取反馈样本数据
    │  │      │  SWEET.py                                  // 读取SWEET本体
    │  │      └─ ThemeVirtualOntology.py                   // 生成Theme虚拟本体
    │  │
    │  ├─util
    │  │  │  FileUtil.py                                   // 文件读取接口
    │  │  │  OntologyUtil.py                               // 本体概念各种度量指标的计算接口
    │  │  └─ RetrievalUtil.py                              // 意图与样本关系判断接口
    │  └─
    │
    └─test
            Apriori_MDL_Test.py                            // MDL运行测试
            SemanticApriori_Test.py                        // Apriori频繁项集挖掘测试
            TestOntologyUtil.py                            // 本体概念与关系查询测试
```