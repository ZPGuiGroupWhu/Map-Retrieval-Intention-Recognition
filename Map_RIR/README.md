# 综合用户正负偏好的地图检索意图识别(Map_RIR)

* ### 项目介绍
  Python3实现的基于最小描述准则（MDL）和Apriori算法的意图识别算法Map_RIR。该算法顾及地理语义，从反馈样本中识别综合用户正负偏好的地图检索意图。
----
* ### 代码目录及文件介绍
```
|-- src
    |-- main    // 代码主体
        |-- experience    // 实验代码
        |   |-- ItemsMerger_Apriori_Experience.py    // Apriori算法的有效性分析
        |   |-- Comparison Experience.py             // 与其他方法的对比实验
        |   |-- Evaluation_IM.py                     // 评价指标计算
        |   |-- Feasibility_Experience.py            // 算法有效性验证
        |   |-- Negative_Experience.py               // 负向偏好表达的有效性验证
        |   |-- Parameter_Sensibility_Experience.py  // 参数影响分析
        |   |-- Result_Analysis.py                   // 实验结果处理分析
        |   |-- Sample_Adjusting_Experience.py       // 样本增强策略有效性

        |-- intention_recognition
        |   |-- Config.py   // 参数设置
        |   |-- DTHF_Kinnunen2018.py    // 基准算法,基于决策树的DTHF算法
        |   |-- Filter_without_FIM.py   // 不使用Apriori挖掘正负偏好的方法
        |   |-- MDL_RM.py               // 基础算法MDL-RM
        |   |-- RuleGO_Gruca2017.py     // 基准算法,基于频繁项集挖掘的RuleGO算法
        |   |-- FrequentItemsets.py     // Apriori算法挖掘正负偏好
        |   |-- IM_Details.py           // 详细记录算法细节
        |   |-- IM_greedy.py            // 使用贪心算法搜索意图的方法
        |   |-- Run_Map_RIR.py          // 本项目算法Map_RIR
        |-- samples   // 样本处理代码
        |   |-- generation    // 样本生成
        |   |-- input   // 样本输入与预处理
        |-- util    // 文件输入输出工具   
|-- resources
    |-- samples   // 预定义意图对应的样本集,各样本集对应意图见该目录下的"各场景具体意图7.30.csv"
    |-- ontologies    // 本体概念从属关系,概念信息量等数据
|-- result    // 实验结果
```
----
* ### 项目依赖
  * 数据处理: numpy, pandas
  * 结果绘图: matplotlib, seaborn
  * Excel输出: openpyxl
----
