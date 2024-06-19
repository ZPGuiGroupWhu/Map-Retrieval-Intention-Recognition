# 基于超图分割的地图检索意图识别

* ### 项目介绍
  Python3实现的基于超图分割的地图检索意图识别。
----
* ### 代码目录及文件介绍
```
|-- Hypergraph Partitioning
    |-- New_HyperGraphCut    // 代码主体
        |-- MDL_RM    // 对比算法代码
        |-- results    // 实验结果
        |-- SweetData    // 本体库数据

        |-- 实验代码
        |   |-- Exp_argminC.py       // CC有效性
        |   |-- Exp_argminC_k.py             // CC自适应k与外部输入k对比
        |   |-- Exp_MDL_RM.py                     // 对比MDL_RM
        |   |-- Exp_Rule_Go.py                     // 对比Rule_Go

        |   |-- Exp_SampleGraph.py           // 超图对比简单图
        |   |-- Exp_NoOntology.py            // 有本体对比无本体
        |   |-- Exp_Ontology.py               // 有本体实验
        
        |   |-- Exp_Para_cc.py          // 参数分析cc
        |   |-- Exp_Para_mu.py                   // 参数分析mu（单意图阈值）
        |   |-- Exp_Para_num.py       // 参数分析，样本数量

        |-- 算法代码
        |   |-- FileUtil.py   // 文件读写
        |   |-- GenerateGeoNames.py    // 本体类GeoNames
        |   |-- GenerateSweet.py   // 本体类Sweet
        |   |-- HyperGraph5.py  
