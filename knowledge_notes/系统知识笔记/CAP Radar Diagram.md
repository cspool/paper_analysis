## CAP Radar Diagram

术语是什么？
CAP雷达图（CAP Radar Diagram）是MoE-CAP提出的三维可视化工具，用于在同一图中对比不同MoE系统方案在Cost（成本）、Accuracy（准确率）、Performance（性能）三个维度的权衡。三个维度分别归一化后绘制在雷达图的三个轴上：Cost轴综合硬件购买成本和能耗成本（C_token = (C_hardware + C_energy × $/kWh) / (T_token × R)），Accuracy轴使用下游任务指标（exact match/F1/win rate），Performance轴根据部署场景可选TPOT（在线推理）或吞吐（离线批处理）。CAP雷达图基于观察"MoE系统通常仅优化CAP三维中的两个、牺牲第三个"，将系统分为三类：PA型（Performance-Accuracy，如SGLang/vLLM）、PC型（Performance-Cost，如K-Transformers量化）、CA型（Cost-Accuracy，如MoE-Infinity offloading）。

从系统架构角度拆解术语：
CAP雷达图的使用流程：(1) 选择目标MoE模型和硬件平台；(2) 对每个候选系统方案运行MoE-CAP自动化评测（加载模型→运行benchmark→采集profiling数据）；(3) 从profiling数据计算Cost维度（C_hardware公式覆盖GPU/CPU/Motherboard/DRAM/SSD全部异构资源 + C_energy公式覆盖GPU/CPU/PCIe/NVLink/C2M通信功耗）、Accuracy维度（在MMLU/GSM8K/MATH/Arena-Hard/LongBench上的得分）、Performance维度（TPOT或吞吐或S-MBU/S-MFU）；(4) 三个维度分别min-max归一化到[0,1]；(5) 在三角雷达图上绘制，面积/形状直观反映系统偏向。最终结合多约束决策矩阵（表7），根据hardware tier、batch size、primary/secondary constraint直接推荐系统。

术语一般如何实现？
MoE-CAP提供Docker镜像+FastAPI服务实现CAP雷达图：POST /cap-profiler在每次forward pass采集数据，GET /cap-results生成最终报告含雷达图。支持6种serving框架+8+模型+5种benchmark的自动化组合评测。CAP雷达图不仅对比系统，还对比方法（如量化vs offloading：SGLang-FP8 vs SGLang-AWQ vs MoE-Infinity在Qwen3-235B-A22B上的功耗-吞吐-准确率权衡）。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
