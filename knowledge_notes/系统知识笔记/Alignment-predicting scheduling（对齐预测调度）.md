## Alignment-predicting scheduling（对齐预测调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-sys 的调度策略：对 GPU 与 CHIME-PIM 两侧的执行延迟分别建模预测，选择使两子批预测延迟对齐（T_GPU ≈ T_PIM）的请求组合，最大化并行执行、最小化空闲气泡。预测模型：GPU 侧用 Random Forest Regression（RFR：增量学习、低延迟、高精度，预测 t_p 与 t_batch）；PIM 侧用线性模型（CHIME-PIM 无干扰、执行时间与计算/传输 token 数线性相关，t_d 取最慢 rank、t_comm 线性）。运行时 profiling 收集 (batch 信息, 延迟) 数据增量更新模型，预测相对误差 <~1%（中位数 <0.5%）。可行性的三个前提：CHIME-PIM 执行无干扰、影响 batch 性能的因素（batch 大小、已处理 token 数）事先已知、GPU 延迟预测有大量先例（Clockwork/INFless 等）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
调度步骤：1) 每个 sub-batch 先加 1 个 prefilling 请求（无则跳过）；2) 每加一个 prefill 请求（T_GPU 增大），向每子批加 N 个 decoding 请求并负载均衡分配到各 rank，预测 T_PIM 与 T_GPU，若 T_PIM < T_GPU 继续加 N 个 decoding 直到 T_PIM > T_GPU；3) 有剩余 prefill 则重复直到 PIM 显存耗尽；显存饱和且气泡在 PIM 侧时，把每子批最后一个 prefill 请求 chunk 化（chunked prefill），动态微调 T_GPU 逼近另一子批的 T_PIM。N 决定调节粒度与 rank 负载均衡的权衡：MHA 头多、可均匀分布到 chips，N=1；GQA N=16。效果（OpenR1 trace）：TBT 最高降 70.93% 且吞吐不降略升；对比"优先填满 PIM 容量"的 baseline，在 MHA + 增大容量时 baseline 选更大 batch 导致 TBT 膨胀而无吞吐收益，CHIME 避免气泡并抑制 TBT 增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器集成 RFR（sklearn 类实现即可）+ 线性回归 + 运行时 profiling 数据集（8:2 划分训练/测试）；模型预测作为组批决策输入。使用方式：面向 AFD 等跨设备并行推理系统；CPU 卸载类系统（NEO）因 CPU 受干扰、缺乏操作级延迟建模而无法对齐，HBM-PIM 类系统（NeuPIMs/AttAcc）PIM 侧延迟恒小于 GPU 侧也难对齐——CHIME 的可预测 PIM 侧 + 建模预测是其适用前提。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
