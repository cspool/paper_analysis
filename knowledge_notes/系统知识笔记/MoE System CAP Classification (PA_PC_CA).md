## MoE System CAP Classification (PA/PC/CA)

术语是什么？
MoE-CAP根据MoE系统在Cost-Accuracy-Performance三维度中的优化倾向，将所有MoE系统分为三种类型：(1) PA型（Performance-Accuracy优化）：优先保证低延迟和高准确率，以高成本为代价，典型如SGLang/vLLM使用高端GPU（H200 NVL 141GB, MI300X 192GB）和Expert/Tensor/Pipeline Parallelism扩展内存容量和计算；(2) PC型（Performance-Cost优化）：通过量化（FP8/INT4等低精度）或稀疏注意力压缩模型以提升吞吐和降低成本，但以精度损失为代价，典型如K-Transformers、SGLang-FP8/AWQ；(3) CA型（Cost-Accuracy优化）：在有限硬件预算下通过expert offloading（MoE-Infinity将冷expert卸载到CPU DRAM/SSD）或CPU辅助GPU（Fiddler用CPU处理低负载expert）保持精度，但以更高延迟为代价。

从系统架构角度拆解术语：
CAP三维分类指导实际部署的系统选型：
- PA型（高成本高性能高精度）：适合延迟敏感的在线推理（如Chain-of-Thought推理），batch size > 8，硬件为数据中心GPU（H100/H200）。技术路径：多GPU TP+EP+PP并行，NVLink/NVSwitch/InfiniBand互联。CAP雷达图示例：SGLang在Qwen3-30B-A3B上解码延迟0.058s/token + 91.1% exact match，但购买成本最高。
- PC型（低成本高性能可接受精度损失）：适合高吞吐批处理（如文档摘要、embedding提取），batch size > 16，硬件为数据中心GPU（H20）。技术路径：FP8/INT4量化 + 优化kernel（Marlin, Ladder）。CAP雷达图示例：SGLang-FP8在Qwen3-235B-A22B上吞吐>75× vs MoE-Infinity, 92.2% accuracy, 2.7× power cost。
- CA型（低成本高精度可接受延迟）：适合单用户/小batch精度敏感场景（如模型评测），batch size 1-8，硬件为工作站/消费级GPU（A5000/A6000/RTX 4090）。技术路径：expert offloading到CPU DRAM/SSD + on-demand loading。CAP雷达图示例：MoE-Infinity在Qwen3-30B-A3B上91.1% accuracy, 60% cost saving, 2.6× latency vs SGLang。

术语一般如何实现？
MoE-CAP通过CAP雷达图和多约束决策矩阵（表7）形式化这一分类：根据Hardware Tier（Workstation GPU/Datacentre GPU）、Batch Size、Primary Constraint（Performance/Cost/Accuracy）和Secondary Constraint，映射到推荐的MoE System（SGLang/vLLM/K-Transformers/MoE-Infinity）和配置（FP16/FP8/AWQ/Expert Offloading）。压力测试显示PA型系统（SGLang连续batching）在batch-size骤增时S-MBU峰值54.5%但饱和后因token eviction急剧下降，CA型系统（MoE-Infinity固定batch）利用更稳定但较低（16.7%）。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
