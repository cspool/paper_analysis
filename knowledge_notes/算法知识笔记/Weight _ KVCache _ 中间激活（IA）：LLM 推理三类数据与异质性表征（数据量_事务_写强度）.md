## Weight / KVCache / 中间激活（IA）：LLM 推理三类数据与异质性表征（数据量/事务/写强度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LLM 自回归推理涉及三类数据：Weight（静态权重矩阵）、KVCache（QKV Generation 层在 decode 每步动态写入的键值缓存，避免重算）、中间激活 IA（层间中间张量）。三者异质性显著：数据量上 Weight 与 KVCache 主导（$D_{Weight}=12d^2L/(p_p p_t)$，$D_{KVCache}=2dsbL/p_t$，d=隐藏维、L=block 数、b=微批、s=序列长、p_p/p_t=流水/张量并行），IA 峰值执行足迹小几个量级（Fig. 4a 跨 GQA/MoE 结构不变）；事务上 Weight/KVCache 读主导（体积大且反复重载）、KVCache 写最少（每 entry 只写一次）、prefill 主导时读下降而长 prompt 的 IA 流量大（Fig. 4b）；写强度 = 写事务/容量/时间，反映 cell 访问频率（Fig. 4c 上界=集中写、下界=理想磨损均衡）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 GPT3-175B（MHA+Dense，d=12288、L=96）微批 64 为例：prefill 处理 prompt token（GEMM，Weight 从 NVM 读、IA 在 DRAM 计算/写回），decode 逐 token 生成（GEMV，从 KVCache 读历史 KV、新 KV 写入 KVCache）；KVCache 随序列/批增长主导内存。三类数据按度量偏好映射到混合内存：Weight/KVCache = 容量+读密集、写稀疏 → NVM；IA = 写密集、容量小 → DRAM（KVCache 溢出放 DRAM）。GQA（Mixtral）KV head 少 → KVCache 写事务进一步被抑制；MoE 粗粒度专家激活 → Weight 大块连续读，保持 NVM 读带宽利用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在算法/系统层面，该表征指导静态数据放置（SHyLA 的 runtime 初始化时按类别放置）与混合内存设计（NVM 面积优先给 Weight 操作、DRAM 保留最小容量给 IA + 其余转带宽）。量化/压缩（INT8 Weight）与分类放置正交。三类数据的体积公式、事务分布与写强度图（Fig. 4）为后续研究提供了"workload–hardware 接口"的数据侧模型。SHyLA 数据（仿真按 Sec. VII-A 方法学，微批 64）论文未开源（联网未找到）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
