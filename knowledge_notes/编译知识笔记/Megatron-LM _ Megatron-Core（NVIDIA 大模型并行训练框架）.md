## Megatron-LM / Megatron-Core（NVIDIA 大模型并行训练框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Megatron-LM 是 NVIDIA 开源的分布式大语言模型训练框架（https://github.com/NVIDIA/Megatron-LM，论文 arXiv:1909.08053；Megatron-Core 为可 pip install megatron-core 的模块化核心库），是 GPT-3、BLOOM、Llama 等 100B+ 预训练的事实标准实现之一。它首创/系统化组合五种正交并行策略（world_size = TP × PP × CP × EP × DP）：Tensor Parallelism（层内切分权重/激活，TP 组内 All-Reduce）、Pipeline Parallelism（层间切分，P2P Send/Recv，1F1B/交错虚拟 stage）、Data Parallelism（模型复制、batch 切分、梯度 All-Reduce）、Context Parallelism（长序列切分、Ring exchange/All-Gather KV）、Expert Parallelism（MoE 专家跨 GPU 分布、All-to-All token 路由）。MoE 支持含 --expert-model-parallel-size、num_moe_experts、moe_router_topk、aux_loss/sinkhorn 负载均衡、token dispatcher（allgather/alltoall/flex）、--overlap-moe-expert-parallel-comm 通信重叠开关。在 MoE-Hub 论文中，Megatron-LM 是全部软件基线的承载框架。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Megatron-LM 是"训练框架/运行时"，而非传统编译器：它把模型定义、并行策略选择与数据流编排组织成可分布式执行的图，由 PyTorch 张量计算图 + NCCL 集合通信驱动。MoE-Hub 论文的用法（baseline 承载）：Megatron-TE（Megatron-LM + NVIDIA Transformer Engine 的优化 transformer kernel）、FasterMoE、Tutel、Comet、CCFuser 五个 SOTA MoE 系统都在 Megatron-LM 之上实现，作为非重叠/图级重叠/细粒度重叠的对比基线；由于各基线共享同一 attention 模块实现，性能差异仅来自 MoE 层执行（含地址解析阶段如系统同步的开销）。流程例子：模型以 EP 切分专家 → 每层 forward 中 routing 打分 → 经 NCCL/通信库发 All-to-All dispatch → 专家 GEMM → All-to-All combine → 加权缩放；Megatron-LM 负责并行策略组装与 kernel 调度，MoE 通信库（DeepEP/Tutel 等）负责 dispatch/combine 实现。MoE-Hub 论文用它测出软件调度+暴露通信占 MoE 层 >24%、距理想层差距显著，作为硬件化方案的动机与对比对象。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
安装使用：git clone + pip 依赖，配置并行维度（--tensor-model-parallel-size、--pipeline-model-parallel-size、--expert-model-parallel-size 等）启动 pretrain 脚本；Megatron-Core 提供 TransformerConfig/ModelParallelConfig 等可组合配置。特色：activation recomputation、distributed optimizer、FlashAttention、FP8/FP4 混合精度（Hopper/Ada/Blackwell）、多 token 预测（MTP）、checkpoint 转换工具（与 HuggingFace 互转）。生态：HuggingFace Accelerate 提供 Megatron-LM 插件，verl 等 RL 框架基于 MegatronEngine。局限：主要面向 NVIDIA GPU 与训练场景，学习曲线陡。MoE-Hub 语境：论文在 Megatron-LM 上实现各软件基线做公平对比，而其提出的硬件方案（st.rowsp + hub 模块）独立于框架——软件侧只增加少量指令/API 即可接入（调度 0 行、通信 <10 条），可移植性好。

涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
