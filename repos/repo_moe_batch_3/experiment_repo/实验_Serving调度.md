# 实验_Serving调度

## Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Aurora 系统，通过三个维度的联合优化来最小化 MoE 推理时间：(1) **通信调度**：为 all-to-all 通信中的 token 传输确定最优顺序，避免接收端 GPU 的带宽竞争（Theorem 4.2, Alg.1）；(2) **专家共置**：将来自**不同模型**的专家放置在同一 GPU 上（而非同模型的多专家），使计算和通信可以完全交错，打破同步 all-to-all 通信限制；(3) **GPU 分配**：在异构集群中将热门专家分配到高性能 GPU（Theorem 5.1）。
  - 四种场景的理论分析：Exclusive+Homogeneous（§4）、Exclusive+Heterogeneous（§5）、Colocating+Homogeneous（§6，转化为瓶颈匹配问题求解）、Colocating+Heterogeneous（§7，3维匹配 NP-hard，通过解耦为两个二分图匹配得到次优解，仅偏离最优 1.07×）。
  - 实验比较：(1) 四种场景下的推理时间对比（vs. SJF、RCS、RGA、Lina、REC）；(2) Colocating 场景下的 GPU 利用率对比（vs. Lina）；(3) Colocating+Heterogeneous 场景下与暴力搜索最优解的差距；(4) 不精确 traffic 输入下（0%-75% 噪声）的性能鲁棒性。

- 硬件平台是什么，配置是什么。
  - 模拟环境（仿真评估）而非真实硬件。
  - 同构集群：网络带宽 100 Gbps。
  - 异构集群：4 种 GPU 类型，带宽分别为 100 Gbps、80 Gbps、50 Gbps、40 Gbps（从高到低性能排列），各类型 GPU 数量相同。
  - 所有 GPU 通过 big switch 模型（无阻塞网络）互联，如 Fig. 4(a) 所示。

- 开源Serving框架是什么。修改了什么。
  - Aurora 不基于现有开源 serving 框架，而是提出了一套理论驱动的优化方法。实现方式为仿真模拟。
  - 修改/设计内容：
    1. **通信调度算法（Alg. 1）**：基于 traffic matrix D，识别瓶颈 GPU（最大流量），确定 token 传输顺序以避免接收端带宽竞争。核心原理：通过添加非负人工 traffic matrix X 将原始 traffic matrix 转换为每行/列均为 b_max 的规整矩阵，再用 Farkas' Lemma 证明 X 的存在性，因此通信时间可压缩至 b_max。
    2. **GPU 分配策略（Theorem 5.1）**：在异构集群中按 expert 处理的 token 数量降序排列，从高到低性能 GPU 分配。
    3. **专家共置策略（§6.2）**：Case I（每 GPU 发送=接收流量）：交替选择热门和冷门 expert（Theorem 6.2）；Case II（发送≠接收）：转化为瓶颈匹配问题，使用二分搜索 + Hopcroft-Karp 算法（复杂度 O(n²√n log n)）。
  - 通信调度可通过在计算操作的 buffer 层调用 NCCL 等通信集体库按所需顺序实现（论文 §3）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未明确说明开源情况。Aurora 为模拟评估，未提及开源代码仓库或开源 link。
  - Aurora 的输入到执行全过程（以 Colocating+Heterogeneous 场景为例）：
    1. **输入**：两个 MoE 模型（各含 n 个 expert）的历史统计信息，包括 traffic matrix D_N（第一个 all-to-all 通信的 token 分布）、D_C（第二个 all-to-all 通信）、以及 Gate/FFN/Aggregation 在各 GPU 上的计算时间。
    2. **优化阶段**：Aurora 接收输入后，依次求解：(a) 专家共置——通过瓶颈匹配将 Model a 和 Model b 的 expert 配对，最小化聚合通信时间的最大列/行和；(b) GPU 分配——将共置后的 expert 对按 token 负载降序分配给高性能 GPU；(c) 通信调度——为每个 GPU 确定 token 传输顺序，确保任何时刻各 GPU 只从单一源接收数据。
    3. **推理执行**：每个 MoE layer 上，Gate 网络计算后触发第一个 all-to-all 通信（按 Aurora 的调度顺序发送 token）→ 各 GPU 上的 FFN 处理到达的 token → 第二个 all-to-all 通信（反向传输）→ Aggregation。两个模型的计算和通信因 expert 来自不同模型而完全交错：当 Model a 做 FFN 计算时，Model b 可同时进行 all-to-all 通信。
    4. **输出**：推理时间 t = E_{A^b} + |G^a|（Eqn. 4），GPU 利用率 = 计算时间/推理时间。

## Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：在 AWS Lambda（CPU serverless 平台）上构建完整的 MoE 模型分布式推理服务系统，包含：(1) 基于贝叶斯决策的专家选择预测器（使用 token ID、position ID、attention ID 三维特征）；(2) 三种 serverless 通信方法（pipelined indirect transfer、non-pipelined indirect transfer、direct transfer）的 scatter-gather 通信设计；(3) 基于 MIQCP 求解的最优部署选择（ODS）算法，联合决策通信方法、内存配置、专家函数副本数；(4) 多维度 ε-greedy search 的贝叶斯优化框架进行全局优化。
  - 实验比较：(1) 不同 scatter-gather 通信方法在不同 token 数下的 billed cost 和吞吐量；(2) ODS vs. MIQCP 直接求解 vs. 随机选择，在不同吞吐量目标下的 billed cost；(3) BO 框架使用不同 acquisition function（multi-dim ε-GS vs. single ε-greedy vs. TPE vs. random vs. no BO）的 billed cost ratio 和专家预测差异；(4) 最终整体性能：Serverless+BO vs. Serverless+real distribution vs. Serverless no BO vs. LambdaML vs. CPU cluster vs. CPU betterTransformer 的 billed cost 和吞吐量。

- 硬件平台是什么，配置是什么。
  - AWS Lambda（CPU-based serverless 平台），14 档内存配置 [128, 768, 960, 1152, 1344, 1536, 1728, 1920, 2112, 2304, 2496, 2688, 2880, 3072] MB。
  - CPU cluster baseline：2×64-core AMD EPYC CPUs，512GB DRAM。
  - 外部存储：2 个 S3 bucket，各 512MB。
  - 最大专家副本数：G=8。

- 开源Serving框架是什么。修改了什么。
  - 论文未基于开源 serving 框架，而是直接在 AWS Lambda 上构建自定义 serverless MoE 推理系统。
  - 使用 PyTorch + transformers 构建 MoE 模型，Optuna 实现 BO 算法，Gurobi 求解 MIQCP 问题。
  - 修改/设计内容：
    1. 设计了三种 scatter-gather 通信方法（pipelined indirect via S3、non-pipelined indirect via S3、direct function invocation），支持按 MoE 层混合选择。
    2. 实现了基于 Bayes 定理的专家选择预测器，使用 token ID、position ID、attention ID 三个特征计算后验概率。
    3. 实现了 ODS 算法：将 MIQCP 分解为三种通信方法各求解一次，然后逐层选择最低 cost 方法，若不满足延迟约束则迭代替换最高延迟层。
    4. 实现了带 feedback processor 的 BO 框架，根据实际 billed cost 反馈调整 key-value dataset table，优化专家预测准确性。
  - 未修改：MoE 模型权重、gating network 路由决策本身。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供独立开源仓库（无公开代码链接）。代码实现基于 PyTorch、Optuna、Gurobi 在 AWS Lambda 上构建。
  - 全过程示例（GPT2 MoE，10240 tokens，Enwik8 dataset，text generation task）：
    1. **离线 Profiling**：在 ≥100 个样本上运行 MoE 推理，记录每个 (token_id, position_id, attention_id) → expert_index 映射的出现频次，构建 key-value dataset table Ω。
    2. **BO 训练循环（~20 iterations）**：
       a. **Expert Selection Predictor**：对每个新 token，提取 token ID f1（已知）、position ID f2（均匀分布）、attention ID f3（用最高 attention score 对应的 token ID 近似），通过两重积分计算后验概率 P(N_{e,i}|f1') = ∫∫ P*(N_{e,i}|f1',f2,f3) · P*(f1',f2,f3)·P'(f3)/P*(f1',f2) · P*(f1',f2)·P'(f2)/P*(f1') df3 df2；取 argmax 得到预测专家。
       b. **Policy Maker**：将三组固定通信方法的 MIQCP 送入 Gurobi 求解（≤60s）；ODS 算法逐层选择最低 cost 通信方法，若不满足 end-to-end latency limit T_limit 则迭代替换最高延迟层。
       c. **部署执行**：各 expert 函数按配置的内存大小（如热门 expert 3008MB、冷门 128MB）和副本数部署到 AWS Lambda；模型参数从 S3 加载到各函数；gating 网络函数按所选通信方法（如 a^e=1 pipelined indirect）将 token minibatch 写入 S3，expert 函数从 S3 下载并计算，结果写回 S3，下一非 MoE 层从 S3 下载聚合。
       d. **Feedback**：收集 J 个 batch 的实际 billed cost c_τ = (1/J)Σc_{τ,j}，更新 BO 历史集 B_τ；多维度 ε-GS 调整 key-value table：低性能 key-value pairs 在 limited range L 内加大探索（ε 衰减更慢），其余在正常 range P 内探索。
    3. **收敛后部署**：BO 收敛后的最优配置部署 MoE 模型，开始服务真实推理请求。

## Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：在 SGLang 推理框架中集成 OEA 路由算法，修改 MoE 层的 decode 阶段路由逻辑：仅在 decode 阶段使用 OEA（不在 prefill 阶段使用），根据 batch 内 token 的 router score 动态决定每个 token 的激活专家集合，最小化 batch 内唯一激活专家数 T。
  - 实验比较：(1) 不同 k0 配置（3/4/5/6/7）下的 MoE 层平均延迟（微秒），对比 vanilla top-8 routing；(2) 平均激活专家数 vs. vanilla；(3) Qwen3-30B 和 Qwen3-235B 两个模型规模下的延迟降低比例（39% 和 15%）；(4) 发现并修复 SGLang 的 CUDA Graph padding 问题：padding token 会激活额外专家导致反向性能损失，解决方案是捕获 CUDA Graph 到 batch size 16（覆盖所有实际 batch size，消除 padding）。

- 硬件平台是什么，配置是什么。
  - Qwen3-30B-A3B：单卡 NVIDIA H100 80GB，bfloat16。
  - Qwen3-235B-A22B：8×H100 80GB，单节点 HGX H100，NVSwitch 互联，tensor parallelism degree=8。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang（Zheng et al., 2024）。
  - 修改内容：
    1. 在 MoE 层的 decode 调用路径中插入 OEA 路由逻辑（替换默认的 top-k 路由）。
    2. 仅 decode 阶段使用 OEA，prefill 阶段保持原始 top-k 路由。
    3. 使用 `--max-running-requests` 限制最大 batch size 为 16（因 KV cache 限制无法到 32）。
    4. 捕获 CUDA Graph 到 batch size 16，避免 SGLang 对不足 batch size 的 padding 行为引入额外专家激活。
  - 未修改：MoE 权重本身、Grouped GEMM kernel、KV cache 管理、prefill 路由。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供独立开源仓库。OEA 路由通过修改 SGLang 的 MoE 层 forward 实现。
  - 全过程示例（Qwen3-30B，batch size=16，k0=5）：
    1. **输入**：SGLang 调度器将 16 个请求的 decode step token 组成 batch，传入 Qwen3-30B 的第 l 层 MoE 模块。
    2. **Router 评分**：对每个 token x_i，路由器 R 输出 N=128 个归一化分数 R(x_i)，排序得到 e_{i,1..128}。
    3. **OEA Phase 1**：每 token 取 top-5 专家作为 baseline → S_i_base，得到 S_base = union_i S_i_base（约 35 个唯一专家）。
    4. **OEA Phase 2**：每 token 遍历其 6-8 位排名的专家，若在 S_base 中则附加（piggybacking），最终每 token 仍 ≤8 个专家，但 T = |S_base| ≈ 35（vs. vanilla 约 48）。
    5. **权重加载**：仅加载 T=35 个专家的权重从 HBM→SRAM（vanilla 需加载 ~48 个），节省约 27% 的 memory fetch 开销。
    6. **计算**：Grouped GEMM 对 35 个专家的权重和对应 tokens 做批量矩阵乘法，输出 shape=(B, D)。
    7. **输出**：路由加权的专家输出求和，送入下一层 transformer block。
    8. **延迟效果**：MoE 层平均延迟从 175.7μs（vanilla）降至 136.0μs（k0=5），降低 23%；k0=3 时降至 106.8μs，降低 39%。

## MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：一个基于 Integer Linear Programming (ILP) 的 MoE 专家放置优化框架，包含三阶段：(1) **Token Routing Profiling**——在任务数据集的采样子集上运行推理，记录逐层的 token-to-expert 路由统计（每个 expert 的 token 处理量 P_{e,l}，以及跨层 expert 对间的 token 路由量 R_{e_1,e_2,l}）；(2) **ILP 优化**——分两步求解：ILP 1 按层将 experts 聚类到 G 个 cluster，最小化各 cluster token 处理负载与层内均值的偏差（min Σ|T_{c,l} - T̄_l|）；ILP 2 将 cluster 分配到 GPU，最小化跨 GPU 通信成本的层间最大值（min Σ max(C_{c_1,c_2,l} / B_{g_1,g_2})），同时保证每个 GPU 的 expert 数量均衡（Σ x_{c,e,l}·y_{c,g,l} = E·L/G）；(3) **Custom Expert Parallelism Initialization**——将 ILP 输出的 expert-to-GPU 映射保存为 PyTorch tensor 文件，初始化模型时替换 Megatron-LM 的默认 contiguous block 放置。
  - 实验比较：(1) End-to-End Speedup：MoETuner vs Megatron-LM default expert placement，单节点（8×H100, 4EP-2TP）和多节点（16×H200, 4EP-4TP）下的推理加速比；(2) Token Processing Time：每层的 token 处理尾延迟（tail latency，最长 GPU 处理时间）和平均延迟；(3) All-to-All Time：每层的 all-to-all 通信尾延迟和平均延迟；(4) Token processing load distribution（箱线图）和 Token dispatching distribution（GPU pair 间 token 分发量箱线图）。

- 硬件平台是什么，配置是什么。
  - 单节点：8× NVIDIA H100 SXM5 80GB，NVLink Gen4 (900GB/s)，双路 Xeon Platinum 8462Y+，2048GB DDR5 4800MHz，NVIDIA ConnectX-7 IB (400Gbps)。
  - 多节点：2 节点共 16× NVIDIA H200 SXM5 142GB，NVLink Gen4 (900GB/s)，双路 Xeon Platinum 8562Y，2048GB DDR5 5600MHz，NVIDIA ConnectX-7 IB (800Gbps)。

- 开源Serving框架是什么。修改了什么。
  - 框架：Megatron-LM (https://github.com/NVIDIA/Megatron-LM) 作为 baseline expert parallelism 框架。
  - 修改内容：
    - **All-to-all 通信模块**：修改 Megatron-LM 的 all-to-all 通信和 expert placement 模块，支持自定义 expert-to-GPU 映射（允许每个 GPU 分配不同数量的 expert，打破默认 contiguous block 分配）。
    - **Custom Expert Parallelism 初始化**：加载 ILP 输出的 expert-to-GPU mapping tensor，按层替换默认 expert placement。
    - **Token Routing Profiling**：在 Megatron-LM 推理流程中插入 profiling hook，记录每个 token 在各层的路由路径，构建路由统计表。
  - 开源情况：论文未明确说明开源链接，web search 未发现公开 GitHub 仓库（论文 2025 年 2 月发布，代码可能尚未公开）。ILP 求解使用 Gurobi 12.0.0 (https://www.gurobi.com)。
  - 使用例子——MoETuner 对 Megatron-LM MoE 推理的优化全流程：
    1. **Profiling 阶段**：在目标 task 数据集（如 WikiText-103）的采样子集上运行 N 轮 Megatron-LM 推理 → 收集每层每个 token 路由到的 expert 对 → 构建路由统计表（P_{e,l}: expert e 在层 l 的处理 token 数；R_{e_1,e_2,l}: 层 l 到 l+1 间 expert e_1→e_2 的 token 路由数）。
    2. **ILP 1 求解**：输入 P_{e,l}、E（expert数）、L（层数）、G（GPU数）→ 决策变量 x_{c,e,l} ∈ {0,1}（expert e 是否分配到 cluster c）→ min Σ|T_{c,l} - T̄_l|，约束每个 cluster 至少一个 expert → Gurobi 求解至 tolerance 0.025 → 输出 x_{c,e,l}（每个 expert 的 cluster 归属）。
    3. **ILP 2 求解**：用 x_{c,e,l} 预计算 C_{c_1,c_2,l} = Σ R_{e_1,e_2,l} · x_{c_1,e_1,l} · x_{c_2,e_2,l} → 输入 y_{c,g,l} ∈ {0,1}（cluster c 是否分配给 GPU g），带宽 B_{g_1,g_2} → min Σ max(C_{c_1,c_2,l} / B_{g_1,g_2} · y_{c_1,g_1,l} · y_{c_2,g_2,l+1})，约束每个 GPU 等量 expert、每个 cluster 一对一映射 → 输出 y_{c,g,l}（每个 cluster 对应哪个 GPU）。
    4. **部署阶段**：Megatron-LM 加载 MoE 模型 → 初始化时读取 ILP 输出的 expert-to-GPU mapping → 替换 contiguous block placement → 推理时按优化后的 layout 执行 all-to-all token dispatching → GPU 间通信量均衡且 token 处理负载均衡。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B（8 experts/layer, top-2 routing, HuggingFace Hub 预训练权重）。
  - 数据集：WikiText-103、MiniPile、LAMBADA、enwik8（均为语言建模数据集）。
  - 并行配置：单节点 8 GPU = 4EP-2TP（4 expert parallel × 2 tensor parallel）；多节点 16 GPU = 4EP-4TP。

## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoESys 的 inference 阶段包含两大核心组件：（1）**Graph Optimization Pipeline**——6 步流程（Graph Fusion → Distillation & Compression → Graph Conversion → Graph Segmentation → IR Optimization → Deployment），将训练的动态图转为静态图后做 kernel fusion、分布式切分并部署；（2）**Ring Memory Offloading**——当 MoE 模型超过单 GPU 显存时，将 CPU-GPU 内存构建为环形内存区，每个 decoder layer 的 expert 参数在 CPU 上存 N 份副本、GPU 上缓存 K 份副本，计算第 i 层时释放第 i 层参数并异步加载第 (K+i) 层参数，形成 "计算-释放-加载" 的流水线。
  - 实验比较：（1）MoE inference throughput：对比 DeepSpeed，不同参数规模（10B/106.5B/209.6B）和 GPU 数（1/8/16）下的 tokens/s；（2）Ring Memory Offloading：48.2B 参数 32-expert MoE 模型在 16×A100(40G) 上，有/无 overlapping offloading 的耗时对比及 GPU memory 节省比例。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A100 80GB（大模型 inference），A100 40GB（ring memory offloading 实验，16 GPU）。
  - CPU: 论文未明确说明型号，用于存储 expert 参数副本并提供 copy engine。
  - Storage: SSD 存储模型参数文件，CPU memory 缓存 expert 参数。

- 开源Serving框架是什么。修改了什么。
  - 框架：PaddlePaddle / PaddleFleetX（https://github.com/PaddlePaddle/PaddleFleetX）。MoESys 作为上层系统集成。
  - 修改内容：
    - Graph Optimization：实现 6 步推理图优化 pipeline——Graph Fusion（合并分布式策略消除参数冗余）、Distillation & Compression（teacher→student 减少 expert 数）、Graph Conversion（动态图→静态图，使用 PaddlePaddle JIT `paddle.jit.to_static`）、Graph Segmentation（手动或自动选择分布式策略切分子图）、IR Optimization（kernel fusion 等 pass）、Deployment。
    - Ring Memory Offloading：为 MoE 模型设计 ring memory 调度器，管理 CPU↔GPU expert 参数传输，利用多个 CUDA stream 实现 compute 与 H2D copy 的部分重叠。
    - Kernel 层：Fused Multi-head Attention（来自 NVIDIA BERT MLPerf 实现）、Custom H2D/D2H kernels（CUDA Pinned Memory）、Custom AlltoAll communication。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - MoESys 基于开源 PaddlePaddle/PaddleFleetX，论文称代码将发布于 PaddlePaddle GitHub，截至搜索未找到独立 MoESys 仓库。
  - MoESys Serving 全流程（以 209.6B MoE 模型在 16×A100 上做 text generation 为例）：
    1. **输入**：用户 query tokens 进入 MoESys inference server。
    2. **Graph Optimization（离线）**：
       a. Graph Fusion：原始动态图 + 分布式策略 → 合并冗余参数。
       b. Distillation & Compression：teacher MoE（多 expert）→ student MoE（少 expert），通过 Mixture-of-Students 方式。
       c. Graph Conversion：`paddle.jit.to_static` 将动态图转静态图。
       d. Graph Segmentation：根据可用 GPU 资源自动或手动选择 expert parallelism + tensor slicing 策略，将静态图切分为分布式子图，插入必要的通信 op。
       e. IR Pass Optimization：应用 kernel fusion（如 fused MHA）提升子图推理性能。
       f. Deployment：优化后的子图部署到各 GPU。
    3. **在线推理（Ring Memory Offloading）**：
       a. 模型含 N 个 decoder layer，每个 layer 的 expert 参数在 CPU memory 存 N 份副本。
       b. GPU memory 划出 ring buffer，容量为 K 份 expert 参数副本 + dense 参数 buffer。
       c. 初始：从 CPU 加载前 K 层的 expert 参数到 GPU ring buffer。
       d. 计算第 i 层：GPU 执行 attention + FFN（含 MoE routing + expert FFN），耗时 T_compute。
       e. 释放：第 i 层计算完成后，释放 Pi 占据的 ring buffer slot。
       f. 异步加载：CPU→GPU copy engine 开始将第 (K+i) 层 expert 参数从 CPU memory 传输到刚释放的 slot（使用独立 CUDA stream），耗时 T_copy。
       g. T_copy 与 T_compute 部分重叠 —— 当 K 足够大且 decoder layer 数足够多时，重叠率极高。
    4. **输出**：生成的 token 序列返回用户。
    5. 效果：GPU memory 消耗降低 ≥30%，推理速度不受 CPU offloading 明显影响（图 12 显示重叠 offloading 的 compute time 仅略高于无 offloading）。

## MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：利用 vLLM 框架的 batched speculative decoding 功能，分析并验证不同 batch size 下 MoE 模型的 SD 加速效果。论文聚焦于（1）中等 batch size（tens of requests）下 SD 对 MoE 的加速潜力——此时所有 expert 已在单步解码中激活，验证多 draft token 不产生额外参数加载开销；（2）Serving 场景分析：private serving（企业内部 chatbot 等中等 batch 场景）、latency-critical 场景（大 batch 不可行）、memory-constrained 场景（MoE 超出 GPU 显存需 offloading）；（3）batch size 对 target efficiency 和 SD speedup 的定量影响趋势。
  - 实验比较：（1）不同 batch size（1-128+）下 SD speedup 曲线——验证先升后降趋势；（2）不同 GPU 平台（2xGPU-A/B, 4xGPU-A/C）的 speedup 对比；（3）MoE vs dense 模型的 end-to-end speedup 随 batch size 变化趋势；（4）不同 sparsity ρ 下 speedup 峰值对应的 batch size 和有效加速范围；（5）不同 dataset/temperature/γ 组合下的 speedup 趋势。

- 硬件平台是什么，配置是什么。
  - 2xGPU-A, 2xGPU-B, 4xGPU-A, 4xGPU-C（论文对 GPU 型号做了匿名化处理）。多 GPU 配置用于评估 inter-GPU parallelization 对 SD speedup 的影响（target model 受益于并行化而 draft model 仍为单 GPU）。

- 开源Serving框架是什么。修改了什么。
  - 框架：vLLM（支持 batched speculative decoding、cudagraph optimization，可报告 T_D, T_T, T_reject, σ 等详细数据）。
  - 修改内容：论文未明确说明对 vLLM 的代码级修改。主要通过修改模型 config.json 中的 `num_experts_per_token` 参数来控制 MoE sparsity 进行实验。vLLM 原生的 batched SD 能力被直接用于验证理论预测。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供独立开源代码仓库，实验完全基于开源 vLLM 框架。
  - 基于开源文档和论文，vLLM Serving 全流程：
    1. **输入**：B 个 user requests（prompt tokens）进入 vLLM server。
    2. **Prefill**：vLLM 对 B 个 prompts 做并行 prefill，计算 KV cache。
    3. **Speculative Decode Loop**：每轮执行 Draft → Verify → Reject 三步：
       a. Draft 阶段：draft model（Qwen2-0.5B 或 Eagle head）在单 GPU 上自回归生成 γ 个 draft tokens（每 token 耗时 T_D(B,1)），γ 次 forward 共耗时 γ × T_D(B,1)。
       b. Verify 阶段：target MoE model 以 batch=B, seq_len=γ 做一次并行 forward。MoE layer 内部：Gate 路由 (B×γ) 个 tokens → 每个 token 激活 K 个 expert → N(Bγ) 个不同 expert 被激活并加载参数 → expert FFN 计算 → 加权汇总。vLLM 的 cudagraph 优化捕获并重放计算图。
       c. Reject 阶段：对比 target logits 与 draft logits 做 rejection sampling，丢弃不匹配的后续 token。
    4. **输出**：每个 request 的生成 token 序列返回给用户。
    5. vLLM 报告各阶段时间分解（T_D, T_T, T_reject）和 σ（接受率相关），论文利用这些数据计算 target efficiency 并验证性能模型。

## MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-SpeQ 的 Serving 调度核心包含三大组件：
    1. **Speculative Governor（自适应控制面）**：基于 Amortization Roofline Model 动态优化 speculative draft length k。将性能建模为两 Roof：Compute Roof（水平线，I/O 完美隐藏时的最大吞吐，高度依赖 k 影响草稿+验证成本）和 I/O Roof（斜率=有效 PCIe 带宽 B_PCIe 的斜线）。在线求解 argmax_k Θ(k) = k_accept(k) / T_cycle(k)，其中 k_accept(k) = Σ∏p_j（EMA 更新的条件接受概率），T_cycle(k) = max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1)。离线 SLO 约束确定 k_max 上限（如 TTFT < 500ms），在线受限搜索 [k_min, k_SLO]。
    2. **Expert Scheduler（数据编排器）**：基于 Expert Lookahead Buffer (ELB) 的 lookahead 驱动三阶段预取流水线——Phase I: 利用 cache 做 locality-aware cache priming；Phase II: ELB 中部选择性预取高置信度 experts；Phase III: 饱和 VRAM cache 消除 verify 阶段 I/O stall。ELB 为 k×L 结构（每个草稿 token×每层包含 (expert_id, confidence_score) 元组），在 CPU 端无阻塞构建。lookahead-aware eviction 策略替换最不可能在后续阶段使用的 experts。
    3. **Execution Engine（执行引擎）**：CUDA multi-stream 调度管理四维并发——多阶段预取、预取与按需加载协调、计算通信 overlap、双向 host-device 传输。pipeline-based 异步加载机制使用 pinned memory + non-blocking CUDA memcpy。verification 阶段 computation reordering 重排 batch tokens 使同 expert 计算连续，最大化 L1/L2 cache 利用率。static shared memory 配置避免运行时分配同步。batched expert selection pattern 处理最小化 D2H 传输。
  - 实验比较：（1）MoE-SpeQ vs HuggingFace Transformers (device_map) vs Mixtral-Offloading-SC vs Mixtral-Offloading-SM，end-to-end TPOT 对比；（2）不同 cache 策略命中率：speculative prefetching vs LRU vs LRU(scaled) vs Single Prefetch(sooner/later)，在 16/24/32GB VRAM 预算下；（3）消融：Full vs without async prefetch vs without fused kernel。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA A100-40GB GPU（PCIe 4.0 x16，理论双向 32GB/s 聚合带宽）。
  - CPU：24-core Intel Xeon Silver 4310，256GB RAM。
  - 实验变量：GPU 内存约束（low-memory / high-memory 两种配置），expert cache 容量（6/16/22/32/48 槽位对应不同 VRAM 预算）。

- 开源Serving框架是什么。修改了什么。
  - 基于 Hugging Face Transformers 框架构建，利用其通用模型接口。未使用 vLLM、SGLang 或 llama.cpp（论文明确指出不同底层框架间不可直接对比）。
  - 修改/新增内容包括：
    (1) **Expert Scheduler**：新增 Expert Lookahead Buffer 数据结构、三阶段预取流水线、lookahead-aware eviction 逻辑，替换原生 Transformers 的粗粒度 device_map offloading。
    (2) **Speculative Governor**：新增 Amortization Roofline Model 建模与在线优化逻辑，动态调整 k。
    (3) **Execution Engine**：新增 CUDA multi-stream 调度的异步执行管理（多阶段 prefetch、计算-通信 overlap、双向传输协调），static shared memory 预配置避免运行时分配同步，pinned memory + non-blocking CUDA memcpy 流水线化异步加载，batched D2H expert selection transfer。
    (4) **Speculative Decoding Loop**：新增 draft→prefetch→verify 三步循环替代原生单步 autoregressive decode。
    (5) **Parameter/KV Cache Sharing**：修改 draft/target 模型管理逻辑，使两者共享 non-expert 参数和 KV cache，每步 verify 后同步 KV cache。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供开源代码仓库链接。
  - Serving 框架全流程（基于论文 §3 和 §4.1.2）：
    1. **输入**：用户 prompt token 序列 → GPU 端 prefill 阶段计算 attention + 加载首层 experts（通过 Transformers model.forward）。
    2. **解码循环**（每步一个 cycle）：
       - **Step 1 - 初始专家加载**：GPU host-to-device 异步传输首组所需 experts（T_pcie,init），同时 CPU 构建初始 ELB。
       - **Step 2 - Draft 生成**：量化的 INT4 draft model 在 GPU 上自回归生成 k 个候选 token，使用 fuseMoE kernel 加速细粒度 MoE 计算。每 token 生成后 CPU 非阻塞解析 router logits → 追加 ELB 条目 (expert_id, confidence_score)。
       - **Step 3 - 预取调度**：Expert Scheduler 读 ELB，Phase I 利用现有 cache hits → Phase II 选择性预取 ELB 中部高置信度 experts（non-blocking H2D transfer） → Phase III 饱和所有缺失 experts。Speculative Governor 根据 Amortization Roofline Model 实时计算最优 k，若 token 接受率下降则动态缩短 k。
       - **Step 4 - Verify**：Target FP16 model 对 [original_prompt + k draft tokens] 单次并行 forward。执行前做 computation reordering（按 expert id 重排 token 计算顺序以最大化 L2 cache 命中）。逐 token 比对 target logits vs draft tokens：接受匹配前缀，从 target 分布采样分歧处新 token，回滚 KV cache/sequence states。
       - **Step 5 - KV Cache 同步**：将 target 的高精度 KV cache 复写到 shared KV cache，供下一步 draft 使用。
    3. **输出**：每个 cycle 输出 ≥1 个有效 token，循环直到 EOS 或 max_len。
    4. **CUDA 多流调度细节**：4 条 CUDA stream 分别管理——(a) draft 模型前向计算流、(b) H2D expert 预取传输流、(c) target 模型 verify 计算流、(d) D2H router logits 回流。通过 CUDA events 管理跨流同步与互斥，CUDA streams 间通过 cudaStreamWaitEvent 确保数据依赖。

## MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-Prism 的 Online Scheduling Engine，基于一次性的 offline benchmark 构建轻量级性能模型 C(k_active)，将激活 sub-expert 数量映射到延迟和内存开销。包含两种调度策略：
    1. **Quality-Constrained Throughput Scheduler（云端吞吐最大化）**：维护 M 个虚拟队列（M=可能的 k_active 数量），每个请求根据 k_min 加入所有满足质量需求的虚拟队列。调度器对各队列计算效用函数 U_m = Σ tokens(R_i) / C(|Q_m|, m)，选择效用最高的队列发射批次。两个硬触发器防饥饿：Batch Full（队列满 B_max）和 Timeout（请求等待超 T_max）。每当从 Q_m 发射批次时，所有批次内请求从所有虚拟队列中原子移除。
    2. **Latency-Optimized Offloading Manager（内存受限设备延迟最小化）**：VRAM Cache Manager 将 GPU VRAM 作为 sub-expert 缓存（LRU 策略），CPU RAM 作为持久存储。Generation Step Orchestrator 逐 token 循环：运行 router 确定所需 sub-expert 集合 S_req(t) → 查询 VRAM cache 得到 miss set S_miss(t) → 异步 CPU→GPU 传输 S_miss(t) → 计算。每步延迟 L(t) = Latency_IO(S_miss) + Latency_compute(S_req)。细粒度 sub-expert 使 I/O 从加载整个 monolithic expert 变为按需只传输所需 sub-expert。
  - 实验比较：与 FullBatch（静态批处理直到 B_max 才发射）和 FIFO（动态非阻塞，先到先服务）两种基线调度器对比。评估指标包括 TTFT（首 token 延迟）、TPOT（每输出 token 延迟）、吞吐量（req/s）、端到端延迟。三个负载等级（low/medium/high），请求到达服从 Poisson 分布，实验时长 300 秒。Offloading 实验在 RTX 4080 (16GB) 和 RTX 4090 (24GB) 上测试。
- 硬件平台是什么，配置是什么：NVIDIA H800 GPU（云端调度），RTX 4080 16GB / RTX 4090 24GB（offloading 实验）。软件环境：PyTorch 2.7.0, CUDA 12.6。
- 开源Serving框架是什么。修改了什么：基于 vLLM 0.9.1 修改，增加了自定义 gating logic 以支持 MoE-Prism 的 proxy gating 和 fine-grained sub-expert 选择。具体修改包括：(1) 修改 MoE layer 的路由逻辑，将原始 top-k expert 选择替换为 fine-grained sub-expert 的 gate neuron proxy 评分机制；(2) 修改 expert 加载/卸载逻辑，支持 sub-expert 粒度的 CPU-GPU 数据传输和 VRAM 缓存管理。
- 开源情况：论文未明确说明开源链接。基于论文描述，Serving 框架使用流程：(1) 部署前执行一次 benchmark，对每种 k_active 值测量延迟/内存，构建查找表 C(k_active)；(2) 加载 refactored model（含 sub-expert 权重和 gate neuron 索引）；(3) 云端场景：请求到达→按 k_min 分配到所有符合条件的虚拟队列→计算各队列效用 U_m→选择效用最高队列或触发硬触发器→发射批次→vLLM 推理→返回结果；(4) Offloading 场景：VRAM cache 初始装载热点 sub-expert→解码循环中 router 输出 S_req(t)→miss=不在 cache 的子 expert→异步 CPU→GPU 传输→GPU 计算→LRU 更新 cache→生成下一个 token。吞吐实验中 MoE-Prism 在 Deepseek 上比 FIFO+原模型提升 19.9% 吞吐（13→15.59 req/s），在 OLMoE 上提升 14.9%（15.57→17.89 req/s）。Offloading 实验中端到端延迟降低约 10%。

## MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-Lens 提出一个面向资源受限环境的 CPU-GPU 混合高吞吐 MoE LLM 推理系统，核心包含五个组件：
    1. **两阶段 holistic performance model**：Stage 1 模型基于 CPU memory capacity、GPU compute、workload 特征（prompt/generation length）推导 PME（Parallelism-Memory Efficiency）指标和理论上界；Stage 2 模型引入 bounded batch size K、paged KV cache、prefill/decode overlapping 调度策略，精确预测端到端 wall-clock 时间（94% 准确率）。
    2. **Resource-Aware Scheduler**：包含 Prefill Scheduler 和 Decode Scheduler 双调度器，均运行在 GPU 上以 GPU memory 维护调度状态。支持 Normal Inference Mode（prefill/decode 并行调度）和 Preemption Mode（KV cache 不足时抢占 decode 序列、释放 KV cache blocks、将抢占序列重新注入 prefill 阶段）。
    3. **Pipeline Profiler**：基于 Equation 2 估算 GPU compute 饱和所需 token 阈值 $n_{real}$，通过测量不同 token 数量的 GPU 时长和单层 weight 传输时间来校准。Scheduler 确保调度 token 数不超过 $n_{real}$。
    4. **VSLPipe 执行引擎**：将 MoE transformer layer 的 compute graph 重组为 `GA (QKV proj + GPU Flash Attn)` → `C (CPU Decode Attn + KV cache store)` → `GB (O proj + MoE layer)`，跨 layer 重组成 execution stage（CPU-only phase → GPU-only phase）。采用 software pipeline（prologue → N-1 main stages → epilogue）将 prefill 和 decode tokens 分两组 $\alpha$/$\beta$ 交替执行，CPU attention 与 GPU GEMM 重叠。
    5. **Contiguous Data Mover**：独立线程运行的 C++ PyTorch extension，以 100MB packet size 分批传输 weight，避免与 PyTorch 计算传输的头线阻塞。Weight Buffer 大小为 $2 \times$ per-layer weight size，仅为原模型大小的几个百分比。
    6. **CPU Decode Attention**：手工 AVX512 SIMD intrinsics 实现的 decode attention kernel（§6.6），manual vectorization + loop unrolling + data prefetching，单线程 4.7×、全线程 3.1× 高于 auto-vectorized baseline。
  - 实验比较：(1) MoE-Lens vs MoE-Lightning 和 vLLM（CPU offload）的 generation throughput（tokens/sec）；(2) 不同模型（Mixtral8x7B 94GB、Mixtral8x22B 282GB、DBRX 264GB）下的吞吐对比；(3) 不同 KV cache 大小（70GB、210GB）对 throughput 的影响；(4) 不同数据集（MTBench 多轮对话、RAG 长 prompt、AIME2024 长生成）下的吞吐对比；(5) 不同 generation length（32、64、128、256 tokens）对 throughput 的影响；(6) 性能模型预测精度（94% accuracy）；(7) 详细执行状态分析（prefill/decode throughput timeline、GPU/CPU/IO 时间分解、preemption 频率、bandwidth contention）。

- 硬件平台是什么，配置是什么。
  - Dual-socket Intel Platinum 8380 CPU（每 socket 8×DDR4-3200 channels，总计 750GB），使用 numactl 限制到单 socket + 单 GPU。实测单 socket CPU memory bandwidth ~150GB/s。
  - NVIDIA A40 GPU（48GB），通过分配随机 tensor 模拟 T4/L4 级别内存（16-24GB effective GPU memory）。
  - PCIe 互连，实测 $B_{IO} \approx 19.5$ GB/s（1GB tensor transfer）。
  - KV cache 配置：70GB-210GB（模拟不同 CPU memory capacity）。

- 开源Serving框架是什么。修改了什么。
  - 论文自身未开源（2504.09345，2025年4月，无公开代码）。
  - 对比 baseline 的 serving 框架：
    - **MoE-Lightning** [9]：state-of-the-art 资源受限 MoE 推理系统，基于 Hierarchical Roofline Model（HRM）将 decode attention offload 到 CPU。开源实现参考 [8]（https://github.com/caoshiyi/artifacts/tree/asplos25）。
    - **vLLM** [26]：基于 paged attention 的 LLM serving 系统，使用 CPU offload 选项运行超大模型。开源 https://github.com/vllm-project/vllm。
  - MoE-Lens 相比 MoE-Lightning 的核心修改：
    1. 用 holistic two-stage model 替代 HRM（HRM 仅建模 arithmetic intensity 和 IO bandwidth，忽略 CPU memory capacity 和 request 特征）。
    2. Resource-Aware Scheduler 替代 MoE-Lightning 的独立 prefill/decode 调度（前者重叠 prefill/decode，后者分离执行）。
    3. VSLPipe + Contiguous Data Mover 替代 MoE-Lightning 的 pipelining（前者最大化 IO bandwidth 利用率，后者存在 IO stall）。
  - 相比 vLLM 的核心修改：vLLM 在 GPU 上计算所有 GEMM 和 attention，仅 page KV cache 到 CPU，受限于 PCIe 带宽。MoE-Lens 将 attention 完全 offload 到 CPU 执行，避免 KV cache 传输。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？
  - **MoE-Lens 未开源**。论文 arXiv: 2504.09345。
  - **框架使用流程（基于论文描述）**：
    1. **部署阶段**：Pipeline Profiler 对目标 GPU 和模型进行 profiling——Equation 2 估算 $n_{real}$，变参测量 GPU time 和 weight transfer time，拟合线性关系得到精确的饱和 token 阈值。
    2. **请求到达**：incoming requests 进入 Prefill Scheduler 队列。
    3. **Normal Inference Mode**：
       - Decode Scheduler 估算现有 decode sequences 的 KV cache block 需求（基于 paged KV cache，每 block b tokens）。
       - 若 KV cache 充足，Decode Scheduler 率先调度所有 decode sequences。
       - Prefill Scheduler 读取活跃 decode 数量，计算可额外调度的 prefill tokens 数（不超过 $n_{real}$），从队列头部调度 prefill requests。
    4. **VSLPipe 执行**：每个 stage 包含 CPU-only phase + GPU-only phase。$\alpha$ 组和 $\beta$ 组交替执行：一组在 GPU 做 GEMM 时另一组在 CPU 做 attention。每个 stage 开始前 Contiguous Data Mover 预取下一 stage 的 weights。
    5. **Preemption Mode**：若 decode 所需 KV cache blocks 不足，preempt 部分 decode sequences → 回收 KV cache → Prefill Scheduler 将抢占序列作为新 prefill 序列重新注入，利用 prefill/decode overlapping 隐藏重计算开销。
    6. **完成**：Decode Scheduler garbage collection 回收 KV cache blocks。
  - **作用**：最大化 CPU memory capacity 利用率（KV cache），通过 prefill/decode overlapping 平滑 GPU 和 PCIe 利用率，减少 GPU idle time。在 MTBench 上以 70GB KV cache 可达 ~90% GPU utilization（$g_{max}=32$），较 MoE-Lightning 平均 4.6× 加速。

## MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-GEN 提出 **module-based batching** 策略，核心修改在离线推理 serving 框架的批处理调度层：
    1. **Module-based batching**：将 MoE 模型分解为 attention 和 expert 两类计算密集型模块，分别为其设置不同的微批次大小（$b_a$ 和 $b_e$）。在 attention 模块以小批次运行，累计多个 attention 批次的 token 后在 expert 模块合并为大批次运行，从而最大化 expert 模块的 GPU 利用率。
    2. **Full KV-cache offloading**：将 KV-cache 完全卸载到 host memory，节省 GPU memory 给更大批次，相比部分卸载减少最高 20× 的 expert weight fetching 流量。
    3. **CPU attention offloading**：将 self-attention 机制（$QK^T$）的计算卸载到 CPU（自定义 AVX 内核），节省 HtoD 带宽给 expert 预取。
    4. **DAG-based batching strategy search**：将 MoE 卸载推理建模为 DAG，通过动态规划求解 critical path 最小化执行时间，自动搜索最优的 $B$、$b_a$、$b_e$、$\omega$（CPU split ratio）、$S_{Expert}$、$S_{Params}$ 配置。
    5. **Single GPU buffer for dense modules**：dense 模块（attention、shared expert）使用单 GPU 预取缓冲区，大小为单层 dense 模块大小即可充分 overlap。
    6. **Engine 实现**：约 3000 行 C++ 和 2000 行 Python 代码，集成 HuggingFace generation pipeline。
  - 实验比较：(1) MoE-GEN(G)（纯 GPU 计算）和 MoE-GEN(H)（CPU+GPU 混合 attention）vs baselines：Llama.cpp、vLLM（continuous batching）、DeepSpeed-Inference、FlexGen*、MoE-Lightning*（model-based batching）；(2) 不同模型（Mixtral-8x7B、Mixtral-8x22B、DeepSeek-V2 236B、DeepSeek-R1 671B）下的 prefill throughput 和 decoding throughput；(3) 不同 context length（512-24K tokens）的 long context performance；(4) 不同 batch size（1, 32）下的小批次性能；(5) 不同 CPU attention ratio $\omega$ 的影响（0-100%）；(6) 完整 dataset 完成时间（MMLU 116K、GSM8K 8.5K、ChatbotArena 36K sequences）。

- 硬件平台是什么，配置是什么。
  - C1: NVIDIA A5000 24GB + AMD EPYC 7453 28-Core + 256GB Host Memory
  - C2: NVIDIA A5000 24GB + AMD EPYC 7453 28-Core + 512GB Host Memory
  - C3: NVIDIA A6000 48GB + AMD EPYC 7313P 16-Core + 480GB Host Memory
  - PCIe 4.0 互连（32 GB/s HtoD 带宽）。

- 开源Serving框架是什么。修改了什么。
  - 开源：https://github.com/EfficientMoE/MoE-Gen。
  - 未基于现有开源 serving 框架修改，而是自研 MoE-GEN Engine。对比的 serving 框架包括：
    - **FlexGen**：model-based batching，按轮次重用已加载的模型权重进行多次 forward pass，未针对 MoE expert sparsity 优化 batch size。
    - **DeepSpeed-Inference**：将 MoE layer 视为 dense MLP 处理，batch size 受 attention peak memory 限制。
    - **MoE-Lightning**：优化 GPU-CPU-I/O overlap 但保留 model-based batching。
    - **vLLM / Llama.cpp**：continuous batching，面向 interactive inference 的 TTFT 优化，解码阶段 batch 更小。
  - MoE-GEN 的核心修改：将 batching unit 从 model level 下沉到 module（attention/expert）level，累计 token 形成大 batch 后才在 GPU 执行。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？
  - **MoE-GEN Engine 执行流程**（以解码阶段为例）：
    1. **Batching Scheduler**：基于硬件/软件 profiling 数据在搜索空间枚举候选配置，对每个配置通过 DAG Constructor 估算 runtime，选择最短完成时间的配置。确定 $B, b_a, b_e, \omega, S_{Expert}, S_{Params}$。
    2. **Engine 初始化**：按配置在 GPU 上分配 KV-cache buffer、expert module buffer、dense module buffer。
    3. **Attention 阶段**：按照 attention micro-batch size $b_a$ 重复执行 attention 模块。对于每一批：
       - GPU 端：执行 Pre-Attention（QKV projection），同时 HtoD engine 预取下一批 attention weights 和对应 KV-cache。
       - CPU 端：按 split ratio $\omega$ 并行执行 self-attention mechanism（$QK^T$），CPU 可直接访问 host memory 中的 KV-cache，无需 HtoD 拷贝。
       - GPU 端：执行 self-attention mechanism（需要先完成 KV-cache HtoD copy）。
       - GPU 端：执行 Post-Attention（output projection）。
    4. **Expert 阶段**：将多个 attention 批次累计的 token 合并为大批次 $B$。由于大 batch 下 token 均匀分配到各 expert，顺序执行所有 experts：HtoD engine 预取下一个 expert weights（利用 PCIe idle 时间），GPU 执行当前 expert 计算。每次只加载一个 expert weights 到 $S_{Expert}$ buffer。
    5. **KV-cache 更新**：DtoH engine 将新生成的 KV-cache 异步写回 host memory。
    6. 以上步骤逐层迭代，直到所有 layers 完成。

## MoE-ERAS: Expert Residency Aware Selection

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-ERAS 在 MoE 推理 serving 场景中，修改 gating network 的 expert 选择策略，使其根据 expert 当前是否驻留在 GPU HBM 中来调整选择决策。核心修改在 serving 框架的 expert dispatch 阶段：
    1. 维护每个 MoE layer 的 expert residency 状态表（记录 expert 在 HBM 还是 CPU DRAM）。
    2. 在每层 gating 输出后、Top-K 选择前，插入 thresholding 或 biasing 操作调整 logits/weights，使路由器倾向选择已驻留的 expert。
    3. 当路由器选择 on-chip expert 而非 off-chip expert 时，避免了一次 CPU→GPU 的 expert 参数传输（在 memory-bound 解码阶段节省显著延迟）。
  - 实验比较：(1) Baseline（Top-K routing + quantization + LRU caching from dvmazur/mixtral-offloading）vs Thresholding（α=0.05, 0.15, 0.25）vs Biasing（β=1）；(2) 不同 offload per layer 设置（offload 1-7 experts per layer）下的 relative speedup；(3) Sequential decoding 的 wall clock latency 和 throughput 对比（100 token sequences, 50 iterations）。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA H100（图 2 展示 H100 HBM vs CPU DRAM 的 expert read time 对比：CPU 读取时间比 GPU 高数个数量级）
  - 主机内存：CPU DRAM 用于 offload 未常驻的 expert 参数
  - 配置文件：baseline 框架可在 Tesla T4 16GB 上运行 Mixtral-8x7B（通过 quantization + LRU caching + expert offloading）

- 开源Serving框架是什么。修改了什么。
  - 基座框架：`dvmazur/mixtral-offloading`（https://github.com/dvmazur/mixtral-offloading），该框架提供 expert 量化、LRU caching 和 expert offloading 功能。
  - 修改内容：
    1. 在 gating network 输出与 Top-K selection 之间插入 residency-aware routing 逻辑（thresholding 或 biasing）。
    2. 维护 expert residency table：跟踪每个 layer 的 expert 当前在 HBM 还是 DRAM。
    3. Profiling 模块：在 inference 前收集 expert activation frequency（用于 biasing 的 freq 参数）。
  - 服务流程：请求进入 → self-attention 计算 → gating network 输出 logits → residency-aware routing (thresholding/biasing) 调整 logits/weights → Top-K 选择 expert → 若选中 off-chip expert 则触发 CPU→GPU 传输 → expert MLP 计算 → 输出 token → 更新 LRU cache 和 residency table。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：MoE-ERAS 自身代码未开源。Baseline `dvmazur/mixtral-offloading` 开源。
  - 框架执行全过程（以 thresholding α=0.15, offload=3 experts/layer 为例）：
    1. **输入**：用户 prompt token 序列，模型权重（expert 参数部分在 HBM，部分在 CPU DRAM）。
    2. **Prefill 阶段**：prompt tokens 经 embedding → attention layers → 到达 MoE layer i。Gating network 计算 Logits = H_i @ W_exp。Residency-aware routing：Weights = Softmax(Logits)，对 HBM 中的 5 个 expert 加 α=0.15。SelectTopK(Weights, k=2) → 若选择的 2 个 expert 均在 HBM 中，零传输开销；若选择 off-chip expert，CPU→GPU PCIe 传输 ~数百 MB 的 expert 权重（图 2：CPU 读取延迟 >> GPU 读取延迟）。Expert MLP 计算输出。LRU cache 更新 residency 状态。
    3. **Decode 阶段**：逐 token 生成，每步经过所有层。当 α=0.15 且 3 experts offloaded 时，约减少 10-13% 的解码延迟；offload 越多、α 越大，节省越显著（最大 21.2% reduction）。
    4. **输出**：生成的 token 序列 + 各层 expert 激活记录（用于 biasing 方法的 freq 更新）。

## MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-GPS 修改了 MoE 多 GPU 推理 serving 流程，在每层 Attention 之前插入 predictor，根据预测结果动态调整 expert 在 GPU 间的分布（dynamic expert duplication）以均衡负载。整体 serving 流程修改如下：
    1. **Predictor 插入点**：在每层 Transformer block 的 Attention 计算之前，predictor 接收当前 batch 的 hidden states，输出 expert 分布预测。Predictor 可为 Distribution-Only（offline 估计）或 Token-to-Expert（在线推理）。
    2. **Expert Duplication**：基于预测的 token-to-expert 分布，使用 Algorithm 1（贪心算法）在 GPU 间动态复制热门 expert。迭代将 overloaded GPU（token > 1/G）上的 expert 复制到 underloaded GPU 直至均衡。
    3. **Token Dispatch**：Distribution-Only 使用 All-to-All Scatter（随机分发，通信开销不变）；Token-to-Expert 使用 Direct Routing（按预测结果直接路由到目标 GPU，跳过 Scatter 阶段，节省通信开销）。
    4. **Expert 移动开销隐藏**：Expert duplication 的 weight 传输（Mixtral 8×7B 单 expert ~47MB FP16）通过 NVLink（~0.1ms）可与 Attention 计算重叠，因此在 moderate batch size 下 latency 可隐藏。
  - 实验比较：(1) Baseline（无 prediction，EP-only FFN + TP Attention）vs Distribution-Only Prediction vs Token-to-Expert Prediction（多 accuracy 点）；(2) NVLink（2TB/s）vs PCIe（32GB/s）下的 prefill latency 对比；(3) 不同 skewness（1.2-2.0）下的 prediction strategy 最优选择；(4) 不同 interconnect bandwidth（600GB/s, 200GB/s, 64GB/s, 32GB/s）下的 savings difference（Distribution-Only minus Token-to-Expert）；(5) Mixtral / LLaMA-MoE / Switch Transformer 横评。

- 硬件平台是什么，配置是什么。
  - GPU：4× NVIDIA A100，fully connected
  - Interconnect：NVLink 3.0（2 TB/s, 600 GB/s per link）或 PCIe 4.0（32 GB/s）
  - 模拟器：LLMCompass [36]（block-level simulator, ISCA 2024），增强支持 MoE + EP + Mixtral 架构
  - 配置：batch size=1, sequence length=512, FP16

- 开源Serving框架是什么。修改了什么。
  - 基座模拟器：LLMCompass [36]（https://github.com/PrincetonUniversity/LLMCompass, ISCA 2024）
  - 修改/增强内容：
    1. **MoE + Expert Parallelism 模块**：新增 EP-specific communication（All-to-All scatter/gather）和 FFN workload 建模。
    2. **Mixtral 架构支持**：实现 Grouped Query Attention (GQA)、SwiGLU activation、Sliding Window attention (4K window)。
    3. **Prediction Strategy 建模**：新增 Distribution-Only 和 Token-to-Expert 两种策略的 runtime 建模，支持可调 accuracy 和 overhead。
    4. **Prediction 性能模拟**：使用 exponential 函数拟合 accuracy-overhead 曲线，polynomial 函数拟合 accuracy-performance 曲线。
  - 论文自身 MoE-GPS 框架代码未开源。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：MoE-GPS 自身代码未开源。LLMCompass 模拟器开源（https://github.com/PrincetonUniversity/LLMCompass）。
  - 框架输入到硬件执行全过程（以 Mixtral 8×7B, 4×A100 NVLink, Distribution-Only Prediction, batch=1, seq_len=512, skewness=1.4 为例）：
    1. **输入**：batch token 序列（1×512 tokens），模型权重分布在 4 个 A100 GPU 上（EP for FFN: 每 GPU 2 experts；TP for Attention）。
    2. **Predictor 阶段**：每层 Attention 前，Distribution-Only predictor 读取 offline MLE 估计的 expert 概率 $\hat{p}_i^l$ → 计算每个 GPU 的目标 token 数 → 触发 Expert Duplication（若某 GPU 负载超阈值则复制热门 expert 到其他 GPU）→ Expert 权重通过 NVLink 传输（~0.1ms, 可与 Attention 重叠）。
    3. **Attention 层（TP）**：输入经 Ring All-Reduce → QKV projection（TP sharded）→ Sliding Window Attention (4K) + GQA → Output projection → Ring All-Reduce → hidden states。Attention 延迟 ~12ms（LLMCompass 保守估计，未使用 FlashAttention）。
    4. **FFN 层（EP）**：Gating network → Top-2 expert selection → All-to-All Scatter（通信，将 token 发送到 hosting GPU）→ expert SwiGLU MLP 计算（因 Distribution-Only 均衡，各 GPU compute 时间相近）→ All-to-All Gather（通信，收集结果）。
    5. **性能**：Distribution-Only Prediction 无 predictor overhead，FFN compute 均衡 → 相比 baseline（无 prediction）节省 FFN compute delay → 相比 Token-to-Expert Prediction（最佳配置）提升 23%。
    6. **输出**：first token 生成（prefill latency），decode 阶段后续自回归生成（论文聚焦 prefill）。

## MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

- 属于Serving调度的实现是什么？实验比较什么？
  MegaScale-Infer 提出解耦式专家并行（Disaggregated Expert Parallelism），将 MoE 模型的 attention 模块与 FFN/Expert 模块分离部署到不同的 GPU 节点上，实现独立扩展和异构部署。核心包含三个机制：
  1. **Disaggregated Expert Parallelism**：将 attention 模块复制到多个 attention node（数据并行），FFN experts 分布在 expert node 上（expert 并行），每个 expert node 包含 1-8 个 GPU（节点内使用 tensor parallelism）。Attention node 聚合来自多个 replica 的请求，增大每个 expert 的有效 batch size，使 FFN 从 memory-intensive 转变为 compute-intensive。
  2. **Ping-Pong Pipeline Parallelism**：将请求 batch 拆分为 m 个 micro-batch，在 attention node 和 expert node 之间形成 ping-pong pipeline。Micro-batch 在 attention 和 expert 之间交替传递，前向计算覆盖通信开销，约束条件为 T_a ≈ T_e, T_c < T_f, m × T_f ≥ 2 × (T_f + T_c)。
  3. **Deployment Plan Search（Algorithm 1）**：枚举 tp_a, tp_e, n_a, m 组合，通过性能模型（基于 roofline 模型的 GEMM 时间估算 + profiling 获取的 k_i 系数 + network bandwidth utilization profiling）SIMULATE binary search 最大 global batch size 满足 SLO 约束。选择最大化 throughput per unit cost 的 deployment plan。
  4. **Heterogeneous Deployment**：attention node 使用高 per-cost 内存带宽/容量的 GPU（如 H20），expert node 使用高 per-cost 计算能力的 GPU（如 L40S）。
  实验比较了 MegaScale-Infer 与 vLLM（仅 tensor parallelism）和 TensorRT-LLM（tensor parallelism + expert parallelism）在不同模型（Mixtral 8x22B, DBRX, Scaled-MoE 317B）和硬件配置（同构 Ampere 集群、异构 H20+L40S 集群）下的 per-GPU decoding throughput、time between tokens (TBT)、end-to-end throughput（含 prefill）、per-cost throughput、per-unit-power throughput。

- 硬件平台是什么，配置是什么。
  同构集群：8 节点，每节点 8×NVIDIA 80GB Ampere GPU（如 A800），128 CPUs，2 TB host memory，8×200 Gbps InfiniBand NICs，节点内 400 GB/s NVLink。
  异构集群：NVIDIA H20（96 GB, 4096 GB/s bandwidth, 148 TFLOPS, 900 GB/s NVLink, 4×400 Gbps NICs）+ NVIDIA L40S（48 GB, 864 GB/s bandwidth, 362 TFLOPS, PCIe intra-node, 2×400 Gbps NICs）。
  bfloat16 用于 weights、activations 和 KV cache。

- 开源Serving框架是什么。修改了什么。
  MegaScale-Infer 是自研系统，与 vLLM 和 TensorRT-LLM 对比。核心修改：
  - 将 MoE 模型的 attention 和 FFN/expert 模块拆分为独立可部署单元。
  - 实现了 ping-pong pipeline 调度器，管理 micro-batch 在 attention node 和 expert node 之间的流水线执行。
  - 实现了 M2N 通信库（PyTorch extension，~4900 行 C/C++ + ~5000 行 Python）替代 NCCL 进行 attention-expert 间的 token dispatch/aggregation 通信。
  - 使用 Flux 实现 TP 通信与相邻 GEMM 的 kernel fusion（如 all-gather + GEMM 融合为单 kernel）。
  - 实现了 sequential memory-intensive operators 的融合（gating + top-k selection + token scatter），减少 kernel launch 和 memory access。
  - 基于 expert 流行度的 on-device redundancy load balancing：将 M 个 expert 分布到 N 个节点，minimize max C_j，使用 greedy approximation 解决。
  - 基座框架：论文未明确说明具体基座，但提到使用 Flux（ByteDance 的 kernel fusion 库）和自研 M2N 通信库。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未声明开源。代码为公司内部部署（已部署到近 10,000 GPU 的生产推理服务中，降低 serving cost 1.5-2.0×）。
  
  **MegaScale-Infer 推理全流程（以 Mixtral 8x22B decoding 为例，tp_a=2, tp_e=1, n_a=4, E=8, m=3）**：
  1. **请求接入与 micro-batch 划分**：用户请求到达 MegaScale-Infer runtime instance。请求被划分为 m=3 个 micro-batch，每个 micro-batch 大小 b_a = B/3。
  2. **Attention 计算（Attention Node，Layer ℓ）**：每个 attention node（包含 2 GPU tensor parallel）对 micro-batch i 执行 attention 计算（QKV projection + attention score + output projection），读取 KV cache（memory-intensive）。Attention node 1..n_a 各持有完整的 attention 参数副本和各自的 KV cache。
  3. **Gating + M2N dispatch**：Attention node 执行 gating network 计算（选择每个 token 的 top-K=2 experts）→ fused kernel（gating + top-k + scatter）准备 token embeddings → 通过 M2N Sender（CUDA event wait → CUDA stream block → Core Sender 通过 RDMA write with immediate 写数据到目标 expert GPU → poll completion queue → unblock stream）将 token embeddings 发送到对应 expert node。
  4. **Expert 计算（Expert Node，Layer ℓ）**：Expert node 通过 M2N Receiver（CUDA event wait → stream block → poll completion queue → GDRCopy flush → unblock stream）接收来自所有 attention node 的 tokens → 按 expert 聚合 tokens 为 batch → 执行 FFN Input GEMM + activation + FFN Output GEMM（compute-intensive，batch size = b_e = b_a × n_a × K/E）。
  5. **M2N aggregation**：Expert node 将 FFN 输出通过 M2N 反向发送回 attention node。
  6. **Ping-Pong Pipeline**：在 Layer ℓ 的 expert 计算期间，attention node 已开始 Layer ℓ 的下一个 micro-batch 或 Layer ℓ+1 的 attention 计算。m=3 时 pipeline 能完全覆盖通信时间（T_c < T_f 时），3 个 micro-batch 在 attention 和 expert 之间交替流动。
  7. **迭代完成**：所有 m 个 micro-batch 完成 L 层 MoE 的 forward pass 后，总 iteration latency T_total = (T_a + T_e + 2T_c) + T_f(mL − 1)。输出 tokens 返回用户。

## Making MoE-based LLM Inference Resilient with Tarragon

- 属于Serving调度的实现是什么？实验比较什么？
  TARRAGON 是一个具备故障恢复能力的 MoE 推理框架（约 16K 行 C++ + 2K 行 Python）。它在解耦的 Attention Worker (AW) 与 Expert Worker (EW) 部署之上构建了三个核心机制：
  1. **可重构数据通路（REFE + ERT）**：Reconfigurable Forwarding Engine (REFE) 是 AW 侧的运行时，通过 Expert Routing Table (ERT) 将逻辑 expert ID 动态映射到物理 EW/GPU，实现故障时的请求重路由，避免全局重启。
  2. **自愈机制（Self-Healing）**：AW 侧通过超时检测 + 重播到健康 EW/shadow expert 来容忍 EW 故障；EW 侧通过部分输入批处理（不等所有 AW）来容忍 AW 故障。
  3. **后台容量恢复（Background Provisioning）**：Orchestrator 在后台启动替换 AW/EW 并集成到在线推理 pipeline 中。
  实验比较了 TARRAGON 与 MegaScale-Infer（解耦基线）和 vLLM（单体基线）在故障场景下的 stall 时间、稳态下的 TTFT/TBT/吞吐量、以及 KV cache checkpointing 的开销。

- 硬件平台是什么，配置是什么。
  Google Cloud (GCP) A3 Ultra 节点，每节点：224 vCPUs, 3 TB RAM, 8x NVIDIA H200 GPUs (141 GB 显存), 8x 400 Gbps ConnectX-7 RDMA NICs（支持 GPUDirect RDMA），节点内 NVLink 3.6 Tbps。实验使用 3 个节点：AWs 占 1 节点 (8 GPUs)，EWs 占 1 节点 (8 GPUs)，checkpoint store 占 1 节点。软件环境：Ubuntu 22.04, Linux 5.15, CUDA 12.8 (driver 580), PyTorch 2.6.0。

- 开源Serving框架是什么。修改了什么。
  基座框架：**vLLM** 作为 AW 侧的 compute engine（处理 prefill 和 decoding 的 attention 计算）。EW 侧从零用 C++ 编写（libtorch for expert computation, libibverbs for RDMA）。Orchestrator 和 Checkpoint Store 均为独立 C++ 服务。
  具体修改：
  - 在 vLLM 上层增加了 REFE（C++ 扩展 + Python shim），负责 AW-EW 间的 RDMA 通信、ERT 查询和请求分发。
  - 实现了双 QP 设计：control-plane QP 用于存活探测和自愈元数据，data-plane QP 用于 token embedding 批量传输（GPUDirect RDMA）。
  - 实现了 ERT 机制：将 expert identity 与 expert location 解耦，允许动态重映射。
  - 实现了 AW 侧自愈：超时后重路由到健康 EW/shadow expert。
  - 实现了 EW 侧自愈：收到足够 AW 的输入即开始 expert 计算，不等所有 AW。
  - 实现了 shadow expert：在 EW GPU 显存中预加载但保持 inactive 的 expert 副本。
  - 实现了异步增量 KV cache checkpointing：利用 AW-EW 通信间隙进行 one-sided RDMA write。
  - 实现了 per-request KV cache restoration：从 checkpoint store 通过 GPUDirect RDMA 直接注入到替代 AW 的 GPU 显存。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文声明将开源（"We will open-source TARRAGON"），截止论文阅读时尚未公开链接。
  
  **TARRAGON 推理全流程（以 decoding 阶段为例）**：
  1. **请求接入**：用户请求通过单一 cluster gateway 到达，被分发到某个 AW。
  2. **Attention 计算**：AW 的 compute engine（vLLM）对当前 layer ℓ 执行 attention 计算，更新 KV cache，产生 token embeddings。
  3. **Gating + 分发**：compute engine 调用 `expert_io(expert_id, layer_id, token_embeddings)` API。REFE 查询 ERT 将 logical expert ID 解析为物理 EW，通过 data-plane QP（GPUDirect RDMA）将 token embeddings 直接写入目标 EW 的 GPU 显存。
  4. **Expert 计算**：EW 收到来自多个 AW 的 tokens 后，按 layer ℓ + expert ID 聚合为 batch，调用 libtorch 执行 expert FFN 前向计算。当收到足够 AW 的输入（或达到最小 batch size）即开始计算。
  5. **结果返回**：EW 将 expert 输出通过 RDMA 写回 AW 的 GPU 显存。AW aggregate 所有 expert 输出（加权求和），进入 layer ℓ+1。
  6. **KV Cache Checkpointing**：在 AW 执行 attention 的间隙（AW-EW link idle 时），REFE 异步将新增的 KV cache segment 通过 one-sided RDMA write 写入 checkpoint store。
  7. **故障处理（AW 故障）**：Orchestrator 检测到 AW 故障 → 从 checkpoint store 恢复该 AW 上所有请求的 KV cache 到健康 AW → 健康 AW 从 committed token 继续 decoding。
  8. **故障处理（EW 故障）**：AW 侧 REFE 探测到 EW 无响应 → 查询 ERT 获取替代 EW（含 shadow expert）→ 重播 token embeddings 到替代 EW → 无需等待 orchestrator。

## MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

- 属于Serving调度的实现是什么？实验比较什么？
  MixServe 提出自动分布式 MoE 推理服务系统，包含三个核心机制：
  1. **Automatic Analyzer（§III-B）**：离线阶段基于模型超参数（hidden dim h, head数, num_layers l, num_experts, top-k）+ 网络硬件配置（计算能力、intra-node/inter-node 带宽和拓扑），通过理论通信模型（AR 通信量 O(bs·h/d)，A2A 通信量 O(bs/d·hk)，式 1-3）和 profiling 数据，自动推导最优 TP-EP-DP 并行策略。包含 queuing-aware latency model（M/M/1 排队模型）预测 TTFT/ITL/Throughput（式 4-11）。
  2. **Hybrid TP-EP Partitioner（§III-C）**：将 Attention block 按 intra-node TP + inter-node DP 切分，MoE block 按 intra-node TP + inter-node EP 切分。解耦 AR 为 RS+AG，重组为 RS→A2A→AG 三段式通信流程，降低通信量和通信规模。自动枚举满足 n_proc × n_node = d_TP × d_EP 的所有可行策略并选最优。
  3. **Fused AR-A2A Communication Algorithm（§III-D）**：通过异步机制将 intra-node RS/AG 通信与 inter-node A2A 通信重叠执行。(a) Fused RS-Combine（Alg 1）：intra-node RS 与 inter-node A2A pairwise 异步重叠，最后 intra-node AG 汇总，时间 O(n_node)，空间 O(bsh·n_proc)；(b) Fused AG-Dispatch（Alg 2）：intra-node AG 与 inter-node Dispatch 异步重叠，时间 O(n_node)，空间 O(1)。
  实验比较了 MixServe 与 vLLM（TP+PP 和 DP+EP 两种配置）和 Tutel（TP+EP）在不同硬件平台（NVIDIA H20 + Ascend 910B）、不同模型（DeepSeek-R1 671B, Qwen3-235B-A22B）、不同并行度（TP=4/8, DP=2/4/8, EP=2/4/16/32）下的 TTFT、ITL、Throughput。还包括 DP vs EP trade-off 消融实验（d_DP = d_EP / d_DP > d_EP / d_DP < d_EP）和同步 vs 异步通信消融实验。

- 硬件平台是什么，配置是什么。
  - **NVIDIA H20 集群**：2 台服务器，每台 8×NVIDIA H20 GPU（96 GB）。Intra-node：NVLink 4.0（最高 900 GB/s）。Inter-node：InfiniBand（400 Gbps）。
  - **Ascend 910B 集群**：4 台 Atlas 800T A2 服务器，每台 8×Ascend 910B NPU（64 GB）。Intra-node：HCCS 全互联（最高 480 Gbps）。Inter-node：RoCE（最高 200 Gbps）。
  精度：论文未明确说明（通常 BF16/FP16）。

- 开源Serving框架是什么。修改了什么。
  基座框架：**vLLM**（Ascend 910B 集群）和 **Tutel**（H20 集群）。
  具体修改：
  - **Offline Stage**：新增 Automatic Analyzer 模块——获取模型超参数 → 用不同 batch size/seq length 的预设 prompt 收集 profiling 数据 → 以网络硬件配置（计算能力 + intra/inter-node 带宽和拓扑）为输入计算理论值 → 综合 observations 和理论值输出最优并行策略（§III-B1 定义的 context-free grammar 形式）。
  - **Online Stage**：新增 Weight Loader 和 Partitioner——根据 offline 输出的最优策略加载对应 weight shards → 初始化 mixed parallel communication groups → 向 MoE model 的 forward method 注入 collective communication operators（RS/AG/A2A）。
  - **Fused AR-A2A Communication**：解耦 TP group 的 AR 为 RS+AG → 重组 EP group 的 A2A → 形成 RS→A2A→AG 三段式流程 → 实现 Async RS-Combine（Alg 1）和 Async AG-Dispatch（Alg 2）算法，使 intra-node 和 inter-node 通信异步重叠。
  - **Serving service**：基于 vLLM 的 memory management 和 request scheduling 机制管理 K/V cache 和调度请求。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未声明开源。经 web search（arxiv 2601.08800）未发现公开代码仓库。基于 vLLM 和 Tutel 实现。

  **MixServe 推理全流程（以 DeepSeek-R1 在 4 节点 Ascend 910B 集群、TP=8 + DP=4/EP=4 为例）**：
  1. **Offline 策略分析**：MixServe 读取 DeepSeek-R1 模型超参数（671B, 256 experts, h=7168, l=61, top-k=8）→ 用不同 batch size（1/2/4/8/16）和 seq length（128/256/512/1024/2048/4096）的预设 prompt 做 profiling 获取 compute/comm latency 观测值 → 输入集群配置（4 node × 8 NPU, HCCS 480 Gbps intra-node, RoCE 200 Gbps inter-node）→ Analyzer 搜索满足内存约束（式 8）的最优 (d_TP, d_EP, d_DP) → 输出 TP=8 + DP=4, TP=8 + EP=4。
  2. **Weight Loading**：Weight Loader 根据 TP=8 切分每节点内 8 NPU 上的 Attention 参数（沿 hidden dim），DP=4 在 4 节点间复制 Attention 参数。MoE 参数按 TP=8（intra-node）+ EP=4（inter-node，256 experts / 4 = 64 experts per node，node 内 8 NPU TP 共享）加载。Partitioner 按此方案注入通信算子。
  3. **请求接入**：用户请求到达 vLLM serving service。Scheduler 使用 continuous batching 管理请求，PagedAttention 管理 KV cache。最大 batch size=16，最大 seq len=4096。
  4. **Attention 计算（Layer ℓ）**：每节点 8 NPU 对 Attention block 做 TP 前向：各 NPU 计算自己持有的 Q/K/V/O 分片 → intra-node AR（RS + AG）同步 hidden states。4 节点间 DP 独立计算（各自有完整 batch 子集）。
  5. **Gating + Expert Routing**：Attention 输出经 gating network 计算 top-8 expert → token 准备 dispatch。
  6. **Fused RS-Combine（MoE block 输入侧）**：每节点内 token hidden states 先做 intra-node RS → 同时启动 inter-node A2A pairwise（4 node, 3 rounds）：每轮中 intra-node RS 结果与 inter-node recv 结果异步处理 → top-k weights 加权 → 最后 intra-node AG 汇总完整 hidden states。
  7. **Expert FFN**：各 NPU 对本地 experts 执行 FFN（MLP 前向），利用 TP 组内共享的 expert 参数。
  8. **Fused AG-Dispatch（MoE block 输出侧）**：FFN 输出在 node 内做 intra-node AG（shard hidden states 到 TP group）→ 同时 inter-node Dispatch 发送到对应 EP rank → 除首轮 pairwise 和末轮 AG 外，其余轮次 intra/inter-node 通信重叠。
  9. **Iteration 完成**：所有 61 层 Decoder 完成 forward → output token 生成。TTFT = queuing delay + prefill latency（s=prompt_len），ITL = decode latency（s=1）。
  10. **Pipeline Parallelism**：论文提到 PP 可用于跨 Decoder layer 分配，式 6 包含 PP 的 P2P 通信项，但实验配置中未明确标注 PP degree。

- 属于Serving调度的实现是什么？实验比较什么？
  FineMoE 基于 Megatron-LM 实现了 FineEP 策略，通过 per-micro-batch 的 token scheduling 实现细粒度 GPU 负载均衡。核心包含四个机制：
  1. **Token Scheduling（§5）**：将负载均衡建模为线性规划问题（LPP 1），目标 min(max GPU load)，约束为每个 expert 的 replica loads 之和等于其 total load。使用 HiGHs 求解器在单 CPU 线程求解（变量数 O(|E|d)），利用 warm-start 跨 micro-batch 复用求解器状态。
  2. **Locality-Aware Routing（§5.2, Algorithm 1）**：优先将 token 路由到本地 expert replica（同一 GPU），减少 all-to-all 通信量。从本地到远程依次分配 token 直到各 replica 达到目标负载 x_e^g。
  3. **Distributed Scheduling（§5.3）**：所有设备通过 all-gather 收集全局 load 信息后各自独立执行确定性调度算法，比集中式减少一次通信操作。
  4. **Overlapping（§5.4）**：CPU 上的 LPP 求解与 GPU 上的 token permutation 操作重叠执行。
  实验比较 FineMoE（w/ and w/o Adaptive Replacement）与 Megatron-LM、SmartMoE、FlexMoE、DeepSpeed 在 GPT（32×1.3B/16×3.2B/8×6.7B）和 Mixtral（16×2B/8×7B）模型上的端到端训练吞吐量、负载均衡能力（Zipfian skewness s∈[0,2]）、执行时间分解、调度开销、DeepEP 集成、ablation study（warm solving/locality-aware routing/overlapping）。

- 硬件平台是什么，配置是什么。
  4 节点，每节点 8×NVIDIA H100 80GB SXM GPU（共 32 GPU），900 GB/s NVLink intra-node，2×400 Gbps InfiniBand NIC per node。训练精度 BF16。PP degree = 节点数（仅用于 inter-node），DP degree = 8，EP degree = 4，FineEP 参数 d=2（1 FineEP group / DP group）。禁用 TP（因其高通信开销）。使用 selective activation recomputation（仅 recompute MoE FFN），distributed optimizers（类 ZeRO-1）。

- 开源Serving框架是什么。修改了什么。
  基座框架：**Megatron-LM**（github.com/NVIDIA/Megatron-LM）。
  修改内容：
  - 修改 MoELayer forward：在 gate network 之后、all-to-all dispatch 之前插入 FineEP Token Dispatcher。
  - 新增 Place Manager（Python，device 0）：生成 expert placement（symmetric: Cayley graphs / asymmetric: greedy + Monte Carlo sampling），broadcast 到所有设备；后台监控 load 分布并触发 adaptive replacement。
  - 新增 Token Dispatcher（C++）：LPP 求解（HiGHs solver）+ locality-aware token routing（Algorithm 1）+ communication-aware scheduling（Appendix A.1）。
  - 扩展 all-to-all 通信组：从 EP group（EP_degree=4）扩展为 FineEP group（d×EP_degree=8）。
  - 实现 Distributed Scheduling：all-gather 收集 `input_e^g` → 各 GPU 独立求解 LPP → 各 GPU 独立 route tokens。
  - 实现 Adaptive Replacement（§6.4）：监控 expert load 分布 → 时间序列预测（moving averages）→ Equation 3 评估 → 触发 asymmetric placement 生成和模型 reinitialization。
  - 额外实现 SmartMoE 和 FlexMoE 在 Megatron-LM 上的版本用于公平对比。
  - 集成 DeepEP（high-performance all-to-all backend）与 FineEP。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未声明开源。经 web search（2025）未发现公开仓库。基于 Megatron-LM 实现（Python + C++ token scheduling）。

  **FineMoE 训练单个 micro-batch 的全流程（GPT 32×1.3B, DP=8, EP=4, d=2, B=4, seq_len=2048）**：
  1. **初始化**：Placement Manager（GPU 0）用 Cayley graph 生成 symmetric expert placement（8 GPUs × 4 experts/GPU），broadcast 到所有 GPU。各 GPU 加载分配的 expert replicas + optimizer states。
  2. **Micro-batch 加载**：Data loader 将 B=4 sequences（2048 tokens）分发给 8 个 DP rank。
  3. **Attention + Gate（Layer ℓ）**：各 GPU 独立执行 self-attention（DP）→ gate network（top-2 routing）→ 产出 `{input_e^g}`（GPU g 上路由到 expert e 的 token 数量，e∈[0,31]）。
  4. **Token Scheduling（与 GPU token permutation 重叠）**：
     a. All-gather：8 GPU 交换 `{input_e^g}` 信息（~32×8 integers，latency ~数 us）。
     b. LPP Solving（CPU, HiGHs）：求解 min max_g Σ x_e^g s.t. Σ_g x_e^g = load_e → 得到 `{x_e^g}`（目标 replica load）。
     c. Locality-Aware Routing（Algorithm 1）：先 route local tokens（同一 GPU 上有 replica 的 expert）→ 再 route 远程 tokens。产出 token-to-(GPU, replica) mapping。
  5. **All-to-All Dispatch**：8 GPU FineEP group 内 all-to-all（NCCL 或 DeepEP），tokens 发送到目标 GPU。
  6. **Expert FFN**：各 GPU 对收到的 tokens 执行本地 expert FFN（SwiGLU: W_gate·x → SiLU ⊗ (W_up·x) → W_down·result）。
  7. **All-to-All Combine**：反向 all-to-all，FFN 输出返回原始 GPU。
  8. **Residual + Next Layer**：MoE output + attention residual → Layer ℓ+1。
  9. **Backward**：autograd 反向传播 + DP gradient all-reduce（BF16 all-to-all + FP32 local reduction）。
  10. **Adaptive Replacement（每 ~50 iterations）**：Placement Manager 检查 load 分布 → 若 Equation 3 预测性能下降 → 生成新 asymmetric placement（greedy replica count + Monte Carlo placement）→ reinitialize（迁移 expert 参数 + optimizer states，~数百 ms）。

## MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems

- 属于Serving调度的实现是什么？实验比较什么？
  实现了一套自动化MoE系统评测流水线，在SGLang和HuggingFace Transformers中植入轻量级expert activation profiler，在每层MoE layer的路由器附近插入probe记录前向传播中的expert激活模式。基于vLLM、SGLang、MoE-Infinity、K-Transformers等serving框架评测CAP三维度（Cost/Accuracy/Performance）的权衡。实验比较：(1) 不同serving系统（SGLang vs K-Transformers vs MoE-Infinity）在Qwen3-30B-A3B上的解码延迟、硬件成本和准确率权衡（CAP雷达图）；(2) 量化vs offloading方法（SGLang-FP8, SGLang-AWQ vs MoE-Infinity）在Qwen3-235B-A22B上的吞吐、功耗和准确率权衡；(3) batch size（1-64）对expert稀疏性和实际带宽需求的定量影响（DeepSeek-V2-Lite, Qwen1.5-MoE, DeepSeek-R1）；(4) S-MBU在多节点推理（2节点×8 H20, InfiniBand 400 GB/s）的精度验证；(5) batch-size骤增压力测试（Microsoft Azure请求trace重放，Poisson分布）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-80G-SXM4（1×, 2×, 8×）, A100-80G-PCIe（1×, 4×, 8×）, H20（8×）, A6000（4×, Ada 300W）, A5000, RTX 4090（450W）；Apple M3 Max；NVIDIA Orin AGX, Orin NX；DGX-H100（10200W）。多节点环境：2节点各配备8×NVIDIA H20 GPU，400 GB/s InfiniBand互联。CPU能耗参考：AMD 777X峰值280W。

- 开源Serving框架是什么。修改了什么。
  评测覆盖六个框架：vLLM, SGLang, MoE-Infinity, K-Transformers, HuggingFace Transformers, Accelerate。核心修改：(1) 在SGLang和HuggingFace Transformers的每层MoE layer路由器附近植入轻量级probe，记录forward pass中每个expert的激活状态（布尔变量𝟙[l,i]），以此计算S_activated = n_layer × S_attn + Σ_l Σ_i 𝟙[l,i] × S_expert；(2) 构建自动化评测流水线，用户提供系统和硬件详情即可自动完成模型加载、数据集评测和CAP指标计算，基于HuggingFace leaderboard设计；(3) 探针兼容CUDA graph编译以最小化性能干扰，最大overhead仅2.7%（TTFT +8ms, TPOT +4ms）；(4) 激活模式数据持久化为activation sheet以便后续复用，避免重复profiling。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/Auto-CAP/MoE-CAP。提供预构建Docker镜像和FastAPI-based CAP分析服务。使用流程：(1) 用户指定MoE模型（如Qwen3-30B-A3B）、serving框架（如SGLang）、硬件（如4×A6000）、数据集（如GSM8K）和batch size；(2) MoE-CAP自动加载模型到指定serving框架，挂载数据集和模型存储卷；(3) 推理每个forward pass时，CAP profiler（POST /cap-profiler）在每层MoE router后记录：router输出的top-k expert索引、每个expert的激活布尔值𝟙[l,i]、当前batch size、解码延迟——这些信息用于计算S_activated；(4) 从activation sheet计算精确的S-MBU = (S_activated + S_KV) / (TPOT × B_peak)，S-MFU = (T_token × (F_attn + 2N_router + 2k_expert × N_expert)) / F_peak；(5) 同时采集硬件成本（C_hardware = C_GPU + C_CPU + C_Motherboard + C_DRAM + C_SSD，覆盖所有异构资源）和能耗成本（C_energy = (P_GPU + P_CPU + P_C2M + P_PCIe + P_NVLink) × R），合成per-token cost C_token = (C_hardware + C_energy × $/kWh) / (T_token × R)；(6) 所有请求结束后，GET /cap-results获取包含CAP雷达图的最终报告，展示Cost（$/token或W）、Accuracy（exact match/F1/win rate）、Performance（TPOT/吞吐/S-MBU/S-MFU）三维权衡对比。

## MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-Inference-Bench 不修改 vLLM 源码，而是将其作为统一推理后端，系统性地评估多种 serving 级配置和优化策略在 MoE 模型上的效果。评估的 serving 级技术包括：
    1. **GPU 并行策略（TP/PP/EP）**：在 vLLM 中配置 Tensor Parallelism（张量并行，按行/列分布权重张量）、Pipeline Parallelism（流水线并行，按层分配）、Expert Parallelism（专家并行，按 expert 分配设备）及 Hybrid Parallelism（TP+PP+EP 混合）。
    2. **Fused MoE**：vLLM 内置的融合 MoE kernel，将 expert 选择、路由和 FFN 计算融合为单个 GPU kernel，减少中间显存传输和 kernel launch 开销。
  - 实验比较：(1) Mixtral-8x7B 和 OLMoE-1B-7B 在 1-4 GPU 上使用 TP-only、TP+EP、PP+EP、PP-only 的吞吐量对比；(2) Mixtral-8x7B 有/无 Fused MoE 在不同 batch size（1/16/32/64）和 input/output length（128/256/512/1024/2048）下的吞吐量对比；(3) Llama-4-Scout-17B-16E 在 H100 vs Cerebras CS-3 上的延迟和吞吐量硬件对比。

- 硬件平台是什么，配置是什么。
  - 主要平台：NVIDIA H100 SXM5 80GB GPU（基于 TSMC 4N 工艺，80GB HBM3，50MB L2 cache，第四代 Tensor Cores，NVLink）
  - 多 GPU：1-4× H100 GPUs（用于 TP/PP/EP 并行策略 scaling 实验），节点内通过 NVLink 高带宽互联
  - 对比平台：Cerebras CS-3 cloud inference system（WSE-3 wafer-scale engine，多数量级更高的内存带宽，减少 inter-device 通信；FP8 weight storage + FP16 computation）
  - 推理框架：vLLM

- 开源Serving框架是什么。修改了什么。
  - 开源框架：vLLM（https://github.com/vllm-project/vllm），论文未修改 vLLM 源码，直接使用其内置的并行策略配置和 Fused MoE kernel。
  - 评估的 vLLM 配置：
    - **Tensor Parallelism (TP)**：`tensor_parallel_size` 参数控制，按行/列分布层权重张量到多设备。vLLM 基于 Megatron-LM 风格的 TP 实现，在 attention 和 FFN 层均支持张量切分。设备间通过 NCCL all-reduce/all-gather 通信。
    - **Pipeline Parallelism (PP)**：`pipeline_parallel_size` 参数控制，按层分配模型到不同设备。vLLM 通过 `Ray` 或 `multiprocessing` 管理各 pipeline stage 间的 micro-batch 调度。
    - **Expert Parallelism (EP)**：`enable_expert_parallel` 参数控制，将 MoE layer 中的 expert 分配到不同设备，各设备激活其持有的 expert 子集。vLLM 通过 all-to-all dispatch/combine 通信收集/分发 tokens。
    - **Fused MoE**：vLLM 内置的 FusedMoE kernel（`vllm.model_executor.layers.fused_moe`），使用 Triton/CUDA 实现，将 router 输出的 token-to-expert mapping 与 expert FFN（silu-gate + up_proj + down_proj）融合为一个 kernel，避免中间结果的 HBM 往返。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文 benchmark 代码未明确说明开源。所有评估基于开源框架 vLLM 和开源模型（Mixtral、DeepSeek、Qwen、Phi、OLMoE 均在 HuggingFace 可获取）。
  - vLLM MoE 推理从输入到 H100 硬件执行的全过程（以 Mixtral-8x7B, TP=4, Fused MoE 为例）：
    ```
    [用户输入] → vLLM API Server (POST /v1/completions)
    
    ① Tokenization & Scheduling:
       - Tokenizer 将 prompt 转为 token IDs [batch_size, seq_len]
       - vLLM Scheduler 分配 KV-cache blocks（PagedAttention）
       - 若启用 TP=4：每张 GPU 获得 1/4 的权重分片
    
    ② Prefill Phase (逐层执行, Layer 0..31):
       For each MoE layer i:
         a) Self-Attention (TP=4):
            - 每张 GPU 计算 QKV_proj(1/4 列切分) → 本地 Q,K,V
            - FlashAttention 计算本地 self-attention
            - All-reduce 聚合同步 output_proj 结果
            - KV-cache 写入 PagedAttention block table
         
         b) Router:
            - hidden_states @ W_gate → logits [B,S,8]
            - Softmax + TopK(k=2) → routed_experts, routing_weights
         
         c) Fused MoE (单 kernel 执行):
            - Kernel 输入：hidden_states [B,S,4096], W_gate_fp8[8,4096,14336], 
              W_up_fp8[8,4096,14336], W_down_fp8[8,14336,4096], routing_map
            - Kernel 内部：
              ① Token-to-expert dispatch: 根据 routing_map 重排 tokens
              ② Grouped GEMM: 合并同 expert 的 tokens → batched matmul
              ③ SiLU(gate_out) * up_out → element-wise activation
              ④ down_proj matmul → expert output
              ⑤ Weighted sum: expert_output * routing_weight → final_output
            - Fused kernel 优势：消除 ① 的中间 tensor HBM write/read
              和 ③④ 之间的 kernel launch 开销（合计节省 12-20% 延迟）
    
    ③ Decode Phase:
       - 逐 token 自回归生成，每个 token 仅需 attention(单 token Q)
         + Fused MoE forward
       - TP all-reduce 通信仅在 attention output 和 MoE output 后各一次
       - KV-cache 追加新的 K,V block
    
    ④ 输出:
       - vLLM 收集所有 GPU 输出 → detokenize → 返回 text
    ```
  - 并行策略选择的关键 insight：在 H100 上 TP 扩展效率最高（1→4 GPU 吞吐量 >2×），因为 NVLink 高带宽使 all-reduce 通信开销被充分掩盖；PP 因 stage imbalance 和同步开销几乎无加速；EP 的 all-to-all dispatch/combine 开销在小 expert activation 场景下抵消了并行收益。

## MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving

- 属于Serving调度的实现是什么？实验比较什么？
  MoE-Infinity 提出了 **Sparsity-Aware Expert Cache**（稀疏感知专家缓存），核心由 Expert Activation Matrix Collection (EAMC) 驱动的激活预测、基于预测的专家预取（prefetching）和缓存淘汰（eviction）三部分组成。在 batch size=1 的个人机器场景下，利用 MoE 模型解码阶段专家激活的高度稀疏性和请求内重用偏斜（skewed reuse），将频繁使用的专家缓存在 GPU 有限显存中，减少 PCIe 上的按需取专家 I/O。实验比较了 MoE-Infinity 与 DeepSpeed-Inference、vLLM、Ollama (Llama.cpp)、Mixtral-Offloading、BrainStorm 在多种 MoE 模型（DeepSeek-V2-Lite、Mixtral-8x7B、Switch-128x0.2B、NLLB-128x0.4B、Arctic-128x4B）和 290 个 LLM 任务（BIGBench/FLAN/MMLU）上的 TPOT（Time Per Output Token）延迟，取得了 3.1–16.7× 的延迟降低。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA RTX A5000 (24GB GPU 显存)，通过 PCIe 4.0 连接主机内存（带宽 32GB/s）。主机内存按模型规模配置：Switch 用 64GB、DeepSeek-V2-Lite 用 32GB、Mixtral 用 128GB、NLLB 用 256GB、Arctic 用 1TB。所有模型参数完整驻留在主机内存中，密集参数（attention 权重和 KV-cache）常驻 GPU 显存，专家参数按需/预取到 GPU。

- 开源Serving框架是什么。修改了什么。
  开源地址：https://github.com/EfficientMoE/MoE-Infinity。MoE-Infinity 基于 PyTorch 自建推理运行时，集成 FlashAttention 等 kernel 优化，支持 PyTorch 和 HuggingFace 格式的 checkpoint。核心修改是在标准 MoE 推理 pipeline 中插入了三层机制：
  1. **EAMC 追踪与匹配**：每次迭代记录 iteration-level EAM（iEAM，L×E 矩阵记录每层每个 expert 被路由的 token 数），累积为 request-level EAM（rEAM），并与 EAMC 中历史 rEAM 用余弦距离匹配找到最相似的激活模式。
  2. **PredictEAM 预测**：将匹配到的历史 rEAM 聚合、行归一化并施加 layer proximity decay（公式 1-(i-l)/L），生成 predicted EAM（pEAM），给出每个 expert 在未来层的激活概率。
  3. **缓存淘汰与预取**：淘汰时计算每个已缓存 expert 的 priority score = n_token / ((pEAM + ε) × (1 - layer_idx/L))，淘汰最低分 expert；预取时根据 pEAM 提前将下一层可能激活的 expert 从主机内存通过 DMA 传输到 GPU。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源（GitHub: EfficientMoE/MoE-Infinity）。以下为一次推理请求的完整执行流程：
  ```
  ① 输入: 用户提交一个 prompt（如 "Explain quantum computing"），batch_size=1，部署模型为 DeepSeek-V2-Lite (64 experts/layer, 每 token 激活约 6 experts)。
  
  ② Prefill Phase:
     - Attention 计算（常驻 GPU）+ Router 计算所有 prompt token 的 expert 分配
     - 根据 Router 输出按需从 CPU 内存 fetch 激活的 experts 到 GPU expert buffer
     - MoE forward（GPU 计算）→ 累积 iEAM → 更新 rEAM
     - EAMC 匹配：将当前 rEAM 与 EAMC 中历史 rEAM 做余弦距离匹配，找到最相似的激活模式组
  
  ③ Decode Phase（逐 token 迭代）:
     每次迭代:
     a. GPU 执行 Attention(当前 token Q, KV-cache) → Router dispatch → 确定当前层激活 expert IDs
     b. Cache lookup: 检查激活 expert 是否已在 GPU cache 中
        - Hit → 直接使用
        - Miss → 触发 FetchOnDemand: CPU→GPU 通过 pinned memory + DMA 传输 expert 参数 (PCIe 4.0 32GB/s)
     c. 预测与预取: PredictEAM(iEAM, EAMC) → pEAM → 预取下一层高概率 expert 到 GPU
        （预取与当前层 MoE 计算重叠，隐藏 PCIe 延迟）
     d. 缓存淘汰: 若 cache 满，按 priority score 淘汰最小概率 expert
        priority = n_token / ((pEAM_prob + ε) × (1 - layer_idx/L))
     e. 更新 iEAM → 累积到 rEAM
     f. MoE forward → 输出 logits → 采样下一个 token
  
  ④ 输出: 生成的 token 序列返回给用户
  
  ⑤ Post-request: rEAM 写入 EAMC，若 EAMC 容量满则替换最相似的已有 rEAM（维持多样性）
  ```
  核心作用：在单张消费级 GPU 上运行远超其显存的 MoE 大模型（如 Arctic 900GB），通过激活感知的智能缓存将 GPU 空闲等待 PCIe 传输的时间从 baseline 的 254–2073ms 降至 51ms，使单 GPU 推理延迟接近多 GPU 全内存部署水平。

## MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-Lightning 是一个面向 GPU 内存受限场景的高吞吐 MoE 批量推理系统，核心包含四个组件：
    1. **CGOPipe（GPU-CPU-I/O Pipeline Schedule）**：一种细粒度的流水线调度策略，将 GPU 计算（QKV projection + O projection + MoE FFN）、CPU 计算（attention softmax）以及四种 I/O 事件（D1: QKV DtoH、D2: Hidden H2D、D3: Weights Transfer、D4: KV cache Transfer）高效重叠。核心创新包括：
       - **CPU Attention**：基于 HRM 分析将 decode 阶段 attention 放在 CPU 执行，仅传输 hidden states（远小于 KV cache），释放 PCIe 带宽给 weight transfer。CPU attention 比 KV cache transfer 快 3-4×。
       - **Weights Paging**：将每层 weights 分 n 页（n = 微批次数），在微批次间交错传输 hidden states H2D 与 next-layer weight pages。每个 expert FFN kernel 通过 page table 访问对应 GPU 上的 weight pages。
       - **双缓冲 Weight Buffer**：分配 2 × sizeof(per-layer-weights-on-CPU) 的 GPU buffer，重叠当前层计算与下一层 weight 预取（CPU→pinned→GPU 两阶段流水线）。
       - **两步超前的 CPU attention**：Algorithm 1 中 GPU PreAttn(i, j+2) 和 CPU Attention(i, j+2) 比当前 PostAttn(i, j) 提前两个微批次，确保 GPU 不被 CPU attention 阻塞。
    2. **HRM (Hierarchical Roofline Model)**：扩展自经典 Roofline Model 的多层内存层次性能模型。引入跨层内存带宽屋顶 $B_{peak}^{j,i} \times I_x^j$ (Eq. 6) 和多个 compute roof ($P_{peak}^i$, $P_{peak}^j$)。定义 turning points $P_1$ (Eq. 9, 低于此则不值得将数据从 CPU 传到 GPU 计算) 和 $P_2$ (Eq. 10, 低于此则受限于 CPU→GPU 带宽)，以及 balance point (Eq. 11, 此时 GPU memory bandwidth × $I_{GPU}$ = CPU→GPU bandwidth × $I_{CPU}$)。基于 HRM 构建性能模型 $T = \max(comm^{cpu\_to\_gpu}, T_{cpu}, T_{gpu})$，以 MILP 搜索最优 6 元组策略 $\mathcal{P} = (N, \mu, A_g, F_g, r_w, r_c)$。
    3. **Dynamic Request Batching (Algorithm 2)**：按 input length 降序排列请求，贪婪地将最长请求分配到当前 token 数最少的微批次，使各微批次大小接近目标 μ，支持 variable-length prompt（无需 padding）。
    4. **Tensor Parallelism**：单节点内 TP 支持，各 GPU 持有权重分片，使用 all-reduce 聚合。TP 下 GPU memory capacity 和 bandwidth 随 tp_size 线性增长，HRM 搜索策略与单 GPU 相同。
  - 实验比较：(1) MoE-Lightning (p) vs FlexGen / FlexGen(c) / DeepSpeed Zero-Inference 的 generation throughput (tokens/sec)；(2) 不同 MoE 模型（Mixtral 8x7B、Mixtral 8x22B、DBRX 132B/16E）下的吞吐对比；(3) 不同 GPU 配置（S1: 1xT4 16G, S2: 1xL4 24G, S6: 2xT4 32G, S7: 4xT4 64G, S8: 2xT4+DBRX, S9: 4xT4+DBRX）下的对比；(4) 不同 workload（MTBench、HELM synthetic reasoning、HELM summarization）和 generation length（32/64/128/256 tokens）下吞吐变化；(5) Tensor Parallelism scaling（2→4 T4，Mixtral 8x22B 2.77-3.38×、DBRX 2.1-2.8× 加速比）；(6) Ablation：Optimizer policy 对比、CPU attention vs KV transfer vs MoE FFN latency、不同硬件配置下最优 policy 变化。

- 硬件平台是什么，配置是什么。
  - 六种 Setting (Table 2)：
    - S1: Mixtral 8x7B + 1xT4 (16GB HBM) + Intel Xeon 2.30GHz 24-core, 192GB DRAM
    - S2: Mixtral 8x7B + 1xL4 (24GB HBM) + Intel Xeon 2.20GHz 24-core, 192GB DRAM
    - S6: Mixtral 8x22B + 2xT4 (32GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
    - S7: Mixtral 8x22B + 4xT4 (64GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
    - S8: DBRX (132B, 16E) + 2xT4 (32GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
    - S9: DBRX (132B, 16E) + 4xT4 (64GB total) + Intel Xeon 2.30GHz 32-core, 416GB DRAM
  - GPU 互联：PCIe（单节点内多 GPU），无 NVLink。
  - Ablation study extra: 2xA100-80GB + 变化的 CPU:GPU bandwidth (100-500 GB/s)。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：基于 **vLLM** [26] (https://github.com/vllm-project/vllm) 和 **SGLang** [56] (https://github.com/sgl-project/sglang)，用 Python 和 C++ 编写。
  - 代码仓库：https://github.com/caoshiyi/artifacts/tree/asplos25（ASPLOS 2025 artifact）
  - 主要修改/新增：
    1. **Pipeline Scheduler (CGOPipe)**：在原有 zigzag execution（FlexGen 风格）基础上实现了 Algorithm 1 的 Prologue→Main Pipeline 调度逻辑——GPU PostAttn + PreAttn 与 CPU Attention 两步超前、paged weights 交错传输。
    2. **CPU GQA Kernel**：基于 Intel MKL 实现自定义 CPU Grouped Query Attention kernel（§6.2），在 CPU 端执行 attention softmax + weighted sum，替代 GPU attention + KV cache H2D 路径。
    3. **Paged Weight Manager**：实现双缓冲 weight buffer（2 × sizeof(per-layer CPU weights)）+ 页表查找 + CPU→pinned→GPU 两阶段异步传输。
    4. **HRM Policy Optimizer**：基于 MILP 的 offline policy 搜索器，输入 H/W spec、model spec、workload params，输出最优 (N, μ, A_g, F_g, r_w, r_c)。
    5. **Variable-Length Batcher**：Algorithm 2 按 input length 降序贪婪分配，无需 padding。
    6. **Tensor Parallelism 支持**：在单节点内实现 Megatron-style TP，权重分片 + all-reduce。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源**：代码公开在 https://github.com/caoshiyi/artifacts/tree/asplos25 。
  - **全流程使用例子（Mixtral 8x7B on 1xT4, MTBench workload）**：
    1. **Offline Policy Search**：HRM 性能模型输入——硬件 H（T4: GPU FLOPS=65T FP16, GPU BW=320GB/s, CPU→GPU BW=16GB/s, GPU memory=16GB, CPU memory=192GB）、模型 M（Mixtral 8x7B: 32 layers, h1=4096, h2=14336, n_e=8, k=2）、workload W（MTBench avg prompt=77, gen_len=128）。MILP 求解 min T 输出策略 (N=504, μ=36, A_g=0, F_g=1, r_w=0, r_c=0)。耗时 < 1 分钟。
    2. **Request Batching**：80 MTBench 问题被复制为数千请求。按 input length 降序排列，贪婪分配到 n_ub = N/μ = 14 个微批次，每微批次最多 μ=36 个请求。若某微批次 KV cache 总量超限，对应请求中止并加入下一批。
    3. **Prefill Stage（全 GPU）**：逐微批次在 GPU 上执行——加载该层所有 experts weights → QKV projection (GPU GEMM) → Flash Attention → O projection → MoE FFN（gate routing + Top-2 expert FFN）→ 输出 KV cache offload 到 CPU pinned memory。预填充为 compute-bound。
    4. **Decode Stage（CGOPipe）**：执行 Algorithm 1。
       - Prologue (i=1, j=1,2)：GPU 做 PreAttn(1,1/2) = LayerNorm + QKV proj → Offload QKV DtoH → CPU Attention(1, 1/2) via MKL GQA kernel → Hidden states H2D → PostAttn(1, 1/2) = O proj + MoE FFN（访问 paged weights 通过 page table）。
       - Main Pipeline (i=2..32, j=1..14)：对 layer i 和 micro-batch j，GPU 执行 LoadH(i,j) (H2D hidden states)、W_PINtoGPU(i+1,j) (从 pinned memory copy weights page j)、PostAttn(i,j) (O proj + MoE FFN)、PreAttn(i,j+2) (提前两步)。CPU 同时执行 CPUAttn(i,j+2) 和 W_CPUtoPIN(i+1,j+2) (weights CPU→pinned)。I/O 调度：D1(QKV DtoH) 和 D2+D3+D4(H2D) 方向相反可并行；D2/D3/D4 同向则通过 paging 交错执行。
    5. **输出**：每个 decode step 生成一个 token，追加到各请求序列。全部 gen_len=128 完成后返回完整输出。Throughput = total_tokens / (T_prefill + T_decode)，MoE-Lightning 达到 30.12 tokens/s（vs FlexGen 9.5 tokens/s，3.17× 提升）。

## NetMoE: Accelerating MoE Training through Dynamic Sample Placement

- 属于Serving调度的实现是什么？实验比较什么？
  - 属于训练调度优化的实现：NetMoE 在 MoE 分布式训练中动态调整训练样本（training samples）在各 GPU 上的放置位置，将跨节点（inter-node）All-to-All 通信转化为节点内（intra-node）通信，从而加速训练。
  - 实验比较：NetMoE vs FastMoE（无 placement 调整的基线）、FasterMoE（动态 expert placement + 通信-计算重叠）、SmartMoE（expert placement + 负载均衡）。端到端加速比最高 1.67×（vs FastMoE）、1.37×（vs FasterMoE）、1.33×（vs SmartMoE）。还比较了 KM 算法 vs PuLP 的求解时间，以及 All-to-All 通信的理论加速 vs 实际加速。

- 硬件平台是什么，配置是什么。
  - 4 节点集群，每节点 8 张 NVIDIA A800-SXM4-40GB GPU（共 32 GPUs）。
  - 节点内：NVLink 互联，带宽 400 GB/s。
  - 节点间：InfiniBand 互联，带宽 100 GB/s。
  - 设备内（intra-device）：内存拷贝，约 2 TB/s（不计入通信建模）。
  - 默认每节点 8 GPUs，附录中也测试了每节点 2/4 GPUs 的配置。

- 开源Serving框架是什么。修改了什么。
  - 基于 **PyTorch** 实现，自定义操作用 C++ 和 CUDA 编写。与 FastMoE / FasterMoE / SmartMoE 对比（均为开源 MoE 训练框架：FastMoE https://github.com/laekov/fastmoe、FasterMoE https://github.com/laekov/FasterMoE、SmartMoE 论文未明确说明开源链接）。
  - 主要修改/新增：
    1. **动态样本放置求解器（Dynamic Sample Placement Solver）**：将 All-to-All 通信建模为 α-β 通信模型，formulate 为组合优化问题（整数规划），再拆分为两个阶段（Stage 1: 全局跨节点优化，Stage 2: 节点内优化），通过将 ILP 转化为加权二分图匹配问题，使用 Kuhn-Munkres (KM) 算法在多项式时间 O(I³) 内求解，其中 I 为全局 batch size。
    2. **Expert Residual Inlining**：将残差连接内联到 expert 计算中（在 scatter 之后、gather 之前执行加法），保证样本放置调整后计算正确性不变（详见论文 Appendix A.1，图 8）。
    3. **CPU Offloading**：KM 算法在 CPU 上执行（因 GPU 不适合并行化该算法），求解过程通过后台线程与 All-to-All scatter + expert 计算重叠，实现零额外开销。
    4. **下一层路由预测**：将当前层的输入传入下一层 router，提前获取下一层路由结果供求解器使用。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源**：论文未明确说明开源链接，在 ICLR 2025 页面和 web 搜索中未找到公开代码仓库。
  - **全流程使用例子（MoE-GPT-S, 4 nodes × 8 A800 GPUs, I=32 samples/iteration）**：
    1. **训练配置**：GPT-2 backbone MoE 模型，序列长度 S=1024，hidden dim H=768，E=2J=16 experts（每 GPU 1 expert），K=2（每个 token 选 top-2 experts），全局 batch I=32（每 GPU 1 sample）。
    2. **前向传播（MoE Layer l）**：
       - **Gating**：每个 GPU 上的 tokens 经过 gating network，得到 routing 结果 `route ∈ N^{I×L×K}`。
       - **计算通信矩阵 num**：统计每个 sample i 需要发给每个 expert e 的 token 数量 `num_{i,e}`（Eq. 2），在 GPU 上计算后传输到 CPU。
       - **Stage 1 求解（跨节点）**：CPU 后台线程构建二分图——左侧 P 为 I 个 samples，右侧 Q 为 N 个 nodes（每个 node 可容纳 B=I/N samples，通过复制 B 次使 |Q|=I）。边权重 `W_{i,n} = c_{i,n}^{(l,gather)} + c_{i,n}^{(l+1,scatter)}`（Eq. 8 计算跨节点通信量）。KM 算法求最小权完美匹配，得到每个 sample 的目标 node。
       - **Stage 2 求解（节点内）**：对每个 node n 独立构建二分图（左侧为该 node 分配到 I/N 个 samples，右侧为该 node 上 J/N 个 GPUs，每 GPU 复制 I/J 次），KM 算法求最小权匹配，得到每个 sample 的目标 GPU。
       - **All-to-All Scatter**：各 GPU 按 routing 结果将 tokens 发送到对应 expert 所在 GPU。此操作使用当前（优化前）的 sample placement。
       - **Expert Computation**：各 GPU 上的 experts 对收到的 tokens 执行 FFN 计算。同时 CPU 后台完成求解。
       - **Expert Residual Inlining**：output = input + expert_output（在 scatter 之后直接执行，而非等 gather 之后）。
       - **All-to-All Gather（使用优化后的 SmpDev）**：token 不再按原位置返回，而是按新求解的 sample placement 目标 GPU 发送——完成动态样本放置，无需额外通信。
    3. **效果**：以 2 nodes/16 GPUs 配置为例，MoE-GPT-S 的 inter-node 通信量从 191.07 MB 降至 116.37 MB（↓39.10%），约 43.66% 的 samples 跨节点交换，91.39% 的 samples 被调整位置。KM 求解时间（I/J=4 时 0.48ms）远小于 scatter+computation 时间（7.13ms），完全被隐藏。

## Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：本文本身**没有实现或修改任何开源 Serving 框架**，而是从**模型侧**分析 MoE 模型对不同 expert offloading 策略的友好程度。核心是提出 SRP 和 SCH 两个指标量化 MoE 模型的"局部路由一致性"（即连续 token 倾向于激活相同 experts 的程度），该属性直接影响 expert offloading/caching 系统（如 LRU/LFU cache）的 hit rate。论文通过理论证明 SCH 可近似 Belady 最优缓存的命中率，并实验验证 SCH 与实际 cache 算法（LRU、LFU）hit rate 高度正相关（m=16 时 Pearson r > 0.88），证明 SRP/SCH 可作为选取/设计 expert offloading 系统的模型级参考指标。此外，论文实现了一个简单的 LRU-based expert offloading 系统（naive 版本，on-demand loading）在 TOY 模型上测量 throughput，验证局部路由一致性与 decoding 阶段 overhead 负相关（r ≈ −0.3）。
  - 实验比较：(1) 20 个 REAL 模型在不同 cache ratio ρ 和 segment 长度 m 下的 SCH，分析 SCH 随 ρ 增长的曲线形状（group 1 模型在 ρ=2 处出现拐点）；(2) SCH 与实际 cache 算法 hit rate 的 Pearson 相关性（m=4/16/64/256, ρ=0.5-3.0）；(3) SCH 与 Belady 最优 hit rate 的相对比较（Baseline TOY 模型），展示 LRU/LFU/SCH 在不同 ρ 下相对于 clairvoyant replacement 的归一化 hit rate；(4) TOY 模型在 LRU-based expert offloading 下的 decoding/prefilling throughput 与 full GPU inference 的相对 overhead。

- 硬件平台是什么，配置是什么。
  - REAL 模型路由决策收集：NVIDIA A100 PCIe 80GB GPU。TOY 模型 expert offloading throughput benchmark：单 GPU（A100）内存不足以容纳完整模型，模拟 memory-constrained 边缘设备场景（Appendix F）。具体配置：GPU 内存足够容纳 activated parameters + 计算，但不足容纳全部 experts；CPU 端有充足 flash memory。

- 开源Serving框架是什么。修改了什么。
  - 论文未基于任何开源 Serving 框架实现（如 vLLM、SGLang 等）。论文关注的是通用 expert offloading 场景下的**模型路由行为分析**，而非特定系统的实现或修改。论文的 naive LRU expert offloading 实现为自行编写，未说明基于何种框架。论文提到主流 expert offloading 系统包括 SwapMoE (Kong et al., 2024)、MoE-Infinity (Xue et al., 2024b)、EdgeMoE (Yi et al., 2025)、AdaPMoE (Zhong et al., 2025) 等作为背景参考。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源代码：https://github.com/ljcleo/moe-lrc
  - 本文分析结论如何指导专家 offloading/caching 系统设计：
    ```
    1. 模型选择（离线/部署前）：
       输入：候选 MoE 模型列表
       过程：用本文代码对每个模型在通用语料上计算 SCH(m=16, ρ=2)
       输出：选择 SCH 最高的模型部署到 memory-constrained 设备
       依据：SCH > 70%（如 LLaMA-MoE-v2）的模型可预期 LRU/LFU cache 获得 > 65% hit rate

    2. 缓存大小选择：
       输入：目标 MoE 模型的 SCH 随 ρ 曲线
       过程：找到 SCH 增长从"快速"变为"平缓"的拐点 ρ
       输出：缓存大小 = ρ × k × expert_size（k = 每 token 激活 expert 数）
       依据：大多数模型 ρ=2 时在 cache 效率（hit rate）与开销（GPU 内存）间取得最佳平衡

    3. 运行时 expert cache 策略（以 LRU 为例）：
       GPU 内存布局：
         - pinned experts: k 个（必须保留，对应每 token 激活数）
         - cached experts: (ρ-1)×k 个（LRU cache pool）
         - 其余 experts 驻留 CPU 内存

       解码阶段每步执行流程：
         a. 当前 token x 输入 router → 得到 top-k experts
         b. 如果 top-k 全部在 cached experts ∪ pinned experts 中 → GPU 直接执行
         c. 否则 → 从 CPU 加载缺失 expert 到 GPU（on-demand load），
            驱逐 LRU pool 中最久未用的 cached expert
         d. GPU 执行 expert FFN 计算
         e. 更新 LRU 访问记录

       论文核心贡献：上述系统的 cache hit rate 上限由 SCH 决定；
       高 SRP 模型（group 1，如 LLaMA-MoE-v2、OLMoE）即使最简单的 LRU
       也能获得高 hit rate，无需复杂的 prediction-based prefetching。
    ```

## Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：基于 Insight 1（prefill-decode 阶段的 expert selection 相关性），提出两种 **prefill-aware expert placement 算法**来指导 decode 阶段的 expert 分布：(1) **Remap-based placement**——保持每 GPU 的 expert 数量不变，按 roofline cost 降序排列 expert，贪心分配给负载最小的 GPU（每 GPU 容量上限 E/G）；(2) **Duplication-based placement**——保留默认连续布局（experts 0-15 on GPU 0, etc.），利用 prefill traces 复制热门 expert 到额外槽位（每 GPU 预留 R 个额外槽位），每次贪心选择能最大减少瓶颈负载 max_g load_g 的 (expert, GPU) 对。
  - 实验比较：(1) Remap 和 Dup vs Default（SGLang/Qwen 标准连续布局，experts 0-15 on GPU-0, 16-31 on GPU-1 等）、Best（oracle decode-stage 选择的最优 placement）、Worst（oracle 最差 placement）；(2) 不同 batch size（64-16,384）下的 MoE 计算时间加速比。

- 硬件平台是什么，配置是什么。
  - 8×NVIDIA H100 80GB GPU，NVLink 互联。
  - 网络：NVLink + 节点内互联。
  - 使用 SGLang 部署 Qwen3-235B（94 MoE layers, 128 experts per layer, top-8 selection）。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：**SGLang** (https://github.com/sgl-project/sglang)。
  - 修改内容：
    1. **Expert placement 接口**：通过 SGLang 的 `init_expert_location` 接口操纵 expert 在各 GPU 上的分布。
    2. **MoE backend**：使用 **DeepEP** 作为 MoE 后端，`ep_dispatch_algorithm` 设为 "dynamic"，使 tokens 均匀分布到复制 expert 的各副本上。
    3. **分布式 profiler**：在 SGLang 中插入 `cuda.Event` timers 独立测量每个 GPU 上的 attention、top-k、all-to-all 和 MoE 操作时间。
  - 开源链接：Case Study 2 代码仓库 https://github.com/zhongkaiyu/moe_exp_placement，DOI: 10.5281/zenodo.19617695。expert selection traces 开源在 https://huggingface.co/datasets/core12345/MoE_expert_selection_trace。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源**：SGLang（Apache-2.0），DeepEP（开源），Case Study 2 代码 + traces 已开源。
  - **全流程（以 Qwen3-235B 在 8×H100 上使用 Duplication-based placement 为例）**：
    1. **输入阶段**：用户通过 HTTP/gRPC 发送文本请求到 SGLang server。请求先经过 prefill 阶段——所有 input tokens 一起处理，Gate 网络记录每个 token 在每层的 expert 选择（prefill traces）。
    2. **Placement 计算**：基于收集的 prefill traces，运行 Duplication-based 算法——(a) 从 traces 计算每层每个 expert 的频率 f_{l,e}；(b) 生成默认连续布局（GPU-0: experts 0-15, ..., GPU-7: experts 112-127）；(c) 每 GPU 预留 R=1 个额外槽位，总计 128+8=136 experts per layer；(d) 贪心迭代：每次选择能最大减少 max_g load_g 的 (expert, GPU) 对，直到额外槽位用完。使用 roofline-based cost model 估算每个 GPU 的负载。
    3. **Expert 重分布**：通过 SGLang 的 `init_expert_location` 接口将新布局加载到各 GPU。DeepEP backend 使用 "dynamic" dispatch 算法确保 tokens 均匀分配到复制 expert 的各副本。
    4. **Decode 执行**：每个 decode step：(a) Attention 计算（各 GPU 处理自己的 KV cache 分片）；(b) Gate/Top-k 路由——选择每个 token 的 top-8 experts；(c) DeepEP all-to-all 通信——将 tokens 发送到目标 expert 所在 GPU；(d) MoE 计算——各 GPU 执行本地 expert 的 3 个 GEMM 操作（gate_proj, up_proj, down_proj）；(e) 第二个 all-to-all——将结果返回原 GPU；(f) 下一层继续。
    5. **性能测量**：通过插入的 `cuda.Event` timers 测量 MoE computation time（3 个 expert linear layers + all-to-all + top-k），排除 attention 时间。Remap 和 Dup 分别实现 15.5% 和 12.5% 的加速（vs Default），均在 Best（oracle）的 10% 以内。
