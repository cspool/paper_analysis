## Serverless MoE Inference（无服务器 MoE 推理）

术语是什么？
Serverless MoE Inference 是将 serverless 计算范式（弹性伸缩、按使用量付费、无服务器管理）应用于 Mixture-of-Experts 大语言模型推理的架构模式。核心挑战：MoE 模型的大量 experts 在 serverless 按内存×时间计费模型下产生高额成本——即使大部分 experts 未被激活，它们仍占用内存并持续产生费用。传统 expert offloading 方法（fMoE, HOBBIT）将所有 inactive experts 缓存在 CPU 内存中，仍需要持续分配的大内存池，未能消除 cost inefficiency。

Remoe 提出了首个面向 serverless 的 MoE 推理系统，核心架构决策：
1. **异构部署**：非 expert 模块（Attention、Gate）部署在 GPU，expert 模块部署在 CPU（GPU 价格远高于 CPU）
2. **Local/Remote Expert 分区**：高频 "local experts" 与主模型同容器常驻 CPU；低频 "remote experts" 部署为独立 serverless function（Kubernetes Pod），仅在被激活时启动和计费
3. **SPS 预测**：请求到达时通过 prompt 语义相似度预测 expert 激活模式，避免 token-by-token 在线预测导致的频繁冷启动
4. **MMP 预分配**：基于 Hoeffding 不等式的最坏情况上界预分配主模型资源以满足 SLO

从系统架构角度拆解术语：
Remoe 的请求处理全流程：
1. 请求到达 → Pre-processing layer 完成 tokenization
2. SPS 算法查询多叉聚类树，用 Soft Cosine Similarity 找到 top-α 语义最相似历史 prompt，softmax 加权预测 expert 激活矩阵 S_pred [L, K_l]
3. MMP 算法用 Theorem 1 上界（per-expert tokens ≤ √(3n)/2 + n/K_l, 95% confidence）计算满足 TTFT/TPOT 的最小主模型内存 w_v
4. Kubernetes 调度主模型 Pod（GPU+CPU），并行启动冷启动
5. Remote Experts Selection：计算每个 expert 的 utility score u_{l,k} = N_in · s̃_{l,k} + N_out · N_topk · s̃_{l,k}，选 b×K_l 个 lowest-utility experts 为 remote
6. Lagrangian 对偶优化：将离散内存规格连续化 → 构造指数衰减拟合 T̃ = θ₁exp(-θ₂ỹ) + θ₃ → 凸性分析（Theorem 2）→ KKT 求解每层 remote expert 最优内存 ỹ_l
7. LPT 算法划分 remote expert 集到 z_l 个 replica → Kubernetes 创建 remote expert Pods（CPU-only）
8. 推理执行：GPU 执行 Attention + Gate → token embedding 通过 gRPC 发送到 remote expert Pods → CPU 并行执行 local + remote experts → 合并输出

Serverless 成本模型（Remoe）：总成本 C = C_loc + C_rem，其中 C_loc = (PT+GT)[c_g·M_g + c_c·Σw_v·m_v]（主模型 GPU+CPU 内存×时间），C_rem = Σ(remote expert Pod 的 CPU 内存 × 执行时间)，且 remote experts 仅在被激活的 token 期间计费。

术语一般如何实现？如何使用？
- 实现平台：Kubernetes（容器编排）+ C++ LibTorch（expert 计算）+ gRPC（GPU↔CPU 数据传输）。要求 token size（7-14KB for BF16）远小于 payload 限制（AWS Lambda: 6MB），避免需要中间存储（S3）引入延迟。
- 适用场景：bursty workload 的 LLM 推理服务；多租户平台需要 pay-per-use 计费；GPU 资源稀缺/昂贵时利用 CPU 执行 expert 计算。
- 当前限制：依赖 serverless 平台的理想化假设（可预测冷启动时间、稳定网络延迟）；对 Attention 计算密集的模型（如长序列）GPU 仍为瓶颈。

涉及论文标题：
- Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing
