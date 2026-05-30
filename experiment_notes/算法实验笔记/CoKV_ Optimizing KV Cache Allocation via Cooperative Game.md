## CoKV: Optimizing KV Cache Allocation via Cooperative Game

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种基于合作博弈论中 Shapley Value 的 attention head 重要性评估方法，称为 Sliced Shapley Value (SSV)，用于评估每个 attention head 在模型推理中的合作贡献，并据此动态分配 KV cache 预算。方法分为两阶段：(1) 预计算阶段：在验证集上采样不同 coalition size 的 head 子集（H={32,64,96,128}），计算 complementary contribution (U(S) - U(N\S))，通过多次采样逼近每个 head 的 SSV 分数；(2) 推理阶段：根据归一化的 SSV 分数按比例分配 cache budget（含 local window 固定部分和 shared budget 按分数分配部分），每个 head 内部使用 SnapKV 的 attention pooling 机制选择保留的 token。实验在 LongBench 16 个数据集上与 SnapKV、PyramidKV、Ada-SnapKV、HeadKV-R2 比较不同 KV cache 大小（64/128/256/512/1024）下的准确率，同时与 Full Cache 对比。还进行了 head masking 实验（按重要性分数 mask top/low groups）和 Needle-in-a-Haystack 检索测试。

- 硬件平台是什么，配置是什么。
  推理实验：NVIDIA A100 40GB GPU；SSV 预计算：8× NVIDIA RTX 3090 GPU 服务器。FlashAttention 默认启用。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3-8B-Instruct（32层×8 KV groups via GQA，共256 groups）、Mistral-7B-Instruct-v0.2（32层×8 KV groups，共256 groups）。数据集：LongBench 16 个数据集，覆盖6类任务——Single-Doc QA (NarrativeQA, Qasper, MultiFieldQA-en)、Multi-Doc QA (HotpotQA, 2WikiMQA, Musique)、Summarization (GovReport, QMSum, MultiNews)、Few-shot Learning (TREC, TriviaQA, SAMSum)、Synthetic (PassageCount, PassageRetrieval-en)、Code (LCC, RepoBench-P)。额外使用 Needle-in-a-Haystack 测试长上下文检索能力（1k-31k tokens）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/nawei1010/CoKV。
  
  **算法 Pipeline 详解（两阶段）：**
  
  **阶段一：Head Importance Evaluation（预计算 SSV）**
  
  输入：Heads N = {h_1,...,h_n}，采样次数 M，coalition size 集合 H={32,64,96,128}
  
  1. 初始化累计矩阵 SV_{i,j} = 0, 计数矩阵 m_{i,j} = 0
  2. 对 k=1 到 M 次迭代：
     a. 随机排列 heads 得到 π^k
     b. 从 H 中随机选 coalition size j
     c. 构造 coalition S = {π^k(1),...,π^k(j)}
     d. 计算 U(S)：mask N\S 中的 heads（仅保留 local window 内的 KV），在验证集上推理得准确率
     e. 计算 U(N\S)：mask S 中的 heads，在验证集上推理得准确率
     f. 计算 complementary contribution u = U(S) - U(N\S)
     g. 对 S 中所有 head π^k(j)，更新 SV_{π^k(j),|S|} += u, m_{π^k(j),|S|} += 1
  3. 对每个 head h_i，SSV_i^H = (1/|H|) * Σ_{j∈H} (SV_{i,j} / m_{i,j})
  
  复杂度：O(M·|H|·T)，T 为单次验证集推理时间。250 samples/coalition size 时 MAE < 1/256，耗时约 20.93 小时（8×3090）。
  
  **阶段二：KV Cache Compression（推理时动态分配）**
  
  输入：shared budget B, local window size s, attention heads, SSV 分数
  
  1. Budget Allocation:
     - 归一化 SSV：NSV_i = (SSV_i - min^α(SSV)) / (max(SSV) - min^α(SSV))，α 为最低分 head 数量超参
     - α 个最低分 head 的 NSV 置 0（不分配除 local window 外的额外 cache）
     - head h_i 的 cache size: c_i = B · (NSV_i / Σ_j NSV_j) + s
  
  2. Token Selection（per head, 基于 SnapKV 机制）：
     - 计算 local window 内 tokens 的 Query: Q_i^{win} = X^{win} · W_i^Q
     - 计算 local window 对所有前缀 KV 的 attention: Ā_i = softmax(Q_i^{win} · K_i^T / √d_h)
     - 对 attention weights 做 max pooling (dim=1) 后取 mean (dim=0)，得到每个非 local window token 的重要性分数
     - 保留 top-c_i 个最高分 token 及其 KV pairs
     - 将保留的 KV 与 local window KV 拼接：{K̂_i, V̂_i} = Cat({selected KV}, {K_i^{win}, V_i^{win}})
  
  张量维度：X^{win} ∈ R^{s×d_model}, W_i^Q/K/V ∈ R^{d_model×d_h}, Ā_i ∈ R^{s×m}, d_h = d_model/num_heads, s=8 (local window), m 为前缀 token 数。
