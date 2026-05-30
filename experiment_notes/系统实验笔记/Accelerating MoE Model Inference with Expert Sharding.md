## Accelerating MoE Model Inference with Expert Sharding

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 MoEShard，一个基于 PyTorch 的 MoE 推理系统，通过 **expert tensor sharding** 替代传统 expert parallelism。核心思想：不再将完整的 expert 分配给不同 GPU，而是将每个 expert 的矩阵 W_i（列切分）和 W_o（行切分）切分到所有 GPU 上，每个 GPU 持有所有 expert 的 shard。所有 GPU 处理所有 token 的 partial computation，最后 pointwise 求和得到等价完整输出，实现 perfect load balancing，无论路由分布如何倾斜都不产生 GPU 空闲或 token dropping。

  实验比较：
  - **Per-layer latency**: MoEShard vs DeepSpeed-MoE (expert parallelism)，batch=250, seq=120, 128 experts → MoEShard 41.5~43.5ms vs DeepSpeed 177~180ms，达 4.25× 加速
  - **TTFT varying experts (8→256)**: MoEShard vs DeepSpeed，fixed batch=250, seq=120, skew (k_r=10%, α_r=0.6) → 峰值加速 6.45× (64 experts), 最低 2.39× (256 experts，因 DeepSpeed CF 丢 token)
  - **TTFT varying batch size (10→450)**: 128 experts → 小 batch(10)时 MoEShard 慢于 DeepSpeed, batch=100 起超越，batch=450 时达 6.24× 加速，接近线性增长
  - **Ablation (with/without MegaBlocks sparse MM)**: expert≥64 时 MegaBlocks 版更优；batch 变化时 MegaBlocks 版始终优于无 MegaBlocks 版

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100 GPU（每卡 80GB HBM），NVLink 互联（双向 600 GiB/s），同一计算节点。CPU: AMD EPYC 7543 32-core @ 3.7GHz，PCIe 连接。CUDA 12.6。

- 开源Serving框架是什么。修改了什么。
  **Baseline 框架**: DeepSpeed-MoE (https://github.com/microsoft/DeepSpeed)，使用 expert parallelism (EP)，每 GPU 持有若干完整 expert，router 分配 token 后通过 all-to-all scatter/gather 通信路由 token 到对应 GPU 执行 expert 计算。DeepSpeed 默认使用 capacity factor (CF) 限制每 expert token 数（实验固定为 min(|E|, 50)），超限 token 被丢弃。

  **MoEShard 开源**: https://github.com/sacs-epfl/moe-inference，Python 3 + PyTorch 实现。

  **MoEShard 修改的核心逻辑**（相对于传统 EP）:
  1. **Expert Sharding 替代 Expert Parallelism**: 将每个 expert 的 W_i ∈ R^(h_i×h_o) 列切分为 |G| 份、W_o ∈ R^(h_o×h_i) 行切分为 |G| 份，每 GPU 持有所有 expert 的 shard。Forward pass 每 GPU 计算所有 token 对所有 expert 的 partial output (x · W_i^g · W_o^g)，最后 all-reduce 求和得到完整结果。
  2. **Token 全复制而非路由**: Step 2-3 中所有 GPU 互相发送 metadata（每 expert token 数）和全部 token（每 GPU 发送 ≈88 MiB for batch=250,seq=120,h=768），由 NVLink 高速带宽吸收（~0.15ms）。
  3. **Step 5 Gather + Pointwise Aggregation**: 各 GPU 将计算的 partial output 发回源 GPU，按元素求和恢复完整 token 输出。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  **MoEShard 推理全过程（以 4 GPU, 128 expert Switch-Base encoder, batch=250, seq=120, h=768 为例）**:

  1. **输入/Token Routing**: batch tokens x ∈ R^(250×120×768) 进入 MoE block。Self-attention 层已在此前完成（复制在所有 GPU 上）。每 GPU 独立执行 ROUTER(x) → m_expert（token→expert 映射）。Router 为 Switch Transformer 的 top-1 gating（或自定义 skew 控制 router）。

  2. **Metadata Exchange**: 每 GPU 按 expert 分组 token → 统计 per-expert token count m_sizes (size=|E|=128) → all-to-all broadcast metadata。每 GPU 现在知道每张卡对每个 expert 有多少 token。

  3. **Token Scatter**: 每 GPU 将其所有 input tokens（concatenated）发送给所有其他 GPU（all-to-all scatter）。接收后组织为 W[g][e]：来自 GPU g 且目标 expert e 的 token 集合。以 batch=250,seq=120,h=768, 4B/element 计，每 GPU 发送 ≈88 MiB，NVLink 3.0 600 GiB/s 下耗时 ~0.15ms。

  4. **Expert Computation (Sharded)**: 每 GPU 遍历 expert e∈E，加载该 expert 在 *本 GPU rank* 的 shard: W_i^g (列 shard, h_i × h_o/|G|) 和 W_o^g (行 shard, h_o/|G| × h_i)。对分配给 expert e 的 token 执行 x · W_i^g · W_o^g → partial output y_g。优化：(a) 将同一 expert 的来自所有 GPU 的 token concatenate，减少 kernel launch 从 |E|×|G| 到 |E|；(b) 使用 MegaBlocks block-sparse MM 将所有 expert shard 计算融合为单次稀疏矩阵乘法。

  5. **Token Gather + Aggregation**: 每 GPU 将其计算的 W[g] (对 GPU g 原始 token 的 partial output) 发回 GPU g。GPU g 收到所有 partial outputs y_g 后 pointwise sum → x_final。等价于未经 sharding 的完整 expert 输出。

  6. **输出**: 聚合后的 token 作为当前 MoE block 输出，传递给下一个 transformer block。

  关键特性：所有 GPU 的计算量完全相等（均处理全部 token × 全部 expert shard），无论路由分布多倾斜，无 token dropping，无 GPU idle time。
