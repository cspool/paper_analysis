## PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  POWERATTENTION 提出一种新型静态稀疏注意力模式，核心思想是让每个 token 仅关注距离为 2 的幂次的位置（power-of-2 distances），配合局部滑动窗口和 sink tokens。具体实现：(1) Power-of-2 Mask：`mask_power = (blk_qk & (blk_qk - 1)) == 0`，即只保留 query 与 key 的 block 索引差值为 2 的幂次的注意力连接；(2) Sliding Window：5-block 局部窗口，保留局部上下文信息；(3) Sink Tokens：1 block 的初始 token 作为 attention sink；(4) Causal Mask：保证自回归因果性。最终 mask = causal & (mask_window | mask_power | mask_sink)。所有模式使用 256-token blocks 对齐 GPU 内存访问。稀疏度约 94%（所有 pattern 保持一致）。理论保证：在 d 层 LLM 中，每个 token 可以访问到距离 ≤ 2^d 的所有 token（指数级感受野增长），且每个 token 的出度不超过 log n。

  实验比较：(a) POWERATTENTION vs Sliding Window / Stride Slash / Dilated Attention / LongNet / Full Attention on PG19 语言建模困惑度（4k-32k context）；(b) Passkey Retrieval 检索任务（32k 和 64k 扩展 context）；(c) RULER benchmark 13 项子任务（NIAH, Variable Tracing, Aggregation, QA）在 4k/8k/16k/32k context 下的平均分；(d) 端到端延迟对比 Full Attention 和 MInference（128K context, 1024 decode steps）；(e) 信息流探针实验（probe analysis）：在每层每位置训练 logistic classifier 检测 passkey 信息传播。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU。基础模型 Qwen2-7B（28 layers, 32K 原生 context length）。训练配置：SlimPajama 1B tokens continued pre-training，ChatQA 2 fine-tuning for long context tasks。POWERATTENTION 超参：5-block local window (5×256=1280 tokens)，1 block sink tokens (256 tokens)，4 个 power-of-2 slash tokens（总计每 token 最多关注 10 blocks = 2560 tokens，即 ~94% 稀疏度 @32K）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2-7B（Yang et al., 2024a），原生 32K context length，28 layers。数据集：(a) 预训练：SlimPajama（Soboleva et al., 2023），1B tokens continued pre-training；(b) 微调：ChatQA 2（Xu et al., 2024），含 long-range dependencies 的自然监督信号；(c) passkey retrieval 合成数据（课程学习：4K→32K，每阶段 200 steps）。Benchmarks：(a) PG19 test set（语言建模困惑度，4k/8k/16k/32k）；(b) Passkey Retrieval（32k/64k）；(c) RULER（Hsieh et al., 2024），14 子任务四类：Needle-in-a-Haystack (NIAH), Variable Tracing (VT), Aggregation (Agg.), Question Answering (QA)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确提供开源 GitHub 链接。算法使用 PyTorch FlexAttention（Dong et al., 2024）实现 mask 定义，Triton（Tillet et al., 2019）结合 RingAttention（Liu et al., 2024）用于序列并行训练以扩展到更长的序列。核心张量计算 pipeline：

  **Step 1 - 构建 Attention Mask（POWERATTENTION 核心）**：
  ```python
  # q_idx [M, 1], kv_idx [1, N] 为 token 索引
  block_size = 256  # CUDA block size
  # Sink token mask: 前 block_size 个 token 全局可见
  mask_sink = kv_idx < block_size  # [1, N]
  # Sliding window mask: 5-block 局部窗口
  blk_qk = q_idx // block_size - kv_idx // block_size  # [M, N]
  mask_window = blk_qk < 5  # [M, N]
  # PowerAttention mask: 仅 block 距离为 2 的幂次
  mask_power = (blk_qk & (blk_qk - 1)) == 0  # [M, N]
  # 因果性 + 组合
  causal = q_idx >= kv_idx  # [M, N]
  mask = causal & (mask_window | mask_power | mask_sink)  # [M, N]
  ```

  **Step 2 - 感受野指数扩展原理**：
  对于距离 d（用二进制表示），d 中为 1 的 bit 最多有 log n 个。设 k₁, k₂, ..., k_m 为 d 的二进制中 1 的位置，则路径为：
  ```
  i → (i - 2^{k₁}) → (i - 2^{k₁} - 2^{k₂}) → ... → j
  ```
  路径长度 = d 的二进制表示中 1 的个数 ≤ log n。因此 d 层内可到达距离 ≤ 2^d 的所有 token。

  **Step 3 - 训练流程**：
  1. Continued pre-training: SlimPajama 1B tokens, Qwen2-7B base model
  2. Fine-tuning: ChatQA 2 data（含跨窗口的 long-range dependencies）
  3. 对于 RULER 评估，采用 hybrid architecture：每 7 层中保留 2 层 Full Attention，其余 5 层使用 POWERATTENTION
