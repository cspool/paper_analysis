## APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 APB，一种结合序列并行与近似注意力的分布式长上下文推理框架。核心设计：(1) Context Splitting：将长文档按 host 数量均分，每 host 持有 local context block + anchor block（包含 query + 文档开头 token）；(2) Block Compression：用 LOCRET 的 retaining heads（训练的小型 MLP）在每个 host 上独立压缩 KV cache，提取 top-l_p 个最重要的 KV pair，无需全局序列视图；(3) Communication：通过 AllGather 在各 host 间共享压缩后的 KV cache，构造 passing block；(4) Computation：用修改后 attention mask 的定制 FLASHATTN kernel 执行 [anchor, passing, local] 三部分联合注意力计算。实验比较与 FLASHATTN、ULYSSES、RINGATTN、MINFERENCE、STARATTN 在 ∞Bench 和 RULER 基准上的任务性能与推理速度。APB 实现最高 9.2×（vs FLASHATTN）、4.2×（vs RINGATTN）、1.6×（vs STARATTN）加速，无观测性能退化。

- 硬件平台是什么，配置是什么。
  8× NVIDIA A800-80GB GPU（NVLink 3.0 互联），搭配 104 核 Intel Xeon Platinum 8470 CPU，跨机通信使用 HDR InfiniBand，运行 CentOS Linux 7 (Core)。FLASHATTN 和 MINFERENCE 实验在单 GPU 上进行，其余方法在 8 GPU 上进行。Yi-34B 因模型较大使用两台机器（layer 均分）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-instruct、Qwen-2.5-14B-instruct、Yi-34B-200K、Llama-3-8B-Instruct-Gradient-1048k（支持 1M 上下文）。
  数据集/Benchmark：
  - ∞Bench（10 个任务，平均上下文 >100K tokens）：Retrieve.PassKey、Retrieve.Number、Retrieve.KV、En.Sum、En.QA、En.MC、En.Dia、Zh.QA、Code.Debug、Math.Find
  - RULER（13 个任务，可控上下文长度）：Single NIAH 1/2/3、Multi-keys NIAH 1/2/3、Multi-values NIAH、Multi-queries NIAH、Variable Tracking、Common Words Extraction、Frequent Words Extraction、Question Answering 1/2

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/thunlp/APB

  **算法 Pipeline 详解（4 阶段）**：

  **Stage 1: Context Splitting（上下文切分）**：
  输入序列 t = {d, q}（文档 d 和查询 q），d 按 H 个 host 均分，每 host 持有 block B_h（长度 l_b = l_d / H）。每 host（除 host 1）的 B_h 前 prepend anchor block A = {q_1, ..., q_{l_q}, d_1, ..., d_{l_a}}（包含 query + 文档首部 l_a 个 token）。APB 使用远小于 STARATTN 的 anchor：l_a = l_b/4 或 l_b/8（STARATTN 为 l_a = l_b）。

  **Stage 2: Block Compression（块压缩）**：
  在每层每 host 上，用 retaining heads R（小型 MLP，中间维度 1024）对 local KV cache 打分：
  $$s_1, \cdots, s_{l_b} = \mathcal{R}([\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h])$$
  取 Top-l_p 个分数最高的 KV pair 作为压缩块：
  $$\{s_{i_1}, \cdots, s_{i_{l_p}}\} = \text{Top-}l_p(s_1, \cdots, s_{l_b})$$
  $$\mathbf{B}_h^C = (\mathbf{K}_h^C, \mathbf{V}_h^C) = (\{\mathbf{k}_h[i_1], \cdots, \mathbf{k}_h[i_{l_p}]\}, \{\mathbf{v}_h[i_1], \cdots, \mathbf{v}_h[i_{l_p}]\})$$
  Retaining heads 基于 LongAlign 数据集前 3000 样本训练 3000 steps（lr=5e-4, AdamW, batch_size=1, 最大输入长度 10240）。

  **Stage 3: Communication（通信）**：
  对压缩后的 KV cache 执行 AllGather（两次，分别对 K^C 和 V^C）：
  $$(\mathbf{K}_{[1:H]}^{C}, \mathbf{V}_{[1:H]}^{C}) = \text{AllGather}(\mathbf{K}_{h}^{C}, \mathbf{V}_{h}^{C})$$
  构造 passing block P_h = (K_p^C, V_p^C) = 前一 host 的压缩块拼接（忽略后续 host）。

  **Stage 4: Computation（计算）**：
  每 host 的 context layout = [A, P_h, B_h]，用修改的 attention mask M' 计算：
  $$\mathbf{Q}^{(i)} = [\mathbf{Q}_a^{(i)}, \mathbf{Q}_h^{(i)}], \quad \mathbf{K}^{(i)} = [\mathbf{K}_a^{(i)}, \mathbf{K}_p^C, \mathbf{K}_h^{(i)}], \quad \mathbf{V}^{(i)} = [\mathbf{V}_a^{(i)}, \mathbf{V}_p^C, \mathbf{V}_h^{(i)}]$$
  $$[\mathbf{A}_a^{(i)}, \mathbf{A}_h^{(i)}] = \text{softmax}\left(\mathbf{M}' \odot \frac{\mathbf{Q}^{(i)} \mathbf{K}^{(i)\top}}{\sqrt{d_m}}\right) \cdot \mathbf{V}^{(i)}$$
  Passing block 在 Attention 计算后丢弃，不参与 FFN。

  **FLOPs 公式**：
  $$\text{APB FLOPs/forward} = L \times [4(1 + \frac{1}{g} + \frac{0.5n}{Hd} + \frac{1.5I}{d})\frac{n}{H}d^2 + 4(H-1)(1 + \frac{1}{g} + \frac{0.5(n/H + l_a)}{d} + \frac{1.5I}{d})(\frac{n}{H} + l_a)d^2 + l_p H(H-1)(\frac{n}{H} + l_a)d]$$

  **Decoding 阶段**：使用 STARATTN 的 stage-2 精确注意力，各 host 独立计算 partial attention，通过 Gather + MergeScore（利用 online softmax 的 lse）合并为全局 attention score。

  **超参数配置（默认 128K 输入，H=8）**：
  - l_b = 16K, l_a = 4K, l_p = 2K
  - 不同输入长度配置见 Table 8

  **Retaining Head 训练**：
  - 数据集：LongAlign（前 3000 样本）
  - 优化器：AdamW（lr=5e-4，β1=0.9，β2=0.95，linear scheduler，warmup=300 steps）
  - Loss：regression loss + smoothing loss（α=0.0025）
  - Gradient clipping：0.5
