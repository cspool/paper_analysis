## Mixture of A Million Experts

- baseline方法是什么？
  Baseline 包括三种：**Dense FFW**（标准 Transformer 中每个 token 激活所有 FFW 参数）、**Coarse-grained MoE**（expert-choice routing, 128 experts, 每个 expert 大小等于原 dense FFW）、**PKM**（product key memory, 检索记忆向量而非可学习函数）。

  以 Dense FFW baseline 的全栈执行路径为例（一个 token 的推理）：
  - **算法层（Dense FFW）**：输入 x ∈ R^(d_model)，经过两层 MLP：h = W_1 x + b_1, y = W_2 σ(h) + b_2。所有 d_model × d_ffn × 2 个参数均参与计算，FLOPs = O(d_model × d_ffn)。参数数量 P 与计算量 C 线性耦合：C ∝ P。增加模型容量直接导致计算开销同比例增加。
  - **系统框架层**：标准 Transformer 训练/推理框架（如 JAX/FLAX 或 PyTorch），FFW 层作为密集矩阵乘法执行。batch 中的每个 token 独立执行相同的 FFW 计算，无稀疏激活。内存占用 = 模型参数 + 激活值 × batch_size × seq_len（激活内存随 batch/seq_len 线性增长）。
  - **编译框架层**：标准 XLA/TVM 编译，将 FFW 矩阵乘法映射为 GPU/TPU 上的 matmul kernel。无特殊编译优化。
  - **kernel 调度层**：使用高度优化的 BLAS 库（cuBLAS/TPU matmul）执行密集矩阵乘法。所有 d_model × d_ffn 权重在每个 token 上都被读取和计算，无 kernel 级稀疏优化。
  - **硬件架构层**：标准 GPU/TPU，矩阵乘法在 Tensor Core/TPU MXU 上执行。内存带宽需求 = 读取 W_1, W_2 + 写入激活值，随模型增大线性增长。
  - Baseline 核心缺陷：
    1. **计算与参数的线性耦合**：Dense FFW 的 FLOPs 和激活内存随 hidden width 线性增长。增大模型容量（更多参数 P）必然导致更高的计算开销。
    2. **Coarse-grained MoE 的专家数量受限**：token-choice 和 expert-choice routing 均需在 N×M 的 gating score 矩阵上执行 top-k 操作，路由复杂度至少 O(N)，限制专家数量通常 < 128。Fine-grained MoE scaling law 预测更高粒度（更多更小专家）可带来更好性能，但现有 MoE 无法扩展到百万级专家。
    3. **PKM 检索记忆向量而非函数**：PKM 检索的是静态记忆向量（不依赖输入变化），而非输入依赖的 expert 网络。记忆向量的表达能力远弱于可学习函数，无法利用输入信息动态调整输出。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PEER 通过 **Product Key 检索路由 + 单神经元 Expert + Multi-Head 检索** 三层设计解决上述缺陷。全栈执行路径（一个 token 通过 PEER 层，以 N=1024², h=8, k=16 为例）：

  - **算法层（Product Key + Singleton Expert）**：
    1. **Query 投影**：输入 x ∈ R^(d_model) 经 h=8 个独立 query network 映射为 8 个 query 向量 q^i(x)，每个 query 维度为 d（product key 维度）。
    2. **Product Key 检索**：每个 query 拆分为 q_1, q_2 ∈ R^(d/2)。在两个子密钥组 C, C'（各 1024 个 d/2 维向量）上分别 top-k=16 检索，获 16²=256 个候选 product keys。计算 q_1^T c_i + q_2^T c'_j 并再次 top-k=16 选出最终 expert 索引。**路由复杂度 O((√N + k²)d) = O((1024 + 256)d)，而 naive 方法为 O(10^6 × d)，加速约 1000×**。
    3. **Expert 计算**：每个 expert e_i(x) = σ(u_i^T x) v_i —— 单神经元（d_expert=1），仅需 2×d_model 个参数。通过 Embedding lookup 检索 u_i, v_i 权重。h×k = 128 个 active expert 共享 expert pool，动态组装成等效 128 神经元 MLP：f(x) = Σ_i Σ_j g_j(x) σ(u_j^T x) v_j。
    4. **Router 权重聚合**：scores 经 softmax 归一化后与 expert 输出相乘求和。总 active 参数 P_active = hk × 2d_model，总参数 P = N × 2d_model = 10^6 × 2d_model。
  - **系统框架层**：PEER 层可直接替换 Transformer 中任意 FFW 层，替换后在 JAX 中以 embedding lookup + einsum 操作执行。训练时 expert 权重存储在 Embedding 层中（类似大词表），通过索引检索而非全量矩阵乘法。batch 扩展时，只有 P_active（active 参数）随 batch_size × seq_len 增加激活内存，P（总参数）仅存储一份。
  - **编译框架层**：论文未明确说明（使用 JAX/XLA 的 embedding lookup + einsum 算子）。论文指出高效实现需要 specialized hardware kernels 加速 embedding lookup 与 einsum 的融合。
  - **kernel 调度层**：核心操作为 embedding lookup（类似大词表查表）和 batched einsum（小矩阵批量乘法）。与 dense FFW 的大矩阵乘法（matmul）不同，PEER 的 compute pattern 是大量独立的小 inner product + 加权求和，对 GPU/TPU 的 Tensor Core 利用率可能偏低。论文未实现专门的 kernel 优化。
  - **硬件架构层**：标准 GPU/TPU。总参数 P（10^6 × 2d_model）存储在高带宽内存中，每个 token 仅需读取 hk 个 expert 的权重（128 × 2d_model），推理时内存带宽需求远低于同参数量的 dense 模型。

  - 对比 baseline 的改进映射：
    - **Dense FFW 的计算-参数耦合 → PEER 解耦**：Dense: P_active = P = O(d_model × d_ffn)。PEER: P_active = hk × 2d_model = 128 × 2d_model（固定），P = N × 2d_model = 10^6 × 2d_model（独立扩展）。增加 N 仅增加参数存储（无额外计算），提高模型容量而不增加 FLOPs。
    - **Coarse-grained MoE 的 O(N) 路由 → PEER 的 O(√N) 路由**：token-choice/expert-choice 须在 N×M gating matrix 上 top-k（min O(N log k)），限制 N < 128。PEER 的 product key 将检索分解为两个 √N 候选集的笛卡尔积，O(√N + k²) 复杂度，支持 N ≥ 10^6。
    - **PKM 的静态记忆 → PEER 的输入依赖函数**：PKM 检索记忆向量 v_i（固定值），输出为 Σ g_i v_i。PEER 检索 expert 函数 e_i(x) = σ(u_i^T x) v_i，输出为 Σ g_i σ(u_i^T x) v_i —— 多了 u_i^T x 的非线性变换，使每个 expert 的输出依赖输入 x。**等价于从"检索记忆"升级为"检索可学习函数"**。
    - **Shared expert 参数 → 知识迁移和参数效率**：multi-head retrieval 共享同一 expert pool，不同 head 可以检索到相同或不同的 expert，隐式实现 expert 间 hidden neuron 共享，提升参数效率和知识迁移。
    - **Expert 负载不均衡 → Query BatchNorm**：直接使用产品密钥可能导致某些 expert 被频繁选中、其他闲置。在 query 上添加 BatchNorm 层使 query 分布更均匀，expert 使用率接近 100%（表 2），unevenness（KL 散度）显著降低。
