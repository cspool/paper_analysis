## DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

- baseline方法是什么？
  **Baseline 是 Vanilla Upcycling**：将预训练 Dense 模型中的原始 FFN 层复制 N 份作为 N 个 MoE 专家的初始权重，再引入一个随机初始化的 Router 进行专家选择。训练后每个专家权重 W_i 独立更新。
  
  **Baseline 的核心缺陷**：引入大量冗余参数。以 MoE-LLaVA-Phi 为例，2.52B 额外参数（占模型的 50%+），这些参数中存在极高的冗余——训练后专家权重与初始 FFN 权重的余弦相似度 > 0.999，专家间余弦相似度也 > 0.999。这意味着 Δ_i = W_i - W_base 是微小且冗余的调整。
  
  **Baseline 全栈执行例子（以 MoE-LLaVA-Phi Vanilla Upcycling 推理一个 visual token 为例）**：
  - **算法层**：图像经过 CLIP-Large 视觉编码器 → visual token embedding → 每间隔一个 Transformer block：attention 计算 → Router 计算 top-2 softmax 路由分数 → 激活 2 个专家 FFN（每个专家是独立复制的 W_i，共 N=4 个独立权重矩阵各占 2560×10240）→ 加权求和输出 → 下一层 → LM head 预测。每个 MoE 层存储 4 个完整 FFN 权重矩阵（每个 ~26M 参数），合计 ~105M/层。
  - **系统框架层**：论文未明确说明（标准 PyTorch/HuggingFace Transformers 推理框架，标准 MoE 前向传播）
  - **编译框架层**：论文未明确说明
  - **Kernel/运行时调度层**：标准 dense GEMM kernel 执行每个激活专家的 FFN 计算；Router 的 top-k 选择为稀疏激活，但专家权重本身为 dense 矩阵
  - **硬件架构层**：NVIDIA A100 80GB GPU 执行标准 CUDA kernel

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：DeRS (Decompose, Replace, Synthesis) 范式，基于 upcycled MoE 专家共享相同初始权重 W_base 的特性，将 N 个专家重构为一个共享基础权重 + N 个轻量 delta 权重的形式，从而消除参数冗余。
  
  **具体设计如何解决 Baseline 缺陷**：
  
  1. **Decompose（分解）**：观察到 W_i = W_base + Δ_i，且余弦相似度 > 0.999 表明 Δ_i 是微小调整。将专家权重显式分解为共享部分和增量部分，使得冗余的 Δ_i 可以被独立压缩。
  
  2. **Replace（替换）**：用轻量表示替换冗余的 Δ_i：
     - DeRS Compression 中：对已训练的 Δ_i 应用后处理稀疏化（随机 drop + rescale）或量化（降低位宽），MoE 层参数从 4×K 降至 K+4×k
     - DeRS Upcycling 中：从训练开始就用稀疏矩阵（紧凑索引+值向量）或低秩矩阵（A@B）表示增量权重，训练参数从 N·d·d_h 降至 d·d_h+N·r·(d+d_h)，实现高达 2270× 参数减少
  3. **Synthesis（合成）**：推理时按需合成 Ŵ_i = W_base + F(Δ_i)，不增加推理延迟的额外开销。
  
  **关键创新点**：这是首次利用 upcycled MoE 特有的"同源初始化"特性进行专家参数去冗余。Vanilla MoE 从 scratch 训练时各专家随机初始化，无法应用此分解方法。
  
  **论文方法全栈执行例子（以 DeRS-LM Upcycling + MoE-LLaVA-Phi 推理一个 visual token 为例）**：
  - **算法层**：图像经过 CLIP-Large 视觉编码器 → visual token embedding → 每间隔一个 block：attention 计算 → Router 计算 top-2 路由分数 → 对于被选中的 2 个专家：
    1. 合成专家权重：W_i = W_shared + A_i @ B_i（A_i: 2560×1, B_i: 1×10240，低秩分解，共仅 ~12.8K 参数 vs 26M）
    2. 使用合成权重执行 FFN 计算
    → 加权求和输出 → 下一层。每个 MoE 层存储 1 个共享 W_shared (26M) + 4×(A_i+B_i) (~51K×4=0.2M)，合计 ~26.2M/层 vs Vanilla 的 ~105M/层。
  - **系统框架层**：论文未明确说明（标准 PyTorch 训练/推理，使用 torch.scatter 实现稀疏矩阵映射，低秩矩阵使用标准 matmul）
  - **编译框架层**：论文未明确说明
  - **Kernel/运行时调度层**：W_shared + A_i@B_i 的合成操作为轻量级加法/矩阵乘法，可在 GPU 上高效执行；稀疏矩阵版本需要在推理时从紧凑向量重构为 sparse/dense 矩阵（torch.scatter）
  - **硬件架构层**：NVIDIA A100 80GB GPU；推理内存从 Vanilla 的 10.5G 降至 DeRS-LM 的 5.9G (43.8% 减少)
