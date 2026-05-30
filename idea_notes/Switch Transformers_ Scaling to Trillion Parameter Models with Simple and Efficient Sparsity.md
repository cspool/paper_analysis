## Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- baseline方法是什么？
  Baseline 方法包括两类：(1) **Dense T5 Transformer**（T5-Base 223M, T5-Large 739M, T5-XXL 11B）——所有 token 共享同一套 FFN 参数，每次 forward 使用全部参数计算。随模型规模增长，FLOPs per token 同比例增加，计算开销大。(2) **Standard MoE Transformer (top-k routing, k≥2)**——使用 Shazeer et al. (2017) 的 Noise Top-k Gating，每个 token 路由到 k>1 个 expert，输出为各 expert 的 gate 值加权求和 y = Σ_{i∈T} p_i(x) E_i(x)。k>1 导致：(a) 路由计算量是 Switch 的 k 倍；(b) 每个 expert 需处理更多 token（expert capacity 增大），增加计算和通信成本；(c) 需要更复杂的 all-to-all 通信模式。
  全栈执行例子（Baseline: Dense T5-Base，224M params，TPUv3，pre-training on C4）：
  - **算法Pipeline层**：输入 token sequence X [B, 768] → Multi-Head Self-Attention (QKV 投影) → FFN: X × W_in [768, 2048] → ReLU → × W_out [2048, 768] → 输出。所有 B 个 token 共享 W_in/W_out，无需路由。单层 FFN 参数量 = 768×2048×2 ≈ 3.1M。12 层总参数 224M，FLOPs per token 固定为 124B。
  - **Serving/系统框架层**：Mesh TensorFlow 的 Data Parallelism（所有 cores 持有完整模型副本，仅 batch 拆分）或 Model + Data Parallelism（d_ff 维度拆分，增加 all-reduce 通信）。无 expert 分发逻辑。每个 TPU core 在 forward/backward 末端进行 all-reduce 梯度聚合。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 TPU XLA 编译的矩阵乘法和 All-Reduce 通信。每个 FFN 层为固定形状的 dense matmul [B,768]×[768,2048]，无动态路由 kernel。
  - **硬件架构层**：TPUv3，32 cores，每个 core 在 forward/backward 全程参与计算，无 idle。通信仅在梯度聚合时发生（all-reduce），中间激活无跨 core 通信。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Switch Transformer**，核心是 **k=1 routing**：将传统 MoE 的 top-k 路由简化为仅路由到单个得分最高的 expert。这直接解决 Baseline 的缺陷：(1) Baseline Dense 模型计算效率低——所有参数对每个 token 都执行计算，Switch Transformer 通过稀疏激活（每个 token 仅用 1 个 expert 的参数），在相同 FLOPs per token 下将参数量从 224M 扩大到 7B+，实现 7x 训练加速。(2) Baseline MoE (k>1) 通信和计算冗余——k=2 时每个 token 需两个 expert 计算，Switch 的 k=1 将 expert capacity 减半，通信量减少约 50%。(3) Baseline 训练不稳定——Switch Transformer 通过 selective precision（router 内部 float32）、reduced initialization scale（0.1x）、expert dropout 三种技术组合稳定训练 bfloat16 稀疏模型。(4) Baseline expert 负载不均——通过 auxiliary load balancing loss（α·N·Σ f_i·P_i）使 token 均匀分发到各 expert，保持 expert 利用率。
  全栈执行例子（Method: Switch-Base，7B params，128 experts，TPUv3 32 cores，训练 on C4）：
  - **算法Pipeline层**：输入 token sequence X [B, 768] → Self-Attention（未修改）→ Switch FFN: Router 计算 logits = X × W_r [768, 128] → softmax → argmax 选 top-1 expert index i → 每个 token 仅路由到 expert i → 对该 expert 执行 FFN_i(X) = ReLU(X × W_in_i)·W_out_i → 输出乘以 gate value p_i → residual add + layer norm。只有被路由到的 expert 参数参与该 token 的计算，其余 127 个 expert 保持 idle。参数量 = 128 × 3.1M（每 expert FFN）+ 共享参数 ≈ 7B，但 FLOPs per token 仍是 124B（同 T5-Base）。
  - **Serving/系统框架层**：Mesh TensorFlow Expert + Data Parallelism（Section 5.4）。32 cores 对应 32 条 data-parallel 路径，每个 core 持有 unique expert（或 experts 子集）。Router 在 local core 上计算各 token 的目标 expert index，产成 binary dispatch tensor [n, B/n, E, C] → einsum 将 tokens gather 到对应 expert → all-to-all 通信交换 tokens（shape [E, C, d_model]）→ 每个 core 上的 expert 执行 FFN → all-to-all 通信返回结果 → combine tensor 加权汇总。额外通信开销：forward 和 backward 各一次 all-to-all，传输量 = E×C×d_model × 2（来/回）× bfloat16。
  - **编译框架层**：Mesh TensorFlow (MTF) 的 SPMD 编程模型。将物理 TPU cores 映射为逻辑 mesh [n, m]，tensor 沿命名维度 shard。Switch layer 的 dispatch/combine 通过 mtf.einsum 和 mtf.reshape 实现，XLA 编译器处理底层通信生成。所有 tensor shape 在编译时静态确定（包括 expert capacity C）。
  - **Kernel调度层**：Router 使用 bfloat16→float32→bfloat16 selective precision，仅 local 计算在 float32。All-to-all 通信传输 bfloat16 精度的 expert 输入/输出 tensor。Expert FFN 为 TPU 上标准 dense matmul kernel，每个 expert 独立执行。论文未使用自定义 GPU/TPU kernel。
  - **硬件架构层**：TPUv3，32 cores 通过高速互联（ICI）连接。All-to-all 通信利用 TPU 的环形拓扑高效完成 tensor 交换。Expert Parallelism 下每个 core 持有 128/32 = 4 个 expert 的完整参数，其余 expert 的参数分布在其他 core 上，通过 all-to-all 按需获取。Extra communication 占总时间比例随 expert 数量增加而增加，但被样本效率提升所抵消。
