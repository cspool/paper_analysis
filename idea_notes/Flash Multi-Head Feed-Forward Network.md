## Flash Multi-Head Feed-Forward Network

- baseline方法是什么？
  **标准Llama-like SwiGLU FFN**：在Transformer block中，FFN层使用单一全连接SwiGLU结构——SwiGLU(X) = (SiLU(X·W_gate) ⊙ (X·W_up)) · W_down。输入X ∈ R^{L×d_model}先通过W_gate和W_up投影到d_ff维（d_ff ≈ 8/3·d_model），计算element-wise gating后通过W_down投影回d_model。FFN被视为"单head"的对参数attention——Q attend over W_1 retrieve from W_2。没有多head分解，所有hidden dimension共享同一套参数。

  全栈执行例子（Llama-like SwiGLU FFN, 370M, d_model=1024, d_ff=2752, L=4096, H100 GPU）：
  - **模型推理算法层**：标准SwiGLU FFN。gate = SiLU(X·W_gate^T) ∈ R^{4096×2752}，up = X·W_up^T ∈ R^{4096×2752}，hidden = gate ⊙ up（element-wise），output = hidden·W_down ∈ R^{4096×1024}。单路径推理（"implicit thinking"的"greedy search"——Chen et al. 2025类比）。
  - **系统框架层**：PyTorch eager execution或torch.compile。FFN层作为nn.Module，由transformers库调用。论文未说明特定serving框架修改。
  - **编译框架层**：论文未明确说明。cuBLAS为GEMM提供高度优化的kernel。
  - **kernel调度层**：cuBLAS GEMM kernel执行三次矩阵乘法：(1) X×W_gate^T (4096×1024×2752)，(2) X×W_up^T (4096×1024×2752)，(3) (gate⊙up)×W_down (4096×2752×1024)。中间激活(gate⊙up) ∈ R^{4096×2752} vollständig materialized in HBM——需先写后读，成为I/O瓶颈。cuBLAS对此有高度优化的数据reuse策略和cache hit。
  - **硬件架构层**：NVIDIA H100 GPU。Tensor core执行FP16/BF16 GEMM。HBM↔SRAM间传输tile。

  Baseline缺陷：
  - (a) **单路径推理限制表达力**：FFN的单一d_ff维中间表示可视为"implicit thinking"的单路径搜索（Chen et al. 2025类比），缺少多路径并行探索的representational diversity（类似multi-head attention中不同subspace的收益）。
  - (b) **中间激活的HBM materialization**：gate⊙up ∈ R^{L×d_ff}必须先写入HBM再读取，成为I/O瓶颈。大序列和大模型下（如L=16K, d_ff=5504），单层中间激活占用HBM约L·d_ff·2 bytes（bf16），大模型下单层可达~180MB。
  - (c) **Scaling ratio固定**：标准设计d_ff/d_model ≈ 8/3，这是单路径下的经验最优。若想增加表达力（如增加"head数"）需要额外参数和计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashMHF（Flash Multi-Head FFN）：Parallel FFN Sub-Networks + I/O-aware Fused Kernel**。核心设计：(1) 将FFN分解为H个head（multi-head design增强表达力），每个head的d_h=d_model/H；(2) 每head内进一步分解为E个并行sub-network（类似dense MoE但不做sparse top-k路由），每sub-network的d_e ≈ 8/3·d_h维持balanced ratio；(3) learned sigmoid gating动态加权聚合sub-network输出；(4) I/O-aware fused kernel (SRAMFFN)避免HBM中materialize中间激活——沿d_ff维度blockwise计算，所有中间结果仅在SRAM中temporary存在。

  全栈执行例子（FlashMHF, 370M, d_model=1024, H=8, d_h=128, E=7, d_e≈342, L=4096, H100 GPU）：
  - **模型推理算法层**：FlashMHF替代SwiGLU FFN。流程变为：(1) Q = split_H(X·W_in) ∈ R^{4096×8×128}；(2) 每head h内计算gating weights R^h = sigmoid(Q_h·W^h)/Σsigmoid ∈ R^{4096×7}；(3) 每sub-network e内独立执行FFÑ(Q_h; K_e^h, U_e^h, V_e^h) = (SiLU(Q_h·K_e^{hT}) ⊙ (Q_h·U_e^{hT}))·V_e^h——注意每sub-network的中间维仅d_e≈342而非d_ff=2752；(4) S_h = Σ_e R^h[:,e] ⊙ FFÑ_output ∈ R^{4096×128}；(5) O = concat_H(S)·W_out。这是"implicit thinking"的"beam search"——多个parallel pathway（H×E个path）并行探索，每path维度小而balanced。本质上类似dense MoE但每head有独立私有sub-network参数。
  - **系统框架层**：PyTorch层替换FFN module为FlashMHF module。论文未说明特定serving框架修改。
  - **编译框架层**：论文未明确说明编译框架修改。Triton和ThunderKittens kernel为手动实现（非编译器自动生成）。
  - **kernel调度层（关键创新）**：SRAMFFN fused kernel（参见Algorithm 1-5）。核心思想——沿d_ff维度（更准确地说，E·d_e维度）blockwise迭代计算，避免materialize intermediate activation：
    - Forward: O ← 0; for block m=1..M: O += (SiLU(Q·K_m^T)⊙(Q·U_m^T))·V_m。每block的结果直接累加到output accumulator（in SRAM），无需保存完整的(gate⊙up) ∈ R^{L×d_ff}。最后一次性将output写入HBM。
    - Hopper实现：warp-group specialization——producer异步prefetch tiles到SRAM ring buffer，CON_WARPGRPS个consumer独立处理不同x-block（sequence partition），在完成一个sub-network的router同步后各自并行计算。
    - Memory从O((d_ff+d_model)·L)降至O(d_model·L)，甚至比标准SwiGLU更小。Memory reduction 3-5x。
    - 通过在SRAM中驻留中间结果+消除HBM round-trip，即使H的总中间维与传统d_ff相等，实际memory和I/O开销大幅降低。Latency speedup 1.00-1.08x（得益于消除I/O瓶颈，但cuBLAS已高度优化GEMM）。
  - **硬件架构层**：NVIDIA H100 GPU。FlashMHF通过降低HBM traffic（消除中间activation的读写）提升memory efficiency。Hopper TMA用于异步prefetch。Warp-group specialization利用SM内并行性。

  关键设计选择与baseline缺陷的对应：
  - **defect (a): 单路径限制表达力** → 方案：Multi-Head设计——FFN分解为H个head，每head有独立private sub-network参数。这是类比multi-head attention的representational subspace specialization。实验验证：FlashMHF-128hdim在370M上比baseline loss低0.016（3.014 vs 3.030），1.3B上低0.050（2.793 vs 2.843）。下游任务平均分370M: 40.48 vs 39.92（+0.56），1.3B: 43.35 vs 41.75（+1.60）。消融实验：Dense-MoE (H=1)甚至差于baseline（3.062 vs 3.030），证明多head分解（而非仅是parallel sub-network）是gain的主要来源。
  - **defect (b): 中间激活HBM materialization** → 方案：SRAMFFN I/O-aware fused kernel。核心trick——沿d_ff维度blockwise计算（每block size = BLOCK_INTER），每block的(SiLU(QK^T)⊙(QU^T))·V直接在SRAM中计算并累加到output accumulator，中间tensor永不写入HBM。结果：peak memory reduction 3-5x（L=4096时FlashMHF≈866MB vs SwiGLU≈2592MB；L=16128时3016MB vs 9966MB）。Memory footprint甚至小于标准SwiGLU（因multi-head design将大中间激活分解为H个窄head的小块accumulation）。
  - **defect (c): Scaling ratio固定** → 方案：Parallel FFN Sub-Networks。Naïve multi-head FFN在模型scale up时d_ff/d_h ratio爆炸（128M: 16, 370M: 21, 1.3B: 45），超过Kaplan et al. 2020的optimal range导致性能退化。FlashMHF通过将d_ff分解为E个d_e维sub-network（每d_e ≈ 8/3·d_h），维持每sub-network internal ratio balanced。关键消融：Naïve MH-FFN在128M上优于baseline但在370M上失效（Table 1: 3.031 vs 3.030，无gain），而FlashMHF-128hdim持续gain（3.014），证明parallel sub-network是scaling成功的关键。
  - **额外设计：Gated aggregation代替简单平均** → Sigmoid-based learned gating（公式11-12）而非softmax-based sparse routing（如标准MoE）。好处：(1) 每token所有sub-network都参与计算（dense activation），避免load imbalance问题；(2) sigmoid + normalization给每token独立的per-sub-network权重，提供更细粒度的动态组合能力；(3) 计算开销可控（仅E维gating vs 全d_ff维FFN操作）。
