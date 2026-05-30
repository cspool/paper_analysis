## Toward Efficient Inference for Mixture of Experts

- baseline方法是什么？
  Baseline 是 Fairseq 实现的 MoE Transformer 推理，采用 **Static Gating + Expert Parallelism**（基于 GShard [2] 和 ELSLM [8]）。核心设计：(1) static gating 函数为每个 expert 预分配固定容量 C（capacity factor），通过 batch matrix multiplication 构建 dispatch mask 进行 token 分配；(2) 使用 NCCL all-to-all 进行固定大小消息的 token 分发和收集（每个 GPU 预知消息大小）。
  
  Baseline 的核心缺陷：
  1. **Computation waste from placeholders**：Static gating 为每个 expert 预设 capacity C，当实际 token 数少于 C 时需填充 zero placeholder。LM（E=512, C=0.05, top-2 gating）的 waste factor 为 12.8×（实际只需 2S tokens，但要计算 25.6S tokens）。MT（E=128, C=1, top-4 gating）的 waste factor 为 64×（实际只需 4S tokens，但要计算 128S tokens）。
  2. **Large memory from dispatch mask**：Batch matmul 构建的 dispatch mask 维度为 (E, S, S×C)，需要大量 GPU 临时内存。论文 memory trace 显示 gating 和 reordering 阶段有瞬时内存尖峰。
  3. **Token dropping risk**：当负载不均衡时，超出 capacity 的 tokens 会被丢弃（只保留 residual connection），损失模型质量。
  4. **Full expert parameter loading**：所有 experts 参数必须常驻 GPU 显存，即使推理中大部分 expert 很少被激活。LM 单 GPU 需 18.9GB，dense 只需 2.2GB。
  5. **Load imbalance**：MoE 在训练时的 token 分布与推理时不同，导致某些 GPU 负载过高（oversubscribed, OOM 风险），某些 GPU 空闲。

  全栈执行例子（Baseline, Fairseq static gating, 单 node 8×V100, LM task）：
  - **算法Pipeline层**：输入 tokens X ∈ R^{8×1024} → gate_linear → top-2 gating 选择 2/512 experts → 为每个 expert 构建 dispatch mask M_e ∈ R^{8×8×25.6=204.8}（多为此维度，大量 zeros）→ batch matmul M @ X（92% FLOPs 为 ×0）→ dispatched tokens per expert → 每个 expert FFN forward → batch matmul reorder → next layer。
  - **系统框架层**：Fairseq (PyTorch) MoE Transformer，NCCL all-to-all 通信（固定消息大小），expert parallelism 跨 8×V100 GPU 分配 512 experts（每 GPU 64 experts）。Expert 参数常驻 GPU 显存。batch size 固定为 8（LM）或 48（MT），受限于显存。
  - **编译框架层**：论文未明确说明（使用 PyTorch JIT 或 eager 模式，未修改编译框架）。
  - **Kernel调度层**：NCCL all-to-all（cudaMemcpy），PyTorch batch matmul（cuBLAS/cuDNN），标准 PyTorch MLP forward（cuBLAS GEMM）。Batch matmul 中 92.2% 计算为 ×0。
  - **硬件架构层**：NVIDIA Tesla V100 (32GB HBM2, NVLink)。CPU: Intel Xeon E5-2698 v4。CPU-GPU: PCIe 3.0 16GB/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出三个正交的 MoE 推理优化技术，从算法、内存管理、负载均衡三个维度解决 baseline 缺陷：

  **1. Dynamic Gating（对应缺陷 1, 2, 3）**
  - 将 static gating 的 batch matmul dispatch 替换为 argsort + bin-count + indexing
  - 复杂度从 O(S²EDC) 降至 O(SD + S log S)
  - 每个 expert 的容量动态设为实际收到的 token 数（不丢 token）
  - 消除 dispatch mask（省内存）和 placeholder 计算（省 FLOPs）
  - 两轮 all-to-all：第一轮通知 sizes（20µs avg），第二轮传可变大小 tokens

  **2. Expert Buffering（对应缺陷 4）**
  - 利用 expert 激活的 temporal locality：仅将热 expert 留在 GPU cache，其余缓存在 CPU
  - LIFO eviction policy（适配 MoE 按 expert ID 顺序执行的特性）
  - 异步 CPU→GPU 参数拷贝，与 token all-to-all 传输重叠
  - Cache miss rate 接近理论最优 Belady's MIN
  - 减少 static GPU memory 达 1.47×

  **3. Load Balancing（对应缺陷 5）**
  - Greedy Balancing：按 expert 历史平均负载排序，贪心分配到最少负载 GPU
  - Anti-Correlation Balancing：针对 decoder 场景 expert 激活相关的情况，在负载估计中加 Pearson 相关系数惩罚
  - 约束每个 GPU 等量 experts，平衡内存和通信
  - 减少 Max load（OOM 风险）和 Avg-Max load（性能退化）

  全栈执行例子（Ours, Dynamic Gating + Expert Buffering + Load Balancing, 单 node 8×V100, LM task）：
  - **算法Pipeline层**：输入 tokens X ∈ R^{8×1024} → gate_linear → top-2 assignments → **argsort** by expert ID (O(S log S)) → **advanced indexing** X[sorted_idx] 重排 token (O(SD)) → **bincount** 计算每个 expert 的实际 token 数 → all-to-all round 1（通知 sizes, ~20µs）→ **split** 按 sizes 切分 → all-to-all round 2（仅传实际 tokens，无 placeholder）→ 各 GPU expert FFN forward（仅计算实际收到的 tokens）→ all-to-all collect → indexing 还原顺序。差异：无 dispatch mask 分配，零 placeholder FLOPs，work factor = 1×（实际需要 = 实际计算）。
  - **系统框架层**：基于 Fairseq (PyTorch) + 开源代码 https://github.com/hyhuang00/moe_inference。Expert Buffering 在 Fairseq MoE forward 前插入 cache check/cudaMemcpyAsync。Load Balancing 在推理前运行 profiling pass 收集 activation 数据，调用 Greedy/Anti-Correlation 算法优化 expert placement。支持可变 batch size（从原始 8 扩展到 64-96）。
  - **编译框架层**：论文未明确说明（基于 PyTorch eager execution，未修改编译层）。
  - **Kernel调度层**：argsort（GPU radix sort kernel）、bincount（GPU reduction kernel）、advanced indexing（GPU gather kernel，O(SD) memory BW bound）、NCCL all-to-all（可变大小）、cudaMemcpyAsync（PCIe stream 与 NCCL stream 并发）。与 baseline 的关键差异：用 indexing 替代 batch matmul，消除 92.2% 浪费计算。
  - **硬件架构层**：NVIDIA Tesla V100 (32GB, NVLink)，NVIDIA RTX A5000 (24GB, Ampere)。CPU-GPU PCIe 带宽是 Expert Buffering 的瓶颈（12GB/s peak），论文指出新技术（如 Grace Hopper）可缓解。

  性能结果（摘要）：
  - Dynamic Gating vs Fairseq static: LM throughput +6.21× (single-node), +11.55× (multi-node)
  - vs Megablock: batch=80 时 1.46× faster（因 dense matmul 优于 BCSR sparse matmul）
  - MT-decoder throughput: +5.75× encoder, +2.58× decoder
  - Dynamic memory (activations): LM -79.6% (6.29→1.28GB), MT -44.2% (1.89→1.05GB)
  - Expert Buffering: static memory -1.47× (~2.25GB)
  - Load Balancing: throughput +1.19× (Greedy, LM multi-node)
  - 允许更大 batch size: LM 8→64, MT 48→96
