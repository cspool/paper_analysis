## MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-SpeQ 实现了 **fuseMoE CUDA kernel**——一个单次发射的融合 kernel，用于加速量化 MoE draft 阶段的细粒度 expert 计算。核心理念是将多个小 GEMM kernel launch（每个 expert 独立的 W_gate, W_up, W_down）融合为一个 monolithic kernel，降低 kernel launch overhead 并提高 GPU 硬件利用率。
    - 背景动机：在细粒度 MoE 模型（如 Qwen2-MoE，K=1408, N=2048）上，Marlin 量化 GEMM 后端性能甚至低于 PyTorch FP16 实现——因为每个 expert 矩阵太小，single kernel 无法占满 GPU SM，大量时间消耗在 kernel launch overhead 上。
    - 融合策略：将 L 层中所有需要计算的 expert 的 INT4 GEMM 操作合并为单一 kernel 调用，一次性完成 gate projection、up projection、SiLU activation、gate×up 逐元素乘、down projection。减少 kernel launch 次数（从 per-expert per-layer 变为 per-layer），提高 GPU occupancy。
  - 实验比较：（1）消融实验：DeepSeek-V2-Lite 上 Full (13.02 tok/s) vs without async prefetch (12.37 tok/s, 95%) vs without fused kernel (8.88 tok/s, 68.2%) vs both off (8.29 tok/s, 63.7%)；（2）图12 量化感知 fused expert kernel 对 throughput 和 latency 的影响。

- 后端平台是什么，配置是什么。
  - NVIDIA A100-40GB GPU（PCIe 4.0 x16，理论双向 32GB/s 聚合带宽）。
  - CUDA multi-stream（4 条独立 stream）+ CUDA events 管理同步互斥。
  - Intel Xeon Silver 4310 CPU（24-core），256GB RAM。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 Hugging Face Transformers 框架，使用 GPTQ 量化库创建 INT4 draft 模型，使用 Marlin 后端做低比特推理。
  - 修改/新增内容：
    (1) **fuseMoE CUDA kernel**：自研的融合 CUDA kernel，将 per-expert 多次 GEMM kernel launch 合并为单次 launch。具体融合：loop over experts in layer → W_gate * h → W_up * h → SiLU(gate_out) ⊙ up_out → W_down * fused_out。所有中间结果保持 on-chip（SM shared memory + register），避免写回 global memory 再读回。
    (2) **Marlin 后端适配**：针对 MoE 细粒度场景（小 K/N 维度）优化 Marlin kernel 调用方式——通过 batch 多个 expert 的 GEMM 为单次更大矩阵运算，使 Marlin 达到接近 4× 加速期望。
    (3) **Computation Reordering**：verify 阶段前分析 ELB，重排 batch tokens 使同 expert 的计算连续执行，最大化 expert weights 的 L1/L2 cache 命中。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未提供开源代码仓库链接。
  - fuseMoE Kernel 评估原理与全流程（基于论文 §3.4.3 和 §4.5）：
    1. **Kernel 输入**：
       - Draft token hidden states: shape [k, d_model]（k 为 draft length，d_model=2048/4096）。
       - MoE layer 的 INT4 quantized weights：每组 expert 的 W_gate[inter_dim, d_model], W_up[inter_dim, d_model], W_down[d_model, inter_dim]，量化参数 scales [group_size] 和 zeros [group_size]（group_size=128）。
       - Router 选定 per-token per-layer 的 top-k expert IDs。
    2. **Kernel 计算流程**：
       - Step 1 - Expert Grouping: 遍历 k 个 token × L 层，按 expert ID 去重分组得到 S = union of selected experts across all tokens。
       - Step 2 - Batch GEMM: 将所有分配给同一 expert 的 tokens 的 hidden states 拼接为 mini-batch [n_tokens_for_expert, d_model] → Marlin INT4 GEMM 乘法 W_gate @ h^T → gate_out [n_tokens, inter_dim]。同理做 W_up @ h^T → up_out [n_tokens, inter_dim]。
       - Step 3 - Fused Activation: SiLU(gate_out) ⊙ up_out → fused [n_tokens, inter_dim]。此步与 Step 2 在同一 kernel 内完成，无中间 global memory write。
       - Step 4 - Down Projection: W_down @ fused^T → expert_output [n_tokens, d_model]。
       - Step 5 - Reduction: 对每个 token 的多个 expert 输出按 router softmax 权重加权求和。
    3. **Kernel 输出**：每层 draft hidden states [k, d_model]，传递至下一层或作为 output logits。
    4. **性能指标**：end-to-end tokens/sec (消融对比: 13.02 vs 8.88 vs 8.29 tok/s)。Prefill/decode latency comparison (图 11：Marlin vs other backends)。
    5. **优化原理**：融合前每个 expert 需要 3 次 GEMM kernel launch（W_gate, W_up, W_down），细粒度 MoE 中 inter_dim 仅 1408（Qwen2-MoE）或 6400（Phi-MoE），单个 GEMM 仅占用极少 SM，launch overhead >> compute。融合后所有 expert 的 GEMMs 在单次 kernel launch 中完成，batch size = Σ n_tokens_per_expert，矩阵维度增大 → GPU occupancy + SM utilization 提升 → 6.8% → 36.3% speedup (from ablation: 68.2% vs 100% normalized speed)。
