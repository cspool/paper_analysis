## Fused MoE Kernel（融合 MoE Kernel，SGLang Triton 后端）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused MoE Kernel 是把 MoE 层的路由后 token-专家映射、多个专家各自的 GEMM 计算（甚至 SwiGLU 的 gate/up/down 三次 GEMM）融合进单个 GPU kernel 的算子实现（代表性：SGLang 的默认 MoE 后端、FlashMoE、MegaBlocks）。相比逐专家独立 launch GEMM，融合后消除了 N-1 次 kernel launch overhead 与中间张量的 HBM 往返，还能利用 grouped-GEMM/block-sparse 布局让不同专家共享一次调度。在 PIPEWEAVE 中它是 6 类被建模 kernel 之一（Triton 语言、BF16/FP16、Tensor pipeline、硬件调度），其 33,264 个样本数据集同时用作"beyond simulation"优化指导的 case study。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused MoE kernel 的关键调度结构（以 SGLang Fused MoE Triton 为例）：
```
# 1. 路由（上一阶段）：每个 token 经 router softmax 选 top-k 专家
# 2. 单 kernel 内按专家分组计算：
for expert e in activated_experts:
    tokens_e = tokens 按路由结果分到 e 的索引集合
    h_gate = tokens_e @ W_e^gate      # GEMM1
    h_up   = tokens_e @ W_e^up        # GEMM2
    h = silu(h_gate) * h_up           # SwiGLU 激活（element-wise）
    out_e  = h @ W_e^down             # GEMM3
# 3. 按原 token 顺序 scatter 回输出
# Triton 中由 @triton.jit kernel + BLOCK_SIZE/num_stages/num_warps 三个参数控制 tile 划分、流水线深度与并行度
```
PIPEWEAVE 对它的解析建模：M ∈ [2,8192]、E ∈ [8,128]、topk ∈ [2,8]、H ∈ [1024,4096]、N ∈ [512,3072]，每 task 的 Tensor ops = α·tile_M·tile_N·tile_K。优化指导流程：P80 分位模型预测执行效率上限 ŷ_p80，perf_gap = ŷ_p80 − y_actual > 0.1 记为 underperforming point；对 A40/L20/A100/H800 各选约 70 个配置 brute-force autotune 三参数，A40 从 921 个 underperforming points（占 30.4%）经调参平均 gap 从 0.187 降到 0.083、几何平均提速 1.61×，且 underperforming points 数与提速成 Pearson 0.86 正相关；残余 gap 归因于 Triton 编程模型/结构设计限制而非参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SGLang 默认 MoE 后端为 Triton Fused MoE kernel；vLLM 侧有集成路径；FlashMoE 用 CUTLASS device-side GEMM 在 persistent kernel loop 内直接调用（单 kernel 完成分布式 MoE）；MegaBlocks 用 block-sparse 布局。使用上，PIPEWEAVE 证明融合 kernel 的性能预测比 element-wise 拆分更难也更重要——融合破坏了"算子边界=launch 边界"假设，只能按 kernel 级 task 分解建模。开源 artifact（github.com/zksainx/pipeweave）提供 Triton MoE 的 roofline 计算器，可对给定 (M,E,topk,H,N) 预测 latency。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
