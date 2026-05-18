## GEMM-based Operator Chain (FFN/Gated FFN/Conv Chain) as Compute-Intensive Fusion Target（作为计算密集融合目标的GEMM算子链）

术语是什么？通过联网搜索让回答具体和精准。

GEMM-based Operator Chain 是 LLM 和 CNN 中由连续 General Matrix Multiplication (GEMM) 构成的算子序列，在 FlashFuser 论文中是 kernel fusion 的主要目标。三种典型模式：(1) Standard FFN——两个连续 GEMM（GEMM(A,B)→C → activation(如ReLU) → GEMM(C,D)→E），如 GPT/OPT/LLaMA 的 Feed-Forward Network 层；(2) Gated FFN——FFN 带两个平行的 Up-FFN 分支（SwiGLU 结构），其中一个分支经 SiLU 激活后与另一分支 element-wise 乘再进入后续 GEMM；(3) Conv Chain——卷积块可通过 im2col 转换为 GEMM chain 形式（Conv1×1 等价于 GEMM）。这些算子链在 LLM 推理中的典型表现为：序列长度 512 时，FFN 层占 GPT-6.7B 约 61% 执行时间、LLaMA-1B 约 57%、OPT-1.3B 约 53%（Table I）。由于中间 tensor C [M×N] 的数据量很大（如 GPT-6.7B: M=128, N=16384, C ≈ 2M floats ≈ 8MB），远超单 SM SMEM (~227KB)，传统方法无法在片上保留完整中间结果，必须通过 HBM round-trip。

从算法pipeline角度拆解术语：

```
// Standard FFN 推理流程 (GPT/OPT/LLaMA 通用):
def feedforward(x):          // x: [M, K]  (M=batch tokens, K=hidden dim)
  C = x @ W_up.T            // GEMM1: [M,K] × [K,N] → [M,N]
  C = activation(C)         // ReLU/GELU/SiLU
  E = C @ W_down.T          // GEMM2: [M,N] × [N,L] → [M,L] (通常 L=K)
  return E

// Gated FFN (SwiGLU, LLaMA 使用):
def gated_ffn(x):           // x: [M, K]
  C1 = x @ W_gate.T         // Gate branch: [M,K] × [K,N] → [M,N]
  C2 = x @ W_up.T           // Up branch:   [M,K] × [K,N] → [M,N]
  C1 = SiLU(C1)
  C = C1 ⊙ C2               // Element-wise multiply
  E = C @ W_down.T          // GEMM2: [M,N] × [N,L] → [M,L]
  return E

// Conv Chain (ResNet, 可 im2col 转换):
def conv_block(x):          // x: [IC, H, W]
  C = Conv1(x, W_conv1)     // Conv1: [IC,H,W] × [OC1,IC,K1,K1] → [OC1,H,W]
  C = ReLU(C)
  E = Conv2(C, W_conv2)     // Conv2: [OC1,H,W] × [OC2,OC1,K2,K2] → [OC2,H,W]
  return E
```

在 GPU 推理中，这些 GEMM chain 的算术强度（Arithmetic Intensity）较低——当 batch size M 较小时（inference 典型 M=128），中间 tensor C 从 HBM 读取/写入的 overhead 相对 GEMM 算力成为瓶颈，呈现 memory-bound 特性。kernel fusion 的目标是将这些算子链合并为一个大 kernel，使中间 tensor C 留在片上(on-chip)而不是写回 HBM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GEMM-based Operator Chain 在不同层次中的实现：
1. **库实现（cuBLAS/CUTLASS）**：每个 GEMM 作为独立 kernel launch，中间 tensor C 必须写回 global memory。PyTorch `torch.compile` 可减少 kernel launch overhead 但数据路径不变
2. **图级 fusion（TASO/TVM Relay）**：通过算子替换和 graph optimization 融合 compute-activation pattern，但不支持 sequential GEMM chain 的 compute-intensive fusion
3. **SMEM-based fusion（BOLT/Chimera/Welder）**：利用 register 和单 SM 的 SMEM 保留中间 tile——但当 C tile > SMEM (~227KB) 时 fusion 失败
4. **DSM-based fusion（FlashFuser）**：利用 DSM（cluster 内多 SM SMEM 互联，~3.6MB）保留更大的 C tile，通过 dsm_comm 原语管理 producer-consumer 数据流

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

