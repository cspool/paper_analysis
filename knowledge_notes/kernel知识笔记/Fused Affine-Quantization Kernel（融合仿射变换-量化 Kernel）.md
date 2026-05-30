## Fused Affine-Quantization Kernel（融合仿射变换-量化 Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Affine-Quantization Kernel 是 FlatQuant 基于 OpenAI Triton 实现的 GPU kernel，将 Kronecker 仿射变换 Q(P₁^T ×₁ X̃ ×₂ P₂) 融合为单个 kernel 调用。设计动机：(1) 使用 Kronecker 乘积后 P₁∈R^{n₁×n₁}、P₂∈R^{n₂×n₂} 尺寸很小（如 64×64），仿射变换为 memory-bound 操作（计算强度低）；(2) 量化也是 memory-bound。传统分开执行会产生两次全局内存往返（先写回 X' 再读取做量化）。融合后：thread block 将 P₁、P₂ 完整加载到 SRAM → slicing tile X̄∈R^{n₁×n₂} → 在 SRAM 内执行 P₁^T X̄ P₂ → 立即对结果量化 → 写回全局内存。三种 SRAM 容量场景：(a) 默认——SRAM 容纳 P₁、P₂、X̄ 及中间结果；(b) Corner Case 1——n₁ 过大，对 P₁ 非规约维 tiling；(c) Corner Case 2——n₂ 过大，分两个 kernel（先 P₁^T X̄ 写回，再乘 P₂ 并量化）。在 RTX 3090 上 hidden_dim≤14336 均使用默认设计。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 FlatQuant 默认设计（hidden_dim=4096, n₁=n₂=64, RTX 3090, batch=1）为例：

```
// Triton kernel (简化为 Python-like 伪代码)
@triton.jit
def fused_affine_quantize_kernel(
    X_ptr, P1_ptr, P2_ptr,        // FP16 inputs from DRAM
    X_q_out_ptr, scale_out_ptr,    // INT4 packed + FP16 scale → DRAM
    n1: int, n2: int, bits: int
):
    pid = tl.program_id(0)          // one program per token (k tokens)
    
    // Phase 1: Load P₁, P₂ into SRAM (once per block, shared across threads)
    P1 = tl.load(P1_ptr + offsets)  // [64, 64] FP16 → 8KB SRAM
    P2 = tl.load(P2_ptr + offsets)  // [64, 64] FP16 → 8KB SRAM
    
    // Phase 2: Load X tile into SRAM
    X_tile = tl.load(X_ptr + pid*64*64 + offsets)  // [64, 64] FP16 → 8KB SRAM
    
    // Phase 3: Affine transformation in SRAM (memory-bound)
    // X' = P₁^T @ X_tile @ P₂
    X_transformed = tl.dot(P1.T, tl.dot(X_tile, P2))  // [64, 64] in SRAM
    
    // Phase 4: Fused quantization (in SRAM, no DRAM write)
    abs_max = tl.max(tl.abs(X_transformed))
    scale = abs_max / (2**(bits-1) - 1)
    X_q_int = tl.round(X_transformed / scale)
    X_q_int = tl.clamp(X_q_int, -2**(bits-1)+1, 2**(bits-1)-1)
    X_q_packed = pack_int4(X_q_int)  // 2×4-bit → 1 byte
    
    // Phase 5: Write to DRAM (single write)
    tl.store(X_q_out_ptr + pid*32*64 + offsets, X_q_packed)  // INT4 packed
    tl.store(scale_out_ptr + pid, scale)                       // scalar FP16
```

**实测性能**（Table 6, hidden_dim=4096, batch=1, seq_len=2048 prefill / decode_1token）：
- 无融合: prefill 0.1956ms, decode 0.0184ms
- 有融合: prefill 0.0625ms, decode 0.0082ms
- 加速比: prefill 3.13×, decode 2.25×

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlatQuant 的融合 kernel 基于 OpenAI Triton 语言编写（https://github.com/openai/triton），编译为 PTX 后在 NVIDIA GPU 上运行。自定义 kernel 位于开源仓库 https://github.com/ruikangliu/FlatQuant。推理时，融合 kernel 的输出（INT4 packed 激活 + FP16 scale）直接送入 CUTLASS INT4 GEMM kernel 进行矩阵乘法。完整的预填充流程：tokens 进入 → 逐 Transformer block → (1) 层归一化 (FP16) → (2) 融合仿射量化 kernel 对激活做在线变换+量化 → (3) CUTLASS INT4 GEMM 执行量化矩阵乘法 → (4) FlashInfer kernel 对 KV cache 执行量化 → (5) 残差连接 (FP16)。端到端加速: prefill 2.30× vs FP16, decode 1.76× vs FP16。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization

---
