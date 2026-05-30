## JIT Attention Variant Compiler (FlashInfer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

JIT Attention Variant Compiler 是 FlashInfer 的 Just-In-Time 编译系统，将 attention 变体的 high-level specification（CUDA 代码定义的 functors）编译为高度优化的 block-sparse FlashAttention CUDA kernel。解决的核心问题：现代 LLM 中 attention 变体快速增长（GQA/MQA、RoPE、sliding window、logits soft-cap、FlashSigmoid、MLA projection 等），为每种变体手工编写专用 CUDA kernel 不可扩展。JIT compiler 通过 parameterized CUDA template + functor-based variant specification + Python-level JIT compilation pipeline，允许用户以约 20 行 CUDA 代码定义新 attention 变体并自动生成优化的 CUDA kernel。

编译器接受 attention variant specification，包含以下 functor slots：
- `QueryTransform(Q, params) → Q'`：Q 在 attention 计算前的变换（fused RoPE、normalization、MLA projection）
- `KeyTransform(K, params) → K'`：K 在 attention 计算前的变换（fused RoPE）
- `ValueTransform(V, params) → V'`：V 在 attention 计算前的变换
- `OutputTransform(O, params) → O'`：attention output 后处理
- `LogitsTransform(S, params) → S'`：logits 后处理（logits soft-cap: `S' = soft_cap * tanh(S / soft_cap)`、sigmoid: `S' = 1/(1+exp(-S))`）
- `LogitsMask(S, mask_info) → S'`：logits masking（sliding window mask、custom sparse mask）

此外，`use_softmax` flag 控制是否使用 softmax（FlashSigmoid 等不使用 softmax 的变体设为 false）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

FlashInfer JIT 编译器的完整编译流程：

```
// ===== Phase 1: Variant Specification (User Input) =====
// ~20 lines CUDA code for Streaming-LLM fused RoPE+attention
struct RoPEFusionVariant {
    // QueryTransform: fuse RoPE into Q loading
    __device__ float query_transform(float q, int head_idx, 
                                      int pos, int dim) {
        float cos_val = __cosf(pos * freq[dim/2]);
        float sin_val = __sinf(pos * freq[dim/2]);
        if (dim < rot_dim / 2)
            return q * cos_val - q_rot * sin_val;  // real part
        else
            return q_rot * cos_val + q * sin_val;  // imag part
    }
    
    // KeyTransform: fuse RoPE into K loading
    __device__ float key_transform(float k, int head_idx, 
                                    int pos, int dim) {
        // ... same RoPE rotation for K ...
    }
    
    // Other functors: identity (pass-through)
    __device__ float logits_transform(float s) { return s; }
};

// ===== Phase 2: Template Instantiation =====
// FlashInfer's CUDA kernel template (simplified):
__global__ void flashinfer_attention_template(
    Q, K, V, O, ..., VariantParams params
) {
    // Load Q tile from GMEM to SMEM
    q_smem[tid] = Q[...];
    //  ↓ Variant injection point: QueryTransform
    q_smem[tid] = query_transform(q_smem[tid], head, pos, dim);
    __syncthreads();
    
    // Load K tile from GMEM to SMEM (BSR for sparse KV-cache)
    k_smem[tid] = load_sparse_kv_tile(K_cache, kv_indices, ...);
    //  ↓ Variant injection point: KeyTransform
    k_smem[tid] = key_transform(k_smem[tid], head, pos, dim);
    __syncthreads();
    
    // QK^T GEMM (Tensor Core WGMMA/HMMA)
    S = WGMMA(q_smem, k_smem);
    
    //  ↓ Variant injection point: LogitsTransform + LogitsMask
    S = logits_transform(S);
    S = apply_mask(S, mask_info);
    
    // Online softmax (or skip if use_softmax=false)
    if (use_softmax) {
        m, l, O_acc = online_softmax(S, V_tile, m_prev, l_prev, O_prev);
    } else {
        O_acc = S × V_tile;  // no softmax, e.g. FlashSigmoid
    }
    
    //  ↓ Variant injection point: OutputTransform
    O_acc = output_transform(O_acc);
    
    // Write output
    O[...] = O_acc;
}

// ===== Phase 3: JIT Compilation =====
// Python level
import torch.utils.cpp_extension as cpp_ext

cuda_source = generate_cuda_source(
    template="flashinfer_fa2_template.cuh",
    variant_class="RoPEFusionVariant",
    variant_params=params,
    task_info=dict(B_r=1, B_c=1, T_q=16, T_kv=64, ...)
)

module = cpp_ext.load_inline(
    name="flashinfer_rope_fusion",
    cuda_sources=[cuda_source],
    functions=["fused_rope_attention"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

// ===== Phase 4: Registration as PyTorch Custom Operator =====
torch.library.define(
    "flashinfer::fused_rope_attention",
    "(Tensor Q, Tensor K_cache, Tensor V_cache, "
    "Tensor kv_indptr, Tensor kv_indices, ...) -> Tensor"
)
torch.library.impl(
    "flashinfer::fused_rope_attention", "cuda",
    module.fused_rope_attention
)

// ===== Phase 5: Caching =====
// Compiled .so cached on disk (keyed by variant spec + task config hash)
// Subsequent runs skip compilation, directly load cached shared library
```

关键设计选择：
- **CUDA code (not Triton)**：FlashInfer 直接生成 CUDA 代码，因为 Triton 在许多 use case（特别是 decode kernel、BSR sparse patterns）仍 underperform hand-tuned CUDA/CUTLASS
- **Template-level injection (not IR-level)**：在 CUDA 源码模板层面插入 variant functors，而非 MLIR/Triton IR 层面。这保留了所有 low-level 优化（register allocation、shared memory layout、async copy scheduling）
- **FlexAttention 兼容**：FlashInfer 可作 PyTorch FlexAttention forward pass 的 CUDA backend（FlexAttention 本身生成 Triton 代码）
- **DLPack 接口**：支持 framework-agnostic tensor exchange，允许编译结果被非 PyTorch runtime 使用

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer JIT compiler 的实现和使用：
- **实现**：核心在 `flashinfer/jit/` 目录下——Python-level template engine (string-based code generation) + PyTorch `cpp_extension.load_inline` 作为编译后端 + disk cache for compiled binaries
- **覆盖范围**：$f_{epilogue}(scan(f_{logits}(f_q(Q) \cdot f_k(K))) \cdot f_v(V))$ 形式的 attention 函数——包括 MLA (DeepSeek-V2, Q/K低秩压缩)、Linear Attention (GLA 的 intra-attention component)、FlashSigmoid、sliding window、logits soft-cap 等
- **使用示例**（Streaming-LLM）：~20 行 CUDA code for QueryTransform + KeyTransform → JIT compile → fused kernel（RoPE + attention in single kernel）→ 1.6-3.7× bandwidth utilization over unfused (RoPE kernel + attention kernel)
- **使用示例**（FlashSigmoid, Figure 5）：`LogitsTransform = sigmoid` + `use_softmax=false` → JIT compile → FlashSigmoid kernel
- **限制**（v0.2）：仅支持 forward pass（不支持 backward/training）
- GitHub: https://github.com/flashinfer-ai/flashinfer

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
