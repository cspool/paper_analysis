## Compilation Template Mapping via Numerical Decoding

术语是什么？通过联网搜索让回答具体和精准。

Numerical Decoding是STOF operator fusion module中将hash-encoded binary fusion scheme映射到concrete compilation template的过程。Binary array中相同number的连续operator组成一个fused segment，numerical decoding解析每个segment的operator组成（operator types和order），查询预定义的template library选择匹配的compilation template（如template_gemm_layernorm处理CI+MI融合、template_gemm_gemm处理CI+CI GEMM chain、template_add_layernorm处理MI+MI融合），并将template.Config中暴露的关键参数（block_size、num_stages、num_warps、blkM/blkN/blkK等）作为该segment的tuning search space。Compilation template本身是基于Triton和TileLang手写的high-performance kernel实现，内部包含tile decomposition（最大化data reuse）、warp-level primitives（高效reduction）、multi-stage pipelining（重叠memory access和compute）等优化。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
// ===== Numerical Decoding Process =====
// Input: binary expression (from hash encoding)
//   [#0 #1 #2 #3 #4 #5 #6] [#7 #8 #9] [#10 #11 #12] [#13 #14]
//     0  1  0  0  0  0  0    1  1  1     0   0   0      1   1
//
// Step 1: Decode segments
// segment 0 (number=0): operators [#0 #1] = [GEMM, Add]
//   → pattern: BiasAdd after GEMM → merged into MHA fused kernel (not template)
//   → but GEMM(#2)-Scale(#3)-Mask(#4)-Softmax(#5)-GEMM(#6) 
//     = MHA subgraph → mapped to custom MHA kernel (not compilation template)
//
// segment 1 (number=1): operators [#7 #8 #9] = [GEMM, Add, Layernorm]
//   → pattern: GEMM(CI) + Add(MI) + Layernorm(MI) = CI+MI mix
//   → mapped to: template_gemm_layernorm
//   → template.Config exposes: block_size, num_stages, num_warps
//
// segment 2 (number=0): operators [#10 #11 #12] = [GEMM, GEMM, Activation]
//   → pattern: GEMM chain + Activation(CI+CI) 
//   → mapped to: template_gemm_gemm
//   → template.Config exposes: blkM, blkN, blkK, blkH, num_stages, num_warps
//
// segment 3 (number=1): operators [#13 #14] = [Add, Layernorm]
//   → pattern: MI+MI
//   → mapped to: template_add_layernorm
//   → template.Config exposes: block_size, num_stages, num_warps

// ===== Template Implementation Example =====
// template_gemm_layernorm (Triton pseudo-code):
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k}, 
                       num_stages=s, num_warps=w)
        for m, n, k, s, w in search_space
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def gemm_layernorm_kernel(A, B, bias, C, M, N, K):
    // Tile-level decomposition for data reuse
    pid = tl.program_id(0)
    // ... tile indexing logic ...
    
    // Accumulator in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    // Multi-stage pipeline: overlap global→shared load with compute
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr, ...)  // global→shared (async)
        b = tl.load(B_ptr, ...)
        acc += tl.dot(a, b)      // Tensor Core MMA
    
    // Add bias (element-wise, fused in same kernel)
    acc += bias
    
    // Layernorm (warp-level reduction)
    mean = tl.sum(acc, axis=1) / BLOCK_N
    var = tl.sum((acc - mean) ** 2, axis=1) / BLOCK_N
    acc = (acc - mean) / tl.sqrt(var + eps) * weight + bias_ln
    
    tl.store(C_ptr, acc)
```

Template选择策略：STOF实现两套template（Triton版和TileLang版），对每个fused operator pattern实际测试两者性能，选择per-case表现更优的backend。这种dual-backend设计利用了两者在不同operator mix下的performance variance。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

STOF的compilation template library手写实现（Python），约5,000 LOC total（含Python template + CUDA MHA kernel）。Template通过Triton的@triton.autotune decorator暴露Config space给search engine，search engine在two-stage tuning的parameter sampling阶段sample Config生成kernel instance并benchmark。与现有template-based approach（如Bolt基于CUTLASS primitives）的区别：Bolt用CUTLASS template但fusion range扩展困难（需程序员手动修改复杂CUTLASS kernel结构），STOF用Triton/TileLang的high-level tile programming interface使template derivation更容易支持wider fusion range。torch.compile兼容性允许STOF复用的PyTorch compilation optimizations（如constant folding、instruction scheduling等）。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU
