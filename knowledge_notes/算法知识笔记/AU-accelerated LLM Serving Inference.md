## AU-accelerated LLM Serving Inference

术语是什么？

AU-accelerated LLM Serving是指利用CPU内置加速器单元（如Intel AMX）来加速LLM推理的serving范式。与传统GPU serving不同，AU-accelerated serving运行在通用CPU上，通过AMX TMUL加速矩阵乘。LLM推理的两阶段在AU上表现截然不同：(1) Prefill phase：GEMM operations (8192×4096×22016, batch=16) 通过AMX达到40.57 TFLOPS，属于compute-bound (92% backend bound)，高AMX cycle ratio (14.4%)，高功耗导致core频率降至2.5 GHz；(2) Decode phase：GEMV operations (16×4096×22016) 通过AMX仅3.87 TFLOPS（小矩阵tile register配置overhead大），属于memory-bound (DRAM 59.9%)，低AMX cycle ratio (1.5%)，此时AVX比AMX更高效。工业界使用xft/xFasterTransformer或ktransformers等框架实现AU-accelerated serving，支持BF16精度和batch size 16。主要SLO指标：TTFT (Time-To-First-Token, prefill) 和 TPOT (Time-Per-Output-Token, decode)。AU-enabled CPU适合serving小模型（Phi-3 3.8B, Llama2 7B/13B, Qwen3-A3B 30B MoE）或作为GPU的补充。

从算法pipeline角度拆解术语：

AU-accelerated LLM serving的pipeline（以llama2-7B, batch=16为例）：
```
// === Prefill Phase (compute-bound, AMX) ===
Input: prompt tokens [0..L-1]

for each transformer layer:
    1. QKV Mapping: input @ W_QKV
       └─ matrix dim: (8192, 4096×22016), batch=16, input_len=512
       └─ AMX GEMM → 40.57 TFLOPS, AMX cycle ratio 14.4%
       └─ Backend bound: 92% (execution port + memory stalls)

    2. Multi-head Attention: Q @ K^T @ V
       └─ FlashAttention-like, compute-bound
       └─ Store KV to KV Cache

    3. Feed Forward: SiLU(W_gate @ x) * (W_up @ x) @ W_down
       └─ Each GEMM uses AMX for large matrix dims

Output: first token generated, KV Cache populated

// === Decode Phase (memory-bound, AVX preferred) ===
Input: last generated token [1, 4096]

for each transformer layer:
    1. QKV Mapping: last_token @ W_QKV
       └─ matrix dim: (16, 4096×22016), batch=16
       └─ AMX GEMV → 3.87 TFLOPS (AMX inefficent for small dim)
       └─ AVX preferred: lower tile register overhead
       └─ AMX cycle ratio: 1.5%, AMX uop ratio: 0.5%
       └─ DRAM bound: 59.9%

    2. Attention: q @ K^T (from KV Cache) → memory-bound
       └─ Load entire KV Cache from memory

    3. Feed Forward → next layer

Output: next token → append KV Cache → repeat

// SLO: TTFT = time from prompt to first token
// SLO: TPOT = average time per subsequent token
```

术语一般如何实现？如何使用？

AU-accelerated serving通过xft或ktransformers调用oneDNN GEMM→oneDNN根据矩阵维度自动选择AMX或AVX。工业界现状：CPU与GPU hybrid部署日益普遍（Alibaba Cloud ~50% CPU核心idle但带AU可利用），AU serving perf-per-dollar优于GPU但perf-per-watt不如GPU（A100比GenA高2.1×）。AUM进一步提出AU CPU与通用workload共享提升平台效率。AU适用于CPU-only serving小模型或CPU-GPU hybrid部署中的轻量AU workload。

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

