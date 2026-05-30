## FlashInfer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashInfer 是高性能 LLM serving 的 attention kernel library，由华盛顿大学、CMU 等开发的开源项目（https://github.com/flashinfer-ai/flashinfer）。提供 attention（prefill/decode）、top-p/top-k sampling、KV cache 管理等 GPU kernel。核心特征：(1) PagedAttention 兼容——支持 paged KV cache layout；(2) Varlen attention——支持不同 sequence length 的 batch 处理；(3) Load balancing——flatten head dim 解决 head 间不平衡；(4) 高效 sampling kernels——包括 top-p/top-k/temperature sampling。Twilight 基于 FlashInfer 构建所有 kernel（SpGEMV、top-p binary search、varlen attention），并修改其 decode attention kernel 支持 INT4 K cache、修改 top-p sampling kernel 用于 attention weights。

从系统架构角度拆解术语，给出具体例子。
FlashInfer 在 LLM serving 中的位置：
```
Serving System (vLLM / SGLang)
  └─ Attention Module
       └─ FlashInfer kernels:
            - Prefill: FlashInfer prefill attention (FlashAttention-style)
            - Decode: FlashInfer decode attention (GEMV optimization)
            - Sampling: FlashInfer top-p/top-k sampling
            - KV Cache: PagedAttention-compatible layout
```
Twilight 集成方式：在 FlashInfer 的 decode attention pipeline 中插入 Pruner kernel（SpGEMV + Top-p），最终 attention kernel 复用 FlashInfer 的 varlen attention。

术语一般如何实现？如何使用？
作为 Python 包安装（pip install flashinfer），提供 PyTorch 接口。CUDA/Triton 实现。Twilight 的修改：(1) SpGEMV kernel——基于 decode attention kernel 改造，加入 INT4 dequantization 和 2-stage pipeline；(2) Top-p kernel——基于 sampling kernel 改造，从"选 token"变为"选 attention weights"。是 vLLM/SGLang 等 serving 框架的底层 kernel 依赖之一。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

---
