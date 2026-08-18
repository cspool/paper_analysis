## FlashInfer（native 与 paged 注意力 kernel 库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashInfer 是 LLM serving 的开源 kernel 库与生成器（github.com/flashinfer-ai/flashinfer，论文 arXiv:2501.01005），提供 prefill/decode/append attention、采样等 CUDA kernel，同时支持 paged 与 ragged 两种 KV 布局：paged 实现即 block-sparse attention（paged_kv_t 用 indptr/indices 数组做页索引，page_size 为块列数，page_size=1 即向量稀疏）；native 实现按 base+offset 在单连续区域访问 KV。wrapper（BatchPrefillWithPagedKVCacheWrapper 等）走 plan()→run() 两阶段：plan 调度变长输入、构建可复用辅助结构，run 跨 transformer 层复用 kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ConServe 的动机实验（同配置只换 KV 放置，Llama-3-8B、8K prefill+1K decode、A100、batch 1–16）：
```
# paged：每 access 查 vLLM block table、散页 gather
ptr = block_table[logical_block]; kv = ptr + intra_block_offset
if cross_block_boundary: ptr = block_table[next_block]   # 边界检查+查表
# native / ConServe：base+offset 连续流式
VA(t,l) = base + seg_off[l] + t * B_layer + delta         # 纯算术
```
结果：FlashInfer-paged prefill kernel 慢 12–24%；Nsight Compute 显示长 scoreboard stall 84.64% vs 79.37%、eligible warps/cycle 0.718 vs 0.825、SM/L2/DRAM 吞吐 −22.4%/−16.7%/−21.1%；多轮（每轮 512 输入+64 decode）paged/native 比从 1.2× 升至 1.75×。ConServe 用其 variable-length 模式 + 紧凑描述符（每序列 KV base 指针 + live 长度）承载连续寻址。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SGLang/vLLM/MLC-LLM/TRT-LLM/TGI 等默认集成；支持 FA2/FA3、CUTLASS/cuDNN 后端自动选择（Turing–Blackwell）、CUDA Graph、cascade attention（共享前缀层级 KV）、POD-Attention、稀疏注意力与 FP8/FP4 量化。使用：动态 batch serving 中替换自研 attention kernel；页大小可配（page_size=1 用于 SGLang 的 token 级 KV 裁剪）。Web 证据：https://github.com/flashinfer-ai/flashinfer 与 https://docs.flashinfer.ai/api/attention.html 确认 wrapper 与 paged KV API。

涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
