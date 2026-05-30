## PagedAttention（分页注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PagedAttention 是 vLLM（UC Berkeley）提出的 LLM serving 系统的 KV cache 内存管理机制，灵感来自操作系统的虚拟内存分页。核心思想：将 KV cache 划分为固定大小的 block（如 16 或 32 token），通过 block table 将逻辑 KV cache 序列映射到非连续的物理 GPU 内存块。这允许按需动态分配 KV cache 内存（类似 OS 的 demand paging），消除预分配连续内存导致的内部碎片（传统方式需预分配 max_seq_len 连续空间，实际利用率仅 20-30%）。PagedAttention 支持 block 级别的内存共享——不同序列的相同 prefix（如 system prompt）可共享同一物理 KV cache block（Copy-on-Write 语义）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
vLLM Serving with PagedAttention:

请求到达 → Scheduler:
  1. 为新请求分配逻辑 KV blocks（Block Manager）
  2. Block Table: [req_id] → [physical_block_0, physical_block_1, ...]
  3. 若共享 prefix，指向已有物理 block（引用计数+1）

Prefill 阶段:
  for each token in prompt:
    compute K,V → 写入当前逻辑 block
    if block full:
      allocate new physical block from free pool
      append to block_table[req_id]

Decode 阶段 (PagedAttention Kernel):
  for each query token q_i:
    for each block_id in block_table[req_id]:
      K_block = KV_cache[block_id]     # [block_size, num_heads, head_dim]
      V_block = KV_cache[block_id]
      scores_block = q_i @ K_block^T / sqrt(head_dim)
      // block内完整计算，跨block通过running statistics合并
      o_i += softmax(scores_block) @ V_block

内存释放:
  当请求完成 → block引用计数-1 → 若count=0则回收至free pool
```

相比朴素 KV cache 管理的优势：(i) 内存浪费从平均 60-80% 降至接近 0；(ii) 支持更大 batch size（可用内存更高效）；(iii) prefix sharing 进一步减少冗余存储——在 chatbot 场景中所有请求共享 system prompt 的 KV cache。实验结果：up to 29× 吞吐提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

vLLM 开源实现（https://github.com/vllm-project/vllm）。需要自定义 CUDA kernel 支持 block 级 indirect memory access（通过 block_table 索引）。S-LoRA 将 PagedAttention 扩展为 Unified Paging，统一管理多个 LoRA adapter 的 KV cache。vAttention 进一步发现可通过修改 CUDA virtual memory API 让 OS 做物理内存重分配（而非自定义 kernel），在 vLLM 基础上提升 1.29× 端到端吞吐。TensorRT-LLM 和 HuggingFace TGI 也集成了 PagedAttention。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
