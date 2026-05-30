## Unified Dynamic KV Cache（统一动态KV缓存）

术语是什么？
Unified Dynamic KV Cache 是 QLLM 提出的一种 KV cache 管理模块，解耦了 sequence-level 和 batch-level 的 KV cache 操作。传统系统中 KV cache 作为 batch concat tensor 的一部分，对单个 sequence 的 cache 操作（如增长、释放、迁移）需要从大 tensor 中定位、split、修改后 re-concat——O(N) 复杂度且易 SxS（shape mismatch）。QLLM 的 Unified Dynamic Cache 通过为每个 Sequence 对象维护独立的 KV cache tensor，在 Batch Facade 层面提供统一访问接口（对外呈现为连续 cache block），避免 split-merge。

从系统架构角度拆解术语：
传统 KV cache（batch-concatenated） vs Unified Dynamic KV Cache（per-sequence）：

```
# 传统 KV Cache (HF TGI / vLLM — logical view)
kv_cache = Tensor[batch_size, num_layers, 2, num_heads, max_seq_len, head_dim]
# 每个 sequence 的 cache 行交错分布在 batch 维度
# Preemption 时需要:
#   1. 找到目标 sequence 的 cache 行索引
#   2. 从 batch tensor 中 split 出该行: cache[idx:idx+1]
#   3. 重新 concat 剩余 batch: cat(cache[:idx], cache[idx+1:])
#   4. 恢复时反向操作: insert + re-concat
# → 每层 KV cache 约 2×32×64×head_dim bytes，split-merge 开销显著

# QLLM Unified Dynamic KV Cache
class UnifiedDynamicCache:
    """Per-sequence KV cache + batch-level unified access"""
    seq_caches: dict[SequenceID, KVCachePerSeq]
    # 每个 Sequence 维护自己的 KV cache:
    # KVCachePerSeq: list[Tensor]  # [layer_0_cache, layer_1_cache, ...]
    #  每个 layer cache: Tensor[num_layers, 2, num_heads, cur_seq_len, head_dim]
    
    def get_batch_view(self, seq_ids, layer):
        """为 batch 构建当前层的 unified view"""
        caches = [self.seq_caches[sid][layer] for sid in seq_ids]
        # 动态 concat 生成 batch view（仅在访问时执行）
        return torch.cat(caches, dim=0)
    
    def grow(self, seq_id, layer, new_k, new_v):
        """Per-sequence 增长：仅操作目标 sequence 的 cache"""
        self.seq_caches[seq_id][layer] = torch.cat([
            self.seq_caches[seq_id][layer], 
            torch.stack([new_k, new_v])
        ], dim=-2)
    
    def preempt_save(self, seq_id):
        """抢占保存：直接持有引用，零拷贝"""
        return self.seq_caches[seq_id]  # 返回引用
    
    def preempt_restore(self, seq_id, saved_cache):
        """抢占恢复：恢复引用，零拷贝"""
        self.seq_caches[seq_id] = saved_cache
```

术语一般如何实现？如何使用？
- **QLLM 实现**：在 Python/PyTorch 层面维护 `dict[SequenceID → list[layer_cache_tensor]]` 映射。每次 attention 计算时，通过 Facade Pattern 动态构建 concat view（`torch.cat` 开销约 <1ms for batch_size=32）。Per-sequence grow 避免跨 sequence 拷贝。
- **与 PagedAttention (vLLM) 的区别**：PagedAttention 使用固定大小的 physical blocks + block table mapping 管理 KV cache，解决碎片化和内存浪费问题。QLLM 的 Unified Dynamic Cache 聚焦于 preemption 场景下的 zero-copy state save/restore，不解决内存碎片问题。两者是互补设计——Unified Dynamic Cache 的 per-sequence logical cache 可部署在 PagedAttention 的 physical block 层之上。
- 适用场景：需要 frequent context switching 或 preemption 的 LLM serving 系统；multi-tenant 推理（不同 tenant 的 sequence cache 隔离）。

涉及论文标题：
- Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference
