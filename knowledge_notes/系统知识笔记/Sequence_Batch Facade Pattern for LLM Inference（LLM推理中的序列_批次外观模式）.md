## Sequence/Batch Facade Pattern for LLM Inference（LLM推理中的序列/批次外观模式）

术语是什么？
Sequence/Batch Facade Pattern 是 QLLM 提出的一种软件设计模式，用于解决 LLM 推理系统中"模型期望看到 concat batch tensor"与"系统需要维护 per-sequence 独立状态"之间的矛盾。通过 Facade Pattern（外观模式，GoF 设计模式之一），Batch 对象对外呈现为单一 concat tensor（兼容现有模型代码），对内维护 per-sequence 独立 tensor——允许系统在任意时刻修改单个 sequence 的状态而不影响其他 sequence，且无需执行昂贵的 tensor split-merge 操作。

从系统架构角度拆解术语：
Facade Pattern 在 QLLM 中的实现原理：

```
# 传统方法（HF TGI / vLLM 使用）
class TraditionalBatch:
    """直接存储拼接后的单一大 tensor"""
    input_ids: Tensor          # shape: [batch_size, seq_len]
    attention_mask: Tensor     # shape: [batch_size, seq_len]
    
    def remove_sequence(self, idx):
        # 需要 split → 移除 → re-concat: O(B*S) 拷贝
        self.input_ids = torch.cat([
            self.input_ids[:idx], 
            self.input_ids[idx+1:]
        ])
        # 类似的 split-merge 发生在 KV cache, hidden states...

# QLLM Facade Pattern
class Sequence:
    """Per-sequence 独立存储"""
    input_ids: Tensor          # shape: [seq_len]
    kv_cache: KVCacheTensor    # 该 sequence 独占的 KV cache
    hidden_states: Tensor      # 当前 hidden state
    routing_weights: Tensor    # router 输出权重
    expert_assignments: list   # 分配的 expert IDs
    experts_completed: int     # 该 token 已完成几个 expert
    mask: Tensor               # attention mask

class BatchFacade:
    """外观层：对外暴露 concat view，对内维护 Sequence 对象"""
    sequences: list[Sequence]
    
    def __getattr__(self, name):
        """拦截模型对 batch tensor 的访问"""
        if name == 'input_ids':
            # 动态构建 concat view（zero-copy via tensor views）
            views = [s.input_ids.unsqueeze(0) for s in self.sequences]
            return torch.cat(views, dim=0)
        # ... 类似地处理 hidden_states, attention_mask 等
    
    def preempt_sequence(self, idx):
        """零拷贝抢占：直接保存 Sequence 引用"""
        seq = self.sequences[idx]
        seq.save_checkpoint()  # 保存 cache_entries, hidden, routing
        self.sequences.remove(seq)  # 从 batch 移除引用
        # 注意：tensor 本身未被拷贝或释放
    
    def restore_sequence(self, seq, idx):
        """零拷贝恢复：插入 Sequence 引用"""
        self.sequences.insert(idx, seq)
        seq.restore_checkpoint()  # 恢复 cache, hidden, routing
```

关键优势：
1. **Zero-copy preemption**：保存/恢复 Sequence 仅需操作引用（指针），不涉及 tensor 数据拷贝。
2. **Model transparency**：模型代码无需修改——`model.forward(batch.input_ids)` 通过 `__getattr__` 拦截获取 concat tensor，与原始实现无差别。
3. **Safe concurrent modification**：每个 Sequence 的 tensor 独立存储，对 sequence A 的状态修改不会污染 sequence B 的数据。
4. **No shape mismatch**：传统 split-merge 可能因 pad/unpad 导致 shape mismatch；Facade Pattern 下每条 sequence 独立维护自己的 shape。

术语一般如何实现？如何使用？
- **QLLM 实现**：在 Python 层面用 `__getattr__` 和方法重载实现。Batch 对象存储 `list[Sequence]`，Sequence 是 dataclass 包含所有 per-token 状态。由于当前 QLLM 是原型系统，Facade 在 Python 层运行；生产部署可将 Facade 下沉到 C++/CUDA 层以减少 Python overhead。
- **与其他方案的对比**：vLLM 的 PagedAttention 通过 block table 管理 per-sequence KV cache pages，但不提供统一的 "inner-layer sequence 操作" 抽象。QLLM 的 Facade Pattern 将 Sequence 抽象扩展到 KV cache + hidden states + routing metadata 全部状态。
- 适用场景：需要 inner-layer preemption 或 context switching 的 LLM 推理系统、支持 partial batch update（如在 decode 中途插入/移除 request）的 serving 框架。
- 局限性：dynamic concat view 构造仍有 Python overhead（`torch.cat` 每层执行一次）；对 batch 内所有 sequence shapes 不同时的 padding 处理需额外逻辑。

涉及论文标题：
- Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference
