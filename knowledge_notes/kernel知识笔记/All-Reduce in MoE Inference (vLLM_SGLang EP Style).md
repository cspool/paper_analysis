## All-Reduce in MoE Inference (vLLM/SGLang EP Style)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

在 vLLM 和 SGLang 的 Expert Parallelism (EP) 推理中，MoE 层的跨 GPU 通信使用 all-reduce 而非训练中的 all-to-all（Dispatch/Combine）。这是因为推理采用"replicated activations + distributed expert weights"方式：所有 GPU 上持有输入 activation 的完整副本，但 expert 权重按 EP 分布（每个 GPU 持有 E/EP 个 expert）。各 GPU 在本地计算自己的 experts 后，通过一次 all-reduce 聚合所有 GPU 的部分结果。这种方式消除了 Dispatch/Combine 所需的 token permutation 步骤。

训练 vs 推理的 EP 通信对比：
- **训练 EP**：Dispatch all-to-all（发送 token）+ routed expert 计算 + Combine all-to-all（收集结果）
- **vLLM/SGLang 推理 EP**：本地 expert 计算 + all-reduce（聚合部分结果）

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

vLLM EP 推理 MoE 层异步化（FarSkip 实现）：

```
# 常规同步模式:
for expert_id in local_experts:
    local_out[expert_id] = ExpertMLP(activation, expert_weights[expert_id])
full_out = all_reduce(sum(local_out))  # 阻塞! GPU 空闲

# FarSkip 异步模式:
local_out = fused_moe(activation, local_expert_weights)
all_reduce_handle = all_reduce(local_out, async_op=True)  # 立即返回

# 不等待 all-reduce，继续执行 attention (FarSkip 架构保证不依赖)
attn_out = attention(hidden_states)

# 同步点——在需要完整输出之前
all_reduce_handle.wait()
output = local_out + shared_expert_out + attn_out + residual
```

CUDA Stream 层面的执行（兼容 HIP/CUDA graphs）：

```
comm_stream = torch.cuda.Stream()
with torch.cuda.stream(comm_stream):
    handle = torch.dist.all_reduce(expert_out, async_op=True)  # PyNCCL
# compute stream 继续执行 → all-reduce 与计算 overlap
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 通信量：all-reduce 仅传输 d_model 维度的 output activation（而非 full token 的 d_model × num_tokens），推理中 activation 已复制
- CUDA Graphs 兼容：使用 PyNCCL（Python NCCL C API binding）替代标准 torch.dist，支持 graph capture
- 重叠率：Llama-4 95.3%, DeepSeek-V2 97.6%（FarSkip 论文数据）

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
