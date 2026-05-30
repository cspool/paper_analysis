## Dynamic All-to-All in MoE Gating（MoE 门控中的动态全交换通信 / 可变大小 All-to-All）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic All-to-All in MoE Gating 是 Huang et al. (NeurIPS 2024) 在 "Toward Efficient Inference for Mixture of Experts" 中提出的 Dynamic Gating 机制中的通信模式创新。传统的 MoE Expert Parallelism 使用 NCCL all-to-all 在等大小消息的假设下（每个 expert 发送/接收固定 S×C tokens），因为所有 GPU 预知消息大小，可直接分配 buffer。Dynamic Gating 消除固定 capacity 后，消息大小变为可变——需要两轮 all-to-all：(1) 第一轮通知 sizes（每个 GPU 告知其他 GPU 将接收多少 tokens）；(2) 第二轮传输实际 tokens（可变大小，按第一轮获得的 sizes 动态分配 buffer）。

第一轮 all-to-all 仅传输标量整数（每个 expert device 对应一个 int64），平均延迟 20µs，几乎可忽略。第二轮 all-to-all 传输可变大小 token tensors（vs 静态的 zero-padding filled tensors）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// Dynamic Gating 的通信调度 (每个 GPU 执行)

// Phase 0: Local computation (gating + argsort + bincount)
vector<int> sizes(E, 0);  // sizes[e] = 本 GPU 将发送到 expert e 的 token 数
Tensor sorted_tokens = argsort_and_bincount(tokens, assignments, sizes);

// Phase 1: All-to-All Round 1 — Size Notification
// 本 GPU 需告知: "我将在 Round 2 向你的 expert e 发送 sizes[e] 个 tokens"
// 每个 GPU 发送 E 个 int64，接收 E 个 int64
NCCL_AllToAll(sizes.data(), sizes.data(), 1, ncclInt64, comm, stream);
// → 现在 recv_sizes[src_gpu * E + e] = GPU src_gpu 将向本 GPU 的 expert e 发送的 tokens 数
// 总 incoming tokens = sum(recv_sizes)
// → 预分配 recv_buffer[total_incoming_tokens * D]

// [期间可与其他计算 overlap — 论文提到 "latency hidden"]

// Phase 2: All-to-All Round 2 — Token Transfer
// 发送: sorted_tokens split by sizes[i % (E/P)] for each target GPU
// 接收: variable-length tokens into pre-allocated recv_buffer
NCCL_AllToAll(send_tokens, recv_tokens, send_counts, recv_counts, ncclFloat, comm, stream);
// send_counts, recv_counts 按 expert 分组，每 GPU one entry per target GPU

// Phase 3: Expert Computation
vector<Tensor> expert_outputs;
int offset = 0;
for (int e = 0; e < num_local_experts; e++) {
    int n = recv_counts[e];  // 实际收到的 tokens 数
    if (n > 0) {
        Tensor tokens_e = recv_buffer.slice(offset, offset + n);
        expert_outputs.push_back(FFN[e](tokens_e));
        offset += n;
    }
}

// Phase 4: All-to-All Round 3 — Output Collection
// 将 expert outputs 送回原始 GPU（sizes 对称，可用相同模式）
NCCL_AllToAll(output_tokens, returned_tokens, ...);
// → 按 inverse permutation 还原 token 顺序
```

对比 Static Gating 的单轮通信：
```
// Static Gating: 单轮 all-to-all (固定大小)
// 每 GPU 向每 target GPU 发送 exactly S×C tokens (包括 zeros)
NCCL_AllToAll(dispatched_tokens, received_tokens, 
              S*C*sizeof(float), ncclFloat, comm, stream);
// 通信量: E × S×C × D × sizeof(float)
// 其中大量 zeros → waste

// Dynamic Gating: 两轮 all-to-all (可变大小)  
// Round 1 通信量: E × sizeof(int64) ≈ 512 × 8B = 4KB (trivial)
// Round 2 通信量: 2S × D × sizeof(float) (仅实际 tokens, 无 waste)
// → 节约: (ECS - 2S) × D × sizeof(float) bytes
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NCCL `ncclAllToAll` 支持可变大小（通过 sendcounts/recvcounts 参数或 `ncclAllToAllv`）。关键实现注意事项：(1) Round 1 与 Round 2 之间可插入其他计算（如 token split/reorder）实现 overlap；(2) 第一轮的 20µs 开销在 multi-node 场景下依然 trivial（因此 multi-node scaling 表现更好）；(3) `ncclGroupStart/End` 可将 Round 2 的 send/recv 分组以减少 collective launch overhead。多节点时，减少的 all-to-all 通信量抵消增加的一轮通信开销，吞吐提升更显著（11.55× vs static）。

涉及论文标题：
- Toward Efficient Inference for Mixture of Experts
- Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference
