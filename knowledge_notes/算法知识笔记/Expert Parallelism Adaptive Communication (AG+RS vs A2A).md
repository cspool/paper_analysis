## Expert Parallelism Adaptive Communication (AG+RS vs A2A)

术语是什么？
Expert Parallelism (EP) 将 MoE 的 expert FFN 分布到多个 GPU 上。标准的 EP 实现使用两次 all-to-all 通信（一次 dispatch + 一次 combine）。MegaScale-MoE 提出自适应通信模式：当 top-k > n（expert 并行度）时，将 all-to-all 替换为 all-gather + reduce-scatter。其原理是：当每个 token 路由到的 expert 数超过并行度时，每个 GPU 上必然有 expert 被某 token 选中，此时 all-gather 可收集所有 token（无信息损失），然后用 local scatter 丢弃不需要的 token。AG+RS 使用环形通信（仅相邻 GPU 通信），而 A2A 需每个 GPU 与其他所有 GPU 通信（全对全），因此 AG+RS 在大 top-k 场景下更高效。通信量公式：EP 为 2k/n × bsh(n-1)/n，TP 为 2bsh(n-1)/n。EP 的优势在于不切分 expert 的 intermediate dimension，保持完整 GEMM 效率。

从算法pipeline角度拆解术语：
EP 通信模式的自适应选择逻辑：
```
if top_k > n:   // n = EP 并行度
    // 使用 AG+RS 模式（环形通信，更高效）
    // Dispatch 阶段:
    gathered = All-Gather(ln2_out)  // [b, s/n, h] → [b, s, h]，每个GPU获得全部token
    ffn_in = Scatter(gathered, routing_map)  // local过滤，仅保留路由到本地expert的token → [b*s*k/n, h]
    // Expert 计算:
    fc2_out = SwiGLU_GroupedGEMM(ffn_in, expert_weights)  // [b*s*k/n, h]
    // Combine 阶段:
    gathered_out = Gather(fc2_out, routing_map)  // local组装 → [b, s, h]
    ffn_out = Reduce-Scatter(gathered_out)  // [b, s/n, h]，环形通信归约
else:
    // 使用 A2A 模式（标准EP实现）
    dispatched = All-to-All(ln2_out, routing_map)  // [b, s/n, h] → 各GPU收到路由给本地expert的token
    fc2_out = SwiGLU_GroupedGEMM(dispatched, expert_weights)
    ffn_out = All-to-All(fc2_out, reverse_routing_map)  // token归还原位
```
在 Mixtral-8×7B 上的实测显示：当 top-k > 6 时，AG+RS 通信时间低于 A2A（Figure 7）。

术语一般如何实现？如何使用？
- 基于 Megatron-LM 实现，自定义 CUDA scatter/gather 算子替代 torch.scatter_add/torch.gather，预计算 routing→memory mapping 实现高效数据传输。
- 每个 MoE layer 的 EP 限制在单 node 内（利用 NVLink 高带宽），跨 node 使用 PP 扩展。
- 负载均衡：使用 auxiliary loss + token dropping，以 GPU 为粒度（而非单个 expert）计算 balance loss 和 capacity。

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production
