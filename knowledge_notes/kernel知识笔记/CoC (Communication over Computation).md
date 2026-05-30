## CoC (Communication over Computation)

术语解释
CoC 是 ETR 论文中用于优化 Ascend NPU 上 MoE 训练的通信-计算融合策略。通过 MTE (Memory Transfer Engine) 的远程内存访问将串行的矩阵乘法和 All-to-All 集合通信融合为统一细粒度 kernel，实现流水线并行执行。

术语是什么？
传统 MoE 训练中 expert FFN 的 MatMul 和 All-to-All 通信串行执行，计算和通信间有显著 idle bubble。CoC 利用 MTE 的远程 DMA 能力，在 AI CORE 执行当前 micro-batch 的 MatMul 时，MTE 同时发起下一 micro-batch 的 token 数据传输，实现计算-通信 pipeline overlap。

从kernel调度角度拆解：
```
# 传统串行
for micro_batch in batches:
    output = MatMul(tokens, expert_weights)     # AI CORE
    dispatch_tokens(output)                      # HCCL All-to-All
    # Total = T_compute + T_comm (idle bubbles)

# CoC 流水线
stream_comp = create_stream(AI_CORE)
stream_comm = create_stream(MTE)
for i, mb in enumerate(batches):
    launch_matmul_on(stream_comp, tokens[i])
    if i+1 < len(batches):
        prefetch_on(stream_comm, tokens[i+1])   # MTE 预取
    synchronize()
    # Total ≈ max(T_compute, T_comm) (重叠)
```

术语一般如何实现？如何使用？
在 Ascend CANN 上通过 MTE 实现。MTE 是 Ascend NPU 内的专用数据传输引擎，支持类似 RDMA 的远程内存直接访问，不占用 AI CORE 计算资源。CANN 编译器将 MatMul 和 All-to-All 标记为可融合算子对生成融合 kernel。论文仅简要描述此优化，未开源实现细节。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
