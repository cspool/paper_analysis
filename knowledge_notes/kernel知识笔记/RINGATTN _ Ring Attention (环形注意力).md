## RINGATTN / Ring Attention (环形注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

RINGATTN (Li et al., ACL 2023) 是一种基于 ring-style P2P 通信的序列并行方法，用于跨多个 GPU 分布式计算精确注意力。核心思想：将长序列按长度均分到 H 个 GPU，GPU 排列成逻辑环。每个 GPU 计算自己的 local Q 与传递来的 K/V 之间的 partial attention，计算完当前块后将 K/V 传给下一个 GPU。经过 H-1 轮传递，每个 GPU 积累了完整序列的 attention 结果。通信与计算可重叠：一个 GPU 在计算当前 K/V 的 attention 时，同时接收下一个 K/V block。

RINGATTN 保持精确注意力的计算语义不变（FULLATTN），但通过分布序列并行来减少单 GPU 的计算量和显存占用。主要设计用于训练场景的长序列处理，也适用于推理。

从kernel调度角度拆解术语。

**RINGATTN 的 Ring Communication 流程**：

```
// H 个 GPU，每个 GPU 持有 local K_i, V_i (i=0..H-1)
// GPU i 的 local Q 为 Q_i

// 初始化
O_i = zeros(n/H, d)        // 累积输出
lse_i = -inf                // log-sum-exp for online softmax

send_K = K_i, send_V = V_i  // 初始发送自己的 K, V

// Ring loop: H-1 轮
for step in 0..H-1:
    recv_K, recv_V = recv_from_prev()      // 从上一个 GPU 接收
    // 通信可与前一 step 计算重叠

    // 计算 Q_i 与接收到的 K, V 的局部 attention
    A_partial, lse_partial = flash_attn(Q_i, recv_K, recv_V)

    // Online softmax 合并
    lse_new = max(lse_i, lse_partial)
    O_i = exp(lse_i - lse_new) * O_i + exp(lse_partial - lse_new) * A_partial
    lse_i = lse_new

    // 将收到的 K, V 继续传给下一个 GPU
    send_to_next(recv_K, recv_V)

// 最终归一化
O_i = O_i / sum(exp(lse_i))
```

**Wall-time 分解（128K, Llama-3.1-8B, 8 GPUs, per block）**：
- QKV Projection: 3.21 ms
- Communication: 18.40 ms（P2P ring, ~9% total）
- Attention: 152.12 ms
- FFN: 24.40 ms
- Total: 205.19 ms/block

术语一般如何实现？如何使用？

RINGATTN 通过 PyTorch 的 NCCL P2P send/recv 原语实现 ring communication。在 HuggingFace Transformers 中，替换 Attention 层的 forward 为 ring-attention 版本。主要超参数：序列并行度 H、block/chunk size。开源参考实现：https://github.com/zhuzilin/ring-flash-attention。APB 论文中使用 RINGATTN 作为 baseline。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

---
