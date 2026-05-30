## Passing Block (传递块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Passing Block 是 APB 特有的设计：在每 host 上，由前序所有 host 的压缩 KV cache 拼接而成的上下文块。它在 attention 计算中作为 local context block 和 anchor block 之间的中间层，使当前 host 能"看到"前序 host 中最关键的 KV pair，从而弥补序列并行中 context 不可见的问题。Passing block 在 attention 计算后被丢弃，不参与 FFN 计算，也不持久化存储。

构造方式：Host h 通过 AllGather 获取所有 host 的压缩 KV cache (K^C_{1:H}, V^C_{1:H})，然后取前序 host 的压缩块拼接：P_h = (K_p^C, V_p^C) = (K^C_{1:h-1}, V^C_{1:h-1})。后续 host 的压缩块被忽略。

从算法pipeline角度拆解术语。

**Passing Block 在 APB 层的生命周期**：

```
// Step 1: 本地压缩
K_h^C, V_h^C = compress(K_h, V_h, l_p)    // 每 host 独立压缩

// Step 2: 全局收集
K_all = AllGather(K_h^C)                  // [H*l_p, d]
V_all = AllGather(V_h^C)

// Step 3: 构造 passing block（仅取前序 host）
K_p = K_all[0:(h-1)*l_p]                  // host 1 的 P_1 为空
V_p = V_all[0:(h-1)*l_p]

// Step 4: Attention 计算
K = [K_a; K_p; K_h]                       // anchor + passing + local
V = [V_a; V_p; V_h]
A = flash_attn(Q, K, V, mask=M')

// Step 5: 丢弃 passing block
// P_h 不进入 FFN，不缓存
H_a^out, H_h^out = FFN(A_a, A_h)          // 仅 anchor 和 local 通过 FFN
```

**Passing Block 的作用**（消融实验 Table 3）：
- 有 passing block (No.0)：E.MC = 72.00
- 无 passing block (No.4)：E.MC = 64.00（-8%）
- 有 passing block 但随机选择 KV (No.2)：E.MC = 66.00（retaining heads 关键）

**多 host 扩展性**（Table 4）：有了 passing block，APB 在 32K 输入下 H=2→8 性能稳定在 92-94；而 STARATTN（无 passing block）在 H=8 时降至 84，因 middle context 不可见。

术语一般如何实现？如何使用？

Passing block 通过 NCCL AllGather 通信实现。通信开销极小（0.62ms/block，<1% total），因为只传输 l_p=2K 个 token 的 KV cache（vs l_b=16K 的完整 local context）。AllGather 后本地拼接 KV tensors，送入修改的 FLASHATTN kernel。开源：https://github.com/thunlp/APB。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

---
