## STARATTN (Star Attention / 星形注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

STARATTN（Star Attention，NVIDIA, 2024）是一种结合序列并行与近似注意力的两阶段分布式推理方法。Phase 1（Context Encoding）：将长上下文划分为连续 block，每 block prepend 一个与 block 等大的 anchor block（文档首部），各 host 独立计算 anchor↔local 之间的 block-sparse attention，无跨 host 通信。Phase 2（Query Processing）：query 广播到所有 host，各 host 对本地 KV cache 计算 partial attention，通过 distributed softmax 合并各 host 结果获得全局精确 attention。

APB 相比于 STARATTN 的改进：(1) 使用更小的 anchor block（l_a = l_b/4 vs l_b），减少 FFN 开销；(2) 引入 passing block（压缩前序 host KV cache）弥补 STARATTN 中 middle context 不可见的问题；(3) 使用 retaining heads 而非 anchor-only 做 KV 选择。

从算法pipeline角度拆解术语。

**STARATTN Phase 1（Context Encoding）**：

```
// 序列划分
blocks = split(doc, H)  // [B_1, ..., B_H], each of size l_b
// 每 host（除 host 1）prepend anchor block
for h in 2..H:
    context[h] = [A, B_h]  // A = doc[0:l_b]，与 block 等大

// 每 host 独立计算，无通信
for h in 1..H:
    Q = qkv_proj(context[h])[0]
    K, V = qkv_proj(context[h])[1:]  // anchor + local 的 KV
    A_h = flash_attn(Q, K, V)
    H_h = FFN(A_h)
    // 仅保留 B_h 的 KV cache，A 的 KV cache 丢弃
```

**STARATTN vs APB 对比**：
| 维度 | STARATTN | APB |
|------|----------|-----|
| Anchor 大小 | l_a = l_b (16K) | l_a = l_b/4 (4K) |
| 跨 host 通信 | Phase 1 无 | AllGather (K^C, V^C) |
| Passing Block | 无 | 有（前序 host 的压缩 KV） |
| 多 host 扩展 | 退化（middle context 不可见） | 稳定（passing block 补偿） |
| 速度 (128K, 8 hosts) | 29,600 tok/s | 37,575 tok/s |
| 开源 | https://github.com/NVIDIA/Star-Attention | https://github.com/thunlp/APB |

术语一般如何实现？如何使用？

STARATTN 以两阶段方式集成到 HuggingFace Transformers 的 `model.generate()` 中：prefill 阶段替换为 Phase 1 block-sparse attention，decode 阶段替换为 Phase 2 distributed softmax。APB 在 Phase 1 基础上增加了 block compression（retaining heads）、AllGather 通信和 passing block 构造。STARATTN 开源：https://github.com/NVIDIA/Star-Attention (ICML 2025)。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
