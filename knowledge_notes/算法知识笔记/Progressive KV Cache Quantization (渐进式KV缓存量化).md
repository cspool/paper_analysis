## Progressive KV Cache Quantization (渐进式KV缓存量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Progressive KV Cache Quantization 是 PM-KVQ 提出的一种面向 long-CoT LLM 的 KV Cache 量化策略。核心思想是不在每个 decoding step 直接将 KV Cache 量化到目标 Fbit（Final bit-width），而是从 16-bit 开始逐步缩减位宽。当预留给当前 block 的显存预算被占满时，通过位宽缩减操作将已存储的 KV Cache 从当前位宽降一档（16→8→4→2 bit），腾出空间后继续以新的低位宽存储后续 token。最终位宽 Fbit 由块级内存分配确定。

与 baseline（KIVI 等在每个 step 直接将 KV Cache 量化为目标位宽）相比，渐进量化充分利用了生成初期的空闲显存。例如在 Fbit=2、max context=32K 的场景下：前 ~2K token 以 16-bit 存储（零量化误差）→ 随后 ~2K token 以 8-bit 存储（极低误差）→ 后续 ~4K token 以 4-bit → 最后部分以 2-bit 存储。由于 long-CoT 推理的误差传播特性，前期 token 的精度比后期更重要——渐进策略恰好保证了这一需求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

渐进量化在 LLM 推理 decoding 阶段的执行流程：

```
# 对每个 transformer block，目标 Fbit 已知
# KV Cache 按位宽分段存储: segment_16, segment_8, segment_4, segment_2

for each decoding step t:
    K_new, V_new = compute_KV(current_token)  # FP16
    
    current_usage = Σ memory_of(all segments)
    
    if current_usage + FP16_token_size > budget:
        # 找当前最高位宽的非空段并降档
        if segment_16 not empty:
            segment_8_new = equivalent_right_shift(segment_16, 16→8)
            segment_8 = concat(segment_8, segment_8_new)
            segment_16 = empty; current_bit = 8
        elif segment_8 not empty:
            segment_4_new = equivalent_right_shift(segment_8, 8→4)
            segment_4 = concat(segment_4, segment_4_new)
            segment_8 = empty; current_bit = 4
        elif segment_4 not empty and Fbit == 2:
            segment_2_new = equivalent_right_shift(segment_4, 4→2)
            segment_2 = concat(segment_2, segment_2_new)
            segment_4 = empty; current_bit = 2
    
    store(K_new, V_new, bit=current_bit)
```

**Annotations**: 位宽档位按 2 的幂次排列。降档触发条件为 `current_usage + new_token > budget`。整个推理过程中仅执行 log2(16/Fbit) 次降档（每次影响大量已有 token），而非每个 step 降档。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

位宽缩减通过 Equivalent Right Shift 实现（纯整数乘加+移位）。渐进量化与块级内存分配正交：每个 block 有独立的 Fbit 和独立的渐进量化过程。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs
