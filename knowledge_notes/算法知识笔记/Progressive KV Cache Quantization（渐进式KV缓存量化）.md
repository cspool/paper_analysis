## Progressive KV Cache Quantization（渐进式KV缓存量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Progressive KV Cache Quantization 是 PM-KVQ 为长 CoT LLM 提出的一种逐步降低 KV Cache 位宽的量化策略。与传统的每步解码直接量化到目标位宽不同，渐进式量化在推理初期以高精度（FP16/INT16）存储 KV Cache，当内存预算被完全占用后，通过位宽收缩（bit-width shrinking）逐步将已存储的 KV Cache 降级到更低位宽（16→8→4→2 bit），为新 token 腾出内存空间。核心思想是"以时间换精度"：前期内存未满时保持零量化误差，后期再有损压缩早期 token，充分利用目标硬件内存预算，从而在相同总内存约束下降低累积量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**渐进式量化在长 CoT 推理中的执行流程（以 Fbit=2-bit, max_len=32K 为例）：**

```
// === Prefill 阶段 ===
K_cache = []  // 空缓存
V_cache = []
memory_used = 0
memory_budget = max_len * 2 * hidden_dim * 2_bytes  // 按Fbit=2计算内存预算
current_bitwidth = 16  // 初始以INT16精度存储

// === Decoding 阶段（逐步生成 token） ===
for t in 1..max_len:
    // Step 1: 计算当前 token 的 KV
    K_new, V_new = self_attention_layer.compute_kv(hidden[t])

    // Step 2: 检查内存——是否需要收缩？
    required_mem = memory_used + 2 * hidden_dim * current_bitwidth/8
    while required_mem > memory_budget:
        // 位宽收缩：16→8, 8→4, 4→2
        old_bitwidth = current_bitwidth
        current_bitwidth = current_bitwidth / 2  // 依次16→8, 8→4, 4→2

        // Equivalent Right Shift: X_b = ((2^{2b}-2^b+1)(X_{2b}+2^{b-1})) >> 3b
        // 例如 8bit→4bit: X_4 = ((2^8-2^4+1)(X_8+2^3)) >> 12
        for each stored_token in K_cache, V_cache:
            K_cache[stored_token] = eq_right_shift(K_cache[stored_token], current_bitwidth)
            V_cache[stored_token] = eq_right_shift(V_cache[stored_token], current_bitwidth)

        memory_used = len(K_cache) * 2 * hidden_dim * current_bitwidth/8
        required_mem = memory_used + 2 * hidden_dim * current_bitwidth/8

    // Step 3: 以当前位宽存储新 token
    K_cache.append(quantize(K_new, bits=current_bitwidth))
    V_cache.append(quantize(V_new, bits=current_bitwidth))
    memory_used = len(K_cache) * 2 * hidden_dim * current_bitwidth/8

    // Step 4: 带解量化的 attention 计算
    K_deq = dequantize(K_cache, bits=current_bitwidth)  // 含保留的首token和recent 128 tokens为INT16
    V_deq = dequantize(V_cache, bits=current_bitwidth)
    output[t] = attention(Q[t], K_deq, V_deq)
```

**位宽收缩阶段图（Gantt-style 描述）：**

以 32K 最大输出长度为目标，实际 token 生成为时间轴：
- Phase 1 (token 1 ~ ~8K): current_bitwidth=16, 零量化误差
- Phase 2 触发 (~8K): 内存预算耗尽，收缩到 8-bit → Equivalent Right Shift 压缩早期 token
- Phase 3 (~16K): 再次耗尽，收缩到 4-bit
- Phase 4 (~24K): 最终收缩到 Fbit=2-bit

与传统方案（每步都 2-bit）对比：前期~8K token 以 16 倍精度存储，累积误差显著降低。

术语一般如何实现？如何使用？

实现关键：(1) 位宽选择遵循 2 的幂（16/8/4/2），确保整数移位操作高效；(2) Equivalent Right Shift 等价于反量化→再量化，但仅通过整数加法和移位实现，避免浮点转换开销；(3) 保留首 token 为 INT16（attention sink 效应），最近 128 tokens 用滑动窗口保留 INT16（继承 KIVI/SKVQ 设计）；(4) 量化粒度为非对称分组量化 group_size=128。适用场景为长 CoT 推理（max output >8K tokens），尤其在 2-bit 极低精度下有显著收益。局限：(1) 需要已知目标内存预算和最大输出长度来规划位宽收缩节点；(2) 未覆盖 MLA 注意力机制。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---
