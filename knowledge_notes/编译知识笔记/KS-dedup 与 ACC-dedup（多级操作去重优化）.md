## KS-dedup 与 ACC-dedup（多级操作去重优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 FlashTFHE 编译器在 MLIR FHELinAlg dialect 之上实现的两个程序级去重优化（论文 Section V"Multi-level Operation Deduplication"）：(1) KS-dedup（key-switching 去重）——multi-bit 程序常对同一 ciphertext 应用多个不同 LUT（fanout 结构），FlashTFHE 采用 key-switching-first 的 PBS 顺序（区别于 Boolean TFHE 的 blind-rotation-first），使 key-switching 结果可作为多个后续盲旋转的公共输入被广播复用，最多减少 47.12% 的 key-switching 操作；(2) ACC-dedup（GLWE accumulator 去重）——程序常对多个 tensor element 应用同一 accumulator（如逐元素标量乘/加），共享 accumulator 使 GLWE 存储需求减少 91.54%、显著缩小程序体积与 DRAM 容量需求。两者都依赖"把 PBS 视为非原子操作、拆开其内部步骤做程序级分析"这一前提（Takeaway 6）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译管线中的位置：输入 FHELinAlg dialect → batch 展开（matmul/multi-LUT → per-ciphertext primitives）→ 依赖分析（构建 ciphertext 间数据依赖与 fanout 图）→ KS-dedup：识别同一 LWE ciphertext 的多个 LUT 消费点，把 KS 结果缓存并广播到多个 BRU，而非每 LUT 重做一次 KS → ACC-dedup：识别共享 accumulator 的 tensor 元素，合并其 GLWE 存储 → adaptive batching 决策 → 输出指令流。伪代码示意：
```
for c in ciphertexts:
    ks_c = KeySwitch(c)                 # 一次 KS
    for lut in luts[c]:                 # fanout：多个 LUT 复用 ks_c
        BlindRotation(ks_c, lut)        # 广播到不同 BRU（KS-dedup）
for e in tensor_elements:
    acc[e] = shared_acc                 # 共享 accumulator（ACC-dedup）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译期 pass（依赖分析 + 公共子表达式消解），依赖 key-switching-first 执行顺序与多 BRU 广播能力。使用：面向 LUT 密集、张量元素结构化的 multi-bit 程序；与硬件侧 adaptive batching 协同（去重后并行度信息喂给 batch 决策）。评估：消融（Figure 20）显示两级去重在 temporal 设计 + 异构 FFT + adaptive batching 之上进一步改善 delay/EDP/EDAP；KS-dedup 最高 47.12%、ACC-dedup 91.54% 的存储削减。论文未开源编译器（联网未找到仓库）。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
