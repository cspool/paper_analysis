## Log-Distributed Token Selection for KV Cache (对数分布Token选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Log-Distributed Token Selection 是 LogQuant (Chen et al., 2024) 提出的 KV Cache token 保留策略，利用 base-2 对数分布选择哪些 token 保留为全精度（BF16），哪些量化为 INT2。核心观察是：LLM 中 attention spikes（高注意力分数的位置）遵循对数分布——距离当前位置越远的 token，其 attention spikes 的密度越稀疏。基于这一观察，LogQuant 以几何递减的密度保留 token：最新 W 个 token 密度 p，次新 W 个 token 密度 p/2，再次新 W 个 token 密度 p/4……

这与 KiVi 的"均匀最近窗"（仅保留最近 R 个 token）形成对比：均匀窗在远处硬截断会丢失关键的远距离 token，而对数窗在远处仍有稀疏保留。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Log-Distributed Token Selection 算法伪代码（Algorithm 1 from LogQuant）**：
```
Input: A (list of full-precision tokens), a* (new token), W (window length)
Output: A (updated list of tokens)

procedure APPENDTOKEN(A, a*, W):
  if length(A) < 3W:               // Cache未满：直接追加
    A ← concat(A, a*)
  else:                            // Cache已满：压缩 + 追加
    A ← concat(A[0:2W:2], A[2W:3W])  // 前2W个token步长=2子采样（密度减半）
    A ← concat(A, a*)                // 追加新token
  end if
  return A
end procedure
```

**具体执行流程**（W=42，以 Llama3.1-8B 为例）：
```
Step 1-125: 直接追加全精度token，cache长度递增
Step 126:   cache长度=126=3W，触发压缩
            A[0:84:2] → 保留42个（前84个中隔1取1）
            A[84:126] → 保留42个（全保留）
            追加新token → cache长度=85
Step 127-168: 直接追加，cache长度增至3W
Step 169:   再次压缩：A[0:84:2] → 效果：Window_0密度p/4，Window_1密度p/2
...
```

密度演化：最初 3W 个 token 均为全精度 → 第一次压缩后，旧 W 个保留密度 1/2，新 W 个密度 1 → 第二次压缩后，最旧 W 个密度 1/4，次旧 W 个密度 1/2，最新 W 个密度 1。自然形成 log₂ 递减密度。

Token Coverage 评估（公式 1）：Coverage = Σ(所选 token 的 attention score) / (3W)。该指标衡量选择方案捕获注意力质量的能力——越高表示保留的高注意力 token 越多。实验（Figure 4）显示 LogQuant 的对数选择在 Llama3-8B、Qwen1.5-7B、Phi3-mini 上均优于 KiVi（均匀窗）、StreamingLLM 和 H2O。

术语一般如何实现？如何使用？

在 LogQuant 实现中，对数分布选择通过 HuggingFace transformers 的 Cache 派生类完成。W = ⌊KiVi_R/3⌋（确保全精度 token 数不超过 KiVi 的 R）。对于 R=128，W=42，LogQuant 最多保留 126 个全精度 token（< KiVi 的 128）。未被对数选择保留的 token 被量化为 INT2（通过 Quanto 的 Key-per-channel 量化，group_size=64）。

使用方式：(1) 替代 KiVi 的均匀窗——直接替换 Cache 类；(2) 与 compression-aware 量化后端（Quanto/HQQ）结合；(3) 与 position-agnostic 重排结合——选择保留的 token 被连续存储以改善内存局部性。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

---
