## 在线激活/KV 量化 kernel（crest factor 计算 + 与 GEMM 执行重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 在线激活/KV 量化 kernel 是量化 LLM 推理中负责把运行期产生的激活（以及 KV cache 的 K/V）按 group 量化到低比特浮点格式的运行时计算 kernel。UNICORE 论文（ISCA'26）在硬件模拟器层面评估其开销：激活量化在 Llama-2-7B 上占 prefill 时延 7.1%–20.7%、decode 仅 0.3%–1.6%（序列 512–8192），且可大部分与 GEMM 执行重叠。UNICORE 的量化 kernel 额外融合 crest factor（CF）计算（max-abs 归约 + RMS 归约 → κ=峰值/RMS → 阈值查表选 DynFP E/M 布局），使算术强度（arithmetic intensity）从 0.63 提到 0.87（额外 reduction、sqrt、division），但仍 memory-bound、无可见开销；对 L≥2K 序列 CF 计算占 QKᵀ FLOPs 不足 0.2%。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 在线量化 kernel 伪代码（每 group 流式处理）：
    ```
    # 对每个 group 的激活张量 g（32 元素 group）
    # 单趟 streaming 归约：
    max_abs = max(|g|)                 # reduction 1：峰值
    sq_sum  = sum(g*g); rms = sqrt(sq_sum/n)   # reduction 2 + sqrt：RMS
    kappa   = max_abs / rms            # crest factor
    layout  = cf_threshold_map(kappa)  # 阈值查表 → DynFP E/M 布局
    scale   = max_abs / max_rep(layout)  # 8-bit scale
    q       = dynfp_quantize(g, layout, scale)  # 量化输出 + 格式索引/scale 元数据
    ```
  - 调度/重叠：量化 kernel 与后续 GEMM 在硬件流水上重叠执行（prefill 中激活量化占时延 7.1%–20.7% 但大部分重叠、decode 中占 0.3%–1.6%），量化后的低比特激活/缓存以原始格式存储传输（E3M2 仅存在于 UNICORE 计算数据通路内）；CF 计算使算术强度 0.63→0.87 但仍 memory-bound（QKᵀ FLOPs <0.2%）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：UNICORE 中该 kernel 运行在加速器 Vector Unit/在线量化数据路径（artifact Software/Accuracy/ 的 unicore_kernel 提供 PyTorch 实现），与 GEMM 在硬件流水上重叠；K/V 与 softmax 输出 P 用与激活相同 group size/位宽量化，仅 K、V 做 crest factor 在线格式选择。使用：在 prefill/decode 全流程对激活（及 KV cache）逐 group 量化，配合离线权重量化（贪心 palette 搜索）形成完整 W/A/KV 低比特推理管线；其 memory-bound 特性使量化开销在带宽受限的 decode 阶段几乎可忽略。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
