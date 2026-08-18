## Crest Factor（峰值因数，在线 K/V 格式选择代理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Crest Factor（峰值因数）κ 是信号处理中峰值与 RMS 之比（κ=max|x|/RMS(x)），衡量波形/分布尖峰程度。UNICORE 用它作为在线 K/V 激活量化时选择 DynFP 格式的轻量代理：不同 DynFP 格式（不同 E/M 布局）的量化信噪比（QSNR）随 κ 呈不同行为，可预计算一组阈值把 κ 映射到最合适的 E/M 布局。动机：K/V 激活在线量化（权重是离线搜索），逐 group 穷举全部 DynFP 候选代价过高，需要免穷举的轻量选择方法。κ 计算只需单趟 max-abs 归约 + RMS 归约（每元素 4 次标量运算），对 L≥2K 序列占 QKᵀ FLOPs 不足 0.2%，且完全 memory-bound，可无缝融合进量化 kernel。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 在线 K/V 量化流程：
    ```
    # 对每个 K/V group 激活张量 g
    max_abs = reduction_max(|g|)      # 一次 max-abs 归约
    rms     = reduction_rms(g)        # 一次 RMS 归约
    kappa   = max_abs / rms           # 峰值因数
    layout  = threshold_map(kappa)    # 预计算阈值查表 → 最优 E/M 布局
    q_g     = dynfp_quantize(g, layout)  # 按选定布局量化
    ```
    例：κ 高（分布尖峰/长尾）→ 选更大指数位布局（如 E2M1）保动态范围；κ 低（分布平坦/紧聚）→ 选更多尾数位布局（如 E1M2）保精度。对比离线权重侧用贪心搜索选格式（可穷举 96 候选），在线侧用 κ 阈值映射避免逐 group 评估候选。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：量化 kernel 内融合（max-abs + RMS 归约，每元素 4 次标量运算：reduction、sqrt、除法），阈值表离线预计算；UNICORE 的 CF 计算使量化 kernel arithmetic intensity 从 0.63 升到 0.87，仍 memory-bound 无可见开销（Llama-2-7B 激活量化占 prefill 时延 7.1%–20.7%、decode 0.3%–1.6%，且与 GEMM 大部分重叠）。使用：K/V 与 softmax 输出 P 用与激活相同 group size/位宽的量化，仅 K、V 做在线格式选择；是 DynFP 在 K/V cache 上的免校准在线落地方式。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
