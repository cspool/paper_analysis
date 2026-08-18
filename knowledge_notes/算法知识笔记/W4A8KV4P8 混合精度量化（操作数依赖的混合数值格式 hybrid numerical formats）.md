## W4A8KV4P8 混合精度量化（操作数依赖的混合数值格式 hybrid numerical formats）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4A8KV4P8 是 P3-LLM（ISCA 2026，Cornell/KU Leuven/Stanford）为 edge LLM inference 设计的混合精度量化方案，含义为：权重（W）4-bit、激活（A）8-bit、KV-cache 4-bit、注意力分数（P，attention-scores）8-bit。其核心思想是 operand-dependent quantization——不为所有操作数套用同一数值格式，而是根据每个 LLM 操作数的数值分布特征（动态范围、outlier 模式、符号性、数值范围）分配专属数值格式，同时兼顾"内存压缩率-模型精度-硬件计算效率"三者的平衡。四个操作数各自采用不同格式：权重用 4-bit BitMoD（FP4 负零编码重映射特殊值 {±5,±8}，group 128）；KV-cache 用 4-bit INT4-Asym（非对称整数，per-head 量化，128 元素共享 16-bit scale + 4-bit zero-point，有效精度 4.16 bit）配合动态输入感知平滑抑制 key cache outlier；激活用 FP8-E4M3（per-token，宽动态范围容纳 outlier，无需 Hadamard 变换或 SmoothQuant 式平滑）；注意力分数用无符号 FP8-S0E4M4（softmax 后值域 [0,1]，无需符号位，4-bit 指数 bias −15 覆盖 [−14,−1]，4-bit 尾数提供数值保真）。动机依据（Fig.3 分析）：权重和 KV-cache 占解码阶段内存主导、对量化不敏感（4-bit 可接受），激活与注意力分数占内存小但对量化敏感（保持 8-bit），且 8-bit 注意力分数使 P·V 能完全跑在低精度 PIM 上（否则 value cache 需搬回 NPU 用 FP16 计算）。通用量化公式（Eq.1）：Δ=|X|_max/Q_max，X_Q=Round(X/Δ,Q)，X̃=X_Q·Δ。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 decode 迭代中 W4A8KV4P8 的计算流（以 Llama-3.1-8B post-RoPE 为例）：
```
# 预填充阶段（离线/在线各一次）：
for c in 0..H-1:  s_c = Max(|K_prefill[:,c]|)        # 动态平滑因子（无校准数据集）
for c in 0..H-1:  K_S[:,c] = K[:,c] / s_c             # 平滑后 key cache
# 权重离线：BitMoD 4-bit，group=128，搜索最优特殊值替换负零
# 解码阶段（每 token）：
h = layer_input(token)
q = Q_linear(h); q = RoPE(q)                          # NPU 高精度
q = FP8_E4M3(q * s_K)                                 # SSF 融合进 query 缩放
for head:  K_new = K_linear(h); K_new = RoPE(K_new)
           Kq = INT4_Asym(K_new / s_c)                # per-head 量化
           P = PCU_GEMV(q, K_S^T)                     # Q·K^T 在 PIM
           P = softmax(P)                             # NPU
           P8 = FP8_S0E4M4(P * S^V / S^V_max)         # 融合 value 缩放
           O = PCU_GEMV(P8, V_INT4) * S^V_max         # P·V 在 PIM
out = O_linear(O)                                     # 线性层 GEMM，dequant 后置
```
效果：平均 perplexity loss 仅 0.25（Wikitext-2）/0.31（C4），比 QuaRot（0.30/0.48）与 QoQ（0.30/0.38）更低；MMLU/ARC-C/GSM8K 平均准确率比 QuaRot/QoQ 高 2.57%/3.05%；decode 内存相对 FP16 降 3.7×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/yc2367/P3-LLM（MIT，wkvaq_quant/ 代码库 + AWQ、LM-Eval 子模块）。使用流程：`run_awq.sh`（`wq_dtype=int|bitmod`，4-bit group 128）做权重量化 → `run_awq_save_4b_model.sh` 保存 fake-quant 模型 → `test_ppl_template.sh`（flag：`--kv_quant_method KTVT/KCVT`、`--apply_k_scale`、`--k_quant_post_rope`、`--p_bits`）测 Wikitext-2/C4 perplexity（仅支持 Llama/Mistral）。在硬件侧，P3-LLM 把这套格式与低精度 PCU 协同设计：6-bit 定点乘法器（5-bit 尾数含隐藏位 + 符号位）同时解码 BitMoD 权重与 INT4-Asym KV，4-bit 指数移位替代浮点乘法中的指数对齐，从而使 PIM 在等面积下获得 4× 计算吞吐（详见本库 PIM/DRAM-PIM 与 Throughput-Enhanced PCU 条目）。

涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
