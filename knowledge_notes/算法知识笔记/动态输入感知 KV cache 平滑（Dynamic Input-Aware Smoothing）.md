## 动态输入感知 KV cache 平滑（Dynamic Input-Aware Smoothing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
动态输入感知平滑是 P3-LLM 抑制 key cache outlier 通道的量化预处理技术。LLM 的 key cache 在固定通道上呈现明显 outlier（Fig.5(b)(f) 显示 pre-RoPE key cache 有结构化 outlier 通道，而 value cache 无 outlier），直接 4-bit 量化误差大。P3-LLM 对每个 key 通道除以该通道的绝对最大值（平滑因子），把数值压到 [−1,1]：K_S[:,c]=K[:,c]/Max(|K[:,c]|)。与既有方案的两个关键区别：(1) 无需校准数据集——Oaken 靠离线校准定 outlier 阈值、QoQ/SmoothQuant 靠校准定平滑因子，都会对新数据集过拟合（Fig.8 显示 QoQ 用 Pile 校准在 Wikitext-2/C4 上量化误差最高）；P3-LLM 的平滑因子直接在 prefilling 阶段从当前输入计算，动态感知输入；(2) 同时研究 pre-RoPE 与 post-RoPE 两种量化位置——通过 profiling 发现 RoPE 旋转对 key cache 分布的影响取决于最大序列长度：Llama-2（4K 序列）post-RoPE key cache 结构化 outlier 被打散（不利量化），故用 pre-RoPE 量化；Llama-3（128K 序列）在典型 4K context 下旋转角很小、post-RoPE 分布几乎不变（保留结构化 outlier 利于量化），故用 post-RoPE 量化。开销分析：每个通道一个平滑因子（所有 token 共享），额外内存 <1%（与 context 长度成反比）；平滑因子计算仅需 prefilling 上下文，A6000 上 Llama-3.1-8B 全层 <5ms（即使 32K context），相对 250ms TTFT SLO 仍 <2%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Prefilling（每层每 head）：
for c in range(H):                       # H = head dimension
    s_c = max(abs(K_prefill[:, c]))      # 每通道绝对最大值
    K_S[:, c] = K_prefill[:, c] / s_c    # 平滑到 [-1,1]
save s_c                                 # decode 阶段复用（内存 <1%）
# Decoding（每 token）：
K_new = RoPE(K_linear(h))
Kq_new = INT4_Asym(K_new / s_c)          # 用同一平滑因子后量化
# 硬件融合：Q·K^T 时把 s_c (SSF) 元素乘进 query，再 FP8 量化 query
q8 = FP8_E4M3(q * s_c)
score = PCU_GEMV(q8, Kq^T)               # 无需在线对 K 解量化
```
效果：消融（Table VI）显示动态平滑把 Llama-2-7B/Llama-3.1-8B 的 Wikitext-2 PPL 降低 0.10/0.17；相对 Oaken（有效 4.8 bit）在更低有效精度（4.16 bit）下 perplexity 更好，且避免校准过拟合（Fig.8 在 C4 上误差最低）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：算法侧在 PyTorch 中实现（开源仓库 https://github.com/yc2367/P3-LLM 的 `--apply_k_scale` 与 `--k_quant_post_rope` flag 组合），每个 head 内按通道求 max 并保存 scale 张量，decode 阶段对新 key 做除法后 INT4-Asym 量化（per-head，128 元素共享 16-bit scale + 4-bit zero-point，有效精度 4.16 bit）；硬件侧把平滑因子融合进 query 的 FP8 量化缩放（SSF fusion），使 Q·K^T 在 PIM 上直接消费量化 key 而无在线解量化。适用场景：任何 KV-cache 4-bit 量化的 LLM 推理（尤其 key cache 含结构化 outlier 通道、且希望避免校准数据集依赖的部署）；pre/post-RoPE 的选择需按模型最大序列长度 profiling 决定（短序列 pre-RoPE、长序列 post-RoPE）。


涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
