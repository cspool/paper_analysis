## PivotKV

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PivotKV 是 RETAKE 提出的 training-free KV cache 压缩方法，在 VideoLLM 的 chunked prefilling 过程中，对每个视频 chunk 的 KV cache 进行 token 级剪枝。核心创新在于 pivot-guided 压缩策略：(1) DPSelect 选出的 pivot frames 的 visual tokens 被强制保留（通过在 token 重要性分数上加无穷大），保证关键低层时空细节不丢失；(2) 非 pivot frames 中，基于 LLM 层内的 self-attention 权重分布计算 token 重要性分数，低注意力 token 被剪枝——注意力分布由 LLM 的多模态高层知识隐含地确定 token 冗余性（knowledge redundancy）。因此 PivotKV 同时保留了 pivot frames 的全部信息（低层时序不变性）并通过 LLM 注意力去除非 pivot frames 的冗余 token（高层语义冗余）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PivotKV 在每个 chunk prefilling 后执行（每个 chunk 包含 tau 帧），逐层独立操作：
```
# 输入: chunk_i 的 KV cache K_i, V_i
#       pivot mask s (当前 chunk 的 pivot token 标记)
#       alpha_kv: PivotKV 压缩比

for each attention layer:
    # Step 1: 计算 chunk 内的 self-attention 权重
    A = Softmax(Q_i K_i^T / sqrt(d_h))

    # Step 2: 计算 token 重要性分数
    # 对所有 query 位置求和，对所有 head 取均值
    a_bar[j] = sum_{all queries} mean_{all heads} A[:, j]

    # Step 3: 强制保留 pivot tokens
    a_bar = a_bar + s * inf

    # Step 4: Top-k 选择
    I = ArgTopK(a_bar, k=alpha_kv * l_q)
    K_hat_i = K_i[:, I, :]
    V_hat_i = V_i[:, I, :]

    # Step 5: 更新历史 KV cache
    K = Concat(K, K_hat_i)
    V = Concat(V, V_hat_i)
```
文本 chunk（prompt tokens）不参与压缩。效率优化：使用额外 CUDA stream 将第 l 层的 PivotKV 压缩与第 l+1 层的 prefilling 重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PivotKV 基于 PyTorch 实现，以即插即用方式集成到现有 VideoLLM（QWen2VL-7B, LLaVA-Video-7B），无需额外训练。压缩比 alpha_kv 按视频自适应设置以确保总 context length <= 32K。在 A100 GPU 上，alpha_kv=0.5 时：QWen2VL 的 FLOPs 降低 9%、TPOT 降低 19%（优化后 TTFT 仅增加 8%）；LLaVA-Video 的 FLOPs 降低 18%、TPOT 降低 26%（优化后 TTFT 仅增加 11%）。消融实验证明 PivotKV 与 DPSelect 互补——仅用 DPSelect（w/o PivotKV）在低压缩比下性能显著下降，而 PivotKV 通过利用知识冗余缓解了这一问题。代码开源在 https://github.com/SCZwangxiao/video-ReTaKe。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding
