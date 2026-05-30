## State-Observation-Dynamics Triplet（状态-观测-动态三元组）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
State-Observation-Dynamics Triplet 是 Owl-1 的核心数学形式化，用三个变量 (s_t, o_t, d_t) 的闭环交互来模拟世界的自回归演化。State s_t：隐式世界状态，由 128 个 learnable query embeddings 实现，无 ground truth，通过 causal self-attention 聚合所有历史观测信息。Observation o_t：从状态解码的 4s 视频片段，由 VideoDM 作为 state decoder 生成，以 s_t 为 cross-attention condition、o_{t-1} 为短期平滑参考。Dynamics d_t：文本形式的世界动态预测，从 (s_t, o_t) 推断即将发生的事件。三者构成完整闭环：s_t → o_t (解码) → d_t (预测) → s_{t+1} (更新)。链式展开：s_{t+1} = g(s_t, f(s_t, D(s_t, o_{t-1}))) = h(s_0, o_0, ..., o_t)，即当前状态包含所有历史信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LMM 序列输入格式：Seq = [..., s_t_queries (128 tokens), o_t_VQ (2 frames sampled), d_t_text (tokenized), ...]。每个时间步的计算：
```
# LMM forward
s_t_queries, d_t_logits = LMM.forward([...history..., 
    s_{t-1}_queries, o_{t-1}_VQ, d_{t-1}_text])
d_t_tokens = argmax(d_t_logits)                            # 预测动态

# VideoDM denoising
z_T ~ N(0, I)
for m in T..1:
    z_{m-1} = denoise(z_m, m,
                      cross_attn_cond=s_t_queries,         # 长期一致性
                      concat_cond=o_{t-1}.last_frame)      # 短期平滑
o_t = z_0
```
s_t 训练两阶段：Stage 1 用 MSE 对齐 L_align = MSE(s_t, T(t))；Stage 2 用 L_pretrain = ||ε - ε_θ(o_{t,m}, m, s_t, o_{t-1})||²。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
o_t 的 visual tokens 由预训练 VQ-VAE 编码关键帧获得（每 4s clip 采样 2 帧）。d_t 由 LMM text tokenizer 编码，训练时用 dense captions teacher-forcing（交叉熵），推理时 LMM 自主生成。跨场景切换时丢弃 image condition，仅依赖 s_t 信息。s_t 长度 128 是固定超参数，平衡表达能力与计算开销。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation
