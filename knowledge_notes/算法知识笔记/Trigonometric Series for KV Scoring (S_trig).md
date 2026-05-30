## Trigonometric Series for KV Scoring (S_trig)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

S_trig 是 TriAttention 提出的 KV cache 重要性评分组件，利用 pre-RoPE Q/K 聚集现象，通过三角函数级数预测 key 在未来 query 位置会收到多少 attention。核心公式（Eq 6）：

$$S_{\text{trig}}(k, \Delta) = \sum_f \|\mathbb{E}[q_f]\| \cdot \|k_f\| \cdot \cos(\omega_f \Delta + \phi_f)$$

其中 E[q_f] 是 Q 中心（校准数据均值），k_f 是 cache 中 key 在频段 f 的 pre-RoPE 复数表示，ω_f = θ^{-2f/d} 是 RoPE 频率，Δ = p_q - p_k 是 Q-K 距离，φ_f = arg(E[q_f]) - arg(k_f) 是相位差。

物理意义：当 Q 高度聚集时，用 Q 中心替代未来任意位置的 query，三角函数级数给出该 key 在距离 Δ 处收到的平均 attention。与 post-RoPE 方法的根本区别：S_trig 是 model-intrinsic 预测——仅依赖 Q 中心和 key 自身的 pre-RoPE 表示——不依赖观测任何实际 attention scores，因此不受 RoPE 旋转限制的"小观察窗口"问题影响。

术语一般如何实现？如何使用？

实现：(1) 离线校准阶段——收集校准数据的 pre-RoPE Q 向量，计算 Q 中心 E[q_f]。(2) 推理时每 128 tokens 触发一次 scoring，遍历 cache 中所有 key，对每个 key 和每个 future offset δ∈{1,2,4,...,2^16} 计算 S_trig(k, Δ+δ)，取平均。计算量 O(|cache| × 17 × d/2)，但因仅每 128 步执行一次，实际 overhead 极低。

S_trig 能捕获距离偏好——某些 head 偏好近距离 key（S_trig 在 Δ 小时 peak），某些 head 偏好远距离 key（attention sink, S_trig 在 Δ 大时 peak）。跨域校准验证：校准数据用 coding data 时，AIME24 准确率 44.2%（vs reasoning 校准 42.1%）——证明 Q 中心是模型内在属性。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---
