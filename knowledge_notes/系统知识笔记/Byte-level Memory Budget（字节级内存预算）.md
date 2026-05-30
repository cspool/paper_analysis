## Byte-level Memory Budget（字节级内存预算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Byte-level Memory Budget 是 StreamingEval 提出的统一资源约束方法，将流式视频评估中不同模型之间的比较从 token 数量约束转换为字节级统一预算。传统做法按 "最多 N 个 visual tokens" 限制上下文窗口，但不同模型 visual token embedding 维度不同（如 3584 vs 5120），导致相同 token 数实际 GPU 内存占用差异可达 2-3×。Byte-level budget 消除了维度不匹配造成的不公平比较。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
StreamingEval 的计算公式：

$$\operatorname{Mem}_i(B) = B \cdot d_i \cdot s_{\text{emb}} + B \cdot 2L_i \cdot h_i^{\text{kv}} \cdot s_{\text{kv}}$$

$$B_i = \left\lfloor \frac{M_{\text{bytes}}}{d_i s_{\text{emb}} + 2L_i h_i^{\text{kv}} s_{\text{kv}}} \right\rfloor$$

其中 $M_{\text{bytes}} = M \cdot 10^9$（如 M=0.5 → 0.5GB），$d_i$ 为投影后 visual token embedding 维度，$s_{\text{emb}}$ 为 embedding 存储精度（BF16=2），$L_i$ 为 LLM decoder 层数，$h_i^{\text{kv}}$ 为 per-layer KV channel width（$n_i^{\text{kv}} \cdot d_i^{\text{head}}$），$s_{\text{kv}}$ 为 KV cache 精度。因子 2 覆盖 Key 和 Value 两个张量。两项分别覆盖 (1) 内存银行中的 visual token embedding 和 (2) 这些 visual token 关联的 Transformer KV cache——两者均随上下文长度线性增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 StreamingEval 中，默认 budget M=0.5GB，支持 sensitivity 测试（0.1G, 0.3G, 0.5G, 1.0G, 1.5G）。实验发现 accuracy 在 1.0G 以上近乎饱和（diminishing returns），而 0.1G 极端预算下模型差距缩小。此方法适用于：(1) 公平基准评估——不同 embedding 维度的模型在统一字节约束下比较；(2) 实际部署资源规划——给定 GPU VRAM 预算，扣除模型权重和运行时开销后可计算最大可行 visual token 上下文窗口。

涉及论文标题：
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding
