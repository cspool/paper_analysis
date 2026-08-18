## MLA（Multi-Head Latent Attention，多头潜在注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLA 是 DeepSeek 系列（DeepSeek-V2/V3/R1）提出的注意力机制：把每个 token 每层的 K/V 投影压缩进一个低秩 latent 向量（down-projection 得到 c_KV ∈ R^{d_c}，d_c ≪ n_heads×d_head），推理时缓存低维 latent 而非完整 K/V 张量，配合解耦 RoPE（decoupled rotary position encoding，K_R 单独加旋转位置）与 weight absorption 技巧在计算时把 up-projection 吸收进 Q/O 权重。效果：KV cache 尺寸从 2×n_heads×d_head 降到 ≈2×d_c+d_R，与注意力头数解耦——本论文给出的核心论点是"MLA 将 KV cache 大小与 head 数量解耦"，使 R1 能在极长 reasoning 上下文下维持远低于同规模密集模型的 KV footprint（对比 GQA 的 8 KV head 仍随层数线性增长）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MLA 前向（每层，token t）
c_KV = W_DKV @ h_t              # 压缩: h ∈ R^d → latent c_KV ∈ R^{d_c}（缓存 c_KV，而非 K/V 全张量）
k_R  = W_KR @ h_t               # 解耦 RoPE key（d_R 维，带位置编码，单独缓存）
q    = W_Q @ h_t; q_R = W_QR @ h_t
# attention 时用权重吸收后的 W_UQ 展开 latent（KV 不再显式存储/读取全头张量）
score = attention(q, q_R, c_KV, k_R)   # c_KV 经吸收后的上投影参与 QK^T 与 SV
```
Annotations：d_c=低秩 latent 维（DeepSeek 通常 ≈512，远小于 GQA 的 8×128），d_R=解耦 RoPE 维；缓存量 ≈(d_c+d_R) 而非 n_heads×d_head×2；MLA 通过把 KV 压缩进 latent 使 KV 足迹与 head 数无关，是 DeepSeek-R1-671B 长 CoT 推理容量可行的算法根因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DeepSeek 开源实现（DeepSeek-V2/V3 技术报告、FlashMLA kernel 库）；vLLM/SGLang 等 serving 引擎原生支持 MLA 模型。本论文的用法与发现：(1) MLA 是 DeepSeek-R1-671B 维持长上下文推理的架构前提——论文实测 R1 参数为 70B 模型的 10×，但 KV cache 消耗速率反而"适度"（对比密集 70B 的激进容量消耗与提前请求限流），直接支撑"架构级 KV 压缩与硬件容量同等重要"的结论；(2) MLA 与 Pipeline Parallelism 协同：压缩后的 KV 使每 PP stage 可容纳更高 micro-batch 深度、填满 pipeline bubble，这是 R1 在 PP=4+TP=2 下优于纯 TP=8（1663s vs 2047s）的机制之一；(3) 对比 GQA（head 共享、仍线性于层数）与 MHA（无压缩），MLA 是把 KV 与 head 数解耦的算法级容量解药。

涉及论文标题：
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
