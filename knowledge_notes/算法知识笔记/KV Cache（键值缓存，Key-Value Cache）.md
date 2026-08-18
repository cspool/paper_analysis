## KV Cache（键值缓存，Key-Value Cache）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KV Cache 是自回归 LLM 推理（prefill/prompt 与 decode/token 生成两阶段）用于避免重复计算历史 token 的 Key/Value 激活缓存：prefill 阶段并行处理整个输入序列并计算各层 K/V 存下；decode 阶段每步只算新 token 的 Q，与缓存的历史 K/V 做注意力。代价是内存占用随序列长度线性增长（长上下文下成为主要片上/片外开销），且 K/V 访问量随序列增长而增大访存流量。SMOOTH（ISCA'26）在移动 NPU 场景把它作为运行期动态因素与内存压力源：① KV cache 大小随用户序列长度变化，编译期无法预知，静态 tile size 因此失效（延迟最多恶化 2.9×）；② decode 期 KV cache 数据不断涌入片上，其块（如 V_cache 单 block）在 S×V 计算进行中逐块消费——SMOOTH 的 block 级分配允许"V_cache 单块一空出就用于预取"，且长序列（32K）下 KV cache 内存开销显著，SMOOTH-ER 较 Gemmini 平均收益从 2K 的 50.1% 增至 32K 的 66.8%。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- decode 单步执行拆解（含 KV cache）：输入 token x_t → 与缓存 K_{1:t-1}、V_{1:t-1} 拼接做注意力：
```
Q_t = x_t @ W_Q            # 只算新 token
K_t = x_t @ W_K; V_t = x_t @ W_V
K_1:t = concat(K_1:t-1, K_t)   # KV cache 追加（随序列增长）
V_1:t = concat(V_1:t-1, V_t)
score_t = Q_t @ K_1:t^T / sqrt(d)   # 矩阵从 l×l 缩为 l×1
ctx_t = softmax(score_t) @ V_1:t
```
注意矩阵从 l×l（prefill）缩为 l×1（decode），配合权重矩阵的 d×d GEMV 形成低 OI 的 I/O-bound 执行。SMOOTH 的关注点：K/V 数据的片上放置与预取——V_cache 块在 attention 计算中被逐块消费后可 early reclaim，块级预取把 KV 流量摊平。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：KV cache 在移动 SoC 上由编译器分配进 SPM 或随权重混排；SMOOTH 的 block 表/bitmap 管理其片上驻留，use_cnt/end_cmd 驱动其消费后回收，N_preload 预取后续 KV 块。评估模型（TinyLLaMA 1.1B–GPT-3 13B，w4a8/int8）在 batch=1 下，KV cache 随输入序列（1K–32K）增长成为 generation 期延迟的主要来源；SRAM 占用/每 token 延迟实验中，无融合时 KV cache 增长使带宽饱和、各策略收益受限，融合后预取机会显现。论文未对 KV cache 做压缩/剪枝类算法改动（相关技术如 H2O、KV 量化属正交方向）。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
