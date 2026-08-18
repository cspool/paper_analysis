## Token Pruning（动态 token 剪枝，LazyLLM 式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Token pruning（token 剪枝/动态 token 丢弃）是长上下文 LLM 推理的加速优化：在每个 transformer 层，根据注意力输出动态选出相关性最高的 token 子集（例如按注意力分数取 top-k），只对这些 token 做后续层（自注意力、FFN/MoE）计算，而跳过被剪掉的 token，从而把随序列长度线性/平方增长的计算量降下来。代表工作 LazyLLM（arXiv:2407.14057）在 prefill 阶段逐层剪枝、被剪 token 的 KV 不写入 cache。它与 KV cache 量化的区别：剪枝减少"参与计算的 token 数"（计算量），量化减少"每个 KV 的位数"（容量/带宽）。它是近似/有损优化（丢弃 token 可能影响精度，属"精度保持的算法改动"谱系，通常按比例控制）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 算法流程（以 LazyLLM/IroKnight 描述为例）：每个 transformer 层先正常计算 attention（MatMul Q·K^T → 缩放 → softmax → MatMul softmax·V）；然后对注意力分数做 Top-K 选择——顺序扫描分数数组、与运行阈值比较、保留分数最高的 k 个 token 索引（k 由剪枝比例决定，如保留 20%-100%）；被保留 token 才继续进入下一层（其 KV 写入 cache），被剪 token 的后续计算跳过。这个 Top-K 扫描是规则、顺序的仿射访问（步长 1 遍历数组），与 MatMul/softmax 的 tiled/vectored 执行同属细粒度规则访问。IroKnight 的视角：正因为剪枝只新增 Top-K 这种规则扫描算子、不改变算子级细粒度仿射访问，全状态加密（见 Pad/PadGen）照常成立。
  - 伪代码：
```
for layer l in model:
    scores = softmax(Q_l @ K_l^T / sqrt(d_k))   # 注意力分数
    keep = topk_by_scan(scores, ratio)          # 顺序扫描+阈值比较，规则仿射访问
    x = layer_l(x[keep])                        # 只计算保留 token
    # KV cache：只写保留 token 的 K/V
```
  - Annotations：topk_by_scan 是剪枝引入的唯一新算子；扫描是步长 1 的顺序数组访问，pad 可预计算；剪枝比例 0% 即不剪（全部保留）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：软件 serving 层（vLLM/SGLang 的 token 级调度、LazyLLM 的 per-layer 剪枝器）；评估指标为剪枝比例（0-80%）下的延迟/能量与精度权衡。IroKnight 的评估（LLM，1024 in/out token、batch 1、剪枝 0-80%）：Llama4-Scout 加密变体延迟开销 0.3%-0.4%、GPT-OSS-120B 0.5%-0.7%，加认证 3.3%-3.4% / 3.6%-3.7%——因剪枝不破坏细粒度规则执行；能量随剪枝比例升高而下降（Llama4-Scout 加密 13.6%→9.7%、GPT-OSS-120B 9.6%→6.4%；认证 17.1%→14.2% / 13.3%→11.1%），因为计算量下降而权重 HBM 流量不变、加密成本被摊销。作用：长上下文 serving 里减少每层计算量、降低延迟与能耗，且与加密/完整性保护正交兼容。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
