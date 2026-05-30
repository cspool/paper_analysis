## KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing

- 属于算法pipeline的实现是什么？实验比较什么？
  KVSharer 提出一种 plug-and-play 的层间 KV cache 共享方法，无需额外训练。核心实现基于一个反直觉的发现：共享不相似的 KV cache（而非相似）可更好地保持模型性能。方法分为两个阶段：(1) Strategy Searching——在校准数据集上执行推理，计算任意两层 KV cache（分别 flatten keys 和 values 为 1D 向量后平均）之间的欧氏距离，按降序排列（距离大=相似度低），依次尝试替换 KV cache 对（将靠近输出端的层用靠近输入端的层替换），若替换后最后一层 hidden state 与原始模型的余弦相似度超过阈值 T=0.5 则保留该替换，直到达到预定的压缩层数 C；(2) Inference with KV Cache Sharing——获得共享策略 Z 后，在 prefill 和 generation 过程中直接将被替换层的 KV cache 从前层拷贝过来，跳过本层的 KV cache 计算。实验比较在 Llama2-7B/13B/70B（Chat 和 Base 版本）、InternLM2-7B/20B、Mistral-7B-Instruct-v0.3 上，不同压缩率（12.5%/25%/37.5%）下与 Full KV Cache 的性能对比（perplexity + OpenCompass 多 benchmark 评分），以及与 H2O、PyramidInfer 等 intra-layer 压缩方法的组合效果，以及与相似度共享（+Sim.）和随机共享的消融对比。

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100 80GB GPU 服务器。PPL 评估使用 Wikipedia 数据集 200 句，每句 2048 tokens。策略搜索使用 30 句随机 Wikipedia 句子（每句 64 tokens）作为校准数据集。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama2-7B-Chat、Llama2-13B-Chat、Llama2-70B（部分实验）、InternLM2-7B-Chat、InternLM2-20B-Chat、Mistral-7B-Instruct-v0.3，以及各模型的 Base 版本。数据集/Calibration：Wikipedia（30 句随机采样的 64-token 句子）、BookCorpus（同等大小子集）。Benchmark（通过 OpenCompass 评估框架）：Reasoning（CMNLI, HellaSwag, PIQA）、Language（CHID, WSC）、Knowledge（CommonSenseQA, BoolQ）、Examination（MMLU, CMMLU）、Understanding（Race-High/Middle, XSum, C3）。PPL 在 Wikipedia 200 句、每句 2048 tokens 上评估。评估模式包括 PPL 和 GEN 两类。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/yangyifei729/KVSharer。算法 pipeline 如下：

  **阶段一：Strategy Searching（Algorithm 1）**

  输入：LLM M、目标共享层数 C、校准数据集 D、相似度阈值 T
  输出：共享策略 Z（包含哪些层用哪些层的 KV cache 替换）

  ```
  1. S ← Euclidean_KV_Distance(M, D)
     // 在 D 上执行前向传播，保存每层 KV cache
     // 将每层的 keys 和 values 分别 flatten 为 1D 向量，取平均作为该层 KV cache 表示
     // 计算任意两层之间的欧氏距离
  2. R ← Descend_Rank(S)
     // 按欧氏距离降序排列（距离大 = 不相似度高，优先尝试）
  3. Z ← ∅, P ← 0
  4. for each r in R:  // r = (layer_src, layer_dst)
       Z ← Z ∪ {r}
       // 替换时：layer_dst（靠近输出端）的 KV cache ← layer_src（靠近输入端）的 KV cache
       // 不可逆方向：靠近输入的层更敏感，不做替换
       M_tmp ← Sharing_KV(M, Z)
       // 在 D 上计算 M_tmp 和 M 的最后一层 hidden state 的余弦相似度 s
       s ← Avg_CosSim(M_tmp, M, D)
       if s ≤ T:
           Z ← Z \ {r}   // 丢弃该替换对
       else:
           P ← P + 1
           if P == C: return Z
  5. return Z
  ```

  **阶段二：Inference with KV Cache Sharing**

  在 prefill 和 generation 阶段，根据 Z 中记录的替换关系：
  ```
  for each layer l in model.layers:
      if l in Z.dst_layers:
          src_layer = Z.get_src(l)
          K_cache[l], V_cache[l] = K_cache[src_layer], V_cache[src_layer]  // 直接拷贝
      else:
          K_cache[l], V_cache[l] = compute_KV(l, input_hidden)
  // 后续 attention 和 FFN 正常进行
  ```

  **关键设计**：替换只发生在靠近输出的层用靠近输入的层替换，因为靠近输入的层更敏感（修改会导致更大性能退化）。一次搜索可通用于所有下游任务（非 task-specific）。

  **组合 intra-layer 压缩**：将 H2O 或 PyramidInfer 应用于各层 KV cache 的稀疏化，然后 KVSharer 共享的层直接拷贝已稀疏化的 KV cache。Hyperparameters 先在全注意力模型上调至约 20% 压缩率，然后直接应用于与 KVSharer 的组合。
