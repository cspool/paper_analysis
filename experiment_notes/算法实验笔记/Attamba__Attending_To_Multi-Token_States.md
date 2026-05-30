## Attamba__Attending_To_Multi-Token_States

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Attamba——一种将State-Space Models (SSMs) 直接集成到Transformer自注意力机制内部的混合架构。核心设计：用SSM block替换标准的Key (K) 和Value (V) 投影矩阵，SSM将连续的P个token压缩为单个表示，attention仅在这些压缩后的chunk边界表示上进行计算。Query (Q) 投影保持标准方式以保留自回归性质。具体技术包括：(1) 训练时保留所有SSM输出用于causal next-word prediction，推理时仅缓存chunk边界输出（P× KV-Cache缩减）；(2) cyclic chunk boundary——不同层使用不同的chunk边界偏移，减少固定边界引入的偏差；(3) 支持leading tokens (L) 保持对最近token的完整attention（类似sliding-window attention）；(4) pseudo-chunking模式（不裁剪attention map，仅替换K/V投影为SSM）可略微提升Transformer困惑度。实验对比：(a) standard Transformer（iso-parameter/iso-KV/iso-FLOPs变体，通过调整attention model dimension F来匹配Attamba的KV-Cache或FLOPs）；(b) SSM类模型Mamba、minGRU、Hawk；(c) 消融实验：KV投影矩阵有无、SSM state dimension大小、chunking方法（Uniform/Random/Cyclic/FAttn/FSSM）、chunk size (4/8/64/128)、leading tokens数量。

- 硬件平台是什么，配置是什么。
  单张NVIDIA RTX A6000 GPU。训练模型约60M参数（8层、8 heads、512 model dimension），batch size=16，sequence length=1024，约982M tokens训练（100k步≈1B tokens）和100k步8B tokens的扩展实验。框架使用Meta Lingua（Facebook开源的PyTorch LLM训练库）。

- 模型是什么。数据集和bench分别是什么。
  模型：60M参数的Attamba（8层、8 heads、512 model dimension）。默认配置：chunk size P=4/8，leading tokens L=0/P，SSM state dimension D_s=16（总SSM参数开销约4M）。对比Transformer baseline同为8层8 heads但attention model dimension F根据iso-KV/iso-FLOPs条件调整（如表1，P=4时isoF=128/isoFLOPs=160，P=8时isoF=64/isoFLOPs=104）。对比模型：Mamba、minGRU、Hawk均在60-64M参数预算内。数据集：10%subset of dclm-baseline-1.0（Li et al., 2024），最大训练8B tokens。Benchmark：WikiText2 test-set perplexity。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/abdelfattah-lab/attamba（BSD-3-Clause license，基于Meta Lingua框架，Python 99.8%）。训练日志：https://wandb.ai/akhauriyash/attamba_arxiv。论文明确标注"experiments should be reproducible from the current state of this repo."

  Attamba算法pipeline核心——单个attention head的前向传播（训练时）：
  ```
  Input: X ∈ R^{n×e}  (n: sequence length, e: model dimension, P: chunk size)
         Mask ∈ R^{n×n} (causal + chunk boundary mask)

  // 1. Query projection保持不变（标准线性投影）
  Q = X W_Q    // W_Q ∈ R^{e×e}, Q ∈ R^{n×e}

  // 2. Key和Value通过SSM block而非线性投影
  //    X被分为n/P个chunk: X^{(1)}, X^{(2)}, ..., X^{(n/P)}  each ∈ R^{P×e}
  For each chunk p = 1, 2, ..., n/P:
    // SSM_K: 对chunk内P个token做autoregressive SSM扫描
    //          使用Mamba-style selective SSM:
    //          h_t = A_t * h_{t-1} + B_t * x_t  (recurrent state更新)
    //          k_t = C_t * h_t
    K^{(p)} = SSM_K(X^{(p)})  // ∈ R^{P×e}, 每行是causally valid的压缩表示

    // SSM_V: 同理处理value
    V^{(p)} = SSM_V(X^{(p)})  // ∈ R^{P×e}

  // 3. 拼接所有chunk的SSM输出
  K_SSM = concat[K^{(1)}; K^{(2)}; ...; K^{(n/P)}]  // ∈ R^{n×e}
  V_SSM = concat[V^{(1)}; V^{(2)}; ...; V^{(n/P)}]  // ∈ R^{n×e}

  // 4. 构造chunk attention mask (Equation 5)
  //    M_train[i,j] = 0 iff:
  //      - j和i在同一chunk内且j≤i (chunk内causal self-attention), OR
  //      - j≤i且j是某个chunk的最后一个token (跨chunk仅attend chunk边界)
  //    M_train[i,j] = -∞ otherwise

  // 5. Attention（仅attend chunk边界+当前chunk内）
  S = Q K_SSM^T / √d           // S ∈ R^{n×n}, d = e/num_heads
  A = Softmax(S + M_train)     // 仅chunk边界和当前chunk内非-masked
  Y = A · V_SSM                // Y ∈ R^{n×e}

  Output: Y (经过output projection和residual后送入MLP)
  ```

  推理时（test-time）简化版：
  ```
  // 推理仅需保存每个chunk的最后一条SSM输出
  For each chunk p:
    K_SSM_test.append(K^{(p)}[-1])  // 仅最后一个token的K, ∈ R^{1×e}
    V_SSM_test.append(V^{(p)}[-1])  // 仅最后一个token的V

  // Attention mask (Equation 7): 每个query仅attend到已完成的chunk边界
  M_test[i,j] = 0 if j ≤ floor(i/P), else -∞

  // KV-Cache大小: n/P × e × 2  (vs Transformer: n × e × 2)
  // Attention FLOPs: O(n²/P)  (vs Transformer: O(n²))
  // 当L>1时，额外保存每个chunk最后L个token的KV用于sliding window attention
  ```

  具体配置参数含义：
  - P (Chunk size): 每个SSM压缩的token数，P=4→4× KV-Cache缩减，P=8→8×缩减
  - L (Leading tokens): 保持完整attention的最近token数，L=P时保留整个最新chunk
  - D_s (SSM state dimension): 默认16，>32后对P=8收益递减（<1% perplexity差异）
  - Cyclic chunking: 第layer层chunk边界偏移layer个token位置，使不同层处理不同token分组
  - Pseudo-chunking: L=seq_len时的退化情况，SSM替代K/V投影但保持全注意力mask

  复杂度分析（vs 标准Transformer）：
  - KV-Cache: (2n/P + 2L)E vs 2nE（L为leading tokens数）
  - Attention FLOPs: O(n²/P) vs O(n²)
  - SSM FLOPs overhead: O(n × D_s × E) per SSM block (linear, 可忽略不计)
  - SSM参数开销: ~4M（60M模型的6.7%）

  Cyclic chunking的实现：
  ```
  # 第layer_num层的chunk边界从 layer_num % P 位置开始
  For layer in range(num_layers):
    offset = layer % P
    # chunks: [{offset, offset+1, ..., offset+P-1},
    #          {offset+P, offset+P+1, ..., offset+2P-1}, ...]
    # 不同层从不同偏移量开始划分chunk，打破固定边界偏差
  ```

  论文主要结果（WikiText2）：
  - Attamba (P=4) vs iso-KV+SWA Transformer: 困惑度显著改善（~24%）
  - Attamba (P=8) vs iso-KV Transformer: 5% perplexity trade-off for 8× KV-Cache压缩
  - 8B tokens训练：Attamba (P=4, L=4) 困惑度 ~18.5 vs Mamba ~20.2 vs Transformer ~20.8
  - Cyclic chunking比Uniform chunking提升约5%
  - 随机chunk边界工作与均匀分块相当（说明SSM对chunk边界鲁棒）
  - Pseudo-chunking（替换K/V投影但不裁剪attention）比标准Transformer困惑度略优
