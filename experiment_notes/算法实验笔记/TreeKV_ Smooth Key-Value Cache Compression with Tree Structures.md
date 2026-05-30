## TreeKV: Smooth Key-Value Cache Compression with Tree Structures

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 TreeKV —— 一种 training-free 的 KV cache 压缩方法，使用树形结构（tree structure）实现平滑的 cache 压缩。核心实现：(1) 通过 wavelet 分析发现 token 贡献从远到近呈现平滑递增，越靠近序列末尾信息越丰富且与相邻 token 差异越大，因此设计"左侧稀疏、右侧密集"的树形结构；(2) decoding 阶段：当 KV cache 达到容量上限 c 时，使用循环 eviction scope {idx, idx+1} 决定待淘汰的相邻 token 对，比较两者的平均 attention 权重（importance score），淘汰较低分者，然后 idx 循环递增 (idx mod c + 1)，使得淘汰范围从远到近循环移动，形成从 coarse-grain 到 fine-grain 的平滑过渡树结构；(3) prefilling 阶段：将 prompt 切分为 blocks（block size = b），以最后一个 block 作为 observation window 查询整个序列得到各 block 的 attention-based importance score，然后在 block 级别执行与 decoding 相同的树形淘汰策略，同时计算（而非逐 token 顺序处理）以提升效率；(4) position encoding re-assignment：淘汰后按缓存中剩余 token 的相对顺序重新分配位置编码。

  实验比较：
  (1) Language modeling on PG19 & OpenWebText2（Table 1-2）：与 StreamingLLM、H2O、TOVA、Full Attention 对比，在 4k/8k/16k 上下文下使用 cache size=1024（含 4 sink + 508 recent + 512 selected），TreeKV 在 16k PG19 上 PPL 6.91（vs TOVA 7.15，提升 3.6%），在 16k OpenWebText2 上 PPL 5.18（vs TOVA 5.24，提升 1.1%）；
  (2) 超长序列（10M tokens，Figure 4）：在 PG19 拼接的 10M token 序列上，TreeKV 的 NLL 最低且随长度增长稳定，而 TOVA 和 H2O 性能退化；
  (3) Longbench 长上下文理解（Table 3）：使用 Llama-3.2-1B-Instruct，与 H2O、SnapKV、FullKV 对比，cache size 2k/8k 下 TreeKV 平均得分 31.70/32.80，优于 H2O (30.23/32.79) 和 SnapKV (31.50/32.04)；
  (4) Ablation study（Figure 5）：对比 H2O、TreeKV 和 TreeKV_Select_Left_Token（仅用树结构不用 attention 权重），验证树结构本身对性能的贡献远超 attention-weight-based selection。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 4090 GPU（24GB），使用 bf16 精度。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B（pretrained with 4K context length，用于 language modeling），Llama-3.2-1B-Instruct（用于 Longbench 长上下文理解）。
  数据集：PG19 test set（100 本完整书籍，平均 113k tokens/本）、OpenWebText2（从 Pile 随机抽取 100 个 test set 样本，平均 18k tokens/样本）。
  Benchmark：Longbench（16 任务 6 类别：Single-Document QA、Multi-Document QA、Summarization、Few-shot Learning、Synthetic Tasks、Code Completion，平均长度 ~11k tokens）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明开源：https://github.com/ZiweiHe/TreeKV（截至检索时仓库为空，代码尚未发布）。

  算法 Pipeline 伪代码（对应论文 Algorithm 1）：

  ```
  # TreeKV Decoding Stage
  # 输入: x^{(1..T)} ∈ R^{1×d}, cache size c
  # 初始化
  S = zeros(c)          # 累积 attention scores
  C = zeros(c)          # 计数
  K_cache = []           # KV cache 存储
  V_cache = []
  idx = 1                # eviction scope 起始索引

  for t in 1..T:
      # 1. 标准 QKV projection（不变）
      q = x[t] @ W_Q    # [1, d]
      k = x[t] @ W_K    # [1, d]
      v = x[t] @ W_V    # [1, d]

      # 2. 追加新 KV 到 cache
      K_cache.append(k)
      V_cache.append(v)

      # 3. 标准 attention 计算
      a = softmax(q @ K_cache^T / sqrt(d))  # [1, len(cache)]

      # 4. 更新 importance scores
      C = (C ∪ {0}) + 1           # 每个 token 的参与计数 +1
      S = (S ∪ {0}) + a           # 累积 attention weights

      # 5. 若超出容量，执行树形淘汰
      if len(K_cache) > c:
          S_avg = S / C            # 平均 attention weight

          # 在 eviction scope {idx, idx+1} 中淘汰较低分者
          if S_avg[idx] > S_avg[idx+1]:
              淘汰 K_cache[idx+1], V_cache[idx+1], C[idx+1], S[idx+1]
          else:
              淘汰 K_cache[idx], V_cache[idx], C[idx], S[idx]

          # idx 循环右移，形成树结构
          idx = (idx + 1) mod c + 1

      # 6. Position encoding re-assignment
      # 按 cache 中剩余 token 的相对顺序重新分配位置编码
  ```

  **Wavelet 分析驱动设计**：
  - 将 attention 输出 s = a^T ∘ V 沿序列长度维度做 multi-level Haar wavelet 分解
  - 观察：越靠近序列末尾，所有频率分量的幅度逐渐增大，尤其高频分量增长显著
  - 含义：token 贡献递增，且与邻居 token 的差异性递增 → 从远到近呈现平滑过渡
  - 因此设计"左侧稀疏(远)、右侧密集(近)"的树形淘汰结构

  **Prefilling Stage 差异**：
  - Prompt 切分为 block_size = b 的多个 block
  - 以最后一个 block 作为 observation window，query 整个序列得到 attention weights
  - 对每个 block 内所有 token 的 attention weights 取均值作为该 block 的 importance score
  - 在 block 级别执行与 decoding 相同的树形淘汰（所有 block 并行处理）

  **Cache 组成**：1k cache = 4 sink tokens + 508 recent tokens + 512 TreeKV-selected tokens
