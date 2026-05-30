## Mixture of A Million Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出了 PEER（Parameter Efficient Expert Retrieval）层，使用 product key retrieval 技术实现从超过一百万（1024²）个 tiny expert（单神经元 MLP）中稀疏检索 top-k 专家。核心创新：(1) product key 将 N 个 d 维 key 分解为两组各 √N 个 d/2 维 sub-key 的笛卡尔积，将 top-k 检索复杂度从 O(Nd) 降至 O((√N + k²)d)；(2) 每个 expert 是 singleton MLP（仅一个隐藏神经元）：e_i(x) = σ(u_i^T x) v_i，权重存储在 Embedding 层中通过索引检索；(3) multi-head retrieval：h 个独立 query network 各自检索 k 个 expert，共享同一 expert pool，输出直接求和，等效于动态组装一个 h 神经元 MLP。
  - 实验通过 isoFLOP 分析（固定 FLOP 预算 6e18 和 2e19）比较 PEER vs Dense FFW vs Coarse-grained MoE（expert-choice routing, 128 experts）vs PKM（1024² memories, h=8, k=32）。在 C4 验证集上绘制 isoFLOP 曲线（模型大小 vs perplexity），并评估 compute-optimal 模型在 Curation Corpus、Lambada、Pile、Wikitext、C4 上的 perplexity。
  - Ablation 研究：(1) 变化总 expert 数量 N（128², 256², 512², 1024²）保持 hk=128 不变；(2) 变化 active expert 数量 hk（32, 64, 128, 256, 512）保持 N=1024² 不变，联合变化 h 和 k；(3) Query BatchNorm 对 expert usage 和 unevenness 的影响。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号或硬件配置。作者单位为 Google DeepMind，致谢中提到使用内部代码库训练模型，推测使用了 Google 内部 TPU/GPU 集群。
  - 训练配置：batch size=128, sequence length=2048。
  - 精度：论文未明确说明训练精度（BF16/FP32），推测为标准混合精度训练。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Transformer backbone，将中间 block 的 FFW 层替换为 PEER/MoE/PKM 层。模型大小通过变化层数、attention heads 和模型维度来控制（具体范围论文未列详细表格）。
  - PEER 配置：N=1024² experts, h=8 heads, k=16 experts/head, query BatchNorm 启用。MoE 配置：expert-choice routing, 128 experts, 每个 expert 大小与对应 dense 模型 FFW 相同。PKM 配置：1024² memories, h=8 heads, k=32 memories/head, query BatchNorm 启用。
  - 数据集：C4（预训练 isoFLOP 分析），Curation Corpus、Lambada、Pile、Wikitext、C4（语言建模评估）。Benchmark 指标：perplexity（验证集）。
  - Expert usage 评估指标：Expert Usage（被检索 expert 比例）、Unevenness（expert 分布与均匀分布的 KL 散度）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供专门的开源仓库。作者引用 PKM 实现为参考：https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb。PEER 的 get_indices 和 query_proj 函数实现可参考该 PKM 实现。
  - 算法 pipeline（基于论文 Algorithm 1 和公式）：

  **Step 1: Query 投影** — 输入 x ∈ R^(b×t×d_model)，通过 h 个独立 query network 映射为 h 个 query 向量 q^i(x) ∈ R^(b×t×d)，其中 d 为 product key 维度。

  **Step 2: Product Key 检索** — 将每个 query q 拆分为两个子查询 q_1, q_2 ∈ R^(d/2)。计算子查询与两组子密钥 C, C'（各含 √N 个 d/2 维向量）的内积：
  ```
  I_C = TopK({q_1^T c_i | c_i ∈ C})  # k 个候选子密钥索引
  I_C' = TopK({q_2^T c'_j | c'_j ∈ C'})  # k 个候选子密钥索引
  ```
  候选 product key 集合 K' = {(c_i, c'_j) | i ∈ I_C, j ∈ I_C'}，共 k² 个候选。计算每个候选 key 与完整 query 的内积 = q_1^T c_i + q_2^T c'_j，再次 TopK 选出最终 k 个 expert 索引。总复杂度 O((√N + k²)d)。

  **Step 3: Expert 权重检索** — 通过 Embedding 层按索引检索 expert 的 down/up projection 权重：
  ```python
  w_down = w_down_embed(indices)  # shape: (b, t, h, k, d_model)
  w_up = w_up_embed(indices)      # shape: (b, t, h, k, d_model)
  ```

  **Step 4: Expert 计算与聚合** — 每个 expert 为单神经元 MLP: e_i(x) = σ(u_i^T x) v_i:
  ```python
  x = einsum('btd, bthkd -> bthk', x, w_down)  # 等价于 u_i^T x
  x = activation(x)                              # σ 非线性
  x = x * softmax(scores)                        # router score 加权
  x = einsum('bthk, bthkd -> btd', x, w_up)    # 输出投影
  ```
  其中 scores 来自 query-key 内积经 softmax/sigmoid 归一化。h 个 head 的输出直接求和（已在 einsum 的 h 维度上隐式完成）。
  - 论文指出"efficient implementation may require specialized hardware kernels to accelerate embedding lookup and fusion with the einsum operations"，当前实现为 JAX 原型。
