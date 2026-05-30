## Mixture-of-Experts with Expert Choice Routing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Expert Choice（专家选择）路由方法：与传统的 token-choice routing（每个 token 选择 top-k 专家）相反，让每个专家独立选择 top-k 个 token。该方法通过 Softmax 计算 token-to-expert affinity 矩阵 S，然后对 S^T 的每一行（每个专家）取 TopK，实现完美负载均衡并允许每个 token 获得可变数量的专家。
  - 实验比较：(1) EC-CF2（容量系数 c=2）vs Switch Transformer Top-1 gating vs GShard Top-2 gating，评估预训练 perplexity 收敛速度和下游 GLUE/SuperGLUE 11 任务 fine-tuning 性能；(2) 扩展专家数量（16→32→64→128）对 perplexity 的影响；(3) 变体：EC-CAP2/CAP3（限制每个 token 最多 2/3 个专家）vs 无约束 EC-CF2；(4) 对比 Hash Layer 路由；(5) 容量系数 ablation（c=0.5, 1, 2）；(6) 与同规模 Dense 模型的预训练比较。

- 硬件平台是什么，配置是什么。
  - 训练平台：Google TPU V4 chips。最大模型（8B/64E）使用 512 TPU V4 chips。
  - 使用 GSPMD 的 2D sharding 算法进行模型分区，充分利用 TPU 集群的 2D 拓扑。
  - 训练精度/优化器：Adafactor optimizer（β1=0, β2=0.99），无 auxiliary load balancing loss，dropout rate=0。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 Transformer 架构，每两层替换一层 FFN 为 MoE 层。100M 规模系列（expert size=100M，专家数 16/32/64/128）和 8B/64E（8B activated params，每 token 9.8B，总参数 143B，32 layers, M=4096, H=16384, 32 heads, d_head=128）。非 MoE FFN 层使用 Gated Linear Unit (GLU) + GeLU。使用 per-layer relative positional bias。SentencePiece tokenizer（vocab 256K）。
  - 数据集：GLaM 数据集——1.6 trillion tokens，由高质量网页子集、书籍、Wikipedia、对话、论坛和新闻混合而成（详见 GLaM 论文 Table 3）。
  - Benchmark：GLUE 和 SuperGLUE 的 11 个任务——BoolQ, CB, CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST2, WiC, WNLI。主要指标为 accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文本身来自 Google，未在论文中提供直接开源链接，但相关实现在 Google Research 仓库中：`google-research/sparse_mixers/routing.py` 包含 ExpertsChoose 路由逻辑（JAX/Flax）；Flaxformer/T5X 中也包含 "Experts choose" routing 机制。
  - 第三方 PyTorch 实现：`pytorch-mixtures`（`pip install pytorch-mixtures`）提供 `ExpertChoiceRouter`，可直接插入 MoE 层。

  **算法 pipeline（张量计算伪代码）**：

  ```
  输入: X ∈ R^{n×d}  (n = batch_size × seq_len, d = hidden_dim)
        W_g ∈ R^{d×e}  (专家嵌入矩阵, e = 专家数)
        c (容量系数), e (专家数)

  Step 1: 计算 token-to-expert affinity
      S = Softmax(X @ W_g)  ∈ R^{n×e}

  Step 2: 专家选择 token（对 S^T 的每行取 top-k）
      k = n × c / e  （每个专家的容量）
      G, I = TopK(S^T, k)
      # G ∈ R^{e×k}: 门控权重
      # I ∈ R^{e×k}: I[i,j] = 第 i 个专家选择的第 j 个 token 的索引
      P = OneHot(I)  ∈ R^{e×k×n}  (排列矩阵)

  Step 3: 按专家排列输入
      X_in = P @ X  ∈ R^{e×k×d}
      # X_in[i] ∈ R^{k×d}: 第 i 个专家的输入 token 集合

  Step 4: 每个专家独立计算 FFN
      for i in range(e):
          X_e[i] = GeLU(X_in[i] @ W_1[i]) @ W_2[i]^T
      # X_e ∈ R^{e×k×d}

  Step 5: 反排列回原始 token 顺序
      X_out[l, d] = Σ_{i,j} P[i,j,l] × G[i,j] × X_e[i,j,d]
      # 等价于: X_out = unshuffle(G ⊙ X_e, P)
  ```

  **可选的约束版本（EC-CAP）**：
  使用熵正则化线性规划限制每个 token 最多 b 个专家：
  ```
  max_A ⟨S^T, A⟩ + λH(A)
  s.t. Σ_j A[i,j] = k (每个专家选k个), Σ_i A[i,j] ≤ b (每个token最多b个专家)
  求解: Dykstra's 交替投影算法 (λ=0.001, max 100 iterations)
  然后: I = TopK(A, k)
  ```

  **使用例子（pytorch-mixtures）**：
  ```python
  from pytorch_mixtures.routing import ExpertChoiceRouter
  from pytorch_mixtures.moe_layer import MoELayer

  router = ExpertChoiceRouter(dim=768, num_experts=64)
  moe = MoELayer(
      num_experts=64,
      router=router,
      experts=experts,        # 64 个 FFN 专家
      capacity_factor=2.0     # 匹配 GShard top-2 的计算量
  )
  output = moe(input_tokens)  # input_tokens shape: (batch, seq, dim)
  ```
