## TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **Expert Group Approximation（专家组近似）** 方法，在 MoE 训练中为路由器提供稠密梯度信号。核心思想：MoE 的 Top-K 路由只激活 K 个专家，未被路由到的专家不产生梯度，导致路由器学习信号稀疏。论文提出用已被路由到专家 i 的其他 token 的输出来近似 token x 对专家 i 的输出 $E_i(x)$。具体公式：$\forall x \in X_R : \hat{E}_i(x) = \frac{1}{K} \sum_{j \in R} \frac{1}{|X_{\{i,j,\cdot\}}|} \sum_{x' \in X_{\{i,j,\cdot\}}} E_i(x')$，产生 $N^2$ 个总近似。在前向传播中保持不变，在反向传播中通过 stop-gradient 操作将近似梯度注入：$y := y + y' - \operatorname{sg}(y')$。同时更新路由器参数和专家参数以保证一致性。
  - 实验比较：(a) Expert Group Approximation vs Top-K (K=2) baseline 在 FineWeb 200B tokens 上的 training loss 和 validation perplexity（Figure 5）；(b) 多 benchmark 评估：mathqa, logiqa2, mmlu, openbookqa, logiqa, arc challenge, arc easy, hellaswag, copa, piqa（Table 1，平均 +0.9%）；(c) 与 Sparsemixer (Liu et al., 2023) 对比（Section 4.3）；(d) Expert scaling（8 vs 32 experts）和 Batch Size scaling（$2^{19} / 2^{20} / 2^{21}$）ablation（Table 2），改善随稀疏度和 batch size 增大而提升（最高 1.5%）；(e) K=1 vs K=2 ablation（Table 3），方法在 K=1 时仍有效；(f) Accurate vs Viable 加权变体对比（Table 3）；(g) 与 K=3 baseline 对比（Table 5）：K=2 + Expert Group Approx. 达到与 K=3 相同的 perplexity 但不增加激活参数。

- 硬件平台是什么，配置是什么。
  - 单 GPU（用于 throughput 测量，reproducibility）；8 节点多节点训练集群（用于主实验）。具体 GPU 型号论文未明确说明。通过 NCCL 进行数据并行通信。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) Fine-grained MoE（DeepSeek 风格）：32 experts，hidden dim 1024，每个 expert 为 bottleneck MLP（intermediate size 704），2B total params，K=2 时 470M active params；(2) 标准 MoE：8 experts，hidden dim 1024，MLP intermediate size 2816，2B total params，780M active params。均使用 24 层 Transformer、16 attention heads（dim 64）、SwiGLU MLP、LayerNorm、RoPE。
  - 数据集：FineWeb（Penedo et al., 2024），200B tokens 训练，Llama3 tokenizer。
  - Benchmarks：mathqa, logiqa2, mmlu, openbookqa, logiqa, arc_challenge, arc_easy, hellaswag, copa, piqa。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：论文声明代码为开源（"Our code is currently open-source and will be linked here upon publication"），但为匿名投稿，链接未提供。使用开源框架 GPT-NeoX (Andonian et al., 2023) + Megablocks (Gale et al., 2022)。
  - 算法 Pipeline 的伪代码描述：

```
# === 前向传播（与标准 Top-K MoE 相同） ===
输入: token x, 路由器权重矩阵 W (shape: [N, d_token])
路由logits = W @ x                        # [N]
π = Softmax(路由logits)                     # [N], 专家权重
R(x) = TopK(π, K)                         # 选出的 K 个专家索引
y = Σ_{i∈R(x)} π_i * E_i(x)               # 标准 MoE 输出

# === 构造近似（Expert Group Approximation） ===
对于每个路由决策 R（共 C(N,K) 种可能）:
  对于每个未激活的专家 i ∉ R:
    近似 = 0
    对于每个激活专家 j ∈ R:
      相邻token集 = X_{i,j,·}  # 同时被路由到专家i和j的tokens
      近似 += mean({E_i(x') for x' ∈ X_{i,j,·}})  # 取平均
    近似 /= K  # 对 K 个组取平均
    ŷ_i(x) = 近似  # 对属于 X_R 的所有 x 使用同一近似

# === 构造稠密近似输出 ===
y' = Σ_{i∉R(x)} ŷ_i(x)                    # 所有未激活专家的近似输出之和

# === 注入近似梯度（前向不变，反向有梯度） ===
y = y + y' - stop_gradient(y')            # Eq.(6)
# stop_gradient 确保前向输出不变
# 反向时 ∂y/∂π = [E_1(x), ..., E_N(x)] 包含所有专家的梯度

# === 参数更新 ===
# 路由器 W: 接收来自所有 N 个专家的稠密梯度
# 专家 E_i: 除自身处理的 K/N 比例 token 外，还接收 (N-K)/N 比例的近似梯度
# 跨数据并行 workers 执行 all-reduce 聚合近似梯度
```

  - 张量计算关键步骤：路由器前向（matmul + softmax + TopK）不变；反向时用 stop-gradient 技巧将 N² 个专家组近似插入计算图，使路由器接收到稠密的 ∂y/∂π 梯度向量。近似在数据并行 workers 之间 all-reduce，增加样本量以估计稠密梯度。
