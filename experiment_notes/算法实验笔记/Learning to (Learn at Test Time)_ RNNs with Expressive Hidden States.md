## Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **TTT (Test-Time Training) 层**——一种将 RNN 隐藏状态建模为机器学习模型本身、更新规则为自监督学习梯度步的序列建模层。两个实例：**TTT-Linear**（隐藏状态为线性模型 `f(x)=Wx`）和 **TTT-MLP**（隐藏状态为两层 MLP，hidden dim 4×，GELU 激活，含 LN 和残差连接）。关键创新包括：mini-batch TTT（b=16）实现内循环并行化；dual form 将梯度计算转化为矩阵乘法以提高 GPU/TPU 硬件利用率；可学习的 reconstruction views（θ_K, θ_V, θ_Q）；可学习的初始权重 θ_init = W_0；可学习的逐 token 学习率 η(x) = η_base · σ(θ_lr · x)。论文在 Mamba backbone（含时序卷积）和 Transformer backbone 下评估。
  - 实验比较：TTT-Linear 和 TTT-MLP vs. **Transformer**（Llama-based Transformer++，含 RoPE、SwiGLU、RMSNorm）和 **Mamba**（现代 RNN baseline），在 125M/350M/760M/1.3B 四个规模下，使用 matched training FLOPs。消融实验展示从 linear attention 逐步加入 learnable W_0、LN+residual in f、mini-batch TTT(b=16 vs b=T)、learnable η、Mamba backbone 的改进过程（Table 1）。额外消融 mini-batch size b 对 perplexity 和 wall-clock time 的影响（Figure 7）。

- 硬件平台是什么，配置是什么。
  - 训练：TPU v5e-256 pod
  - 推理延迟评测：NVIDIA A100 GPU 80G HBM，PCIe 连接
  - Transformer 推理 baseline 使用 vLLM serving 系统

- 模型是什么。数据集和bench分别是什么。
  - 模型规模：125M（12层/d=768）、350M（24层/d=1024）、760M（24层/d=1536）、1.3B（24层/d=2048）参数
  - 数据集：**The Pile**（标准 2k 和 8k 上下文实验）、**Books3**（Pile 子集，用于长上下文 1k-32k 实验）
  - 评估指标：perplexity（PPL）、scaling law 曲线（FLOPs vs. PPL）、每 token 平均 PPL 随 token index 变化、wall-clock latency（forward/prefill 和 generate/decode）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：JAX 实现 https://github.com/test-time-training/ttt-lm-jax，PyTorch 实现 https://github.com/test-time-training/ttt-lm-pytorch。基于 EasyLM 框架。
  - 算法 pipeline 核心伪代码（TTT-Linear 单层，第一个 mini-batch，dual form）：

    ```
    # 输入: X = [x_1,...,x_b] ∈ R^{d×b} (mini-batch of tokens)
    # 可学习参数: θ_K, θ_V, θ_Q ∈ R^{d×d} (reconstruction views)
    #             θ_init = W_0 ∈ R^{d×d} (初始权重)
    #             θ_lr ∈ R^d (学习率参数)
    # 超参: η_base = 1.0 (TTT-Linear), b = 16

    # Step 1: 生成 training view, label view, test view
    X̂ = θ_K @ X          # training view, 低秩投影
    Y  = θ_V @ X          # label view
    X̄ = θ_Q @ X          # test view

    # Step 2: 自监督损失 (MSE reconstruction)
    # ℓ(W_0; x_t) = ||W_0 x̂_t - y_t||²
    # 对每个 token 的梯度: G_t = ∇ℓ(W_0; x_t) = 2(W_0 x̂_t - y_t) x̂_t^T

    # Step 3: mini-batch 更新 (dual form, 无需显式计算 G_t)
    # W_b = W_0 - η Σ_{t=1}^b G_t = W_0 - 2η (W_0 X̂ - Y) X̂^T
    W_b = W_0 - 2 * η * (W_0 @ X̂ - Y) @ X̂.T

    # Step 4: 输出 token 计算 (dual form)
    # z_t = W_t x̄_t = (W_0 - η Σ_{s=1}^t G_s) x̄_t
    # Z = [z_1,...,z_b] = W_0 X̄ - 2η (W_0 X̂ - Y) mask(X̂^T X̄)
    # 其中 mask 是上三角 mask（类似 attention mask, 但用 0 替代 -∞）
    Δ = (W_0 @ X̂ - Y) * mask(X̂.T @ X̄)    # mask 为上三角 1/下三角 0
    Z = W_0 @ X̄ - 2 * η * Δ
    ```

  - 内循环/外循环双层训练：
    - **内循环 (TTT)**：对每个序列，从 W_0 开始，对每个 mini-batch 计算梯度并更新 W。目标是最小化 reconstruction loss ℓ(W; x_t) = ||f(θ_K x_t; W) - θ_V x_t||²。
    - **外循环 (常规训练)**：优化 θ_rest（网络其余参数）、θ_K, θ_V, θ_Q, θ_init, θ_lr。目标是最小化 next-token prediction loss。训练配置遵循 Chinchilla recipe（AdamW, cosine schedule, warmup, weight decay 0.1, gradient clipping 1.0, mixed precision）。
  - 时间/空间复杂度：每个 token O(d²)（与序列长度 T 无关），dual form 将内循环 mini-batch 计算转化为 matmul 以利用 TensorCores。
