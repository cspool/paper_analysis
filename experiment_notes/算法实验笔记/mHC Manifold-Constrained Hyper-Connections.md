## mHC Manifold-Constrained Hyper-Connections

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **Manifold-Constrained Hyper-Connections (mHC)**，将 HC 的残差连接矩阵投影到 Birkhoff polytope（双随机矩阵流形）上，恢复 identity mapping 性质。核心设计：
    - **残差映射约束**：将 HC 的 $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ 通过 Sinkhorn-Knopp 算法投影为双随机矩阵（行和=列和=1，元素非负），使 $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ 成为特征的凸组合，保证信号均值的保持和范数的正则化。
    - **输入/输出映射非负约束**：对 $\mathcal{H}_l^{\text{pre}}$ 和 $\mathcal{H}_l^{\text{post}}$ 施加 Sigmoid 函数约束（前者经过 $\sigma(\cdot)$，后者经过 $2\sigma(\cdot)$），防止正负系数组合导致的信号抵消。
    - **参数化**：先 flatten $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ 为 $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$，经 RMSNorm 后通过线性投影 $\varphi_l^{\text{pre}}, \varphi_l^{\text{post}} \in \mathbb{R}^{nC \times n}$ 和 $\varphi_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$ 获得动态映射，加上可学习 bias 作为静态映射，再进行约束投影得到最终映射矩阵。
    - **Sinkhorn-Knopp 迭代**：从 $\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{\text{res}})$ 开始，交替行归一化和列归一化，$t_{\text{max}} = 20$ 次迭代得到近似双随机矩阵。
    - **理论保证**：(1) 谱范数 $\|\mathcal{H}_l^{\text{res}}\|_2 \leq 1$，非膨胀；(2) 双随机矩阵的乘法封闭性保证跨层复合映射仍为双随机；(3) Birkhoff polytope 是所有置换矩阵的凸包，残差映射可解释为"置换的凸组合"，反复作用趋向单调增加跨流信息混合。
  - 实验比较：
    - **Baseline**：标准残差连接（Pre-Norm Transformer）
    - **HC**：Hyper-Connections（Zhu et al., 2024），expansion rate n=4
    - **mHC**：本文方法，expansion rate n=4，Sinkhorn-Knopp 20 次迭代
    - 下游 benchmark：BBH (3-shot EM), DROP (3-shot F1), GSM8K (8-shot EM), HellaSwag (10-shot Acc.), MATH (4-shot EM), MMLU (5-shot Acc.), PIQA (0-shot Acc.), TriviaQA (5-shot EM)
    - 缩放实验：Compute Scaling（3B→9B→27B 参数模型）、Token Scaling（3B 模型训练 1T tokens）

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号和集群规模。系统级开销评测指出训练引入了仅 6.7% 的额外时间开销（n=4），暗示使用大规模 GPU 集群进行训练。
  - 利用 DualPipe pipeline parallelism schedule（DeepSeek-V3 技术），涉及 NVLink 和 NIC 通信。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 DeepSeek-V3 架构的 MoE 模型，使用 MLA（Multi-Head Latent Attention）、Loss-Free Load Balancing、RMSNorm、RoPE。
    - 3B：12 layers, 1280 dim, 64 routed experts + 2 shared experts, 6 active experts, 16 attention heads
    - 9B：18 layers, 1920 dim, 64 routed experts + 2 shared experts, 6 active experts, 24 attention heads
    - 27B：30 layers, 2560 dim, 72 routed experts + 2 shared experts, 6 active experts, 32 attention heads
  - 数据集：论文未明确说明预训练数据集名称，仅提及按参数比例缩放训练 tokens。
    - 3B: 39.3B tokens, 9B: 105B tokens, 27B: 262B tokens
    - 3B (1T): 1.05T tokens 用于 token scaling 实验
  - Benchmarks: BBH, DROP, GSM8K, HellaSwag, MATH, MMLU, PIQA, TriviaQA

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明是否开源。DeepSeek 系列工作的代码尚未在公开仓库完全发布。
  - mHC 算法 pipeline 的伪代码：
    ```
    # 输入: x_l (n, C) - n-stream residual hidden state at layer l
    # 参数: phi_pre (nC, n), phi_post (nC, n), phi_res (nC, n^2)
    #        b_pre, b_post (1, n), b_res (n, n)
    #        alpha_pre, alpha_post, alpha_res (scalar)
    
    # Step 1: Flatten and normalize
    x_flat = flatten(x_l)  # shape: (1, nC)
    x_norm = RMSNorm(x_flat)  # shape: (1, nC)
    
    # Step 2: Compute raw mappings
    H_pre_raw  = alpha_pre  * (x_norm @ phi_pre)  + b_pre   # (1, n)
    H_post_raw = alpha_post * (x_norm @ phi_post) + b_post  # (1, n)
    H_res_raw  = alpha_res  * reshape(x_norm @ phi_res, (n, n)) + b_res  # (n, n)
    
    # Step 3: Manifold projection
    H_pre  = sigmoid(H_pre_raw)          # (1, n), non-negative
    H_post = 2 * sigmoid(H_post_raw)     # (1, n), non-negative
    H_res  = SinkhornKnopp(H_res_raw)    # (n, n), doubly stochastic
    
    # Sinkhorn-Knopp: H_res_raw -> M_0 = exp(H_res_raw)
    #   for t=1..20: M_t = normalize_rows(normalize_cols(M_{t-1}))
    #   return M_20
    
    # Step 4: Apply mappings
    layer_input  = H_pre @ x_l                    # (C,) - aggregate n streams to 1
    layer_output = F(layer_input, W_l)            # (C,) - standard layer computation
    x_{l+1}      = H_res @ x_l + H_post^T * layer_output  # (n, C) - update stream
    ```
  - **张量计算关键特性**：
    - 当 n=1 时，H_res 退化为标量 1，H_pre=H_post=1，mHC 完全恢复为原始残差连接。
    - 当 n>1 时，H_res 的双随机性保证行和=列和=1，$\|H_res\|_2 \leq 1$，跨层不影响信号范数。
    - 复合映射 $\prod_{i=1}^{L-l} H_{L-i}^{res}$ 仍为双随机矩阵（封闭性），Amax Gain Magnitude 约 1.6（vs HC 的 ~3000）。
