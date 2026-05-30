## PARQ Piecewise-Affine Regularized Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PARQ 是一种 QAT 方法，核心贡献包括：(1) **凸分段仿射正则化（PAR）**——定义 PAR(w)=max_k{a_k(|w|-q_k)+b_k}，其中斜率 a_k 满足 0≤a_0<a_1<...<a_m=+∞，具有严格递增斜率的分段仿射凸函数，在非光滑点（±q_k）处产生聚类效应；(2) **AProx（Aggregate Proximal Gradient）算法**——用累积步长 γ_t=Ση_s 缩放 proximal map 替代标准 Prox-SGD 中逐次缩放的 proximal map，使软量化（slanted segments）随训练逐步收敛到硬量化（hard quantization），解决了 Prox-SGD 中随 η_t→0 的 diminishing regularization 问题；(3) **PARQ 实用实现**——使用 LSBQ（Least Squares Binary Quantization）在线估计目标量化值 Q，通过独立的逆斜率 schedule ρ_t^{-1}（cosine decay 或 sigmoid decay）从软量化渐近到硬量化，无需预先指定 λ 和 {a_k}。
  - 实验比较：PARQ vs **STE/BinaryConnect**（Courbariaux et al. 2015，全程使用硬量化映射）vs **BinaryRelax**（Yin et al. 2018，非凸正则化，slanted segment 斜率逐步减小至 0），在 5 种位宽（ternary T、1-4 bits）下的分类准确率比较。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号和硬件配置。使用 PyTorch 框架进行训练，训练设备为标准 GPU（论文作者来自 Meta FAIR，使用公司内部 GPU 集群）。开源代码 https://github.com/facebookresearch/parq 可在单 GPU 或多 GPU 环境运行。

- 模型是什么。数据集和bench分别是什么。
  - **模型**: ResNet-20、ResNet-56（CIFAR-10 实验）；ResNet-50（ImageNet 实验）；DeiT-Ti（5M 参数）、DeiT-S（22M 参数）、DeiT-B（86M 参数）（ImageNet 实验）。所有模型权重均被量化，卷积/注意力 block 权重 per-channel 量化（row-wise over tensors）。对 DeiT，embedding、layer normalization 参数和最终 projection 权重保持全精度。
  - **数据集/Benchmark**: CIFAR-10（图像分类）、ImageNet（ILSVRC 2012，图像分类）。每项实验 3 次随机种子取平均，报告 mean ± std。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：PyTorch 包 https://github.com/facebookresearch/parq，实现 PARQ 及多种主流 QAT 方法（STE/BinaryConnect、BinaryRelax），可复现论文结果。
  - PARQ 算法核心流程（Algorithm 1）：
    ```
    输入: w¹ ∈ R^d, 量化比特数 n, 步长 {η_t}_{t=1}^T, 逆斜率 schedule {ρ_t^{-1}}_{t=1}^T
    初始化: u¹ = w¹
    for t = 1, 2, ..., T-1 do
        u^{t+1} = u^t - η_t ∇f(w^t, z^t)        // 前向步: 在量化参数 w^t 处计算梯度
        Q^{t+1} = LSBQ(u^{t+1}, n)               // 在线估计目标量化值
        w^{t+1} = prox_PARQ(u^{t+1}, Q^{t+1}, ρ_t) // 聚集 proximal 映射
    end for
    输出: w^T
    ```
  - LSBQ 估计 Q：对于 n-bit 量化，将 u∈R^d 近似为 w_i = Σ_{j=1}^n v_j s_j(u_i)，其中 v_j 为递减的正标量，s_j(u_i)∈{-1,1} 为二进制函数。解通过 greedy foldable representation 获得：s_j(u_i)=sgn(u_i-Σ_{ℓ=1}^{j-1} v_ℓ s_ℓ(u_i))。Q={±q_1,...,±q_m} 为 v_j 的组合（如 q_m=v_1+...+v_n），|Q|=2^n。
  - prox_PARQ 结构（图 9a）：与 AProx 的渐进性不同，PARQ 使用独立斜率 ρ_t。ρ_t^{-1} 从 1 单调递减到 0（cosine decay 或 sigmoid），使得 proximal map 从接近 identity（训练初期，slope≈1）逐步过渡到硬量化（训练末期，slope→∞）。这避免了因为有限训练迭代次数导致 γ_t 不够大的问题。
  - AProx 理论核心：与 Prox-SGD 的区别在于，Prox-SGD 中的 u^{t+1}=w^t-η_t g^t（w^t 已含过往正则化贡献），使用近端正则化 η_t λ Ψ 平衡单步梯度；而 AProx 中的 u^{t+1}=u^t-η_t g^t（u^t 仅累加梯度），使用聚集正则化 γ_t λ Ψ 平衡所有过往梯度。AProx 等价于 ProxConnect（Dockhorn et al. 2021），但通过凸 PAR（而非任意单调 proximal map）提供了更强的收敛保证。证明了最后迭代（last-iterate）收敛 O(ln(t)/√t)，而非仅平均迭代收敛。
  - 训练细节：ResNet 使用 SGD（momentum=0.9, weight decay=2e−4），200 epochs，lr=0.1 在 epoch 80/120/150 除以 10；DeiT 使用 AdamW（lr=5e−4, weight decay=0.05），300 epochs，最后 20 epochs 将 lr 固定在 1e−8。DeiT 使用 RandAugment、mixup 和 CutMix（不含 repeated augmentation）。
