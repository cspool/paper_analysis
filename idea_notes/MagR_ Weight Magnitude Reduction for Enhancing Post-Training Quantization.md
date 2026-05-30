## MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

- baseline方法是什么？
  - **直接 PTQ（RTN / OPTQ）**：对预训练权重直接做 uniform quantization，不进行预处理。量化步长 δ = (max(w) − min(w))/(2^b − 1) 由原始权重的最大/最小值决定。由于预训练权重中存在 outliers 和大范围分布，δ 偏大，导致量化网格稀疏，量化误差大，尤其在 sub-4bit 时 perplexity 急剧上升。OPTQ 虽用二阶 Hessian 信息逐列贪心补偿量化误差，但无力改变权重本身的分布范围。
  - **线性变换预处理方法（AWQ / OmniQuant / QuIP）**：对权重施加可逆线性变换 T（通道缩放、随机正交变换等），使 TW 比 W 更"量化友好"（幅度小、无 outliers），然后量化 TW。推理时需在特征上施加 T⁻¹（XT⁻¹），额外引入计算和存储开销（QuIP 比 OPTQ 慢约 1.5×）。
  - 全栈执行例子（以 OPTQ baseline 为例）：
    - 算法层：加载 FP16 预训练 LLaMA 模型 → 逐层用 calibration 数据计算 Hessian H = XᵀX → 对每列权重贪心取整并更新未量化权重的 Hessian 补偿 → 得到 INT-k 量化权重。
    - 系统框架层：基于 PyTorch + HuggingFace Transformers，block-wise 加载（每次 7 个 linear layer 到 GPU），推理时执行量化矩阵乘法（FP16 反量化 × FP16 激活）。
    - 编译框架/kernel调度/硬件架构层：论文未明确说明（依赖 PyTorch 默认的 CUDA kernel 和 NVIDIA A100 Tensor Core）。
  - Baseline 核心缺陷：权重幅度大 → 量化步长大 → 量化误差大；线性变换方法虽能降幅度，但引入推理开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MagR 方法**：在量化前，对每层权重做 channel-wise ℓ∞-regularized 最小二乘优化（式2），通过 Proximal Gradient Descent（式3）+ ℓ₁-ball 投影（Moreau 分解，式4）迭代求解新的权重 W'，使 ‖W'‖∞ 最小化且保持 ‖XW' − XW_hat‖ ≤ ε。预处理后的 W' 直接替代原始权重，后续量化（RTN / OPTQ / QuIP）无需对特征做任何逆变换。
  - **核心洞察**：特征矩阵 X 近似秩亏（表2，fraction rank 均值 70%–84%，最低仅 0.1%），意味着 X 的核空间非平凡，存在大量 w 满足 Xw = Xw_hat。MagR 在核空间中寻找 ‖w‖∞ 最小的解，从而缩小量化步长 δ，而不改变层输出。
  - **如何解决 Baseline 缺陷**：
    - 针对"权重幅度大"：ℓ∞-regularization 直接将每列权重的最大绝对值作为优化目标压到最小，Figure 1 显示列最大幅度通常可减半以上；Table 1 表明 MagR 预处理后的 FP16 模型 perplexity 几乎无损（LLaMA2-7B WikiText2: 5.47→5.52）。
    - 针对"推理开销"：MagR 是非线性变换，不产生 T⁻¹，不修改特征/激活路径，推理时零开销。
    - 针对"sub-4bit 精度差"：通过缩小 δ（含 β 缩放因子进一步聚拢量化网格），大幅降低量化误差（Figure 2 显示各层量化 RMSE 显著下降）；W2A16 下 MagR+OPTQ† 在 LLaMA2-70B 上达 WikiText2 PPL 5.95，优于 OmniQuant (7.81) 和 QuIP (6.33)。
  - 全栈执行例子：
    - 算法层：加载 FP16 预训练权重 W_hat → 逐层用 calibration 数据计算 Hessian H = XᵀX → 运行 proximal gradient descent（K=150 迭代，每次迭代：梯度步 V^k = W^k − η·H·(W^k − W_hat)，然后列级 ℓ₁-ball 投影 W^{k+1} = V^k − ηα·proj_{‖·‖₁≤1}(V^k/(ηα))）→ 得预处理权重 W' → 对 W' 做 OPTQ/RTN 量化（含 β-scaled δ）→ 输出 INT-k 模型。
    - 系统框架层：与 baseline 相同，PyTorch + HuggingFace，block-wise GPU 加载，推理时标准量化矩阵乘法（无额外变换）。
    - 编译框架/kernel调度/硬件架构层：论文未明确说明（与 baseline OPTQ 同，依赖 PyTorch 默认 CUDA kernel）。
  - **MagR 预处理开销**：LLaMA2-7B ~15 min，13B ~30 min，70B ~3.5 hr（单 A100），仅一次性预处理；总量化时间（MagR+OPTQ）约为 OmniQuant 的一半。
