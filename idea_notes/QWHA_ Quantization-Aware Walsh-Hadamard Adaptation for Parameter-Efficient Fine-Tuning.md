## QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

- baseline方法是什么？
  - Baseline 方法分为两类：(1) LoRA-based QA-PEFT（以 CLoQ 为代表）：使用 SVD 分解量化误差 ΔW_Q，用低秩矩阵 BA 近似补偿，初始化时最小化层输出误差。适配器参数限制在 rank-r 子空间内，表示能力有限（normalized rank < 6.3%）。(2) FT-based 适配器（LoCA/DCA, SSH/DHA）：将权重更新表示为 ΔW = H'^{-1} F H^{-1}（双变换），F 为稀疏系数矩阵。初始化时参数位置 E 随机选择或部分随机+部分幅值选择（SSH），系数 c 初始化为零，不做量化误差的显式补偿。在 QA-PEFT 场景下，这类方法缺乏量化感知初始化，往往表现不如 LoRA-based 方法。
  - 全栈执行例子（Baseline: CLoQ on LLaMA-3.2-3B, 4-bit, P(r=64)）：
    - **算法pipeline**：GPTQ + MagR 4-bit 量化 W_0 → 收集 WikiText-2 校准集激活 X → 计算 R = (XX^T)^{1/2} → 最小化 ||ΔW_Q R - BA R||_F^2 初始化 A, B（低秩近似）→ Alpaca 数据集 fine-tuning 3 epoch → 推理时 y = (W_Q + BA)x。缺陷：(i) LoRA 的 rank ≤ r，对复杂量化误差模式（尤其是异常值）重建能力有限；(ii) 低秩结构限制 fine-tuning 的表示能力，增加参数 budget P(r) 也无法缩小与 WHA 的 gap（Figure 6 显示 QWHA P(r>32) 已超越 CLoQ 最大评分）。
    - **系统框架**：PyTorch + HuggingFace Transformers + PEFT，AdamW optimizer，cosine LR scheduler。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 PyTorch CUDA kernel（矩阵乘法 forward 和 LoRA 低秩分解）。推理吞吐 188.1 tok/s。
    - **硬件架构**：NVIDIA A100 80GB GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QWHA 提出三大创新组件解决 baseline 缺陷：
    (1) **WHA (WHT-based Adapter)**：将权重更新定义为 ΔW = F H^{-1}（仅单变换），H 为 WHT 矩阵（±1 元素）。WHT 基函数为方形波（sharp transitions），相比 DCT/DHT 的正弦基（smooth transitions），天然更适合捕获量化误差的异常值结构。理论分析：WHT 系数的 Pareto hill index η 最小（能量分布最陡），即 WHT 将最大比例的误差能量集中在最少系数中，使得稀疏适配器可以用少量参数高效补偿量化误差。实验验证 WHA 捕获最多异常值系数（avg 18.12% vs DCA 7.23%/DHA 17.06%）。
    (2) **AdaAlloc**：通道级自适应参数分配 p_i ∝ ||(ΔW_Q X)_{i,:}||_F^t，高误差输出通道获得更多参数，同时保证每个通道 ≥2 参数以维持 full rank（满足 Coja-Oghlan et al. 的稀疏随机矩阵 full-rank 条件）。在每个通道内，选取 |(ΔW_Q H)_{i,j}| 最大的 p_i 个系数位置。对比：纯幅值选择（Magnitude）过度集中参数于少数通道，导致 low-rank F 和 fine-tuning 能力下降（rank 接近 0）；随机选择（Random/LoCA/SSH）虽保持 high rank，但初始化误差大（Table 2: avg error 5.96/4.57 vs AdaAlloc 3.86）。AdaAlloc 是唯一同时实现 high rank 和 low init error 的策略（Figure 4 + Table 2）。
    (3) **Refinement**：对已选参数位置通过 v B'^T (B' B'^T)^{-1} 重新投影，使选中 basis vectors 的线性组合能补偿未选中向量。无 Refinement 时系数直接取自稠密解，忽略列间相关性，层输出误差仅略微降低（avg 7.21→7.06）；加入 Refinement 后层输出误差大幅下降（avg 7.21→3.86，约 46.5% 降幅）。
  - 全栈执行例子（QWHA on LLaMA-3.2-3B, 4-bit, P(r=64)）：
    - **算法pipeline**：
      1. GPTQ + MagR 量化 W_0 → W_Q（4-bit, group size 64）
      2. WikiText-2 128 条序列前向收集激活 X → 计算 R = UΣ^{1/2}, B = H^{-1}R
      3. AdaAlloc：p_i = floor(p × ||(ΔW_Q X)_{i,:}||_F / Σ||(ΔW_Q X)_{j,:}||_F)，每通道 ≥2
      4. Per-channel：v = (ΔW_Q)_{i,:} R → 选最大 |(ΔW_Q H)_{i,j}| 的 p_i 个位置 → E
      5. Refinement：B' = B[E], c = v B'^T (B' B'^T)^{-1}
      6. Alpaca fine-tuning：Y = (W_Q + α ΔW) X，ΔW = Scatter(c, E) H^{-1}
      7. 推理：WHT 通过 fast Hadamard kernel（仅加减法，O(n log n)），184.6 tok/s
    - **系统框架**：PyTorch + fast-hadamard-transform (Dao-AILab) + GPTQ。AdamW, lr=3e-5 (4-bit LLaMA-3.2-3B Alpaca)。训练时间 6.0h (batch=4)，远快于 LoCA (30.1h, DCA 双变换) 和 SSH (26.1h, DHT 双变换)，与 CLoQ (5.0h) 相当。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：fast Hadamard kernel（Dao-AILab）：无需显式矩阵构造，通过递归 fused kernel 实现 H^{-1} X 仅用加法和减法。WHT 比 DCT/DHT 更快因其无需复数运算。1D WHT vs 2D WHT 训练时间：batch=1 时 18.2h vs 25.3h，batch=4 时 6.0h vs 8.0h。
    - **硬件架构**：NVIDIA A100 80GB GPU。推理显存 QWHA 52.68GB vs CLoQ 59.53GB（减少 13.0%），因稀疏适配器 scatter ops 无额外内存开销，fast Hadamard kernel 无矩阵乘法。
  - **Baseline 缺陷 → 方法设计映射**：
    - (i) LoRA 低秩限制（rank ≤ r << d_min）→ WHA 的 full-rank 适配器（rank ≈ d_min），稀疏矩阵 F 的每行/每列 ≥2 非零元即保证 full rank，P(r≥4) 下 100% 满足条件。Figure 6 验证增加 LoRA rank 无法追上 WHA。
    - (ii) LoRA 低秩结构对量化误差异常值重建不足 → WHT 的方形波基函数在频域中天然适合表示突变/尖峰（异常值），Pareto hill index η 最小使 WHT 系数能量最集中，稀疏适配器用等量参数捕获更多误差能量（Figure 2(b), Figure 3）。
    - (iii) FT-based adapters 无量化感知初始化（参数随机/零初始化），导致 QA-PEFT 表现差甚至不如 LoRA → AdaAlloc + Refinement 实现误差驱动的参数位置选择和值优化，直接最小化层输出误差 ||ΔW_Q R - F H^{-1} R||_F^2。
    - (iv) 传统幅值选择（Magnitude）导致 low-rank F，随机选择初始化误差大 → AdaAlloc 通道级分配保证 full rank + 通道内幅值选择最小化误差，同时兼顾 fine-tuning 能力和初始化质量。
    - (v) 双变换（H'^{-1} F H^{-1}）计算开销大（DCT/DHT 需 63.3h/45.8h batch=1）→ WHA 单变换 + fast Hadamard kernel（仅加减法，无矩阵乘法），训练时间分别降至 18.2h/9.7h（batch=1/2），接近 CLoQ 的 12.5h/7.1h。
