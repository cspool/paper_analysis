## Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - **Student Float (SF4)**：基于 Student's t-distribution 推导的理论最优 4-bit 查找表数据类型。对 30+ DNN 进行 weight/activation 分布 profiling，发现大多数 DNN 分布由 Student's t-distribution（自由度 ν≈5）最优近似，而非正态分布。SF4 将概率质量均匀划分 16 份（固定 p₈=0.5 确保零无损表示），经 t-distribution 分位数函数 Q_S(p;ν) 映射后归一化到 [-1,1]，作为 NF4 的高精度替代品。
  - **Supernormal Support**：针对 E2M1 FP4 因正负零冗余浪费 1/16 值的问题，将负零重映射为额外超常值。提出两个变体：(a) super-range (SR) — 在分布边缘分配一个点扩展动态范围；(b) super-precision (SP) — 在分布内部增加一个点提升精度。同时将 SP 扩展到 APoT4。
  - 实验比较 baselines：
    - Lookup: NF4（Normal Float, 假设正态分布）
    - Integer: INT4, INT3
    - FP4 variants: E2M1-I (Intel), E2M1-B (bitsandbytes), E2M1 (standard), E3M0
    - Alternative: APoT4
  - 评估场景：weight-only PTQ（block size 128, RTN ± MSE clipping）、W4A4（± SmoothQuant）、GPTQ 比较、subchannel block size sweep (16/32/64/128/256)、3-bit 格式、CNN/ViT vision models

- 硬件平台是什么，配置是什么。
  - 量化实验：基于修改版 Intel Neural Compressor 库 + PyTorch；GPU 具体型号论文未明确说明
  - 硬件评估：Synopsys Design Compiler，TSMC 28nm 工艺，SystemVerilog RTL 综合 MAC 单元

- 模型是什么。数据集和bench分别是什么。
  - LLM：Mistral-7B, LLaMA2-7B, OPT-1B, OPT-6.7B, Phi-2, BLOOM-7B, Yi-6B, LLaMA-7B (多语言)
  - Vision：ResNet18, ResNet50, DenseNet121, ViT-B-16
  - 数据集/Benchmark：
    - LAMBADA（准确率 ↑）、WikiText-2（困惑度 ↓）
    - 零样本常识推理：HellaSwag, Winogrande, PIQA, BoolQ, ARC-c
    - 多语言 LAMBADA（EN/FR/DE/IT/ES）、ImageNet-1K（vision, Top-1 acc）
  - 校准方法：RTN（round-to-nearest）和 MSE clipping calibration；sub-channel block size 128；总计 4000+ 数据点

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/cornell-zhang/llm-datatypes
  - **SF4 导出算法（Algorithm 1）**：
    ```
    # 离线一次性计算
    δ = 0.5 * (1/32 + 1/30) = 0.0323
    # 在概率空间均匀分布 16 个点，p₈=0.5（零无损）
    p₁=δ, p₂,..., p₇, p₈=0.5, p₉,..., p₁₅, p₁₆=1-δ
    # 通过 t-distribution (ν=5) 分位数函数映射
    s̃ᵢ = Q_S(pᵢ; ν=5)   # Q_S = Student's t 分位数函数
    sᵢ = s̃ᵢ / maxᵢ|s̃ᵢ|    # 归一化到 [-1, 1]
    # s₁...s₁₆ 即为 SF4 的 16 个量化层级
    ```
  - **SF4 量化流程（类似 NF4/QLoRA 的 block-wise quantization）**：
    ```
    # Block-wise weight-only quantization
    W_flat = W.reshape(-1)
    blocks = W_flat.reshape(B, 128)     # block size 128
    for b in 1..B:
        w_max[b] = maxᵢ |blocks[b,i]|
        for i in 1..128:
            x = blocks[b,i] / w_max[b]       # normalize to [-1,1]
            idx = argminⱼ |x - sⱼ|           # nearest SF4 codebook entry
            Ŵ[b,i] = w_max[b] * s_{idx}      # decode
    # 存储：每个 weight 4-bit index + 每 block 一个 FP16/BF16 w_max
    ```
  - **Supernormal E2M1 原理**：
    ```
    # 标准 E2M1: 1 sign + 2 exp + 1 mantissa → 正值: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # 负零 = 冗余，浪费 6.25% (1/16) 位数空间

    # Super-precision (SP): 负零 → 5.0
    # 正值: {0, 0.5, 1, 1.5, 2, 3, 4, 5, 6}  — 分布内部增加 1 个层级

    # Super-range (SR): 负零 → 8.0
    # 正值: {0, 0.5, 1, 1.5, 2, 3, 4, 6, 8}  — 扩展动态范围
    ```
  - **APoT4 SP 原理**：APoT4 = (-1)^S (2^E + 2^Ẽ)，其中 E∈{0, 2⁻¹, 2⁻², 2⁻⁴}、Ẽ∈{0, 2⁻³}。SP 变体在同样的集合框架下复用负零位增加一个额外求和组合。
  - 核心洞察：E2M1 的形状分段逼近 SF4，这解释了为何 E2M1 比 INT4 精度更高——它对分布中心的密集区域分配了更多量化层级。
