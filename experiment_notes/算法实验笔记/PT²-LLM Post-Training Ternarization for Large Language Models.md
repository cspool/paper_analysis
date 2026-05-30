## PT²-LLM Post-Training Ternarization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PT²-LLM 是一种后训练三值量化（Post-Training Ternarization）框架，将 LLM 权重量化到 {−1, 0, +1} 三值（等效 1.58-bit）。核心包含三个组件：(1) **非对称三值量化器（Asymmetric Ternary Quantizer, ATQ）**——引入逐行偏移 μ 捕获权重非对称分布，通过两阶段无训练优化：**Iterative Ternary Fitting (ITF)** 交替求解最优三值网格参数 (α, μ)（闭式解）和最优三值赋值 T（逐元素弹性舍入），迭代到收敛（~10 轮）；**Activation-aware Grid Alignment (AGA)** 利用校准数据激活统计量 C = Σ XX^T，以输出误差 E_x = ||WX - ŴX||² 为目标闭式求解更优的网格参数 (α, μ)，冻结 T 避免过拟合。(2) **结构相似性重排序（Structural Similarity-based Reordering, SSR）**——在 GPTQ 逐块量化框架中，每次选择下一块时基于残差矩阵的列间余弦相似度选取 top-k 最相似的列组成量化块，使块内权重分布更紧凑、方差更小，抑制离群值影响。
  - 实验比较：PT²-LLM (1.58-bit) vs **GPTQ** (2-bit) vs **AWQ** (2-bit) vs **QuIP** (2-bit) vs **Slim-LLM** (混合精度 2-bit SOTA) vs **PB-LLM** (1.7-bit，位宽最接近的 baseline)，在 LLaMA-7B/13B/65B、LLaMA-2-7B/13B/70B、LLaMA-3-8B、Qwen3-14B-Base 上比较 WikiText2/C4 困惑度（PPL）和 7 个零样本 QA 任务（PIQA, ARC-e, ARC-c, HellaSwag, Winogrande, OBQA, BoolQ）的准确率。消融实验验证 ITF、AGA、SSR 各自贡献，以及校准集大小和类型的影响。额外比较压缩时间、模型大小（Size Reduction）、推理吞吐（llama.cpp 上 prefill/decode/end-to-end 吞吐对比 2-bit 模型）。

- 硬件平台是什么，配置是什么。
  - 单张 Nvidia A800-80GB GPU。PTQ 校准阶段：WikiText2（128 个随机 2048-token 片段作为校准集），block size=128，无需训练或梯度反传。压缩 LLaMA-7B 耗时约 32 分钟。推理测速：llama.cpp（https://github.com/ggml-org/llama.cpp）在 Nvidia A800 GPU 上，序列长度 128（prefill）、256（decode）、128+256（end-to-end）。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LLaMA-1（7B, 13B, 65B）、LLaMA-2（7B, 13B, 70B）、LLaMA-3（8B）、Qwen3-14B-Base。所有线性层权重被量化。
  - **校准数据集**：WikiText2（128 个 2048-token 片段）。
  - **评估数据集（PPL）**：WikiText2、C4。
  - **评估 Benchmark（零样本 QA）**：PIQA、ARC-easy、ARC-challenge、HellaSwag、Winogrande、OpenBookQA、BoolQ。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/XIANGLONGYAN/PT2-LLM（Apache 2.0，论文被 ICLR 2026 接收，代码和量化模型标记为将发布，截至检查尚未上传完整实现）。arXiv: 2510.03267。
  - PT²-LLM 量化流程（以 LLaMA-7B 单层线性层 W∈R^{n×m} 为例，block size=128）：
    ```
    # === SSR: Structural Similarity-based Reordering ===
    # 输入: W ∈ R^{n×m}, block_size k=128
    # 逐块处理，每次根据残差矩阵列间余弦相似度选块：
    for col_start in range(0, m, k):
        # 计算残差矩阵的列均值参考向量
        w_bar = mean(W_remaining, axis=1)  # (n,)
        # 计算剩余每列与 w_bar 的余弦相似度
        sim_j = (W_remaining[:,j] · w_bar) / (||W_remaining[:,j]||₂ * ||w_bar||₂)
        # 选 top-k 相似列作为当前量化块
        B = top_k_columns(sim_j, k)
        # 对该块执行 ATQ 量化 + GPTQ 误差补偿
        quantize_and_compensate(B)
    
    # === ATQ: Asymmetric Ternary Quantizer (对每个 block) ===
    # 输入: W_block ∈ R^{n×k}, X_calib ∈ R^{B×L×k}
    # Step 1: 非对称初始化
    μ = row_mean(W_block)                          # (n,): 逐行均值
    W_tilde = W_block - μ                           # 中心化
    Δ = 0.75 * row_mean(|W_tilde|)                 # (n,): 阈值估计
    T_ij = 1 if W_tilde_ij > Δ_i else (-1 if W_tilde_ij < -Δ_i else 0)
    α = Σ_j(T_ij * W_tilde_ij) / Σ_j(|T_ij|)       # (n,): 最优缩放
    
    # Step 2: ITF — 迭代三值拟合 (约10轮收敛)
    while T != T_prev:
        # 闭式求解最优网格 (α*, μ*)
        α* = (m*(W∘T)1 - (T1)∘(W1)) / (m*(T∘T)1 - (T1)²)
        μ* = ((T∘T)1∘(W1) - (T1)∘((W∘T)1)) / (m*(T∘T)1 - (T1)²)
        # 弹性舍入更新 T
        Z_ij = (W_ij - μ*_i) / α*_i
        T_ij = argmin_{t∈{-1,0,1}} |Z_ij - t|
    
    # Step 3: AGA — 激活感知网格对齐
    C = Σ_b Σ_l X_{bl} X_{bl}^T                        # 激活协方差
    d = 1^T C 1                                        # 标量
    v = T C 1                                           # (n,)
    α* = (d*(W∘T)S1 - v∘(WS1)) / (d*T²S1 - v²)         # 激活感知闭式解
    μ* = (T²S1∘(WS1) - v∘((W∘T)S1)) / (d*T²S1 - v²)
    # T 冻结不更新，避免过拟合
    
    # 输出: Ŵ = α* T + μ*  (每行仅3个可能值: {-α_i+μ_i, μ_i, α_i+μ_i})
    ```
  - 关键公式：量化后权重 Ŵ = αT + μ，其中 α 为逐行缩放因子、μ 为逐行偏移、T∈{−1,0,+1}^{n×m}。存储时仅需保存 α（n个float）、μ（n个float）和 T（n×m 个 2-bit 索引），理论位宽 ≈ 1.58 bit/权重。
