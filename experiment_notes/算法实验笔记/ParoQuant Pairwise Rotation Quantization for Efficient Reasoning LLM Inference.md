## ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：ParoQuant 是一种后训练权重量化（PTQ）方法，结合**hardware-efficient independent Givens rotations**（独立 Givens 旋转）与 **channel-wise scaling**（逐通道缩放），在量化前对权重进行可学习变换以抑制离群值（outliers）。核心包含三个设计：(1) **Scaled Pairwise Rotation Transform**——将权重矩阵按 group size=128 分组，每组内先应用逐通道缩放（diag(α)·W）拉平各通道幅值，再依次应用 K=8 个独立旋转（每个旋转由最多 64 对互不重叠的 Givens 旋转组成），使同一量化组内动态范围收窄；(2) **两阶段逐层优化**——Stage 1 用 AdamW 优化旋转角度 θ 和缩放因子 α 以最小化量化层输出误差 ||Q(l)(X') - l(X)||，Stage 2 采用 QAT-like 方式微调权重和量化参数 (s, z) 进一步消除残留离群值；(3) **算法-系统联合设计**——旋转核与反量化 GEMM kernel 融合，推理时对激活应用逆变换 T^{-1} = (R_1^{-1}...R_K^{-1})·diag(1/α)，计算仅涉及成对向量化乘加指令。
  - 实验比较：ParoQuant (W4A16, INT4, group=128) vs **AWQ** (channel-wise scaling, grid search)、**EfficientQAT** (layer-wise fine-tuning, SOTA 线性量化)、**QTIP** (随机 Hadamard + trellis 量化, SOTA 向量量化)、**QuIP#** (Hadamard + lattice codebook)、**OmniQuant** (可学习平滑参数)、**SpinQuant** (可学习旋转)。在 LLaMA-2-7B、LLaMA-3-8B/70B、LLaMA-3.1 Instruct 8B、DeepSeek-R1-Distill-LLaMA-3.1-8B、Qwen3-1.7B/4B/8B/14B 上评估 WikiText2/C4 困惑度、MMLU-Pro/GPQA Diamond/AIME-24/AIME-25（推理任务）、BoolQ/ARC-C/ARC-E/HellaSwag（非推理任务）。消融实验验证各组件（scaling, rotations, Stage 2 fine-tuning）的贡献，以及校准集大小（128-2048）、旋转数量（2-8）的影响。

- 硬件平台是什么，配置是什么。
  - 训练/量化优化：单张 NVIDIA H200 GPU。PyTorch 2.8.0 + Transformers 4.55.2 + Datasets 3.6.0。校准集 2048 samples × 2048 tokens（均匀采样自 WikiText2、C4、RedPajama），验证集 64 samples from Pile。Batch size=16（70B 模型减半），AdamW 优化器，learning rate: 旋转角度和缩放 0.05、权重 10^{-5}、scales/zero-point 10^{-6}，cosine 衰减至 1/20。量化 LLaMA-3-8B 耗时约 9 小时。
  - 推理测速：NVIDIA RTX A6000 (48GB)、RTX 6000 Ada (48GB)、RTX 4090 (24GB)，batch size=1 decode，PyTorch 2.6.0 + torch.compile max-autotune + CUDA Graphs。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LLaMA-2-7B、LLaMA-3-8B、LLaMA-3-70B、LLaMA-3.1-Instruct-8B、DeepSeek-R1-Distill-LLaMA-3.1-8B、Qwen3-1.7B/4B/8B/14B（Base 预训练模型）。所有线性层权重被量化为 W4A16（INT4 group=128）。
  - **校准数据集**：WikiText2 + C4 + RedPajama 各 1/3，共 2048 样本 × 2048 tokens。验证集：Pile 64 样本。
  - **PPL 评估**：WikiText2、C4（test split，LLaMA-3/Qwen3 context 8192，LLaMA-2 context 4096）。
  - **推理 Benchmark**：MMLU-Pro (12k samples，seed=42)、GPQA Diamond (198 samples，seeds=42/0/1)、AIME-24 (30 samples，seeds=42/0/1)、AIME-25 (30 samples，seeds=42/0/1)。使用 Lighteval 0.8.1 + vLLM 0.10.1。
  - **非推理 Benchmark**：BoolQ、ARC-Challenge、ARC-Easy、HellaSwag，使用 lm-eval-harness 0.4.9.1，batch size=32。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/z-lab/paroquant（MIT License，ICLR 2026 接收），PyPI `pip install paroquant`，HuggingFace 上提供已量化模型（如 z-lab/Meta-Llama-3-8B-PARO、z-lab/DeepSeek-R1-Distill-Llama-8B-PARO）。
  - ParoQuant W4A16 量化 pipeline（以 LLaMA-3-8B 单层为例，W ∈ R^{C_in×C_out}，group=128）：
    ```
    # === Stage 0: 分组与配对选择 (Algorithm A1) ===
    # 将 W 沿 channel 维度按 group size g=128 切分为子矩阵
    # 对每个 group (128×C_out)，独立选择 K=8 个 independent rotations，
    #   每个 rotation 选 N=64 对互不重叠的 channel pairs
    for each group W_g ∈ R^{128×D}:
        P_all = shuffle({(i,j) | 1≤i<j≤128})  # 所有可能的配对
        A[i,j] = 1 for i≠j  # 全局可用性矩阵
        for r = 1 to K:
            A_rot = copy(A)  # 当前 rotation 内的可用性
            for each pair (i,j) in P_all:
                if |P_r| ≥ N: break
                if A_rot[i,j]==0: continue
                append (i,j) to P_r
                A_rot[i,:]=0; A_rot[:,i]=0  # 禁用通道 i
                A_rot[j,:]=0; A_rot[:,j]=0  # 禁用通道 j
                A[i,j]=0; A[j,i]=0  # 全局禁用

    # === Stage 1: 优化旋转和缩放 (Algorithm A2) ===
    # 对每个 decoder layer l，逐层优化：
    for each layer l with calibration input X' (已量化的前层输出):
        Y = l(X)  # 原始层输出作为标签
        for each linear in l:
            # 对每个 group 应用 scaled pairwise rotation:
            α = ones(128)       # channel-wise scaling (初始化为1)
            θ = zeros(K×N)     # rotation angles (初始化为0)
            T(W_g) = (∏_{t=1}^{K} R(P_t, θ_t)) · diag(α) · W_g
            # R(P_t, θ_t) 是第 t 个 independent rotation，
            #   由 |P_t| 个互不重叠的 Givens 旋转组成
            #   每个 Givens 旋转: 
            #     W'[i,:] = cosθ·W[i,:] - sinθ·W[j,:]
            #     W'[j,:] = sinθ·W[i,:] + cosθ·W[j,:]
        
        # 量化变换后的权重:
        s = (max(T(W)) - min(T(W))) / 15   # INT4 量化步长
        z = -round(min(T(W)) / s)           # 零点
        W_q = clamp(round(T(W)/s) + z, 0, 15)
        
        # 优化目标: min_{α,θ} ||l'(X') - Y|| (SmoothL1Loss)
        #   where l' 使用量化后的权重，推理时对激活应用 T^{-1}:
        #     T^{-1}(X) = X · diag(1/α) · (R_1^{-1}) · ... · (R_K^{-1})
        #     R_t^{-1}: 逆序 Givens 旋转，角度取 -θ
        optimize α, θ with AdamW (lr=0.05, 10 epochs)

    # === Stage 2: 权重和量化参数微调 ===
    # Stage 1 后大部分离群值已消除，但仍可能有孤立离群值
    # 进一步微调权重 W 和量化参数 (s, z):
    for each layer l:
        optimize W, s, z with AdamW (lr=1e-5 for W, lr=1e-6 for s/z, 10 epochs)
        # 损失函数同 Stage 1, SmoothL1Loss

    # === 推理时 (Inference) ===
    # 对每个 linear layer Y = X·W + b:
    # 1. 应用逆变换到激活: X' = T^{-1}(X)
    #    - 在 fused CUDA kernel 中完成 (见 kernel调度)
    # 2. INT4 GEMM: Y = dequant(W_q) @ X'^T + b
    #    - 使用 AWQ 的 W4A16 GEMM kernel
    ```
  - 最终效果：在推理任务上 ParoQuant 平均精度仅下降 0.9%，相比 AWQ 提升 2.4%、比 EfficientQAT 提升 6.3%，且仅比 AWQ 慢约 10%，比 QTIP 快约 25%。
