## Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 **LTDR (Long-Tailed Distribution-aware Router)**，包含两个核心模块：
    1. **Modality-specific Distribution-aware Router (MsDaR)**：针对 LVLM 中 vision tokens 服从 long-tailed 分布、language tokens 服从 uniform 分布的模态差异，移除 vision TER 的 load balancing 约束（只保留 language TER 的 load balancing），让 vision tail tokens 能路由到专业化 experts 而非被负载均衡打散。
    2. **Vision-specific Dynamic Expert Activation (VsDEA)**：定义 vision tail token 为 RPV (Routing Probability Variance) 高于所有 vision tokens 平均 RPV 的 token（约 13%），对这些 tail tokens 激活更多 experts（Top-a, a > k），采用 renormalized softmax 权重，作为 data-augmentation 策略提升 tail tokens 的学习效果。
  - 实验比较：
    - **Vision-Language 任务**：vs MoE-LLaVA (4Top2, StableLM-1.6B / Phi2-2.7B)、Molmo (64Top8, OLMoE-1B-7B)
    - **Vision 任务**：vs GMoE (4Top2 / 6Top2, ViT-S/16)
    - **路由策略对比**：Task routing、Cluster routing、Instruct routing、Dynamic routing、STGC、Modality routing、Distribution routing、DeepSeekMoE
    - **Ablations**：MsDaR vs MsDaR+VsDEA 模块有效性、Vision Token Selection 策略（VHTs vs IATs vs 不同固定比例 VTTs）、策略互换（language+MsDaR vs vision+MsDaR）、跨 router 性能（Top-1/Top-2/Top-a/Dynamic）、load balancing 系数消减（0.01→0.001 vs 移除）
    - 结果：Vision-Language 平均提升 1.2%/2.1%，Vision 平均提升 1.6%，inference time 不显著增加

- 硬件平台是什么，配置是什么。
  - **V100-30G**：NVIDIA V100 30GB
  - **A800-80G**：NVIDIA A800 80GB
  - 训练配置：epoch=1, learning rate=2e-5, cosine schedule, weight decay=0.0, load balancing loss coefficient=0.01, text max length=2048, batch size per GPU=16, precision=FP16
  - MoE-LLaVA train steps=original（MoE-LLaVA 默认设定），Molmo train steps=2000
  - VsDEA 中 a=4（others, MoE-LLaVA 上）/ a=12（Molmo 上, others use Top8）

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - **MoE-LLaVA**：StableLM-1.6B / Phi2-2.7B 作为 LLM backbone，MoE 层替换 FFN，4 experts Top2 routing，CLIP visual encoder
    - **Molmo**：OLMoE-1B-7B backbone，64 experts Top8 routing
    - **GMoE**：ViT-S/16 backbone，4 或 6 experts Top2 routing
  - **Instruction-tuning 数据集**：
    - MoE-LLaVA 用：LLaVA (158K) + ShareGPT (40K) + VQAv2 (83K) + GQA (72K) + OKVQA (9K) + OCRVQA (80K) + A-OKVQA (66K) + TextCaps (22K) + RefCOCO (48K) + VG (86K)，合计 665K
    - Molmo 用：VQAv2 (440K) + TextVQA (35K) + OKVQA (9K) + ChartQA (28K) + DocVQA (39K) + InfographicVQA (24K) + AI2D (15K) + A-OKVQA (17K) + AndroidControl (300K) + ScienceQA (6K) + TabWMP (23K) + ST-VQA (25K) + TallyQA (250K)
    - 扩展数据集：Open-LLaVA-NeXT（额外 350K，总计 1021K）
  - **Benchmarks**：
    - Vision-Language：GQA, ScienceQA-IMG, TextVQA, POPE, MME, MMBench, MM-Vet（MoE-LLaVA 路径）；ChartQA, DocVQA, AI2D, VQAv2, AndroidControl, CountBenchQA（Molmo 路径）
    - Vision：PACS, VLCS, Office-Home（GMoE 路径）
    - High vision-token scenario：NLVR2（多图 VQA）
  - Training loss：Auto-regressive generative loss `L = -Σ log p(y_i | V, T, Y_{<i}, θ)`，仅计算生成文本部分

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确给出开源代码链接。基于 MoE-LLaVA（open source）、Molmo（open weights）、GMoE（open source）实现。
  - **LTDR 算法 Pipeline 伪代码**：
    ```
    # === 输入 ===
    # V ∈ R^{M×D}: vision tokens (M ≈ 576 from CLIP)
    # T ∈ R^{N×D}: language tokens
    # K = 4 (MoE-LLaVA) / 64 (Molmo): total experts
    # k = 2 (MoE-LLaVA) / 8 (Molmo): default activated experts
    # a = 4 (MoE-LLaVA) / 12 (Molmo): activated experts for tail tokens
    # W ∈ R^{D×K}: router weight matrix (trainable linear layer)

    # === Step 1: 标准 Router Forward ===
    f_v = V @ W           # [M, K] vision logits
    f_t = T @ W           # [N, K] language logits
    P_v = softmax(f_v)    # [M, K] routing probabilities
    P_t = softmax(f_t)    # [N, K] routing probabilities

    # === Step 2: MsDaR — Load Balancing（仅 language） ===
    for i in 1..K:
        F_i = count(T routed to expert i) / N    # language token fraction
        G_i = mean(P_t[:, i])                     # mean routing prob
    L_balancing = K * sum_{i=1..K} F_i * G_i
    # vision tokens 不参与 L_balancing 计算

    # === Step 3: VsDEA — RPV 计算与 Tail 分类 ===
    RPV_v = Variance(P_v, dim=1)              # [M], per-token variance
    threshold = Mean(RPV_v)                    # scalar, ~13th percentile
    is_tail = (RPV_v > threshold)              # [M] boolean mask

    # === Step 4: VsDEA — 差异化 Expert Activation ===
    # Vision tail tokens: activate a experts (a > k)
    for v_i in V where is_tail[i]:
        indices = TopK(f_v[i], a)              # select top-a experts
        w = softmax(f_v[i][indices])            # renormalized weights
        output = sum_{j=1..a} w_j * Expert_j(v_i)

    # All other tokens (vision head + all language): activate k experts
    for x in [V_head, T]:
        indices = TopK(f(x), k)
        w = softmax(f(x)[indices])
        output = sum_{j=1..k} w_j * Expert_j(x)
    ```
  - **关键数学关系**：
    - RPV: `Var(P(v)) = (1/K) Σ (P(v)_i - μ)²`，其中 μ = (1/K) Σ P(v)_i
    - Vision head/tail 阈值: `Mean(RPV(V)) = (1/M) Σ RPV(v_i)`，自适应动态阈值
    - 实验测得 vision tail tokens 约占 13%，head tokens 约占 87%
  - **Inference 性能**：all-to-all 通信速度由最慢 expert 决定，VsDEA 激活更多 experts 但不显著增加最慢 expert 负载，因此 inference time 几乎不变（V100 上 avg 1100s vs 1108s baseline，A800 上 846s vs 917s）
