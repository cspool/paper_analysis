## CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - **实现**：将 Top-K 稀疏门控 MoE 块通过 Co-Upcycling 策略集成到多模态 LLM 的 **CLIP 视觉编码器**（ViT-L）和 **MLP 连接器**（两层线性 MLP）中。具体为：(1) 将 MLP 连接器的单个 MLP 块替换为 Top-2-in-4 稀疏 MoE 块；(2) 将 CLIP ViT 每个 transformer encoder 层的 MLP 块替换为 Top-2-in-4 稀疏 MoE 块；(3) 每个 MoE 专家的权重从预训练/预微调后的同位置 MLP 块初始化（Co-Upcycling）；(4) 三阶段训练：MLP 连接器预训练 → 全参数预微调（ALLaVA 标注数据）→ 含 MoE 块的视觉指令微调；(5) 辅助损失：负载均衡损失 L_b（α_b=0.1）+ Router z-loss L_z（α_z=0.01）。
  - **实验比较**：在 VQA（VQAv2, GQA, ScienceQA-IMG, TextVQA）和指令跟随（POPE, MME, MMBench, SEED-Bench, LLaVA-Wild, MM-Vet, MMMU, MathVista）benchmark 上与各尺寸组的 SOTA 多模态 LLM 对比（7B/13B/7B-MoE），并消融 MLP-MoE、CLIP-MoE、LLM-MoE（upcycled vs pre-trained Mixtral）、多分辨率输入、预微调阶段。

- 硬件平台是什么，配置是什么。
  - **预训练阶段**：8×A100 GPU，每 GPU batch size 32，ZeRO-2
  - **预微调阶段**：16×A100 GPU，每 GPU batch size 8，ZeRO-3
  - **视觉指令微调阶段**：32×A100 GPU，每 GPU batch size 8，总 batch size 256，ZeRO-3-offload
  - **学习率**：预训练 1e-3 → 预微调 2e-6 → 指令微调 4e-6（最终模型），Cosine 调度
  - **优化器**：AdamW

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Baseline = CLIP ViT-L（视觉编码器）+ 两层 MLP（连接器）+ Mistral-7B / Mixtral-8×7B（LLM）。CuMo 变体：
    - CuMo Mistral-7B：激活参数 7.80B，总参数 8.26B
    - CuMo Mixtral-8×7B：激活参数 13.45B，总参数 47.71B
    - CLIP-MoE 激活参数 0.50B（总 0.91B），MLP-MoE 激活参数 0.05B（总 0.10B）
  - **预训练数据**：LLaVA-558K
  - **预微调数据**：ALLaVA-Caption 708K（高质量图像标注）
  - **指令微调数据**：LLaVA-665K + ShareGPT4V 102K + LAION-GPT-V 11K + DocVQA 10K + SynDog-EN 50K + ChartQA 4K + DVQA 50K + AI2D 2K + InfoVQA 4K + ALLaVA 708K + LIMA 1K + ALLaVA-Text 143K，总计约 1.65M（全部开源）
  - **Benchmark**：ScienceQA-IMG, TextVQA, GQA, POPE, MME, MMBench(EN/CN), MM-Vet, VQAv2, LLaVA-Wild, SEED-IMG, MMMU(val), MathVista

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：代码 Apache 2.0，模型权重 CC BY-NC 4.0。GitHub: https://github.com/SHI-Labs/CuMo
  - **算法 Pipeline 伪代码**（以 CLIP 中一层 MoE 为例）：

    ```
    # 输入: X ∈ R^{N×C_in}（N 个 visual tokens, C_in 通道数）
    # 稀疏 MoE 块替换标准 MLP 的计算流程:

    # === Step 1: Router 计算专家权重 ===
    W = Softmax(Linear(X))           # W ∈ R^{N×S}, S 个 experts

    # === Step 2: Top-K 选择 ===
    W_K_indices, W_K_values = TopK(W, K)   # 选 K 个最高分 expert
    W_K = Softmax(W_K_values)              # ∈ R^{N×K}

    # === Step 3: 每个 token 仅通过选中的 K 个 expert ===
    X_out = zeros(N, C_out)
    for i in 1..K:
        expert_idx = W_K_indices[:, i]          # 每个 token 的第 i 个选中 expert
        tokens_i = X[expert_idx]                 # 路由到该 expert 的 tokens
        expert_out = MLP_expert[t](tokens_i)     # 通过对应 expert MLP
        X_out += W_K[:, i:i+1] * expert_out      # 加权累加

    # === Step 4: Co-Upcycling 初始化 ===
    # 每个 expert MLP 权重 = 同位置预训练/预微调阶段 MLP 权重
    for t in 1..S:
        MLP_expert[t].weight = MLP_pretrained[t].weight

    # === Step 5: 辅助损失 ===
    L_total = L_ce + 0.1 * L_balance + 0.01 * L_z_loss
    ```
  - **使用方式**：`pip install -e ".[train]"` 安装后，训练用 `python -m cumo.serve`；推理 CLI: `python -m cumo.serve.cli --model-path checkpoints/CuMo-mistral-7b --image-file <img>`，支持 4-bit/8-bit 量化。框架基于 PyTorch + LLaVA + flash-attn。
