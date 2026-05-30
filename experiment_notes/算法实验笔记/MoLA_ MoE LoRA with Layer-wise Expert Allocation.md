## MoLA: MoE LoRA with Layer-wise Expert Allocation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoLA 提出一种层级别专家分配的 MoE-LoRA 参数高效微调方法。在 Transformer 的每层中，将 LoRA 适配器作为 MoE expert（即每层有多个低秩矩阵对 {A_i, B_i}），通过 router 进行 top-K 路由选择。关键创新是为不同 Transformer 层分配不同数量的 LoRA expert，而非传统 MoE 中各层 expert 数量相同。具体公式：S_i^{jt}(x) = TopK(Softmax(W_r^{jt}x), K)_i / Σ TopK(Softmax(W_r^{jt}x), K)_i，h^{jt} = W_0^{jt}x + Σ_{i=1}^K S_i^{jt}(x) A_i^{jt} B_i^{jt} x。对每个 dense weight matrix（attention 的 W_q/W_k/W_v/W_o 和 MLP 的 W_gate/W_down/W_up）都应用 LoRA expert。提出五种层级别配置：MoLA-△(8642, triangle，底层多 expert)、MoLA-▽(2468, inverted-triangle，高层多 expert)、MoLA-▷◁(8228, hourglass，底层和高层多)、MoLA-✸(2882, diamond，中层多)、MoLA-□(5555, rectangle，各层相同)。
  - 实验比较：（1）与 baseline PEFT 方法（Prompt Tuning、LLaMA-Adapter、LoRA rank=64）和 Full-Parameter Fine-tuning 在 6 个 benchmark 上的精度对比；（2）五种 MoLA 层级别配置之间的对比（每种总 expert 数相同，仅分配方式不同，总 config sum 为 20 或 16）；（3）极端配置（10-2-2-2 / 2-10-2-2 / 2-2-10-2 / 2-2-2-10）分析各层段 expert 冗余度；（4）Transfer Learning：instruction tuning → downstream fine-tuning；（5）Continuous Learning：跨 5 个 ScienceQA 领域连续学习，用 OP 和 PD 指标评估；（6）Frobenius Norm 分析各层 expert 相似度以量化冗余；（7）PiSSA 初始化方法的兼容性实验。

- 硬件平台是什么，配置是什么。
  - GPU: 8× NVIDIA A100-40G + 3× NVIDIA A6000
  - 训练时间: COLA 数据集约 4 小时
  - 精度: 论文未明确说明（推断为 Hugging Face Transformers 默认混合精度设置）
  - 分布式框架: 论文未明确说明（基于 Hugging Face Transformers + PyTorch 训练）

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA-2-7B（32 层）、Mistral-7B（32 层）、Gemma（28 层，Appendix E）
  - NLP 数据集：MRPC（5801 句对，二分类）、RTE（2490 train / 277 val，二分类）、COLA（8551 train / 1043 val，二分类）
  - Commonsense QA 数据集：ScienceQA（6508 train / 2224 test text-only）、CommonsenseQA（9740 train / 1221 val）、OpenbookQA（4957 train / 500 val / 500 test）
  - Instruction Tuning 数据：OpenOrca 随机采样 50,000 条

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/GCYZSL/MoLA（179 stars，5 commits）
  - 开源内容：训练脚本（mola_training.py、mola_training_instruction.py）、评估脚本（evaluation_scienceqa.py）、推理 notebook、数据预处理脚本。基于 Hugging Face Transformers、mm-cot、alpaca-lora 构建。
  - 伪代码示意（MoLA 单层前向传播，对每个 linear module 执行）：
    ```
    # 超参数: rank=8, top_k=2, N_j=该层expert数
    # 对于第j层，预训练权重W_0冻结

    # 1. 原始预训练前向
    h_base = W_0 @ x  # [d_q, d_p] @ [B, L, d_p] -> [B, L, d_q]

    # 2. Router 计算每个 expert 的选择概率
    router_logits = W_r @ x  # W_r: [d_q, N_j], -> [B, L, N_j]
    router_probs = Softmax(router_logits, dim=-1)  # [B, L, N_j]

    # 3. Top-K 选择
    topk_vals, topk_idx = TopK(router_probs, K=2)  # 各 [B, L, K]
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # 归一化

    # 4. 每个 expert 计算 LoRA delta
    h_expert = 0
    for i in range(N_j):
        # A_i: [d_q, r], B_i: [r, d_p], r=8
        if i in topk_idx:
            delta_i = A_i @ B_i @ x  # [d_q, r] @ [r, d_p] @ [B,L,d_p] -> [B,L,d_q]
            weight_i = topk_vals[topk_idx == i]
            h_expert += weight_i * delta_i

    # 5. 输出 = 预训练 + LoRA expert 组合
    h = h_base + h_expert  # [B, L, d_q]

    # 6. Load balancing loss（每层计算）
    # f_i = 1/T Σ_t Indicator(token_t选择expert_i)
    # P_i = 1/T Σ_t router_probs[t][i]
    # L_balance = N_j * Σ_i f_i * P_i
    ```
  - 关键超参数：LoRA rank=8, top-K=2, LoRA alpha=16, LoRA dropout=0.05, optimizer=AdamW, lr=3e-4, batch_size=128, cutoff_length=256, epochs={10,15,20}, seed=10。可训练参数量（config sum=20, 即 5555 等配置）：105,635,840（LLaMA-2-7B 总参数的 ~1.5%）。
