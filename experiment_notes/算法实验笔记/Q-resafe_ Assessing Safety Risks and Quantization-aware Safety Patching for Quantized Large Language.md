## Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Q-resafe——量化感知的安全修补框架，在已量化的 LLM 上通过三个关键步骤恢复安全能力：(1) **安全修补数据集构建**：使用预量化全精度 LLM 作为教师，对校准数据集的每个 prompt x，生成 preferred response y_w ~ π_W(·|x)（全精度模型）和 dispreferred response y_l ~ π_Q⁰(·|x)（量化后模型），构建 DPO 偏好三元组 (x, y_w, y_l)，无需人工标注；(2) **DPO 对齐损失**：L = -E_{(x,y_w,y_l)} log σ(β log(π_Q(y_w|x)/π_Q⁰(y_w|x)) - β log(π_Q(y_l|x)/π_Q⁰(y_l|x)))，以量化模型 π_Q⁰ 为参考模型；(3) **周期性安全关键权重识别与选择性更新**：每 K 次迭代使用 SNIP score I(W_ij, x) = |W_ij · ∇_{Q_ij} L(x)|，对校准集 D_calib 取均值得到 SafeScore(Q)，选择 top-τ 百分比的权重作为安全关键权重，通过掩码矩阵 M_Q 只更新这些权重，其余权重保持不变，更新约束为 Q = Q⁰ + Quant(M_Q ⊙ AB)（LoRA 低秩分解 + 再量化）。算法 1 给出了完整流程。
  - 实验比较（安全风险评估）：对四种代表性量化方法（AWQ (PTQ w/o FT)、AQLM (PTQ w/ FT)、LLM-QAT (QAT w/ FT)、QLoRA (QAT w/ LoRA FT)）在 INT4/INT8/3-bit/2-bit 下，使用三种风险等级校准数据集（Risk-I: UltraChat benign、Risk-II: AOA indirectly harmful、Risk-III: AdvBench directly harmful），评估 ASR↓、MT-Bench↑、AlpacaEval↑。
  - 实验比较（安全修补）：Q-resafe vs 各 baseline 量化方法在三种风险数据集上的安全恢复效果，包含消融实验（τ 从 0.0 到 1.0 的安全关键权重比例、SFT vs DPO vs Q-resafe 的对比、2-bit 到 8-bit 的 bit-width 影响、LLM.int8()/NF4/FP4 的方法泛化性、不同解码策略 τ/top-k/top-p 下的鲁棒性）。

- 硬件平台是什么，配置是什么。
  - 4 × NVIDIA A100 40GB GPU。框架：PyTorch + HuggingFace Transformers。预训练模型权重从 HuggingFace Hub 获取。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-2-7B-Chat（安全对齐对话强）、Gemma-7B-Instruct（结构化任务强），均为经过 instruction tuning 和 RLHF 的安全对齐模型。
  - 校准数据集：AdvBench（520 条有害指令，覆盖亵渎、威胁、错误信息等）、UltraChat（大规模多域安全对话）、AOA（绝对服从 agent 提示 + 10 条 AdvBench 样本构造的间接有害数据集）、Alpaca-cleaned（用于安全关键权重识别的消融实验）。
  - 安全评估 Bench：ASR（Attack Success Rate，响应对有害指令的攻击成功率↓），辅助指标包括 HarmBench 分类器和 Harmfulness Score (1-5 GPT-4 评分)。编码策略攻击评估用 ASR_Decoding（变 temperature/top-k/top-p 采样）。
  - 效用评估 Bench：MT-Bench（160 题，8 领域，双轮对话，GPT-4 评分 1-10）、AlpacaEval（805 题，单轮，GPT-4 对比 text-davinci-003 的 win rate）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：项目主页 https://thecommonirin.github.io/Qresafe/，代码仓库 https://github.com/Thecommonirin/Qresafe。论文声明发布所有评估模型和修改后的 Q-Resafe benchmark。
  - **算法 pipeline 伪代码（对应 Algorithm 1）**：
    1. 构建安全修补数据集 D_patch：
       for each prompt x in D_calib:
           y_w ~ π_W(·|x)   # 全精度模型生成 preferred response
           y_l ~ π_Q⁰(·|x)  # 量化模型生成 dispreferred response
           D_patch ← D_patch ∪ {(x, y_w, y_l)}
    2. 初始化：量化权重 Q⁰ (来自 AWQ/AQLM/LLM-QAT/QLoRA)，LoRA 矩阵 A∈R^{d_in×r}, B∈R^{r×d_out}（r=128），超参数 τ=0.6, K=1000, η=5e-6, β=0.01。
    3. for t = 0 to T-1:
         if t % K == 0:  # 周期性重新识别安全关键权重
           对于每层权重矩阵 W，计算 SNIP score:
             SafeScore(Q^t) = E_{x∈D_calib} |W_ij · ∇_{Q_ij} (-log p(y|x))|
           M_Q = 1[SafeScore(Q^t) ∈ Top-τ]  # 选 top-τ% 权重
           (M_A, M_B) = MapMask(M_Q)        # 映射到 LoRA 维度的掩码
         # 选择性 SGD 更新（仅更新安全关键权重对应的 LoRA 行/列）
         A^{t+1} = M_A ⊙ (A^t - η∇_A L_DPO) + (1-M_A) ⊙ A^t
         B^{t+1} = M_B ⊙ (B^t - η∇_B L_DPO) + (1-M_B) ⊙ B^t
         Q^{t+1} = Q⁰ + Quant(A^{t+1} B^{t+1})  # LoRA 更新后再量化到同精度
    4. 输出：安全修补后的量化 LLM π_{Q^T}
  - **张量计算示例**：对 Llama-2-7B-Chat 的某层权重 W∈R^{4096×4096}，INT4 量化后 Q⁰∈Q^{4096×4096}，LoRA r=128 时 A∈R^{4096×128}, B∈R^{128×4096}，参数量仅为全量微调的 6.25%。SNIP 计算：对校准 batch 的每个 token 计算交叉熵损失 L，反向传播得 ∇_Q L，逐元素乘 |Q_ij · ∇_{Q_ij} L|，跨 batch 求均值排序，取 top-60%(τ=0.6) 的权重索引生成 M_Q∈{0,1}^{4096×4096}。MapMask 将 M_Q 中有 1 的行/列映射为 M_A∈{0,1}^{4096×128}（对应行有 1 则整行标记）和 M_B∈{0,1}^{128×4096}（对应列有 1 则整列标记）。更新时非掩码位置的 LoRA 参数保持为零初始化不变。
