## Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

- 属于算法pipeline的实现是什么？实验比较什么？
  - **MoE Experts Compression Suite (MC-Suite)**：从四个维度（权重、推理行为、激活、梯度）提出 16 种专家重要性评估准则，用于识别可安全丢弃的冗余专家：
    1. **Weight-Guided（4种）**：Expert Weight Similarity (EWS) — 计算专家权重的 pairwise cosine similarity；Router Weight Norm (RWN) — 路由矩阵中对每个专家的 l2-norm；Expert Weight Stable Rank (WSR) — 权重矩阵的 stable rank = Σσ_i²/σ_1²；Expert Weight Norm (EWN) — 专家权重的 l2-norm。
    2. **Inference-Guided（4种）**：Expert Usage Frequency (EUF) — 专家被多少 token 激活的比例；Expert-Expert Collaboration (ECC) — 两专家共同被路由到同一 token 的次数；Expert Vocabulary Coverage (EVTC) — 专家处理的唯一 token 占词表比例；Expert Input Token Similarity (ETS) — 跨专家输入 token 的重叠数。
    3. **Activation-Guided（4种）**：Expert Activation Similarity (EAS) — 专家激活的 pairwise cosine similarity；Expert Activation Entropy (EAE) — H(A_Ep) ∝ Σ_j log[σ(A_Ep^j)]，各隐藏维度标准差对数之和；Expert Activation Distribution Outliers (EAO) — μ ± 3σ 之外的激活异常值计数；Expert Activation Norm (EAN) — 累积激活的 l2-norm。
    4. **Gradient-Guided（4种）**：Expert Gradient Similarity (EGS) — 专家梯度的 pairwise cosine similarity；Expert Gradient Entropy (EGE) — H(W_Ep^g) ∝ Σ_i log[σ(W_Ep^{g^j})]；Expert Gradient Outliers (EGO) — 梯度异常值计数；Expert Gradient Norm (EGN) — 梯度的 l2-norm。
  - **MoE Lottery Subnetworks**：提出迭代 estimate-prune-finetune 三阶段流程，替代传统 one-shot pruning：
    1. 对每个 MoE layer，使用 MC-Suite 准则估算专家重要性
    2. 每轮丢弃 s/k% 的专家（k 轮总丢弃 s%），每轮丢弃后重新估算剩余专家重要性
    3. 丢弃后使用 task-agnostic budget finetuning（next-token prediction on C4）校正子网络的次优状态
    4. 仅需 ~1M training tokens 即可饱和 finetuning 收益
  - 实验比较：
    - **Baselines**：Random Dropping (one-shot, iterative, w/ MoE Lottery)、prior expert pruning methods (Lu et al., 2024; Muzio et al., 2024)
    - **LLM Weight Pruning 对比**：Random Pruning, Magnitude Pruning, Wanda (Sun et al., 2023) 在 2:4 structured sparsity 下
    - **消融实验**：One-shot vs Iterative vs MoE Lottery 三种剪枝策略对比
    - **Sparsity ratios**：12.5%, 25.0%, 37.5%, 50.0%, 62.5%, 75.0%
    - **评估指标**：C4/Wikitext Perplexity、MMLU accuracy、ARC-c accuracy、ARC-e accuracy、HellaSwag accuracy、WinoGrande accuracy、BoolQ accuracy、CommonsenseQA accuracy
    - **Instruction-following 恢复实验**：zero-shot → k-shot examples → SFT（supervised fine-tuning with instruction-tuning dataset）
    - **关键发现**：(1) Min-EAN (最小激活范数) 和 Min-EGE (最小梯度熵) 是最优准则，50% sparsity 下 perplexity 从 15.21 (Random One-shot) 降至 9.99 (Min-EAN MoE Lottery)；(2) MoE Lottery 在 ≥50% sparsity 下仍保持 robust 性能（≥1.27× speedup, ≤0.55× memory）；(3) Expert dropping 主要损害 instruction-following 能力，可通过 k-shot 或 SFT 恢复；(4) 迭代剪枝与 one-shot 剪枝选出的专家高度不一致（Figure 5a），而 MoE Lottery 与迭代剪枝选出专家高度一致（Figure 5b）；(5) Expert dropping 在 Base model 上执行优于在 Instruct model 上执行。

- 硬件平台是什么，配置是什么。
  - **8×NVIDIA A100 GPU**（论文 Appendix C 明确说明："With the availability of 8×A100, we use a batch size of 8"）
  - 使用 HuggingFace Transformers 加载 Mixtral-8×7B checkpoint

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Mixtral-8×7B（Base 和 Instruct 两个版本），32 层 MoE，每层 8 个 experts，top-2 routing，总参数 ~46.7B，激活参数 ~12.9B，float32 下 180GB 内存（激活 28GB/token）
  - **Calibration 数据集**：C4 validation set（256 samples, max_seq_len=2048），用于 MC-Suite 准则估计和 task-agnostic finetuning
  - **Evaluation Benchmarks**：MMLU（14042 test samples）、ARC-Challenge（ARC-c）、ARC-Easy（ARC-e）、HellaSwag、WinoGrande（1267 test samples）、BoolQ（3270 test samples）、CommonsenseQA（1221 test samples）
  - **Finetuning**：AdamW optimizer, cosine LR scheduler, max LR=1e-6, batch size=4~8，具体超参见 Table 6

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源状态**：截至 2025/05，论文未发布官方代码（Papers with Code 显示 "No code implementations yet"）。arXiv: 2504.05586。作者来自 UT Austin、Apple、UNC Chapel Hill。
  - **算法 Pipeline 伪代码**：

    ```
    # ========== MC-Suite: Expert Importance Estimation ==========
    # 给定 MoE 模型 M，层 l，n 个专家 E = {E_1,...,E_n}，router G with W_G^{d×n}
    # Calibration 数据集 X_calib

    # 示例准则 1：Minimum Activation Norm (Min-EAN)
    def estimate_activation_norm(M, l, X_calib):
        for each expert E_p in layer l:
            A_Ep = []  # accumulated activations
            for batch in X_calib:
                # register forward hook on E_p output
                a_Ep = forward_hook(M, layer=l, expert=p, batch)
                A_Ep.append(a_Ep)
            A_Ep = concat(A_Ep, dim=0)
            score[p] = sum(norm_l2(A_Ep, dim=0))
        drop_idx = argmin(score)  # 最小范数→最可丢弃

    # 示例准则 2：Minimum Gradient Entropy (Min-EGE)
    def estimate_gradient_entropy(M, l, X_calib):
        for batch in X_calib:
            loss = next_token_prediction(M(batch), batch_labels)
            loss.backward()  # 累积梯度
        for each expert E_p in layer l:
            W_grad_p = W_Ep.grad  # 形状与权重相同
            # H ∝ Σ_j log[σ(W_grad_p^j)]
            stds = [std(W_grad_p[j, :]) for j in range(d_hidden)]
            score[p] = sum(log(s) for s in stds if s > 0)
        drop_idx = argmin(score)  # 最小梯度熵→最可丢弃
    ```

    ```
    # ========== MoE Lottery Subnetworks ==========
    # 输入：full MoE model M, target sparsity s, k rounds, MC-Suite criterion c

    def moe_lottery_pruning(M, s, k, c, X_calib):
        drop_per_round = s / k  # e.g., 50% in 4 rounds → 12.5%/round
        for round in range(k):
            # Step 1: Estimate importance using criterion c
            for each MoE layer l in M:
                scores = estimate_criterion(c, M, l, X_calib)
                # 每个 layer 均匀丢弃 expert (per-layer uniform)
                n_drop = int(n_experts * drop_per_round)
                drop_experts[l] = argsort(scores)[:n_drop]

            # Step 2: Prune experts (delete from router + weights)
            for each MoE layer l in M:
                W_G_l^{d×n} → W_G_l^{d×(n-n_drop)}  # 从 router 删除对应列
                remove expert weights from memory

            # Step 3: Task-agnostic budget finetuning
            tokens = 0.2M * (2^round)  # progressive schedule
            for batch in X_calib:
                loss = next_token_prediction(M(batch), batch_labels)
                optimizer.step()
                if tokens_processed >= tokens: break
        return M  # MoE lottery subnetwork
    ```

  - **张量计算示例（Min-EAN 准则, Mixtral-8×7B, layer l）**：
    - 输入：X_calib 经 layer l-1 后的 hidden states H^{t×d}（t tokens, d=4096）
    - Router: G(H) = softmax(H @ W_G^{4096×8}) → top-2 routing → 每个 expert E_p 获得 tokens 子集 X_p^{t_p×4096}
    - Expert forward: A_p = SiLU(X_p @ W_{gate}^{4096×14336}) * (X_p @ W_{up}^{4096×14336}) @ W_{down}^{14336×4096}（标准 SwiGLU FFN）
    - Expert Activation Norm: score[p] = ||A_p||_2 = sqrt(Σ_{i=1}^{t_p} Σ_{j=1}^{4096} (A_p[i,j])²)
    - 选择 score 最小的 expert 丢弃 → 从 router 删除 W_G[:,p] → W_G^{4096×7}
    - 50% sparsity 时（4/8 experts 丢弃）: memory 180GB→99GB, speedup 1.27×

  - **关键结论张量化**：
    - Perplexity trend（w/ MoE Lottery）: Random 75% sparsity→33.05, Min-RWN→17.26, Min-ETS→16.03, Min-EGE→15.08, Min-EAN→14.02（Table 1, Mixtral Instruct）
    - Expert dropping 对 instruction-following 的影响：zero-shot MMLU@50% sparsity 18.91(one-shot)→40.79(MoE Lottery); 加 k-shot 后可接近 full-MoE baseline
    - 迭代 vs one-shot 专家选择差异：Figure 5a 显示 Dark pink（不一致）大面积存在，表明 one-shot 与迭代选出的子网络完全不同
