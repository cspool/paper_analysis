## MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Roster of Experts (RoE)，一种无需训练的 MoE 推理算法。核心为三个组件：
    1. **Gumbel-Top-K 随机路由**：在标准 MoE router logits $\mathbf{R} \in \mathbb{R}^E$ 上添加 Gumbel 噪声后做 TopK 选择——$\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$，其中 $\mathbf{G}$ 为 Gumbel(0,1) i.i.d. 采样向量，$\tau$ 为逐层温度超参数。当 $\tau=0$ 时退化为标准确定性路由；$\tau>0$ 时产生受控随机性，等价于从 router 隐含的 categorical 分布中无放回采样 k 个 expert。
    2. **多路径聚合**：对每个 token 生成 n 个候选输出 logits（n 次独立的 Gumbel-Top-K 采样），通过概率平均（probability averaging）聚合成最终预测 logits。
    3. **Clean Cache 策略**：将 n 个样本的 forward pass 合并为一个 batch，batch 中第一个样本（index 0）使用 $\tau=0$ 确定性路由产生"clean path"，其 KV-cache 被其余所有样本共享。由此 KV-cache 内存开销与单样本完全相同。
  - 实验比较：(1) RoE vs baseline standard MoE greedy decoding 在三类模型的 12 个 benchmark 上的准确率（exact match 和 pass@1）；(2) 计算开销分析：不同样本数 K 下的 GPU 内存、功耗、延迟增长；(3) 效率对比：RoE with K=32 的 OLMoE-7B vs 等价的 10.5B 标准 MoE，对比延迟和内存。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB（单卡）。RoE 计算开销实验使用单卡 A100 跑 GSM8K 前 100 题。
  - 超参数搜索框架：Optuna（TPE，Tree-structured Parzen Estimator）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：OLMoE-1B-7B-Instruct（7B total, 1B active）、Mixtral-8x7B-Instruct-v0.1（47B total, ~13B active）、GPT-OSS-20B。
  - 数学推理 benchmark：GSM8K, SVAMP, AddSub, SingleEQ, MultiArith。
  - 常识推理 benchmark：ARC-Easy, ARC-Challenge, OpenBookQA, Social-I-QA, Hellaswag。
  - 代码生成 benchmark：HumanEval, HumanEvalPlus（pass@1）。
  - 效率分析数据集：WikiText-103（perplexity 评估 equivalent model size）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供官方开源代码仓库。作者来自 Apple 和 UCSD，arXiv:2509.17238。
  - 算法 pipeline 伪代码（单 token 生成步骤）：

```
# ===== 单 Token 生成: RoE Forward =====
# 输入: hidden_state h (当前 token), KV-cache (共享)
# 参数: 逐层温度 tau[l] (经 Optuna TPE 调优), 样本数 n
# 输出: 下一个 token 的 logits

def roe_forward(h, kv_cache, tau, n, model):
    all_logits = []

    # Step 1: 将 n 个样本的 forward 打包为 batch
    batch_h = h.repeat(n, 1)  # (n, d_model)

    for layer in model.layers:
        # Step 2: Attention (共享 KV-cache)
        # Sample 0 使用 tau[l]=0 (clean path); 其余样本使用相同的 KV-cache
        if is_first_sample:
            out, new_kv = attention(batch_h[0], kv_cache)  # 单次计算
            kv_cache = new_kv  # 更新 clean KV-cache

        # Step 3: MoE Layer with Gumbel-Top-K Routing
        for each MoE layer l in model.moe_layers:
            router_logits = W_router[l] @ batch_h  # (n, E)

            # Gumbel noise for diversity
            for i in range(n):
                if i == 0 and use_clean_cache:
                    tau_eff = 0.0   # Clean path: deterministic
                else:
                    tau_eff = tau[l]  # Temperature from TPE tuning
                G = sample_gumbel(E)
                noisy_logits = router_logits[i] + tau_eff * G

                # Top-K 选择
                topk_vals, topk_idx = topk(softmax(noisy_logits), k)
                expert_out = 0
                for idx, weight in zip(topk_idx, topk_vals):
                    expert_out += weight * expert_ffn[idx](batch_h[i])
                batch_h[i] += expert_out

    # Step 4: Logit 聚合 (probability averaging)
    for i in range(n):
        all_logits.append(model.lm_head(batch_h[i]))
    final_logits = mean(softmax(all_logits), dim=0)  # 概率平均

    return final_logits, kv_cache

# ===== 温度搜索: Optuna TPE =====
# 搜索空间: 每 MoE 层一个 tau_i ∈ [0, 0.5]
# 中间层 (skip first/last k layers) 参与搜索
# 优化目标: validation perplexity (数学任务) 或 validation accuracy (常识/代码任务)

def search_temperature(model, val_data, task_type):
    def objective(trial):
        tau = []
        for l in range(L_moe):
            if l < skip_first or l >= L_moe - skip_last:
                tau.append(0.0)  # 首尾层固定为 0
            else:
                tau.append(trial.suggest_float(f"tau_{l}", 0.0, 0.5))
        score = evaluate_roe(model, val_data, tau)
        return score

    study = optuna.create_study(
        direction="minimize" if task_type == "math" else "maximize",
        sampler=TPESampler()
    )
    study.optimize(objective, n_trials=50)
    return study.best_params
```

  - 关键张量维度与计算：
    - Router logits: $\mathbf{R} \in \mathbb{R}^{E}$ (E 为 expert 数)
    - Gumbel 噪声: $\mathbf{G} \sim \text{Gumbel}(0,1)$，即 $G_i = -\log(-\log(U_i)), U_i \sim \text{Uniform}(0,1)$
    - 温度控制方程: $\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$
    - 当 $\tau=0$ 时退化为 Standard Top-K；当 $\tau$ 中等时保留高 logit expert 被选中的优势（Gumbel-Max 性质）
    - Clean Cache: batch[0] 使用 $\tau=0$ 产生共享 KV-cache，batch[1:] 使用 TPE 调优的 $\tau_l$
    - 最终 logits: $\text{logits}_{\text{final}} = \text{softmax}^{-1}(\frac{1}{n}\sum_{i=1}^{n} \text{softmax}(\text{logits}_i))$
  - 超参数配置（Table 1）:
    - 数学任务：OLMoE N=32/T=0.5/L=1, Mixtral N=64/T=0.25/L=5, GPT-OSS N=64/T=0.2/L=5，PPL 优化
    - 常识任务：OLMoE N=32/T=0.5/L=3, Mixtral N=64/T=0.3/L=3, GPT-OSS N=64/T=0.2/L=5，Accuracy 优化
    - 代码任务：OLMoE N=32/T=0.5/L=1, Mixtral N=64/T=0.25/L=5, GPT-OSS N=64/T=0.2/L=5，Accuracy 优化
