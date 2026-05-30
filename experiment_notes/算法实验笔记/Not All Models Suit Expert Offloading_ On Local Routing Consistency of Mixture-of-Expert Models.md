## Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出两种量化 MoE 模型局部路由一致性（local routing consistency）的指标：（1）**Segment Routing Best Performance (SRP)**——基于 segment 的路由器对原始 token-level 路由器决策的逼近上限 F1 分数。对 single expert，定义激活序列为 binary classification tasks，segment estimator 在长度为 m 的 segment 内统一预测（全激活或全不激活），对所有 segment 计算 F1，当且仅当对所有 activation frequency ≥ α_e^m 的 segment 给出"激活"预测时 F1 最大化。对 expert group（layer 或 model level），用 segment router 预测 group 内所有 expert 的激活情况，同样求最大 F1。辅助指标 ρ̂（segment routing size ratio）衡量为达最佳 F1 所需激活 expert 数与原有激活数的比值。（2）**Segment Cache Best Hit Rate (SCH)**——模拟 oracle segment cache：缓存上限为 ρ·k（k 为原始每 token 激活 expert 数），驱逐未来 m 个 token 中激活次数最少的 expert，SCH 为其 hit rate，桥接 SRP 与实际 expert offloading 系统。
  - 实验比较：(1) 20 个 MoE LLM（3B-57B 参数）在不同 segment 长度 m（4/16/64/256）下的 SRP 和 ρ̂，按 SRP 将模型分为 4 组；(2) 11 个 TOY 模型（基于 OLMoE 修改，~1.43B 参数）验证 load balance、shared experts、expert combination space 等因素对 SRP 的影响；(3) 领域级（11 domains）SRP 与 expert specialization 分析（domain/prediction/vocabulary specialization）；(4) SCH 与实际 cache 算法（LRU、LFU）hit rate 的相关性（Pearson correlation），以及 SCH vs. 最优 Belady cache 的相对差距；(5) base vs. post-trained 模型 SRP 一致性；(6) layer-wise、position-wise、per-expert SRP 细粒度分析。

- 硬件平台是什么，配置是什么。
  - REAL 模型：NVIDIA A100 PCIe 80GB GPU（用于 router decisions 收集和 offloading throughput benchmark）。TOY 模型训练：基于 Megatron-DeepSpeed 框架，使用 OLMoE 预训练代码，序列长度 4096，全局 batch size 1024（~4M tokens/batch），10000 steps（约 40B tokens），learning rate cosine decay from 4×10⁻⁴ to 5×10⁻⁵，bfloat16 混合精度。

- 模型是什么。数据集和bench分别是什么。
  - REAL 模型（20 个，3B-57B total params）：LLaMA-MoE-v2（3.80B/8.03B act/total）、Yuan2.0-M32（3.70B/39.94B）、PowerMoE-3B（0.88B/3.30B）、Qwen3-30B-A3B（3.35B/30.53B）、Phi-3.5-MoE（6.64B/41.87B）、OLMoE-1B-7B（1.28B/6.92B）、GRIN-MoE（6.64B/41.87B）、Mixtral-8x7B（12.88B/46.70B）、MiniCPM-MoE-8x2B（4.32B/13.87B）、JetMoE-8B（2.33B/8.52B）、LLaMA-MoE-v1-3.5B（3.50B/6.74B）、XVERSE-MoE-A4.2B（4.23B/25.78B）、Jamba-Mini-1.6（12.11B/51.57B）、DeepSeek-V2-Lite（2.66B/15.71B）、DeepSeekMoE（2.83B/16.38B）、Qwen2-57B-A14B（14.25B/57.41B）、NLLB-MoE-54B（3.75B/54.50B）、Qwen1.5-MoE-A2.7B（2.69B/14.32B）、OpenMoE-8B（3.80B/11.86B）、SwitchTransformers-Base-128（0.22B/7.42B）。
  - TOY 模型（11 个，~1.43B total params，从 OLMoE 配置修改）：Baseline（8 layers, hidden=1280, 64 experts activate 8）、FewerExp（32 experts, activate 4）、ActMore/ActFewer（activate 16/2）、1ShrExp/2ShrExp（1 or 2 shared experts）、DenseFst/DenseHlf（第 1 层或第 1/3/5/7 层替换为 dense MLP）、NoLB（load balance loss coeff = 0）、OverLB（load balance loss coeff = 0.1）。
  - 数据集：从 RedPajama（C4、CommonCrawl、Books、Wikipedia、ArXiv、StackExchange、GitHub）和下游应用数据（LMArena arena-human-preference-140k、OpenMathInstruct-2、OpenCode-Instruct、OpenScienceReasoning-2）中抽取，每域 2048 个 512-token 样本，总计 22,528 输入样本。Benchmark 为 SRP 和 SCH 指标本身。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/ljcleo/moe-lrc（论文明确给出），承诺发布采样 corpus 和确定性实验配置。
  - 算法 pipeline 伪代码（SRP 计算）：
    ```
    # Input: MoE model M, corpus S (22,528 samples × 512 tokens), segment length m
    # Output: SRP(E, m), ρ̂

    for each sequence T in S:
        for each MoE layer l in M:
            record expert activation matrix A_l[T] where A_l[T][i][e] ∈ {0,1}

    # Step 1: Per-expert SRP
    for each expert e in each layer l:
        # count activation frequency f in every segment
        for each segment [p, p+m-1] in each T:
            f[e,T,p,m] = Σ_{i=p}^{p+m-1} A_l[T][i][e]

        # find α that maximizes F1 (proved in Appendix C.3)
        for α in [0, m]:
            TP_α = Σ f[e,T,p,m] for segments where f[e,T,p,m] >= α
            FP_α = Σ m for segments where f[e,T,p,m] >= α  minus TP_α
            FN_α = Σ f[e,T,p,m] for segments where f[e,T,p,m] < α
            F1_α = 2*TP_α / (2*TP_α + FP_α + FN_α)
        SRP(e, m) = max_α(F1_α)

    # Step 2: Expert group SRP (layer/model level)
    for each expert group E:
        # Joint optimization over all experts in E
        # Equations 5-6: F1 maximized iff segment router predicts "active"
        # for all (e, segment) where f[e,T,p,m] >= α_e^m
        SRP(E, m) = joint_max_F1_over_all_e_in_E

    # Step 3: ρ̂ computation
    ρ̂ = (avg predicted active experts at optimal F1) / (avg original active experts)
    ```

  - SCH 计算伪代码：
    ```
    for each layer l, segment length m, cache ratio ρ:
        cache_size = ρ * k  # k = number of active experts per token
        for each segment start position p in all T:
            cache = empty_set()
            for token t in segment[p : p+m]:
                demanded = top_k(router_weights_l[t])
                hit = True
                for expert e in demanded:
                    if e not in cache:
                        evict_k = e  # mark as missed for this expert
                        hit = False
                if not hit:
                    # evict experts least activated in remaining future of segment
                    future_activation_counts = count_activations_past_t(expert, T, t+1, p+m)
                    evict_experts = bottom_N(cache, future_activation_counts, N=|missed|)
                    cache = (cache \ evict_experts) ∪ missed_experts
                    record miss_count
                else:
                    record hit_count
        SCH = hit_count / (hit_count + miss_count)
    ```
