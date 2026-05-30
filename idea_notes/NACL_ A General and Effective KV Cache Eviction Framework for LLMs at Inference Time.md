## NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

- baseline方法是什么？
  Baseline 是 H2O（Heavy-Hitter Oracle）和 MSRNN 等基于 attention score 的 KV cache 淘汰方法。H2O 在 generation 阶段每步贪心淘汰 KV cache：对每个 token 计算 F_score = Σ_{all past tokens} Softmax(A[i, :])（累加全部历史 attention scores），保留 top-C 最高分 token + 最近 token。MSRNN 仅用当前 token 的 attention score 做淘汰。

  全栈执行例子（H2O baseline, LLaMA2-7B-Chat, 4K context, 单 A100 80GB）：

  - **算法层**：输入 x_prompt ∈ R^{4096×4096}。encoding 阶段正常 prefill 存储完整 KV cache。generation 阶段每生成一个 token，计算当前 token 对所有 cache 中 key 的 attention scores，累加到历史 accumulated attention scores，按总分排序淘汰低分 token。核心缺陷：
    1. **Attention bias problem**：attention scores 高度集中在初始 token 和最近 token，中间 token 即使关键（如 passkey）也因低 attention 被 H2O 淘汰（Fig. 2）。
    2. **Step-by-step 贪心淘汰**：每步基于局部信息做淘汰决策，无法全局优化；时间复杂度 O(p+T) per token。
    3. **冗余信息干扰**：H2O 累加全部 token 的 attention scores，大量无关 token 的 scores 引入噪声，稀释了真正重要的 task-specific scores。
    4. **Perplexity 不可靠**：H2O 用 PPL 作为主要指标，但在 long-text 实际任务（如 LongBench passkey retrieval）中 PPL 表现好但任务准确率差。

  - **kernel调度层**：使用标准 FlashAttention-2 计算 attention，KV cache 淘汰操作为纯 PyTorch tensor indexing。无自定义 kernel。

  - **Serving/框架层**：论文未明确说明 serving 框架。淘汰逻辑在 HuggingFace Transformers 推理 pipeline 中以 Python hook 实现。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Attention bias（H2O 偏向初始/最近 token，MSRNN 仅看当前 token）→ 中间关键 token 被误淘汰
  2. Step-by-step 贪心淘汰 O(p+T) → 长 context 下淘汰本身成为瓶颈
  3. 全量 attention score 累加引入冗余信息 → 评分不精准
  4. PPL 评估不反映真实 long-text 任务性能 → 方法效果被高估
  5. 确定性淘汰缺乏鲁棒性 → 一旦关键 token 被淘汰无法恢复

- 论文方法是什么？如何对应解决Baseline的缺陷？
  NACL 提出混合 KV cache 淘汰框架：PROXY-TOKENS EVICTION（基于 proxy tokens 的全局最优淘汰）+ RANDOM EVICTION（per-head 概率采样淘汰），在 encoding 阶段一次性完成淘汰。

  NACL 全栈执行例子（LLaMA2-7B-Chat, 4K context, C=20%, C_p=6%, C_r=12%, 单 A100 80GB）：

  - **算法层**：
    1. **Encoding phase one-eviction**：将淘汰从 generation 阶段移至 encoding 阶段一次性完成。encoding 阶段计算完整 attention matrix A ∈ R^{p×p}，利用全局信息做最优淘汰 S_encoding = F_score(A, C)，随后 compressed KV cache 用于全部 generation steps。时间复杂度从 O(p+T) 降至 O(1)（T ≪ p）。
    2. **PROXY-TOKENS EVICTION**：选取输入末尾 ~10% token 作为 proxy tokens P（对应用户问题部分），F_score = Σ_{x_p∈P} Softmax(A[x_p, :])，仅聚合 proxy tokens 的 attention 信号。proxy tokens 天然携带 task-specific 信息，其 attention pattern 更精准地反映哪些 token 对任务关键。淘汰建模为组合优化：S_t = argmax_{S⊂R} Σ_{x∈S} F_score(A, C_p) ∪ P（proxy tokens 默认保留）。
    3. **RANDOM EVICTION**：将 F_score 经 Softmax 归一化得到概率分布 P_prompt，从该分布中采样 C_r 个 token 保留。每个 head 使用不同 seed → head-wise 多样化采样。在 LLaMA-7B 32层×32头、budget=20% 下，token 在至少一个 head 中保留概率为 1-(C_h)^l，即使 C=1% 也 >99.99%。
    4. **Hybrid budget allocation**：C = C_p + C_r，典型比例 20% total = 6% proxy eviction + 12% random + 2% protect proxy（Tab. 4）。

  - **kernel调度层**：实现 Reduce Attention Scores CUDA kernel 兼容 FlashAttention-2。利用 FlashAttention-2 forward 输出的 log-sum-exp 重算 attention scores 并做 column-wise reduce（Algorithm 2）。或仅对 proxy tokens（~10%）重算 attention scores，开销可忽略。128K context 下 evict 20% 维持 ~15GB 稳定显存。

  - **Serving/框架层**：论文未明确说明 serving 框架。NACL 作为 pluggable eviction policy 可在 HuggingFace Transformers 中以 hook 形式实现。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 缺陷 → NACL 设计映射：
  1. **Attention bias（H2O 偏向初始/最近）→ PROXY-TOKENS EVICTION 用 task-specific proxy tokens**：H2O 累加全部 token attention（含大量非任务相关的 attention scores），导致评分偏向位置而非语义重要性。NACL 仅用末尾 ~10% 的 proxy tokens（用户问题）评分，这些 token 的 attention pattern 反映"哪些 prefix token 对回答当前问题有用"，从而更精准定位关键信息。实验证据：NACL 在 passkey retrieval（PR-Zh/PR-En）上显著优于 H2O（30% budget: NACL 6.8/9.0 vs H2O 3.7/5.0），证明 proxy tokens 能保留位于中间的 passkey 而 H2O 的 attention bias 将其淘汰。
  2. **Step-by-step 贪心 O(p+T) → Encoding phase one-eviction O(1)**：H2O 每 generation step 做一次淘汰，长 context 下淘汰开销占总推理时间可观。NACL 在 encoding 阶段利用完整 attention matrix 一次性全局优化淘汰，generation 阶段仅每 m 步做轻量淘汰。消融：移除 global eviction 导致 short-text -1.3%、long-text -1.5%。
  3. **全量 attention 累加引入噪声 → Proxy tokens 子集精准评分**：H2O 的 F_score = Σ_{all} Softmax(A[i,:]) 中大量 non-task-related token 的 attention 贡献噪音。NACL 的 F_score = Σ_{P} Softmax(A[x_p,:]) 仅聚合 proxy tokens。消融：移除 PROXY-TOKENS EVICTION 后 short-text -28.1%、long-text -6.0%，证明该策略是最核心贡献。极端情况：0% proxy budget = MSRNN（仅当前 token），100% = H2O（全量 token），~10% 最优。
  4. **确定性淘汰无鲁棒性 → RANDOM EVICTION head-wise 多样化采样**：H2O 的确定性 top-K 一旦丢弃关键 token 无法恢复。NACL 的 head-wise probability sampling 确保每个 token 在多个 head 的 KV cache 中有独立被保留的机会。消融：移除 RANDOM EVICTION 后 short-text -1.2%、long-text -9.2%（long-text 下随机性更重要！）。Uniform sampling 替代 attention-weighted sampling 后 long-text -1.1%。证明随机性 + attention 引导的组合最优。
  5. **PPL 不可靠 → LongBench + lm-eval-harness 真实任务评估**：论文重新评估了 H2O、MSRNN、Attention Sink 在 short-text（7 任务，5-shot/25-shot）和 long-text（LongBench 7 任务，budget 10%/20%/30%）上的真实表现，揭示 PPL 与实际任务准确率的系统性偏差。

  **关键性能**：
  - NACL 20% short-text avg 63.8 vs Full 64.6 (-0.8) vs H2O 60.3 (-4.3) — 80% improvement over H2O
  - NACL 20% long-text avg 30.8 vs Full 31.5 (-0.7) vs H2O 28.6 (-2.9) — 76% improvement over H2O
  - KV cache up to 5× reduction with >95% performance maintenance
  - LLaMA2-7B, batch=4, 32K seq: 64GB → NACL 20% ≈ 12.8GB
