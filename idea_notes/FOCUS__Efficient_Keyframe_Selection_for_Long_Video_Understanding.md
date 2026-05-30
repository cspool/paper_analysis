## FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding

- baseline方法是什么？
  Baseline 方法是 (1) uniform sampling（当前 MLLM 默认策略：固定间隔采样 N 帧，完全 query-agnostic）；(2) Top-K keyframe selection——用 Vision-Language encoder（如 BLIP）计算每帧与 query 的 relevance score，选 top-K 最高分帧，但需要预过滤（pre-filtering: downsampling to 1 fps）以控制计算开销；(3) AKS (Tang et al., 2025)——adaptive keyframe sampling，通过 split-and-judge 递归策略平衡 relevance 和 coverage，同样依赖预过滤（1 fps）降低候选帧数。

  Baseline（以 Top-K / AKS 的典型流程为例）全栈执行例子：
  - 算法层：输入一小时长视频 (108K frames @ 30fps) + query → 预过滤阶段（downsampling to 1 fps → 3600 候选帧，丢弃 97% 原始帧）→ 逐候选帧经 BLIP ITM 计算 r_t = cos_sim(e_t, e_q) → Top-K 选最高分 k 帧 / AKS 递归 split-and-judge 分配 → k frames → Vision Encoder + LLM decoder → 答案。预过滤丢弃了大量未评分帧，可能恰好漏掉关键帧。
  - 系统框架层：HuggingFace Transformers，BLIP 逐帧串行或小 batch forward
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 H100 (80GB)

  Baseline 的缺陷：
  1. **预过滤导致信息丢失**：Top-K 和 AKS 都先 uniform downsample 到 1 fps 才评分——对于一小时视频意味着丢弃 97% 的帧不做评分。关键视觉信息可能恰好存在于被丢弃的帧中（如一个持续仅 0.5 秒的短暂事件），预过滤从根本上违背了 keyframe selection 的目标（从全部帧中选最优帧）。
  2. **全帧评分计算不可行**：若不做预过滤，108K 帧全用 BLIP 评分需约 10^11-10^12 FLOPs（论文估算），对应 AKS w/o pre-filtering 的 255 GPU hours——在实际应用中完全不可行。
  3. **均匀采样无法感知 query 相关性**：Uniform sampling 对所有 query 返回相同帧集合，在 LongVideoBench（问题聚焦于特定场景/事件）上准确率极低（如 Long 视频上仅 51.8% with LLaVA-Video-7B）。
  4. **AKS 需要调节 coverage vs relevance 的超参数（s_thr, L）**：不同 benchmark（LongVideoBench vs VideoMME）需要不同的超参数组合才能达到最优，泛化性受限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：FOCUS 通过将 keyframe selection 建模为 Combinatorial Pure-Exploration (CPE) multi-armed bandit 问题，消除预过滤并提供理论保证：
  (1) **Clip-level bandit 建模**：将视频按固定长度（16s）划分为 M 个 clip 作为 bandit arms，目标是选出 top-m 个预期 relevance 最高的 arm。基于视频帧间强时间相关性（Figure 1: median ACF > 0.5 for ~5 seconds）的观察，clip 内相邻帧高度相似，因此只需少量采样即可估计整个 clip 的 relevance。
  (2) **Bernstein confidence radius + 乐观探索**：每个 arm 维护经验均值 μ̂_a 和方差自适应 Bernstein confidence radius β_a(n)，用 optimistic mean μ̃_a = μ̂_a + β_a 指导 arm 选择——既利用高均值 arm 又探索高不确定性 arm。
  (3) **两阶段并行批处理**：将原始串行 bandit 算法（Algorithm 1）简化为两次并行批处理（Algorithm 2）：Stage I 所有 arm 各采样 q 帧 → Stage II 仅对 top α*m optimistic arm 再采样 z 帧 → 选 top-m arm by empirical mean。充分利用 GPU 批处理能力。
  (4) **无需预过滤**：FOCUS 直接对所有 clip 并行采样，不预先丢弃任何帧。通过只对少量帧做 BLIP forward 实现高效探索——仅处理 ~1.6% 的总帧数，5.5 GPU hours。

  对比 baseline 的全栈执行例子（FOCUS + LLaVA-Video-7B, k=64, clip=16s, α=0.25, 同一小时长视频）：
  - 算法层：
    1. 将视频划分为 M 个 16s clip → M 个 bandit arms
    2. Stage I: 每个 arm 并行采样 q 帧 → 批量 BLIP forward 计算 r_t —— 仅需 ~1.0% 总帧数的 BLIP forward
    3. 计算每个 arm 的 μ̂_a, σ̂_a², β_a(n), μ̃_a(n)
    4. A_coarse = TopM(μ̃, α*m) ← 选 optimistic mean 最高的 α*m arms
    5. Stage II: 对 A_coarse 中每个 arm 并行采样 z 帧 → 批量 BLIP forward
    6. A_fine = TopM(μ̂, m) ← 基于无偏经验均值选最终 m 个 arms
    7. 在选中 arms 内 nearest-neighbor 插值 + 概率采样 k_a 帧 → 64 keyframes
    8. Vision Encoder + LLM decoder → 答案
    → LongVideoBench Long videos (>20min) accuracy: 63.7%（vs Uniform 51.8%, +11.9%; vs Top-K 60.5%, +3.2%）
    → GPU hours: 5.5h（vs AKS w/ pre-filtering 9.3h, vs AKS w/o pre-filtering 255h）
  - 系统框架层：HuggingFace Transformers，BLIP 批量 forward 在单卡 H100 上
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 NVIDIA H100 (80GB)

  解决对应关系：
  | Baseline 缺陷 | FOCUS 解决方案 |
  |---|---|
  | 预过滤导致信息丢失（丢弃 97% 帧） | 无需预过滤：所有 arm（覆盖所有帧）都参与 Stage I coarse exploration，通过只对少量帧做 BLIP forward（~1.6%）实现高效探索。LPU hours 5.5h vs AKS 255h（w/o pre-filtering） |
  | 全帧评分计算不可行（10^11-10^12 FLOPs） | Bandit 采样：实际只有 ~1.6% 帧被 BLIP 评分。Table 3 对比：FOCUS 1.6% frames seen vs AKS 3.7%（w/ pre-filtering）vs 100%（w/o pre-filtering） |
  | Uniform 忽略 query 相关性 | Bandit arms 的 reward 基于 BLIP 计算的 query-frame cosine similarity，选出 query-relevant top-m arms。LongVideoBench Long: +11.9% over uniform |
  | AKS 需要 tune coverage 超参数 | Bandit 框架的探索机制（UCB/confidence radius）自动平衡 exploration 和 exploitation，无需 coverage 约束。α 虽可调但对 accuracy 影响小（0.1-0.5 范围内 accuracy 62.9-63.6%） |

  核心技术贡献：
  - **Bernstein confidence radius 的方差自适应探索**：比标准 UCB 更鲁棒。Table 8 消融：FOCUS-M（仅用经验均值）62.3/58.1/63.0 vs FOCUS（加 Bernstein）63.5/60.7/63.5（LLaVA-Video/Qwen2-VL/LLaVA-OV）
  - **理论保证**：Bernstein confidence bound 保证 |μ̂_a - μ_a| ≤ β_a 以 ≥ 1-6/n 概率成立（Theorem B.1）；Algorithm 2 以 ≥ 1-6(M-m)/n 概率返回 oracle top-m set（Theorem C.1）
  - **Two-stage 批处理设计**：FOCUS-C（仅 coarse）61.7/58.4/62.3, FOCUS-F（仅 fine）61.5/57.7/62.5, FOCUS（两阶段）62.3/60.7/63.5 — 两阶段互补，coarse localization + fine exploitation
