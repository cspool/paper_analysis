## Optimization Memory (AccelOpt)

术语是什么？
Optimization Memory（优化记忆）是 AccelOpt 系统中的核心 self-improving 组件，它是一个固定容量（ExpN=16）的优化经验队列，存储从 beam search 探索过程中提取的 slow-fast kernel pairs 及其对应的通用优化策略。每条 experience item 包含三部分：(1) slow kernel 的伪代码片段（优化前），(2) fast kernel 的伪代码片段（优化后），(3) LLM Summarizer agent 生成的通用优化策略描述。Memory 的来源包括 positive rewrites（baseline kernel → 更快的生成 kernel）和 negative rewrites（生成 kernel → baseline kernel 更慢），两者都代表性能改进方向（从慢到快）。与 LessonL 不同，AccelOpt 的 memory 以 evolving candidate kernels 为 anchor，而非总是以 baseline 为 anchor，因此更 diversity。

从编译框架角度拆解术语：
Optimization Memory 的 curation 流程（Algorithm 2）：
```
输入: 本轮所有生成 kernel 的 profiled 结果 K, 上轮 memory E_{i-1}
阈值: tpos=1.04 (正向 speedup), tneg=1.15 (负向 slowdown), TopK=8

1. Group kernels by (candidate, plan):
   - 每个 (candidate, plan) 子组内找最快 kernel
   - 若子组 max_speedup > tpos → 加入正向候选 Rpos
   - 若子组 max_speedup < 1/tneg → 加入负向候选 Rneg

2. Summarizer agent 处理:
   - 对 Rpos 中 speedup 最高的 TopK/2 个 slow-fast pair
     生成 experience item (优化策略 + pseudo-code)
   - 对 Rneg 中 slowdown 最严重的 (TopK - |Epos|) 个 pair
     同样生成 experience item

3. 更新 memory 队列:
   E_i = [Epos_1, ..., Eneg_1, ..., E_{i-1}[:(ExpN - |Epos| - |Eneg|)]]
   - 新经验追加到队尾
   - 最旧经验当超出 ExpN 时丢弃
```

Memory 中的经验在下一轮被注入 Planner agent 的 prompt（"Past experiences" 字段），引导 Planner 提出受已有成功策略启发的优化计划。

术语一般如何实现？如何使用？
Memory 由 AccelOpt 框架自动管理，用户无需手动操作。Experience items 以自然语言 + pseudo-code 形式存储。实验表明：(1) 增加 ExpN 比增加 TopK 更 cost-efficient（图 15），(2) optimization memory 使达到相同 speedup 的迭代数减少 16-17%（图 13），(3) 当采样足够多 kernel 时，memory 不影响最终最优 kernel 性能但显著提升 cost efficiency（图 14）。ExpN 的效果依赖于 executor 模型能力——gpt-oss-120b 增加 ExpN 仅带来 0.6% speedup 提升，而 Qwen3-Coder-30B 带来 4.6%。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
