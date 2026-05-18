## Round-Based Deadline-Aware Scheduling（基于轮次的截止时间感知调度）

术语是什么？

Round-Based Deadline-Aware Scheduling是TetriServe提出的DiT serving调度框架：将连续时间离散化为固定时长τ的rounds，每个round内通过动态规划(DP)进行request packing以最大化满足deadline的请求数。该设计的核心动机是DiT serving的step-level调度问题是NP-hard，连续时间全局优化不可行，而round-based离散化使问题在per-round粒度上可用DP高效近似求解（O(RN)时间, O(N)空间）。

从系统架构角度拆解术语：

Round-based调度在TetriServe中的运转流程：

1. **Round定义**：每个round时长为τ（由step granularity决定，如5或10 steps的等效时间）。round在固定时间边界开始，新到达请求在下一round初被纳入调度。

2. **Per-Round Scheduling Problem**（Algorithm 1 DP Round Scheduler）：
   - 输入：pending requests R，每个request i有候选allocation set {(s^m_i, A^m_i)}（步数, GPU数），per-step time T_i(·)，GPU容量N，round时长τ，当前时间t_r
   - 对每个request i的每个allocation m：计算q^m_i = min(s^m_i, floor(τ/T_i(A^m_i)))（本round可完成的步数）
   - 构建option set O_i = {none} ∪ {m | q^m_i > 0 ∧ A^m_i ≤ N}
   - 对每个option o：计算更新后剩余步数s̃^m_i(o)，剩余时间下界LB_i(o) = Σ s̃^m_i(o) × T_i^min
   - sv_i(o) = I[t_{r+1} + LB_i(o) ≤ D_i]（请求在下一round初是否尚未definitely late）
   - DP：dp[c] = max surviving requests consuming c GPUs，transition为next[c] = max(next[c], dp[c - w_i(o)] + sv_i(o))

3. **"Definitely Late"判定**：使用最快可能步时T_i^min计算LB_i(o)，若即使以最快速度执行也无法在deadline前完成，该请求被标记为definitely late，在本round必须被调度。

4. **Round Duration选择**：τ在scheduling overhead和响应性之间权衡。TetriServe根据step granularity确定τ（如5 steps），使异构步长的请求在相似round边界完成，最小化idle bubbles。

术语一般如何实现？如何使用？

TetriServe的round-based scheduler用C++实现核心DP循环，绑定了Python接口。DP使用rolling array优化到O(N)空间。对于已超时请求，分配最多1 GPU best-effort执行，不参与DP竞争。scheduler通过与xDiT的GPU workers通过async logic（复用vLLM）通信，在每round结束时收集完成状态并触发下一round调度。Round duration在实验中sweep了{1,2,5,10,15} steps，中等granularity（5 steps）在调度灵活性和overhead间取得最优平衡。

涉及论文标题：
- TetriServe: Efficiently Serving Mixed DiT Workloads

---
