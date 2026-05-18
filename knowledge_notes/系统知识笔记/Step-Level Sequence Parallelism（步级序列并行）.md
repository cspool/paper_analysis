## Step-Level Sequence Parallelism（步级序列并行）

术语是什么？

Step-Level Sequence Parallelism是TetriServe提出的DiT serving核心调度策略：在一个请求的多个denoising steps之间动态调整sequence parallelism的GPU degree（如从SP=2切换到SP=4再到SP=1），而非使用固定SP度贯穿整个请求。与固定SP策略不同，step-level SP根据每个请求的剩余步数和剩余deadline时间，在每步或每几步骤粒度上决定最优GPU分配——紧急大分辨率请求临时scale-up更多GPU加速，小分辨率或无紧迫deadline的请求仅用少量GPU以节省资源。

从系统架构角度拆解术语：

Step-Level SP在TetriServe中的运转流程：

1. **Offline Profiling**：对每种(分辨率, GPU数k∈{1,2,4,8})组合预先profiling单步耗时T(k)，构建cost model lookup table。同时profiling每种分辨率的最快单步耗时T_min = min_k T(k)。

2. **Deadline-Aware GPU Allocation**（每round初）：对于请求i有S_i个剩余steps和deadline D_i，计算满足deadline的最小GPU分配：min Σ(A_ij × T_ij(A_ij)) s.t. Σ(Q_ij + T_ij(A_ij)) ≤ D_i。通过枚举lookup table中的GPU候选值求解。

3. **Round-based Execution**：Scheduler选择(m, A^m_i)对——分配A^m_i个GPU执行q^m_i = min(s^m_i, floor(τ/T_i(A^m_i)))个steps（τ为round时长）。请求在round边界可被重新调度，切换SP度。

4. **Elastic Scale-Up**：在placement后利用空闲GPU为有余量steps的请求临时增加并行度（如从SP=2→SP=4），前提是T_i(k'_i) < T_i(k_i)。

5. **Placement Preservation**：跨round尽量保持同一请求在同一GPU集合上，避免state migration开销。latent transfer overhead低于0.05% per step。

关键挑战：step-level SP的scheduling问题是NP-hard（证明见TetriServe Appendix A），因为需要全局优化所有请求的每步GPU分配以最大化deadline满足数。TetriServe通过round-based DP heuristic近似求解。

术语一般如何实现？如何使用？

在TetriServe中，step-level SP通过C++实现的scheduler核心决策循环执行，达到毫秒级控制面延迟。NCCL process groups预warmup策略：预创建常用GPU组合的通信groups并warmup NCCL channels，不常用组合按需warmup，平衡启动延迟和显存。Latent transfer通过Future-like抽象实现异步非阻塞传输，使调度器可忽略传输时间进行deadline accounting。与xDiT固定SP相比，step-level SP在Uniform workload下SAR平均提高10%（tight SLO下提高28%），Skewed workload下提高15%（最高32%）。

涉及论文标题：
- TetriServe: Efficiently Serving Mixed DiT Workloads

---
