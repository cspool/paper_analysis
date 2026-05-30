## Beam Search for Kernel Optimization

术语是什么？
Beam Search for Kernel Optimization 是将 beam search（束搜索）策略应用于 AI 加速器 kernel 优化空间探索的方法。与传统的 repeated sampling（重复采样）不同，beam search 每轮维护 B 个 candidate kernel（beam width），对每个 candidate 生成 N 个优化计划、每个计划尝试 K 次实现（生成 B×N×K 个 kernel），然后从 (B + B×N×K) 个 kernel 中选择 Top-B 进入下一轮。核心机制是 candidate selection function β：首先在每个 (candidate, plan) 子组内选出最快的正确 kernel（确保每个探索方向都贡献其最优结果），然后按 latency 排序选择全局 Top-B 作为下一轮 candidates。若 valid kernel 数不足 B，用上一轮 candidates 填充。

从编译框架角度拆解术语：
Beam Search 算法流程（Algorithm 1）：
```
输入: E_{i-1} (上轮优化经验), C_i (本轮 B 个 candidates)
参数: θ_p (planner), θ_e (executor), θ_s (summarizer), r (profiler)

for each candidate c in C_i:
    // 1. Planner: 为每个 candidate 生成 N 个优化计划
    P = {p | p ~ θ_p(p | c, E_{i-1})}     // |P| = N
    for each plan p in P:
        // 2. Executor: 每个计划尝试 K 次实现
        A_p = {(a, p, r(a)) | a ~ θ_e(e | p, c)}  // |A_p| = K
        // 每个 kernel 编译、在硬件上运行、profiling

// 3. Summarizer + Memory curation
E_i = σ(K, E_{i-1}; θ_s)         // 从 K = ∪A_p 中提炼优化经验

// 4. Candidate selection
C_{i+1} = β(K ∪ C_i, B)          // 选择下轮 B 个 candidates
```

与 repeated sampling 的关键区别：repeated sampling 每次从同一 baseline 出发独立采样，而 beam search 每轮基于上轮最优结果继续探索（cumulative improvement）。实验证明 beam search 比 repeated sampling 更有效（图 13），使用相同 LLM 获得更高 speedup。

术语一般如何实现？如何使用？
在 AccelOpt 中，beam search 的 hyperparameters 为 B=6, N=12, K=2, T=16（总计每问题约 2304 个 kernel 被采样）。B 控制探索广度（diversity），当 B 减至 1 时性能显著下降（图 13 中 B=4 vs B=6）。β 函数是 search 效果的关键：group-by-candidate-and-plan 的 design 确保了多样性（diversity），每个 (candidate, plan) 只贡献一个最优 kernel。该机制对其他 AI 加速器平台（如 GPU Triton kernel）同样适用，平台适应性仅需 profiling service + platform-specific base prompts。

涉及论文标题：
- AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
