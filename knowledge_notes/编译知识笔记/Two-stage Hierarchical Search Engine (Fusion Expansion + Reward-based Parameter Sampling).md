## Two-stage Hierarchical Search Engine (Fusion Expansion + Reward-based Parameter Sampling)

术语是什么？通过联网搜索让回答具体和精准。

Two-stage Hierarchical Search Engine是STOF中用于在hierarchical optimization space（fusion scheme × kernel parameters）中自动搜索最优配置的搜索引擎。Stage 1 (Fusion Expansion)：基于初始fusion scheme（由neural hashing + predefined rules生成），使用DFS (Depth-First Search)逐步扩大各fused segment的边界，每次扩展后随机采样参数配置比较pre-fusion vs post-fusion性能，有gain保留否则回退。Stage 2 (Parameter Sampling)：对已确定的fusion scheme，用reward-based算法动态分配各segment的参数采样配额——首轮各segment等量采样，后续对带来最高overall gain的segment增加配额。两个阶段共享performance data cache，避免重复执行相同(scheme, parameters)组合。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
// ===== Stage 1: Fusion Expansion (DFS with feedback) =====
// Three expansion rules (围绕CI operator, 每segment最多2个CI):
//
// expand: 合并相邻segment（不破坏其他segment结构）
//   S0: [0 1][0 0 0 0 0][1 1][0 1 1][1 0 0]  
//       → expand segment 2: [0 1][0 0 0 0 0][1 1 1 1 1][0 0]
//       → S1: 0 0 1 1 1 1 1 0 0 1 0 0 0 1 1
//
// seize: 含CI的segment抢占纯MI segment的operator
//   S1: 0 0 1 1 1 1 1 0 0 1 0 0 0 1 1
//       → segment 2 (CI+MI) seize from segment 3 (MI-only):
//   S2: 0 0 1 1 1 1 1 0 0 1 0 0 0 0 1
//
// compete: 两个segment竞争一个operator → CI少的segment优先
//   S2: 0 0 1 1 1 1 1 0 0 1 0 0 0 0 1
//       → segment 2 (1 CI) vs segment 3 (0 CI):
//       → segment 3 wins, extends left:
//   S3: 0 0 1 1 1 1 1 0 0 0 1 1 1 1 0

// DFS with performance feedback:
function fusion_expand(scheme S):
    for rule in [expand, seize, compete]:
        S' = apply_rule(S, rule)
        if S' in performance_cache:
            continue  // 已尝试过, skip
        // 随机采样 pre-fusion和post-fusion各N个参数配置
        params_pre = random_sample(N, S.segments)
        params_post = random_sample(N, S'.segments)
        perf_pre = max(benchmark(S, p) for p in params_pre)
        perf_post = max(benchmark(S', p) for p in params_post)
        performance_cache[(S', params)] = perf_post
        if perf_post > perf_pre:
            S = S'  // 接受 expansion
        else:
            rollback  // 拒绝, 保持原scheme
    return S

// ===== Stage 2: Reward-based Parameter Sampling =====
// 输入: 已确定fusion scheme S with k segments
// 总采样配额: T (per iteration)
//
// Iteration 1: equal allocation
//   each segment gets T/k samples
//   benchmark all → record per-segment gain
//
// Iteration i+1: reward-based reallocation
//   gains = [gain_0, gain_1, ..., gain_{k-1}]  // from previous iteration
//   best_segment = argmax(gains)
//   samples[best_segment] += Δ  // reward: 增加配额
//   // other segments keep same or slightly reduced
//
// Convergence: 当所有segments的gain improvement < ε 或达到max iterations

// Performance Cache:
// Key: (scheme_hash, segment_id, param_config_hash)
// Value: measured_latency
// 作用: 避免重复执行相同配置——尤其是大input scale下
//   (16, 2048) setting下, cache使STOF tuning time比MCFuser快6.7×
```

与已有auto-tuning方法的对比：
- TVM/Ansor: loop-based construction + ML cost model → 通用但性能不如vendor library
- Bolt: template-based但fusion expansion需要程序员手动修改CUTLASS kernel
- MCFuser: loop-based CI chain tuning但搜索空间受限
- STOF: template-based + hash encoding自由扩展 + reward-based采样 → 更小tuning time (6.7× faster than MCFuser, 6.9× faster than Bolt)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

STOF以Python实现search engine。DFS in Stage 1用递归+visited set（基于scheme hash）实现，每次expansion调用Triton/TileLang的JIT compilation + CUDA execution来benchmark。Reward-based采样在Stage 2用简单的贪心策略——记录每次iteration各segment对overall runtime的贡献，贡献最大的segment获得额外采样配额。Performance cache使用内存中的dict (key=hash(scheme+params), value=latency)，在tuning session生命周期内有效。Overhead analysis (STOF Figure 13)显示: analytical model占tuning time <0.5%，hash encoding+numerical decoding占<1%，reward algorithm占<1.5%，总计<3% tuning time overhead。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU
