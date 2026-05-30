## Buddy Expert Substitution (Buddy 专家替代)

术语解释
Buddy Expert Substitution 是 BuddyMoE 的核心运行时机制：在 MoE 推理中，当 router 选择的 expert 不在 GPU 显存（cache miss/prefetch miss），不等待同步 CPU→GPU 传输（~10ms），而是用 GPU 显存中功能相似的 "buddy expert" 即时替代（~0ms），以极小精度损失换取显著吞吐提升。

术语是什么？
Buddy Expert 通过离线共激活模式分析和 CFT 构建。运行时替代流程（Algorithm 1）：对于每个 token 的每个 CPU-resident expert e_id，按 B_ℓ[e_id] 的 buddy ranking 查找 GPU-resident 的 buddy b_id，通过 atomic CAS 操作确保 uniqueness constraint 后替换到 S'。替代通过三个 safety gate 控制：TAE Gate（token 级敏感度）、Distribution Gate（batch 级 CPU expert 比例）、Buddy Priority Score Ψ（全局相似性 × 局部兼容性 × 拓扑感知）。CUDA kernel 并行化：grid(T,1,1) × block(K,1,1)，shared memory U_t 维护 token 的已分配 expert set，atomicCAS 保证无锁唯一性。

从算法pipeline角度拆解术语：
```
for token t in batch:
    S = Router(x_t).TopK(k)
    for e_id in S where M[e_id] == false:  # CPU-resident
        for r in range(H):
            b_id = B[e_id][r]
            if M[b_id] and b_id not in S':
                S'[e_id] = b_id; break
    output = Σ weight_i * FFN_i(x_t)  # with S'
```

术语一般如何实现？如何使用？
- 集成到 llama.cpp serving 框架，作为 router 和 expert execution 间的中间层
- Buddy profile 离线生成并序列化随 model checkpoint 加载，O(K_max · E_ℓ) 存储可忽略
- 与现有 prefetching 互补：prefetch 成功时正常工作，失败时用 buddy 避免 stall
- 最大收益在极端内存约束下：cache rate=0.375 时 +10.3% t/s vs original baseline

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

---
