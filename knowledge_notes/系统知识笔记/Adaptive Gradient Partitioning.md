## Adaptive Gradient Partitioning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Adaptive Gradient Partitioning 是 FSMoE 的反向传播梯度同步优化技术。Gradient-AllReduce 和 AlltoAll 均为节点间通信，无法直接重叠。FSMoE 两阶段算法将梯度自适应分配到各 MoE 层的 overlappable parts：Phase 1 贪心分配梯度到空闲时间段，Phase 2 差分进化优化剩余梯度跨层分配。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
Phase 1: for each layer i (last→first):
    r_i = Algorithm1(t_gar=0)               # 先以无GAR优化
    t_olp_i = overlappable_time(r_i)         # MoE+dense空闲时间
    n_grad_i = g_grad_inv(min(t_grad(n_rem), t_olp_i))

Phase 2: if n_rem > 0:
    minimize Σ f_moe^i(t_grad(x_g^i))       # 差分进化求解
    # f_moe^i = Algorithm1(t_gar=t_grad(x_g^i))
```

Overlappable parts 的三种形态：Case2: r·t_exp+t_ag+t_rs-2(r-1)t_a2a; Case3: t_ag+t_rs; Case4: r·t_ag+r·t_rs-2(r-1)t_a2a。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 使用 scipy 的 differential_evolution 算法。优化训练前执行一次。对比 PipeMoE+Lina（固定 30MB chunk），FSMoE 自适应分区在多变配置下均更优（加速 1.14× vs PipeMoE+Lina）。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
