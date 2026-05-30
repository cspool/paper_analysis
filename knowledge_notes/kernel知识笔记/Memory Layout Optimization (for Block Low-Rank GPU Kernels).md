## Memory Layout Optimization (for Block Low-Rank GPU Kernels)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Layout Optimization是通过改变权重tensor在内存中的存储顺序（contiguous dimension ordering）来消除在线推理时permutation操作的技术。核心原理：静态权重以"对后续bmm友好的layout"存储时，原本需要独立kernel的permutation可被完全消除（数据已是目标顺序）。区别于permutation fusion（在kernel内完成重排），layout optimization将重排移到离线阶段→零运行时开销。

论文实例：(1) Monarch V重排布——V从沿b₂ first改沿r' first contiguous，消除r'↔b₂ permutation；(2) Pre-permute下游权重——Monarch输出后被静态权重消费时，pre-permute该权重rows→消除最终permutation；(3) BLAST S/U转置——离线转置S/U消除所有在线permutation。

从kernel调度角度拆解术语：
```
// Monarch ①: V重排布 (offline, one-time)
V_old: [b₁, b₂, r', p] → permute(0,2,1,3) → [b₁, r', b₂, p]
V_new: reshape(b₁, r'*b₂, p) with r' first contiguous

// 效果: 第一批bmm(X_blocks @ V_new^T) 直接产生目标layout
//       消除独立的 r'↔b₂ permutation kernel
//       消除128MB (b₁×n×r'b₂) 中间permuted tensor

// Monarch ③: Pre-permute downstream weight
// 若Y_Monarch ∈ (b₂, n, q)后接W_down @ Y
// offline: W_down rows重排为Monarch-friendly order
// → 跳过在线 (b₂,n,q)→(n,q,b₂) permutation kernel
```

术语一般如何实现？如何使用？
静态权重pre-processing：checkpoint加载后一次性转换（<1秒）。限制：(1)仅适用于静态权重，不适用动态激活；(2)改变存储格式→与下游kernel协同设计（tile尺寸需调整以保证coalesced access）；(3)layout变化后需re-tune kernel tile sizes。对于BLAST ⑤的S/U转置，关键是保持n在contiguous维度以确保后续tensor core访问效率。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
