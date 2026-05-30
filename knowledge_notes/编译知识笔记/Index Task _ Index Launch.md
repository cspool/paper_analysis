## Index Task / Index Launch

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Index Task（在 Legion 中称为 Index Launch [50]）是一种表示分布式并行 task group 的编程抽象。一个 IndexTask(d, A) 包含：(1) launch domain d：一个多维矩形 domain，每个点对应一个 processor；(2) 参数列表 A：一组 (store, partition, privilege) 三元组，定义每个点上的 point task 操作哪些 sub-store 以及以何种权限访问。这样，一个 IndexTask 声明即代表了所有 processor 上的并行计算，而无需逐 processor 重复声明。这是实现 scale-free representation 的核心构件：task 数量不随 processor 数增长。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
IndexTask(d=(4,), [  // 4个GPU, launch domain为4个点
  (center, P_center, Read),    // P_center 将 center 按 4 个 GPU 分区
  (north,  P_north,  Read),    // P_north 是 center 的 aliasing partition (offset shift)
  (t1,     P_t1,     Write)    // t1 同样按 4 个 GPU 分区
])

展开为 Point Tasks (每个 GPU 一个):
  GPU 0: PointTask(0, [SubStore(center, P_center, 0), SubStore(north, P_north, 0), SubStore(t1, P_t1, 0)])
  GPU 1: PointTask(1, [...])
  GPU 2: PointTask(2, [...])
  GPU 3: PointTask(3, [...])

每个 PointTask 内部: 对 sub-store 中的元素执行 element-wise ADD:
  for i, j in sub_store:
      t1[i,j] = center[i,j] + north[i,j]
```

Dependence Map D(T1, T2) 定义为 domain(T1) → P(domain(T2)) 的函数，表示 T2 中哪些 point task 依赖 T1 的每个 point task。当 D(T1, T2)[p] ⊆ {p} 对所有 p 成立时，T1 和 T2 fusible（只有 point-wise dependency，无跨 processor 通信）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Index Task 由高层库（如 cuPyNumeric, Legate Sparse）的开发者构造，终端用户不直接操作。在 cuPyNumeric 中，每个 NumPy 操作产生一个或多个 IndexTask。Diffuse 在 fusion 过程中分析 IndexTask 的参数并构造 fused IndexTask。最终 IndexTask 翻译为 Legion 的 Index Launch 执行。Index Launch 的正确性由 Legion runtime 的 visibility algorithm [14] 保证——它计算每个 point task 的精确依赖关系并调度通信。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
