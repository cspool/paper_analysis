## Layout Inference (Layout 推断机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Layout Inference 是 TileLang 编译器中的核心自动化 pass，负责推断 kernel 中所有 buffer 的 Fragment Layout（即 block 级 register file 到各 thread 的映射方式）和 Thread Binding（循环如何映射到 threads）。Layout Inference 基于两个关键观察：(1) 多个 tile operators 可能共享同一 buffer，它们的 layout 和 thread binding 策略互为依赖；(2) 不同 operator 对 layout/thread binding 的严格性要求不同 — GEMM（使用 Tensor Cores）要求最严格的 layout 约束，而 element-wise 操作允许更大灵活性。

Layout Inference 的算法：维护 LayoutMap（记录所有 buffer 的 layout 信息）→ 定义三层优先级（Gemm > Element-wise > Copy）→ 从高到低逐层推断，每层遍历所有待定 buffer 尝试推断 → 直到该层无更多 buffer 可推断 → 进入下一层。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

以 GEMM + bias add 为例（图 7 场景）：
```
场景: C_local = GEMM(A_shared, B_shared) + D
  - C_local: GEMM 输出，Fragment Layout 已知（MakeMMASTMatrixLayout）
  - D: bias buffer，layout 待定

Layout Inference 过程 (按优先级):
Level 1 — Gemm (最高优先级):
  - A_shared: 推断为 MakeSwizzleLayout
  - B_shared: 推断为 MakeSwizzleLayout  
  - C_local: 推断为 MakeMMASTMatrixLayout
    (GEMM 的 thread binding 已确定: C_local[4x4] 分布在 8 threads, 每 thread 2 elements)

Level 2 — Element-wise (bias add):
  - 观察 C_local 的 Fragment Layout: 每行由 2 threads 处理 → 两 thread 需相同 D 元素
  - 推断 D 的 layout: 需 replicate — 每行 D 元素需复制到处理该行的两个 thread 的 register
  - 通过 repeat/repeat_on_thread/replicate 四种 Fragment primitive 实现

Level 3 — Copy (最低优先级):
  - 剩余的 global→shared copy 操作的 thread binding 基于已确定的 thread 数自动 parallelize
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Layout Inference 在编译器内部实现，用户通常无需手动介入。当默认推断不足以产生最优 layout 时，用户可通过 T.annotate_layout 显式覆盖特定 buffer 的 layout。Layout Inference 的输出同时驱动自动向量化和 loop parallelization pass（图 8 展示 multi-stage 自动 thread binding inference）。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---
