## Memoization of Fusion Analysis (De-Bruijn Index for Task Streams)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Memoization of Fusion Analysis 是 Diffuse 用于复用 task fusion 分析和编译结果的优化技术。由于科学计算应用常包含循环，同一 pattern 的 task stream 会在每次迭代中重复出现——但 store ID 不同（allocated to different IDs each iteration）。Diffuse 将 task stream 的匹配问题建模为 alpha-equivalence：将 store arguments 视为 bound variables，通过 canonical De-Bruijn index-like representation 将具体 store ID 替换为 canonical indices，使 isomorphic task streams 映射到同一 canonical representation，从而复用 fusion decision 和已编译的 fused kernel 代码。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Memoization 的 canonical representation 转换流程：

```
原始 task stream (Iteration 1):          原始 task stream (Iteration 2):
  T1([(S1,R), (S2,W)])                    T1([(S5,R), (S6,W)])
  T2([(S2,R), (S1,W)])                    T2([(S6,R), (S5,W)])
  T3([(S1,R), (S3,W)])                    T3([(S5,R), (S7,W)])
  T4([(S3,R), (S1,W)])                    T4([(S7,R), (S5,W)])

        ▼ Canonical Conversion                ▼ Canonical Conversion

  Canonical:                              Canonical:
    T1([(0,R), (1,W)])                   → 与 Iteration 1 相同!
    T2([(1,R), (0,W)])                      → 复用 fusion analysis
    T3([(0,R), (2,W)])                      → 复用 compiled fused kernel
    T4([(2,R), (0,W)])

  不同 pattern (非 isomorphic):
    T1([(S5,R), (S6,W)])
    T2([(S6,R), (S5,W)])
    T3([(S5,R), (S7,W)])     ← S7 替代了 S1 的位置? 不是!
    T4([(S7,R), (S5,W)])     ← 实际 pattern 相同（S7=S3 的新映射）
    
  对比 canonical: T3 uses (0,R) and (2,W) → pattern matches: isomorphic!
```

Canonical conversion 方法：从左到右遍历 task stream，第一次遇到每个 store 时分配从 0 递增的 canonical index。类似的 canonicalization 技术曾被用于避免枚举等价于 register renaming 的指令序列 [9]。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 Diffuse 中，canonicalization 在 fusion analysis 之前执行。对于每个进的 task window，先转换为 canonical representation，然后查询 memoization table。若命中（canonical representation 已存在），跳过 fusion analysis 和 MLIR compilation，直接复用 fused task 和 compiled kernel；否则执行 full analysis + compilation pipeline 并将结果存入 memoization table。由于科学计算应用的迭代性质，循环中的大部分 task stream 都会命中 cache。Paper 中报告该方法是实现 practical implementation 的必需组件。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
