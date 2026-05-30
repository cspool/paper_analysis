## SMAWK Algorithm（SMAWK 全单调矩阵行最小值搜索算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SMAWK 是 Aggarwal, Klawe, Moran, Shor 和 Wilber 于 1986 年提出的算法（以五位作者姓氏首字母命名），用于在 O(n) 时间内找到 n×n 隐式定义的 totally monotone matrix 中每行的最小值位置。Totally monotone matrix 定义为：对于任意 a < b 和 c < d，若 A[a,c] > A[b,c] 则必有 A[a,d] > A[b,d]。该性质保证矩阵的行最小值列索引单调非递减。SMAWK 利用这一性质通过"剪枝 + 递归归约 + 合并"将复杂度从 O(n²) 降至 O(n)，且不需要显式存储整个矩阵——只需要能 O(1) 时间求值矩阵任意位置的函数即可。QUIVER 利用 C[k,j] 满足 quadrangle inequality 的性质证明 DP 矩阵 A[k,j]=MSE[i-1,k]+C[k,j] 是 totally monotone，从而用 SMAWK 以 O(d) 时间找到 min_k A[k,j]，替代原 DP 中每步 O(d²) 的枚举。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
SMAWK 的四个阶段（以在 QUIVER DP 中的应用为例）：

```
输入: 隐式矩阵 A ∈ R^{d×d}（totally monotone），函数 f(k,j) 可在 O(1) 求值 A[k,j]
输出: 每列的 argmin 索引 r[1..d]

// Phase 1: Pruning（剪枝）
// 利用 totally monotone 性质消除不可能包含行最小值的列
stack = [(A[1,1], 1)]  // (值, 列号)
for j = 2 to d:
    // 比较相邻列：若 A[k,j] < A[k,j-1] 对所有行k成立，则列j-1可被剪掉
    while stack not empty:
        用 totally monotone 性质判断栈顶列是否被支配
    push j onto stack
// 剪枝后剩余列数 ≤ 行数

// Phase 2: Recursive Reduction（递归归约）
// 只考虑偶数行，在剪枝后的列集上递归求解
if rows == 1:
    return naive_min over all remaining columns
else:
    r_even = SMAWK(even_rows, pruned_columns)

// Phase 3: Candidate Set（候选列集）
// 利用 r_even 的结果为奇数行提供候选列
// 因行最小值列索引单调，奇数行 k 的 argmin 必在 [r_{k-1}, r_{k+1}] 之间

// Phase 4: Merge（合并）
// 在候选区间内 O(1) 枚举求各奇数行的最小值
for each odd row k:
    r[k] = argmin_{j ∈ [r[k-1], r[k+1]]} A[k,j]
```

复杂度递归：T(n) = T(n/2) + O(n)，其中 O(n) 来自剪枝和合并阶段，解得 T(n) = O(n)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SMAWK 有多种开源实现：Python recipe by David Eppstein（https://github.com/pombredanne/code-5），以及其他语言实现。QUIVER 使用 C++ 优化实现。SMAWK 的前提条件是矩阵必须 totally monotone，这一性质可通过基础函数 C 满足 quadrangle inequality 来保证。在 QUIVER 之外，SMAWK 也用于 concave 1D DP（如最优二叉搜索树）、序列分割、几何优化等问题。使用方式：(1) 证明目标矩阵 totally monotone（通常通过 quadrangle inequality），(2) 实现 O(1) 的矩阵元素求值函数，(3) 调用 SMAWK 获取每列的 argmin 索引。

涉及论文标题：
- Optimal and Approximate Adaptive Stochastic Quantization
