## Depth First Tree Traversal (DFTT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Depth First Tree Traversal（深度优先树遍历，DFTT）是 TUSQ 的计算复用模块：把 ECM 输出的待模拟电路集合（含各自频率）按"共享前缀门"组织成一棵树——节点是电路某时刻的中间态矢量，边是门；根到叶子的路径对应一条电路。用深度优先遍历计算所有叶子输出：沿边正向乘 U（compute）计算，反向乘 U†（uncompute，回滚到公共祖先）后再走另一分支，从而共享前缀计算。例如电路 U₁U_c 与 U₂U_c 共享前缀 U_c：算完第一个输出后回滚到公共节点，再乘 U₂ 得到第二个输出，公共部分只算一次。
- 渐近优势：设树边数 |E|、树高 h、叶子数 N_l，DFTT 每条边最多遍历两次，T_dftt = 2|E| = O(|E|)；naive 每条叶子都从根重走，T_naive = N_l·h = O(|E|log_b|E|)（b 为噪声通道分支数，DEP b=4、测量 b=2）。DFTT 把操作数从 O(|E|log_b|E|) 降到 O(|E|)，且不占用额外内存（不像 TQSim 缓存中间态），速度不依赖可用内存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DFTT 树遍历伪代码（图5B 的 S1/S6 例子）：
  ```
  # 树：根 a，公共前缀边 a→...→d，S1 分支 d→...→f1，S6 分支 d→...→f6
  stack = [root_state]                 # 当前态矢量
  def dfs(node):
      for child in node.children:
          stack.push(apply_gate(stack.top, edge(child)))   # compute：乘 U
          if child.is_leaf: sample_and_accumulate(stack.top, freq[child])
          else: dfs(child)
          stack.push(apply_gate_inv(stack.top, edge(child)))  # uncompute：乘 U†
  dfs(root)                            # 每条边正反各走一次 = 2|E| 次矩阵向量乘
  ```
- 复杂度推导：b+b²+...+b^h = |E| ⟹ h = log_b((b-1)|E|+b)-1，N_l = b^h = (1-1/b)|E|+1，故 T_naive = O(|E|log_b|E|)。
- 额外并行：内存富余时（如只用 25% 内存），可把根态拷贝到子树并行 DFTT（n 倍并行 n 倍内存），进一步提速。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TUSQ 中 DFTT 调度 cuStateVec kernel 在 GPU 上执行矩阵向量乘，逐边正向/反向遍历；前提是所有边对应幺正门（有逆）。对非幺正通道（mid-circuit measurement、erasure），用 DFTT+Caching：把 non-invertible 边之前的态缓存进 LIFO（容量 K，受内存约束），回滚跨非幺正边时取缓存而非求逆；同一层 MCM 合并为一条边以降低缓存需求；K=3 即可恢复 60%-100% 的 DFTT 性能（surface code 电路，d=3/5/7、p=10^-2/10^-3/10^-4）。DFTT 是无损优化，平均贡献 50.79%（最大 83.58%）的速度提升（消去 log|E| 因子）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
