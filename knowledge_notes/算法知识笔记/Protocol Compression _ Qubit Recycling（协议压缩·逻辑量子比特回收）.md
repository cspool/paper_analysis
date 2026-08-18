## Protocol Compression / Qubit Recycling（协议压缩·逻辑量子比特回收）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 把蒸馏协议（triorthogonal 矩阵）改造为更少"峰值并发活动逻辑 qubit"的等价协议，而不改变蒸馏性能（错误抑制阶数 t 与输出数 k 不变）。核心观察：triorthogonal 矩阵常含大片全零子块——偶权重行在首个 1 之前无需初始化（$f_i$ 前 idle）、在末个 1 之后（$\ell_i$ 后）可测量释放；奇权重行编码输出、一旦初始化不可释放。已释放的偶行 qubit 可"回收"给稍后初始化的行使用。衡量指标 $C(G)=\max_j |W(j)|$。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- ```
  输入: triorthogonal G ∈ {0,1}^{m×n}（前 k 行奇权重=输出，余 m−k 行偶权重=check）
  每行 i: f_i=首个 1 的列（无则 +∞），ℓ_i=末个 1 的列（无则 −∞）
  working(i,j) = (j≥f_i) 且 (偶行: j≤ℓ_i；奇行: 恒真)
  C(G) = max_{j∈[n]} |{i: working(i,j)}|
  允许变换（保持 triorthogonality）:
    列置换（重排对易的 π/8 旋转）/ 块内行置换（奇行间、偶行间）/ F_2 行加法
  目标: min C(G')，s.t. G' 与 G 等价（同蒸馏性质）
  ```
  效果：49-to-1 从 13→7 个逻辑 qubit、51-to-3CS 从 18→9、64-to-2CCZ 从 17→10，使这些协议可装进单个 gross/two-gross 块（pivot 注入方案容量上限 11 逻辑 qubit）。最优解 NP-hard（k=0、偶行权重 2 时化简为 cutwidth），实际用贪心聚类行起止 + 定向行加法（编译 <5 s）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 该方法是"协议级"优化：不依赖具体电路实现细节，对任何 triorthogonal 生成的协议（|T⟩/|CS⟩/|CCZ⟩）通用。相关并行工作（arXiv:2606.07734 "Exploring the landscape of compact magic-state distillation factories"）用经典纠错码 + SAT 求解器把 49T-to-1T 压到仅 5 个活动 qubit（mid-circuit measurement + reinitialization 回收），说明压缩方向仍是活跃前沿；其"压缩 + 检测"二分法（压缩：含多 T 门的电路完成简单任务；检测：错误产生可观测症状）是对本技术的互补视角。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
