## Y 算子合成（Y-Synthesis / Pauli Operator Cancellation，Y 算子分解与抵消）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Y-Synthesis 是 O3LS Module 2 的电路合成算法：把 Pauli-Y 算子分解为 X/Z 组合并挖掘 Pauli 算子抵消机会，减少 lattice surgery 操作数。动机：测量 Pauli-Y 需要同时访问 X 与 Z 算子，但在 squeezed/紧凑布局中 X/Z 不能同时访问，必须把 Y 分解为等价 X/Z 旋转组合；此前编译器（[32][52]）固定按 [Z_{π/4}⊗(Z^{⊗N−1})_{π/4}](X^{⊗N})_{−π/8}[Z_{−π/4}⊗(Z^{⊗N−1})_{−π/4}] 分解偶数个 Y，错过分解选择带来的门吸收抵消机会（Fig.6）。O3LS 改为按"每组奇数个 Y"二分分组，构造左/右 Z 旋转算子，检查分组能否被前驱/后继算子吸收，选吸收算子最多的分组（Fig.9）。
- 从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Algorithm 1（Y-synthesis）伪代码：
  ```
  输入：Pauli 算子序列 S={P_1..P_l}，输出：合成后序列 S'
  1 初始化 S'={}
  2 for P_i in S do
  3    取 P_i 的 Y 下标集合 Y_indices_i
  4    if size(Y_indices_i)==0 then S'.append(P_i); continue
  5    else if size(Y_indices_i) 为奇数 then b_1=Y_indices_i, b_2=∅
  6    else 找 Y_indices_i 的二分 b_1,b_2（各含奇数个 Y）
  7    按 b_1,b_2 构造左 Z 旋转算子 L_i^{(1)},L_i^{(2)} 与右 Z 旋转算子 R_i^{(1)},R_i^{(2)}
  8    分解 P_i 得无 Y 算子 P_i'，S'.append(L_i^{(1)},L_i^{(2)},P_i',R_i^{(1)},R_i^{(2)})
  9  end for
  10 对 S' 做 Pauli 算子合成（合并相邻同基算子，实现抵消）
  11 return S'
  ```
  关键在 Step 10：对每个分组检查其派生的算子能否被前驱算子吸收（输入拓扑排序，前驱已完成 Y 分解）；对后继算子检查是否存在潜在 Y 分解机会使分组可被吸收。每算子复杂度 O(n)（检查 n 个前驱 + n 个后继），总体 O(nl)。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器的合成 pass，与 O3LS-IR（PDAG）配合：Pauli 算子为节点，依赖为边，记录旋转角/Pauli word/前驱/后继，入度为 0 的节点可执行。效果：单独 Y-synthesis（O3LS-2）相对 prior passes 时间步降 18.33%、LER 降 18.30%；叠加松散调度（O3LS-2+3）达 37.74%/34.34%；叠加 EA 映射（O3LS-2+3+4）达 38.62%/35.17%（ablation Fig.21）。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
