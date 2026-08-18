## 自动数据布局搜索（Automatic Layout Search / Squeezed Layout，自动布局搜索与压缩布局）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 自动数据布局搜索是 O3LS Module 1：自动设计逻辑 qubit 数据布局（patch 放置），生成"压缩"（squeezed）布局——在保持路由连通的前提下最大化每个数据 patch 暴露的 X/Z 边缘数，同时用密度因子调节紧凑/稀疏，找到"时间步 vs ancilla 路径长度"的 sweet spot（Fig.1）。痛点：现有编译器用固定布局（compact 顺序放置、sparse/standard 稀疏放置，board 通常 10×10/9×15）——布局过大移动距离长（idle 记忆错误多、LER 高），过小并行度低（旋转瓶颈 >50% 时间步）；手动设计不规则布局困难。
- 从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 评分函数与设计流程：
  $$S(B) = C(B) \times \big(N_x(B) + N_z(B) - \alpha_e\, N_e(B)\big)$$
  C(B)=board B 上是否存在连接指定数据 patch 至少一边的路由路径（且 ancilla 的 X/Z 边缘都连到路由空间以支持 Y 测量），有=1 无=0；N_x/N_z=被路由路径连接 X/Z 边缘的 qubit 数；N_e=数据 patch 到路由空间的边缘数（惩罚项），α_e=密度因子。流程：①board 角上放 ancilla patch A；②每步生成候选 board 集 {B_i}（尝试把下一个数据 patch 放到各位置），选最高分者；③放置后做单步移动后处理（评估能否通过一步移动提升分数，如 Fig.8 Step5 重定位 q_5 增加边缘数）；④迭代到全部 qubit 放置完成。复杂度 O(n|B|)。α_e 敏感性：0.1~0.3 最优（小值偏好稀疏、大值偏好紧凑，两端都次优）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器的布局设计 pass，输出 squeezed 布局供后续 Y-synthesis/映射/调度使用。效果：相对 standard/sparse 布局空间减少 28.0%/46.7% 且不增时间步；相对大布局（12×12）LER 降最高 16.9%；与 SPARO [28] 布局对比（O3LS-1 = O3LS 布局 + SPARO 调度）LER 改善 3.05%（ancilla 路由空间降 17.35%），完整 O3LS 栈再降时间步 78.24%、路由空间 27.17%、LER 77.1%。资源估计：平均空间+时间双降 23.63%，最大空间减 44%（约省 7000 物理 qubit，d=9）。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
