## Tile Propagation (WELDER)

术语是什么？
Tile Propagation是WELDER的核心形状推断机制：给定tile-graph中output node的output tile shape，通过链式tensor expression分析自动推导所有node的input/output tile shape。数学基础：每个operator-tile的tensor expression定义output→input区域的精确映射，因此反向可推导input tile shape。对不规则访问（Gather、带stride Conv）提供conservative upper bound。

从编译框架角度拆解术语：
逆拓扑序遍历tile-graph：output tile → tensor expression仿射变换 → input tile → 传播至前驱node输出端 → 循环。示例：MaxPool output [1,1,F] → input [2,2,F] (kernel 2×2) → ReLU input [2,2,F] (1:1) → Conv input [4,4,C] (kernel 3×3, out+2=4) → weight [3,3,C,F]。

术语一般如何实现？如何使用？
Propagate通过逆拓扑序+仿射变换实现。对标准operator精确计算，不规则operator返回bounding box，reduction轴支持partition size配置。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---
