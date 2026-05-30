## Operator-Tile (WELDER)

术语是什么？
Operator-Tile是WELDER中DNN计算的最小调度单元。一个DNN operator（如Convolution、MatMul）被分解为多个同质的operator-tile，每个operator-tile接收输入tensor的一个数据tile并计算输出tensor的一个数据tile，计算逻辑由index-based tensor expression描述。与Roller/Triton的tile不同：Roller/Triton的tile是kernel内部的循环分块（intra-operator tiling），WELDER的operator-tile是跨operator的调度单元。

从编译框架角度拆解术语：
Conv operator: input[H×W×C] × weight[3×3×C×F] → output[H'×W'×F] 分解为多个 Conv-tile[i,j]，每个取input tile [3×3×C] 计算output tile [1×1×F]。tensor expression: output[1,1,f] = ΣΣΣ input[kh,kw,c] × weight[kh,kw,c,f]。关键属性：(1) input tile由output tile+tensor expression唯一确定；(2) 多tile可并行；(3) 相邻operator-tile通过shared reuse-tile连接消除中间物化。

术语一般如何实现？如何使用？
由TVM tensor expression自动推导。code generation映射为ComputeTile调用。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---
