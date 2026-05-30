## Tile-Graph (WELDER)

术语是什么？
Tile-Graph是WELDER编译器提出的核心抽象——一种tile级数据流图，将DNN计算从传统的operator级dataflow graph（DFG）下沉到更细粒度的operator-tile级别。在tile-graph中，每个节点代表一个operator-tile（处理输出tensor的一个数据分区），每条边代表tile间的数据依赖。Tile-graph的关键特性是：通过SetConnect接口在边上指定数据复用的memory level（register、shared memory、global memory），通过Propagate接口从output tile shape反向推导所有input tile shape。这使得WELDER能够显式控制数据在memory hierarchy中的流动——哪些中间数据在shared memory中复用（避免DRAM往返）、哪些在register中复用（避免shared memory往返）。

tile-graph与传统DFG的本质区别：DFG中operator间通过完整tensor通信，所有中间结果必须物化在最低memory layer（如GPU global memory）；tile-graph中operator-tile间通过数据tile通信，中间tile可以在更高memory layer中复用。

从编译框架角度拆解术语：
Tile-graph在WELDER编译流程中：输入ONNX Graph → 每个operator分解为多个operator-tile → 相邻operator pair通过SetConnect建立tile级连接（指定memory level）→ Propagate从output tile shape推断所有tile shape → MemTraffic evaluation选择最优配置。示例：Conv→ReLU→MaxPool中，Conv-tile和ReLU-tile在shared memory连接，中间tile [2×2×F]直接复用，无需DRAM往返。

术语一般如何实现？如何使用？
实现在WELDER编译器内部（5.2k行代码，基于TVM/Roller/Rammer）。用户透明——输入ONNX graph，WELDER自动完成。核心API：SetConnect、Propagate、MemFootprint、MemTraffic。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---
