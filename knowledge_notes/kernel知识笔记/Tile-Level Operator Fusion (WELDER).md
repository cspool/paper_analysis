## Tile-Level Operator Fusion (WELDER)

术语是什么？
Tile-Level Operator Fusion是WELDER提出的一种统一融合框架，将DNN算子的融合从传统的"规则匹配（rule-based matching）"抽象为"tile-graph连接调度"。核心理念是：不限制哪些operator类型可以融合，而是通过SetConnect在tile-graph的edge上指定数据复用的memory level，Propagate自动对齐tile shape，然后由traffic cost model判断融合是否有益（减少total traffic）。如果两个reduction-based operator（如Matmul+Softmax）在shared memory level连接后，traffic减少了，WELDER就自动生成fused kernel，无需预定义的融合规则。

与Ansor/TVM的register-level element-wise融合和AStitch的shared-memory rule-based融合的本质区别：WELDER不预先限制可融合的operator类型——任何能用tensor expression描述的operator都可以参与tile-level fusion。这使WELDER自动发现89种含两个以上reduction operator的非常规融合模式，最大将48个operator融合为单个kernel。

从kernel调度角度拆解术语：
Tile-Level Operator Fusion的执行流程：

```
输入: DNN Graph G = {Conv, ReLU, MaxPool, ...}

For each edge (op_i → op_j) in topological order:
  For each memory level L in [Register, SharedMem, GlobalMem]:
    SetConnect(edge, L)            // 尝试在L层连接
    
    subgraph = ExtractSubgraph(op_i, level=0)
    // 提取所有connect_level > 0的连续边形成的连通子图
    
    // 对每个subgraph搜索最优tile配置
    for subtile in EnumerateSubtiles(subgraph):
      config = Propagate(subgraph, subtile)  // 推断所有tile shape
      if MemFootprint(subgraph) > L.capacity: continue
      configs.push(config, priority=MemTraffic(subgraph))
    
    top_configs = TopK(configs, k)
    for config in top_configs:
      // 递归上层sub-graph调度
      for node in subgraph.nodes:
        upper = ExtractSubgraph(node, level+1)
        SubGraphTiling(upper, level+1, config)
    
    latency = Min(Profile(top_configs))
    if latency < best_latency:
      best_level = L

  SetConnect(edge, best_level)  // 选择最优连接层
```

融合示例（BERT attention: Matmul Q*K + Softmax）:
- Ansor: 无法融合（两个都是reduction-based operator，无匹配规则）
- WELDER: SetConnect(Matmul→Softmax, SharedMem) → Propagate → MemTraffic评估 → 发现[16×128] tile节省69% traffic → 自动生成fused kernel

融合示例（NAFNet: pointwise Conv + Norm + pointwise Conv）:
- TensorRT: 无对应fusion rule
- WELDER: 在shared memory连接 → 自动fuse，3.09× speedup

术语一般如何实现？如何使用？
WELDER将融合问题转化为图优化问题。在code generation阶段，shared memory level的连接通过Load/Store Rewriting实现：原独立kernel的global memory load/store被改写为shared memory access，然后组合为一个fused kernel。register level的连接通过TVM compute_inline实现。用户仅需提供ONNX graph，WELDER自动完成融合决策和codegen。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---
