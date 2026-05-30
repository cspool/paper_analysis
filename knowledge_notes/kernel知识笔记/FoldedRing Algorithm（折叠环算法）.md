## FoldedRing Algorithm（折叠环算法）

术语是什么？
FoldedRing 是针对环拓扑上单链路故障的容错 All-to-All 算法。当环上某链路故障时，FoldedRing 利用故障链路两端节点之间的所有反向（逆时针）物理链路构建逻辑补偿连接——将这些反向链路"折叠"为故障链路的替代路径。结合其余健康的顺时针链路，恢复 Ring 算法的逻辑通信模式。FoldedRing 可扩展到其他集合通信（All-Reduce、Reduce-Scatter、All-Gather）。

从kernel调度角度拆解术语：
FoldedRing 的路径构建过程（4 节点环，节点 1-4 间链路故障）：

```
Procedure FoldedRing_Gen(Ring_Nodes, Link_state):
    N_nodes = size(Ring_Nodes)
    for stage = 1 to N_nodes - 1:
        for node = 0 to N_nodes - 1:
            dest = (node + stage) % N_nodes
            if Link[node][dest] exists:     // 直接链路存在
                FoldedRing_Comm[stage][node] = dest
            else:                           // 链路故障，构建折叠路径
                path = []
                curr = node
                while curr != dest:
                    next = (curr - 1 + N_nodes) % N_nodes  // 逆时针绕行
                    if Link[curr][next] exists:
                        path.append(next)
                        curr = next
                    else:
                        break               // 无法到达，失败
                FoldedRing_Comm[stage][node] = path
```

性能特征（单链路故障场景）：传输时间 = (N-1)/2 · S/B（Ring 的传输时间为 (N-1)/2 · S/(2B)，即 FoldedRing 性能为 Ring 的 0.5×）。启动时间 = (N-1)α（因需要建立绕行路径连接，远超 Ring 的 α）。仅能处理单维环上的单一链路故障——对于同一环上两个或更多故障，FoldedRing 无法构建折叠路径。

术语一般如何实现？如何使用？
FoldedRing 作为 MATE 调度中故障环上的基础容错传输机制使用。在正常 phase 中（MATEe 模式），FoldedRing 传输部分数据；在加速 phase 中与 HalfRing（通过健康维度链路构建的逻辑连接）并行执行。实现时需要离线预计算故障环上的 FoldedRing 通信路径表，该路径表取决于故障位置但不受数据量影响（可复用）。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks
