## HalfRing Algorithm（半环算法）

术语是什么？
HalfRing 是针对环（1-D torus）拓扑上 All-to-All 集合通信的带宽与延迟最优算法。通过利用双向链路的最短路径通信原则，HalfRing 在每个阶段根据收发节点间实际距离选择传输方向（而非固定双方向发送），使得配对阶段可同时利用两条方向相反的链路，消除 Ring 算法的非最短路径带宽浪费。在 N 节点环上，当 N=2k+1 时有 2k 个阶段配对为 k 对同时执行；当 N=2k 时有 2k-1 个阶段，剩余一个未配对阶段将数据等分后双向发送。HalfRing 保证无死锁（仅单跳传输）、无活锁（无绕路）、无网络争用（逐跳显式编排链路分配）。

从kernel调度角度拆解术语：
HalfRing 的逐跳 store-and-forward 调度过程（以 4 节点环为例）：

```
Procedure HalfRing_Generator(Ring_Nodes[N_nodes], Data_Size S):
    // N_nodes 为节点数；每个节点初始持有 S 大小数据，All-to-All 完成时每节点持有所有节点的数据
    if N_nodes % 2 == 1:
        Stage_Num = (N_nodes - 1) / 2   // 奇数节点：Stage_Num 对
    else:
        Stage_Num = N_nodes / 2          // 偶数节点

    comm_size = S / N_nodes
    for stage = 0 to Stage_Num - 1:
        if stage == Stage_Num - 1 and N_nodes % 2 == 0:
            comm_size = comm_size / 2    // 最后一个未配对阶段数据减半

        Sub_Stage_Num = stage            // 该阶段需 stage 次逐跳转发
        for sub = 0 to Sub_Stage_Num - 1:
            for each node in Ring_Nodes:
                // 顺时针方向：node → (node+1)%N_nodes
                Dest_CW[stage][sub] = (node + 1) % N_nodes
                // 逆时针方向：node → (node-1+N_nodes)%N_nodes
                Dest_ACW[stage][sub] = (node + N_nodes - 1) % N_nodes
                Comm_Size[stage][sub] = comm_size
            // 每个子阶段：所有相邻节点对同时执行单跳传输
```

线性成本模型性能分析（S=单节点数据量, N=节点数, B=单向带宽, α=每跳延迟）：
- Ring: 传输时间 = (N-1)/2 · S/(2B)，启动时间 = α
- HalfRing (N 偶数): 传输时间 = N/8 · S/B，启动时间 = α，加速比 1~2×
- HalfRing (N 奇数): 传输时间 = (N²-1)/8 · S/(NB)，启动时间 = α，加速比 1.5~2×

术语一般如何实现？如何使用？
HalfRing 通过离线预计算通信时间表实现——对给定拓扑的所有阶段、子阶段、节点对生成确定的发送方→接收方+转发映射表。运行时将该时间表下发到通信后端（如 MPI 的 Isend/Irecv 或 PyTorch Distributed 的 send/recv），每个节点按其时间表在对应子阶段执行单跳数据传输和转发。适用于 N-D torus 的单维 All-to-All 阶段，需配合 DimRotation 等多维调度使用。PyTorch Distributed 实现中，通信对和传输顺序预先离线计算，CPU 侧 kernel launch 开销显著降低。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks
