## Data Flow Graph (DFG) for GPU Kernel Scheduling

术语是什么？
Data Flow Graph (DFG) 是一种有向图结构，用于表示GPU程序中各kernel之间的数据依赖关系。在HuntKTm的stream scheduler中，DFG的每个节点是一个kernel调用，有向边表示数据依赖（RAW/WAW/WAR）。DFG的构建是自动kernel并行化的基础——通过DFG可识别哪些kernel没有相互依赖，可以分配到不同CUDA stream上并发执行。DFG还支持分层（levelization）：同一层内的kernel互相无依赖，可以安全并发。

从kernel调度角度拆解术语：
HuntKTm的DFG constructor通过以下流程自动构建DFG：
1. 开发者仅在每个kernel参数列表开头插入一个常量，标注writable参数个数N_out，并将writable参数重排到前N_out位置。
2. DFG constructor逆序遍历kernel调用序列，使用BFS算法识别每个kernel的直接前驱（predecessor）。
3. 依赖判断规则：若两个kernel访问同一数据对象，且至少一个访问是写操作，则存在依赖。具体而言：
   - kernel A（在B之前执行）写data，kernel B读data → RAW依赖 A→B
   - kernel A读data，kernel B写data → WAR依赖 A→B
   - kernel A写data，kernel B写data → WAW依赖 A→B
   - 两者均只读data → 无依赖
4. 指针别名处理：通过同一base address派生的指针参数被视为访问同一数据，确保指针算术不会导致遗漏依赖。

```
构建DFG伪代码:
Function buildDFG(kernelList):
    dfg = empty DAG
    for i = kernelList.length-1 down to 0:  // 逆序遍历
        kernel_i = kernelList[i]
        for j = i-1 down to 0:  // BFS搜索前驱
            kernel_j = kernelList[j]
            for each data object d:
                if kernel_i.writes(d) OR kernel_j.writes(d):
                    if kernel_i.accesses(d) AND kernel_j.accesses(d):
                        dfg.addEdge(kernel_j → kernel_i)
                        break  // 找到直接前驱后停止
    return dfg
```

DFG分层（Levelization）：
```
level = {0, 0, ..., 0}
for each node in topological_order:
    for each predecessor p of node:
        level[node] = max(level[node], level[p] + 1)
// 同level的kernel无相互依赖，可分配到不同stream并发
```

术语一般如何实现？如何使用？
HuntKTm通过LLVM pass在host IR中定位`__cudaPushCallConfiguration`调用模式来识别kernel launch，对识别到的kernel执行DFG构建。相比GrSched的运行时动态DFG构建，编译期DFG构建可以获得全局视图，支持更激进的同步剪除优化。对复杂控制流（如kernel调用在循环或条件分支中），HuntKTm退化为保守的依赖分析。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
