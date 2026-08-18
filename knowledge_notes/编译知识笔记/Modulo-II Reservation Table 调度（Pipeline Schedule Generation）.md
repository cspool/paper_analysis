## Modulo-II Reservation Table 调度（Pipeline Schedule Generation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modulo-II Reservation Table（II 模保留表）是 PipeComm 管道调度生成阶段（Algorithm 1）的核心数据结构：为每条网络链路维护一个按 II 取模的时间槽占用表 RT[link][0..II-1]，调度器据此为每个数据传输分配无冲突的稳态时隙，保证跨迭代（跨 chunk）重叠执行时链路资源不被超订。这是经典 modulo scheduling（Rau & Glaeser 1981，VLIW 编译器）在通信领域的直接类比：软件流水线中每硬件资源一个 modulo reservation table、迭代以 II 间隔重叠发射；PipeComm 中每条链路即"资源"、每个 chunk 的传输即"迭代"、II 即启动间隔。与经典 modulo scheduling 的区别：通信中无跨迭代数据依赖，调度器无需复杂依赖消解（只有 pattern 内的结构性拓扑序依赖），因此推迟（deferral）只平移执行偏移、不会产生循环逻辑停顿，算法必然收敛。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Algorithm 1 伪代码（heap-based scheduler）：
```
输入: R=通信 pattern 集, II=启动间隔; 输出: step(每数据传输的执行步)
Heap 初始化为各 pattern 的根节点（depth=0）
while Heap 非空:
    (s, node, depth) ← Heap.pop()          # 取当前最小 depth 的待调度节点
    for 该节点在 pattern s 中的每条出边 link:
        if (s, link) 未分配 step:
            if RT[link][depth mod II] 空闲:   # 检查 II 模相位
                step(s, link) ← depth
                for i in 0..w(link)-1: RT[link][(depth+i) mod II] ← 占用  # w=链路传输延迟
                if 边终点就绪: Heap.push((s, end(link), depth + w(link)))
            else: finished ← False
    if not finished: Heap.push((s, node, depth+1))   # 冲突则推迟一 depth 重入堆
```
例子（3×3 2D mesh 全 AllReduce，II=2）：调度器产出 6 步 schedule（Fig.5a），每传输静态绑定奇/偶时隙（II=2 意味着每传输独占奇或偶时隙）；II 容量约束（Eq.5）保证稳态总有足够空槽，仅在 prologue/epilogue 出现部分欠占用（Fig.5b）。RT 的模结构使跨迭代（i 与 i+1）的资源复用确定：Step 1(mod 2) 与 Step 3(mod 2) 复用同一组链路。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：C++ schedule.cpp（开源仓库 https://github.com/pku-liang/pipecomm，编译为 ./schedule），Python 前端 pipecomm.py 先跑 HiGHS ILP 生成 pattern 集，再调用 C++ 调度器产出最终 schedule。使用：作为 PipeComm 综合流程的第二阶段，输入是第一阶段 MILP/增量策略产出的 pattern 集（已在 II 容量约束下保证稳态可行），输出是每数据传输的执行步号；随后 schedule 注入 ASTRA-sim 仿真或翻译为 GPU send/recv/reduce kernel 序列执行。作用：把"无冲突的跨迭代重叠"从组合搜索问题简化为贪心堆调度，配合 MILP 的容量保证获得确定、收敛、最小化 pipeline depth 的稳态调度。

涉及论文标题：
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis
