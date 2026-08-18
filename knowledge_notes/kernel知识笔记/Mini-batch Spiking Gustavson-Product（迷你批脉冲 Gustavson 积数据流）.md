## Mini-batch Spiking Gustavson-Product（迷你批脉冲 Gustavson 积数据流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Gustavson 积（Gustavson's algorithm，F.G. Gustavson 1978, ACM TOMS）是稀疏矩阵乘（SpGEMM）的经典行式（row-wise）公式：逐输出行计算 C(i,*) ← Σ_j A(i,j)·B(j,*)，即"对 A 的第 i 行每个非零元，把 B 对应行的缩放累加进输出行"。与内积式（inner-product，逐元素点积累加，需读满稠密权重）和外积式（outer-product，每次只算单点积、反复读写输出）相比，Gustavson 的行式累加使每个输出行（SNN 中即膜电位行）只读/写一次，显著降低访存。
- ELSA 把它适配到 SNN：SNN 里权重 4-bit、spike 1-bit、膜电位 12-bit（膜比权重大得多），因此减少膜访问收益最大。但 SNN 是异步事件驱动——spike 生成即前传、按行无规律到达；直接套用 ANN 的 Gustavson 会因行切换频繁而丧失行驻留收益。ELSA 的解决方案是 mini-batch 化：利用 Bundled AER（BAER）提供的行对齐，把同一膜行的 spike 捆成 mini-batch，每批只读一次膜行、并行累加多权重行、写回一次，在"不打同步屏障、维持 spine/token 流水"的前提下恢复 Gustavson 的低访存优势（Fig.7、Fig.23）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 单批（一个 BAER flit）的 PE 内计算流程（ELSA Fig.10）：
```
# 输入：一个 BAER mini-batch = 同行 spike { (x, y_i, q_i) }，共享行地址 x
1) 控制模块解析 spike 编码位置：x → 膜地址；{y_i, q_i} → N-way 权重 buffer 地址
2) N-way 权重 buffer 读 N 行权重 w[y_i]（负 spike 取二补码）
3) 16 输入加法树并行累加 → 膜电位行 V[x]
4) fire 组件读 spike tracer S[x]，判激发 y_t = Θ(V[x], V_thr, S[x])
5) update 组件写回膜 V[x] 与 tracer S[x+1]
# 关键：同一 x 行的全部 spike 一次读膜、一次写膜（行驻留），而非逐 spike 反复切换
```
- 例（ELSA Fig.10c）：mini-batch (0,1),(0,3) 触发读 W 第 2 行 [2,2,3,3] 与第 4 行 [1,3,1,1] → 加法树得膜行 [3,5,4,4] → fire+update。Fig.23 能量分解：inner-product 权重 buffer 能耗占 76.2%（ResNet34），outer-product 膜 buffer 占 70.3%，Gustavson 把二者合并压到 43.1%，平均比 inner 省 2.7×、比 outer 省 1.9×。
- Annotations：x 是 spike 的行地址（膜行号）、y_i 是列地址（权重行号）、q_i 极性位；N=每 mini-batch spike 数（≤PE 的加法树并行度 16）；"行驻留"= 该行膜只被读/写一次；无同步屏障是相对 TrueNorth 1kHz 全局 tick 的关键差异（ELSA 借此把吞吐从 58 GOPS 提到 4135 GOPS）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件侧：ELSA 的 PE 控制模块 + N-way 权重 buffer + 膜 buffer + tracer buffer + 16 输入加法树实现该数据流；BAER 提供行对齐的 mini-batch；每 PE 128 个 ST-BIF 电路、每周期 1024 次加法。软件/系统侧：映射算法按"层内不跨核"（partition 阶段把整层放同核）避免 spike 广播与跨 PE 归约，使 mini-batch 能保持行对齐。Web 证据：Gustavson 数据流在 GPU/加速器领域是主流稀疏方案——ZeD（ASPLOS）用 row-wise product + 稀疏累加器（SPA）工作空间、Opal（16nm CGRA）以"unioner + accumulation loop"链式实现 Gustavson 模式（最多 -79% 运行时间 vs inner-product）、RELL-STC（SIGMETRICS 2026）把 Tensor Core 改为 Gustavson 数据流（vs cuSPARSE 平均 3.54×）。区别：ELSA 针对 SNN 的"异步 + 行不对齐"问题，用 BAER 行捆绑产生 mini-batch，这是 ANN 侧没有的适配。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
