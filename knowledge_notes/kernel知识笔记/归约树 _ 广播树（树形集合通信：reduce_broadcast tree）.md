## 归约树 / 广播树（树形集合通信：reduce/broadcast tree）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在互连网络上以二叉树并行完成 N 路归约：16 宽归约 = 4 层二叉树，2^N 个节点恰好需要 2^N−1 个中间节点（每个中间节点都被利用）、每非叶节点累加两个子节点结果，深度 log2 N。广播是归约在树结构上的逆操作（根分发到叶）。LLM 里 Softmax 的 max/sum、input-split 的部分和、TP 的跨设备汇总都需要归约/广播；传统 PIM 靠 global buffer 做串行归约，带宽受限且需串行访问 bank，成为 input-split 的阻碍。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的实现：bank 为归约粒度，ArgReg 作每非叶节点的累加器；行级指令 NoC_Reduce(OP, Addr, Addr, Mask, DstBank) 由编译期按固定树模板 + bank id 实例化为各 bank 的 packet 序列；支持 4 棵并行树（64-bit Mask 决定宏参与）。Softmax 例子（16 bank）：
```
# 每 bank 本地 exp 部分和
for b in 16 banks: part[b] = sum(exp(x)) over local slice
# 4 层归约树（每层 NoC_Reduce packet，ArgReg += 子节点）
for level in 1..4: tree_reduce('+', part)      # 深度 log2(16)=4
# 根 bank 得总和 → NoC_BCast 沿树反向下发
for b in banks: softmax[b] = part[b] / total
```
相对集中式 NLU：数据无需搬出 bank 再搬回，通信与计算合流、避免根节点拥塞。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件：NoC/交换机内归约树（CompAir-NoC、NVLink SHARP 交换机内归约）；软件：ring/tree all-reduce。使用方式：input-split 映射的配套机制——归约树效率直接决定 input-split 是否可行；中间节点复用（2^N−1 个中间节点对应 2^N 叶）保证满利用率；广播与归约共享同一套树硬件（互为逆）。

Raptor 补充视角（ISCA'26，层级 collectives）：Raptor 的 all-reduce 用层级分解（hierarchical decomposition）实现——reduce-scatter + all-gather 两阶段：数据归约的主体在 chiplet 内经 on-chip NoC 局部完成，越往 MCM（D2D）与卡（PCIe）层级交换的消息越小；allgather 实现为广播，利用源侧多播（source-side multicast）能力降低高层拥塞。每 transformer 层触发两次 TP all-reduce（attention 投影后一次、FFN 后一次），各传输 O(h·b)（h=hidden dim、b=micro-batch）。collective 数据量由并行度决定：3D-DRAM 的高每卡容量使部署用更少卡/更低 TP（Llama-70B TP=4 或 1 vs SRAM 的 TP=8），参与者少、传输量小 → 对网络延迟/带宽不敏感（0.01-10µs 扫描中 SRAM 因高 TP/PP 下降最快，3D-DRAM 在 4K 上下文现实网络 0.5µs/1TB/s 下 4.38× vs HBM、3.15× vs SRAM）。MoE disaggregated 部署每层最多四种 collective：attention 组内 all-to-all 交换部分注意力输出与 log-sum-exp（~16KB/card @TP=4）、post-attention all-reduce（∝h·b）、dispatch many-to-many（数百 KB/card）与 combine many-to-many（MB/card），dispatch/combine 随激活专家数与 EP 度缩放。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis

PipeComm 补充视角（ISCA'26，拓扑感知综合中的树形 pattern）：PipeComm 的通信 pattern 构建以"有向 spanning tree"为基本单元——每个 pattern 是一棵从根节点广播/归约的有向树，树的边（链路）选择由 MILP 决策变量 x_{s,e} 决定、深度由 l_{s,v} 建模（Eq.4）；广播与归约利用对偶性（reverse 原语：交换 reduce/broadcast 方向、翻转边）互为逆操作。关键差异：①相比固定二叉树模板（log2 N 层），PipeComm 的树由求解器按拓扑异构带宽最优选边（不假设 uniform/对称）；②多个 pattern 在 II 容量约束（Σx≤II/w）下并行共存并跨迭代重叠（流水线化），而不是单棵树顺序执行；③AllReduce 用"多棵广播树 + 多棵归约树交错（interleave）"完成（3×3 2D mesh 上 II=1 可容纳 2 broadcast + 1 reduce 三个 pattern），而非经典的 ReduceScatter+AllGather 对称分解——这使 reduce 与 broadcast 相位在同一 pipeline 内重叠，比单相位 AllGather 有效提速 1.45×/1.16×。真实 GPU（16×L20）上 Pipe-Sol 的树形 schedule 平均 1.24× over NCCL。
