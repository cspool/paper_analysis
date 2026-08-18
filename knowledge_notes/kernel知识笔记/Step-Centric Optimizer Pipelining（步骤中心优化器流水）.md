## Step-Centric Optimizer Pipelining（步骤中心优化器流水）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PS 上把一次参数更新拆成固定顺序的若干「步骤」，CPU 线程按步骤分配（而非按层分配）的软件流水技术。反向阶段 6 步：①从 SmartNIC pull 2B 梯度到 CPU 内存 ②SSD 读 12B 模型状态（Adam m/v 各 4B + 主权重 4B）③CPU Adam（读 2B 梯度 + 12B 状态、17 次浮点运算/参数、写 12B 更新状态 + 2B 参数副本）④12B 状态 + 2B 参数写回 SSD ⑤push 2B 参数给 worker；前向 2 步（SSD 读 2B 参数 + push）。
- 对比 layer-centric（按层分配线程）：每层需 32 线程才够 CPU Adam 线速，并行层数受总线程数限制 → 流水气泡；step-centric 给算力密集的 Adam 步骤 32 线程、其余步骤各 1 线程，共 37 线程即满流水（layer-centric 需 104 线程）。
- 资源账（100Gbps 网络，DisDP 表 III）：前向 0 FLOPs、23.3 GB/s 内存带宽、11.6 GB/s SSD 带宽；反向 99 GFLOPS、349 GB/s 内存带宽、81.4 GB/s SSD 带宽。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 流水调度：每步骤同时只处理一层，层按 L1→L6 依次从步骤①旋转到步骤⑥；线程按步骤静态分配。伪代码：
```
threads = allocate_by_step()          # step3 (CPU Adam) = 32 线程，其余步骤各 1
for layer in layers:                  # 层顺序旋转经过各步骤
    step1.pull_grad(layer)            # 1 线程：SmartNIC -> 内存
    step2.read_state(layer)           # 1 线程：SSD -> 内存
    step3.adam(layer)                 # 32 线程：17 FLOPs/参数
    step4.write_state(layer)          # 1 线程：内存 -> SSD
    step5.push_param(layer)           # 1 线程：内存 -> SmartNIC
# 各步骤同时服务不同层 = 深度流水；吞吐 = 瓶颈步骤吞吐（Adam 32 线程达线速）
```
- 性能目标：PS 线速消费 100Gbps 聚合梯度，使吞吐瓶颈在网络而非优化器。消融：DisDP vs DisDP-LC（layer-centric 变体）1.10~1.17×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PS 进程（双 Xeon Gold 5320：1.83 TFLOPS、375 GB/s 内存带宽、PCIe Gen4）多线程 + SIMD + 循环展开的 CPU Adam（沿 ZeRO-Offload 做法）；SSD 读、CPU Adam、网卡收发三者重叠（Web 证据：DeepSpeed ZenFlow 同思路的原生进程重叠 CPU 优化器）。扩展：6730P（Gen5：2.56 TFLOPS、819 GB/s）可支撑 200Gbps。使用场景：所有「单机吃线速流式状态更新」场景（PS 优化器、流式 KV 后端）。信息缺口：论文未给出步骤间缓冲队列的实现（ring buffer/同步原语）。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
