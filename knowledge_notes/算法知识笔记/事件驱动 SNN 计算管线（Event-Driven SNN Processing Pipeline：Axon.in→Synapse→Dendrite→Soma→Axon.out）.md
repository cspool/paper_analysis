## 事件驱动 SNN 计算管线（Event-Driven SNN Processing Pipeline：Axon.in→Synapse→Dendrite→Soma→Axon.out）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
事件驱动 SNN 计算管线是脉冲神经网络（SNN）在大规模神经形态平台上的基本计算模型：与 ANN 的逐层稠密前向不同，SNN 在离散时间步（1ms 生物学仿真步）内处理稀疏 spike 事件，每个时间步执行一条连续的事件驱动流水线（WaferBRAIN 论文 Fig.2a）：**Axon.in** 汇集输入事件 → **Synapse** 做稀疏权重查找与衰减（memory-bound：每个 spike 触发对其目标突触集合的访问）→ **Dendrite** 通过分段归约（segmented-reduce）累加突触后电流 → **Soma** 积分膜电位状态并判断阈值发放（compute-bound）→ **Axon.out** 通过本地广播与边界触发散射发出 spike，同时 NoC 路由包（communication-bound）。全脑规模（10^11 神经元、10^14 突触）下每步执行 trillions 级事件操作，且 compute/memory/communication 三端异质瓶颈并存——这决定了神经形态硬件必须在存储层次、互连与通信范式上协同设计（WaferBRAIN 的 NAHP + 3D-WSI + switchless dragonfly 正是为此）。生物学实时约束：人脑平均发放率 15-30Hz、1ms 时间步对应 ~3% 每步发放活动，系统须在 1ms step 预算内完成全部事件处理（WaferBRAIN 以"1ms 红线"评估可持续 firing rate）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
事件驱动管线与稀疏 spike 传播的计算过程（一个时间步，WaferBRAIN 建模；spike 传播本身可视为稀疏矩阵-稀疏向量乘 SpMSpV——见 kernel 层 SpMSpV 条目）：
```
# 每个时间步，对每个发放神经元 n_i（发放率 λ，fanout F=256）：
# 1) Axon.in：汇集本节点收到的 spike 事件（AER 包）
# 2) Synapse（memory-bound）：按 spike 的目标神经元/轴突 ID 查稀疏突触表
#    → 取 (DstNeuron, Weight) 邻接表（本地 LNid 索引 或 全局 GAid 索引）
# 3) Dendrite：segmented-reduce 按 DstNeuron 分组累加突触电流 I_syn[ds] += w
# 4) Soma（compute-bound）：膜电位 V[ds] = V[ds]·decay + I_syn[ds]
#    若 V[ds] > V_th → 发放 spike，置 refractory
# 5) Axon.out（communication-bound）：打包 spike →
#    本地目标：mode=0 广播 LNid（区域内路径复用）
#    全局目标：边界 owner 发 mode=1 单播 <POD,Wafer,Node>+GAid
# 6) 路由/传播时延计入 1ms 步预算，最拥塞路由器服务时间 T_max/Θ_router + 最长路径 δ_max 决定步时延
```
具体例子（100B 全脑模型）：95% 突触为区域内稠密连接（广播）、5% 为跨区域长程投射（单播）；每神经元每步 256 个目标（fanout），其中 ~243 本地（广播 1 次覆盖）、~13 跨区域（单播到 5 个随机区域）。评价指标：每节点路由负载 R（packet/s）、流量 T（bit/s）、每步通信时延（对比 1ms）、可持续 firing rate。对照实验：14,012 GPU 的神经仿真慢 65-118.8×、82,944 CPU 节点慢 578×——算法管线本身（LIF 等）不构成瓶颈，事件吞吐由 memory/communication 端决定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：算法侧 = SNN 模型（LIF/IF 等神经元模型、突触权重与延迟、区域化连通矩阵）+ 每步事件驱动调度（只有发放神经元触发计算，无发放则跳过）；硬件侧 = BPU 四模块流水（axon-in 查表 / dendrite DMA+分段归约 / soma 膜电位 / axon-out 打包）+ SRAM 热状态 + 3D-DRAM 突触邻接表 + NoC 广播/单播（见硬件架构库 BPU/NAHP 条目）。使用方式：生物实时交互与闭环研究（dynamic clamp、电刺激）要求 1ms 步预算内的端到端事件处理；系统设计目标 = 在固定实时步预算下最大化可仿真神经规模（容量）。WaferBRAIN 的评估：1B（单 wafer）/16B 皮层（4×4 mesh 或 14×1 dragonfly）/100B 全脑（10×10 mesh 或 14×7 dragonfly）三模型、每节点 1.30-1.49M 神经元、区域 8×8 节点网格（83-95M 神经元/区域）、firing rate 0.1%-4.9% 扫描；NAHP 在 100B dragonfly 下 1ms 界内可持续 3.8% firing rate，超过人脑 15-30Hz 对应的 ~3% 阈值。

涉及论文标题：
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
