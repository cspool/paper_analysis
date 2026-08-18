## In-Switch Computing（交换机内计算）与 NVLink SHARP (NVLS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
In-Switch Computing（交换机内计算，in-network computing）指在互连交换机芯片内嵌入轻量计算能力（多播复制、归约求和、匹配/聚合），使数据穿过交换机时被"就地处理"，从而消除跨链路的冗余搬运。NVLink SHARP（NVLS）是其多 GPU 商用形态：NVIDIA 自 Hopper 世代起在 NVSwitch 上集成计算单元（学术原型为 ISCA 2020 Klenk 等的 in-network shared-memory collectives 架构，论文参考文献 [19]），提供 multimem 指令族——multimem.st（in-switch multicast，加速 AllGather）、multimem.ld_reduce（in-switch reduction，加速 Reduce-Scatter）、multimem.red（原子归约）。NVLS 依赖 multimem addressing：请求包只携带一个 multimem 地址，交换机依靠预配置的 target set 决定复制目的地，以极小 header 开销实现多播/归约。其根本限制是"静态"：只支持 target set 固定、地址对称的静态集合通信，无法处理 MoE 中目标集随 token 变化、地址不对称的动态通信（本论文的核心观察）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
NVSwitch 芯片是多 GPU 节点的互联中心，NVLS 把"复制器/加法器"做进交换芯片：multimem.st 时源 GPU 只向交换机发一份数据，交换机按预配置 target set 在输出端口逐份复制多播；multimem.ld_reduce 时交换机收集各 GPU 响应、在端口 reduction buffer（本论文建模 64KB）内累加，计数归零后仅把最终结果返回发起 GPU。本论文在 GH200 NVL32-like 系统上实测/建模：DeepSeek-V3 的通信占 MoE 层执行时间 70.4%，冗余流量占总流量近 50%（topk≥8）；若强行用静态 NVLS（AllGather 仿真 Dispatch、Reduce-Scatter 仿真 Combine）会产生 340% 无用流量，反而抵消 in-switch 收益。DySHARP 的芯片级扩展：交换机 Route 模块按 target list 复制/裁剪转发、Reduction Logic 计数归零返回，仅 +1 cycle、<0.01mm²（<0.1% NVSwitch die）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用使用：sm_90+（Hopper）GPU + 第三代 NVSwitch（NVLink4 系统），PTX ISA 8.1 起提供 multimem.st / multimem.ld_reduce / multimem.red（可选 .ldsem/.stsem/.redsem 同步语义、.acc_prec 累加精度）；跨 GPU 可见性需 __threadfence_system() 或 acquire/release 语义；NVSHMEM 提供 nvshmemx_mc_ptr() 等多播地址 API；NCCL 用 NVLS 实现 AllReduce 等（github.com/NVIDIA/nccl Issue #2072/#843：NVLS 只支持包含发起者自身的 GPU 组，跨节点可与 IB SHARP 三级组合）。研究线：SHARP（IB 树内归约）、CAIS、TRACI（DLRM 动态通信）等；DySHARP 将 NVLS 的 multimem 寻址动态化，是该方向首个支持 MoE 动态通信的工作。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
