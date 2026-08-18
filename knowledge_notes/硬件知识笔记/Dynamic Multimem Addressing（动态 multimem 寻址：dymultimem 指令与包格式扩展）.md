## Dynamic Multimem Addressing（动态 multimem 寻址：dymultimem 指令与包格式扩展）

术语解释
DySHARP 把 NVLS 静态 multimem 寻址扩展为支持动态不规则通信的全栈方案：请求包携带"单个 multimem 地址（代数索引）+ 轻量 target expert list"，配合目的端代数-布局映射，使单个地址可表示一次操作的多个不规则目的地。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
静态 multimem（NVLS）成立的前提是 target set 固定、地址对称；MoE 的 Dispatch/Combine 目标集随 token 变化、各 GPU 布局独立，静态 multimem 不可用。DySHARP 分两层扩展：(1) 包格式——flit0 把 64-bit 地址重定义为 48-bit multimem 地址（代数索引，支持 128TB 地址空间）+ 1-bit stage（Dispatch/Combine）+ 15-bit target count，后接 target extension flits（每个 16-bit expert ID，每 flit 8 个）；(2) ISA——dymultimem.st（Dispatch 多播）/ dymultimem.ld_reduce（Combine 归约），寄存器 r2=multimem 地址（代数索引）、r1=数据操作数（.st）或归约结果（.ld_reduce）、r3=target count、r4=连续 target list 基址。对比替代方案 explicit addressing（显式嵌入全部目的地地址）：payload 效率从理想 80% 降至 69%（8 目标时），且软件须跟踪远端内存状态、预计算地址（>5% 性能损失 + 10-20% GPU 算力）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流（论文 Fig. 9 的 Step 1-9）：SM 执行 dymultimem 指令 → 进入独立 MultimemQ（32 项，与 LSQ 分离，存放已发 target 取指请求的指令）→ LSU 按 r3/r4 从 shared/global memory 取连续 target list → 组装 flit0 + target flits 请求包经片上网络出 GPU → NVSwitch 的 Route 按 OutPort^i = Target^i / 每 GPU 专家数计算输出端口、逐端口复制并裁剪 replica（仅保留发往该端口的 targets）→ 目的 GPU Hub 经 MV translation 得虚拟地址后写内存（.st）或读数据回包，交换机 Reduction Logic 记录 target 数、随部分响应递减、归零即返回归约结果（.ld_reduce）。加权归约处理：dymultimem.ld_reduce 不做加权（避免高硬件复杂度），由 GEMM-2 epilogue 先对输出乘 gate 权重，交换机做无权重加法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件接口：扩展 CUDA Runtime 的 multicast object 管理——cuDyMulticastCreate（CUDymulticastObjectProp 指定 bsize、stage、hsize 与 2 个共享代数-布局映射的区域集）与 cuDyMulticastBindAddr（指定 multimem 空间 ntoken 与各专家接收 token 数 nactive[expert]，后者由 gate 路由在 Dispatch 前给出）。target list 通常预载 shared memory 以降低取指开销。评估：纯通信性能平均 >90% ideal；vs explicit addressing 在 payload 效率、通信算子与 MoE 层三个层面全面占优。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
