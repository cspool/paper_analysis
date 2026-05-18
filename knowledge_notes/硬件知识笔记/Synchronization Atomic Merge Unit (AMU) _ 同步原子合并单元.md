## Synchronization Atomic Merge Unit (AMU) / 同步原子合并单元

术语是什么？

Synchronization Atomic Merge Unit (AMU) 是LRM-GPU在每个chiplet网络中嵌入的硬件模块，用于在network内检测和合并跨chiplet的同步atomic请求，减少inter-chiplet atomic同步流量。AMU由四部分组成：(1) Merge Table——包含多个entries，每entry含Status (valid/reserve/invalid)、Opcode (原子操作类型和操作掩码)、Address、SM list (记录发出相同地址atomic请求的SM ID，最多8个)、Data (32B)等字段。key (status/opcode/address)使用CAM存储以支持单周期并行匹配，data使用SRAM存储，CAM和SRAM均采用dual-port以支持并发读写。(2) Instruction Decoder——识别atomic操作类型和操作掩码。(3) ALU——执行atomic请求的合并计算（如两个atomicAdd(addr,1)合并为atomicAdd(addr,2)）。(4) Multicast Unit——在响应返回时根据SM list向所有参与SM广播atomic操作结果。

从硬件架构角度拆解：

AMU的硬件微架构和执行流程（以4-chiplet GPU为例）：

1. **配置参数**：每chiplet一个AMU，16 channels并发处理，merge table 2K entries/16 banks (128 entries/bank)，每entry SM list深度8，data 32B，total area 1.84mm²/power 301.44mW (TSMC 40nm)。

2. **请求流程**：
   - SM发出atomic请求 → 进入network → 判断是否为跨chiplet请求（目标地址在远端chiplet的LLC/memory partition）
   - 若是跨chiplet → 路由到AMU；否则直接发送。
   - AMU内：查询merge table CAM（匹配status=valid + opcode可合并 + address相同）
   - 命中：ALU合并数据（如data_new = data_entry OP data_request），SM list追加当前SM ID
   - 不匹配但有free entry：分配新entry，设置status=valid，启动countdown timer
   - 不匹配且无free entry：请求直接发送（绕过AMU，不记录）
   - Timer到期或SM list满：entry中的合并请求发送到LLC → status变为reserve（禁止继续合并，等待响应）

3. **响应流程**：
   - AMU接收来自LLC的atomic响应 → 以响应地址查询merge table
   - 命中reserve entry：通过multicast unit向SM list中所有SM广播响应 → entry释放（status=invalid）
   - 未命中：响应直接转发给目标SM

4. **合并规则**：atomicAdd/Sub/Min/Max/And/Or/Xor可自由合并（可交换/无序）；atomicCAS仅在比较数据相同时才允许合并（选一个请求作为combined request发送到LLC，其余请求等待返回fail结果）；跨cache line对齐的同一coarse-grained地址区域内不同偏移的请求可按operation-mask合并为一个transaction。

5. **面积功耗分解**（TSMC 40nm, 1GHz, 1.1V）：merge table: 185.51mW/1.52mm² (定制电路, Cadence Virtuoso)；其他逻辑(decoder+ALU+multicast): 115.93mW/0.32mm² (Verilog RTL + Synopsys DC综合)；合计301.44mW/1.84mm²。

术语一般如何实现？如何使用？

AMU的实现和使用要点：(1) 嵌入位置——AMU位于chiplet内部network中，在crossbar或router后、inter-chiplet link前，拦截所有出/入chiplet的atomic请求。不同于ARC[11]和LAB[10]在SM/warp内合并atomic（不跨SM），AMU利用跨SM的atomic locality在网络级合并。(2) Merge table设计——CAM+SRAM dual-port是高并发需求的关键：GPU大量SM同时发atomic请求，单端口merge table会成为瓶颈。16 banks × dual-port提供32个并发读写通道。(3) Countdown timer——控制合并窗口长度的关键参数：太短合并机会少，太长增加单个atomic请求等待时间。论文未给出最优timer值或timer参数灵敏度分析。(4) 与cache coherence的交互——AMU合并后的atomic响应需正确广播到所有参与SM，确保每个SM都收到其atomic操作的结果。对于atomicCAS，只有combined request实际执行，其余请求从multicast获得失败结果。(5) 适用性——AMU对高竞争同步场景收益最大（多SM竞争同一lock或更新同一共享变量产生大量同地址atomic请求）；对低竞争或无同步场景几乎无额外开销（AMU只在检测到跨chiplet atomic时才介入）。与LRM-GPU的LRC机制互补：LRC减少L1.5 coherence overhead，AMU减少inter-chiplet atomic bandwidth pressure。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

