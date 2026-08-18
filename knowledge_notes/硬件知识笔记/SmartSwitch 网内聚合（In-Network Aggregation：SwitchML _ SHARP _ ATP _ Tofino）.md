## SmartSwitch 网内聚合（In-Network Aggregation：SwitchML / SHARP / ATP / Tofino）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 网内聚合 = 在可编程交换机数据面（而非端主机）上执行 ML 训练集合的归约/广播，减少端到端流量。代表：SwitchML（NSDI'21，P4/Tofino 上做 AllReduce 聚合，端侧做浮点-整数格式转换，开源 p4lang/p4app-switchML）、SHARP（Mellanox 交换芯片内硬件归约，闭源、InfiniBand 专用）、ATP（Tofino 上多租户聚合）、iSwitch（FPGA 交换机，RL 场景）（Web 证据）。
- 硬件限制：Tofino 只支持定点运算（浮点梯度需转换，如 SwitchML 策略）；流水 ≤20 级（无法做 ≥50 级硬件阶段的可靠协议端点）；片上 SRAM 有限。
- DisDP 的 SmartSwitch（Edgecore Wedge100BF-32X，Tofino）：梯度 many-to-one 聚合（N 路部分梯度 → 1 份聚合梯度）+ 参数 one-to-many 广播 + 心跳聚合（heartbeat table 对 Ack/Credit 做 min 聚合）。资源：11 级流水(91.7%)、MAT 99(51.6%)、TCAM 111B(28.9%)、VLIW 1.64Kb(12.0%)、Register 37(77.1%)、SRAM 48MiB(39.5%)。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 梯度聚合流程（反向）：各 worker SmartNIC 把 bf16 梯度转整数后发包 → 交换机入口流水按集合 ID 匹配（MAT/TCAM）→ 聚合单元把该包梯度累加到 SRAM 中的中间值 → 最后一包到达时输出聚合结果到 PS 端口。参数广播（前向）：PS 发一份参数 → 交换机复制到 N 个 worker 出口端口。
- 为什么单独 AG/RS 用 SmartSwitch 无效：AG 只能把发送从 (N-1)S/N 降到 S/N、接收仍 (N-1)S/N；RS 反之——集合时间由未减小的一侧主导。DisDP 把语义改成 push/pull（聚合/广播），双向流量同时降到 S 与 S/N，才能利用全双工带宽。
- 与 SHARP 对比：SHARP 仍依赖 GPU 侧 NCCL 管数据分块 → 仍有算通干扰（DisDP 在 OPT-30B 上 2.38~3.35× ZeRO-Infinity+SHARP）；DisDP 全流程在 SmartNIC/SmartSwitch 上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：P4（TNA）+ BFRuntime 控制面（SwitchML 开源）；SHARP 为闭源 InfiniBand 交换芯片功能。DisDP 用 100Gbps 以太网 + Tofino 交换机，worker/PS 侧配 FPGA SmartNIC（U50）配合格式转换与可靠协议。使用要点：多机架时用层级交换机（ToR SmartSwitch 聚合本机架部分梯度 → 集群 SmartSwitch 再聚合），参数广播反向层级展开；支持更多 worker 只需加深层级。信息缺口：论文未给出聚合表项数上限与多集合并发度。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
