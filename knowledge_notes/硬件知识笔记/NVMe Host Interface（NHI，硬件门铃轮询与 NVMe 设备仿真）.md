## NVMe Host Interface（NHI，硬件门铃轮询与 NVMe 设备仿真）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NHI = DPU 卡上把"DPU 背后的远端存储"仿真成本地标准 NVMe PCIe 设备给主机看的硬件逻辑（源自同组 FVM，OSDI'20 [69]）。主机侧用未修改的 NVMe 驱动即可使用；主机看到的 SQ/CQ 位于主机内存，NHI 负责硬件轮询 SQ doorbell、DMA 取命令、写 CQ 完成。
- 关键点：传统 NVMe 设备由控制器主动经 PCIe 取命令，NHI 反其道——它是"设备仿真 + 门铃监听"硬件，命令取回后不在本地消费而是转交给 NVMe/TCP Engine 发网。admin 命令（如 Identify、Get Log Page、AER、ANA 状态查询）转发板载 sidecore 软件处理，I/O 命令留在 FPGA。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 命令处理流（论文 §V-A，Fig.6 ①）：主机写 SQ 并更新 doorbell → NHI 硬件轮询检测 doorbell 变化 → DMA 从主机内存取 NVMe 命令 → admin 命令送 sidecore、I/O 命令送 Command Scheduler → ……完成路径：PDU Decapsulator 提取 NVMe completion → NHI 写主机 CQ → 主机侧获得完成。
- 资源账本：NHI 仅 36K LUT/103K FF/76 BRAM——轻量；其对"免内核驱动"的贡献：NTI 移除 kernel NHI 驱动（§V-C），使卡在 Linux/Windows/VMware ESXi/QEMU-KVM 上开箱即用，这是 UNH-IOL 认证与 NVMe Integrator's List 在列的硬件基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：门铃寄存器地址截获（PCIe BAR 映射）、命令 DMA、CQ 写回、MSI-X 中断或轮询完成通知；协议状态（已处理命令数、namespace ANA state）由硬件维护、软件并行处理 host 请求的协议操作（I/O counting、AER、ANA 状态迁移）。
- 使用场景：任何"远端资源本地化"的 DPU 前端（NVMe-oF、virtio-blk 仿真），与后端引擎（NVMe/TCP Engine/TOE）经标准化接口对接。信息缺口：论文未说明 NHI 相对 FVM 原设计的改动。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
