## A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics

- 属于芯片设计的实现是什么？实验比较什么？
  - 实现：4nm 工艺硅片原型（含控制器 block 与硬件自动化每端口流水线，论文图 15 给出 floorplan 与交换机芯片显微照片），CXL 3.2 硅片配 PCIe 6.0 PHY（64G PAM4）；交换机以端口 bank（统一控制器+转换/路由逻辑）经片内非阻塞 NoC 互连，宽端口可聚合为单一高带宽接口或拆分为窄端口，支持 HBR/PBR 混合模式与多级交换机级联。
  - 实验比较：硅片实测对比传统 HBR 设计实现约 2.1× 延迟降低（跨层整合后 RTT 低于 50ns、链路带宽 +25%）；系统级对比传统 HBR 交换机（SWbasic）、MHD 直连（4N1M_private/shared）等 7 种配置，SWopt 平均延迟较 SWbasic 降 42%、p99 降 58%、吞吐最高 4.8×、64 节点近线性扩展。功耗给出 TDP ~20W（聚合 SoC 功耗），逐模块面积/功耗数据论文未明确说明。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 论文未明确说明：硅片实测 + 自研 cycle-accurate RTL emulation 校准，无公开模拟器链接。
- 模拟器模拟什么的性能，修改了什么。
  - 以硅片实测标定（总线时序、流水线深度）的 RTL 仿真模拟 CXL 3.2 控制器/交换机多主机内存池化的端到端延迟、吞吐与带宽利用率，替代当前缺失的 PCIe 6.0 商用硬件环境。
- 开源情况。
  - 论文未说明芯片 RTL 开源；Panmnesia 为商业公司，官网确认该 PBR 交换机为全球唯一 CXL 3.2 全合规 ASIC 交换机（预发布硅片 2026 年送样）。基于论文描述的使用例子：4 个主机节点经 200Gbps OSFP 直连接入 CXL 3.2 PBR 交换机与 10TB 存储节点，各节点将 CXL 内存（daxctl/mmap）映射为 PostgreSQL 17 缓存；修改版 PostgreSQL 利用 CXL.cache 硬件一致性支持跨节点并发写与跨节点缓存复用，替代传统单主节点串行写；YCSB-A 写密集下 SWopt 达 >95% 带宽利用率。
