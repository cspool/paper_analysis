## Shoreline Bandwidth / Memory IO Shoreline（岸边带宽 / 内存IO岸边）

术语是什么？通过联网搜索让回答具体和精准。

Shoreline Bandwidth（也称Memory IO Shoreline）是指芯片边缘（die edge/perimeter）上用于连接外部内存（如HBM stack）的高速IO接口所提供的总带宽。由于每个HBM stack需要沿芯片边缘布置一个密集的高带宽PHY接口区域（约102.5 GB/s/mm in HBM-CO），芯片能从HBM获取的总bandwidth直接受限于chip edge（perimeter）的长度——而非芯片面积（area）。这被称为"shoreline bandwidth constraint"：memory bandwidth scales with die perimeter, not die area。

从芯片设计角度拆解：

该约束的物理机理：
1. 每个HBM stack通过~1024个数据IO引脚（组成一个宽PHY macro）与compute die通信，PHY需要布置在chip edge紧邻HBM stack的位置。
2. 在给定工艺节点下，每mm chip edge上可容纳的PHY IO密度有上限（约102.5-128 GB/s/mm for advanced packaging）。
3. Monolithic大芯片的perimeter-to-area ratio低——H100 (~814mm² die)仅~60mm有效memory IO shoreline→最多支持~6 TB/s theoretical peak（实际H100配置~3.35TB/s from 5-6 HBM3 stacks）。
4. Chiplet-based设计（如RPU）通过将单一monolithic die拆为多个小chiplet提高总perimeter-to-area ratio：相同total compute die area下，RPU暴露~600mm shoreline（vs H100 ~60mm），即约10× memory IO shoreline。

Shoreline bandwidth的另一个维度是BW/mm效率：HBM-CO达102.5 GB/s/mm（via advanced packaging），而UCIe-S达128 GB/s/mm，NVLink-GRS仅32 GB/s/mm（PCB-routed, <10mm）。

术语一般如何实现？如何使用？

在芯片物理设计中，memory IO shoreline是floorplanning的核心约束之一：architect需要在chip perimeter上分配space给HBM PHY、inter-chiplet interconnect (UCIe)、PCIe和其他IO。Advanced packaging技术（如CoWoS-S/R/L、EMIB、InFO）提供不同级别的shoreline密度和功耗/bit。RPU的chiplet架构通过将compute拆为多个边长较小的die，最大化aggregate perimeter长度并紧密coupling each die with dedicated HBM-CO stacks。

涉及论文标题：
- RPU - A Reasoning Processing Unit
