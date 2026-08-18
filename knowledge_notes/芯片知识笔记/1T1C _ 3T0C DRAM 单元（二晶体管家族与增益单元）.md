## 1T1C / 3T0C DRAM 单元（二晶体管家族与增益单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 1T1C 是经典 DRAM 单元：1 个访问晶体管 + 1 个存储电容，电荷存于电容、读出经电容电压/电荷传感；3T0C 是电容式（capacitor-less）增益单元（gain cell）：3 个晶体管（写晶体管、读晶体管、存储晶体管）构成，电荷直接存于存储晶体管的栅电容（gate capacitance）上、经电流传感读出，无需离散电容。3T0C 相比 1T1C 的优劣势：面积更小（无电容、密度约为 6T-SRAM 的 2 倍）但存储电荷量小、retention 短，需要更高刷新频率（硅工艺下 3T0C 因而很少实用）；而 2D 材料的超低泄漏使 3T0C 的短 retention 问题被缓解——TDMSim 中 2D-3T0C 刷新周期 0.1s（2D-1T1C 为 0.5s，硅 1T1C 为 64ms），且 3T0C 以 float node discharge 读出、32MB 时访问延迟最低。业界类比：NEO Semiconductor 的 3D X-DRAM 家族（IGZO 沟道）同样提出 1T1C/3T0C 结构，宣称 retention >450s（IGZO 低 off-current），与 TDMSim 的 2D 材料思路一致。
- 从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 TDM-Memory 中两类单元的实现差异直接影响外围电路：3T0C 的 decoder 多一个输入、driver 数量翻倍（读/写晶体管分离），1T1C 则保持标准 decoder/driver。单元级验证（Fig.7）：对 100 个流片样本测单 cell 访问延迟与能耗，与 TDM-Memory 预测吻合。芯片级流程：2D-FET 参数 → 按 cell 类型（1T1C/3T0C）计算 gate 电容（量子电容主导）、bitline 网络（含 Schottky 接触电阻 RC）、decoder/driver → 在 32MB SRAM 面积预算下 DSE：2D-1T1C 达 512MB、2D-3T0C 约 64MB；128MB 时 2D-1T1C 访问延迟与 32MB SRAM baseline 相当；2D-3T0C 因 gate 寄生电容存电荷、retention 短、刷新频繁，静态功率高于 2D-1T1C。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：1T1C = 标准 DRAM 工艺（晶体管+深槽/堆叠电容）；3T0C = 增益单元（gate 电容存储、电流传感、无电容），读出时序与 1T1C 不同（float node discharge）。TDMSim 在修改版 CACTI 中实现 1T1C/3T0C 的延迟/能量/面积模型，并分别以流片阵列（小规模）与 Hspice 参考（32MB 大规模）验证（偏差 <6%）。使用要点：选 cell 结构 = 密度 vs retention 的权衡；2D 材料低泄漏同时拉高两类单元的 retention 上限，其中 1T1C 因面积效率最高（等面积 512MB）成为论文基准配置（2D 1T1C 128）。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
