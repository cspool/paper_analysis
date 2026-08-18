## BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing

- 属于芯片设计的实现是什么？实验比较什么？
  - 实现 = 在 DRAM 芯片银行级物理结构层面，将每 bank 的常规 6T SRAM scratchpad（WRAM）替换为 6T push-rule 可重构单元（Jeloka 等 JSSC'16 [30]，28 nm 流片验证）：每 cell 增加一根额外 wordline（WLL/WLR 各驱动一个存取管），灵敏放大器在差分（SRAM）与单端（CAM：BL/BLB 分测、多行同时激活）间为控制级切换（同一对晶体管两种接法），使同一 SRAM 宏片上支持 BCAM/TCAM/SRAM 动态重配置；外围仅加每 BL/BLB 一个 AND 门 + tag 锁存。不改动 DRAM subarray、灵敏放大器与外部时序接口，新增结构全部为短距离 pitch-matched 走线。实验比较：AP 化存储 vs 普通 scratchpad 的面积/能量开销（FN-CACTI 7 nm 基线 vs Design Compiler+ASAP7 综合+SPICE，1.2281×，容量按 c·p/122.81 折算）、逐 DPU 功耗 vs UPMEM 实测 150 mW TDP（按 [49] 缩放方程悲观缩至 65 nm 平面 CMOS）、频率 225–350 MHz 与 VL 灵敏度、DRAM 工艺约束（约 3 层金属、逻辑密度 1/10、pitch 4×、晶体管慢 3×）下的可集成性。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 电路级：Synopsys Design Compiler + ASAP7 PDK（https://github.com/The-OpenROAD-Project/asap7）综合 AP 链 RTL，SPICE 仿真 32x36 6T push-rule 子阵列含外围逻辑；存储基线 FN-CACTI（https://github.com/SLAM-Lab-Gatech/FN-CACTI，7 nm）。系统级：gem5（https://github.com/gem5/gem5）+ uPIMulator（https://github.com/VIA-Research/uPIMulator）。论文未给出 BAAP 芯片级 RTL/版图开源链接（无法确认）。
- 模拟器模拟什么的性能，修改了什么。
  - 电路级工具链评估 AP 存储单元相对普通 6T SRAM 的面积/能量比（1.2281×），据此把被重配置的 scratchpad 容量折算为 AP 向量长度（如 25% WRAM → VL=96）；系统级模拟器按表 IV 每指令 CAM 搜索/更新周期数评估银行级性能。论文综合/仿真的是自研 AP 链外围逻辑（tag、累加、归约、中间结果传播），系统级为 gem5 前端扩展；uPIMulator 未说明修改。DRAM 工艺可集成性为定性论证（改动局限于 DRAM 厂商本已放置 SRAM 的 bank 旁区域）+ 350 MHz 降频与 65 nm 功率缩放保守建模。
- 开源情况。
  - BAAP 芯片级 RTL/版图未开源；基座开源信息：6T push-rule 可重构单元为 Jeloka 28 nm 已流片验证（JSSC'16 [30]），UPMEM 为商用产品（SDK：https://sdk.upmem.com）。基于论文描述的使用例子：写 CSR 切换灵敏放大器差分↔单端（控制线仅换接法、无微架构状态需排空），SRAM 模式行读写作常规差分信号、可重构性对性能影响可忽略；DirectAP 模式列连续布局实现整字单周期 CAM 匹配，SIMD 模式位切片布局做跨子阵列链进位传播；单 bank AP 存 32×96 元素向量，16 芯片 DIMM 同时操作 12,288 个 32-bit 结果；功耗按 65 nm 平面 CMOS 悲观折算后 DLP 负载仍低于 150 mW/DPU。
