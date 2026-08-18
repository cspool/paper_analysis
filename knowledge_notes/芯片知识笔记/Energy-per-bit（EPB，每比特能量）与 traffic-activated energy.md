## Energy-per-bit（EPB，每比特能量）与 traffic-activated energy

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EPB（energy-per-bit，pJ/bit）是互连/存储链路传输单比特数据所消耗的能量，是评估 die-to-die、片上、内存链路的能耗密度指标。Omelet（ISCA'26）对每条链路离线计算 EPB：把 I/O 电流-电压乘积对一周期积分（∫I·V dt over 1 clock cycle），连同 latency 与带宽存入技术表，仿真时作为链路元组 ⟨W, t_cyc, E_bit⟩ 的一部分。Omelet 区分两类能量指标：intrinsic EPB（每链路平均能量，与链路激活无关，反映拓扑/材料固有特性）与 traffic-activated energy（在饱和点测量，由链路利用率决定实际激活的能量）。相关笔记：knowledge_notes/芯片知识笔记/Hybrid Bonding DRAM for 3D NMP（hybrid bonding ~0.88 pJ/b）、Die-to-die SerDes PHY 与 PAM4 调制（SerDes 链路能耗）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
EPB 把封装/链路的物理能耗转成可比较的芯片级指标。Omelet 的拆解：(1) 物理侧——HFSS 提取 RC 寄生 → SPICE 链式仿真（super-buffer 驱动，VDD=1V，FreePDK45 45nm）→ 对每链路积分得 EPB（如硅 interposer 单链路 EPB 高于有机，因硅 lossy 电学特性）；(2) 系统侧——仿真在饱和点统计每链路利用率，traffic-activated energy = Σ(利用率 × EPB × 传输 bit)。关键发现（Takeaway 3）：单看 intrinsic EPB 会误导——mesh 拓扑 per-link 最省能但拥塞激活大量短链路，总 switching activity 高；Kite 家族把流量集中到少数长距离路径，单比特能耗高但激活链路少，网络总能耗反而更低。验证数据：与 Jangam et al. SuperCHIPS 对比，100µm/2µm pitch 链路 0.287 pJ/bit（-4.33%）、500µm/10µm pitch 链路 0.35 pJ/bit（+11.75%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：EPB 由电路仿真（SPICE 积分电流电压）或实测（商用链路规格）获得；Omelet 通过 I/O driver/接收器/ESD 电路模型 + 技术表实现。使用方式：DSE 引擎把 EPB 作为评估指标之一（连同延迟/吞吐/利用率输出 Pareto 前沿），能量评估需考虑流量驱动（traffic-activated energy）而非仅链路固有值；鲁棒性验证用 EPB 扰动 β 0.8×–1.2× 确认 Pareto 前沿稳定。作用：让"单链路能耗 vs 网络级能耗"的差异成为设计变量——为拓扑选择（Kite 式少路径 vs mesh 式多路径）提供能耗维度依据，支撑低功耗 chiplet 系统的 co-design。

涉及论文标题：
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
