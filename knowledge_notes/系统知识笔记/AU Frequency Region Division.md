## AU Frequency Region Division

术语是什么？

AU Frequency Region Division是AUM处理Variation-2 (Frequency Interference)的核心机制，将AU-enabled CPU物理核按AU使用率划分为三个独立频率区域：(1) **High-AU region C_H**：高AU usage core (prefill tokens)，频率F_H ~2.1-2.5 GHz（AMX→高功耗→TDP→频率降）；(2) **Low-AU region C_L**：低AU usage core (decode tokens)，频率F_L ~2.8-3.1 GHz（AVX为主，AMX usage低→频率降幅小）；(3) **None-AU region C_N**：无AU usage core (shared SPECjbb/OLAP)，频率F_N ~3.2 GHz。U_AU threshold基于server-level AU usage分布设定。

从系统架构角度拆解术语：

Region Division设计逻辑（避免Frequency Interference Cascade）：
1. **隔离频率惩罚**：若decode core与prefill core混在同一region→prefill高AMX触发的2.5 GHz频率降幅会惩罚decode→decode性能不必要退化
2. **保护shared apps**：None-AU region不受AU频率惩罚→shared app可按最高频率运行
3. **Core Switcher优化**：通过调整各region的core数量(C_H, C_L, C_N)并查表取对应频率(F_H, F_L, F_N)→最大化E_CPU = (α×P_H + β×P_L + γ×P_N)/W_CPU

术语一般如何实现？如何使用？

AUM做region-level频率控制（而非per-core）：频率设为each region的maximal level below TDP。Paper承认fine-grained per-core frequency capping可进一步提升效率但会显著扩大优化空间→需集成intelligent algorithms（future work）。runtime通过turbostat monitor实际频率→Core Switcher调整region分配。

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving
