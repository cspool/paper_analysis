## Tensor Engine（TE）与 Weight Buffer（WB）及 slice/gang/chiplet 层级（Raptor 加速器逻辑层级）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Raptor 加速器的逻辑 die 层级，把计算与 3D-DRAM 内存按"计算-内存共置"组织：slice（基本局部性域）= 4×4 tensor engine（TE）阵列 + SIMD 核（辅助向量/超越函数操作）+ slice 级 SRAM 全局内存（staging 激活）+ 16 个 weight buffer（WB）+ 16 个独立 3D-DRAM channel；gang（中间执行域）= 4 个 slice 组成执行岛，协调联合处理更宽张量分片；chiplet = 4 个 gang，权重驻留（weight-stationary）chiplet，背后是 840 个 3D-DRAM bank（映射成每 slice 16、每 chiplet 256 个平衡 channel）。每加速卡集成 2-4 个 MCM，每 MCM 4 个 1.2GHz chiplet。TE 做主要乘加运算并在输出 buffer 累加部分和；3D-DRAM 存权重与 KV cache，分区成专属 channel 独立喂每个 TE 的 WB，保留跨 TE 的 bank 级并行、防止 refresh/scrub 跨 TE 停顿。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Raptor 数据流中：decode 时每 slice 的 TE 从专属 3D-DRAM channel 以 128B flit 顺序流式读权重/KV tile（stream blocking + stream flipping 编码）→ 权重进入 WB → TE 做注意力/GEMV 的 MAC 并累加到输出 buffer，激活经 slice SRAM 全局内存 staging、流式进 per-TE 输入 buffer → 部分和经 gang 内协调、chiplet 内 NoC 局部 reduce → 跨 chiplet 走 D2D（32Gbps/lane）、跨 MCM/卡走 PCIe Gen7/ESUN。层级设计的必要性：Raptor 的架构贡献（stream blocking 的 bank-group→channel 映射、stream flipping 的 F2F 接口、topology-preserving redundancy、thermal-aware refresh/ECC）都实现在逻辑-内存接口与层级调度上，slice/gang/chiplet 分解使这些机制可解释、可扩展（TE 局部权重投递 → 包级张量并行）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：逻辑 die 用 TSMC N4P，与 3D-DRAM die face-to-face（F2F）36µm-pitch µbump 键合；slice 含 4×4 TE 阵列 + SIMD 核 + 16 WB + 16 channel；chiplet 840 bank/256 channel，D2D 32Gbps/lane 连 MCM 内 chiplet。使用方式：作为"权重驻留 + channel 化访存"的加速器数据流范式——每 TE 有独立 WB 与独立 3D-DRAM channel，软件栈按 slice 把权重/KV 切成 per-channel 16KB tile；10 PFLOPS 张量计算（XPU 逻辑）配 100TB/s 3D-DRAM 构成 Raptor 产品，同一逻辑配 SRAM（150TB/s/4GB）或 HBM（18TB/s/192GB）作为外部内存基线。热设计上 TE 阵列是热点（~92°C @ 106W/chiplet，422W/MCM ÷ 4），SRAM 缓冲/跨 gang crossbar/PHY 区低 4-6°C。

涉及论文标题：
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
