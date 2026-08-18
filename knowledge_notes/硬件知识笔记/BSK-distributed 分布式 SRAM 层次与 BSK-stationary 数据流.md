## BSK-distributed 分布式 SRAM 层次与 BSK-stationary 数据流

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BSK-distributed 是 CASCADE 的分布式内存策略：把全部 BSK（126 MB，足够容纳 128-bit 安全参数）分布在 12 个 HC 的私有 SRAM 中（每 HC 10.5 MB BSK buffer），替代集中式 HBM 内存层次。BSK-stationary、RLWE-flowing 数据流：BSK 作为驻留数据不动（避免移动最大数据分量），中间 RLWE 密文（ACC）在 chiplet 间流动。动机：跨 HMUX 流水线并行要求 n 个 HMUX 并发访问各自 BSK，集中式 HBM 无法支撑该带宽（Morphling 单 HBM stack ≈30W、接近 die 功耗 56%；MP-PP 使能流水后利用率仅 14.3%；MP-PP-HBM 需 8 个 stack 且每瓦效率低 3.7×）。分布式 SRAM 把并发访问分散并限制在各 chiplet 内（消除跨 chiplet BSK 访问冲突），并消除 BSK 片外搬运。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：OIFS 离线把 n 个 HMUX 划分/交错成 f(t,c) 映射并据此放置 BSK → 每个 HC 从本地 BSK SRAM 取 BSK_i 执行 HMUX（VMA 单元以 IP=256 的并行度取 256 个 BSK 系数，即内部 BSK SRAM 带宽 = IP）→ ACC 经 D2D 环形拓扑传给下游 HC（同融合组内回馈本地）→ RLWE 在环上多次循环直到 n 次 HMUX 完成 → HC0 的 VPU 做 key-switching。对比集中式：BSK 从片外 HBM 取回（MP-PP 带宽饱和）。可扩展性：片外带宽压力不随 chiplet 数增长；SRAM 容量敏感性（28-160 MB 扫描）显示容量不足被迫片外取数性能骤降、超过临界阈值（126 MB）性能达峰。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：每 HC 集成 11.5 MB SRAM（10.5 MB BSK + 768 KB 局部 + 128 KB 输入 + 128 KB 输出；HC0 为 12 MB 加存 KSK）；封装采用成熟 2.5D 无源硅中介层（避免激进封装技术，规模/复杂度在 Intel Sapphire Rapids 已验证范围内）。使用：BSK 因可跨多次 BSP 复用而适合常驻；BSK-stationary 保证 ICT 只发生在物理相邻 chiplet 间（避免通信拥塞）；容量可经增加 chiplet 线性扩展以容纳更大 BSK（无需架构重构）。评估中 126 MB 配置足以容纳参数集 III（112 MB）与 IV（90 MB）。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
