## Ramulator 2.0（cycle-accurate DRAM 模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CMU-SAFARI 开发的现代、模块化、可扩展、快速的 cycle-accurate DRAM 模拟器（IEEE CAL 2023，arXiv:2308.11030，MIT 许可，github.com/CMU-SAFARI/ramulator2）。架构核心：(i) Interface（抽象 C++ 类）+ Implementation（具体实现类）解耦，组件经接口指针互连，换实现不改其他代码；(ii) YAML 配置 + 自注册工厂实例化；(iii) 内存控制器**插件**机制——RowHammer 缓解（PARA、Graphene、Hydra、TRR、RFM）与统计/命令 trace 分析都以 update(DRAM_CMD, ADDR) 接口的插件实现；(iv) 用字符串字面量 + 查找表定义组织层级（channel/rank/bankgroup/bank/row/column）、命令（ACT/PRE/RD/WR/REF）与时序约束（tRCD/tRAS/tRP 等），编译期 consteval 优化。支持 DDR3/4/5、LPDDR5、HBM3、GDDR6，可与 gem5 集成；经 Micron Verilog 模型校验时序合法性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 本论文将 CK-D/CK-P/CK-S/SALP 版全部实现为 Ramulator 2.0 控制器插件，在命令路径上完成计数与刷新注入。模拟配置（Table 2）：DDR4 16GB、1 通道、2 rank/通道、4 bank group、4 bank/BG、64 subarray/bank、1K 行/subarray、3200 MT/s；处理器 1/4 核 3.6GHz、4-wide、128 项指令窗口；控制器 64 项读写队列、FR-FCFS + open-row、RoBaRaCoCh 映射；LLC 每核 2MB；4KB Buddy 分配器 35% 碎片化。评估流程：62 条单核 trace（SPEC CPU2006/2017、TPC、MediaBench、YCSB，按 RBMPKI 分三档）+ 60 个四核混合（LLLL/MMMM/HHHH/LLMM/MMHH/LLHH），每核 ≥100M 指令；插件在每个 ACT 上更新 CT/抛硬币、在达阈值/命中时向命令流插入单行 ACT+PRE，模拟器逐周期推进 DDR4 时序，输出 IPC/加权加速比；能耗由 DRAMPower（drampower.info）按命令流估算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：clone 后用 CMake 构建，YAML 写组织/时序/控制器/前端配置，trace-driven 或 gem5 前端生成请求流，插件挂到 controller 的 update 回调；ColumnKeeper artifact（github.com/CMU-SAFARI/ColumnKeeper，Zenodo DOI 10.5281/zenodo.19446517）进一步提供 setup.sh + Slurm 调度 + E*.py/P*.py 一键复现 Figure 2/5-15（约 12 小时 / 300 CPU），并用 `python -m plotting.GO_eval_params` 自动重算论文 §6/§8 数值。适用场景：DRAM 标准、RowHammer/ColumnDisturb 缓解、PIM、刷新策略等内存系统研究的 cycle 级评估。DCC（ISCA'26）以修改版 AttAcc 模拟器（Ramulator 2.0 为基座）做 GPU-PIM 时序仿真：用 DRAM LD/ST 命令仿真 Host↔PIM 数据搬移成本、用后端特定 DRAM 计算命令仿真 PIM core 计算，按 HBM3 参数（tCK=0.79、tRCD=19、tRP=19、tCL=19、tCCD=4、BL=2、5.2Gbps/pin、333MHz）逐周期推进；DCC 编译器对每个 tiling draft 生成指令 trace 全量注入，端到端时间拆分为计算时间与数据重排时间。能耗侧：算术单元用 Synopsys Design Compiler + ASAP7 7nm 预测 PDK 综合、SRAM buffer 用 FinCACTI、DRAM die 数据路径按 HBM 芯片显微照片估算。DejaVu（ISCA 2026）用官方 Docker 镜像（Dockerhub: richardluo831/ramulator2）跑 perf_eval 流程：在控制器实现 PARA 与 PRAC 两种读干扰缓解并把阈值 N_RH 做成 −20%…+20% 可扫描参数（run_artifact.sh → parse_results.sh → plot_all_figures.sh），输入 57 条单核 trace（SPEC CPU2006/CPU2017、TPC、MediaBench、YCSB）构成的 60 个随机四核 mix，输出归一化系统性能随 N_RH 的变化（Fig.24/25），量化 DejaVu 引起的阈值 guardband 代价（N_RH −20% 时 PARA 平均性能开销 6.3%）。

GenZA 补充视角（ISCA'26，ZKP 加速器端到端模拟的 DRAM 建模）：GenZA 自建 cycle-accurate 模拟器（kernel 级正确性与性能经 RTL 验证），片上计算延迟按 kernel 分析模型建模，访存用 Ramulator 2.0 精确建模 2×HBM2e（共 1 TB/s）的延迟与带宽，并显式考虑计算-访存依赖推导整体性能。使用方式：电路/调度配置（门数、PE 分配、NTT 管线长度、MSM window c）+ 架构配置（128 PEs、每 PE 128 kB、1 GHz）喂入 → 逐 kernel 推进（MSM/NTT/sumcheck/Poseidon 的计算模型 + Ramulator2 的 HBM 模型）→ 输出端到端证明时间、off-chip 流量、PE 利用率；另为 NoC 分析建 packet-level 2D mesh 模拟器（喂 MSM dispatch 流量评估 stall 与 link 利用）。

PrISM 补充视角（ISCA'26，Loaded Dice 论文）：PrISM 在 Ramulator 2.0 中实现 per-bank 概率采样缓解逻辑（SSQ/SHQ/PMQ + ABO 交互），使用 Ramulator2 内部 OoO core 模型（8 核 4GHz、512-entry ROB、16MB LLC、FR-FCFS cap 1、MOP 映射，单通道单 rank 32Gb DDR5-8000B，时序按 JEDEC DDR5：tRC 48ns、tREFI 3.9µs、tRFMab 350ns/tRFMsb 190ns），并验证内部 core 模型与开源 ChampSim–Ramulator2 集成（github.com/STAR-Laboratory/PRAC_TC_ISCA25）在 3200–8000MT/s 下性能一致（<1.2%）。复现入口：github.com/STAR-Laboratory/prism 的 perf_analysis/（`./run_artifact.sh --method slurm --artifact all`，复现 Fig.7/8/9、Table VI），输入 57 个开源 workload（SPEC2006/2017、TPC、Hadoop、MediaBench、YCSB，按 RBMPKI 分档，8 核同构混合、每核 250M 指令），输出加权 speedup 与每 tREFI RFM 频率；模拟器插件在每个 ACT 上做窗口采样、SHQ 交集、PMQ 计数与 Alert 触发，逐周期推进 DDR5 时序得到 PrISM 的 0.2%（TRH-D=500）与 1.5%（250）平均 slowdown。

PVAC 评估视角（ISCA'26）：PVAC 用 Ramulator 2.0 做 cycle-accurate 性能评估并对比 5 种缓解方案。评估配置（Table VI）：4.2GHz 4 核 OoO（128-entry 指令窗口）、8MB LLC（64B 行 8 路）、MC 64-entry 读/写队列、FR-FCFS [72] + MOP4CLXOR 地址映射 [34]、DDR5-4800 2 sub-channel 1 rank、8 bank group×4 bank、64K 行/bank、1KB/chip（共 64 bank，每核 16 bank 2MB LLC）。在模拟器内实现 PRAC（原生 aggressor 计数 + PRAC 时序 tRP 36ns）、Chronus（CSA 并发计数 + 单 Alert 自适应缓解全部超阈值行）、QPRAC（proactive 阈值 NBO/2 周期 1×tREFI）、MOAT（tRFC=410ns、NMit=1、proactive 周期 4×tREFI）、PVAC（victim 计数、双 CSA 布局、proactive 阈值 NBO/2 周期 1×tREFI、优先级队列），并按 victim 侧 HC 统一重导各方案 NBO（HC 32–2048 扫描、NMit=1/2/4 各取满足 HC 的最低开销配置）。负载：SPEC CPU2006/2017、TPC、MediaBench、YCSB 按 RBMPKI 分 High/Mid/Low 三档、30 个四核混合负载；输出归一化 weighted speedup（Fig.9/11/12）、归一化平均能耗（Fig.10，CSA 能耗用 SPICE 按 [7],[22],[55],[73],[80] 标定）、每 tREFW RFM 数（Fig.11/12）。攻击场景：单核 round-robin 激活 n 行（n∈{8,32,128,512,1K,4K,8K}）、aggressor stride 1–5 扫描，HC 32–512。关键结果：PVAC 良性负载比 Chronus 高 1.3%（HC=64）至 6.9%（HC=32），攻击下最高高 29.4%（HC=64、n=128）。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting

MERIDIAN 补充视角（ISCA'26，扩展 Ramulator 2.0 的 PIM 系统级仿真）：MERIDIAN 在 Ramulator 2.0 上扩展出 32 设备去中心化 PIM 系统的 cycle-accurate 模拟器：加入 LPDDR5X 时序（t_RC=60/t_RAS=40/t_CL=23/t_RP=20/t_RCDRD=17/t_RCDWR=8，8.5 Gb/s/pin，128 channels，64GB/package）、CXL 3.0 over PCIe Gen5 ×16 链路模型（128 GB/s/链路、端到端 165ns、switch 处按活跃设备共享带宽并串行化超限传输）、每 bank 16-lane PU（1 GHz，32 TFLOPS/设备）、NMU/softmax 单元/BOOMv2 核、DAC/CEC 集群、tensor/pipeline/hybrid 映射与 ICE 交错调度/动态负载迁移；DRAM 能量把 DRAMPower 集成进 Ramulator 计算。方法学：算术单元 Verilog + Synopsys Design Compiler 28nm 综合、DRAM 内 PIM 逻辑按 10nm-class DRAM 工艺缩放并 10× 膨胀、controller 侧按 7nm、SRAM buffer 按 AttAcc 估计、Micron LPDDR5/LPDDR5X datasheet + DRAMPower 算 DRAM 能量。输入=系统配置 + 模型层/head 到设备映射 + 请求流 + KV 分片布局 + ICE 调度策略；输出=各阶段（通信/prefill/decode）周期与能耗，汇总为吞吐、每请求延迟、通信占比（≤6.34%）与能量效率（相对 CPU-GPU 7.48×/9.24×、相对 CENT/PAPI 4.48×/4.54×）。

SegFold 补充视角（ISCA'26，SpGEMM 加速器的 DRAM 后端）：SegFold 的 csegfold 模拟器在构建时经 CMake FetchContent 自动拉取 Ramulator 2.0，配置 HBM2-8Gb @ 2 Gbps 作为 offchip 内存模型，与片上 1.5 MiB cache（16-way、128B 行）一起对 SpGEMM 的 A/B/C 访存逐周期计时（bank/row 冲突、带宽），输出 simulated cycles 作为主指标；非方阵实验中还把同一 Ramulator2 HBM2 后端接入 Spada 的开源模拟器以保证内存系统公平比较（Flexagon baseline 则经 STONNE + 同一 Ramulator 后端）。使用方式是标准 Ramulator2 集成：YAML 写 HBM2 组织/时序配置，作为内存后端库嵌入 csegfold 的 request 流，而非修改其插件或控制器。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
