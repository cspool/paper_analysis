## DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation

- 属于芯片设计的实现是什么？实验比较什么？
  - 实现：面向 chiplet 物理层（PHY）的端到端仿真建模。建模 AMD EPYC 风格 chiplet 架构（4 个 CCD × 8 核，中心 1 个 IOD 内含 2×2 PHY 路由器连接 8 个内存控制器），并完整建模芯片间 SerDes PHY 数据通路：QC-LDPC FEC（flit 级 128-bit + 16 parity bits，R≈0.88）→ PAM4 Gray 映射（±50/±150 mV interposer 摆幅）→ AWGN 信道（jitter 26 dB、crosstalk 20 dB 与 SNR_base 35 dB 谐和求和为 SNR_eff ≈ 19 dB）→ LLR 解调 → 迭代解码（预算 4 次）→ flit 级 ACK/NACK 重传；参数对齐 IEEE HIR 2024、UCIe 2.0（68B flit 格式、最高 32 GT/s）、PCI-SIG jitter 规格等公开规范。
  - 实验比较：(1) 固定延迟链接 HG vs PHY 建模 DICE vs 单片 Mono 的包延迟组成与 IPC（DICE 平均 IPC 偏移 6.8%、最高 27.6%）；(2) 对照真实芯片 AMD EPYC 9454P / ThreadRipper 3960X / EPYC 7R13 的 C2C 延迟 RMSE（相对 HG 减少 7.1%–10.7%）；(3) chiplet DSE：符号率（2–32 symbols/cycle，>16 收益递减）、IOD 路由器延迟（模拟 IOD 用落后制程 14nm vs CCD 5nm 的工艺差）、全局共享 vs 本地共享 LLC（AMD EPYC 式 vs Sapphire Rapids/3D V-Cache 式）对平均包延迟与执行时间的影响。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - gem5（含 Garnet 网络模型与 Ruby 内存模型），https://www.gem5.org ；对比基线 HeteroGarnet（gem5 的 chiplet 扩展，固定延迟+带宽限流模拟 SerDes）。DICE 代码开源：GitHub https://github.com/RashidAGP/DICE-Simulator ，Zenodo https://zenodo.org/records/19428665 ，arXiv:2607.24221。
- 模拟器模拟什么的性能，修改了什么。
  - 模拟 chiplet 处理器上应用的 IPC、执行时间、平均包延迟与尾延迟、pre/post-FEC flit 错误率，及 PHY 参数（SNR、符号率、FEC 码率）对上述指标的影响。修改：在 gem5 中新增 PHY 输出/接收单元、AWGN 错误注入器、LLR 计算与 layered min-sum 解码器、cut-through 流控与 flit 级重传；并用 Yosys/OpenSTA（TSMC 40nm）综合 QC-LDPC 编解码器校准逐 cycle 延迟（编码 175 cells 达 2.0 GHz；packet 级 768-bit 编码 2320 cells 不满足时序，故选 flit 级）。
- 开源情况。已开源（GitHub + Zenodo artifact，ISCA 2026 Artifact Available/Functional/Results Reproduced 三枚徽章）。
- 基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？
  - 使用：gem5 SE 模式加载 14 个 benchmark（每 CCD 一进程），或 FS 模式跑 Linux + C2C benchmark；通过 Python 配置切换 CCD/IOD 拓扑、符号率、SNR_base、jitter/crosstalk、FEC 码率与迭代预算、GS/LS LLC 等旋钮。作用：为 chiplet 架构师提供 PHY 保真的性能-可靠性联合评估——例：LLC miss 触发的 CCD→IOD 请求逐 flit 经 FEC 编码→PAM4 调制→噪声信道→迭代解码（2N+1 cycles）→失败重传后才到达内存控制器，揭示固定延迟模型抹掉的尾延迟可变性（bfs 中 HG 尾延迟 61 vs DICE 104 cycles），支撑符号率、IOD 制程、LLC 共享范围等物理/架构级设计空间探索。
