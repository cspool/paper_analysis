## Democratizing and Accelerating Hardware Verification with Software-Native Optimization

> 近似层次匹配说明：本文不是 GPU/加速器 kernel 论文，而是以 RTL 模拟器为后端的软件原生验证平台；按"运行时计算与调度"最接近的层次（kernel调度/运行时）归类，其核心正是把验证逻辑的运行时调度（事件循环、事务传输线程池）从模拟器内部搬到软件运行时。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：① XClock——软件声明的时钟与事件循环，作为 host 与模拟器共享的规范时间基（频率/边沿/相位），在语言原生异步运行时（asyncio、Boost.Asio）上调度软硬件交互；每轮迭代 = 处理 T0 时刻软件事件 → 提交缓冲写 → 调 HWStep(T1≥T0) 进入模拟器推进 → 返回后按需读信号、回调派发新事件 → 入队继续；T1=T0 时只推进零时相位至静止（观测组合逻辑/δ-cycle）。② XData——绑定 XClock 的时序感知数据类型，只在声明边沿/时间点传输（默认上升沿）。③ XSocket——把 TLM 事务 transport 派发到有界线程池，将同步阻塞等待转为异步阻塞，避免"模拟器等软件线程、软件等模拟器推进"的相互等待死锁。④ 符号隔离支撑单进程内多线程实例化同一设计（软件类风格动态组合）。
  - 实验比较：Q4 事件驱动时序开销（vs 裸 Verilator）；Q3 XSocket 事务吞吐（UCV+ vs UVM 共享内存 relay）；Q2 多实例可扩展性（单进程多线程 vs 多进程多实例）。
- 后端平台是什么，配置是什么。
  - 后端 = RTL 模拟器（Verilator 为主，GHDL/VCS 兼容），主机 2× AMD EPYC 7773X 64 核、16×64GB DDR4 3200 MT/s、Ubuntu 20.04.1、GCC 11.4.0；Verilator v5.034 配置 8 线程；Python 3.8.10 / Golang 1.23.4 / OpenJDK 17（artifact：Ubuntu 22.04.5、Verilator v5.026、cocotb v1.9.2、Golang v1.25.1）。
- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 UCV/Picker 运行时 + artifact 测试脚本：./scripts/A without XS.sh（quick check，Fig.11 cocotb 对比）、./scripts/A with XS.sh 与 ./scripts/B without/with XS.sh（full check，Fig.11+Fig.13 多语言，XS=XSocket 相关实验）；Docker 内 cd /home/xyl/exp。未修改模拟器与第三方库（XClock 是普通库，从既有事件循环调用）。指标：仿真吞吐 cycle/s、峰值 RSS、编译 CPU 时间、峰值内存。
- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：平台 https://github.com/XS-MLVP/picker ；artifact https://github.com/Makiras/UnityChipExp （MIT，DOI 10.5281/zenodo.19447034，Docker ghcr.io/makiras/unitychipexp:latest；8 核/64GB/20GB，2×EPYC 7773X 全量评估约 25 小时，其中 XiangShan 23 小时）。评估原理：同一组激励与检查下，软件测试线程经 XClock/XData 驱动模拟器按 T0→T1 步进（HWStep），统计周期数与墙钟时间得 cycle/s，与裸 Verilator、cocotb 对比。结果：XiangShan(3,451,036 LOC) 吞吐损失 ≤3%，CoupledL2(86,104 LOC)/RocketChip(51,721 LOC) 14%–55%（小设计每秒周期数高、软件事件调度占比大）；峰值内存主要随语言变化（JVM/解释器开销）。XSocket：NoC(13,036 LOC) UCV+ 编译 24.06s/执行 15.41s vs UVM 15.32s/18.47s，ICache(5,163 LOC) 13.69s/94.36s vs 19.14s/106.12s，约 16.6% 执行加速、验证代码量少 12%。多实例：XiangShan 单进程多线程内存较多进程降 52%，CoupledL2 收益微弱（状态存储主导）。
