## 单主机/双主机 PIM 控制接口（Single-Host / Two-Host PIM Design）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM 控制接口的两种形态（COSM §2.2 的二分法）：双主机（two-host）——PIM 单元是独立于 CPU 的"第二主机"，自带指令序列器与本地 DRAM 访问（代表 UPMEM DDR4-PIM：DPU 为多线程 in-order RISC 核、本地 IRAM/WRAM、DMA 搬运），PIM 操作期间 CPU 对该 bank 的访问被阻塞以防 DRAM 状态损坏，完成靠轮询状态寄存器，结束后内存控制器须重同步 DRAM → CPU/PIM 切换开销大。单主机（single-host）——PIM 单元用扩展 DRAM 命令精确控制（GDDR6-AiM、HBM2 FiM、AsyncDIMM、Chopim 等），配合翻译表把高层操作映射为命令；内存控制器全程知晓 DRAM 状态，集中式调度允许 CPU/PIM 命令细粒度交错，无需保守时序或轮询。COSM 基于 §3.1 观察选择单主机（CPU 对内存延迟敏感，须快速抢占 PIM；双主机切换延迟大、粗粒度命令阻碍及时 CPU 访问）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
COSM 的单主机并发时序例子（LPDDR5-6400，2ch×2rank）：ACT 打开 bank 行 → 过 tRCD 发 PIM_Exec → PIM 单元按 tCCD 自主逐列执行（无需再发触发命令）→ CPU 请求到达该 bank → Command Arbiter 发 PIM_Pause，PEE 完成当前列（≤tCCD）后冻结状态、释放总线 → CPU 访问完成 → 仲裁器重发 PIM_Exec，PIM 从冻结列地址恢复执行。双主机下同一场景须等整个 PIM 任务完成 + 轮询 + 重同步，CPU 延迟不可控。单主机的代价：命令编码空间有限（依赖预留/RFU 编码与翻译表），灵活性低于自带指令序列器的双主机。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DRAM 命令集扩展（模式寄存器、RFU 命令编码）+ 内存控制器内 PIM 队列/调度器；COSM 的命令族 = PIM_Exec（载入型 L/存储型 S，执行长度 nPTL）+ PIM_Pause + 带宽解耦传输命令（PIM_RdBuf/WrBuf、PIM_LdBuf/StBuf），时序约束：ACT→PIM_Exec/PIM_LdBuf/StBuf 需 tRCD；PIM_Exec→PIM_Pause 需 tCCD；PIM_Pause(Ld)→PRE 需 tRTP；PIM_Pause(St)→PRE 需 tCCD+tWR。使用方式：CPU-PIM 并发执行、细粒度抢占（见本库"可抢占 PIM 执行命令"）；对 SIMD/all-bank PIM 扩展到单主机需处理多 bank 同步（COSM 留作未来工作）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
