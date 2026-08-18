## 双端口计算 SRAM（Dual-Port Computing SRAM）与三型端口冲突

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
双端口计算 SRAM 是 PipeIMC 为支持流水化执行而提出的计算存储结构：在普通（单端口）计算 SRAM 上**额外增加一组 bitline 和 wordline**（类似 true dual-port SRAM），形成两个端口——原端口（memory port，执行 memory phase，空闲时也可执行计算 phase）与新端口（calculation port，只执行 calculation phase）。两个端口各有独立的多行访问支持、独立的计算外围电路（8 组 1-bit 外围电路跨 8 条 bitline 组成 8-bit 外围电路）与独立的微码 sequencer，使同一个计算 SRAM 阵列能并行执行两个算术操作（一个取数/写回、一个计算）。PipeIMC 每个 IMC 执行单元配一个 8KB（256 wordline × 256 bit）dual-port 计算 SRAM slice，支持 32 个并发线程（每 4 条 wordline 为一个物理寄存器）。选择双端口而非三端口的原因：三端口以上额外 bitline/wordline 的面积与功耗代价高，且多个 warp 共享同一前端流水，难以产生足够无冲突操作喂满更多计算端口。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
双端口在硬件层面带来的问题是**端口冲突**，论文归纳为三类（Fig.5）：(1) **write-first read-write 冲突（数据依赖）**——操作必须等其所有操作数就绪（一个端口写结果、另一端口想读该结果）；(2) **双端口写冲突**——同一 wordline 不能被两个端口同时写（与普通 dual-port SRAM 相同）；(3) **read-first read-write 冲突**——仅存在于双端口计算 SRAM：由于 in-situ 计算，操作数不能提前读出，且复杂算术（乘法 32 次迭代）在计算 phase 期间反复读取源 wordline，其他操作在此期间不能修改这些源 wordline。执行流程例子：dispatcher 发现某个端口空闲时，遍历操作表选出可调度 phase，检查其与在飞 phase 的端口冲突（Algorithm 1 第 17 行只检查数据依赖——因为重命名已消除另外两类冲突）后派发到空闲 sequencer；每周期至多派发一个 phase 以防双端口竞态（两端口同时空闲时只调度 memory port）。重命名机制保证同窗口内无数据依赖的操作永不会冲突，使调度更自由（见本库显式寄存器重命名条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：用 Cadence Virtuoso 全定制 256×256 dual-port 计算 SRAM 阵列（TSMC 40nm、1.1V、Spectre TT corner 25°C 仿真）。电路评估结果：相对 vanilla SRAM 面积 +55.7%、相对 dual-port vanilla SRAM +18.2%；静态功耗相对单端口计算 SRAM +48.1%；多行访问操作能耗相对读/写 +54.7%；多行访问频率相对 vanilla 读/写慢 2%（但仍低于分两次读单行的能量/延迟）。相对三端口阵列，双端口面积 -19.6%、静态功耗 -23%。论文援引商业 dual-port 计算 SRAM 的流片成功案例说明布线拥塞可由 EDA 流程管理，并提出 bitline 多路复用缓解布线压力、1:1 bitline-to-SA 比例。使用：作为 CPU cache 阵列中的计算 bank（每个 IMC 执行单元一个 slice），支持"计算 phase 与 memory phase 流水重叠"这一 PipeIMC 的核心思想。Vault 无专门笔记证据（omnisearch 对 dual-port SRAM 无命中）。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
