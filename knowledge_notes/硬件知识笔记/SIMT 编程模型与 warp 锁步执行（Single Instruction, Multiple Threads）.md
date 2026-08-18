## SIMT 编程模型与 warp 锁步执行（Single Instruction, Multiple Threads）

术语解释
SIMT 编程模型：程序员编写单线程顺序代码，硬件把线程分组为 warp，同一 warp 内线程锁步执行同一条指令，同时支持发散控制流与独立访存；GPGPU 用 SIMD 后端 + active mask/mask-stack + 内存合并实现该模型。DICE 保留 SIMT 编程模型（软件兼容性）但换掉 SIMD 执行后端。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SIMT 与纯 SIMD 的关键区别：SIMT 允许线程级控制发散（不同线程走不同分支）、允许独立访存地址（无需显式 gather/scatter）。硬件实现链条：kernel 启动 → CTA/block 分配至 SM → 线程按 warp（32 线程）分组 → warp 调度器逐指令 fetch/decode/issue 广播至 32 lane → 发散路径以 active mask 屏蔽执行（mask-stack 记录重收敛点）→ warp 内同时发出的访存由 coalescer 合并为宽事务。代价（DICE 动机）：每条指令的操作数都经集中式 RF 读写，中间值多次往返 RF，还需 MOV/S2R 等数据搬移指令；RF + 控制逻辑占 SM 动态功耗 >40%（NN 上 RF 32.4%、控制 18.1%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Baseline（RTX2060S 建模配置，Table II）：34 SMs、4 subcores/SM、每 subcore 16 CUDA cores（INT/FP 可同拍执行）、1024 threads/SM、256KB RF/SM、96KB L1/SM；SM 有 160 个功能单元（64 INT+64 FP+16 LD/ST+16 SFU）但每周期至多 dispatch 128 线程（利用率上限 80%）。执行例子：NN kernel 一条 FFMA 指令 → 操作数从 RF 读出 → 32 lanes 锁步乘加 → 结果写回 RF → 下一条指令重复；发散分支 inactive lanes 空转浪费 ALU 周期。DICE 侧（同一 SIMT 程序）：线程不绑定 lane、不锁步——以 CTA 为粒度、e-block 为单位把活跃线程以 II=1 流水派进 CGRA，inactive 线程被选择性跳过（selective dispatch）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与使用：CUDA 编程模型暴露 thread/block/grid 层级与 __syncthreads() 同步，nvcc 编译后硬件负责 warp 分组、调度与发散管理。现代演进：Volta 起引入独立线程调度（ITS）+ 显式 BSSY/BSYNC 重收敛指令替代隐式掩码栈。Web sources：NVIDIA Turing Architecture Whitepaper（论文引用 [35]）；CUDA C Programming Guide。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
- PipeIMC a Pipelined In-SRAM Computing Architecture

PipeIMC 合并视角（in-SRAM 计算的 SIMT 控制方案）：PipeIMC 在 in-SRAM 计算架构中采用 in-order SIMT 控制方案组织计算 SRAM 阵列——把最后一级 cache 变成大 SIMT 风格寄存器文件，多个 warp 共享一个 schedule-fetch-decode 前端流水（每周期选一个活跃 warp 取指，操作预写入 tag array，取回后经解码缓冲交给 decoder，再按 warp ID 派遣到对应 IMC 执行单元）。与 SIMD 控制相比，SIMT 的多个独立控制流更灵活、可扩展、适合组织更大 SRAM 阵列（对比 Duality Cache 的 in-order SIMT 与 EVE 的 in-order vector/SIMD 方案）。发散处理：scheduler 借助 IMC 执行单元提供的控制流信息（分支方向/目标）与 SIMT control stack 处理 warp 分歧；硬件支持 SPLIT/JOIN 操作（操作 IPDOM 栈：SPLIT 按条件把 warp 拆分、压入含当前 PC/线程掩码/else 分支掩码的栈帧，JOIN 检查帧指向的 else 分支是否已执行、未执行则切换 else 分支否则弹出栈帧重收敛）、BARRIER（同一 control block 内 warp 同步，到达数等于操作数时释放）、FENCE（warp 内同步）、WSPAWN/TSPAWN（warp/线程激活控制）。每个 warp 32 线程、每线程一个 IMC 执行单元执行 in-SRAM 计算操作；warp 停取指直到取到的操作进入 decoder，若操作改变控制流则该 warp stall 到操作在 commit unit 提交（保证窗口内操作不回滚）。效果：SIMT 控制 + in-order issue/out-of-order execution 使 PipeIMC 相对 EVE/Duality Cache 控制 hazard stall 减少。
