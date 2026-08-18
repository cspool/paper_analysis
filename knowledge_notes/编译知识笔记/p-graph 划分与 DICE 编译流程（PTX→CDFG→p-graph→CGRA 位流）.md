## p-graph 划分与 DICE 编译流程（PTX→CDFG→p-graph→CGRA 位流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
p-graph 是 DICE 编译器的程序划分单元：把从 PTX 构建的 CDFG 按四条约束切成子图，每个子图编译为一个 CGRA 配置位流；运行期动态依赖（变延迟访存、数据相关控制流）的生产-消费边被放到 p-graph 边界，交由硬件（RF、LDST Unit、PDOM 栈）外部处理，从而每个 p-graph 都是静态可分析、定长（固定 LAT）、可静态调度的。划分约束（Fig.4）：① 控制流约束——p-graph 内无 branch/jump（同基本块约束，可被谓词执行放宽）；② 访存约束——p-graph 内无 load-to-use 依赖（访存生产者-消费者分离到不同 p-graph）；③ barrier 约束——同步 barrier 终止 p-graph；④ 资源约束——资源用量须容于 CGRA 容量。逻辑链：约束①②把"变延迟访存"与"数据相关控制流"两类动态性挡在 p-graph 之外 → 每个 p-graph 有固定静态排程 → PE 间无需 FIFO/握手/弹性互连 → 静态 spatial-only CGRA 即可执行通用 SIMT 程序。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
编译流程（Fig.5）：CUDA 源码 → NVCC → PTX IR → DICE 编译器 (1) 构建 CDFG（LD/ST、branch、barrier 为边界节点；Fig.3 例：BB1/BB2/BB3 三个基本块）→ (2) 按四约束划分 p-graph → (3) 生成 p-graph IR + metadata（Table I：BITSTREAM_ADDR 32b、BITSTREAM_LENGTH 8b、UNROLLING_FACTOR 2b、LAT 8b、IN_REGS/OUT_REGS 34b 位图、LD_DEST_REGS 4×6b、NUM_STORES 3b、BRANCH_* 32b、BARRIER 1b、PARAMETER_LOAD 1b）→ CGRA mapper 对每 p-graph 做 PE 布局与静态互连布线 → 输出配置位流。编译期优化 pass：(a) predication 合并——把因发散拆开的路径用 1-bit 谓词并入单 p-graph（省配置数）；(b) 线程展开（thread unrolling）——编译器感知硬件 swizzled bank 映射 bank=(r+T) mod N_r，按输入寄存器集合静态求最大安全 unroll factor（bank 冲突时可调整寄存器分配换更高 unroll）；(c) 数据搬移指令消除——空间直连消灭 MOV/S2R，静态指令数在 NN/BFS-2/GE-1 上最多 −17.8%。结果：各 kernel 划分为 4–13 个 p-graph（Table III）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：自研编译器（不改既有开源编译框架，NVCC 仅作 PTX 前端），大部分 kernel 自动编译、个别辅以手工优化以保证架构评估不受编译器成熟度限制；产物随 artifact 开源（GitHub https://github.com/jiayi-wang98/DICE-test-collection ，Zenodo https://doi.org/10.5281/zenodo.19278715 ，dice-test-gpu-rodinia/cuda/dice_test/sw_* 提供 metadata/位流 bundle）。使用：编译产物（p-graph 划分粒度、位流、metadata）直接驱动 Accel-sim 中 CP 四阶段流水执行。定位与边界：p-graph 大小受限的根源是共享内存访问延迟不可静态预知（bank conflict/LDST stall）与频繁 barrier——共享内存访问的流水优化论文留作未来工作；scale-up（DICE-U 32 PE）收益有限（0.97×）亦因可映射 p-graph 尺寸受内存依赖与发散约束。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
