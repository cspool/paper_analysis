## 7-A_Mess_of_Memory_System_Benchmarking_Simulation_and_Application_Profiling.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Mess benchmark包含两个汇编级kernel：(1) **Pointer-chase benchmark**：在单个CPU核或单个GPU SM上运行，通过pointer-chase方式测量内存访问延迟。pointer-chase遍历随机链接的指针链，确保每次内存访问都依赖于前一次访问结果，从而精确测量load-to-use延迟。(2) **Memory traffic generator**：在其余CPU核/GPU SM上并发运行，每个核遍历两个独立数组（一个load数组、一个store数组），通过控制并发核数和数组访问强度来调节内存带宽利用率，通过load/store比例来控制读写比（100%-loads到100%-stores）。traffic generator产生复杂的内存访问模式：各数组内部顺序访问，但不同数组和核之间交叉访问，覆盖大范围row-buffer hit/empty/miss率（Intel平台: 35/43/22% 至 84/13/3%）。两个kernel均用汇编实现以避免编译器干预，数据结构分配在大页内存中以最小化TLB miss/Page walk开销，运行时通过硬件计数器监控并减去这些开销。
  - 实验比较：(1) 不同实际硬件平台的内存bandwidth-latency曲线对比：Intel Skylake/Cascade Lake/Sapphire Rapids (DDR4/DDR5)、AMD Zen2 EPYC (DDR4)、IBM Power 9 (DDR4)、Amazon Graviton 3 (DDR5)、Fujitsu A64FX (HBM2)、NVIDIA H100 (HBM2E)；(2) 不同模拟器内存模型的bandwidth-latency曲线对比 vs 实际硬件：ZSim fixed-latency/M/D/1 queue/Internal DDR/DRAMsim3/Ramulator, gem5 Simple memory/Internal DDR/Ramulator2, OpenPiton Metro-MPI；(3) 应用benchmark模拟误差（IPC error）：STREAM/LMbench/Google multichase 在 ZSim+各内存模型 vs 实际Intel Skylake，gem5+各内存模型 vs 实际Graviton 3；(4) Row-buffer statistics对比：实际Intel vs DRAMsim3 vs Ramulator；(5) CXL memory expander bandwidth-latency curves对比：Manufacturer SystemC model vs Mess simulator (ZSim/gem5/OpenPiton集成)。

- 后端平台是什么，配置是什么。
  - CPU: Intel Skylake (Xeon Platinum, 24核@2.1GHz, 6×DDR4-2666, 128 GB/s); Intel Cascade Lake (Xeon Gold, 24核@2.1GHz, 6×DDR4-2666, 128 GB/s); Intel Sapphire Rapids (Xeon Platinum, 56核@2GHz, 8×DDR5-4800, 307 GB/s); AMD Zen2 EPYC 7742 (64核@2.25GHz, 8×DDR4-3200, 204 GB/s); IBM Power 9 02CY415 (20核@2.4GHz, 8×DDR4-2666, 170 GB/s); Amazon Graviton 3 (64核@2.6GHz, 8×DDR5-4800, 307 GB/s); Fujitsu A64FX (48核@2.2GHz, 4×HBM2, 1024 GB/s)
  - GPU: NVIDIA H100 (132 SMs@1.1GHz, 4×HBM2E, 1631 GB/s)
  - 模拟器: ZSim (event-based, 24核Intel Skylake模型, 6×DDR4-2666); gem5 (cycle-accurate, 64核Graviton 3 Neoverse N1模型, 8×DDR5-4800); OpenPiton Metro-MPI (RTL, 64核Ariane RISC-V, Verilator加速); DRAMsim3; Ramulator; Ramulator2

- 评估性能的软件/脚本是什么。修改了什么。
  - Mess benchmark: C/C++ with inline assembly kernel，compilation: GCC, G++, ICX, MPI++, Python 3。使用Linux perf tool测量uncore硬件计数器获取内存带宽，可选Intel VTune和LIKWID。Mess simulator: 集成到ZSim、gem5、OpenPiton的反馈控制循环模块（C++），基于bandwidth-latency曲线解析计算内存延迟。
  - 修改：(1) ZSim集成：在原有fixed-latency/M/D/1 queue/Internal DDR模型基础上新增Mess simulator模块，通过标准CPU-memory接口连接；(2) gem5集成：在Simple memory/Internal DDR模型基础上新增Mess memory model；(3) OpenPiton Metro-MPI集成：在fixed-latency模型基础上新增Mess memory model；(4) Mess simulator作为feedback controller收敛到正确的(bandwidth, latency)工作点。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：Mess benchmark: https://github.com/bsc-mem/Mess-benchmark (MIT License); Mess simulator: https://github.com/bsc-mem/Mess-simulator (MIT License); DOI: 10.5281/zenodo.13748673
  - **评估原理与全过程**：
    1. **输入**：配置参数包括目标内存带宽利用率范围、读写比(100:0至50:50)、并发核数。
    2. **Pointer-chase kernel执行**：在Core 0/SM 0上运行。分配大页内存中的链表数组，每个节点存储下一节点地址。kernel以汇编实现，循环执行 `load [ptr]; mov ptr, [ptr]`，每次访存依赖前一次结果，串行化内存访问。硬件计数器记录未加载内存系统中的总延迟。测量unloaded latency（仅pointer-chase运行）和loaded latency（pointer-chase + traffic generator并发）。
    3. **Traffic generator kernel执行**：在其余核上运行。每个核分配独立load数组和store数组，顺序遍历。通过控制活跃核数和数组访问步长来调节带宽。生成的读写比 = load数组访问:store数组访问（写分配缓存策略下store=1次read+1次write，100%-store kernel实际产生50%read/50%write流量）。
    4. **数据采集**：Linux perf读取uncore硬件计数器（如Intel iMC counters: UNC_M_CAS_COUNT.RD/WR）获得内存带宽(x轴)，pointer-chase测得的延迟为y轴。每个(bandwidth, latency)点对应特定读写比和流量强度。
    5. **输出**：bandwidth-latency曲线族（数百个测量点），每条曲线对应一个读写比。导出.csv文件并通过Python脚本(convert.py)生成PDF图表。
    6. **Mess模拟器评估**：CPU模拟器产生内存读写操作→Mess feedback controller每秒/每1000次内存操作采样模拟带宽(cpuBW)→与当前bandwidth-latency曲线位置(messBW)比较→若不一致则按比例-积分控制器公式 `messBW_{i+1} = messBW_i + convFactor × (cpuBW_i - messBW_i)` 调整应用在曲线上的位置→从曲线读出对应Latency→减去CPU simulator已建模的core/cache/NoC延迟→将Memory部分的延迟反馈给CPU simulator作为下一窗口的内存延迟。

## 83-PAPI- Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System .pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PAPI dynamic parallelism-aware task scheduling framework——一个在线（runtime）kernel调度机制，动态将FC kernel调度到GPU PUs或FC-PIM设备。调度核心为低开销的kernel bottleneck predictor：通过估算FC kernel的arithmetic intensity (AI)来判定其是memory-bound还是compute-bound。AI ≈ RLP × TLP（RLP=Request-Level Parallelism即batch size，TLP=Token-Level Parallelism即speculation length）。调度算法：(a) **Initial Scheduling**：serving开始前，用初始RLP×TLP估算AI，与memory-boundedness threshold α比较；AI>α→FC到PUs (GPU)，AI≤α→FC到FC-PIM。α通过offline iterative evaluation确定（在PIM和PU上运行FC kernel，用不同parallelization level观察execution time确定最优阈值）；(b) **Runtime Scheduling**：每次decoding后收集所有request的<|eos|> tokens→若count>0则表示有request完成→RLP变化→重新估算AI→与α比较→决定是否reschedule FC kernel。TLP变化通过专用寄存器监测，host CPU修改TLP时通知scheduler更新。整个scheduling过程在host CPU上执行，开销极低（仅需乘法+比较+token counting）。
  - 实验比较：(1) End-to-end speedup: PAPI (dynamic scheduling) vs A100+AttAcc (static: FC always GPU, attention always PIM) / A100+HBM-PIM (static) / AttAcc-only (static PIM-only) across three LLMs, batch sizes={4,16,64}, speculation lengths={1,2,4}；(2) Sensitivity analysis: speedup with varying RLP (batch 4-128, TLP=1) and varying TLP (speculation length 1-8, batch=4) using LLaMA-65B；(3) Performance breakdown: decode-stage execution time per token for AttAcc-only vs PIM-only PAPI, showing FC kernel speedup contribution。

- 后端平台是什么，配置是什么。
  - GPU: 6 NVIDIA A100 GPUs (312 TFLOPS FP16, 1935 GB/s HBM2E bandwidth, 80GB each) with tensor cores as PUs
  - PIM: 90 HBM3 devices (5.2 Gbps/pin, 333MHz), 30 FC-PIM (4P1B, 96 banks each, 12GB) + 60 Attn-PIM (1P2B, 16GB each)
  - Interconnect: NVLink (GPU↔FC-PIM high-speed), PCIe/CXL (host↔Attn-PIM)
  - Host CPU

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于Ramulator2 + AttAcc custom simulator实现dynamic scheduling逻辑。
  - 修改：(1) 在host CPU simulator中实现token-level scheduling loop：collect <|eos|> → count → if RLP changed → estimate AI = RLP×TLP → compare α → decide GPU or FC-PIM；(2) 新增TLP专用寄存器及host CPU通知机制；(3) 实现offline α threshold determination流程（iterative evaluation on both PIM and PU under varying parallelism）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：Ramulator2开源（https://github.com/CMU-SAFARI/ramulator2）；AttAcc simulator开源[74]；PAPI scheduler实现论文未明确说明是否开源。
  - **评估原理与全过程**（以GPT-3 66B, batch=16, speculation=4为例）：
    1. **输入**：模型参数（hidden=9216, layers=64），parallelism参数（RLP=16, TLP=4）, α threshold（offline预确定≈32）。
    2. **Initial scheduling**：AI_est = RLP×TLP = 16×4 = 64 > α(32) → FC kernel识别为compute-bound → initial placement到GPU PUs。
    3. **Runtime monitoring**：Decoding iteration 1结束→收集<|eos|> tokens，假设0个→RLP保持16→不reschedule。Iteration N：3个requests完成→RLP=13，AI_est=13×4=52>α(32)→仍compute-bound→继续GPU。Iteration M：RLP降至7，AI_est=7×4=28<α(32)→memory-bound→reschedule FC kernel到FC-PIM（weight parameters通过NVLink从FC-PIM加载到PUs的流程停止，改为直接在FC-PIM in-situ计算）。
    4. **Simulation**：GPU model模拟PUs上FC GEMM延迟（基于A100 tensor core）；Ramulator2+AttAcc model模拟FC-PIM上FC计算延迟（4P1B FPU并行计算+GEMV tiling）；Attn-PIM上attention kernel始终执行，采用AttAcc的数据映射方案（K^T column-partitioned, V row-partitioned across pseudo-channels/bank-groups/banks）。
    5. **输出**：每个decoding iteration的per-token latency breakdown (FC/attention/communication/other)，end-to-end speedup和energy efficiency vs baselines。

## 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：COMET-W4Ax kernel——高性能混合精度 W4Ax GEMM kernel（约7000行 CUDA/C++）。核心优化包括：(1) **SIMT-enhanced software pipeline**：两级重叠策略——第一级隐藏 off-chip memory load 在 data transformation 和 tensor core 计算中，第二级使用双缓冲（double buffering）shared memory（buffer0 存 A0+W0，buffer1 存 A1+W1）隐藏 tensor core 计算和 data transfer/transformation。使用 async_copy_barrier 和 sync_threads 确保正确性；(2) **Weight interleaving for W4A8 GEMM**：重排 INT4 权重数据布局为交错存储（如 thread T0 加载地址 0-3 和 8-11 而非连续 0-7），消除 shared memory conflict，将 ldmatrix 指令从 2 条减至 1 条；(3) **Fast INT4→INT8 conversion**：使用 location switch（交换 W1/W2 存储位置）+ zero extension（替代 sign extension）将每值转换开销从 10 条指令降至 2 条（zero extension 乘 16 后在 scale factor 中除以 16 补偿）；(4) **Fine-grained SM scheduling**：包含 barrier minimization（仅结果写回前同步）、tile remapping（INT4/INT8 mma 计算均匀分配到各 SM）、tile decomposition（实现 one-to-many tile-to-SM binding，idle SM 通过 task-stealing 从邻近 busy SM 窃取计算任务）；(5) **Mixed-precision data layout**：激活 tensor 按 block 以不同精度编码存储，tile 大小 128×128×128 (m×n×k)，INT4 warp 形状 64×64×128，INT8 warp 形状 64×64×64（INT4 tile 的 warp 数仅为 INT8 的一半）。
  - 实验比较：(1) Kernel 性能 (batch sizes 2/4/8/16/64/256)：COMET-W4Ax vs cuBLAS-W16A16 / TRT-LLM-W4A16 / TRT-LLM-W8A8，覆盖 LLaMA 系列、Mistral-7B、Qwen2-72B、OPT-175B 各模型各层 GEMM shape；(2) Ablation：逐步移除 software pipeline / weight interleaving / fast conversion naÏve W4Ax / +remapping / COMET-W4Ax / Oracle W4A4 的 kernel latency；(3) SM scheduling ablation：W4A8 baseline / W4Ax w/o optimization / W4Ax w/ remapping / COMET-W4Ax / Oracle W4A4 的归一化 speedup。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100-80GB-SXM4 (108 SMs, 80GB HBM2e, 2.0TB/s bandwidth, 312 TFLOPS FP16 / 624 TOPS INT8 / 1248 TOPS INT4 tensor core, 78 TFLOPS CUDA core)
  - CUDA 12.1
  - Kernel profiling: NVIDIA Nsight Compute [37]
  - End-to-end profiling: NVIDIA Nsight Systems [38]

- 评估性能的软件/脚本是什么。修改了什么。
  - 基础：TensorRT-LLM v0.10.0 [40] + CUTLASS [36]（约7000行新增 C++/CUDA）
  - COMET-W4Ax kernel 编译为独立 .so 动态库，提供 Python 接口（pybind）和 C++ API
  - 修改：全新实现 W4Ax kernel，包括自定义 data layout、software pipeline 双缓冲、weight interleaving 加载、快速 INT4→INT8 conversion、SM scheduling（tile remapping + tile decomposition + task-stealing）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/rhmaaa/COMET-LLM
  - Kernel 执行全过程（以 LLaMA-3-8B, GEMM shape [B, M, K]×[N, K], batch=64, W4A4比例 75% 为例）：
    1. **数据准备**：FMPQ 量化后的 INT4 权重（interleaved layout）和混合精度激活（INT4 + INT8 blocks，k=128 分块）存储在 HBM 中。激活 tensor 按 block 精度标记，写入 COMET-W4Ax 的 tile descriptor。
    2. **Tile 划分与 SM 映射**：GEMM 按 128×128×128 tile 分解。假设共 18 个 tile，通过 tile remapping 将 INT4 和 INT8 tile 均匀分配到各 SM。每个 SM 内的 warp scheduler 根据 tile 精度发射不同数量的 warp（INT4: 64×64×128 形状 warp，INT8: 64×64×64 形状 warp，INT4 tile 的 warp 数 = INT8 tile 的一半）。
    3. **SIMT-enhanced pipeline 执行（以 SM0 上某 tile 为例）**：
       - **Iteration 0（init）**：cp.async 从 HBM 加载 A0（激活 tile）和 W0（权重 tile）到 shared memory buffer0。若 A0 为 INT8 block，CUDA core 执行 fast INT4→INT8 conversion（location switch + zero extension，2 inst/value）并将结果写回 shared memory。若需 permutation（channel permuted 层），在此阶段执行。async_copy_barrier 等待数据就绪。
       - **Iteration 1**：tensor core 从 buffer0 通过 ldmatrix 加载数据（INT4 weight：interleaved layout 单指令加载；INT8 activation：标准加载）。执行 mma 指令（INT4 mma 或 INT8 mma）。同时 cp.async 从 HBM 加载 A1+W1 到 buffer1，并行执行 dequant/permutation。sync_threads 同步。
       - **Iteration 2+**：buffer0 和 buffer1 交替作为 compute buffer / load buffer。双缓冲隐藏所有 data load + transformation 开销。
    4. **SM 间负载均衡**：INT8 tile 计算时间 ≈ 2× INT4 tile（因 INT8 tensor core 吞吐为 INT4 的一半但 warp 数加倍）。tile remapping 确保各 SM 的 INT4:INT8 tile 比例均匀。若某 SM 的 INT8 tile 较少（任务轻），其空闲 cycles 通过 task-stealing 协助邻近 SM 计算部分 tile（one-to-many binding）。
    5. **Reduction 与输出**：所有 tile 计算结果在最后 barrier 后 reduction 累加，写回 HBM。仅此一次全局同步。
    6. **性能指标**：NVIDIA Nsight Compute 采集 kernel latency (μs)，计算 TOPS = (2×M×N×K) / latency。归一化 speedup = latency_baseline / latency_COMET。典型结果：batch=256 时 2.88× over cuBLAS-W16A16。

## 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Anda架构中的三个kernel/运行时计算组件：(1) **Anda-enhanced bit-serial Processing Unit (APU)**：包含Anda PE（处理单元）+ FP Accumulator。Anda PE采用bit-serial计算方式执行变长Anda格式激活与INT4权重的点积运算。计算流程：存储符号和共享指数到内部寄存器→INT权重双缓冲加载到PE→按bit-plane顺序加载尾数（从MSB到LSB）→first-element-then-bit-plane reduction pattern：每个bit-plane内所有元素通过加法树累加得到partial sum（仅存一个partial sum per bit-plane而非全部中间结果）→各bit-plane partial sum串行累加完成点积→根据Anda尾数长度动态移位→使用共享指数转换为FP16→乘以INT权重的group-wise scale factor→跨组FP累加器累加→FP32结果转FP16输出。(2) **Runtime Bit-plane Compressor (BPC)**：16并行lane，每lane处理64个FP16值。FP Field Extractor分解符号/指数/尾数→Max Exp Catcher确定组内最大指数→计算各元素指数差值→Parallel-to-Serial Mantissa Aligner：每cycle各元素指数差减1，差为0时shift out最高尾数位（否则输出0）→打包为bit-plane aligned mantissa→Sign/Exp/Mant经Data Packager组装为bit-plane压缩格式写回。Bit-serial aligner仅需1个比较器+1个移位器（vs bit-parallel需多个），硬件面积更小。(3) **Bit-plane Data Layout Scheme**：在片上buffer中将64个Anda值的sign/mantissa/exponent按bit-plane view transpose布局（同权重的bits打包），变长尾数仅影响地址深度不影响带宽利用率。
  - 实验比较：(1) PE-level：FP-FP (FP16 Tensor Core)、FP-INT (增强Tensor Core with FP-INT单元)、iFPU (bit-serial, FP→BFP动态转换)、FIGNA (bit-parallel BFP)、FIGNA-M11 (11-bit)、FIGNA-M8 (8-bit)、Anda-M4到Anda-M13 (可变尾数4-13 bit) 的面积、功耗、面积效率 (TOPS/mm²)、能效 (TOPS/W)；(2) System-level：各架构在9个LLM模型上的speedup、area efficiency、energy efficiency（均归一化到FP-FP baseline）；(3) Energy breakdown：LLaMA-13B推理中Compute/SRAM/DRAM各部分的归一化能耗。

- 后端平台是什么，配置是什么。
  - RTL synthesis: Cadence Genus @ 16nm, 285 MHz, 0.8V
  - Cycle-accurate simulator (自研) verified against functional simulations
  - HBM2: 3.9 pJ/bit, 256 GB/s
  - MXU: 16×16 APU array, output stationary dataflow
  - 片上SRAM: Activation Buffer 1MB (Mantissa) + 0.125MB (Exponent), Weight Buffer 1MB

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研Cycle-Accurate Simulator（论文未开源），验证对象为SystemVerilog RTL综合网表的VCD仿真。
  - Power evaluation: Cadence Genus基于VCD文件的功耗分析。
  - 修改/设计：全新设计Anda PE（bit-serial dot-product with variable precision）、BPC（runtime bit-plane compressor）、bit-plane data layout memory controller。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：论文未提供simulator或RTL源码。arXiv: https://arxiv.org/abs/2411.15982。PE/system baselines对照FIGNA、iFPU等已发表工作的公开参数配置。
  - Kernel评估全过程（以LLaMA-13B, A_qkv模块, M=7, GS=64, INT4 weights为例）：
    1. **输入**: 64个Anda格式激活值（每值1-bit sign + 7-bit mantissa + 5-bit shared exponent），按bit-plane布局存储在Activation Buffer中；64个INT4权重值存储在Weight Buffer中（双缓冲）。
    2. **Anda PE bit-serial dot-product**: Cycle 0: 加载sign和shared exponent到PE寄存器，预取INT4权重到double-buffer。Cycle 1-M: 对每个bit-plane（M=7共7个bit-plane），从Activation Buffer读取64个元素的第b位尾数bit（64-bit word直接读取）→64个1-bit×4-bit乘法（1-bit AND gating on INT4）→adder tree累加64个结果得1个INT32 partial sum→存入partial sum register。下一bit-plane读取时partial sum左移1位后累加新partial sum。7个bit-plane处理完成得1个INT32 dot-product。
    3. **动态移位与格式转换**: dot_product << (M - actual_mant_used) 对齐尾数宽度 → 乘以2^shared_exponent转为FP16 → 乘以INT4 weight的group scale factor → FP Accumulator跨组累加。
    4. **BPC输出压缩**: FP32 accumulator输出→BPC的FP Field Extractor分解→确定输出Anda格式的目标尾数长度M_out→Parallel-to-Serial Mantissa Aligner执行exponent对齐和bit-serial化→Data Packager组装为bit-plane格式→写入Activation Buffer。
    5. **性能输出**: cycle-accurate simulator统计总execution cycles, compute utilization (APU active cycles / total), memory access energy (SRAM bit + DRAM bit × access count), speedup = T_baseline / T_Anda。Anda采用output stationary dataflow：权重broadcast到行方向APU，activation bit-plane broadcast到列方向APU最大化input reuse。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：InstAttention 在 CSD 的 FPGA 上实现了硬件 SparF Attention 引擎，包含以下 kernel 级组件：(1) **argtopk 单元**：从 q 向量（1×dh, FP16）中选出 top-r 绝对值最大的通道索引 i，在运行时标识稀疏通道模式；(2) **NFC Filter（4 个）**：每个 NAND Flash Controller 内嵌入 filter 逻辑，在从 flash 加载 KV cache 时执行双步过滤——第一步 page 粒度过滤（若 group 内所有条目为稀疏则跳过整页读取），第二步 token 粒度过滤（从含混合 token 的已加载 page 中只保留非稀疏条目）；(3) **Attention Kernel ×2（含 GeMV + Softmax 单元）**：两组相同的 attention kernel，每组合多个 GeMV unit 和 Softmax unit。Kernel ① 处理近似注意力分数 ŝ（步骤 8 in Algorithm 1: q[i] × K[:,i]^T→softmax→归一化）。Kernel ② 处理精确注意力输出（步骤 18-19: q × K[j,:]^T→softmax→α·s·V[j,:]+(1-α)·v̄）。两个 attention kernel 可根据实时负载调度——若 Kernel ① 先完成则并行启动 Kernel ② 的 K,V 加载；(4) **Sum/Normalization 单元**：用于 softmax 的求和和 (1-α)·v̄ 的加权补偿计算。
  - 实验比较：(1) SparF engine 各单元延迟分解：Logit (dense attention 的注意力分数计算), Attend (dense 的 V 加权), Filter (NFC 过滤开销), Logit-0 (SparF 第一步近似分数，对应算法步骤 4), Argtop-k, Sum unit 在 SparF vs Dense 下的百分比分布；(2) 解码延迟分解：KV Cache Access、Weight Access、qkvo Transfer 在不同配置（InstA/InstA-2, bs=4/64/256, dense/sparse）下的占比；(3) SparF engine 性能统计：GeMV unit 12.7 GFLOPS (real) / 13.3 GFLOPS (virtual), Softmax 14.2 MFLOPS (real), Filter 1.85 GB/s (real)。

- 后端平台是什么，配置是什么。
  - InstCSD (Daisyplus OpenSSD): Xilinx ZU17EG MPSoC FPGA + 4-core ARM, 2GB DRAM, PCIe 3.0x4。SparF engine 和 NFC filters 实现在 FPGA 部分，@285MHz。FTL 运行在 ARM 处理器上。
  - 原型平台 (Zynq7045): Xilinx Zynq7045 FPGA SoC (更经济, edge computing targeted)。DSP 资源用于最大化 attention 计算性能。
  - 虚拟化平台 (NVMeVirt): 软件定义的 InstCSD 模拟器，运行在 host CPU 上，模拟 8 个 flash channel × 1.4GB/s 带宽，PCIe 4.0x4 (7GB/s max)，用于大规模扩展性测试。
  - GPU: NVIDIA A6000 48GB VRAM, PCIe Gen4x16。

- 评估性能的软件/脚本是什么。修改了什么。
  - 框架: FlexGen + 自研 InstCSD driver（基于 [44] 定制 NVMe driver）
  - 修改: (1) FPGA 上实现 SparF Attention engine（GeMV + Softmax + argtopk + NFC filter 硬件加速器）；(2) FPGA 上实现 FTL with dual address mapping（token-indexed + channel-indexed L2P tables）；(3) ARM 上运行 FTL 软件管理 KV cache 逻辑地址到物理地址映射；(4) 编写 GPU-CSD P2P DMA driver（替换 GPU-Direct Storage 的 host filesystem 路径）；(5) 扩展 NVMe 协议：DWord10 承载 32-bit 自定义逻辑地址，控制面新增 config/attend/reclaim 命令。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源: https://github.com/ChaseLab-PKU/InstAttention
  - Kernel 评估全过程（以 OPT-13B 单个 attention head, dh=128, S=1024, r=16, k=256, 1/8 sparse, FPGA@285MHz 为例）：
    1. **输入**: q ∈ FP16^{1×128}（当前 token 的 query 向量，GPU 计算后经 P2P DMA 传输至 InstCSD）
    2. **argtopk Unit**: 接收 q → 计算 |q_i| for i=1..128 → top-16 最大绝对值通道索引 i = {i_1,...,i_16}。输出 i indices 送入 NFC 和 Attention Kernel ①。
    3. **Channel-indexed K cache 双步加载**: NFC 使用 i 索引从 flash 读取 K[:,i]（channel-indexed mapping）。第一步 coarse filtering——i 中的通道索引被分组（每 group 2-8 个连续通道对齐 4KB page），若某 group 全部通道在 q 中均为 0（非 top-r），skip 整个 flash page。第二步 fine filtering——已加载的 flash page 中逐 token 过滤稀疏条目。输出 K[:,i]（dense, 约 S×r FP16 ≈ 32KB）。
    4. **Attention Kernel ① (GeMV + Softmax)**: 读入 q[i] (16×FP16) + K[:,i] (1024×16 FP16) → GeMV unit 计算 q[i]·K[:,i]^T 得 1×1024 scores → Softmax unit 归一化 → 乘以 ||q[i]||_1/||q||_1 得 ŝ ∈ FP16^{1×1024}。
    5. **argtopk Unit (第二轮)**: 接收 ŝ → 选 top-k=256 最大 token 索引 j = {j_1,...,j_256}。输出 j 送入 NFC 和 Attention Kernel ②。
    6. **Token-indexed KV cache 双步加载**: NFC 使用 j 索引从 flash 读取 K[j,:], V[j,:]（token-indexed mapping，K cache 存两份方向）。group = 16 tokens/page (2048 FP16 = 4KB)。第一步 coarse filter——若 group 内全部 token 均为稀疏（不属 top-k），skip 整页。第二步 fine filter——从已加载 page 中过滤稀疏 token。输出 K[j,:] (256×128 FP16 ≈ 64KB), V[j,:] (256×128 FP16 ≈ 64KB)。
    7. **Attention Kernel ② (GeMV + Softmax)**: 读入 q (128 FP16) + K[j,:] (256×128 FP16) → GeMV: q·K[j,:]^T → 1×256 → Softmax → s。并行：V[j,:] prefetch 到 buffer overlapping with GeMV。GeMV: s·V[j,:] → 1×128 → 乘 α → + (1-α)·v̄ → out ∈ FP16^{1×128}。
    8. **P2P DMA 输出**: out (1×128 FP16 = 256B) 经 PCIe P2P DMA 传回 GPU VRAM → GPU 继续 O Proj. + FFN。
    9. **性能输出**: 端到端 decoding latency = GPU 计算延迟 + PCIe 传输延迟 + CSD attention 计算延迟。Throughput (tokens/s) = batch_size × num_decode_steps / total_wall_time。Latency 分解：对 bs=64 dense inference, FlexGen 中 KV Cache Access 占 98.9%，InstA 降至 80.7%，InstA-2 降至 76.4%。SparF engine 内部：Logit-0（新增的近似分数步骤）占比主要的额外 overhead。

## 72-ALISA_Accelerating_Large_Language_Model_Inference_via_Sparsity-Aware_KV_Caching.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SWA的attention kernel在GPU上进行token-level sparse attention计算，涉及四个关键kernel操作：(1) **Local Attention Sum**: 在最近k步的attention weights矩阵AW[n-k:n, :]上沿head和step维度求和（reduce sum），得到长度为n的向量S表示每个prior token的重要性分数。k=⌊nr/2⌋由caching ratio r动态决定；(2) **Argmax/K选择**: 从S[0:n-k]中选top-k个全局动态token indices I_g，与局部静态token indices I_l=[n-k, ..., n-1]合并为I；(3) **Gather操作**: 使用I从完整K/V tensors中gather稀疏KV——K_s=K[I,:], V_s=V[I,:]，将不规则稀疏模式下的KV打包为dense tensor；(4) **Dense Matrix Operations**: 对dense K_s/V_s执行QK^T、softmax、AW×V_s标准矩阵乘法。
  - 实验比较：(1) 单attention module执行时间分解: QK^T / Local Attention Sum / Sparse KV (gather) / Softmax & Attention Score 各操作的耗时和TFLOPS，OPT-6.7B/30B，batch_size=64, seq_len=128，不同KV sparsity (0%/20%/40%/60%/80%)；(2) LLM推理全流程分解: FlexGen vs ALISA各phase的执行时间（compute vs memory access）和GPU/CPU内存使用量，OPT-30B, batch_size=64；(3) Recomputation影响: Phase III中重计算on/off对执行时间的影响；(4) Ablation: 不同技术（SWA/Dynamic Scheduling/KV Compression）在不同KV sparsity下的独立贡献。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA Tesla V100 (16/32 GB HBM) 和 NVIDIA H100 (80 GB HBM)。V100用于≤13B模型，H100用于30B模型。
  - CPU: 2.60 GHz Intel Xeon, 128 GB DRAM。CPU-GPU 20 GB/s PCIe带宽。
  - 精度: FP16（所有变量），KV compression独立使用INT8。

- 评估性能的软件/脚本是什么。修改了什么。
  - 框架: FlexGen + HuggingFace Transformers。Attention kernel修改：
    (1) 在FlexGen的attention forward中插入SWA的token选择逻辑——在每个attention head中计算local attention sum（沿最近k步求和）→top-k argmax选全局动态token→与局部静态token merge→gather稀疏KV；
    (2) 修改FlexGen的KV tensor内存管理，支持token-level的GPU/CPU分配（原为head-level静态offload）；
    (3) 添加Phase III的KV recompute kernel（GPU重计算被删除的旧token K/V值）；
    (4) 集成INT8 quantize/dequantize kernel用于CPU-stored KV tensors。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源: 论文未提供独立开源链接。基于FlexGen (https://github.com/FMInference/FlexGen) 实现。
  - Kernel评估全过程（以OPT-6.7B single attention head, r=0.4, d=4096, n=512, head_dim=128, num_heads=32为例）：
    1. **输入**: Q ∈ R^{1×128} (当前token query), K ∈ R^{512×128}, V ∈ R^{512×128}, AW_prev ∈ R^{512×512}（前k步的attention weight历史，稀疏存储仅最近k行）。
    2. **Local Attention Sum kernel**: 从AW_prev取[n-k:n, :] = [410:512, :]共k=102行，沿row和head维度sum reduce → S ∈ R^{512}。实质是从GPU shared memory/HBM读入AW_prev的最近k行，每次对(dim=0, dim=head)做加和。由于k≤n较小，此操作为vector级加法，TFLOPS低（~10 TFLOPS for OPT-30B vs ~60 TFLOPS for QK^T）。
    3. **Argmax Kernel**: 对S[0:n-k] = S[0:410]做top-k=102 selection → I_g (102个indices)。与I_l=[410,...,511]合并为I (204个indices)。
    4. **Gather Kernel**: K_s = gather(K, I, dim=0)，从HBM中以indices I读取204行×128维×FP16 = ~51KB数据。不规则访存但因token级gather粒度适中（非element-level scatter），仍有合理memory coalescing。
    5. **Dense MatMul**: QK_s^T [1×128]×[128×204] = [1×204] → softmax → AW×V_s [1×204]×[204×128] = [1×128]。使用cuBLAS or FlashAttention-style tiled matmul。
    6. **Profiling**: 测量每个kernel的CUDA event elapsed time。发现核心瓶颈：QK^T的FLOPS不随KV sparsity成比例下降（GPU core未充分利用），local attention sum在更大模型中耗时接近甚至超过QK^T（vector vs matrix operation的data reuse差异）。
    7. **Throughput输出**: 端到端tokens/sec = total generated tokens / wall-clock time。对OPT-6.7B, ALISA (80% KV sparsity) 相比FlexGen加速1.4–3.0×。

## 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ACES在SpMM加速器上实现了三层kernel调度/运行时计算优化：(1) **Adaptive Execution Flow（Condensing Adapter）**：运行时根据sparse pattern在三种condensing degree间动态选择——traverse CSR offsets将矩阵A按row length分区为bands，大band通过sampling（3次32-row trial pass，计时选最快condensing）决定执行流，小band默认moderate。condensing degree控制A的非零元素如何在columns间重排，从而影响MPE并行计算的B row reuse和APE merging同步——none=高B reuse+高sync冲突，aggressive=低B reuse+低sync冲突，moderate=折中。(2) **PureFiber Cache Replacement Policy**：对global cache中每条cache line计算RD（Reuse Distance，下次被请求的预期时间）和FD（Fiber Density，所属fiber的cache line数=并发访问指示器），驱逐max(RD+FD)的line，tie-break选higher FD。目标是最大化pure fiber数量（所有concurrent cache line access全部hit，无stall）。同时管理B fibers和C partial fibers。(3) **Non-Blocking Buffer + Non-Blocking Cache**：NB buffer追踪所有outstanding cache miss，支持miss-hit-under-miss和hit-under-miss，允许多个并发miss不阻塞后续访问。每个entry对应一个missing cache line，含多个subentry处理同line的多次miss。数据从DRAM返回后更新cache并notify等待的PE/fetcher。(4) **Synchronization Scheduler**：运行时跟踪APE正在merge的row，从SQ中选择不存在同步冲突的fiber分配给APE，支持skip top fiber选next available。(5) **Merging Scheduler**：用Huffman tree（优先队列）调度final merging——每次取两个最小weight的partial fiber合并（minimizing total comparisons + memory traffic），合并结果重新入队直到row完成。
  - 实验比较：(a) ACES vs SIGMA(25.5×)/SpArch(8.9×)/SPADA(2.1×)的speedup；(b) Ablation：四种condensing degree × 两种cache policy (LRU/PureFiber)的speedup，证明adaptive condensing最优；(c) 三种cache policy (LRU/RD/PureFiber) × 三种condensing degree的速度和cache stall reduction（PureFiber avg -21.8% stall vs LRU）；(d) 不同NB buffer subentry数 (8-128)的speedup（64 subentries时达+35.8%）。

- 后端平台是什么，配置是什么。
  - 自研SpMM专用加速器ACES：16 MPEs (1GHz) + 16 APEs (1GHz)，16 SQs (2KB/queue)，Global Buffer 0.5KB，Global Cache 1MB (16 banks, 16-way associative)，16×16 swizzle-switch crossbar，NB Buffer 0.5KB (64 subentries)，HBM 128 GB/s (16×64-bit channels, 8GB/s per channel)。TSMC 28nm工艺，总area 3.52mm²，总功耗2.83W。Baseline accelerator（SIGMA/SpArch/SPADA）统一配置为16 multipliers + 1.5MB SRAM buffer + 1GHz + same HBM module，64-bit double-precision。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研cycle-accurate simulator（C++/SystemC，论文未给出具体语言），模拟SpMM在ACES及baseline加速器上的cycle-level执行。**未开源**。
  - 修改：ACES的simulator是从头构建的。对于baseline（SIGMA/SpArch/SPADA），按照各自论文描述实现其execution flow和硬件架构，统一标准化到相同配置（16 multipliers, 1.5MB SRAM, 1GHz, 64-bit double-precision, same HBM）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：ACES simulator和RTL**均未开源**（论文未提供任何URL或repository链接）。仅使用开源工具CACTI 7.0 (https://github.com/HewlettPackard/cacti)建模cache SRAM。
  - **评估原理与流程**：输入为SuiteSparse矩阵（square乘自身，non-square乘transpose，CSR格式）→ Simulator读取矩阵，Condensing Adapter执行adaptive condensing（band分区 + sampling选condensing degree）→ 模拟执行：每个cycle，A Fetcher/B Fetcher从DRAM prefetch数据入Global Buffer/Cache → MPE从buffer取A element + 从cache取B row fiber → 乘法cycle-by-cycle产出partial fiber → SQ buffer → Synchronization Scheduler分配fiber给APE → APE immediate merging（从cache读existing partial + merge + 写回cache）→ Cache访问触发的miss由NB Buffer非阻塞处理（登记miss entry → 发DRAM请求 → 不阻塞后续cache access → 数据返回后更新cache + 通知等待方）→ PureFiber policy在每次cache eviction时计算RD+FD选择victim → 全部MPE完成后，Merging Scheduler用Huffman tree调度final merging → 统计total execution cycles、DRAM read/write bytes、PE active cycles → 输出speedup (= T_baseline / T_ACES)、memory traffic、PE utilization。Workload：SuiteSparse 18矩阵，密度范围1.4e-06到1.9e-03。


## 63-Spindle- Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Spindle是一个面向多任务多模态（MT MM）大模型分布式训练的wavefront调度系统。核心将模型执行分解为waves（最小调度单元），通过联合优化实现异构感知的工作负载并行化和依赖驱动的执行调度。包含五大组件：(1) **Graph Contraction（MetaOp图收缩）**：将原始计算图G收缩为MetaGraph G_M，连续同构operator融合为MetaOp（条件：连续OP间出入度为1 + 相同operator type + 相同input data size）。通过BFS将MetaOp解耦为MetaLevels（同level内无dependency），将全局优化问题(1)分解为per-MetaLevel子问题。(2) **Scalability Estimator（可扩展性估计器）**：对每个MetaOp采用分段α-β建模估计执行时间T_m(n)（n为分配的GPU数）。先profiling离散点(n_i, T_m(n_i))，再拟合分段α-β曲线。profiling耗时<5分钟。(3) **Resource Allocator（资源分配器，§3.3）**：将per-MetaLevel子问题形式化为Malleable Project Scheduling Problem (MPSP)，用bisection search求解连续最优C*（所有MetaOp同时开始、同时结束），再通过bi-point discretization将连续分配转化为两个离散ASL-tuple ⟨n_m, ·, l_m⟩, ⟨n_m, ·, l_m⟩（满足L_m = l_m + l_m和C* = T_m(n_m)·l_m + T_m(n_m)·l_m），最后rounding l为整数。(4) **Wavefront Scheduler（波前调度器，§3.4, Alg 1）**：greedy迭代构造waves——①Propose: 贪心选择ASL-tuple填充设备；②Extend: 若未占满设备则对剩余执行时间大的MetaOp扩展资源；③Align: 以最短ASL-tuple执行时间为wave时长，截断其他tuple使时间跨度对齐；④Commit: 设定start time并移除已完成部分。(5) **Device Placement（设备放置，§3.5）**：优先intra-device-island放置（NVLink高带宽内）、优先放置高通信量MetaOp到岛内、device memory balance防OOM。(6) **Runtime Engine（运行时引擎，§3.6）**：四步——Localization（各device实例化所属MetaOp）、Intra-task Data Dependency（插入transmission operators处理wave间数据流依赖）、Inter-task Model Dependency（parameter device group pool管理跨任务参数同步）、Training Step（wave-by-wave forward/backward + group-wise parameter sync）。
  - 实验比较：(1) 端到端iteration time对比：Spindle vs Megatron-LM、DeepSpeed、DistMM-MT（DistMM多任务扩展）、Spindle-Optimus（任务级marginal gain分配），在Multitask-CLIP（4/7/10任务）、OFASys（4/7任务）、QWen-VAL 10B（3任务）上，集群规模8/16/32/64 GPUs；(2) Case study：Multitask-CLIP 4任务16 GPU上的cluster平均利用率时间线、per-device和per-MetaOp利用率spider chart对比；(3) Time breakdown：forward/backward、parameter sync、inter-wave send/recv占比分析，以及device placement ablation（sequential placement vs Spindle placement，通信开销27%→6%）；(4) Optimality analysis：Spindle iteration time vs 理论最优C*（MPSP连续松弛上界），偏差<7%；(5) Execution planner时间开销：所有实验<3秒。

- 后端平台是什么，配置是什么。
  - GPU集群：8-node cluster，每node 8×NVIDIA A800 80GB GPUs（NVLink intra-node互联），node间400 Gbps InfiniBand互联。实验从1 node（8 GPUs）扩展到8 nodes（64 GPUs）。
  - Baseline系统手工调优parallel configurations（DP/TP/PP degree、ZeRO stage、activation checkpointing等），取最优性能上报。Spindle自动生成execution plan。

- 评估性能的软件/脚本是什么。修改了什么。
  - 分布式训练框架：Spindle基于PyTorch实现，共~10K LoC Python——2.1K LoC execution planner + 7.9K LoC runtime engine。数据流传输用NCCL batched P2P primitives，parameter device groups用NCCL communication groups管理。
  - Spindle API：用户定义SpindleTask（每个task为PyTorch module），通过add_flow API连接不同task的modules，或用PyTorch FX Tracer自动从统一模型中拆分modules。
  - Baseline系统：(1) Megatron-LM [50] 和 DeepSpeed [62]——SOTA单任务训练系统，将MT MM模型的sub-models在时间维度解耦（各sub-model顺序占满全集群执行）；(2) DistMM-MT——DistMM [27] 的多任务扩展，对每个MM task分配适当资源并顺序执行；(3) Spindle-Optimus——任务级workload-aware资源分配，类似Optimus [53] 用marginal gain (T_m(n) - T_m(n'))/(n' - n) 贪心分配设备。
  - 评估指标：iteration time（100次迭代平均），speedup ratio vs DeepSpeed，TFLOPs/s利用率。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：Spindle代码**未开源**。论文supplemental material仅包含Appendix PDF：https://github.com/AFDWang/ASPLOS25-Spindle-Supplemental-Material（含README.md + Appendix.pdf）。底层依赖PyTorch + NCCL（开源）。
  - 评估全过程：
    1. **输入**：(a) 用户通过SpindleTask API定义MT MM训练任务——每task包含PyTorch modules（modality encoders + cross-modal module + task-specific heads），通过add_flow连接；(b) GPU集群配置（设备数N，inter-node bandwidth，intra-node NVLink带宽，per-GPU memory capacity）。
    2. **Graph Contraction（§3.1）**：Spindle扫描所有task的计算图，按拓扑序将连续同构operators融合为MetaOps（同一operator type + 相同input data size + 出入度为1），得到MetaGraph G_M。再经BFS分配MetaLevel编号（同level内MetaOp无依赖）。
    3. **Scalability Profiling（§3.2）**：对每个MetaOp m，用不同GPU数n_i ∈ {1,2,4,8,...} profile执行时间T_m(n_i)（不同parallel config: DP/TP/SP），拟合分段α-β函数作为scaling curves。Profiling总耗时<5分钟。
    4. **Resource Allocation（§3.3）**：对每个MetaLevel独立求解MPSP——bisection search找C*使得Σ T_m^{-1}(C*/L_m) = N。Bi-point discretization：对每个MetaOp选n_m, n_m（满足n_m ≤ n*_m ≤ n_m的最接近合法整数），计算l_m, l_m使C* = T_m(n_m)l_m + T_m(n_m)l_m，round l为整数。
    5. **Wavefront Scheduling（§3.4）**：greedy迭代：每wave贪心提案ASL-tuples填充N个设备→资源不足时扩展→以最短执行时间对齐wave时长→commit并移除。所有MetaLevel的wavefront schedules按时序merge。
    6. **Device Placement（§3.5）**：wave-by-wave贪心放置，优先intra-island（同node NVLink），优先高通信量数据流，平衡device memory。OOM时回溯调整。
    7. **Runtime Execution（§3.6）**：每iteration——各device按wave顺序执行local MetaOps → 插入NCCL send/recv处理wave间activation/gradient传输 → backward后group-wise parameter sync（同group内all-reduce梯度）→ optimizer step。100 iterations平均上报iteration time。
    8. **输出**：iteration time（ms），speedup ratio vs DeepSpeed，TFLOPs/s利用率timeline，time breakdown（fwd/bwd/sync/comm占比）。

## 64-POD-Attention- Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：POD-Attention 是首个针对 hybrid batch（prefill + decode 混合批）高效计算 attention 的 GPU kernel。核心创新包括：(1) **CTA-parallel fusion**：在 CTA 粒度融合 prefill 和 decode attention 计算——不同 CTA 独立执行 prefill 或 decode，避免 warp-parallel fusion 的 straggler 问题和 intra-thread fusion 的 barrier 限制；(2) **SM-aware CTA scheduling**：leader thread 读取 SMID hardware counter 获取 CTA 所在 SM，通过 atomically increment SM counter 获得 ticket，根据 scheduling policy（50:50 或 proportional）决定该 CTA 执行 prefill 还是 decode，保证每个 SM 内 prefill 和 decode CTA 并发执行；(3) **性能优化**：decode tile size 从 FA 的 128 降至 16（CUTLASS A100 tensor op 最小值），将 decode compute utilization 降至 ~10% 释放 tensor cores 给 prefill；virtual decode CTA 将 decode CTA 拆为 warp 粒度虚拟 CTA，平衡 prefill 和 decode 的 shared memory 需求；limiting prefill splits 限制 chunked-prefill 沿 K/V 维度的 splits 数最多填满 2 waves（避免与 decode 的 memory bandwidth 争抢）；2 CTA/SM vs 4 CTA/SM 自适应选择（prefill-dominant 用 2 CTA/SM 以使用更大 tile，decode-heavy 用 4 CTA/SM 以支持细粒度调度）。POD-Attention 基于 FlashAttention v2.6.1 构建，将 prefill 和 decode 函数改造为 generic device function（移除 CUDA 提供的 blockIdx，改为函数参数），通过 wrapper kernel 调用并传入计算后的 CTA ID 实现灵活 remapping。
  - 实验比较：(1) 单层 attention 计算 latency：FA_Serial（FlashAttention prefill+decode 串行）、FA_Streams（两个 CUDA streams 并行执行）、FA_HFuse（HFuse 工具 warp-parallel fusion）、FI_Serial（FlashInfer 串行）、FI_Batched（FlashInfer prefill kernel 同时计算 prefill+decode）、POD（本文方法），在 Yi-6B 上 32 个 prefill chunk 各 co-scheduled with decodes 场景；(2) >1000 个 hybrid batch sweep：context length 4K–20K，chunk size 512–2K，三种模型；(3) 各优化 ablation：CTA per SM 配置（2 vs 4）、scheduling policy（50:50 vs proportional）、limiting prefill splits vs vanilla split。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB HBM，Yi-6B 部署在 1×A100，Llama-2-7B 和 Llama-3-8B 部署在 2×A100（tensor parallelism）。
  - CUDA 12.4，GCC 11.4，Python 3.12，PyTorch 2.4，Ubuntu 22.04。
  - CUTLASS 用于 tensor core 操作。

- 评估性能的软件/脚本是什么。修改了什么。
  - 代码基于 FlashAttention v2.6.1 (https://github.com/Dao-AILab/flash-attention)，修改如下：
    1. 将 FA 的 prefill 和 decode kernel 转换为 device function（移除 CUDA blockIdx 依赖，改为参数传入）。
    2. 新增 SM-aware CTA scheduling wrapper kernel（Figure 9 伪代码）——通过 `asm volatile("mov.u32 %0, %smid;")` 读取 SMID，atomicAdd 操作 SM counter 数组决定 CTA 绑定的操作类型。
    3. 修改 decode tile size：QSL 维度 tile length 从 64–128 降至 16。
    4. 实现 virtual decode CTA：将 decode CTA 内 warp-level barrier 替代 CTA-level barrier，每个 warp 作为独立 virtual CTA 执行，shared memory 使用量降至原 1/4。
    5. 限制 prefill splits：chunked-prefill 的 K/V 维度 splits 数限制在填满 2 waves 以内。
    6. 自适应 CTA/SM 配置选择：运行时根据 batch composition 选择 2 CTA/SM 或 4 CTA/SM。
  - 基线对比系统：FlashInfer v0.2.0 (https://flashinfer.ai)，HFuse [42] warp-fusion toolchain。
  - Micro-benchmark（Figure 7）：自建 compute-bound kernel（array scalar multiplication + barrier）+ memory-bound kernel（three-array addition + barrier），评估 CTA-parallel、kernel-parallel、intra-thread 融合方式。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/microsoft/vattention/tree/main/pod_attn，Zenodo DOI: 10.5281/zenodo.14770840，Docker image: rnp1910/pod_attention:asplos_25_pytorch_run。
  - 评估全过程：
    1. **安装**：`git clone https://github.com/microsoft/vattention.git` → `cd vattention/pod_attn/` → `make install_miniconda` → `conda activate pod_attn` → `conda install cuda-toolkit=12.4.0` → `pip install -r requirements.txt` → `make install_all`。或直接使用 Docker：`docker run --gpus all -it rnp1910/pod_attention:asplos_25_pytorch_run`。
    2. **入**：GPU kernel 入为 hybrid batch 的 Q/K/V tensors——prefill 分含 chunk_size 个 query token（Q 维度 [chunk_size, num_heads, head_dim]），decode 分含 batch_size 个 query token（Q 维度 [batch_size, num_heads, head_dim]）。K/V tensors 含全上下文缓存的 key/value（维度 [context_length, num_kv_heads, head_dim]）。
    3. **Kernel Launch**：wrapper kernel 以 prefill_CTAs + decode_CTAs 总数 launch。每个 CTA 到达 SM 后，leader thread 执行 SM-aware scheduling——读 SMID → atomicAdd SM counter 得 ticket → 按 proportional policy（prefill_ratio = prefill_CTAs / (prefill_CTAs + decode_CTAs)）或 50:50 policy 决定操作类型 → atomicAdd 对应操作 CTA counter 获得该操作内的 CTA ID → 写入 shared memory → 经 __syncthreads 后所有 thread 读取分配 → 调用 prefill_op(cta_id) 或 decode_op(cta_id)。
    4. **Prefill Attention 执行**：以分配的 cta_id 调用 FA prefill device function。使用 QSL tile length 128（compute-optimized），K/V 维度 splits 数受限于填满 at most 2 waves。计算 tile-by-tile softmax attention，使用 tensor cores 执行 GEMM。
    5. **Decode Attention 执行**：以分配的 virtual CTA ID 调用 decode device function。QSL tile length 16（最小化冗余 compute，仅 ~10% compute utilization），warp-level barrier 替代 CTA-level barrier。每个 virtual CTA 独立处理 decode batch 的一个子集。
    6. **SM 资源管理**：prefill CTA 和 decode CTA（含 virtual CTA）共享 SM 的 shared memory（kernel launch 时分配 max(prefill_smem, decode_smem)）。virtual decode CTA 使用 1/4 原始 shared memory 使总量与 prefill 持平。
    7. **出**：对 prefill 分输出 prefill Q 的 attention output（[chunk_size, num_heads, head_dim]），对 decode 分输出 decode Q 的 attention output（[batch_size, num_heads, head_dim]）。通过 `torch.cuda.Event` 记录 kernel 开始/结束时间差得 latency。ncu profiler 采集 compute utilization（Tensor Core %）和 HBM BW utilization。
    8. **执行命令**：`make figure1`（2 min），`make figure6`（2 min），`make figure7`（2 min），`make figure11`（2 hours sweep >1000 batches），`make figure13`（CTA/SM ablation），`make figure14`（scheduling policy ablation）。输出原始数据和日志到 `output/`，图表到 `graphs/`。

## 62-SCAR- Scheduling Multi-Model AI Workloads on Heterogeneous Multi-Chiplet Module Accelereators.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SCAR是一个面向异构数据流MCM AI加速器的多模型工作负载调度框架，包含三个核心引擎：(1) **MCM-Reconfig Engine**：顶层时间窗口划分与层分配引擎。首先生成候选时间窗口划分策略——通过采样离散时间点作为边界点，设置nsplits=4（默认5个时间窗口）。层到窗口的分配采用First-Fit Greedy Packing（Algorithm 1）：对每个模型的每层，计算expected execution latency（基于各chiplet数据流类型的平均执行时间 E(Lat(l))=Σ(ndf_i/|C|)×Lat(l→i)，其中Lat(l→i)由MAESTRO离线数据库提供），若层可在当前窗口内完成则分配，否则推迟到下一窗口。(2) **Provisioner Engine (PROV)**：在每个时间窗口内，为各模型初步分配chiplet（作为计算node）数量。支持exhaustive search和rule-based两种模式。Uniform distribution规则：N_i = round(E(P_i)/ΣE(P_j) × |C|)，其中E(P_i)为目标优化指标（latency/energy/EDP）的期望值。(3) **Segmentation Engine (SEG)**：在每个时间窗口内，将已拓扑排序的模型层分割为layer segments（Definition 5），映射到计算node进行独占执行。segmentation空间复杂度O(Π N_i^(L_i-1))，通过Product-to-Summation Reduction（Heuristic 1：将分割搜索分解为两步——先垂直分割到模型，再水平分割每模型内层）、Inter-Layer Pipelining Heuristic（Heuristic 2：促进连续层在不同chiplet间的流水线执行以增强封装内数据复用）、和Segment Size Balancing Heuristic（Heuristic 3：平衡node间segment大小以最小化straggler影响）来管理复杂度。最终通过chiplet mapping将segment分配到具体chiplet，支持inter-chiplet pipelining、dynamic chiplet regrouping等高级调度技术。
  - 实验比较：(1) 7种异构MCM配置（Het-Sides、Het-CB等变体）+ 3种同构baseline（Simba-NVDLA、Simba-Shi-diannao、Simba-Hybrid）在10个多模型场景上的EDP/energy/latency对比；(2) Rule-based search vs EDP search的调度结果对比；(3) 不同MCM topology（Grid、Sides、CB）对性能影响；(4) Ablation on greedy packing algorithm（SCAR first-fit vs uniform packing baseline）；(5) 不同优化目标（latency/energy/EDP）对最优MCM策略选择的影响；(6) SCAR vs NN-baton（单模型scheduler）对比；(7) 不同chiplet数量（4x4到6x6）和topology对heterogeneous策略的scaling影响。

- 后端平台是什么，配置是什么。
  - MCM AI加速器：基于Simba [64]架构，2D mesh Network-on-Package（NoP）拓扑，chiplet配置为6×6（36 chiplets）作为主要评估规模，也评估4×4、5×5变体。每个chiplet为独立AI加速器，包含PE array、memory、NoC，支持特定dataflow类型。
  - Chiplet数据流类型：(a) NVDLA-style dataflow [52]——NVIDIA开源深度学习加速器的dataflow风格；(b) Shi-diannao-style dataflow [16]——面向视觉处理的近传感器加速器dataflow风格。两种数据流在loop ordering、parallelization、tiling策略上不同。
  - 封装接口：off-chip bandwidth（BW_offchip）、NoP bandwidth（BW_nop）。chiplet位于MCM两侧（左右）具备off-chip接口。
  - 性能估算工具：MAESTRO [35][36]（https://github.com/maestro-project/maestro）用于离线分析每层在各chiplet数据流类型上的latency和energy。MAESTRO是data-centric DNN mapping分析工具，建模PE array、buffer hierarchy、NoC的data reuse和performance。

- 评估性能的软件/脚本是什么。修改了什么。
  - 调度框架：SCAR scheduler（Python/C++实现，论文未说明是否开源）。框架输入多模型工作负载描述（每模型每层的类型、参数、依赖关系）和MCM硬件配置（|C|、各df类型chiplet数ndf_i、BW_offchip、BW_nop），输出完整调度方案（time window划分→每窗口layer-to-chiplet assignment→segment-to-chiplet mapping with pipelining）。
  - 性能估算：MAESTRO [35][36] 离线生成latency/energy数据库——对每种chiplet数据流类型，对模型中的每层，MAESTRO分析其dataflow mapping的reuse efficiency、compute time、data movement time，输出Lat(l→df_i)和Energy(l→df_i)。SCAR在线调度时直接查询此数据库，不进行在线cycle-level simulation。
  - 修改：SCAR在MAESTRO之上新增三个调度引擎模块。MAESTRO本身未修改——作为offline analysis tool使用。SCAR的调度决策（time window partitioning、layer-to-chiplet assignment、segmentation）通过Python scripts实现，最终生成execution timeline和EDP/latency/energy统计。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未明确说明SCAR代码是否开源（MICRO 2024）。MAESTRO开源：https://github.com/maestro-project/maestro。Simba架构参考：Simba [64]（MICRO 2019）。
  - 评估全过程：
    1. **输入**：(a) 多模型工作负载场景定义（Sc）——包含M个模型，每模型L_i层，每层的operator type（Conv/FC/Attention等）和tensor shape（H/W/C/R/S/K等参数）；(b) MCM硬件配置（H）——chiplet数|C|，各dataflow类型的chiplet数ndf_i，BW_offchip，BW_nop，NoP topology；(c) 优化目标（latency/energy/EDP）。
    2. **MAESTRO离线分析**：对每层l在每个数据流类型df_i上运行MAESTRO analysis——MAESTRO输入：layer tensor dimensions + dataflow description (loop order, tiling, spatial unrolling) + hardware resource constraints (PE array size, buffer sizes, NoC BW)。MAESTRO输出：Lat(l→df_i)、Energy(l→df_i)。构建完整latency/energy数据库。
    3. **MCM-Reconfig**：设置nsplits=4 → 5个periodic time windows。对每model按layer顺序依次用Greedy Packing Algorithm（Algorithm 1）分配层到window：计算E(Lat(l))=Σ(ndf_i/|C|)×Lat(l→i)，若layer latency ≤ window slack则分配，否则推迟。window boundaries基于最坏情况模型延时设定。
    4. **PROV**：每window内，按Uniform Distribution规则 N_i = round(E(P_i)/ΣE(P_j)×|C|) 为各模型分配node数。PROV agnostic to chiplet dataflow properties——仅按node数分配。
    5. **SEG**：每window内，先vertical segment（每模型独立搜segment划分），再horizontal segment（search跨node的inter-layer pipelining机会）。应用三个heuristics降低复杂度。最终chiplet mapping将segment绑定到具体chiplet（考虑dataflow compatibility和NoP hop distance）。
    6. **性能评估**：从chiplet mapping结果计算整体execution timeline：每个chiplet上各segment的执行时间（查MAESTRO数据库）+ inter-chiplet communication delay（data_size/BW_NOP × n_hops）+ off-chip traffic delay（data_size/BW_offchip）。累积所有chiplet和所有time windows的timeline得到端到端latency。Energy = Σ(per-segment compute energy + data movement energy via NoP + off-chip access energy)。EDP = Latency × Energy。
    7. **输出**：端到端EDP/latency/energy数值，各model的per-layer execution breakdown，chiplet utilization timeline，inter-chiplet data movement volume。与同构baseline（所有chiplet同dataflow）比较EDP reduction百分比。

## 57-HotTiles_Accelerating_SpMM_with_Heterogeneous_Accelerator_Architectures.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：HotTiles是一个IMH-aware（Intra-Matrix Heterogeneity aware）的稀疏矩阵建模与分区框架，用于在异构加速器上调度SpMM kernel。核心包括两部分：(1) **IMH-Aware Performance Modeling**：对每个稀疏矩阵tile，分别估算hot worker和cold worker的执行时间和主存访问字节数。对每个worker的5个任务（read sparse input、read dense input、read dense output、SIMD MAC、writeback dense output）建模计算时间和访存时间，考虑4种数据复用类型（Inter-tile、Intra-tile stream、Intra-tile demand、None）和不同稀疏格式（COO-like、CSR-like）的访存量差异。引入data-driven参数vis_lat（visible latency per byte）捕捉worker的latency hiding能力——通过少量profiling runs自动搜索最优值。(2) **IMH-Aware Partitioning Heuristic**：将稀疏矩阵tile按hot-cold性能差异排序，通过cutoff index线性扫描确定最优分区点。4种启发式（MinTime Parallel/Serial、MinByte Parallel/Serial）分别最小化时间或字节数，覆盖并行/串行两种执行模式。最终选预测runtime最低的分区方案，复杂度O(NlogN)。分区后为不同worker类型生成对应的稀疏压缩格式（COO/CSR变体）。HotTiles独立于底层异构硬件——只需profiling得到vis_lat即可适配新架构。
  - 实验比较：(1) HotTiles heterogeneous execution vs HotOnly/ColdOnly homogeneous execution、IUnaware heterogeneous execution、BestHomogeneous（per-matrix best selection）——在SPADE-Sextans和PIUMA架构上比较speedup；(2) 4种HotTiles heuristics在不同SPADE-Sextans system scale下的互补性分析；(3) HotTiles heterogeneous（scale 4）vs homogeneous with double workers（scale 8）；(4) gSpMM不同arithmetic intensity下HotTiles的性能（SPADE-Sextans+PCIe）；(5) 高密度矩阵集上的额外评估；(6) Architecture exploration：9种iso-scale异构架构的预测vs实际性能对比；(7) 预测误差分析（Homogeneous和HotTiles）；(8) Preprocessing overhead分析。

- 后端平台是什么，配置是什么。
  - **SPADE-Sextans架构**：SPADE PE（cold worker，OoO non-speculative vector engine，无scratchpad，COO-like格式，untiled row-ordered traversal）+ Sextans PE（hot worker，streaming access + scratchpad，COO-like格式，tiled row-ordered traversal），集成于同一die，共享内存。System scale可扩展（1/2/4/8），baseline为scale 4：16 SPADE PEs（1 SIMD MAC/cycle，L1 32KB）+ 1 Sextans PE（20 SIMD MACs/cycle，scratchpad 2MB）。PE频率0.8 GHz，cache line 64B，主存BW 205 GB/s（最大理论值），最大观测BW 161 GB/s。
  - **SPADE-Sextans+PCIe架构**：on-chip SPADE PEs + off-chip Sextans通过PCIe连接（max BW 32GB/s）。Sextans增强计算能力至20 nonzeros/cycle（与AI无关），用于gSpMM高arithmetic intensity评估。
  - **PIUMA架构**：4 MTPs（cold worker，fine-grained round-robin multithreading，CSR-like格式，untiled row-ordered traversal）+ 2 STPs（hot worker，in-order single-threaded + scratchpad + DMA engines，CSR-like格式，tiled row-ordered traversal）。Atomic engine支持read-modify-write无data race。双精度浮点。所有PE共享同一memory subsystem。
  - **预处理主机**：dual-socket 48-core Intel Xeon Platinum 8260M CPU。

- 评估性能的软件/脚本是什么。修改了什么。
  - **SPADE-Sextans评估**：使用SST [57]（Structural Simulation Toolkit）+ DRAMSim3 [46] 进行cycle-level模拟。论文未说明具体修改SST的细节，但建模了SPADE PE的OoO pipeline、Sextans PE的scratchpad流式访问、Merger模块的SIMD ADD操作、以及不同tile format下的memory access pattern。
  - **PIUMA评估**：使用基于Sniper [16][17] 的in-house simulator。论文未给出修改细节（PIUMA微架构细节为proprietary）。
  - **面积/功耗分析**：CACTI [9] 评估Merger模块memory结构，SIMD arithmetic数据来自[22]。面积和功耗按[60]缩放至10nm。
  - **评估原理**：HotTiles framework读取MatrixMarket格式稀疏矩阵→扫描tiles并应用cold/hot performance model→执行partitioning heuristic（4种候选）→选最优分区→生成各worker类型的sparse format→simulator接收format执行SpMM→输出runtime和memory bandwidth utilization。预测误差通过比较framework预测的execution time与实际simulated runtime计算（SPADE-Sextans geomean error: HotOnly 4.8%, ColdOnly 19.6%, HotTiles 12.4%）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未说明HotTiles代码是否开源。HotTiles使用SST simulator（开源：https://github.com/sstsimulator/sst-core）和DRAMSim3（开源：https://github.com/umd-memsys/DRAMSim3），以及Sniper（开源：https://github.com/snipersim/sniper）。SPADE和Sextans为已发表架构（ISCA'23和FPGA'22），PIUMA为Intel proprietary架构。
  - **评估全过程**：
    1. **输入**：MatrixMarket格式稀疏矩阵（benchmark matrices from SuiteSparse [20]），K=32（dense matrix columns），框架配置参数（worker数Nhw/Ncw、计算吞吐GFLOP/s、scratchpad大小、主存BW GB/s、复用类型、sparse format、任务重叠方式）。
    2. **Profiling阶段**：用小型test matrices单独执行hot/cold workers homogeneous运行，搜索vis_lat参数使预测误差最小（一次性per-machine）。
    3. **HotTiles preprocessing**：(a) Matrix Scan——对每个tile，调用cold/hot performance model计算thi/tci（执行时间）和bhi/bci（访存字节数）。执行时间估计：计算时间=2*K*tile_nnzs/worker_GFLOPs，访存时间=bytes*vis_lat，总时间取max（重叠）或sum（无重叠）。bytes按Table I的4种复用类型计算。(b) Partitioning——4种heuristics分别排序tiles并线性扫描cutoff index，选预测runtime最低的分区。(c) Format Creation——为hot/cold workers生成对应的COO/CSR-like sparse formats。
    4. **Simulator执行**：Simulator读取partitioned sparse formats+Din dense matrix→worker按指定traversal order处理tiles→模拟5个SpMM tasks的重叠→account for memory bandwidth contention（BW作为shared resource limit）→若并行模式则Merger module合并output buffers。
    5. **输出**：runtime (ms)、bandwidth utilization (GB/s)、cache lines accessed per nonzero、worker computational utilization (GFLOP/s)、以及相对于baseline的speedup。

## 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ORCHES提出三项运行时kernel调度技术来优化TTC-based LLM推理在GPU-PIM异构系统上的执行。(T1) Adaptive Assignment：基于离线分析模型（以batch size W、hidden dim D、compute capability CC、memory bandwidth BW为参数）和在线roofline调度补偿，动态将Linear operator和Attention operator（shared KV query和unique KV query部分）在GPU和PIM之间分配。当batch size小→全部分配给PIM；中等→shared KV query由GPU执行，其余PIM执行；大→Linear和shared attention由GPU执行，unique KV query保留在PIM。在线补偿通过引入ratio α动态调整每层的GPU/PIM workload分配，求解 T_PIM({α})=T_GPU({α}) 来最小化max延时。(T2) Branch Prediction Pipelining：受CPU分支预测启发，用小型PRM预判候选分支选择结果，使PIM上的generation能提前speculative执行，与GPU上large PRM的verification重叠。若预测错误则回滚。历史对齐机制用large PRM的historical scores替换small PRM的scores以提升预测准确率。Pipelined Verification在candidate tokens部分生成后即启动pre-verification，仅当GPU idle且有足够tokens保证arithmetic intensity时触发。(T3) Memory Structuring：address cache映射logical candidate ID到physical location避免DRAM遍历；动态memory reorganization用fragmentation ratio β触发碎片整理（compact到连续空间）；controller die buffer优化GPU对PIM KV cache的访问模式。
  - 实验比较：(1) ORCHES vs GPU baseline (AGX Orin) 在不同model sizes/search tree widths/SoC bandwidth下的归一化加速比；(2) ORCHES vs SOTA GPU-PIM baseline (AttAcc[ASPLOS'24]、Duplex[MICRO'24]) 的加速比；(3) 消融实验——ORCHES-A(全PIM)/ORCHES-B(自适应Linear)/ORCHES-C(B+动态补偿) vs baseline和AttAcc；(4) T1 Only/T2 Only/T1+T2 vs GPU baseline；(5) GPU和PIM的utilization分析；(6) T2 prediction accuracy和T3 memory footprint saving。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA AGX Orin [23] (edge GPU, 204.8 GB/s off-chip bandwidth)。
  - PIM: 模拟器扩展自AttAcc [ASPLOS'24]，32GB总容量，2048个memory banks，每bank 16个multipliers+adders (GEMV units)；Controller Die含Accum Units (parallel adders)、Softmax Units (fixed-function pipelined datapath)、Address Cache (SRAM)、Shared KV Buffer、State Machine。off-chip bandwidth 204.8 GB/s与AGX Orin匹配。
  - 模拟器设置：100%/75%/50% SoC bandwidth三种配置进行鲁棒性评估。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：扩展的AttAcc simulator [25,26]，基于Ramulator2 [20] 建模memory系统。开源地址：https://github.com/scale-snu/attacc_simulator
  - 修改：增强frontend (task scheduling) 和 backend (PIM memory system simulation) 支持TTC-based reasoning pipeline；实现performance counters统计data volume和不同computation counts，乘以unit energy values进行系统级energy评估。GPU modeling和PIM estimating准确度已在prior work [25]中由real hardware验证。
  - 算法pipeline：text-based TTC pipeline [18] (policy model: Llama3.2-1B, Qwen2.5-1.5B, Qwen2.5-3B；PRM: Qwen2.5-1.5B/7B-PRM-Tuned, Llama3.1-8B-PRM-Tuned；9种组合)，vision-based TTC pipeline [36] (Llama-3.2-11B-Vision-Instruct for both policy & PRM)。
  - 数据集：MATH500 [8] (text math), LiveCodeBench [11] (code), MATHVista [19] (vision math)。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：基于开源AttAcc simulator (https://github.com/scale-snu/attacc_simulator)扩展。ORCHES自身代码论文未明确说明是否已开源（MICRO 2025，2025年10月发表）。
  - 评估原理：模拟器输入包括：(1) 模型配置（layer数、hidden dim、attention type）；(2) TTC pipeline配置（search tree width=2~8、reasoning steps数）；(3) GPU/PIM硬件参数（compute capability CC、BW_PIM_IO、BW_PIM_internal、BW_GPU）。推理过程按step展开：每个step中，generation phase在policy model上进行decoding（每token调用一次Linear + Attention），verification phase在PRM上进行prefilling（批量处理所有candidates）。模拟器按T1的roofline model决定每层operator的GPU/PIM分配（α ratio），按T2的branch predictor决定是否提前启动下一step的generation，按T3管理memory allocation/fragmentation。模拟器输出各operator的latency breakdown、total runtime、device utilization、data movement volume、energy consumption（通过multiplying unit energy values）。最终aggregate所有steps得到端到端推理延时和能耗。

## 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Pimba在PIM端实现state update和attention的kernel级运算。核心是SPU四阶段流水线执行的state update kernel：(Stage1) 从DRAM fetch state sub-chunk；(Stage2) 并行计算state decay (d_t ⊙ S)和outer product (k_t v_t^T)；(Stage3) 加法更新state (S_decay + outer)；(Stage4) dot product (S_t^T q_t)同时writeback更新后的state到DRAM。每个sub-chunk迭代中，SPU读取共享的d_t/q_t/k_t向量和第i个v_t元素。Command scheduling通过在ACT4的tFAW空闲间隙插入REG_WRITE、在PRECHARGES的tRP间隙插入RESULT_READ实现数据传输与计算重叠。Chunk group内的chunks共享d_t/q_t/k_t，仅变更v_t，最大化operand复用。
  - 实验比较：(1) 各SU-LLM在不同batch size下的延迟分解（state update占比）；(2) GPU vs time-multiplexed PIM vs pipelined PIM的归一化吞吐和面积开销；(3) 端到端延迟分解（state update I/O + compute、attention I/O + compute、GEMM、others）；(4) Pimba vs GPU/GPU+PIM的state update延迟加速比14.6×/6.9×。

- 后端平台是什么，配置是什么。
  - PIM: 40 HBM2E stacks, 1,512MHz memory bus。SPU clock: 378MHz (4× tCCD_L)。Per-bank row buffer与SPU交互。HBM时序参数: tRP=14, tRAS=34, tCCD_S=2, tCCD_L=4, tWR=16, tRTP_S=4, tRTP_L=6, tREFI=3900, tFAW=30。
  - GPU: NVIDIA A100 80GB, 也评估H100。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研cycle-accurate simulator基于Ramulator2 [44]，扩展GPU和NVLink模拟基于开源simulator [54]。修改: 新增PIM子系统建模（SPU pipeline、MX算术单元、access interleaving），整合DRAM时序约束和refresh scheme。
  - 面积/功耗: Synopsys Design Compiler + FreePDK 45nm，DeepScaleTool缩放到10nm。SRAM buffer用CACTI7 @22nm缩放到10nm。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源: https://github.com/casys-kaist/pimba, 运行 `uv run python scripts/run.py` 生成accuracy_result.yaml和performance_result.yaml，`uv run python scripts/draw.py` 生成图表。
  - 评估原理:
    1. **Simulator输入**: 模型spec (dim_head, dim_state, num_heads, num_layers)，HBM配置 (频率、时序参数、banks数)，SPU配置 (pipeline stages, MX format, chunk/group layout)，batch size和sequence length。
    2. **State update kernel模拟过程**: 对每个chunk group的每个chunk，按sub-chunk iteration逐一模拟SPU四阶段pipeline。每个iteration：计算Stage1 DRAM read latency (tCCD_L per column)→Stage2 MX multiply latency (element-wise decay + outer product parallel)→Stage3 MX add latency→Stage4 dot product + writeback latency (tWR)。Access interleaving建模: 交替从upper/bottom bank读取/写入sub-chunk。
    3. **Command scheduling模拟**: ACT4激活4 banks (tFAW约束)→REG_WRITE传输operands (data bus latency)→COMP执行pipeline→RESULT_READ取回结果 (tRTP+tWR后)→PRECHARGES (tRP)。
    4. **输出**: 各operation的cycle count→转换为latency (ms) 和throughput (tokens/s)。Energy: DRAM activation/read能量来自[52]，compute能量来自合成结果。Total energy = Σ(各操作cycles × power)。

## 51-Crane- Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture..pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Crane的核心kernel调度/运行时计算在于将DNN workload跨tile的sub-batch级执行调度建模为层级化block+pipeline state系统，由MILP优化。具体调度维度：(1) **Execution Scheme调度**：对每个composite block的N个子block，在2N-1个pipeline-derived states中分配state workload s_i。每个state内，若干子block同时活跃（根据pipeline state定义J_i），tile资源按计算量比例分配。通过灵活设置s_i值，Crane可实现sequential（s_1=s_{N+1}=...=BS/B_sub）、pipeline（所有s_i>0按流水线分配）、parallel（s_1=s_{N+1}=BS/B_sub，其余为0）或混合模式。这比SET的rigid batch-level约束（sub-batch处理顺序严格绑定到batch-level pattern）更灵活——例如可生成A1A2→(A3,B1)→(B2B3,C1C2)→C3等非标准顺序。(2) **Fusion Strategy调度**：通过MeT表隐式编码。ScT[i,j]提供sub-batch处理的upper bound，MeT_S和MeT_D提供SRAM/DRAM中stored sub-batch的lower bound。当某layer的输出保留在SRAM（MeT_S区间）而非写入DRAM时，即实现了layer fusion——下一layer可直接从SRAM读取数据避免DRAM访问。MILP在Eq.11-12约束下自动决定哪些sub-batch的中间结果保留在SRAM（fusion）vs写入DRAM（no fusion）。(3) **Recomputation调度**：前向传播后，MeT_FW[2N-1,j]记录子block B_j在DRAM中保留的activation区间。Step-1调度backward消费DRAM中stored activation（ScT_BW1构建受Eq.14约束）；Step-2调度forward recomputation恢复被丢弃的activation + backward计算（ScT_BW2受Eq.15约束，recompute仅恢复已丢弃的sub-batch）。MILP自动决定哪些activation保留在DRAM（checkpoint）vs丢弃后recompute。(4) **Batch Splitting调度**：通过候选sub-batch size B_sub的枚举+MILP内ScT变量实现。不同B_sub改变ScT的行/列规模和memory容量约束的松紧度，Crane选取K1/K2个B_sub候选按Cost_comp/Cost_traffic排序后分别优化，最终选最优B_sub下的调度方案。
  - 实验比较：(a) Inference调度性能：vs SET/Tangram/TileFlow在相同硬件下的latency/energy/EDP对比（详见编译框架条目）。(b) Training调度性能：vs MBS/hypothetical SET+Tangram，CRANE通过全面探索E+F+R+B四因素实现EDP降低3.62×–64.73×。(c) Scheduling runtime：Crane 2.82× faster than SET，156.20× faster than Tangram on AMD EPYC 7402P。(d) Ablation——batch splitting消融验证sub-batch size对PE utilization和数据movement的trade-off；recomputation消融验证DRAM capacity 2.2× reduction；execution scheme消融验证4.7× EDP reduction vs fixed scheme。(e) 细粒度sub-batch优化案例：Inception-ResNet-V1 inference (BS=2, B_sub=1)，Crane生成tile-time schedule消除了SET coarse-grained sub-batch scheduling的bubble overhead。

- 后端平台是什么，配置是什么。
  - **Inference**: NVDLA-style tiled accelerator，edge 16-tile / cloud 144-tile。每tile: 1024 int8 MACs (2 TOPS @1GHz), 1MB SRAM。Mesh NoC: 24 GB/s bidirectional。DRAM: BW_D = 0.5 GB/TOPs。TSMC 12nm, 1GHz。MAC energy: 0.018 pJ/op, NoC energy: 0.7 pJ/bit/hop, DRAM energy: 7.5 pJ/bit。Tile utilization modeling: 1D-tiling 4 factors mapped to T tile grid，utilization = product of dim_q/ceil(dim_q/k_q)。
  - **Training**: Two-tiled systolic array，每tile 128×128=16384 MACs, 10MB SRAM。NoC: 100 GB/s, 0.7 pJ/bit。32GB HBM2: 300 GB/s, 3.9 pJ/bit。
  - **Scheduling runtime evaluated on**: AMD EPYC 7402P CPU。

- 评估性能的软件/脚本是什么。修改了什么。
  - Crane自身是C++实现的scheduling framework。Cost model内部集成：(a) PE computation latency model (Eq.16-18)：根据tile utilization ratio (u_{i,j})、MAC count (P)、workload (FLOPs)计算每state的compute latency。(b) Data traffic latency model (Eq.19)：根据data dependency hop count (H_C/S/D)和data volume (V_m)计算NoC transfer latency。(c) Energy model：compute energy = FLOPs × E_comp；NoC energy = traffic bits × hops × E_NoC；DRAM energy = DRAM access bits × E_DRAM。(d) EDP = Latency × Energy作为MILP目标函数。(e) zsim [30] + DRAMSim2 [29] 用于cost model validation（cycle-accurate simulation基线）。
  - 修改：Crane复用SET的brute-force intra-layer exploration方法，未修改现有simulator。inter-layer部分完全由Crane的MILP formulation替代。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - Crane未开源。评估原理：给定DNN model + hardware config → 构建层级化block → 对每block构建MILP (ScT+MeT约束+EDP目标) → MILP solver求解 → 提取调度方案和性能预估。输入→输出全过程：(1) 输入DNN模型层定义（每层FLOP count、tensor size、层间依赖d_{m,j}）、硬件spec（tile数、MAC/tile、SRAM/DRAM容量和BW、energy参数）、batch size BS。(2) Intra-layer exploration阶段：对每种sub-batch size候选，遍历tiling factor组合（将sub-batch的4维映射到tile grid的4因子），计算每个layer的tile utilization u_{i,j}，得到单层compute latency和data traffic特征。(3) Per-block MILP阶段：构建ScT和MeT的整数规划模型——decision variables为ScT[i,j]（整数，0~BS/B_sub）和MeT[i,j]^S/D（整数，0~BS/B_sub），约束为Eq.1-12（forward）或Eq.1-15（training含recomputation）。目标函数EDP = (Σ_i s_i × max_j L_comp,i,j + Σ traffic latency) × (Σ compute energy + Σ NoC energy + Σ DRAM energy)。(4) MILP solver (Gurobi/CPLEX) 求解得到最优{ScT, MeT, s_i}。(5) 层级化refine：top-level block最优解→分解到lower-level→各自MILP优化→汇总cost→迭代至收敛。(6) 输出：优化后的schedule（每state的各block active状态、处理sub-batch数、tile分配、SRAM/DRAM中的sub-batch存储区间），以及预估的latency (cycles)、energy (mJ/pJ)、EDP (ms×mJ)、DRAM traffic (GB)。评估中Crane直接用cost model给出的EDP/latency/energy作为最终性能指标，不经过实际硬件执行。

## 48-Meta_s Second Generation AI Chip- Model-Chip Co-Design and Productionization Experiences.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MTIA 2i的kernel级优化覆盖GEMM、TBE和模型特定kernel：(1) **FC kernel auto-tuning**：构建kernel generator生成input/output/weight stationary三种变体，每变体可调block size、DMA scheduling、circular buffer usage。建立performance database + approximate nearest neighbor search替代exhaustive tuning，tuning时间缩短1000x，性能在最优5%以内。(2) **GEMM custom instruction优化**：新增multi-context custom instructions避免重复写custom registers；auto-increment offset feature使矩阵乘法指令在tight loop中高效issue；DMA_IN prefetch feature使数据从DRAM→SRAM预取后再加载到Local Memory。2K×2K×2K GEMM达到>92% peak FLOPS。(3) **TBE kernel优化**：新DMA_IN指令支持index输入自动计算地址+对齐地址处理；SE accumulation新指令支持最多128 rows（原32 rows）减少embedding pooling指令数。(4) **DRAM带宽bound优化**：当activation fit in PE Local Memory时，decouple activation loading（从LLS prefetch）和weight loading（broadcast across PE columns），利用hardware broadcast read消除NoC contention，weight tiles prefetch到LLC隐藏DRAM延迟→latency改善45%，shape 512×26592×2048 (109MB weight tensor) 达到>95% DRAM带宽。(5) **HSTU ragged attention kernel**：bias计算涉及table index computations+gather操作→repurpose SE的lookup table (LUT) 分片加载weights和timestamps表。(6) **LayerNorm/SoftMax kernel**：LayerNorm 3步pipeline（row-wise mean→variance→element-wise result）混合fixed-function+RISC-V vector；SoftMax 5步pipeline跨RISC-V scalar+vector core平衡DMA fetch和computation。
  - 实验比较：(a) GEMM效率：2K×2K×2K达到>92% peak FLOPS；(b) DRAM带宽优化前后latency改善45%；(c) FC autotuning：近似搜索vs exhaustive tuning性能差异<5%但时间缩短1000x；(d) Section 6 case study：关键ranking model Perf/TCO从50%→180% vs GPU baseline，其中kernel优化贡献包括FC kernel variant selection、graph fusion、TBE consolidation；(e) 生产9个模型的效率数据（Figure 6），高复杂度模型(480-1000 MFLOPS/sample)因DRAM bandwidth-bound效率较低。

- 后端平台是什么，配置是什么。
  - MTIA 2i芯片：TSMC 5nm, 64 PEs (8×8), 1.35GHz, 256MB SRAM (2.7 TB/s), 64-128GB LPDDR5 (204.8 GB/s), GEMM 177 TFLOPS/s FP16/BF16, 354 TFLOPS/s INT8。
  - Server: Grand Teton platform, 24 MTIA 2i chips/server, 2× Intel CPU (96核 each), 2×1.15TB DDR5, 2×200Gbps NIC, PCIe Gen5。

- 评估性能的软件/脚本是什么。修改了什么。
  - PyTorch 2.0 + Triton: MTIA Triton compiler优化Triton kernel compilation以高效利用fixed-function units。Triton kernel编译为MTIA custom instructions下发到PE的CP→CP编排DPE/RE/SE/MLU异步执行。
  - Autotuning framework: 自动选择FC kernel variant、batch size、request coalescing参数。Offline replayer进行traffic-replay测试评估throughput/latency。
  - MTIA Streaming Interface + MTIA Firmware Driver: 提供底层device API（MTIATensor, Device Memory Allocator）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 不开源。MTIA软件栈不开源，但基于开源PyTorch 2.0和Triton [26]。Kernel执行过程：PyTorch eager mode或TorchDynamo trace→TorchInductor生成Triton代码→MTIA Triton Compiler编译为custom instructions→RISC-V cores issue到Command Processor→CP按依赖解析调度DPE(GEMM)、RE(accumulate)、SE(quantize/activate)、MLU(transpose/reshape)→Fabric Interface DMA通过NoC与SRAM/LPDDR交互→结果返回Local Memory→CP通知完成。Autotuning评估原理：给定模型graph→sweep batch size/FC kernel variant/data placement→offline replayer回放生产流量→收集throughput/P99 latency→选最优配置。

## 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：VGA协处理器将H3 block的ROI (Region of Interest) 计算完全从GPU/TPU offload到专用硬件。ROI包含四个kernel：(1) **FFTConv**——使用Generalized Cooley-Tukey算法进行2D-FFT（Column-wise FFT→CTF Multiplication→Row-wise FFT→IFFT），配合state passing实现chunk-wise convolution；(2) **State Update**——利用Mux Vandermonde矩阵与输入chunk相乘后加到AL·x_{c-1}更新状态向量；(3) **Output Projection**——Mxy Vandermonde矩阵与前状态向量相乘产生输出投影；(4) **Pointwise operations**——复数/实数乘法与加法。核心创新是**on-the-fly Vandermonde矩阵生成**：CTF/Mxy/Mux矩阵均从1-2行/列通过循环复数乘法在线生成，避免存储完整矩阵（SRAM需求减少5×），使多kernel融合成为可能。
  - 实验比较：(a) GPU baseline: H3模型在A100-40GB上运行自定义FFT convolution CUDA kernel + 论文自实现的state passing算法（官方H3 repo未提供state passing）；FlashAttention2作为self-attention baseline。(b) TPU baseline: H3 ported版本含state passing；TensorFlow2 multi-head attention。(c) Speedup: VGA (128 PE GPU / 32 PE TPU) vs baselines，序列8K-128K。(d) Breakdown: FFTConv/State Passing/Pointwise各自加速比。(e) Memory traffic: VGA相比GPU 9.7× DRAM traffic reduction。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA A100-40GB (826mm², 400W TDP, HBM2 1555GB/s)
  - TPU: TPUv3单个core (half chip, 324mm², 225W, HBM2 450GB/s)
  - VGA (GPU集成): 128 PEs, 每PE k=32 CCUs, 1GHz, TSMC 40nm→7nm scaled (52.82mm², 41.1W)
  - VGA (TPU集成): 32 PEs, 每PE k=32 CCUs, 1GHz, TSMC 40nm→16nm scaled (42.35mm², 18.92W)
  - Chunk size L=2048（使2D-FFT作为方阵执行，fit in SRAM）

- 评估性能的软件/脚本是什么。修改了什么。
  - GPU baseline: 基于H3官方repo (https://github.com/danfu09/H3) 的FFT convolution CUDA kernel。论文自行实现了state passing算法（官方repo未提供），作为完整H3 GPU baseline。FlashAttention2 (https://github.com/Dao-AILab/flash-attention) 作为self-attention baseline。
  - TPU baseline: H3 state passing版本移植到TPUv3；TensorFlow2 multi-head attention。
  - VGA simulator: 自研cycle-accurate simulator + Ramulator2模拟DRAM。
  - 模型：H3-GPT (GPT-125M-like, h=768, m=64, 12 H3 blocks, WikiText-103), H3-Speech (h=128, m=64, 6 H3 layers, SC10 dataset)。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - H3模型开源：https://github.com/danfu09/H3。VGA simulator + RTL未明确说明开源。
  - GPU baseline评估流程：
    1. **输入**：WikiText-103/SC10数据，序列长度8K-128K，batch size由各平台最大power-of-two决定(GPU: H3-GPT b=8, H3-Speech b=16; TPU: H3-GPT b=2, H3-Speech b=16)。
    2. **H3 GPU kernel执行**：Q/K/V经FC层→1D Conv on K→PointMult(K,V)→SSMConv(state passing chunk-wise FFT convolution)→PointMult(Q, result)→FC输出。
    3. **FFTConv kernel**：输入chunk u_c (L=2048)→Column-wise FFT (BF mode多次)→CTF生成与乘法(CTFGen/CMult mode)→Row-wise FFT→乘filter K_f (CMult)→IFFT。
    4. **State Passing kernel**：CMult mode计算A^L·x_{c-1}→Update mode：广播u_c元素到各CCU，各行on-the-fly生成Mux元素乘积累加到A^L·x_{c-1}产生x_c→Projection mode：各列on-the-fly生成Mxy元素乘x_{c-1}元素累加。
    5. **Pointwise kernel**：RMult mode (M7) 实数乘法；Residual mode (M6) 加法。
    6. **性能输出**：GPU baseline用CUDA event timer测量各kernel时间。VGA用cycle-accurate simulator测量ROI各operation cycle count + Ramulator2 DRAM延迟。端到端 = GPU frontend/backend时间 + VGA ROI时间（for VGA case）。

## 35-PIM-MMU_A_Memory_Management_Unit_for_Accelerating_Data_Transfers_in_Commercial_PIM_Systems.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PIM-aware Memory Scheduler (PIM-MS)，一个集成在PIM-MMU Data Copy Engine (DCE)内的硬件级fine-grained memory调度器，专门优化DRAM↔PIM数据传输时的PIM read/write吞吐量。核心调度算法（Algorithm 1）：
    1. **初始化**：将address buffer中每个PIM core的传输进度offset归零（pim_cores[id].offset = 0）。
    2. **Channel-level并行最大化**（outer loop: do-parallel channel）：同时向所有PIM channels发出memory requests，最大化channel-level parallelism。
    3. **Bank-group interleaving优先级**（middle loop over bankgroups）：连续column command targeting不同bank groups，最小化tCCD（column-to-column DRAM timing delay）。
    4. **Bank-level parallelism + row buffer conflict最小化**（inner loop over banks）：通过AGU翻译后的DRAM地址信息，对同bank group内的不同banks进行调度。
    5. 关键洞察：DRAM↔PIM传输时不同PIM core的目标地址是互斥的（无true data dependency），因此可安全地进行任意reordering而不影响正确性。不同于baseline的OS线程调度（round-robin fairness policy, preemption quantum ~几ms），PIM-MS在硬件层以单cycle粒度调度，将所有PIM core的传输请求统一管理，远优于baseline的software-level/coarse-grained/multi-threaded调度。
  - 实验比较：(a) per-channel write throughput breakdown：baseline software coarse-grained DRAM→PIM vs hardware fine-grained DRAM→DRAM (memcpy)，用Intel VTune测量；(b) Ablation study逐级添加Base → Base+D(DCE无PIM-MS) → Base+D+H(+HetMap) → Base+D+H+P(+PIM-MS full PIM-MMU)下吞吐量和能耗变化；(c) 16个PrIM benchmark端到端执行时间中的DRAM→PIM和PIM→DRAM传输延迟对比。

- 后端平台是什么，配置是什么。
  - 真实系统（表征+PIM kernel执行）：Intel Xeon Gold 5222 CPU，3通道DDR4-3200 DRAM (76.8 GB/s) + 3通道DDR4-2400 UPMEM-PIM DIMM (57.6 GB/s)，每通道1 DIMM。
  - 周期级模拟器：Ramulator扩展。CPU model: 8-core, 3.2GHz, 4-wide OOO, 224-entry instruction window, 64 MSHRs/core, 8MB shared LLC, 64-entry read/write request queues, FR-FCFS baseline scheduling。DRAM: DDR4-2400, 4 channels, 2 ranks/channel。PIM: DDR4-2400, 4 channels, 2 ranks/channel (512 PIM cores)。PIM-MMU: 3.2GHz, 16KB data buffer + 64KB address buffer。Baseline OS线程调度建模：8线程，1.5ms round-robin preemption quantum。

- 评估性能的软件/脚本是什么。修改了什么。
  - Microbenchmark: PrIM [43] benchmark suite中的CPU-DPU microbenchmark（测量DRAM↔PIM数据搬运吞吐量）；自定义memcpy microbenchmark使用AVX-512 _mm512_stream_si512多线程向量指令（测量DRAM→DRAM吞吐量）。
  - Real-world benchmarks: PrIM [43] 16个内存密集型PIM workloads: BFS, BS, GEMV, HST-L, HST-S, MLP, NW, RED, SCAN-RSS, SCAN-SSA, SEL, SpMV, TRNS, TS, UNI, VA。
  - Ramulator修改：实现DCE的AGU/PIM-MS/HetMap cycle-level行为；将AVX指令模拟为64B wide non-cacheable read/write；实现OS thread round-robin preemption模型（1.5ms quantum）；双memory mapping函数（MLP-centric和locality-centric）。
  - Intel VTune [56] 用于测量per-channel write throughput breakdown；Intel PCM [55] 测量系统功耗。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明其Ramulator扩展是否开源。UPMEM生态开源：UPMEM SDK [106] https://sdk.upmem.com；PrIM benchmarks [43] https://github.com/CMU-SAFARI/prim-benchmarks。
  - Ramulator开源：https://github.com/CMU-SAFARI/ramulator。
  - 评估流程：
    1. **Baseline dpu_push_xfer trace提取**：在真实UPMEM系统上编译runtime library (gcc 9.4.0)，用CPU tracing工具提取dpu_push_xfer执行的instruction trace（AVX load/store + 循环迭代）。
    2. **Ramulator trace-driven模拟**：将instruction trace输入Ramulator。CPU model按OOO执行trace中的指令，生成memory requests。Memory controller的FR-FCFS scheduler处理这些请求，按locality-centric memory mapping翻译地址。
    3. **PIM-MS调度模拟**：对于PIM-MMU，不执行CPU trace。Address buffer填充所有目标PIM core的base addresses和offsets。PIM-MS algorithm按Algorithm 1的嵌套循环（do-parallel channel → bank group interleaving → bank interleaving）生成memory requests序列。AGU将物理地址翻译为DRAM地址并放入memory controller command queue。
    4. **DRAM timing模拟**：Ramulator's DRAM model根据DDR4-2400时序参数计算每个ACTIVATE/READ/WRITE/PRECHARGE命令的cycle-level延迟。PIM-MS的发序列方式通过bank group interleaving最小化tCCD开销。
    5. **性能输出**：总transfer cycle count → 吞吐量 (GB/s)；per-channel bandwidth utilization对比（软件coarse-grained vs 硬件fine-grained）；端到端加速比 = (baseline dpu_push_xfer cycles + PIM kernel wall-clock time) / (PIM-MMU transfer cycles + PIM kernel wall-clock time)。
    6. **能耗输出**：McPAT模拟计算各组件（core/cache/DRAM/PIM-MMU DCE buffer）的动态和静态功耗 × 执行时间，chip-wide energy。

## 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：HLX提出两个新型kernel数据流——**PipeFlash**和**PipeSSD**，专为Hybrid Transformer-Mamba模型中的FA-2和SSD计算加速设计。
  - **PipeFlash (针对FA-2)**：
    - 传统FA-2: block-level同步执行QKT→local softmax→PV→update O，非MatMul(softmax/update O)无法与MatMul(QKT/PV)重叠，导致compute utilization在A100饱和于~61%、H100~49%。
    - PipeFlash: fine-grained pipelined执行，每次处理Q block中的2行。4-stage pipeline：DPE#0做QKT(2 rows)→RVPE做local softmax(1 row per Q row)→DPE#1做PV(2 rows)→UpE做update O(1 row)。通过流水线重叠，softmax和update O被MatMul隐藏。K/V block被Q block的所有行复用。中间数据减少4.8×（score/prob矩阵128KB→1KB）。
    - 在HLX上达到最高97.5% compute utilization，相比A100平均提升1.83×，相比H100平均提升2.03×。FA-2 speedup: 1.75× vs A100, 2.78× vs H100。
  - **PipeSSD (针对SSD)**：
    - 传统SSD: 5个独立kernel (chunk cumsum→chunk state→state passing→BMM chunk→chunk scan)，中间数据无法复用，memory-bound (Op/B极低)，compute utilization ~27% on A100, ~38% on H100。
    - 先提出fused SSD: 类似FA-2的block-level fusion，将6个operation合并为单kernel，但中间数据642KB/block，超出A100/H100 per-SM内存（256KB reg + 164/224KB shared mem），导致register spilling和occupancy下降，GPU上性能反而恶化1.74×。
    - PipeSSD: 在fused SSD基础上加入fine-grained pipelining，分3个stage：(1st stage) dA预处理(sdt=softplus(dt+dtbias) → dACS=cumsum(dA) → decay_states=exp(dACS[-1:]-dACS) → d2t=decay_states×sdt)；(2nd stage) CBT=C×BT → CBTLdt=CBT×L×sdt → YDiag=CBTLdt×x；(3rd stage) 并行执行dCOff=exp(dACS)×C → YOff=dCOff×states(j-1) 以及dBdt=d2t×B → statesN=dBdtT×x → YFinal=YDiag+YOff + update states。
    - 2nd stage YDiag映射: DPE#0(CBT 2 rows)→RVPE(CBTLdt 1 row, 经Local NoC的element-wise mult)→DPE#1(CBTLdt×x 2 rows, MatMul); 3rd stage YOff/statesN映射: RVPE→dCOff(8 rows)和dBdtT(4 rows)经MUX/DEMUX分别到DPE#0(YOff=8 rows)和DPE#1(statesN=4 rows)→UpE(YFinal 8 rows + update states 4 rows)。
    - 中间数据减少11× (642KB→58.5KB)，DRAM流量减少6.8×。Compute utilization达~78.4%，平均提升2.84× vs A100、2.04× vs H100。SSD speedup: 2.91× vs A100, 4.95× vs H100。
  - 实验比较：(a) FA-2 compute utilization和speedup随seq len (1K-128K)对比A100/H100；(b) SSD compute utilization和speedup对比；(c) FA-3 on H100 vs PipeFlash on HLX60；(d) 批量大小扫掠(1-128 @1K seqlen)的compute utilization和speedup；(e) End-to-end Hybrid-2.7B延迟和batch吞吐对比；(f) TPUv3 baseline对比。

- 后端平台是什么，配置是什么。
  - HLX硬件平台: HLX60配置(60 URSCs, 614.4 TFLOPS FP16, 2000 GB/s DRAM带宽, 30.4MB on-chip SRAM, 14nm @625MHz, 缩放至7nm后169mm²/201.8W)对标H100；HLX30配置(30 URSCs, 307.2 TFLOPS, 1935 GB/s, 15.2MB SRAM, 83.9mm²/108.47W)对标A100；HLX6配置(6 URSCs, 61.44 TFLOPS, 450 GB/s, 3.04MB SRAM, 47.16mm²/35.06W)对标TPUv3。
  - GPU baseline: NVIDIA A100 80GB (312 TFLOPS FP16, 1935 GB/s HBM2E, 84.3MB on-chip, 826mm²/300W, 7nm)；NVIDIA H100 80GB (756 TFLOPS FP16, 2000 GB/s HBM2E, 103.9MB on-chip, 814mm²/350W, 4nm)。
  - TPU baseline: TPUv3 (half-chip/single core: 61.5 TFLOPS, 450 GB/s HBM2, 16MB on-chip, 324mm²/225W, 16nm)。

- 评估性能的软件/脚本是什么。修改了什么。
  - GPU baseline: FA-2/FA-3 CUDA kernels from [11, 47] GitHub; NVIDIA Nsight Systems/Nsight Compute测延迟和compute utilization。HLX simulator: 自研cycle-level simulator建模URSC中DPE/RVPE/UpE的pipeline执行和DRAM时序。
  - PipeFlash相对FA-2的修改: 将block-level同步执行改为row-level (2 rows) fine-grained pipeline，Q/K/V分块粒度不变但内部以行为单位流水。softmax和update O与MatMul在时序上重叠。
  - PipeSSD相对SSD/fused-SSD的修改: 将5个独立kernel合并为单一fused kernel → 再分解为3个pipeline stage。解决了column-wise dependency (state passing: states(j-1)→states(j)) 和row-wise dependency (YOff+YDiag→YFinal, statesint+statesN→states) 的流水调度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - HLX simulator未明确开源。GPU baseline kernel开源: https://github.com/state-spaces/mamba [11], https://github.com/Dao-AILab/flash-attention [47]。
  - 评估原理:
    1. **输入**: Hybrid-2.7B模型参数 (attention: 30 heads × dhead 128; SSD: 80 heads × dhead 64, dstate 128, blocksize 256)。输入tensor shapes: Q/K/V [batch, nheads, seqlen, dhead]; dt/x/B/C [batch, ..., seqlen, dstate/...]; A [nheads, dstate]。Sequence length: 1K-128K; batch size: 1-128。
    2. **PipeFlash执行流程 (per attention layer)**: (a) Load Q_block(2 rows×dhead), K_block(blocksize×dhead) from GS→DPE#0 IMEM/WMEM; (b) DPE#0: QKT MatMul [2, blocksize]; (c) Forward result to RVPE→RVPU执行softmax: rowmax→exp(S_i-max)→rowsum→P_i=P_i/rowsum; (d) Forward P_i到DPE#1做PV MatMul [2, dhead]; (e) UpE: update O_i = diag(exp(m_i(j-1)-m_i(j)))⁻¹×O_i(j-1) + P_i×V_j; (f) 所有KV blocks遍历完成后UpE计算final O_i = diag(l_i(Tc))⁻¹×O_i(Tc)→写GS→DRAM。
    3. **PipeSSD执行流程 (per Mamba-2 layer)**: (a) Load dt(j), dtbias, A, x(j), B(j), C(j) from GS; (b) 1st stage RVPE: softplus→sdt→dA=sdt×A→dACS=cumsum(dA)→decay_states=exp→d2t=decay_states×sdt; (c) 2nd stage DPE#0: CBT=C×B^T → RVPE: CBTLdt=CBT×L×sdt → DPE#1: YDiag=CBTLdt×x; (d) YDiag存GS; (e) 3rd stage并行: RVPE→dCOff=exp(dACS)×C→DPE#0: YOff=dCOff×states(j-1); RVPE→dBdt=d2t×B→DPE#1: statesN=dBdt^T×x; (f) UpE: YFinal=YDiag+YOff, states(j)=exp(dACS[-1])×states(j-1)+statesN→GS。
    4. **输出**: Per-layer latency (cycles/625MHz), compute utilization = ΣFLOPs/(peak_FLOP_rate×latency), end-to-end latency = Σ(all layers + RMSNorm/Conv1D/Proj), speedup = GPU_latency/HLX_latency。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PIM-STM，一个为UPMEM PIM系统提供7种STM（Software Transactional Memory）实现的运行时库，在DPU本地以WRAM/MRAM为工作内存进行事务性并发控制。7种STM覆盖STM设计空间中的可行组合：(1) NOrec（基于全局sequence lock，粗粒度元数据，Commit-Time Locking + Write-Back）；(2) Tiny ETLWB（基于Orecs的版本时钟验证，Encounter-Time Locking + Write-Back + Invisible Reads）；(3) Tiny ETLWT（Tiny + Write-Through）；(4) Tiny CTLWB（Tiny + Commit-Time Locking + Write-Back）；(5) VR ETLWB（Visible Reads + rw-lock表 + ETL + WB）；(6) VR ETLWT（VR + ETL + WT）；(7) VR CTLWB（VR + CTL + WB）。所有实现基于UPMEM的acquire/release原子指令（256-bit硬件原子寄存器）模拟CAS——先acquire目标地址lock bit，再检查值是否匹配expected，然后更新并release。其中VR系列的rw-lock使用32-bit word编码：2bit锁模式 + 6bit reader计数 + 24bit reader身份/owner地址（write模式时用30bit存owner地址），避免访问写集检查reads-after-writes。
  - 实验比较：(a) 7种STM实现在单DPU上的吞吐量、abort rate、时间阶段分解（Read/Write/Validating(Exec)/Validating(Commit)/Other(Exec/Commit)/Time Wasted）；(b) STM元数据放置于MRAM vs WRAM的对比（2.46x-5.1x加速，几何平均2.86x）；(c) 多DPU扩展性：KMeans（共享centroid合并）和Labyrinth（独立实例）在1-2560 DPU上的speedup vs CPU，以及能效对比（RAPL测量CPU能耗，UPMEM TDP 370W估算DPU能耗）。

- 后端平台是什么，配置是什么。
  - UPMEM PIM server：2× Intel Xeon Silver 4215 CPU（单DPU实验）；1× Intel Xeon Gold 5218 CPU（32硬件线程，190GB DRAM，多DPU实验）。256GB主DRAM + 160GB PIM-enabled内存，共计2560 DPU。每DPU含：64MB MRAM (DRAM bank)、24KB IRAM（指令内存）、64KB WRAM（高速scratchpad）、1个核心支持24硬件线程（有效并行度11，pipeline深度限制）。DPU间无直连通信，跨DPU通信须经CPU转发。acquire/release原子指令基于256-bit硬件原子寄存器，通过hash函数映射地址到bit array index。多DPU实验用2500 DPU。
  - 多DPU实验CPU机器：Intel Xeon Gold 5218 CPU，190GB DRAM，RAPL测量CPU+内存能耗。

- 评估性能的软件/脚本是什么。修改了什么。
  - ArrayBench：自研合成基准，操作共享数组。A工作负载（N=12500，Y=2500只读+K=10000读写，事务读100随机+写20随机，低竞争），B工作负载（K=10，事务写4个元素，高竞争）。
  - Linked-List：基于TM的并发链表（add/remove/contains），LC（90% contains，低竞争）和HC（50% contains，高竞争）。
  - KMeans（STAMP移植）：DPU并行处理各自分片输入点，分别维护centroids私有副本，每轮结束后CPU合并centroid更新再广播新centroids。LC（k=15, N=14）和HC（k=2, N=14）。多DPU版用200K点/DPU。
  - Labyrinth（STAMP移植）：基于Lee算法的3D网格并发路径布线。CPU将独立实例分配给各DPU。S（16×16×3）、M（32×32×3）、L（128×128×3）三种网格。多DPU版用NOrec + 最优tasklet数。
  - 修改：所有benchmark移植到DPU执行，用PIM-STM API替代原始CPU STM API调用（start/abort/commit事务 + 读写WRAM/MRAM地址）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/Andre12Lopes/PIM-STM.git（包含完整PIM-STM库、benchmark移植代码）。
  - 评估原理：编译时通过宏选择STM实现和元数据放置（WRAM/MRAM）。Benchmark程序调用pim_stm_tx_begin()、pim_stm_read(addr)、pim_stm_write(addr, val)、pim_stm_tx_commit() API。每个benchmark在单DPU上以1-11 tasklet运行，多次迭代（共10轮取平均）。吞吐量以committed transactions/秒计。时间分解通过内部instrumentation采集各阶段CPU周期数。
  - Kernel输入到性能输出全过程（以NOrec为例）：tasklet调用pim_stm_tx_begin()→检查全局sequence lock是否空闲（busy则backoff等待）→执行read/write操作：read检查写集是否有之前写入值（WB特性）→若sequence lock递增则触发readset value-based validation→write缓冲到私有写集（WB不直接写入）→commit时acquire全局sequence lock（通过acquire指令）→将写集值写回MRAM/WRAM→release sequence lock。吞吐量=committed_tx / wall_clock_time。abort rate=aborted/(committed+aborted)。时间分解：Read时间（含validation开销）、Write时间、Validating (Exec/Commit)时间、Time Wasted（处理aborted事务）。

## 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：pSyncPIM在PIM processing unit上通过手写汇编实现SpMV和SpTRSV kernel的运行时计算调度。(1) **SpMV kernel**：COO格式稀疏矩阵，matrix compression策略——先row-wise partition稀疏矩阵，移除all-zero columns（compact），再将压缩子矩阵分布到各bank。通过-1填充空位作为CEXIT触发条件。Host负责外部accumulation（仅非零输出），消除远程bank访问需求。子矩阵大小限制为1KB（单memory row）以匹配输入/输出向量。Algorithm 2展示了SpMV的PIM kernel工作流：Read SpVQ0←Bank（读row/col/value）→Loop: IndMOV SRF←Bank[SpVQ0.col]（indirect scalar read）→SSpV SpVQ1←SRF⊗SpVQ0（vector multiply）→SpVDV DRF0←SpVQ0⊕Bank（vector accumulate）→Write DRF0→Bank（输出）→Read next SpVQ0→CEXIT when SpVQ1 empty。(2) **SpTRSV kernel**：采用scalar multiplication-based算法（Algorithm 3）替代dot product-based算法（Algorithm 1）以避免随机远程bank访问。使用recursive block algorithm [1] 将三角矩阵递归分解为L0/M/L1子块——L0和L1递归SpTRSV求解，M用SpMV求解。Unitriangular矩阵L*=L-I和U*=U-I（省略对角线元素），column-first COO存储。column-wise batch内划分为多个level（所有columns相互独立），对每个level：SB模式读输入向量→AB模式广播→AB-PIM模式执行kernel（对每个non-zero element执行 x[re] = x[re] - scale × ve）→SB模式切换下个level。(3) **矩阵格式**：COO格式，论文称可支持CSR/CSC（需加4个32-bit index寄存器和一个32-bit integer adder）。多精度支持从INT8到FP64。
  - 实验比较：(a) SpMV kernel：pSyncPIM vs RTX 3080 GPU (cuSPARSE) vs per-bank execution model vs SpaceA，26个稀疏矩阵，速度提升1.96× (GPU)、6.26× (per-bank)、0.56× (SpaceA)；(b) SpTRSV kernel：pSyncPIM vs cuSPARSE on GPU，6个FP64矩阵（上下三角），geomean 3.53× speedup；(c) 密集BLAS kernel (INT8/FP64) throughput：pSyncPIM vs per-bank，平均9.6× speedup；(d) 7个真实应用端到端：pSyncPIM + SpGEMM accelerator [4]，geomean 51.6× (graph apps)、2.2× (linear system solvers) vs GPU；(e) 消融：TC benchmark中SpGEMM accelerator-only vs accelerator+pSyncPIM（2.0× boost）。

- 后端平台是什么，配置是什么。
  - pSyncPIM PIM架构（仿真）：HBM2-based，4 bank groups × 4 banks/group = 16 banks/channel，16 pseudo-channels，共256 banks/cube。每bank配一个processing unit（共256 PE/cube），250MHz，32B datapath，INT8: 25.6 GIOPS / FP64: 3.2 GFLOPS per PE。External bandwidth: 256GB/s (1x) / 768GB/s (3x)，Internal bandwidth: 2TB/s。128B control register (32 PIM instructions)，16B scalar register，3×32B dense vector registers，3×192B sparse vector queues。
  - Baseline GPU: NVIDIA GeForce RTX 3080 (760GB/s external bandwidth)，CUDA 11.8，cuSPARSE，GraphBLAST library。
  - Area: PE 0.967mm² per unit，total 68.99mm² per cube。Power: max 5.0W (SpMV)。

- 评估性能的软件/脚本是什么。修改了什么。
  - 修改DRAMsim3 simulator（https://github.com/umd-memsys/DRAMsim3）添加PIM执行支持。
  - 手写PIM kernel汇编代码（15条指令：DMOV/IndMOV/SpMOV/SpFW/GthSct数据移动 + SDV/SSpV/Reduce/DVDV/SpVDV/SpVSpV向量操作 + NOP/JUMP/EXIT/CEXIT控制）。对代码重排和预加载输入减少data dependency stall和ALU latency。
  - GPU baseline: CUDA Runtime 11.8，NVIDIA Nsight Compute 2023.2.2 测量kernel执行时间。cuSPARSE library用于SpMV/SpTRSV。GraphBLAST library用于graph应用，GPU_Timer wrapper结构测量性能。
  - 数据集: 26 sparse matrices from SuiteSparse Collection [7] 和 SNAP datasets [26]。5 graph applications (BFS/CC/PR/SSSP/TC) + 2 linear system solvers (P-BCGS/P-CG)。SpTRSV矩阵：2cubes_sphere, offshore, parabolic_fem, poisson3Da, crankseg_2, rma10。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - pSyncPIM的DRAMsim3修改未明确说明开源。DRAMsim3基础版开源。
  - 评估原理：(1) 预处理——host CPU将COO格式稀疏矩阵row-wise partition、移除all-zero columns、按1KB子矩阵尺寸分布到各memory bank；对SpTRSV额外执行ILDU分解（归一化对角线）、row reordering最大化独立行数、recursive block decomposition；(2) Host通过memory command序列控制PIM执行：SB mode下发输入数据 → AB mode广播→ AB-PIM mode执行PIM kernel（processing unit运行infinite loop，各自独立CEXIT终止）→ host检测所有bank完成后切换SB mode读结果；(3) Kernel execution time = mode switching latency + PIM kernel programming time + actual computation cycles + external I/O time；(4) Wall clock time测量（匹配GPU测量方式），排除initial matrix mapping时间但包含mode switching和programming开销；(5) GPU端用wall clock time，CUDA Events计时，排除data transfer和preprocessing。

## 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：VQ-LLM在GPU kernel层面的三项核心优化，(1) **Codebook Cache分层放置**：基于offline profiling的codebook entry access frequency，将entries分为hot（>µ+3σ，存thread local registers以消除bank conflict）、medium（存shared memory）、cold（存global memory）。通过reorder-based static mapping实现，entry index < nreg→register，nreg ≤ index < nshared→shared memory，index ≥ nshared→global memory。利用GPU resource slack（不降低occupancy的resource余量）自适应确定nreg/nshared。(2) **Codebook-Centric Dataflow**：沿codebook switch axis（如Attention的H,C axis；GeMM的M,N axis for GPTVQ）切分并行化task，每个thread block仅需load一个codebook，消除多thread block重复加载codebook的off-chip traffic。split factor通过平衡Traffic_Reduce（=split_factor × Output_Size）和Traffic_Codebook（=Original_Codebook_Traffic/split_factor）自适应确定。(3) **Codebook-Centric Hierarchical Fusion**：利用GPU intra-warp shuffle (shfl_xor) API实现register-level数据重排——当dequantized data layout与后续computation所需layout不匹配时（如CQ-2 dequantize row-wise 4元素但attention需column-wise accumulation），通过mini-warp内shuffle操作在register中直接重排，消除shared memory round-trip。自适应选择：nshuffle ≤ 5用register fusion，否则用shared memory fusion。
  - 实验比较：(1) 消融实验：GC→SC→O1→O2→O3→O4 latency breakdown，覆盖GeMM/GeMV/Attention (Decode) kernel，分析各优化在不同VQ配置（QuiP#-4/AQLM-3/GPTVQ-2/CQ-2）下的效果差异；(2) 与开源实现对比（AQLM open-source kernel等）；(3) 与element-wise quantization kernel（AWQ-4bit for GeMM/GeMV, QoQ-4bit for Attention）的latency对比；(4) 不同attention baseline对比（Flash Attention, Paged Flash Attention, Paged Flash Decoding）；(5) 不同sequence length (1k/2k/4k)和batch size (1/8)下的latency scaling。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX 4090 24GB（Ada Lovelace, 128 SM, shared memory per block=48KB, registers per thread=255, L1 cache line=128 bytes）
  - Tesla A40 GPU（Ampere, ~696 GB/s memory bandwidth ≈ 67% of RTX 4090）
  - 论文指出结果可泛化到其他vendor GPU（AMD CDNA, NVIDIA H100, MTT S4000等共享相似memory hierarchy概念的GPU）

- 评估性能的软件/脚本是什么。修改了什么。
  - Baseline kernels：cutlass [43]（FP16 GeMM/GeMV），flash-attn [8]（FlashDecoding attention），qServe [31]（AWQ-4bit, QoQ-4bit element-wise quantization）
  - 评估涵盖：GeMM kernel（prefill, weight quantization）、GeMV kernel（decode, weight quantization, batch size 1/16）、Attention decode kernel（KV cache quantization, batch size 1/8, seq_len 1k/4k）
  - 修改：VQ-LLM生成优化的fused kernel替换原始kernel。对attention使用FlashDecoding [10]作为baseline dataflow，在此基础上应用codebook-centric dataflow改造。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未给出VQ-LLM独立开源链接。依赖的开源组件：AQLM (https://github.com/vahe1994/AQLM), cutlass, flash-attn, qServe, LMEval (https://github.com/EleutherAI/lm-evaluation-harness)。
  - 以CQ-2 Attention (Decode) kernel的评估为例（VQ<4,8,1> on RTX 4090）：
    1. **Kernel输入**：Quantized KV cache tensor（shape: [num_tokens, num_heads, channels/vector_size] = [4096, 32, 32]，dtype: 3-bit indices packed），codebooks（shape: [num_codebooks=32, #Entry=8, vector_size=4]，dtype: FP16），query tensor（shape: [batch_size, num_heads, head_dim] = [8, 32, 128]）。
    2. **评估原理**：用CUDA Events (cudaEventRecord) 测量单个kernel launch的wall-clock latency。对每种配置（VQ算法×kernel类型×shape）多次运行取平均。
    3. **Kernel执行流程**（VQ-LLM optimized）：
       - Grid launch: parallel_blocks = num_tokens × num_heads × (num_codebooks/split_factor)
       - Per block: (a) CBcached, boundary ← Load(CB, Budget) —— 按access frequency将hot entries load到register，medium到shared memory； (b) for each token in block's tokens: quantized_index → Access(CBcached, boundary, CB, index) —— 查表获取FP16 centroid，得到dequantized K/V； (c) for K cache: dequantized data layout与row-wise accumulation对齐，无需reorder；for V cache: dequantized data layout (4 elements/thread, row-wise) 与 required layout (column-wise weighted accumulation) 不匹配 → Reg_Fusion via shfl_xor reorder in mini-warp； (d) Attention compute: Q·K^T → softmax → weighted sum of V
    4. **性能输出**：Kernel latency (µs)，SM utilization (%), Shared memory usage (bytes), Bank conflict rate, Global→Shared traffic (bytes), Shared→Reg traffic (bytes)。Speedup = baseline_latency / optimized_latency。

## 31-UM-PIM_DRAM-based_PIM_with_Uniform_amp_Shared_Memory_Space.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：UM-PIM在kernel/运行时层面提供CPU-PIM协同计算调度和inter-PIM-unit通信API：(1) **Zero-copy PIM task offloading**：通过dual-track memory management实现CPU直接访问PIM pages，消除PIM-Ion/Ioff中的data transfer步骤（CPU-PIM和PIM-CPU内存拷贝），CPU compute和PIM compute直接共享uniform memory space。(2) **Inter-PIM unit通信API**（类似NCCL collective communication）：Scatter（读contiguous block→分散到各bank）、Broadcast（读一个bank→写所有bank，利用RC broadcast mode单burst写入所有devices）、Gather（每bank读一块→汇聚到一个bank）、All-Gather（每bank读一块→拼接→广播到所有bank）。(3) **Nested loop ordering优化**：利用RC filtered data locality，原则为read access时br和l loop在dr loop之外，write access时l/br/dr loop在dw loop之外。(4) **malloc_pim API**：分配PIM page内存，通过THP (Transparent Huge Pages) mmap+madvise分配256MB chunks，mlock防止swap out，读取/proc/pid/pagemap获取CPN，通过ACN指令写入RCL。(5) **Instruction支持**（MMIO配置）：ACN（append chunk to RCL&PCL）、CCL（clear RCL&PCL）、CRC（flush RC to DRAM）、BCM（set/unset RC broadcast mode）。(6) **Offloading decision优化**：低offloading overhead使compiler可offload更多program segments到PIM（平均+7.8%），fork-join模式中CPU遍历PIM results的nested loop order利用RC hit rate优化。
  - 实验比较：(a) UM-PIM vs PIM-Ion（interleaving on + software addr translation & re-layout）和PIM-Ioff（interleaving off + software）——端到端PIM workload计算时间及CPU compute/Xfer/PIM compute breakdown。(b) Scatter/Broadcast/Gather不同block size（64-16K bytes）下的延时。(c) CPU-PIM和PIM-CPU数据传输不同block size（128-8M bytes）下的延时。(d) 不同PIM rank数（1/2/4/8/16）和CPU core数（1/2/4/8）下BFS和PageRank的scaling。(e) All-Gather不同nested loop order下的时间、RC hit rate和读带宽。(f) 不同memory interleaving方法下的speedup对比。

- 后端平台是什么，配置是什么。
  - 主机CPU：8-Core O3CPU @3.2GHz (GEM5模拟X86 ISA)，L1I/L1D 32kB/32kB (Assoc:8)，L2 1MB (Assoc:16)，L3 22MB (Assoc:22)，Cache Line 64B。
  - DRAM：DDR4-2400，8 channels × 4 ranks，8 banks per rank，8 devices per rank，8GB/rank。Timing: tBURST=3.32ns, tRCD=tCL=tRP=14.16ns, tRAS=32ns, tRRD=3.32ns。
  - PIM Unit：UPMEM DPU @500MHz，16 Tasklets，64 DPU per rank（bank-level PIM）。真实硬件用于测量PIM task执行时间。
  - UM-PIM系统：{4 DRAM channels + 4 PIM channels} × 4 ranks，共32 PIM ranks、2048 PIM units。

- 评估性能的软件/脚本是什么。修改了什么。
  - CPU workloads：SPEC CPU 2006（lbm, mcf, sjeng, dealII, xalancbmk, GCC），评估不同interleaving策略下的CPU性能。
  - PIM workloads：12 benchmarks——BFS (#node:196608, sparsity:5e-5), PageRank (Kronecker gen, #node:8k, Deg:12), MLP (neuron:16k, layers:3), NW (size:64k), UNI (len:256M), SCAN-RSS/SCAN-SSA (len:8M), SEL (len:32M), HST-S (len:1.5M, bins:256), TC (#node:8k, Deg:12), WFA (len:20k×110), RL (iter:100)。来源：PRIM [24], GAP [4], AiM [16], PIM-ML [23]。
  - 修改：(a) CPU host program中使用UPMEM SDK的数据转移函数（地址翻译+数据重排多线程API）模拟PIM-Ion/Ioff的software overhead。(b) UM-PIM中CPU host program通过malloc_pim()分配PIM pages，直接以virtual address访问PIM pages（不用显式数据转移）。(c) Inter-PIM通信使用scatter/broadcast/gather/all-gather API替代原始inter-bank通信模式。(d) PIM task latency通过UPMEM真实DPU测量后插入GEM5 task launch point。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明UM-PIM软件栈是否独立开源。依赖开源组件：GEM5 simulator, Ramulator2, UPMEM SDK（商业产品，文档公开于https://sdk.upmem.com/2023.2.0/）。
  - 评估原理（以BFS PIM workload为例）：(1) **程序编译**：host CPU program用gcc编译（含malloc_pim() API调用→分配PIM chunk→THP mmap+madvise→mlock→读取/proc/pid/pagemap获取CPN→ACN指令写入RCL）；DPU kernel用UPMEM SDK clang编译。(2) **执行流程**：CPU program初始化数据→malloc_pim()为每个PIM unit分配PIM page→CPU直接写PIM page（zero-copy，无显式Xfer步骤）→offload DPU task→DPU读取本地PIM page执行BFS计算（利用PCL翻译VAddr→PAddr→HWAddr）→DPU完成后CPU直接读PIM page result→CPU执行无法offload的操作（如transcendental functions）→fork-join同步→CPU按优化nested loop order遍历PIM results（遵守br/l外层、dr内层规则，利用RC hit rate）→下一轮offload。(3) **性能测量**：GEM5 trace CPU compute cycle、DRAM access cycle（含UM-PIM hardware overhead: tUI for RCL+ATM, tRC for RC access, RC hit/miss分支延时）、data transfer cycle。PIM task latency从UPMEM真实DPU测量插入。(4) **输出**：端到端墙钟时间，CPU compute/Xfer/PIM compute三部分breakdown（ms），speedup vs PIM-Ion/Ioff。Data transfer时间曲线（time vs block size: 64-8M bytes for CPU-PIM/PIM-CPU，64-16K bytes for scatter/broadcast/gather）。


## 27-CoCoTree: A Computation-Capable Architecture for Collective Communication in Scalable PIM

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：CoCoTree将DIMM PIM系统上的集体通信操作（All-Reduce, Reduce-Scatter, Reduce, Broadcast, All-Gather等）从host CPU forwarding offload到树形硬件网络，通过in-network computation在每个Co-Node进行byte-granular reduction，实现kernel级集体通信的硬件加速。支持多种reduction操作（sum/bitwise AND/OR/XOR/unsigned min/max，以及FP32 sum/min/max），支持任意整数宽度、动态宽度扩展和流水线并行（pipeline execution across Co-Nodes，允许多个通信round在不同tree level并发执行）。Programming interface为two-phase模型（configuration + computation），PE通过API（CoCoTree::initConfig(), CoCoTree::configTree(), CoCoTree::send(), CoCoTree::waitReceive(), CoCoTree::getReceived()）直接在PIM kernel中表达集体通信。
  - 实验比较：与Baseline UPMEM（CPU forwarding）、DIMM-Link、PIMnet比较5种集体通信操作（Broadcast/All-Gather/Reduce/Reduce-Scatter/All-Reduce）在64-2048 PE下的吞吐量，以及8个PIM workload（BFS/HST/RED/MLP/GEMV/SpMV/CC/EMB）的端到端性能。CoCoTree在All-Reduce达95.6× speedup，Reduce和Reduce-Scatter平均54.5×/54.4×，Broadcast和All-Gather达1.4× vs PIMnet。Ablation study验证Tree Network (N)、In-network Computation (C)、Pipelining (P)各组件的贡献。

- 后端平台是什么，配置是什么。
  - 主机：2×Intel Xeon Silver 4216 (2.2GHz, 32核)，256GB内存
  - PIM：20×UPMEM BC021B DIMM (DDR4-2400)，DPU频率350MHz，总计2530 PEs，160GB PIM内存。每个PIM chip含8个独立64MB DRAM bank，每个bank配一个32-bit scalar in-order RISC DPU含24KB IRAM和64KB WRAM。
  - SDK: upmem-2023.2.0-Linux-x86_64
  - 编译器: G++ 12.3.0

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研C++ cycle-accurate模拟器，通过DPI-C接口集成Verilator生成的CoCoTree RTL模型。模拟器替换了UPMEM host-forwarding通信路径为CoCoTree（及DIMM-Link/PIMnet）的集体通信模型。
  - Benchmark workloads来自PrIM benchmark suite [37]和SparseP [35]：BFS (rMat graph), HST (1536×1024), RED (6.3M elements), MLP (256×256), GEMV (1024×64), SpMV (rtn matrix), CC (rMat graph), EMB (RM2 dataset)。所有host端程序使用OpenMP并行。
  - 修改：通过CoCoTree API替换原有host-forwarding collective通信代码。例如BFS的node bitmap frontier reduction：原UPMEM用host CPU显式DMA gather→process→scatter，CoCoTree改为PE内直接调用CoCoTree::send()注入本地bitmap→Co-Nodes做bitwise OR reduction→CoCoTree::getReceived()获取结果。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明CoCoTree模拟器是否开源。使用开源UPMEM SDK和PrIM benchmark suite。
  - 评估原理：C++模拟器读取PIM workload binary，识别其中的CoCoTree API调用。对于每次集体通信操作，模拟器使用Verilator生成的RTL模型进行cycle-accurate仿真：PE通过DMA将数据传输到Co-Leaf → Packing Unit按byte-granularity重排并pack为32-bit packets → packets经handshake-based link (valid/ready)发送到Co-Node → Co-Node根据配置执行routing（7种模式）和FU computation（byte-granular add/OR/max等）→ 结果沿树上行到root → 根据ADDR/STH向下broadcast到目标PEs → Co-Leaf Unpacking Unit解包还原 → PE通过polling获取结果。
  - 输入→性能输出：Workload源码（含CoCoTree API）→ 编译为DPU binary + host binary → C++模拟器load binary并执行 → 对每个集体通信API调用触发CoCoTree RTL仿真 → 统计各操作耗时（configuration phase + computation phase）→ 输出：各集体通信吞吐量（operation data size / execution time），端到端workload wall-clock时间，通信/计算时间breakdown。功耗数据通过Yosys+iEDA从RTL门级网表提取。

## 26-SoMa: Identifying, Exploring, and Understanding the DRAM Communication Scheduling Space for DNN Accelerators

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SoMa通过DLSA Exploration Stage（第二阶段）实现细粒度的DRAM tensor调度优化——在运行时层面为每个DRAM tensor（ifmaps/weights/ofmaps）调整DRAM Tensor Order（访问顺序）和Living Duration（Start/End生命周期，控制prefetching和delayed storing的时机）。具体而言：(1) 通过prefetching（提前将weights/ifmaps从DRAM加载到GBUF）利用DRAM空闲时段；(2) 通过delayed storing（延迟将ofmaps写回DRAM）平滑DRAM带宽峰值压力。这实际上是对DRAM通信kernel执行时序的调度优化——在总执行timeline上调整每个load/store操作相对于compute tile的位置，以最大化compute和DRAM access的overlap。
  - 实验比较：与Cocco baselin对比(Cocco使用经典double-buffer策略，即前一tile prefetch、后一tile store)。展示第一阶段(LFA)和第二阶段(DLSA)逐步优化后Computing Resources Utilization和Average Buffer Utilization的提升，并与Theoretical Maximum Computing Resources Utilization（蓝色菱形，理想情况所有DRAM tensor或compute tile无stall连续执行）对比。第二阶段平均仅距理论上限3.1%。

- 后端平台是什么，配置是什么。
  - Edge: 16 TOPS, 8MB GBUF, 16GB/s DRAM BW, TSMC 12nm, 1GHz
  - Cloud: 128 TOPS, 32MB GBUF, 128GB/s DRAM BW, TSMC 12nm, 1GHz

- 评估性能的软件/脚本是什么。修改了什么。
  - 使用SoMa框架内置的Evaluator模块进行性能评估，未使用外部独立benchmark工具。
  - Evaluator采用local-to-global评估方法：对每个compute tile调用Core Array Scheduler & Evaluator（采用经典数据流调度方法[32][42]）评估tile内的并行度、数据复用和能耗；对每个DRAM tensor按读/写数据量×单位能耗计算DRAM能耗和传输时间。总延迟通过三种依赖条件推导：(a)前序DRAM tensor完成；(b)ifmaps/weights的Start≤当前tile ID；(c)ofmaps需等生成compute tile完成。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/SET-Scheduling-Project/SoMa-HPCA2025
  - 评估原理：Evaluator从local到global递进评估。以五层网络(ABCDE)为例：(1) 每个compute tile（如A1）：Core Array Scheduler在GBUF→L0 buffer级别探索tile内sub-tile划分，考虑各core并行和数据复用，评估GBUF-L0交互和计算能耗/延迟；(2) 每个DRAM tensor（如WA、IA1）：按数据量(Bytes)×单位DRAM读/写能耗(Joule/Byte)计算DRAM能耗；按数据量/DRAM带宽计算传输时间；(3) 总延迟根据依赖图合并所有compute tile和DRAM tensor时间线。Compute tile开始条件：所有所需数据已就绪 且 所有End≤当前tile的DRAM tensor已完成。DRAM tensor开始条件：前序DRAM tensor已完成 且 (ifmaps/weights的Start≤当前tile 或 ofmaps的生成tile已结束)；(4) 总能耗=Σ各compute tile能耗+Σ各DRAM tensor能耗。优化目标：Energy^n × Delay^m。

## 24-Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management.pdf (FlashGen)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：FlashGen修改了vLLM中的Flash-Attention（prefill phase）和Flash-Decoding（generation phase）kernel，以支持非连续物理内存中存储的KV blocks。在PagedAttention内存管理下，前序轮次的history KV分布在多个不连续的物理page中。标准的Flash-Attention/Flash-Decoding kernel假设KV在GPU内存中连续存储，不适用于multi-turn场景下的KV cache恢复。FlashGen的修改将attention kernel扩展为支持gather-style的非连续KV加载——类似于FlashInfer库的实现方式，允许kernel从多个不连续的KV block地址读取并计算attention。
  - 此外，FlashGen实现了layer-by-layer pipelined KV restoration：当从host memory恢复KV时，GPU在第L层decoder layer执行attention计算的同时，DMA engine在后台传输第L+1层的KV（类似于PipeSwitch的pipeline context switching方法）。这隐藏了host→GPU的KV传输延迟。Batch-aware KV restoration进一步优化：当迭代batch仅包含generation phase（计算量小），若某请求的KV恢复未完成，scheduler将其从当前batch中排除，避免GPU compute unit因等待KV传输而stall，直到KV就绪后再重新编入batch。
  - 实验比较：与vLLM（未修改的Flash-Attention/Flash-Decoding）、CachedAttention对比。通过TPOT CDF（Section 5.2, Figure 13）评估kernel修改对token生成延迟的影响——FlashGen的P99 TPOT为103ms（vLLM为608ms）。通过prompt phase时间分解（Section 5.2, Figure 12）评估KV传输vs重算的延迟构成。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 80GB GPU（Azure Standard_NC48ads_A100_v4实例）
  - 单GPU: OPT 13B, Llama-2 13B；双GPU tensor parallelism: OPT 30B, Llama-2 70B
  - CUDA 12.1, PyTorch v2.3
  - Host: 440GB DRAM + 2×960GB NVMe SSD RAID-0，但kernel修改本身在GPU端执行

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于vLLM的end-to-end serving framework评估（非独立kernel microbenchmark）。FlashGen-Cache的KV restoration pipeline在GPU端通过修改的Flash-Attention/Flash-Decoding kernel + CUDA stream异步DMA实现。kernel修改的核心：vLLM原版Flash-Attention kernel假设Q·K^T·V中K/V在连续内存，FlashGen修改后支持从多个PagedAttention block指针列表gather非连续KV数据——实现方式与FlashInfer（https://flashinfer.ai）的grouped attention / paged attention kernel设计类似。
  - 评估通过控制client数量sweep负载，测量端到端TPOT、TTFT等指标。KV hit rate通过内置统计收集，分解为GPU hit/CPU hit/SSD hit/recomputation四部分。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - FlashGen代码开源情况：论文未明确说明是否开源，未提供代码仓库链接。vLLM为Apache 2.0开源，FlashInfer为开源库。
  - kernel输入到性能输出的全过程（以OPT 30B多轮对话prefill phase + KV restoration为例）：
    1. **输入**：Session的history KV在host memory中以PagedAttention block格式存储（每block固定大小，如16 tokens × num_kv_heads × head_dim × 2 (K+V) × 2 bytes (FP16)）。KV blocks的物理地址列表（block table）由vLLM的block manager维护。
    2. **Prefill调度**：Scheduler决定调度该请求时，检查block table确定哪些block已在GPU、哪些需从host传输。发起host→GPU DMA传输（cudaMemcpyAsync on a separate CUDA stream），传输目标为GPU上新分配的KV cache blocks。
    3. **Pipeline执行（Layer-by-layer）**：Decoder Layer 1开始——修改的Flash-Attention kernel接收Q（当前prompt tokens）、K/V地址列表（GPU上已有的blocks + 新传输的blocks，可能不连续）。Kernel内按block粒度gather K/V：对每个attention head，遍历block table，从每个block的物理地址偏移读取对应K/V片段，拼接为逻辑连续的K/V序列后执行标准Flash-Attention tiling（Q·K^T→softmax→×V，使用GPU SRAM tiling减少global memory访问）。同时后台CUDA stream传输Layer 2的KV blocks。
    4. **非连续KV处理**：修改后的kernel关键差异——原版Flash-Attention内层循环遍历连续K/V地址，修改版增加一个外层循环遍历block table entries，内层对每个block执行原版tiled attention。类似FlashInfer's "paged attention"设计：将block table作为额外输入参数传入kernel。
    5. **Generation Phase**：每step生成1个new token，Flash-Decoding kernel处理——Q为单个token，K/V为所有已缓存token（含history和已生成token）。修改后的kernel同样支持非连续KV blocks。若KV传输未完成且batch仅含generation phase，该请求被排除出batch。
    6. **性能输出**：TPOT（每个generation step的wall-clock时间）通过kernel launch到completion的CUDA event计时。FlashGen P99 TPOT=103ms vs vLLM P99=608ms（OPT 30B on ShareGPT），主要收益来自消除history token重算。

## 21-Tandem Processor: Grappling with Emerging Operators in Neural Networks

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Tandem Processor（32-lane SIMD处理器）运行时执行非GEMM算子，覆盖五大类：(1) element-wise数学运算（Add, Sub, Mul, Exp, Sqrt, Floor, Ceil, Pow, Reciprocal等），(2) element-wise激活函数（ReLU, LeakyReLU, Clip, Tanh, Sigmoid, GeLU），(3) reduction类算子（Depth-wise Conv, MaxPool, GlobalAveragePool, ReduceMean, Softmax），(4) data layout transformation（Transpose, Reshape, Concat），(5) datatype cast（FXP32/FXP16/FXP8/FXP4）。复杂算子（如GeLU、Softmax）通过基本ALU指令组合实现（INT32精度），例如GeLU用5次乘法+3次加法+sign+absolute+minimum。
  - Tandem Processor与GEMM unit通过tile粒度软件流水实现overlap执行：GEMM unit完成tile写入Output BUF后，Tandem Processor取得Output BUF ownership直接计算，避免显式数据拷贝。采用double-buffering：GEMM unit处理tile N+1时Tandem Processor处理tile N。
  - 实验比较：与off-chip CPU fallback、dedicated units + CPU、Gemmini（RISC-V + dedicated units）、TPU+VPU、A100 CUDA Cores进行端到端和非GEMM-only性能比较。

- 后端平台是什么，配置是什么。
  - 后端平台：NPU-Tandem，包含：(1) 32×32 systolic array GEMM unit（384KB scratchpads, 128KB accumulators, INT8乘法/INT32累加, 1GHz）；(2) Tandem Processor（32 SIMD lanes, 128KB Interim BUF 1&2 scratchpads, 32-slot IMM BUF, INT32 ALU, 1GHz）；(3) Execution Controller FSM + Instruction Buffer + Output BUF（GEMM unit与Tandem Processor间的共享缓冲区）。
  - RTL使用Synopsys DC综合（65nm/15nm），布局布线验证，CACTI-P建模片上内存功耗。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研cycle-accurate模拟器：读入编译器生成指令，以cycle精度模拟Tandem Processor的pipeline各阶段（Fetch→Decode/Iterator Tables→Strided Address Calculation→Scratchpad Read/Execute→Write Back）。模拟器追踪每周期每种硬件资源的活跃状态和能耗。验证：模拟器输出与RTL仿真输出误差≤5%。
  - 对比baseline的评估方法：(a) CPU baseline使用ONNX Runtime测量Intel Core i9-9980XE CPU时间；(b) Gemmini使用FireSim cycle-accurate FPGA加速模拟器；(c) GPU baseline使用TensorRT v7.2.3和ONNX Runtime CUDA Execution Provider；(d) TPU+VPU行为在自研模拟器内按Google VPU专利[58]建模；(e) PCIe通信延迟使用Xilinx Alveo u280 FPGA实测。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GeneSys项目（https://actlab-genesys.github.io/）。以BERT中GeLU+GEMM subgraph执行为例：(1) ONNX图输入编译器，识别MatMul→Add→GeLU→MatMul subgraph；(2) 编译器进行layer fusion（将前MatMul+Add+GeLU融合为一个block），uniform tiling（例如沿sequence维度分tile）；(3) 编译生成混合指令流：GEMM instructions（配置systolic array的MatMul）+ synchronization instructions（标记GEMM/SIMD区域边界）+ Tandem Processor instructions（Load tile→ALU Add→ALU GeLU分解原语→Store tile→Sync release OBUF）；(4) 模拟器加载指令，Execution FSM按状态机调度：Inst.Dispatch→GEMM-Tandem状态（tile-0 GEMM compute→tile-0 done→Tandem Processor取得OBUF并执行非GEMM→同时GEMM tile-1 compute→...）→Block Done；(5) 模拟器逐cycle追踪systolic array MAC利用率、Tandem Processor SIMD lane利用率、OBUF occupancy、scratchpad读写冲突，输出各tile的cycle计数和能量统计；(6) 最终输出端到端总延迟、非GEMM算子alone延迟（通过仅执行非GEMM-only block测量）、各算子类别runtime/energy breakdown。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：consumption-centric（以消费为中心）的subgraph-level execution scheme，替代传统production-centric/production-at-once方案。该scheme包含三阶段flow来决定subgraph中每个node（layer）的execution behavior：
    - Stage-1：确定输出nodes的tile size（类似传统单层scheduling，优化compute utilization）。
    - Stage-2：反向推导（reverse topological order）确定各node的data update offset Δ和memory allocation size x。使用LCM对齐来自不同consumers的input offset需求，计算每个producer update对应的consumer update次数。tile size x(u)=max_v∈children{χ(u,v)}，取所有output所需的最大input tile。
    - Stage-3：确定upd_num（每个subgraph elementary operation中各node的memory更新次数）和execution sequence。使用最小co-prime的upd_num解（unique solution）对应最小elementary operation。
  - Memory管理方面：引入MAIN region和SIDE region双区域管理。MAIN region存储PE source data（tile of P0×Q0×C）。SIDE region保留水平方向overlap data。滑动卷积窗口时：垂直overlap data在MAIN region本地复用；水平overlap data从SIDE region加载（path ①）；底部水平slice新数据写入SIDE region供下一row loop用（path ②）。
  - 实验：通过图3对比L=1/3/5层fusion的EMA和bandwidth reduction，证明subgraph-level相对于layer-level的优势（EMA reduction 42.3%~74.7%, BW reduction 26.8%~67.8%）；通过图11的graph partition实验对比Cocco与其他方法（Greedy/DP/Enumeration）的EMA cost和BW requirement。

- 后端平台是什么，配置是什么。
  - SIMBA-like分层加速器：每core含1MB global buffer、1.125MB weight buffer、4×4 PE array（每PE含8×8 MAC）。12nm工艺，1GHz。DRAM energy 12.5pJ/bit，每core DRAM bandwidth 16GB/s。
  - 硬件实现（Section 3.2, Figure 8）：buffer region manager为2N-depth register file（N=64, 272-Byte, 17-bit address），逻辑分区global buffer为多个MAIN/SIDE/output region。面积ratio仅0.18%（在1MB 64bit-width global buffer旁）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于Timeloop和MAESTRO评估器开发的自定义simulator，实现：(1) consumption-centric tile size derivation代替output-centric tiling；(2) MAIN/SIDE region memory allocation model；(3) 无padding data的memory access计算；(4) subgraph-level EMA/energy/latency聚合计算，延迟取计算周期和外部通信周期的最大值。
  - 算术和内存overhead从12nm库综合RTL实现提取。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：Cocco框架（含simulator和GA optimizer）未开源。Timeloop和MAESTRO评估器为开源项目，但Cocco基于它们进行的subgraph-level修改未公开。
  - 评估原理和流程（以ResNet50某3-layer subgraph {Conv3_1, BN3_1, ReLU3_1} 在separate buffer配置下执行为例）：
    1. **输入**：subgraph DAG（node信息含kernel_size F、stride s、tensor维度H×W×C×N）。Buffer约束：global buffer size, weight buffer size。PE配置：8×8 MAC/PE, 4×4 PE array。
    2. **Stage-1（tile size of output nodes）**：对subgraph的output nodes（无后继nodes或需writeback到DRAM的nodes），按PE array利用率和buffer容量选择output tile size（如P0=4, Q0=4, C_tile=64）。
    3. **Stage-2（反向推导）**：从output nodes沿DAG反向遍历——对每对(producer v, consumer u)计算: Δ(v)=lcm{Δ(u)·s(u)}（对齐offset要求），x(v)=max_u{χ(v,u)}，χ(v,u)=F(v)+(Δ(u)/s(v)-1)×s(v)。例如output tile Δ(output)=2→Conv3_1(3×3/2 kernel)的Δ=4, x=6（需6-wide tile生产供2-wide output）。
    4. **Stage-3（upd_num和执行序列）**：确定co-prime upd_num解→确定subgraph elementary operation的完整步骤序列→每个步骤中哪些nodes需要新数据load、哪些compute、哪些数据更新到SIDE region。
    5. **Memory allocation检查**：MAIN region总需求=Σ各node x(u)×C_tile的activation buffer + weights buffer。SIDE region需求=水平overlap数据（≈Σ(Fy-sy)×C_tile）。若总需求>buffer容量→报超限（Cocco通过split-subgraph处理）。
    6. **EMA计算**：weight_load=Σ每个node的weight大小（从DRAM加载），input_load=subgraph输入nodes的activation从DRAM加载量，output_store=需writeback nodes的结果写回DRAM量。总EMA=weight_load+input_load+output_store bytes。
    7. **Energy计算**：EMA_energy=EMA×12.5pJ/bit，buffer_static_energy=buf_size×leakage_pJ/cycle，compute_energy=MAC ops×MAC_energy_pJ。总energy=EMA_energy+buffer_energy+compute_energy。
    8. **Latency计算**：compute_cycles=M×N×C×K²/(8×8×16)（MAC array parallel），comm_cycles=EMA_bytes/(16GB/s÷1GHz)。Latency=max(comm_cycles, compute_cycles)。
    9. **输出**：该subgraph的EMA、Energy、Latency值，汇总到genome fitness评估。

## 16-RELIEF_Relieving_Memory_Pressure_In_SoCs_Via_Data_Movement-Aware_Accelerator_Scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：RELIEF (RElaxing Least-laxIty to Enable Forwarding) 在线加速器调度策略，运行于SoC硬件管理器（Hardware Manager）之上。RELIEF扩展Least Laxity First (LL)调度，新增两个关键机制：(1) 转发节点优先级提升——当producer加速器执行完成时，其child节点（consumer）被标记为"转发节点"(forwarding node)，因为其可直接从producer的scratchpad memory转发数据。RELIEF在就绪队列中将这些转发节点提升到队列前端，绕过其他已就绪的低laxity节点，优先利用硬件数据转发机制；(2) is_feasible()可行性检查——在提升转发节点优先级前，遍历就绪队列中所有比候选节点优先级更高的任务，检查每个任务的laxity是否大于候选节点的运行时间。如果任意高优先级节点的剩余laxity不够容忍候选节点的运行延迟，则不进行优先级提升（避免deadline miss）。若is_feasible()返回false，节点按原始laxity位置插入就绪队列。
  - RELIEF算法流程（Algorithm 1+2）：
    1. 当加速器完成一个节点执行时，遍历该节点的所有child节点。
    2. 对每个child，若all parents已完成，预测其运行时间(runtime)，计算laxity = deadline - runtime，将其插入候选转发队列fwd_nodes（按laxity排序）。
    3. 对每种加速器类型，max_forwards = 当前空闲加速器实例数。当fwd_nodes非空时，弹出队首node，调用is_feasible(ready_queue, node, index)检查：从ready_queue头部开始，找到第一个非转发节点且laxity>0的节点，若其laxity > node.runtime，则安全提升。若可行，将node插入ready_queue前端；否则插入laxity排序位置。
  - 实验比较：5种学术界SOTA调度策略——FCFS (GAM+中的round-robin)、GEDF-D (按DAG deadline)、GEDF-N (按临界路径分析分配node deadline)、LAX (Least Laxity First变体，负laxity节点de-prioritize)、HetSched (laxity + sub-deadline ratio)，加上LL和RELIEF-LAX两种消融变体。4种不同争用水平：低争用（单个application）、中争用（2-application组合）、高争用（3-application组合）、连续争用（3-application循环执行至50ms截止）。

- 后端平台是什么，配置是什么。
  - gem5-SALAM模拟器 [47]：cycle-accurate加速器建模，消耗LLVM IR描述的加速器和配置文件，提供执行时间、能耗等统计。
  - 模拟移动SoC配置（Table VI）：ARM Cortex-A7 1.6GHz单核顺序CPU（作为硬件管理器微控制器），32KB 2-way L1-I + 32KB 4-way L1-D cache。LPDDR5-6400主存：1个16-bit通道，1 rank，BG模式，tCK=1.25ns，burst length=32，峰值带宽12.8GB/s。Full-duplex bus互联：宽度16B，峰值带宽14.9GB/s。此外评估crossbar switch互联（Section V-H）。
  - 7个图像处理加速器（各含scratchpad memory，均工作在1GHz）：
    - ISP (SPAD 115,204B): demosaicing, color correction, gamma correction
    - grayscale (SPAD 180,224B): RGB转灰度
    - convolution (SPAD 196,708B): 最大5×5 filter卷积
    - elem-matrix (SPAD 262,144B): 元素级矩阵操作（add, mult, sqr, sqrt, atan2, tanh, sigmoid）
    - canny-non-max (SPAD 262,144B): 非极大值抑制
    - edge-tracking (SPAD 98,432B): 基于阈值的边缘标记和增强
    - harris-non-max (SPAD 196,608B): 3x3网格角点值增强
  - 每个加速器使用ED²最小化原则设计（类似gem5-SALAM前序工作[47, 53]），足够的scratchpad处理128×128输入+double buffered输出。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于gem5-SALAM模拟器（开源，GitHub: Sacusa/gem5-SALAM, HPCA_2024分支，BSD-3许可）。硬件管理器运行于Cortex-A7上裸机C代码，实现完整的端到端执行：中断处理(ISR) → 调度器运行 → driver功能 → DMA传输 → 加速器执行。
  - 修改/新增内容：
    1. 调度器：在gem5-SALAM中实现RELIEF调度算法（Algorithm 1+2），替换原有的round-robin/FCFS调度。在每个加速器完成执行时调用RELIEF()函数，传入finishing node参数。
    2. 转发机制：在硬件管理器metadata中新增字段——producer_acc和producer_spm（child node数据结构中，告知driver从哪个producer加速器和哪个scratchpad partition读取）、output[]（加速器metadata中，跟踪哪次执行产生了当前scratchpad分区的输出）、ongoing_reads[]（记录正在从该scratchpad分区读取数据的consumer数量，防止写后读冲突）。consumer DMA engine可直接从producer scratchpad读取数据，无需经主存。
    3. 预测器：实现compute time预测（固定函数加速器，通过输入大小+请求操作类型查表）、data movement预测（分析DAG结构+节点状态，预测colocation和forward）、memory bandwidth预测（支持Last value/Average/EWMA三种）。
    4. 基准程序：5个应用的C++ gem5-SALAM基准——Canny edge detection、Richardson-Lucy deblur、Harris corner detection、GRU、LSTM。
  - 运行脚本（来自Appendix artifact）：`./run_combinations_3.sh nproc` 启动高争用模拟，结果存于$M5_PATH/BM_ARM_OUT/comb_3。绘图脚本：plot_forwards.py, plot_data_movement.py, plot_accelerator_occupancy.py, plot_slowdown.py, plot_deadlines_met.py。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：是。gem5-SALAM代码在GitHub https://github.com/Sacusa/gem5-SALAM/tree/HPCA_2024，Zenodo DOI: https://doi.org/10.5281/zenodo.10237117，BSD-3 license。
  - 安装步骤：pip install -r requirements.txt → 按README.md构建gem5 → export M5_PATH=`pwd` → (可选) 构建benchmarks binary和accelerator LLVM IR。
  - 评估原理：gem5-SALAM是一款LLVM-based加速器建模系统架构模拟器。加速器以LLVM IR描述（C语言高层描述→clang编译为LLVM IR），模拟器解析LLVM IR为cycle-accurate的执行模型，同时模拟内存系统（LPDDR5 controller model）、互联（bus/crossbar）、DMA engine和硬件管理器CPU。模拟器统计execution time、memory traffic（DRAM和scratchpad）、energy consumption（使用gem5-SALAM内置energy model）。
  - 端到端执行流程（以Canny edge detection: ISP→grayscale→convolution→elem-matrix→canny-non-max→elem-matrix→edge-tracking为例）：
    1. Host CPU程序构建DAG节点（arm node data structure: acc_id, inputs[], children[], parents[], deadline），写入共享内存中的就绪队列。
    2. 硬件管理器（Cortex-A7）从就绪队列中读取root节点（ISP），按调度策略排序后，通过driver函数将任务启动到ISP加速器上——driver写加速器的MMIO寄存器配置计算参数，写DMA engine的MMIO寄存器启动数据加载（从主存加载128×128原始图像）。
    3. ISP执行（34.88us compute, 8.71us memory），完成后发中断给硬件管理器。
    4. 硬件管理器ISR被触发，读取ISP的输出状态，调用RELIEF(finishing_node=ISP_node)：检查ISP的children（grayscale节点），若其所有parents已完成，预测runtime，计算laxity，检查is_feasible()决定是否优先级提升。若提升，将grayscale插入ready_queue头部；否则按laxity插入。
    5. grayscale被调度执行——若数据转发被启用，consumer driver直接通过DMA从ISP的scratchpad memory读取原始图像数据（不需要经主存回写再读取）。grayscale执行（10.26us compute + data transfer），完成→中断。
    6. 依次类推，每个后续节点（convolution→elem-matrix→...→edge-tracking）均经过RELIEF调度和数据转发决策。consumer若能与producer共址（使用同一类型加速器），硬件管理器在producer完成后直接将consumer部署到同一加速器（colocation），完全消除数据搬移。
    7. 末端节点edge-tracking完成后，硬件管理器更新node status（标记完成），host CPU轮询或等待中断获知整个DAG完成。
    8. gem5-SALAM输出statistics dump：每加速器的compute time、memory access time、DMA time、idle time；DRAM traffic（read/write bytes）；scratchpad traffic（forward/colocation bytes）；调度器延迟（push task to ready queue latency）；每条互联的占用率。绘图脚本解析这些statistics生成论文中的Figure 4-13。

## 15-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：DMX运行时系统将数据重构kernel从CPU offload到DRX执行，运行时组件包括：(1) DRX driver——初始化command queue，管理RX/TX数据队列对（每个加速器一对，每对100MB，200MB总空间支持最多40个加速器），维护head/tail指针跟踪enqueue数据；(2) point-to-point DMA——通过dma-buf API在加速器与DRX间直接传输数据，绕过CPU系统内存，《发送方DRX读取RX队列数据→执行数据重构→写入TX队列→DMA到目标加速器》，目标加速器的DRX可被绕过（通过内部PCIe multiplexer pass-through）；(3) 中断处理——低中断率时使用interrupt coalescing合并burst中断，高中断率时切换为polling模式（Linux NAPI风格设计）；(4) 集合通信支持——one-to-many broadcast（源DRX通过back-to-back point-to-point DMA将重构后数据发送到多个目标加速器/DRX），many-to-one all-reduce（scatter-reduce + all-gather，DRX执行reduction操作）；(5) 控制平面——host CPU运行OpenCL-style host程序，通过daemon协调加速器和DRX上的kernel执行，每个加速器/DRX有独立的command queue和execution context。
  - 实验比较：(i) Multi-Axl baseline（数据重构在CPU上执行，加速器间数据经CPU搬移），(ii) DMX（数据重构在DRX上执行，point-to-point DMA绕过CPU），(iii) broadcast在4-32个加速器下的性能对比，(iv) all-reduce在4-32个加速器下的性能对比，(v) 扩展到3 kernel pipeline的performance scaling。

- 后端平台是什么，配置是什么。
  - AWS F1实例（多个Xilinx UltraScale+ VU9P FPGA，PCIe x16连接Intel Xeon Platinum 8260L CPU@2.4GHz，64GB内存，hyperthreading disabled）。1-30个加速器对应1-15个并发应用，每应用2个kernel。DRX配置128 RE lanes @250MHz(FPGA)/1GHz(ASIC)，64KB scratchpad + 64KB I-cache + 8GB DDR4 3200 per DRX。PCIe Gen3 baseline，灵敏度扩展至Gen4/Gen5。

- 评估性能的软件/脚本是什么。修改了什么。
  - DMX驱动栈：基于GEM（Graphics Execution Manager [107, 108]）管理命令执行和内存操作，通过ioctl syscall执行命令和内存映射。使用dma-buf API [109]建立加速器-DRX point-to-point DMA。DRX driver在PCIe枚举时发现所有连接加速器，静态分配内存地址空间。
  - 修改内容：新增DRX driver（Linux kernel module）——枚举DRX和加速器→分配RX/TX数据队列→共享queue offset给其他DRX→编排数据重构。对比Multi-Axl baseline（数据重构在CPU用户态用Intel MKL）vs DMX（offload到DRX），控制平面不变（OpenCL-style，CPU host program）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明DMX驱动和runtime软件栈开源情况。
  - 端到端数据流（以Brain Stimulation的FFT→Proximal Policy Optimization为例）：(1) Accelerator1（Xilinx Vitis DSP Library实现FFT）处理(256,1024,8)电磁信号输入，完成后发中断→(2) Accelerator1 driver捕获中断，找到DRX1上对应Accelerator2的RX2队列，共享queue offset给Accelerator1→(3) Accelerator1通过point-to-point DMA将FFT输出写入DRX1的RX2队列→(4) DRX1处理单元从RX2读取数据，128 RE lane并行执行数据重构（Pow/Div/Mul/Cast），结果写入TX2队列→(5) DRX1发中断通知CPU数据重构完成→(6) DRX1配置point-to-point DMA（经内部PCIe multiplexer bypass DRX2）将数据传到Accelerator2→(7) Accelerator2（DNN Accelerator [13]）执行Proximal Policy Optimization→完成。性能评估：emulation infrastructure整合三阶段cycle级latency（kernel执行+数据重构+数据搬移），计算端到端延迟和pipeline吞吐（吞吐=bottleneck stage决定）。能耗：CPU RAPL + FPGA post-synthesis功耗×执行时间 + PCIe switch功耗 + PCIe传输能耗。

## 14-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：DMX运行时系统将数据重构kernel从CPU offload到DRX执行，包含：(i) DRX driver管理command queue、RX/TX数据队列对（每个加速器一对，200MB总空间支持最多40个加速器）、head/tail指针跟踪，(ii) point-to-point DMA通过dma-buf API在加速器和DRX间直接传输数据（绕过CPU），(iii) 中断处理采用Linux NAPI风格（低中断率时interrupt coalescing，高速率切换polling），(iv) 支持one-to-many broadcast和many-to-one all-reduce等集合通信模式（通过back-to-back DMA + DRX reduction操作），(v) 数据重构kernel在DRX上执行——从RX队列读取加速器输出→执行数据重构（reshape/cast/transpose/concat/flatten等）→写入TX队列→DMA到目标加速器。
  - 实验比较：(i) Multi-Axl baseline（数据重构kernel在CPU上执行，加速器间通过CPU拷贝数据），(ii) DMX（数据重构kernel在DRX上执行，point-to-point DMA绕开CPU），(iii) broadcast和all-reduce在4/8/16/32加速器下的性能对比，(iv) 不同DRX放置方案（Integrated/Standalone/Bump-in-the-Wire/PCIe-Integrated）的性能对比。

- 后端平台是什么，配置是什么。
  - 后端平台：AWS F1实例（Xilinx UltraScale+ VU9P FPGA × 多个，PCIe x16连接Intel Xeon Platinum 8260L CPU@2.4GHz，64GB内存，hyperthreading disabled）。
  - 配置：加速器数量1-30（对应1-15个并发应用×2 kernel），DRX配置128 RE lanes@250MHz(FPGA)/1GHz(ASIC)，DRX每加速器含64KB scratchpad + 8GB DDR4，PCIe Gen3 baseline（扩展到Gen4/Gen5灵敏度研究）。

- 评估性能的软件/脚本是什么。修改了什么。
  - DMX驱动栈：基于GEM（Graphics Execution Manager）管理命令执行和内存操作，通过ioctl syscall执行命令读写映射。使用dma-buf API建立加速器-DRX间的point-to-point DMA。DRX driver在PCIe枚举时发现所有连接的加速器，为每个加速器分配数据队列。
  - 修改内容：新增DRX driver（Linux kernel module）→ 初始化command queue和RX/TX数据队列 → 共享队列start/end指针给其他DRX → 编排数据重构操作。Multi-Axl baseline中数据重构在CPU用户态执行（Intel Math Kernel Library），DMX中offload到DRX硬件。控制平面仍运行在CPU上（OpenCL-style编程模型）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明DMX驱动和runtime软件栈开源情况。
  - 运行时流程（以Video Surveillance的H.264解码→Object Detection为例）：(1) Accelerator1 (Xilinx Video Codec Unit) 完成H.264解码（输出(960,540,3)视频帧）→ 通过中断通知CPU → (2) Accelerator1 driver捕获中断，在DRX1上找到对应Accelerator2 (DNN Accelerator)的TX数据队列 → 共享RX2数据队列偏移给Accelerator1 → (3) Accelerator1通过point-to-point DMA将视频帧写入DRX1的RX2队列 → (4) DRX1处理单元从RX2队列读取数据，执行数据重构（Mul/MaxPool/Reshape/Cast），将结果写入TX2队列 → (5) DRX1通过中断通知CPU数据重构完成 → (6) DRX1配置point-to-point DMA（经内部PCIe multiplexer绕过DRX2）将重构后数据传输到Accelerator2 → (7) Accelerator2执行Object Detection kernel → 完成。性能指标：端到端延迟（各阶段cycle级latency求和）、吞吐（pipeline瓶颈stage决定）、能耗（CPU RAPL + FPGA post-synthesis功耗 × 执行时间 + PCIe switch功耗 + PCIe传输能耗）。

## 17-MAGIS- Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MAGIS中的图调度（Graph Scheduling）组件——包含Re-materialization（evict中间tensor后在backward时重新计算）、Swapping（将tensor通过异步CUDA Stream swap到CPU内存后再swap-in）、Re-ordering（调整operator拓扑排序顺序）三种运行时调度技术。MAGIS将这些调度技术分解为图变换规则（Re-materialization Rule/Swapping Rule及对应的De-规则，§5.2）和重排序（re-ordering）。在NVIDIA GPU上通过PyTorch CUDA Stream API实现异步Store和Load操作，数据在GPU显存（24GB RTX 3090）和CPU内存间通过PCIe交换。增量调度算法（Algorithm 2）基于dominator tree的narrow waist (NW)值确定需重调度的子图范围，GraphPartition用nw(v)≤1节点分割子图为独立部分，DpSchedule（基于Serenity [3]的DP算法）对各部分独立寻优重排序，相对full scheduling实现4-30×加速。
  - 实验比较：(1) PyTorch baseline（简单拓扑排序+立即释放未来不用tensor），(2) POFO（DP-based re-mat + swapping组合优化），(3) DTR（heuristic动态re-mat，MegEngine eager模式），(4) XLA（贪心re-mat，HLO-level），(5) TVM/Relay + Torch-Inductor（基本内存回收）。评估7个workload：ResNet-50 (b64), BERT-base (b32), ViT-base (b64), U-Net (b32), U-Net++ (b16), GPT-Neo-1.3B (b32), BTLM-3B (b32)。

- 后端平台是什么，配置是什么。
  - NVIDIA GeForce RTX 3090 GPU (24GB显存)，Intel Xeon Silver 4210R CPU (20核)。GPU与CPU间通过PCIe Gen3进行数据交换（swapping）。CUDA 11.6 + cuDNN 8.4.0. PyTorch 2.1.0。

- 评估性能的软件/脚本是什么。修改了什么。
  - MAGIS自研simulator：内置operator性能cache（存储每个operator在RTX 3090上的实际执行延迟），通过模拟执行全图调度来估算整体延迟和峰值内存——跟踪每个operator在调度中的执行顺序、输出tensor生命周期（start=operator完成时间，free=max(child完成时间)），计算每个时间步的活跃内存量。异步swapping时，Store/Load operator的re-ordering策略是Store尽早放置、Load尽可能晚放置以使数据传输延迟恰好被隐藏。
  - 不修改现有框架。MAGIS为独立框架，图调度通过M-Rules（Scheduling-based Rules）和增量调度实现，代码生成后端生成调用PyTorch CUDA Stream API的Python代码实现异步GPU↔CPU数据交换。
  - 关键设计：(1) 将re-mat/swapping分解为图变换+重排序——memory-performance trade-off完全移入图变换阶段（M-Rules），调度阶段仅做不影响总延迟的重排序；(2) Schedule-based规则仅在包含memory hot-spot的子图上匹配，减小搜索空间；(3) 增量调度：GetRescheduleInterval基于NW值在先前调度上找到需重调度的interval，通过ExtendBound向两侧扩展直到NW值满足阈值条件（nw<4或iteration>20），仅对受影响的子图部分调度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：是。代码位于 https://github.com/pku-liang/MAGIS。
  - 评估原理：MAGIS simulator使用profile阶段在RTX 3090上采集的每个operator实际CUDA执行延迟（存储于operator cache），通过模拟执行全图调度来估算整体延迟和峰值内存。模拟器不实际执行operator，而是基于调度中的operator顺序和tensor生命周期信息计算内存时间线：对调度序列中的位置i，operator vi完成后其输出tensor开始占用内存（Si=i），在max(child_j完成时间)时刻释放（Fi=max(child_j)）。时间步i的活跃tensor集合Ai={v_j | Sj ≤ i ≤ Fj}，峰值内存Mpeak=max_i Σ|u| for u in Ai。
  - 端到端流程（以U-Net b=32训练，memory limit=60% PyTorch峰值为例）：
    1. **Profile阶段**：在RTX 3090上执行U-Net各operator（Conv2d, ReLU, MaxPool, UpSample, Concat等），记录每个operator的CUDA执行时间到cache。同时记录PCIe传输带宽用于估算Store/Load延迟。
    2. **初始调度**：对U-Net原始图做简单拓扑排序作为initial schedule，simulator评估baseline peak memory = 1056 units（33个tensor各size=32同时活跃）和总latency。
    3. **图变换阶段**：M-Optimizer对图应用M-Rules：(a) F-Trans Enabling——沿batch-dim将encoder/decoder的conv block分裂为n=2的split parts，每个part处理batch=16→峰值内存从1056降至约560；(b) Swapping Rule——在skip-connection tensor A的consumer B处插入Store(A, CPU)和Load(CPU, B)，使tensor A在forward后swap到CPU，backward需要时再load回GPU；(c) Re-mat Rule——对部分decoder activation，删除其forward保存结果，改为backward时重新compute。
    4. **增量调度**：对变换影响的子图区域执行增量调度：GetRescheduleInterval找到变换在初始调度中对应的operator区间[beg, end]→ExtendBound向前后扩展边界至NW值<4→获取new graph中对应子图Snew→GraphPartition按nw(v)≤1节点将Snew分割为独立部分→DpSchedule对各部分DP-based最优重排序（Store操作被尽早放置到forward后，Load操作延迟到consumer需要前刚好隐藏PCIe延迟的时刻）。
    5. **评估**：simulator按新调度顺序模拟执行→跟踪各时间步active memory→累加operator延迟→输出peak_memory和total_latency。若满足memory constraint且latency overhead < 5%，标记为有效M-State。
## 18-SmartMem- Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SmartMem在kernel层面的优化包含三部分：(1) Reduction Dimension-based Layout Selection生成优化后的tensor layout，通过将consumer的reduction dimension强制映射为producer输出tensor的连续存储维度，使后续kernel的数据访问沿reduction dimension顺序进行（stride=1），提升数据局部性和SIMD效率；(2) 2.5D Texture Memory Mapping——将优化后的tensor layout映射到移动GPU的2.5D纹理内存。对含多个reduction dimension的tensor：沿一个reduction dimension按4元素partition（匹配2.5D内存的vector宽度），每个partition内沿另一个reduction dimension连续存储k×|D|元素块，使consumer kernel可沿任意一个reduction dimension进行连续SIMD load和reduction操作；(3) 优化后的内存访问模式——通过Operator Elimination的Index Comprehension Strength Reduction简化kernel内部的取模和除法索引计算（如i%8%4简化为i%4），减少GPU上昂贵的取模/除法指令开销；(4) Genetic Algorithm auto-tuning——搜索最优GPU kernel执行配置（block dimension、unrolling factor、tiling shape）。
  - 实验比较：(a) 与MNN/NCNN/TFLite/TVM/DNNFusion在18个模型上的端到端延迟对比（Table 8）；(b) Memory access count和cache miss count对比（Figure 7）；(c) 优化消融分析——DNNF → +LTE → +Layout Selecting → +Other Opt逐级叠加（Figure 8）；(d) Memory和cache的消融分析——LTE对减少memory access count效果更显著，Layout Selecting对减少cache miss count效果更显著（Figure 9）；(e) Roofline分析——Swin/ViT/ResNext/SD-VAEDecoder对比理论峰值（Figure 12）；(f) 不同batch size下的可扩展性（Figure 10，batch 1-16）；(g) 可移植性评估——在Mali-G57和Adreno 540上的性能（Figure 11）；(h) 桌面GPU评估——在NVIDIA V100上的TorchInductor对比（Table 9）。

- 后端平台是什么，配置是什么。
  - 主平台：Qualcomm Snapdragon 8 Gen 2（Adreno 740 GPU，16GB unified memory）。GPU global memory bandwidth: 55 GB/s，2.5D texture memory bandwidth: 511 GB/s，peak computation: 2.0 TMACs/s。
  - 可移植性平台：(a) Qualcomm Snapdragon 835（Adreno 540 GPU，6GB unified memory），(b) MediaTek Dimensity 700（Mali-G57 GPU，4GB unified memory）。
  - 桌面GPU：NVIDIA Tesla V100（global memory bandwidth 900 GB/s），通过PyTorch 2.1.0 + TorchInductor实现。

- 评估性能的软件/脚本是什么。修改了什么。
  - SmartMem在DNNFusion基础上新增kernel层面的优化：(1) 新增Index Comprehension模块——将计算图中消除的Reshape/Transpose等layout op替换为简化的索引计算（identity/split/merge三类映射关系 + Strength Reduction），在fused kernel内部直接计算数据访问索引而非执行独立的reshape/transpose kernel；(2) 新增2.5D Texture Memory Layout Mapping——将tensor按reduction dimension映射到纹理内存的宽度和高度维度，配置纹理采样器参数；(3) 新增Genetic Algorithm-based auto-tuning——搜索GPU kernel的block dimensions、unrolling factors和tiling shapes。
  - 评估使用Qualcomm Adreno GPU hardware counter收集memory access count和cache miss count。优化前后对比operator数量（fusion rate），每个model执行50次取平均值，方差可忽略。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明SmartMem开源情况。
  - Kernel执行全过程（以Swin Transformer中Conv+Reshape+Transpose+MatMul pattern为例）：
    1. **输入**：原始计算图含Conv(ILD-Variable)→Reshape(ILD-Fixed)→Transpose(ILD-Fixed)→MatMul(ILD-Variable)。输入tensor形状为[2, 256, 4]（即M,N,K）。
    2. **Operator Elimination（LTE）**：查Table 5——Reshape是ILD-Fixed，Transpose是ILD-Fixed→均被消除。消除通过Index Comprehension实现——分析[2,256,4]→Reshape→[16,8,4,4]→Transpose→[16,4,8,4]的索引依赖链：i'=i*8+j/(4*8), j'=j%4, k'=j%(4*8)//4, l'=k。Strength Reduction简化：消除冗余取模运算后将fused kernel内部的索引计算简化为直接映射。
    3. **Layout Selection**：MatMul是consumer，其reduction dimension是k→强制Conv(producer)生成沿k维连续存储的输出layout。对于2.5D memory：将Conv输出tensor按reduction dimension映射——取一个reduction dim按4元素partition（vector size），沿另一个dim连续存储，使MatMul kernel可用stride=1的方式load数据。
    4. **Kernel执行**：FusedConv-MatMul kernel在Adreno 740 GPU上执行——workgroup从2.5D纹理内存以vector load（4元素）方式读取Conv输出（已被layout优化为沿reduction dim连续），沿reduction dim做SIMD reduction（如dot product累加），结果写回1D buffer。不再执行独立的Reshape kernel和Transpose kernel（原本各需一次global memory round-trip）。
    5. **性能输出**：Hardware counter记录memory access count和cache miss count→计算GMACS（GMACs per second）→与baseline对比延迟/吞吐。Swin上SmartMem 149 GMACS vs DNNFusion 34 GMACS（4.4× speedup），memory access减少约1.8×，cache miss减少约2.0×。

## 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Oaken的Quantization Engine和Dequantization Engine硬件模块，在加速器compute core内部以pipeline方式执行KV cache的在线量化/反量化kernel操作。这些模块作为kernel-level计算单元，与Vector Processing Unit (VPU)和Matrix Processing Unit (MPU)协同工作，在token生成过程中实时完成量化/反量化计算。
  - Quant Engine kernel pipeline：Threshold Comparator（分组判断，比较T_lo/T_hi）→ Scale Calculator（计算per-group量化scale = max(|v|)/ (2^bits-1)）→ Quantizer（INT4/INT5量化round操作）→ Splitter（分离dense/sparse路径）→ OR Gate + Shifter（融合编码为8-bit对齐格式）→ 通过DMA写入Device Memory。
  - Dequant Engine kernel pipeline：Decomposer（从8-bit encoded数据解析group flag和value index）→ 分离Dense/Sparse dequantizer路径→ Min&Max恢复单元→ Scale Calculator（重建scale = (max-min)/(2^bits-1)）→ Dequantizer输出FP16值→ 送入MPU计算attention。
  - 实验比较：baseline GPU（A100）上的vLLM（FP16，无量化kernel）、KIVI（GPU上2-bit量化，软件dequant kernel）、QServe（GPU上4-bit量化，软件dequant kernel）。Oaken-HBM和Oaken-LPDDR加速器 vs GPU baselines。

- 后端平台是什么，配置是什么。
  - Oaken自研加速器：270 FP16 TFLOPS, TSMC 28nm综合。Compute Core包含VPU（22.86%面积）、MPU（6.03%）、Quant Engine（1.86%）、Dequant Engine（6.35%）、DMA、MMU。HBM配置：80GB, 2.0 TB/s；LPDDR配置：256GB, 1.1 TB/s。
  - GPU baseline：NVIDIA A100 (312 TFLOPS, 80/160GB HBM, 2.0 TB/s)

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估代码：https://github.com/casys-kaist/oaken。GPU上运行accuracy evaluation（eval_perplexity.py, eval_workload.py），通过--quant-method oaken启用Oaken量化算法模拟。
  - 修改：Oaken accuracy evaluation在GPU上用PyTorch实现量化/反量化kernel的软件模拟——oaken_main.py调用quantizer/oaken/中的量化器对KV cache执行grouping+quantization+encoding+dequantization，替换原始FP16 attention计算中的K/V值完成精度评估。
  - 硬件性能评估：论文使用自研加速器模拟器评估throughput，硬件的Quant/Dequant Engine在TSMC 28nm综合评估面积和功耗。论文未详细公开硬件模拟器名称和实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：accuracy evaluation开源（https://github.com/casys-kaist/oaken），硬件加速器性能模拟器未开源。
  - Kernel执行全过程（以Generation阶段单token的attention计算为例）：
    1. 输入：new token的Q向量（FP16, [batch, num_heads, d_head]）、当前layer所有past tokens的量化KV cache pages（存储格式：[6-bit idx | 1-bit group flag | 1-bit sign or 5-bit val] per entry, 8 bit/entry total）。
    2. Dequant Kernel启动：MMU根据management table找到请求对应的KV cache pages物理地址→DMA从Device Memory读取encoded KV数据（8 bit/entry）→送入Dequant Engine。
    3. Decomposer：解析每8-bit entry的group flag field——flag=0→middle group（4-bit INT4 value），flag=1→outer group（5-bit INT5 value），flag=sparse→inner group（value=0，跳过计算）。
    4. Dequantization：对middle group——val_fp16 = val_int4 * scale_mid + T_lo；对outer group——val_fp16 = val_int5 * scale_out + shift；对inner group——val_fp16 = 0。
    5. Attention计算：Dequantized FP16 K/V值直接送入Matrix Processing Unit计算attention score = softmax(Q·K^T/√d)·V。
    6. Quant Kernel（写回）：同时new token的K/V经Quant Engine——Threshold Comparator按T_lo/T_hi分组→Scale Calc计算scale→Quantizer量化→Dense-Sparse Encoder编码→DMA写回KV cache pages。
    7. 性能输出：throughput = batch中完成的token数 / elapsed time。Oaken-HBM在Llama2-7B batch=256时实现1.79× over vLLM (FP16) 的throughput提升。

## 20-PIM-DL- Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PIM-DL的LUT-NN Inference Engine和Auto-Tuner。核心是将LUT-NN的LUT算子（table lookup + reduction）映射到DRAM-PIM的数千个PE上并行执行，并自动搜索最优映射参数。包含：
    1. **Sub-LUT Partition**（工作负载分区）：将CCS输出的index matrix沿N dim切分为N/N_{s-tile}个tile，LUTs沿F dim切分为F/F_{s-tile}个tile。PEs逻辑分组为#PE/(N_{s-tile}×F_{s-tile})个group——同group的PEs共享同一index tile（广播），跨group同位置的PEs共享同一LUT tile（广播）。每个PE计算(N_{s-tile}, F_{s-tile})大小的output tile。该分区策略避免了inter-PE通信（CT dim不切分，无需partial sum merging）并确保load balance（均匀tiling）。
    2. **Micro Kernel Execution**（微内核执行）：每个PE将本地的(N_{s-tile}, CB) index tile和(CB, CT, F_{s-tile}) LUT tile进一步切分为MTiles：(N_{m-tile}, CB_{m-tile}) index MTile 和 (N_{m-tile}, F_{m-tile}) output MTile。PE加载一个index MTile和对应的output MTile → traverse所有CB_{m-tile} blocks检索LUTs → reduce得到完整output MTile。
    3. **Three LUT Load Schemes**（三种LUT加载策略）：
       - Static Load：整块LUT MTile < on-chip buffer时，一次性加载全部LUT并在execution期间reuse（需buffer容纳CB_{s-tile}×CT×F_{s-tile}元素）
       - Coarse-grain Load：每次加载CT个candidates的CB_{load-tile}×CT×F_{load-tile}元素到buffer，按CB_{load-tile}分组复用
       - Fine-grain Load：按需加载F_{load-tile}个LUT值，适合多硬件线程并发（如UPMEM PE的多个threadlets可各自发送独立memory request）
  - 实验比较：在UPMEM PIM-DIMM上对BERT-large FFN1层的mapping space可视化（Figure 13），展示不同tiling factors和load schemes下的性能分布。auto-tuner给出的参数相比实测best仅≤6%性能退化，平均估计误差3.44%，max error 13.73%。end-to-end对比：PIM-DL (V=4/CT=16) vs GEMM-based inference on PIM-DIMM: 18.91× geomean speedup。latency breakdown显示LUT operator占总延迟的51.52%~60.41%。

- 后端平台是什么，配置是什么。
  - UPMEM PIM-DIMM（真实硬件）：8× DIMMs，1024 PEs total（每DIMM 2 ranks×64 PEs）。每PE为programmable RISC core@350MHz，64KB on-chip buffer，8-bit data path per PE（8 PEs组成64-bit rank data path）。PE支持多个硬件线程（threadlets），各自可发起独立memory request。
  - Samsung HBM-PIM（模拟）：4× Cubes，512 PEs，8GB HBM2 memory。使用Samsung PIMSimulator。
  - SK-Hynix AiM（模拟）：16× Chips，512 PEs，16GB GDDR6 memory。扩展PIMSimulator支持AiM功能。

- 评估性能的软件/脚本是什么。修改了什么。
  - UPMEM SDK (Version 2021.3.0)：基于clang 10.0.0的PIM kernel编译器，提供PIM binary部署和runtime library。PIM-DL Engine的PIM operator包含：(1) Host-side PIM kernel——hypervisor触发PIM执行；(2) PIM binary——LUT lookup + reduction kernel，实现在UPMEM ISA指令集中。
  - Samsung PIMSimulator（https://github.com/SAITPublic/PIMSimulator）：官方开源的HBM-PIM模拟器。PIM-DL扩展其功能以支持AiM产品的behavior模拟。
  - GGML tensor library：Host侧tensor操作，使用AVX intrinsics加速x86 CPU上的CCS算子（GEMM）和element-wise算子。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/leesou/PIM-DL-ASPLOS (MIT License)。PIMSimulator开源（https://github.com/SAITPublic/PIMSimulator）。
  - PIM Auto-Tuner评估原理（Algorithm 1，以BERT-large FFN1层为例，workload shape (N=32768, CB=256, CT=16, F=4096)）：
    1. **输入**：workload size (N, CB, CT, F) 和 target platform的硬件参数（BW_host_index, BW_host_lut, BW_host_output, BW_pim_index, BW_pim_lut, BW_pim_output, #PE）。
    2. **遍历sub-LUT tiling factors**：对每个合法的(N_{s-tile}, F_{s-tile})对（约束：#PE = N/N_{s-tile} × F/F_{s-tile}），计算sub-LUT partition overhead：
       - t_index_sub = S_IndexTile × #PE / BW_host_index（index广播的host→PIM传输）
       - t_lut_sub = S_LUTTile × #PE / BW_host_lut（LUT广播的host→PIM传输）
       - t_output_sub = S_OutputTile × #PE / BW_host_output（PIM→host结果fetch）
       - t_sub-lut = t_index_sub + t_lut_sub + t_output_sub
    3. **搜索micro kernel最优参数**：对每个(N_{s-tile}, F_{s-tile})，在micro kernel mapping space中搜索：(a) micro kernel tiling factors (N_{m-tile}, F_{m-tile}, CB_{m-tile})；(b) tile traversal order；(c) LUT load scheme（static/coarse-grain/fine-grain）。用analytical model估计t_micro-kernel：
       - t_transfer = LC_index × MTileSize_index / BW_pim_index + LC_lut × MTileSize_lut / BW_pim_lut + LC_output × MTileSize_output / BW_pim_output + SC_output × MTileSize_output / BW_pim_output
       - t_reduce = RCount × t_single-reduce（single-reduce latency为profiled值，与底层PE架构相关）
       - t_micro-kernel = t_transfer + t_reduce
    4. **输出最优映射**：min(t_sub-lut + t_micro-kernel*)，生成MappingParams = {N_{s-tile}, F_{s-tile}, Kernel*}。每个模型仅需tune一次（~1s/model on dual-socket Xeon 4210 CPU），参数fed入PIM-DL Inference Engine执行。LUT operator: PIM PE加载index MTile → fetch对应LUT entries（根据load scheme）→ 累加到output MTile → store output结果。最终output tiles从PIM fetch回host，与CCS结果combine完成整个linear layer的LUT-NN推理。

## 23-Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Orojenesis基于Snowcat代理架构穷举mapspace搜索，计算张量算法在不同buffer容量下的数据移动下界（backing store access bound）。该bound为mapping-independent——提供任何mapping（tiling, parallelization, schedule, fusion）在给定buffer容量下均无法超越的数据移动/操作强度极限。单Einsum分析覆盖：(1) GEMM各种shape——M=K=N从2k到8k，矩形shape（tall-and-skinny），揭示maximal effectual buffer size≈smallest operand size + smallest rank + 1；(2) Convolution各种配置——filter size（1×1到5×5）、stride 2、dilation 2，C=K=64，P=Q=16；(3) Batched MM——不同的head数H（1-128）、reduction dim K（4k down to 32），固定计算量128 GOPs；(4) Grouped BMM——groups数G（1-32），H=32，M=N=4k，K=128。多Einsum fusion分析覆盖GPT-3-6.7b LLM的完整building block（sequence length 2048 × batch 16 = effective l=32768，d=4096，h=32，f=128，c=16384），包含MHA（Q_proj, K_proj, V_proj, bmm_QK, bmm_QKV, Final_proj）和FFN（mm_0, mm_1）的全部Einsum。
  - 实验比较：(a) vs Algorithmic-minimum accesses（所有operand sizes之和），显示Orojenesis bound tighter——4k GEMM上algorithmic-minimum与A100 DRAM实测traffic相差6.5×，与L2-to-L1 traffic相差32.3×；(b) 不同fusion strategy对比——No Fusion vs Untiled Fusion vs Tiled Fusion vs Segmented Tiled Fusion；(c) MHA fusion策略对比——FlashAttention-like（TiledN→TiledK）vs FLAT-like（TiledK→TiledN）；(d) 不同chain长度fusion效果——6-Einsum chain vs segmented chains；(e) 完整LLM block analysis——unfused vs fused bounds，以及50MB last-level cache下2.5× traffic reduction，320MB buffer下最大5.6× reduction（6GB absolute reduction）。

- 后端平台是什么，配置是什么。
  - GPU验证平台（用于Orojenesis bounds validation而非analysis target）：
    - NVIDIA A2 GPU：2MB last-level cache
    - NVIDIA A30 GPU：24MB last-level cache
    - NVIDIA A100 GPU：40MB last-level cache
    - NVIDIA H100 GPU：50MB last-level cache
    - 对比SIMT core vs Tensor Core的DRAM访问量
  - Simba加速器模型（用于bounds验证）：5种不同Global Buffer配置（128B, 1KB, 8KB, 64KB, 512KB），通过Timeloop的Simba analytical model评估
  - 性能模型参考平台：GF100-like chip（40nm tech, 529mm² die area, 700MHz, DRAM bandwidth 149GB/s），Accelergy area estimation（332.25 μm² per MAC, 2.59 μm² per byte SRAM）

- 评估性能的软件/脚本是什么。修改了什么。
  - Orojenesis自身：Jupyter notebooks（orojenesis_single.ipynb, orojenesis_multi.ipynb）调用Timeloop mapper在Snowcat架构上穷举搜索生成bounds，生成ski-slope/OI mesa/performance mesa图。
  - GPU验证：NVIDIA CUTLASS（https://github.com/NVIDIA/cutlass）——在A2/A30/A100/H100上运行优化后的4k×4k×4k GEMM schedule，测量DRAM访问量。通过profiling工具（nvprof/nsys）收集各memory level的read/write traffic。
  - Simba验证：Timeloop的Simba analytical model（基于[66]）——配置5种Global Buffer size，对每个配置搜索最优mapping并记录DRAM access count。
  - 修改：Orojenesis自身基于Timeloop修改（见编译框架部分）；GPU验证和Simba验证使用现成工具profiling/analytical modeling，未修改。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://timeloop.csail.mit.edu/orojenesis（BSD-3-Clause），Zenodo DOI: 10.5281/zenodo.10850531。
  - Orojenesis bounds评估原理：对给定Einsum（如4k×4k×4k GEMM），Snowcat架构上的mapspace由tile loop bounds（如M0, M1, K0, K1, N0, N1）和loop order（所有合法permutations）组成。每个mapping的buffer size requirement = tile size(A) + tile size(W) + tile size(B)（即M0×K0 + K0×N0 + M0×N0），backing store accesses = M1×K1×K0×N0（A）+ K1×N1×K0×N0（W）+ M1×N1×M0×N0（B）。穷举所有mapping后，按buffer size分组取min(accesses)，得Pareto curve。
  - GPU验证评估原理：在A100/H100等GPU上运行CUTLASS优化的tiled GEMM kernel，通过硬件性能计数器（nvprof metrics: dram_read_bytes, dram_write_bytes, l2_read_bytes, l2_write_bytes）采集各memory level的实际读取/写入字节数。对比Orojenesis bound与实测值：实测值应≥bound（验证bounds有效性），同时观察A100/H100的优化schedule是否接近bound。
  - 完整流程（4k GEMM验证）：编写CUTLASS GEMM kernel（FP16, tile size=256×128×32等优化参数）→编译→在GPU上运行→nvprof收集dram_read_bytes/sectors + dram_write_bytes/sectors→计算total DRAM accesses→与Orojenesis ski-slope上对应LLC size点的bound比较→确认实测accesses ≥ bound。

## 25-Be CIM or Be Memory- A Dual-mode-aware DNN Compiler for CIM Accelerators.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：CMSwitch在CIM accelerator上对DNN算子进行dual-mode-aware的kernel调度与资源分配。核心调度由MIP (Mixed-Integer Programming) 求解器（Gurobi）驱动，在每个network segment内联合优化：(a) 每个CIM array (x,y)对每个算子Oi的mode分配（compute mode λ_c / memory-input λ_min / memory-output λ_mout），(b) 算子间pipeline调度——segment内所有算子同时映射到chip上并行执行。调度遵循三个关键机制：
    1. **Operator dependency reuse**：若Oi的输出是Oj的输入，Oi的output memory array可直接复用为Oj的input memory buffer，避免数据搬移。
    2. **In-place computation**：如attention中K值计算后保留在memory mode array，切换到compute mode后QK^T直接在原位计算，消除数据搬移overhead。
    3. **Latency-driven mode balancing**：每个算子Oi的延迟LOi ∝ OPOi / min(ComOi·OPcim, (MemOi·Dcim + Dmain)·AIOi)，即高arithmetic intensity算子分配更多compute array，低arithmetic intensity算子分配更多memory array。
  - 实验比较：在CIM-MLC functional simulator + 修改版NeuroSim/MNSim组成的simulator上执行生成的meta-operator流，测量端到端延迟。与PUMA、OCC、CIM-MLC三种固定CIM array模式（全部作compute）的编译调度结果比较。MIP求解每次segment resource allocation，segment内算子以pipeline方式并行执行。

- 后端平台是什么，配置是什么。
  - 后端平台：DynaPlasia双模CIM芯片（28nm eDRAM），96个switchable CIM array（320×320），每个array在compute mode可执行bit-serial MAC（MVM/MMM），memory mode可用作scratchpad buffer。Compute-mode operation rate OPcim ∝ array_size（array提供1次MAC/cycle的并行度），memory-mode data rate Dcim由架构设计和用户定义topology决定，外部main memory bandwidth Dmain ∝ extern_bw + internal_bw。模式切换通过GIA/GIAb信号修改（1 cycle延迟），数据写回/加载bandwidth由Dmain决定。
  - 评估simulator：基于CIM-MLC functional simulator（功能验证）+ 修改版NeuroSim [7]和MNSim [50]（延迟评估），增加了对dual-mode switch的模拟支持。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于CIM-MLC compiler附带的功能模拟器，以及修改版NeuroSim/MNSim。修改包括：(a) 增加CIM array mode切换的cycle级模拟（切换信号配置延迟、模式切换后的memory/compute行为差异）；(b) 按DEHA参数配置DynaPlasia硬件规格。
  - 延迟评估：functional simulator执行编译生成的meta-operator流，NeuroSim/MNSim计算每步operation的cycle数（包括MAC并行度、memory bandwidth、mode switch overhead等），累加得到end-to-end latency。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明CMSwitch评估脚本是否开源。CIM-MLC为已知开源CIM编译框架，NeuroSim（https://github.com/neurosim/neurosim）和MNSim为开源CIM模拟器。
  - kernel调度到性能输出的全过程（以OPT-6.7B单层attention在DynaPlasia上的执行为例）：
    1. **输入**：DP分割后的segment Si,j（如包含QKV projection各矩阵乘 + QK^T attention + SV output projection + FFN），MIP求解后输出per-operator的λ分配方案。
    2. **QKV Projection调度**：MIP为W_Q、W_K、W_V的MMM各分配若干个compute array（如W_Q: 24 array compute + 8 array memory-input + 0 array memory-output），compute array预先load对应权重。memory-input array从buffer/外部加载Q/K/V输入activation。K值计算后，部分memory-output array保留K数据（为后续QK^T准备）。
    3. **QK^T Attention调度**：MIP检查operator dependency——Q projection的output memory array可直接作QK^T的input memory buffer。K保留在memory array中→CM.switch(TOM)将这些array从memory切为compute→Q与K^T直接在原位执行MVM/MMM（无需额外数据搬移）。softmax结果S写入memory array。
    4. **SV Output调度**：S保留在memory array→CM.switch(TOM)切为compute mode→与V做MVM。MIP为后层FFN分配不同的compute/memory array比例（如FFN FC1: 40% compute + 60% memory用于大activation数据加载）。
    5. **Pipeline并行**：segment内多算子通过pipeline重叠执行。如QKV projection中W_Q compute的同时，W_K所需的memory-input从外部预取。MIP目标函数min max(LOi)保证pipeline瓶颈最小化。
    6. **延迟计算**：每个算子的LOi根据allocated compute/memory array数量计算——compute capacity = ComOi × OPcim（每个compute array提供的MAC/cycle），memory bandwidth = MemOi × Dcim + Dmain（memory array + 原有buffer/外部带宽），有效computation rate = min(compute capacity, memory bandwidth × AI_Oi)，LOi = total MAC operations / effective rate。Segment延迟 = max(各算子LOi) + pipeline overlap后。总延迟 = 所有segment延迟之和 + inter-segment switch overhead（3%-5%）。
    7. **性能输出**：总end-to-end latency与CIM-MLC baseline对比，CMSwitch的MIP调度使memory mode array占比根据operator AI动态调整（QKV计算33%~67% memory mode，attention计算更多compute mode），平均加速1.31×。

## 28-MIMDRAM: An End-to-End Processing-Using-DRAM System for High-Throughput, Energy-Efficient and Programmer-Transparent Multiple-Instruction Multiple-Data Computing

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MIMDRAM继承SIMDRAM的μProgram框架（bbop指令→MAJ/NOT优化→AAP/AP命令序列），新增：(1) mat scheduler (online first fit) 在bbop buffer中扫描待执行bbop→查mat scoreboard bitmap确定目标mat是否空闲→分配空闲mat并启动μProgram engine；(2) mat scoreboard (128-bit bitmap) 追踪每mat占用状态；(3) 8个μProgram processing engines并发执行不同bbop→各自向DRAM发出AAP/AP/GB-MOV/LC-MOV命令；(4) PUD vector reduction——两步：GB-MOV跨mat搬移部分和→intra-mat adder tree用LC-MOV实现最终约简至4 elements；(5) MIMD并行模式：同一subarray内不同mat range并发执行不同μProgram（fine-grained PUD在mat1执行add，mat2执行mul等）。
  - 实验比较：SIMD利用率（实际使用SIMD lanes/总SIMD lanes）；性能/能效（CPU-normalized perf/Watt）；吞吐量（weighted speedup）；周转时间（harmonic speedup）；公平性（maximum slowdown）。与SIMDRAM（固定全subarray SIMD）对比，MIMDRAM的mat粒度假定下SIMD利用率平均提升15.6×。

- 后端平台是什么，配置是什么。
  - 模拟PUD平台：DDR4-2400，1ch 8chips 4ranks，16 banks/rank，16 mats/chip=128 mats total，1K rows/mat=1024，512 columns/mat。每个mat = 512 columns × 1K rows = 512K cells。总SIMD宽度 = 128 mats × 512 columns/mat = 65536 bit（baseline SIMDRAM全subarray模式）。VMIMDRAM支持可变宽度 = 1~128 mats × 512 columns。
  - 控制单元：8-entry mat queue，2KB bbop buffer（容纳1024条bbop），8个μProgram processing engines

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估基础设施：gem5系统仿真集成MIMDRAM模型；12个C/C++应用从117个候选应用中筛选（memory-bound + loop可auto-vectorize）→ x264(SPEC 2017), hw/km/bp(Rodinia), pca(Phoenix), 2mm/3mm/cov/dg/fdtd/gmm/gs(Polybench)。495个multi-programmed mixes（8应用/mix）按最大VF分三类：low(<16K)、medium(16K-64K)、high(>64K)。
  - 仿真修改：在gem5中建模AAP/AP/GB-MOV/LC-MOV的cycle-accurate延迟，建模mat scheduler的online first fit调度决策和mat scoreboard bitmap查询开销，建模transposition unit的H→V和V→H数据layout转换延迟。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/CMU-SAFARI/MIMDRAM
  - 例子：以kmeans(km)应用执行为例
    1. **输入**：kmeans源码→MIMDRAM编译器三Pass→生成含bbop指令的二进制。km有4个vectorizable loops，max VF=16384→每个操作占16384/512=32 mats（而非全部128 mats）。
    2. **执行原理**：CPU dispatch bbop_add(targeting mats [0,31])到MIMDRAM control unit→mat scheduler查scoreboard→mats[0,31]空闲→分配μProgram engine 0→engine 0将bbop_add翻译为μProgram（8n+2个AAP/AP for n-bit addition）→发出ACT-enqueue/PRE-enqueue/ACT-dequeue含mat range→DRAM chip通过mat selector仅激活mats[0,31]的local wordline→在32 mats×512 cols=16384 SIMD lanes上执行bit-serial addition。同时，另一个独立bbop_mul(targeting mats[32,63])被不同的μProgram engine并发执行。
    3. **PUD操作类型与μProgram**：16种bbop (abs, add, bitcount, div, max, min, mult, ReLU, sub, and/or/xor-reduction, equal, greater, greater_equal, if_else)。每个bbop的μProgram是预先计算好的AAP/AP序列。如1-bit full adder需要5 AAPs+3 APs/iteration，n-bit addition共(8n+2)个AAPs/APs。vector reduction先GB-MOV搬移数据跨mat，再LC-MOV做intra-mat adder tree。
    4. **性能输出**：gem5记录每个bbop的完成cycle→计算execution time→normalize到baseline CPU执行时间→得normalized performance。CACTI根据激活的mats数量、row activation次数和命令类型计算动态能耗→perf/Watt。mat scoreboard记录整个执行期间各mat的busy/idle比例→SIMD utilization。
  - 作用：在PUD硬件模型中评估bit-serial SIMD操作在可变mat粒度下的运行时性能，量化MIMD并行和vector reduction对多个真实应用的加速效果。

## 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ASADI在ReRAM arrays上的四类in-situ计算kernel：(1) In-situ S×V (SpMM)——基于DIA格式的稀疏-稠密矩阵乘法kernel；(2) In-situ Q×K^T (SDDMM)——基于DIA格式mask矩阵的采样稠密-稠密矩阵乘法kernel；(3) In-situ Linear Layer——基于analog in-situ computing的VMM（vector-matrix multiplication）kernel；(4) In-situ Softmax——基于ReRAM的max和exponential bit-wise操作kernel。
  - 实验比较：(a) ASADI vs PIM baseline (Samsung FIMDRAM + Ramulator-PIM)在全部数据集上的speedup (1.9×-63.7×) 和energy saving (1.5×-5.2×)；(b) ASADI vs DIA-PIM (DIA计算在PIM上) 和 CSR-ASADI (CSR计算在ReRAM上)的消融实验；(c) ASADI vs GPU/SPRINT/CPSAA；(d) Latency breakdown (OCT/Linear/QK/Softmax/SV/CTRL各部分占比)；(e) Energy breakdown (OCT/Linear/QK/Softmax/SV/CTRL)；(f) 对角线局部性影响(60%-10%)和稀疏度影响(1.5τ-4τ)的性能评估。

- 后端平台是什么，配置是什么。
  - ASADI架构配置：12 En-PE + 12 De-PE，每个En-PE含12 Tiles (12 attention heads)，每个Tile含2 analog modules + 1 digital module + 1 microcontroller。
  - Analog module: 32×3个64×64 ReRAM arrays (1-bit per cell, read-only, 存W_Q/W_K/W_V)，96 arrays total，16个6-bit ADC (6 arrays共享1 ADC)，49KB，IR 64B。
  - Digital module: 64×(8192/1024)个1024×1024 ReRAM arrays (1-bit per cell, write-enable, 存Q/K/V/S)，512 arrays total，1024 DRV/array，512 S&A units，67.2MB。
  - ReRAM: 1GHz 1T1M，SET/RESET 1.62/3.63V，column-parallel read/write，1000GB/s OCI (inner-Encoder)，PCIe-6.0 128GB/s (cross-Encoder)。
  - ASADI总面积 279.8mm², 总功耗 538.9K mW，总容量 9.7GB。
  - Baseline PIM: Samsung FIMDRAM, 10GB HBM2, 500MHz logic/bank, 使用Ramulator-PIM模拟。
  - GPU: NVIDIA RTX A6000, 46GB, 300W TDP, CUDA v11.6, PyTorch v2.0.0。

- 评估性能的软件/脚本是什么。修改了什么。
  - 修改ZSim [30]模拟ReRAM array的行为（memory-level simulation）。
  - 自研in-house cycle-accurate模拟器获取ASADI的latency和energy，遵循[42]的数学证明。
  - Ramulator-PIM [14]用于baseline PIM平台的latency和energy评估。
  - Pre-processing代码修改自Sanger [22]的GitHub项目（PyTorch）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明ASADI自研模拟器是否开源。Pre-processing代码基于Sanger开源项目。
  - 评估原理（in-house cycle-accurate模拟器）：
    - **In-situ S×V kernel评估**：输入DIA format S (n×ω) + dense V (n×d) → 模拟器追踪每个ReRAM array的vector-vector multiplication cycle（parallel within each row）→ transfer DIA vectors between arrays → decompression (Rd/Ro list lookup + column shift) → vector-vector accumulation → 输出Z矩阵+cycle count。
    - **In-situ Q×K^T kernel评估**：输入DIA format mask M (n×ω) + dense Q/K (n×d) → 按DI迭代：shift Q up/down (O(1) cycle per DI) → memory copy per (Rd,Ro) → parallel vector-vector multi for all arrays → 恢复Q → 下一DI迭代 → gather SlicesS_i → vector-vector add → 输出DIA format S矩阵+cycle count。
    - **Latency计算**：OL (总体延迟) = LOI (单次迭代延迟) × NI (迭代次数)。对于ASADI，LOI为常数（所有ReRAM rows并行计算），NI与ω成正比（长序列ω=MSL/8）→总体O(n)。Baseline PIM的LOI随序列长度增加（因cross-bank transfer overhead），NI也随序列长度增加→总体O(n²)。
    - **Energy计算**：各模块（DRV/S&A/ADC/ReRAM read-write/OCI/PCIe）的能量=功率×执行时间。Digital module占总能耗>98%（因大规模并行ReRAM rows的DRV功耗）。
    - **输入→性能输出**：pre-processed模型（weight matrices + 压缩DIA format mask/S matrices）→配置模拟器（sequence length, ω, model dimensions）→对每个Encoder/Decoder执行三个阶段模拟（phase 1 analog linear layer → phase 2 digital Q×K^T + Softmax + S×V → phase 3 analog feed-forward）→输出：speedup vs baseline, energy savings, latency/energy breakdown。
  - 作用：在无需真实ReRAM硬件的情况下，验证DIA-based in-situ computing在full-flow Transformer加速器中对sparse attention的性能和能效提升。

## IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PIM Access Scheduling (PAS)，包含三个层面的调度：(1) **Workload Mapping**——Algorithm 1基于analytical model在compile time决定FC操作映射到MU（Matrix Unit）还是PIM。Analytical model估计MU执行时间（含weight loading column-tiling pipeline + VU prefetching overlap）和PIM执行时间（= n × PIM(row, col)），选择时间更短的单元。若FFN的第一个FC映射到PIM，则GELU也分配到PIM（PIM硬件支持FC后直接执行GELU via LUT）。对于attention heads，采用head-wise partitioning将Q/K/V weight分布到不同PIM channel，利用attention head parallelism。(2) **Mapping-Aware Scheduling for Multi-Head Attention**——Summarization阶段：FC for Q/K/V映射到MU（matrix-matrix乘法，利用intra-head parallel + inter-head pipeline），weight从PIM通过DMA加载，优先生成Key以并行执行on-chip key transposition（利用DMA做转置，AM→WM数据搬运即partial transpose）；Generation阶段：FC映射到PIM加速matrix-vector乘法，QK^T和SV根据workload特征选择映射到PIM或MU。(3) **Memory Access Scheduling**——当macro PIM command ready时，command scheduler强制未issue的DMA commands进入wait状态，保证PIM执行不被normal memory access打断。PCU将macro command解码为micro commands，PIM MC按GDDR6 timing约束发出bank-level commands。PIM执行完成后释放DMA commands。Scheduling同时考虑resource conflict（PIM memory bank在normal access和PIM computation间互斥）和data dependency（NPU computation和PIM computation间的依赖关系）。
  - 实验比较：(1) PAS vs naive scheduling（view PIM computation as normal memory access, no workload mapping）；(2) QK^T/SV mapped to PIM vs mapped to MU with scheduling；(3) Unified memory + PAS vs Partitioned memory + scheduling；(4) Algorithm 1 FC mapping vs always-PIM / always-MU mapping；(5) 与A100 GPU、DFX、NPU-MEM的端到端性能对比。

- 后端平台是什么，配置是什么。
  - NPU：4 cores, 128×64 systolic array MU (46 TFLOPS@BF16), 16 VLIW processors VU, 700 MHz
  - PIM：GDDR6-AiM based, 8 channels, 16 banks/channel, 1 PU/bank, 1 GHz PU, 32 GFLOPS/PU, 1024 GB/s internal BW, 256 GB/s external BW
  - Scratch-pad：Activation Memory 12MB + Weight Memory 4MB
  - Host：PCIe 5.0 ×16
  - GDDR6 Timing：tCK=0.5ns, tCCDS=tCCDL=1ns, tRAS=21ns, tWR=36ns, tRP=30ns, tRCDRD=36ns, tRCDWR=24ns

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研cycle-accurate in-house simulator（集成商业NPU simulator + AiM PIM simulator），新增PCU、双模memory controller、macro/micro PIM command调度逻辑。模拟器输出latency、throughput、utilization、dynamic energy。
  - 对比平台：(1) NVIDIA A100-SXM-80GB GPU, PyTorch 2.0, CUDA 11.8, HuggingFace+Megatron-LM, torch.cuda.Event API测量延迟；(2) DFX 4-FPGA appliance；(3) NPU-MEM（同NPU+标准GDDR6）。
  - 能耗模型：NPU core动态能耗 + PIM计算能耗（3× DRAM read功耗）+ 标准DRAM操作能耗。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：论文未开源。基于商业NPU（SAPEON X330）和商业PIM（GDDR6-AiM），模拟器需商业IP微架构信息。
  - Kernel输入→性能输出流程：
    1. **输入**：LLM模型（GPT-2 M/L/XL/2.5B, BERT B/L/1.3B/3.9B, BF16）+ workload配置（input 128/256/512 tokens, output 1/8/64/512 tokens, batch=1）
    2. **编译阶段**：Compiler生成ordered commands → Algorithm 1遍历每个command：若cmd.type==MU_FC，估计MU时间（考虑column-tiling with tile size T、pipelined weight loading+computation、VU prefetching overlap）和PIM时间（= n × PIM(row, col)），选时间更短者 → 若FFN首FC→PIM则GELU→PIM → head-wise partitioning分配Q/K/V weight到各PIM channel → mapping-aware scheduling确定QK^T/SV映射（PIM或MU）及流水线调度
    3. **执行阶段**：Command scheduler检查dependency→issue命令到各unit → 当macro PIM cmd ready：PCU解码为micro cmds → NoC广播到所有PIM channel → PIM MC按GDDR6 timing发出bank-level cmds → 全bank全channel并行执行matrix-vector乘法（weight在bank, input vector在global buffer, 16 banks×8 channels同时计算）→ bank内PU执行MAC+accumulate → 完成后释放DMA cmds恢复normal memory access
    4. **输出**：end-to-end latency（ms）、layer-wise latency breakdown（FC/FFN/self-attention/LayerNorm）、throughput（TFLOPS）、compute utilization（%）、dynamic energy（J）、energy efficiency improvement over NPU-MEM
  - 作用：验证PAS在统一内存NPU-PIM系统中的调度效果——Unified memory + scheduling vs Partitioned memory达1.4-1.6× speedup（GPT-2 M-XL），PIM throughput翻倍（因unified可用2× PIM chips）。mapping-aware scheduling带来平均34%性能提升。Algorithm 1在4-16 input tokens范围内达94%准确率选择最优计算单元。

## 34-Stream-Based_Data_Placement_for_Near-Data_Processing_with_Extended_Memory.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：NDPExt的software runtime在host processor上周期性执行cache configuration，将NDP DRAM cache空间分配到各个data stream，co-optimize capacity sizing、spatial placement和data replication。核心运行时计算组件：(1) **Max-Flow Sampler Assignment**——每epoch结束，各NDP unit的512-bit bitvector发送host，runtime构建bipartite graph（units→streams），edges weight=4（每unit 4 samplers），用Edmonds-Karp max-flow算法最大化覆盖stream数。分配<0.5 ms for 512 streams。(2) **Set-based Miss Curve Sampling**——每个sampler用k=32 sample sets同时捕获c=64种capacity case（32 kB到256 MB, geometric partition 1.16× per step）的miss curve，4 samplers × 8 kB per unit = 32 kB SRAM。因为NDPExt使用hash-based set partitioning（不满足stack property），无法用传统utility monitor，改用set sampling + 按K/k scale外推。(3) **Configuration Algorithm (Algorithm 1)**——迭代式co-optimization：每轮找miss curves上steepest slope（最大utility margin）→分配cache space到accUnits[sid]的每个unit（初始每unit单独replication group，最大化replication）→当某unit空间不足时，比较group extending（用nearby unit空间，attenuation factor惩罚remote latency）和group merging（合并两个replication group释放space，重新分布元素）→选utility增益最大的操作→直到无space可分配。每个stream可独立演化出不同replication scheme（通过RGroups实现），比传统NUCA的global replication degree更灵活。(4) **Consistent Hashing优化**——reconfiguration时将65536×64个DRAM row位置映射到hash ring，remap stream data到最近spot减少data movement，减少9.4% invalidation traffic + 3.7% speedup vs bulk invalidation。(5) **Read-Write Coherence**——readOnly bit per stream初始为1，首次写触发exception→host更新remap table→invalidate所有replication group copies→后续写由单一NDP unit直接处理。每stream最多触发一次exception，overhead minor。
  - 实验比较：(a) NDPExt vs Jigsaw/Whirlpool/Nexus baselines（各baseline均有128 kB dual-granularity metadata cache per unit），HBM3-style和HMC2-style NDP memory各自独立评估；(b) NDPExt vs NDPExt-static（no runtime reconfiguration）；(c) Performance breakdown analysis：interconnect latency reduction（NDPExt vs Nexus，如hotspot 38 ns vs 113 ns）、miss rate对比；(d) 消融：sampler assignment time scaling（64-512 streams）、reconfiguration method（Static/Partial/Full）和interval、NDP core count scalability（32-256 cores）、CXL link latency impact（50-250 ns）。

- 后端平台是什么，配置是什么。
  - NDP系统：128-core（8 stacks × 16 cores per stack with 4×4 intra-stack mesh），2 GHz in-order cores。两种NDP memory：HBM3-1600MHz（RCD-CAS-RP: 24-24-24, 256 MB/unit, 16 GB total）和HMC2-1250MHz（RCD-CAS-RP: 14-14-14, 256 MB/unit, 16 GB total）。L1I 32 kB 2-way，L1D 64 kB 4-way，64B cachelines, LRU。
  - Extended memory：DDR5-4800, 4 channels × 2 ranks × 16 banks, RCD-CAS-RP: 40-40-40。CXL Type-3 multi-headed device, 16-lane, default 200 ns link latency。
  - Interconnect：Intra-stack 128-bit link, 1.5 ns/hop, 0.4 pJ/bit。Inter-stack 32 GB/s per dir, 10 ns/hop, 4 pJ/bit。
  - Baseline non-NDP host：64-core CPU + DDR5 + Jigsaw NUCA for 32 MB LLC。
  - SRAM structures (per NDP unit)：SLB 4544 B, ATA 64 kB, samplers 32 kB, bitvector 64 B。Total per-unit SRAM ~100 kB, aligned with prior NDP designs。

- 评估性能的软件/脚本是什么。修改了什么。
  - Simulator: zsim（https://github.com/s5z/zsim），extended for multi-stack mesh NDP + CXL modeling。
  - Workloads: (a) Tensor workloads——recsys (DLRM-style inference), mv (matrix-vector), gnn (GCN with Reddit, sparsedense matmul), SIMD optimized；(b) Rodinia——backprop, hotspot, lavaMD, lud, pathfinder；(c) GAP——bfs, pr, cc, bc, tc (2nd-5th iterations)。Total footprint exceeds NDP memory to stress cache。
  - Stream annotation：手动insert stream configuration hints via configure stream() API，平均4.3行代码修改per workload。Stream类型：affine（statically determined addresses）和indirect（dynamically determined by input data）。Stream数4-256。
  - Baseline Jigsaw/Whirlpool/Nexus adapt to DRAM cache with 128 kB dual-granularity metadata cache（512B block metadata + 64B fine-grained migration, similar to Bi-Modal DRAM Cache）。
  - SRAM modeling: CACTI 7。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - NDPExt的zsim修改和runtime configuration软件未明确说明开源。基础zsim开源（https://github.com/s5z/zsim）。
  - Runtime configuration评估原理与输入→输出流程：
    1. **Stream宣告**：程序员在workload中插入configure stream(affine/indirect, base, size, elemSize [, stride, length, order])声明每个data structure的stream属性。例如PageRank：vertex list (affine stream), edge list (affine stream), rank scores (indirect stream, depend on edge list), visited array (indirect stream)。>99% accesses captured by streams。
    2. **Epoch循环**（每50M cycles）：(a) 硬件samplers在epoch内采集各stream的set-based miss curve——每个sampler用k=32样本sets，对c=64种capacity case各统计hit/miss，4B address per set，总计8 kB per sampler；(b) Epoch结束时，512-bit bitvector + miss curves从所有NDP units发送到host processor。
    3. **Sampler Assignment**：Host runtime读取bitvectors → 构建bipartite graph（unit nodes weighted 4 → stream nodes weighted 1）→ Edmonds-Karp max-flow求解 → 获得下个epoch的sampler-to-stream分配。时间<0.5 ms for 512 streams。
    4. **Cache Configuration**：Host runtime执行Algorithm 1——初始化allocCap全零 → 循环：NextSteepestSlopeSeg(missCurves)找最大utility margin → 分配space到accUnits[sid]各unit → space不足时CalcUtil比较group extending vs group merging → 选更大utility → AdjustAlloc更新allocCap/RShares/RGroups → 直到无space可分配。Output: stream remap table（RShares + RRowBase + RGroups）。
    5. **Reconfiguration应用**：Stream remap table从host发送NDP stacks → consistent hashing remap数据到nearby DRAM rows减少movement → bulk invalidation清理reassigned space（up to 300k cycles, 在50M cycle epoch中占比minor）。
    6. **Data Access路径**：L1 miss → SLB TCAM lookup（range match确定sid + element ID）→ hash element ID确定目标NDP unit → interconnect route → remote SLB lookup RRowBase → afffine stream查ATA/indirect stream hash得DRAM column → DRAM access → miss则CXL access to extended memory。
    7. **输出**：speedup over baselines（geomean 1.41× over Nexus, up to 2.43× on recsys）、energy breakdown（40.3% energy savings vs Nexus）、interconnect latency per workload、miss rate、replication statistics（mv up to 33% space used by replicated data, gnn 27%）。

## 36-Leviathan_A_Unified_System_for_General-Purpose_Near-Data_Computing.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Leviathan的**near-cache engine硬件调度器**负责在正确的时机和位置执行NDC action，是四种NDC范式统一执行的核心运行时机制：
    **(1) Task-offload dynamic调度（DYNAMIC scheduling）**：invoke指令发送task-offload请求后，engine的task-offload scheduler执行层级探测定位actor——先检查本地L2是否缓存该actor→若miss则forward到actor的LLC bank engine→若带EXCLUSIVE flag则进一步检查是否存在remote L2持有exclusive permissions并forward。特殊机制：DYNAMIC任务在需远程执行时以1/32概率改在本地执行，使高temporal locality的object逐步上移到私有cache（data migration），减少后续访问延迟。Invoke buffer（每core 4-entry）在task提交速度超过执行速度时提供背压——engine NACK invoke请求时，core spill该task回core执行（类Livia [47]）。含Future的offload task跳过invoke buffer因wait on future自带背压。
    **(2) Data-triggered调度**：cache controller在cache miss/insertion和eviction时检查TLB bits确定是否触发constructor/destructor。Engine的data-triggered scheduler维护actor buffer（存pending action的actors，此时actors不能被其他线程访问）和vtable cache（映射地址范围到关联的action即Morph's vtable）。对于small objects（< cache line），scheduler对line内所有objects并行执行action；对于large objects（> cache line），单个action插入/逐出多行。
    **(3) Stream调度**：Engine stream scheduler管理producer-consumer协调——track每个active stream的circular buffer size、phantom head/tail指针。Producer（long-lived NDC thread on engine）调用push写入circular buffer，满时block。Consumer（core thread）通过Stream::next()从phantom地址空间load，触发data-triggered constructor从circular buffer copy数据。Core执行pop指令递增head指针，跨cache line时发送消息到engine unlock producer。Deadlock prevention：OOO core speculative load可能全部指向stream end导致所有L1 MSHR reserved——系统NACK超出stream tail的speculative load，仅在commit时re-execute（此时必定指向有效head）。
    **(4) 跨范式交互调度**：Leviathan支持不同NDC范式间的直接交互。PHI案例：core offload RMW task（task offload）→ cache miss触发constructor（data-triggered）初始化phantom data → RMW更新object → eviction触发destructor apply或log updates。Stream案例：long-lived producer（long-lived + data-triggered constructor copy）实现解耦访问-执行。这是首个支持所有范式交互的系统。
    **(5) 内存分配器运行时**：Allocator<T>在运行时的三个任务：(a) pad objects to next power-of-two for cache alignment；(b) LLC object mapping——修改bank-index函数对大于cache line的object zero out LSBs确保单bank映射；(c) DRAM compaction——cache中pad但DRAM中紧凑存储，通过translation buffer在LLC miss/writeback时做地址翻译（与tag lookup并行，无额外延迟）。
  - 实验比较：(a) **PHI PageRank**：Leviathan DYNAMIC + data-triggered调度 vs Baseline vs täkō（data-triggered only, with/without relaxed atomics）——验证task offload消除memory fence overhead并减少NoC traffic 40%；(b) **Decompression**：Leviathan data-triggered调度 vs OL (task offload, L2-local decompress后不retain在L1) vs No Pad（类täkō，无数据布局支持）——验证data-triggered比task offload更适合数据转换场景；(c) **Hash-table lookups**：Leviathan DYNAMIC offload + LLC object mapping vs Baseline vs Livia-like [47]（无padding/mapping）——验证跨object size（24/64/128B）的调度效率；(d) **HATS streaming**：Leviathan streaming vs Baseline BDFS vs täkō data-triggered BDFS——验证dedicated streaming比pseudo-streaming减少engine instructions per edge；(e) **Sensitivity**：invoke buffer size对PHI性能影响、stream buffer size对HATS性能影响、hash table input size对LLC fitting影响、system size（16-100 tiles）对hash table性能影响。

- 后端平台是什么，配置是什么。
  - 周期级模拟器：SwarmSim [36] (https://github.com/CMU-SAFARI/swarm)，大幅修改以支持NDC engine和cycle-level timing。不再使用真实硬件。
  - 系统配置（Table V）：16-core x86-64 Skylake-like OOO, 2.4GHz；16 near-cache engines（per-tile L2+LLC），dataflow fabric 5×5（15 int PE + 10 mem PE, 1-cycle latency, single-issue），32 thread contexts per engine（均分offloaded/data-triggered防死锁），8KB L1d + 256-entry rTLB；L1 32KB 8-way；L2 128KB 8-way（2-cycle tag/4-cycle data）；LLC 8MB（512KB/tile, 16-way, 3-cycle tag/5-cycle data），tr̃rîp replacement, strided prefetcher at L2；Mesh NoC 128-bit flits/links, 2-cycle router/1-cycle link delay；DRAM 4 controllers, 100-cycle latency, 11.8 GB/s/controller, 32-entry FIFO cache per controller。

- 评估性能的软件/脚本是什么。修改了什么。
  - **PHI (PageRank)**：在4M顶点/40M边合成图上16线程执行PageRank。Leviathan实现task offload（RMW action）+ data-triggered（constructor初始化zero / destructor conditionally log or writeback）。täkō baseline用relaxed atomics [9, 70] 模拟RMO（因täkō不支持task offload）或fenced atomics。
  - **Near-cache data decompression**：16K个6B Pixel数组（Zipfian分布 [17] 32K accesses），对每个access的解压数据计算均值。Leviathan实现base+delta lossy decompression constructor（图15，类似BDI [57]）。对比OL（task offload at L2）和No Pad（无padding的data-triggered）。
  - **Hash-table lookups**：16线程各1K次lookup，hash table 4MB（padded），32 nodes/bucket，uniform/Zipfian key分布。Leviathan实现continuation-passing style task offload链（图17，Node::Lookup action递归invoke next node）。object size 24/64/128B三种。对比Livia-like [47]（无padding和LLC mapping）。
  - **HATS graph traversal**：uk-2002图 [21] 上执行PageRank，BDFS traversal producer push Edge到stream，core consumer process edge。Leviathan实现Stream<Edge> + genStream action（图19）。对比täkō data-triggered BDFS（每cache line触发新action重新初始化stack）和software BDFS。
  - **SwarmSim修改**：在baseline SwarmSim上增加engine数据通路（dataflow fabric execution model）、NDC scheduler逻辑（task-offload DYNAMIC probing / data-triggered actor buffer / stream head-tail management）、ISA扩展（invoke/flush/pop）、cache tag扩展、DRAM compaction地址翻译、invoke buffer背压逻辑。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - SwarmSim开源：https://github.com/CMU-SAFARI/swarm。论文未明确说明Leviathan修改是否开源，代码与Livia [47] 和 täkō [66] 共享基础设施（同CMU SAFARI组）。
  - 评估原理和流程：
    1. **NDC action定义**：程序员用C++编写actor class，定义各paradigm对应的action方法。例如task offload：class Node { int64 key, value; Node* next; int64 Lookup(key) { if match return value; else if next invoke next->Lookup(key); } }。例如data-triggered：class Pixel { uint16 colors[3]; Pixel(Decompressor* decomp) { /* base+delta decompression logic */ } }。例如streaming：class LeviathanHATS extends Stream<Edge> { void genStream() { /* BDFS traversal + push */ } }。
    2. **编译与加载**：应用编译为x86-64 binary（含增强ISA的invoke/flush/pop指令）。Simulator加载binary，core按OOO模型执行指令流。actor实例通过Leviathan::Allocator<T>分配，保证cache padding和DRAM compaction。
    3. **Cycle-level调度模拟**：Simulator对每条NDC action执行cycle-level跟踪——invoke指令经invoke buffer→engine task-offload scheduler→DYNAMIC probing（L1D→L2→LLC）→确定执行位置→dataflow fabric执行action指令（PE级并行，数据就绪fire）。Data-triggered actions由cache controller在miss/eviction时触发→engine data-triggered scheduler取actor buffer entry→执行constructor/destructor。Stream producer在engine持续运行push loop→consumer core端pop增量head。Clustered coherence维持engine L1d与core L1d/L2的一致性。
    4. **性能收集**：cycle计数器记录各阶段耗时——core execution cycles、engine execution cycles、cache access cycles、NoC traversal cycles、DRAM access cycles。Energy由各component的dynamic energy model（基于Jenga [75] 和 [60]）乘以activity factor累加。
    5. **输出**：speedup vs baseline（total cycles ratio）、normalized energy（各component breakdown）、附加指标——DRAM accesses breakdown、branch mispredictions per edge、engine instructions per edge、NoC traffic reduction%。

## 38-SpecInfer- Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SpecInfer实现tree-based parallel decoding CUDA kernel用于高效计算token tree中所有节点的tree attention。核心kernel技术包括：(1) **Topology-aware causal mask**——将token tree的拓扑结构编码为causal mask矩阵，使得每个token只attend到其在tree中的祖先节点。与sequence-based causal mask（每个token attend到前面所有token）不同，topology-aware causal mask按tree topology设置attention允许/屏蔽关系。这使得所有tree nodes的attention计算可以fuse到单个kernel中执行。(2) **Depth-first search KV-cache management**——使用DFS遍历顺序更新shared KV-cache，避免多序列独立KV-cache的冗余存储和冗余计算。DFS traversal保证每次计算一个token attention时，其祖先节点的KV已经在cache中。(3) **FasterTransformer-based fused attention kernel**——基于FasterTransformer的attention kernel修改，每个thread block计算单个request的单个attention head，query tensor加载到shared memory，各线程并行计算query/key product分段，broadcast结果用于max和exponential sum计算。
  - 实验比较：(a) Tree-based parallel decoding vs Sequence-based parallel decoding的per-token latency对比（BS=1/2/4/8/16, LLaMA-7B + LLaMA-68M）——tree-based方法对小batch与sequence-based持平，对大batch最多1.8×加速，改善来自消除共享prefix的冗余attention计算和kernel fusion。(b) 不同token tree width对per-token latency的影响（greedy和stochastic decoding，5个数据集）——验证tree width增大增加verified tokens但增加verification latency的tradeoff。

- 后端平台是什么，配置是什么。
  - NVIDIA A10 24GB GPU（AWS g5.12xlarge），CUDA 12.1。单GPU或多GPU（1/4/8 GPU）场景。
  - 软件栈：CUDA 12.1, NCCL, cuBLAS, cuDNN, cuTLASS。基于FasterTransformer的attention kernel修改。
  - 多节点：UCX + MPI + 100 Gbps Ethernet。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于FlexFlow runtime的spec_infer.cc和incr_decoding.cc程序。FlexFlow compiler将DNN computation graph从layer→operator→task三层抽象编译为异步执行的task graph。
  - 关键kernel修改：
    1. **Attention kernel（基于FasterTransformer）**：原始sequence-based attention kernel被修改为支持topology-aware causal mask。原始kernel按sequence topology linearize tokens后做标准causal attention（每个token只attend到前面的tokens）。修改后的kernel按tree topology linearize tokens，使用tree causal mask替换sequence causal mask（mask值根据token在tree中的祖先关系而非sequence位置关系设定），通过单次kernel launch完成整个token tree的attention计算。
    2. **KV-cache管理**：从per-sequence KV-cache改为shared KV-cache + DFS traversal。原始方法为每个token sequence维护独立KV-cache（造成冗余存储和kernel launch overhead）。修改后使用DFS order遍历token tree、复用同一KV-cache空间，不同分支按DFS进入时写入、回溯时恢复。
  - 评估脚本：server_gpu_experiments.sh（服务端GPU评估，单节点+多节点），offloading_experiments.sh（offloading评估）。实验结果输出到FlexFlow/inference/output/路径。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/goliaro/specinfer-ae（含FlexFlow fork、FasterTransformer修改版）
  - Tree attention kernel评估原理和全过程（以LLaMA-7B, 单A10 GPU, BS=1为例）：
    1. **输入准备**：Prompt tokens经SSM speculation后生成token tree 𝒩，tree nodes linearize为token序列（按tree拓扑而非sequence顺序），同时构造topology-aware causal mask矩阵 M_tree ∈ {0, -∞}^(N×N)，其中N为token tree总节点数，M_tree[j,k]=0当且仅当k是j的祖先节点。
    2. **Kernel执行**（单次launch）：
       - Query: Q = X @ W_Q, shape: [N, d_model]
       - Key: K = X @ W_K, 使用共享KV-cache中已缓存的祖先节点keys
       - Value: V = X @ W_V, 使用共享KV-cache中已缓存的祖先节点values
       - Attention scores: A = Q @ K^T / sqrt(d_head), shape: [N, N]
       - Masked scores: A_masked = A + M_tree（-∞位置softmax后为0）
       - Output: O = softmax(A_masked) @ V
    3. **Thread block分工**：每个thread block负责1个request的1个attention head。query vector加载到shared memory。各thread并行计算query/key dot-product的一个segment。然后broadcast partial results做max reduction和exp sum。
    4. **KV-cache更新**：按DFS order计算每个tree node的attention output后，将其K,V写入共享KV-cache的对应位置。当DFS回溯时，对应位置的KV被后续branch的K,V覆盖。这样保证计算每个node时cache中恰好是其祖先的KV，避免cache conflict。
    5. **性能输出**：记录从token tree输入到所有tree node attention output计算完成的wall-clock时间（per-iteration latency），与sequence-based parallel decoding对比（sequence-based方法需要为每个sequence单独launch kernel，产生N_kernel × kernel_overhead的额外开销）。
  - Kernel的性能优势来源：(1) 消除共享prefix的冗余attention计算——多个token sequence共享公共prefix时只需计算一次；(2) 单kernel fusion减少kernel launch overhead——从O(#sequences)次launch降为1次；(3) 共享KV-cache减少GPU memory访问和allocation overhead。

## 3-AMALI- An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：AMALI是一个GPU kernel解析性能模型，从SASS指令级建模CUDA kernel在GPU上执行的cycle数。核心kernel执行建模包括：
    1. **多级层次建模**：Kernel → SM → Sub-core → Warp。总cycle C_kernel = (∑C_i)/numSMs，每个SM的C_i = (∑subC_j)/numSubcs + S_i，其中S_i包含compute/memory resource contention + MSHR + NoC + DRAM stall。
    2. **Warp interval分析**：将warp执行划分为intervals（基于AMAT和数据依赖），建模四类stall：
       - selected（base execution，无hazard）：S_selected = I_warp / IssueRate
       - wait（compute data dependency）：操作数未就绪时的stall
       - short_scoreboard（shared memory access dependency）
       - long_scoreboard（global memory access dependency）
       - 各stall按公式 S_i = ∑ S_k · P(i, flag[k]) 分类求和
    3. **资源争用建模（Interval Parser）**：三类throttle
       - math_pipe_throttle：使用AMALI Tensor Core Model（II_TC = FMA_count/TP_dt）替代GCoM的II = warp_size/FU_lanes，正确建模不同tensor size modifier的吞吐差异
       - lg_throttle：L1 D Cache access contention，沿用GCoM的memory structural hazard模型
       - mio_throttle：MSHR + NoC + DRAM memory侧争用，继承MDM模型
    4. **Kernel Launch Latency (KLL)**：建模之前被所有GPU解析模型忽略的imc_miss（immediate constant cache miss）和no_instructions（instruction cache miss）stall。KLL = s·GS + k，s = α·BS² + β·BS + γ。这两类stall在Llama2-7B推理中占CPI stack高达70%。
    5. **Instruction Divergence (ID)**：针对LLM推理中warp间指令数差异巨大（同kernel中有的warp仅数百条指令，有的超过6000条），引入ID = (max_subCoreInstr - I_SC_Repr.warp) / IssueRate 建模warp间load imbalance。

  - 实验比较：AMALI vs GCoM vs MDM在A100上预测kernel cycle的MAPE。评估Llama3-8B推理（prompt长度128~6144 tokens，batch size 1~8）、Llama3-15B推理（不同prompt长度），以及非LLM benchmark（DeepBench CONV/GEMM, Rodinia BP/B+/DWT/PF）。区分prefill和decode phase。消融分析GCoM→+KLL→+ID→+TCM→AMALI各阶段的MAPE贡献。H100 tensor core design space exploration（128/256/512 FMAs/clk）。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB（Ampere架构），108 SMs，每SM 4 sub-cores，Tensor Core 256 FMAs/clk，HBM2e 80GB
  - 设计空间探索GPU：NVIDIA H100（Hopper架构），Tensor Core 512 FMAs/clk
  - CPU：2x AMD EPYC 7543
  - 软件：Ubuntu 22.04.5 LTS，CUDA 11.7

- 评估性能的软件/脚本是什么。修改了什么。
  - **NVBit**：GPU动态二进制插桩框架，AMALI基于NVBit开发SASS Tracer，在kernel执行时插桩收集每warp的SASS指令trace、内存访问地址、寄存器使用、warp/SM ID等信息。与GCoM使用相同trace收集方法。
  - **Nsight Compute (NCU)**：NVIDIA官方profiling工具，使用`gpc__cycles_elapsed.avg` metric获取硬件ground truth，每个kernel重复测量10次取平均。
  - **cuAssembler**：用于修改SASS trace中HMMA modifier（16816→1688），验证tensor size对CPI的影响（HMMA.16816 CPI=8, HMMA.1688 CPI=4）。
  - **自定义micro-benchmark**：使用pointer chase方法测量GPU FU延迟、内存层次参数（L1/L2/DRAM延迟），以及KLL系数（A100: α=0.0036, β=0.0366, γ=1.1891）。
  - AMALI核心修改（相对GCoM）：(1) TCM：II_TC = FMA_count/TP_dt替换II = warp_size/FU_lanes；(2) KLL：新增constant/instruction cache stall建模；(3) ID：新增warp指令分布建模。

- 开源情况。AMALI核心分析工具链代码未开源（搜索未找到公开仓库）。依赖的开源工具：NVBit (https://github.com/NVlabs/NVBit)，cuAssembler (https://github.com/cloudcores/CuAssembler)。评估原理：NVBit在kernel运行时插桩→收集SASS trace（指令序列、内存地址、warp分布）→ AMALI Cache Simulator计算AMAT→ Interval Analyzer基于数据依赖划分intervals→ Interval Parser建模资源争用stalls→ KLL comp计算launch延迟→ ID组件修正warp差异→输出predicted kernel cycles→与NCU硬件ground truth对比计算MAPE。模型输入为SASS trace；中间计算包括AMAT、interval/profile、资源争用cycles；输出为kernel cycles的预测值及CPI stack breakdown（按selected/wait/scoreboard/math_pipe_throttle/lg_throttle/mio_throttle/imc_miss/no_instructions分类）。

## 43-Trapezoid- A Versatile Accelerator for Dense and Sparse Matrix Multiplications.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Trapezoid实现了四种dataflow，根据输入矩阵的sparsity level（Dense, Mildly Sparse, Highly Sparse）动态选择最优dataflow，在2D spatial array硬件上统一调度矩阵乘法计算：(1) **Dense IP dataflow**：用于D×D，标准inner-product dataflow——A元素stationary在PEs, B列垂直流式传播，MRN reduction mode累加partial products，与TPU一致但用MRN替代horizontal links做reduction；(2) **TrIP dataflow**：用于MS×D和MS×MS，关键创新是多fiber Intersection（intersect 4 A rows × 4 B columns per cycle）——MFIU每cycle产生4×4=16个pairwise fiber intersections，prefix sum计算effectual computation indices，双Benes网络路由A/B值到multiplier，MRN sliced为多个subtrees每subtree产出一个C元素。动态packing：用A/B bitmask预计算各PE row的effectual computations数（popcount on A&B bitvector），取不超过multiplier数(128)的最大B列数，避免overflow；(3) **TrGT dataflow**：用于HS×HS，Gustavson-based temporal dataflow——PE row分为4个PE subrows，每条subrow处理1行A，从global cache gather对应的B row elements（temporal, one element at a time），MRN merge mode（comparator按n坐标归并partial output fibers），写回cache。Multi-level memory hierarchy（global cache→local buffer）提供4 gather reads/cycle；(4) **TrGS dataflow**：用于HS×MS和HS×D，Gustavson-based spatial dataflow——PE row（非subrow）处理1行C，A row的nonzero按k坐标broadcast，B row elements从cache spatial stream（16 contiguous nonzeros/cycle），B Benes网络按n坐标路由到对应multiplier，MRN reduction mode累加partial results。对每个workload自动选取best-performing dataflow（类似Flexagon的做法）。
  - 实验比较：(a) Per-dataflow per-workload perf/area对比——IP-based（Dense IP, TrIP, SIGMA-IP, Flexagon-IP）vs Gustavson-based（TrGT, TrGS, Flexagon-Gustavson）在5个代表性workload（llama0.4-1.0 MS×D, Res0.27-0.15 MS×MS, ca-1.0 HS×D, ca-0.6 HS×MS, ca-ca HS×HS）；(b) 6类sparsity组合上的overall perf/area，自动选最优dataflow vs TPU/SIGMA/Flexagon；(c) Roofline：各dataflow在compute-memory roofline上的位置（TrIP♦在dense端接近compute roofline→TrGS▲在中段saturates memory bandwidth→TrGT■在HS端接近memory roofline）。

- 后端平台是什么，配置是什么。
  - 自建cycle-level模拟器（非真实芯片），模拟配置：128×128 FP32 MACs, 1GHz, 17MB SRAM（16MB cache 4×4MB clusters+1MB local buffers 128×8KB）, 2TB/s HBM, 峰值32 TFLOP/s。面积：81.9 mm² at 16nm。功耗：25-191W（HS inputs低端，MS/D inputs高端），平均110W。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自建cycle-level simulator（C/C++实现，论文未命名），建模：(1) 4 dataflows的完整执行——含loop nest映射（spatial/temporal dimensions）、tiling策略、dataflow selection heuristic；(2) Tiling：coordinate-space tiling on K（TrIP），occupancy-based tiling on M（TrGT/TrGS）；(3) 动态B列packing：对TrIP，预处理阶段用A/B bitmask预计算effectual computations，取不超过128的max B列数；(4) MFIU cycle级行为：AND gate→prefix sum tree→shift unit→distribution routing；(5) MRN mode切换：reduction mode（subtrees）vs merge mode（comparator tree）；(6) Memory hierarchy：cache hit/miss、bank conflict、crossbar contention、local buffer bank access arbitration。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明开源（搜索未找到公开仓库）。评估流程：(1) **输入**：矩阵A/B（CSR/CSC或bitmask格式，含坐标+值）、sparsity level判断（density阈值：>10%→D/MS, <1%→HS）、硬件配置；(2) **Dataflow selection**：D×D→Dense IP（A stationary in PEs, B vertical stream, MRN reduction）；MS×D/MS×MS→TrIP（MFIU 4×4 intersection→prefix sum→shift→Benes route→multiply→MRN subtrees→buffer scatter-write）；HS×HS→TrGT（PE subrow gather B→MRN merge→cache write-back）；HS×MS/HS×D→TrGS（PE row spatial stream B→Benes route→multiply→MRN reduction→accumulate）；(3) **Cycle-level模拟**：trace每个PE/MAC/MFIU/MRN node/buffer bank/cache bank/HBM channel逐cycle的状态，建模contention（multiplier不足时stall、cache miss时等待refill、HBM bandwidth饱和时queue）；(4) **输出**：total execution cycles、perf (TFLOPs, normalized per area)、compute utilization（effectual MACs / peak MACs）、off-chip traffic (bytes, breakdown by A/B/C data type)、energy per component。评估原理：performance = OPs / (cycles / frequency)，area-normalized by RTL-synthesized die area；compute utilization反映sparsity-handling efficiency；off-chip traffic反映dataflow的memory efficiency。

## 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MECLA PE array的on-the-fly matrix regrouping + dual-mode mapping + PSume reuse机制。三个核心kernel调度技术：
    1. **On-the-fly Matrix Regrouping**（Figure 6）：SSMP分解后，每个SS子矩阵以间隔[x·nx, y·ny]散布在权重矩阵中。为最大化硬件利用率和PSume reuse，MECLA在数据映射时进行on-the-fly matrix regrouping——将有关联的子矩阵行/列（间隔nx或ny）集中到一起，使reuse计算发生在同一或相邻PE cluster内，避免冗余计算和数据搬运。
    2. **Dual-mode Source/Derived Sub-matrix Gathering**：根据SSMP配置动态选择最优reuse策略。若nx > ny（outer-product reuse更显著）→SS映射到PE weight buffer，DS scaling scalar映射到scale buffer——利用PE array的PSume重用输出通道间的计算结果；若ny > nx（inner-product reuse更显著）→交换映射：DS scaling scalar映射到weight buffer，SS sub-matrix weight映射到scale buffer——改变矩阵乘法顺序以完全重用SS权重（inner-product reuse）。
    3. **Outer-product Reuse & Inner-product Re-association**（Figure 8）：
       - Outer-product模式（8a）：4×4 weight矩阵视为4×1 scaling vector × 1×4 weight vector。1×4 weight vector先与4×1 input vector乘得1×1 shared PSum，再与4个scaling scalar乘。4个weight data存于4个PE（一行4×4 PE array），scaling factor a/b/c存于scaling multiplier array，未使用的multiplier gated节能。
       - Inner-product模式（8b）：4×4 weight矩阵视为4×1 weight vector × 1×4 scaling vector。先计算1×4 scaling vector与4×1 input vector的乘积，再与4×1 weight vector乘。
       - 效果：给定示例中操作数从28次乘法+16次加法降至4次乘法+4次加法，功耗降低85.6%。
  - 实验比较：(1) MECLA各优化feature的computation和memory reduction ablation——无优化 vs SSMP only vs SSMP+MECLA硬件（Figure 11）；(2) V100 GPU上SSMP vs MECLA processor上的throughput improvement——SSMP on GPU 2.32-2.88× vs SSMP+MECLA 4.25-5.28×；(3) 能量效率breakdown分析（Figure 12）：+SSMP优化（GPU上1.83-2.01×）→+MECLA硬件（34.3-48.6×）的逐级improvement。

- 后端平台是什么，配置是什么。
  - MECLA处理器：28nm CMOS，1GHz，1.0V，8 PE clusters，每cluster 16组4×4 PE + 16 scaling accumulators。PE array内部：每个4×4 PE单元做matrix-vector multiplication（input broadcast + weight unicast），产生4个32-bit PSum→送入4×4 scaling multiplier array（每PSum对应4个multiplier，乘以最多4个scaling factor）。Scaling multiplier间通过内部crossbar通信，支持max 32次PSume reuse。On-chip SRAM: 256KB data buffer（支持≤16 tokens存储，每token embedding dim≤16384 →16KB per token），512KB SS buffer，512KB DS scaling scalar buffer
  - 对比GPU：NVIDIA V100 32GB（125 TOPS INT8 peak）作为baseline
  - MECLA集群配置：32 MECLA processors，900GB/s inter-processor bandwidth

- 评估性能的软件/脚本是什么。修改了什么。
  - 软件侧：PyTorch + HuggingFace library实现SSMP模型→fine-tune后捕获runtime weight（分解为SS+DS）和activation data→生成RISC-V指令→送入VCS post-layout simulation
  - 硬件侧：Synopsys VCS（post-layout gate-level cycle-accurate simulation）+ Synopsys PrimeTime（power analysis with actual running data waveforms）+ DDR4 simulator（memory access simulation）
  - 修改：kernel调度是MECLA硬件的新设计，不是对现有kernel scheduler的修改。PE array的dual-mode mapping和regrouping策略由RISC-V core根据SSMP配置(x,y,nx,ny)在运行时动态选择。
  - PE cluster的8组scaling multipliers通过内部crossbar通信以支持超过4次的PSume reuse（max 32 reuse times，根据algorithm evaluation结果设定）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明开源。评估原理基于post-layout gate-level simulation。
  - Kernel评估流程（以Bloom-7B FFN linear层，SSMP配置(8,8,4,4)，outer-product reuse模式（nx=4 > ny=4相等但选outer-product为例））：
    1. **输入**：activation vector x ∈ R^{1×4096}（token embedding，INT8）；SSMP weight: WSS ∈ R^{128×344×8×8}（INT8），scaling scalar S ∈ R^{128×344×4×4}（FP16）；SSMP配置：x=8, y=8, nx=4, ny=4
    2. **Mode Selection**（RISC-V决策）：比较nx vs ny。若nx≥ny→outer-product模式：WSS→PE weight buffer，S→scale buffer；否则inner-product模式：S→PE weight buffer，WSS→scale buffer
    3. **Matrix Regrouping**（数据重排）：按output channel重排——将同一SS对应的4个DS（沿output channel方向，nx=4）的output channels集中到同一PE cluster。实际映射：SS [8,8] weight data → 4×4 PE array（2 cycles per SS row），S [4,4] scaling → scaling multiplier array
    4. **Outer-product PSum Reuse计算**：
       ```
       for each SS_block in WSS [128×344]:
           # Step A: 1×8 weight slice × 8×1 input slice → 1 shared PSum
           for row in 0..7:  # SS rows (output channels)
               w_slice = SS_block[row, :]  # [8] weight values
               x_slice = x[corresponding_input_indices]  # [8] activation
               PSum[row] = dot(w_slice, x_slice)  # 1 MAC int8→int32
           # Step B: 4×4 PSum × 4×4 scaling → 16 outputs
           for i in 0..3:    # nx
               for j in 0..3:  # ny
                   out_ch = row_base + i*8 + row  # actual output channel
                   out[out_ch] += PSum * S[region_i, region_j, i, j]
       ```
       - 传统方法需每output channel重复完整w×x计算（16次8-MAC=128 MACs），PSume reuse仅需8次MACs（8 weight）+ 16次scaling mult = 24 ops，计算量降至18.75%
    5. **Scaling Accumulator**：scaling multiplier接收4个不同PSum（4 PE rows），每个multiplier乘以各自的scaling factor，结果accumulate
    6. **Cross-cluster Reduction**：8 PE clusters各输出部分结果→sum得最终output activation vector ∈ R^{1×11008}
    7. **VCS Cycle Counting**：gate-level simulation记录每个operation的cycle数：weight load cycles (DDR4→SS buffer) + compute cycles (PE array) + scaling cycles (scaling multiplier) + accumulation cycles + output writeback cycles
    8. **性能输出**：total_cycles → Throughput = total_MAC_ops / cycles * 1GHz GOPS；Computation reduction = 1 - MECLA_ops / naive_ops；Memory reduction = (naive_weight_bytes - SSMP_weight_bytes) / naive_weight_bytes；Power = PrimeTime dynamic + static power from switching activity trace

## 46-Fast On-device LLM Inference with NPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - **实现**：llm.npu 提出 out-of-order subgraph execution——将 LLM prefill 阶段的 Transformer blocks 按 layer 和 chunk 两个维度拆分为多个 subgraph（NPU 子图：INT8 Linear/FFN；CPU/GPU 子图：LayerNorm、Attention、shadow outlier computation），在跨 chunk 依赖约束下进行乱序调度。关键设计：(1) 两层依赖建模——cross-chunk 依赖：Attention 等依赖前序 chunks 的 KV Cache；intra-chunk 依赖：LayerNorm→Linear→Quantize 等顺序依赖。(2) 在线贪心调度算法——定义每个 subgraph 对减少 NPU stall 的贡献值 C：若 g 在 CPU/GPU 上执行，C = Σ T_i（g 完成后可释放到 NPU 的 subgraph 集合 S 的总执行时间）；若 g 在 NPU 上执行，C = -Σ T_i。每调度决策取 max C 的 subgraph。(3) profiling 阶段离线收集各 subgraph 的执行时间和依赖关系，运行时微秒级在线调度。
  - **实验比较**：(a) Ablation study：naive NPU offloading → +chunk-sharing → +shadow outlier → +OOE（out-of-order execution），OOE 额外降低 prefill 延迟 18%–44%；(b) 整体 vs 5 个 baseline（llama.cpp-CPU、MNN-CPU、MLC-GPU、TFLite-GPU、PowerInfer-V2-NPU）。
  - 后端平台是什么，配置是什么。
    - Redmi K70 Pro（Snapdragon 8gen3，Hexagon NPU 73 TOPS，24GB RAM）、Redmi K60 Pro（Snapdragon 8gen2，16GB RAM）。Android 13。Qualcomm QNN SDK。CPU 和 NPU 共享统一物理内存但使用独立地址空间，通过 shared buffers 同步中间结果。
  - 评估性能的软件/脚本是什么。修改了什么。
    - 评估软件：基于 MLLM（https://github.com/UbiquitousLearning/mllm）和 QNN（Qualcomm Neural Processing SDK）构建。新增约 10K 行 C/C++ 和汇编代码。
    - 修改内容：
      1. 实现 KVCache、SiLU、RMSNorm、ROPE 等 QNN 原生不支持的 LLM 算子。
      2. Chunk-sharing graph：将 LLM 拆分为 static operators（Linear、LayerNorm，跨 chunk 共享）和 dynamic operators（Attention，不同 chunk 独立），static 部分构建一次后复用，减少内存占用 75%（7.2GB）。
      3. 对 Linear 层进行 tensor shape 优化（如将 2048×2048 等价转为 32×32×2048 以更好利用 NPU CNN 架构偏好），profiling 阶段选择最高效的等效 shape。
      4. 内存管理：优先将计算密集型任务（FFN）调度到 NPU 4GB 有限内存中。
      5. Out-of-order scheduler：在 CPU 上运行微秒级在线调度器，根据 NPU stall 贡献值 C 选择下一 subgraph。
  - 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
    - 开源：https://github.com/UbiquitousLearning/mllm（MIT license），Zenodo DOI: https://doi.org/10.5281/zenodo.14392760。
    - 评估原理与全流程：
      ```
      # === 准备阶段 ===
      # 1. 量化模型生成（需 A100 服务器，见 accuracy_results/）
      python llmnpu_get_int8_weights_finalmodel_int8bias_ns.py \
          --model Qwen1.5-1.8B --quant_method max-min_symmetric \
          --calib_data wikitext --outlier_prune_rate 0.85

      # 2. 为每个 LLM 模型生成 chunk-sharing compute graph（固定 chunk_length=256）
      #    对于每个 layer，将 LLM 分解为 144 个 subgraph，
      #    其中 120 个为 static（跨 chunk 共享），24 个为 dynamic（如 Attention）
      subgraphs = decompose_LLM_into_subgraphs(model, chunk_length=256)
      static_subgraphs = subgraphs.filter(type in [Linear, LayerNorm, FFN])  # 120 个
      dynamic_subgraphs = subgraphs.filter(type in [Attention])              # 24 个

      # 3. 离线 profiling：对所有 static+dynamic 子图在 NPU/CPU 上分别测时
      for g in subgraphs:
          g.time_npu = profile_on_NPU(g)
          g.time_cpu = profile_on_CPU(g)
          g.processor = "NPU" if g.contains_int8_linear else "CPU"
          # 记录依赖关系
          g.deps_cross_chunk = [all_cross_chunk_deps[g.chunk][g.id]]
          g.deps_intra_chunk = [g.chunk * M + g.id - 1]

      # === 运行时推理阶段 ===
      def llm_npu_prefill(prompt_tokens, chunk_length=256):
          num_chunks = ceil(len(prompt_tokens) / chunk_length)
          pending_subgraphs = Queue()

          # 1. 将所有 chunk 的第一个 subgraph 入队
          for c in range(num_chunks):
              pending_subgraphs.push(Subgraph(chunk=c, sub_id=0))

          results = {}
          while pending_subgraphs:
              # 2. 在线调度：对每个 pending subgraph 计算贡献值 C
              best_g = None
              best_C = -inf
              for g in pending_subgraphs:
                  S = get_new_ready_subgraphs_after(g)  # g 完成后可释放的 subgraphs
                  if g.processor == "CPU":
                      C = sum(s.time_npu for s in S)  # 释放 S 到 NPU 的总时间
                  else:  # NPU
                      C = -sum(s.time_cpu for s in S)  # S 在 CPU 上的代价
                  if C > best_C:
                      best_C, best_g = C, g

              # 3. 在相应处理器上执行选中的 subgraph
              result = execute_on_processor(g=best_g, processor=best_g.processor)
              results[best_g.id] = result

              # 4. 将新就绪的 subgraph 入队
              for next_g in best_g.successors:
                  if all(dep in results for dep in next_g.dependencies):
                      pending_subgraphs.push(next_g)

          return results

      # 3. 性能指标采集
      # - Prefill speed (tokens/s): prompt_length / max(NPU_time, CPU_time)
      # - Energy (J): 通过 /sys/class/power_supply 每 100ms 采样
      # - Memory (GB): 通过 Android dumpsys meminfo 获取
      ```
    - 调度核心：NPU 执行时间常构成 critical path（如 256-token prompt 下 Qwen1.5-1.8B NPU 315ms vs CPU 约一半），因此贪心算法以"最大化 NPU 利用率 + 最小化 CPU/GPU 占用对 NPU 的干扰"为目标，而非最大化并行度。Naive 重叠导致 37% bubble rate，OOE 将 bubble rate 降至 0.7%。

## 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - SpecEE 提出 **Context-aware merged mapping for predictors**（上下文感知的合并映射）用于 speculative decoding 场景下的 early exiting。核心是自定义 GPU kernel：将 speculative decoding token tree 中每条 path 的多个 token 合并为一个 hyper-token，通过 block-wise GEMM（基于 cutlass 的 group GEMM，参考 MegaBlocks 设计）一次计算 draft token logits，从 exponential mapping complexity 降为 linear complexity。此外，支持与正交加速技术（quantization/AWQ、sparse activation/PowerInfer）集成。实验比较 SpecEE+EAGLE vs EAGLE（speculative decoding baseline）：Llama2-7B 上平均 speedup 1.05×，Llama2-13B 上 1.06×（均在 NVIDIA A100-80GB）。
  - 自定义 GPU operator 位于 feature extraction 阶段：计算 speculative token logits 需要 hidden_states × speculative_lm_head，对于 token tree 中多个 token path 的 hidden states，通过 merged mapping 将多个 path 的特征批量计算。

- 后端平台是什么，配置是什么。
  - NVIDIA Tesla A100-80GB GPU，CUDA 12.1
  - NVIDIA RTX 4090 24GB GPU，CUDA 12.1
  - Lenovo Legion Y7000，NVIDIA RTX 4060 Laptop 8GB，CUDA 12.6

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：Cloud 场景基于 Pytorch + C++/CUDA 自定义后端；PC 场景基于 llama.cpp C++/CUDA
  - 自定义 GPU kernel 修改：
    1. 基于 cutlass group GEMM 实现 block-wise 矩阵乘法，用于计算 hyper-token 对应的 speculative token logits（hidden_states[batch_paths × hidden_dim] × speculative_lm_head[hidden_dim × num_spec_tokens]）
    2. 参考 MegaBlocks block-sparse 设计思想：token tree 中不同 path 的 feature computation 具有相同 pattern（计算 hidden_states × speculative_lm_head 的指定 columns），通过 group GEMM 合并为一次 kernel launch
    3. Merged mapping 原理：token tree 中 path (root→I→thank) 等所有 path 共享同一个 speculative_lm_head，不同 path 的不同 hidden states 通过 batch dim 组织，合并为 hyper-token batch，将复杂度从 O(tree_size × vocab) 降为 O(tree_depth × num_spec_tokens)
  - 与正交技术的集成（kernel 层面）：AWQ 量化模型使用量化后的 lm_head columns；PowerInfer sparse activation 下 predictor 仅访问 hot neurons 对应的权重列

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/infinigence/SpecEE，Zenodo DOI: https://doi.org/10.5281/zenodo.15102802
  - GPU kernel 输入到性能输出全过程（Cloud scenario, speculative decoding with EAGLE）：
    1. **输入**：EAGLE DLM 生成 token tree（如 depth=2, branching=3 → 最多 1+3+9=13 tokens）。Prompt + 已生成的 token sequence 构成 input context
    2. **Merged Mapping**：token tree 中每条 path（如 path_1: ?→I→thank, path_2: ?→I→am, ...）被合并为 hyper-token 的概念。对于给定 layer i，所有 path 的 hidden states 组织为 batch [num_paths × hidden_dim]
    3. **Custom GEMM Kernel（基于 cutlass group GEMM）**：`draft_token_logits = group_gemm(hidden_states_batch, speculative_lm_head)`。Group GEMM 将多个小 GEMM 合并为一次 block-wise 调用，利用 GPU tensor core 并行。speculative_lm_head 是 lm_head 的 4 列（对应 4 个 speculative tokens），shape [hidden_dim × 4]
    4. **Feature Extraction Kernel**：对 draft_token_logits 执行 softmax（local probabilities）、计算与前一层 local probs 的 difference（probability variation），拼接为 12-dim feat
    5. **MLP Predictor Kernel**：W₁(12×512) 和 W₂(512×1) matmuls 在 GPU tensor core 执行，Sigmoid 输出 exit probability。CUDA kernel 以 batch 方式处理所有 path 的 predictor
    6. **Verification**：若某 path 的 hyper-token 满足 exit 条件，则通过 lm_head（global projection）验证 global top token 是否在 speculative set 中
    7. **性能输出**：tokens/sec measured by dividing total generated tokens by total inference time。SpecEE+EAGLE 在 Llama2-7B 上实现 120.8 tokens/s（vs EAGLE 114.5 tokens/s on MT-Bench），~1.05× speedup
  - 关键：merged mapping 将 mapping 复杂度从指数（每 token 独立 per-tree_node）降至线性（per-path 合并处理），且 custom GEMM kernel 利用 GPU 并行性将 predictor cost 压缩至 5.6% 总推理时间

## 49-HeterRAG- Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：HeterRAG在两种PIM设备上运行不同的kernel级计算：(1) **AccelDIMM上的距离计算kernel**：ANNS检索中distance computation占retrieval总时间的>80%。每个RPM (Rank-level Processing Module) 内的Distance Computation Unit执行query vector与DRAM中vertex vector的内积距离（inner product），支持FP32精度的32个并行乘法器和32个加法器。数据路径：DIMM接收PIM-Inst→DPM解析Rank ID并分发到对应RPM→RPM的Inst Decoder解析指令→检查Vertex Cache (LRU)→cache miss时DDR-C/A Generator生成DDR ACT/RD/PRE命令从DRAM读取vector数据→Distance Computation Unit计算距离→结果写入DPM的Distance Buffer→DPM轮询方式回传TPM的Functional Block→Functional Block的Top-k Priority Queue排序更新。(2) **AccelHBM上的GEMV kernel**：LLM decode阶段以GEMV为主，全部offload到HBM内的BPM (Bank-level Processing Module)。BPM含2个向量内积计算单元（与AccelDIMM距离计算单元类似），输入：bank-level row buffer数据 + channel-level global buffer数据。GEMV映射方案同AttAcc [64]的KV矩阵和权重矩阵组织方式，确保并行性并最小化数据移动。(3) **AccelHBM TPM上的GEMM和其他操作kernel**：GEMM在TPM的Matrix Unit (128×128 Systolic array × 8) 上执行，element-wise和normalization (LayerNorm/RMSNorm/Softmax) 在Vector Unit (16 4-wide VLIW processors × 8) 上执行。这些操作因算术强度较高不适合offload到内存。(4) **Locality-aware retrieval的cache kernel**：每个RPM的Vertex Cache (128KB, 分local/global两部分) 缓存频繁访问的vertex vectors，LRU替换。cache hit直接使用缓存数据计算distance，skip DRAM ACT/RD/PRE。迭代RAG时将上一轮result buffer的vertex vectors写入initial buffer作为下一轮搜索的起始点。(5) **Locality-aware generation的KV处理kernel**：Tree Search Unit在prefix tree中匹配新文档序列→KV Substitution Unit合并matched文档的dense+sparse KV→Token Filtering Unit为unmatched但有cached dense KV的文档选择important tokens (10-20%) 进行selective recompute。(6) **Fine-grained parallel pipeline调度**：host按固定间隔聚合AccelDIMM的中间检索结果→已完成/高置信度部分结果提前发送AccelHBM→AccelHBM立即开始prefilling (利用locality-aware generation的prefix tree匹配)→检索完全完成后进入decoding→AccelDIMM和AccelHBM同时工作，平均资源利用率提升到44.5% (AccelDIMM) 和38.6% (AccelHBM)。
  - 实验比较：(a) Distance computation offload到RPM vs 全在TPM/CPU执行——retrieval阶段bandwidth利用率提升；(b) GEMV offload到BPM bank-level并行 vs GPU上执行——decode阶段memory wall缓解；(c) Vertex Cache (128KB LRU) hit rate和DRAM access reduction效果；(d) KV cache的dense/sparse分离 + selective computation vs 全量KV recompute——prefilling时间减少；(e) parallel pipeline vs sequential execution——overall hardware utilization对比（Table 3）。

- 后端平台是什么，配置是什么。
  - AccelDIMM配置：DDR4-3200 MT/s, 16Gb × 8, 2 Ranks。tRC=72, tRCD=22, tCL=22, tRP=22, tBL=4, tRRD_S=4, tRRD_L=6, tFAW=30, tCCD_S=4, tCCD_L=6。每DIMM: TPM (128-entry Inst Queue, 16MB Visited List Buffer, 1KB Initial Buffer, 2KB Neighbor Buffer, 128 FP32 Comparator ×8 Top-k PQ), DPM (256-entry PIM-Inst Queue, 32KB Distance Buffer), RPM (128-entry PIM-Inst Queue, 128KB Vertex Cache, 32 FP32 Mult + 32 FP32 Adder)。
  - AccelHBM配置：8-Hi HBM2 stack, 8 Channels。tRC=45, tRCD=16, tCL=16, tWR=16, tRAS=29, tRRD=2, tCCD_S=2, tCCD_L=4。TPM (128×128 SA ×8, 16 VLIW ×8, 1 TSU/KVSU/TFU ×8, 24MB Scratchpad)。BPM (32 FP16 Mult, 32 FP16 Adder, 8Kb Buffer per BPM)。每两个bank一个BPM。
  - CPU-GPU baseline: Intel Xeon Gold 5117 (2.00GHz, 256GB DDR4) + NVIDIA Tesla V100 (32GB HBM2)。
  - 硬件单元Verilog综合：65nm工艺@500MHz → power/area scaling到22nm + 50%面积补偿。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研cycle-accurate仿真框架：扩展Ramulator（增加TPM和in-memory processing modules建模）+ 修改ZSim（offload RAG到memory侧）。DDR4和HBM2两种memory-side配置。
  - CPU-GPU baseline评估：Intel RAPL (CPU能耗) + nvprof (GPU能耗) 测量。
  - 硬件实现评估：Synopsys Design Compiler综合Verilog RTL (65nm, 500MHz)，CACTI 7.0建模buffer/cache延迟和能耗，MICRON DDR4 Power Calculator计算DDR4功耗，TPU-v4i参考计算HBM2功耗。
  - 修改：Ramulator增加PIM指令处理、距离计算单元建模、vertex cache建模；ZSim增加ANNS和LLM推理的memory offload路径。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明仿真框架和RTL是否开源。
  - Kernel执行到性能输出的全过程：(1) **检索阶段距离计算kernel**：query vector加载到所有RPM的Query Reg → PIM-Inst指定vertex ID和DRAM地址 → RPM Inst Decoder解析 → Vertex Cache tag比较 (cache hit: 128KB LRU → 直接送入Distance Computation Unit; cache miss: DDR-C/A Generator发出ACT→RD→PRE命令读DRAM) → 32 FP32 Mult + 32 FP32 Adder并行计算内积距离 → 距离值写入DPM Distance Buffer → DPM轮询传回TPM → Functional Block的Top-k Priority Queue (128 FP32 Comparator ×8) 插入排序 → 更新邻接表。(2) **生成阶段GEMV kernel**：TPM Request Generator发出PIM请求broadcast到所有HBM Ctrl → BPM接收weight vector (from bank row buffer) 和activation vector (from channel global buffer) → 两个向量内积单元 (32 FP16 Mult + 32 FP16 Adder) 并行计算 → 部分和累加 → 结果传回TPM→ Vector Unit做activation → 下一层。(3) **性能原理**：Ramulator cycle-accurate跟踪DDR/HBM command时序——每个ACT/RD/PRE command和tRC/tRCD/tCL等timing constraint决定memory access延迟；CACTI建模buffer/cache的SRAM访问延迟；RTL综合给出compute unit延迟。对每个RAG request：host embedding时间 + AccelDIMM检索时间（distance computation次数×每次延迟 + neighbor fetching DDR延迟 + queue updating时间）+ interconnect传输时间（32GB/s PCIe 5.0 ×8）+ AccelHBM生成时间（prefill GEMV/GEMV次数×每次延迟 + decode GEMV次数×每次延迟）。多次request并发得到throughput (QPS)。

## 50-S-DMA- Sparse Diffusion Models Acceleration via Spatiality-Aware Prediction and Dimension-Adaptive Dataflow..pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：S-DMA的Dimension-Adaptive Dataflow (DAD) 是runtime计算调度机制，在扩散模型每个denoising timestep内动态选择PE阵列的数据流模式以最大化稀疏跳过效率。核心调度逻辑：(1) **Sparsity统计**：SAP预测模块输出per-layer的sparsity mask后，DAD controller统计spatial维度(H×W)和channel维度(C)各自的稀疏率(sparsity ratio = zero_activations / total_activations)。(2) **模式选择**：若spatial sparsity > channel sparsity → 选择S-first (Spatial-first) 数据流——PE阵列按spatial位置维度外层循环、channel维度内层循环遍历，在spatial维度上zero-skip跳过零值空间位置；若channel sparsity > spatial sparsity → 选择C-first (Channel-first) 数据流——PE阵列按channel维度外层循环、spatial维度内层循环遍历，在channel维度上zero-skip跳过零值通道。(3) **Router配置**：DAD controller通过可配置的PE间router/network-on-chip重配置数据路径，将activation broadcasting和weight stationary映射到选定数据流模式，保证数据复用的正确性和最大化。(4) **多kernel切换**：UNet每层（Conv、Attention、Upsample/Downsample）可能具有不同的稀疏性分布特征，DAD在每层计算前重新评估并切换数据流。(5) **SAP与DAD的协同**：SAP的预测结果直接驱动DAD的模式选择——预测mask中sparse位置被DAD调度跳过，dense位置正常计算，形成predict→schedule→execute的流水线。
  - 实验比较：(a) S-DMA (SAP+DAD) vs 仅DAD无SAP的版本——验证SAP预测准确性对调度效率的影响；(b) DAD自适应数据流 vs 固定C-first数据流 vs 固定S-first数据流——验证维度自适应的收益；(c) S-DMA vs GPU baseline的per-layer加速比；(d) 不同denoising timestep下DAD模式切换的频率和有效性；(e) DAD dataflow切换的overhead（reconfiguration cycles）占总执行时间的比例。

- 后端平台是什么，配置是什么。
  - S-DMA加速器ASIC：PE阵列规模论文未明确说明（典型16×16或32×32），片上SRAM buffer（activation buffer + weight buffer + sparsity mask buffer），片外DRAM。FPGA原型验证使用Xilinx FPGA平台（具体型号论文未明确说明）。GPU baseline使用NVIDIA GPU（具体型号论文未明确说明，推测为V100或RTX 3090级别）。
  - 对比S-DMA加速器：相同PE阵列规模的dense accelerator（无sparsity支持）、以及已有稀疏加速器（SCNN、SparTen等）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 论文自研cycle-accurate simulator评估S-DMA性能。Simulator建模：(a) SAP预测器延迟（轻量CNN前向推理cycle数）；(b) DAD controller的sparsity统计和模式选择延迟；(c) PE array cycle-accurate执行——逐MAC操作追踪，zero-skip逻辑由sparsity mask控制；(d) 片上buffer读写延迟；(e) 片外DRAM访问时序（集成DRAM时序模型）。修改：相比通用cycle-accurate simulator，S-DMA simulator新增SAP模块建模、DAD controller状态机、可配置PE router建模和zero-skip PE执行逻辑。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明simulator和RTL是否开源。
  - DAD调度kernel执行到性能输出的全过程：
    1. **输入阶段**：扩散模型UNet权重加载到片外DRAM，当前timestep t的带噪图像x_t加载到activation buffer。SAP模块读取x_t → 轻量CNN前向推理 → 输出per-position binary sparsity mask存入sparsity mask buffer。
    2. **调度阶段**：DAD controller读取sparsity mask → 分别统计spatial维度(H×W)和channel维度(C)的sparsity ratio → 比较两者大小 → 选择C-first或S-first数据流 → 发送配置信号到PE array router → router重配置PE间数据路径（activation broadcast方向、weight stationary映射、partial sum reduction路径）。DAD切换overhead为configurable router的reconfiguration cycles（通常数个cycle）。
    3. **执行阶段**：PE array按选定数据流执行UNet层计算。以Conv层为例：(a) C-first模式：外层循环遍历channel→内层循环遍历spatial位置，sparsity mask为0的channel整层跳过（所有PE跳过该channel的全部spatial位置）；(b) S-first模式：外层循环遍历spatial位置→内层循环遍历channel，sparsity mask为0的spatial位置跳过（所有PE跳过该位置的全部channel）。非零位置的MAC正常执行：activation × weight → partial sum accumulation → activation function (SiLU/ReLU)。
    4. **性能输出**：Simulator统计每timestep各层的active MAC operations（实际执行的非零MAC）和total cycles（含跳过位置节省的cycle）。Throughput = total_active_MACs / total_cycles → GOPS。Energy = PE_dynamic_energy (active cycles × per-MAC energy) + PE_idle_energy (skipped cycles × gating power) + SRAM_access_energy + DRAM_access_energy。Speedup = dense_accelerator_cycles / S-DMA_cycles。SAP prediction accuracy = (correctly predicted zeros + correctly predicted non-zeros) / total predictions → 影响DAD调度有效性。
    5. **评估原理**：Cycle-accurate仿真逐cycle追踪每个PE的状态（active/idle/gated）、buffer访问、DRAM命令。S-DMA的加速来源于：(a) SAP准确预测零激活→减少无效MAC计算；(b) DAD根据当前层稀疏性分布自适应选择最优遍历维度→最大化zero-skip粒度（C-first可整channel跳过，S-first可整spatial位置跳过）→减少PE idle bubbles和data movement overhead。

## 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - **实现**：MNM将RALM系统的两大memory-bound kernel分别调度到最合适的硬件位置执行——(1) **语言模型MHA kernel调度到PIM**：将Self-Attention和Chunked Cross-Attention的GEMV-dominant操作（Score = Q·K^T逐token逐head, Context = softmax_out·V）从GPU offload到HBM core die的PIM logic。每HBM bank含16-bit FP MAC unit，利用bank-parallel执行：数据分布——K^T和V矩阵的16个FP16 elements per column存PIM row buffer，Q vector的16-element segment存global vector buffer→PIM_MAC_AB做16-element dot-product across all banks，结果经PIM_MV_SB送logic die Softmax calculator→Softmax输出写回global vector buffer→再次PIM_MAC_AB执行Context操作。(2) **检索器IVF-PQ kernel调度到PNM**：将PQ code scan（LUT lookup + residual dot-product）和top-k selection（Odd-Even Merge Sort）从GPU offload到HBM logic die的PNM logic。16个并行PQ code scanner，每个含24个FP16 MAC units（合计384-dim dot-product per scanner），每nCCDL=4 cycles从HBM3读取1024B的PQ codewords（64个1B indices）。GPU只负责cluster selection（计算||x-y_c||²选出top nprobe）和最终merge多cluster的top-k结果。两个kernel通过dual row buffer实现并发——PIM MAC操作占1条row buffer时，PNM可通过第2条row buffer独立读PQ codewords，避免bank contention。
  - **比较目标**：GPU Baseline (H100 NVL)、2×GPU、AttAcc (PIM-only MHA加速) [80]、ChamVS-D (PNM-only DIMM-based检索加速) [34]、ChamVS-H (PNM-only HBM-based检索加速)、PipeRAG (co-execution scheduling，非kernel offload) [35]、4×GB200/8×GB200。

- 后端平台是什么，配置是什么。
  - **主GPU**：NVIDIA H100 NVL, 94GB HBM3, 132 SMs @ 1GHz, L2 cache 60MB, shared memory 132KB per thread block, max 32 thread blocks per SM, max thread block size 1024, interconnect NVLink 4.0。
  - **MNM HBM**：6× MNM-enabled HBM3 stacks（+6× standard HBM3 stacks on GPU side = 12 total）。每MNM stack: 8-Hi 16GB, 1024-bit interface @5.2Gb/s/pin, 16Ch/2pCh/2Ra/4BG/4BA organization, page size 1KB。Timing: tCL=7.308ns, tRP=7.308ns, tRAS=17.308ns, tCCDS=0.769ns, tCCDL=1.538ns, tRC=24.231ns。
  - **PIM logic per core die**: FP16 MAC @650MHz (4× slower than HBM3 I/O 2.6GHz), area 0.11mm², power 0.00105W per MAC unit; dual row buffer area 1.38mm², power 0.154W。
  - **PNM logic per logic die**: 16× PQ code scanners @650MHz, area 1.00mm² total, power 0.99W; Top-k sorter area 0.003mm², power 0.00395W; Softmax calculator power 0.154W。
  - **Host CPU**：Intel Xeon Gold 6526Y。

- 评估性能的软件/脚本是什么。修改了什么。
  - **Profiling工具**：NVIDIA Nsight Systems [74] + Nsight Compute [73] 对RETRO-0.5B/1.5B/7.5B + FAISS-GPU IVF-PQ进行GPU workload profiling，提取各kernel的latency breakdown（SA MHA/CCA MHA/FFN/QKV gen/Projection/PQ code scan/Top-k selection/Cluster selection）和memory utilization/arithmetic intensity，绘制roofline model。
  - **Simulator**：基于AttAcc simulator [80]（GPU-PIM cycle-level simulator）修改，扩展包含：(a) IVF-PQ性能模型——FAISS-GPU [37]的检索kernel（cluster selection dot-product、PQ code scan LUT lookup+dot-product、top-k sort）的GPU cycle/energy建模，经Nsight profiler验证；(b) MNM PIM command set执行模型——PIM_ACT_AB（tRCD-timing row激活）、PIM_WR_GB（global vector buffer写入，1 cycle per 64B）、PIM_MAC_AB（16个FP16 MAC并行16 cycles完成256-dim accum，memory-bound→total cycles由HBM bandwidth决定）、PIM_MV_SB（score buffer转移经global bus到logic die）；(c) MNM PNM command set执行模型——PNM_WR_MMIO（MMIO register写入，host→DMA→HBM reserved address）、PNM_RET_INIT（PQ code buffer加载nCCDL=4 cycles per 1024B, MAC module 24×16=384-dim dot-product in pipeline, Full sorter 16→16 stage, Partial sorter 32→16 merge stage）；(d) Selective batching + Early generation调度——T_gen/T_ret,max测量→N_ret=⌊T_gen/T_ret,max⌋ request group→Retrieval Group Queue with On-Ret Group tracking→updateQueue重排基于early generated tokens count。
  - **Workload配置**：RETRO语言模型（0.5B/1.5B/7.5B, 12 decoder blocks, embedding_dim=384, 各模型Numhead不同）、IVF-PQ检索器（Config 1-5: nlist=32768→2048, nprobe=16→256, interval=64→8, top-k=1→16, M=64固定）、数据集（Wikipedia 80M vectors、Realnewslike C4 140M vectors）。Scalability用89B/175B/310B GPT-style decoder-only模型projection。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **论文未开源MNM simulator**。基础组件AttAcc [80]和Ramulator2 [62]为已知开源项目，但MNM扩展（IVF-PQ模型、MNM commands、调度模型）未找到公开仓库。可在GitHub搜索AttAcc（AttAcc! Unleashing the power of PIM for batched transformer-based generative model inference, ASPLOS 2024）获取上游simulator，但无MNM相关代码。
  - **评估原理与全过程（Config 5, RETRO-7.5B, Batch 32, Realnewslike 140M）**：
    1. **Kernel识别与profiling**：NVIDIA Nsight Systems采集RETRO推理trace→提取kernel时间线：(a) MHA Score: batched GEMV per token per head（Q[1×64]·K^T[64×seq_len]，seq_len增长→memory-bound）; (b) MHA Context: softmax_out[1×seq_len]·V[seq_len×64]; (c) IVF-PQ: Cluster selection (GEMV nlist×dim→memory-bound), PQ code scan (LUT access per codeword→low compute utilization), Top-k selection (sort→GPU-unfriendly)。Roofline: H100 NVL peak 990 TFLOPS (FP16 Tensor), 3.35 TB/s HBM bw; MHA ~1.0 Op/B, PQ scan ~0.1 Op/B, Top-k ~0.01 Op/B → all in memory-bound region。
    2. **Kernel分派**：基于profiling结果→MHA Score/Context kernel分派到PIM（高bank-level parallel bandwidth: ~4TB/s per HBM stack internal bw vs ~3.35TB/s GPU-side)→PQ code scan+Top-k kernel分派到PNM（logic die专用hardware LUT/sorter vs GPU低利用率的shared memory LUT+warp-level sort）。GPU保留：QKV generation（GEMM，compute-bound→GPU tensor core高效）、FFN（GEMM）、Cluster selection（GEMV但仍适合GPU batch parallel）、LM Head。
    3. **PIM kernel执行（MHA Score为例）**：GPU issue PIM_WR_GB→global vector buffer加载Q[head i]的16个FP16 segment→PIM_ACT_AB→对应row激活，K^T[head i] column segment加载到PIM row buffer→PIM_MAC_AB→16 FP16 MAC units per bank并行：MAC[bank_j] = Σ(Q[k]×K^T[j][k]), k=0..15 → result register j = partial dot-product → across all banks produce 64 partial results → aggregate via global bus reduction tree → final Score vector。每token per head的32 elements Score向量需2次PIM_MAC_AB（32×16=512 FP16 MAC）。
    4. **PNM kernel执行（PQ code scan+Top-k）**：Cluster selection on GPU→对每个selected cluster：GPU DMA→PNM_WR_MMIO写query x和||x-y_c||²到Query Register，写precomputed LUT（256×64=16K entries）到PNM SRAM→PNM_RET_INIT：加载PQ codewords（1024B per 4 cycles per channel）→每64个1B PQ indices构成64B index vector→MAC module每个MAC unit读取1 index→PQ codebook lookup得到8-bit encoded sub-vector→decoder扩展为FP16→24 MAC units并行做x·y_R[sub] dot-product→累加所有M=64 sub-vectors的partial dot-products→加上LUT查表值||y_R||²+2(y_c·y_R)→加上Query Register中||x-y_c||²→最终距离d。16个PQ scanner每cycle产16个距离d[0:15]→Full sorter (Odd-Even Merge Sort, 16 elements parallel compare-swap)→已排序16个距离→Partial sorter (merge 16 sorted + 16 from Top-k register→32, Odd-Even→取最小16)→更新Top-k register。完成cluster后GPU读最终top-k=16 IDs。
    5. **性能输出**：(a) Kernel-level: PIM MHA Score/Context latency vs GPU baseline per token→MHA加速比=GPU_MHA_cycles / MNM_PIM_MHA_cycles；(b) PNM retrieval latency vs GPU FAISS IVF-PQ latency per cluster→retrieval加速比=GPU_retrieval_cycles / MNM_PNM_retrieval_cycles；(c) System-level: E2E speedup（GPU complete pipeline vs MNM complete pipeline, 含kernel offload overhead+GPU-MNM communication latency）→最高29.2×；(d) Energy: 各component（GPU Compute/On-chip Mem/Off-chip Mem/MNM PIM/PNM Logic/GPU-MNM Comm）energy→最高71.5% savings。Speedup来源：(i) PIM利用HBM bank内部~4TB/s带宽为MHA GEMV提供远高于GPU off-chip 3.35TB/s的有效带宽；(ii) PNM在logic die上用专用LUT+sorter pipeline替代GPU上低效率的shared-memory-LUT+warp-sort，每cycle处理64个PQ codes（GPU每SM warp仅32 threads且LUT工作集>shared memory capacity→多次global memory access）；(iii) Dual row buffer使PNM检索与PIM GEMV并发执行，隐藏PNM延迟。

## 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：AxCore将LLM推理中的GEMM操作实现为weight-stationary systolic array上的mpFPMA kernel。其kernel调度/数据流如下：
    **(1) Weight-stationary数据流**：低比特量化权重Wq（FP4/INT4）预加载至所有PE，因权重在所有token间共享而在列方向保持静止（pre-loaded & held stationary within PEs of each column）。高精度activation A（FP16/BF16/FP32）沿行方向逐列传播。中央PreAdd单元每条row计算共享中间值T=A-B1+C1一次并沿行广播（correction advancing），PE内仅需轻量整数加法R=T+Align(Wq)。
    **(2) 多格式并发支持**：SNC单元将不同FP4子格式（E3M0/E2M1/E1M2）统一转换为S1E3M2内部格式，使同一array可同时处理不同format group的权重块；FormatSel信号选择对应格式解码通路，normal值bypass直接输出。
    **(3) 流水线PE微架构**：Approx Mult（SNC→mantissa对齐bit-shift→low-bit integer adder产生R）→Guard Unit（检测zero operand并强制R=0）→Partial FP Adder（列方向累加R到partial sum Psum，保持un-normalized，mantissa宽度为NMa+2避免溢出）。
    **(4) 共享后处理kernel**：列累加完成后→Norm模块流水线（Abs→LZD→left shift→Round→Expo/Sub更新→组合sign/exponent/mantissa为标准FP16）→AxScale（FPMA-based dequantization: O=Oq+S-B+C2, 两次整数加法）→Accumulator（累加scaled partial sums到已存储值）。
    **(5) GEMM tiling**：64×64 systolic array with 4× tiling。矩阵乘法M×K×N → 沿M方向tile, K方向systolic传播, N方向weight preload。
  - 实验比较：(1) Compute density (TOPS/mm²) across 6 datatype configurations for PE array only; (2) Energy efficiency (TOPS/W) on OPT-13B/30B decoding phase (batch=32, output_len=1); (3) Energy breakdown by component (Compute/DRAM/Buffer/Other); (4) Area decomposition (PE array vs Others preprocessing/postprocessing); (5) SNR vs matrix size (128-32768) for numeric accuracy of kernel operations.

- 后端平台是什么，配置是什么。
  - 加速器硬件：28nm TSMC工艺，1GHz目标频率。64×64 systolic array，4× tiling。SpinalHDL→Synopsys Design Compiler→Verilog RTL。
  - Simulator: 基于DNNWeaver v2.0扩展的cycle-accurate simulator，SRAM功耗由CACTI 7.0建模。所有baseline designs配置为相同SRAM大小、相同peak throughput (TOPS)归一化以便公平比较。

- 评估性能的软件/脚本是什么。修改了什么。
  - **RTL功能验证**：Synopsys VCS仿真 (Hardware/AxCore/README.md)。
  - **LLM准确率评估**：PyTorch-based Software/AxCore框架（修改HuggingFace OPT/LLaMA2实现，插入FPMA-aware quantization+mpFPMA inference路径）+ lm-eval-harness zero-shot框架。Shell脚本自动执行→生成Table 2 (PPL)和Table 3 (zero-shot)数据。
  - **性能/能耗模拟**：Software/axcore_simulator（基于DNNWeaver v2.0修改）：新增mpFPMA PE延迟模型、SNC/PreAdd/Norm/AxScale/Accumulator流水线模型、weight-stationary systolic array数据流模型、多格式并发配置。运行脚本→results/fig_17.pdf (energy breakdown + TOPS/W)。
  - **GEMM操作占比分析**：Profile/ 目录包含profiling脚本计算OPT/LLaMA各sequence length下的GEMM vs Attention OP比例→Figure 2。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub (https://github.com/CLab-HKUST-GZ/micro58-axcore) + Zenodo DOI (https://doi.org/10.5281/zenodo.16895417)
  - 使用步骤：(1) `conda env create -f Software/AxCore/environment.yml` 创建PyTorch环境；(2) 运行对应shell脚本，自动从HuggingFace下载模型和数据集；(3) 脚本执行weight quantization (format selection + FPMA-aware quantization)→加载AxCore logic→对WikiText-2逐层推理计算PPL→zero-shot benchmark推理→输出accuracy。
  - Simulator评估原理与全过程（W4A16 GEMM kernel, OPT-13B decoding为例）：
    1. **输入配置**：`axcore_simulator` 读取硬件config（PE array=64×64, tile=4, SRAM sizes=Weight Buffer/Unified Buffer/Accum Buffer各KB数, DRAM BW, frequency=1GHz）、模型config（OPT-13B: 40 layers, hidden_dim=5120, FFN_dim=20480, num_heads=40）、workload config（batch=32, output_seq_len=1）。
    2. **GEMM tiling与数据流**：对每个Linear层 weight_matrix[K×N]: 沿N方向按64列分组（对应array columns），沿K方向按systolic propagation。Activation[32×K]: 每行(32 tokens)沿K方向broadcast至所有PE列。Weight预加载→activation流入→per row: PreAdd产生T→PE执行 SNC(Wq)→Align(Wq<<10-NM_FP4)→R=T+Align(Wq) (7b adder)→列累加(Psum+=R, unnormalized)→Norm→AxScale(O=Oq+s-B+C2)→Accumulator。
    3. **延迟估算**：每个systolic cycle完成一次activation传播和一次PE内加法。K维度需(K/1)个cycle per row（一次处理K的一列）。Norm/AxScale/Accumulator流水线各占固定stage数。总cycle = (K × systolic_propagation_stages) + row_parallel × pipeline_overhead。
    4. **面积计算**：从DC synthesis得出PE area (含SNC+adder+accumulator)、PreAdd area、Norm area、AxScale area、Buffer area (CACTI)→total area mm²。
    5. **能耗计算**：各模块activity factor × unit_energy_per_op (DC synthesis power report at 1GHz) × cycle count + SRAM read/write energy (CACTI) × access count + DRAM energy × data volume / DRAM bandwidth。Decoding phase batch=32、output_len=1使buffer energy主导（weight已在on-chip stationary）。
    6. **输出**：TOPS = (2×M×K×N / total_cycle_time) / 10^12; TOPS/mm² = TOPS / area; Energy = Σ(component energy) in nJ; TOPS/W = TOPS / (energy / cycle_time); normalized对比 baseline FPC-FP32。

## 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：FuseMax的kernel调度核心是通过**Cascade of Einsums**抽象形式化attention算法，分析不同算法的pass数（3-pass/2-pass/1-pass），并通过mapping和binding将最优1-pass cascade映射到spatial array架构上。具体包括：(1) **Pass分析（Section III）**：基于Einsum cascade的is-ﬁbertree依赖分析，形式化lower bound on pass count——给定任意mapping，若存在read-after-write dependency使得某tensor的整个fiber必须在后续Einsum读取前完成，则形成额外pass。3-pass cascade（PyTorch/TensorFlow/FLAT）需对每个M fiber做3次遍历；2-pass cascade（TileFlow/Choi et al.）先compute local max→partition→second pass用global max修正；1-pass cascade（FlashAttention-2）用running max替代local max，迭代式building numerator/denominator。(2) **Pass Reduction via Reassociation（Section III-C）**：两种方法——(a) Deferring multiplication：利用分配律将对Ak的reduction提前到乘以Y之前（减少1 pass + 减少乘法次数）；(b) Iterative construction：类似prefix-sum迭代构建中间RY和RZ（1-pass但增加compute）。(3) **FuseMax Mapping（Mapping 1）**：对M和P double tiling（M1×M0, P2×P1×P0），M0×P0 = #2D Array PEs，将所有Einsum在M0/P0层maximally fuse（除最后的AV除法在P2层fuse）。2D array同时执行tensor product（BQK）和softmax exponentiation（SLNV），1D array执行sum/max reduce。(4) **FuseMax Binding**：两级interleaving——Epoch间pipeline（software pipelining，不同epoch处理不同tile-relative坐标a<b<c<d的tiles，例如Epoch i完成BQK/LM的tile d，而RM tile c在更晚epoch）；Epoch内cycle-level interleaving（BQK|SLNV交错于2D array，SPNV|RNV交错于1D array，每cycle所有neighbor links active）。
  - 实验比较：(1) 1D/2D PE array利用率对比（Unfused Baseline vs FLAT vs +Cascade vs +Architecture vs +Binding，四种模型，seq len 256~1M）；(2) Speedup（FuseMax vs Unfused vs FLAT，attention only + full inference）；(3) Energy对比；(4) 2D array utilization breakdown by Einsum（Figure 7：FLAT vs +Cascade vs +Architecture vs +Binding，BERT各seq len）。

- 后端平台是什么，配置是什么。
  - 模拟的target hardware：TPUv2/TPUv3风格spatial array accelerator，45nm技术节点，940MHz频率。2D PE array维度与FLAT cloud accelerator匹配（论文未给出具体维度，但PARETO分析扫参16×16至512×512）。1D PE array: 256 PEs。Global buffer容量设为使总芯片面积与FLAT相等。DRAM off-chip。
  - 评估的模型：BERT-Base [17]、TrXL-wt103 [13]、T5-small encoder [49]、XLM [13]，batch size B=64。Sequence lengths: 256 to 1M tokens。
  - Timeloop + Accelergy software toolchain运行于x86-64 machine（artifact Docker环境，5GB磁盘空间，完整实验~9小时）。

- 评估性能的软件/脚本是什么。修改了什么。
  - **Timeloop [41]**（https://github.com/NVlabs/timeloop）+ **Accelergy [56]**（https://github.com/nelliewu95/accelergy）。
  - 修改：(1) 修正FLAT原作者代码中的bugs（含conceptual errors），创建并验证FLAT的Timeloop模型（误差<1%）；(2) 增补FLAT baseline的softmax建模（原FLAT忽略softmax的data transfer cost，默认230 PEs）；(3) 创建FuseMax三种配置的Timeloop模型，每个Einsum独立搜索mapping；(4) 实现full cascade性能组合heuristics（TeAAL[35]方法）；(5) FP division modeling: Xia et al. [59]设计缩放到45nm，via Accelergy。
  - 评估过程：对每个Einsum→Timeloop搜索最优mapping→输出cycles + activity counts→Accelergy计算energy→合并为full cascade（考虑pipeline overlap）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/FPSG-UIUC/micro24-fusemax-artifact（Docker + Jupyter notebooks）。
  - **评估原理与过程**：
    1. **输入：Cascade of Einsums规范**：FuseMax使用Cascade 5（1-pass attention cascade）定义kernel计算。例如BQK Einsum: `BQKm1,m0,p = Qe,p × BKe,m1,m0` ——对iteration space E×M0×P每个点执行multiply。Mapping 1定义loopnest：`for p2: for m1: for p1: parallel_for p0: parallel_for m0: ComputeRNVTile(...)`。每个Einsum的computational intensity（FLOPs/byte）由iteration space决定——BQK为E×M0×P0次MAC，读Q (E×P0) 和BK (E×M0) tiles。
    2. **Timeloop搜索**：对每个Einsum，Timeloop以minimize runtime为目标搜索：
       - Spatial mapping: M0×P0循环映射到2D PEs（output/weight/input stationary dataflow search）
       - Temporal tiling: 其余循环维度（E/M1/P1/P2/B/H）按buffer hierarchy分tile（DRAM→global buffer→PE buffer）
       - 约束：buffer capacity > tile footprint；NoC bandwidth足以在compute cycles内完成data movement
    3. **Binding层级组合**：
       - +Cascade: 使用1-pass cascade定义，但保持FLAT architecture（2D做tensor product，1D做softmax，无interleaving）
       - +Architecture: 添加FuseMax PE（支持exponentiation in 2D），但仍用简单的tile-by-tile binding（fully produce/consume one M0×P0 tile before next）
       - +Binding: 添加epoch级pipeline + cycle级interleaving——Timeloop中手动编码为loop order和spatial schedule
    4. **Performance输出**：各Einsum的execution cycles综合考虑：compute cycles、memory transfer cycles（受制于BW）、pipeline fill/drain overhead、tile dependency stall。Full cascade runtime = max(compute-bound time, memory-bound time)，考虑pipeline重叠后的effective time。Energy = Σ(PE MAC/exp/div/max/sum energy + buffer access energy + DRAM access energy + NoC energy)。
    5. **验证**：FuseMax的FLAT model与FLAT原作者修正后的代码误差<1%。

## 61-Duplex- A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Duplex提出两项runtime kernel调度技术来最大化xPU和Logic-PIM的利用率：(1) **Expert Co-processing**：在MoE层gate确定每token-to-expert assignment后，runtime统计每个expert处理的token数。由于不同expert处理不同数量的token（token distribution non-uniform），Duplex使用预存的lookup table（含不同token数下xPU和Logic-PIM的预估处理时间），在runtime快速决定哪些expert分配给xPU、哪些分配给Logic-PIM。算法从"all experts on xPU"开始，逐步将token数最少的expert移给Logic-PIM，寻找最优组合以最小化MoE layer总延迟。通过tensor parallelism for experts (Duplex+PE+ET)使每device处理更多expert（而非expert parallelism下每device仅处理Nex/Ndevice个expert），增强co-processing的调度粒度。(2) **Attention Co-processing**：在mixed stage中，利用attention operation per-request独立的特性，将prefilling sequence的attention kernel调度到xPU（高Op/B GEMM: Lin Q slices × KV matrices），将decoding sequences的attention kernel调度到Logic-PIM（低Op/B GEMV/narrow GEMM）。两个kernel在不同processing units上并行执行。Bank bundle-aware memory allocation确保xPU和Logic-PIM的memory access不冲突。(3) **Load Balancing via All-Reduce**：每个expert FFN的计算（FC1/GatedAct/FC2）分布到所有Logic-PIM stacks（每stack持有expert weight的不同列slice），在所有experts处理完成后由xPU执行一次all-reduce得到FC3(down-projection)的完整输入，最小化inter-Logic-PIM通信。
  - 实验比较：(1) Duplex vs Duplex+PE (expert+attention co-processing enabled) vs Duplex+PE+ET (tensor parallelism for experts) 的throughput消融分析；(2) Duplex vs Bank-PIM在不同模型（Mixtral with MoE+GQA, Llama3 with GQA only, OPT with MHA only）上的throughput对比；(3) Expert co-processing在不同Lin下的tail latency (p99 TBT/T2FT) 表现；(4) 不同QPS (4-16) 下的latency scalability测试。

- 后端平台是什么，配置是什么。
  - xPU: 等同NVIDIA H100 [35], 80GB HBM3 per device, 1GHz computing units。
  - Logic-PIM: 每stack含32 GEMM modules (512 FP16 MACs each, 650MHz), activation module, softmax module, 2×1MB buffers。4x HBM3 bandwidth, 8 Op/B peak (21.3 TFLOPS per stack)。
  - System interconnect: HGX NVLink 900GB/s bidirectional (≤8 devices), InfiniBand 400GB/s (inter-node)。
  - Default: Mixtral 4 devices/1 node, GLaM 8 devices/1 node, Grok1 16 devices/2 nodes。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：自研cycle-accurate simulator基于Ramulator 2.0 [12][33] (https://github.com/CMU-SAFARI/ramulator2)。修改：(1) DRAM controllers支持bank bundle模式的simultaneous bank reading；(2) 实现Logic-PIM compute modules的时序模型；(3) 实现serving scheduler的continuous batching调度逻辑；(4) 实现expert/attention co-processing的runtime调度算法；(5) 实现HGX/InfiniBand通信延迟和带宽模型。
  - 面积/功耗评估：Synopsys Design Compiler + ASAP7 7nm PDK [6] 综合算术单元（Verilog实现）；FinCACTI [48] 建模SRAM buffer energy；HBM energy参考[37]。
  - 评估原理：Simulator的serving scheduler按stage管理request，每stage分发任务到device components。Expert co-processing算法在gate后runtime查lookup table分配expert到xPU/Logic-PIM。Attention co-processing按request类型（prefilling/decoding）分配。各compute module的execution time = (computation cycles based on FLOPS) + (memory access time based on BW and burst size)。Device间通信时间按HGX/InfiniBand参数计算。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未明确说明Duplex simulator是否开源。基于Ramulator 2.0 (开源) 扩展。
  - **Expert Co-processing Kernel调度全过程**（以Mixtral on 4 Duplex devices, decoding-only stage, batch=64为例）：
    1. **Kernel输入**：MoE layer的gate output——64个decoding tokens各自选择2个expert（top-k=2）。Expert FFN weights（expert 0-7, 每个含FC1/FC2/FC3 matrices, FP16, intermediate dim 14336）。
    2. **Token Distribution统计**：统计每个expert的token分配——例如expert 0: 18 tokens, expert 1: 12, expert 2: 8, expert 3: 15, expert 4: 22, expert 5: 6, expert 6: 10, expert 7: 14。总计128 expert-token pairs。
    3. **Lookup Table查表**：预存的lookup table给出各expert在各处理单元上的预估时间。例如expert 5 (6 tokens): T_xPU = 180us (memory-bound, 6 tokens × D×D_interm weights loading), T_Logic-PIM = 90us (4x BW reduces memory time)。expert 4 (22 tokens): T_xPU = 350us (compute-bound for high token count), T_Logic-PIM = 580us (compute-bound earlier due to fewer MACs)。
    4. **Greedy Assignment**：初始T_total = max(T_xPU所有experts)。逐步将token最少的expert移给Logic-PIM：先移expert 5 (6 tokens)→Logic-PIM timeline增加90us, xPU timeline减少180us→T_total降低。继续移expert 2 (8 tokens)、expert 6 (10 tokens)、expert 1 (12 tokens)... 直到找到T_xPU和T_Logic-PIM最平衡的分配。
    5. **Kernel并行执行**：xPU执行assigned experts（如expert 0/3/4/7）→GEMM kernel on H100-equivalent SMs。Logic-PIM执行assigned experts（如expert 1/2/5/6）→GEMM kernel on 32 GEMM modules per stack, 512 MACs each。每个Logic-PIM GEMM module: 读weight chunk (D×D_interm/4 via column slice across 4 stacks) 和input tokens→MAC计算→activation (SiLU)→读FC2 weight→MAC计算→partial sum存入buffer。
    6. **All-Reduce**：所有expert的Logic-PIM results（partial sums）由xPU从4 Logic-PIM stacks读取→all-reduce→得FC3输入→xPU执行FC3(down-projection) GEMM。
    7. **性能输出**：MoE layer execution time (cycles)→contributes to TBT。Logic-PIM和xPU的utilization (active time / total MoE time)。Speedup over baseline (all on xPU or all on Logic-PIM)。
  - **Attention Co-processing Kernel调度**（mixed stage, batch=64 decoding + 1 new request with Lin=2048）：
    1. Prefill attention kernel（2048 tokens, Q: 2048×4096, K: 2048×4096 per head）→调度到xPU（高Op/B GEMM: 2048 Q slices share same K/V）。
    2. Decoding attention kernel（64 requests, Q: 64×1×4096, KV cache: 每request unique matrix, 每head independent）→调度到Logic-PIM（低Op/B GEMV/narrow GEMM for GQA）。
    3. 两kernel并行执行在不同processing units上→bank bundle-aware memory allocation确保无bank冲突。
    4. Softmax (decoding) 在Logic-PIM softmax module执行；Softmax (prefill) 在xPU执行。

## 65-MagiCache- A Virtual In-Cache Computing Engine.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MagiCache 的 virtual engine 实现了一套完整的运行时 vector register 调度与 cacheline 空间管理系统：(1) **Lazy Initialization（Algorithm 1）**：仅在 vector register 被指令实际使用时才分配 segment——检查 VRMT entry valid bit，若为 0 则在对应 fused array 中通过 FFA 找候选 cacheline → evict if dirty → set computing bit → 填充 VRMT entry。未使用的 register 和 segment 不占用任何 cache 空间。(2) **FFA Allocation Policy（Find-First-Available）**：从随机起始位置循环扫描 256 cachelines（每 cycle 检查 32 个连续的 tag state，最多 8 cycles 完成一个 array 的分配），优先选 free cacheline（valid=0），其次选 available cacheline（computing=0）。不引入额外 LRU 状态更新，miss rate 增加 <1%。设 minimum associativity threshold 防止某一 set 的可用 cacheline 被耗尽。(3) **Life Cycle Management**：通过标准 liveness analysis [28] 预提取 vector register 的生命周期，在生命周期结束时编译器插入 `vsetvli zero, zero` 指令触发 release——clear VRMT entries → clear computing bit → set LRU bits = least recently used 使 computing line 恢复为普通 cacheline。Release 指令 overhead <0.5%。(4) **Instruction Chaining**：运行时冲突检测（configuration/permutation/store 地址范围交叉三类）→ 无冲突连续指令组成 group → 各 fused array 独立异步执行 group 内所有指令，仅 group 间同步。同步次数减少 45.3%，memory access time 减少 2%-27%。(5) **Q/k Parameter Tuning**：通过 Q（每 register segment 数）和 k=Q/N 在计算并行度与 cache capacity 间 trade-off。最大 vector length = Q × W bits，最大 occupancy = 32×Q×W/(N×H×W)。k 从 1 到 4。(6) **OS Context Switch**：新增 CSR vreg_valid（32-bit）记录已初始化 register 子集，仅保存/恢复 valid register 保留 lazy initialization 优势。
  - 实验比较：(1) MagiCache (Fused-1/2/4, Chain-1/2/4) vs SplitCache (Split-8) speedup；(2) Execution breakdown 各阶段 cycle（Fig. 9）；(3) Instruction chaining sync time 减少（-45.3%）；(4) MSHR usage 对比（Table 7: Split-8 overall 5.59→Chain-4 8.23）；(5) Unit-stride vs strided 应用（k-means/backprop）性能差异；(6) Cross-element 指令（slide）对 chaining 的影响（Chain-1 性能略降 1%）。

- 后端平台是什么，配置是什么。
  - 模拟平台：gem5 cycle-approximate simulator。Processor: O3CPU out-of-order 8-issue 8-commit RV64GC, 192-entry ROB。L1I/L1D: 32KB 4-way 2-cycle-hit。L2 MagiCache: 512KB 8-way 1024-sets 8-cycle-hit 32 MSHRs，8 个 256×256 fused arrays（super array pairing 为 256×512）。LLC: 8MB 16-way。Memory: DDR4-2400。
  - Virtual Engine 硬件（TSMC 28nm @1GHz）：面积 26434 μm²，功耗 27.01 mW。各模块 breakdown: Instruction Queue 5970/4.84mW, CSRs 246/0.31mW, Request Generator 19279/19.51mW, VRMT Control Logic 939/2.35mW。
  - Vector ISA：RISC-V Vector Extension 1.0 全部 32-bit 整数指令。各指令 cycle 数由独立 C++ micro-code simulator 验证（Table 3: vadd 2 cycles, vmul 161-164, vdiv 360, etc）。
  - Benchmark：Rodinia [7]（vvadd/matmul/jacobi-2d/pathfinder/backprop）+ RiVEC [31]（k-means），32-bit integer RISC-V vector intrinsics，LLVM 17 + RISC-V GNU toolchain。Liveness analysis 预处理。

- 评估性能的软件/脚本是什么。修改了什么。
  - **gem5 修改**：(a) L2 tags 扩展 computing/presence bits；(b) 实现 virtual engine 全部模块（VRMT, instruction queue, CSRs, request generator, FFA 分配）；(c) 实现 micro-code execution framework（bit-line computation model）；(d) 实现 instruction chaining 冲突检测、group 管理、异步执行；(e) 实现 cache coherence（presence bit + snoop L1）和 fence；(f) 实现 register release 逻辑；(g) 新增 per-stage cycle counters。
  - **独立 C++ Micro-code Simulator**：验证各 fused array 的 micro-code program，提供 per-instruction cycle count lookup table 供 gem5 使用。
  - **Baseline SplitCache**：仿照 EVE [3]，L2 8-way 中 4-way 转为 computing arrays，rows 均匀划分给 32 vector registers。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未明确说明 MagiCache gem5 代码是否开源。底层 gem5 开源。
  - **MagiCache Vector Register 调度全过程**（matmul Chain-4, k=4, Q=32, N=8, VL=16384 bits=512 elements）：
    1. **输入**：RISC-V vector assembly——vsetvli → vle32.v v1 (load B行) → vle32.v v0 (load C行) → vmacc.vx v0, a5, v1 → vse32.v v0 (写回C)。LLVM 17 编译。
    2. **Lazy Initialization（首 iteration）**：vle32.v v1 首次执行 → VRMT[v1][0..31] 全部 invalid → 触发 Algorithm 1 对 32 segments 分配。Segment j→array j%8 → FFA circular scan 找 free/available cacheline → evict if dirty → set computing bit → VRMT[v1][j]=(valid=1, index=RowIndex)。各 array 并行分配 <8 cycles。
    3. **Vector Load**：Request generator 计算 512 element addresses → cache controller 按 cacheline 合并（32 cachelines）→ 查 L2 tags → hit: 8-cycle response → miss: allocate MSHR→LLC fill。32 MSHRs 容纳 1 batch。
    4. **Instruction Chaining**：四指令无冲突（无 config/perm，vle32.v v0 与 vse32.v v0 地址范围相同）→ 组为一个 group。Array 0 完成 v1 segment load 后立即开始 compute vmacc.vx（不等 Array 3），Array 3 若有 miss 则自动推迟 compute。各 array 时间线交错（Fig. 7(b)）。
    5. **Group Sync**：所有 arrays 完成 group 内全部指令 → sync → 下一 iteration。
    6. **Register Release（每 iteration 末尾）**：`vsetvli zero, zero` 触发 release → clear VRMT → clear computing bit → LRU=LRU → cachelines 恢复。
    7. **输出**：Per-stage execution cycles → speedup = Split-8 cycles / MagiCache cycles（Fig. 8, Table 6）。Cache utilization（Table 8）。MSHR usage（Table 7）。L2 miss rate（Fig. 10）。

## 68-SPARK_Scalable_and_Precision-Aware_Acceleration_of_Neural_Networks_via_Efficient_Encoding.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SPARK将变长编码的decoder/encoder集成到output-stationary systolic array中，实现mixed-precision PE（MPE）的variable-speed矩阵乘法执行。核心设计：(1) **Decoder**（MUX+OR+NOT gates实现）：每周期读入4-bit + enable信号；EN=0时若c0=0输出低精度值、若c0=1根据c3输出3或4-bit；EN=1时直接输出4-bit作为高精度后段。Decoder沿PE array边沿放置，m×n的array仅需m+n个decoder。(2) **Mixed-Precision PE（MPE）**：默认INT4模式（4-bit MAC），按需切换到INT8模式。INT4×INT8乘法分2周期（cycle t: 高4-bit×低4-bit <<4，cycle t+1: 低4-bit×高4-bit累加）。INT8×INT8乘法分4周期。(3) **Variable-speed dataflow**：低精度值全速（1 cycle/op），高精度值插入stall（2或4 cycles），PE间通过stall同步维持systolic array节奏。(4) **Encoder**：5-bit Leading Zero Detector（LZD）判断输入可否低精度编码——LZD=0时输出最后4-bit为short code，LZD=1时按规则生成8-bit long code（prev part + post part），post part由XOR(b0, b3)决定是否舍入。(5) 输出SPARK编码经encoder缩减bit-width后回写global buffer，减少下一层数据传输开销。
  - 实验比较：(1) 各架构normalized latency对比（Eyeriss/BitFusion/OLAccel/BiScaled/AdaFloat/ANT/Olive vs SPARK），6个网络上SPARK平均4.65× vs AdaFloat；(2) Normalized energy breakdown（DRAM+Global Buffer+Core），SPARK ResNet-50节电74.7% vs Eyeriss；(3) 面积breakdown：decoder 0.251% + encoder 0.261% + PE 99.49%；(4) 各架构iso-area PE数量与面积对比（28nm）；(5) 不同model size下energy efficiency趋势（scalability）；(6) SPARK+DBB pruning联合优化的执行周期减少。

- 后端平台是什么，配置是什么。
  - 自研cycle-accurate simulator模拟SPARK PE array。PE array规模4096个4-bit MPE（iso-area下可扩展）。频率200MHz，28nm TSMC工艺。Global buffer 5MB（CACTI估算）。Encoder/decoder带宽~50 GB/s（非阻塞，峰值需求~25 GB/s）。DeepScaleTool统一缩放至28nm进行iso-area对比。CACTI [1]评估内存结构。

- 评估性能的软件/脚本是什么。修改了什么。
  - Cycle-accurate simulator（自研，论文未说明开源）：模拟PE array的cycle-level执行——含decoder解码、MPE混合精度计算、variable-speed dataflow的stall编排、accumulation、encoder编码全流程。RTL实现在Verilog中，通过Synopsys Design Compiler + 28nm TSMC工艺库综合评估面积和功耗。
  - 模拟器修改/设计要点：(1) 将标准固定精度systolic array改为支持variable-length编码输入的架构，增加m+n个decoder在array边沿；(2) MPE支持INT4/INT8模式动态切换，含shifter+adder扩展；(3) Encoder集成在activation/pooling后，将INT8结果编码回SPARK格式。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：论文未明确说明代码开源。SPARK encoding在PyTorch中实现（software端），RTL用Verilog + Synopsys DC评估（hardware端），cycle-accurate simulator为自研。
  - 评估全过程：
    1. **输入准备**：从PyTorch Model Zoo加载pretrained模型（VGG16/ResNet18/ResNet50/ViT/BERT），执行layer-wise INT8量化。Software端执行SPARK encoding——对量化后的weight离线编码，activation的encoding模拟在hardware端。
    2. **SPARK编码（Offline）**：对每层weight张量逐元素执行SPARK编码（算法详见算法pipeline），将INT8值映射为4-bit或8-bit SPARK code，统计各层short code占比和有效bit-width。
    3. **Accuracy验证（PyTorch）**：将SPARK-encoded weight加载回模型，forward pass模拟SPARK decoding后的值参与计算。记录accuracy vs original FP32 baseline。对activation的SPARK encoding做online模拟。
    4. **Cycle-accurate模拟**：将SPARK-encoded weight和activation加载到simulator。Simulator模拟systolic array执行：
       - ① Global buffer读取SPARK-encoded数据（4-bit对齐存储）；
       - ② Decoder沿array边沿每周期解码4-bit输入+enable信号，输出8-bit operation；
       - ③ MPE array执行混合精度MAC：低精度值1 cycle完成，高精度值根据配对情况2或4 cycles，PE间通过stall维持同步；
       - ④ Partial sums在accumulation unit累加；
       - ⑤ 输出经activation/pooling后由encoder重新编码为SPARK格式；
       - ⑥ 编码结果写回global buffer。
       Simulator记录每层的execution cycles、PE利用率、stall cycles。
    5. **面积/功耗评估**：Verilog RTL用Synopsys DC综合（28nm TSMC），提取decoder（6.42 μm²）、encoder、PE（79.57 μm²）的面积和功耗。Global buffer用CACTI [1]建模。所有baseline用DeepScaleTool缩放到28nm进行iso-area比较。
    6. **性能输出**：total execution cycles（normalized latency）、energy breakdown（DRAM/Global Buffer/Core）、area breakdown（decoder/encoder/PE）、energy efficiency（TOPS/W或normalized energy）、speedup vs baselines。

## 69-Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MikPoly的运行时micro-kernel polymerization调度机制：(1) **Offline micro-kernel生成与性能建模**：从两阶段模板的offline loops提取micro-kernel template K_hat（3层tiled loops: for ui.0 in uTM.0/for uj.0 in uTN.0/for uk.0 in uTK.0 → for ui.1 in uTM.1/for uj.1 in uTN.1/for uk.1 in uTK.1 → C[...]+=A[...]*B[...]），使用TVM auto-scheduler对给定platform H生成一组固定尺寸micro-kernels S_K~，每个micro-kernel尺寸为(uM=uTM.0×uTM.1, uN=uTN.0×uTN.1, uK=uTK.0×uTK.1)，专为M_local优化（例如GPU上uM=256,uN=128,uK=32）。为每个micro-kernel K~在单PE上运行t∈[1,5120]次reduction loop的pipelined task实验，学习分段线性performance model g_predict(t,K~,H)，预测含t个K~实例的pipelined task在单PE上的执行开销。(2) **Runtime polymerization**：当operator shape (M,N,K) 已知时，MikPoly用polymerization patterns将online loops重组为多个region R_i（每个R_i内包含一个parameterized micro-kernel），通过polymerization strategy从S_K~中选fixed-size micro-kernels实例化各R_i的micro-kernel。Cost model: Cost(S,H)=Σ f_wave(R_i,K~_i,H) × f_pipe(R_i,K~_i,H)，其中f_pipe用g_predict预测单PE pipelined task开销；f_wave=ceil(f_parallel/|P_multi|)估算wave数。GPU上为减少runtime overhead仅使用Pattern I和II（9种pattern中），利用hardware scheduler动态分配thread blocks到SMs；NPU上使用全部9种pattern并通过max-min static allocation分配micro-kernels到DaVinci Cores。(3) **Pipelined task执行模型**：每个micro-kernel在reduction loop中的执行采用pipelining技术分三阶段——(a) 从M_global加载数据到M_local，(b) PE使用micro-kernel在M_local上计算，(c) 结果从M_local写回M_global。中间结果存M_local减少memory traffic。优化目标: arg min Cost(S,H)。(4) **Search pruning**：在遍历polymerization strategies时，若某(R_i,K~_i)的预估cost已超过当前最优strategy的cost，跳过相关strategies，大幅缩减搜索空间（runtime overhead ~2μs vs oracle穷举1.6s）。
  - 实验比较：(a) MikPoly vs MikPoly-Oracle（穷举搜索最优）：MikPoly达到Oracle的96%性能，search time仅~2μs vs ~1.6s。(b) MikPoly-Wave（仅优化wave数→大micro-kernel）：speedup为Oracle的0.81x；MikPoly-Pipe（仅优化pipelined task→小micro-kernel）：speedup为Oracle的0.72x；CUTLASS（无cost model指导）：0.45x。(c) GEMM在GPU上MikPoly vs cuBLAS avg 1.47x (max 4.82x)，vs CUTLASS avg 3.02x。(d) Convolution在GPU上MikPoly vs cuDNN avg 1.98x (max 5.38x)。(e) load-imbalance case study：单micro-kernel GEMM-A在M=4096时sm_efficiency从86.67%降至58.90%（因thread blocks从96增至128导致wave数从1变2，最后一波underutilize GPU），而MikPoly的multi-micro-kernel GEMM-AB通过polymerize两个micro-kernels（A处理M=3072，B处理M=1024）将整体wave数调至3，sm_efficiency恢复到96.06%。

- 后端平台是什么，配置是什么。
  - GPU平台：Nvidia A100 (80GB, Ampere Tensor Cores, 108 SMs, 64 max warps per SM)，CPU Intel Xeon Gold 6348, 256GB Host, Ubuntu 18.04, CUDA 11.5 + cuBLAS + cuDNN。
  - NPU平台：Ascend 910 (32GB, Da Vinci Cores + Cube Unit), CPU Kunpeng 920, 128GB Host, EulerOS 2.8, CANN SDK v5.1.RC1。
  - LLM平台：4×A100 with NVLink, tensor parallelism=4。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估使用PyTorch v1.11 (GPU CNN)、TurboTransformers (GPU language models)、MindSpore v1.7 (NPU)。将标准GEMM/Convolution operator替换为MikPoly生成的tensor programs。Micro-kernel生成使用TVM auto-scheduler + CUTLASS v2.9 templates（GPU）或manual templates（NPU）。
  - 修改：MikPoly在TVM auto-scheduler基础上添加：(a) 两阶段模板中offline loops→micro-kernel template提取；(b) 单PE上的pipelined task performance model学习（g_predict函数）；(c) RankAndPrune筛选（Top-40 micro-kernels）；(d) Runtime Polymerization组件（pattern匹配+strategy搜索+cost model估计+code instantiation）；(e) GPU上动态分配（利用hardware scheduler）、NPU上max-min static allocation。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - MikPoly未开源。以GPU GEMM为例，kernel输入到性能输出的全过程：(1) **Micro-kernel binary**：offline生成的每个micro-kernel（如uM=256,uN=128,uK=32）编译为CUDA binary，tensor地址和loop iteration counts为runtime参数。Micro-kernel内部执行tiled GEMM pipeline：加载A/B tile到shared memory → warp-level matrix multiply-accumulate on Tensor Cores → 结果存register→写回global memory。(2) **Runtime shape输入**：GEMM shape (M,N,K)=(4096,1024,4096)在runtime已知→MikPoly的Runtime Polymerization组件开始执行。(3) **Pattern exploration**（GPU仅Pattern I/II）：Pattern I保留单一micro-kernel、Pattern II分割为两region（R1处理M的top部分，R2处理M的bottom部分）→对每个pattern/region组合遍历S_K~中micro-kernels→用g_predict和f_wave估算cost→启发式剪枝跳过劣解→选cost最低的组合。(4) **Code instantiation**：将选中micro-kernels的参数（地址偏移、迭代次数）通过scalar assignments写为完整kernel launch参数，dispatch到GPU。(5) **GPU执行**：CUDA runtime将thread blocks分配到108个SMs，每个SM上每个warp执行micro-kernel的pipelined GEMM：从global memory加载A/B tile→Tensor Core FP16 GEMM→结果accumulate→写回global memory。多wave的thread blocks按硬件scheduler依次调度。(6) **性能测量**：用PyTorch/TurboTransformers wrapper调用MikPoly operator，warmup后平均20次运行测量end-to-end latency，与cuBLAS/cuDNN同类测量对比得出speedup。Nvidia profiling tools（nsight）收集sm_efficiency、elapsed_cycles_sm、grid_size等指标分析load imbalance。

## 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MCBP设计了bit-grained计算dataflow和runtime kernel调度——(1) **Bit-slice GEMM dataflow**：将INT8 GEMM分解为8个1-bit bit-slice矩阵的shift-and-accumulate操作。Weight矩阵沿bit维度分解后重组为以group size m=4的Group Matrix，在PE cluster内进行bit-slice并行计算。每个PE cluster分配T_M=64 × T_K=256 weight tile和T_K=256 × T_N=32 activation tile，8个PE并行处理同一weight tile的不同bit-slice。(2) **CAM-based fast match**：512B Content Addressable Memory在单周期内识别Group Matrix中的重复列向量。4-bit search key分高2位和低2位分别查询，AND生成bitmap标记匹配位置，实现O(1)重复检测。CAM时钟门控：search key=4'b0000时关闭。(3) **Index Converter + Addition Merge Unit (AMU)**：16个index converter将bitmap翻译为activation索引→AMU从Group Sum Buffer (GSB)读取psum，与fetched activation相加后写回。一个Reconstruction Unit (RU)采用固定数据路径时分复用16个AMU。(4) **BSTC在线编解码**：BSTC decoder含1-bit CMP、5-bit SIPO (m=4)、leading one eliminator——'0'检测输出4个0，否则缓存至SIPO满后输出。Encoder含4-bit CMP和MUX——非零加1'b1前缀输出，零输出1'b0。Segmented data layout支持并行解码：weight沿H维分割为sub-weights存储于独立bank，起始地址记录于address area。(5) **BGPP runtime filter**：16个bit-serial inner product units (64-input AND-based adder tree)逐bit计算partial QK→Progressive Filter中Threshold Updating模块找Max/Min→Clipping模块与threshold比较生成binary mask→仅mask=1的Keys索引参与下一轮bit加载和计算。Sign Decision Unit (SDU)处理SM格式的符号位。
  - 实验比较：(1) BRCR/BSTC/BGPP在MCBP accelerator上的ablation latency reduction (Dolly & MBPP, 1k/4k prompt/decoding)；(2) latency breakdown: compute vs memory access vs bit shift overhead (baseline value-level vs MCBP bit-level)；(3) 软件增益（GPU直接部署算法）vs 硬件增益（ASIC），分解为BRCR(1.2×→2.88×), BSTC(1.44×→2.19×), BGPP(1.23×→1.48×)；(4) 与A100 GPU throughput/energy efficiency对比 (batch 8/128, 26 benchmarks)；(5) SOTA加速器对比(FuseKNA, Bitwave, Spatten, FACT, SOFA, Energon)的prefill throughput+energy和decoding throughput+energy；(6) 与INT4加速器Cambricon-C的prefill/decoding对比 (Dolly, Llama7B/13B, Bloom1B7)。

- 后端平台是什么，配置是什么。
  - MCBP ASIC：TSMC 28nm, 1GHz, 20 PE Clusters (160 PEs)。每PE cluster含512B CAM单元、16 index converters、16 Add Merge Units、1 Reconstruction Unit。BSTC: 20×4 decoders, 10×4 encoders。BGPP: 64个64输入AND-based adder tree、4个Clock-gated Progressive Filters。On-chip SRAM 1248KB (768KB Weight + 384KB Token + 96KB Temp)。HBM2: 8×128-bit @2GHz, 8GB, 512-bit/cycle带宽, 4 pJ/bit。
  - GPU平台：NVIDIA A100 80GB, TensorRT-LLM, INT8 compute 624 TOPS。MCBP scaled to 148 processors = 622 TOPS@INT8 for iso-peak-compute对比。

- 评估性能的软件/脚本是什么。修改了什么。
  - RTL验证：Verilator (开源，https://github.com/verilator/verilator) 仿真RTL提取各stage cycle count。Custom cycle-level simulator评估end-to-end性能，集成CACTI 6.0 (SRAM), Ramulator (HBM), Cadence Virtuoso (CAM cell级设计)。GPU profiling: cudaEvent计时, nvprof排除非计算阶段, nvidia-smi测量功耗, 每实验运行2k次去掉top/bottom 15%取平均。
  - 修改：(1) Verilator仿真的RTL为MCBP自研，包括CAM-based BRCR unit、BSTC CODEC、BGPP unit的SystemVerilog实现。(2) Custom cycle-level simulator建模MCBP的8-step pipeline workflow（BSTC decode→CAM match→Index Convert→Merge→Reconstruct→Accumulate→Quantize→Writeback），BGPP与BRCR并发执行。模拟HBM row activation和burst访问。(3) GPU对比用TensorRT-LLM直接运行INT8模型，MCBP算法在GPU上通过PyTorch custom kernel实现（无专用硬件加速）测量software gain。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - MCBP RTL和simulator未明确说明开源。以Verilator+Custom Simulator评估Llama7B Dolly任务为例，kernel输入到性能输出的全过程：
    1. **模型准备**：HuggingFace Llama7B PyTorch checkpoint → BSTC offline encoding：8-bit weights分解为8个BS matrices→3rd-7th BS矩阵按m=4列向量进行two-state coding→low-SR的1st/2nd/8th矩阵无压缩→写入offline compressed weight file。
    2. **输入**：prompt tokens (Dolly, S=1k-8k)、compressed BS weights decoded为8个BS矩阵、decoding长度48 tokens。Config: m=4, batch=8, INT8 GEMM, α_r=0.5-0.6。
    3. **Verilator仿真**：SystemVerilog RTL作为输入，Verilator编译为C++ cycle-accurate model→仿真的信号级行为包括：BSTC decoder的SIPO填充和leading one消除→CAM的4-bit search key并行匹配→Index Converter地址翻译→AMU的GSB读写→RU的固定数据路径adder reordering。输出per-stage cycle counts。
    4. **Custom Simulator**：Per-stage cycles + DRAM latency (Ramulator) → 总latency计算。8-step pipeline: (1) Data Fetcher根据token indices和BS weight physical address从HBM加载到SRAM；(2) BSTC Decoder并行解压sub-weights；(3) CAM-based BRCR Unit识别重复列向量，返回indices；(4) Fetcher按indices取activation；(5) AMU merge重复activation→GSB；(6) RU重建为GEMV结果；(7) Quantizer施加scale+bias；(8) 结果写回HBM。BGPP并发执行：QK预测→vital KV indices→仅这些KVs参与attention computation。
    5. **性能输出**：(a) Total latency → throughput (tokens/s)；(b) Energy: Core energy (PE switching activity from Verilator + Synopsys DC power report) + SRAM energy (CACTI, read/write counts) + HBM energy (Ramulator + 4pJ/bit IO)；(c) Area: Synopsys DC综合 (总9.52mm², 28nm)；(d) Utilization: 活跃cycles/总cycles (平均78%)。
    6. **GPU软件增益测量**：同样的BRCR/BSTC/BGPP算法逻辑在PyTorch custom CUDA kernel中实现→cudaEvent测时→nvprof profile→nvidia-smi测功耗→与MCBP ASIC对比计算hardware gain。

## 77-Make_LLM_Inference_Affordable_to_Everyone_Augmenting_GPU_Memory_with_NDP-DIMM.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Hermes在NDP-DIMM侧实现了GEMV kernel（处理cold neurons的FC层计算）和attention kernel，与GPU侧的GEMM kernel协同完成LLM推理。具体kernel级实现：（1）**NDP-DIMM GEMV Kernel**：每个DIMM的GEMV unit含256个multipliers，每个multiplier负责128-bit bit-serial乘法，可同时计算8个FP16值。数据流：DRAM cell→center buffer→multiplier array（读取weight参数）→reduction tree-based accumulator（partial sum累加）→256KB buffer。Activation input来自GPU（hot neuron结果通过PCIe传输）或上一层DIMM输出。bit-serial设计使256个multiplier达到约hundreds of GFLOPS总量（论文引用类似设计[6][14][26][68]为hundreds of GFLOPS）。（2）**NDP-DIMM Activation Kernel**：Activation unit含256 FP16 exponentiation units + 256 FP16 addition units + 256 FP16 multiplication units + comparator tree + adder tree + divider，用于LLM中的softmax和ReLU等非线性操作。（3）**GPU GEMM Kernel**：hot neurons在RTX 4090 Tensor Cores上执行标准GEMM，用NVIDIA Nsight Compute [40] profiling。（4）**Merge Kernel**：在NDP-DIMM侧汇总GPU传来的hot neuron部分结果和本地cold neuron GEMV结果。由于GPU结果仅几KB，hidden在DIMM计算延迟中不增加额外开销。（5）**Inter-DIMM Data Movement**：通过DIMM-link（25GB/s per link, bidirectional）实现neuron remap的数据传输。相比host CPU中转，DIMM-link提供62× speedup（以OPT-66B为例，migration overhead从5.3%降到<0.2%）。（6）**Kernel Pipeline Flow**：每层执行顺序——QKV generation (GPU hot GEMV ∥ DIMM cold GEMV) → merge in DIMM → attention in DIMM (QK^T + softmax + ×V) → projection in GPU (DIMM idle期间做neuron remap) → MLP (GPU hot GEMV ∥ DIMM cold GEMV) → merge in DIMM。GPU和DIMM间通过host CPU插入同步barrier。
  - 实验比较：（1）Performance breakdown：Hermes vs Deja Vu vs Hermes-base的FC operator、Attention operator、Predictor、Prefill、Communication各部分延迟分解（batch sizes 1-16），显示Deja Vu通信占89%，Hermes predictor <0.1%；（2）Ablation: Hermes-random vs partition vs token-adjustment vs layer-adjustment vs adjustment vs Hermes（含remapping），逐步证明各kernel调度模块贡献；（3）Sensitivity: DIMM数量（1-16）、GPU型号（T4/3090/4090）、GEMV unit multiplier数量（32-512）、batch size（1-16）；（4）Hermes vs TensorRT-LLM（5×A100）: LLAMA2-70B bs=1达79.1%, bs=16仍24.4%。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA RTX 4090 24GB GDDR6, 330 Tensor TOPS (FP16), 936 GB/s带宽
  - NDP-DIMM: 8× DDR4-3200, 32GB/DIMM, 每个DIMM含1个NDP core (GEMV: 256 multipliers @1GHz + Activation unit)，DIMM-link: 25Gb/s/Lane × 8 lanes = 25GB/s per link
  - PCIe 4.0 ×16: 64GB/s

- 评估性能的软件/脚本是什么。修改了什么。
  - GPU kernel profiling: NVIDIA Nsight Compute [40]
  - NDP-DIMM simulator: 基于Ramulator 2.0 [35][48]修改的in-house simulator，复现NDP-DIMM内存访问和计算行为——修改Ramulator 2.0以支持center buffer-based NDP设计、GEMV unit的bit-serial乘法建模、DIMM-link通信延迟和带宽建模、activation unit功能仿真
  - RTL synthesis: Synopsys Design Compiler [56], TSMC 7nm工艺（验证GEMV unit面积1.23mm² per core，功耗和时序）
  - 端到端评估: 自研Hermes runtime（未开源），Python/C++混合实现

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - Ramulator 2.0 开源: https://github.com/CMU-SAFARI/ramulator2。Hermes修改版simulator未开源。
  - 评估原理和过程（以评估LLaMA2-70B单层MLP的cold neuron GEMV kernel为例）：
    1. **输入**: LLM weight matrix的一列cold neuron（如MLP FC1层的W[:, j], j为cold neuron index），对应activation input vector x（维度H），mapping table指示该neuron在哪个DIMM。
    2. **DRAM模拟（Ramulator 2.0修改版）**: 建立DIMM timing模型（DDR4-3200: tRC=76, tRCD=24, tCL=24, tRP=24, tBL=4, tCCD_S=4, tCCD_L=8, tRRD_S=4, tRRD_L=6, tFAW=26）。模拟DRAM row activation→column read→data to center buffer的全过程。Center buffer带宽由DIMM内部data bus宽度和频率决定。
    3. **GEMV计算模拟**: 模拟256 multipliers的bit-serial FP16乘法：每个multiplier从center buffer读取weight（128-bit），与activation input（FP16）逐bit相乘累加。Reduction tree accumulator合并256个partial products。运行时记录：multiplier利用率、计算周期数、pipeline stall（等待center buffer数据）。
    4. **DIMM-link通信模拟**: 若触发neuron remap（window-based scheduling），模拟DIMM-link 25GB/s数据搬移：source DIMM读DRAM→DIMM-link controller→bidirectional external link→target DIMM bridge→写DRAM。记录migration数据量和延迟。
    5. **性能输出**: 给定batch_size=B, input activation sparsity rate=s（如80% cold neurons中仅部分激活），输出：DIMM侧GEMV cycles、GPU-GEMM cycles（Nsight Compute实测）、merge kernel cycles、总层延迟 T = max(T_GPU, T_DIMM) + T_merge。端到端tokens/s = 序列长度 / 总时间。
  - GPU kernel使用NVIDIA Nsight Compute [40] profiling：launch GEMM kernel (cudaLaunchKernel) → 记录SM占用率、Tensor Core利用率、内存带宽利用率 → 与模拟的DIMM延迟合并计算end-to-end。

## 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：BitMoD提出**统一bit-serial处理单元(PE)**，在同一data-path上支持INT8、INT6、FP4-EA/ER、FP3-EA/ER多种低精度权重与FP16激活的混合精度矩阵乘法。核心kernel设计：(1) **统一bit-serial表示**：每个数字分解为一系列bit-serial terms，每term包含4个部分——1-bit sign、2-bit exponent、1-bit mantissa、shared bit-significance(bsig)。INT8/INT6使用Booth编码将二进制串分解为4/3个Booth strings（每string 3-bit→生成sign/exp/man/bsig）；扩展FP4/FP3先转为fix-point(sign-magnitude)→比较并替换冗余负零为special value→Leading-One Detector(LOD)生成2个bit-serial terms。(2) **混合精度bit-serial PE**：每cycle执行4-way dot product between 4个bit-serial weight terms和4个FP16 activations。四步pipeline：❶Exponent Alignment（对齐weight和activation指数计算delta exponent δe）→❷Bit-Serial Multiplication（1-bit weight mantissa × 11-bit activation mantissa + hidden bit，右移对齐δe，3-bit reserve for round-to-nearest-even）→❸Group Accumulation（adder tree→乘以weight bsig→累加到accumulator mantissa→normalize更新eACC）→❹Bit-serial Dequantization（accumulator mantissa逐bit乘以per-group INT8 scaling factor Δ的每一位→shift-and-add →输出dequantized partial sum的eGRP和mGRP）。由于INT6含3个bit-serial terms（3 cycles）、FP4/FP3含2个terms（2 cycles），BitMoD PE比FP16 MAC实现1.33×（INT6）和2×（FP4/FP3）吞吐量提升。(3) **bit-serial dequantization**：per-group scaling factor为8-bit，需8 cycles dequantization；而最低精度FP3的group dot-product（G=128, PE dot-product size=4）需128/4×2=64 cycles→dequantization从不stall计算pipeline。(4) **self-attention支持**：key和value tensors quantize到INT8/INT4（因softmax归一化使KV高度可量化），由同一bit-serial PE处理。
  - 实验比较：(1) BitMoD PE vs FP16 baseline PE在TSMC 28nm下的Tile面积(99,509 vs 95,498 µm²)和功耗(39.36 vs 36.96 mW)——BitMoD PE小24%；(2) 不同精度下的speedup over baseline FP16 accelerator：lossless(INT6)和lossy(4-bit/3-bit)配置；(3) 能量消耗分解(DRAM+Buffer+Core)；(4) Perplexity-EDP Pareto图 for Phi-2B和Llama-2-7B；(5) 与bit-parallel混合精度PE(FIGNA-like)的面积/功耗对比。

- 后端平台是什么，配置是什么。
  - Custom ASIC accelerator (TSMC 28nm): RTL设计用SystemVerilog实现，Synopsys Design Compiler综合@1GHz。512KB activation buffer + 512KB weight buffer (CACTI建模)。DRAM: DDR4 model from DRAMSim3。
  - 对比baseline: FP16 accelerator (同样TSMC 28nm, iso-compute area)。ANT加速器，OliVe加速器。

- 评估性能的软件/脚本是什么。修改了什么。
  - Custom cycle-level simulator：基于RTL综合参数(area, power, timing)建模端到端LLM推理性能。模拟PE array(4×4 tiles, 每tile 8×8 PEs, output-stationary dataflow)执行LLM各层(batch_size=1, input seq_len=256)的cycle-accurate时延。
  - GPU量化实现：PyTorch实现BitMoD量化框架，从HuggingFace加载预训练模型。量化Llama-2-7B仅需~10秒 on A6000。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源: https://github.com/yc2367/BitMoD-HPCA-25, artifacts: DOI: 10.5281/zenodo.14252531
  - Kernel评估全过程（以FP3 weight, G=128, PE dot-product size=4, 8×8 PE tile为例）：
    1. **输入**: FP16 activation A ∈ R^{8×4} (PE tile一行8个PE×4 activations/PE); 4-bit-serial weight terms per PE，每term含(ws, we, wm, wbsig); per-group INT8 scaling factor Δ。Bit-serial Term Generator从weight buffer接收原始权重→解码为bit-serial terms。
    2. **PE pipeline Step ❶ - Exponent Alignment**: 对4对(activation, weight term), 计算ae[j]=exponent(A[j]) (5-bit), we[j]=weight exponent (2-bit)。MAX unit选择最大(ae[j]+we[j])→对齐各对的指数差δe[j]=(max_exp - ae[j] - we[j])。同时生成符号ys[j]=sign(A[j]) XOR ws[j]。
    3. **PE pipeline Step ❷ - Bit-Serial Multiplication**: 4个activation mantissas am[j] (11-bit含hidden bit) × weight mantissas wm[j] (1-bit)→通过AND gate实现(1-bit乘法)→4个11-bit结果右移δe[j]位→adder tree求和得到bit-serial dot product (14-bit, 3 extra bits for rounding)。
    4. **PE pipeline Step ❸ - Group Accumulation**: dot product (14-bit)左移wbsig位→与accumulator mantissa mACC相加(通过adder)→normalize：检测leading one位置更新eACC和归一化后的mACC。
    5. **PE pipeline ❹ - Bit-serial Dequantization**(每group完成后触发): accumulator mantissa mACC逐bit乘以Δ[i] (i=0..7)→shift-and-add累积→normalize→输出(mGRP, eGRP)为FP16格式的dequantized partial sum。
    6. **PE Column Accumulation**: 各PE的dequantized partial sums通过PE column accumulator累加→得到最终per-channel output activation。
    7. **Self-attention处理**: Key和Value tensors各自quantize到INT4/INT8→送入bit-serial PE(INT格式, Booth encoding)→与FP16 Query进行QK^T→softmax→×V。
    8. **性能输出**: Total cycles per layer = max(bit-serial multiply cycles, dequantization cycles) × systolic pipeline depth。Throughput (ops/cycle) = PE数量 × 4 ops/PE / cycles_per_op (FP3/FP4: 2 cycles → 2 ops/cycle/PE; INT6: 3 cycles → 1.33 ops/cycle/PE)。End-to-end speedup = T_baseline_FP16 / T_BitMoD。


## 84-MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现分为两个层面：(A) GPU CUDA kernel优化—register caching和动态GEMM kernel调度处理分布式离群值；(B) 专用加速器上的ReCoN NoC—多级蝶形网络对离群值partial sum进行重排序和FP处理，实现multi-precision INT PE阵列上的离群值计算。实验比较：(GPU) TensorRT-LLM FP16 baseline、W4A4 Atom kernel；(加速器) OliVe、GOBO、OLAccel、AdaptivFloat的latency、throughput、energy和compute density。

- 后端平台是什么，配置是什么。
  GPU: NVIDIA A100 (real GPU + GPGPU-Sim/AccelSim模拟), NVIDIA H100 (量化算法执行)。加速器: 自研Verilog RTL→TSMC 7nm synthesis→1GHz，64×64 PE array，HBM2 off-chip (256 GB/s)，2MB L2 global SRAM，OCP-SRAM interface (64 GB/s)。对比baseline时使用iso-bandwidth (off-chip: 2 TB/s, on-chip: 1.5kb/cycle/warp)和iso-compute (55,296 multipliers)。

- 评估性能的软件/脚本是什么。修改了什么。
  (A) GPU kernel：基于CUDA实现优化的MicroScopiQ kernel，PyTorch前端。扩展GPGPU-Sim和AccelSim以支持修改后的tensor core (添加variable right shifter支持INT+FP co-issue)。Energy估算是AccelWattch/GPUWattch。修改Tensor Core：添加variable右移器 (Inliers: >> 0, Outlier Upper: >> 1, Outlier Lower: >> 2) 处理FP mantissa。
  (B) 加速器：基于DnnWeaver和BitFusion构建cycle-accurate simulator。修改PE支持multi-precision (2/4-bit)，添加同步缓冲器和ReCoN NoC。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  算法代码开源：MicroScopiQ-LLM-Quantization.git（无完整URL）。加速器和GPU simulator修改未明确说明是否开源。

  **GPU Kernel执行流程** (以W4A4 MicroScopiQ GEMM为例)：
  1. **Weight Loading**: 每个thread block负责计算T_m × T_n输出tile。权重从global memory加载到shared memory（按off-chip layout组织：weights + metadata按MaB存储，包含I_sf、per-μB的MXScale和permutation list）。
  2. **Register Caching (Register-Level Outlier Distribution)**: 使用CUDA `shfl_sync(m, r, t)` 原语进行warp内部寄存器通信。B_μ=8时每个warp(32 threads)包含4个μB。基于permutation list，warp内部线程交换register值完成离群值Upper/Lower halves的合并。
  3. **Mixed-Tile GEMM Dynamic Dispatch**: 
     - Iter 0 (混合tile): 包含离群值的tile→先执行register-level outlier merging→MX-FP离群值dequantize到FP16→Tensor Core FP16 GEMM→FP32 accum。
     - Iter K-1 (纯inlier tile): 无离群值→直接使用INT Tensor Core执行MX-INT-4 GEMM→INT32 dequantize到FP16→累加到FP32。
     每次迭代沿K维度执行block-level dynamic decision选择GEMM路径。
  4. **Modified Tensor Core (HW)**: 每个tensor core执行16-bit FEDP (four-element dot products)，4-bit时16 EDPs。添加variable right shifter支持离群值FP mantissa处理(Inliers >>0, Upper >>1, Lower >>2)。Overhead ∼0.1%。
  5. **性能输出**: Token generation throughput (tokens/s), normalized latency vs FP16 TensorRT-LLM baseline。

  **加速器执行流程** (以64×64 systolic array为例)：
  1. **Weight Stationary Mapping**: μB weights映射到PE行。每行PE接收高精度(4-bit)权重或打包低精度(2×2-bit)权重。
  2. **iAct流入**: PE row 0接收8-bit INT iAct从左侧流入，接收iAcc从上方流入。PE执行INT乘法(weight × iAct)，与iAcc累加。
  3. **Multi-Precision PE (MODE signal)**:
     - MODE_2b: Res = {(P_11 << 2 + P_10), (P_01 << 2 + P_00)} → 两个独立2-bit乘积累加并行输出
     - MODE_4b: Res = (P_11 << 4 + P_00) + (P_01 << 2 + P_10 << 2) → 单4-bit乘积累加
     ADD stage: MODE_2b时两个加法器独立并行运算；MODE_4b时通过multiplexer传播carry联合运算。
  4. **Outlier Detection & ReCoN Routing**: PE row中若μB含有离群值，controller通过OAcc_NoC/PE信号将partial sum引导至ReCoN。
  5. **ReCoN Processing** (多级蝶形NoC, time-multiplexed across PE rows):
     - Synchronization Buffer: 补偿systolic array skewed data flow，同步各列partial sum到达时间。
     - ReCoN Switch操作 (3-bit配置):
       - Pass(=): 直通inlier partial sum到下一级
       - Swap(×): 交换left/right输入端口→将Lower Half outlier重定向到Upper Half对应列
       - Merge(||): 接收O_Upper,Res和O_Lower,Res→分离Res和iAcc→将两个halves的mantissa bit右移(Upper>>1, Lower>>2)→相加→加上hidden-bit贡献(iAct)→输出完整FP-outlier partial sum
     - 每级ReCoN同时接收与PE相同的iAct，用于hidden-bit处理。
  6. **Output Interface**: 重排序后的partial sum送入下一PE行或oAct buffer。
  7. **Post-Processing**: oAct scale factor = O_sf + iAct_sf (power-of-two加法)。纯inlier计算的oAct通过in/out信号右移补偿scale差异。oAct缩放(右移)→MX-INT-4/8_{128}量化→送external memory或回iAct buffer供下一层计算。
  8. **性能输出**: Total cycles per layer → normalized latency/speedup；Compute density (TOPS/mm²)；Energy (static + dynamic)；Power breakdown (PE array/on-chip memory/ReCoN)。


## 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：**AQPIM的PQ-based attention kernel**——在HBM-PIM上运行的新型attention计算kernel，将传统GEMV操作转换为codebook查表+求和操作。核心kernel组件：
    (1) **Distance Calculation (DC) kernel**：在BankPE上执行，计算KV子向量与centroids的欧氏距离（ADD+MUL+SUM），所有bank并行计算。
    (2) **Cluster Assignment (CA) kernel**：在BufferPE上执行，对所有bank的distance结果取MIN确定centroid assignment。
    (3) **Centroid Calculation (CC) kernel**：分子（加权向量和）在BankPE计算（MUL+SUM），分母倒数在BufferPE计算（SUM→DIV取倒数），最终除法退化为BankPE上一次乘法。
    (4) **Attention (ATNK) kernel**：在BankPE上执行，query子向量×key codebook子矩阵乘（MUL+SUM），生成inner product matrix。
    (5) **Softmax (SFM) kernel**：在BufferPE上执行（ADD+SUM+MAX+DIV+EXP），处理后返回BankPE。
    (6) **Attention (ATNV) kernel**：在BankPE上执行，attention scores × value codebook，使用intra-row indirection查表。
    (7) **专用PIM命令**：PIM_SET_CONFIG（广播PQ配置参数）、PIM_MAC_AB（所有bank的MAC操作）、PIM_SFM（BufferPE softmax）、PIM_RET（intra-row indirection检索）、PIM_MV_BA（BankPE→BufferPE数据移动）、PIM_MV_BF（BufferPE→BankPE数据移动）、PIM_RD/PIM_WR（I/O）、PIM_ACT_AB（所有bank row激活）。
    (8) **数据映射策略**：Head-wise HBM映射（每个attention head分配独立HBM，消除inter-HBM传输）；Subvector-wise bank映射（每个subvector分配独立bank，最大化BankPE利用率）。
    (9) **Sequence-by-sequence pipelining**：GPU逐sequence生成qkv并offload到PIM后立即处理下一个sequence，隐藏GPU-PIM串行延迟。
    (10) **Memory allocation**：codebook区域固定、prefilling buffer固定、PQ indices按layer/page粒度分配，prefilling后buffer区域被解码阶段indices覆盖复用。
  - 实验比较：(1) Total execution time (Fig.11) 架构对比 GPU+HBMs vs AttAcc! vs AQPIM，算法对比 PQCache/SKVQ/SnapKV/AQPIM on GPU，输入4K、输出128~1024 tokens；(2) Per-decoding-step latency (Fig.12) 在不同sequence length (4096/8192/16384/32768) 下对比；(3) Speedup breakdown (Fig.13)：gpu+cpu vs gpu∞ vs gpu+pq vs AQPIM（batch=32）；(4) Intra-row indirection cycles (Table V)：On BankPE vs On BufferPE for Keys和Values（seq=4K）；(5) Accuracy & Speedup vs Memory Reduction (Fig.15)：Best-case (TREC) 和 Worst-case (LCC) 的accuracy-speedup-memory tradeoff。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA H100 GPU（1 core, HBM 3.35TB/s bandwidth）
  - PIM：4×16GB HBM-PIM modules（HBM3架构），BankPE在每个DRAM bank旁（1KB row buffer），BufferPE在buffer die
  - 额外HBMs：1×16GB conventional HBM（模型参数存储）
  - CPU：Intel Xeon Platinum 8480+ Processor
  - PCIe bandwidth：256GB/s（GPU↔CPU）

- 评估性能的软件/脚本是什么。修改了什么。
  - Simulator: Customized GPU-PIM simulator for LLM inference, built upon AttAcc! simulator [55]（https://github.com/scalesnu/attacc_simulator），后者是Ramulator [20][36][48]的修改版
  - 修改：(1) 在AttAcc! simulator基础上新增PQ codebook generation的DC/CA/CC pipeline建模；(2) 新增PQ-based ATNK/ATNV attention kernel的cycle-accurate模拟；(3) 新增intra-row indirection机制的DRAM时序建模（tCCDL不受影响，column decoder input被latch用于pipeline）；(4) 新增PIM命令集（PIM_SET_CONFIG等）；(5) DRAM traffic和contention由DRAM controllers管理并在simulator中建模
  - 面积/功耗评估：Intra-row indirection逻辑用ASAP7 PDK [9]综合，与AttAcc!相同的DRAM die density ratio缩放。BankPE/BufferPE已有计算单元（ADD/MUL/SUM/DIV/EXP等）沿用AttAcc!设计

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：Simulator https://github.com/scalesnu/attacc_simulator（AttAcc! simulator，论文在此之上修改）；AQPIM-specific修改未见独立开源仓库
  - **评估原理与全过程**：
    1. **输入**：(a) System configuration: GPU type (H100), HBM count/PIM count, HBM capacity, PCIe bandwidth, BankPE/BufferPE count & specs；(b) Model details: architecture (Mistral-7B), layer config (d_model=4096, n_heads=32, n_kv_heads=8 for GQA), context length；(c) Input config: batch size (default 16), input/output token lengths, PQ params (m=32 subvectors, K=512 centroids)
    2. **Prefilling simulation**：GPU generates QKV→latency按GPU计算能力model；KV offloading→HBM write bandwidth model；PIM codebook generation→DC kernel cycles（BankPE并行计算distance）+ CA kernel cycles（BufferPE argmin）+ CC kernel cycles（4 iterations，每iteration内BankPE+BufferPE协同）；GPU attention/projection/FFN→与PIM并行执行，取max
    3. **Decoding simulation (per token)**：GPU generates qkv→GPU compute model；qkv offload to PIM→HBM write；PQ-based attention kernel：(i) ATNK: query×codebook → BankPE cycles（每个subvector的矩阵乘）；(ii) MV_BA: results→BufferPE；(iii) SFM: softmax → BufferPE cycles（EXP/DIV/SUM）；(iv) MV_BF: softmax results→BankPE；(v) ATNV: intra-row indirection lookup+accumulation → BankPE cycles（包含1次row activation + column decoder streaming）；(vi) 输出回传GPU→HBM read；GPU projection/FFN→与PIM串行执行（decode阶段GPU idling通过sequence-by-sequence pipelining缓解）
    4. **时序模型**：DRAM时序基于HBM3标准（tRC, tRCD, tCCD等），BankPE和BufferPE cycle counts基于AttAcc! synthesis结果。Intra-row indirection: 1次row activation + column decoder sequential output stream，tCCDL unaffected
    5. **能耗模型**：各PE operation energy来自AttAcc! synthesis报告（ASAP7 PDK），DRAM access energy基于HBM3 datasheet
    6. **输出**：(a) Total execution time (ms) 分解为 d_attn/d_pq/d_fc/d_comm/d_etc；(b) Per-decoding-step latency (ms)；(c) Energy per decoding step (normalized) 分解为 g_attn/g_fc/g_comm/g_etc；(d) KV cache memory footprint (GB)

## 8-SambaNova_SN40L_Scaling_the_AI_Memory_Wall_with_Dataflow_and_Composition_of_Experts.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SN40L streaming dataflow runtime通过以下机制实现kernel调度和运行时计算：
    - **Spatial Kernel Fusion**：将20+ operators融合为单一kernel，以coarse-grained pipeline在PCUs/PMUs上流式执行。Fused kernel包含mixed systolic GEMM (PCU systolic array mode) + SIMD element-wise ops (PCU SIMD mode) + transpose (PMU diagonal striped format) + reductions (PCU cross-lane reduction)。Pipeline stages通过PMU双缓冲decoupling，RDN vector fabric传输tensor tiles。
    - **Hardware-Orchestrated Kernel Launch (HO)**：AGCU硬件实现kernel schedule的自主执行（Program Load→Argument Load→Kernel Execute序列），消除host software scheduling overhead。对decode阶段（kernel执行时间极短，dominated by weight loading）实现1.4×-8× speedup。
    - **Software-Orchestrated Kernel Launch (SO)**：Host CPU通过PCIe触发每个kernel的AGCU command序列。灵活但overhead大，适用于prefill/training（kernel执行时间长，overhead amortized，仅~1.1× speedup from HO）。
    - **Streaming P2P Collective Communication**：P2P protocol使collective communication operators (AllReduce等) 被fused和pipelined到同一kernel中，避免数据经过HBM hops。
    - **Dynamic Memory Swapping (CoE Runtime)**：运行时dynamic memory manager在DDR中分配每个compiled model binary所需空间。当expert被请求时，CoE runtime "activates" it by copying HBM-intended memory segments from DDR→HBM，然后transfer control到compiled kernel。LRU eviction管理HBM中的active experts（编译器annotated read-only weights跳过copy back to DDR）。
  - 实验比较：(1) Fused+SO vs Fused+HO kernel launch speedup——decode benchmarks 1.4×-8×, prefill/training ~1.0×-1.1×；(2) Fused vs Unfused kernel launch count ratio——Llama7B-prefill 11× reduction, FlashFFTConv ~125× reduction, sparseGPT ~100× reduction；(3) Operator fusion speedup vs unfused baseline——FlashFFTConv 13×, Mistral decode 13×, Llama7B decode ~8×, Llama70B decode ~2×；(4) Llama3.1 8B/70B/405B token generation speed on 16 SN40L sockets；(5) CoE model switching time (DDR→HBM) vs DGX；(6) >85% HBM bandwidth utilization during decode vs GPU <50%。

- 后端平台是什么，配置是什么。
  - SN40L RDU：TSMC 5nm, 638 BF16 TFLOPS, 1040 PCUs + 1040 PMUs, 520MB SRAM, 64GB HBM (1.8TB/s), 1.5TB DDR (200GB/s), <2GHz clock。实验使用8-socket SN40L Node（FlashFFTConv使用单socket）。Host: x86 CPU, PCIe connection。
  - 对比平台：NVIDIA DGX A100 (8×A100 80GB, 32GB/s host-to-GPU) 和 DGX H100 (8×H100 80GB, 64GB/s host-to-GPU)。DGX对比数据为基于已发布延迟和规格的估计。

- 评估性能的软件/脚本是什么。修改了什么。
  - Benchmarks（Table III）：Llama2-7B/70B [20], Bloom-176B [51], Mistral-7B [21], Falcon-40B [52], Llava1.5-7B [53], sparseGPT-13B [38] (87.5% sparse training), FlashFFTConv [40] (FFT Convolution, 1M sequence)。所有benchmarks在三种配置下测量：Unfused (per-op kernel, results materialized to DDR/HBM, SO)、Fused+SO、Fused+HO。
  - Samba-CoE部署：150个Llama2-7B experts + 1 router (均源自Llama2-7B)，~1T total parameters，deployed on single 8-socket SN40L Node。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - SN40L runtime为闭源商业软件。Benchmarks使用的模型均为开源（Llama2, Mistral, Bloom, Falcon, Llava1.5, sparseGPT, FlashFFTConv）。
  - **Kernel执行与性能评估全流程（以Llama2-7B decode on 8-socket SN40L为例）**：
    1. **Kernel Input准备**：SN40L compiler将Llama2-7B decoder layer编译为fused streaming dataflow kernel binary。Weights从DDR pre-loaded到HBM（via CoE runtime activation）。KV cache常驻HBM（autoregressive loop复用）。
    2. **AGCU Kernel Launch (HO mode)**：AGCU autonomous执行kernel schedule——Program Load (loaded once, cached) → Argument Load (input token embedding addresses, weight buffer addresses, KV cache addresses) → Kernel Execute trigger。
    3. **Streaming Dataflow Pipeline Execution**：Q/K/V projection GEMMs → PCU systolic array模式，weights从HBM stream进PMU buffers → broadcast buffer分发 → systolic MAC → 结果stream到下游PMU stage buffers。Attention score computation → PCU SIMD模式执行softmax原语。Attention output × V projection → PCU systolic array。FFN (FC1→activation→FC2) → mixed PCU pipeline (systolic + SIMD)。Layer norm / residual add → PCU SIMD模式 (FP32 accumulation)。所有stages通过PMU双缓冲pipeline并行，数据在RDN vector fabric上credit-based flow control传输。Transpose通过PMU diagonal striped format实现全带宽读写。
    4. **Multi-Socket P2P**：TP=8 partitioning下，每个socket计算partial results → P2P protocol streaming AllReduce across 8 sockets（不经HBM）→ 完整output tensor回HBM。
    5. **Performance Measurement**：Wall-clock time测量（从kernel launch到completion）→ token generation throughput (tokens/sec/user) = batch_size × tokens_generated / total_time。HBM bandwidth utilization = actual_bytes_read / (peak_bandwidth × time)。Kernel launch count ratio = unfused_kernel_count / fused_kernel_count。Speedup = unfused_latency / fused_latency。
    6. **Benchmark评估原理**：每个benchmark在三种配置下运行固定workload（如decode 20 tokens from 4K context），测量end-to-end latency。Unfused baseline：每个PyTorch op编译为独立kernel，intermediate results materialize to DDR/HBM，SO scheduling。Fused+SO：compiler自动fusion，但仍由host software scheduling。Fused+HO：同一fused kernel，AGCU hardware autonomous scheduling。Speedup = Unfused / Fused (Figure 10)。

## 90-AUM- Unleasing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving.pdf

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：AUM Runtime AU Controller——系统级运行时资源调度器，管理AU-enabled CPU核心的kernel执行资源分配。核心机制：(1) 按AU使用强度将CPU核心划分为三个频率区域（High-AU/AMX密集@低频、Low-AU/AVX轻量@中频、None-AU/无AU@全turbo频），基于TDP限制下的强制频率降低规律；(2) 通过Intel RDT（Resource Director Technology）的CAT（Cache Allocation Technology）和MBA（Memory Bandwidth Allocation）动态调整LLC way分配和内存带宽上限，控制AMX/AVX kernel执行时的后端资源bound；(3) 用LAG指标（Σ(dTPOT-e_token)）量化decode阶段各request相对于SLO的进度偏差，实时判定AU kernel需要更多或更少资源；(4) Allocation Tuner监控AU kernel实际性能P^m，若达标则aggressive harvest资源（使用P^a平均性能指导），超标则conservative返回（使用P^t tail性能指导），δ_AU超threshold时触发Core Switcher重新分核心区。
  - 实验比较：(1) CPU效率对比：AUM vs ALL-AU/SMT-AU/RP-AU across Compute/OLAP/SPECjbb共享负载；(2) 分解性能对比：AUM vs AU-UP/AU-FI/AU-RB variants across three shared workloads；(3) 不同硬件平台的效率gain：GenA/GenB/GenC with SPECjbb；(4) SLO保证率对比：AUM vs ALL-AU/SMT-AU/RP-AU on TTFT prefll + TPOT decode；(5) 资源分配CDF：LLC way和Memory BW分配的累积分布。

- 后端平台是什么，配置是什么。
  - GenA: Intel Xeon 8475B (Sapphire Rapids), 48 cores/2 sockets, AMX 206.4 TFLOPS (BF16, 1024 ops/cycle TMUL, 8×1KB TILECFG registers), AVX-512 25.6 TFLOPS, 2.7 GHz base, L1-I 32KB/L1-D 48KB/L2 2MB per core, LLC 97.5MB/socket, DDR5 1TB (233.8 GB/s)
  - GenB: Intel Xeon Max 9468 (Sapphire Rapids+HBM), 48 cores/2 sockets, AMX 206.4 TFLOPS, 2.1 GHz base, LLC 105MB, HBM 128GB (588 GB/s)
  - GenC: Intel Xeon 6982P-C (Granite Rapids), 120 cores/1 socket, AMX 344 TFLOPS, 2.8 GHz base, L1-I 64KB/L1-D 48KB/L2 2MB, LLC 504MB, MCR 768GB (600 GB/s)
  - 每物理核心含1个AMX单元（TMUL accelerator: 1024 BF16 ops/cycle, 8×2D TILECFG registers@1KB each）和AVX-512单元，AU资源不跨hyperthread共享

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估工具：Linux perf [19]（PMU event采集: tma_amx_busy, tma_fp_amx, tma_fp_arith, avx_insts）、Intel pmu-tools [38]（top-down methodology分析前端/后端bound）、Intel pqos [26]（CAT/MBA资源分区控制）、turbostat [45]（核心频率记录）
  - 修改：(1) xFasterTransformer集成AU usage profiling hooks，记录各LLM serving阶段的AMX cycle ratio/µop ratio；(2) AUM Runtime Controller作为Python daemon新增，监控AU kernel SLO并调用pqos调整CAT/MBA配置；(3) Background Profiler新增离线profiling脚本，自动化3 division × 3 sharing × 5 resource config × 10 repetitions的450次AU执行测量

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：论文未明确说明开源链接。AUM基于开源xFasterTransformer [33]和Intel RDT工具链 [26]构建。
  - **评估原理与全过程**（以GenA SPR, llama2-7b decode with SPECjbb sharing为例）：
    1. **AU Kernel性能采集（perf + pmu-tools）**：Linux perf采集PMU硬件计数器：(a) tma_amx_busy → AMX忙碌周期占比，decoder阶段~1.5%；(b) tma_fp_amx / tma_fp_arith → AMX完成的浮点操作占比，decoder 0.5%/1.5%；(c) avx_insts → AVX指令计数，decoder更高因小矩阵GEMV更适合AVX；(d) top-down methodology四级分解（Retiring/Bad Speculation/Frontend Bound/Backend Bound → Core Bound/Memory Bound → Port Utilization/Serializing Operations/Divider/L1/L2/LLC/DRAM Bound）识别AU执行瓶颈。
    2. **频率干扰测量（turbostat）**：turbostat记录每核心实时频率 → 对比仅AU核心跑decode (3.1GHz) vs AU核心+power stressor共享 (2.54GHz) → 量化frequency interference程度 → 输入AUV Model的FAU维度。
    3. **资源分区控制（pqos/CAT/MBA）**：pqos配置LLC CAT（按way数分配，如给AU应用12 ways给共享应用4 ways）→ 测量AU应用TPOT变化（<5% degradation for decode）→ MBA限制内存带宽（如AU 60% / share 40%）→ 测量带宽分配对AU性能影响 → 确定RAU最小资源需求。
    4. **AUM Runtime Decision Loop**：每控制迭代(~1ms, 单核执行)：(a) 采集token生成latency和SLO → 计算LAG_i = Σ(dTPOT - e_token) → SLOL = dTPOT + LAG_i → (b) 查AUV Model表获取当前division/resource下的P^a和P^t → (c) P^m < SLOL → aggressive harvest: R_AU ← M(P^a_H, P^a_L)，优先回收LLC way（AU应用affinity低）→ (d) P^m > SLOL → conservative return: R_AU ← M(P^t_H, P^t_L) → 若δ_AU > 2则调整核心division。
    5. **Performance Output**：CPU efficiency = (1.8×P_H + 0.2×P_L + 3e-5×P_N) / W_CPU → 输出Perf-per-Watt提升4.7% over SOTA sharing baseline (RP-AU)，8.8% over exclusive ALL-AU → SLO violation减少7-11% → runtime overhead <1ms查表延迟。