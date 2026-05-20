## PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出PIMphony orchestrator，包含三项协同的PIM runtime kernel调度技术：(1) Token-Centric PIM Partitioning（TCP）：将Attention的QK^T和SV并行维度从head/batch转为token维度，在单个PIM module内沿token维度切分，每个channel处理一段token的Key/Value cache，与同一query做部分dot-product，结果在module内通过PIM HUB/GPR做inter-channel reduction（不跨module，避免跨module同步开销）；(2) Dynamic PIM Command Scheduling（DCS）：在PIM controller中增加I/O-aware buffering（multi-entry GBuf+expanded dual-port OBuf）和dependency-aware scheduling（Dependency Table D-Table记录每个GBuf/OBuf entry最近访问命令，Status Table S-Table记录命令ID/完成时间/OBuf is-MAC flag），WR-INP、MAC、RD-OUT命令在真实依赖满足时乱序/提前发射，重叠数据搬运和计算；(3) Dynamic PIM Access（DPA）：引入Dyn-Loop（loop bound来自请求当前token length而非编译期最大值）和Dyn-Modi（loop内按stride修改row/col operand field）两类动态PIM指令，on-module dispatcher（含instruction buffer、configuration buffer、VA2PA table）在PIM HUB内做运行时VA-to-PA翻译，实现KV cache按1MB chunk lazy allocation。三项技术分别解决长上下文PIM的三个低效：channel underutilization、I/O bottleneck、静态KV cache容量浪费。实验比较PIM-only CENT baseline和xPU+PIM NeuPIMs baseline，以及GPU baseline (A100-80GB with flash-decoding + paged-attention)，在LLM-7B/72B、context 32K-1M、LongBench和LV-Eval benchmark上评估吞吐、延迟、MAC utilization、能耗、容量利用率。

- 后端平台是什么，配置是什么。
  PIM后端：(1) CENT（PIM-only）：每module 16GB、16TB/s internal BW、PNM (3 TFLOPS)、32 PIM channels。7B使用8 modules (128GB)，72B使用32 modules (512GB)。(2) NeuPIMs（xPU+PIM heterogeneous）：每module 32GB、32TB/s internal BW、8 Matrix Units (256 TFLOPS)、32 PIM channels。7B使用4 modules (128GB)，72B使用16 modules (512GB)。(3) GPU baseline：NVIDIA A100-80GB，7B使用2张、72B使用8张（内存容量匹配PIMphony配置）。PIM channel配置：16-channel、16-bank commercial PIM module。建模使用validated Ramulator-based cycle-accurate simulator，结合AiMX PIM specification校准DRAM command timing和resource contention。

- 评估性能的软件/脚本是什么。修改了什么。
  评估使用Ramulator-based cycle-accurate simulator，集成AiMX架构参数。修改：(1) 在CENT和NeuPIMs simulator中集成PIMphony的on-module dispatch logic、I/O buffering、DCS dependency table/status table、expanded output buffer；(2) DRAM command timing和resource contention按AiMX PIM specification校准。MLIR compiler/runtime生成PIM instruction sequences，编译离线完成不计入inference latency。硬件overhead通过CACTI估计：OBuf为MAC unit area的0.47%/bank；DCS control blocks带来0.5% area和1.3% power增加；on-module dispatcher内部buffer <200KB、4% area overhead。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未开源。PIMphony kernel调度执行流程（以一次长上下文decode step的Attention为例）：
  1. DPA dispatcher：新请求进入→host初始化request ID、当前token index Tcur、VA2PA table→dispatcher在decode instruction时将virtual row/col映射到已分配物理chunk。KV cache增长超过当前1MB chunk→host分配新chunk并更新VA2PA。
  2. TCP执行：QK^T时，query被广播/写入GBuf，每个channel读取自己负责的Key cache token段计算部分score→各channel score segment在PIM HUB/EPU拼接并进入Softmax。SV时，score segment与对应Value cache段在各channel做partial context→module内reduction得完整context vector。论文指出16-channel/16-bank配置下，QK^T token length>256、SV token length>32即可full channel activation。
  3. DCS执行（以FP16 GEMV为例）：compiler生成WR-INP W0/W1/W2写入GBuf 0/1/2→MAC M3/M4/M5读取GBuf entry和DRAM row/col累加到OBuf→RD-OUT R6读出output。静态scheduler按固定顺序等待所有命令；DCS中M3到达时从D-Table查到只依赖GBuf 0的W0→S-Table显示W0完成后立即发射M3，不等W2。M7与R6不冲突时先于R6发射。论文示例从34 cycles缩短到22 cycles。
  4. GQA row-reuse下DCS：GQA多query heads共享K/V→优先在当前open DRAM row上处理所有共享query→减少ACT/PRE overhead→但增加WR-INP压力。DCS利用dual-port GBuf/OBuf在MAC消费当前entry时预取下一批query/score，或在MAC写OBuf其他entry时读出已完成结果。ping-pong buffering baseline因hand-off pipeline stalls，DCS up to 1.4× higher compute-unit utilization。
  5. 整体加速效果：PIM-only最高11.3× speedup (CENT baseline)、xPU+PIM最高8.4× speedup (NeuPIMs baseline)。Context 1M tokens时CENT baseline退化到2% utilization，PIMphony达46.6× speedup。DPA将capacity utilization从静态31.0%-40.5%提升到75.6%。

