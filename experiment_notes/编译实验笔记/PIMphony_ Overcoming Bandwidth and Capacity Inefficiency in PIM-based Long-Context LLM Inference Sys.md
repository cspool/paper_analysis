## PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

- 属于编译框架的实现是什么？实验比较什么？
  实现基于MLIR的PIMphony compiler和IREE runtime stack，将长上下文LLM decoding的Transformer subgraph编译为PIM-specific instruction sequences，并嵌入token-centric partitioning和dynamic memory allocation metadata。核心编译实现：(1) MLIR Dialect扩展：扩展MLIR dialect以表达PIM-amenable kernels（QK^T、SV、FFN），通过custom pattern-matching识别Transformer decoder pattern中的attention和feed-forward子图，生成PIM-specific code。(2) Code Generation Passes：针对PIM primitive（WR-INP/MAC/RD-OUT）生成优化命令序列，embed TCP token partitioning metadata（token segment range per channel）和DPA dynamic memory metadata（Dyn-Loop bound/Dyn-Modi stride/VA2PA mapping hints），支持DCS所需的entry-level dependency annotations。(3) IREE Runtime HAL：基于IREE runtime stack和Hardware Abstraction Layer扩展，对接commercial PIM SDK，在multi-node PIM部署中自适应响应context length变化。编译离线完成，不计入inference latency。实验比较CENT baseline和NeuPIMs baseline（两者均无PIMphony compiler优化），GPU baseline为A100 with flash-decoding+paged-attention，评估end-to-end throughput、TP/PP组合优化、capacity utilization。

- 硬件平台是什么，配置是什么。
  PIM后端：CENT (PIM-only, 16GB/module, 16TB/s internal BW, 32 PIM channels) 和 NeuPIMs (xPU+PIM, 32GB/module, 32TB/s internal BW, 32 PIM channels)。PIM channel配置16-bank commercial PIM module。GPU baseline: NVIDIA A100-80GB。模型: LLM-7B (32 layers, 32 heads, dh=128), LLM-72B (80 layers, 64 heads, dh=128)。LLM-7B-32K/128K和LLM-72B-32K/128K，涵盖non-GQA和GQA (group size 2/4/8)变体。系统容量: 7B 128GB, 72B 512GB。

- 开源编译框架是什么。修改了什么。
  基于MLIR [64] 和 IREE [67] runtime stack。未修改MLIR/IREE core，在其上新增：(a) Custom MLIR dialect extension for PIM operations——定义PIM-specific op和type表示WR-INP、MAC、RD-OUT primitive及GBuf/OBuf resource；(b) Pattern-matching passes——识别Transformer decoder中的QK^T、SV、FFN子图，标记为PIM-amenable kernels；(c) PIM code generation——生成PIM instruction sequences，嵌入TCP token partitioning（per-channel token segment assignment）、DCS dependency metadata（entry-level hazard annotations for D-Table/S-Table population）和DPA dynamic addressing（Dyn-Loop/Dyn-Modi instruction encoding）；(d) IREE HAL extension——对接commercial PIM SDK，运行时根据当前请求token length和VA2PA table状态adaptively dispatch instructions。Compiler不依赖新硬件primitive，确保与CENT和NeuPIMs simulator兼容。

- 开源情况。编译框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未开源。PIMphony compiler使用流程：
  1. 输入：Transformer decoder model graph（如LLaMA架构的decoder layer，含Q/K/V projection、attention QK^T/SV、output projection、FFN）→MLIR frontend将模型lowering到MLIR dialect。
  2. Pattern-matching pass：遍历MLIR IR→识别attention子图（Q×K^T matmul + softmax + score×V matmul）和FFN子图→标记为PIM-amenable kernels。非PIM-amenable kernels（如LayerNorm、embedding lookup）留在host/xPU执行。
  3. TCP partitioning：对QK^T和SV operator，compiler根据模型config（num_heads, head_dim, PIM module/channel count）计算token-centric partition scheme——将token sequence切成nChannel段，每段分配给一个PIM channel。生成per-channel instruction sequences时嵌入token segment range metadata。
  4. DCS-aware code generation：对每个PIM primitive sequence，compiler分析GBuf/OBuf entry-level data dependency→生成dependency annotations（which command writes/reads which GBuf/OBuf entry）→供runtime DCS controller填充D-Table/S-Table。
  5. DPA code generation：compiler生成Dyn-Loop指令（loop bound = runtime token length而非Tmax）和Dyn-Modi指令（按stride修改row/col operand）→嵌入VA2PA table index reference→runtime dispatcher在decode时查询VA2PA table完成地址翻译。
  6. IREE runtime部署：compilation产物（PIM instruction sequences + metadata）通过IREE HAL提交到commercial PIM SDK→PIM SDK根据当前系统状态（request batch、token lengths、VA2PA mappings）dispatch到target PIM modules。Compilation离线完成，每model编译一次。
  Compiler的主要作用：将高层次Transformer graph自动映射到PIM-native primitive sequences，自动生成TCP partition scheme、DCS dependency annotations和DPA dynamic addressing metadata，免去手写PIM指令序列的工作量，并保证与multi-node PIM系统的兼容性。

