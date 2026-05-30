## Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

- baseline方法是什么？
  **Baseline方法有两类：编译器方法（Triton、Ladder）和手写kernel（QuantLLM、Marlin）。**

  **Triton [53]**（tile-oriented compiler）：提供tile级编程模型但缺乏低精度原生支持。用户需手动从uint32中通过bitwise操作解包sub-byte数据。Triton不暴露GPU内存层次（registers vs shared memory由编译器隐式管理），导致低精度kernel的weight loading pipeline存在关键瓶颈（图1a）：① cp.async异步global→shared拷贝 → ② shared→registers加载 → ③ 解包和casting → ④ **shared memory上的layout conversion**（将register tensor layout转换为Tensor Core指令要求格式）——Step 4是主要瓶颈。Triton的编程模型抽象掉tensor layout，使得通过改变global memory layout避免此瓶颈的优化不可行。

  **Ladder [58]**（schedule-oriented compiler）：扩展TVM调度系统，引入低精度原语将低精度数据（如4-bit ints）pack至更大类型（如8-bit ints）。但有两个关键缺陷：（1）type-level packing限制——只能处理power-of-two位宽，无法高效支持3/5/6/7 bit等非标准位宽；（2）primitive-style scheduling不支持software pipelining [26]，导致weight loading与computation串行执行（图1b）：① global→registers加载（无pipelining）→ ② 向量化casting → ③ 结果存至shared memory → ④ ldmatrix从shared memory加载到registers → Tensor Core计算。Step ①-②的串行和额外的shared memory往返（Step ③-④）浪费内存带宽。

  **QuantLLM [60]**、**Marlin [21]**（手写kernel）：仅为特定quantization方案（FP6、INT4）手工优化，缺乏通用性。QuantLLM仅支持浮点5/6-bit不支持sub-channel量化粒度；Marlin仅限于4-bit signed integer且不支持Hopper GPU。

  **Baseline全栈执行例子（以Triton uint4×FP16 matmul, BS=1, decode stage为例）：**
  - 算法层：A16W4量化推理，weights为uint4压缩存储，activations为FP16
  - 框架层：Triton kernel，auto-tuning搜索tile configuration
  - 编译框架层：Triton将Python kernel编译为PTX → SASS，但uint4 loading通过手工uint32 bitwise操作实现
  - Kernel调度层：Triton cp.async异步加载权重到shared memory → 每个线程从shared memory读4个uint4（packed in uint32）→ bitwise unpack + casting → shared memory layout conversion（因unpack后的register layout与Tensor Core要求的mma.m16n8k16 layout不匹配，必须通过shared memory中转）→ ldmatrix加载 → Tensor Core计算。layout conversion是瓶颈，尤其在batch=1时受限于memory bandwidth。
  - 硬件架构层：NVIDIA L40S (Ada Lovelace)，Tensor Core mma.m16n8k16

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Tilus方法：tile级GPGPU DSL + VM，核心三大创新——代数layout系统、thread-block级编程模型含显式内存层次、原生任意位宽低精度支持。**

  **解决Triton的layout conversion瓶颈**：代数layout系统通过Kronecker product构建复杂register tensor layout。关键洞察：当两个tensor的"每线程bit数"相同且thread数相同时，可以在registers内零开销reinterpret。Tilus的View指令利用此性质——例如uint4 weight tile由32 threads持有，每thread 16 bits（4×u4），可直接reinterpret为每thread 16 bits的Tensor Core兼容layout，完全消除shared memory layout conversion。

  **解决Ladder的pipelining缺失**：Tilus的VM指令集提供CopyAsync/CopyAsyncCommitGroup/CopyAsyncWaitGroup显式异步拷贝指令，使开发者可精确控制global→shared pipelined拷贝与computation的overlap。在decode stage batch>1时这是关键性能optimization。

  **解决Ladder的type-level packing限制**：Tilus采用tile-level reinterpretation而非type-level packing。通过预处理变换权重global memory layout（如i6[K,N]→u8[K/BK,N/BN,BK*BN*6/8]），将低精度tile的紧凑比特流映射为连续u8字节序列，然后用标准LoadGlobal高效加载，再通过View做零开销类型+layout同时reinterpret（图9）。此方法通过参数化n_bytes_per_thread和GCD计算（§7.2）支持任意1-8 bit位宽。

  **解决手写kernel的通用性缺失**：所有低精度类型共享同一参数化程序模板，仅改变tile大小和数据类型参数。200个configurations per operator，auto-tuning完成。

  **Tilus方法全栈执行例子（uint4×FP16 matmul, BS=1, decode stage，对应图1c/图2）：**
  - 算法层：A16W4量化推理，weights预变换（u8紧凑存储）
  - 框架层：Tilus Python DSL程序 + vLLM集成
  - 编译框架层：Tilus VM IR → 优化passes → Hidet IR → 低精度lowering（PRMT/LOP3/bitwise指令选择）→ CUDA C → nvcc → .cubin binary
  - Kernel调度层：① CopyAsync异步global→shared拷贝（pipelined with上一iteration的computation）→ ② CopyAsyncWaitGroup同步 → ③ LoadShared从shared memory加载u8 register tensor → ④ **View零开销reinterpret**（u8→i4，layout同时转换为Tensor Core兼容格式，完全在寄存器内完成）→ ⑤ Cast向量化i4→f16（PRMT+LOP3+bitwise，寄存器内）→ ⑥ Dot Tensor Core mma.m16n8k16 → 循环k维 → ⑦ StoreGlobal写出。对比Triton：消除shared memory layout conversion（Step ④ vs Triton Step ④）；对比Ladder：加入software pipelining（Step ①-② vs Ladder的Step ①-②串行）且消除shared memory往返。
  - 硬件架构层：同baseline（NVIDIA L40S），但Tilus kernel通过自动向量化（cp.async.v4, lds128, ldg128）和指令选择（ldmatrix vs lds按layout兼容性自动选择）更充分利用硬件带宽。
