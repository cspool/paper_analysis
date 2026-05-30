## Iris: First-Class Multi-GPU Programming Experience in Triton

- baseline方法是什么？
  **Baseline方法：PyTorch torch.matmul + RCCL AllGather，即bulk-synchronous多GPU执行。**

  在baseline中，计算和通信被严格分离为两个独立kernel：首先是GEMM kernel完成全部本地矩阵乘法，所有workgroup同步、kernel结束、中间结果写入global memory；然后RCCL AllGather kernel启动，从global memory读取结果、通过Infinity Fabric分发到所有远程GPU。两个kernel之间是hard synchronization barrier——所有GEMM workgroup必须完成才能启动任何通信操作。

  Baseline全栈执行例子（以8×MI300X上GEMM+All-Gather，M=8192, N=3584, K=14336为例）：
  - 算法层：数据并行GEMM——输入A[M,K]分片在本地，B[K,N/8]各GPU持有N维的1/8。各GPU计算本地A×B得到C_local[M,N/8]。All-Gather将各GPU的C_local沿N维拼接为完整C[M,N]。
  - 系统框架层：PyTorch Distributed初始化多进程（每GPU一个rank），torch.matmul调用cuBLAS/ROCBlas GEMM，RCCL调用AMD集体通信库。CPU端host code序列：output = torch.matmul(A, B_local) → torch.distributed.all_gather(C_full, output) → CPU wait for GPU completion between each step。
  - 编译框架层：无跨kernel优化。GEMM和AllGather是独立编译的二进制，编译器看不到通信操作，无法co-optimize。
  - Kernel调度层：bulk-synchronous执行——(a) GEMM kernel launch，304 CU全部用于GEMM tile计算，所有tile完成后kernel结束；(b) global barrier（kernel teardown + CPU launch overhead + next kernel setup）；(c) RCCL AllGather kernel launch，通信操作分发数据到各GPU。存在显著的execution "bubble"——GEMM完成后GPU等待kernel teardown、CPU coordination、AllGather kernel startup，期间SM和Infinity Fabric均空闲。中间结果必须经global memory写出再读入（write→read round-trip），浪费HBM带宽。
  - 硬件架构层：8×AMD MI300X GPU，7条Infinity Fabric Link/GPU全连接mesh。Infinity Fabric在GEMM执行期间完全空闲（因通信仅在GEMM完成后才启动）。

  Baseline核心缺陷：
  1. **Bulk-synchronous bubble**：kernel barrier强制所有计算完成后才开始通信，GPU在barrier期间存在idle bubble，计算和通信资源（CU、Infinity Fabric）交替闲置而非并发利用。
  2. **Intermediate global memory traffic**：GEMM结果必须先写入global memory再被AllGather读回，增加不必要的HBM带宽消耗，而这部分数据本可直接从register或shared memory传递。
  3. **Compiler blindness to communication**：RCCL作为外部二进制库被调用，Triton编译器无法看到通信操作，无法做computation-communication co-optimization、intelligent scheduling或跨kernel boundary融合。
  4. **Kernel launch/teardown overhead**：每次kernel launch产生CPU→GPU dispatch延迟和kernel prologue/epilogue开销，在fine-grained场景（小tile、多kernel）下不可忽略。
  5. **No tile-granularity control**：开发者无法在tile级别控制"算完一块立即传出"，只能等整个GEMM kernel完成。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Iris——第一个从底层为Triton tile编程模型设计的多GPU通信库，纯Python+Triton实现，无需外部通信库依赖。

  **(1) 原生Triton实现 + 编译器全可见性**——解决"Compiler blindness"缺陷：
  所有通信操作（load/store/get/put/copy/atomic_*）都是纯Triton代码，Triton编译器可看到全部计算和通信操作。指针翻译（__translate）将本地symmetric heap指针转换为远程地址的计算逻辑对编译器透明，使编译器可co-optimize computation+communication、做unified instruction scheduling和register allocation。这与wrapper-based方法（Triton-Distributed、PyTorch Symmetric Memory）形成根本对比——后者将通信库作为opaque bytecode链接，编译器完全无法跨边界优化。

  **(2) Tile级symmetric memory API**——解决"Bulk-synchronous bubble"和"Intermediate global memory traffic"缺陷：
  提供值语义（load/store——register↔remote memory）和指针语义（get/put/copy——buffer↔buffer）两套API，均操作于tile粒度(BLOCK_SIZE_M × BLOCK_SIZE_N)。值语义的关键优势：GEMM tile产出后可直接从register iris.store到远程GPU，无需先写local global memory再读回——消除intermediate HBM round-trip。这使Fused Sequential模式成为可能：在GEMM的主循环末尾插入几行iris.store代码，每个tile一产完即发。

  **(3) 融合kernel模式分类（Taxonomy of Fused Patterns）**——解决"No tile-granularity control"缺陷：
  Iris提供了完整的compute-communication overlap策略谱系，均通过minimal code changes实现：
  - **Fused Sequential**（最简单）：在GEMM loop结束后附加iris.store将tile scatter到所有远程GPU。仅需几行代码修改。适用于通信占比小（小输出tile + 大K）的场景。缺点是GEMM和All-Scatter仍为顺序依赖——最后一个GEMM tile完成后仍需执行其通信，增加tail latency。
  - **Fused Workgroup Specialization**（最高效）：单persistent kernel内按program_id划分workgroup——前256个workgroup做GEMM(tl.dot)，完成后atomic_cas(release)发信号；后48个workgroup spin-lock(atomic_cas with acquire)等信号，获取后iris.put通信。GEMM和通信在不同CU上并发执行，通信可完全隐藏在GEMM后面（尤其是小N大K场景——因为N/8后每个tile通信量极小）。代价是需要worst-case resource allocation——fused kernel的资源分配（shared memory、VGPRs、thread count）受GEMM（资源密集型操作）约束，即使通信操作本不需那么多资源。
  - **Unfused Producer-Consumer**：与Fused Workgroup Specialization对称，但使用两个独立kernel在不同CUDA stream上执行——避免worst-case resource allocation（通信kernel可独立配置更优化的资源分配），代价是额外的kernel launch latency和更少的调度控制。

  **(4) 成熟的C++/HIP memory model + GPU-scoped atomics**——解决同步正确性问题：
  使用acquire/release ordering + gpu/sys scope控制跨GPU可见性，而非引入新的同步语义。这一设计基于AMD的SC-HRF (Sequentially Consistent Heterogeneous Race Free)内存模型，使开发者使用熟悉的primitive做多GPU同步。

  **(5) Cache-aware programming**——解决chiplet架构性能问题：
  cache_modifier(".wt")控制write-through策略适配chiplet间coherence，chiplet_swizzle将workgroup映射到XCD分组（spatial locality for LLC），GROUP_SIZE_M做L2-cache-friendly tile grouping。

  论文方法全栈执行例子（以Fused Workgroup Specialization GEMM+All-Scatter，8×MI300X，M=8192, N=3584, K=14336为例）：
  - 算法层：同baseline数据并行GEMM+All-Scatter，算法不变。
  - 系统框架层：iris.init()通过PyTorch Distributed + HIP IPC建立跨GPU symmetric heap。单次launch wg_specialized_gemm_all_scatter[(304,)]，无multi-kernel coordination、无CPU-side host code between steps。
  - 编译框架层：Triton编译整个fusion kernel——编译器同时看到gemm_loop的tl.dot计算和iris.put的远程指针翻译+tl.store，可做unified register allocation、instruction scheduling和memory coalescing。通信不是opaque binary blob而是first-class Triton code。
  - Kernel调度层：256 GEMM workers持续执行gemm_loop（tile级软件pipeline: global→shared→register→Tensor Core MMA），每个tile完成后atomic_cas(release)通知；48 COMM workers持续spin-lock等待信号，获取后执行iris.put：__translate(ptr, from_rank, to_rank, heap_bases) → tl.load(heap_bases+to_rank)获取remote heap base → offset = ptr_int - from_base → remote_ptr = to_base + offset → 通过Infinity Fabric直接写remote GPU memory。GEMM和通信在304个CU上并发——计算资源(256 CU)和通信资源(48 CU + Infinity Fabric)同时饱和。
  - 硬件架构层：Infinity Fabric在GEMM执行期间不再空闲——每个GEMM tile完成后立即被COMM worker通过Infinity Fabric传输。N=3584/8=448每GPU，小输出tile使通信带宽需求远低于Infinity Fabric容量，通信完全隐藏在GEMM后面（Figure 10深色区域几乎覆盖整个时间线）。speedup达2.5×。

  设计思路核心：
  Iris的根本洞察是**fine-grained compute-communication overlap的真正障碍不是硬件能力，而是抽象层级的mismatch**。当通信原语与计算原语生活在同一语义空间（tile-based Triton）时，overlap pattern从"需要独立kernel、host-side coordination、manual resource partitioning的heroic engineering"退化为"在同一kernel内加几行代码即可实现"。这验证了一个更广义的论点：编译器可见性（而非纯粹的手工汇编优化）是高效多GPU编程的关键——当编译器能同时看到计算和通信，co-optimization自然发生。Iris的1.79× peak speedup的深层意义在于：它是在纯Python+Triton（通常被认为比手写CUDA/HIP性能差）中实现的，但通过abstraction alignment（而非lowering到更低级语言）达到了超越手写RCCL的性能。这暗示高性能多GPU编程的未来方向是"raising the abstraction level to match the problem structure"而非"lowering to bare metal"。
