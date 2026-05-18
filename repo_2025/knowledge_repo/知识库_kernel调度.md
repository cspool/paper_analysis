## Data-Centric Dataflow Abstraction (for Heterogeneous NMP Accelerators)

术语解释
Data-Centric Dataflow Abstraction是H2-LLM提出的面向异构NMP加速器的数据流设计方法。与先分配computation engine（centralized processor或NMP PEs）的compute-centric方法不同，data-centric方法先bind memory channels到operators，再决定computation engine分配。

术语是什么？
Data-Centric Dataflow Abstraction解决了NMP-based heterogeneous accelerator上operator-to-hardware mapping的核心问题：如何在保持full external bandwidth对centralized processor可用的同时，灵活利用NMP加速能力。相比SpecPIM [47]的compute-centric mapping（先分配engine后决定channel），data-centric方法的优势是：(1) 所有正常channels的external bandwidth始终对centralized processor可用，避免prefill operators因bandwidth不足变为memory-bound；(2) 支持flexible operator fission——operator可同时分配到normal和NMP channels沿output feature dimension分割；(3) 支持exploring inter-operator parallelism（如parallel transformer中attention block和FFN block的并发执行）。

从kernel调度角度拆解：
Data-Centric Dataflow Abstraction的三阶段operator-channel binding过程：

**Stage 1: Memory Access Group (MAG) Partition**
```
Input: Transformer layer operator graph V
Output: A0 ⪯ A1 ⪯ ... ⪯ A_{M-1}
Constraints:
  - A0 ∪ A1 ∪ ... ∪ A_{M-1} = V  (cover all operators)
  - A_i ∩ A_j = ∅ (∀i ≠ j)       (no overlap)
  - A_i ⪯ A_j means all ops in A_i do not depend on ops in A_j
```
每个MAG包含可并行执行的operators组，同一MAG内的operators共享相同输入tensor的会被合并以探索fission策略。所有normal + NMP channels被分配给每个MAG。

**Stage 2: Coarse-Grain Binding (GCMap)**
```
For each MAG A_i:
  1. Extract weakly connected components → MPGs P_{(i,j)}
  2. GCMap_{A_i}: {P_{(i,0)}, ..., P_{(i,N-1)}} → P(C) - ∅
     where C = {PC0,...,PC_{P-1}, NC} (NC = normal channels集合)
  Constraints:
    - GCMap(P_{(i,j)}) ∩ GCMap(P_{(i,j')}) = ∅ (channel exclusive)
    - ∪_j GCMap(P_{(i,j)}) = C                  (channel utilization)
```
每个MPG被分配独立的channel subset，不同MPG间无channel交叠，保证并行执行互不干扰。

**Stage 3: Fine-Grain Binding (OCMap)**
```
For each MPG P_{(i,j)}:
  1. Stratify into operator tiers: T^{(i,j)}_k based on dependency
  2. OCMap_{T^{(i,j)}_k}: operators → P(GCMap(P_{(i,j)})) - ∅
  Each operator tier occupies all channels of its MPG,
  operators within same tier are mutually independent and execute concurrently.
```

**Operator Fission**：当某operator同时被bind到normal和NMP channels时：
- GEMM: 沿output feature dim N在centralized processor和NMP PEs间分割
- Attention: 不同GEMMs（QK, SV等）分配到两个computation engine

**Decoding Stage Execution Flow**：
```
for each MAG A_i in order A_0 to A_{M-1}:
    for each MPG P_{(i,j)} in parallel:        // distinct VPUs
        for each tier T^{(i,j)}_k in order:
            for each operator in T^{(i,j)}_k in parallel:
                execute on bound computation engine
            synchronize within tier             // case (2)
        synchronize across MPGs                 // case (1)
    synchronize fission operators               // case (3)
```

术语一般如何实现？如何使用？
H2-LLM的Data-Centric Dataflow通过其DSE framework实现——genetic algorithm搜索MAG partition、GCMap、OCMap和operator fission ratio的最优组合。Model compiler将搜索到的dataflow design编译为execution flow（包括tiling factors、buffer allocation、synchronization points），由centralized processor的controller在runtime调度执行。Controller通过deterministic timing（memory access + buffer access + computation cycles）管理同步，无需runtime dynamic decisions。代码开源：https://github.com/leesou/H2-LLM-ISCA-2025。

涉及论文标题：
- 9-H2-LLM- Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference.pdf

## Operator Fission (for Heterogeneous NMP Accelerators)

术语解释
Operator Fission是将一个LLM operator的计算workload在heterogeneous accelerator的centralized processor和NMP PEs之间分割执行的技术。与duplicating weights for both engines不同，fission沿output feature dimension分割而不复制weights。

术语是什么？
Operator Fission解决了异构加速器中单个operator如何同时利用centralized processor和NMP PEs的问题。在H2-LLM中，当data-centric dataflow binding将某operator分配到both normal和NMP channels时触发fission。对于GEMM operator (M,K)×(K,N)：output feature dim N被split，centralized processor负责部分output channels，NMP PEs负责剩余部分，无weight duplication。对于attention operators：不同GEMMs（如QK、SV、O projection）被分配到不同computation engine，而非split单个GEMM。Fission后两部分独立执行，通过synchronization等待双方完成后再merge结果。相比weight duplication方案（需复制完整模型weights到两个engine），fission在edge-side accelerator有限的memory capacity下更practical。

从kernel调度角度拆解：
H2-LLM中operator fission的kernel执行流程：
```
Operator: FC layer with shape (M, K) × (K, N)
Fission: Split N → N_cpu + N_nmp

// Centralized Processor Part
CPU_Part:
  input = gather(M, K) from external memory    // shape (M, K)
  weight_cpu = load weight[:, :N_cpu]           // shape (K, N_cpu)
  output_cpu = GEMM(input, weight_cpu)           // on systolic array/VPU
  // output_cpu shape: (M, N_cpu)

// NMP Part (in parallel)
NMP_Part:
  scatter input to HB-NMP channels              // (M, K) scattered
  weight_nmp = already resident in PE buffers   // (K, N_nmp)
  for each output tile in (M, N_nmp):
    load input_tile to global buffer (❶)
    for each PE: load weight_tile (❷), MAC accumulate (❸)
    write output_tile back to DRAM (❹)
  // output_nmp shape: (M, N_nmp)

// Synchronization (case 3)
Barrier:
  wait CPU_Part and NMP_Part both done

// Merge
output = concat(output_cpu, output_nmp) along dim N  // shape (M, N)
```

对于attention operators：Q/K/V projections → 不同GEMMs分配到不同engine（如Q→NMP, K→CPU, V→NMP）→ QK在更优engine上执行→ SV和O类似分配。

术语一般如何实现？如何使用？
Operator fission由H2-LLM的dataflow binding自动决定——当OCMap将某operator同时分配到normal和NMP channels时触发。Fission ratio（N_cpu vs N_nmp）是dataflow design space的一部分，由genetic algorithm搜索。Model compiler根据fission decision生成两部分各自的execution flow（NMP部分用operator templates + Eq.1 tiling，CPU部分用Tileflow），并插入synchronization barriers。H2-LLM实验表明flexible fission比fixed fission (AttAcc [60]仅FFN的固定fission)提升1.11× (geomean)。开源：https://github.com/leesou/H2-LLM-ISCA-2025。

涉及论文标题：
- 9-H2-LLM- Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference.pdf

## SIMT-enhanced Software Pipeline for Mixed-Precision GPU GEMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SIMT-enhanced Software Pipeline是COMET-W4Ax kernel中的两级重叠流水线技术，用于在GPU上隐藏混合精度GEMM中的数据加载、反量化（dequantization）和格式转换（permutation）开销。传统GEMM流水线仅重叠global memory load和tensor core计算，但W4Ax kernel需要额外步骤——CUDA core上的dequantization和permutation——这些步骤的吞吐远低于tensor core（如A100上CUDA core仅78 TFLOPS vs INT4 tensor core 1248 TOPS），若未有效overlap将成为瓶颈。

COMET的两级流水线设计：(1) **第一级：off-chip memory load隐藏在data transformation + tensor core计算中**。使用cp.async从HBM异步加载数据到shared memory，加载期间CUDA core可并行执行dequantization/permutation。(2) **第二级：双缓冲shared memory隐藏计算与数据传输**。分配两个shared memory buffer（buffer0、buffer1），tensor core从buffer0读取数据执行mma时，buffer1从global memory预取下一tile数据并执行transformation。使用async_copy_barrier确保数据就绪，sync_threads确保线程同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
COMET-W4Ax的双缓冲流水线伪代码（GEMM tile 128×128×128, A100）：
```
// 共享内存双缓冲
__shared__ int4_tile smem_A[2][TILE_M][TILE_K];  // buffer0, buffer1
__shared__ int4_tile smem_W[2][TILE_N][TILE_K];
register_fragment frag_A, frag_W, frag_C;

// Iteration 0: 初始化——加载A0+W0到buffer0
cp.async(&smem_A[0], &gmem_A_tile0);  // async global→shared
cp.async(&smem_W[0], &gmem_W_tile0);
// 若需要dequant/permutation: CUDA core并行处理
if (need_dequant_A0) dequant_INT4toINT8_CUDA_core(&smem_A[0]);
if (need_perm_A0)    permute_channels_CUDA_core(&smem_A[0]);
async_copy_barrier();  // 等待所有cp.async完成

// Iteration 1+: 双缓冲交替
for (int iter = 1; iter < num_tiles; iter++) {
    sync_threads();
    
    // Compute: tensor core从buffer_prev读取
    ldmatrix(&frag_A, &smem_A[(iter-1)%2]);  // reg←shared
    ldmatrix(&frag_W, &smem_W[(iter-1)%2]);
    if (is_INT4_tile) 
        mma.sync.m16n8k64.INT4(&frag_C, &frag_A, &frag_W);
    else
        mma.sync.m16n8k32.INT8(&frag_C, &frag_A, &frag_W);
    
    // Prefetch: 同时加载下一tile到buffer_cur
    cp.async(&smem_A[iter%2], &gmem_A_next);
    cp.async(&smem_W[iter%2], &gmem_W_next);
    if (need_dequant) dequant_CUDA_core(&smem_A[iter%2]);
    if (need_perm)    permute_CUDA_core(&smem_A[iter%2]);
    
    async_copy_barrier();
}
```
关键同步原语：(a) `cp.async` 异步拷贝不阻塞warp；(b) `async_copy_barrier()` 等待所有pending async copy完成；(c) `sync_threads()` block内线程同步。该设计隐藏了所有CUDA core data transformation开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Software pipeline是GPU GEMM优化的通用技术（如CUTLASS的ping-pong buffer、ALCOP的自动compiler级pipelining）。COMET的独特之处是将SIMT pipeline适配到混合精度场景：INT4和INT8 tile的pipeline结构相同，但INT8 tile额外需要CUDA core上的INT4→INT8格式转换（fast conversion, 2 inst/value），该转换通过pipeline隐藏在tensor core计算中。pipeline效果：ablation显示去掉software pipeline后kernel latency退化1.69×（LLaMA-3-8B, batch=16-256平均）。编译为CUDA .so动态库后可在TensorRT-LLM等框架中热加载使用。

涉及论文标题：
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

---

## Weight Interleaving for Mixed-Precision GPU GEMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Interleaving是COMET-W4Ax kernel中针对W4A8 GEMM的一种权重数据布局优化，通过重排INT4权重的存储顺序，消除多个线程同时加载时产生的shared memory bank conflict，并将ldmatrix指令数从2条减至1条。

问题背景：W4A8 GEMM使用INT8 tensor core（mma.m16n8k32），但权重以INT4存储（节省4×存储）。GPU tensor core要求所有操作数为同精度，因此INT4权重必须先转换为INT8再参与计算。在标准W8A8 GEMM中，32个线程的ldmatrix数据加载是连续的——线程T0加载地址0-3(值b0-b3)、T1加载4-7(b4-b7)。但在INT4存储格式下，2个4-bit值打包在1个byte中，导致T0需要加载b0-b7（含b4-b7）而T1也需要b4-b11（含b4-b7），产生重叠和shared memory conflict。COMET的interleaving方案：将INT4权重按交错而非连续方式存储——T0加载地址0-3和8-11（b0,b1,b8,b9等非连续值），T1加载4-7和12-15——使各线程访问的地址不重叠，消除conflict。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Weight Interleaving的数据布局对比：
```
// Naive layout (连续存储，每2个INT4打包1 byte→16-bit):
// 32-bit word 0: [W3(4b) | W2(4b) | W1(4b) | W0(4b)]
// 32-bit word 1: [W7(4b) | W6(4b) | W5(4b) | W4(4b)]
// Thread T0 loads word0(word0[0-31]) → covers W0-W3
// Thread T1 loads word0(word0[32-63])? NO, T1 loads next 32-bit → W4-W7
// 但INT4下T0需要W0-W7(2 words), T1需要W4-W11(2 words) → 重叠

// Interleaved layout (COMET):
// 32-bit word at idx 0: [W1(4b) | W0(4b) | W1(4b) | W0(4b)] from different positions
// Thread T0: loads non-contiguous chunks → addresses 0-3, 8-11
// Thread T1: loads non-contiguous chunks → addresses 4-7, 12-15
// 所有线程访问的地址不重叠 → 无shared memory conflict

// PTX实现（简化）:
// ldmatrix.sync.aligned.x4.shared.b16  → 单条指令完成交错加载
```
该设计使W4A8 GEMM的ldmatrix指令从2条（标准INT8 GEMM需要2次ldmatrix）减至1条。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Weight interleaving在COMET中通过CUDA kernel的全局内存数据布局实现：量化权重时即按interleaved pattern存储（离线完成，无运行时开销）。实现为CUTLASS风格的tile iterator中自定义data layout functor。Ablation显示去掉weight interleaving后kernel latency退化1.27×。该优化与fast INT4→INT8 conversion协同：interleaving解决数据加载效率，fast conversion解决格式转换效率。

涉及论文标题：
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

---

## Fast INT4-to-INT8 Conversion on GPU CUDA Cores

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fast INT4-to-INT8 Conversion是COMET-W4Ax kernel中的高效数据格式转换技术，用于在CUDA core上将INT4权重转换为INT8以便INT8 tensor core进行W4A8 GEMM。传统naïve转换需10条PTX指令（数据移位+sign extension需多指令模拟），COMET通过两个策略降至2条指令：(1) **location switch**：交换数据存储位置使得转换后的8-bit地址自然对齐，避免移位；(2) **zero extension替代sign extension**：PTX不直接支持4-bit sign extension，但支持zero extension单指令（零填充）。zero extension后值被乘以16，通过scale factor除以16补偿。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Naive INT4→INT8 (10 inst per value):
// 输入：16-bit packed [W3|W2|W1|W0] (各4-bit signed)
// 1. 提取W0: AND R0, packed, 0x0F
// 2. 提取W1: SHR R1, packed, 4; AND R1, R1, 0x0F
// 3. Sign extend W0: SHL R2, R0, 28; SHR R0, R2, 28  (3 inst)
// 4. Sign extend W1: SHL R2, R1, 28; SHR R1, R2, 28
// ... W2, W3类似 → 总计 ~10 inst/value

// COMET Fast Conversion (2 inst per value):
// Step 1: Location Switch —— 离线重排打包顺序
// 原: [W3(12-15) | W2(8-11) | W1(4-7) | W0(0-3)]
// 新: [W1(12-15) | W0(8-11) | W3(4-7) | W2(0-3)]
// 使W1在bits 12-15, W0在bits 8-11 → 天然对齐8-bit边界

// Step 2: Zero Extension (PTX: lop3 / CUDA intrinsic)
// 提取W0: PRMT R0, 0x5410, packed, 0x0  → 单指令zero-extract到8-bit
// 提取W1: PRMT R1, 0x7632, packed, 0x0  → 同上
// Zero extension使值乘以16 → scale factor需除以16:
// result = tensor_core_mma(A_INT8, W_zeroext) * (scale / 16.0)
```
性能对比：naïve 10 inst/value × 32 threads = 320 inst；COMET 2 inst/value × 32 threads = 64 inst。在CUDA core瓶颈场景下（78 TFLOPS vs 624 TOPS INT8 TC），指令数减少5×大幅缓解瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该技术利用NVIDIA PTX ISA的字节置换指令（PRMT - Permute Bytes）实现快速数据提取。PRMT可从32-bit源中选择任意4个bytes并按任意顺序输出。location switch + zero extension的组合使格式转换从瓶颈转变为可被software pipeline完全隐藏的轻量操作。Ablation显示去掉fast conversion后kernel latency退化1.53×。该设计也适用于未来GPU的FP4→INT8转换（H100支持FP4）：FP4的sign和mantissa位保持原始位置，仅exponent需移位转换。

涉及论文标题：
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

---

## Fine-Grained SM Scheduling for Mixed-Precision GPU GEMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-Grained SM Scheduling是COMET-W4Ax kernel中解决混合精度GEMM（W4A4 + W4A8 tile混用）导致的SM负载不均问题的调度策略。当INT4 tensor core（1248 TOPS）和INT8 tensor core（624 TOPS）在同一kernel中混合使用时，负责INT4 tile的SM计算快2×但必须等INT8 tile的SM完成才能同步——导致INT4 SM大量空闲。COMET的三层调度：(1) **Barrier Minimization**：仅在最终结果写回全局内存前插入一次同步，而非每个mma iteration都同步；(2) **Tile Remapping**：重新映射tile到SM，使每个SM的INT4:INT8 tile比例均匀；(3) **Tile Decomposition + Task-Stealing**：打破一对一的tile-SM绑定，idle SM可主动从邻近busy SM窃取部分计算任务（one-to-many binding）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 场景: GEMM 256×256×384, 2个activation blocks (1 INT4, 1 INT8)
// 共18个tile (128×128×128 each), 4个SM

// === Naive调度 (tile按顺序映射SM, 每迭代同步) ===
// SM0: tile 0(INT8) tile 4(INT4) tile 8(INT4)  tile 12(INT4) tile 16(INT8)
// SM1: tile 1(INT4) tile 5(INT8) tile 9(INT4)  tile 13(INT4) tile 17(INT4)
// SM2: tile 2(INT4) tile 6(INT4) tile 10(INT8) tile 14(INT4)         -
// SM3: tile 3(INT4) tile 7(INT4) tile 11(INT4) tile 15(INT8)         -
// 问题: SM1 INT4 tile多→快→等SM0 INT8 tile→利用率低

// === COMET调度 (Tile Remapping + Task-Stealing) ===
// Step 1: Barrier minimization —— 仅最终写回前一次sync
// Step 2: Tile Remapping —— 均匀分配INT4/INT8
// SM0: tile 0(INT8) tile 5(INT4) tile 8(INT4)  tile 13(INT4) tile 16(INT4)
// SM1: tile 1(INT4) tile 4(INT4) tile 9(INT8)  tile 12(INT4) tile 17(INT4)
// SM2: tile 2(INT4) tile 7(INT4) tile 10(INT4) tile 15(INT8)        -
// SM3: tile 3(INT4) tile 6(INT8) tile 11(INT4) tile 14(INT4)        -
// 每SM INT4:INT8 ≈ 3:1 或 4:0  → 相对Naive更均匀

// Step 3: Tile Decomposition —— 最后iter时SM2/SM3空闲
// SM0: tile 16(INT4, full)      → SM0处理全部
// SM1: tile 17(INT4, full)      → SM1处理全部
// SM2: steal tile_16_half from SM0   # one-to-many binding
// SM3: steal tile_17_half from SM1
// SM2/SM3从shared memory直接读取数据协助SM0/SM1计算
```
效果：Tile remapping使speedup从1.31×升至1.56×（LLaMA-3-8B），加入tile decomposition后升至1.71×，达Oracle W4A4 kernel的92.7%-97.8%性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SM scheduling在COMET中作为编译期优化实现：在LLM compilation阶段（模型加载时），根据各GEMM shape和FMPQ确定的INT4/INT8 tile比例，预先计算tile-to-SM映射表。task-stealing通过shared memory实现——idle SM从shared memory直接读取邻近tile的数据（同一thread block内的shared memory对所有warp可见），无需额外HBM访问。该设计灵感来自Stream-K的work-centric parallel decomposition，但适配了混合精度场景。

涉及论文标题：
- 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

---

## Pointer-Chase Latency Measurement (指针追踪延迟测量)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pointer-Chase是测量内存系统load-to-use延迟的经典microbenchmark技术。核心原理：构建一个链表（pointer chain），每个节点仅包含指向下一个节点的指针（地址），且节点按随机顺序链接。benchmark kernel在单线程中循环执行 `tmp = *tmp` 操作——每次load的目标地址依赖上一次load的结果值，从而完全串行化内存访问（无法被硬件prefetcher预测、无法被out-of-order执行并行化）。总执行时间除以pointer chase次数即为每次内存访问的平均load-to-use延迟。随机link order避免硬件stride prefetcher检测固定步长模式。为测量loaded system延迟，Mess benchmark将pointer-chase放在一个core上，同时在其余core上运行memory traffic generator产生背景流量——pointer-chase测得的延迟反映在该背景流量水平下的实际内存响应时间。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Pointer-Chase kernel的汇编级伪代码（x86示例）：
```
// Setup: 分配N个cache-line大小的node，随机排列链接
// 每个node: [next_pointer (8B)] [padding to 64B]
// 随机排列使得连续访问的nodes在物理地址上不连续

// Kernel (one core):
mov rax, [start_address]    // 加载起始地址
mov rcx, MEASUREMENT_COUNT  // 测量次数
loop:
  mov rax, [rax]            // pointer chase: load next address
  dec rcx                   // 减少计数器
  jnz loop                  // 循环

// 延迟 = total_cycles / MEASUREMENT_COUNT
```
GPU版本的pointer-chase同理：单SM执行 `LD R0, [R0]`，但GPU上pointer-chase延迟与CPU有很大不同（H100负载延迟363ns，远高于CPU的85-129ns，因GPU的片内频率低、复杂内存层次、大量SM竞争内存带宽）。Mess benchmark的关键设计选择：(a) kernel用汇编实现而非C，避免编译器引入额外指令或reorder导致的干扰；(b) 数据分配在大页内存（2MB/1GB pages）以最小化TLB miss和page walk开销；(c) 运行时通过硬件计数器监控TLB-related事件，从总latency中减去这些非内存开销。这保证了测量的latency接近纯硬件内存系统的响应时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Pointer-chase可用于：(1) **内存系统characterization**：测量不同平台的unloaded/loaded latency，构建bandwidth-latency曲线；(2) **微架构reverse engineering**：通过改变working set size和stride，探测cache hierarchy各层的大小和延迟（如经典的LMbench lat_mem_rd测试）；(3) **模拟器验证**：比较模拟和实际的pointer-chase结果来校准内存模型。开源工具：LMbench（lat_mem_rd）、Google multichase（https://github.com/google/multichase）、P-chase（GPU用）。Mess benchmark（https://github.com/bsc-mem/Mess-benchmark）提供完整的多平台pointer-chase实现（汇编，支持x86/ARM/Power/RISC-V/PTX），并与traffic generator结合用于bandwidth-latency曲线测量。

涉及论文标题：
- 7-A_Mess_of_Memory_System_Benchmarking_Simulation_and_Application_Profiling.pdf

## Memory Traffic Generator (内存流量生成器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Traffic Generator是Mess benchmark的核心组件之一，在多个CPU核/GPU SM上并发运行以生成可控的内存背景流量。每个核遍历两个独立数组——一个load数组（顺序读取）和一个store数组（顺序写入）——通过控制并发核数和访问步长来调节总内存带宽利用率，通过load/store数组访问比例来控制读写比。由于多个核并发访问不同数组，实际产生的内存访问pattern并非简单顺序流，而是由多核交叉和不同数组访问交错组成的复杂混合模式，覆盖了大范围的DRAM row-buffer hit/empty/miss率。在Mess paper的Intel平台测量中，row-buffer状态分布在35/43/22%（低带宽）到84/13/3%（高带宽，高读比）之间，反映了真实workload的访问特征。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Traffic Generator kernel的汇编级伪代码（每个worker核）：
```
// 每个worker核分配两个独立数组:
// load_array[N] - 用于load操作的数组
// store_array[N] - 用于store操作的数组

// Kernel (per core, 汇编实现):
mov rcx, ITERATIONS
load_loop:
  mov rax, [load_array + rdx]   // load: 产生读流量
  mov [store_array + rdx], rax  // store: 产生写流量
  add rdx, STRIDE               // 步进（可配置控制带宽强度）
  dec rcx
  jnz load_loop
```
关键设计考虑：(a) 流量可控性：调整活跃worker核数（0到所有核）→遍历bandwidth范围；调整STRIDE（跨步访问）→改变cache miss率进而影响bandwidth；调整load vs store比例→控制读写比（100%-load kernel=纯读，100%-store kernel在write-allocate cache下=50%读/50%写，streaming store kernel可产生>50%写）。(b) 复杂访问模式：每个worker顺序访问自己的数组→对DRAM row-buffer是顺序友好的；但多worker并发+不同数组地址→全局memory controller看到的是高度交错的多流访问→row-buffer hit率偏离简单顺序模式的理想值。这与真实多进程HPC workload有可比性。(c) 与pointer-chase无干扰：pointer-chase运行在专用core/SM上，traffic generator在其余资源上——两者通过共享的DRAM memory system竞争，而不会直接干扰对方的cache/execution units。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Memory Traffic Generator通常作为memory system characterization benchmark套件的一部分使用：(1) 与延迟测量（pointer-chase或其他方法）配合生成bandwidth-latency曲线；(2) 评估内存系统的饱和行为——增加traffic直到延迟开始非线性增长；(3) 评估read/write ratio对性能的影响——不同比例的load/store产生不同DRAM bank/rank的时序约束效应。Mess benchmark的traffic generator已开源（https://github.com/bsc-mem/Mess-benchmark），支持x86/ARM/Power/RISC-V CPU和NVIDIA GPU PTX。与STREAM benchmark的对比：STREAM测量应用级可持续带宽（4个kernel: Copy/Scale/Add/Triad），每个kernel固定有一个特定的读写比且不控制延迟测量；Mess traffic generator可连续调节带宽和读写比，与延迟测量同时进行，提供全面的bandwidth-latency关系。

涉及论文标题：
- 7-A_Mess_of_Memory_System_Benchmarking_Simulation_and_Application_Profiling.pdf

## STREAM Benchmark (STREAM可持续内存带宽基准测试)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STREAM是由John McCalpin提出的经典内存带宽benchmark，至今仍是HPC领域衡量可持续内存带宽的de facto标准。它包含四个向量操作kernel：(1) **Copy**: `c[i] = a[i]`（2读1写，但实际是1读1写因为c不需要先读）；(2) **Scale**: `b[i] = scalar * c[i]`（1读1写）；(3) **Add**: `c[i] = a[i] + b[i]`（2读1写）；(4) **Triad**: `a[i] = b[i] + scalar * c[i]`（2读1写，融合乘加）。每个kernel的工作集大小设为远大于最后一级cache，确保所有访问都到主存。报告的带宽是"应用级"（application-level）——基于应用执行时间和数据结构大小估算的"有用数据流量"，不包括微架构产生的额外流量（如write-allocate引发的额外read、cache eviction、prefetch等）。Mess paper深入对比了STREAM应用级带宽和Mess架构级带宽（基于uncore计数器，包括所有微架构流量），揭示了两种测量方法在不同平台上的系统偏差。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
STREAM Copy kernel的伪代码（简化）：
```
// Setup: a[N], c[N] 数组 (N >> LLC size)
#pragma omp parallel for
for (i = 0; i < N; i++) {
    c[i] = a[i];  // 1 read + 1 write per element
}
// Bandwidth = (N * sizeof(element) * 2) / exec_time
// 其中 2 = 1 read + 1 write (应用视角)
```
但实际在write-allocate cache的CPU上，`c[i] = a[i]` 的microarchitecture级流量可能是：(a) 若c[i]的cache line不在cache中→产生一次read（read-for-ownership加载c的完整cache line）+一次write（eviction时写回）；(b) a[i]的读取也可能触发eviction。因此架构级总流量可能高于应用级计算值。Mess paper发现不同平台STREAM与Mess最大带宽的差异不同：IBM Power 9和Intel平台的差异最大（STREAM明显低于Mess，暗示write-allocate cache），而Amazon Graviton 3和NVIDIA H100的差异很小（暗示write-through cache policy）。因此建议同时使用两种带宽分析方法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
STREAM使用C语言+OpenMP实现。标准用法：(1) 编译：`gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream`；(2) 执行：`OMP_NUM_THREADS=N ./stream`（N=物理核数）；(3) 输出：四个kernel各自的带宽值（MB/s或GB/s）。关键注意事项：(a) 数组大小需至少4×LLC大小以避免cache effect；(b) 需要在编译时指定数组大小（-DSTREAM_ARRAY_SIZE），运行时不可变；(c) 多核执行需要注意NUMA effect和thread pinning。STREAM自1990年代发布以来在HPC社区广泛使用，是每台新系统验收测试的标准项目。开源：https://github.com/jeffhammond/STREAM。

涉及论文标题：
- 7-A_Mess_of_Memory_System_Benchmarking_Simulation_and_Application_Profiling.pdf

## Bit-plane Data Layout Scheme

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-plane Data Layout Scheme是Anda（HPCA 2025）提出的一种片上存储数据布局方案，用于高效组织变长BFP激活数据。与传统以element为单位的固定长度数据布局（将每个FP数据作为原子单元按行存储）不同，bit-plane布局将一组值（64个Anda元素）的sign、mantissa和exponent按bit-plane view分离重组：相同位权重的bit被打包为连续的64-bit word。例如，64个元素的第k位mantissa bit被打包为一个64-bit word存储。这种transposed data arrangement类似bit-interleave但更规整——mantissa的每个bit-plane作为一个独立的可寻址行，variable mantissa length仅改变address depth（bit-plane数量）而不影响memory bandwidth利用率。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bit-plane布局下的Anda PE读取流程（M=7 bit mantissa, GS=64）：
```
# Memory layout for Group #k (64 Anda values, each M-bit mantissa):
#   addr_offset + 0: sign[63:0]    (64-bit word, 1 bit per element)
#   addr_offset + 1: exp[4:0]      (5-bit shared exponent)
#   addr_offset + 2: mant_bp[0]    (bit-plane 0: 64 MSB bits)
#   addr_offset + 3: mant_bp[1]    (bit-plane 1)
#   ...
#   addr_offset + M+1: mant_bp[M-1] (bit-plane M-1: 64 LSB bits)
# Total depth: 2 + M (sign+exp rows + M bit-plane rows)
# For variable M, total depth varies but bandwidth per access = 64b always

# Kernel: Bit-serial dot-product using bit-plane layout
for bp in range(M):  # M bit-planes, MSB to LSB
    mant_bits = load(mant_bp_addr[bp])    # 64-bit word, 1 bit per element
    for i in range(64):
        partial[i] += mant_bits[i] * weight_int4[i]  # 1b×4b multiply
    partial = partial << 1  # next bit-plane, shift for bit significance
result = partial × 2^shared_exp × weight_scale
```

对比传统element-based layout（每个FP16元素需访问16-bit），bit-plane layout每次访问获取64个元素的1-bit（共64-bit），带宽利用率100%且访问模式规整。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit-plane layout的硬件实现：Activation Buffer采用multi-bank SRAM设计，address generator根据当前处理的mantissa bit-plane索引产生读地址（addr = base + group_idx × (2+M_group) + bp_offset）。由于每个bit-plane是64-bit word（对齐常见SRAM bank宽度），无alignment浪费。Variable mantissa长度由per-module的M参数控制address depth，在address generation时M作为输入参数。该布局与bit-serial PE（APU）天然契合：PE每cycle处理1个bit-plane，无需decompress/reformat，直接从SRAM读取64-bit mantissa bits送入adder tree。

涉及论文标题：
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## First-Element-Then-Bit-Plane (FETB) Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
First-Element-Then-Bit-Plane (FETB) Reduction是Anda PE（APU）中采用的bit-serial点积累加模式。传统的bit-serial累加有两种方案：(1) element-first: 先对每个元素的所有bit-plane累加得到per-element partial sum，再跨元素累加——需要存储多个per-element intermediate results；(2) bit-plane first: 对每个bit-plane内所有元素累加后得到一个partial plane sum，shift后跨plane累加——仅需存储每个bit-plane的一个partial sum。FETB属于bit-plane first模式：每个bit-plane内64个元素的1-bit×INT4乘法结果通过adder tree累加为1个INT32 partial plane sum，存入单个register；所有M个bit-plane的partial plane sums进行带移位的顺序累加得到最终dot-product结果。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FETB reduction在Anda PE中的伪代码（GS=64, INT4 weight, M-bit mantissa）：
```
# FETB dot-product kernel
# Input: bit_plane_mant[M][64] (M bit-planes, 64 elements each)
#        weight_int4[64] (INT4 weights)
# Output: dot_product (INT32)

partial_plane_sum = [INT32(0)] * M

# Phase 1: Per-bit-plane first-element reduction
for bp in range(M):  # MSB to LSB
    for i in range(64):
        # 1-bit mant × 4-bit weight per element
        elem_prod = mant_bits[bp][i] ? weight_int4[i] : 0
    # Adder tree: 64→1 reduction
    partial_plane_sum[bp] = tree_add(elem_prod[0:64])

# Phase 2: Cross-bit-plane sequential accumulation
dot_product = 0
for bp in range(M):
    dot_product = (dot_product << 1) + partial_plane_sum[bp]  # shift for bit significance

# Phase 3: Exponent and weight scale application
result = dot_product × (1 << shared_exp) × weight_group_scale
```

FETB的核心优势：(1) 仅需M个partial sum register（vs element-first需64个）；(2) shift操作仅作用于M个plane sums（vs 64个element sums）；(3) 单个shared accumulator顺序处理所有plane（vs per-element accumulator），硬件面积显著减小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FETB在Anda PE中的硬件实现：adder tree（64:1 reduction）→ 1个partial sum register → 1个shifter → 1个accumulator。每cycle：从Activation Buffer读1个64-bit mantissa bit-plane→64个1b×4b product→adder tree→plane sum→shift and accumulate。M个cycle完成1个64元素group的dot-product。相比Bitlet/Bitlet-X的bit-interleave方案（需要复杂交错控制逻辑），FETB的规整bit-plane layout使控制逻辑极大简化。FETB的局限：仅适合GS为2的幂次（方便adder tree），且需bit-plane layout支持。对于bit-parallel PE（如FIGNA），不需要FETB因为所有mantissa bits在同一cycle处理。

涉及论文标题：
- 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

## Adaptive Execution Flow for SpMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Execution Flow是ACES SpMM加速器的核心运行时调度机制，在运行时动态选择矩阵A的condensing degree以平衡input data reuse和parallel computing efficiency。与传统SpMM加速器使用固定执行流（InP/OutP/ROW）不同，adaptive execution flow根据输入矩阵的sparse pattern自动在三种condensing degree间切换，使每个矩阵band在最佳trade-off点执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ACES Adaptive Execution Flow的运行时调度（以single SpMM A(M×K) × B(K×N)为例）：
```
// Step 1: Band Partitioning (based on CSR offsets)
bands = []
current_band_start = 0
for i in 1..M-1:
    if |row_len[i] - row_len[i-1]| > 10:  // threshold
        bands.append((current_band_start, i-1))
        current_band_start = i
bands.append((current_band_start, M-1))

// Step 2: Per-band condensing degree selection
for each band in bands:
    if band.size() >= 256:  // large band
        best_degree = SAMPLING_PHASE(band)
    else:  // small band
        best_degree = MODERATE  // default

// Step 3: Sampling phase (3 trials × 32 rows each)
SAMPLING_PHASE(band):
    sample_rows = band.first_32_rows()
    for degree in [NONE, MODERATE, AGGRESSIVE]:
        total_time = 0
        for trial in 1..3:
            condense_matrix_A(sample_rows, degree)
            time = execute_SpMM_trial()  // includes MPE mult + APE immediate merge
            total_time += time
        avg_time[degree] = total_time / 3
    return argmin(avg_time)

// Step 4: Execute with selected condensing
for each band in bands:
    condense_matrix_A(band, band.degree)
    // A Fetcher loads condensed columns → Global Buffer
    // B Fetcher loads corresponding B rows → Global Cache
    // MPEs execute scalar-vector multiplications
    // APEs perform immediate merging
```
三种condensing degree对执行流的影响：
- **None**：column-by-column traversal，高B reuse（同column的A元素共享B row），但同row partial fibers多→sync冲突高。
- **Aggressive**：所有非零左移→condensed columns变dense，B reuse低（不同column混合），但sync冲突低。
- **Moderate**：columns分两组各自condense→折中B reuse和sync冲突。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 轻量级condensing adapter硬件（RTL状态机）读取CSR offsets，partition bands，执行sampling，选择condensing degree；(2) 采样结果存入少量寄存器；(3) 每个band的condensing通过在读取A非零元素时按selected degree重排column traversal order实现，不修改原始CSR存储。Sampling开销已包含在ACES总执行时间中。对比SPADA的Window-based Adaptive (WA) 执行流（introduces collective dependency among multipliers），ACES每个MPE独立工作，消除collective dependency。对比Flexagon的offline analysis-based execution flow selection，ACES完全在线（runtime sampling），无需pre-processing。

涉及论文标题：
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## PureFiber Cache Replacement Policy

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PureFiber是ACES专门为SpMM加速器global cache设计的并发感知缓存替换策略。与传统locality-only策略（如LRU、Belady's OPT）不同，PureFiber在驱逐cache line时同时考虑RD（Reuse Distance，下次访问的时间距离=locality指标）和FD（Fiber Density，该line所属fiber的总cache line数=concurrency指标），目标是最小化因cache miss导致的total stall cycles。

一个关键设计概念是 **Pure Fiber**：指一个fiber（B行或C partial row）的所有所需cache lines在当前访问中全部hit（无任何miss）。PureFiber通过优先保留低FD高locality的lines来最大化pure fiber数量。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PureFiber的eviction决策（在cache set满时触发）：
```
// Each cache line maintains:
//   RD (Reuse Distance): time until next request
//   FD (Fiber Density): number of cache lines in this fiber

// On cache line insertion or hit:
cache_line.RD = initialize_or_reset()   // from prefetch info
cache_line.FD = fiber_total_lines()     // known from fiber metadata

// On each cycle, all cached lines:
cache_line.RD -= 1  // decrement until next reuse

// On cache eviction (cache set full):
victim = argmax(cache_line.RD + cache_line.FD)  // primary score
if multiple candidates with same score:
    victim = argmax(cache_line.FD)  // tie-break: evict higher FD first
evict(victim)
```
设计原理：
- 高RD值→该line很久后才会被请求→驱逐它对未来performance影响小（类似Belady's OPT驱逐max reuse distance）。
- 高FD值→该line属于一个大fiber（多cache lines并发请求）→保留它不足以使整个fiber成为pure fiber（还需同时保留该fiber的其他lines）→驱逐该line "sacrifices"一个不太可能成为pure的fiber。
- 低FD值→该fiber仅需少量lines即可成为pure→优先保留→提高pure fiber达成率。

以图5案例（cache容量8 lines，B fibers B0-B4各有2/5/2/6/4 lines）：
- Belady's OPT（仅RD）：1个pure fiber
- PureFiber（RD+FD）：3个pure fiber ——通过优先驱逐高FD fiber lines，保留低FD的B0、B2 lines使其成为pure。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 每个global cache line附加RD counter（每次hit或insert时从prefetch info初始化，每cycle递减）和FD register（存储该fiber的总line数）；(2) eviction时计算RD+FD，选最大值驱逐；(3) 硬件overhead轻量——RD decrement per cycle可pipelined实现。PureFiber在ACES的1MB 16-bank 16-way set-associative cache中实现，管理B row fibers和C partial fibers。实验显示：与LRU比，PureFiber在不同condensing degree下提供+17-20% speedup；与RD-only（仅locality）比，提供+10% speedup和额外-8.8% cache stall reduction。

涉及论文标题：
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## Immediate Merging in SpMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Immediate Merging是ACES SpMM加速器中的pipelined merging策略：每当MPE完成一个scalar-vector乘法产出partial output fiber（到SQ），对应APE立即开始将其与global cache中已有的同row partial fiber合并，而非等待所有乘法完成后再集中merge。这实现了multiplication和merging的pipeline overlap，且row-granularity partial results避免了OutP的large partial matrix transfer开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ACES Immediate Merging的执行过程：
```
// MPE side:
mpe_execute(a_elem, B_fiber):
    partial_C_fiber = []  // sorted by coordinate
    for each (coord, val) in B_fiber:
        partial_C_fiber.append((coord, a_elem.val * val))
    SQ[my_id].enqueue(partial_C_fiber)  // push to selective queue

// APE side (triggered when SQ not empty and APE idle):
ape_immediate_merge():
    fiber_from_SQ = sync_scheduler.select_fiber(SQ[my_id])
    row_id = fiber_from_SQ.row_id
    
    cached_fiber = global_cache.lookup(row_id)  // look for existing partial fiber
    
    if cached_fiber exists:
        // Two-pointer walk merge (both fibers sorted by coordinate)
        merged = []
        p1 = 0, p2 = 0
        while p1 < len(fiber_from_SQ) and p2 < len(cached_fiber):
            if fiber_from_SQ[p1].coord == cached_fiber[p2].coord:
                merged.append((coord, fiber_from_SQ[p1].val + cached_fiber[p2].val))
                p1++; p2++
            elif fiber_from_SQ[p1].coord < cached_fiber[p2].coord:
                merged.append(fiber_from_SQ[p1]); p1++
            else:
                merged.append(cached_fiber[p2]); p2++
        // Append remaining
        merged.extend(fiber_from_SQ[p1:])
        merged.extend(cached_fiber[p2:])
        global_cache.write(row_id, merged)  // write back merged fiber
    else:
        // No matching fiber in cache → write directly
        global_cache.write(row_id, fiber_from_SQ)
    
    // If cache full → PureFiber eviction
    // If cache miss during lookup → NB Buffer handles non-blocking
```

关键特征：(1) MPE和APE形成一比一pipeline——MPE产出partial fiber后立即可被对应APE merge，无需等待其他MPE；(2) row-granularity merge使得partial fibers size小，可在on-chip cache内完成，减少DRAM writeback；(3) Synchronization Scheduler从SQ中选择无sync冲突的fiber分配。

术语一般如何实现？如何使用？
Immediate merging依赖：一对一MPE-APE配对（16 MPE→16 APE）、SQ缓冲、Synchronization Scheduler的sync避免、以及global cache的partial fiber存储。当cache满时，部分partial fibers被writeback到DRAM → 全部乘法完成后进入final merging stage（由Huffman Tree Merging调度）。与SpArch的pipelined merge（high-radix merger，但集中式bottleneck）相比，ACES的distributed APEs实现更fine-grained parallelism。与SPADA的WA（multipliers collective dependency → merge等所有window完成）相比，Immediate merging消除了等待。

涉及论文标题：
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## Huffman Tree Merging for SpMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Huffman Tree Merging是ACES在SpMM的final merging阶段使用的合并调度算法。将每个output row的partial fibers视为Huffman tree的叶子节点（权重=该fiber含非零元素数），通过优先队列每次合并两个最小weight的fiber，最小化总memory load/store操作和比较次数。该技术源自SpArch [61]，ACES在此基础上增加merging scheduler动态避免sync冲突。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ACES Huffman Tree Merging的调度伪代码（以单output row所有partial fibers的final merge为例）：
```
// Input: partial_fibers = list of (fiber, weight=nnz_count)
// Output: single merged output row fiber

priority_queue = MinHeap()
for each (fiber, weight) in partial_fibers:
    priority_queue.push(weight, fiber)

while priority_queue.size() > 1:
    // Extract two smallest fibers
    (w1, fiber1) = priority_queue.pop()
    (w2, fiber2) = priority_queue.pop()
    
    // APE merges them (two-pointer walk merge)
    merged = ape_merge(fiber1, fiber2)  // coordinate-sorted merge + add
    merged_weight = nnz_count(merged)   // ≤ w1+w2 due to coordinate overlaps
    
    priority_queue.push(merged_weight, merged)

final_output_row = priority_queue.pop().fiber
```
设计原理：优先合并小fiber → 较小的intermediate results → 较少的data movement和比较。例如图4：Row C0有fibers (w=23,10,5,13,5,18) → 先合并两个w=5 → internal node w=7 → 继续与w=10合并 → ... → 最终root w=47。对比naive sequential merge（按任意顺序）减少total operations。

术语一般如何实现？如何使用？
ACES实现：(1) Merging Scheduler维护一个priority queue（MinHeap），初始填入所有partial fibers的weight；(2) 每次pop两个最小weight fiber→封装为merge task→放入small task buffer→分配给下一个可用APE执行；(3) APE合并完成后将merged fiber weight重新push入优先队列；(4) 重复至该row完成→复用同一priority queue处理下一row。Merging Scheduler还检查sync冲突：若两个最小weight fiber对应的merge task可能与其他APE冲突，则选择次小weight组合（adhere to Huffman order within sync constraint）。相比直接全部fiber sequential merge，Huffman tree scheduling在ACES中贡献了显著的off-chip traffic reduction。

涉及论文标题：
- 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

## Partial Tensor Parallelism (部分张量并行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Partial Tensor Parallelism（部分张量并行）是ExeGPT提出的一种LLM推理trade-off机制。与传统的tensor parallelism（对所有GPU应用相同的TP degree）不同，partial TP允许tensor parallelism仅应用于GPU子集：固定TP degree（如TP=4），但仅对前K个GPU应用tensor parallelism，剩余N-K个GPU以TP=1独立执行。这引入了一个额外的控制维度——applied GPU count（参与TP的GPU数量）独立于TP degree——使得可以在latency和throughput之间进行细粒度trade-off。增加TP-applied GPU数量（给定fixed TP degree）减少pipeline depth→降低latency，但因all-reduce同步开销增加→降低throughput。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Partial TP在ExeGPT中的执行（以16 GPU, TP degree=4, applied on 8 GPUs为例）：
```
GPUs 0-7 (TP-applied group):
  for each transformer layer L in model_layers:
    // Split layer along hidden dimension
    [W_q, W_k, W_v, W_o, W_ff1, W_ff2] split across 4 GPUs (within each 4-GPU TP group)
    // Forward pass:
    partial_q = W_q_partial @ input  // on each GPU
    all-reduce(partial_q) → q  // sync #1
    partial_k = W_k_partial @ input
    all-reduce(partial_k) → k  // sync #2 (only for decoder layers)
    partial_v = W_v_partial @ input  
    all-reduce(partial_v) → v  // sync #3 (only for decoder layers)
    attention_output = attention(q, k, v)
    partial_o = W_o_partial @ attention_output
    all-reduce(partial_o) → o
    // FFN similarly with split weights and all-reduce
  // each layer has 2 all-reduce for encoder, 3 for decoder

GPUs 8-15 (non-TP):
  for each transformer layer L:
    // Full layer on single GPU, no all-reduce
    output = full_layer(input)  // TP=1 execution
```
Monotonicity: 给定fixed TP degree，增加TP-applied GPU count → pipeline stage数减少（pipeline depth减小）→ latency单调递减，但因synchronization overhead占比增大→ throughput单调递减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Partial TP的实现：(1) 在FasterTransformer中，TP group由用户配置的TP degree决定，ExeGPT的XRunner修改了group assignment逻辑：前K个GPU按TP degree分组，剩余GPU独立执行；(2) Pipeline parallelism在此基础上将不同stage映射到不同GPU组；(3) XScheduler的branch-and-bound算法将TP degree作为固定常数，对applied GPU count进行搜索——多次运行算法（每次不同TP degree值）后选取最优解；(4) 适用场景：需要同时控制pipeline depth和synchronization overhead的multi-GPU推理场景，特别是latency bound较紧但又不愿完全牺牲throughput时。

涉及论文标题：
- 58-ExeGPT- Constraint-Aware Resource Scheduling for LLM Inference.pdf

---

## Decoder Micro-batch for Pipeline Bubble Reduction (减少流水线气泡的解码微批次)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Decoder Micro-batch是ExeGPT在WAA Scheduling中引入的控制变量，将较大的decoding batch (B_D)拆分为多个更小的micro-batch (B_m)以pipeline方式overlap执行。目的是减少pipeline bubble——在pipeline parallel decoding中，后续GPU必须等待前一GPU完成整个batch的decoding iteration才能开始，导致pipeline深度随batch size增大而增大、latency增加。Micro-batch将大batch拆小，使pipeline stage间可以更早开始overlap执行。例如，B_D=32 with B_m=8时，将32个query拆为4个micro-batch of 8，GPU2可在GPU1完成第一个micro-batch后立即开始，而非等待全部32个完成。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Decoder Micro-batch的timeline效果（WAA, 3 decoder GPUs, B_D=32, B_m=8）：
```
// 无micro-batch (B_m=32):
GPU2: |===== Dec iter 1 (batch 32) =====|===== Dec iter 2 (batch 32) =====| ...
GPU3:    |===== Dec iter 1 (batch 32) =====|===== Dec iter 2 =====| ...
GPU4:       |===== Dec iter 1 (batch 32) =====|===== Dec iter 2 =====| ...
// Pipeline depth per token: 7 stages (3 GPUs × 2 iterations + bubble)
// Large bubble between GPU2 finishing iter 1 and GPU4 starting iter 2

// 有micro-batch (B_m=8, 4 micro-batches):
GPU2: |mb1|mb2|mb3|mb4|mb1|mb2|mb3|mb4| ...
GPU3:    |mb1|mb2|mb3|mb4|mb1|mb2|mb3|mb4| ...
GPU4:       |mb1|mb2|mb3|mb4|mb1|mb2|mb3|mb4| ...
// Pipeline depth per token: ~3.67 stages (reduced overlap gap)
// Smaller bubbles, reduced latency
```
Trade-off: 减小B_m→更多micro-batch→更少pipeline bubble→更低latency；但更多micro-batch→更多kernel launch overhead和更小的per-kernel batch→更低GPU利用率→更低throughput。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 在FasterTransformer的pipeline execution中插入micro-batch loop——将B_D个query的decoding batch切分为ceil(B_D/B_m)个micro-batch，每个micro-batch独立执行attention+FFN，micro-batch间通过pipeline overlap；(2) Micro-batch size B_m受限于：下限=1（per-query执行，throughput最低），上限=B_D（退化回无micro-batch状态）；(3) XScheduler将B_m作为控制变量参与branch-and-bound搜索，利用其monotonicity（B_m减小→latency降低、throughput降低）；(4) 适用于WAA Scheduling中decoding batch较大（B_D值大）而latency bound较紧的场景。

涉及论文标题：
- 58-ExeGPT- Constraint-Aware Resource Scheduling for LLM Inference.pdf

---

## Computation-Communication Overlap via Dual CUDA Streams (双CUDA Stream计算-通信重叠)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Computation-Communication Overlap是分布式深度学习训练的基础优化技术，利用GPU硬件特性——计算核（CUDA Cores/Tensor Cores）和通信引擎（NVLink/InfiniBand NIC）是独立的硬件单元，可并行工作。通过在独立的CUDA Stream中分别执行计算kernel和通信kernel（NCCL collective），使通信时间被计算时间"隐藏"。Concerto的运行时（Runtime）将此机制自动化：编译器生成优化后的拓扑序列后，轻量运行时遍历拓扑序列，将计算算子dispatch到默认CUDA Stream（由GPU SM执行），通信算子dispatch到专用通信CUDA Stream（由NCCL + NIC执行），实现自动overlap，无需手工管理异步通信。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Concerto的运行时执行模型（以GPT backward pass为例）：
```
// 编译器生成的拓扑序列（已优化）
topo_order = [MatMul_grad_W, all_reduce_g, LayerNorm_grad, MatMul_grad_X, getitem, view, all_reduce_h, ...]

// 运行时dispatch（双CUDA Stream）
compute_stream = torch.cuda.default_stream()
comm_stream = torch.cuda.Stream()

for op in topo_order:
    if op.is_communication():
        with torch.cuda.stream(comm_stream):
            nccl_collective(op)  // all-reduce / all-gather / reduce-scatter
    else:
        with torch.cuda.stream(compute_stream):
            op.execute()

// End-of-communication marker:
// 当compute_stream需要通信结果时，插入event等待
// 在编译阶段Decoding已确保通信被调度在计算之前（避免SM被占满延迟launch）
```
关键设计：
1. **拓扑序保证正确性**：编译器确保使用通信结果的计算节点在拓扑序中位于通信之后（依赖保留），无需runtime做额外同步。
2. **End-of-Communication Marker**：对ZeRO-2（optimizer末尾的同步all-gather），Concerto引入异步返回机制——允许计算图直接返回未同步的通信tensor，到下次计算图使用时（下一forward）再同步，从而将all-gather重叠到下一forward的前向计算中。
3. **辅助算子前移**：getitem/view等零GPU时间的辅助算子被提前到相邻通信之前（在Decoding阶段），腾出更多通信融合空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
传统的手工实现（Megatron-LM, PyTorch DDP, DeepSpeed）需要开发者显式管理CUDA Stream和同步——调用ncclAllReduce时指定stream，用cudaEventRecord/cudaStreamWaitEvent控制依赖。Concerto将这一过程完全自动化在编译器中。PyTorch本身的torch.cuda.Stream API和NCCL（NVIDIA Collective Communications Library, https://github.com/NVIDIA/nccl）提供了底层支持——NCCL v2.18+支持per-stream collective操作，允许在同一GPU上同时进行多个独立的collective通信。Concerto当前仅用2个Stream（计算+通信各1），论文讨论未来可扩展到3类资源（compute, intra-node comm, inter-node comm）重叠不同层次的通信。

涉及论文标题：
- 37-Concerto- Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning.pdf
- 81-Klotski- Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline .pdf

Klotski 将此技术扩展到 MoE offloading 推理场景，使用四个 CUDA Stream 实现更细粒度的 I/O-compute overlap：(1) weight prefetch stream——预取下一层权重（gate + hot experts）；(2) expert transfer stream——根据 gate 结果实时传输未被 prefetch 的冷 experts；(3) KV cache prefetch stream——预取下一 batch 的 KV cache；(4) KV cache store stream——存储当前 batch 的 KV cache。四 stream 均异步执行，通过 `sync()` 函数在各关键计算点前同步所需数据。例如 expert layer 计算前 sync(weight_prefetch_stream) 确保 hot experts 已到位，非 expert layer 每个 batch 前 sync(load_cache_stream) 确保 KV cache ready。

## Decomposition Degree (分解度 N)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Decomposition Degree（N）是Concerto auto-decomposition中的关键超参数，表示将一组计算算子（Decomposition Context）沿某一轴切分为多少个partition（chunk）。N越大，通信被切分得越细（每个chunk的通信量为TC/N），与计算的交错overlap机会越多——前驱计算可提供(N-1)/N比例的overlap时间，后继计算同理。但N增大也带来开销：(1) 硬件利用率下降——分解后单个子算子的并行度降低、Tensor Core利用率下降、kernel launch次数增加；(2) HBM traffic增加——需要额外读写中间结果。因此N的选择需要在overlap收益和分解开销间权衡。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以GPT Feed-Forward的auto-decomposition为例（all-gather的前驱LayerNorm和后继MatMul1→GeLU→MatMul2均被分解为N chunk）：
```
// N=4, 原始算子执行时间: T_LN=2, T_AG=8, T_MM1=3, T_GeLU=1, T_MM2=3
// 重叠效率衰减因子 α=1.2

// 分解后Pipeline执行:
Stream_Compute: [LN0][LN1|MM1_0][LN2|GeLU0|MM2_0][LN3|MM1_1|GeLU1|MM2_1][MM1_2|GeLU2|MM2_2][MM1_3|GeLU3|MM2_3]
Stream_Comm:                [AG0            ][AG1                      ][AG2            ][AG3            ]

// cost计算 (Section 5.3公式):
TPRE = T_LN = 2, TPOST = T_MM1 + T_GeLU + T_MM2 = 7
// cost1 = TC - α*(N-1)*(TPRE+TPOST)/N = 8 - 1.2*3*9/4 = 8-8.1 = -0.1
// cost2 = TC/N - α*(N-1)*TPRE/N = 2 - 1.2*3*2/4 = 0.2
// cost3 = TC/N - α*(N-1)*TPOST/N = 2 - 1.2*3*7/4 = -4.3
// final cost = max{cost1, cost2, cost3, 0} = max{-0.1, 0.2, -4.3, 0} = 0.2
```
Concerto通过micro-benchmark实测得到α≈1.2（分解后算子性能下降约20%），涵盖三类算子：(1) General Matrix Multiply（~18.2%降速），(2) Batch Reduction（~21.9%），(3) Element-wise（~23.8%）。N的取值最终由策略选择ILP求解器自动决定——cost最低的策略被选中，若分解总开销超过非分解则选择不分解策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Concerto的分解执行通过将大算子替换为N个并行子算子实现。例如，将MatMul(input_shape=[B*S, H], weight=[H, 4H])沿batch轴N=4分解为4个MatMul(input_shape=[B*S/4, H], weight=[H, 4H])，然后用CombineSpec指定的合并函数（如gather或reduce）合并结果。在CUDA层面，这等价于将原本一个CUBLAS gemm调用拆为N个较小的gemm调用。论文的benchmark显示：N从1增至32时，GPT Feed-Forward的Achieved TFLOP/s从~225降至~175（约22%下降），HBM Traffic从~125GB增至~210GB（约68%增加）。因此N通常取较小值（4-16），实际值由cost model自动决定。

涉及论文标题：
- 37-Concerto- Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning.pdf

## Bucket Communication in Distributed Data Parallelism (分布式数据并行中的桶通信)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bucket Communication是分布式数据并行训练中广泛使用的通信优化技术。基本思想：将模型的所有参数梯度按size分组（"分桶"），当某一桶中所有参数的梯度都在backward中计算完成时，立即对该桶发起异步all-reduce（而非等所有参数梯度计算完才统一通信），从而实现backward计算与梯度all-reduce通信的overlap。PyTorch DDP默认bucket size为25MB，但此值并非对所有模型最优——Concerto论文实验显示GPT 2.5B在bucket size=400MB时overlap率更高（迭代延迟更低），VGG19在70-200MB区间最优。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PyTorch DDP的bucket通信执行流程：
```
// DDP初始化
bucket_size = 25MB  // 默认
buckets = bucket_partition(model.parameters(), bucket_size)

// Backward pass
for param in reversed(parameters_in_backward_order):
    param.grad = compute_gradient(param)
    bucket = find_bucket(param)
    mark_param_ready_in_bucket(bucket)
    if bucket.all_params_ready():
        // 异步发起all-reduce
        torch.distributed.all_reduce(bucket.grads, async_op=True)

// Optimizer step前
for bucket in buckets:
    wait_all_reduce(bucket)  // 确保所有通信完成
    update_parameters(bucket.params)
```
DeepSpeed ZeRO的bucket机制更复杂：(1) ZeRO-2：weight不分片但optimizer state分片，需要all-gather收集optimizer state → 计算更新 → reduce-scatter分散结果；(2) ZeRO-3：weight/gradient/optimizer state全分片，forward/backward每层前all-gather weight → 计算 → 丢弃weight → 下一层，反向all-gather weight → 计算梯度 → reduce-scatter梯度。ZeRO的bucket类似DDP但涉及多种通信原语（all-gather + reduce-scatter），调度更复杂，固定策略难以在所有场景下最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bucket通信在NCCL层面实现为collective communication over multiple streams。Concerto通过RCPSP solver + communication fusion自动确定最优的通信调度策略和融合决策，优于手工设置的固定bucket size——编译器层面的全局优化可发现手动难以察觉的overlap机会（例如发现backward pass中某些与all-reduce无关的计算可以在通信等待期间执行）。实际效果：Concerto在ZeRO-2上相比DeepSpeed（固定bucket）最大加速42.9%，在非NVLink环境下（通信更慢）优势更显著。

涉及论文标题：
- 37-Concerto- Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning.pdf

## Consumption-Centric Subgraph Execution

术语解释
Consumption-Centric Subgraph Execution 是一种从输出端（consumer）反向推导各层tile size和数据更新offset的子图执行方案，替代传统从输入端（producer）一次生产所有中间结果的production-centric方案，解决多分支子图中因kernel size/stride不同导致的unbalanced data production和冗余缓存问题。

术语是什么？
Consumption-Centric（以消费为中心）是Cocco提出的子图执行调度范式。在传统production-centric方案中，producer layer以自身tile size一次性生产所有数据供多个consumer使用，但不同分支的kernel size和stride不同会导致某些consumer暂时无法消费的数据被额外缓存（Figure 4a示例：一个3×3/s=2 conv和1×1/s=1 conv并行分支，5×5 input生产后1×1分支仅有部分数据被即时消费，其余占buffer浪费）。Consumption-centric方案改为从output node（子图的最终输出层）出发，确定其output tile size后，沿DAG反向拓扑序遍历所有producer nodes，使用LCM操作对齐不同consumer的input offset需求，推导每个producer的tile size（取所有consumer所需的最大值）和data update offset。

从kernel调度角度拆解：
Cocco的consumption-centric三阶段flow（以ResNet某3-node子图{Conv(-1), Conv(0), Add(1)}为例，1D简化描述）：

```
// Stage-1: 确定输出node的tile size（类似传统单层scheduling）
// 对子图的output node(s)，按PE利用率和buffer约束选择:
x(output) = Δ(output) = 2  // output tile size为2

// Stage-2: 反向推导各producer的Δ和x（reverse topological order）
// 对每对(producer u, consumer v):
for (u in reverse_topo_order):
    // LCM对齐来自不同consumers的offset需求
    Δ(u) = lcm_{v in children(u)} { Δ(v) * s(v) }
    // tile size取所有consumers所需的最大input size
    x(u) = max_{v in children(u)} { F(v) + (Δ(u)/s(v) - 1) * s(v) }
    
    // 示例：output → node(0)(F=3,s=2): Δ(0)=lcm{2*2}=4, x(0)=3+(4/2-1)*2=5
    //      node(0) → node(-1)(F=5,s=2): Δ(-1)=lcm{4*2}=8, x(-1)=5+(8/2-1)*2=11

// Stage-3: 确定upd_num和执行序列
// upd_num表示每elementary operation中各node的memory更新次数
// 满足 upd_num(v)*Δ(v)*s(v) = upd_num(parent)*Δ(parent)
// 取最小co-prime解得到最小elementary operation
upd_num(-1)=1, upd_num(0)=1, upd_num(1)=2 (co-prime解)
// 表示：1次producer(-1) update → 1次node(0) update → 2次output(1) update
```

实际的2D-CONV扩展类似，额外处理H和W两个维度的LCM推导。

术语一般如何实现？如何使用？
Consumption-centric flow的实现集成在Cocco的simulator中（基于Timeloop/MAESTRO修改）。在编译器/部署流程中：(1) 编译时确定子图划分后，对每个子图调用consumption-centric flow推导各layer的tile size、内存分配和执行序列；(2) 生成子图级的elementary operation序列（类似传统layer级tiling的指令序列，但跨多个layer）；(3) 运行时，硬件按生成的序列执行：output tile drive → 反向推导的producer tile更新 → PE阵列计算 → MAIN/SIDE region数据更新。该方案对任意拓扑的DAG（包括NASNet和RandWire等不规则网络）均适用，因为三阶段flow仅依赖DAG的拓扑和layer kernel参数（F, s），不假设特定网络结构。

涉及论文标题：
- 19-Cocco- Hardware-Mapping Co-Exploration towards Memory Capacity-Communication Optimization

## Subgraph-Level Execution for DNN Accelerators

术语解释
Subgraph-Level Execution（子图级执行）是指DNN加速器将多个连续layer作为一个融合的子图（subgraph）整体执行，在on-chip buffer内完成inter-layer intermediate data的全复用，避免逐层写回DRAM再读出的冗余外部访存，是相对于传统layer-level execution的更高层次优化。

术语是什么？
在DNN加速器部署中，Layer-Level Execution逐层执行每个layer（load weights → load input activations → compute → write output to DRAM → next layer），intermediate data必须经DRAM往返，EMA（External Memory Access）随模型深度线性增长。Subgraph-Level Execution将若干连续layer融合为一个子图，在子图内部：input从DRAM加载一次→所有融合的layer在on-chip buffer内完成数据生产和消费→最终output（或需跨子图复用的中间结果）才写回DRAM。Figure 3显示fusing 3 layers（L=3）即可减少EMA 42.3%~74.7%、BW requirement 26.8%~67.8%。融合更多层（L=5 vs L=3）的边际收益递减（ResNet50仅进一步减少13.6% EMA），因为更大的子图要求更大buffer但也提供更多data reuse机会，存在最优trade-off点。

从kernel调度角度拆解：
Subgraph-level execution的典型执行流程（以ResNet50某3-layer subgraph为例）：
1. **子图加载**：从DRAM加载subgraph inputs（首层输入activations）和所有layer weights到global buffer和weight buffer。
2. **Tile级执行循环**：
   a. 确定output tile（consumption-centric flow的Stage-1）。
   b. 反向推导各layer所需的input tile范围和update offset。
   c. PE阵列按MAIN region数据计算当前tile → 结果写入MAIN/SIDE region。
   d. Sliding window：更新overlap数据（SIDE→MAIN, new→SIDE），移动至下一tile。
3. **子图内全复用**：intermediate data在MAIN/SIDE region间流转，不经DRAM。只有被后续子图依赖的layer output才writeback到DRAM。
4. **子图间调度**：按topo序执行子图，prefetch下一个子图的weights（overlap计算和数据传输）。

术语一般如何实现？如何使用？
Subgraph-level execution的实现取决于加速器硬件能力：(1) 需要灵活的on-chip buffer管理（如Cocco的buffer region manager）以在buffer内为不同layer维护独立的数据区域；(2) 需要子图级tiling和调度能力（编译器/调度器需推导跨layer的tile依赖关系）；(3) 已有工作包括Fused-CNN[4]（手工规则融合，仅适用plain CNN）、SR-CNN[38]（selective caching融合）、LCP[42]（按cluster融合）、Irregular-NN[73]（DP-based partition但搜索空间受限）、DNNFuser[29]（RL-based但针对1D layer-fusion）。Cocco将subgraph execution与GA-based partition co-exploration结合，支持任意拓扑的自动最优划分。

涉及论文标题：
- 19-Cocco- Hardware-Mapping Co-Exploration towards Memory Capacity-Communication Optimization

## Point-to-point DMA between Heterogeneous Accelerators

术语解释
Point-to-point DMA between accelerators是指在多加速器系统中，一个加速器直接通过DMA将数据写入另一个加速器（或DRX）的内存空间，而不经过host CPU系统内存中转的数据搬移机制。DMX通过dma-buf API和PCIe fabric特性实现加速器-DRX-加速器间的直连DMA传输。

术语是什么？
Point-to-point DMA（P2P DMA）是一种使PCIe设备之间可以直接传输数据而无需CPU介入的机制。在标准PCIe拓扑中，设备间的数据交换通常需要CPU作为中介（device→CPU memory→device），但PCIe规范支持peer-to-peer transaction——一个PCIe endpoint可直接访问另一个PCIe endpoint的BAR（Base Address Register）空间。Linux内核通过dma-buf API和GEM提供P2P DMA的软件支持：dma-buf是内核中用于在多个设备间共享buffer的框架，允许一个设备访问另一个设备的DMA buffer。DMX利用P2P DMA实现加速器↔DRX间的直接数据搬移：加速器1完成后发中断，driver配置DMA将其输出直接写入DRX的RX数据队列，DRX处理后再通过DMA将重构数据直接发送到加速器2，全程不经CPU系统内存，消除CPU的PCIe upstream带宽瓶颈。

从kernel调度角度拆解：
DMX中P2P DMA的端到端执行流程（以Accel1→DRX1→Accel2为例）：

```
// Step 1: Accel1完成kernel执行，发中断到CPU
Interrupt_from_Accel1 → CPU_driver_interrupt_handler()

// Step 2: Accel1 driver查找目标Accel2对应的RX队列
rx_queue = drx1.find_rx_queue(target_accel=Accel2)
buf_offset = rx_queue.next_free_buffer()
drx1_driver.share_rx_offset(accel1_fd, rx_queue.id, buf_offset)

// Step 3: 配置Accel1→DRX1的P2P DMA
dma_buf = dma_buf_export(accel1_output_buffer)
dma_buf_map_attachment(dma_buf, drx1_device)
configure_dma_engine(accel1_dma, src=accel1_output, dst=drx1_rx_buf)
dma_async_memcpy()  // Accel1 DMA直写DRX1的RX队列

// Step 4: DRX1硬件从RX队列读取，执行数据重构kernel
drx1_execute_restructuring_kernel(rx_queue, restructuring_ops)
// 结果写入TX队列

// Step 5: DRX1完成，发中断，配置DRX1→Accel2的P2P DMA
Interrupt_from_DRX1 → CPU_driver
configure_dma_engine(drx1_dma, src=drx1_tx_buf, dst=accel2_input)
dma_async_memcpy()  // DRX1 DMA直写Accel2

// Step 6: Accel2执行下一个应用kernel
accel2_execute_application_kernel()
```

在broadcast模式下Step 5扩展为back-to-back多次P2P DMA到多个目标；在all-reduce模式下多个源DRX→单一目标DRX（含目标DRX执行reduction操作）。

术语一般如何实现？如何使用？
P2P DMA依赖PCIe ACS (Access Control Services) 配置和Linux dma-buf子系统。关键步骤：(1) 内核启用PCIe ACS P2P转发（或禁用ACS P2P blocking）；(2) DMA引擎支持peer-to-peer物理地址寻址；(3) 使用dma_buf_attach()和dma_buf_map_attachment()建立跨设备DMA mapping。DMX中P2P DMA与DRX数据队列系统结合——RX/TX队列作为中间buffer，DRX在队列间执行数据重构。此设计与GPU Direct RDMA和NVMe P2P DMA一脉相承。

在RELIEF论文的移动SoC架构中，P2P DMA的表现形式为consumer加速器的DMA engine直接从producer加速器暴露的scratchpad memory读取数据，不需要经过主存。关键是加速器scratchpad memory被暴露到系统互联的non-coherent DMA plane上——consumer DMA向producer scratchpad发起read transaction，硬件管理器通过ongoing_reads跟踪读操作以避免producer覆盖仍在转发的数据。此机制是RELIEF数据转发（data forwarding）策略的硬件基础。

涉及论文标题：
- 14-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators
- 15-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators
- 16-RELIEF_Relieving_Memory_Pressure_In_SoCs_Via_Data_Movement-Aware_Accelerator_Scheduling

---

## Laxity

术语解释
Laxity（松弛度）是实时任务调度中的核心概念，定义为任务距离其deadline的时间余量：`laxity = deadline − runtime − current_time`。Laxity为正表示任务可以在当前不执行的情况下仍满足deadline，laxity为负表示任务即使立即开始也将miss deadline。

术语是什么？
Laxity量化了一个任务在必须开始执行前的等待能力。在LL（Least Laxity First）调度中，laxity最小的任务具有最高优先级——因为其可等待的时间最少。当多个任务争用资源时，scheduler根据laxity值分配资源，确保最紧迫的任务优先执行。在RELIEF中，laxity扮演双重角色：(1) 标准排序——任务在就绪队列中按laxity升序排列（laxity越小的越优先）；(2) 可行性检查——is_feasible()通过检查高优先级任务的laxity是否大于候选转发节点的runtime，来决定优先级提升是否会导致deadline miss。Laxity的计算依赖于对task runtime的准确预测（RELIEF中通过compute time预测+memory time预测）。

从kernel调度角度拆解：
以RELIEF Algorithm 2的is_feasible()伪代码为例，展示laxity如何在调度决策中使用：

```
Function is_feasible(ready_queue, fnode, index):
    can_forward = True
    for node in ready_queue:
        if ready_queue.index(node) == index:  // 找到candidate insert位置
            break
        curr_laxity = node.laxity - curTick()  // 减去当前时间得到剩余laxity
        if not node.is_fwd and curr_laxity > 0:  // 第一个非转发且正laxity的节点
            can_forward = curr_laxity > fnode.runtime  // 其laxity是否足够容忍candidate的执行？
            break
    
    if can_forward:
        // 更新所有更高优先级节点的laxity（减去fnode.runtime）
        for node in ready_queue:
            if ready_queue.index(node) == index:
                break
            node.laxity -= fnode.runtime
    
    return can_forward
```

关键逻辑：因为就绪队列已按laxity排序，第一个正laxity节点的剩余laxity如果大于candidate runtime，则所有后续节点的laxity都足够大，优先级提升是安全的。

术语一般如何实现？如何使用？
Laxity计算需要两个输入：(1) deadline——可通过critical-path method（GEDF-N方式）为DAG中每个node分配sub-deadline，或使用HetSched的SDR（sub-deadline ratio）方式按贡献加权分配；(2) runtime prediction——RELIEF分别预测compute time（固定函数加速器通过查表，误差仅0.03%）和memory time（data movement predictor + bandwidth predictor）。对于bandwidth predictor，论文评估了Last value（最近n个任务的均值）、Average（算术平均）和EWMA（指数加权移动平均）三种方案，Average在mean error(0.68%)和max error(3.95%)上最佳。

涉及论文标题：
- 16-RELIEF_Relieving_Memory_Pressure_In_SoCs_Via_Data_Movement-Aware_Accelerator_Scheduling

---

## Least Laxity First (LL) Scheduling

术语解释
Least Laxity First（LL）是最小松弛度优先调度策略：在每加速器类型的就绪队列中，按laxity = deadline − runtime − current_time升序排列任务，laxity最小的任务最先被调度。LL是uniprocessor实时系统中的最优调度算法（与EDF并列），但在多处理器和多加速器场景下并非最优。LAX [59]是LL的变体——将负laxity任务deprioritize以减少浪费资源在已经无法满足deadline的任务上。

术语是什么？
LL调度在RELIEF中作为基础调度框架：(1) 所有任务按其laxity排序在就绪队列中；(2) RELIEF在LL基础上新增"转发节点优先级提升"——当producer完成时，child节点被标记为forwarding node，优先提升到就绪队列前端（即使其laxity并非最小）；(3) 通过is_feasible()确保提升不会导致其他任务miss deadline。当is_feasible()返回false时，任务回退到标准LL位置。LL的laxity在DAG调度中有两种使用方式：(a) LL-based：每个节点直接使用DAG的整体laxity（不做分布），这使得所有节点都有大量laxity可用，但也可能导致不公平；(b) HetSched-style：按SDR将laxity分布给各节点，限制了单个节点的laxity但也限制了优先级提升幅度。

从kernel调度角度拆解：
以RELIEF Algorithm 1的调度流程伪代码为例：

```
Function RELIEF(finishing_node):
    // Step 1: 标记新就绪的forwarding candidates
    for child in finishing_node.children:
        child.completed_parents += 1
        if child.completed_parents == child.num_parents:  // 所有依赖满足
            child.runtime = predict_runtime(child)
            child.laxity = child.deadline - child.runtime  // LL核心：计算laxity
            index = find_pos(fwd_nodes[child.acc_id], child)  // 按laxity排序插入
            fwd_nodes[child.acc_id].insert(index, child)
    
    // Step 2: 对每种加速器类型，尝试转发节点优先级提升
    for each acc_id:
        max_forwards = num_idle_accelerators[acc_id]  // 限流：不超过空闲实例数
        while not fwd_nodes[acc_id].empty():
            node = fwd_nodes[acc_id].pop_front()  // 按laxity取出最紧迫的forwarding node
            index = find_pos(ready_queue[acc_id], node)  // 按laxity的标准位置
            
            if max_forwards > 0 and is_feasible(ready_queue[acc_id], node, index):
                ready_queue[acc_id].push_front(node)  // 提升到front
                node.is_fwd = true
                max_forwards -= 1
            else:
                ready_queue[acc_id].insert(index, node)  // 回退到LL标准位置
                node.is_fwd = false
```

术语一般如何实现？如何使用？
LL调度需要runtime prediction（RELIEF用查表+predictors）和per-accelerator ready queue数据结构（硬件管理器维护）。与EDF不同，LL需要知道task的runtime（EDF只需要deadline），但LL提供了更精细的紧迫性度量。在加速器丰富的SoC中，LL跟踪相对laxity（而非绝对deadline）更能在多个并发DAG间平衡资源分配。LL的硬件管理器实现在Cortex-A7微控制器上bare-metal C代码：调度延迟在0.15-1.2us范围（取决于加速器类型数量），其中RELIEF略高（约+0.1us）但仍远小于加速器compute time（10us-1545us），可被计算overlap。

涉及论文标题：
- 16-RELIEF_Relieving_Memory_Pressure_In_SoCs_Via_Data_Movement-Aware_Accelerator_Scheduling

---

## Forwarding Node (in Accelerator Scheduling)

术语解释
Forwarding Node（转发节点）是RELIEF调度策略中的核心概念：指一个加速器DAG节点，其所有parent节点都已完成执行，且至少一个parent刚刚完成（即该节点是"新就绪"的）。因其parent的数据仍在producer加速器的scratchpad memory中尚未被覆盖，该节点可以通过硬件数据转发（data forwarding）机制直接从producer scratchpad读取输入数据，无需从主存重新加载。

术语是什么？
Forwarding Node是RELIEF设计的关键创新——调度器通过识别新就绪的child节点并将其标记为forwarding candidates来利用硬件转发机制。Forwarding node的数据读取路径有两种：(1) Forward：consumer DMA engine从producer scratchpad直接读数据（绕过主存）；(2) Colocation：若child与parent使用同一类型加速器，child被直接部署到同一加速器实例上（数据已在scratchpad中，完全消除数据搬移）。RELIEF通过max_forwards（不超过空闲加速器实例数）确保forwarding nodes总是下一个被调度的，保证producer scratchpad数据在覆盖前被有效利用。

从kernel调度角度拆解：
Forwarding node的生命周期：

```
// 当accelerator完成节点n的执行时：
ISR_handler(completed_node=n):
    for child in n.children:
        child.completed_parents += 1
        if child.completed_parents == child.num_parents:
            // child成为"就绪节点"
            // 因为parent n刚完成，child也是"forwarding candidate"
            child.runtime = predict_runtime(child)
            child.laxity = child.deadline - child.runtime
            // 插入候选转发队列fwd_nodes（按laxity排序）
            fwd_nodes[child.acc_id].insert_by_laxity(child)
    // 调度器随后尝试将forwarding nodes提升到就绪队列前端
    schedule_all_accelerators()

// forwarding node被调度后：
launch_node(forwarding_node fn):
    // Consumer driver检查producer_acc和producer_spm字段
    if fn.producer_acc is not NULL:
        // 数据仍在producer scratchpad中
        // Consumer DMA直接从producer SPAD读取（forward）
        dma_read(src=fn.producer_acc.spm[fn.producer_spm], dst=consumer_spad)
        // 硬件管理器increment producer的ongoing_reads计数
        producer_acc.ongoing_reads[fn.producer_spm] += 1
    else:
        // 数据已写回主存，从DRAM加载
        dma_read(src=DRAM_addr, dst=consumer_spad)
    
    // 执行consumer computation
    consumer_acc.execute(fn)
```

术语一般如何实现？如何使用？
Forwarding node的实现需要：(1) 硬件——加速器scratchpad memory暴露到系统互联non-coherent DMA平面，consumer DMA engine支持peer-to-peer read；(2) 软件/硬件管理器metadata——node结构中的producer_acc和producer_spm字段记录producer位置，加速器metadata中的ongoing_reads[]数组跟踪并发读计数以防止数据被覆盖，output[]指向当前活跃输出。在生产就绪系统中，可基于PCIe resizable BAR和Linux P2PDMA接口实现类似机制。

涉及论文标题：
- 16-RELIEF_Relieving_Memory_Pressure_In_SoCs_Via_Data_Movement-Aware_Accelerator_Scheduling

---

## Data Restructuring Kernel (DRX Runtime)

术语解释
Data Restructuring Kernel是在DRX硬件上执行的数据重构运行时程序，由DRX Compiler从高级kernel描述编译为DRX ISA指令流，运行在DRX的decoupled access-execute pipeline上，负责将上游加速器输出重构为下游加速器的输入格式。

术语是什么？
Data Restructuring Kernel是DMX数据平面的核心可执行单元，类似GPU kernel（在CUDA core上执行）或CPU程序（在通用core上执行）。每个Data Restructuring Kernel对应一个特定的数据格式转换操作组合，例如"对(8192,768)频谱矩阵执行Pow→Add→Mul→Div→Log10→Cast得到mel spectrogram"。kernel由DRX Compiler编译为DRX ISA指令流（Loop/Compute/Off-chip Memory/Synchronization四类指令），加载到DRX 64KB I-cache，由Instruction Repeater循环发射执行。DRX支持多个kernel的串行执行（command queue），kernel间通过Synchronization指令和中断同步。

从kernel调度角度拆解：
DRX Runtime中Data Restructuring Kernel的执行流程（以mel spectrogram kernel为例，伪代码级DRX ISA表示）：

```
// 1. Synchronization: 标记kernel开始
SYNC START, group=mel_spec_kernel

// 2. Loop: 配置instruction repeater遍历(8192,768)矩阵
LOOP dim0: base=0, iter=8192, stride=768   // 行遍历
LOOP dim1: base=0, iter=768, stride=1        // 列遍历

// 3. Off-chip Memory: 预取输入tile到scratchpad bank[0]
MEM_LOAD src=ddr4_fft_output, tile_size=(128,128), dst=spad[0]

// 4. Compute: 128 RE lanes并行执行变换操作序列
COMPUTE POW, src1=spad[0], result=spad[1]     // spad[1]=spad[0]^2
COMPUTE ADD, src1=spad[1], src2=const_eps, result=spad[1]
COMPUTE MUL, src1=spad[1], src2=const_scale, result=spad[1]
COMPUTE DIV, src1=spad[1], src2=const_ref, result=spad[1]
COMPUTE LOG10, src1=spad[1], result=spad[1]   // log10
COMPUTE CAST, src1=spad[1], type=fp32_to_int16, result=spad[2]

// 5. Off-chip Memory: 将重构结果写回DDR4
MEM_STORE src=spad[2], dst=ddr4_mel_spectrogram

// 6. Synchronization: 标记kernel完成，触发中断
SYNC DONE, group=mel_spec_kernel
```

实际执行以tile为单位流式处理（tile大小由compiler根据scratchpad容量确定），Loop指令控制tile遍历顺序，Memory指令预取下一tile实现计算-访存overlap。

术语一般如何实现？如何使用？
Data Restructuring Kernel由DRX Compiler从高级语言描述编译生成，通过DRX driver的command queue提交到DRX硬件。DRX driver管理kernel执行顺序（FIFO），处理kernel完成中断并触发下游操作（DMA到目标加速器）。kernel输入来自上游加速器的RX数据队列，输出写入TX数据队列等待DMA到下游加速器。

涉及论文标题：
- 14-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators
- 15-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators

---

## Re-materialization (DNN Memory Optimization)

术语解释
Re-materialization（重物化/重计算）是DNN内存优化的核心技术之一：在forward pass中故意不保存某些中间tensor（evict），当backward pass需要这些tensor计算梯度时，重新执行对应的operator来重新生成它们的值。这是一种以额外计算换取显存空间的策略。

术语是什么？
在DNN训练的计算图中，forward pass的中间激活（activation）必须保存到backward pass才能计算梯度。对于深层网络或大batch训练，这些激活的累积size可能远超GPU显存。Re-materialization通过在forward pass中选择性地不保存某些operator的输出，并在backward需要时重新计算它们，来减少同时驻留显存的tensor数量。例如，U-Net中encoder路径的Conv+ReLU输出的feature map可以不被保存，在backward时重新执行Conv+ReLU来重新生成。

MAGIS将Re-materialization重新表述为图变换规则（Re-materialization Rule，§5.2）：将一个operator A（有多个user）的输出分离——一个user B改为使用重新计算的operator A'（A'与A类型相同、输入相同），从而让A的输出可以在B做forward时被释放。De-re-materialization Rule是其对偶——将两个类型相同、输入相同的operator A和A'合并回单个operator。MAGIS仅对包含memory hot-spot的子图应用re-mat规则，控制搜索空间。

从kernel调度角度拆解：
在GPU上，re-materialization的运行时执行（伪代码）：

```
// Forward pass: 标记某些operator输出不保存（remat_set）
for v in forward_operators:
    output = execute_op(v, inputs)
    if v not in remat_set:
        save_for_backward(v, output)  // 正常保留
    else:
        release(output)  // evict，不保存

// Backward pass: 对remat operator先重计算再求梯度
for v in reversed(forward_operators):
    if v in remat_set:
        saved_inputs = reload_inputs(v)  // 从forward保存的inputs中恢复
        output = execute_op(v, saved_inputs)  // 重计算
    grad = compute_gradient(v, output, upstream_grad)
    if v in remat_set:
        release(output)  // 重计算结果的中间tensor可立即释放
```

在MAGIS中，哪些operator参与re-mat由M-Rules在编译阶段决定，增量调度负责将re-mat引入的额外operator（A'）与原有operator一起进行re-ordering。

术语一般如何实现？如何使用？
成熟的Re-materialization实现：(1) Checkmate [24]：使用Integer Programming (IP)离线求解最优方案，精确但慢；(2) DTR [27]：eager模式下heuristic动态决策——基于tensor size（越大越优先evict）、compute cost（越小越优先recompute）和staleness；(3) XLA [46]：HLO-level贪心算法，按heuristic顺序逐个标记operator为recompute；(4) MONeT [47]：co-optimize re-mat与operator implementation。

在MAGIS中，re-mat通过子图替换实现，与F-Trans、swapping统一优化。代码生成后端将最终决策转为PyTorch代码——re-mat的operator在backward中重新执行。

涉及论文标题：
- 17-MAGIS- Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN

---

## Swapping / Tensor Offloading (GPU-CPU)

术语解释
Swapping（张量交换/卸载）是DNN内存优化的核心技术之一：将GPU显存中暂时不用的tensor异步拷贝到外部存储（通常是CPU内存），当后续operator需要该tensor时再异步拷贝回GPU显存。这是一种以PCIe数据传输开销换取GPU显存的策略。

术语是什么？
当GPU显存不足以容纳所有需要同时活跃的tensor时，Swapping将一部分tensor暂时offload到CPU内存。与Re-materialization不同，Swapping不引入额外计算——tensor的值被保留，只是存储位置从GPU变为CPU。但代价是PCIe数据传输的延迟和带宽限制（如RTX 3090 PCIe Gen3 x16: ~16GB/s，远低于GPU显存带宽~936GB/s）。

MAGIS将Swapping重新表述为图变换规则（Swapping Rule，§5.2）：在operator A与其user B之间插入Store(A)和Load(A)节点——Store将A的输出异步拷贝到CPU内存（pin_memory），Load在B需要时将数据从CPU异步拷贝回GPU。De-swapping Rule是其对偶——移除Store/Load并恢复A→B直连。

MAGIS的关键设计：(1) 使用PyTorch CUDA Stream API实现异步Store/Load，使数据传输与GPU计算overlap；(2) re-ordering策略是Store尽早放置（尽早开始swap-out）、Load尽可能晚放置（刚好在user需要前完成swap-in以隐藏PCIe延迟）。

从kernel调度角度拆解：
在GPU上，swapping的异步执行（以U-Net skip-connection为例，伪代码）：

```
// Forward: encoder输出tensor A (batch×C×H×W)，显存占用大
A = encoder_forward(x)  // GPU计算

// Store: 异步拷贝到CPU（独立CUDA stream，不阻塞compute stream）
with torch.cuda.stream(swap_stream):
    A_cpu = torch.empty(size_A, device='cpu', pin_memory=True)
    A_cpu.copy_(A, non_blocking=True)  // GPU→CPU DMA
del A  // GPU显存释放

// ... forward/backward继续在compute stream上计算 ...

// Load: decoder需要A时，异步拷贝回GPU
with torch.cuda.stream(swap_stream):
    A_gpu = torch.empty(size_A, device='cuda')
    A_gpu.copy_(A_cpu, non_blocking=True)  // CPU→GPU DMA
torch.cuda.synchronize()  // 确保传输完成
decoder_backward(A_gpu, grad)  // 使用swap-in的tensor
```

MAGIS的re-ordering算法（DpSchedule）通过预测PCIe传输时间（size/bandwidth）计算Load的最晚启动时间，确保传输延迟被计算完全隐藏。

术语一般如何实现？如何使用？
成熟的Swapping实现：(1) vDNN [42]：GPU内存虚拟化，自动管理GPU-CPU tensor迁移；(2) Capuchin [38]：基于tensor访问模式的swapping策略；(3) POFO [5]：DP组合re-mat和swapping；(4) ZeRO-Offload [41]：将优化器状态和参数offload到CPU；(5) TFLMS [30]：使用特殊operator和control-flow edge表示swapping。

在MAGIS中，swapping通过Swapping Rule实现，与F-Trans和re-mat统一在M-Rules中优化。使用torch.cuda.Stream实现异步传输，pin_memory确保CPU buffer不被swap out。

## 2.5D Texture Memory (Mobile GPU, 移动GPU 2.5D纹理内存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2.5D Texture Memory（2.5D纹理内存）是移动GPU（如Qualcomm Adreno系列和ARM Mali系列）上的一种特殊内存层次，介于传统1D buffer memory和完全2D texture memory之间，因其每个cache line为固定长度向量（通常4个元素，即"0.5D"），且cache本身具有二维的空间局部性（"2D"），故称"2.5D"。其主要特性：cache line为4元素向量（适合SIMD vector load），cache组织为2D tile结构（height×width），支持自动边界检查和硬件插值（图形渲染特性），通过坐标寻址（coordinate-based addressing）而非线性指针，且是read-only cache。与1D buffer memory（连续线性地址空间、指针寻址、无自动边界检查）形成对比。在移动GPU上，2.5D texture memory的带宽远高于1D global memory——以Snapdragon 8 Gen 2的Adreno 740为例，texture memory bandwidth高达511 GB/s，而global memory bandwidth仅55 GB/s（约9.3×差距）。使用texture memory进行convolution操作可将延迟减少3.5×相比1D buffer memory。但由于纹理内存采用多维坐标寻址（而非线性指针），Reshape操作在2.5D memory中代价极高——需要实际重新排列数据的空间位置（而非像1D memory中仅修改维度元数据），因为空间关系在2.5D memory中对数据局部性至关重要。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在SmartMem中，2.5D texture memory用于存储经过layout优化后的tensor，使consumer kernel能高效访问。具体的kernel层面的内存访问过程（以MatMul消费Conv输出的场景为例）：

```
// 传统1D buffer方式（Conv输出 [M, N, K]）
// MatMul沿k维做reduction，但数据沿N或M连续 → stride访问 → cache miss
for i in range(M):
    for j in range(N):
        sum = 0
        for k in range(K):
            // 1D buffer: offset = i*(N*K) + j*K + k
            // 沿k访问stride=1（好），但i和j的切换需要大stride
            sum += buffer[i*(N*K) + j*K + k] * weight[k][j]
        output[i][j] = sum

// SmartMem 2.5D texture方式（Conv输出按reduction dim k重排layout）
// 将k维映射到texture的宽度方向（0.5D vector = 4 elements）
// 每个work-item处理_k_group = k/4组vector load
for i in range(M):
    for j in range(N):
        sum = 0
        for kg in range(K/4):
            // texture load: read4(x=kg, y=i*N+j)
            // 沿k维连续访问texture宽度 → 2D spatial locality
            vec4 = texture_load(x=kg, y=i*N+j)
            sum += dot(vec4, weight[kg*4:kg*4+4][j])
        output[i][j] = sum
```

在2.5D texture memory中，SmartMem将reduction dimension映射到texture坐标的连续访问方向——如果reduction dim沿texture的x方向存储（stride=1），workgroup访问纹理时获得最佳spatial locality和cache利用率。同时，因为texture memory使用坐标寻址（而非线性偏移），SmartMem的Index Comprehension将Reshape/Transpose的维度变换直接编码为texture坐标映射，无需实际重组数据。

术语一般如何实现？如何使用？
在移动GPU（OpenCL/OpenGL ES/Vulkan）上使用2.5D texture memory的方式：(1) 创建Image2D对象分配texture内存（clCreateImage2D/glTexImage2D），指定数据格式（如CL_RGBA/CL_FLOAT对应4元素vector）；(2) Kernel中使用image read/write指令替代buffer load/store——read_imagef(texture, sampler, (float2)(x, y))一次读取4个float；(3) 利用texture专用L1 cache的2D spatial locality——相邻work-item访问相邻(x,y)坐标时cache命中率高。SmartMem的layout mapping策略——将reduction dimension沿2.5D memory中连续访问的方向存储，并利用4元素vector匹配0.5D宽度——可推广到其他利用移动GPU texture memory的DNN优化框架（如TMModel [ICS 2025]和FlashMem [2026]）。在Qualcomm Adreno上，使用Image2D（2D texture）比Image1D（1D texture）提供约2×更高的吞吐量。关键限制：texture memory是read-only的（对compute shader而言读取高效但写入受限），因此SmartMem主要用于优化consumer的读访问。

涉及论文标题：
- 18-SmartMem- Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile

## Tensor Layout Mapping and Layout-Aware Memory Access (张量布局映射与布局感知内存访问)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tensor Layout Mapping是将编译期选定的tensor逻辑布局映射到物理内存层次（1D buffer或2.5D texture memory）的过程，而Layout-Aware Memory Access是在kernel执行时按照该布局最优方式访问数据的策略。SmartMem的layout mapping解决的核心问题：给定一个含多个reduction dimension需求的tensor（如Figure 5的L0含D1和D3两个reduction dim），如何将其物理存储到2.5D memory中以使所有consumer都能高效访问。具体策略：(1) 对含两个reduction dim的tensor——沿一个reduction dim按k=4元素（0.5D vector长度）partition，每个partition内沿另一个reduction dim连续存储4×|D|大小的chunk；(2) 将partition序列映射到texture坐标的x方向（连续访问方向），将另一个reduction dim映射到y方向；(3) 对无reduction dim的tensor（如被element-wise op消费）——按2.5D的连续访问方向选择映射方式，避免在fused kernel中出现额外的索引计算（因为Add操作可与Conv等fuse）。Layout-aware memory access则确保每个consumer kernel以stride=1方式沿其关心的reduction dimension读取数据——通过texture坐标计算而非线性指针重映射，消除Reshape/Transpose引起的碎片化内存访问。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Figure 6展示优化前后的对比：(a) 原始计算图——FusedLayerNorm输出经Reshape→Transpose→Reshape后送入MatMul，内存访问碎片化；(b) Operator Fusion——将layout op与MatMul融合但数据访问顺序仍复杂（跨stride访问）；(c) SmartMem layout mapping——将MatMul的输入tensor按其reduction dim重新映射到2.5D memory后，MatMul kernel可以沿reduction dim以stride=1顺序访问数据（texture坐标连续递增）。以伪代码表示layout-aware memory access的kernel执行：

```
// 优化后的MatMul kernel（2.5D texture + reduction-dim layout）
// 输入tensor已按MatMul的reduction dim (k) 映射到texture的x方向
// y方向用于batch*M维度的遍历
kernel matmul_with_layout(Image2D input, Buffer weight, Buffer output,
                           int M, int N, int K):
    gid_x = get_global_id(0)  // 映射到M维度group
    gid_y = get_global_id(1)  // 映射到N维度group
    
    // Layout-aware: k沿texture x方向连续 → stride=1 load
    sum = 0
    for kg in range(K/4):
        float_x = kg  // 沿reduction dim连续递增
        float_y = gid_x  // M维固定
        // texture vector load, spatial locality好
        vec4 a = read_imagef(input, sampler, (float2)(float_x, float_y))
        vec4 w = vload4(0, weight + gid_y*K + kg*4)
        sum += dot(a, w)
    output[gid_x*N + gid_y] = sum
```

相比优化前（无layout mapping），数据访问从stride>1的非连续访问变为stride=1的顺序texture访问，SIMD向量load效率提升，cache miss减少——SmartMem论文Figure 9显示Layout Selecting主要减少cache miss count（2.0×平均），因为更好的layout提高了2.5D texture cache的空间局部性。

术语一般如何实现？如何使用？
Layout mapping的实现：(1) 编译期——根据消费该tensor的所有operator的reduction dimension需求生成layout schedule（每个tensor的维度→memory坐标映射关系）；(2) Kernel代码生成期——将layout schedule转换为texture坐标计算公式嵌入生成的GPU kernel代码（替代硬编码的线性地址偏移计算）；(3) 运行时——kernel执行时，memory access按layout schedule确定的顺序和步幅进行。在SmartMem中，此过程集成在DNNFusion的code generation后端中，配合Genetic Algorithm自动搜索最优的tiling shape和unrolling factor。更广泛地，layout-aware memory access原则可适用于任何需要将编译期layout决策落实到kernel执行中的场景——如TVM的tensorization、Triton的block pointer等。

涉及论文标题：
- 18-SmartMem- Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile

涉及论文标题：
- 17-MAGIS- Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN

## Per-token Online KV Cache Quantization Kernel

术语解释
Per-token Online KV Cache Quantization Kernel是Oaken中在运行时对每个新生成token的Key和Value向量执行在线量化的kernel操作，利用offline预计算的per-layer量化参数（thresholds, scales），在token生成pipeline中实时完成分组→量化→编码→写入KV cache的操作。

术语是什么？
在LLM serving的generation阶段，每步生成1个new token→该token经过所有decoder layer计算后产生K和V向量。Per-token Online Quantization Kernel在每层decoder的K/V计算完成后立即执行在线量化：将FP16的K/V向量根据offline预计算的per-layer阈值（T_lo, T_hi）分组→各组分别量化（INT4/INT5）→融合编码（8-bit aligned）→写入KV cache。该kernel的关键特征是：(1) per-token粒度——每个token独立量化，无跨token依赖；(2) online执行——在serving时实时量化，不增加prefill/serving延迟（通过pipeline overlap）；(3) 无online profiling开销——所有量化参数预计算，kernel仅执行查表和算术操作。

从kernel调度角度拆解：
Per-token Online Quantization Kernel的伪代码（以单token单layer的K/V量化为例，对应Quant Engine硬件pipeline）：

```
// Kernel: per_token_quantize_kv
// 输入: K_new, V_new ∈ R^{d_head}（FP16）, layer_idx
// 输出: encoded_K, encoded_V（8-bit aligned format）写入KV cache

KERNEL per_token_quantize_kv(K_new, V_new, layer_idx):
    // Load per-layer quantization parameters from control register
    T_lo = CTRL_REG[layer_idx].T_lo       // inner group上界阈值
    T_hi = CTRL_REG[layer_idx].T_hi       // outer group下界阈值
    scale_mid = CTRL_REG[layer_idx].scale_mid
    scale_out = CTRL_REG[layer_idx].scale_out
    shift = CTRL_REG[layer_idx].shift
    
    for vec in [K_new, V_new]:  // 分别处理K和V
        // Stage 1: Threshold Comparison（分组）
        for i in 0..d_head-1:
            val = vec[i]
            abs_val = abs(val)
            if abs_val > T_hi:
                group[i] = OUTER      // top 4% outliers
            elif abs_val < T_lo:
                group[i] = INNER      // bottom 6%, → 0 (sparse)
            else:
                group[i] = MIDDLE     // 90% inliers
        
        // Stage 2: Quantization per group
        for i in 0..d_head-1:
            if group[i] == INNER:
                quant_val[i] = 0                     // sparse, filled with zero
            elif group[i] == MIDDLE:
                // INT4 uniform quantization
                val_clamped = clamp(vec[i], T_lo, T_hi)
                quant_val[i] = round((val_clamped - T_lo) / scale_mid)  // [0, 15]
            elif group[i] == OUTER:
                // Group Shift + INT5 quantization
                val_shifted = vec[i] - shift
                quant_val[i] = round(val_shifted / scale_out)  // [-15, 15]
        
        // Stage 3: Fused Dense-and-Sparse Encoding
        encoded = []  // 输出8-bit aligned entries
        for i in 0..d_head-1:
            if group[i] == INNER:
                // Inner group: skipped in encoding (zero-filled in final layout)
                continue
            elif group[i] == MIDDLE:
                // [6-bit idx | 1-bit flag=0 | 1-bit: 0(dense)]
                entry = (i << 2) | (0 << 1) | 0
                encoded.append((entry, quant_val[i]))  // 8-bit entry + 4-bit val
            elif group[i] == OUTER:
                // [6-bit idx | 1-bit flag=1 | 1-bit: sign]
                sign = 1 if quant_val[i] < 0 else 0
                entry = (i << 2) | (1 << 1) | sign
                encoded.append((entry, abs(quant_val[i])))  // 8-bit entry + 5-bit val
        
        // Write to KV cache via DMA
        DMA_WRITE(kv_cache_addr, encoded)
```

Dequant Kernel（attention计算时读取）：

```
KERNEL per_token_dequantize_kv(kv_cache_addr, layer_idx):
    // Load per-layer parameters
    T_lo = CTRL_REG[layer_idx].T_lo
    T_hi = CTRL_REG[layer_idx].T_hi
    scale_mid = CTRL_REG[layer_idx].scale_mid
    scale_out = CTRL_REG[layer_idx].scale_out
    shift = CTRL_REG[layer_idx].shift
    
    // DMA Read encoded KV from Device Memory
    encoded_data = DMA_READ(kv_cache_addr)
    
    // Initialize output as zeros (inner group default)
    vec_fp16 = zeros(d_head)
    
    // Dequantize each encoded entry
    for entry, val in encoded_data:
        idx = (entry >> 2) & 0x3F      // extract 6-bit index
        group_flag = (entry >> 1) & 0x1 // extract group flag
        
        if group_flag == 0:  // MIDDLE group
            vec_fp16[idx] = val * scale_mid + T_lo
        else:  // OUTER group
            sign = entry & 0x1
            vec_fp16[idx] = (val if sign == 0 else -val) * scale_out + shift
        // INNER group: already zero (default)
    
    return vec_fp16  // FP16 K或V，送入MPU计算attention
```

术语一般如何实现？如何使用？
Per-token Online Quantization Kernel的实现：(1) 硬件实现（Oaken）——Quant Engine和Dequant Engine以专用硬件pipeline执行上述kernel操作，Threshold Comparator/Scale Calculator/Quantizer/Splitter/Shifter/Decomposer等stage在硬件中并行流水线化，与compute和memory传输overlap；(2) GPU软件实现（Oaken accuracy evaluation开源代码）——在PyTorch中实现量化函数，对每个token的K/V tensor执行masking+quantization+encoding操作，替换原始FP16 K/V后执行标准attention。GPU实现存在量化/反量化overhead（额外的GPU kernel launch和memory access），但用于精度验证；(3) 性能特征——per-token量化延迟为d_head × (quant_cycles_per_entry) / frequency，以d_head=128, 1 cycle/entry, 1GHz为例≈128ns，远小于attention计算延迟（batch=256, context=2K时约μs级），通过pipeline完全隐藏。该kernel是Oaken系统实现"online quantization without profiling overhead"的关键——所有参数量化阈值预计算，online仅执行fast compare-and-quantize操作。

涉及论文标题：
- 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf

## Sub-LUT Partition（子LUT分区）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sub-LUT Partition是PIM-DL提出的将LUT-NN推理中LUT算子（table lookup + reduction）的工作负载分配到DRAM-PIM数千个PE上的并行分区策略。核心目标：利用DRAM-PIM的massive memory-level parallelism，同时避免三大架构限制（constrained host-PIM communication、no inter-PE datapath、load-balancing）。

策略设计：将CCS输出的index matrix沿N dim切分为N/N_{s-tile}个tile（每个tile大小为N_{s-tile}×CB），LUTs沿F dim切分为F/F_{s-tile}个tile（每个tile大小为CB×CT×F_{s-tile}）。PIM PEs被逻辑分组为#PE/(N_{s-tile}×F_{s-tile})个group——同group内PEs共享同一index tile（通过host→PIM广播），跨group同位置PEs共享同一LUT tile（通过host→PIM广播）。每个PE独立计算(N_{s-tile}, F_{s-tile})大小的output tile。

该分区的triple-win设计：
1. **解决L1 (host-PIM communication)**：tile复用提升broadcast bandwidth temporal locality。同group共享index tile，跨group同位置共享LUT tile。
2. **解决L2 (inter-PE communication)**：CT dim（centroid dimension）不在PE间分割→每个PE对给定index tile持有完整的CT维度的LUT数据→无需partial sum merging→零inter-PE通信。
3. **解决L3 (load balancing)**：tile size uniform distribution→所有PE workload identical→最慢PE不成为瓶颈。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以BERT-large FFN1层在UPMEM PIM-DIMM上的Sub-LUT Partition为例，workload shape (N=32768, CB=256, CT=16, F=4096), N_{s-tile}=16384, F_{s-tile}=8：

```
# PIM-DL Sub-LUT Partition Scheme
# HW: 1024 PEs, UPMEM PIM-DIMM (8 DIMMs×128 PEs each)

# Parameters
N_s_tile = 16384   # index tile size
F_s_tile = 8       # LUT/output tile size  
# PE count constraint: #PE = (N/N_s_tile) × (F/F_s_tile) = 2 × 512 = 1024 ✓

num_groups = N / N_s_tile = 32768/16384 = 2    # groups along N dim
pes_per_group = F / F_s_tile = 4096/8 = 512     # PEs per group along F dim

# Step 1: Host→PIM broadcast index tiles
for group_g in range(num_groups=2):
    # Index[g*N_s_tile:(g+1)*N_s_tile, :] shape = (16384, 256)
    index_tile = Index[g*16384:(g+1)*16384, :]
    broadcast_to_PE_group(group_g, index_tile)  # same tile to all 512 PEs in group g

# Step 2: Host→PIM broadcast LUT tiles
for f_idx in range(pes_per_group=512):
    # LUTs[:, :, f_idx*F_s_tile:(f_idx+1)*F_s_tile] shape = (256, 16, 8)
    lut_tile = LUTs[:, :, f_idx*8:(f_idx+1)*8]
    broadcast_to_PE_position(f_idx, lut_tile)  # same tile to PE f_idx in both groups

# Step 3: Parallel PIM execution
for group_g in [0, 1]:
    for pe_j in group_g.PEs:  # 512 PEs in parallel
        # Compute output_tile = LUT_lookup(index_tile_g, lut_tile_j)
        # Shape: (N_s_tile=16384, F_s_tile=8)
        output_tile[g][j] = PIM_LUT_kernel(index_tile, lut_tile)

# Step 4: Host←PIM fetch output tiles
for group_g in [0, 1]:
    for pe_j in group_g.PEs:
        fetch_output_from_PE(group_g, pe_j, output_tile[g][j])

# Final output: (32768, 4096) = 2×512 = 1024 tiles assembled from all PEs
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sub-LUT Partition的实现和使用：
1. **实现**：在PIM-DL Engine的PIM Runtime中实现。Host侧执行sub-LUT partition逻辑——根据model shape和target platform #PE计算合法N_{s-tile}×F_{s-tile}组合（约束：#PE = N/N_{s-tile} × F/F_{s-tile}），在Auto-Tuner中搜索最优组合。PIM侧每个PE的micro kernel独立执行局部LUT lookup。
2. **与GEMM数据分区的区别**：GEMM typical tiling沿K dim（reduction dim）分割产生partial sums需merge（all-reduce），而Sub-LUT Partition的CT dim（analogous to K in GEMM）不分割→无merge overhead→天然适合no-inter-PE-communication的DRAM-PIM。
3. **性能建模**：PIM-DL Auto-Tuner用analytical model估算t_sub-lut = Σ STileSize_x × #PE / BW_x_host，搜索最小化t_sub-lut + t_micro-kernel的参数组合。

涉及论文标题：
- 20-PIM-DL- Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization.pdf

## LUT Load Scheme（LUT加载策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT Load Scheme是PIM-DL在DRAM-PIM PE的on-chip buffer（如UPMEM PE的64KB WRAM）上管理LUT数据加载的策略，用于micro kernel执行时从PE local DRAM bank加载LUT entries到on-chip buffer。由于LUT访问由centroid index驱动（on-demand），加载策略影响buffer利用率、memory bandwidth利用率和数据复用。

PIM-DL定义了三种LUT Load Scheme：
1. **Static Load（静态加载）**：当PE分配的LUT MTile total size ≤ on-chip buffer时，一次性将全部LUT加载到buffer并驻留整个execution。适用条件：CB_{s-tile} × CT × F_{s-tile} ≤ buffer_size。优点：最高数据复用，仅load一次。缺点：F_{s-tile}受buffer size严格限制。
2. **Coarse-grain Load（粗粒度加载）**：每次加载CB_{load-tile}×CT×F_{load-tile}个LUT元素到buffer。因每个index从CT个candidates选1个，这些CT个candidates可buffer起来复用。CB_{load-tile}控制沿codebook维度的复用粒度。
3. **Fine-grain Load（细粒度加载）**：按需加载，每处理一个新index时load F_{load-tile}个LUT值。适合PE有多个硬件线程（如UPMEM PE支持up to 24 tasklets/threadlets）——每线程可独立issue memory request，利用concurrent memory access隐藏延迟。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以UPMEM PIM-DIMM单PE（64KB WRAM, 8 hardware threadlets）为例，workload (N_s_tile=512, CB=256, CT=16, F_s_tile=128)：

```
# ===== Static Load Scheme =====
# Condition: CB×CT×F_s_tile = 256×16×128 = 524288 elements > 64KB → NOT applicable
# 仅当F_s_tile ≤ 8时可用（256×16×8=32768 ≤ 64KB）

# ===== Coarse-grain Load Scheme =====
# CB_load_tile=64, F_load_tile=32
# Per CB_load_tile block load: 64×16×32 = 32768 LUT elements → fits in 64KB

for cb_block in range(0, CB=256, CB_load_tile=64):
    # Load LUT block: shape (64, 16, 32)
    lut_block = load_from_DRAM(LUT_addr + cb_block * 16 * F_s_tile, 64*16*32)
    
    for idx_row in range(0, N_s_tile=512, N_m_tile):
        idx_mtile = load_index_tile(idx_row, N_m_tile, cb_block, 64)  # MTile load
        out_mtile = load_output_tile(idx_row, N_m_tile)  # load output MTile for accumulation
        
        for cb_i in range(64):  # traverse codebook tiles within block
            for in_i in range(N_m_tile):
                centroid_id = idx_mtile[in_i][cb_i]  # ∈ [0, CT=16)
                # Fetch 32 F-dim values from buffer (reuse CT-dim)
                out_mtile[in_i] += lut_block[cb_i][centroid_id][:]  # 32-wide F accumulation
        
        store_output_tile(idx_row, N_m_tile, out_mtile)  # write back partial output
```

```
# ===== Fine-grain Load Scheme =====
# F_load_tile=2 (per threadlet buffer), 8 threadlets concurrent

for idx_row in range(N_s_tile=512):
    for cb_i in range(CB=256):
        centroid_id = index[idx_row][cb_i]  # ∈ [0, 16)
        
        # Launch 8 threadlets, each fetches F_load_tile=2 elements
        for t in range(8):
            threadlet[t].async_load(
                LUT[centroid_id][cb_i][t*F_load_tile : (t+1)*F_load_tile]
            )
        # Wait for all loads, then accumulate 16 F-dim elements
        for t in range(8):
            output[idx_row][t*2:(t+1)*2] += threadlet[t].get_result()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LUT Load Scheme的实现和使用：
1. **实现**：在PIM-DL Auto-Tuner中作为搜索参数P4。Auto-Tuner根据模型shape和硬件参数（on-chip buffer size、PE threadlet数量、local memory bandwidth）自动选择最优方案。UPMEM PIM-DIMM上——每个PE的RISC core执行ISA级LUT kernel指令，通过UPMEM SDK compile的PIM binary实现具体的load scheme logic。
2. **性能特征**（UPMEM PIM-DIMM实测）：不同load scheme间性能gap可达1.91×（Figure 13）。Fine-grain load对threadlet数量敏感——threadlet多时可更好利用concurrent memory access隐藏latency。Coarse-grain load对CB_{load-tile}选择敏感——需要在CB re-traversal overhead和buffer reuse间权衡。
3. **应用**：不仅适用于DRAM-PIM，任何有on-chip scratchpad memory的计算单元（GPU shared memory、NPU local buffer）在执行index-driven table lookup时都可使用类似策略。

涉及论文标题：
- 20-PIM-DL- Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization.pdf

---

## Tile-Granularity GEMM-NonGEMM Software Pipelining

术语解释
Tile-Granularity GEMM-NonGEMM Software Pipelining是Tandem Processor论文提出的将GEMM操作和非GEMM操作以tile（子张量）粒度进行overlap执行的软件流水技术。与传统的layer粒度串行执行（GEMM完→整层非GEMM）或operand粒度overlap相比，tile粒度在资源利用率和复杂性之间达到最优。

术语是什么？
软件流水的基本机制：(1) 编译器在layer fusion后对融合block执行uniform tiling——为GEMM和非GEMM层的输入/输出选择一致的tile划分方案；(2) 生成混合指令流，在GEMM指令和Tandem Processor指令之间插入synchronization instructions标记边界；(3) 执行时，GEMM unit和Tandem Processor以double-buffering方式overlap：GEMM处理tile N+1时Tandem Processor处理tile N。Tiling策略的关键约束：(a) 不对GEMM的reduction维度做tiling（否则GEMM输出partial result，Tandem Processor无法执行完整非GEMM操作而stall）；(b) tile size需足够大以包含非GEMM操作的完整邻域（如Depth-wise Conv 5×5 kernel需5×5 patch完整数据）；(c) tile size需足够小以适应128KB Interim BUF。论文测量：tile粒度overlap相比layer粒度使GEMM unit利用率提升20%，Tandem Processor利用率提升13%。

从kernel调度角度拆解：
以BERT中MatMul→Add→GeLU block的4-tile执行为例（伪代码）：
```
// compiler-generated schedule (simplified)
Block: {GEMM: MatMul, SIMD: Add, SIMD: GeLU}
Tiling: 4 tiles along sequence dimension

// Instruction stream:
Sync SIMD_START_EXEC       // mark SIMD region start
GEMM_CONFIG MatMul tile_0  // configure GEMM for tile 0
// ... GEMM configuration instructions ...
Sync GEMM_START_EXEC       // dispatch to GEMM unit

// Tile 0: GEMM
GEMM_COMPUTE tile_0        // GEMM unit writes to OBUF bank_0

// Tile 0 non-GEMM + Tile 1 GEMM overlap
Sync SIMD_START_BUF        // Tandem takes OBUF bank_0 ownership
TILE_LD_ST LOAD ...        // load additional data to Interim BUF
SET_ITER 0, ...            // configure loops
ADD InterimBUF2,0, ...     // Add operation
// GeLU decomposition:
MUL ..., MUL ..., MUL ..., MUL ..., MUL ...
ADD ..., ADD ..., ADD ...
SIGN ..., ABS ..., MIN ...
Sync SIMD_END_BUF          // release OBUF bank_0
// At the same time: GEMM unit processing tile_1 → OBUF bank_1

// Tile 1 non-GEMM + Tile 2 GEMM overlap
Sync SIMD_START_BUF        // Tandem takes OBUF bank_1 ownership
... (repeat for tiles 1-3) ...
Sync SIMD_END_EXEC          // mark block end
```

术语一般如何实现？如何使用？
Tile粒度软件流水的实现关键：(1) **编译器tiling决策**：基于scratchpad容量（128KB Interim BUF）、非GEMM算子的邻域需求（如DW-Conv kernel size）、GEMM reduction维度约束，通过分析式（非ML-based）计算最优tile size。对于非GEMM-only block，编译器还优化tile size以最大化Data Access Engine和Tandem Processor pipeline的overlap。(2) **硬件同步**：Execution Controller FSM通过两个handshaking信号（OBUF_done→GEMM unit通知Output BUF已释放，Tandem_done→Exec FSM通知非GEMM tile完成）管理状态转换。不需要复杂的内存一致性和缓存协议——因为scratchpad是软件管理的且所有权转换是编译器确定性的。(3) **与传统软件流水的区别**：CPU上的软件流水（如Intel IA-64的modulo scheduling）通过循环展开+寄存器重命名实现指令级并行，而Tandem Processor的软件流水是在processor-level（GEMM unit vs Tandem Processor）的粗粒度任务级overlap。

涉及论文标题：
- 21-Tandem Processor: Grappling with Emerging Operators in Neural Networks

## Data Movement Bound (Algorithmic-minimum vs Attainable Bound)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Data Movement Bound（数据移动下界）指给定张量算法在特定buffer capacity约束下理论上能达到的minimal backing store data movement。通常有两个层次的bound：(1) **Algorithmic-minimum bound**（传统上在"speeds and feeds"分析中使用）= 所有input和output operand sizes之和——等价于cache中的compulsory misses，完全不考虑buffer capacity或data reuse。对4k×4k×4k GEMM，algorithmic-minimum = 96MB（FP16, 3 operands各32MB）；(2) **Attainable bound**（Orojenesis提供）= 在给定buffer capacity下，任何mapping均无法超越的backing store access下界——考虑buffer capacity如何限制data reuse exploitation。Orojenesis通过Snowcat架构上的穷举mapspace search推导此bound。论文揭示：对于4k GEMM，A100 GPU DRAM实测traffic为algorithmic-minimum的6.5×，L2-to-L1 traffic为32.3×——algorithmic-minimum与实际可达data movement之间的gap可达数量级。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
以4k×4k×4k GEMM在buffer capacity=50MB（≈H100 L2 cache）下的data movement分析为例：
```
// Algorithmic-minimum (compulsory misses only):
accesses_algo_min = size(A) + size(W) + size(B)
                  = 4096×4096×2B + 4096×4096×2B + 4096×4096×2B
                  = 96 MB

// Orojenesis attainable bound (Snowcat exhaustive search, buffer=50MB):
// Best mapping: M0×K0×N0 tile that fits in 50MB
// M=M0×M1, K=K0×K1, N=N0×N1
// buffer_size = M0×K0 + K0×N0 + M0×N0 ≤ 50MB
// accesses = M1×K1×K0×N0 (A reads) 
//          + K1×N1×K0×N0 (W reads)
//          + M1×N1×M0×N0 (B writes)
// Pareto-optimal: ~200 MB at 50MB buffer (vs 96MB algo-min)
// Gap 0 = 200MB - 96MB = 104MB excess (2.08× algo-min)

// A100实测 (40MB L2, CUTLASS optimized schedule):
accesses_measured ≈ 6.5× algo_min for DRAM
                  ≈ 32.3× algo_min for L2-to-L1
```
对于GPT-3-6.7b LLM building block：algorithmic-minimum w/o fusion ≈ 20+ GB, attainable bound at 50MB buffer with optimal fusion ≈ 10 GB, reduction factor = 2.5× over unfused。若buffer=320MB，fused bound可降低5.6× relative to unfused algo-min。

术语一般如何实现？如何使用？
Data movement bound的使用场景：(1) **Early DSE**: Architect用attainable bound代替algorithmic-minimum做roofline performance model——得出更准确的compute vs memory-bound判定；(2) **Mapping quality check**: 若compiler/mapper生成的mapping实测accesses远超Orojenesis bound，说明mapping sub-optimal或存在架构inefficiency；(3) **Buffer sizing**: 通过分析不同buffer size下的access reduction rate（ski-slope斜率），确定buffer capacity的diminishing returns point；(4) **Fusion evaluation**: 对比fused vs unfused的attainable bounds，量化fusion在不同buffer capacity下的potential benefit。Orojenesis的Snowcat-based approach使得bound derivation无需per-architecture mapspace search，一次run结果portable到所有遵循Timeloop描述框架的架构。

涉及论文标题：
- 23-Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms

---

## Layer-by-layer Pipelined KV Restoration (逐层流水线KV恢复)

术语是什么？
Layer-by-layer Pipelined KV Restoration是FlashGen-Cache中从host memory恢复attention KV到GPU的pipeline技术：在Transformer decoder的prefill阶段，将第L+1层decoder的KV从host→GPU的DMA传输与第L层decoder的attention计算并行执行，隐藏host→GPU的传输延迟。该技术借鉴了PipeSwitch的pipeline context switching思想。

从kernel调度角度拆解：
以OPT 30B（32 decoder layers, prefill 256个prompt tokens + 2048个history tokens从host memory恢复）为例：
```
// 标准同步KV restoration（无pipeline）:
for L in 0..num_layers-1:
    cudaMemcpyAsync(KV_gpu[L], KV_host[L], size_per_layer, stream_data)
    cudaStreamSynchronize(stream_data)  // 等待传输完成
    flash_attn_prefill(Q[L], KV_gpu[L], ...)  // 执行attention
// 总延迟 = Σ(dma_latency[L] + compute_latency[L])

// Pipelined restoration（FlashGen方法）:
for L in 0..num_layers-1:
    if L == 0:
        cudaMemcpyAsync(KV_gpu[0], KV_host[0], size, stream_data)
        cudaStreamSynchronize(stream_data)
        flash_attn_prefill(Q[0], KV_gpu[0], ...)  // Layer 0: 计算
    else:
        // Layer L KV传输与Layer L-1计算并行
        cudaMemcpyAsync(KV_gpu[L], KV_host[L], size, stream_data)
        // Layer L-1的attention在上一步已开始，此时与传输并行
        flash_attn_prefill(Q[L-1], KV_gpu[L-1], ...)  // 使用stream_compute
    // 最后：等待最后一个transfer完成 + 执行最后一层attention
// 总延迟 ≈ dma_latency[0] + Σ(max(dma_latency[L], compute_latency[L-1])) + compute_latency[last]
```
Pipeline的有效性取决于每层KV传输量和计算量的比例。当prefill batch包含足够多的token（compute-bound），计算时间足以完全覆盖下一层KV传输。当batch仅含generation phase（1 token per request per step, memory-bound），传输可能超过计算——FlashGen通过batch-aware exclusion将KV未就绪的请求排除出当前batch来解决。

术语一般如何实现？如何使用？
实现需要：(1) 多个CUDA stream——至少一个data stream（host→GPU DMA）和一个compute stream（kernel launch），利用A100的copy+compute concurrency能力；(2) layer维度展开——不等待全部layer的KV传输完成再开始prefill，而是逐层进行；(3) 同步点最小化——仅在每层compute开始前检查该层KV传输是否完成（cudaEvent），无需全量sync。该技术的约束：(a) 需要模型按layer串行执行（decoder layer-by-layer）才能pipeline——对Pipeline Parallelism等跨layer并发策略有冲突；(b) 每层KV大小 = 2 (K+V) × num_heads × head_dim × history_seq_len × 2 bytes (FP16)，需大到传输时间有意义；(c) prefill batch size需足够大（计算量）来覆盖传输时间。FlashGen额外优化：修改Flash-Attention kernel支持non-contiguous KV blocks（因为PagedAttention下host memory中的KV blocks物理不连续，需gather-style读取）。

涉及论文标题：
- 24-Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management.pdf (FlashGen)

---

## Non-contiguous KV Block Attention (非连续KV块注意力)

术语是什么？
Non-contiguous KV Block Attention是FlashGen对Flash-Attention和Flash-Decoding kernel的修改，使其支持从物理上不连续的KV cache blocks读取并计算attention——因为在PagedAttention中，history KV分布在多个物理上不连续但逻辑上连续的page中，标准Flash-Attention kernel假设KV在GPU global memory中连续存储。

从kernel调度角度拆解：
标准Flash-Attention内层循环（简化）：
```
// 标准Flash-Attention: K/V连续存储
float Q_tile[TILE_Q][HEAD_DIM];  // Q tile in SRAM
float O_tile[TILE_Q][HEAD_DIM] = {0};
float m[TILE_Q] = {-inf};
float l[TILE_Q] = {0};

for kv_block_start in 0..seq_len step TILE_KV:
    // 从连续地址加载K/V tile
    float K_tile[TILE_KV][HEAD_DIM] = load_contiguous(K + kv_block_start * head_dim);
    float V_tile[TILE_KV][HEAD_DIM] = load_contiguous(V + kv_block_start * head_dim);
    // Q·K^T → softmax → ×V (标准tiling)
    ...
```

修改后的Non-contiguous Flash-Attention：
```
// 修改后: K/V分布在非连续物理blocks中
float Q_tile[TILE_Q][HEAD_DIM];
float O_tile[TILE_Q][HEAD_DIM] = {0};
float m[TILE_Q] = {-inf};
float l[TILE_Q] = {0};

int* block_table;  // [num_logical_blocks], 每个entry是物理block index
int page_size = 16; // tokens per block (PagedAttention page)
int kv_offset = 0;

for logical_page in 0..num_logical_pages:
    int physical_page = block_table[logical_page];
    // 从物理page基址 + 内部偏移计算实际地址
    K_ptr = &K_cache[physical_page * page_size * num_kv_heads * head_dim];
    V_ptr = &V_cache[physical_page * page_size * num_kv_heads * head_dim];
    
    for inner_offset in 0..page_size step TILE_KV:
        // Gather-style非连续加载
        float K_tile[TILE_KV][HEAD_DIM] = load_from(K_ptr + inner_offset * ...);
        float V_tile[TILE_KV][HEAD_DIM] = load_from(V_ptr + inner_offset * ...);
        // 后续标准Flash-Attention tiling相同
        ...
```
关键修改：外层增加block table遍历循环（遍历logical pages→映射到physical pages），内层对每个page内的连续tokens执行标准Flash-Attention tile操作。实现方式与FlashInfer（https://flashinfer.ai）的paged attention kernel设计类似——block table作为额外kernel参数传入，由每个thread block独立解析address。

术语一般如何实现？如何使用？
实现方式：(1) **Block table参数**——将PagedAttention的block table（int32数组, [num_requests, max_num_blocks_per_seq]）作为kernel额外参数传入，kernel内根据logical block index查表获取physical block base pointer；(2) **Gather semantics**——每个thread block在遍历logical blocks时计算实际global memory地址：`physical_addr = kv_cache_base + physical_block_id * block_size_bytes + intra_block_offset`；(3) **Alignment约束**——page size需对齐Flash-Attention的KV tile size（如page_size=16, TILE_KV=16/32/64/128），以确保单page内访问连续；(4) **Head dimension处理**——block table通常在batch×seq维度做映射，head维度连续存储（NHD layout），每个attention head独立读取相同的block table。该技术的应用场景：多轮对话KV cache恢复、beam search共享prefix KV、page-based KV offloading/reloading等所有需要从不连续物理地址读取KV的场景。FlashInfer和FlashGen均基于此设计，TensorRT-LLM的paged KV cache kernel也采用类似思路。性能开销：相比连续KV的Flash-Attention，非连续版本增加block table查表开销（≈1次global memory load per logical page）和地址计算，但通常在prefill的compute-bound场景下可忽略。

涉及论文标题：
- 24-Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management.pdf (FlashGen)

---

## CIM Array Mode Allocation (λ-based MIP)

术语解释
CIM Array Mode Allocation是在dual-mode CIM架构上通过Mixed-Integer Programming (MIP)为每个DNN算子分配CIM array的compute/memory模式的方法。通过λ变量体系（λ_c, λ_min, λ_mout）形式化array-operator绑定关系，在资源约束下最小化pipeline执行延迟。

术语是什么？
在双模CIM芯片上，每个CIM array可独立设为compute mode或memory mode（input/output）。CMSwitch将这个问题形式化为MIP：(1) **决策变量**：λ_c(i,x,y)=1表示CIM array(x,y)作为算子Oi的compute array；λ_min(i,x,y)=1表示作为Oi的input memory buffer；λ_mout(i,x,y)=1表示作为Oi的output memory buffer。(2) **关键约束**：array overlap（每个array每时刻只能是一种mode）、operator dependency（Oi的输出memory array可复用为依赖算子Oj的input memory buffer，减少数据搬移）、resource limit（所有算子使用的array总数≤Ncim=96）。(3) **目标函数**：`min max(LOi)`，最小化segment内最慢算子的延迟（pipeline瓶颈）。(4) **延迟模型**：`LOi ∝ OPOi / min(ComOi·OPcim, (MemOi·Dcim+Dmain)·AIOi)`——高AI算子bottleneck在compute capacity，低AI算子bottleneck在memory bandwidth。

从kernel调度角度拆解：
对于OPT-6.7B单层attention，MIP求解输出的λ分配导致以下kernel调度策略：
```
Segment 1: QKV Projection + Attention
  - Arrays [0:23]: λ_c(W_Q) = 1 → load W_Q weights, compute Q = X × W_Q
  - Arrays [24:31]: λ_min(W_Q) = 1 → buffer Q input activation
  - Arrays [32:47]: λ_c(W_K) = 1 → load W_K weights, compute K = X × W_K
  - Arrays [48:55]: λ_mout(W_K) = 1 → retain K output (reuse for QK^T)
  - Arrays [56:71]: λ_c(W_V) = 1 → load W_V weights, compute V = X × W_V
  - Dependency reuse: λ_mout(W_K) → λ_min(QK^T) for K
  - Arrays [48:55]: λ_c(QK^T) after CM.switch(TOM) → in-place QK^T computation
```
MIP的operator dependency constraint使K值从memory mode array直接原位切换为compute执行QK^T，无需数据搬移。不同算子的AI差异导致compute/memory array比例不同（QKV约40-50% compute，attention约60-70% compute，FFN最后一层约33% compute）。

术语一般如何实现？如何使用？
MIP使用Gurobi求解器求解，输入为segment内算子顺序、算子AI和OP数、芯片Ncim/OPcim/Dcim/Dmain。求解输出为λ分配矩阵→经post-allocation optimization（weight duplication）→转为DMO meta-operator流。MIP求解为offline操作，编译时间与search space相关但通过constraints pruning保持线性扩展。

涉及论文标题：
- 25-Be CIM or Be Memory- A Dual-mode-aware DNN Compiler for CIM Accelerators.pdf

---

## Inter-segment Mode Switch Overhead (CIM)

术语解释
Inter-segment Mode Switch Overhead是在dual-mode CIM编译优化中，相邻网络segment之间因CIM array模式切换而产生的额外延迟。由三个步骤组成：数据写回（write-back）、模式切换（mode switch）和权重重载（weight reload）。

术语是什么？
当DNN被DP分割为多个segment串行执行时，从一个segment切换到下一个需要：(1) **数据写回（Write-back）**：若前一个segment使用了更多memory array存储中间数据，且这些数据后后续需要使用但不能在原位保留，必须先写回off-chip main memory（或保留在未切换的memory array中）。写回延迟T_wb根据数据量和external bandwidth估算。对于不会被重用的数据（如softmax结果），CIM array可直接切换而不写回。(2) **模式切换（Mode Switch）**：`T_swc = L_M→C × Switch_m→c + L_C→M × Switch_c→m`，其中Switch_m→c和Switch_c→m分别表示从memory切到compute和从compute切到memory的array数量，L为每个array的切换延迟（DynaPlasia=1 cycle）。通过GIA/GIAb信号修改实现。(3) **权重重载（Weight Reload）**：不同segment处理不同算子，compute array中存储的权重需要更新。`T_rw = max_{Ol in S_i,j}(Com_Ol × latency_write)`，取segment内各算子中需reload权重的最大compute array数量×单个array权重写入延迟。

从kernel调度角度拆解：
对于LLaMA2-7B attention层被分为Segment 1 (QKV+QK^T)和Segment 2 (SV+FFN)：
```
Segment 1 执行完毕 → Inter-segment switch → Segment 2 开始
Step 1 (Write-back): Segment 1的output (softmax结果S) 在memory array中 → S被QK^T原位消费后可丢弃 → T_wb≈0
Step 2 (Mode Switch): 部分compute array 需要切为memory mode为Segment 2的大activation数据提供bandwidth → Switch_c→m = 16 arrays × 1 cycle = 16 cycles
Step 3 (Weight Reload): Segment 2需要W_O和FFN的W1/W2权重 → Com_O(SV) = 32 arrays需要reload → T_rw = 32 × write_latency
Total T_inter ≈ T_wb + T_swc + T_rw
```
在DynaPlasia上，mode switch overhead仅占总执行时间的3%-5%。

术语一般如何实现？如何使用？
Inter-segment overhead在DP segmentation阶段被计算和优化——DP在决定分割点时考虑T_inter，避免在不划算的位置分割。DP的cost function为`L[j][A] = min(L[i][A'] + T_intra(A) + T_inter(A',A))`。通过impossible-case pruning和segment粒度控制来平衡segmentation收益与switch cost。

涉及论文标题：
- 25-Be CIM or Be Memory- A Dual-mode-aware DNN Compiler for CIM Accelerators.pdf

## Prefetching and Delayed Storing (DRAM Communication)

术语解释
Prefetching and Delayed Storing是SoMa[HPCA 2025]识别的一种被已有DNN调度文献忽视的DRAM通信优化范式，通过将数据从DRAM的加载（prefetching）提前到DRAM空闲时段、将数据向DRAM的写回（delayed storing）推迟到DRAM空闲时段，平滑DRAM带宽利用的不均匀性，最大化compute和DRAM access的overlap，从而减少计算stall和DRAM带宽浪费。

术语是什么？
在DNN加速器处理layer fusion后的tile序列时，不同tile的DRAM带宽需求与计算需求之比差异极大——权重加载tile（每层首tile）DRAM密集、后续tile因weights已on-chip而几乎无DRAM需求（仅compute），形成seesaw pattern。传统double-buffer策略（前一tile prefetch、后一tile store）无法有效利用这种不均衡——DRAM带宽在计算密集tile时段闲置，在DRAM密集tile时段拥堵导致计算stall。SoMa通过显式控制每个DRAM tensor的存取时机来利用DRAM空闲时段：将大权重tensor提前多个tile开始prefetch（在DRAM空闲的compute-only tile期间），将ofmaps延迟存储（错开峰值时段与其他DRAM tensor竞争）。

从kernel调度角度拆解：
SoMa中prefetching and delayed storing通过Living Duration属性实现：
```
// 传统double-buffer（Cocco）
// Tile序列: ..., B2, B3, B4, C4, A'1, B'1, C'1, ...
// DRAM row (Cocco):
//   [WA'    ][WB'    ][WC'    ]  ← A'1前集中加载，A'1 stall
// COMPUTE row (Cocco):
//   B2  B3  B4  C4  [--STALL--]  A'1 B'1 C'1

// SoMa的prefetching and delayed storing（第二阶段优化后）
// 原始情况：LG2的3个大权重WA'(up to 2304KB INT8)在LG2开始时集中加载
//   导致A'1之前的heavy computing stall
// 优化后：
//   - WA'_1, WA'_2（两个最大的权重tensor）的DRAM Tensor Order移至FLG2末尾DRAM空闲期
//   - 其余权重tensor的Living Duration Start向前推（如从A'1推到B2/D2）
//   - FLG2末层ofmaps延迟一个tile存储，与LG2首层ifmaps交换DRAM Tensor Order位置
// DRAM row (SoMa):
//   [WA'/WB'/WC' prefetched在B2-B4的DRAM空闲期]
// COMPUTE row (SoMa):
//   B2  B3  B4  C4  A'1 B'1 C'1  ← stall消除
```

术语一般如何实现？如何使用？
该范式由SoMa编译框架的第二阶段SA自动实现，不需手工调优。SA operators包括Change DRAM Tensor Order和Change Living Duration，每次随机选一个DRAM tensor（选择概率正比于tensor大小）调整其存取顺序或生命周期。优化受buffer容量约束——prefetch提前越多，buffer占用越久，存在buffer-latency trade-off。SoMa实验显示在8MB GBUF约束下，第二阶段利用prefetching and delayed storing将性能较第一阶段再提升1.16×（平均距理论上限3.1%）。该范式的优化潜力依赖于：(1) DRAM access和compute的比值不均衡程度（越大潜力越大）；(2) 可用buffer容量（越大可prefetch越早）；(3) batch size（越大compute时间越长，越有机会隐藏传输）。对于GPT-2 Decode等极低compute density场景（延迟完全由weight和KV cache加载主导），该范式的优化空间几乎为零——没有足够的compute时间来"隐藏"DRAM传输。

涉及论文标题：
- 26-SoMa: Identifying, Exploring, and Understanding the DRAM Communication Scheduling Space for DNN Accelerators

## Living Duration (DRAM Scheduling Attribute)

术语解释
Living Duration是SoMa Tensor-centric Notation中DLSA（DRAM-Load-and-Store-related Attributes）的核心属性之一，为每个DRAM tensor定义一个(Start, End) tile对，同时控制该tensor在on-chip buffer中的存活时间区间和其从DRAM加载/向DRAM存储的调度时机。

术语是什么？
每个需要与DRAM交互的tensor（ifmap、weight、ofmap）都有独立的Living Duration = (Start, End)，其中Start和End是tile ID而非绝对时间。该属性有双重语义：(1) **Buffer分配**——tensor在buffer中的存活区间为从Start tile到End tile，在此期间buffer被该tensor占用；(2) **存取时机调度**——对ifmaps/weights，End固定为最后消费该数据的tile的下一个tile（表示release时间），Start表示可以开始从DRAM加载的最早tile（提前Start=prefetching）；对ofmaps，Start固定为生产该数据的tile（表示生成完成），End表示必须完成存储的最晚tile（延迟End=delayed storing，若存储未完成则对应consumer tile stall）。

从kernel调度角度拆解：
以SoMa论文Fig.4为例的Living Duration设定与效果：
```
Tensor WE (weight of layer E):
  Living Duration = (B, D2)
  含义：从tile B开始即可加载（prefetch），在tile D2之前可释放buffer
  效果：WE在tile B到D2期间占据buffer，但实际加载时机由DRAM Tensor Order决定

Tensor WB (weight of layer B):
  Original Living Duration (double-buffer): Start = B (前一tile A2之后)
    → A2→B的计算stall（WB加载未完成）
  SoMa优化后: Start = A2 (提前一个tile)
    → WB loading与A2 compute重叠，A2→B的stall消除
    → Buffer占用增加：WB在A2期间也在buffer中

Tensor OE1 (ofmap of E1):
  Living Duration = (E1, D1)
  含义：E1计算完成后即可开始store（Start=E1），必须在D1开始前完成store（End=D1）
  效果：若OE1的store未在D1开始前完成，D1必须stall等待
  SoMa优化：延迟OE1的DRAM Tensor Order（错开与IC2的竞争），或延长End
```

Living Duration的合法性约束：对ifmaps/weights，Start ≥ 生产者tile（ifmap的产生tile或网络输入），End ≤ 最后消费tile+1；对ofmaps，Start ≥ 生产者tile（ofmap的产生tile），End ≤ 消费者tile且≥Start。所有Living Duration区间对应的buffer总使用量 ≤ 硬件buffer容量。

术语一般如何实现？如何使用？
Living Duration由SoMa的DLSA Exploration Stage通过SA搜索自动确定。SA operator `Change Living Duration`随机选择一个DRAM tensor（概率正比于tensor大小），对ifmaps/weights随机修改其Start（向前推=更早prefetch，增加buffer占用但可能消除stall），对ofmaps随机修改其End（向后推=更晚store，减少峰值DRAM竞争）。SA的cost function评估各tensor的Living Duration组合在buffer容量约束下的整体延迟/能耗。该机制已在SoMa开源框架中实现：https://github.com/SET-Scheduling-Project/SoMa-HPCA2025。Living Duration的搜索空间受buffer容量和依赖图的严格限制，SA通过probabilistic acceptance（以概率p接受更差方案）跳出局部最优。

涉及论文标题：
- 26-SoMa: Identifying, Exploring, and Understanding the DRAM Communication Scheduling Space for DNN Accelerators

## Collective Communication Operations in PIM (PIM集体通信操作)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Collective Communication Operations（集体通信操作）是并行计算系统中多个处理节点之间协调数据交换和计算的机制。在 DIMM PIM 系统中，常见集体通信操作包括：(1) Broadcast：一个 PE 的数据分发到所有参与 PEs。(2) All-Gather：每个 PE 贡献数据，所有 PE 收集全部数据。(3) Reduce：多个 PE 的局部数据经 reduction 操作（如 sum/min/max/bitwise OR）汇总到指定 PE。(4) All-Reduce：每个 PE 贡献数据，所有 PE 获得 reduction 后的全局结果。(5) Reduce-Scatter：各 PE 的局部数据经 reduction 后，结果按块分散到不同 PE。这些操作在 graph processing（BFS frontier sync）、machine learning（梯度聚合）、推荐系统（embedding lookup）等 PIM workload 中广泛使用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 CoCoTree 上 BFS 的 All-Reduce (bitwise OR) 为例：

```c
// === Configuration Phase (由1个PE发起) ===
CoCoTree::initConfig(num_pes=1024, op_type=ReduceOR, subtree_height=10);
CoCoTree::configTree();      // command packet 沿树上行配置所有 Co-Node
CoCoTree::waitConfigReady(); // barrier: 等待所有 PE 收到配置

// === Computation Phase (所有PE并发) ===
for (int i = 0; i < local_frontier_size; i++) {
    CoCoTree::send(&local_bitmap[i], sizeof(bitmap_t));
}
CoCoTree::waitReceive();
bitmap_t global_frontier = CoCoTree::getReceived();

// CoCoTree 硬件在 kernel 执行期间自动完成:
// 1. Co-Leaf 将 bitmap 按 byte 拆解 pack 为 32-bit packets
// 2. Level-0 Co-Node 对左右子 PE 的 packet 执行 byte-wise OR
// 3. 结果逐层 reduce 至 root，产生全局 OR 结果
// 4. Root broadcast 结果到所有参与 PE subtree
// 5. Co-Leaf unpack 恢复到 PE 本地 WRAM
```

传统 UPMEM host-forwarding（对比）：Host CPU 通过 DMA gather 各 PE 数据 → CPU 执行 reduction → DMA scatter 回 PE。CoCoTree 将 O(N) 次 host-PIM DMA 压缩为 O(log N) 层 tree reduce，All-Reduce 达 95.6× speedup。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 DIMM PIM 中实现集体通信的方式：(1) CPU Forwarding（UPMEM baseline）：host CPU DMA gather→compute→scatter，简单但受限于内存通道带宽。(2) ABC-DIMM [ISCA 2021]：multi-drop bus 实现 broadcast-based 直连通信。(3) DIMM-Link [HPCA 2023]：专用点对点链路。(4) PIMnet [HPCA 2025]：多层级互连网络。(5) CoCoTree [HPCA 2026]：分层二叉树 + in-network computation，每层流量减半，支持 64-2048 PE linear scaling。

涉及论文标题：
- 27-CoCoTree_A_Computation-Capable_Architecture_for_Collective_Communication_in_Scalable_PIM.pdf

## Two-Phase Configuration-Computation Communication Model

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Phase Configuration-Computation Communication Model（两阶段配置-计算通信模型）是 CoCoTree 提出的解耦控制流与数据流的通信协议。每次集体通信操作分为两阶段：(1) Configuration Phase：发送 command packet 预先配置所有参与 Co-Node 的 routing 路径和 FU 操作类型，command packet 到达所有参与 PE 时充当同步 barrier。(2) Computation Phase：已配置的 Co-Node 和 Co-Leaf 按预设 routing 和 FU 处理 data packet，无需再传输操作类型、目标地址等元数据。此分离消除了数据阶段元数据的冗余传输，提高带宽效率，并允许配置一次后执行多轮计算（pipelining）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Two-Phase Model 执行流程（以 Reduce 为例）：

```
// === Phase 1: Configuration ===
// PE#0 构造 command packet: DC=0, INSTR=redSumU, ADDR=目标地址,
//   STH=subtree高度(如10→1024 PE), DFD=STH
// Command packet 沿树上行: 每 Co-Node 减 DFD, 左旋 ADDR
//   DFD>0: LSB(ADDR) 决定左/右子转发
//   到达 root (DFD==1): 完成上行
// Root broadcast command 到整个 subtree (DFD=ADDRL 下行至 0)
//   沿途 Co-Node 解析 INSTR:
//     FU Controller → FU_i=add; Routing → merge-up + broadcast-down
// Command 到达所有 PE → barrier, PE 准备发送数据

// === Phase 2: Computation ===
// 各 PE: CoCoTree::send(local_data)
//   Co-Leaf pack 为 32-bit data packet (DC=1, TAIL=1)
// Data packet 沿已配置 Co-Node 上行: merge + FU(add) 执行 reduce
// 结果到 root → routing 切换 broadcast-down
// 结果分发到目标 PE → Co-Leaf unpack → PE polling 获取
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) Packet 格式：32-bit，DC(1bit) 区分 command(0)/data(1)。Command 含 INSTR(5bit)/ADDR(15bit)/ADDRL(4bit)/STH(4bit)/DFD(4bit)；Data 含 DATA(32bit)/TAIL(1bit)。(2) Routing：基于 perfect binary tree 的地址驱动路由——ADDR 编码目标地址，STH 定义 subtree 高度，DFD 逐跳递减。(3) 同步：command packet 同时到达所有 PE 作 implicit barrier。(4) Pipeline 支持：配置一次后连续发送多轮数据，不同 tree level 并发处理不同轮次。

涉及论文标题：
- 27-CoCoTree_A_Computation-Capable_Architecture_for_Collective_Communication_in_Scalable_PIM.pdf

## bbop (Bulk Bitwise Operation) μProgram Execution in PUD

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
bbop (Bulk Bitwise Operation)是SIMDRAM/MIMDRAM框架定义的PUD ISA扩展指令，每条bbop对应一种在DRAM内执行的bulk bit-serial操作（共16种：abs, add, bitcount, div, max, min, mult, ReLU, sub, and/or/xor-reduction, equal, greater, greater_equal, if_else）。每条bbop指令被翻译为一个μProgram——一个预先计算好的最优AAP/AP（row copy/TRA）命令序列——由memory controller中的μProgram processing engine按严格时序发出到DRAM芯片。SIMDRAM的三步框架：(1) 将期望操作转为AND/OR/NOT→再优化为MAJ/NOT表示以减少TRA次数；(2) 生成AAP/AP序列；(3) 运行时由control unit执行。MIMDRAM在此框架上增加mat range字段（ML/VF），使bbop可指定目标mats范围和支持可变SIMD宽度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以bbop_add (n-bit addition, target mat_i, VF=1024) 的μProgram执行为例：
```
μProgram for bbop_add(n, mat_range=[0,1]):  // 2 mats × 512 cols = 1024 lanes
  // n-bit full adder: per bit iteration
  for bit in 0..n-1:
    AAP(T0 ← A[bit])     // copy A's bit-row to compute row T0
    AAP(T1 ← B[bit])     // copy B's bit-row to compute row T1
    AAP(T2 ← Cin)        // copy carry-in to T2
    // MAJ(A,B,Cin) → Sum via 3 APs
    AP(T3 ← MAJ(T0,T1,T2))  // compute sum = A⊕B⊕Cin via MAJ chain
    // Carry = MAJ(A,B,Cin) computed during same TRA sequence
    AAP(Cout ← T3)       // write-back carry for next iteration
    AAP(Y[bit] ← T3)     // write-back result bit
  // Total: (8n+2) AAPs/APs per addition
```
MIMDRAM的mat-level调度伪代码：
```
mat_scheduler(bbop_buffer):
  for bbop in bbop_buffer (oldest first):
    for each μprogram_engine in [0..7]:
      if engine.idle:
        mat_range = bbop.mat_range
        if mat_scoreboard.check_free(mat_range):  // all mats in range free
          mat_scoreboard.mark_busy(mat_range)
          engine.load_μprogram(bbop)
          engine.execute()  // issue AAP/AP with mat_range to DRAM
          bbop_buffer.remove(bbop)
          return
        // else: try next bbop (online first fit)
```
当mat_scoreboard显示mats[0,1]空闲时，bbop_add被调度到engine 0执行，同时mats[2,3]可被engine 1用于并发bbop_mul——这就是MIMD模式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
bbop μProgram的实现：
1. **μProgram预计算**：在SIMDRAM/MIMDRAM框架构建时，16种bbop的μProgram已根据bit-serial算法预先计算并存储在control unit中。不同操作有不同的AAP/AP数量：bit-serial addition需(8n+2)步，multiplication O(n²)步（quadratic scaling），comparison只需few steps。
2. **μProgram processing engine**：每engine维护状态机——track当前执行的AAP/AP index，管理DRAM timing（tRAS, tRP, tWR等），发出ACT-enqueue/PRE-enqueue/ACT-dequeue命令。MIMDRAM配置8个engines支持最多8路并发PUD操作。
3. **mat scoreboard**：128-bit bitmap tracking每mat的busy/free状态。mat scheduler用online first fit算法扫描bbop buffer→查scoreboard→分配。
4. **Row分组管理**：Ambit的row分组（Data/Control/Bitwise groups）是每subarray的固定分配。bbop执行前需确保操作数在Data group中，计算结果写入Bitwise group。
5. **MIMDRAM vs SIMDRAM μProgram**：μProgram本身相同，区别在于MIMDRAM的AAP/AP携带mat range→TRA只激活目标mats→剩余mats可并发执行其他μProgram。

涉及论文标题：
- 28-MIMDRAM_An_End-to-End_Processing-Using-DRAM_System_for_High-Throughput_Energy-Efficient_and_Programmer-Transparent_Multiple-Instruction_Multiple-Data_Computing.pdf

## MIMD Execution Model in Processing-Using-DRAM (Mat-Level Parallelism)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MIMD (Multiple-Instruction Multiple-Data) Execution in PUD是MIMDRAM提出的PUD执行模型突破。传统PUD（如SIMDRAM）采用固定宽度的SIMD模型——每次PUD操作覆盖整个DRAM subarray的所有columns，且同时只能执行一种操作。MIMDRAM通过fine-grained DRAM将每个DRAM mat变为独立可控的SIMD单元（512 SIMD lanes/mat），允许同一subarray内不同mat range并发执行不同的bbop指令——即MIMD模式。这类似将DRAM subarray从单核SIMD处理器升级为多核MIMD处理器，每个"核"是一组mats组成的可变宽度SIMD引擎。MIMDRAM支持最多8路并发PUD操作（对应8个μProgram processing engines）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MIMD PUD的执行例子（应用含独立bbop的循环）：
```
// 原始C代码
for(i=0;i<1024;i++){
  C[i]=A[i]+B[i];    // add, 1024-wide → mats[0,1]
  F[i]=D[i]*E[i];    // mul, 1024-wide → mats[2,3]  
  G[i]=C[i]-F[i];    // sub, depend on add & mul → mats[0,1]
}
```
DDG: add→sub, mul→sub（cross-mat dependency via bbop_mov）

MIMDRAM的MIMD执行timeline：
```
Time →
Mats[0,1]: |--bbop_add(C,A,B)--|....idle....|--bbop_sub(G,C,t)--|
Mats[2,3]: |--bbop_mul(F,D,E)--|--bbop_mov(t←F, mat2→mat0)--|
                ↑ 并发执行 ↑              ↑ 数据搬移 ↑
```
- 第一阶段：add和mul在不同mats上**并发执行**（MIMD）——mat scheduler同时分配engine 0→mats[0,1]和engine 1→mats[2,3]
- 第二阶段：bbop_mov通过GB-MOV将mul结果从mat[2,3]搬到mat[0,1]的临时行t
- 第三阶段：bbop_sub在mats[0,1]上执行

对比SIMDRAM（仅SIMD）：
```
Time →
All Mats: |--bbop_add--|--bbop_mul--|--bbop_sub--|
           ↑ 顺序执行，mats[2,3]在add期间完全空闲 ↑
```

MIMD的关键增益：当应用中存在independent bbops（如pca, 3mm, fdtd），MIMDRAM将执行时间缩短2.8×。对于只有dependent bbops的应用，MIMD不带来并行增益但fine-grained mat activation仍节省能耗（仅激活所需mats）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MIMD执行模型的实现要素：
1. **mat-level独立性**：每mat有独立local row decoder + local SA + isolation transistor →不同mats可激活不同rows执行不同AAP/AP序列。
2. **control unit调度**：mat scheduler的online first fit算法+mat scoreboard bitmap确保无mats冲突。8个μProgram engines物理上支持最多8路并发。DDR4 module通常有128 mats（8 chips × 16 mats/chip）→足够承载大量并发操作。
3. **限制**：MIMDRAM仅允许object操作在物理连续的mat range上（简化mat addressing和state management）；数据依赖通过bbop_mov（GB-MOV/LC-MOV）显式处理；当前最多8路并发（受限于μProgram engines数量）。
4. **扩展到SALP/BLP**：MIMDRAM还可与subarray-level parallelism (SALP, 8-64 subarrays/bank) 和bank-level parallelism (BLP, 8-16 banks/rank)结合→实现三级并行 (mat × subarray × bank)。此模式下MIMDRAM用64 subarrays × 16 banks提供13.2× CPU性能。
5. **与应用并行原语的交互**：OpenMP outer loop并行→MIMDRAM compiler将outer iteration的bbops分配到不同mat group→实现SIMT over MIMD。

涉及论文标题：
- 28-MIMDRAM_An_End-to-End_Processing-Using-DRAM_System_for_High-Throughput_Energy-Efficient_and_Programmer-Transparent_Multiple-Instruction_Multiple-Data_Computing.pdf

## SDDMM and SpMM in Sparse Attention (稀疏注意力中的SDDMM与SpMM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SDDMM（Sampled Dense-Dense Matrix Multiplication）和SpMM（Sparse-Dense Matrix Multiplication）是稀疏注意力中替代标准dense attention两个矩阵乘的算子：(1) SDDMM——用稀疏mask矩阵M控制密集Q与K^T的逐元素点积：S[i][j]=Q[i]·K[j] if M[i][j]=1，等价于在密集DDMM(Q·K^T)上施加采样/剪枝；(2) SpMM——稀疏score矩阵S乘以密集V矩阵：Z=S·V，仅NNZ个非零有效计算。在Transformer attention中，SDDMM输出稀疏S→softmax→SpMM输出Z。两者将attention计算从O(dn²)降至O(d·NNZ)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
**CSR-based SDDMM (ASADI Fig.11b)**：
```
for i=0 to n-1:                    // n次迭代
    load Q[i]; for each non-zero M[i][j]:
        for dim=0..d-1: S[i][j]+=Q[i][dim]*K[j][dim]
// 每次仅平均sparsity×n有效计算; 迭代间共享K行需串行
```
**DIA-based SDDMM (ASADI Algorithm 2)**：
```
for i=0 to ω-1:                    // ω次(≪n)
    send M_i(DI,Rd,Ro); shift Q by DI; copy Q per (Rd,Ro)
    for all arrays parallel: SlicesS_i = Q×K  // vec-vec multi
    restore Q; gather SlicesS_i; S=ΣSlicesS_i
// Longformer场景: 7.5× latency节省
```
**CSR-based SpMM (ASADI Fig.9b)**：
```
for i=0..n-1: remap CSR row→align col coords→累加Z[i]+=S[i][j]*V[j]
// n次迭代,每次仅sparsity×n有效计算
```
**DIA-based SpMM (ASADI Algorithm 1)**：
```
distribute S的DIA到d arrays; rotate diagonals; for all arrays: I+=S×V(并行vec-vec)
decompress I (Rd/Ro); Z=ΣI
// ω次迭代,7.5×迭代节省(Longformer)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPU: SDDMM用block-sparse格式+CUTLASS; PyTorch用`torch.sparse.mm`。CPU用MKL `mkl_sparse_d_mm`。PIM/加速器：CSR-based用row-sequential decoder+DMA gather→PE multiply-accumulate; DIA-based用diagonal-parallel decoder+in-situ vector-vector multi (ReRAM行并行)。ASADI的DIA-based不需remapping→硬件控制简化和迭代数减少。

涉及论文标题：
- 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf

## DIA-based In-situ Computing Paradigm (基于DIA的存内计算范式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DIA-based in-situ computing paradigm是ASADI提出的在ReRAM阵列上利用DIA格式对角线局部性直接执行稀疏矩阵乘法的计算范式。核心思想：(1) DIA格式的稀疏矩阵与密集矩阵共置同一ReRAM阵列；(2) 利用ReRAM行并行能力，一次vector-vector multi同时处理对角线上所有非零元素；(3) 消除CSR的row-wise remapping（DIA天然保持列坐标连续性）。覆盖Transformer attention全流程：Linear layer(analog in-situ VMM)、Q×K^T(DIA-based SDDMM)、Softmax(in-situ max/exp)、S×V(DIA-based SpMM)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
In-situ S×V (Algorithm 1, n=6,d=2,ω=2):
```
Step1 Mapping: V每维存独立ReRAM array; DIA S均分到d arrays
Step2 Vec-Vec Multi(Iter1): Arr0: I_col0=DI_{-1}⊙V[:,0]; Arr1: I_col1=DI_0⊙V[:,1]
Step3 Transfer: rotate DIA vectors to next array
Step4 Vec-Vec Multi(Iter2): Arr0: I_col2=DI_0⊙V[:,0]; Arr1: I_col3=DI_{-1}⊙V[:,1]
Step5 Decompress: center diagonals恢复; grey cells按(Rd,Ro)映射
Step6 Accumulate: Z[:,0]=I_col0+I_col2; Z[:,1]=I_col1+I_col3
```
In-situ SDDMM (Algorithm 2):
```
for each DI_i: shift Q per DI→copy modified rows per (Rd,Ro)→parallel vec-vec multi→SlicesS_i
gather SlicesS_i→accumulate to S matrix
```
关键：所有ReRAM rows并行计算→每迭代O(1)延迟→总迭代=ω≪n→O(n)总延迟(vs CSR O(n²))。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
依赖三个硬件原语：(1) ReRAM行并行vec-vec multi——激活WL DRV，BL读取电流/电压或数字状态，1 cycle完成整行；(2) Memory controller shift——按DI调整行选择器地址，O(1) cycle；(3) (Rd,Ro) memory copy——DMA-like intra-array copy。不限于ReRAM——任何支持行并行in-situ计算的非易失存储器(PCM, STT-RAM, Modified DRAM)均可适用。

涉及论文标题：
- 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf

---

## Codebook Cache (Hierarchical GPU Memory Placement for VQ Dequantization)

术语解释
Codebook Cache是VQ-LLM提出的GPU kernel级软件抽象，将VQ算法中codebook entries按access frequency分层放置在GPU memory hierarchy的不同层级（register、shared memory、global memory），以优化dequantization时的codebook访问效率。

术语是什么？
Codebook Cache利用了一个关键observation：codebook entries的access frequency高度不均匀——例如AQLM-3中近一半entries低于平均frequency，但有26个hot entries的访问频率超过µ+3σ。基于此，Codebook Cache采用分层放置策略：(1) Hot entries（frequency > µ+3σ）→ thread-local registers——消除bank conflict；(2) Medium entries → shared memory——低延迟on-chip访问；(3) Cold entries（frequency < average）→ global memory——节省有限的shared memory。实现采用reorder-based static mapping：offline对entries按frequency降序排序并重编号索引（最频繁=index 0），runtime通过简单index比较定位entry所在level（index < nreg→register，nreg ≤ index < nshared→shared，≥nshared→global），无tag array或complex eviction policy。分层边界nreg和nshared通过GPU resource slack（不影响SM occupancy的resource余量）自适应确定。

从kernel调度角度拆解：
Codebook Cache的运行时操作（per thread block in VQ dequantization kernel）：

```
// Kernel launch前（offline）:
sorted_entries = sort_by_frequency(codebook_entries)       // 按access frequency降序
nreg = min(hot_count, free_regs / entry_size)              // register中放hot entries
nshared = min(hot_count + medium_count, free_smem / entry_size)  // shared memory中放medium
// Reorder codebook: entries[0:nreg]→register, entries[nreg:nshared]→shared, entries[nshared:]→global

// Kernel runtime (per thread):
__device__ float4 codebook_access(int index) {
    if (index < nreg) {
        return reg_codebook[index];        // Thread-local register, 0 bank conflict
    } else if (index < nshared) {
        return smem_codebook[index - nreg]; // Shared memory
    } else {
        return gmem_codebook[index - nshared]; // Global memory
    }
}

// Dequantization per quantized element:
for each quantized_index in thread's tile:
    centroid = codebook_access(quantized_index)
    dequantized_data[offset] = centroid
```

术语一般如何实现？如何使用？
Codebook Cache利用CUDA的register和shared memory作为可编程cache——不同于硬件管理的L1 cache（对codebook entries仅12.45% hit rate，因entry size和irregular access pattern与128B cache line不匹配），Codebook Cache由软件精确控制。Adaptivity通过resource slack实现：不同computation kernel有不同register/smem使用模式，Codebook Cache使用不影响occupancy的slack空间。Cache提供三个API：Load(CB, Slack)→缓存codebook并返回边界；Access(CB_cached, boundary, CB, index)→定位并返回entry；Switch(New_CB_Ptr)→切换codebook（适配per-channel codebook算法如GPTVQ）。

涉及论文标题：
- 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf

---

## Codebook-Centric Dataflow

术语解释
Codebook-Centric Dataflow是VQ-LLM提出的GPU kernel并行化策略：沿codebook switch axis切分并行化任务，使每个thread block仅需load一个codebook，消除多thread block重复加载相同codebook导致的off-chip memory traffic。

术语是什么？
Naive并行化（如FlashDecoding沿token axis切分attention）导致不同thread block访问相同codebook→重复的Global→Shared traffic，甚至高于FP16版本的off-chip traffic。Codebook-Centric Dataflow改为沿codebook switch axis切分——例如Attention沿(H, C) axis切分（每个codebook对应一组channels），GeMM沿codebook所在的M或N axis切分（GPTVQ per-channel codebook）。切分后每个thread block仅覆盖一个codebook的范围，无需切换或重复加载codebook。代价是原本沿该axis的temporal reduction变为跨thread block的global reduction——split_factor通过平衡公式自适应确定（Traffic_Reduce = Traffic_Codebook → split_factor = sqrt(Original_Codebook_Traffic / Output_Size)）。

从kernel调度角度拆解：
```
// Baseline (FlashDecoding沿token axis)：
Parallel_For(tokens):
    For each codebook (4 channels each):
        Load codebook from global to shared   // 多block重复load相同codebook！
        Dequantize → Partial Q·K^T
    Global softmax

// VQ-LLM Codebook-Centric Dataflow:
Parallel_For(codebook_switch_axes=(H,C), split_factor):
    Load one codebook to cache  // 仅一次，无重复
    For each token in block's range:
        Dequantize K/V → Partial attention
    Reduce(partial_result, reduce_axes ∩ codebook_switch_axes)  // Global reduction
```

术语一般如何实现？如何使用？
split_factor由adaptive heuristic确定：Trade-off between global reduction overhead和codebook duplicate traffic。在GeMV上效果显著（output size小、reduction overhead小），在GeMM上需谨慎（output size大、reduction overhead可能与codebook traffic saving抵消）。对于不同VQ配置，codebook switch axes不同：QuiP#-4/AQLM-3的GeMM中=R；GPTVQ-2=M,N；CQ/Attention中=H,C。

涉及论文标题：
- 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf

---

## Intra-Warp Shuffle for Register-Level Dequantization-Computation Fusion

术语解释
Intra-Warp Shuffle是NVIDIA GPU的CUDA编程特性，允许同一warp内32个线程直接交换register值而无需通过shared memory。VQ-LLM首次将此特性应用于VQ dequantization-computation融合：当dequantized data layout（VQ vector_size决定）与后续computation所需layout（如mma指令要求layout=2）不匹配时，通过shfl_xor在mini-warp内交换register数据完成重排，消除shared memory round-trip traffic。

术语是什么？
NVIDIA GPU的shuffle指令（shfl_sync, shfl_xor_sync等）允许warp内线程交换32-bit register值，通过hardware crossbar在单cycle内完成。VQ-LLM利用shfl_xor实现register-level fusion：将dequantization threads按预定mapping分组为mini-warp（共享相同dequantized data的线程子集），通过数次shuffle将dequantized data重排为computation所需layout。例如VQ<8,...> dequantization + mma computation（mma.sync要求每线程hold 2 elements，而VQ dequantized 8 elements/thread）→通过3次shfl_xor在mini-warp（4 threads）内完成8→2重排。

从kernel调度角度拆解：
```
// VQ<8,...> + mma.sync (layout: 2 elements/thread)
// Mini-warp: 4 threads sharing the same 8 dequantized elements
// Before: tid0=[0,1,2,3], tid1=[0,1,2,3], tid2=[4,5,6,7], tid3=[4,5,6,7]
// After:  each thread holds 2 elements matching mma

reg[tid^1] = __shfl_xor_sync(mask, reg[tid^1], 1);  // tid0[1]↔tid1[0], tid2[3]↔tid3[2]
reg[tid^2] = __shfl_xor_sync(mask, reg[tid^2], 2);  // tid0[2]↔tid2[0], tid1[3]↔tid3[1]
reg[tid^3] = __shfl_xor_sync(mask, reg[tid^3], 3);  // tid0[3]↔tid3[0], tid2[1]↔tid1[2]
// Ready for mma.sync computation
```

术语一般如何实现？如何使用？
Adaptive选择：profiling结果表明shared memory access latency ≈ 5× (register + shuffle latency)，因此nshuffle ≤ 5时使用register-level fusion（shuffle），否则fallback到shared memory-level fusion。Thread mapping在offline确定——对每个dequantized element，找到需要它的compute threads并分组为mini-warp，确保所有shuffle局限在mini-warp内。Shuffle API的通用用途包括warp-level reduction、data broadcast和但terfly permutation——VQ-LLM是首次将其用于VQ dequantization layout transformation。

涉及论文标题：
- 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf

## PIM Access Scheduling (PAS, 存内计算访问调度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM Access Scheduling (PAS) 是IANUS提出的面向NPU-PIM统一内存系统的调度框架，解决PIM memory同时服务NPU normal memory access和PIM computation时的资源冲突和数据依赖问题。在统一内存系统中，同一PIM bank不能同时执行normal read/write和PIM compute——两者互斥。PAS包含三层调度：(1) Workload Mapping——基于analytical model在compile time决定每个FC操作映射到MU还是PIM；(2) Mapping-Aware Scheduling——为multi-head attention的QK^T和SV操作选择执行单元（PIM或MU）并排布流水线；(3) Memory Access Scheduling——运行时协调macro PIM command和DMA normal access的分时复用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PAS的核心调度逻辑（以GPT-2 decoder的generation step为例）：

```
// Phase 1: Compile-time Workload Mapping (Algorithm 1 in IANUS)
for each cmd in ordered_commands:
    if cmd.type == MU_FC:
        prev_cmd = CMDs[i-1]
        if prev_cmd.type == VU:
            t_prefetch = VU_model(n, prev_cmd.dim)  // weight prefetch overlap with VU
        w_cfg = cmd.weight_config
        w_load = DMA_weight(w_cfg.row, T)           // DMA weight loading time per tile
        mu_tile = MU_FC(n, w_cfg.row, T)            // MU compute time per tile
        mu_unpipe = mu_tile + w_load                 // First tile: load + compute
        mu_pipe = (w_cfg.col / T) * max(w_load, mu_tile)  // Pipeline: overlap load+compute
        mu_total = mu_unpipe + mu_pipe - t_prefetch
        pim_time = n * PIM_model(w_cfg.row, w_cfg.col)  // PIM: n tokens × per-token time
        if pim_time < mu_total:
            cmd.type = PIM
            if cmd is first_FC_of_FFN:
                next_GELU_cmd.type = PIM  // Chain GELU to PIM

// Phase 2: Mapping-Aware Scheduling for Multi-Head Attention
// Summarization: FC_Q/K/V → MU (matrix-matrix), QK^T/SV → MU or PIM
// Generation: FC_Q/K/V → PIM (matrix-vector), QK^T/SV → PIM
// Schedule: generate K first → key transposition (DMA) parallel with V generation
for each head i in [0, h-1]:
    MU: W_Q^i @ x → Q_i            // Head i Q projection
    MU: W_K^i @ x → K_i            // Head i K (earliest)
    DMA: AM→WM K_i transposition   // On-chip, via streaming buffer, parallel with V
    MU: W_V^i @ x → V_i            // Head i V
MU/PIM: Q_i @ K_i^T → S_i           // Inter-head pipeline
VU: softmax(S_i, 1-bit mask)
PIM/MU: S_i @ V_i → Z_i

// Phase 3: Runtime Memory Access Scheduling
scheduler:
    when macro_PIM_cmd.ready:
        for each unissued DMA_to_offchip: DMA.state = WAIT
        PCU.issue(macro_PIM_cmd)
    PCU:
        micro_cmds = decode(macro_PIM_cmd)  // → 128 bank-level cmd sequences
        NoC.broadcast(micro_cmds)
        wait_all_completions() → signal scheduler → release DMA
```

从kernel调度角度，PAS的关键权衡在Algorithm 1体现：input tokens少（generation, n=1）→PIM无weight loading overhead；input tokens多（summarization, n=512）→MU 128-wide systolic array parallelism + weight loading pipelining优势。Algorithm 1在4-16 input token范围内达94%准确率。QK^T/SV到PIM时省去K/V cache loading（PIM直接从memory读取previous K/V），GPT-2 2.5B (head_dim=96)获24% improvement。整体mapping-aware scheduling平均34%提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAS实现依赖：(1) Compiler生成ordered commands含dependency和weight dimension供analytical model使用；(2) Analytical model需calibrate MU (128×64 systolic array with column-tiling), DMA (256 GB/s external BW), VU (16 VLIW), PIM (1024 GB/s internal BW, GDDR6 timing)；(3) Command scheduler pending queue (256 slots) + per-unit issue queues (4 slots each) tracking状态；(4) PCU macro→micro decode table。PAS的mapping-aware scheduling在GPT-2 XL上FC speedup 4.1× vs NPU-MEM, FFN 5.1×, self-attention 4.3×（因weight prefetch替代K/V cache loading）。类似概念：无完全相同的NPU-PIM统一内存调度方案已公开。

涉及论文标题：
- 30-IANUS- Integrated Accelerator based on NPU-PIM Unified Memory System.pdf

## NOrec STM Algorithm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NOrec（No Ownership Records）是由 Dalessandro et al. [PPoPP 2010] 提出的 STM 算法，完全取消 Ownership Records（ORec），仅使用**单个全局 sequence lock**（带时间戳的锁）来序列化所有 update 事务的提交事件。NOrec 属于 coarse-grained metadata 设计：所有冲突检测通过 sequence lock 的递增和 readset value-based validation 完成，无需像 ORec-based 方案那样为每个 memory word/region 维护独立的 lock 和 version 元数据。

核心权衡：i) 优势——元数据量极小（仅一个全局 sequence lock），读写操作的 instrumentation overhead 低，在高争用场景下因事务执行快（少 overhead）而降低冲突概率；ii) 劣势——每当检测到并发 update 事务提交（sequence lock 递增），必须验证整个 readset（逐地址重新读取检查值是否被覆盖），readset 大且低争用场景下这些 validation 频繁但无效。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

NOrec 在 UPMEM DPU 上的伪代码执行流程（PIM-STM 实现）：

```
// === 事务开始 ===
tx_begin():
    while sequence_lock.is_locked():  // backoff等待
        delay()
    tx.snapshot = sequence_lock.value  // 记录当前lock值
    tx.readset = []   // 读集（地址+值）
    tx.writeset = []  // 写集（地址+新值）

// === 读操作 ===
tx_read(addr):
    // 1. 检查写集（Write-Back: 先查自己是否改过）
    if addr in tx.writeset:
        return tx.writeset[addr].new_value
    
    // 2. 检查是否有并发update事务提交
    if sequence_lock.value != tx.snapshot:
        // 3. Value-based validation: 逐地址检查readset
        for (r_addr, r_val) in tx.readset:
            if *r_addr != r_val:
                abort()  // 值被覆盖→冲突→abort
        tx.snapshot = sequence_lock.value  // 更新snapshot
    
    // 4. 执行实际读取
    val = *addr  // MRAM/WRAM load
    tx.readset.append((addr, val))
    return val

// === 写操作（Write-Back策略）===
tx_write(addr, new_val):
    tx.writeset[addr] = new_val  // 缓冲写，不直接写入共享内存

// === 提交 ===
tx_commit():
    if tx.writeset is empty:
        return SUCCESS  // 只读事务无需lock
    
    // 1. Acquire全局sequence lock（UPMEM: acquire原子指令模拟CAS）
    acquire_lock(&sequence_lock)
    
    // 2. 写回所有缓冲值到共享内存
    for (addr, new_val) in tx.writeset:
        *addr = new_val  // MRAM/WRAM store
    
    // 3. Increment sequence lock → 通知所有并发事务
    sequence_lock.value += 1
    
    // 4. Release lock
    release_lock(&sequence_lock)
    return SUCCESS

// === 中止 ===
tx_abort():
    tx.writeset.clear()   // WB: 直接清空写集（未写入共享内存）
    tx.readset.clear()
    retry()  // 重新执行事务
```

NOrec 的三个关键机制：
1. **Backoff 等待**：事务开始时若 sequence lock 被持有则等待。这在高争用下作为隐式争用管理——减少同时活跃的 update 事务数。
2. **Value-based validation**：每读操作后检查 sequence lock 是否递增→若递增则逐地址对比 readset 中的值是否被修改。这是 NOrec 替代 ORec 的核心机制——用时间换元数据开销。
3. **Commit 时的全局序列化**：所有 update 事务串行 commit（通过 sequence lock），保证了 opacity。

在 UPMEM 上的适配：用 acquire/release 模拟 CAS 来操作 sequence lock——先 acquire 对应 bit→检查 lock 值→修改→release。sequence lock 的 contention 管理对 UPMEM 的有限并行度（有效 11 tasklet）尤其有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NOrec 的实现要点：
- **CPU 实现**：使用硬件 CAS 指令操作 sequence lock。Dalessandro 的原始 NOrec [PPoPP 2010] 额外支持简单的 priority-based contention management（记录事务的 restart 次数）。
- **PIM 实现**（PIM-STM）：适配 UPMEM 无 CAS 限制，用 acquire/release 原子原语模拟 CAS。metadata 可按编译宏置于 WRAM（64KB，加速）或 MRAM（64MB，节省 WRAM 给应用数据）。
- **适用场景**：高争用 + 小 readset/writeset 事务（如 Linked-List HC、KMeans HC、ArrayBench B）。PIM-STM 实验中 NOrec 在 75% workloads 中达最佳峰值吞吐量。
- **不适用场景**：低争用 + 大 readset 事务（如 ArrayBench A，read 120 个位置但极少冲突）——NOrec 的频繁全 readset validation 触发 expensive MRAM reads，比最佳 VR 变体慢 2.5×。

类似算法：TinySTM、SwissTM、TL2 均采用 ORec 方案（细粒度 metadata），与 NOrec 形成 coarse vs fine metadata granularity 对比。

涉及论文标题：
- 31-PIM-STM- Software Transactional Memory for Processing-In-Memory Systems.pdf

## STM Concurrency Control Design Space

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

STM 并发控制设计空间指 STM 实现中的四个关键正交维度，它们的组合决定了 STM 在不同 workload 特性下的性能表现。PIM-STM 系统化探索了这 4 个维度在 UPMEM PIM 架构上的 7 种可行组合：

1. **Metadata Granularity（元数据粒度）**：决定冲突检测的粒度。
   - **ORec-based（细粒度）**：每个 memory word/region 维护独立的 Ownership Record（含 lock + version），事务可检测到具体哪一地址冲突→减少 unnecessary abort。代表：Tiny [Felber et al.]、TL2 [Dice et al.]。代价是更多 metadata 访问 overhead（每个 read/write 需访问 lock table）。
   - **NOrec-based（粗粒度）**：仅一个全局 sequence lock，无 per-address metadata。代表：NOrec [Dalessandro et al.]。优势是 instrumentation overhead 低，代价是无差别 validation 多。

2. **Read Visibility（读可见性）**：
   - **Invisible Reads**：读操作不留下可被其他事务检测的痕迹，事务不知道"谁在读什么"。代价：需要 readset validation 验证观察到状态的有效性。代表：Tiny、NOrec。
   - **Visible Reads**：读操作通过获取 rw-lock 的 read-mode 在元数据中注册——其他事务发现该数据正被读取。优势是避免 readset validation，代价是读操作需写共享元数据（acquire rw-lock 的 read bit）。代表：VR (Visible Reads) 系列。

3. **Lock Timing（锁时机）**：
   - **Encounter-Time Locking (ETL)**：事务执行期间即获取写锁。优势是冲突早检测→减少 waste work；代价是锁持有时间长→增加争用。
   - **Commit-Time Locking (CTL)**：推迟到 commit 时才获取写锁。优势是锁持有时间短→减少争用；代价是冲突晚检测→更多 waste work + 每次读都需扫描写集（检查 reads-after-writes）。

4. **Write Policy（写策略）**：
   - **Write-Back (WB)**：写缓冲在私有写集，commit 时一次性写回共享内存。优势是 abort 时无 undo cost；代价是 commit 时有 copy overhead。
   - **Write-Through (WT)**：写直接到共享内存，旧值存入 undo log。优势是读操作直接从共享内存获取最新值（无需写集查找）；代价是 abort 时需 undo log 恢复旧值。WT 仅与 ETL 兼容（否则 uncommitted writes 被外部可见→违反 opacity）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PIM-STM 分类学中的 7 种可行组合及在 UPMEM MRAM 上的性能特征：

```
Design Space Taxonomy（图2，PIM-STM论文）：
                    Read        Lock        Write
                    Visibility  Timing      Policy
NOrec       ────  Invisible    CTL         WB
Tiny ETLWB  ────  Invisible    ETL         WB
Tiny ETLWT  ────  Invisible    ETL         WT
Tiny CTLWB  ────  Invisible    CTL         WB
VR ETLWB    ────  Visible      ETL         WB
VR ETLWT    ────  Visible      ETL         WT
VR CTLWB    ────  Visible      CTL         WB
```

不可行组合：
- WT + CTL → 不可行（uncommitted write 直接写入 shared memory 违反 opacity）
- NOrec + Visible Reads → 不切实际（track reader 但不 track 读什么无意义，仅增 overhead）
- NOrec + ETL → 不希望（跟踪进行中事务的 write → 频繁 atomic update sequence lock + 更多无效 validation）

在 UPMEM 架构上的关键发现（metadata 置于 MRAM）：
- **Read Visibility**：Invisible reads (Tiny) 在高争用下优于 VR（VR 的 rw-lock upgrade 导致 spurious abort），但 VR 在低争用+大 readset 下快 2×——VR 的 read-mode lock acquisition 在 UPMEM 上极轻量（基于寄存器 atomic 操作，不访 MRAM/WRAM），省去了 Tiny/NOrec 的昂贵 MRAM readset validation。
- **Lock Timing**：ETL 整体优于 CTL——CTL 减少 abort rate 的收益被更大的 waste work 和更高的 read cost（写集扫描）抵消。
- **Write Policy**：WB vs WT 在 metadata 在 MRAM 时影响极小（MRAM access 数相近）; metadata 在 WRAM 时 WB 在高争用下优势显现（up to 14% throughput increase）。
- **Metadata Granularity**：NOrec 在 average/median 上最稳健，但 workload 特性决定最优选择→"no one-size-fits-all"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

设计空间的选择由 workload 特性驱动：
- **大 readset + 低争用**（如 ArrayBench A, KMeans LC）→ VR + ETL + WB/WT。避免昂贵 validation，read-mode lock overhead 低。
- **小 writeset + 高争用**（如 ArrayBench B, Linked-List HC）→ NOrec。backoff 机制 + 低 abort cost + 简单快速执行。
- **中等争用**（如 Linked-List LC）→ NOrec 或 Tiny ETL 均可，write policy 影响不显著。
- **metadata 在 WRAM**：Tiny ETL 与 NOrec 竞争最激烈——ORec 的 WRAM 访问加速显著缩小与 NOrec 的差距。

PIM-STM 提供编译宏切换实现（如 `-DSTM_NOrec` / `-DSTM_Tiny_ETLWB` / `-DSTM_VR_ETLWB` 等），开发者可在不修改应用代码情况下测试不同 STM 设计。

涉及论文标题：
- 31-PIM-STM- Software Transactional Memory for Processing-In-Memory Systems.pdf

## Sequence Lock in STM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sequence Lock（序列锁）是 STM 中用于序列化 update 事务提交事件的核心同步原语。它是一个带时间戳的互斥锁：包含一个 lock bit（标记是否有事务正在 commit）和一个 version counter（每次成功的 update 事务 commit 后递增）。读事务通过观察 sequence lock 的值来检测并发 update 事务的提交——若 sequence lock 的值与事务开始时记录的 snapshot 不同，说明可能有冲突。

Sequence lock 的核心作用是替代细粒度的 per-address metadata（ORec）：不跟踪"哪个地址被修改"，只捕获"是否有事务提交了修改"。读事务检测到 sequence lock 变化后，通过逐地址验证 readset 来确认自己读取的值是否仍有效——这是一种 coarse-grained conflict detection。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Sequence lock 在 NOrec 中的运作机制：

```
// Sequence lock 结构
struct seq_lock {
    uint64_t value;  // bit[0]: lock flag, bit[63:1]: version counter
};

// 事务开始时记录 snapshot
tx_begin():
    while (seq_lock.value & 1):  // wait if locked
        backoff();
    snapshot = seq_lock.value;   // snapshot = current version

// 每次读后检查
tx_read(addr):
    if (seq_lock.value != snapshot):  // 有并发update事务commit?
        // 逐地址验证readset
        for each (r_addr, r_val) in readset:
            if *r_addr != r_val:
                abort_and_retry();
        snapshot = seq_lock.value;  // 更新snapshot
    
    val = *addr;
    readset.append((addr, val));
    return val;

// Commit时更新sequence lock
tx_commit():
    acquire_lock(&seq_lock);       // set bit[0]=1
    for each (addr, val) in writeset:
        *addr = val;               // write back
    seq_lock.value += 2;           // increment version, clear lock bit
    // 等价于: clear bit[0], increment version
```

Sequence lock 的两种用途：
1. **冲突检测信号**：读事务通过 `seq_lock.value != snapshot` 检测到并发 update 事务的 commit→触发 readset validation。
2. **Commit 序列化**：所有 update 事务的 commit 阶段串行化（通过 acquire sequence lock），保证 opacity。

在 UPMEM 上：sequence lock 通过 acquire/release 原子指令操作——acquire 对应 bit→检查 lock 状态→CAS 式更新→release。因 sequence lock 操作极短（不访 MRAM/WRAM，仅寄存器级操作），lock aliasing（多个地址 hash 到同一 atomic bit）对性能影响 negligible。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CPU 实现使用硬件 CAS 或 atomic_fetch_add 指令。PIM (UPMEM) 实现使用 acquire/release 模拟。Sequence lock 的 backoff 策略（NOrec 中事务开始前等待 lock 空闲）作为 implicit contention management：减少同时活跃的 update 事务数→降低冲突概率。类似概念：Linux kernel 的 seqlock 用于保护读多写少的数据结构（如 jiffies 时钟计数器）。

涉及论文标题：
- 31-PIM-STM- Software Transactional Memory for Processing-In-Memory Systems.pdf

## Version Clock Validation in STM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Version Clock Validation（版本时钟验证）是 Tiny STM [Felber et al.] 使用的冲突检测机制，基于 per-address ORec 中的 version number。每个事务维护一个 snapshot 窗口：lower bound（事务开始时的全局时钟）和 upper bound（可扩展的上界）。当事务尝试读取一个 version 高于 upper bound 的地址时，执行 snapshot extension——验证之前读取的所有地址的 version 未变，若通过则扩展 upper bound 到当前全局时钟。此机制允许事务在 snapshot extension 成功后继续执行而非直接 abort，比简单设计（如 TL2）更灵活。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Tiny STM 的 version clock validation 伪代码（UPMEM 适配）：

```
// 全局变量
global_clock = 0;         // 全局时钟（单调递增）
lock_table[N_ENTRIES];    // 每个entry含: lock_bit + version

// 事务元数据
struct tx_snapshot {
    lower_bound;  // 事务开始时的global_clock
    upper_bound;  // 当前可见的最大version
};

tx_begin():
    tx.lower_bound = global_clock;
    tx.upper_bound = global_clock;

tx_read(addr):
    entry = lock_table[hash(addr)];
    
    // 1. 检查地址version是否在可见窗口内
    while entry.version > tx.upper_bound:
        // 2. Snapshot extension: 验证整个readset
        for each (r_addr, r_orec) in readset:
            lock = lock_table[hash(r_addr)];
            if lock.version != r_orec.saved_version:
                abort();  // 有地址被修改→extension失败→abort
        
        // 3. Extension成功→扩展upper bound
        tx.upper_bound = global_clock;
    
    // 4. 检查该地址是否被写锁
    if entry.is_locked():
        abort();  // 有并发事务正在写→冲突
    
    // 5. 记录此次读
    val = *addr;
    readset.append((addr, entry.version));  // 保存entry version供后续验证
    return val;

tx_write(addr, val):
    entry = lock_table[hash(addr)];
    // ETL: 立即acquire write lock
    acquire_write_lock(entry);
    writeset.append((addr, val, entry.old_version));

tx_commit():
    // CTL or ETL: 获取所有写地址的lock（CTL时此时才lock）
    // Increment global_clock
    new_clock = global_clock + 1;
    // 对每个write地址: 写入新值, entry.version = new_clock
    for each (addr, val) in writeset:
        *addr = val;
        lock_table[hash(addr)].version = new_clock;
        release_lock(lock_table[hash(addr)]);
    global_clock = new_clock;
```

Snapshot extension 的关键：当被读地址的 version 高于 upper bound 时，不直接 abort（这是 TL2 的做法），而是验证 readset→若所有之前读的地址 version 未变→扩展 upper bound。这让 Tiny 在某些场景下避免不必要的 abort，比 simpler invisible-read 设计更高效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Tiny 的 version clock validation 属于 ORec-based + invisible reads 方案，require lock table（编译时确定大小，trade off memory vs aliasing）。在 UPMEM 上：lock table 置于 MRAM 或 WRAM（WRAM 版本因低访问延迟显著受益）。Version clock 的优势在中等争用 + 中等 readset 场景最明显——足够大的 readset 使 snapshot extension 有意义，足够低的争用使 extension 成功概率高。UPMEM 实验中 Tiny ETL 在 metadata 置于 WRAM 时与 NOrec 竞争（在 25% workloads 中最佳）。

涉及论文标题：
- 31-PIM-STM- Software Transactional Memory for Processing-In-Memory Systems.pdf
- 31-UM-PIM_DRAM-based_PIM_with_Uniform_amp_Shared_Memory_Space.pdf

## Zero-copy PIM Task Offloading (零拷贝PIM任务卸载)

术语解释
Zero-copy PIM task offloading是一种PIM系统架构设计，使CPU能直接以virtual address访问PIM unit的local memory，消除传统PIM系统中CPU-PIM data transfer的显式内存拷贝、地址翻译和数据重排步骤。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

在传统PIM系统（如UPMEM）中，PIM task offloading需三步：(1) CPU-PIM transfer: CPU从main memory拷贝数据到PIM memory space，同时执行软件地址翻译（将virtual address逐byte映射为PIM unit的HWAddr）和数据重排（将interleaved data layout转换为PIM unit可见的contiguous layout）；(2) PIM execution: PIM unit读取本地数据执行计算；(3) PIM-CPU transfer: CPU从PIM space拷贝结果回main memory，再次地址翻译和数据重排。地址翻译+数据重排占总transfer时间~70%（UPMEM SDK测量），每次transfer有fixed overhead ~50μs。UM-PIM通过uniform & shared memory space实现zero-copy：CPU pages和PIM pages共存于同一物理地址空间，CPU通过malloc_pim()分配的PIM pages可直接以standard virtual address读写——CPU写PIM page的数据已被硬件（RCL+ATM+RC）自动映射为PIM unit可见的contiguous non-interleaved格式，CPU读PIM page时RC自动聚合device-level分散数据。Step 1和Step 3完全消除，CPU computation时间平均减少4.93×。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Zero-copy offloading在UM-PIM中的kernel调度流程（以BFS PIM workload为例）：

```
// === CPU Host Program (zero-copy) ===
// Step 1: Allocate PIM pages
pim_pages = malloc_pim(total_data_len);  // THP mmap→mlock→ACN to RCL/PCL
// Step 2: CPU directly writes input data to PIM pages (zero-copy)
for (i = 0; i < num_pim_units; i++) {
    memcpy(pim_pages[i], input_data[i], page_size);
    // No address translation: TAD1+UM-PIM Interface handles mapping
    // No data re-layout: PIM pages have non-interleaved chunk layout
}
// Step 3: Offload to PIM units
for (i = 0; i < num_pim_units; i++) {
    launch_pim_task(i, pim_task_bfs_kernel);
}
// Step 4: CPU directly reads PIM results (zero-copy)
wait_all_pim_tasks();
for (i = 0; i < num_pim_units; i++) {
    // Direct read PIM page — RC auto-filters device-level data
    process_result(pim_pages[i]);
}
// Step 5: Inter-PIM communication via shared memory space
gather_pim_results(dest_buf, pim_pages);   // CPU memcpy in PIM page space
broadcast_frontier(pim_pages, frontier);   // RC broadcast mode
```

对比传统PIM-Ioff的流程：
```
// === CPU Host Program (traditional PIM-Ioff) ===
// Step 1: Allocate isolated PIM memory space
pim_space = pim_alloc_reserved(total_data_len);
// Step 2: CPU-PIM data transfer (Copy + Translate + Re-layout)
for (i = 0; i < num_pim_units; i++) {
    temp_buf = malloc(page_size);  // temporary buffer for re-layout
    for (byte = 0; byte < page_size; byte++) {
        // Software address translation: VAddr→PAddr→HWAddr
        hw_addr = software_translate(input_data[i] + byte);
        // Software data re-layout: interleaved→contiguous
        reorder(temp_buf, input_data[i], hw_addr, page_size);
    }
    memcpy(pim_space[i], temp_buf, page_size);  // actual copy
}
// Step 3-5: Same PIM execution, then PIM-CPU reverse transfer
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Zero-copy PIM offloading依赖三个层面：(1) **系统层**：dual-track memory management — CPU pages用interleaved mapping (TAD0)，PIM pages用non-interleaved chunk-based mapping (TAD1)，malloc_pim()通过THP mmap+madvise分配256MB chunks。(2) **硬件层**：RCL识别page type→ATM执行below-bank address mapping→RC过滤和聚合device-level数据→PCL使PIM unit独立翻译VAddr。(3) **API层**：malloc_pim(len)替代标准malloc以分配PIM pages，CPU用标准C pointer访问PIM pages（memcpy, pointer dereference等均可用）。通信API：scatter/broadcast/gather/all-gather提供inter-PIM-unit数据传输优化。

涉及论文标题：
- 31-UM-PIM_DRAM-based_PIM_with_Uniform_amp_Shared_Memory_Space.pdf

## Inter-PIM Unit Collective Communication (PIM单元间集合通信)

术语解释
Inter-PIM unit collective communication是UM-PIM为PIM系统中跨PIM unit数据交换提供的集合通信API（Scatter/Broadcast/Gather/All-Gather），利用shared memory space和RC硬件加速实现高效的PIM-to-PIM数据传输。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

在传统PIM系统（如UPMEM）中，PIM unit间无直连通信，cross-PIM数据交换必须经CPU：PIM unit→CPU memory（PIM-CPU transfer）→CPU compute reduction/merge→CPU memory→PIM unit（CPU-PIM transfer）。每个collective操作需两次data transfer加一次CPU computation。UM-PIM中PIM pages在shared memory space中可直接被CPU访问→inter-PIM communication变为CPU在PIM page空间内的memcpy操作（无需穿过PIM-to-CPU memory boundary）。四种通信模式：(1) **Scatter**：从source bank读contiguous block→写入destination banks的PIM pages。(2) **Broadcast**：从source bank读block→利用RC broadcast mode（单burst写所有devices）写入所有destination banks。(3) **Gather**：从所有source banks读→汇聚到destination bank。(4) **All-Gather**：每bank读block→拼接→广播到所有banks。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以Scatter和Broadcast在UM-PIM中的kernel调度为例：

```
// === Scatter: src_bank[0:len] → dst_banks[0..#pim-1][*] ===
for (l = 0; l < len / 64; l++) {           // l: outermost (data block offset)
    for (p = 0; p < num_pim_units; p++) {   // inner loop
        // RC locality: same offset l → adjacent SB cells → MUX2 reuse
        memcpy(dst_pim_pages[p][l*64], src_pim_page[l*64], 64);
        // CPU memcpy from PIM page to PIM page — no CPU↔PIM boundary crossing
    }
}

// === Broadcast: src_bank[0:len] → all dst_banks[*] ===
rc_set_broadcast_mode(true);  // BCM instruction → RC broadcast mode
for (l = 0; l < len / 64; l++) {
    // Single memcpy broadcasts to all devices via RC broadcast mode
    memcpy(dst_pim_pages[0][l*64], src_pim_page[l*64], 64);
    // RC writes same 64B to all 8 device cells simultaneously
    // Each destination bank's RC propagates to its local DRAM
}
rc_set_broadcast_mode(false);

// === Gather: all src_banks[*] → dst_bank[0:len_aggregated] ===
for (l = 0; l < len / 64; l++) {           // l: outermost
    for (p = 0; p < num_pim_units; p++) {   // inner loop
        memcpy(dst_pim_page[l*num_pim_units*64 + p*64], 
               src_pim_pages[p][l*64], 64);
    }
}
```

Nested loop顺序优化原则：read access时br（bank）和l（offset）在外层、dr（device）在内层；write access时l/br/dr在外层、dw（write device）在内层。此顺序使相邻iteration的read/write命中同一device的相邻offset→RC中已缓存数据被复用→提升RC hit rate→DRAM read bandwidth接近CPU pages水平（仅~1.6× slower）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

UM-PIM提供四种API：(1) scatter(void* src, void* dst[], size_t len)；(2) broadcast(void* src, void* dst[], size_t len)；(3) gather(void* dst, void* src[], size_t len)；(4) all_gather(void* dst[], void* src[], size_t block_size)。API类似NCCL（NVIDIA Collective Communications Library）的设计风格，但运行在CPU-PIM shared memory space而非GPU multi-node环境。BCM指令在broadcast前设置RC broadcast mode→broadcast后unset。CPU程序使用这些API时，hardware configuration（bank/device数）从OS获取以自动确定nested loop顺序参数。在UM-PIM评估中，scatter/broadcast/gather分别达9.84×/12.6×/7.90× speedup vs PIM-Ioff（后者需2次data transfer = PIM-CPU + CPU-PIM per operation）。

涉及论文标题：
- 31-UM-PIM_DRAM-based_PIM_with_Uniform_amp_Shared_Memory_Space.pdf

---

## Partially Synchronous Execution for PIM (PIM部分同步执行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Partially Synchronous Execution（部分同步执行）是pSyncPIM [ISCA 2024]提出的核心执行模型，在commercial all-bank PIM架构的lock-step同步约束下，允许各bank的processing unit有限度地分歧执行路径以处理不规则稀疏矩阵操作。核心思想：所有bank仍共享同一memory command（保持JEDEC interface兼容），但每个PE的执行行为取决于自身状态（sparse vector queue空/满、CEXIT条件满足否）——同一command对某些PE执行实际操作，对另一些PE成为NOP（predicated execution）；每个PE在各自workload完成时独立退出无限循环（conditional exit）。这介于commercial PIM的完全同步all-bank执行和学术提案的完全异步per-bank执行之间。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
pSyncPIM的partially synchronous SpMV kernel伪代码（基于Algorithm 2和Section V描述）：

```
// Host预处理: sparse matrix A → row-wise partition → compact (remove all-zero columns)
// → distribute to 256 banks, 每个bank的sub-matrix ≤1KB (one memory row)

// 所有bank PE执行相同的PIM kernel (infinite loop):
1: Read row, col, values → SpVQ0  // DMOV: bank→sparse queue
2: loop:
3:     IndMOV scalar SRF ← Bank[SpVQ0.col_idx]  // indirect读input vector x
                                                  // 若SpVQ0.col_idx==-1 (sentinel): predicated skip
4:     SSpV SpVQ1 ← SRF ⊗ SpVQ0.val   // scalar×sparse vector multiply
5:     SpVDV DRF0 ← SpVQ1 ⊕ DRF0      // dense-sparse vector accumulate
                                        // 若SpVQ1 empty: predicated skip
6:     if (DRF0 full or row boundary):
7:         Write DRF0 → Bank          // 写partial output
8:     Read next batch → SpVQ0        // 读下一批non-zero
9:     CEXIT when SpVQ0 is empty      // 各自独立exit (不同bank不同timing)
10: end loop

// Host: 所有bank CEXIT后, host发出序列 → switch to SB mode
// Host: read各bank的partial output, accumulate only non-zero outputs → final y
```

关键调度特征：所有PE共享同一`IndMOV/SSpV/SpVDV/CEXIT` command stream。Predicated: PE_i的SpVQ0.col_idx==-1时IndMOV自动跳过；PE_i的SpVQ1为空时SpVDV自动跳过。CEXIT timing per-bank: 处理完分配的全部non-zero的PE立即退出；仍有元素的PE继续循环。空白位置用-1 sentinel填充以使各bank消耗相同memory row数量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
pSyncPIM的partially synchronous execution实现依赖三个硬件机制协同：
1. **Predicated Execution**：每个PE的指令执行逻辑检查本地状态flag（queue full/empty, register ready等）→ 状态不满足时指令变为NOP。Host发送同一command→部分PE执行，部分PE跳过。
2. **CEXIT（Conditional Exit）**：PE运行infinite loop processing kernel，当指定sparse vector queue为空时设置CEXIT flag。Host需检测所有PE是否全部exit。-1 sentinel在index queue中触发empty flag。
3. **Matrix Compaction**（矩阵压缩）：host-side预处理确保各bank获得压缩子矩阵（row-wise cut + all-zero column removal），使每个PE的workload虽有大小差异但能收敛于有限循环迭代。子矩阵≤1KB确保input/output vector fits in single memory row。
此执行模型相比完全异步PIM（如SpaceA [HPCA 2021]）的tradeoff：deployability↑（保持JEDEC interface）但performance↓（0.56× of SpaceA for SpMV）。

涉及论文标题：
- 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

---

## Conditional Exit (CEXIT, 条件退出指令)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CEXIT（Conditional Exit）是pSyncPIM [ISCA 2024]引入的PIM控制指令（属于C format，4B），用于实现partially synchronous execution中的per-bank独立kernel termination。与常规EXIT指令（所有PE同时退出）不同，CEXIT允许每个PE在自身sparse workload完成时独立退出无限循环——当CEXIT指定的sparse vector queue为空时，该PE设置CEXIT flag并停止执行循环体，而其他仍有workload的PE继续执行。这打破了all-bank PIM的lock-step termination约束。CEXIT使用-1 sentinel值填充index queue的空位来触发empty condition。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CEXIT在pSyncPIM SpMV kernel中的调度流程：

```
// PIM kernel instruction layout in 128B control register (max 32 instructions):
//   [0] DMOV  SpVQ0 ← Bank[row_addr]    // Load first batch
//   [1] JUMP  label_loop, ORDER=0, CNT   // Enter loop with counter
//   [2] label_loop:
//   [3] IndMOV SRF ← Bank[SpVQ0.col]    // Predicated - skip if col==-1
//   [4] SSpV  SpVQ1 ← SRF ⊗ SpVQ0      // Predicated - skip if empty
//   [5] SpVDV DRF0 ← SpVQ1 ⊕ Bank       // Predicated
//   [6] DMOV  Bank ← DRF0               // Write partial result
//   [7] DMOV  SpVQ0 ← Bank[next_addr]   // Read next batch
//   [8] CEXIT SpVQ0                      // Exit when SpVQ0 empty (-1 sentinel)
//   [9] JUMP  label_loop, ORDER=0, CNT-1 // Loop back if CNT>0

// Execution timeline (3 PEs with different workloads):
// PE_0 (workload=3 batch): ... → CEXIT not triggered → JUMP → continue
// PE_1 (workload=5 batch): ... → CEXIT not triggered → JUMP → continue  
// PE_2 (workload=0, all -1): ... → CEXIT triggered → PE_2 idle
// ... PE_0 exits at cycle T₁₀₀ → PE_1 continues ...
// ... PE_1 exits at cycle T₁₅₀ → all PEs CEXITed → host proceeds
```

CEXIT idle PE仍响应memory row activate/precharge commands（保持DRAM protocol），但不改变bank data。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CEXIT实现：C format 32-bit指令，OpCode field识别CEXIT操作，Imm0 field指定监测的sparse vector queue编号（0-2）。PE硬件中每个sparse vector queue有empty flag——queue中所有entry为-1 sentinel或pointer到达boundary时flag置位。CEXIT执行时检查目标queue的empty flag→若True则置PE的global CEXIT flag并进入idle；若False则CEXIT成为NOP。Host检测机制（论文未详细说明）可能方案：per-channel CEXIT status register（AND of all PE CEXIT flags）或timeout-based polling。CEXIT和EXIT共存——EXIT用于所有PE必然同时完成的同步点，CEXIT用于不均匀workload的分歧点。

涉及论文标题：
- 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

---

## Predicated Execution in PIM (PIM谓词执行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Predicated Execution（谓词执行）在pSyncPIM语境中指：多个PIM processing unit同时接收并decode同一条memory command/instruction，但实际是否执行该指令取决于每个PE的本地状态条件（predicate）。类似于GPU SIMT model中的predicated execution——warp内所有thread执行同一指令，但per-thread predicate mask决定哪些thread active。在all-bank PIM中，predicated execution是打破lock-step同步的关键机制之一：host发送一次command到所有bank，各PE根据自身sparse vector queue空/满状态、CEXIT flag等predicate条件自主决定执行或skip。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SpMV kernel中predicated execution的两个典型场景：

```
// 场景1: DMOV (load to sparse queue) predication
// Host: one DMOV SpVQ0 ← Bank[row_i] command for all 256 PEs
// PE_0..PE_100: SpVQ0有32B空位 → predicate=true → 执行load
// PE_101..PE_200: SpVQ0满 → predicate=false → 指令变NOP
// PE_201..PE_255: SpVQ0有32B空位 → predicate=true → 执行load

// 场景2: IndMOV (indirect scalar read) predication  
// Host: one IndMOV SRF ← Bank[SpVQ0.col_idx] command for all PEs
// PE_0: SpVQ0.col_idx=15 → predicate=true → read Bank[15]
// PE_1: SpVQ0.col_idx=-1 (sentinel) → predicate=false → skip
// PE_2: SpVQ0.col_idx=42 → predicate=true → read Bank[42]
```

PE idle期间仍响应memory row activate/precharge commands（保持bank DRAM protocol活跃），但不修改data。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
pSyncPIM的predicated execution实现于PE的instruction decode/execute stage：每条指令执行前检查opcode关联的硬件predicate signal——DMOV/SpMOV类push指令检查queue not_full；IndMOV/SSpV/SpVDV类pop指令检查queue not_empty且index≠-1。Predicate检查采用组合逻辑（不增加额外cycle），predicate=false时指令在1 cycle内变为NOP（不stall pipeline）。与GPU SIMT predicate的区别：GPU通过SIMT stack管理divergent branch reconvergence；pSyncPIM因每个PE仅处理自己的子矩阵、无reconvergence需求，predication仅用于all-bank共享command下的per-PE workload自适应。结合CEXIT，predication确保即使不同PE有不同数量的sparse elements，同一command stream可正确处理所有PE。

涉及论文标题：
- 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

## Set-based Miss Curve Sampling for DRAM Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Set-based Miss Curve Sampling是一种为low-associativity DRAM cache（如direct-mapped DRAM cache）设计hardware miss curve profiler的技术。传统LLC（高associativity）的utility monitor [Qureshi et al., MICRO'06]利用stack property（若access在capacity C hit，则对所有>C的capacity也hit）实现：单个size-C tagless monitor即可推导1到C的完整miss curve。但DRAM cache使用hash-based set mapping + low associativity（甚至direct-mapped），按set而非way partition，不满足stack property。Set-based sampling解决此问题：(1) 假设hash function使各sets的access分布均匀，则采样少量sets（k=32）的miss behavior可外推total behavior；(2) 对每种capacity case独立采样k=32 sets，而非试图用stack property推导。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NDPExt的Set-based Miss Curve Sampler的伪代码和运转：
```
// Hardware sampler structure (per sampler, monitors one stream)
struct MissCurveSampler {
    // c=64 capacity cases: 32KB to 256MB, geometric step factor 1.16
    struct CapacityCase {
        addr_t sample_sets[32];  // k=32 sampled set addresses, 4B each
        uint64_t hits[32];       // per-set hit count
        uint64_t misses[32];     // per-set miss count  
    } cases[64];
};

// Per-access processing (at target NDP unit, after SLB+AFA lookup)
void sampler_process_access(sid, elem_id, is_hit):
    sampler = unit.samplers[assigned_stream_of(sid)]
    elem_set = hash(elem_id)  // hash determines which set this element maps to
    
    for i in 0..63:  // for each capacity case
        // Static interleaving: element_set mod total_sets_in_case determines
        // if this set is one of the k=32 sampled sets
        if is_sampled_set(elem_set, capacity_case_i, k=32):
            idx = sampled_set_index(elem_set, k=32)
            if is_hit:
                sampler.cases[i].hits[idx]++
            else:
                sampler.cases[i].misses[idx]++

// End-of-epoch: scale and aggregate
function generate_miss_curve(sampler):
    for i in 0..63:
        // Scale: sampled k sets → total K sets for this capacity
        scale_factor = total_sets_in_case(i) / 32
        total_misses = sum(sampler.cases[i].misses) * scale_factor
        total_accesses = total_misses + sum(sampler.cases[i].hits) * scale_factor
        miss_curve[i] = (capacity_cases[i], total_misses / total_accesses)
    return miss_curve
```

关键参数：(1) k=32 sample sets per capacity case（更多sets提高精度但增加SRAM cost）；(2) c=64 capacity cases from 32 kB to 256 MB（geometric partition with factor 1.16=∛(256MB/32kB)）；(3) total SRAM per sampler = 32×64×4B = 8 kB（仅存储address和counters，不存储actual data）；(4) per-unit配置4个samplers, total 32 kB SRAM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hardware实现：(1) 每个sampler 8 kB SRAM（32 sets × 64 capacity cases × 4B addresses），包含set selection logic（static interleaving）和hit/miss counters；(2) 地址比较：存储sample set的起始地址 + interleaving stride，每个access通过modulo运算确定是否属于sample set；(3) 与prior work (Jigsaw [Beckmann et al., MICRO'16], Nexus [Zhang et al., ISCA'17]) 的sampler配置相似，精度经验证足够（Section VII-C sensitivity study证明不同k值对overall performance影响negligible）；(4) 限制：set-based sampling假设hash function使访问在各sets间均匀分布——对workloads with skewed set访问可能引入error，但NDPExt evaluated workloads均未出现此问题；(5) 相比way-based utility monitor (UMON)，set-based sampler的overhead更高（需为每种capacity case独立采样而非用stack property推导），但在low-associativity DRAM cache场景下是必要的权衡。

涉及论文标题：
- 34-Stream-Based_Data_Placement_for_Near-Data_Processing_with_Extended_Memory.pdf

## Max-Flow Sampler Assignment for Stream Monitoring

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Max-Flow Sampler Assignment是一种将有限数量的hardware miss curve samplers高效分配给workload中众多data streams的调度算法。在NDPExt中，每个NDP unit有S=4个samplers（每个sampler可监控1个stream），但workload可能有up to 256 streams。直接将samplers静态分配给streams可能导致部分stream未被覆盖。Sampler assignment的约束：(1) 每个sampler只能监控被local NDP unit访问的stream（该unit的512-bit bitvector中对应bit=1）；(2) 每个stream只需被任意1个访问它的unit监控即可。问题形式化：给定U个NDP units（每unit up to S streams to sample）和T个target streams，找最大数量的streams使每个被至少1个unit覆盖且每unit不超过S个streams。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Max-Flow Sampler Assignment的建模与求解伪代码：
```
// Input: 
//   bitvectors[U][512] - each unit's accessed streams (only T ≤ 512 active)
//   S = 4 - samples per unit
// Output:
//   assignment[U][S] - which stream each unit's sampler should monitor

// Step 1: Build flow network
Graph G:
    // Nodes
    source = SuperSource
    sink = SuperSink  
    for u in 0..U-1: unit_nodes[u]
    for t in active_streams: stream_nodes[t]
    
    // Edges
    for u in 0..U-1:
        add_edge(source → unit_nodes[u], capacity=S)  // max S streams per unit
    for u in 0..U-1:
        for t in active_streams:
            if bitvectors[u][t] == 1:
                add_edge(unit_nodes[u] → stream_nodes[t], capacity=1)  // unit can sample stream
    for t in active_streams:
        add_edge(stream_nodes[t] → sink, capacity=1)  // each stream needs 1 sampler
    
// Step 2: Run Edmonds-Karp max-flow
max_flow = EdmondsKarp(G, source, sink)
// Complexity: O(V*E²) where V = U+T+2 ≤ 768, E ≤ U*T ≤ ~32k
// Runtime: < 0.5ms on host processor for 512 streams (shown in Fig. 4b)

// Step 3: Extract assignment from flow result
for u in 0..U-1:
    sampler_idx = 0
    for t in active_streams:
        if flow(unit_nodes[u] → stream_nodes[t]) == 1:
            assignment[u][sampler_idx++] = t
```

例（Fig. 4a）：4 units × 4 streams。
- Edge weights: source→unit_i = 4, unit_i→stream_j (if accessed) = 1, stream_j→sink = 1
- Max flow = 4 (all streams covered): unit0→stream0, unit1→stream1+stream2, unit2→stream3
- Unit3 not assigned any stream initially — will be used in subsequent epochs if stream access patterns change

若无法fully cover（too many streams），NDPExt采用round-robin buffering：先采样subset，存储结果到host memory，下epoch采样剩余streams，直到覆盖所有streams。论文evaluated workloads中未遇到无法fully cover的情况。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) Edmonds-Karp算法在host processor runtime（非NDP）执行，因需全局bitvector信息，runtime <0.5 ms even for 512 streams；(2) 每epoch（50M cycles）结束时各unit的512-bit bitvector通过interconnect发送host；(3) Host runtime build flow network并以Edmonds-Karp求解 → 输出每个unit的sampler-to-stream assignment → 下发NDP units → samplers在下一epoch按assignment采集miss curves；(4) 实现依赖bitvector的compact表示（512 bits = 64 bytes per unit）使传输开销minimal；(5) 算法基于标准max-flow——也可替换为更高效算法如Dinic等但Edmonds-Karp在此scale下已足够快。此方法generalizable到任何需要assign finite profilers to monitored entities in distributed system的场景。

涉及论文标题：
- 34-Stream-Based_Data_Placement_for_Near-Data_Processing_with_Extended_Memory.pdf

---

## PIM-aware Memory Scheduler (PIM-MS)

术语解释
PIM-aware Memory Scheduler (PIM-MS)是PIM-MMU Data Copy Engine (DCE)内部的硬件级fine-grained memory调度器，利用DRAM↔PIM数据传输时不同PIM core目标地址互斥的特性，在cycle粒度对memory requests做aggressive reordering以最大化memory-level parallelism。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PIM-MS的调度策略基于一个关键insight：DRAM↔PIM数据传输过程中，不同PIM core的memory transactions的目标地址是互斥的（程序员在数据传输前已将输入数据partition并唯一分配给各PIM core），因此不存在true data dependency，可以安全地对所有memory requests进行任意reordering而不影响程序正确性。这与常规memory controller的FR-FCFS（First-Ready First-Come-First-Serve）调度有本质区别——常规memory controller必须保守地处理不同请求之间的潜在data dependency。在baseline UPMEM-PIM中，数据传输通过多线程 + OS CFS调度器实现——OS以~ms级quantum调度数据搬运线程，线程调度策略优先fairness而非memory channel的均衡利用，导致某些channel长期流量拥塞而其他channel空闲。PIM-MS将所有PIM core的传输请求统一在硬件层管理，以cycle粒度调度，远比OS线程调度精细。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PIM-MS的调度算法（Algorithm 1 from PIM-MMU paper）：

```
// Input: N PIM core entries, each = (src_base_addr, dst_base_addr)
// Stored in DCE's address buffer as "Offset" per PIM core
// Output: Ordered sequence of (src_addr, dst_addr) for memory controller

// Helper: Get PIM core ID from DRAM hierarchy coordinates
procedure get_pim_core_id(ra, bg, bk):
    return ra * num_banks * num_bankgroups + bg * num_banks + bk

// Helper: AGU translates physical→DRAM address and manages progress
procedure AGU(id):
    src_base, dst_base = base_addrs[id]
    src_addr = src_base + pim_cores[id].offset
    dst_addr = dst_base + pim_cores[id].offset
    pim_cores[id].offset += min_access_granularity
    return src_addr, dst_addr

// Initialization: reset all PIM core transfer progress
do-parallel channel:
    for ra ← 0 to num_ranks:
        for bg ← 0 to num_bankgroups:
            for bk ← 0 to num_banks:
                id = get_pim_core_id(ra, bg, bk)
                pim_cores[id].offset = 0

// Main scheduling loop: maximize MLP through fine-grained reordering
do-parallel channel:                          // (1) exploit all channels
    for bk ← 0 to num_banks:                  // (3) bank-level parallelism
        for ra ← 0 to num_ranks:
            for bg ← 0 to num_bankgroups:     // (2) bank-group interleaving
                id = get_pim_core_id(ra, bg, bk)
                src_addr, dst_addr = AGU(id)
                addrs.append(src_addr, dst_addr)
```

调度目标的三层优化：
1. **Channel-level parallelism（最外层do-parallel channel）**：同时向所有PIM channels发出memory requests，确保每个channel在任何时刻都有pending requests。
2. **Bank-group interleaving（中间层bg循环）**：连续column commands targeting不同bank groups → 最小化tCCD——DDR4在same bank group内连续column access需要tCCD_L（~4 cycles），不同bank group间仅需tCCD_S（~1 cycle）。
3. **Bank-level parallelism + row buffer conflict最小化（内层bk + AGU）**：同一bank group内对不同banks发请求；AGU翻译后的DRAM地址信息用于避免同bank的连续不同row access（activate-precharge overhead）。

对比baseline的OS CFS调度（~ms quantum, thread间channel流量不均衡），PIM-MS以64B cache line粒度交替服务所有PIM core。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PIM-MS的实现要点：(1) **硬件集成**——PIM-MS位于DCE内部，紧邻memory controller的request scheduler。接收address buffer中所有PIM core条目作为调度输入。(2) **与memory controller的接口**——PIM-MS通过标准request queue接口（64-entry per queue）发出请求，不改变memory controller的command scheduling逻辑。(3) **与FR-FCFS对比**——常规FR-FCFS调度器read优先于write，row-hit优先于row-miss，老请求优先于新请求。PIM-MS在此基础上增加了"跨所有PIM core均匀轮转"的top-level policy。(4) **实现开销**——PIM-MS主要由logic gates实现（调度状态机和少量comparator logic），面积可忽略。(5) **效果**——ablation study显示，单独DCE无PIM-MS时DRAM↔PIM吞吐量反而下降（失去AVX-512 wide vector并发优势）；加PIM-MS后PIM read/write吞吐量完全解锁，端到端数据传输加速4.1×（平均）。

涉及论文标题：
- 35-PIM-MMU_A_Memory_Management_Unit_for_Accelerating_Data_Transfers_in_Commercial_PIM_Systems.pdf

---

## Memory-Level Parallelism (MLP) in PIM Memory Systems

术语解释
Memory-Level Parallelism (MLP)指同时有多个outstanding memory requests在不同DRAM层级（channel/rank/bank-group/bank）上并行服务的能力。在PIM系统中，MLP面临独特挑战：PIM的bank-level架构要求输入数据localized在特定PIM core的bank内，与最大化MLP所需的fine-grained data interleaving across the DRAM subsystem相矛盾。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MLP是现代DRAM系统性能的基石。DRAM subsystem在多个层级提供并行性：(1) channel-level parallelism——多个独立memory channel可同时传输数据；(2) rank-level parallelism——同channel内的多个rank可interleave操作；(3) bank-group/bank-level parallelism——多个banks可同时处于不同状态。为最大化MLP，memory mapping function将物理地址中的频繁变化bits（接近LSB）映射到channel/bank地址，使连续访问均匀分散到所有可用资源。然而PIM系统与最大化MLP有直接冲突：PIM core仅能访问其本地bank数据，因此数据必须被localized在特定bank（而非fine-grained interleaved）。PIM-MMU论文揭示了PIM中MLP受限的三个根因：(a) 软件coarse-grained multi-threaded数据搬运受限于OS调度器的fairness优先policy；(b) PIM-specific BIOS memory mapping（locality-centric）将channel bits置于MSB，使所有访问都无法充分利用channel-level parallelism；(c) CPU中转（含transpose预处理）的OOO执行窗口和MSHR数限制了可同时发出的memory request数量。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以baseline UPMEM-PIM的DRAM→PIM数据搬运为例展示MLP如何受限：

```
// Baseline: Software multi-threaded (limited MLP ~15.5% utilization)
Thread_0: PIMcpy(ch=0, bk=0)  // 仅target channel 0
Thread_1: PIMcpy(ch=1, bk=0)  // 仅target channel 1
...
Thread_7: PIMcpy(ch=7, bk=0)  // 仅target channel 7

// OS CFS Scheduler (~ms quantum):
// T1: schedules Thread_0,1,3 → Ch 0,1,3 busy, Ch 2,4,5,6,7 idle
// T2: schedules Thread_2,4,5 → 流量倾斜至Ch 2,4,5
// Result: ~3-4/8 channels active per quantum → 15.5% BW utilization

// PIM-MMU with PIM-MS: Hardware fine-grained (MLP fully recovered)
// Iteration 1: serve bank 0 of ALL channels → Ch0-7 all active
// Iteration 2: serve bank 1 of ALL channels (bank-group interleaving)
//   → tCCD_S only 1 cycle (cross bank-group)
// Iteration N: continuously rotate through all banks × channels
// Result: all channels continuously active → high BW utilization
```

MLP在不同配置下的表现：PIM-MMU Figure 14展示2C-4R/4C-8R/4C-16R下DRAM→DRAM memcpy吞吐量——baseline的locality-centric mapping使吞吐量不随rank数增加，PIM-MMU的MLP-centric mapping使吞吐量随channel数线性增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Conventional DRAM controller中MLP通过以下机制实现：(1) Memory mapping function的XOR hashing确保连续cache lines映射到不同channels和banks；(2) FR-FCFS scheduling维护per-bank request queue，优先row-hit请求同时维持足够row-miss以利用bank-level parallelism；(3) Request reordering——不同bank的请求可安全reorder（不同bank间无hazard）。在PIM context中，PIM-MMU通过三重策略恢复MLP：(1) HetMap双套mapping恢复DRAM侧的MLP；(2) PIM-MS fine-grained reordering恢复PIM侧的MLP（利用无data dependency特性）；(3) DCE单线程offload消除CPU OOO window和MSHR限制。关键原则：PIM中的MLP优化需要HW/SW co-design——纯软件优化（多线程+AVX-512）或纯硬件优化（DMA offload）单独都不足以完全恢复MLP。

涉及论文标题：
- 35-PIM-MMU_A_Memory_Management_Unit_for_Accelerating_Data_Transfers_in_Commercial_PIM_Systems.pdf

---

## DRAM↔PIM Data Transfer with Chip Interleaving Transpose

术语解释
DRAM↔PIM数据传输是memory bus integrated PIM系统中的关键操作——在DRAM和PIM物理地址空间之间搬运数据，包括因DIMM chip interleaving引入的数据transpose预处理步骤。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

在DDR4 DIMM-based PIM系统中（如UPMEM-PIM），DRAM↔PIM数据传输因chip interleaving引入额外复杂性：在x8配置下，每个8-byte data word被以1-byte粒度分散到8个DRAM chip（chip 0存byte 0，chip 1存byte 1等）。这是因为每个DDR4 chip仅提供x8 data width——8 chips ×8 = 64-bit bus。常规DRAM中这对CPU透明（CPU通过cache读写），但PIM架构中每个chip含独立PIM core（bank-level design），PIM core仅能访问其本地chip/bank的数据。因此数据传输前必须执行transpose：将从8个chips读取的interleaved data重组为per-PIM-core contiguous格式——将数据排列为8×8 byte matrix（行=连续8-byte words即8 burst，列=8 chips），transpose该矩阵使同一chip的bytes聚合为连续8-byte data word。Baseline UPMEM-PIM用CPU执行此transpose（AVX-512），PIM-MMU将其offload到DCE hardware preprocessing unit。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Baseline UPMEM-PIM的DRAM→PIM数据传输三阶段（CPU-driven）：

```
// Stage 1: CPU reads from DRAM (AVX-512 load)
for (i = 0; i < num_cache_lines; i++):
    __m512i chunk = _mm512_stream_load_si512(&src[i*8])  // 512-bit = 64B
    
// Stage 2: CPU transpose (8×8 byte matrix)
for (j = 0; j < 8; j++):         // for each chip (0-7)
    for (k = 0; k < 8; k++):     // for each burst (0-7)
        transposed[j*8 + k] = chunk[k*8 + j]

// Stage 3: CPU writes to PIM (AVX-512 store)
for (i = 0; i < num_cache_lines; i++):
    _mm512_stream_store_si512(&pim_dst[i*8], transposed_chunk)
```

CPU开销：Stage 1+3利用AVX-512但non-cacheable PIM writes导致stall，core utilization ~100%，系统功耗~70W。输出带宽仅15.5%理论峰值（8.9 GB/s vs 57.6 GB/s）。

PIM-MMU DCE硬件offload流程——CPU进入sleep，全部三阶段在硬件中完成：
```
// DCE handles: read from DRAM → on-the-fly transpose → write to PIM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：(1) **纯软件（UPMEM baseline）**——dpu_push_xfer API，AVX-512 + 多线程，CPU负责全部三阶段。(2) **DMA offload（PIM-MMU DCE基础）**——AGU + data buffer + preprocessing unit释放CPU。Preprocessing unit用crossbar switch实现8×8 byte transpose。(3) **DMA + fine-grained scheduling（完整PIM-MMU）**——加PIM-MS和HetMap。(4) **数据搬运占比**——16个PrIM benchmark中DRAM↔PIM数据传输占端到端执行时间平均63.7%，最高99.7%。对于多数PIM workload，数据传输而非PIM kernel计算是真正bottleneck。(5) **关键参数**——address buffer大小（64KB）决定单次offload覆盖的PIM core数（512 cores × ~128B/entry ≈ 64KB）。

涉及论文标题：
- 35-PIM-MMU_A_Memory_Management_Unit_for_Accelerating_Data_Transfers_in_Commercial_PIM_Systems.pdf

## Task Offload (NDC Paradigm)

术语解释
Task Offload是Near-Data Computing (NDC)的核心范式之一：CPU core或另一个NDC action显式地将一个短任务（short task）发送到cache hierarchy中靠近目标数据的位置执行，完成后通常快速返回结果。任务是short-lived的，直接与调用方通信。Leviathan [MICRO 2024] 通过invoke指令和DYNAMIC scheduling实现该范式。

术语是什么？
Task Offload是Leviathan NDC taxonomy中的第一个范式，核心特征为：(1) **任务小而短**——通常为单个操作如atomic RMW、hash table node comparison、pointer chasing；(2) **显式触发**——由core或另一个NDC action通过invoke语义显式发送；(3) **与调用方通信**——通过Future<R>返回结果给调用方，或通过continuation-passing style链式调用。典型例子：Remote Memory Operations (RMOs) [39, 67]——core请求对cache/dRAM中的数据执行单个原子操作（如atomicAdd），避免数据在core间ping-pong。更复杂的task offload可涉及pointer chasing [31, 35]（遍历链表时每个节点触发下一个lookup task）和hash table bucket traversal [95]。Leviathan统一task offload和long-lived workloads的软件接口（均为invoke actor->action()），区别仅在于[location]参数和任务时长。

从kernel调度角度拆解术语：
以Leviathan hash-table lookup的task offload实现为例（Fig. 17）：
```
// Actor definition: a hash table node
class Node:
    int64 key, value
    int64 metadata[N]  // arbitrary padding, Leviathan handles layout
    Node* next

    // Action: offloaded lookup, runs near 'this' Node
    int64 Lookup(key):
        if this->key == key:
            return value          // found, return via Future<R>
        if next == nullptr:
            return -1              // not found
        return invoke next->Lookup(key)  // continuation-passing: chain to next node
```
核心调度特点：(1) 每个Lookup action在目标Node的LLC bank engine上执行（DYNAMIC probing确定位置）；(2) 若key不匹配，action通过`invoke next->Lookup(key)` spawn下一个task到next node的LLC bank——形成continuation-passing chain；(3) 所有中间tasks在LLC内完成，无需回core往返；(4) 最终结果（found value or -1）通过Future<R>返回给core。对比baseline：core必须反复从LLC fetch每个node（每次~100 cycle NoC round-trip），Leviathan将多条NoC traversal替换为单次invoke chain + 单次future return。

术语一般如何实现？如何使用？
实现需：(1) ISA扩展：invoke指令（含location参数LOCAL/REMOTE/DYNAMIC/EXCLUSIVE）；(2) Invoke buffer（per-core, 4-entry）提供背压——满时spill task回core执行；(3) Engine task-offload scheduler支持DYNAMIC probing（L1D→L2→LLC）和remote forwarding；(4) Future<R>通信原语——engine通过store-update指令 [30, 47] 将结果push到waiting core的register。在Leviathan中，带Future的offload tasks跳过invoke buffer因future wait自带背压。Task offload适用于需要快速响应的短操作（atomic RMW、lookup、pointer chase），不适合长时间计算（应使用long-lived workloads）。

涉及论文标题：
- 36-Leviathan_A_Unified_System_for_General-Purpose_Near-Data_Computing.pdf

## Data-Triggered Actions (NDC Paradigm)

术语解释
Data-Triggered Actions是NDC的核心范式之一：计算在数据于cache hierarchy中移动时隐式触发，而非由软件显式调用。触发点为cache insertion（数据被加载到cache）和cache eviction（数据被逐出cache）。Leviathan将此范式实现为actor constructor/destructor，通过Morph注册。

术语是什么？
Data-triggered actions的特征是：(1) **隐式触发**——不是由软件显式invoke，而是由cache controller在数据移动时自动触发；(2) **不与core直接通信**——action对core透明，core不感知其执行；(3) **短生命周期**——action仅处理当前数据移动事件（构造或析构对象）。典型应用包括：hardware prefetching（监控cache miss并触发额外数据请求 [5, 6, 74, 88]）、data compression/decompression（数据从DRAM进入cache时解压 [8, 24, 56, 57, 64, 77]）、PHI commutative scatter-update（cache insertion时初始化partial update buffer、eviction时conditionally apply或log updates [52]）。Leviathan中data-triggered paradigm操作的是phantom data——数据仅存在于cache中，不backed by off-chip memory [18, 66]，在insertion时构造、eviction时析构。

从kernel调度角度拆解术语：
以Leviathan decompression workload为例（Fig. 15），data-triggered constructor实现base+delta解压：
```
class Pixel:
    uint16 colors[3]  // 6B object, doesn't evenly divide 64B cache line

    // Constructor: data-triggered action, triggered on L2 cache miss
    Pixel(Decompressor* decomp):
        idx = decomp->getOffset(this)     // get object index in array
        bases = decomp->bases
        deltas = decomp->deltas
        for i in range(len(colors)):
            base = bases[i][idx >> 3]     // 1 base per 8 pixels
            delta = deltas[i][idx]
            mantissa = delta & 0b1111
            exponent = delta >> 4
            colors[i] = base + (mantissa << exponent)  // reconstruct pixel

class Decompressor extends Leviathan::Morph<Pixel>:
    uint16* bases[3]
    uint8* deltas[3]
    // registered at L2 level
```
调度流程：(1) Application注册Morph<Pixel> at L2 cache level，TLB entries标记该地址范围已注册；(2) Core load Pixel时L2 miss，L2 cache controller检查TLB bits发现Morph registered → 暂停正常fill → 发event到engine data-triggered scheduler；(3) Scheduler查vtable cache获取Pixel constructor地址，从actor buffer分配entry，rTLB获取virtual address；(4) Dataflow fabric执行constructor：从compressed bases/deltas数组恢复6B Pixel值，写入cache line；(5) 对于小于cache line的objects（6B），scheduler对line内所有objects并行执行constructor；(6) Core load完成，获得解压后的Pixel值在L1中。后续同一Pixel reuse直接L1 hit。

术语一般如何实现？如何使用？
实现需：(1) Morph抽象——封装data-triggered地址范围的注册/注销，含per-engine local state（Morph::views）；(2) Constructor/destructor action定义（类似täkō's onMiss/onEviction [66]，但操作object而非cache line）；(3) TLB bits（2 bits）指示Morph注册状态和位置（L2/LLC）；(4) Cache tag bit（1 bit）指示eviction时是否触发destructor；(5) vtable cache in engine scheduler映射地址范围到action函数指针；(6) rTLB物理→虚拟地址翻译。Data-triggered actions不适用所有场景——如decompression workload中task offload (OL)在L2解压但不retain在L1，反比baseline慢2.8×。Data-triggered因保留解压数据在L1而达成2.4× speedup。

涉及论文标题：
- 36-Leviathan_A_Unified_System_for_General-Purpose_Near-Data_Computing.pdf

## NDC Streaming (Near-Data Computing Streaming Paradigm)

术语解释
NDC Streaming是Leviathan NDC taxonomy中的第四种范式：一个near-data producer（运行在engine上的long-lived线程）持续生成数据流，推送给core上的consumer处理。它专为解耦访问-执行（decoupled access-execute）模式设计，producer可run ahead隐藏内存延迟。

术语是什么？
NDC Streaming的特征：(1) **解耦**——producer（数据生成/遍历）和consumer（数据处理）独立运行，producer在cache hierarchy中，consumer在core上；(2) **持续通信**——producer频繁push数据给consumer，consumer处理后acknowledge；(3) **long-lived**——producer在整个computation期间持续运行。典型应用包括HATS [51]（BDFS graph traversal producer + edge processing consumer）、Stream Dataflow [54]（affine stream patterns）、Near-Stream Computing [80]（近缓存流计算）。Leviathan的stream实现结合了long-lived workloads（producer线程）和data-triggered actions（phantom address space copy）两种范式。不同于传统affine-only streaming [54, 79-81]，Leviathan的stream支持任意数据访问模式和可变object大小。

从kernel调度角度拆解术语：
以Leviathan HATS BDFS streaming为例（Fig. 19）：
```
class LeviathanHATS extends Leviathan::Stream<Edge>:
    Stack bdfs = {Vertex* vec, uint top}

    // Producer: long-lived NDC action on engine
    void genStream():
        while True:
            if bdfs.top == 0:
                root = G.getNextRootVertex()
                if root == INVALID: return
                bdfs.vec[++bdfs.top] = root
                active[root++] = false
            dst = bdfs.vec[bdfs.top]
            while dst.nextNeigh < dst.inDegree:
                src = dst.neighbors[dst.nextNeigh++]
                push(Edge(src, dst))  // blocks when buffer full
                if bdfs.top < depth and !active[src]:
                    bdfs.vec[++bdfs.top] = src
                    active[src] = false
            --bdfs.top

// Consumer: regular thread on core
for range(G.numEdges):
    Future<Edge> future = stream.next()
    processEdge(future.wait())
```
调度流程：(1) Stream初始化：分配circular buffer在shared memory（bufferSize在构造时指定），分配phantom address space用于consumer load；(2) Producer (genStream) 作为long-lived action在tile's local engine上运行，执行BDFS遍历生成Edge，调用push写入circular buffer；(3) buffer满时push block，等待consumer pop释放空间；(4) Consumer core执行stream.next() → load phantom address → 触发data-triggered constructor（系统自动生成）copy下一个entry从circular buffer到phantom space → 返回Future<Edge>；(5) Consumer core执行pop指令递增head pointer → head跨cache line时发消息到engine → engine stream scheduler更新head, unblock producer if blocked；(6) 关键优势对比täkō pseudo-streaming：Leviathan的stream producer是continuously running action，而täkō每8 edges触发一次新action需重新初始化BDFS stack——Leviathan减少engine instructions per edge，且producer可run far ahead of consumer（不依赖consumer load触发）。

术语一般如何实现？如何使用？
实现需：(1) Stream<Edge> class（继承Morph<T>）封装producer (genStream/push)和consumer (next/pop)接口；(2) Circular buffer in shared memory；(3) Phantom address space——consumer通过此空间sequential load，hardware通过data-triggered constructor copy数据；(4) Engine stream scheduler管理head-tail指针和buffer状态；(5) Core ISA扩展：pop指令递增head并通知engine；(6) Deadlock prevention：OOO core speculative load past end-of-stream → NACK speculative loads, re-execute on commit。Stream buffer大小评估：performance plateaus at 64 entries（HATS workload），buffer在memory中不占额外硬件。NDC streaming适用于有producer-consumer解耦潜力的应用（graph traversal、data decompression pipeline），不适合需要core实时决策的紧密耦合场景。

涉及论文标题：
- 36-Leviathan_A_Unified_System_for_General-Purpose_Near-Data_Computing.pdf

## DYNAMIC Task Scheduling for Near-Data Computing

术语解释
DYNAMIC Task Scheduling是Leviathan task-offload paradigm中的自适应任务定位机制：invoke指令不静态指定执行位置，而是运行时逐级探测cache hierarchy（L1D→L2→LLC）以定位目标actor数据，并在数据所在位置执行action。这是Leviathan实现"where to compute"自动化的核心调度策略。

术语是什么？
DYNAMIC是invoke指令的默认location参数（此外还有LOCAL和REMOTE选项）。其工作原理：(1) Core执行invoke时首先probe本地L1D——若actor cached in L1D，直接在core本地执行action（最快，无offload overhead）；(2) L1D miss → core发送packet到local engine；(3) Engine task-offload scheduler检查local L2是否cached该actor——若cached，在L2 engine执行action；(4) L2 miss → engine forward packet到actor的LLC bank（通过mapping actor pointer到LLC bank index）；(5) LLC bank engine执行action。额外机制：(a) EXCLUSIVE flag——LLC engine检查是否有remote L2持有exclusive permissions，若是则forward到该remote L2 engine；(b) Data migration——DYNAMIC任务在需远程执行时以1/32概率改在本地执行，使高temporal locality的object逐步上移到私有cache（类似cache promotion），减少后续访问延迟。

从kernel调度角度拆解术语：
以hash table lookup中DYNAMIC scheduling的逐级定位过程（伪代码）：
```
// Core executes invoke
invoke node->Lookup(key)  // location = DYNAMIC (default)

// Step 1: Core probes L1D
if L1D.contains(node_address):
    execute Lookup on core locally  // fastest path
    return

// Step 2: Core sends packet to local engine
packet = {data_ptr: node, func_ptr: Lookup, flags: DYNAMIC, args: [key]}
send_to_local_engine(packet)

// Step 3: Local engine task-offload scheduler
if local_L2.contains(node_address):
    execute Lookup on local engine dataflow fabric
    return

// Step 4: Forward to actor's LLC bank
llc_bank = map_to_llc_bank(node_address)
forward_to(llc_bank, packet)

// Step 5: LLC engine checks EXCLUSIVE flag (if set)
if flags & EXCLUSIVE:
    remote_L2 = directory.get_exclusive_owner(node_address)
    if remote_L2:
        forward_to(remote_L2, packet)  // execute near exclusive owner

// Step 6: LLC engine executes action (or remote L2 if forwarded)
execute Lookup on llc_engine dataflow fabric

// Data migration (probabilistic, 1/32):
if should_migrate():  // 1/32 probability
    execute locally instead  // pull data up the hierarchy
```

术语一般如何实现？如何使用？
实现需：(1) Core侧——invoke指令的L1D probe逻辑（复用现有load pipeline）；(2) Engine task-offload scheduler——含L2 tag lookup（确定是否cached）和LLC bank mapping逻辑（hash actor address→bank index）；(3) NoC packet routing支持engine-to-engine forwarding；(4) Data migration概率计数器（简单LFSR实现1/32概率）。DYNAMIC scheduling确保action总是在数据最近的位置执行，同时通过data migration自动适应access pattern变化（热点数据逐步上移到L2甚至L1D）。对比static LOCAL/REMOTE：LOCAL可能迫使remote data迁到local engine，REMOTE总是LLC执行可能错过L2 locality。DYNAMIC在hash table workload上达成2.0× speedup vs 1.5×（without padding/mapping）和0.91×（128B without mapping）。

涉及论文标题：
- 36-Leviathan_A_Unified_System_for_General-Purpose_Near-Data_Computing.pdf

## Morph (Phantom Data Abstraction for NDC)

术语解释
Morph是Leviathan中封装data-triggered NDC范式地址范围管理的核心抽象。它管理phantom data——仅存在于cache中、不backed by off-chip memory的数据区域，允许软件注册constructor/destructor actions在cache insertion/eviction时自动触发。Morph是täkō [66] onMiss/onEviction concept的evolution，关键改进是操作object而非cache line。

术语是什么？
Morph<T>是一个模板类，封装：(1) 一个contiguous phantom address range（在cache中分配、不占DRAM）；(2) per-engine local state（Morph<T>::views[]——每个LLC bank的engine有自己的view副本，存engine-local metadata如bases指针、状态等）；(3) vtable映射——将地址范围映射到actor type T的constructor和destructor函数指针；(4) 注册/注销接口（register/unregister）。Morph解决的问题：传统data-triggered NDC（如täkō [66]）要求action操作cache lines——程序员必须手动处理line内object layout/alignment。Morph使action操作单个object：constructor接收object指针（而非line指针），destructor额外接收isDirty flag。Morph内部处理object→cache line的映射——小于line的objects并行构造/析构，大于line的objects一次性多行操作。Morph的address range可能跨多个LLC banks，因此每个bank的engine持有独立view。

从kernel调度角度拆解术语：
以decompression workload的Morph注册和执行流程：
```
// Define Morph subclass for decompression
class Decompressor extends Leviathan::Morph<Pixel>:
    uint16* bases[3]
    uint8* deltas[3]

    // Register at L2 cache level
    Decompressor* decomp = Morph<Pixel>::register(
        morphType = DECOMPRESSOR,
        cacheLevel = L2,
        numActors = 16384  // 16K Pixels
    )

    // Morph creates phantom address range for 16K Pixels in cache
    // Each LLC bank's engine gets a Morph::view with:
    //   - bases/deltas pointers (engine-local state)
    //   - TPadded* actors (base address of padded actors)
    //   - int size (number of actors in range)

    // On cache miss → constructor triggered:
    //   Cache controller checks TLB bits → Morph registered at L2
    //   Engine scheduler finds Morph vtable → Pixel constructor address
    //   Constructor receives Morph::view pointer → can access bases/deltas
    //   Constructor calls decomp->getOffset(this) → object index
    //   Constructor decompresses and initializes Pixel

    // On cache eviction → destructor triggered:
    //   Cache tag bit indicates destructor registered
    //   Destructor receives (Morph::view, isDirty)
    //   Can conditionally writeback (if dirty) or discard (if clean)
```

术语一般如何实现？如何使用？
实现需：(1) Morph<T>基类——管理phantom address range、per-engine views、vtable注册；(2) Allocator集成——Morph通过Leviathan::Allocator分配actors以保证intra-bank locality；(3) TLB bits + Cache tag bits——标记Morph注册和destructor触发；(4) Engine data-triggered scheduler——vtable cache映射address range→action函数指针，actor buffer管理pending constructions/destructions；(5) rTLB——物理→虚拟地址翻译给constructor/destructor。使用方式：程序员继承Morph<T>并定义T的constructor/destructor，调用register在指定cache level注册。注销时（unregister）触发flush指令清空Morph的address range。Morph对phantom data的生命周期完全管理——constructed on insertion, destructed on eviction，无DRAM backing。对比täkō's onMiss/onEviction：Leviathan的constructor/destructor操作object而非cache line，代码更简单且transparent to object size。

涉及论文标题：
- 36-Leviathan_A_Unified_System_for_General-Purpose_Near-Data_Computing.pdf

## Tree Attention (树形注意力机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tree Attention是将Transformer的standard attention mechanism从序列（sequence）结构推广到树（tree）结构的注意力计算机制。对于一棵token tree 𝒩，每个节点u ∈ 𝒩关联一个token t_u。Tree Attention的定义为：TreeAttention(u) = Attention(S_u)，其中S_u是从tree root到节点u的路径上所有token组成的序列（即u的祖先序列）。与standard sequence attention的区别在于：(1) 输入结构从线性序列变为树，多个序列共享公共前缀；(2) 因果性从"token只能attend到序列中前面的token"变为"token只能attend到tree中其祖先节点"；(3) 通过topology-aware causal mask将所有tree nodes的attention计算fused到单个kernel，消除共享前缀的冗余计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Tree Attention的kernel级计算过程（以SpecInfer的FasterTransformer-based实现为例）：
```
输入: Token tree N (N个节点，按tree topology linearize)
      共享KV-cache (已包含所有祖先节点的K, V)
输出: 每个节点u的attention output O[u]

Kernel配置: 每个thread block处理1个request的1个attention head
1. Query计算:
   X = 所有tree node的hidden states + prompt tokens, shape [batch_size+N, d_model]
   Q = X @ W_Q, shape [batch_size+N, d_head]
   
2. Key/Value从共享KV-cache读取:
   K = KV_cache.keys[0:N], shape [N, d_head]  // 按tree topology排列
   V = KV_cache.values[0:N]
   
3. Attention scores:
   A = Q @ K^T / sqrt(d_head), shape [N, N]
   
4. Topology-aware causal mask:
   M[j][k] = 0     if token_k是token_j在tree中的祖先节点
           = -∞     otherwise (包括token_k不在j的祖先路径上)
   A_masked = A + M

5. Attention output:
   O = softmax(A_masked) @ V  // -∞位置经softmax后为0
```

Tree Attention kernel的关键优化：
- **单kernel fusion**：所有N个tree node的attention在单次kernel launch中完成（vs sequence-based需要O(#sequences)次launch）
- **共享prefix消除冗余**：共享公共前缀的token sequences无需重复计算其attention
- **Shared KV-cache**：所有tree node共享同一KV-cache空间，DFS traversal保证正确性
- **Topology-aware mask**：将tree拓扑结构编码为attention mask，在标准attention kernel中实现tree attention语义

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tree Attention的实现依赖两个关键组件：
1. **Topology-aware causal mask矩阵**：形状[N, N]，值域{0, -∞}。Mask[i][j]=0当且仅当token_j是token_i在tree中的祖先（即j的路径是i路径的前缀）。这个mask在GPU kernel中与attention scores相加后经softmax处理（-∞→0），物理上实现了tree causality。与sequence causal mask（下三角矩阵）的区别是：tree mask是稀疏矩阵，只保留祖先关系对应的位置。
2. **Depth-first search KV-cache management**：使用DFS顺序遍历token tree，按访问顺序写入共享KV-cache。当DFS进入新分支时，祖先KV已在cache中；DFS回溯时被覆盖。这样保证：(a) 任何时候cache中存储的是当前节点到root路径上所有token的KV；(b) 只需要一份KV-cache空间（O(tree_depth)而非O(#sequences × sequence_length)）。

Tree Attention vs Standard Sequence Attention的性能：
- GPU kernel launch次数：1（tree）vs O(#sequences)（sequence-based）
- 冗余attention计算：消除共享prefix的重复计算
- SpecInfer实验：tree-based parallel decoding vs sequence-based decoding在large batch（BS=16）下实现高达1.8×加速（LLaMA-7B, A10 GPU）

涉及论文标题：
- 38-SpecInfer- Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.pdf

## Posit Approximate Softmax

术语解释
Posit Approximate Softmax是利用Posit数的bitwise运算特性（而非精确浮点算术）来近似Transformer中softmax函数的计算kernel。核心思路：用bitwise NOT最高位+右移2位近似sigmoid（仅对es=0有效）、用bitwise XOR with negated signmask近似reciprocal、通过e^x = 1/sigmoid(-x) - 1组合构造指数函数，从而消除面积大功耗高的浮点除法器和精确指数单元。

术语是什么？
Softmax公式：σ(z)_i = e^(z_i) / Σ e^(z_k)。在传统实现中，这需要：(1) 精确浮点指数e^x（通常通过CORDIC或查找表多项式和Taylor级数实现）；(2) 精确浮点除法（乘法逆或Goldschmidt/SRT除法器）。Posit Approximate Softmax利用Posit (es=0)的两个bitwise近似函数：(a) Approx Sigmoid: S(x) ≈ bitwise NOT of MSB + right shift 2 (仅对es=0的Posit有效); (b) Approx Reciprocal: 1/x ≈ x XOR negated_signmask (对所有es有Posit有效, piece-wise linear)。组合构建exp：e^x = 1/S(-x) - 1。对于Posit(8,1)（es=1），需先转换到es=0或用移位模拟sigmoid近似。

从kernel调度角度拆解：
以Transformer attention中softmax kernel的伪代码为例（Posit8版本，NVIDIA GPU或加速器vector unit上执行）：

```
// === Posit8 Approximate Softmax Kernel ===
// Input:  attention_scores [seq_len] (Posit8 decoded → E5M4)
// Output: softmax_probs [seq_len] (Posit8 format)

// Stage 1: Numerical stability — find max
max_val = attention_scores[0]
for i in 1..seq_len:
    max_val = max(max_val, attention_scores[i])

// Stage 2: Compute exp via approximate sigmoid + reciprocal
sum_exp = 0
for i in 0..seq_len:  // 可向量化 — N-lane SIMD
    x = attention_scores[i] - max_val  // x ≤ 0, numerically stable
    
    // Step 2a: Approximate sigmoid S(-x)
    // For Posit(8,1): convert to es=0 equivalent, apply bitwise NOT + shift
    neg_x = -x  // x ≤ 0 → -x ≥ 0
    s_val = approx_sigmoid_posit(neg_x)
    // Hardware: bitwise NOT of MSB, right shift by 2 positions
    
    // Step 2b: Construct exponential: e^x = 1/s - 1
    // 1/s via approximate reciprocal (bitwise XOR)
    recip_s = approx_reciprocal_posit(s_val)
    // Hardware: s_val XOR 0b10000000 (negated signmask for 8-bit)
    
    exp_val = recip_s - 1.0  // subtraction in accumulator
    
    // Step 2c: Threshold optimization (for attention masking)
    if x < -3.0:  // θ = -3 (optimal from Table 3)
        exp_val = 0  // 截断为0, 实现attention mask, 可通过bitwise mask实现
    // Step 2d: Shift optimization (Table 3: ε = -1.188)
    exp_val = exp_val - (-1.188)  // 向下平移curve以更好匹配精确指数
    
    exp_vals[i] = exp_val
    sum_exp += exp_val  // accumulate in BF16

// Stage 3: Softmax output via approximate reciprocal (division)
recip_sum = approx_reciprocal_posit(sum_exp)  // 1/Σexp, bitwise XOR

for i in 0..seq_len:  // 可向量化
    softmax_probs[i] = exp_vals[i] * recip_sum  // 用乘法替代除法
    // Encode back to Posit8
    softmax_probs[i] = encode_posit8(softmax_probs[i])

// === Backward pass (for training) ===
// Custom gradient for approximate softmax
// f': piecewise linear derivative of approximate reciprocal
// f' = -2^(-floor(log2(Σexp)) · 2 - 1)
L = floor(log2(sum_exp))  // 可用leading one count实现
f_prime = -2^(-L * 2 - 1)  // piecewise linear, power-of-2 steps

for i in 0..seq_len:
    for j in 0..seq_len:
        if i == j:
            grad_softmax[i][j] = softmax_probs[i] + exp_vals[i] * f_prime * exp_vals[i]
        else:
            grad_softmax[i][j] = exp_vals[i] * f_prime * exp_vals[j]
// 与传统softmax backward相同结构 → 可复用相同向量化硬件
```

核心硬件原语：
- **approx_sigmoid_posit**: `output = input XOR (1 << 7) >> 2`（bitwise NOT of MSB + shift right 2）
- **approx_reciprocal_posit**: `output = input XOR 0b10000000`（所有bits XOR with negated signmask, 等价于NOT all bits except sign）
- **Threshold mask**: `mask = (x < theta) ? 0 : 1`，通过比较器+AND gate实现

术语一般如何实现？如何使用？
本论文在Vector Unit的Stage 2 (exp)和Stage 3 (reduce+divide)中使用这些近似：Stage 2的指数单元用bitwise logic替换浮点exp单元（面积↓62%, 功耗↓44% at 200MHz），Stage 3的除法器用bitwise XOR reciprocal替换浮点除法器（面积↓85%, 功耗↓75% at 200MHz）。在GPU上模拟时，通过PyTorch自定义autograd Function实现：`apply()`调用bitwise操作，`backward()`实现revised gradient formula。近似softmax的accuracy影响：MobileBERT F1从89.9降至88.6（thresholding + shifting优化后，仅↓0.3% with quantization, Table 4）。推理时仅需threshold+shift优化即可保持<1% accuracy loss；训练时需额外custom backward pass。

涉及论文标题：
- 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

## Initiation Interval (II, 发射间隔)

术语是什么？
Initiation Interval（II，发射间隔/启动间隔）是计算机体系结构中的经典概念（Hennessy & Patterson），指同一类型功能单元（FU）连续两次发射操作之间所需的时钟周期数。在GPU上下文中，II = warp_size / functional_unit_lanes。例如A100的CUDA core有64个FP32 lane，warp_size=32，FP32指令的II = 32/64 = 0.5，即每cycle可发射2个warp的FP32指令（4 cycles/warp inst对应II=4）。II的本质是FU吞吐量的倒数——值越小，吞吐越高，资源争用越少。

从kernel调度角度拆解术语：
在GPU解析模型中，II用于计算compute resource contention导致的stall cycles。GCoM的Issue cycle计算公式：
```
C_Issue_{k,m}(x) = (I_m · II_m · x) / (numActSCs(x) · IssueRate)
```
其中II_m计算为 warp_size/FU_lanes。但这个公式对Tensor Core不适用，因为Tensor Core使用per-warp register scheme（32线程协作而非竞争），实际II远小于公式计算结果。AMALI的改进——Throughput-based II for Tensor Cores：
```
II_TC = FMA_count / TP_dt
```
例如：
- A100 Tensor Core TP_fp16 = 256 FMAs/clk
- HMMA.16816 (FMA_count=2048): II_TC = 2048/256 = 8 cycles
- HMMA.1688 (FMA_count=1024): II_TC = 1024/256 = 4 cycles
- 对比GCoM的II = 32/4 = 8（假设4个tensor core/sub-core），恰好与HMMA.16816一致但无法区分HMMA.1688

这个修正使AMALI的GEMM kernel MAPE从183.95%降至17.84%。

术语一般如何实现？如何使用？
在实际GPU中，II不是一个可配置参数而是微架构的设计结果。在解析模型中，II的确定方式：(1) 对于CUDA core指令：通过micro-benchmark测量各FU的实际吞吐（如pointer chase或连续发射相同指令）；(2) 对于Tensor Core指令：通过micro-benchmark测量不同modifier组合的HMMA指令执行时间（或用cuAssembler修改SASS验证）；(3) 使用时，II被代入issue cycle计算公式预测resource contention造成的stall。设计空间探索中可改变FU count或tensor core throughput来评估不同架构配置。

涉及论文标题：
- 3-AMALI- An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf

## HMMA Instruction (Half Matrix Multiply Add)

术语是什么？
HMMA（Half Matrix Multiply Add）是NVIDIA GPU SASS指令集中用于编程Tensor Core的指令族，执行半精度矩阵乘加：D = A × B + C。HMMA指令名中的"H"表示half-precision输入。完整格式为HMMA.shape.dtype，其中shape编码M×N×K维度（如16816=16×8×16, 1688=16×8×8），dtype指定累加器类型（如F32=FP32 accumulate, F16=FP16 accumulate, BF16=BF16 accumulate）。不同shape的FMA数量不同：16816=2048 FMAs，1688=1024 FMAs。

从kernel调度角度拆解术语：
HMMA指令的kernel级执行流程：
```
// 伪代码：warp在Tensor Core上执行HMMA.16816.F32
// 实际SASS: HMMA.16816.F32 R0, R108, R140, R0

1. Warp scheduler选择ready warp，发出HMMA指令
2. 32个线程协作：各自从共享寄存器读取A/B矩阵片段
   - R108→R111 (4 registers) = A矩阵16×16片段（per-thread view）
   - R140→R141 (2 registers) = B矩阵8×16片段（per-thread view）
   - R0→R1 (2 registers) = C累加器 16×8片段
3. Tensor Core阵列在8 cycles内完成2048个FMAs
   (A100 256 FMAs/clk/Tensor Core)
4. 结果写回R0→R1

// AMALI建模：II_TC = 2048/256 = 8 cycles
```

HMMA与wmma/mma的关系：
- PTX wmma.mma.m16n16k16（高层API）编译为2条HMMA.16816
- PTX mma.sync.aligned.m16n8k16（低层API）编译为1条HMMA.16816
- 不同shape对应不同tiling策略和register用量

AMALI的关键发现：GCoM使用II=32/4=8统一建模所有HMMA，无法区分16816（CPI=8）和1688（CPI=4），而AMALI的throughput-based model精确区分两者。

术语一般如何实现？如何使用？
HMMA指令不能直接由CUDA C++产生，需要通过：(1) wmma intrinsics（如nvcuda::wmma::mma_sync）——编译器自动映射到HMMA；(2) PTX inline assembly（asm volatile("mma.sync.aligned.m16n8k16...")）；(3) 第三方库如CUTLASS——通过模板元编程生成最优HMMA序列。在性能分析和建模中，cuobjdump -sass反汇编cubin可查看实际生成的HMMA序列。cuAssembler可修改SASS中的HMMA modifier用于micro-benchmark。

涉及论文标题：
- 3-AMALI- An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf

## Kernel Launch Latency (KLL, Kernel启动延迟)

术语是什么？
Kernel Launch Latency（KLL，Kernel启动延迟）是指从host端调用CUDA kernel launch API到GPU上kernel实际开始执行第一条指令之间的延迟。这个延迟包括：host→device的launch command传输、GPU command processor处理launch请求、为所有thread block分配SM资源、加载kernel指令到instruction cache、初始化constant memory/constant cache等。KLL建模的主要挑战是：延迟不是常数，而是随grid size（thread block数量）和block size（每block线程数）变化。

从kernel调度角度拆解术语：
AMALI的KLL模型基于micro-benchmark实验建立：
```
KLL = s · GS + k
其中 GS = grid size, s = slope（随block size变化）
s = α · (BS)² + β · BS + γ
A100实测系数: α=0.0036, β=0.0366, γ=1.1891
```

KLL的物理含义：(1) 线性项s·GS捕获为每个thread block分配SM资源的开销，grid越大总launch时间越长；(2) s对BS呈二阶关系——block size越大，每个block的launch overhead越高（更多的register allocation、shared memory初始化等）；(3) 固定项k捕获command传输和GPU command processor处理的固定开销。

KLL在AMALI中用于建模两类NCU stall：imc_miss（constant cache miss）和no_instructions（instruction cache miss）——这两类stall本质上发生在kernel启动阶段（指令/常量尚未加载到cache），而非kernel执行阶段。KLL将这两个之前被所有GPU解析模型忽略的stall分量直接建模。

术语一般如何实现？如何使用？
KLL通过micro-benchmark测量：(1) 编写一个空kernel（无实际计算指令）；(2) 以不同(grid_size, block_size)组合反复launch该kernel；(3) 使用CUDA event（cudaEventRecord）精确测量每种组合的wall-clock launch时间；(4) 对固定BS的数据点做线性拟合得到s，再对s~BS做二次拟合得到α/β/γ。KLL可跨GPU架构变化（不同架构的command processor和资源分配机制不同），因此每个目标GPU需单独标定。

涉及论文标题：
- 3-AMALI- An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf

## Math Pipe Throttle

术语是什么？
Math Pipe Throttle是NVIDIA GPU中一种特定的stall类型，在NCU的Scheduler Statistics中定义。当warp准备执行一条需要特定执行流水线（execution pipeline）的指令，但该流水线正忙（被其他warp占用）时，warp被stall，NCU将其归类为math_pipe_throttle。这本质上是一种compute structural hazard（计算结构冒险）——请求的计算资源（CUDA core、Tensor Core、SFU等）当前不可用。在GPU解析模型中，math_pipe_throttle对应compute resource contention造成的stall（S_ComStruct）。

从kernel调度角度拆解术语：
math_pipe_throttle在不同kernel类型中的来源不同：
- **GEMM kernel（使用Tensor Core）**：math_pipe_throttle主要来自Tensor Core的contention。传统解析模型用II = warp_size/FU_lanes高估了它（如图2a），AMALI用II_TC = FMA_count/TP_dt修正。
- **Vector kernel（使用CUDA Core）**：math_pipe_throttle来自FP32/INT单元的contention，传统II公式相对准确。
- **设计空间探索**：AMALI验证了Tensor Core throughput加倍（256→512 FMAs/clk）时math_pipe_throttle精确减半，因此可以通过改变TP_dt参数预测不同Tensor Core配置下的math_pipe_throttle变化。

AMALI中建模：
```
S_math_pipe = S_ComStruct  (来自GCoM的compute resource contention模型)
其中 issue cycle C_Issue_{k,TC}(x) 使用 II_TC = FMA_count/TP_dt 替代 II = warp_size/FU_lanes
```

术语一般如何实现？如何使用？
在NCU中，math_pipe_throttle是硬件性能计数器直接报告的stall原因。在解析模型中，通过计算各FU的issue cycle和contention来预测。减少math_pipe_throttle的方法：(1) 提高Tensor Core occupancy（更多active warp可隐藏延迟）；(2) 使用更大的tile size减少HMMA指令密度；(3) 混合使用CUDA core和Tensor core避免单类型FU饱和。

涉及论文标题：
- 3-AMALI- An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf

## Loop Perforation for Sparse Computation（面向稀疏计算的循环打孔）

术语是什么？
Loop Perforation（循环打孔）是近似计算领域的技术，通过有选择地跳过循环迭代来提升计算效率，以精度为代价换取性能。Fractal将这一概念应用于DNN稀疏计算：将结构化稀疏模式重新解释为对稠密GEMM的多级tiling循环进行"穿孔"——在特定tiling层级跳过（perforate）部分迭代，跳过的迭代对应的计算被完全省略。穿孔循环记作`loop_length^nnz`，穿孔外层循环跳过整列/整块（含所有内层循环），穿孔内层循环跳过单个元素。转换后的循环使用index vector间接寻址只执行保留的非零迭代。

从kernel调度角度拆解术语：
以1024×1024 GEMM在A100 GPU上的kernel执行（3级K轴穿孔：K0 32选28, K1 8选5, K2 dense）为例：
```cuda
// Sparse GEMM kernel伪代码（perforated GEMM on GPU）
__global__ void sparse_gemm(
    float* C, float* A_vals, int* K0_idx, int* K1_idx,
    float* B, int M, int N, int K) {
  // thread block沿outer dense axes (I0,I2)和J并行
  int i0 = blockIdx.x / 32;  // I0: outer M-tile
  int i2 = threadIdx.x;       // I2: inner M-element
  int j  = blockIdx.y * blockDim.y + threadIdx.y; // J
  float acc = 0;
  for (int k0_iter = 0; k0_iter < 28; k0_iter++) {     // K0 sparse
    int k0 = K0_idx[i0 * 28 + k0_iter];
    for (int k1_iter = 0; k1_iter < 5; k1_iter++) {    // K1 sparse
      int k1 = K1_idx[k0 * 5 + k1_iter];
      // Inner dense compute (K2, I2, J) using Tensor Core
      for (int k2 = 0; k2 < 4; k2++) {
        int k = k0 * 32 + k1 * 4 + k2;
        acc += A_vals[compute_a_offset(i0,i1,k0_iter,k1_iter,k2,i2)]
             * B[k * N + j];
      }
    }
  }
  C[(block_row + i2) * N + j] = acc;
}
```
GPU执行特点：K0_indices决定coarse block选择（控制block级workload），K1_indices决定block内fine tile选择（控制warp级计算量），内层dense axes K2/I2/J在warp内正常并行。

术语一般如何实现？如何使用？
Fractal中循环打孔代码由PatternIR→SparseTIR→TVM自动生成：(1) SparseTIR的`with sp_iter`将穿孔sparse axes和dense axes组合生成正确嵌套循环；(2) 每个sparse loop的index vector在Fractal-ELL中连续存放，kernel启动时加载到shared memory；(3) GPU thread block沿外层dense/sparse axes并行分配，warp/thread在内层dense axes上并行；(4) 关键约束：spatial axes后继reduction axes的穿孔被禁止（防止并行线程写冲突），必要时转为atomic操作；(5) Condense原语在kernel执行前预聚集非连续数据，使穿孔后循环内实现稠密连续访问。

涉及论文标题：
- 40-Fractal- Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning.pdf

## Per-Layer Asynchronous KV-Cache Transfer with MSCCL++

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Layer Asynchronous KV-Cache Transfer（逐层异步KV-cache传输）是Splitwise中优化跨机KV-cache传输延迟的kernel/communication-level技术。核心思想：在prompt machine上每完成一个transformer layer的forward pass计算并生成该层的KV-cache后，立即通过MSCCL++的异步RDMA put将该层KV-cache推送到token machine，同时prompt machine继续执行下一layer的计算。这与naive的serialized transfer（所有层计算完成后一次性传输）相比，将传输延迟与prompt computation重叠，使可见的传输overhead从线性增长（随prompt size）降为常数（约为单层传输的non-overlapped tail部分，A100上~8ms, H100上~5ms）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Per-Layer Asynchronous KV-Cache Transfer的伪代码执行流程（以BLOOM-176B on 8×H100, 1500-token prompt为例）：

```
// Prompt Machine (GPU i, tensor parallelism rank i/8)
// B = batch_size_total_tokens (prompt tokens across all requests in batch)
// L = n_layers = 70
// H = n_heads_per_TP_rank, d = d_head

for layer_id in 0..L-1:
    // Step 1: Forward pass for this layer
    // Self-attention: compute Q, K, V from hidden_states
    Q = Linear_Q(hidden_states)   // shape: [B, H, d]
    K = Linear_K(hidden_states)   // shape: [B, H, d]
    V = Linear_V(hidden_states)   // shape: [B, H, d]

    // Attention computation
    attn_output = Attention(Q, K, V)  // uses FlashAttention
    hidden_states = Linear_O(attn_output)

    // FFN: two linear layers with activation
    hidden_states = Linear_FFN1(hidden_states)
    hidden_states = GELU(hidden_states)
    hidden_states = Linear_FFN2(hidden_states)

    // Step 2: K and V for this layer are now final → trigger async transfer
    // K, V are in GPU HBM
    if prompt_size >= 512:  // use per-layer transfer for large prompts
        for each request in batch:
            target_token_machine = request.token_machine
            kv_blocks = get_kv_blocks_for_layer(layer_id, request)
            // MSCCL++ zero-copy one-sided put
            // Non-blocking: GPU RDMA engine handles transfer
            mscclpp_put_async(
                src=K.data_ptr(), size=K.nbytes,
                dst=target_token_machine, dst_offset=layer_kv_offset,
                semaphore=request.semaphore_id
            )
            mscclpp_put_async(
                src=V.data_ptr(), size=V.nbytes,
                dst=target_token_machine, dst_offset=layer_kv_offset + K.nbytes,
                semaphore=request.semaphore_id
            )
    // Transfer proceeds in background while next layer computes

// After all layers complete, signal completion via semaphore
for each distinct target_token_machine:
    mscclpp_signal_semaphore(target_token_machine, semaphore_ids)

// --- Token Machine (GPU j) ---
// Wait for all layers to arrive
mscclpp_wait_semaphore(semaphore_ids)
// KV-cache is now fully in local GPU HBM → begin token generation
```

关键kernel调度要点：
1. **MSCCL++ put是GPU-driven的**：GPU通过RDMA直接推送数据到远端GPU HBM，不经过CPU，不打断GPU compute pipeline。
2. **Per-layer overlap窗口**：每层传输时间 ~ KV_layer_size / InfiniBand_bw。BLOOM-176B每层KV ~ batch_tokens × n_heads × d_head × 2(K+V) × 2(FP16) bytes。对batch=1500 tokens、112 heads、d_head=128：每层约6.7MB。H100 InfiniBand 400Gbps (~50GB/s) → 每层传输 ~0.13ms，而该层计算 ~10-20ms → 传输可完全隐藏。
3. **Semaphore同步**：使用同一InfiniBand连接实现semaphore signal/wait，避免额外TCP/IP或gRPC同步开销。
4. **Block连续性优化**：vLLM中KV-cache按block（16 tokens）分配，物理上可能不连续。MSCCL++合并连续block为单次put减少传输次数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Per-Layer Asynchronous KV-Cache Transfer的实现要素：
1. **MSCCL++**：Microsoft开源GPU驱动通信库（https://github.com/microsoft/mscclpp），提供one-sided put/get/semaphore等原语，专为GPU集群高效通信设计。Splitwise使用put原语实现prompt→token的单向数据推送。
2. **vLLM集成**：修改vLLM的model execution loop（在每层forward后插入MSCCL++ put调用），保持PagedAttention block table兼容。PR: https://github.com/vllm-project/vllm/pull/2809。
3. **模式自动选择**：Splitwise根据prompt大小自动决定传输模式——serialized transfer（总KV-cache小时serialized更简单，无fine-grained同步overhead）vs per-layer transfer（总KV-cache大时overlap收益显著）。H100上threshold为512 tokens。
4. **Per-layer non-overlapped tail**：即使per-layer overlap，仍有最后几层的传输无法完全overlap（无更多计算可并行）。A100上non-overlapped时间约8ms，H100上约5ms（因为400Gbps vs 200Gbps InfiniBand）。
5. **实际效果**：Per-layer transfer将second-token latency overhead从serialized的64%降至16.5%，E2E overhead仅0.8%。

涉及论文标题：
- 41-Splitwise_Efficient_Generative_LLM_Inference_Using_Phase_Splitting.pdf

## Gustavson Dataflow (for Matrix Multiplication)

术语解释
矩阵乘法的三种基本dataflow之一，coiteration loop位于中间层（k loop middle），逐行产生输出。特点是平衡intersection和reduction成本，适合highly sparse (HS)矩阵乘法。

术语是什么？
Gustavson dataflow的loop nest为：
```
for m in [0, M):          // 逐输出行
    for k in [0, K):      // coiteration in middle loop
        for n in [0, N):  // inner
            C[m,n] += A[m,k] * B[k,n]
```
与Inner-Product（coiteration innermost, element-level intersection）和Outer-Product（coiteration outermost, matrix-level intersection）对比，Gustavson执行row-level intersection：每遍历A的一个非零元素，与B的一整行做intersection。Intersection次数 = nnz(A)（而非IP的M×N次或OP的K次），且每个intersection只要B行非空即成功。Reduction在partial output row级别进行，比OP的matrix-level reduction简单得多。

从kernel调度角度拆解：
Trapezoid中Gustavson-based TrGT dataflow（HS×HS）：
```
for n1 in [0, N1):                    // 输出tile
    for m2 in [0, M2):                // B tile on-chip
        B_tile = load_B_tile(n1, m2)
        for m1 in [0, M1):            // spatial Y: PE row
            C_tile = zero_init()
            for m0 in [0, M0):        // spatial Y: PE subrow (4×)
                B_tmp = zeros([K, N0])
                for k in [0, K):      // leader-follower over A nonzeros
                    A_val = A[m2,m1,m0,k]
                    for n0 in [0, N0):
                        B_tmp[k,n0] = B[n1,k,n0]  // gather B row
                // merger: transpose B_tmp [K,N0] -> [N0,K]
                for n0 in [0, N0):    // reduction
                    for k in [0, K):
                        C[n1,m2,m1,m0,n0] += A[m2,m1,m0,k] * B_tmp_t[n1,n0,k]
```

关键特征：(1) B matrix-level reuse：B tile on-chip，多行A复用同一B tile；(2) A element-driven：仅遍历A的非零元素，天然skip A zeros；(3) Leader-follower pattern：A[k]为leader，B[k]为follower被gather；(4) Cache critical：B行访问不规则（A的k坐标决定访问哪行B），需要高gather bandwidth的cache。

术语一般如何实现？如何使用？
1. **Memory traffic优势**：Gustavson仅需读A一次（element-by-element），读B次数由A的非零模式决定（nearby rows of A → repeated B rows → cache hit），写C row by row。对比OP需读A column一次、B row一次但写K个partial result matrices（high write traffic）。对HS输入，Gustavson是traffic最优的dataflow。
2. **Intersection-Reduction trade-off**：IP element-level intersection (M×N次, 稀疏时99%+ ineffectual) vs Gustavson row-level intersection (nnz(A)次, 仅B行空时ineffectual)。Gustavson在HS时远优于IP；但MS时IP的cheap intersection（bitvector AND）仍可接受，且IP的element-level reduction（仅accumulator）比Gustavson的row-level reduction（需merge tree）更简单。
3. **Accelerator实现**：Gamma [79]用fibercache捕获B行reuse；Flexagon [50]用MRN merge mode做归并；Trapezoid用multi-level memory hierarchy（global cache + local buffer）提供4 gather reads/cycle + MRN merge mode。所有HS×HS accelerator最终都收敛到Gustavson-based design。
4. **变体**：TrGT（temporal B row gather, 1 MAC/cycle/subrow, minimize traffic） vs TrGS（spatial B row stream, 16 MACs/cycle/PE row, higher throughput for HS×MS/HS×D）。

涉及论文标题：
- 43-Trapezoid- A Versatile Accelerator for Dense and Sparse Matrix Multiplications.pdf

## Inner-Product (IP) Dataflow for Matrix Multiplication

术语解释
矩阵乘法的三种基本dataflow之一，coiteration loop位于最内层（k loop innermost），逐元素产生输出。特点是high compute intensity and reuse，适合dense和mildly sparse矩阵乘法。

术语是什么？
IP dataflow的loop nest为：
```
for m in [0, M):          // 行遍历
    for n in [0, N):      // 列遍历
        for k in [0, K):  // coiteration innermost
            C[m,n] += A[m,k] * B[k,n]
```
每个(m,n)输出元素通过reducing A的一行与B的一列（element-level intersection + reduction）。在dense情况下，每个A元素被reuse N次（所有B列），每个B元素被reuse M次（所有A行），达到极高compute intensity。但在稀疏情况下，每个intersection（M×N次）需要检查k坐标匹配，稀疏度越高匹配概率越低。

从kernel调度角度拆解：
Trapezoid中TrIP dataflow（MS×MS, 创新点）：
```
for n1 in [0, N1):                            // N tile
    for m1 in [0, M1):                        // M tile
        C_tile on-chip
        for k1 in [0, K1):                    // K tile
            for [n_l, n_h) in dynamic_split(N0):  // 动态选B列数(1-4)
                for [m_l, m_h) in static_split(M0): // spatial Y: PE row
                    for m0 in [m_l, m_h):          // spatial X: local buf bank
                        for n0 in [n_l, n_h):      // spatial X: local buf word
                            for k0 in [0, K0):     // spatial X: MRN leaf
                                // MFIU intersection
                                C[n1,m1,m0,n0] += A[m1,k1,m0,k0] * B[n1,k1,n0,k0]
```

在2D spatial array上的映射（以TPU为例）：M mapped to spatial Y (PE rows)，K mapped to spatial X (PE columns)，A stationary in PEs，B columns vertical stream。每个PE row计算一个dot product (1 A row × 1 B column)。Quadratic compute (P² MACs/cycle) + linear I/O (P elements/cycle)。

术语一般如何实现？如何使用？
1. **Dense加速器标准dataflow**：TPU、Tensor Cores均使用IP-based 2D spatial array。D×D时reaches compute roofline with minimal control overhead（仅systolic forwarding）。
2. **稀疏adaptation**：SIGMA [60] adds Benes network for A-side sparsity routing；DSTC [70] uses OP-based alternative for dual-side sparsity但牺牲quadratic compute特性；Trapezoid's TrIP adds MFIU for dual-side sparsity while maintaining IP's advantages。
3. **关键trade-off**：IP的intersection cost随sparsity增长而exponentially增长。例如20% density→1/25 intersections effectual (on average)，但IP's bitvector AND cost远低于一次MAC，在MS range仍acceptable。但HS时（<1% density），>99% intersections ineffectual → IP完全不可行 → 需切换到Gustavson。
4. **Tiling**：实际实现中IP dataflow被多层tiled：M→M1×M0, N→N1×N0, K→K1×K0。Trapezoid使用coordinate-space tiling on K和occupancy-based tiling on M/N。

涉及论文标题：
- 43-Trapezoid- A Versatile Accelerator for Dense and Sparse Matrix Multiplications.pdf

## On-the-fly Matrix Regrouping（在线矩阵重组）

术语解释
MECLA PE array的一种kernel数据映射技术——在数据加载到PE时动态重排SSMP划分后散布的权重行/列，将具有PSum重用关系的数据集中到同一或相邻PE cluster，避免跨cluster通信并最大化计算复用效率。

术语是什么？
SSMP将权重矩阵划分为多个region，每个SS子矩阵以间隔[x·nx, y·ny]散布在完整权重空间中。若直接按原始顺序将权重映射到PE array，相关的SS及其对应的DS会被分配到不同PE cluster甚至不同数据加载批次中，导致：(1) PSume无法在cluster内复用——同一SS的PSum需跨cluster传输（带宽瓶颈）；(2) scaling scalar需多次从buffer读取——同一SS对应的16个DS需各自独立读取scaling factor。On-the-fly regrouping在数据从SS/DS buffer加载到PE weight buffer/scale buffer时，动态重排数据顺序——将同一SS对应的所有DS行（间隔nx）集中、同一SS对应的所有DS列（间隔ny）集中，使计算发生在同一PE cluster内。

从kernel调度角度拆解：
On-the-fly Matrix Regrouping的kernel数据映射流程（outer-product模式，nx=4, ny=4）：
```
输入：
  原始SSMP weight布局（逻辑视角）:
    Region(rx,ry): SS(rx,ry) [8×8]
                    + 15个DS(rx,ry,p,q) [8×8], (p,q)∈[0..3]²
    在完整矩阵中的位置:
      DS(rx,ry,p,q) 位于行 [rx·32 + p·8 : rx·32 + (p+1)·8]
                         列 [ry·32 + q·8 : ry·32 + (q+1)·8]

Regrouping操作（数据重排伪代码）：
  // Stage 1: 按output channel regroup（outer-product优先）
  for each region (rx, ry):
      for p in 0..nx-1:  // p=output channel group index
          // 收集同一SS对应的第p组DS行
          regrouped_weight[p] = [
              SS(rx,ry, :, :),          // SS本身（DS(0,0)）
              DS(rx,ry, p, 1),          // DS(p,1)的weight（=SS × S[p,1]）
              DS(rx,ry, p, 2),          // DS(p,2)的weight
              DS(rx,ry, p, 3)           // DS(p,3)的weight
          ]
          // 这4个[8,8]矩阵共享相同的8 output channels矩阵结构
          // → 产生4个仅差scalar倍的PSum → 可被同一SA row处理

  // Stage 2: PE cluster分配
  clusters_per_region = ceil(nx * ny / PE_cluster_capacity)
  // 将regrouped数据分配到8个PE clusters
  for each PE_cluster in 0..7:
      PE_weight_buffer[cluster] ← regrouped_weight[assigned_slices]
      scale_buffer[cluster] ← S[assigned_slices]  // 对应scaling scalar

  // Stage 3: 可补偿的非SSMP partition处理
  for residual_matrix in non_SSMP_partitions:
      // 这些少数矩阵区域无法用SSMP分解
      // 按标准dense MatMul映射，无regrouping
      PE_weight_buffer[cluster] ← residual_matrix

硬件实现：
  // RISC-V core在数据加载阶段执行regrouping
  for addr in weight_load_sequence:
      base_region = compute_region(addr)
      (p, q)       = compute_ds_offset(addr)
      new_addr     = regroup_addr(base_region, p, q, mode=OUTER_PRODUCT)
      load_to_PE(new_addr, weight_data[addr])
```

术语一般如何实现？如何使用？
1. **地址重映射**：Regrouping本质上是权重地址的重映射——RISC-V core根据SSMP配置(x,y,nx,ny)和当前mode（outer/inner product）计算每个权重元素的target PE地址。不需要额外的数据搬移硬件——在weight加载阶段直接routing到正确PE buffer位置。
2. **Mode-dependent regrouping**：Outer-product模式按output channel regroup（优先最大化PSum行间reuse）；Inner-product模式按input channel regroup（优先最大化PSum列间reuse）。RISC-V在解码SSMP配置时自动选择。
3. **Regrouping粒度**：以[x, y]（SS尺寸）为基本grouping单位。一个region内的nx·ny个[x,y] block被重排为连续访问序列，消除原始散布引入的memory访问不连续开销。
4. **Buffer占用**：Regrouping后的weight layout在PE weight buffer中连续存储，一个PE cluster处理一个或多个完整regrouped group。512KB SS buffer可容纳LLaMA-7B单层的regrouped weight（压缩后~100KB/层）。
5. **边界处理**：当矩阵维度不是x·nx的整数倍时，最后不完整的region用padding或标准dense MatMul处理。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## Outer-product Partial Sum Reuse（外积部分和重用）

术语解释
MECLA PE array在outer-product模式下利用SSMP的核心性质——SS和DS对相同input channels的partial sum仅差一个scalar倍——通过先计算SS与input的shared PSum，再用scaling multiplier乘不同scalar产生多个output channel结果，避免对每个DS重复计算完整MAC。

术语是什么？
从output channel（外积）视角看矩阵乘法：权重矩阵W的每一行（一个output channel）与input vector x做内积。SSMP将权重行分为SS行和DS行——SS行[i] = w_i，DS行[k·x+i] = w_i × S[p,q]（k=1..nx-1）。传统GEMV为每个output channel独立计算w_row × x。Outer-product PSum Reuse的核心洞察：SS行[i]和DS行[k·x+i]作用于相同的input slice（相同的y个input channels），因此它们的PSum满足：PSum_DS = PSum_SS × S[p,q]。所以只需计算SS行的PSum一次，DS行的结果通过scalar乘法得到——calc reduction from O(nx·ny·x·y) MACs to O(x·y) MACs + O(nx·ny) scalar mults。

从kernel调度角度拆解：
Outer-product PSum Reuse的kernel伪代码（4×4 PE，SSMP配置(8,8,4,4)，nx>ny选outer-product）：
```
// 单PE cluster处理一个regrouped group
// 输入: activation x[32]（对应y*ny=32 input channels）
//       SS[8][8]（x=8, y=8 source sub-matrix）
//       S[4][4]（nx=4, ny=4 scaling scalars）

// === 传统GEMV（Baseline）===
for p in 0..3:         // nx output channel groups
    for i in 0..7:     // x rows per group
        out_channel = p*8 + i
        for q in 0..3: // ny input channel groups
            for j in 0..7:  // y cols per group
                in_channel = q*8 + j
                // 对每个DS重复完整MAC
                weight = SS[i][j] * S[p][q]  // DS weight = SS × scalar
                out[out_channel] += weight * x[in_channel]
// 操作数: 4×8 × 4×8 = 1024 MACs（multiplications + additions）

// === Outer-product PSum Reuse（MECLA）===
// Step 1: 计算shared PSum（仅SS×input）
for i in 0..7:         // x SS rows
    for q in 0..3:     // ny input channel groups
        psum_ss[i][q] = 0
        for j in 0..7: // y SS cols
            in_channel = q*8 + j
            psum_ss[i][q] += SS[i][j] * x[in_channel]  // INT8 MAC
// 操作数: 8×4×8 = 256 MACs

// Step 2: Scaling PSum to get DS outputs
for p in 0..3:         // nx output channel groups
    for i in 0..7:     // x rows
        out_channel = p*8 + i
        for q in 0..3: // ny input groups
            // DS[p]的PSum = SS_PSum × S[p][q]
            scaled_psum = psum_ss[i][q] * S[p][q]  // scalar mult, much cheaper
            out[out_channel] += scaled_psum
// 操作数: 4×8×4 = 128 scalar multiplications + 128 additions
// 总计: 256 MACs + 256 scalar ops vs baseline 1024 MACs
// 计算量降至: (256+256)/1024 = 50%（示例），理论最大72% reduction

// 硬件映射到4×4 PE + 4×4 SA:
// PE row 0: 计算psum_ss[0][0..3]（4个32-bit PSum）
// PE row 1: 计算psum_ss[1][0..3]
// PE row 2: 计算psum_ss[2][0..3]
// PE row 3: 计算psum_ss[3][0..3]
// → 4×4 SA: 每个SA row处理一个PE row的PSum
//   SA[row=0][col=q]: psum_ss[0][q] × S[p][q] for p in 0..3
//   (4个multiplier对应4个不同的p值，但需多次cycle完成)
```
论文示例中power reduction 85.6%（从28 muls + 16 adds降至4 muls + 4 adds）。

术语一般如何实现？如何使用？
1. **PE Array设计配合**：需要PE array（MAC）+ scaling multiplier array（scalar mult）两级计算。PE做weight×activation的PSum，scaling multiplier做PSum×scalar的扩展。
2. **Mode选择**：当nx > ny时outer-product reuse更有效——更多output channels共享SS的PSum。RISC-V在解码SSMP配置后动态选择。
3. **Gating**：未使用的scaling multiplier（DS数量不足nx·ny时）clock gated节能。例如配置(8,8,2,2)时仅需4个multiplier active。
4. **局限性**：仅适用于SSMP partition——标准dense weight无法利用此reuse。非SSMP partition区域走标准dense MatMul datapath。
5. **与inner-product的互补**：当ny > nx时inner-product re-association（见下一条）更有效，MECLA通过dual-mode mapping自适应切换。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## Inner-product Computation Re-association（内积计算重关联）

术语解释
MECLA PE array在inner-product模式下利用SSMP的input channel方向PSum共性，通过改变矩阵乘法顺序——先计算scaling×input的乘积（输入缩放），再乘以SS weight——来实现计算复用，是对outer-product PSum reuse的互补优化。

术语是什么？
当SSMP配置满足ny > nx（更多input channel方向的DS扩展），从input channel（内积）视角看：权重矩阵W的不同列组作用于相同的input activation但被不同scaling scalar缩放。传统方法为每个input channel组独立计算SS_weight × x_slice。Inner-product re-association改变计算顺序：首先将所有DS scaling scalar与对应的input activation slice相乘（scaled_input = S × x），然后SS weight与scaled_input做标准内积。由于scalar×input的计算量远小于weight×input的MAC，重关联后计算量显著降低——相当于将scaling从weight侧"移动"到input侧，利用input vector较小（batch=1时仅1个token）的特点，scalar乘法开销极小。

从kernel调度角度拆解：
Inner-product Re-association的kernel伪代码（4×4 PE，ny > nx选择inner-product模式）：
```
// 输入: activation x[32]（y*ny=32 input channels）
//       SS[8][8]（SS weight）
//       S[4][4]（scaling scalars, ny=4 > nx=2为例）

// === 传统GEMV（Baseline）===
for p in 0..1:         // nx=2 output groups
    for i in 0..7:     // SS rows
        for q in 0..3: // ny=4 input groups
            for j in 0..7: // SS cols
                weight = SS[i][j] * S[p][q]
                out[p*8+i] += weight * x[q*8+j]
// 操作数: 2×8×4×8 = 512 MACs

// === Inner-product Re-association（MECLA）===
// Step 1: 先计算 scaled_input = S × x（scaling乘法）
for p in 0..1:         // nx
    for q in 0..3:     // ny
        for j in 0..7: // SS cols
            scaled_x[p][q*8+j] = S[p][q] * x[q*8+j]  // scalar mult
// 操作数: 2×4×8 = 64 scalar multiplications

// Step 2: SS weight × scaled_input（标准内积）
for p in 0..1:         // nx
    for i in 0..7:     // SS rows
        for q in 0..3: // ny
            for j in 0..7:
                out[p*8+i] += SS[i][j] * scaled_x[p][q*8+j]  // MAC
// 操作数: 2×8×4×8 = 512 MACs（与baseline相同）

// 但是！关键优化：scaled_x被nx=2个p值共享
// 更优的重关联（利用SS weight跨p的sharing）：
// Step 1': 对所有p，scaled_x once
for q in 0..3:
    for j in 0..7:
        scaled_x_base[q*8+j] = S[0][q] * x[q*8+j]  // p=0作为base
        // DS(p)的input = scaled_x_base × (S[p][q]/S[0][q])
        // 进一步减少scalar乘法

// 硬件映射（inner-product mode）:
// 与outer-product相反：S[1×4] → PE weight buffer, SS[4×1] → scale buffer
// PE array: 计算S_scaling_row × x → 产生scaled input PSum
// SA: scaled_input_PSum × SS_weight → output
```
论文中两种模式互补：nx>ny用outer-product，ny>nx用inner-product，相等时任一均可。

术语一般如何实现？如何使用？
1. **重关联条件**：当ny > nx时inner-product re-association更优——因input channel方向有更多scaling共享。在极限情况（ny=16, nx=2），inner-product可将计算量降至接近仅计算SS本身的开销。
2. **与outer-product共硬件**：同一套4×4 PE + 4×4 SA硬件支持两种模式——仅需交换SS和S在PE weight buffer/scale buffer中的映射位置，由RISC-V根据SSMP配置控制。
3. **Scaled input cache**：如果同一input activation被多个SSMP layer的不同DS scale使用，scaled_input可被cache在data buffer中跨层复用（论文未明确实现此优化）。
4. **Precision考虑**：scalar mult在FP16精度下进行（或混合精度：INT input × FP16 scalar），可能引入额外rounding error。论文通过fine-tuning过程中的SSMP weight adaptation吸收这些误差。
5. **极限case**：当ny极大（如16）而nx极小（如2）时，inner-product重关联几乎消除所有冗余MAC，PSum reuse接近理论极限。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## Dual-mode Mapping for PE Array（PE阵列双模映射）

术语解释
MECLA PE array的运行时自适应映射策略——根据SSMP配置中nx（output channel扩展因子）与ny（input channel扩展因子）的相对大小，动态选择outer-product PSum reuse或inner-product re-association模式，使PE array在任何SSMP配置下都能最大化计算复用效率。

术语是什么？
由于SSMP配置(x,y,nx,ny)因模型、层、任务而异（通过grid search确定），不同层可能有不同的nx/ny比例。例如某层nx=4, ny=2（outer-product reuse更优），另一层nx=2, ny=8（inner-product reassociation更优）。MECLA通过dual-mode映射策略自适应处理：(1) RISC-V core在解码每层SSMP配置时计算nx vs ny；(2) 若nx≥ny→Mode=OUTER_PRODUCT：SS weight→PE weight buffer，S→scale buffer；(3) 若nx<ny→Mode=INNER_PRODUCT：S→PE weight buffer，SS weight→scale buffer。模式切换通过改变DMA加载顺序和PE buffer映射完成，不改变PE/SA硬件本身——同一套硬件支持两种计算流程。

从kernel调度角度拆解：
Dual-mode mapping的运行时决策和执行流程：
```
// RISC-V core: 每层inference前执行mode selection
function select_mode(ssmp_config):
    (x, y, nx, ny) = ssmp_config
    if nx >= ny:
        return OUTER_PRODUCT
    else:
        return INNER_PRODUCT

// Mode-specific data loading（OUTER_PRODUCT mode）
function load_weights_outer_product():
    // 1. SS weight → PE weight buffer（4×4 PE array per group）
    for pe_row in 0..3:
        for pe_col in 0..3:
            PE_group[group_id].weight[pe_row][pe_col] = SS[ss_row][ss_col]
    
    // 2. Scaling scalar → scale buffer（4×4 SA per group）
    for sa_row in 0..3:
        for sa_col in 0..3:
            SA_group[group_id].scale[sa_row][sa_col] = S[ds_p][ds_q]
    
    // 3. 执行流程
    // PE: PSum = Σ weight[pe_row][pe_col] * x[input_channel]
    // SA: output = PSum × scale[sa_row][sa_col]

// Mode-specific data loading（INNER_PRODUCT mode）
function load_weights_inner_product():
    // 1. Scaling scalar → PE weight buffer（角色互换）
    for pe_row in 0..3:
        for pe_col in 0..3:
            PE_group[group_id].weight[pe_row][pe_col] = S[ds_p][ds_q]
    
    // 2. SS weight → scale buffer（角色互换）
    for sa_row in 0..3:
        for sa_col in 0..3:
            SA_group[group_id].scale[sa_row][sa_col] = SS[ss_row][ss_col]
    
    // 3. 执行流程（重关联）
    // PE: scaled_input = Σ S × x  （scaling × input first）
    // SA: output = scaled_input × SS_weight  （then × weight）

// 边界case处理
if nx == ny:
    // 任一模式等效，默认选OUTER_PRODUCT
    // 或根据其他因素（如buffer容量、batch size）选择
```

双模映射的关键是regrouping策略也随mode改变：
- Outer-product: regroup by output channel → 同SS的nx个output DS行聚集
- Inner-product: regroup by input channel → 同SS的ny个input DS列聚集

术语一般如何实现？如何使用？
1. **硬件复用**：PE array和SA的硬件完全复用——模式切换仅改变数据加载路径和buffer映射，不涉及硬件重配置。这是MECLA PE array设计的关键约束：4×4 PE + 4×4 SA需同时支持两种计算模式。
2. **RISC-V控制**：Mode select逻辑在RISC-V core中以软件指令实现（非硬件FSM），提供灵活性——可在不同层之间动态切换而无需重新综合。
3. **SSMP配置依赖性**：dual-mode的有效性依赖于SSMP的nx、ny参数有足够的多样性。论文实验表明标准配置(8,8,4,4)（nx=ny=4）下两种模式等效，更aggressive配置如(8,8,8,2)或(8,8,2,8)才使dual-mode选择有意义。
4. **Overhead**：模式切换overhead仅是一次buffer flush + DMA重配置（~数十cycles），相比单层推理的数千cycles可忽略。
5. **与静态dataflow对比**：传统加速器（如Eyeriss、TPU）使用固定dataflow（weight stationary/output stationary等）；MECLA的dual-mode介于静态和fully reconfigurable之间——仅两种模式但每种覆盖SSMP的互补reuse pattern。

涉及论文标题：
- 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

## FFT-based Convolution Operator (FFTConv)

术语解释
FFTConv是SSM-based全局卷积模型中利用快速傅里叶变换（FFT）在频域执行卷积的算子。将时域卷积O(l²)降为频域逐点乘法O(l log l)，是H3等模型实现sub-quadratic复杂度的关键。

术语是什么？
FFT-based convolution利用卷积定理：两个向量的卷积等价于它们各自傅里叶变换的逐点乘积再逆变换。算子流程：FFT(input) ⊙ FFT(filter) → IFFT → output。H3模型中使用Generalized Cooley-Tukey算法执行FFT/IFFT（因chunk size L可能非2的幂次——实际选L=2048=2^11使用Radix-2，但通用算法支持任意可分解长度）。FFTConv占H3 ROI时间的~42%（序列长度128K），是ROI中最大的单一操作。尽管计算复杂度已从O(l²)降至O(l log l)，在GPU上仍memory-bound——因为每stage FFT需全序列读写和barrier同步，即使数据fit in shared memory，DRAM bandwidth utilization极低，compute utilization远低于峰值。

从kernel调度角度拆解：
Generalized Cooley-Tukey FFT在chunk size L上的执行（L = L1 × L2，以L=6为例L1=2, L2=3）：
```
Function FFTConv_2D(input_chunk u[L], filter K[L]):
    // Step 1: Reshape u → U_2D[L1][L2]
    
    // Step 2: Column-wise FFT (L2次独立长度L1的FFT)
    for j in 0..L2-1:
        FFT_1D(column U_2D[:, j])  // BF mode, log2(L1) stages
    
    // Step 3: CTF Multiplication
    for i in 0..L1-1:
        for j in 0..L2-1:
            U_2D[i][j] *= CTF[i][j]  // CMult/CTFGen mode
    
    // Step 4: Row-wise FFT (L1次独立长度L2的FFT)
    for i in 0..L1-1:
        FFT_1D(row U_2D[i, :])  // BF mode, log2(L2) stages
    
    // Step 5: Reshape back to 1D
    U = reshape(U_2D)
    
    // Step 6: Multiply with filter in frequency domain
    U = U ⊙ FFT(K)  // CMult mode
    
    // Step 7: IFFT (same structure as FFT with conjugate twiddle)
    u_conv = IFFT_2D(U)
    
    return u_conv
```

在VGA硬件上的kernel调度：
- Column-wise FFT: 每个CCU处理一列，BF mode用Register File存twiddle factor（跨stage复用）
- CTF Generation: 偶数CCU为CTFGen mode生成CTF→传递给奇数CCU（CMult mode）与列FFT结果相乘
- Row-wise FFT: 需SRAM行/列访问模式切换（D-SRAM的circular rotation消除bank conflict）
- 乘filter K_f: 全CMult mode
- IFFT: 同FFT但twiddle factor取共轭

术语一般如何实现？如何使用？
1. **GPU实现（CUDA kernel）**：使用cuFFT或自定义kernel，数据fit in shared memory时利用warp shuffle。但随序列增长需spill到global memory（DRAM bandwidth bottleneck）。FlashFFTConv [20]使用矩阵乘法（Tensor Core）实现FFT以提升GPU利用率，但仅支持FP16。
2. **Chunk size选择**：L=2048（=2^11）使得2D-FFT可作为32×64或64×32方阵执行，最大化CCU阵列利用率。
3. **Twiddle factor复用**：同一FFT stage内所有BF操作使用相同twiddle factor，存入CCU Register File跨cycle复用，无SRAM访问。
4. **精度考虑**：FP32全精度——因FFT/IFFT的累加误差在长序列中会传播放大，FP16不够。
5. **与Parallel Scan关系**：FFTConv比parallel scan快~3.3×（Table II），但FFT需要滤波器预定义（parallel scan更灵活）。

涉及论文标题：
- 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf

## Butterfly Operation in FFT Hardware（FFT硬件蝶形运算）

术语解释
Butterfly (BF) operation是Cooley-Tukey FFT算法的基本运算单元，将两个输入(E_i, O_i)通过一次复数乘法和一次加减法变换为两个输出(U_i, U_{i+L/2})。在VGA中，BF由CCU的M3 mode在单cycle内完成。

术语是什么？
Radix-2 Cooley-Tukey FFT的Butterfly操作定义为：
```
Input: E_i, O_i, twiddle factor z^{2i}
Output: U_i = E_i + z^{2i}·O_i
        U_{i+L/2} = E_i - z^{2i}·O_i
```
其中z是满足z^L=1的复数单位根，z^{2i}称为twiddle factor。BF操作将长度为L的DFT分解为两个L/2子DFT的结果组合，通过log2(L)级递归分解使复杂度从O(L²)降至O(L log L)。每级包含L/2个独立BF操作，可完全并行。

从kernel调度角度拆解：
在VGA CCU的BF mode (M3) 中的硬件执行：
```
// 每个CCU独立处理一对(E_i, O_i)
// Register File预存twiddle factor T = z^{2i}

Cycle:
  1. CMult Unit: O_i × T → twisted_O (复数乘法, 4mul+2add)
  2. Reconfigurable Array:
     - Adder pair 1: E_i + twisted_O → U_i
     - Adder pair 2: E_i - twisted_O → U_{i+L/2}
  3. 两个结果(U_i, U_{i+L/2})写入output vector

// Twiddle factor T在同一FFT stage的所有L/2个BF操作中保持不变
// 因此T只需在stage开始时load到Register File一次
```

在GPU CUDA kernel中的BF实现：
```
// 典型warp级实现
__shared__ float2 shared_data[L];
float2 T = twiddle_factors[thread_idx];
float2 E = shared_data[2*thread_idx];
float2 O = shared_data[2*thread_idx + 1];
float2 twisted = complex_mul(O, T);  // 2次FMA (FP32)
float2 U_0 = complex_add(E, twisted); // 2次FADD
float2 U_1 = complex_sub(E, twisted); // 2次FADD
shared_data[2*thread_idx] = U_0;
shared_data[2*thread_idx+1] = U_1;
__syncthreads();  // 每stage结束需barrier同步
```
GPU的瓶颈在于__syncthreads() barrier和shared memory bank conflict，限制compute utilization。

术语一般如何实现？如何使用？
1. **FFT stage流水**：log2(L)个BF stage串行执行（每stage依赖前一stage输出）。Stage间通过D-SRAM传递数据（写入当前stage结果→下stage读取），无DRAM访问（数据全在D-SRAM内）。
2. **Bank conflict消除**：D-SRAM circular row rotation确保row/col访问均跨不同bank。
3. **BF vs CMult throughput**：BF mode每CCU每cycle产出2个复数结果（vs CMult的1个），因此FFT stage的执行效率较高。
4. **IFFT**：与FFT使用相同BF硬件，仅twiddle factor取共轭（z^{-2i}），并在最后除以L。

涉及论文标题：
- 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf

## CTF (Compensated Twiddle Factors)

术语解释
CTF (Compensated Twiddle Factors) 是Generalized Cooley-Tukey FFT算法中用于补偿行FFT和列FFT之间phase差异的复数因子矩阵。在2D-FFT流程中，列FFT之后、行FFT之前需将中间结果逐元素乘以CTF矩阵。CTF也是Vandermonde矩阵，可在VGA的CCU上on-the-fly生成。

术语是什么？
Generalized Cooley-Tukey算法将长度为L=L1×L2的1D FFT分解为：列FFT→CTF乘法→行FFT。数学上，1D DFT矩阵可写为F_L = (F_{L1} ⊗ I_{L2}) · diag(vec(CTF)) · (I_{L1} ⊗ F_{L2}) · P，其中P是reshape permutation。CTF矩阵的元素CTF[i][j] = ω^{i·j}（ω = e^{-2πi/L}），分别补偿L1行FFT和L2列FFT之间的"距离"。CTF矩阵具有Vandermonde结构：CTF[i+1][j] = CTF[i][j] × ω^j（行方向循环关系），CTF[i][j+1] = CTF[i][j] × ω^{i·L1}（列方向循环关系）。

从kernel调度角度拆解：
在VGA上CTF的生成与乘法的硬件执行：
```
// CTFGen mode (M2) on even-indexed CCUs
// CMult mode (M1) on odd-indexed CCUs

初始化:
  CCU_0 (CTFGen): Register File存 CTF[0][0]=(1+0i), gen_factor=ω^0
  CCU_1 (CMult): 接收CCU_0传来的CTF[0][0]
  ...

Cycle 0:
  CCU_0: output CTF[0][0] → CCU_1 (unidirectional连接)
  CCU_1: C_i × CTF[0][0] → 写回 (C_i为列FFT结果)
  
Cycle 1:
  CCU_0: next_CTF = CTF[0][0] × gen_factor → CTF[0][1]
         output CTF[0][1] → CCU_1
  CCU_1: C_{i+L1} × CTF[0][1] → 写回
  ...
```

由于CTFGen仅需偶数CCU，整个CCU阵列按(CTFGen, CMult, CTFGen, CMult, ...)交替配置，最大化阵列利用率。

术语一般如何实现？如何使用？
1. **存储开销**：存储完整CTF矩阵需要L1×L2个复数（对L=2048为~32KB FP32 complex）。VGA的on-the-fly生成仅需存初始元素+生成因子（~几个复数）。
2. **生成速率匹配**：CTFGen每cycle生成一个元素→传给相邻CMult CCU消费，生成与消费速率1:1匹配，无bubble。
3. **行/列方向生成**：列FFT后用列方向CTF生成（沿行循环），行FFT前用行方向CTF生成。两者使用相同的CCU mode（M2+M1 pair）但不同的生成因子。

涉及论文标题：
- 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf
- 46-Fast On-device LLM Inference with NPUs.pdf

## Out-of-order Subgraph Execution

术语解释
Out-of-order subgraph execution 是 llm.npu 提出的 CPU/NPU 异构调度算法。在 LLM prefill 被 chunk-sharing graph 拆分为多个 chunk×subgraph 后，通过打破 chunk 序列顺序约束，对就绪子图按"最大化减少 NPU stall"的贪心策略在线调度到 CPU/GPU 或 NPU 执行，将 bubble rate 从 37% 降至 0.7%。

术语是什么？
LLM prefill 的量化推理流程中，NPU 负责 INT8 Linear/FFN（计算量大），CPU 负责 LayerNorm、Attention（Softmax 等浮点操作）和 shadow outlier MatMul（计算量小）。Naive 重叠（按 chunk 顺序依次执行）导致大量 CPU 等待 NPU 的空闲 bubble（37% bubble rate）。Out-of-order 调度的核心观察：(1) NPU 执行时间常构成 critical path（如 256-token prompt 的 Qwen1.5-1.8B，NPU 315ms vs CPU ~一半）；(2) 各 subgraph 之间存在两层依赖——跨 chunk 依赖（Attention 依赖前序 chunks 的 KV Cache）和 chunk 内依赖（LayerNorm→Linear 的顺序）；(3) 满足依赖前提下，多个 chunk 的子图可跨 chunk 乱序执行。

从kernel调度角度拆解：
```
# 离线阶段：profiling 各 subgraph 在 NPU/CPU 的执行时间及依赖关系
for each subgraph g in chunk_sharing_subgraphs:
    g.time[NPU] = profile_on_NPU(g)
    g.time[CPU] = profile_on_CPU(g)
    g.processor = NPU if g.type in [INT8_Linear, INT8_FFN] else CPU
    g.deps_cross_chunk = [...]  # cross-chunk dependency edges
    g.deps_intra_chunk = [...]  # intra-chunk sequential edges

# 在线调度：贪心选择减少 NPU stall 最大的子图
def online_scheduler(prompt_tokens, chunk_length=256, all_subgraphs):
    num_chunks = ceil(len(prompt_tokens) / chunk_length)
    ready_queue = PriorityQueue()
    results = {}
    
    # 初始化：所有 chunk 的第一个子图入队
    for c in range(num_chunks):
        ready_queue.push(subgraph(c, 0))
    
    while not ready_queue.empty():
        best_g = None
        best_contribution = -inf
        
        for g in ready_queue:
            # 计算 g 完成后可释放到 NPU/CPU 的子图集合 S
            S = get_newly_ready_subgraphs_after_completion(g, results)
            total_npu_time_of_S = sum(s.time[NPU] for s in S if s.processor == NPU)
            total_cpu_time_of_S = sum(s.time[CPU] for s in S if s.processor == CPU)
            
            if g.processor == CPU:
                # CPU 上的子图：完成后释放 S 到 NPU → 贡献 = S 的 NPU 执行时间
                contribution = total_npu_time_of_S
            else:  # NPU
                # NPU 上的子图：完成后释放 S 到 CPU → 贡献 = -S 的 CPU 执行时间
                contribution = -total_cpu_time_of_S
            
            if contribution > best_contribution:
                best_contribution = contribution
                best_g = g
        
        # 执行选中的子图
        execute_on_processor(best_g, best_g.processor)
        results[best_g.id] = True
        
        # 将 g 的 successors 中所有依赖满足的入队
        for succ in best_g.successors:
            if all(dep in results for dep in succ.dependencies):
                ready_queue.push(succ)
    
    return results
```
调度算法的直观理解：优先执行那些"完成后能为 NPU 释放最多工作"的 CPU 子图（最大化 NPU 利用率），或优先执行那些"完成后仅释放最少额外 CPU 工作"的 NPU 子图（最小化对 NPU 的干扰）。由于该调度问题是 NP-hard（可规约为 TSP 变体），论文采用此在线贪心算法，决策开销在微秒级。

术语一般如何实现？如何使用？
在 llm.npu 中，该调度器作为 MLLM 运行时的推理调度层实现：(1) 离线 profiling 阶段：用 QNN 的性能 profile API + Android systrace 获取各子图在各自处理器上的精确延迟；(2) 运行时调度：以 microsecond 级开销在线计算 pending 子图的贡献值 C，选择 max C 执行。适用条件：存在多个可并行处理的独立子图（长 prompt → 多 chunk → 大调度空间）、CPU/NPU 异构且非抢占式（每处理器一次仅一个任务）、NPU 执行时间 dominant。局限性：短 prompt（64 token）下 chunk 数少，调度空间不足（仅 12% 额外加速 vs naive）；需离线 profiling 收集准确的每子图执行时间。

涉及论文标题：
- 46-Fast On-device LLM Inference with NPUs.pdf

## Context-aware Merged Mapping for Early Exiting (上下文感知的早退合并映射)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Context-aware Merged Mapping 是 SpecEE 为 speculative decoding 场景下的 early exiting 设计的一种映射方法，用于将 token tree 中每条 path 的多个 token 合并为一个 hyper-token，将 predictor 的 mapping complexity 从指数级降为线性级。核心问题：在 speculative decoding 中，token tree（如 root → 3 branches → 9 tokens）中的每个 token 在当前 early exiting mapping 下被独立处理——每个 tree node 都需要单独执行 predictor feature computation（hidden_states × speculative_lm_head 等），导致 mapping complexity 随 tree size 指数增长。Merged Mapping 利用 token tree 中 path 内各 token 共享 contextual dependency（同一 path 上的 token 具有相似的 exit layer，context similarity hit rate >70%）和相同的 speculative_lm_head，将所有 path 合并为 hyper-token batch 统一计算，将复杂度从 O(tree_size × hidden_dim) 降为 O(tree_depth × num_spec_tokens)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Custom GPU kernel 基于 cutlass group GEMM 的 merged mapping 计算流程：
```python
# 输入: token_tree (depth=D, branch=B), hidden_states per layer
#       speculative_lm_head [hidden_dim, num_spec_tokens] (e.g. [4096, 4])

# Step 1: 提取所有 token path 为 hyper-tokens
hyper_tokens = extract_paths(token_tree)  # [num_paths], e.g. 1+3+9=13 tokens grouped by path

# Step 2: 收集每条 path 的 hidden states
hidden_batch = gather_hidden(hyper_tokens, layer_idx)  # [num_paths, hidden_dim]

# Step 3: Custom CUDA group GEMM（基于 cutlass block-wise GEMM + MegaBlocks 设计）
# 所有 path 共享同一个 speculative_lm_head，通过 batch GEMM 一次计算
spec_logits = cutlass_group_gemm(hidden_batch, speculative_lm_head)  # [num_paths, 4]

# Step 4: 批量 feature extraction + MLP predictor
local_probs = batch_softmax(spec_logits)          # [num_paths, 4]
prob_var = local_probs - prev_probs               # [num_paths, 4]
feats = concat([spec_logits, local_probs, prob_var])  # [num_paths, 12]

# MLP predictor 对 batch 执行（GPU tensor core 并行）
h = relu(feats @ W1.T)      # [num_paths, 512]  — W1 [512, 12]
preds = sigmoid(h @ W2.T)   # [num_paths, 1]    — W2 [1, 512]

# Step 5: 对 pred > 0.5 的 path 执行 verification
# Cannikin law: path 的 exit layer = max(各 node exit layer)
for i, path in enumerate(hyper_tokens):
    if preds[i] > 0.5:
        global_logits = hidden_batch[i] @ lm_head  # [1, vocab_size]
        if argmax(global_logits) in spec_tokens:
            exit_paths.append(path)
```

术语一般如何实现？如何使用？
具体实现基于：(1) cutlass group GEMM——将多个小 GEMM（每条 path 的 hidden × spec_lm_head）合并为一次 block-wise matrix multiplication，利用 A100 Tensor Core 的 FP16 算力（312 TFLOPS）；(2) MegaBlocks block-sparse 设计——不同 path 的 hidden states 属于不同的 "block"，通过 group GEMM 统一处理，避免 per-token kernel launch overhead；(3) Cannikin law 处理——若 path 中某 node 未 exit，该 path 继续推理，exit position 为最远 node 的 exit layer（类似短板效应）；(4) 结果：SpecEE+EAGLE 在 Llama2-7B/A100 上实现 ~1.05× speedup vs pure EAGLE（120.8 vs 114.5 tokens/s on MT-Bench）。合并映射将 mapping 开销从 exponential 降为 linear，使得 speculative decoding 下 predictor overhead 可控。

涉及论文标题：
- 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting.pdf

## Hyper-token in Speculative Decoding (推测解码中的超token)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hyper-token（超token）是 SpecEE 在 speculative decoding + early exiting 场景下引入的抽象概念，将 token tree 中一条完整 path 上的所有 token 合并为一个语义单元。在 speculative decoding 中，DLM 生成的 token tree 包含多层多分支的 token 结构（如 1 root → 3 level-1 → 9 level-2 tokens）。传统的 early exiting mapping 将每个 tree node 视为独立搜索空间，hyper-token 则将每条从根到叶的 path 压缩为一个抽象 token。这种抽象利用了 path 内各 token 的 context similarity——相邻 token 的 exit layer 高度相关（最近 5 tokens ±2 层 hit rate >70%），因此可以统一处理。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hyper-token 的计算抽象：
```
Token Tree Structure:
        ? (root)
      /  |  \
     I   It  Thank (level 1, 3 tokens)
    /|\  ...  ... (level 2, 9 tokens)
   
Hyper-token Paths:
  Path 1: ? → I → thank
  Path 2: ? → I → am
  Path 3: ? → I → can
  Path 4: ? → It → is
  ...

对每条 path，hyper-token 的特征计算：
  spec_logits = path_hidden @ spec_lm_head  # 提取该 path 最后 token 的 hidden states
  # spec_lm_head 对所有 path 共享（同一组 speculative tokens）
  
  merged_hidden_batch = stack([path1_hidden, path2_hidden, ...])  # [P, hidden_dim]
  batch_spec_logits = merged_hidden_batch @ spec_lm_head         # [P, 4]
```
Custom GPU kernel 实现：基于 cutlass group GEMM，将 batch 维度（P）和 speculative token 维度（4）组织为 block-wise GEMM。参考 MegaBlocks 的 block-sparse matmul 设计——不同 path 的特征计算 pattern 一致（hidden × 相同 spec_lm_head columns），通过 group GEMM 合并为一次 kernel launch。

术语一般如何实现？如何使用？
Hyper-token 映射的使用方式：(1) 映射阶段——在每层 decoder layer 后，收集 token tree 所有 path 的 hidden states，组织为 batch [num_paths, hidden_dim]；(2) 计算阶段——通过 cutlass group GEMM 一次计算所有 path 的 draft token logits → softmax → features → MLP predictor；(3) 决策阶段——逐 path 判断 exit，未 exit 的 path 继续下一层推理，exit 的 path 通过 verification algorithm 确认输出；(4) Cannikin law——path 的最终 exit layer = max(path 内各 node 的 exit layer)。Hyper-token 映射复杂度从 O(tree_nodes × hidden_dim) 降至 O(paths × num_spec_tokens) ≈ O(tree_depth × num_spec_tokens)。Limitation：需要 GPU batch 处理能力，不适合单 token 推理场景（如纯 autoregressive decoding 不构建 token tree 时不需要 hyper-token 抽象）。

涉及论文标题：
- 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting.pdf

## FC (Fully Connected) Kernel Auto-Tuning with Performance Database

术语解释
FC Kernel Auto-Tuning是MTIA 2i对全连接层kernel的自动优化方法，通过kernel generator生成多种stationary变体→构建performance database→使用approximate nearest neighbor search快速选择最优kernel variant，将exhaustive tuning时间缩短1000x。

术语是什么？
推荐模型中的FC (Fully Connected) layer（即MLP层/Linear层）是矩阵乘法GEMM占主导的层。MTIA 2i上FC kernel的效率高度依赖于：(1) tensor shape (M×K, K×N)；(2) 数据放置（activations在LLS还是LPDDR，weights在LLC还是LPDDR）；(3) 数据精度（FP16/INT8）；(4) batch size（影响M维度）。Kernel generator可生成三种stationary变体：input stationary（activation常驻DPE cache）、output stationary（output在Local Memory中累加）、weight stationary（weight tile常驻DPE cache）。每种变体可调参数包括：block size（tile size）、DMA调度顺序、circular buffer大小。直接对所有shape和所有变体做exhaustive test寻找最优kernel耗时过长。MTIA的解决方案：(1) 预先构建performance database——对各类(shape, data placement, precision)组合采样测试kernel性能；(2) 对新遇到的FC shape→将shape特征向量化→在performance database中使用approximate nearest neighbor (ANN) search找最相似的已知shape→复用其最优kernel variant；(3) tuning时间缩短1000x，最终性能在exhaustive tuning最优的5%以内。

从kernel调度角度拆解术语：
FC kernel auto-tuning的执行流程伪代码：
```
# Phase 1: Offline Database Building
for shape in sample_shape_space:  # 采样shape空间
    for variant in [input_stat, output_stat, weight_stat]:
        for placement in [LLS, LLC, LPDDR]:
            # 运行kernel并记录性能
            latency = run_kernel(shape, variant, placement)
            db.insert(shape_features, variant, placement, latency)

# Phase 2: Online Selection (per model FC layer)
fc_shape = (M=2048, K=1024, N=4096, dtype=FP16)
features = [M, K, N, flops_intensity, arithmetic_intensity, dtype_encoding]
nearest = ann_search(db, features, k=5)  # 找5个最近邻
# 对nearest中的候选做少量实测验证
best = min(nearest, key=lambda x: run_kernel(fc_shape, x.variant, x.placement))
compile_and_cache(best)  # 编译为custom instructions并缓存
```
关键要素：(1) Shape特征向量——不仅包括(M,K,N)绝对尺寸，还包括compute intensity (FLOPS/byte)、SRAM occupancy ratio等归一化特征；(2) ANN algorithm——论文未明确说明具体算法（可能是LSH或IVF-PQ）；(3) Kernel variant selection影响——不同shape条件下最优变体不同，如大M（大批量）时output stationary有利，小M时weight stationary有利。

术语一般如何实现？如何使用？
FC kernel auto-tuning的使用方式：(1) 模型onboarding流程——新模型通过TorchDynamo trace→识别所有FC shapes→对每个unique shape调用ANN search选择kernel variant→autotuning同时选择batch size和data placement；(2) 与PyTorch编译栈集成——MTIA Graph Compiler在编译时调用auto-tuning→生成compiled subgraph含选定的kernel variants；(3) 局限——ANN search假设shape相似→最优variant也相似，对非常out-of-distribution的shape可能不准确（但论文报告5%以内）。类似系统：TVM AutoTVM/AutoScheduler使用ML-based cost model预测kernel性能；Triton的autotuning使用grid search+early pruning。MTIA的特殊性在于异构PE架构使kernel variant空间更大（3种stationary × continuous DMA/CB parameters），而GPU tuning主要关注tile size/thread block配置。

涉及论文标题：
- 48-Meta_s Second Generation AI Chip- Model-Chip Co-Design and Productionization Experiences.pdf

## Activation/Weight Decoupling for Memory-Bandwidth-Bound GEMM

术语解释
Activation/Weight Decoupling是MTIA 2i针对DRAM bandwidth-bound GEMM的kernel优化技术：当activation fit in PE Local Memory但weight超出SRAM时，将activation loading（从LLS prefetch）和weight loading（broadcast across PE columns）解耦为独立pipeline阶段，利用硬件broadcast read消除NoC contention，并通过weight tile prefetch到LLC隐藏DRAM延迟。

术语是什么？
在推荐模型的复杂FC层中，weight tensor可能非常大（如109MB），无法完全fit在256MB SRAM中（因SRAM还需容纳activation buffer等其他数据）。此时GEMM kernel变为DRAM bandwidth-bound——DPE等待weight tile从LPDDR加载。Activation/Weight Decoupling的核心思想：(1) Activation通常fit在PE的Local Memory中（384KB per PE，distributed across 64 PEs→合计约24.6MB distributed LS）；(2) 将activation loading（从LLS到PE LS）和weight loading（从LPDDR到LLC再到PE LS）的时间线解耦——activation提前加载到PE LS（prefetch from LLS），weight loading独立进行（从LLC→broadcast到PE columns），两者不争抢NoC资源；(3) 利用MTIA 2i的hardware broadcast read——weight tile读一次从LLC broadcast到同一列的8个PE，消除NoC per-PE独立读取的contention；(4) Weight tile prefetch——在DPE计算当前tile时，DMA engine预取下一个weight tile从LPDDR到LLC中，隐藏DRAM延迟。

从kernel调度角度拆解术语：
以shape为512×26592×2048（weight tensor 109MB）的FC layer为例，伪代码：
```
// Decoupled GEMM pipeline on MTIA 2i PE grid
// 假设activations (512x26592, ~52MB) fit in distributed PE Local Memory

// Phase 1: Preload activations to PE LS (one-time cost)
for pe_col in 0..7:          // 8 PE columns
    for pe_row in 0..7:       // 8 PE rows
        FI_DMA(pe[pe_row][pe_col].LS, LLS_offset(activation_tile[pe_row][pe_col]))

// Phase 2: Pipeline weight loading + GEMM execution
for weight_tile_idx in 0..num_weight_tiles:
    // Prefetch next weight tile: LPDDR -> LLC (hide DRAM latency)
    if weight_tile_idx + 1 < num_weight_tiles:
        FI_DMA_PREFETCH(LLC, LPDDR_addr(weight_tile[weight_tile_idx + 1]))
    
    // Broadcast current weight tile: LLC -> all PEs in column
    for pe_col in 0..7:
        // hardware broadcast: one LLC read, all 8 PEs in column receive
        FI_DMA_BROADCAST(pe[*][pe_col].LS, LLC_addr(weight_tile[weight_tile_idx]))
    
    // GEMM on each PE: activation (cached) x weight tile (just loaded)
    for pe in all_64_PEs:
        CP_issue_DPE_GEMM(pe.LS.activation_tile, pe.LS.weight_tile, RE_accum)
    
    // Barrier: all PEs finish current tile before loading next tile
    CP_wait_all_complete()
```

关键点：(1) activation preload是一次性的——在所有weight tiles计算期间reuse；(2) weight broadcast读取一次LLC，同时发送到8个PE（同一column），NoC bandwidth需求降至1/8；(3) prefetch与compute overlap——weight_tile_idx的GEMM执行期间，DMA engine预取weight_tile_idx+1到LLC。

术语一般如何实现？如何使用？
此技术的使用条件：(1) Activation能fit在PE distributed Local Memory中；(2) Weight tensor超出SRAM→DRAM bandwidth-bound场景。对于更大batch size（activation不fit LS→需额外tiling level on activation dimension）。通用化：此类优化思想可应用于任何具有distributed scratchpad memory+shared cache+off-chip DRAM层次的AI加速器，核心是解耦不同数据的加载路径以减少资源竞争。MTIA 2i实现效果：latency改善45%，达到>95% DRAM bandwidth。

涉及论文标题：
- 48-Meta_s Second Generation AI Chip- Model-Chip Co-Design and Productionization Experiences.pdf

## PIM-offloaded Distance Computation for Graph-based ANNS

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM-offloaded Distance Computation for Graph-based ANNS 是将graph-based Approximate Nearest Neighbor Search中的distance computation操作从host CPU offload到DIMM内的PIM处理单元执行的kernel调度技术。在HeterRAG中，该操作占retrieval阶段执行时间的>80%。传统CPU-GPU方案中，CPU需从DDR4 DRAM读取scattered vertex vectors到CPU cache/register再计算distance→DDR4 bandwidth (~19.2GB/s)和random access pattern成为瓶颈。PIM offload通过在DRAM rank内部嵌入Distance Computation Unit (FP32 Mult+Adder)，利用DIMM内部远高于外部接口的有效带宽，在数据驻留地执行计算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PIM-offloaded Distance Computation的伪代码（以一个RPM处理一次distance computation为例）：
```
// 主机侧：ANNS主循环 (在AccelDIMM TPM Functional Block上执行)
priority_queue PQ = {starting_vertices} // 初始化为起始顶点
visited_list = {}

while not termination:
    // Step 1: Neighbor Fetching (在TPM Functional Block执行)
    cur_vertex = PQ.pop_nearest_unvisited()
    neighbors = DDR_READ(cur_vertex.neighbor_addr) // 标准DDR读
    new_neighbors = neighbors.filter(visited_list)

    // Step 2: Distance Computation (PIM-offloaded到RPM执行)
    for each vid in new_neighbors:
        pim_inst = encode_pim_inst(
            rank_id = hash(vid) % num_ranks,  // vertex分布的rank
            dram_addr = vector_table[vid],     // vector在DRAM中的地址
            ddr_cmds = [ACT, RD, PRE]          // 预编码的DDR命令
        )
        send_to_rpm(rank_id, pim_inst)         // 通过DPM分发

    // Step 3: Queue Updating (在TPM Functional Block执行)
    distances = collect_from_all_rpms()        // 从DPM Distance Buffer收集
    for each (vid, dist) in distances:
        PQ.insert_or_update(vid, dist)

// RPM侧：PIM-Inst执行
rpm_execute(pim_inst):
    vid = extract_vid(pim_inst.dram_addr)
    // Vertex Cache lookup
    if vid in vertex_cache:
        vector = vertex_cache[vid]             // Cache hit
    else:
        // DDR-C/A Generator发出DDR命令
        ddr_activate(pim_inst.bank, pim_inst.row)      // ACT
        vector = ddr_read(pim_inst.column)              // RD
        ddr_precharge(pim_inst.bank)                    // PRE
        vertex_cache.insert_lru(vid, vector)            // 更新cache

    // Distance计算 (inner product)
    dist = 0
    for i in 0..vector_len PARALLEL(32):   // 32 FP32 lanes
        dist += query_reg[i] * vector[i]   // FP32 Mult + Adder tree
    write_to_distance_buffer(vid, dist)
```
关键调度决策：
- **offload粒度**：仅distance computation offload到RPM；neighbor fetching (需Visited List Buffer 16MB, RPM面积不够) 和queue updating (需全局排序) 保留在TPM。
- **并行度**：RPM级（2 ranks/DIMM × N DIMMs）并发执行distance computation。DPM内的PIM-Inst Queue (256-entry) 和RPM内的PIM-Inst Queue (128-entry) 缓冲和解耦TPM与RPM的执行。
- **指令优化**：PIM-Inst采用RecNMP式压缩——将多DDR command (ACT/RD/PRE) 编码为单一PIM指令。传输采用TRiM两阶段技术——先发PIM-Inst到DPM Queue (phase 1)→DPM再转发到RPM Queue (phase 2)，隐藏传输延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PIM-offloaded Distance Computation的实现：(1) 硬件——每个RPM集成32 FP32 Mult + 32 FP32 Adder（支持128-dim vectors需4次向量内积操作，32 lanes×4 passes），可扩展支持L2距离等度量。(2) 内存组织——vector data和graph neighbors采用unified memory organization存储，vertex-level partition到各rank确保负载均衡。(3) 性能——PIM offload消除了CPU-DDR4的external bandwidth瓶颈，利用DIMM内部有效带宽（多个bank并行），实现retrieval throughput显著提升。类似的思想也见于MemANNS和DRIM-ANN（基于UPMEM RISC cores），但HeterRAG的专用distance computation硬件比通用RISC core更高效（平均25.15×和28.42× higher QPS）。

涉及论文标题：
- 49-HeterRAG- Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation.pdf

## GEMV PIM Offloading (Bank-level GEMV in HBM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GEMV PIM Offloading 是将LLM decode阶段dominant的General Matrix-Vector Multiplication (GEMV) 操作从外部计算单元（GPU SM/TPU MXU）offload到HBM bank内部的PIM处理单元执行的技术。GEMV的算术强度极低（单token输入下FLOP/Byte比远低于roofline knee），使其成为典型的memory-bound操作——受限于HBM external bandwidth而非计算能力。通过在HBM bank内部嵌入BPM (Bank-level Processing Module)，GEMV计算直接在weight和activation数据所在的DRAM bank内完成，利用HBM bank级内部带宽（远高于HBM外部I/O bandwidth），消除memory wall。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GEMV PIM Offloading在HeterRAG AccelHBM中的执行流程（以LLaMA2 decoder layer的一次MHA QKV projection为例）：
```
// 单层Transformer Decoder Block的GEMV offload
// Weight矩阵W_qkv: [d_model, 3*d_head*n_head] 驻留在HBM bank中
// Activation向量x: [1, d_model] 在channel global buffer中

// Step 1: TPM将GEMV操作编码为PIM请求
for each HBM channel in parallel:
    pim_request = {
        op: GEMV,
        weight_addr: W_qkv_channel_partition,    // 该channel管理的weight片段
        act_vector: x,                            // 从global buffer广播
        output_buffer: channel_partial_sum_addr   // 部分和写回地址
    }
    send_to_channel(channel_id, pim_request)

// Step 2: Channel内Bank级并行执行 (每个BPM)
for each bank in channel in parallel:
    // BPM从bank row buffer读weight片段
    weight_part = bank_row_buffer[weight_addr]    // 已在之前ACT中加载
    act_part = channel_global_buffer[act_offset]   // activation对应片段
    // 向量内积计算 (2个向量内积单元)
    partial_sum_0 = 0; partial_sum_1 = 0
    for i in 0..vec_len/2 PARALLEL(32 + 32):     // 2个向量内积单元
        partial_sum_0 += weight_part[i] * act_part[i]          // Unit 0
        partial_sum_1 += weight_part[i+vec_len/2] * act_part[i+vec_len/2] // Unit 1

    // 写回部分和
    write_partial_sum(partial_sum_0 + partial_sum_1)

// Step 3: 层次化部分和聚合
    // 同channel内bank间聚合 → channel内聚合
    // 跨channel聚合 (if tensor parallelism, 在host或designated HBM上)
    final_gemv_result = aggregate_all_channels()

// Step 4: 非PIM操作在TPM执行
    // LayerNorm, Softmax等 → Vector Unit (VLIW processors)
    // GEMM (prefill阶段) → Matrix Unit (Systolic Array)
```

关键调度原则：
- **操作分类offload**：GEMV → BPM (低AI, memory-bound)；GEMM → TPM Systolic Array (高AI, compute-bound)；Element-wise/Norm → TPM VLIW (中等AI, 不适合memory offload)。
- **并行度**：HBM channel级并行 (8 channels) × Bank级并行 (per channel, 每个BPM服务2 banks)。
- **Weight/KV组织**：采用与AttAcc相同的mapping scheme，确保weight和KV矩阵在bank间的parallel access和minimal data movement。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GEMV PIM Offloading的实现：(1) BPM硬件——每两个bank一个BPM，含2×32 FP16 Mult + 2×32 FP16 Adder + 8Kb buffer，总面积6.016mm² (22nm)。(2) 与GEMM的协调——BPM仅处理GEMV；GEMM由TPM的128×128 systolic array处理（systolic array对GEMM有高吞吐，但对GEMV效率低——大量PE idle）。这是HeterRAG的operation-level offloading决策的核心。(3) 使用场景——特别适合RAG的decode阶段（batch size小，单token autoregressive generation），GEMV在decode中占主导。该技术在AttAcc [ASPLOS'24]、NeuPIMs [ASPLOS'24]、Newton [MICRO'20]和TransPIM [HPCA'22]中也有类似设计，是HBM-based PIM加速LLM decode的标准做法。(4) ORCHES (MICRO'25) 在TTC reasoning场景下使用GEMV PIM offloading：每bank配备multiplier-adder tree实现的GEMV units（16 multipliers+adders per bank，2048 banks total），专门处理memory-bound的unique KV attention query和small-batch linear operators。ORCHES的创新在于用roofline model指导哪些operators分配到PIM GEMV units（T1），与传统方法固定将attention全部offload到PIM不同。

涉及论文标题：
- 49-HeterRAG- Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation.pdf
- 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration.pdf
- 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

**MNM paper补充（PIM-based MHA Score & Context with MNM Command Set）：**
在MNM架构中，GEMV PIM Offloading专门针对RALM中RETRO模型的MHA Score和Context操作（而非HeterRAG中的QKV projection），使用不同的PIM命令集和执行流程：
```
// MNM PIM-based MHA Score & Context (1 head, 1 token)
// K^T, V在HBM bank中, Q由GPU提供

// === Score: Q · K^T ===
for seg in 0..d_head/16:  // Q分段, 每段16 FP16
    PIM_WR_GB(seg, Q[seg*16 : (seg+1)*16])
    PIM_ACT_AB(K^T_row)       // 激活K^T row→PIM row buffer
    PIM_MAC_AB()              // bank-parallel dot-product
    // result[j] = Σ(Q[k] * RowBuffer[j][k]), k=0..15
    // 16 banks × 16 elems = 256 partial per command

PIM_MV_SB()  // results → logic die Softmax via global bus

// Softmax on logic die (PNM hardware):
// exp_val[i]=exp(score[i]); sum+=exp_val[i]; softmax_out[i]=exp_val[i]/sum

// === Context: softmax_out · V ===
for seg in 0..d_head/16:
    PIM_WR_GB(seg, softmax_out[seg*16 : (seg+1)*16])
    PIM_ACT_AB(V_row)
    PIM_MAC_AB()  // context[j] = Σ(softmax_out[k] * V[j][k])
// GPU reads final context output
```

MNM的GEMV PIM与HeterRAG的关键区别：(1) **目标操作** — HeterRAG offload QKV projection, MNM offload MHA Score/Context pure GEMV；(2) **Softmax** — HeterRAG在GPU执行, MNM在logic die PNM执行→消除GPU roundtrip；(3) **Dual row buffer** — MNM使PIM GEMV与PNM retrieval并发, HeterRAG无此特性；(4) **命令集** — MNM的PIM_ACT_AB/WR_GB/MAC_AB/MV_SB vs HeterRAG的BPM CXL MMIO控制。

## Dimension-Adaptive Dataflow (DAD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dimension-Adaptive Dataflow (DAD) 是S-DMA加速器中根据每层计算前预测的sparsity mask动态选择PE阵列数据流遍历顺序的运行时调度机制。核心原理：当扩散模型U-Net某层的spatial维度（H×W）稀疏率高于channel维度（C）时，采用Spatial-first (S-first) 数据流——外层循环遍历spatial位置、内层循环遍历channel，在spatial维度上粗粒度跳过零值位置；反之采用Channel-first (C-first) 数据流——外层循环遍历channel、内层循环遍历spatial位置，在channel维度上粗粒度跳过零值通道。DAD的关键优势是最大化zero-skip粒度——维度自适应的粗粒度skip（跳过整channel或整spatial位置）比GPU warp-level的细粒度per-element check更高效，因为避免了warp divergence（同一warp内不同thread遇到不同zero pattern→部分thread idle但无法整体跳过warp）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DAD在S-DMA加速器中的kernel调度伪代码（以U-Net某一Conv层为例）：
```
// SAP已预测并存储sparsity mask M ∈ {0,1}^{C×H×W}
// M[c][h][w] = 0: 该位置为零，可跳过
// M[c][h][w] = 1: 该位置非零，需计算

// DAD Controller: 统计sparsity ratio
spatial_sparsity = 0   // 统计每个spatial位置的总体稀疏度
channel_sparsity = 0   // 统计每个channel的总体稀疏度
for c in 0..C-1:
    for h in 0..H-1:
        for w in 0..W-1:
            spatial_sparsity += M[c][h][w]
            channel_sparsity += M[c][h][w]

// 归一化为稀疏率
spatial_sparsity_ratio = 1.0 - spatial_sparsity / (C * H * W)
channel_sparsity_ratio = 1.0 - channel_sparsity / (C * H * W)

// 模式选择
if spatial_sparsity > channel_sparsity:
    mode = S_FIRST     // 外层遍历spatial位置，可跳过整位置
else:
    mode = C_FIRST     // 外层遍历channel，可跳过整channel

// Router配置：按选定mode重配置PE间互联路径
configure_router(mode)

// PE Array执行：按选定数据流遍历
if mode == C_FIRST:
    // Channel-first: 外层遍历channel
    for c in 0..C-1:
        if channel_all_zero(M, c):   // 整channel全零→跳过整个channel
            skip_channel(c)
            continue
        for h in 0..H-1:
            for w in 0..W-1:
                if M[c][h][w] == 0:
                    gate_pe(h, w)    // gating: skip this MAC
                else:
                    pe_array[h][w].mac(activation[h][w], weight[c][h][w])
else:
    // Spatial-first: 外层遍历spatial位置
    for h in 0..H-1:
        for w in 0..W-1:
            if spatial_all_zero(M, h, w):  // 该位置所有channel全零→跳过
                skip_position(h, w)
                continue
            for c in 0..C-1:
                if M[c][h][w] == 0:
                    gate_pe(c)         // gating: skip this channel
                else:
                    pe_array[h][w].mac(activation[h][w], weight[c][h][w])
```

DAD调度的关键决策点：
1. **Sparsity统计**：通过sparsity mask buffer读取per-layer的M，O(C×H×W)统计两个维度的稀疏率。统计可硬件并行化（per-row/col accumulator）。
2. **模式选择**：简单的comparator逻辑（spatial_sparsity > channel_sparsity → S-first），无复杂启发式。
3. **Router重配置**：切换数据流模式需重配置PE间互联路径——activation broadcast方向（S-first: spatial维度broadcast；C-first: channel维度broadcast）、partial sum reduction direction、weight fetch pattern。Reconfiguration overhead为固定cycle数（数个cycle），远小于每层计算时间（数百至数千cycle）。
4. **每层独立调度**：U-Net不同层具有不同稀疏性分布（downsample层spatial size小但channel多→可能C-first更优；upsample层spatial size大→可能S-first更优），DAD在每层计算前重新评估并切换。
5. **协同SAP**：SAP的预测准确率直接影响DAD调度有效性——若SAP false negative率高（预测为零但实际非零），DAD会选择不当的skip策略导致精度损失。

术语一般如何实现？如何使用？
DAD的硬件实现：(1) DAD Controller——硬件状态机实现，含sparsity ratio accumulator（per-dimension计数器）→comparator→mode select signal→router configuration signal。(2) Configurable Router——PE间可配置互联网络（类似Benes network或crossbar），支持两种数据流模式的activation broadcast和partial sum reduction路径切换。Router的配置由mode select signal控制，切换延迟为固定clock cycle数。(3) PE Gating Logic——每个PE含zero-skip gating电路，接收sparsity mask的对应bit→若为0则gating关闭PE的MAC单元时钟/电源→节省动态功耗同时释放计算带宽用于处理下一个非零位置。(4) 使用方式——DAD在扩散模型每步去噪的每个U-Net层自动运行：SAP预测mask→DAD统计选择→Router配置→PE执行。DAD对扩散模型特有的适配：不同timestep的同一层可能有不同的sparsity分布→DAD在每个timestep/layer组合独立决策。DAD的overhead（统计+配置时间）占总执行时间的比例应<1%，否则自适应收益被overhead抵消。

涉及论文标题：
- 50-S-DMA- Sparse Diffusion Models Acceleration via Spatiality-Aware Prediction and Dimension-Adaptive Dataflow..pdf

## Pipeline-Derived Execution States for DNN Scheduling (基于流水线衍生的执行状态用于DNN调度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pipeline-Derived Execution States是Crane提出的用于表达DNN层间调度行为的核心抽象机制。其洞察是：对N个子block的composite block，若按canonical pipeline模式调度，会产生2N-1个distinct execution states，这些states天然覆盖了sequential、parallel及任意混合执行模式中所有可能的active layer组合。具体而言，state-i（i∈[1,2N-1]）的活跃子block集J_i定义为：当i≤N时，J_i = {B_1, B_2, ..., B_i}（pipeline启动阶段，逐个子block激活）；当i>N时，J_i = {B_{i-N+1}, ..., B_N}（pipeline稳定+排空阶段，尾端子block逐出）。每个state-i有一个state workload s_i ∈ R≥0，表示该state处理的sub-batch数量。通过灵活分配s_i值，pipeline-derived states可退化表达为：sequential（s_1 = s_{N+1} = BS/B_sub，其余为0）、parallel（s_1 = s_N = BS/B_sub，其他为0）、标准pipeline（所有s_i > 0均匀分布）、或优化的混合模式。

从kernel调度角度拆解：
Pipeline-Derived Execution States在Crane中的调度伪代码：

```
# 对composite block B (N个子block B_1,...,B_N)
# 生成2N-1个pipeline states
S_B = {1, 2, ..., 2N-1}

# 每state确定活跃子block集
for i in 1..2N-1:
    if i <= N:
        J_i = {B_1, B_2, ..., B_i}       # pipeline fill阶段
    else:
        J_i = {B_{i-N+1}, ..., B_N}     # pipeline drain阶段

# MILP优化分配每state的workload s_i
# 满足: Σ_{i in A_ℓ} s_i = BS/B_sub  (每layer处理全batch)

# 每state内的tile分配和执行:
for each state i:
    # 活跃子block按计算量比例分配tile
    total_workload_i = Σ_{j in J_i} Workload_j
    for each active sub-block B_j in J_i:
        T_{i,j} = T × Workload_j / total_workload_i  # tile分配
    
    # 并行执行活跃子block
    parallel for each active B_j:
        # 处理 s_i 个sub-batch
        for sub_batch in 1..s_i:
            # PE array执行compute
            L_comp = Workload_j / (u_{i,j} × T_{i,j} × P)
            # NoC transfer (若B_j依赖其他block的输出)
            L_traffic = Dep_data × V_m / BW_NoC
            # SRAM/DRAM memory management per ScT和MeT
```

具体例子——3-layer模型(B_1, B_2, B_3), BS/B_sub=3：
- **Pipeline模式**：State-1(B_1 only)→State-2(B_1+B_2)→State-3(B_1+B_2+B_3)→State-4(B_2+B_3)→State-5(B_3 only)。每state处理1个sub-batch(s_1=s_2=s_3=s_4=s_5=1)。
- **Sequential退化**：s_1=3 (B_1 only)→s_2=s_3=0→s_4=3 (B_2 only)→s_5=0。实际退化为B_1完全执行→B_2→B_3。
- **混合模式**(Crane may discover)：s_1=2 (B_1 processes 2 sub-batches alone)→s_2=1 (B_1+B_2 each 1 sub-batch)→s_3=0→s_4=2 (B_2+B_3 each 1 sub-batch)→s_5=1 (B_3 alone)。这种schedule是SET的rigid ratio-tree永远无法生成的。

术语一般如何实现？如何使用？
Pipeline-derived execution states在Crane中作为MILP的state space实现：(1) N较小的block直接枚举所有2N-1个states；(2) 每state的活跃子block集J_i由pipeline定义预先确定；(3) s_i通过MILP求解确定具体值；(4) 对fan-out/fan-in DAG结构，Theorem 1证明通过递归层级化可以表达任何拓扑的任意调度。与现有方法的关系：SET的ratio-tree也导出states，但受限于repeated pattern across same node的约束（sub-batch处理顺序强制绑定batch-level pattern）；Crane的pipeline-derived states + 灵活s_i分配完全解耦了这一绑定。

涉及论文标题：
- 51-Crane- Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture..pdf

## Sub-batch Scheduling and Batch Splitting for DNN Accelerators (子批次调度与批次分割)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sub-batch Scheduling（子批次调度）是DNN加速器调度中的一项技术：将原本的完整batch BS划分为更小的sub-batch B_sub逐块处理，以降低单次执行的内存容量需求，同时为layer fusion创造更多机会。在Crane中，Batch Splitting (B)是inter-layer scheduling的四大design factors之一。Sub-batch的核心trade-off：较小的sub-batch→更好的on-chip buffer utilization（每sub-batch需SRAM buffer更小）→更aggressive的layer fusion→降低DRAM访问；但同时→lower PE utilization（tile利用率下降，因小tile分配导致MAC利用率不足）→增加weight reload overhead（同一weight需为多个sub-batch重复加载）。

从kernel调度角度拆解：
Sub-batch scheduling在Crane中的执行过程（以ResNet-50 BS=64为例）：

```
# Batch Splitting调度流程
B_sub候选集 = {1, 2, 4, 8, 16, 32, 64}  # BS的因子
num_sub_batches = BS / B_sub            # 需处理的sub-batch数

for each B_sub in 候选集:
    # 1. Intra-layer exploration per sub-batch
    for each layer ℓ:
        # 将B_sub的4维映射到T tile grid
        for each tiling (k1,k2,k3,k4):
            u_ℓ = Π_q dim_q / (ceil(dim_q/k_q) × k_q)   # tile utilization
            L_comp_ℓ = Workload_ℓ[B_sub] / (u_ℓ × T × P)
            V_ℓ = intermediate_tensor_size[B_sub]         # per-sub-batch output size
    
    # 2. MILP构建 (ScT, MeT维度: 2N-1 × N)
    # MeT容量约束与B_sub相关:
    # Σ_j (ScT[i,j] - MeT^S[i,j]) × V_j[B_sub] ≤ Cap_S
    
    # 3. MILP求解最优{ScT, MeT, s_i}
    # 目标: min EDP
    
    # 4. 按Cost_comp和Cost_traffic排序选择Top-K候选

# 最终选择min EDP的B_sub及对应schedule
```

Sub-batch对memory层次的具体影响：
- **SRAM-level**：每sub-batch的activation fit更可能fit在tile SRAM中，降低DRAM spill。例如ResNet-50某层输出tensor为H×W×C×4B，BS=64时整个batch输出=64×H×W×C×4B可能溢出1MB SRAM；B_sub=16时per-sub-batch=16×H×W×C×4B可fit→允许fusion。
- **DRAM-level**：B_sub影响DRAM中需同时存活的activation总量。Smaller B_sub→更少sub-batch同时active→更低DRAM容量需求。对于training，这一点尤为关键（activation需保存到backward）。
- **PE utilization**：B_sub=1时每个tile上仅1个sub-batch的workload mapped→tile内tiling factor选择受限→u_ℓ低（可能<50%）。B_sub=BS/B_sub较大时→更多parallelism→u_ℓ高（接近100%）。

术语一般如何实现？如何使用？
Sub-batch scheduling的实现方式：
- **Crane**：B_sub作为外层枚举参数（候选集为BS的因子），每种B_sub构建独立MILP求解，top-level block的优化中保留Top-K1/K2候选沿不同度量（Cost_comp, Cost_traffic），通过层级化refine最终收敛到最优B_sub。
- **SET**：支持B但受限于rigid batch-level constraint——sub-batch处理顺序强制与batch-level execution pattern绑定。
- **MBS (Mini-Batch Serialization)** [Lym et al., MLSys 2019]：training专用，将activation沿batch维度切分后逐sub-batch处理，sequential execution pattern。sub-batch间通过SRAM保留前层输出实现layer fusion。
- **Gpipe** [Huang et al., NeurIPS 2019]：将micro-batch思想应用于分布式训练pipeline parallelism——每个micro-batch完成forward后立即触发backward，减少activation stall时间。
- **实际使用**：Sub-batch size通常选择为BS的因子或接近因子；对于training，B_sub越小越容易fit in DRAM但因recomputation可能需要更多compute（丢弃的activation需重算）；Crane的MILP联合优化B_sub与E/F/R以找到全局最优trade-off。

涉及论文标题：
- 51-Crane- Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture..pdf

## SPU Four-Stage Pipeline for PIM State Update

术语解释
SPU Pipeline 是 Pimba 在 PIM 内部执行的 state update 操作的 4 阶段流水线，将 state decay、outer product、update 和 output GEMV 四个操作流水化，每 iteration 处理一个 state sub-chunk，配合 access interleaving 实现连续无气泡执行。

术语是什么？
SPU Pipeline 包含四个阶段：(1) **Stage 1 — Fetch**：从 DRAM row buffer 读取一个 state sub-chunk（16 个 MX8 元素，256-bit column 宽度）到 SPE 寄存器。(2) **Stage 2 — Decay + Outer Product（并行）**：MX Multiplier 同时执行 S_decay[i] = MX_mul(d_t[k], S_reg[i]) 和 outer[i] = MX_mul(k_t[k], v_t[j])，两者无数据依赖可并行。(3) **Stage 3 — Update**：MX Adder 执行 S_new[i] = MX_add(S_decay[i], outer[i])。(4) **Stage 4 — Dot Product + Writeback**：Dot Product Unit 累加 S_new[i]·q_t[k] 到 accumulator，同时将 S_new[0:15] 写回 DRAM row buffer。Pipeline 深度为 4，fill/drain 各需 3 iterations。稳定状态下每 iteration 产出一个 sub-chunk 的完整 state update + partial dot product。

从kernel调度角度拆解：
单 head state update 的 SPU pipeline 执行伪代码：
```
// 输入: S_prev 在 HBM DRAM 中, d_t/k_t/q_t 在 SPU 寄存器, v_t[j] 按 sub-chunk 索引
acc = 0  // dot product accumulator
for chunk in chunk_group:
    q_reg = q_t[chunk.head_range]
    for sub_iter in 0..N_sub:
        bank = (sub_iter % 2 == 0) ? UPPER : BOTTOM
        
        // Stage 1: Read sub-chunk (tCCD_L)
        S_reg = bank.row_buffer.read(chunk.col_offset + sub_iter * COL_WIDTH)
        
        // Stage 2: Parallel decay + outer product (MX MUL, 1 cycle per 16 elements)
        for i in 0..15 PARALLEL:
            S_decay[i] = MX_MUL(d_t[i], S_reg[i])
            outer[i]    = MX_MUL(k_t[i], v_t[chunk.v_offset + sub_iter])
        
        // Stage 3: Update (MX ADD, 1 cycle)
        for i in 0..15 PARALLEL:
            S_new[i] = MX_ADD(S_decay[i], outer[i])
        
        // Stage 4: Dot product + Writeback
        acc += Σ_{i=0}^{15} DOT(S_new[i], q_reg[i])
        bank.row_buffer.write(chunk.col_offset + sub_iter * COL_WIDTH, S_new)
```
Pipeline 关键约束：COMP 命令连续 issue 需遵守 tCCD_L（每 column read 4 memory bus cycles）。SPU 频率 = memory_bus_freq / tCCD_L。

术语一般如何实现？如何使用？
- 在 Pimba cycle-accurate simulator（基于 Ramulator2）中建模：每 sub-chunk iteration 延迟 = max(stage_latency) × pipeline_stages = 4 cycles（稳态）。Total latency = (N_sub_chunks + 3) × 4 cycles。
- 与 time-multiplexed PIM 对比：无 pipeline，四个操作顺序执行，state update latency 约 4× longer（面积更小）。
- Pipeline 与 access interleaving 协同：Stage 1 bank 选择和 Stage 4 writeback bank 由 SPU controller 每 iteration 自动交替切换。

涉及论文标题：
- 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

## Chunk Group State Data Layout for PIM

术语解释
Chunk Group Data Layout 是 Pimba 为 state 矩阵在 HBM DRAM 中的物理存储设计，将 state 沿 dim_head 维度切为 sub-chunk（对齐 DRAM column width），沿 dim_state 维度组织为 chunk（对齐 DRAM row size），共享 operands 的 chunk 归为 chunk group 分配至同一 bank 连续 row。

术语是什么？
数据布局的三层组织：(1) **Sub-chunk**：沿 dim_head 按 DRAM column width（256-bit = 16×MX8）切分。dim_head=128 则 8 个 sub-chunk/row。(2) **Chunk**：沿 dim_state 维度组合 sub-chunk 为对齐 row size 的 chunk。state 按 column-major 存储：同一 dim_head 范围的连续 dim_state 元素在同一 row 中。(3) **Chunk Group**：共享 d_t/k_t/q_t 的多个 chunk 归入同一 group，分配至同一 bank 连续 row。每 chunk 仅需不同的 v_t 元素（如 dim_state=128 需 128 个 v_t 值）。优化效果：(a) 共享 operands 每 chunk group 仅传输一次（非每 chunk），大幅减少 REG_WRITE 数据量；(b) 连续 row 减少 ACT 开销；(c) SPU 通过 sub-chunk 迭代逐列读取，利用 DRAM burst 特性。

从kernel调度角度拆解：
State 数据布局的地址映射：
```
// State S[dim_head][dim_state] in MX8
// DRAM column = 256-bit = 16 × MX8 values

N_sub_per_chunk = dim_state 内的 sub-chunk 数（由 row_size 决定）
N_group = dim_head / 16  // 每 16 个 head element 为一个 chunk group

for each chunk_group g in 0..N_group:
    bank = g % num_SPU_groups
    for each row r in 0..(dim_state/sub_per_chunk):
        // 存储 S[g*16 : g*16+15][r*N_sub : (r+1)*N_sub]
        DRAM[bank][g * rows_per_group + r][0:256-1] = pack_MX8(S[g*16:g*16+15][r*N_sub...])
```
KV cache 采用类似布局，支持 attention score/attend 模式的 PIM 访问。

术语一般如何实现？如何使用？
- Pimba compiler/runtime 根据 model dims 和 HBM 几何参数在编译时计算 chunk group 分配；GPU kernel 根据 layout 计算 PIM 操作地址。
- Operand 复用：per chunk group 仅传一次 d_t/k_t/q_t（dim_head-size vector），per chunk 仅传 v_t 的 1 个元素。
- 与传统 row/column-major 对比：为 PIM bank 级并行和 operand 复用专门设计，不适用于 GPU（无 bank constraint）。

涉及论文标题：
- 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

## Custom DRAM Command Scheduling with Transfer Overlap

术语解释
Pimba 的 Custom DRAM Command Scheduling 通过在 ACT4 的 tFAW 空闲间隙插入 REG_WRITE、在 PRECHARGES 的 tRP 间隙插入 RESULT_READ，实现 PIM 计算与 host-PIM 数据传输的重叠，最小化 offloading 通信 overhead。

术语是什么？
Pimba 定义了 5 条 custom DRAM 命令：(1) **ACT4**：同时激活 4 个 bank 的 target row，受 tFAW 约束。(2) **REG_WRITE**：host GPU 将 operands 以 MX8 格式写入 SPU 寄存器。(3) **COMP**：触发 SPU pipeline 计算，连续 issue 遵守 tCCD_L（4 cycles）。(4) **RESULT_READ**：从 SPU accumulator 取回 partial sums，需等待 tRTP+tWR。(5) **PRECHARGES**：所有 bank 的 row buffer precharge，耗时 tRP。Command Scheduling 的关键优化：在 ACT4 之间的 tFAW 间隙插入 REG_WRITE（data bus 空闲）；在 PRECHARGES 的 tRP 期间插入 RESULT_READ（precharge 不占用 data bus）。所有命令时序确定——GPU 可连续发射无需同步，仅在需要结果时阻塞。

从kernel调度角度拆解：
Pimba Command Sequence 时序图（一个 chunk group）：
```
Time →
C/A Bus:  [ACT4_0]--tFAW--[ACT4_1]--tFAW--...--[COMP_0]-tCCD_L-[COMP_1]-...--[PRECHARGES]
Data Bus: .....[REG_WRITE_0].....[REG_WRITE_1]..........................[RESULT_READ].......
PIM:      ......|--SPU pipeline (COMP_0)--|--SPU pipeline (COMP_1)--|......|--writeback--|

// REG_WRITE 在 tFAW 间隙 → 与 activation 重叠
// RESULT_READ 在 tRP 期间 → 与 precharge 重叠
// COMP 连续 issue 遵守 tCCD_L = 4 cycles/iteration
```
总 COMP cycles = N_sub_chunks × tCCD_L = (dim_state × dim_head / (16 × chunk_size)) × 4。考虑 pipeline fill/drain：total = (N_sub_chunks + 3) × 4。

术语一般如何实现？如何使用？
- GPU kernel 根据预计算 schedule（确定性时序）issue 命令序列，编译时验证满足所有 DRAM 时序约束（tFAW/tCCD_L/tRP/tRTP/tWR）。
- 与标准 DRAM 兼容：5 条命令扩展自标准 DRAM 命令集，不改变基础 DRAM 状态机。ACT4 = 4×ACT 批处理；COMP 复用 column read I/O gating 时序。
- 通信 overhead：通过传输-计算重叠，REG_WRITE 和 RESULT_READ 的 data bus 占用几乎完全被隐藏（>90% overlap rate for large models）。

涉及论文标题：
- 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

## PNM-based IVF-PQ Retrieval Offloading

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PNM-based IVF-PQ Retrieval Offloading是将FAISS IVF-PQ向量检索的memory-bound且GPU-unfriendly kernel（PQ code scan和top-k selection）从GPU SM offload到HBM logic die的PNM（Processing Near Memory）专用硬件上执行的技术。GPU仅保留cluster selection（计算||x-y_c||²选top nprobe clusters，compute-bound GEMV适合GPU tensor core）和最终的multi-cluster top-k merge。核心offload的两个kernel：(1) PQ code scan — 对每个cluster中所有PQ codeword计算L2距离，含codebook lookup、residual dot-product（x·y_R）、预计算LUT加和；(2) Top-k selection — 从cluster内所有计算出的距离中选出最小的k个。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PNM-based IVF-PQ的kernel执行伪代码（以1个cluster为例）：
```
// === GPU side: Cluster Selection (保留在GPU) ===
for each cluster c in 0..nlist-1:
    dist_to_centroid[c] = ||x - y_c||^2  // GPU GEMV, batched over dimensions

selected_clusters = topk(dist_to_centroid, nprobe)  // nprobe=256 for Config 5

// === PNM side: per-cluster retrieval (offload到PNM) ===
for each cluster c in selected_clusters:
    // Step 1: GPU→PNM setup (via MNM Controller/DMA)
    // PNM_WR_MMIO: write ||x-y_c||^2 to Query Register
    // PNM_WR_MMIO: write precomputed LUT[c] to PNM SRAM  
    //    (LUT[c][codebook_idx][subvec] = ||y_R_sub||^2 + 2(y_c_sub · y_R_sub))
    //    size: ksub × M = 256 × 64 = 16K FP16 entries
    
    // Step 2: PNM_RET_INIT — 发起PNM retrieval
    // PQ code scanner hardware执行:
    
    for each batch of 64 codewords loaded from HBM:
        // 16 parallel scanners, each processes different codewords
        for scanner_id in 0..15:  // 16 scanners parallel
            // Load 64 PQ indices from HBM via TSV
            indices[0..63] = load_pq_codewords_from_hbm()  // 64 × 1B indices
            
            for each idx in indices:
                // Reconstruct residual vector from PQ codebook
                y_R = [0] * dim  // 384-dim FP16
                for subvec_id in 0..M-1:  // M=64 sub-vectors
                    code_id = pq_codeword[idx][subvec_id]  // 8-bit index
                    y_R[subvec_id * dsub : (subvec_id+1) * dsub] = 
                        PQ_codebook[subvec_id][code_id]  // dsub = dim/M elements
                
                // Compute x · y_R (24 MAC units, 384-dim dot-product)
                dot_product = 0
                for i in 0..383:  // parallel across 24 MACs
                    dot_product += x[i] * y_R[i]
                
                // Lookup precomputed value
                lut_val = LUT[idx]  // ||y_R||^2 + 2(y_c · y_R)
                
                // Final distance
                distance = query_reg.centroid_dist  // ||x-y_c||^2
                         + lut_val
                         - 2 * dot_product
                
                distances.append((idx, distance))
    
    // Step 3: Top-k selection in PNM hardware
    // Full sorter: sort 16 new distances (from 16 scanners)
    sorted_16 = odd_even_merge_sort(distances[0..15])
    
    // Partial sorter: merge sorted_16 with top-k register (16 items)
    merged_32 = merge(sorted_16, topk_register)
    topk_register = odd_even_merge_sort(merged_32)[0..15]  // keep top 16
    
// === GPU side: Final merge ===
global_topk = merge_all_stacks(topk_registers_from_6_MNM_stacks)
final_topk = global_topk[0..k-1]  // k=16 for Config 5
```

关键kernel调度要点：
- **Data movement**：PQ codewords从core die DRAM bank读→sense amp→global data bus→TSV→logic die PQ code scanner（每nCCDL=4 cycles传输1024B）。无需将codewords传输到GPU→节省GPU off-chip memory bandwidth。
- **Concurrency**：16个PQ code scanner独立并行工作，每个处理不同的codeword batch。scanner内24个MAC units并行计算384-dim dot-product（24×16=384）。
- **LUT access pattern vs GPU**：GPU FAISS IVF-PQ的LUT访问受限于shared memory容量（132KB/SM < 16K entries × 2B = 32KB per cluster，但多cluster并存时超出）→频繁global memory access。PNM的dedicated SRAM LUT（per cluster单次加载）消除了此bottleneck。
- **Sorting**：GPU上top-k sorting通常用warp-level bitonic sort或atomicCAS→irregular memory access和divergent branches→低SM utilization。PNM的Odd-Even Merge Sort network [6]固定pipeline，每cycle compare-and-swap一对→deterministic latency。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **硬件实现**：ASAP 7nm PDK合成，16个PQ code scanner（650MHz），每scanner含24个FP16 MAC units = 384 MAC total per scanner × 16 = 6144 MAC units total per MNM stack。Top-k selector含Full sorter（16-input odd-even network, 4 stages）、Partial sorter（32-input merge, 5 stages）、Top-k register（16× FP16 distance + 16× ID）。
- **FAISS兼容性**：PNM offloading的IVF-PQ算法与FAISS-GPU [37]接口兼容——输入：query vector x, cluster centroid table, PQ codebook, precomputed LUT, PQ codewords；输出：top-k (distance, ID) pairs。GPU cluster selection的输出（selected cluster list + ||x-y_c||^2）作为PNM input。
- **ID remapping优化**：原始FAISS IVF中每72B chunk（64B PQ codeword + 8B unique ID）与HBM3 32B access granularity不对齐→MNM在preprocessing阶段按sequential order重分配codeword IDs→host maintain ID mapping table→PNM仅处理64B-aligned codewords→GPU读回reordered IDs后通过mapping table还原原始data IDs。
- **Energy benefit**：PNM offloading将GPU上的LUT access (global memory, ~100pJ/bit) 替换为logic die SRAM access (~1pJ/bit)→per distance calculation energy显著降低。同时减少GPU↔HBM的data movement→PHY switching power降低。

涉及论文标题：
- 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration.pdf

## PIM+PNM Concurrent Execution (Dual Row Buffer-Based)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM+PNM Concurrent Execution是利用HBM3 core die的dual row buffer实现PIM-based GEMV计算（MHA Score/Context）和PNM-based memory read（IVF-PQ codeword loading）在同一bank上同时进行的并发执行机制。标准HBM每bank仅1条row buffer→PIM MAC操作占用row buffer期间，任何memory read/write（包括PNM需要的PQ codeword读取）必须等待→形成sequential bottleneck。Dual row buffer通过在每bank物理上设置2条独立row buffer（各带独立sense amp和I/O gating），使PIM和PNM可同时访问同一bank的不同row→实现generation和retrieval的物理并发。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dual row buffer并发执行的schedule伪代码：
```
// MNM Controller schedule — 每HBM channel独立执行
// 假设：HBM channel ch0有16 banks (BA0-BA15)

// === Concurrent Phase: PIM generation + PNM retrieval ===
// Timeline (per HBM cycle):
// cycle 0-3: PIM_WR_GB → write Q segment to global vector buffer
//            (uses global bus for write, does NOT conflict with row buffers)

// cycle 4:   PIM_ACT_AB → activate row R1 for K^T data
//            load K^T column into Row Buffer A (dedicated PIM row buffer)
//   simultaneously:
//            PNM read command → activate row R2 for PQ codewords
//            load PQ codeword data into Row Buffer B (dedicated PNM row buffer)
//   buffer_select = {PIM→A, PNM→B}  // no conflict

// cycle 5-20: PIM_MAC_AB → MAC units compute:
//             dot_product = Σ(Q[i] * RowBufferA[i]), i=0..15
//             // 16 cycles for 256-dim partial, memory bandwidth bound
//   simultaneously (cycle 5-8):
//             PNM read → Row Buffer B data sent to logic die via TSV
//             // 4 cycles to transfer 1024B PQ codewords (nCCDL=4)

// cycle 21:  PIM_MV_SB → MAC results move to score buffer
//            then to logic die Softmax calculator via global bus
//   simultaneously:
//            Next PNM read batch starts (Row Buffer B reloading)

// cycle 22-37: Softmax calculator on logic die computes:
//              softmax_out[i] = exp(score[i]) / Σ exp(score[j])
//              // latency = pipeline depth of exp+div units

// cycle 38-54: Softmax_out written back to global vector buffer
//              PIM_MAC_AB → Context operation: softmax_out · V
//              // 16 cycles, similar to Score operation
//   simultaneously:
//              PNM continues loading next PQ codeword batch
//              Top-k sorter processes previous batch results

// === Non-concurrent phases (when PNM not active) ===
// When no retrieval is pending (all requests in GENERATING state):
//   PIM uses either Row Buffer A or B
//   Other row buffer idle → standard DRAM mode
```

关键并发细节：
- **Row buffer allocation**：Row Buffer A dedicated to PIM operations（激活row→MAC unit reads）；Row Buffer B dedicated to PNM/standard I/O operations（激活另一row→数据通过global bus/TSV传输）。Buffer select信号由MNM Controller根据当前command type动态产生。
- **Bank conflict avoidance**：PIM和PNM在并发时必须访问不同row（不同row address）→若PIM和PNM操作恰好指向同一row→需serialize（但正常情况下PIM访问K^T/V weight rows，PNM访问PQ codeword rows→不同address range→极少冲突）。
- **Global bus sharing**：PIM_MV_SB和PNM的data transfer共享global data bus→在bus level可能需要arbitration。MNM通过time-multiplexing解决：PIM结果在compute完成后burst transfer→PNM在interleaved cycle传输codewords。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **硬件cost**：Dual row buffer增加3.8mm² per core die（vs standard single row buffer的baseline）≈ 4.2% core die area overhead。Power overhead: 98.2mW per core die（active standby + data transfer energy [49,78]）。这是MNM vs AttAcc的主要面积差异（AttAcc [80] 10.8% vs MNM 15.0% core die overhead）。
- **对比time-multiplexed方案**：无dual row buffer时，PIM和PNM必须time-multiplex bank access（如先完成PIM MAC→释放row buffer→PNM读取→PNM完成后PIM恢复）→introduce idle cycles on both sides。Selective batching+Early Gen的调度增益在不具备dual row buffer时将大幅缩减（因为retrieval和generation的真实物理并发被serialization取代）。
- **对比NeuPIM [24] dual row buffer**：NeuPIM也使用dual row buffer但目的不同——用于NPU和PIM交替访问同一bank的不同row以支持sub-batch interleaving。MNM的dual row buffer服务于PIM+PNM并发而非PIM+NPU并发→两者的target concurrency pair不同但底层机制相同。
- **Limitation**：当PIM MAC操作和PNM read同时访问同一bank→row buffer可并发但sense amp共享→仍需serialize activation（但不同row→不同sense amp group→可真正并发）。Dual row buffer不解决同一row的concurrent access冲突（不属于典型workload pattern）。

涉及论文标题：
- 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration.pdf

## CAM-based Fast Match for Bit-Slice Repetition Detection

术语是什么？
CAM-based Fast Match是MCBP加速器中用于在单周期内识别bit-slice Group Matrix中重复列向量的硬件加速技术。采用512B Content Addressable Memory (CAM)按group size m=4组织：4-bit列向量的高2位(HO)和低2位(LO)分别管理在不同CAM bank中。Search时，对每种可能的4-bit search key（0000到1111），HO两bit查询MSB bank、LO两bit查询LSB bank→读出的bitmap按位AND→生成最终bitmap标记匹配列位置。例如search key=0001→HO读地址'00'、LO读地址'01'→AND得bitmap='1001'（表示x0和x3匹配）。CAM对search key=4'b0000实行clock-gating省电。Basic block为2-bit匹配粒度，可通过re-matching多个basic block的输出适应不同group size。

从kernel调度角度拆解术语：
CAM在MCBP GEMM dataflow中的操作流程：
```
# 输入: Decompressed Group Matrix G (m=4, each column is 4-bit)
# 输出: Per-search-key bitmap 标记哪些activation列需merge

# Step 1: Address orchestration for decompressed weight
for col in 0..H-1:
    ho = G[col][3:2]  # Higher-order 2 bits
    lo = G[col][1:0]  # Lower-order 2 bits
    MSB_bank[ho].append(col)  # Store col index at HO address
    LSB_bank[lo].append(col)  # Store col index at LO address

# Step 2: CAM search (枚举所有16种search key)
for search_key in 0..15:
    ho = search_key >> 2
    lo = search_key & 0x3
    ho_row = MSB_bank.read(ho)  # 单cycle, returns bitmap of H bits
    lo_row = LSB_bank.read(lo)  # 单cycle
    bitmap = ho_row & lo_row   # AND: both HO and LO must match
    # bitmap[j]=1 → column j matches search_key
    results.append((search_key, bitmap))

# Step 3: Index conversion
for (search_key, bitmap) in results:
    if search_key == 0: continue  # clock-gated, save power
    indices = bitmap_to_indices(bitmap)  # 16 Index Converters parallel
    # indices指向需merge到MAV[search_key]的activation位置
```

术语一般如何实现？如何使用？
CAM单元在MCBP中每PE Cluster一组（512B），21个PE Clusters共享。Index Converter（16个/PE Cluster）将bitmap转换为activation SRAM地址→Data Fetcher按地址取activation→送入AMU merge。CAM的面积为BRCR Unit的~25%，功耗占~47%，但BRCR整体使net area减少45%、net power减少72%。CAM basic block为2-bit匹配粒度，可通过级联支持m=2/3/4/5等不同group size配置。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## Merge-and-Reconstruct Dataflow for Bit-Grained GEMM

术语是什么？
Merge-and-Reconstruct是MCBP BRCR的bit-grained GEMM dataflow，将传统GEMV的H次乘加运算分解为两个阶段：(1) Merge阶段——识别Group Matrix中重复列向量，将对应activation累加到Merged Activation Vector (MAV)，复杂度H×(1-bs̃)次加法（与m无关）；(2) Reconstruct阶段——用Enumeration Matrix（固定pattern的bit matrix，记录每行含哪些列类型）乘以MAV恢复m行GEMV结果，复杂度m×2^{m-1}次加法。该dataflow的核心trick：Enumeration Matrix format是固定的（因H>>2^m,可假设所有2^m种列向量都出现），因此Reconstruct的数据路径可完全固定化——每个adder的输入register是预绑定的。

从kernel调度角度拆解术语：
MCBP PE cluster内Merge-and-Reconstruct的执行过程（m=4, H=4096）：
```
# PE Cluster: 1 CAM + 16 Index Converter + 16 AMUs + 1 RU
# Tiling: T_M=64, T_K=256, T_N=32

# === Merge Phase (16 AMUs parallel) ===
# CAM outputs 16 (search_key, bitmap) pairs per cycle
for each (search_key, bitmap) in [0001..1111]:
    indices = IndexConverter(bitmap)  # 16 converters parallel
    activations = ActivationFetcher.fetch(indices)
    # AMU: read GSB[search_key], add all fetched activations, write back
    psum = GSB.read(search_key)
    for act in activations:
        psum += act
    GSB.write(search_key, psum)

# === Reconstruct Phase (1 RU time-multiplexed over 16 AMUs) ===
# Fixed reconstruction formula (for m=4):
# y3 = z1 + z3 + z5 + z7 + z9 + z11 + z13 + z15
# y2 = z2 + z3 + z6 + z7 + z10 + z11 + z14 + z15
# y1 = z4 + z5 + z6 + z7 + z12 + z13 + z14 + z15
# y0 = z8 + z9 + z10 + z11 + z12 + z13 + z14 + z15

# RU executes in reverse order (y3→y0) to maximize data reuse:
# Computing y3 first: adder3 reads z15
# Computing y2 next: adder2 ALSO reads z15 (already in adder3 pipeline)
# Then y1: adder1 needs z7,z12-z15
# Then y0: adder0 needs z8-z15
# This reordering reduces switching activity by 75%
```

术语一般如何实现？如何使用？
MCBP PE Cluster：每cluster 8个PE并行处理同一weight tile的不同bit-slice（bit-slice parallelism）→每PE内含一个AMU set (16 AMUs per group, 多个groups time-multiplexed)。GSB (Group Sum Buffer) 存16个MAV entries (per group)。RU采用fixed datapath：adder inputs预绑定→仅需MUX选择读取哪个GSB register。RU的fixed nature使得控制逻辑极简，适合high-throughput bit-serial pipeline。对于INT8 GEMM，8 bit-slices间需inter-PE accumulation（bit-shift加和）。

涉及论文标题：
- 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

## PipeFlash (Fine-grained Pipelined Dataflow for FlashAttention-2)

术语解释
PipeFlash是HLX提出的针对FlashAttention-2的fine-grained pipelined数据流，通过将FA-2的block-level同步执行细化为row-level 4-stage流水线来隐藏non-MatMul延迟。

术语是什么？
传统FA-2按block级别同步执行：对每个Q block，遍历所有K/V blocks执行QKT→local softmax→PV→update O，这4个step存在严格顺序依赖→step 8(softmax的m/l update) 和step 10(O update) 等非MatMul操作无法与step 7(QKT) 和step 9(PV) 等MatMul操作重叠→compute utilization饱和于~61% (A100)。PipeFlash将执行粒度细化为每次处理Q block中的2行，形成4-stage pipeline：(DPE#0) QKT [2 rows, Bc] → (RVPE) local softmax [1 row] → (DPE#1) PV [2 rows, dhead] → (UpE) update O [1 row]。4个stage在时间上重叠——当某一行在DPE#1做PV时，下一行同时在RVPE做softmax，再下一行在DPE#0做QKT→non-MatMul(softmax/update O) 被MatMul(QKT/PV)完全隐藏。K和V block被同一Q block中的所有行复用。中间数据(score/prob矩阵)从128KB降至1KB (4.8× reduction)。

从kernel调度角度拆解术语：
PipeFlash在HLX URSC上的pipeline调度 (per Q block of 2 rows, per K/V block j):
```
Cycle timeline (各stage并行处理不同Q行):
DPE#0: | QKT_row0 | QKT_row1 | QKT_row0' | QKT_row1' | ...
RVPE:  | idle     | softmax0 | softmax1  | softmax0'  | ...
DPE#1: | idle     | idle     | PV_row0   | PV_row1    | ...
UpE:   | idle     | idle     | idle      | updO_row0  | ...

per Q row (以row0为例):
S = Q[row0] @ K[j]^T                 # [1, Bc]
m_new = max(m_old, rowmax(S))
P = exp(S - m_new)                   # [1, Bc]
l_new = exp(m_old - m_new)*l_old + rowsum(P)
O_new = diag(exp(m_old - m_new))^{-1}*O_old + P @ V[j]  # [1, dhead]
m_old, l_old, O_old = m_new, l_new, O_new
Finalize: O_final = diag(l_Tc)^{-1} @ O_Tc
```
Pipeline balance: QKT (MatMul [1, Bc]×[Bc, dhead]→[1, dhead]) ≈ PV (MatMul [1, Bc]×[Bc, dhead]→[1, dhead]) 相同FLOPs，各处理2 rows。Softmax (rowmax + exp + rowsum) 和Update O (element-wise scale + add) 各处理1 row（FLOPs低）。

术语一般如何实现？如何使用？
PipeFlash需要在支持fine-grained pipeline的硬件上实现（如HLX URSC）。GPU上因SIMT限制和缺乏专用pipeline path无法高效实现：FA-3的2-stage warp-specialized异步pipeline因register doubling+occupancy下降→utilization仍饱和于~61%。PipeFlash证明：更细的粒度(4-stage, row-level) + 专用硬件资源(DPE/RVPE/UpE) → 可达97.5% compute utilization，1.75× speedup over A100 FA-2, 2.78× over H100 FA-2, 1.84× over H100 FA-3。

涉及论文标题：
- 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

## PipeSSD (Fine-grained Pipelined Fused State-Space Duality)

术语解释
PipeSSD是HLX提出的首个针对Mamba-2 SSD计算的融合+流水线化数据流，将fused SSD分解为3-stage fine-grained pipeline以大幅减少中间数据和提升compute utilization。

术语是什么？
传统SSD分5个独立kernel，memory-bound（GPU compute utilization ~27%）。直接fused SSD（合并为单kernel）在GPU上因中间数据642KB/block超per-SM memory而性能退化1.74×。PipeSSD在fused SSD基础上加入fine-grained pipelining：将6个operation归入3个pipeline stage——(1st) dA预处理、(2nd) YDiag、(3rd) YOff∥statesN→YFinal+update states。处理粒度：YDiag 2 rows, statesN 4 rows, YOff 8 rows（由pipeline cycle平衡决定）。中间数据从642KB降至58.5KB (11×)，DRAM流量减少6.8×。Compute utilization达~78.4%（受限因素：Hybrid-2.7B中dhead (64)是dstate (128)的一半→未能达100%）。

从kernel调度角度拆解术语：
PipeSSD在HLX URSC上的3-stage pipeline调度 (per chunk block j):
```
1st Stage (RVPE):
sdt = softplus(dt[j] + dtbias)
dA = sdt * A
dACS = cumsum(dA)
decay_states = exp(dACS[-1:] - dACS)
d2t = decay_states * sdt

2nd Stage (DPE#0→RVPE→DPE#1): YDiag calculation
DPE#0: CBT = C[j] @ B[j]^T         # [cl, cl], 2 rows
RVPE:  CBTLdt = CBT * L * sdt      # element-wise, 1 row
DPE#1: YDiag = CBTLdt @ x[j]       # [cl, dhead], 2 rows
(YDiag stored to GS)

3rd Stage parallel:
# Path A (RVPE→DPE#0→UpE):
dCOff = exp(dACS) * C[j]           # RVPE, 8 rows
YOff = dCOff @ states(j-1)         # DPE#0, 8 rows
# Path B (RVPE→DPE#1→UpE):
dBdt = d2t * B[j]                  # RVPE, 4 rows
statesN = dBdt^T @ x[j]            # DPE#1, 4 rows
# UpE:
YFinal = YDiag + YOff              # 8 rows
states(j) = exp(dACS[-1]) * states(j-1) + statesN  # 4 rows
```
与PipeFlash pipeline对比：PipeFlash各stage处理相同行数(2 rows)因QKT=PV FLOPs；PipeSSD各stage行数不同(2/1/8/4)因CBT(MatMul [dstate, dhead]) 和CBTLdt×x(MatMul [cl, dhead]) FLOPs不同→行数按总cycle平衡分配。

术语一般如何实现？如何使用？
PipeSSD是首个将fusion和fine-grained pipelining结合用于SSD的工作。实现依赖HLX URSC的Local NoC可重构data forwarding：2nd stage(DPE#0→RVPE→DPE#1 forward path) 和3rd stage(RVPE→DPE#0 and RVPE→DPE#1 并行forwarding via MUX/DEMUX)。GPU上无法实现——即使H100 TMA支持异步搬运，fine-grained row-level forwarding无法通过GPU shared memory高效实现。PipeSSD达成4.95× speedup over H100 SSD, 2.91× over A100 SSD。

涉及论文标题：
- 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

## Fine-grained Pipelined Dataflow (for AI Accelerators)

术语解释
Fine-grained pipelined dataflow是一种将计算操作分解为细粒度（如row-level或sub-tile level）流水线stage并在专用硬件上同时执行不同stage的dataflow设计方法，解决传统block-level同步执行中non-MatMul延迟无法隐藏的问题。

术语是什么？
在AI加速器中，传统的kernel执行通常是block-level同步的：一个operation的所有数据计算完成后才开始下一个operation（如FA-2中全部QKT→全部softmax→全部PV→全部update O）。Fine-grained pipelined dataflow将数据分解为更小单元（如Q的2行、CBT的2行等），形成多个pipeline stage——每个stage由专用硬件单元处理（如DPE做MatMul, RVPE做element-wise, UpE做update）——不同数据单元的不同stage在不同硬件单元上同时执行。关键设计参数：(1) pipeline depth（stage数）、(2) per-stage data granularity（行数/元素数）、(3) pipeline balance（各stage cycle匹配）。HLX中PipeFlash是4-stage (QKT→softmax→PV→update O)，PipeSSD是3-stage (dA预处理→YDiag→YOff∥statesN+finalize)。

从kernel调度角度拆解术语：
Fine-grained pipelining在HLX URSC上的调度示例 (PipeFlash 4-stage with 2 rows/block):
```
Time →   t0    t1    t2    t3    t4    t5
DPE#0:  Q0K0  Q1K0  Q0K1  Q1K1  Q0K2  Q1K2
RVPE:   -     S0    S1    S0'   S1'   S0''
DPE#1:  -     -     P0    P1    P0'   P1'
UpE:    -     -     -     U0    U1    U0'
```
其中Q_i = Q row i, K_j = K block j, S = softmax, P = PV, U = update O。同一时间点t3：DPE#0做Q1K1（MatMul）、RVPE做S0'（第二个K/V block的softmax）、DPE#1做P0（第一个K/V block的PV）、UpE做U0（第一个K/V block的update O）。Pipeline fill/drain: 前3个和最后3个时间步有部分单元idle→但整体计算密度大幅提升。

术语一般如何实现？如何使用？
GPU上FA-3的warp-specialized pipeline是coarse-grained (2-stage)，因register pressure（per-warp register翻倍→occupancy下降）和SIMT限制（heterogeneous warps scheduling overhead）→utilization仅~61%。Fine-grained pipelining需要专用硬件：(1) 每个stage有独立计算单元（避免resource contention）、(2) stage间直接wire forwarding（避免shared memory中转延迟）、(3) per-stage可调节并行度（通过row数控制cycle balance）。HLX URSC是首个支持fine-grained pipelining的统一架构。类似概念见于SambaNova SN40L的coarse-grained fusion（520MB SRAM, 但无fine-grained pipeline depth）。PipeFlash/PipeSSD证明：fine-grained > coarse-grained → 97.5% vs 61% utilization。

涉及论文标题：
- 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

## Online Arithmetic Intensity Estimation for FC Kernel Scheduling in LLM Decoding

术语解释
PAPI提出的低开销在线kernel bottleneck预测方法：通过 runtime parallelism metrics (RLP × TLP) 直接估算FC kernel的arithmetic intensity (AI)，以决定将FC kernel调度到GPU (compute-bound) 还是PIM (memory-bound)，无需profiling或硬件计数器。

术语是什么？
Online Arithmetic Intensity Estimation for FC Kernel Scheduling 是PAPI dynamic scheduler的核心mechanism。FC kernel的arithmetic intensity可通过公式简化：AI ≈ RLP × TLP（其中RLP=Request-Level Parallelism即batch size，TLP=Token-Level Parallelism即speculation length）。推导：FC kernel的weight matrix维度为(h, h)，输入为(RLP×TLP, h)，FLOPs = RLP×TLP×h×h×2，Bytes = (2×RLP×TLP×h+h²)×2。当h很大时（如GPT-3 175B的h=12288），AI ≈ RLP×TLP。调度器将此估计值与offline确定的memory-boundedness threshold α比较：AI_est > α → FC到GPU PUs (compute-bound)；AI_est ≤ α → FC到FC-PIM (memory-bound)。

从kernel调度角度拆解：
PAPI Runtime Scheduling伪代码：
```
// === Initial Scheduling (serving start前) ===
RLP_0 = batch_size               // e.g., 64
TLP_0 = speculation_length        // e.g., 4
AI_est = RLP_0 × TLP_0           // = 256
α = offline_determined_threshold  // e.g., 32 (from iterative eval on GPU vs PIM)
if AI_est > α:
    place_FC_on_GPU_PUs()
else:
    place_FC_on_FC_PIM()

// === Runtime Scheduling (each decoding iteration后) ===
for each decoding_iteration:
    // Step 1: 收集所有request的output tokens
    tokens = gather_output_tokens(all_requests_in_batch)
    
    // Step 2: 统计<|eos|> tokens, 追踪RLP变化
    eos_count = count_eos(tokens)
    if eos_count > 0:
        RLP_current = RLP_previous - eos_count  // requests completed
    else:
        RLP_current = RLP_previous
    
    // Step 3: 检查TLP变化 (from dedicated register)
    TLP_current = read_TLP_register()
    
    // Step 4: 估算AI并比较α
    AI_est = RLP_current × TLP_current
    if (AI_est > α) and (FC_currently_on_PIM):
        reschedule_FC_to_GPU()    // memory→compute transition
    elif (AI_est <= α) and (FC_currently_on_GPU):
        reschedule_FC_to_FC_PIM() // compute→memory transition
    // else: no change needed
    
    // Step 5: 执行decoding
    execute_decoding_iteration()
```

α threshold确定（offline）：
```
for each config in {(RLP, TLP) combinations}:
    T_GPU = measure_FC_latency_on_GPU(config)
    T_PIM = measure_FC_latency_on_FC_PIM(config)
    // 若GPU更快, FC为compute-bound
    // 若PIM更快, FC为memory-bound
α = smallest RLP×TLP where T_GPU < T_PIM
```

术语一般如何实现？如何使用？
实现：(1) Host CPU上维护RLP计数器（token-level scheduling循环中decrement on <|eos|>）；(2) TLP专用寄存器，host CPU软件修改TLP时通过指令通知PAPI hardware更新；(3) α离线通过iterative evaluation在simulation/real hardware上确定。开销极低——每个decoding iteration仅需乘法（RLP×TLP）+ 比较（vs α）。PAPI在GPT-3 66B上的evaluation显示AI_est与实际measured AI高度匹配（仅在RLP=128时轻微高估，但此时FC correctly identified as compute-bound）。与其他scheduling方法对比：SpecPIM的genetic algorithm+MCTS需要50轮GA+10K leaf node搜索，在static scenario有效但在dynamic scenario重复运行开销巨大；PAPI的RLP×TLP估计则适用于real-time dynamic scheduling。

涉及论文标题：
- 83-PAPI- Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System .pdf

## Adaptive GPU-PIM Workload Assignment with Roofline Model

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive GPU-PIM Workload Assignment（自适应GPU-PIM负载分配）是ORCHES提出的在GPU-PIM异构系统上将TTC reasoning pipeline中的Linear和Attention operators动态分配到GPU或PIM执行的调度策略。由offline分析模型和online scheduling compensation两部分组成。核心依据是roofline model：以batch size W、hidden dim D、device compute capability CC、memory bandwidth BW（PIM有IO带宽BW_PIM_IO和bank内部带宽BW_PIM两个独立指标）为输入，计算各operator在GPU和PIM上的预估latency，根据比较结果分配。引入ratio α表示operator输出维度中α比例分配给GPU，其余在PIM执行，实现GPU-PIM协同处理（co-processing），数据movement overhead仅占~8.3% total runtime。与prior work（AttAcc固定将全部Linear分配GPU、全部Attention分配PIM）不同，ORCHES支持三种分配策略随batch size自适应切换。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Offline分配决策模型（以Linear operator为例，给定W, D, CC_GPU, BW_GPU, CC_PIM, BW_PIM_IO, BW_PIM）：
```
// Step 1: 计算GPU和PIM上的latency
T_GPU = W*D²/CC_GPU + (2*W*D + D²)/BW_GPU         // 公式1
T_PIM = W*D²/CC_PIM + 2*W*D/BW_PIM_IO + D²/BW_PIM  // 公式2
// PIM多一项IO带宽（controller→banks）

// Step 2: 三种分配策略选择
if T_GPU < T_PIM:         // GPU更快
    if W is large:         // compute-bound（如PRM prefilling W=100+）
        assign Linear→GPU, Shared Attention→GPU, Unique Attention→PIM
    else if W is medium:   // 部分compute-bound
        // 引入协同处理ratio α
        solve T_PIM(α) = T_GPU(α) for α  // 公式3,4
        assign Shared KV Query→GPU, Linear→PIM, Unique KV Query→PIM
else:                      // W is small (memory-bound, 如policy decoding W=4)
    assign All operators→PIM
```
在线补偿策略（T1B，应对shared KV累积导致的动态变化）：
```
// 每step执行前，根据L_i（各KV fragment长度）更新分配
for each layer i: α_i = 1  // 初始：所有层分配给GPU
// 按W_i从小到大排序layers
for each layer i in sorted_by_W_i:
    α_i = 0  // 逐层从GPU reassign到PIM
    update T_PIM({α}), T_GPU({α})  // 公式5,6
    if T_PIM >= T_GPU: break
// 找到critical layer t，求解T_PIM = T_GPU得α_t
// 其他层保持α=0或α=1
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖于：(1) offline profiling确定device的CC和BW数值（GPU modeling和PIM estimating已由prior work [25]在real hardware上验证）；(2) 在simulator frontend (task scheduling) 中实现roofline model计算和分配决策逻辑；(3) online compensation在每step开始前根据当前shared KV累积长度L_i重新求解α。适用场景：任何具有动态batch size、混合memory/compute-bound特征的heterogeneous系统中operator-to-device mapping。

涉及论文标题：
- 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

## Branch Prediction for TTC Reasoning Pipelining

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Branch Prediction for TTC Reasoning Pipelining是ORCHES借鉴CPU branch prediction思想提出的TTC推理pipeline优化技术（T2）。在标准TTC中，generation（policy model decoding）和verification（PRM prefilling）之间存在严格的step间数据依赖——step N的verification必须完成后，step N+1的generation才能开始（C2）。ORCHES使用小型PRM（large PRM的前若干层，如前10层）在GPU idle slot预判当前step的branch selection结果，使PIM上的generation能够speculative提前启动，与GPU上的large PRM full verification重叠执行。若预测错误→rollback到正确selection重新generate。预测准确率通过History Alignment机制从~52%提升至~78%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Branch Prediction + Pipelined Verification的调度伪代码：
```
// Step N Generation已完成在PIM上，candidates C_N已生成
// --- Prediction Phase (GPU idle时触发) ---
// Small PRM = large PRM的前10层（0额外computational overhead）
history_scores = load_from_buffer(steps_1_to_N-1)  // large PRM的historical scores
small_scores = SmallPRM.prefill(P + C_N, history=history_aligned(history_scores))
predicted_branch = argmax(small_scores)

// --- Speculative Generation (PIM上，与verification overlap) ---
PIM_launch(PolicyModel.decode(P + predicted_branch))

// --- Large PRM Verification (GPU上，同步执行) ---
actual_scores = LargePRM.prefill(P + C_N)
actual_selected = argmax(actual_scores)

// --- Pipelined Verification (T2B) ---
// partial tokens ready→GPU idle→启动pre-verification
if generated_tokens_count >= threshold AND GPU_is_idle:
    launch_pre_verification_on_partial_tokens()

// --- Decision ---
if predicted_branch == actual_selected:
    continue  // 预测成功，no rollback
else:
    PIM_rollback()  // 预测错误→重新generate
    PIM_launch(PolicyModel.decode(P + actual_selected))

// --- History Update ---
store_to_buffer(large_PRM_scores_for_step_N)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) Small PRM使用large PRM的子集（前10层），zero额外参数和计算开销；(2) History Alignment——用large PRM的历史scores替换small PRM的历史scores，将预测准确率从~52%提升至~78%；(3) Rollback开销低——PIM decoding快且large PRM verification latency长（3×–5×），有充足时间窗口；(4) Pipelined Verification触发条件：GPU idle AND partial tokens count ≥ threshold。适用于任何multi-branch verification+generation交替的推理pipeline。

涉及论文标题：
- 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

## Memory Structuring for Branch Pruning Fragmentation in TTC

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Structuring是ORCHES应对TTC推理中branch pruning引起的memory fragmentation的三管齐下策略（T3）。TTC每步生成W个候选branch→PRM验证后仅保留top-K（K<W）→其余branch从内存移除→产生memory holes（fragmentation）。candidate长度变化大（10~1000 tokens）进一步加剧问题。传统PIM对非连续内存访问效率低（需额外DRAM indexing），ORCHES通过(1) Address Cache (SRAM)、(2) Dynamic Memory Reorganization、(3) Controller Buffer系统性解决。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory Structuring的调度流程：
```
// (1) Address Cache：logical ID→physical address映射
// 避免DRAM pointer（2次DRAM access→1次SRAM+1次DRAM access）
function get_physical_addr(candidate_id):
    addr_entry = AddressCache[candidate_id]  // SRAM read (1-2 orders faster)
    return (addr_entry.start_addr, addr_entry.length)

// (2) Dynamic Memory Reorganization
fragmentation_ratio = Total_Holes / Memory_for_Reasoning
// 每3-5个reasoning step触发一次
if fragmentation_ratio >= threshold:
    valid_blocks = collect_valid_candidate_blocks()
    compact_to_contiguous(valid_blocks)     // 后台执行，不阻塞主流程
    update_address_cache(valid_blocks)      // 更新SRAM地址映射

// (3) Controller Buffer (Shared KV Buffer)
// QKV aggregation后→KV segments cached in controller die buffer
// Reorganization时KV从buffer写回banks（纯PIM-internal，无host transfer）
// GPU从buffer读取KV cache（减少bank activations）
```
效果：节省平均65% context memory footprint，runtime overhead仅0.12%，area overhead 12%。

术语一般如何实现？如何使用？
Address Cache：SRAM位于Controller Die，由State Machine管理。仅存储sequence起始位置和长度（<1000 data points），area overhead minimal。Memory Reorganization：硬件compaction逻辑，track dirty/free block lists。Controller Buffer：Controller Die集成的额外SRAM buffer。与vLLM PagedAttention不同：PagedAttention解决跨request的KV cache动态allocation，ORCHES T3解决同一request内部branch pruning导致的intra-request fragmentation。

涉及论文标题：
- 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

## Weight-Stationary Systolic Array GEMM with Multiplier-Free Processing

术语解释
Weight-stationary是deep learning accelerator中systolic array的一种经典dataflow模式，权重预加载至PE阵列并保持静止，activation沿行方向逐cycle传播，部分和沿列方向累加。AxCore在此dataflow上实现multiplier-free mpFPMA计算：weight以压缩低比特格式(FP4)静止在PE列中，activation(FP16)流经PreAdd修正后沿行传播，PE内通过整数加法而非乘法完成MAC。

术语是什么？
在weight-stationary dataflow中：(1) Step 1(Preload): 量化权重Wq沿列方向预加载至所有PE并保持静止——因LLM推理中权重在所有token间共享，避免重复加载。(2) Step 2(Propagation): Activation A沿行方向流入PreAdd→计算T=A-B1+C1→沿行广播至该行所有PE。(3) Step 3(Computation): 每PE执行 R=T+Align(Wq) (7-bit adder)→列向累加 Psum+=R (un-normalized)。(4) Step 4(Post-processing): 列累加完成后→Norm(归一化)→AxScale(FPMA-based dequantization: O=Oq+S-B+C2)→Accumulator(writeback)。AxCore配置为64×64 systolic array with 4× tiling。

从kernel调度角度拆解：
```
// AxCore Weight-Stationary mpGEMM Kernel (W4A16, M×K×N)
// A[M,K]: FP16 activation; Wq[K,N]: FP4 quantized weight; S[N]: FP16 scale

// Phase 1: Weight Preload
for col in 0..N step 64:         // 沿输出channel tiling
    for k in 0..K step tiling:   // 沿输入dimension systolic propagation
        load Wq[k:k+tile_size, col:col+64] into PE columns
        // weights remain stationary across all M rows

// Phase 2: Activation Propagation (systolic)
for row in 0..M step 32:         // batch/groups of activation rows
    T = A[row:row+32, :] - B1 + C1  // PreAdd: per-row correction
    for k in 0..K:                // systolic shift along K dimension
        for col_pe in 0..63:     // 64 PE columns in parallel
            wq = PE[col_pe].Wq[k]         // stationary weight
            wq_aligned = SNC_and_align(wq) // subnormal→normal + bit-shift
            R[col_pe] = T[row][k] + wq_aligned  // 7-bit integer adder (not multiplier!)
            Psum[row][col_pe] += R[col_pe]      // column-wise accumulation (unnormalized)

// Phase 3: Post-processing (per column, pipelined)
for col in 0..N step 64:
    O_norm = Norm(Psum[:, col:col+64])     // Shared normalization: LZD→shift→round
    O_final = O_norm + S[col:col+64] - B + C2  // AxScale: FPMA-based dequantization
```
关键性能：weight仅加载一次（preload），activation逐K维度systolic shift。与传统FPC kernel的主要差异：PE内是7-bit adder而非24-bit FP multiplier→允许在相同面积下集成更多PE或运行更高频率→compute density 6.7×-12.5× over FPC。

术语一般如何实现？如何使用？
- Weight-stationary的优势：适合weight量>activation量的场景（LLM推理正好如此——weight固定且量大，activation batch较小）。LLM decoding phase (batch=32, output_len=1)下weight reuse极高。
- 与output-stationary对比：output-stationary保持partial sum在PE中→适合activation reuse高的场景（如CNN feature map reuse）；weight-stationary适合weight reuse高的场景（LLM linear layers）。AxCore选择weight-stationary的原因：(1) 量化后weight窄(4-bit)→PE存储开销小；(2) LLM推理中weight per-token reuse远高于activation reuse。
- Tiling策略：64×64 array + 4× tiling处理大矩阵。Tiling沿M(activation batch)、K(input dim)、N(output dim)三个维度。要求group_size和block_size均为array size倍数以保证对齐。
- 实现工具：DNNWeaver v2.0 (cycle-accurate simulator) 建模数据流延迟；SpinalHDL RTL合成面积/power。

涉及论文标题：
- 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

## Intra-Matrix Heterogeneity (IMH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Intra-Matrix Heterogeneity (IMH) 是HotTiles论文提出的概念，描述稀疏矩阵中非零元素（nonzeros）不均匀分布的特性——非零元素倾向于在矩阵的某些区域形成密集子区域（dense regions），而其他区域高度稀疏（sparse regions），而非均匀散布在整个矩阵中。例如power-law graph的邻接矩阵中，大多数边关联于少数节点，导致相应的稀疏邻接矩阵中非零元素形成聚类。IMH直接决定了SpMM kernel的arithmetic intensity：密集区域有更多nonzeros→更高的计算密度→compute-bound；稀疏区域nonzeros少→dominated by memory accesses→memory-bound。HotTiles的核心洞察是：传统homogeneous SpMM加速器对所有矩阵区域使用相同的PE（Processing Element），无法针对不同IMH特征的区域使用最优处理策略，导致性能次优。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IMH在SpMM kernel调度中的影响（以coPapersCiteseer矩阵，8192×8192 tile size为例）：
```
// IMH对tile特征的影响
for each tile_i in partitioned_sparse_matrix:
    tile_nnzs = count_nonzeros(tile_i)
    tile_density = tile_nnzs / (tile_height * tile_width)
    
    // tile的arithmetic intensity决定其特性
    arithmetic_intensity = (2 * K * tile_nnzs) / bytes_accessed(tile_i)
    // K = dense matrix columns (通常32)
    
    if arithmetic_intensity is high:  // 如coPapersCiteseer对角附近tiles
        // compute-bound → 适合Hot Worker
        bottleneck = compute_throughput
    else:  // 如coPapersCiteseer上三角稀疏region
        // memory-bound → 适合Cold Worker
        bottleneck = memory_bandwidth
```
HotTiles分析发现：coPapersCiteseer的IMH模式表现为对角线和左上角的密集子社区（dense sub-communities沿对角线），这些计算密集型tile应分配给Sextans（hot worker），而矩阵外围的稀疏区域分配给SPADE（cold worker）。IMH-aware分区使hot worker处理的nonzeros从52%（IMH-unaware随机分配）提高到72%。

术语一般如何实现？如何使用？
- **IMH检测**：通过扫描稀疏矩阵的tile-level density统计。不需要全局矩阵density，而是per-tile density分布（tile_nnzs/tile_area的方差）。
- **IMH利用**：HotTiles通过per-tile performance modeling利用IMH——为每个tile分别估计hot worker和cold worker的执行时间（thi/tci），而非使用holistic roofline model假设均匀nonzeros分布。
- **适用场景**：任何nonzeros非均匀分布的稀疏矩阵都表现出IMH，包括power-law graph、社交网络、引文网络、Web graph等。均匀随机稀疏矩阵IMH效应弱，此时HotTiles退化到与IUnaware相近。
- **IMH-unaware的后果**：假设均匀分布的roofline model(如AESPA)会高估或低估特定tile的工作负载特征→导致次优的worker分配→如在SPADE-Sextans架构上IUnaware的带宽压力增加而无compute time改善→性能劣于best homogeneous执行。

涉及论文标题：
- 57-HotTiles_Accelerating_SpMM_with_Heterogeneous_Accelerator_Architectures.pdf

## IMH-Aware SpMM Tile Partitioning Heuristic

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IMH-Aware SpMM Tile Partitioning Heuristic是HotTiles的核心调度算法，将稀疏矩阵的tiles分配给异构PE类型（Hot/Cold Workers）以最大化总体性能。该启发式将NP-hard的分区优化问题（2^Ntiles种组合）分解为4个O(NlogN)复杂度的独立子问题，每个生成一个候选分区方案，最终选择预测runtime最小的方案。4个启发式分别优化：MinTime Parallel（最小化并行执行时间，忽略带宽压力）、MinTime Serial（最小化串行执行时间）、MinByte Parallel（最小化并行模式的访存字节数）、MinByte Serial（最小化串行模式的访存字节数）。每个子问题的求解基于tiles排序+线性扫描cutoff index——复杂度来自排序O(NlogN)，cutoff placement为线性O(N)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HotTiles partitioning heuristic的调度流程：
```
// 输入：对每个tile i，已知 hot_time[i], cold_time[i], hot_bytes[i], cold_bytes[i]
//       Nhw (hot worker数), Ncw (cold worker数), BW (总内存带宽)
//       以及是否支持并行(parallel_possible), t_merge (merge cost)

// === MinTime Parallel Heuristic ===
tiles_sorted = sort_by(hot_time[i] - cold_time[i], increasing)
// 左端=hot favored, 右端=cold favored
best_runtime = INF
for cutoff in range(0, Ntiles):
    hot_tiles = tiles_sorted[0:cutoff]
    cold_tiles = tiles_sorted[cutoff:Ntiles]
    th_total = sum(hot_time[t] for t in hot_tiles) / Nhw
    tc_total = sum(cold_time[t] for t in cold_tiles) / Ncw
    b_total = sum(hot_bytes[t] for t in hot_tiles) + sum(cold_bytes[t] for t in cold_tiles)
    runtime = max(max(th_total, tc_total), b_total / BW) + t_merge
    if runtime < best_runtime:
        best_runtime = runtime
    else:
        break  // 收敛——后续cutoff只会更差

// === MinByte Serial Heuristic ===
// 类似但排序依据为 hot_bytes[i] - cold_bytes[i]，目标函数不同
// 串行模式：th_total和tc_total分别与各自带宽项取max后相加
...

// 最终选择：4个heuristic中预测runtime最小者
best_heuristic = argmin(predicted_runtime of {MinTime_P, MinTime_S, MinByte_P, MinByte_S})
```
**Heuristic选择策略**：
- 低带宽压力（one worker type可轻松饱和BW）→ MinTime Parallel优
- 高带宽压力（多个worker竞争shared BW）→ MinByte或Serial优
- 大merge cost → Serial heuristics优（避免双output buffer合并开销）
- PIUMA架构（Atomic engine消除merge cost）→ 仅使用Parallel heuristics

术语一般如何实现？如何使用？
- **输入准备**：需先运行IMH-aware performance model为每个tile计算hot/cold worker的thi/tci和bhi/bci。vis_lat参数需通过profiling runs提前标定。
- **Sorting键选择**：MinTime heuristics按thi-tci排序（优先给hot worker处理hot-favored tiles），MinByte heuristics按bhi-bci排序。
- **收敛性**：cutoff index线性扫描在预测runtime不再下降时立即停止→实际扫描数远小于Ntiles。
- **HotTiles Framework集成**：partitioning结果决定每个tile→Cold/Hot的assignment，进而决定format creation（hot worker用tiled COO/CSR，cold worker用untiled COO/CSR）。
- **局限性**：假设maximum reuse（忽略tile assignment order对inter-tile reuse的精确影响），且忽略cache reuse→预测误差geomean 12.4%（HotTiles heterogeneous），其中cold worker的caching effect ignored是最主要的误差来源（ColdOnly误差19.6%）。

涉及论文标题：
- 57-HotTiles_Accelerating_SpMM_with_Heterogeneous_Accelerator_Architectures.pdf

## Data Reuse Types in Sparse Tile Processing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Data Reuse Types in Sparse Tile Processing是HotTiles IMH-aware performance model中用于估算主存访问次数的分类体系。在SpMM的tiled执行中，dense matrix (Din/Dout)的rows被不同tiles或同一tile内的不同nonzeros多次访问。HotTiles定义了4种reuse类型，每种对应不同的主存访问量（table lookup式估算）：(1) Inter-tile reuse——先前tile已将dense rows带入fast local memory (FLM)，当前tile无需访问主存→0 rows accessed；(2) Intra-tile (stream)——worker将整个dense tile stream入scratchpad后处理该sparse tile的所有nonzeros→tile_width (Din) 或 tile_height (Dout) rows；(3) Intra-tile (demand)——通过registers/caches按需复用，对row-ordered traversal，Dout仅需tile_uniq_rids个rows（同一r_id的nonzeros复用同一Dout row）→tile_uniq_cids (Din) 或 tile_uniq_rids (Dout)；(4) None——每个nonzero独立触发主存访问，无任何复用→tile_nnzs rows。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Reuse类型决定主存访问量的计算（以SPADE-Sextans架构为例）：
```
// 对sparse tile i，估算Din和Dout的主存访问行数
// 输入：tile_height, tile_width, tile_nnzs, tile_uniq_rids, tile_uniq_cids
//       worker_type (hot/cold), traversal_order (tiled/untiled)

if worker == HOT (Sextans, tiled row-ordered traversal):
    Din_reuse = Intra-tile-stream  // scratchpad preloads tile_width Din rows
    Dout_reuse = Inter-tile         // 假设非该row panel的第一个hot tile
elif worker == COLD (SPADE, untiled row-ordered traversal):
    Din_reuse = None                // 无FLM，on-demand access per nonzero
    Dout_reuse = Intra-tile-demand  // registers复用同一r_id的Dout rows

// 主存访问的dense rows估算 (Table I)
ReuseType_to_rows = {
    Inter-tile:       0,                          // 已在FLM中
    Intra-tile-stream:  tile_width (Din) / tile_height (Dout),
    Intra-tile-demand:  tile_uniq_cids (Din) / tile_uniq_rids (Dout),
    None:              tile_nnzs                  // per-nonzero独立访问
}

// 总主存访问字节数
bytes_Din = ReuseType_to_rows[Din_reuse][Din] * K * element_size
bytes_Dout_read = ReuseType_to_rows[Dout_reuse][Dout] * K * element_size
bytes_Dout_write = tile_uniq_rids * K * element_size  // 仅modified rows写回
bytes_A = sparse_format_bytes(sparse_format, tile_nnzs, tile_height)
// COO-like: 3*tile_nnzs*4; CSR-like: (tile_height + 2*tile_nnzs)*4 (fp32)
```

**关键设计决策**：
- Hot worker (scratchpad-based)的Din reuse=Intra-tile-stream：必须一次stream入tile_width行，即使其中部分是冗余的→在sparse tile (tile_uniq_cids << tile_width)时hot worker产生冗余访存→此时cold worker的on-demand access更优。
- Cold worker (cache/register-based)的Dout reuse=Intra-tile-demand：仅访问uniq_rids个Dout行→当tile_uniq_rids很小时访存高效，但当tile_uniq_rids ≈ tile_nnzs（极稀疏且row-unique nonzeros）时退化为None。

术语一般如何实现？如何使用？
- **Reuse类型选择规则**：取决于worker的FLM类型（scratchpad vs cache/register）、traversal order（tiled vs untiled）、以及sparse format（COO vs CSR）。HotTiles model在partitioning前假设maximum reuse（非首次tile），partitioning后调整实际reuse类型。
- **Accuracy trade-off**：Intra-tile-demand复用假设所有r_id相同的nonzeros都能被cached/registers捕获→忽略cache capacity miss和conflict miss→这是model误差来源之一（特别是PIUMA架构中）。
- **通用性**：该分类体系不仅适用于HotTiles的异构SpMM调度，也可用于任何tiled sparse kernel的访存分析，包括SpMV、SDDMM等具有类似access pattern的kernel。

涉及论文标题：
- 57-HotTiles_Accelerating_SpMM_with_Heterogeneous_Accelerator_Architectures.pdf

## Algorithmic Minimum Live Footprint (from Cascade of Einsums)

术语解释
Algorithmic Minimum Live Footprint是从Cascade of Einsums的dependency分析中推导出的、在任意mapping下kernel中各tensor所需的最小on-chip buffer size。它是mapping-independent的lower bound——任何scheduling/dataflow/mapping都无法使tensor的live footprint低于此值。

术语是什么？
Live footprint指某tensor在执行过程中同时需要保持在on-chip memory中的数据量。给定Einsum cascade，通过分析is-ﬁbertree的read-after-write dependencies可以确定：(1) 哪些tensor的哪些fiber需要在多个Einsum间跨越pass边界（即必须在完成该fiber的所有element处理后才能被下一Einsum re-read）；(2) 该fiber的shape决定了algorithmic minimum live footprint。例如，3-pass attention cascade中QK tensor（shape M×P）的M fiber在Pass 1（QK+GM）完成后需在Pass 2（SN计算）re-read → live footprint至少为M×P（完整的QK tensor）。1-pass cascade通过running max消除了跨pass dependency → QK的live footprint仅为M0×P0（tile size），与总M无关。

从kernel调度角度拆解术语：
以Cascade 1（pedagogical 2-pass cascade）的live footprint分析为例：
```
Cascade 1:  Y = A_k × B_k    (Einsum 5)
            Z = Y × A_k      (Einsum 6)
```
- Dependency: Einsum 6（read A_k）必须在Einsum 5（also read A_k）完成所有K fiber元素后才能start——因为Einsum 5产生Y，而Einsum 6同时需要Y和A_k。即使fusion can produce/consume tiles of Y，也无法避免re-read A_k的K fiber。
- Live footprint of A: 至少K fiber size（shape K）。若buffer不够 → 必须spill+reload → memory traffic ∝ K。
- 若reassociate为Cascade 2（defer Y×）：Einsum 7 Y=A_k×B_k + Einsum 8 X=A_k（reduce away K rank）→ same pass → 1-pass of K。
- Live footprint of A: 可小至tile size → buffer requirement不随K growth。

FuseMax将此分析应用于attention：
```
3-pass attention: QK_{m,p} live footprint = M × P  (whole tensor)
                   → seq len = 1M, P = 64, fp16 → ~128 MB
                   → exceeds on-chip buffer → spill to DRAM
1-pass attention: QK_{m0,p0} live footprint = M0 × P0  (tile on 2D array)
                   → M0 ≈ 16, P0 ≈ 16 → ~512 B
                   → always fits on-chip
```

术语一般如何实现？如何使用？
Live footprint分析用于：(1) accelerator on-chip buffer sizing——给定target sequence length range，确定所需最小buffer capacity；(2) algorithm selection——为给定buffer budget选择合适pass数的cascade；(3) fusion scheduling——确定哪些Einsum可安全fuse而不超出buffer capacity。FuseMax使用此分析证明1-pass cascade的buffer需求与seq len无关（仅取决于tile size M0×P0和P0×P2 partition），因此可处理任意长seq（1M+ tokens）而无需spill。此分析是mapping-independent的——对给定cascade，任何mapping/scheduling/binding都受此live footprint lower bound约束。

涉及论文标题：
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

---

## Mapping and Binding for Spatial Accelerator Architectures

术语解释
Mapping和Binding是描述computation（由cascade of Einsums指定）如何具体调度到spatial accelerator硬件上的两个层级概念。Mapping定义logical tasks和它们之间的dependency graph；Binding定义这些tasks到具体硬件的assignment（compute unit、time、memory location）。

术语是什么？
**Mapping**：将Einsum cascade的iteration space points分组为logical tasks（可小至单点或大至整个iteration space），形成task graph（DAG，nodes=tasks, edges=dependencies）。Mapping spec包含loop order、partitioning/tiling、work scheduling（sequential vs parallel）等。这些dependencies可以是：(1) true dependencies（由cascade的Einsum间的data dependency强制）；(2) ordering constraints（由mapping spec额外添加，如serialization for correctness）。例如Mapping 1（FuseMax mapping）定义了对M和P的双重tiling（M1×M0, P2×P1×P0），M0×P0映射到2D PE array，并在M0/P0层maximally fuse所有Einsum（ComputeRNVTile包含Einsums 44-54，ComputeAVTile为Einsum 55，在P2层fuse）。

**Binding**：将task graph中的每个task分配到具体hardware resources——哪个compute unit执行、何时执行、inputs/outputs在memory hierarchy中的位置。Binding需遵守：(1) task graph的dependencies；(2) architecture的物理限制（PE数量、buffer sizes、NoC topology、communication latency）。FuseMax的binding通过两级interleaving实现：(a) Epoch-level pipelining——不同epoch处理不同tile-relative coordinates的tiles；(b) Cycle-level intra-epoch interleaving——2D array内BQK和SLNV交替计算，1D array内SPNV和RNV交替计算。

从kernel调度角度拆解术语：
FuseMax Mapping 1的loopnest结构：
```
for p2 in 0..P2:         # P = P2 × P1 × P0
  for m1 in 0..M1:       # M = M1 × M0
    for p1 in 0..P1:
      parallel_for p0 in 0..P0:   # mapped to 2D array columns
        parallel_for m0 in 0..M0: # mapped to 2D array rows
          # Fused Einsums 44-54 (ComputeRNVTile):
          BQK[m1,m0,p] = Q[e,p] × BK[e,m1,m0]     # Einsum 44
          LM[m1,p] = max_m0(BQK[m1,m0,p])          # Einsum 45
          RM[m1+1,p] = max(RM[m1,p], LM[m1,p])     # Einsum 46
          SLN[m1,m0,p] = exp(BQK - RM[m1+1,p])     # Einsum 47
          ...
    for p1 in 0..P1:
      parallel_for p0 in 0..P0:
        # Einsum 55 (ComputeAVTile):
        AV[f,p] = RNV[f,M1,p] / RD[M1,p]           # Einsum 55
```
Binding的关键设计决策：
- M0×P0 = #2D PEs（而非更小或更大）→ 最大化parallelism；
- BQK和SLNV在2D array上cycle-level interleave → 隐藏BQK的fill/drain latency → 每cycle所有links active；
- RM需等LM drain → 安排在后续epoch执行（而非stall当前epoch）→ epoch间pipeline重叠；
- 最终division AV在第2个p1 loop单独执行（因RNV/RD需完整M1 traversal后才能使用）。

术语一般如何实现？如何使用？
Mapping和binding的形式化概念源于TeAAL [35]和Timeloop [41]。Timeloop提供automated mapping space search（对单个Einsum搜索loop order/tiling/parallelization）；但cascade-level的binding（如FuseMax的epoch-level + cycle-level interleaving）需手动设计，因Timeloop不原生支持cross-Einsum interleaving。Looptree [21]和TileFlow [62]探索了fused-layer dataflow search但限于有限的transformation space，FuseMax的inter-Einsum interleaving type尚不可被这些工具自动discover。在FPGA systolic array compiler（如AutoSA [54]）中，mapping tuning聚焦于PE array allocation和I/O scheduling。在GPU上，mapping对应于grid/block/thread decomposition + shared memory tiling，binding对应于warp scheduling + register allocation。

涉及论文标题：
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

---

## Fine-grain Intra-Epoch Interleaving (Cycle-level Spatial Array Binding)

术语解释
Fine-grain Intra-Epoch Interleaving是FuseMax在spatial array的单个epoch内部、以cycle粒度交错执行不同类型Einsum的binding技术，使2D和1D PE array的utilization接近100%。

术语是什么？
在spatial array architecture上，每个tile的computation需要fill（加载数据到PEs）、compute（PE内计算）、drain（spatial reduction输出）三个阶段。传统non-interleaved binding（如+Architecture配置）fully produce和consume一个tile后再start下一个→fill/drain期间PEs idle。FuseMax的intra-epoch interleaving在单epoch内：(1) 2D array上cycle-level交替BQK（Q×K^T matmul）和SLNV（SLN×V matmul+reduce）→'BQK|SLNV' notation；(2) 1D array上同时进行SPNV（correction of old running num×V）和RNV（accumulation of new num×V）→'SPNV|RNV' notation。每cycle所有neighbor-neighbor links都active（carrying either BQK or SLNV data）。Interleaving依赖tight coupling between 2D and 1D arrays实现static schedule，无需GPU所需的frequent synchronization。

从kernel调度角度拆解术语：
以2×2 toy array的intra-epoch interleaving为例（Figure 5 of FuseMax）：
```
Time:    t=0        t=1        t=2        t=3        t=4
2D Array:
  PE(0,0): BV0→     -          SLN0×BV0   -          ...    (BQK|SLNV alternating)
  PE(0,1): -         BV1→      -          SLN1×BV1   ...
  PE(1,0): BK0→      BK0       BK0→       BK0        ...    (BK stationary for BQK)
  PE(1,1): BK1→      BK1       BK1→       BK1        ...
1D Array:
  PE(0):   -         -          SPNV0      SPNV0+...   ...   (SPNV|RNV alternating)
  PE(1):   -         -          RNV_old    RNV_new    ...

Link activity (each cycle):
  link(0,0)→(0,1): BV0 move    -          SLN0×BV0 move   -          ...
  link(1,0)→(1,1): BK0 move    -          BK0 move        -          ...
  link(0,0)→(0,1): Q broadcast Q          Q broadcast     Q          ... (BQK phase)
  link(0,0)→(0,1): -           V broadcast -              V broadcast ... (SLNV phase)
```
Key: (1) BQK的E×M0×P0 iteration space需要E cycles→fill/drain各需~4E cycles（output stationary dataflow）。若无interleaving，2D array在fill+drain期间idle（utilization = E/(E+8E) ≈ 11%）。Interleaving使SLNV的compute overlap在BQK的fill/drain期间→utilization → ~100%。(2) 1D array在等待2D drain时不idle，而是处理前一轮的SPNV/RNV（使用已在buffer中的旧running values）。

术语一般如何实现？如何使用？
Intra-epoch interleaving实现需要：(1) PE支持dual-mode operation——同一PE可配置为BQK mode（MAC only）或SLNV mode（先exp via 6 sequential MACs，再MAC with BV）；(2) NoC支持packet-level multiplexing——相邻cycle可传输不同Einsum的数据；(3) Static schedule——由于所有tile size和通信模式在编译时已知，可offline生成per-cycle PE和link的调度表→硬件仅需按表执行（no runtime synchronization）。此技术与GPU warp specialization（如FlashAttention-3）不同：GPU通过warp-level software pipelining + async copy隐藏latency，但因SIMT的lockstep执行和高同步成本→utilization仅~61%（vs FuseMax ~100%）。FPGA systolic array可实现类似fine-grain interleaving via modulo scheduling，但受限于FPGA routing congestion。

涉及论文标题：
- 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

## Context Daemon (上下文守护进程)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Context Daemon是SpotServe在每个GPU instance上运行的独立后台进程（C++实现），负责管理两种GPU context：(1) Model Context——LLM的模型权重参数（如GPT-20B的74.5GB参数切分后的per-GPU shard）；(2) Cache Context——per-request的推理中间状态，包括KV cache（各Transformer layer的key和value张量）和intermediate activations。Context Daemon的核心作用是解耦context管理与推理执行：Daemon作为独立进程存在，即使Inference Engine（FasterTransformer）因instance preemption而中断，Daemon进程仍然存活——GPU上的context不会被释放或丢失。这避免了两个关键开销：(a) 重新从磁盘/云存储加载模型参数（GPT model with 120B parameters loading takes >2 minutes on AWS）；(b) 重新计算因preemption丢失的KV cache状态（相当于recomputation of all previously generated tokens）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Context Daemon管理GPU内存中两类tensor的生命周期。伪代码示意：
```
// Context Daemon管理接口
class ContextDaemon:
    model_contexts: Dict[layer_id, Tensor]  // GPU tensors for model weights
    cache_contexts: Dict[req_id, KVState]   // per-request KV cache
    
    def get_model_tensor(layer_id, shard_id) -> Tensor:
        return model_contexts[(layer_id, shard_id)]  // proxy到Inference Engine
    
    def get_cache_tensor(req_id, layer_id) -> Tensor:
        return cache_contexts[req_id][layer_id]
    
    def migrate_context(target_gpu, tensors, stream):
        // NCCL batched async send
        for t in tensors:
            ncclSend(t.data, target_gpu, stream)  // 异步发送
            lock(t)  // mutex lock blocking inference access
        ncclGroupEnd()
    
    def on_migration_complete(tensor):
        unlock(tensor)  // 释放mutex，推理可继续
```

具体执行过程：(1) 推理时，FasterTransformer通过Daemon提供的proxy接口获取model/cache tensor的GPU指针（通过CUDA IPC共享内存实现零拷贝）。(2) Preemption notification到达→Interruption Arranger决定stop decoding→Daemon标记当前cache context state为"committed"。(3) Migration阶段：Daemon通过NCCL batched send/recv将tensor数据传输到target instance的Daemon→target Daemon接收并注册为新context。(4) 若preemption发生在迁移期间（unexpected early preemption）→Daemon回退：如有cache的all replicas丢失→仅保留model context（从其他instance或disk加载）；若model也丢失→full restart。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SpotServe实现：(1) Context Daemon作为独立C++进程（ParamsClient），与FasterTransformer通过CUDA IPC共享GPU memory。使用cudaIpcGetMemHandle()和cudaIpcOpenMemHandle() API实现跨进程共享。(2) Model context在初始加载时分配在Daemon的进程空间，FasterTransformer通过IPC获取tensor指针直接访问（无数据拷贝）。(3) Cache context在推理过程中按需动态分配（per-request KV cache），由Daemon统一管理生命周期。(4) NCCL通信：Daemon初始化NCCL communicator，迁移时调用ncclGroupStart()/ncclGroupEnd()包裹batched send/recv操作。(5) Mutex机制：每个context tensor关联一个mutex lock——迁移开始前Daemon对所有被迁移的tensor加锁（block推理引擎对这些tensor的读写）；迁移完成后释放锁；未被迁移的tensor不受影响（允许推理与迁移overlap）。(6) 内存管理：FasterTransformer原始的内存分配被替换为重定向到Daemon管理的tensor pool，统一分配/回收。(7) 开源实现：https://github.com/Hsword/SpotServe 中ParamsClient目录。

涉及论文标题：
- 60-SpotServe- Serving Generative Large Language Models on Preemptible Instances.pdf

## CUDA IPC for Multi-Process GPU Context Sharing (CUDA进程间GPU内存共享)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA IPC (Inter-Process Communication) 是NVIDIA CUDA提供的API，允许同一机器上不同CPU进程共享GPU内存中的同一块物理内存区域。在SpotServe中，CUDA IPC用于解决Context Daemon和FasterTransformer Inference Engine分属不同进程的问题——两个进程需要访问相同的model/cache context tensor数据，但由于进程隔离，正常情况下的GPU内存分配（cudaMalloc）仅对分配进程可见。CUDA IPC通过导出一个进程分配的GPU内存的"handle"（cudaIpcGetMemHandle），另一个进程通过cudaIpcOpenMemHandle打开handle即可获取对同一物理GPU内存的访问指针，实现零拷贝（zero-copy）跨进程共享。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码流程：
```
// Process A: Context Daemon (分配GPU内存)
float *d_model_context;
cudaMalloc(&d_model_context, model_size_bytes);
load_model_weights(d_model_context, "gpt-20b.bin");

// 导出IPC handle
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, d_model_context);

// 通过Unix domain socket或共享文件传递handle给Process B
send_to_process_b(handle);

// ===========================================

// Process B: FasterTransformer Inference Engine (访问同一GPU内存)
cudaIpcMemHandle_t handle;
recv_from_process_a(&handle);

float *d_model_context;
cudaIpcOpenMemHandle((void**)&d_model_context, handle, cudaIpcMemLazyEnablePeerAccess);

// 现在d_model_context指向与Process A相同的物理GPU内存
// Inference Engine可直接读写model weights，无需任何数据拷贝
run_inference(d_model_context, input_tokens, output_tokens);

// 使用完毕后
cudaIpcCloseMemHandle(d_model_context);
```
关键执行细节：(1) 所有context tensor（model weights + KV cache）由Daemon分配和维护；(2) Inference Engine启动时通过CUDA IPC获取所有model context tensor的指针——访问model weights时直接dereference，与同进程分配的性能相同；(3) KV cache在推理过程中持续增长（每个new token追加），Daemon动态分配新内存→通过IPC共享→Inference Engine可直接追加key/value；(4) 当preemption触发context migration时，Daemon管理NCCL通信，Inference Engine被通过mutex lock暂时blocked访问被迁移的tensor。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA IPC关键API：(1) cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)——导出devPtr所指GPU内存的IPC handle。要求devPtr必须是cudaMalloc分配的base pointer（不能是偏移地址）。(2) cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)——在目标进程中打开IPC handle，获取GPU内存指针。flags可选cudaIpcMemLazyEnablePeerAccess（延迟启用peer access）。(3) cudaIpcCloseMemHandle(void* devPtr)——关闭IPC内存引用。限制：(a) 仅支持统一虚拟地址空间（Unified Virtual Addressing, UVA）的平台；(b) 被共享的内存不能被释放直到所有进程都close了handle；(c) 跨进程共享的GPU内存不支持cudaFree——只能由原始分配进程释放；(d) 同一机器的不同process间共享（不跨网络）。在SpotServe中，CUDA IPC避免了跨进程的GPU→CPU→GPU数据拷贝——Daemon和FT间共享几十GB的model context（如GPT-20B 74.5GB）时零拷贝对性能至关重要。

涉及论文标题：
- 60-SpotServe- Serving Generative Large Language Models on Preemptible Instances.pdf

## NCCL Batched Async Send/Recv for LLM Context Migration (NCCL批量异步传输用于LLM上下文迁移)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NCCL (NVIDIA Collective Communications Library) batched async send/recv 是SpotServe中实现context migration的核心通信机制。NCCL是NVIDIA提供的GPU间高性能集体通信库，原用于分布式训练的all-reduce/all-gather等集体操作。SpotServe使用NCCL的point-to-point send/recv primitives（而非collective operations），通过ncclGroupStart()/ncclGroupEnd()将多个p2p传输操作bundle成一个group，实现batch异步执行——所有send/recv在group内部被NCCL调度器并行或流水线化执行，最小化启动开销。用于在inter-instance 50Gbps网络上传输model context和cache context的tensor数据。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码（每个GPU instance上的Context Daemon执行的迁移逻辑）：
```
ncclGroupStart();
for each tensor t in migration_plan:
    if is_source(t, my_gpu_id):
        ncclSend(t.data, t.size, ncclChar, t.target_gpu, comm, stream);
        lock_mutex(t.mutex);  // 推理引擎被blocked访问正在传输的tensor
    if is_dest(t, my_gpu_id):
        // 预分配接收buffer
        cudaMallocAsync(&recv_buf[t.id], t.size, stream);
        ncclRecv(recv_buf[t.id], t.size, ncclChar, t.source_gpu, comm, stream);
ncclGroupEnd();
cudaStreamSynchronize(stream);  // 等待所有传输完成

// 传输完成后释放锁，注册新context
for each tensor t in received_tensors:
    register_context(t.id, recv_buf[t.id]);
    unlock_mutex(t.mutex);  // 推理引擎可继续访问
```

NCCL group的关键特性：(1) 所有在GroupStart/GroupEnd间的操作作为一个batch提交到NCCL调度器→NCCL内部可能合并小tensor传输、复用网络连接、overlap多个p2p流。(2) 异步执行——cudaStreamSynchronize等待完成后GPU继续工作。(3) 数据类型ncclChar表示按字节传输（context tensor无类型语义，就是raw bytes）。以GPT-20B migration为例：需要传输~2.6GB model context（per-layer weights按plan顺序分批）+ ~400MB cache context。每个tensor传输大小从几MB（单层shard）到几百MB（大层权重）不等。NCCL group化后将所有p2p操作batch提交→在50Gbps网络上利用full bandwidth。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现细节：(1) NCCL communicator初始化：每个GPU instance在启动时创建NCCL communicator（包含所有可能参与的GPUs的拓扑信息），comm通过ncclCommInitRank基于MPI初始化。(2) 通信buffer管理：GPU memory中动态分配通信buffer——每个instance的可用buffer size受U_max约束（在MemOptMigPlanner中定义）。传输完成后buffer立即释放（cudaFreeAsync）。(3) CUDA stream使用：所有NCCL p2p操作在专用stream上执行，与推理计算stream分离——允许推理引擎在未被迁移的tensor上继续计算，实现compute与communication overlap。(4) Mutex lock粒度：per-tensor mutex——迁移中的tensor被加锁（推理访问被阻塞），已完成迁移的tensor解锁（推理可访问），同一GPU上不同tensor可处于不同状态（部分迁移部分推理）。(5) NCCL版本要求：≥2.10。论文提到NCCL batched send/recv primitives在NCCL 2.10+支持grouped p2p操作，更早版本只能逐条send/recv（overhead更大）。(6) 与progressive migration的配合：NCCL传输按migration planner生成的顺序执行（cache优先→layer weights按MemOptMigPlanner顺序）→每完成一个stage的传输即下发<start>到该stage。

涉及论文标题：
- 60-SpotServe- Serving Generative Large Language Models on Preemptible Instances.pdf

## Expert Co-processing (xPU + Logic-PIM Expert-Level Kernel Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Co-processing是Duplex提出的MoE层runtime kernel调度技术，利用MoE层中不同expert处理不同数量token的特性（gate分配后token distribution非均匀），将处理token数较少的expert分配给Logic-PIM（低Op/B处理器，高带宽低计算），将处理token数较多的expert分配给xPU（高Op/B处理器，高计算），使两者并行执行以最小化MoE层总延迟。核心决策基于预存的lookup table（含不同token数下xPU和Logic-PIM的预估处理时间），在runtime gate输出后快速查表决定expert分配，decision overhead negligible。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Co-processing的运行调度伪代码（以Mixtral 47B, 8 experts, top-k=2, batch size=64, decoding-only stage为例）：

```
// Stage: Decoding-only, batch_size=64, N_experts=8, top_k=2
// After Gate computes token-to-expert assignment

Input: tokens[batch=64, D=4096], expert_weights[8][3 matrices each]
       lookup_table[N_experts][max_tokens]

// Step 1: Gate forward pass (on xPU)
gate_logits = Gate_FC(tokens)  // [64, 8]
expert_ids, gate_weights = top_k(gate_logits, k=2)  // [64, 2]

// Step 2: Count tokens per expert
token_counts = zeros(N_experts)  // [8]
for t in range(64):
    for k in range(2):
        e = expert_ids[t][k]
        token_counts[e] += 1
// Example: token_counts = [18, 12, 8, 15, 22, 6, 10, 14]

// Step 3: Estimate processing time using lookup table
T_xPU_all = 0
for e in range(8):
    T_xPU_all += lookup_table[e][token_counts[e]].T_xPU

// Step 4: Greedy assignment
sorted_experts = argsort(token_counts)  // ascending
assignment = ["xPU"] * 8
T_best = sum of all T_xPU[e]

for e in sorted_experts:
    assignment[e] = "Logic-PIM"
    T_xPU_remaining = sum(T_xPU[i] for i where assignment[i]=="xPU")
    T_LP_total = max(T_Logic-PIM[i] for i where assignment[i]=="Logic-PIM")
    T_new = max(T_xPU_remaining, T_LP_total)
    if T_new < T_best:
        T_best = T_new
    else:
        assignment[e] = "xPU"
        break

// Step 5: Dispatch and parallel execution
// xPU: process assigned experts as GEMM on H100 SMs
// Logic-PIM: process assigned experts on GEMM modules @650MHz

// Step 6: All-reduce after all experts complete
xPU_all_reduce(Logic-PIM_output_buffers)
// FC3 (down-projection) on xPU
```

关键参数：lookup_table预存不同token count下每个expert在xPU和Logic-PIM上的预估处理时间——xPU时间考虑GPU SM计算+HBM external BW读weight（低token count时memory-bound），Logic-PIM时间考虑GEMM module计算+bank bundle internal BW读weight（4x BW降低memory time但更多MACs使compute bound早于Bank-PIM）。Tensor parallelism for experts (Duplex+PE+ET) 将每expert weight列切分到所有devices→每device处理全部8 experts→co-processing粒度更细。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **Lookup table预计算**：离线对不同token count (1-128) sweep每个expert在xPU和Logic-PIM上的处理时间。使用cycle-accurate simulator (Ramulator 2.0)。Table size = N_experts × max_tokens × 2 (xPU+LP) × 4B ≈ 8KB，可完全cache。
- **Runtime overhead**：gate forward pass是MoE层标准操作，token counting O(batch×top_k)，lookup查表+greedy O(N_experts·log N_experts)，总overhead << MoE GEMM computation time (O(batch×D×D_interm))。
- **适用条件**：token distribution skewed时最有效（如某些experts处理30+ tokens，某些仅2-3 tokens）。prior work [19]论证real workloads中expert skews常见。若token均匀分布则全部xPU或全部Logic-PIM可能更优。
- **与expert parallelism的关系**：传统expert parallelism将不同expert分布到不同GPUs→每device仅处理Nex/Ndevice个expert→co-processing粒度受限。Duplex+PE+ET用tensor parallelism for experts替代expert parallelism（within node）→每device处理全部Nex experts→co-processing分配粒度提升Ndevice倍。

涉及论文标题：
- 61-Duplex- A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.pdf

## Attention Co-processing (xPU + Logic-PIM Request-Level Attention Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Co-processing是Duplex提出的attention层runtime kernel调度技术，在continuous batching的mixed stage中利用attention operation per-request独立特性，将prefilling sequence（新到达request的长input sequence）的attention kernel调度到xPU执行（高Op/B GEMM：Lin个Q slices共享同一K/V矩阵），将decoding sequences（ongoing requests的单token generation）的attention kernel调度到Logic-PIM执行（低Op/B：每request Q vector × unique KV cache）。两组kernel在不同处理单元上并行执行，bank bundle-aware memory allocation确保无memory access冲突。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Attention Co-processing调度流程（以Mixtral GQA deggrp=4, mixed stage: 1 prefill Lin=2048 + 64 decoding）：

```
// Mixed stage attention scheduling:
// Memory allocation:
//   Bank bundles 1-3: KV cache for decoding requests (Logic-PIM reads)
//   Bank bundle 4: prefill Q/K/V scratch (xPU reads/writes)

// === xPU: Prefill attention (high Op/B) ===
for each head group g (8 groups, deggrp=4):
    Q_pref = slice Q for heads in group g  // [2048, 512] (4 heads × D_head=128)
    // GEMM: [2048, 512] × [512, 2048] via K^T
    // 2048 Q rows share same K/V → high data reuse → high Op/B
    attn_scores = Q_pref × K_cache_pref^T / sqrt(128)
    attn_weights = softmax(attn_scores)
    attn_out_group = attn_weights × V_cache_pref

// === Logic-PIM: Decoding attention (low Op/B, parallel with above) ===
for each (req_id, head_id) pair distributed across Logic-PIM stacks:
    Q_vec = Q[req_id, head_id]        // [1, 128]
    K_mat = KV_cache[req_id, head_id] // [seq_len_req, 128]
    V_mat = KV_cache[req_id, head_id] // [seq_len_req, 128]
    // GEMV: Q_vec × K_mat^T → softmax → × V_mat
    // Low Op/B: single Q vector per request × unique KV
    // Logic-PIM GEMM module (512 MACs) processes this
    // Softmax on Logic-PIM softmax module

// Synchronize after both complete
// xPU migrates newly created prefill K/V from bundle 4 to bundles 1-3
// Projection (FC) on xPU
```

关键：bank bundle 4独立于bundles 1-3→xPU和Logic-PIM访问无冲突。prefill K/V迁移仅发生一次per prefill (overhead negligible)。Request parallelism和head parallelism使Logic-PIM可fully parallelize decoding attention。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **Bank bundle-aware memory分配**：KV cache在bundles 1-3（Logic-PIM读取），prefill scratch在bundle 4（xPU读写）→无冲突并发。Prefill完成后K/V迁移至bundles 1-3（一次性操作，overhead negligible）。
- **GQA影响**：GQA deggrp=4-8使attention Op/B升至4-8→Logic-PIM compute-to-bandwidth ratio=8匹配此范围。MHA (deggrp=1)下Op/B极低→Bank-PIM可能更优（论文在OPT MHA上验证Bank-PIM优于Logic-PIM for MHA）。
- **与expert co-processing协同**：Attention co-processing处理attention层（mixed stage），expert co-processing处理MoE层（所有stages），交替应用于不同layer类型共同提升throughput。
- **GPU-only对比**：GPU-only需sequential执行或因共享HBM bandwidth竞争。Logic-PIM独立bank bundle path消除此瓶颈。

涉及论文标题：
- 61-Duplex- A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.pdf

---

## Inter-Chiplet Pipelining for Multi-Model MCM Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Inter-Chiplet Pipelining是SCAR提出的一种跨chiplet流水线执行技术，用于在异构MCM AI加速器上加速多模型推理。其核心思想是：将同一模型的不同layer segments分配到不同chiplet上，producer chiplet在完成segment的部分计算后（而非等待整个segment完成），立即通过NoP将中间数据以streaming方式发送给consumer chiplet，使两个chiplet的计算在时间上重叠。这种producer-consumer流水线执行减少了在chiplet local memory中完整缓冲中间数据的需求，增强in-package data reuse（数据不经过off-chip DRAM往返，仅通过高带宽NoP传输），并降低整体执行延时。Inter-chiplet pipelining是Simba [64] cross-layer pipelining在异构多模型场景下的泛化——不仅支持同一模型内的跨层流水线，也支持不同模型的layer在不同chiplet类型间流水线执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Inter-Chiplet Pipelining的执行过程（以ResNet-50的Conv5_x→FC layer在2个chiplet上的pipeline为例）：
```
// Producer: NVDLA chiplet (c_prod) executes Conv5_x segment
// Consumer: Shi-diannao chiplet (c_cons) executes FC segment

// Producer:
for h_tile in 0..H_out_tiles:
  for w_tile in 0..W_out_tiles:
    compute_conv_tile(input[h_tile*Th:(h_tile+1)*Th],
                      weight, output_tile[h_tile][w_tile])
    // Stream tile via NoP to consumer immediately
    NoP_send(c_cons, output_tile[h_tile][w_tile], stream_id=layer_id)

// Consumer (overlapping in time with producer):
received_tiles = 0
while received_tiles < H_out_tiles * W_out_tiles:
  tile = NoP_recv(c_prod, stream_id=layer_id)
  fc_compute(tile, fc_weight, fc_output_partial)
  received_tiles += 1
fc_output = accumulate(fc_output_partial)
```
在SCAR的SEG引擎中，Inter-Chiplet Pipelining由Heuristic 2促进：segmentation时优先将连续层分配到不同chiplet并标记为pipeline edge。调度器评估pipelining收益（time_overlap = min(producer_tile_time × n_tiles, consumer_tile_time × n_tiles)）与成本（extra NoP BW consumption）。仅当data transfer time < compute overlap benefit时才启用pipelining。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Inter-Chiplet Pipelining的调度实现：(1) SEG engine识别pipeline candidate——检测连续层pair (l_i, l_{i+1})，若适合映射到不同dataflow类型chiplet（如Conv→NVDLA、FC→Shi-diannao），标记为pipeline opportunity；(2) Chiplet mapping将pipeline pair分配到不同chiplet，确定tile granularity——tile越小overlap越多但NoP transaction overhead更大；(3) Execution timeline中consumer start time = max(producer first tile ready time, consumer ready time)，而非producer全部完成时间。MAESTRO offline database提供per-layer partial output ready latency。Simba原始cross-layer pipelining已验证16% speedup；SCAR扩展到跨dataflow chiplet和跨模型场景。

涉及论文标题：
- 62-SCAR- Scheduling Multi-Model AI Workloads on Heterogeneous Multi-Chiplet Module Accelereators.pdf

## Time Window-based Layer-to-Chiplet Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Time Window-based Layer-to-Chiplet Scheduling是SCAR多层调度框架的核心机制。它将多模型端到端工作负载按periodic time windows切分，在每个时间窗口内独立执行layer assignment→chiplet provisioning→layer segmentation→chiplet mapping的子调度。Time window由start time（Ts）和duration（Ttw）定义，包含一组从各模型分配来的layer（L = {l | l ∈ Sc}）。最坏情况模型延时作为time horizon被划分为nsplits个periodic windows（SCAR默认nsplits=4→5 windows）。该机制将全局O(10^56)的巨大调度空间分解为per-window局部子问题，同时通过dynamic chiplet regrouping使chiplet分配在窗口间可动态变化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Time Window Scheduling的完整流程（Algorithm 1 - Greedy Layer Packing）：
```
Input: M (multi-model workloads), T (time windows), C (chiplets), DF (dataflows)
Output: L2W (Layer-to-Window assignments)

Function LAYER_ASSIGNMENT(M, C, T):
  for each model m in M:
    exec_win = []       // layers in current window
    win_idx = 0
    used_cycles = 0
    
    for each layer l in m:  // topological order
      // Expected latency: average across all chiplet dataflow types
      E(Lat(l)) = Σ_{i=1}^{|DF|} (ndf_i / |C|) × Lat(l → df_i)
      
      while True:
        if win_idx == |T|:
          Slack = None  // last window, no time constraint
        else:
          Slack = T[win_idx] - used_cycles
          
        if Slack == None or E(Lat(l)) <= Slack:
          exec_win.append(l)   // assign to current window
          used_cycles += E(Lat(l))
          break
        else:
          L2W[win_idx][m] = exec_win  // finalize window
          used_cycles = T[win_idx]
          exec_win = []
          win_idx += 1
    
    L2W[win_idx][m] = exec_win
  return L2W
```
窗口内后续步骤per-window：
1. **PROV**：N_i = round(E(P_i)/ΣE(P_j) × |C|)——为每模型分配chiplet数
2. **SEG**：将窗口内layer集分割为segments，搜索inter-chiplet pipelining
3. **Chiplet Mapping**：将segments绑定到具体chiplet（考虑dataflow compatibility和NoP hop distance）
4. **Execution Timeline**：构建per-window Gantt图→计算端到端EDP/latency/energy

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) Time horizon由MAESTRO单模型end-to-end estimation得到；(2) nsplits=4在实验分析中EDP improvement rate趋于平稳；(3) First-fit greedy packing确保低延迟层优先在early window执行（防starvation），跨window边界的层推迟到下一窗口；(4) Dynamic chiplet regrouping——窗口边界是chiplet重新分配时机，窗口k的chiplet-to-model assignment在窗口k+1可完全改变；(5) MCM-Reconfig用expected execution time做粗略assignment，fine-grained scheduling由SEG在per-window内完成；(6) 与MLIR兼容——Time window对应coarse-grained scheduling pass，segment mapping对应fine-grained graph partitioning pass。

涉及论文标题：
- 62-SCAR- Scheduling Multi-Model AI Workloads on Heterogeneous Multi-Chiplet Module Accelereators.pdf

## MetaOp (Meta-Operator) and Graph Contraction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MetaOp（Meta-Operator）是Spindle提出的计算图收缩抽象，将原始计算图G=(V,E)中连续的相同类型operators融合为一个MetaOp节点，得到缩约MetaGraph G_M=(V_M, E_M)。融合条件（§3.1）：(1) operator i和j之间有数据流⟨i,j⟩∈E，且i的出度和j的入度均为1（直接前后继关系）；(2) operator i和j具有相同的operator type和相同的input data size（确保完全相同的workload）。图收缩按拓扑序遍历G，重复应用融合条件直到无法继续融合。每个MetaOp m∈V_M包含L_m个连续operators，这些operators共享相同的执行时间函数T_m(n)（在n个设备上的执行时间），因为它们的workload完全相同。MetaOp将原始计算图中大量operators（如多层Transformer layers）聚合为少数异构workload代表，大幅缩小优化问题规模。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MetaOp的graph contraction流程（以Multitask-CLIP为例，图3）：
```
// Input: Original computation graph G = (V, E) with various operators
// Each operator has attributes: op_type, input_shape, output_shape

GraphContraction(G):
  G_M = G  // start with original graph
  changed = True
  while changed:
    changed = False
    // Topological traversal
    for i in topological_order(G_M):
      for j in successors(i) in G_M:
        if out_degree(i) == 1 AND in_degree(j) == 1       // Cond (1): direct chain
           AND op_type(i) == op_type(j)                     // Cond (2): same type
           AND input_shape(i) == input_shape(j):            // Cond (2): same workload
          // Contract i and j into single MetaOp
          fused = Merge(i, j)  // L_fused = L_i + L_j
          G_M = replace(G_M, i, j, fused)
          changed = True
          break  // restart after each contraction
  return G_M
```
例如Multitask-CLIP中：
- 原始operators A, B, C（均为Audio Op, input [8,229,768]）→ MetaOp 1 (L=3)
- 原始operators D, E（Text Op, [8,77,768]）→ MetaOp 2 (L=2)  
- 原始operators F, G, H（LM Op, [8,512,1024]）→ MetaOp 3 (L=3)
- 类似的Text Op→MetaOp 4, Vision Op→MetaOp 5&6, LM Op→MetaOp 7

每个MetaOp m的每operator执行时间函数T_m(n)是相同的（相同workload），MetaOp总时间开销 = T_m(n) × L_m。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Graph Contraction在Spindle中实现为Python代码（Execution Planner，~2.1K LoC）。基于PyTorch计算图（通过torch.fx Tracer获取），按拓扑序遍历节点，依次检查相邻节点对是否满足融合条件。融合后的MetaGraph G_M保留原始operators的拓扑依赖关系，但将同构连续operators折叠。随后通过BFS在G_M上分配MetaLevel编号（同level内MetaOp无依赖），将全局优化问题(1)分解为per-MetaLevel子问题。MetaOp抽象适用于任何包含大量重复同构层的DL模型（如Transformer的stacked layers），通过减少优化变量数量（从|V|到|V_M|）实现高效的operator-level资源分配和调度。

涉及论文标题：
- 63-Spindle- Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling.pdf

## Malleable Project Scheduling Problem (MPSP) for Operator-Level Resource Allocation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Malleable Project Scheduling Problem (MPSP) 是调度理论中的经典问题——给定一组可塑（malleable）项目（每个项目的执行时间可随分配资源量变化）和连续可分资源，在无项目间依赖约束的条件下，最小化最大完成时间（makespan）。Spindle（§3.3）将per-MetaLevel的资源分配问题形式化为MPSP：将每个MetaOp m建模为一个malleable project（包含L_m个operators，每个operator的执行时间T_m(n)随分配GPU数n变化），目标是找到最优分配方案使所有MetaOp的最大完成时间C*最小。Theorem 1给出MPSP在正非增执行时间函数下的最优解性质：所有MetaOp同时开始（s_m=0）、同时结束（e_m = C*），满足T_m(n_m*)*L_m = C* for all m, 且 Σ n_m* = N。这意味着最优分配下，所有MetaOp的执行时间均对齐到相同值C*。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Spindle中MPSP求解过程（per-MetaLevel, N devices）：
```
// Input: N devices, MetaOps m ∈ V_M with scaling curves T_m(n)
// Output: Optimal C*, allocation n*_m for each MetaOp m

BisectionSearch_MPSP(N, {T_m(n), L_m | m ∈ V_M}):
  // Binary search on C (makespan)
  C_low = 0, C_high = max_m(T_m(1) * L_m)  // upper bound: all serial
  while C_high - C_low > ε:
    C_mid = (C_low + C_high) / 2
    // Check if feasible: can we fit within C_mid?
    total_devices = 0
    for m in V_M:
      // Find min n such that T_m(n) * L_m ≤ C_mid
      n_m = find_min_n_for_deadline(m, C_mid)  // from scaling curve
      if n_m is infeasible:  // T_m(N)*L_m > C_mid
        infeasible = True; break
      total_devices += n_m
    if total_devices ≤ N AND not infeasible:
      C_high = C_mid  // feasible, try smaller C
    else:
      C_low = C_mid   // infeasible, need larger C
  C* = C_high
  // Recover n*_m from final feasible search point
  for m in V_M:
    n*_m = T_m^{-1}(C* / L_m)  // from scaling curve inverse
  return C*, {n*_m}
```
以图5a为例：N=4, 3个MetaOps (L1=8, L2=12, L3=6)。Bisection search找到C*使得T1(n1*)*8 = T2(n2*)*12 = T3(n3*)*6且n1*+n2*+n3*=4。得到n*1=0.96, n*2=1.5, n*3=0.6（连续值），MetaOp3的n*3<1表示其workload较小可以串行执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MPSP求解在Spindle中利用scalability estimator（§3.2）提供的分段α-β scaling curves T_m(n)进行bisection search。由于T_m(n)是分段函数，求逆T_m^{-1}(C*/L_m)时需要在各段内查找。Bisection search收敛速度快（对数复杂度），配合前期的graph contraction降维，整个resource allocation过程在execution planner中runtime <3秒（图12）。MPSP formulation适用于任何需要将异构workload分配到有限可扩展资源的场景，关键前提是workload执行时间对资源量单调非增（SCAR [MICRO'24]也使用类似概念进行chiplet分配）。Spindle将其应用从传统project scheduling扩展到DL operator-level GPU资源分配。

涉及论文标题：
- 63-Spindle- Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling.pdf

## Wave (Smallest Scheduling Unit in Wavefront Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wave是Spindle调度系统中的最小调度单元（§3.4），代表一次并发执行——在一个wave内，来自不同MetaOp的多个切片（sliced ASL-tuples）分布在不同的固定device group上同时执行，且它们的执行时间跨度被对齐以避免device idle。Wave的关键特征：(1) wave内资源分配不变——各MetaOp slice的device数量在wave内固定；(2) 数据流传输仅发生在wave之间——wave内无跨MetaOp通信；(3) wave持续时间由时间对齐步骤确定——以wave内最短ASL-tuple执行时间为基准，截断其他tuple。Wave将连续最优MPSP解的"所有MetaOp同时开始同时结束"性质转化为离散可行的交错执行方案。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Wave的构造过程（Algorithm 1 with Fig. 5b, N=4为例）：
```
WavefrontSchedule(N, T_start, alloc_plan, remaining):
  T_current = T_start
  P = ∅  // final schedule
  
  while remaining not empty:  // craft wave k
    // Step ①: Propose Candidate Set
    S_cand = greedy_propose(N, remaining)
    // e.g., Wave 1: MetaOp1 ASL-tuple <n=4, l=2> (occupies all 4 devices using 2 ops)
    // Wave 2: MetaOp1 <4,9>, MetaOp2 <2,14>, MetaOp4 <2,3>
    
    // Step ②: Extend if needed
    S_cand = extend_resources_if_needed(N, S_cand)
    // e.g., Wave 4: extend MetaOp4 from n=1 to n=2 to fill idle device
    
    // Step ③: Align Time Span
    T_shortest = min_{tuple in S_cand}(exec_time(tuple))
    for each tuple in S_cand:
      // Slice tuple to fit T_shortest
      n_ops = ops_needed_for_time(tuple, T_shortest)
      tuple.scheduled_ops = n_ops
    T_wave = T_shortest  // wave duration
    
    // Step ④: Commit Wave
    for each tuple in S_cand:
      tuple.start_time = T_current
    S_k = S_cand; P = P ∪ S_k
    remaining = remaining - scheduled_ops_of(S_cand)
    T_current += T_wave
  
  return P, T_current
```
以图5b为例：Wave1中MetaOp1 [8 ops×2 ops]全设备执行→Wave2中MetaOp1 [8,·,9]截断为1 op、MetaOp2 [2,·,14]截断为2 ops、MetaOp4 [2,·,3]完整3 ops→时间对齐→Wave3-6类似。最终6个waves在时间轴上紧密排列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Wave概念在Spindle中作为execution plan的基本组成单元。每个wave被表示为时间片上的算子集合 {ASL-tuple → scheduled_operators}，由Wavefront Scheduler（~500 LoC Python）生成。在Runtime Engine执行时，各device按wave顺序逐一执行：当前wave内所有device完成MetaOp slice计算→NCCL send/recv传递wave间数据流→下一wave开始。Wave的time span alignment通过截断ASL-tuple中的operator数量实现——若原tuple包含l_m个operators，wave时长T_wave，则截取前⌈T_wave / T_m(n)⌉个operators在当前wave执行，剩余operators留在后续wave。这种切片-调度模式类似于pipeline parallelism中的micro-batch，但粒度更细（operator-level而非layer-level），且资源分配在wave间可以变化。

涉及论文标题：
- 63-Spindle- Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling.pdf

## Bi-point Discretized Allocation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bi-point Discretized Allocation是Spindle资源分配器（§3.3）中将MPSP连续最优解转化为实用离散分配方案的方法。MPSP连续解给出每个MetaOp m的最优GPU数n*_m（实数），但实际GPU数必须为整数且满足valid constraint（如Data Parallel时batch size整除、Tensor/Sequence Parallel时degree整除）。Bi-point Discretization用两个离散ASL-tuple ⟨n_m, ·, l_m⟩和⟨n_m, ·, l_m⟩线性表示连续最优分配，满足：(Cond 10a) L_m = l_m + l_m（覆盖所有operators）；(Cond 10b) C* = T_m(n_m)·l_m + T_m(n_m)·l_m（总执行时间保持为最优makespan C*）。选n_m, n_m为最接近n*_m的合法整数（n_m ≥ n*_m ≥ n_m），l_m和l_m由公式(10a)(10b)解得实数，最后round为整数。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bi-point Discretization算法（per MetaOp m）：
```
BiPointDiscretize(m, n*_m, C*, L_m, valid_allocations):
  // Step 1: Find nearest valid integer allocations
  n_high = min{ n ∈ valid_allocations | n ≥ n*_m }
  n_low  = max{ n ∈ valid_allocations | n ≤ n*_m }
  // If n*_m is already valid: n_high = n_low = n*_m
  
  // Step 2: Solve for operator counts l_high, l_low
  // From Cond (10a): L_m = l_low + l_high
  // From Cond (10b): C* = T_m(n_low)*l_low + T_m(n_high)*l_high
  // Solve linear system:
  l_high = (C* - T_m(n_low)*L_m) / (T_m(n_high) - T_m(n_low))
  l_low  = L_m - l_high
  
  // Step 3: Round to integers, discard zero-length tuples
  l_low  = round(l_low)
  l_high = round(l_high)
  result = []
  if l_low > 0:  result.append(ASL(n_low, l_low))
  if l_high > 0: result.append(ASL(n_high, l_high))
  
  // Step 4: Handle dummy allocations (n_low = 0)
  // If n*_m < 1, discretize as n_low=0 (dummy, ignored), n_high=1
  return result
```
以图5a中MetaOp2为例：n*_2 = 1.5, L_2 = 12, C*（由bisection得到）。Valid allocations: n ∈ {1,2,4,8,...}。n_low=1, n_high=2。解得l_high = 8.4, l_low = 3.6 → round → ASL(2, 8) + ASL(1, 4)。MetaOp3 (n*_3 = 0.6): n_low=0(dummy ignored), n_high=1 → 仅ASL(1, 6)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bi-point Discretization在Spindle中实现为Python函数，嵌入Resource Allocator模块。Valid constraint检查根据MetaOp的parallel strategy类型确定合法n值集合（DP要求n整除global batch size B_m；TP要求n可被TP degree整除）。Rounding引入的微小偏差（公式10b不严格成立）是Spindle与理论最优C*偏差<7%（图11）的部分来源。Bi-point discretization保证了每个MetaOp最多产生2个ASL-tuple，结合wavefront scheduler中每个wave消耗至少一个MetaOp的全部operators，确保总wave数≤2·|V_M|（线性于MetaOp数）。

涉及论文标题：
- 63-Spindle- Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling.pdf

## Scaling Curves with Piecewise α-β Modeling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scaling Curves with Piecewise α-β Modeling是Spindle的可扩展性估计器（§3.2）采用的方法，用于准确估计异构MetaOp在不同GPU分配量n下的执行时间T_m(n)。传统的统一α-β建模（如Alpa [87]、Galvatron [44]使用的Hockney模型[26]：T(n) = α + β/n）假设所有operators具有同构的scalability特征，在MT MM模型的不同MetaOp间因workload type（不同compute kernel）和per-device workload量差异而失效。Spindle的piecewise α-β approach对每个MetaOp独立profiling多个离散点(n_i, T_m(n_i))，用分段函数拟合——每段[n_i, n_{i+1}]内使用独立的α-β参数，适应不同GPU数区间内kernel行为的非线性变化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Spindle Scaling Curves的profiling和估计流程：
```
// Phase 1: Profiling (per MetaOp m, < 5 min total for all MetaOps)
ProfileMetaOp(m, model, cluster):
  datapoints = []
  dp_degrees  = [1, 2, 4, 8, 16, 32]  // discrete profiling points
  tp_degrees  = [1, 2, 4, 8]
  for n in valid_allocations(dp_degrees, tp_degrees):
    // Execute MetaOp m on n GPUs with optimal parallel config
    t_n = benchmark_forward_backward(m, n_devices=n)
    datapoints.append((n, t_n))
  return datapoints

// Phase 2: Piecewise α-β fitting
FitScalingCurve(datapoints):
  sorted_points = sort_by_n(datapoints)
  segments = []
  for i in range(len(sorted_points) - 1):
    n1, t1 = sorted_points[i]
    n2, t2 = sorted_points[i+1]
    // Fit α + β/n model in segment [n1, n2]
    // t1 = α + β/n1, t2 = α + β/n2
    β = (t1 - t2) / (1/n1 - 1/n2)
    α = t1 - β/n1
    segments.append((n1, n2, α, β))
  return PiecewiseFunc(segments)

// Phase 3: Estimation at arbitrary n
EstimateTime(m, n):
  seg = find_enclosing_segment(m.scaling_curve, n)
  return seg.α + seg.β / n
```
以图4中Multitask-CLIP为例：Task1-Text operator（lightweight）在n=4时T_m≈3ms，n=32时T_m≈1.5ms——scalability远低于线性（理想32/4=8×加速，实际仅2×）。而Task2-Vision operator（heavier）从n=1到n=32表现出接近线性的下降。分段α-β能捕捉到不同n区间内的scalability转折点（如某段内因kernel launch overhead占比变化导致的β突变）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Scaling Curves在Spindle中为每个MT MM模型的MetaOp集合profiling一次（<5分钟），存储在scalability estimator中供resource allocator调用。Profiling使用PyTorch distributed执行forward+backward benchmark，对每个(n, parallel_config)组合测量wall-clock execution time（含compute和通信）。分段α-β参数在求解MPSP的bisection search中被调用用于求T_m^{-1}(C*/L_m)（通过在各段内二分查找）。Piecewise modeling比单一α-β更精确地捕捉compute-bound到communication-bound的regime转换，但也要求更多的profiling datapoints。对于需要快速估计异构workload scalability的调度系统，这种分段方法是实用的折中方案。

涉及论文标题：
- 63-Spindle- Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling.pdf

## SM-aware CTA Scheduling (SM感知CTA调度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SM-aware CTA Scheduling 是 POD-Attention [ASPLOS '25] 提出的一种软件级 GPU kernel 内 CTA（Cooperative Thread Array）调度技术。其核心思想是：在 kernel launch 时不确定每个 CTA 执行何种操作（prefill 或 decode），而是让每个 CTA 在运行时——即已被 GPU 硬件 CTA scheduler 分配到某个 SM 后——通过读取其所在的 SM 硬件 ID（SMID），按预设 policy 决定执行 prefill 还是 decode attention。这解决了传统 GPU 并发执行方案（CUDA streams、CTA-parallel、warp-parallel）无法保证不同操作在同一 SM 内 co-locate 的根本问题。调度 policy 有两种：(1) 50:50——每个 SM 上 CTAs 交替执行 prefill 和 decode；(2) Proportional——按 prefill_CTAs 与 decode_CTAs 的全局比例分配每 SM 的执行比例。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SM-aware CTA Scheduling 的执行流程（对应 Figure 9 的 CUDA 伪代码）：

```
// Kernel Launch: total_CTAs = prefill_CTAs + decode_CTAs
// 每 SM 维护 counter array: sm_ctr[SM_ID], prefill_ctr, decode_ctr

__global__ void pod_attention_kernel(...) {
    if (threadIdx.x == 0) {  // Leader thread
        // Step 1: 读取所在 SM 硬件 ID
        int sm_id;
        asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id));
        
        // Step 2: Atomic increment SM counter, 获取 ticket
        const int ratio = prefill_ratio + decode_ratio;
        int ticket = (atomicAdd(&sm_ctr[sm_id], 1) % ratio);
        
        // Step 3: 根据 ticket 决定操作类型
        int op;
        if (ticket < prefill_ratio) op = PREFILL;
        else op = DECODE;
        
        // Step 4: 获取该操作内的 CTA ID
        int cta_id = atomicAdd(&cta_assign[op], 1);
        
        // Step 5: 若超出最大 CTA 数则切换操作
        if (op == PREFILL && cta_id >= prefill_ctas) {
            op = DECODE;
            cta_id = atomicAdd(&cta_assign[DECODE], 1);
        } else if (op == DECODE && cta_id >= decode_ctas) {
            op = PREFILL;
            cta_id = atomicAdd(&cta_assign[PREFILL], 1);
        }
        
        // Step 6: 写入 shared memory
        shared_mem[0] = cta_id;
        shared_mem[1] = op;
    }
    __syncthreads();
    int cta_id = shared_mem[0];
    int op = shared_mem[1];
    __syncthreads();
    
    if (op == PREFILL) prefill_attention(cta_id);
    else decode_attention(cta_id);
}
```

关键设计要点：(1) SMID 通过 PTX ISA `mov.u32 %0, %smid` 读取 NVIDIA GPU 特殊寄存器 `%smid`（自 Volta 架构引入）；(2) `atomicAdd(&sm_ctr[sm_id], 1)` 保证同一 SM 内多 CTA 并发调度的正确性；(3) Proportional policy 通过 `prefill_ratio` 使 scheduling 自适应不同 batch composition；(4) Overflow 保护——CTAs 超限时自动切换到另一操作。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SM-aware CTA Scheduling 基于 CUDA PTX ISA 的 `%smid` 特殊寄存器，该寄存器自 NVIDIA Volta 架构（SM 7.0+）开始提供。POD-Attention 在 FlashAttention v2.6.1 基础上实现，代码开源在 https://github.com/microsoft/vattention/tree/main/pod_attn。通用化实现需要：(1) 将操作编写为 device function，移除对 CUDA 内置 `blockIdx` 的依赖，改为参数传入；(2) wrapper kernel 启动 total_CTAs = sum(两种操作的 CTAs)；(3) kernel 内实现 SM counter array 的 atomicAdd 调度；(4) 根据 policy 和 batch composition 动态计算操作比例。该技术适用于任意需要 GPU SM 内 co-locate 两类异构操作的场景。

涉及论文标题：
- 64-POD-Attention- Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.pdf

## CTA-parallel Fusion (CTA级别并行融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CTA-parallel Fusion 是 GPU kernel 融合的一种粒度选择——在 Cooperative Thread Array（CTA，即 thread block）级别将两个或多个异构操作融合到单个 kernel 中，不同 CTA 执行不同操作。与 warp-parallel fusion（同一 CTA 内不同 warp 执行不同操作）和 intra-thread fusion（同一 thread 交替执行不同操作的指令）相比，CTA-parallel fusion 有三个关键优势：(1) 不同 CTA 可独立开始和完成，不受其他 CTA 进度的阻塞——避免 warp-parallel fusion 的 "straggler 问题"（一个慢 warp 延迟整个 CTA 完成，进而阻塞下一个 CTA 调度）；(2) barrier 影响限定在各自 CTA 内——`__syncthreads()` 不阻塞其他 CTA，避免了 intra-thread fusion 中 barrier 前后指令无法 overlap 的限制；(3) 编程相对简单——只需将不同操作编写为 device function，在 wrapper kernel 中根据 CTA ID 路由调用。CTA-parallel fusion 的代价是无法保证不同操作的 CTA 在同一个 SM 上 co-locate——这正是 POD-Attention 引入 SM-aware CTA scheduling 解决的核心问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CTA-parallel fusion 的实现范式（以 POD-Attention 为例）：

```
// Step 1: 将 prefill 和 decode attention 改造为 device function
//     移除对 blockIdx 的依赖，改为参数 cta_id
__device__ void prefill_attention(int cta_id, ...) {
    int q_tile_idx = cta_id / num_kv_splits;  // 原 blockIdx 映射
    ...
}
__device__ void decode_attention(int cta_id, ...) {
    int request_idx = cta_id / num_kv_heads_per_cta;
    ...
}

// Step 2: Wrapper kernel 启动 total_CTAs = prefill_CTAs + decode_CTAs
__global__ void fused_attention_kernel(...) {
    // SM-aware CTA scheduling 决定 op 类型和 cta_id
    int cta_id, op;
    __shared__ int smem[2];
    if (threadIdx.x == 0) {
        // ... scheduling logic (见 SM-aware CTA Scheduling) ...
        smem[0] = cta_id; smem[1] = op;
    }
    __syncthreads();
    cta_id = smem[0]; op = smem[1];
    __syncthreads();
    
    if (op == PREFILL) prefill_attention(cta_id, ...);
    else decode_attention(cta_id, ...);
}
```

对比其他融合粒度的性能（基于 POD-Attention micro-benchmark 和 attention kernel 实验）：
- Kernel-parallel (Streams): 无 SM 级 co-location 保证，平均仅 7% 提升。
- Warp-parallel (HFuse): straggler 问题导致 prefill-heavy 场景下性能退化达 13%。
- Intra-thread: CTA-level barrier 限制，平均 13% 提升但无法用于 attention kernel。
- CTA-parallel (POD-Attention): 峰值 59% speedup，均值 28%，永不比串行差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CTA-parallel fusion 的实现要点：(1) 将原始 kernel 重构为 `__device__` 函数，移除对 CUDA built-in `blockIdx`/`blockDim`/`gridDim` 的引用，替换为函数参数；(2) 确保两种操作的 shared memory 用量可调和——通过手动调整 tile size、register usage 或 Virtual CTA 技术平衡需求；(3) wrapper kernel launch 时指定 `total_CTAs = op1_CTAs + op2_CTAs`，shared memory = `max(op1_smem, op2_smem)`；(4) 运行时通过 scheduling 机制（SM-aware scheduling 或 persistent threads）决定每个 CTA 的操作类型和逻辑 CTA ID。该方法已在 POD-Attention 中验证，代码开源。通用化到其他 kernel 对时需注意两种操作的 tile size、register pressure 和 shared memory 兼容性。

涉及论文标题：
- 64-POD-Attention- Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.pdf

## Wave Quantization (波量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wave Quantization（波量化）是 GPU kernel 执行中的性能现象：当 kernel 的总 CTA 数量不能被 GPU 的 SM 数量整除时，最后一批（last wave）CTA 只能部分填充 SM，导致部分 SM 在该 wave 中空闲。例如，GPU 有 108 个 SM，kernel 启动 220 个 CTA——前两 wave 各 108 CTA 填满所有 SM，第三 wave 仅剩 4 CTA（220 - 2×108 = 4），104 个 SM idle。在 worst case 中，仅增加 1 个 CTA 就可使 latency 翻倍——因为多出的 CTA 触发了一个新的、几乎全空的 wave。Wave quantization 使性能对 CTA 数量呈阶梯状而非平滑变化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Wave Quantization 的数学描述（以 POD-Attention A100 实验为例）：

```
GPU: NVIDIA A100, N_SMs = 108
Decode batch size=55, 每 request 4 KV heads × 1 CTA/KV head → 220 CTAs

Wave 计算:
  num_waves = ceil(CTAs / N_SMs) = ceil(220/108) = 3

  Wave 1: 108 CTAs / 108 SMs (全满)
  Wave 2: 108 CTAs / 108 SMs (全满)
  Wave 3:   4 CTAs / 108 SMs (104 SMs idle)

Kernel latency ≈ 3 × max_wave_duration

对比 batch size=54 → 216 CTAs:
  num_waves = ceil(216/108) = 2
  Kernel latency ≈ 2 × max_wave_duration
```

POD-Attention 中 wave quantization 的实际影响：
- FA_Serial 在 decode batch size 54→55 时（CTA 216→220），decode attention 时间增加 >25%，total attention time 增加达 17%。
- Concurrent execution（POD-Attention）可通过填充 idle SMs 缓解——当一操作的 CTA 在 last wave 只占少数 SM 时，另一操作的 CTA 占据剩余 SM。
- 这解释了 FA_Streams 在有 wave quantization 时提升可达 20%，而无 wave quantization 时仅 ~5%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Wave quantization 是 GPU 执行模型的内在特性，缓解策略包括：(1) 选择 CTA 数量使 CTAs % N_SMs ≈ 0（通过调整 tile size 或 thread block size）；(2) 使用 concurrent kernel execution（如 POD-Attention）在 final wave 填充另一类型操作；(3) 使用 persistent threads 编程模型（如 CUTLASS Stream-K [44]）——启动恰好 N_SMs 个 CTA，每个 CTA 内部通过 work-stealing 动态获取 tile 执行，消除 wave 概念；(4) NVIDIA Hopper 架构的 Thread Block Cluster 提供更大粒度的 SM 分组调度。在 LLM 推理中，decode batch size 和 KV head 数的乘积决定 decode CTA 数量——当 batch size 在临界值附近波动时，wave quantization 可导致显著的 latency jitter。

涉及论文标题：
- 64-POD-Attention- Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.pdf

## Virtual Decode CTA (虚拟解码CTA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Virtual Decode CTA 是 POD-Attention 中用于平衡 fused kernel 内 prefill 和 decode 操作 shared memory 需求的技术。在 CTA-parallel fusion 中，kernel launch 时为所有 CTA 分配固定 shared memory = max(prefill_smem, decode_smem)。由于 prefill 需要大 tile（QSL tile=128）→ 大 shared memory，而 decode 用小 tile（QSL tile=16）→ shared memory 需求仅为 prefill 的约 1/4，若按 prefill 标准分配给 decode CTA 会造成 75% 的 shared memory 浪费。Virtual Decode CTA 的解决方案：将一个物理 decode CTA 拆分为多个 warp 粒度的虚拟 CTA（如 4 warps → 4 virtual CTAs），每个 virtual CTA 使用原始 shared memory 的 1/4，所有 virtual CTA 共享同一物理 CTA 的 shared memory 空间——使 decode 总 shared memory 使用量接近 prefill，避免过度分配。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Virtual Decode CTA 的实现流程：

```
// 原始 decode CTA: 4 warps (128 threads), shared_mem = 48KB
// 改造后: 4 virtual CTAs, 各 1 warp (32 threads), shared_mem = 12KB 各

// 关键修改: CTA-level barrier → warp-level barrier
// Before: __syncthreads() 同步所有 4 warps
// After:  __syncwarp()   仅同步当前 warp 的 32 threads

__device__ void decode_virtual_cta(int virtual_cta_id, ...) {
    int warp_id = virtual_cta_id;
    int warp_tile_offset = warp_id * (tile_size / 4);  // 各自的 shared mem 段
    
    // 独立加载 K/V tile 到 warp 专属 shared memory 段
    for (int i = threadIdx.x % 32; i < tile_size / 4; i += 32)
        smem[warp_tile_offset + i] = K[warp_tile_offset + i];
    __syncwarp();  // 仅同步当前 warp
    
    // 使用自己的 shared memory 段计算 attention
    ...
}

// Wrapper kernel 中:
//   物理 CTA decode → 4 virtual CTAs:
//     virtual_cta=0 (warp 0): 处理 requests 0-15
//     virtual_cta=1 (warp 1): 处理 requests 16-31
//     virtual_cta=2 (warp 2): 处理 requests 32-47
//     virtual_cta=3 (warp 3): 处理 requests 48-63
```

Shared memory 平衡效果：无 Virtual CTA 时 decode CTA 占用与 prefill 相同的 48KB 但实际仅需要 12KB（浪费 36KB/CTA）；有 Virtual CTA 后 4 virtual CTAs 各自使用 12KB，总量 48KB = prefill 需求，消除浪费。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Virtual Decode CTA 的实现需要：(1) 将原始 attention kernel 的 `__syncthreads()`（CTA-level barrier）替换为 `__syncwarp()`（warp-level barrier），确保每个 warp 的数据加载和计算独立；(2) decode device function 接收 `virtual_cta_id` 参数定位其处理的 decode requests 子集和 shared memory 段；(3) wrapper kernel 中，根据物理 CTA ID 计算 virtual CTA 范围；(4) 注意 `__syncwarp()` 仅同步 warp 内 32 threads 且要求 warp 内无 divergence。该技术适用于任何 fused kernel 中一方的 shared memory 需求远小于另一方的情况。

涉及论文标题：
- 64-POD-Attention- Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.pdf

## Instruction Chaining for In-Cache Computing (存内计算的指令链接技术)

术语解释
MagiCache提出的硬件技术：将无冲突的连续vector指令打包为group，允许各fused array独立异步执行group内全部指令，仅group间进行同步。

术语是什么？
Instruction Chaining是MagiCache对现有同步执行模式（所有computing arrays必须同步完成每条指令后才执行下一条）的改进。灵感来源于VLIW架构（如Duality Cache的compiler-managed VLIW）使不同computing arrays可执行不同操作，但MagiCache采用硬件实现的轻量级方案——无需复杂compiler设计。核心机制：
1. **Conflict Detection**：Virtual engine运行时检测三类冲突：(a) Configuration指令（vset(i)vl(i)）改变全局vector状态（vl/vtype）；(b) Permutation指令（slide/gather等）跨array移动数据元素；(c) Store指令的address range与其它memory指令交叉——可能导致data hazard。例外：两memory指令的address range完全相同时（如unit-stride load/store同base+stride），各array处理不重叠的address sub-range→无hazard。
2. **Group Formation**：无冲突的连续指令被推入同一group；检测到冲突时，插入synchronization pseudo-instruction作为group边界，后续指令进入新group。
3. **Asynchronous Execution**：每个fused array的array sequencer独立执行group内全部指令序列——Array 0完成其segment的load后可立即开始compute，不等Array 3的load miss完成。仅当某array到达group边界（synchronization pseudo-instruction）时需等待所有其他arrays也到达边界。
4. **Instruction Retirement**：一条指令只有在所有arrays都完成该指令后才可在MagiCache中retire（确保精确异常）。

从kernel调度角度拆解术语
Instruction Chaining的执行流程（以matmul的vle32.v v1 → vle32.v v0 → vmacc.vx → vse32.v为例，Chain-4配置, N=8 arrays）：
```
// Group formation by Virtual Engine:
当前group = []
指令1: vle32.v v1, (a2)  // no conflict → push to group
指令2: vle32.v v0, (a1)  // same base+stride as 指令4? check → same → no hazard
                          // no config/permutation → push to group
指令3: vmacc.vx v0, a5, v1  // no conflict → push to group
指令4: vse32.v v0, (a1)  // addr range = 指令2 range → no hazard → push to group
// End of iteration, group被提交.

// Per-Array异步执行 (伪代码):
for each fused array A in [0..7]:
    // Array A处理 segment_A 的elements
    load segment_A of v1 from memory     // 可能 MSHR stall + cache miss
    load segment_A of v0 from memory     // 不等其他arrays完成v1 load
    compute: segment_A of v0 += segment_A of v1 * scalar_a5  // 11 cycles
    store segment_A of v0 to memory
    barrier_sync()  // group boundary, 等所有arrays完成
```

对比同步执行（SplitCache）：
```
// 同步执行 (SplitCache):
for each instruction I in [vle32.v v1, vle32.v v0, vmacc.vx, vse32.v]:
    for each computing array A in parallel:
        execute I on segment_A
    barrier_sync_all_arrays()  // 每条指令后sync → 累计sync stall大
```

效果：同步时间减少45.3%（Fig. 9），average memory access time减少2%-27%（因不同array的load miss latency被compute overlap）。但strided access（如k-means/backprop）因无法coalesce requests → fused arrays近乎串行执行 → chaining效果减弱。含cross-element指令（jacobi/pathfinder的slide）的应用无法chaining → Chain-1比Fused-1性能略降1%（因异步导致的memory access不连续）。

术语一般如何实现？如何使用？
硬件实现：(1) 扩展array sequencer支持独立执行多指令序列；(2) virtual engine增加conflict detection逻辑（configuration/permutation检测直接看opcode；store address冲突通过比较address range register判断）；(3) 新增synchronization pseudo-instruction和group ID tracking。对上层软件透明——无需修改RVV程序或compiler。限制：chaining仅适用于无上述三类冲突的指令序列；当应用中大量存在permutation指令时，chaining收益降低。

涉及论文标题：
- 65-MagiCache- A Virtual In-Cache Computing Engine.pdf

## FFA (Find-First-Available) Allocation Policy (首次可用分配策略)

术语解释
MagiCache virtual engine中用于在fused array内寻找候选cacheline以分配给vector register segment的硬件分配策略。

术语是什么？
FFA是MagiCache为cacheline-level动态分配设计的分配策略。与cache replacement policy（在2-16 ways的同一set内选victim）不同，FFA在256个cachelines的fused array中寻找候选。FFA的工作方式：
1. 从随机起始位置开始circular scan（避免总是从row 0开始导致front rows被过度分配）。
2. 每cycle扫描32个连续的tag states（读取computing bit + valid bit），最多8 cycles完成一个array的扫描。
3. 优先级：Free cacheline（valid bit=0，该cacheline当前未存储任何数据）> Available cacheline（computing bit=0且valid bit=1，该cacheline存储数据但可被替换）。
4. 选到后立即返回。若256 rows均无free/available→等待（但lazy initialization + low occupancy使此情况极少发生）。
5. 设minimum associativity threshold：当某set的available associativity降至阈值以下时，FFA不再从此set选cacheline→确保每个set保持足够的available cacheline。

从kernel调度角度拆解术语
FFA在执行中的调度流程：
```
Algorithm: FFA_Allocate(array_id, set_threshold):
    start_pos = random(0, H-1)  // H=256 rows
    for i in [0, H-1]:
        row = (start_pos + i) % H
        tag = array[array_id].tags[row]
        if tag.valid == 0:        // free cacheline
            if set_available_ways[tag.set] > set_threshold:
                return row
        elif tag.computing == 0:   // available cacheline
            if set_available_ways[tag.set] > set_threshold:
                return row
        // 每32个rows检查一次, 最多8个cycles
    return FAIL  // all rows occupied or threshold-blocked
```

FFA vs 传统replacement policy (LRU/pseudo-LRU) 的关键差异：
- LRU/pseudo-LRU：在2-16 ways选victim → 需维护access timestamp或tree-based pseudo-LRU bits per set → 额外硬件状态
- FFA：在256 rows选candidate → 仅需扫描现有tag bits（valid + computing）→ 无额外状态维护 → 硬件开销更低
- 性能代价：FFA miss rate增加<1%（实验证明），因为256候选远大于2-16→即使简单策略也有足够选择空间

术语一般如何实现？如何使用？
纯硬件实现：FPGA/ASIC中，FFA逻辑作为virtual engine的一部分。每个fused array一个FFA scan unit（并行到其他arrays）。每个cycle读取连续32个tag entries（tag array通常按row组织便于burst read）→ comparator chain判断free/available→优先encoder选第一个free→若无free选first available。整个scan过程2-8 cycles完成，不在critical path上（因为vector register初始化通常只在首iteration执行一次）。set_threshold同样硬件维护：每set一个counter跟踪available ways数量，分配时decrement，释放时increment。

涉及论文标题：
- 65-MagiCache- A Virtual In-Cache Computing Engine.pdf

## Lazy Initialization for In-Cache Vector Registers (缓存内向量寄存器的延迟初始化)

术语解释
MagiCache的寄存器管理策略：仅在vector register被指令实际使用时才为其分配computing lines（即物理cacheline），未使用的register及segment不占用任何cache空间。

术语是什么？
Lazy Initialization对比于SplitCache/EVE的eager pre-configuration——后者在启动时将computing arrays的所有256 rows全部预分配给32个vector registers（每register固定8 rows），无论这些registers是否会被使用。Lazy initialization的工作方式：
1. **触发条件**：当vector指令的source/destination register在VRMT中对应的segment entry valid bit为0（即该segment尚未分配物理computing line）时，触发初始化。
2. **初始化过程（Algorithm 1）**：(a) 计算需初始化的segment数 = effective vector length / elements_per_segment；(b) 对每个待初始化segment j，在对应fused array (j%N) 中执行FFA找候选cacheline；(c) 若cacheline dirty→evict to next level cache；(d) convertToComputingLine（4步：evict→clear bits→LRU invalid→computing bit=1）；(e) 填充VRMT entry（valid=1, index=RowIndex）。
3. **有效向量长度变化**：vset(i)vl(i)改变vl时——vl增加→为新segment初始化；vl减少→释放多余segment（clear VRMT→clear computing bit→恢复为cacheline）。
4. **释放时机**：通过liveness analysis预提取register生命周期，在生命周期结束时插入vsetvli zero, zero指令→virtual engine检测vl=0→执行release。

从kernel调度角度拆解术语
Lazy Initialization的执行流程（以matmul首次iteration为例）：
```
// 程序代码:
vsetvli t0, a4, e32, m1, tu, mu   // vl = 512 (for Chain-4)
vle32.v v1, (a2)                    // 首次使用 v1
vle32.v v0, (a1)                    // 首次使用 v0
vmacc.vx v0, a5, v1
vse32.v v0, (a1)

// 运行时行为:
执行vsetvli: vl=512, Segments = 512/16 = 32  // 每segment 16个32-bit elements (512/32=16)
// v1/v0尚未初始化 → VRMT entries全invalid

执行vle32.v v1:
  Virtual Engine检测VRMT[v1][0..31]全invalid → 触发初始化
  for j in 0..31:
    array = j % 8
    row = FFA_Allocate(array, threshold)
    if dirty: evict(row)
    set computing bit = 1
    VRMT[v1][j] = (valid=1, index=row)
  // 8 arrays并行分配, <8 cycles total

执行vle32.v v0:
  Virtual Engine检测VRMT[v0][0..32]全invalid → 同上初始化
  // 但v1空间未被释放, cache utilization = 2 registers' segments out of 512 total rows
  // = 2×32/512 = 12.5% (远低于SplitCache固定50%)

... 后续iterations:
  v0/v1的VRMT entries已valid → 不需要重新分配
  每次iteration结束: vsetvli zero, zero → release v0/v1 → cache空间恢复
  下iteration开始: vsetvli t0, a4 → 重新分配v0/v1 → 可能与上次相同/不同rows
```

对比eager pre-configuration（SplitCache）：
- SplitCache: 255 rows/array分配给32 registers（每register 8 rows）→ 256×8=2048 rows used → cache utilization ~56%（即使仅用2/32 registers）
- MagiCache lazy: 仅2×32=64 rows allocated for v0+v1 → 其余192 rows still available as cachelines → cache utilization ~97%

术语一般如何实现？如何使用？
硬件要求：VRMT支持per-entry valid bit + FFA allocation unit + liveness analysis pass（编译器集成/外部预处理）。对软件的影响：需要在编译时或BIOS中运行liveness analysis确定release point并插入release指令（vsetvli zero, zero）。MagiCache实验中release指令overhead <0.5%。Context switch配合：只保存vreg_valid CSR中标记为valid的registers→保留lazy initialization的空间节省效果。

涉及论文标题：
- 65-MagiCache- A Virtual In-Cache Computing Engine.pdf

## MBGMM (Multi-size Batched Gather Matrix-Matrix Multiplication)

术语解释
MBGMM 是 S-LoRA 使用的 CUDA kernel，用于在 multi-adapter LLM serving 中高效批量处理不同 rank 的 LoRA adapter 计算。它将 batch 内多个请求的不同 rank adapter 的 LoRA 两层矩阵乘法（shrink: v=xA + expand: y=vB）统一到一次 kernel 调用中，通过 gather 操作从 non-contiguous adapter weight memory 中按 adapter ID 索引对应矩阵块，实现 heterogeneous batching。

术语是什么？
MBGMM kernel 解决的问题：在一个 batch 中，不同请求可能使用不同 rank 的 LoRA adapter（rank 8/16/32/64/128），对应适配器矩阵 A∈R^(r×d_in)、B∈R^(d_out×r) 尺寸不同。传统方案需为每个 adapter rank 单独 launch kernel（多次 kernel launch overhead + 小矩阵 tensor core 利用率低）。MBGMM 将所有 adapter 的 LoRA 计算统一为一个 batched gather GEMM：对 shrink step (v_i = x_i × A_adapter(i)^T)，kernel 将 batch 内所有 inputs x_i 拼为 batch 维度，用 gather 索引到 unified memory pool 中各自 adapter 的 A 矩阵（不同 rank 的 A 可能在内存中不连续），一次 kernel 调用完成所有 shrink 计算；expand step (y_i = v_i × B_adapter(i)) 类似处理。结果与 base model 的 `W_base @ x` 相加得到最终 `h_i = W_base @ x_i + B_adapter(i) @ (A_adapter(i) @ x_i)`。

从 kernel 调度角度拆解：
伪代码（以 shrink step v = x @ A^T 为例，基于 S-LoRA BGMV kernel + Punica SGMV kernel）：
```
// Inputs:
//   x[batch_size, d_in] — batched inputs (token activations)
//   adapter_ids[batch_size] — adapter index per request
//   adapter_pool — unified memory pool storing A matrices for all adapters
//     (non-contiguous: A matrices of different ranks interleaved with KV cache)
//   adapter_offsets[num_adapters] — byte offset of each adapter's A in pool
//   adapter_ranks[num_adapters] — rank of each adapter
//
// Output: v[batch_size, max_r] — batched low-rank projection result

// SGMV-shrink: v = gather(x) @ A^T (per-adapter A matrix)
// GPU grid: blockIdx.x = adapter_id, blockIdx.y = (unused or rank_slice)
dim3 grid(num_adapters, 1);
dim3 block(256);  // threads per block

__global__ void sgmvin_shrink(
    float* v, const float* x, const float* adapter_pool,
    const int* adapter_offsets, const int* adapter_ranks) {
  
  int adapter_id = blockIdx.x;
  int r = adapter_ranks[adapter_id];
  const float* A = adapter_pool + adapter_offsets[adapter_id]; // A[r][d_in]
  
  // Each block computes v[adapter_id][0:r] = x[adapter_id] @ A^T
  // Tile over d_in dimension (typical: d_in=4096, r=8~128)
  __shared__ float tile_x[TILE_SIZE];
  __shared__ float tile_A[TILE_SIZE][r];
  
  float acc[4] = {0}; // up to r=128 / thread
  for (int tile = threadIdx.x; tile < d_in; tile += blockDim.x * TILE_SIZE) {
    // Cooperative load x tile and A tile into shared memory
    load_tile(tile_x, x + adapter_id * d_in + tile);
    load_tile(tile_A, A + tile * r);
    __syncthreads();
    
    // Tensor core MMA (if using wmma/mma instructions)
    for (int k = 0; k < TILE_SIZE; k++) {
      for (int ridx = 0; ridx < r; ridx++) {
        acc[ridx] += tile_x[k] * tile_A[k][ridx];
      }
    }
    __syncthreads();
  }
  
  // Store
  for (int ridx = 0; ridx < r; ridx++) {
    v[adapter_id * max_r + ridx] = acc[ridx];
  }
}

// SGMV-expand: y += gather(v) @ B (per-adapter B matrix)
// Similar tiling strategy over d_out dimension
__global__ void sgmvin_expand(
    float* y, const float* v, const float* adapter_pool,
    const int* adapter_offsets, const int* adapter_ranks) {
  
  int adapter_id = blockIdx.x;
  int r = adapter_ranks[adapter_id];
  const float* B = adapter_pool + adapter_offsets[adapter_id] + r * d_in * sizeof(float);
  // B[d_out][r] = adapter_pool[offset_of_A + r*d_in : offset_of_A + r*d_in + d_out*r]
  
  // Compute y[adapter_id] += v[adapter_id] @ B^T
  // Tiling over d_out
  for (...) {
    // similar cooperative tiling + tensor core MMA
  }
}
```
实际实现更复杂：S-LoRA 的 BGMV kernel 扩展 Punica SGMV 支持多 rank batch——blockIdx.y 绑定到 rank slice（当不同 adapter rank 差异大时，一个 block per adapter 可能导致 load imbalance），使用 Triton 实现 prefill 阶段 tiling。关键优化：(1) shrink step 中 `v = xA` 用 d_in 维度 tiling（d_in=4096 大），因 A 矩阵每行仅 r 个元素（低秩），tile 沿 d_in 分割；(2) expand step 中 `y = vB` 用 d_out 维度 tiling（d_out=4096 大），B 矩阵每列仅 r 个元素；(3) WARP tile 映射策略：不同 adapter 的 block 映射到不同 SM，同一 adapter 内 threads 协作完成 shrinked r 维度的 reduction。

术语一般如何实现？如何使用？
MBGMM 源于 S-LoRA 开源代码（https://github.com/S-LoRA/S-LoRA），BGMV kernel 通过 PyTorch CUDA extension 或 Triton 实现。使用上对用户透明——S-LoRA 的 inference engine 在遇到 batch 内不同 adapter 时自动调用 BGMV/MBGMM kernel。Chameleon 复用该 kernel 不修改——其 Adapter Cache 和 Scheduler 在 kernel 以上层次优化（减少 PCIe fetch 和优化 batch composition），但底层 LoRA 计算 pipeline 与 S-LoRA 相同。类似工作在 Punica (SGMV kernel)、vLLM LoRA support 和 Unsloth (fused LoRA MLP/QKV/W kernels) 中有不同实现。

涉及论文标题：
- 66-Chameleon- Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments..pdf

## SGMV (Segmented Gather Matrix-Vector Multiplication)

术语是什么？
SGMV（Segmented Gather Matrix-Vector Multiplication，分段聚合矩阵-向量乘法）是 Punica [6] 提出的 CUDA kernel，用于在 Multi-LoRA LLM serving 中高效 batched 处理使用不同 LoRA adapter 的多个 query。由于 Multi-LoRA serving 中 batch 内不同 query 可能使用不同 LoRA adapter（不同 A/B 矩阵、不同 rank），传统 GEMM（需要所有 query 使用相同的权重矩阵）无法直接处理。SGMV 将计算分解为两步：(1) **SGMV-shrink**：`v_i = x_i @ A_i`，将高维 input x_i（d_model）投影到低秩空间（rank r_i），不同 query 使用各自 adapter 的 A_i 矩阵；(2) **SGMV-expand**：`y_i = v_i @ B_i`，将低秩中间结果 v_i 扩展回高维空间（d_model），使用各自 adapter 的 B_i 矩阵。两步中通过 gather 操作从 GPU memory 中按 adapter ID 索引对应权重块（不同 adapter 的 A/B 矩阵在内存中非连续存储），实现 batch 内异构 adapter 的统一 kernel 执行。

从kernel调度角度拆解：
SGMV kernel 的 CUDA 执行模型（以 batch 内含 3 个 query，分别使用 rank=32、64、16 的 LoRA 为例）：
```
# 输入：
# x_batch: [batch_size, d_model]  # 如 [3, 4096]
# adapter_ids: [0, 1, 2]  # 各 query 使用的 adapter
# A_pool: {adapter_0: [r0, d_model], adapter_1: [r1, d_model], ...}  # 非连续存储
# B_pool: {adapter_0: [d_model, r0], adapter_1: [d_model, r1], ...}

# Step 1: SGMV-shrink (x @ A^T)
for each query i in batch (parallel on different thread blocks):
    adapter_id = adapter_ids[i]
    A_i = A_pool[adapter_id]  # gather: [r_i, d_model]
    r_i = A_i.shape[0]  # adapter rank
    v_i = zeros(r_i)  # [r_i]
    # Matrix-vector: v_i = x_i @ A_i^T
    for k in range(d_model):
        v_i[:] += x_i[k] * A_i[:, k]  # or tiled
    store v_i in intermediate buffer

# Step 2: SGMV-expand (v @ B)
for each query i in batch (parallel):
    B_i = B_pool[adapter_ids[i]]  # gather: [d_model, r_i]
    y_i = zeros(d_model)  # LoRA output, added to base model output
    # Matrix-vector: y_i = v_i @ B_i^T  →  y_i[j] = Σ v_i[k] * B_i[j,k]
    for k in range(r_i):
        y_i[:] += v_i[k] * B_i[:, k]  # or tiled

# 最终输出：h_i = W_base @ x_i + y_i  # 加上 base model 的输出
```
SGMV 的关键优化：(1) **Segmentation by adapter**：相同 adapter 的 query 合并为子 batch，共享一次 adapter weight load（减少 HBM 访问）；(2) **Gather with index mapping**：通过 adapter_id → memory offset 的映射表在 kernel 内间接索引非连续 adapter weights，消除显式 memory copy；(3) **Tiling along d_model dimension**：d_model 维度（如 4096）远大于 rank（如 32），tiling 沿 d_model 分割以适配 shared memory 和 register file。与标准 GEMV 相比，SGMV 的 arithmetic intensity 更低（仅 r_i × d_model 乘加而非 d_model × d_model），因此为 memory-bound——性能受限于 HBM bandwidth 而非 compute throughput。

术语一般如何实现？如何使用？
SGMV 最初由 Punica (https://github.com/punica-ai/punica) 提出并使用 CUDA C++ 实现，后被 S-LoRA (https://github.com/S-LoRA/S-LoRA) 扩展为 BGMV/MBGMM（支持更灵活的 multi-rank batch 和 prefill tiling），vLLM 通过 punica_wrapper 集成。在 FAIR LIBRA (ELORA) 中，SGMV kernel 本身不被修改——ELORA 的优化在 kernel 以上层次（通过 unified caching pool + dependency tree 确保 batch 内所有 query 所需 LoRA 已在 HBM 中，消除 kernel 执行前的 PCIe loading 延迟）。SGMV 使用上对用户透明——serving framework 在构建 batch 时根据各 query 的 adapter ID 自动调用对应 kernel。

涉及论文标题：
- 67-ELORA- Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving.pdf

## Variable-Speed Systolic Array Execution (变速脉动阵列执行)

术语解释
Variable-Speed Systolic Array Execution（变速脉动阵列执行）是SPARK提出的一种支持混合精度MAC在统一PE阵列中高效执行的调度机制。由于SPARK encoding产生两种精度的operand（4-bit低精度和8-bit高精度），它们的MAC计算耗时不同（1 cycle vs 2-4 cycles），PE阵列需要以"变速"方式执行——低精度值全速流过，高精度值在特定PE中暂停（stall）多cycle完成计算，其余PE同步等待以维持systolic array的pipeline节奏。这种设计避免了为不同精度使用分离的PE阵列，实现了灵活性与硬件效率的平衡。

术语是什么？
Variable-Speed Systolic Array Execution的核心挑战：在systolic array中，数据按固定节奏在PE间传播（每cycle右移activation、下移partial sum）。当某些PE需要额外cycle完成高精度计算时，必须让整行/整列的PE同步等待（stall），否则会导致数据错位。SPARK的解决方案：
- 默认全速模式（INT4×INT4）：每PE每cycle完成1个MAC，无stall
- 局部变速模式（涉及高精度operand）：高精度PE耗时2或4 cycles，同row/column的其他PE插入对应stall（1或3 cycles）
- Stall传播：通过enable chain在PE间传播stall信号，维持数据同步
- 总体效率：由于~80%操作为INT4，仅少数cycle有stall，整体吞吐接近全INT4阵列

从kernel调度角度拆解：
Variable-Speed execution在SPARK 4×4 MPE array上的具体调度（图9c为例）：

```
// 4 PEs (PE00, PE01, PE02, PE03) compute A[1×4] × B[4×4]
// A = [a0,0 a0,1 a0,2 a0,3]
// B = [b0,0..b0,3; b1,0..b1,3; b2,0..b2,3; b3,0..b3,3]

// operand precision: a0,0(HP), b0,0(HP), others(LP)
// HP = 8-bit, needs 4 cycles for HP×HP; LP = 4-bit, 1 cycle

Cycle 1:  PE00: HP×HP start ────  PE01: LP×LP ────  PE02: idle ────  PE03: idle
Cycle 2:  PE00: HP×HP cont       PE01: stall       PE02: LP×LP    PE03: idle
          // PE01 stalls 1 cycle waiting for PE00's first operand
Cycle 3:  PE00: HP×HP cont       PE01: LP×LP       PE02: idle     PE03: LP×LP
Cycle 4:  PE00: HP×HP done       PE01: stall       PE02: LP×LP    PE03: idle
          // PE01 stalls 1 cycle: PE00's HP×HP still producing partial
Cycle 5:  PE00: LP×LP            PE01: LP×LP       PE02: idle     PE03: LP×LP
          // PE01 also high precision input arrives, needs 2 cycles
Cycle 6:  PE01: HP×LP start      PE00: LP×LP       PE02: LP×LP    PE03: idle
          // PE01 needs 2 cycles for mixed-precision compute
Cycle 7:  PE01: HP×LP cont       PE00: stall       PE02: LP×LP    PE03: LP×LP
          // PE00 stalls 1 cycle to sync with PE01
Cycle 8-19: ... similar pattern ...

// Total: ~19 cycles for 4×4 GEMM (vs 4+2=6 cycles for pure INT4)
// Overhead: ~3.17× cycle increase, but saving 50% storage and transfer
// Net speedup vs INT8 baseline: up to 4.65×
```

关键调度规则：
1. **Stall条件**：某PE需要>1 cycle完成当前MAC时，同行的后续PE和同列的下游PE插入stall
2. **Stall时长**：INT4×INT8需1 stall cycle，INT8×INT8需3 stall cycles
3. **利用率**：由于>80% operands为INT4，stall cycle占比低，整体PE利用率高
4. **无额外硬件**：stall通过现有的pipeline control logic实现

术语一般如何实现？如何使用？
Variable-Speed Systolic Array Execution的实现方式：
1. **Stall logic**：每个PE有stall_in和stall_out信号，stall_in=1时PE暂停当前cycle的data movement和computation。Stall沿systolic array的行和列方向传播。
2. **Precision detection**：PE通过SPARK decoder输出的enable信号和identifier bit自动判断当前operands精度，决定需要的cycle数。
3. **Controller**：Central controller不直接控制每个PE的精度模式——精度由数据本身（identifier bit）驱动，controller仅管理array-level的scheduling。
4. **与DNN结构的关系**：由于SPARK encoding已按value magnitude分配精度，operator在dataflow中的精度分布符合统计规律（Gaussian分布），调度行为可预测。
5. 该方案适用于任何支持混合精度计算的systolic array架构，不限于SPARK的编码方案。

涉及论文标题：
- 68-SPARK_Scalable_and_Precision-Aware_Acceleration_of_Neural_Networks_via_Efficient_Encoding.pdf

## Pipelined Task Execution in Micro-Kernels（微内核流水线任务执行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pipelined Task Execution是MikPoly中micro-kernel在accelerator PE上执行的流水线模型。每个micro-kernel K~在reduction loop中的执行被组织为pipelined task，分三阶段流水线：(1) **Load阶段**：从M_global（如GPU global memory）加载数据到M_local（如GPU shared memory）；(2) **Compute阶段**：PE使用micro-kernel在M_local上执行tiled computation（如Tensor Core GEMM）；(3) **Store阶段**：将计算结果从M_local写回M_global。通过流水线技术，同一pipelined task内的多个micro-kernel实例可重叠load/compute/store操作（类似双缓冲/异步copy）。中间结果（如partial sums）保持在M_local中，减少M_global访问流量。MikPoly为每个micro-kernel在单PE上离线学习pipelined task的执行时间模型g_predict(t,K~,H)，其中t是pipelined task中micro-kernel的实例数。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以GPU A100上GEMM micro-kernel (uM=256, uN=128, uK=32) 的pipelined task执行为例：
```
// pipelined task for reduction loop K:
// 执行 t = K/uK = 4096/32 = 128 个micro-kernel实例
for k_tile in range(0, K, uK):  // 128 iterations
    // Stage 1: Load (async copy from global to shared memory)
    cp_async(A_tile, global_A + k_tile_offset)
    cp_async(B_tile, global_B + k_tile_offset)
    __syncthreads()

    // Stage 2: Compute (on Tensor Cores, using shared memory)
    for m_tile in range(0, uM, 64):
        for n_tile in range(0, uN, 64):
            mma_sync(accum[m_tile:n_tile], A_tile[m_tile], B_tile[n_tile])

    __syncthreads()
    // (Pipeline: next iteration's Load can overlap current Compute)

// Stage 3: Store (write accum to global memory)
store(global_C, accum)
```
MikPoly通过分段线性函数g_predict学习t（micro-kernel实例数）与总执行时间的关系——在t较小时memory latency主导，t增大时compute和memory逐渐balance。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 流水线通过CUDA async copy (cp_async)实现load/compute overlap，类似CUTLASS的pipelining策略。(2) g_predict的学习：offline阶段在单PE（单SM）上以t=1,2,4,...,5120运行pipelined task并测量时间，拟合分段线性函数（每个分段对应不同compute/memory行为区间）。(3) online阶段，f_pipe = g_predict(f_num(R_i,K~_i), K~_i, H)直接查表/计算，无需实际执行。(4) 该模型抽象了GPU shared memory、register file、Tensor Core的使用模式，对NPU的L1 buffer/L0 buffer/Cube Unit同样适用。

涉及论文标题：
- 69-Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization.pdf

## Load Imbalance Mitigation via Multi-Micro-Kernel（多微内核负载均衡）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是MikPoly通过multi-micro-kernel polymerization解决GPU/NPU上因shape不匹配导致的load imbalance问题的机制。当operator shape使单一micro-kernel产生的thread block数不是SM数的整数倍时，最后一批wave中的SM会underutilized（部分SM闲置），显著降低sm_efficiency。MikPoly通过将operator的computation分割为多个region，为不同region分配不同尺寸的micro-kernel，从而调整总thread block数和wave分配，使负载更均衡。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multi-micro-kernel GEMM-AB的wave分配计算：
```
// GEMM-A (single micro-kernel): M=4096, N=1024, K=4096
// (uM=256, uN=128, uK=32), 8 warps per thread block
f_parallel = (4096/256) × (1024/128) = 128 pipelined tasks
total_warps = 128 × 8 = 1024
waves = ceil(1024 / 864) = 2
// A100: 108 SMs × 8 active warps/SM = 864 max warps/wave
// Wave 2: 1024 - 864 = 160 warps (20 blocks) → 88 SMs idle
// sm_efficiency = 58.90%

// GEMM-AB (two micro-kernels):
// A: (uM=256, uN=128, uK=32), 8 warps/block
//     f_parallel_A = (3072/256) × (1024/128) = 96, warps_A = 768
// B: (uM=64, uN=64, uK=64), 4 warps/block
//     f_parallel_B = (1024/64) × (1024/64) = 256, warps_B = 1024
// total_warps = 768 + 1024 = 1792
// waves = ceil(1792 / 864) = 3, 最后一wave仅64 warps
// sm_efficiency = 96.06%
```
当t_A > 2×t_B（t为pipelined execution time），GEMM-AB总时间 = t_A + 2×t_B < 2×t_A = GEMM-A时间，实现加速。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) Polymerization patterns（特别是Pattern II及以上）自动探索multi-region分解，cost model中的f_wave因子自动捕获wave数效应。(2) 不同micro-kernel尺寸(uM,uN,uK)对应不同warp配置（如uM=256→8 warps/block，uM=64→4 warps/block），允许在region间trade-off并行粒度。(3) 该机制在online polymerization中自动运行——cost model的f_wave×f_pipe计算自然偏好wave利用率高的micro-kernel组合。

涉及论文标题：
- 69-Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization.pdf

## Expert Parallelism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Parallelism 是一种专为 Mixture-of-Experts (MoE) 模型设计的分布式部署策略，将 MoE block 中的 Expert FFN 参数按 Expert 维度分割（每个 GPU 存储一部分 Experts 的完整参数），非 MoE 参数（self-attention、layer norm 等）在所有 GPU 上复制。推理时，Gate Function 在所有 GPU 上独立计算→选出需要激活的 Experts→如果选中的 Expert 不在本地 GPU，则通过 all-to-all communication 将对应 token 发送到持有该 Expert 的 GPU→在目标 GPU 执行 expert FFN→再将结果 all-to-all 通信回原 GPU。Expert Parallelism 解决了 MoE 模型总参数量远超单 GPU 容量的问题（多 GPU 聚合内存可存储全部 Experts），但由于 MoE 的稀疏激活特性（如 top-1 of 128 仅激活 0.8%），多数 GPU 在给定请求下可能仅有极少或零个 expert 被激活，导致 GPU compute utilization 极低、TCO 高。Expert Parallelism 主要用于 MoE 训练和部分推理场景。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Parallelism kernel 调度过程（4 GPU 部署 Switch-Base 64 experts, top-1）：

```
# Expert分布: GPU0: experts[0..15], GPU1: experts[16..31],
#             GPU2: experts[32..47], GPU3: experts[48..63]

# Step 1: All GPUs compute gate independently
gate_out = gate(hidden_states)           # 每个GPU独立计算
expert_idx = topk(gate_out, k=1)         # 如token被分配给expert #42

# Step 2: All-to-all scatter (token dispatch)
# 各GPU将tokens按目标expert所属GPU分组
send_to_gpu[0] = tokens_for_experts[0..15]    # → GPU0
send_to_gpu[2] = tokens_for_experts[32..47]   # → GPU2 (expert #42)
# all_to_all_scatter: GPU交换数据

# Step 3: Local expert execution
for token in received_tokens:
    expert_id = token.assigned_expert
    output = expert_ffn[expert_id](token)  # 本地expert FFN kernel执行

# Step 4: All-to-all gather (result return)
# 各GPU将expert计算结果发送回token原始所在GPU
# all_to_all_gather: GPU交换output数据
```

Kernel 调度要点：(a) all-to-all 通信是 collective operation，须全局同步→GPU 间负载不均衡导致 straggler 等待；(b) expert 分布不均时某些 GPU 收到大量 tokens 而另一些空闲（load imbalance）；(c) all-to-all 通信量 = num_tokens × d_model × 2（scatter + gather），可成为瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主要实现框架：(a) **DeepSpeed-MoE [30]**：实现 expert-parallel 推理和训练，基于 ZeRO 优化的 all-to-all 通信，支持 hierarchical all-to-all 减少跨节点通信。(b) **FasterTransformer [25]**：支持 multi-GPU MoE 推理，通过 NCCL 实现 all-to-all communication。(c) **Tutel [16]**：动态 adaptive parallelism switching，在 data/tensor/expert parallelism 间切换。(d) **GShard [19]**：Google 的 MoE 分布式训练框架，expert parallelism 是其核心策略。(e) **Lina [21]**：优化 all-to-all communication scheduling 和 compute kernel fusion 以减少 MoE 推理延迟。Expert Parallelism 的替代方案：CPU Expert Offloading（单 GPU + CPU 内存，避免 all-to-all 通信开销）、Expert Caching（热点 experts 常驻 GPU）。单 GPU CPU offloading 方案（如 Pre-gated MoE）消除了多 GPU Expert Parallelism 的 all-to-all 通信和 load imbalance 问题，但依赖 PCIe 带宽和 compute-communication 重叠能力。

涉及论文标题：
- 71-Pre-gated_MoE_An_Algorithm-System_Co-Design_for_Fast_and_Scalable_Mixture-of-Expert_Inference.pdf

## Unified Bit-Serial Representation for Mixed Data Types

术语是什么？
Unified Bit-Serial Representation是BitMoD提出的统一bit-serial编码方案，将INT8、INT6、FP4（含扩展FP4-EA/ER）、FP3（含扩展FP3-EA/ER）四种不同数值格式的权重全部映射到相同的bit-serial term格式：(ws, we, wm, wbsig)——1-bit符号(sign)、2-bit指数(exponent)、1-bit尾数(mantissa)和共享的bit-significance(bsig)。每个term的值 = (-1)^ws × 2^we × wm × 2^wbsig。不同数据类型的编码路径不同：(1) INT8/INT6通过Booth编码——每3-bit Booth string生成一个term（INT8: 4 strings→4 terms, INT6: 3 strings→3 terms），Booth string的truth table映射{000,111}→(sign=0, exp=0, man=0, op=0), {001,010}→(+x), {110,101}→(-x), {011}→(+2x), {100}→(-2x)，相邻Booth string的bsig差2。(2) 扩展FP4/FP3通过fix-point转换+LOD——先将FP值转为1-bit sign + 4-bit integer + 1-bit fraction的fix-point（覆盖FP4-EA的最大特殊值±8和FP4的±0.5），然后Leading-One Detector找到两个'1' bit位置生成2个bit-serial terms。设计中：特殊值寄存器(SV_reg)存储4个可编程special values，LOD解码前将冗余负零替换为assigned special value。

从kernel调度角度拆解：
以FP4权重解码为bit-serial terms的具体计算过程：
```
# Input: FP4 weight w (含sign, exp, mantissa bits + per-group special value selection)
# Output: up to 2 bit-serial terms: [(ws1,we1,wm1,wbsig1), (ws2,we2,wm2,wbsig2)]

def decode_FP4_to_bit_serial(w, sv_reg):
    # Step 1: FP4 → 6-bit fix-point (sign-magnitude)
    # FP4 format: S | E1 | E0 | M
    # basic values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    # fix-point: 1 sign bit + 4 integer {I3,I2,I1,I0} + 1 fraction {F0}
    fix_val = fp4_to_fixpoint(w)  # 6-bit: {sign, I3,I2,I1,I0, F0}
    
    # Step 2: Check for redundant -0 and replace with special value
    if fix_val == negative_zero:
        fix_val = sv_reg[group_sv_idx]  # replace with assigned special value
    
    # Step 3: LOD (Leading-One Detector) → get first two '1' bits
    # LOD scans bits {I3,I2,I1,I0} and {I2,I1,I0,F0} for '1's
    for bit_pos in range(4):
        if fix_val[bit_pos] == 1:
            term_i = (sign=fix_val.sign, exp=bit_pos, man=1, bsig=0)
            # bsig adjusted by column position in PE array
            break
    
    for bit_pos in range(4):
        if fix_val[bit_pos + 1] == 1:  # shifted window {I2,I1,I0,F0}
            term_i = (sign=fix_val.sign, exp=bit_pos, man=1, bsig=0)
            break
    
    return terms  # 最多2个terms

# INT8 Booth encoding example: weight 8'b00101101
# Booth strings: {001, 011} → decode to terms
# 001→(+x, op=0, exp=0, man=1), 011→(+2x, op=0, exp=1, man=1)
# bsig of consecutive Booth strings differ by 2
```
Key insight: 统一表示使得同一PE hardware data-path处理所有精度，仅通过改变bit-serial terms数量切换精度（INT8:4, INT6:3, FP4:2, FP3:2 terms），吞吐量反比于terms数量。

术语一般如何实现？如何使用？
硬件实现：Bit-serial Term Generator模块（RTL, SystemVerilog）位于weight buffer和PE array之间。包含：(1) Booth Encoder（INT8/INT6路径）——combinational logic根据3-bit Booth string输出sign/exp/man/bsig；(2) Fix-Point Converter + LOD + Special Value Comparator（FP4/FP3路径）——FP→fix-point LUT + NOR-based zero-detector + mux选择SV_reg值 + priority encoder找leading ones。面积仅占PE tile的2.5%（2,419 µm² vs tile 97,090 µm² in TSMC 28nm）。扩展性：如需新special value（如7），可修改LOD→bit-serial表达为2^3 - 2^0（两个terms），无需改变硬件架构。

涉及论文标题：
- 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

## Bit-Serial Dequantization

术语是什么？
Bit-Serial Dequantization是BitMoD提出的低开销per-group dequantization硬件方案：利用bit-serial computing原理，将per-group INT8 scaling factor ∆逐bit与累加器中的partial sum相乘并shift-add，替代传统需要FP multiplier的dequantization。核心设计：group dot product在accumulator中累加完成后，dequantization unit逐cycle取出∆的一个bit（从MSB到LSB或LSB到MSB），若∆[i]=1则ACC<<i加到结果寄存器；若∆[i]=0则跳过。8-bit ∆需8 cycles dequantization。由于per-group dot product（G=128, PE dot-product size=4, FP3 needs 2 terms→64 cycles）远大于dequantization cycles→dequantization**从不stall计算pipeline**。该设计避免了在ASIC中引入FP multiplier完成per-group rescaling的面积开销。

从kernel调度角度拆解：
以BitMoD PE的Step❹ dequantization为例（group size G=128, 4-way PE, FP3 weight）：
```
# Input: Accumulator (mACC, eACC) after group dot product completion
#        INT8 scaling factor ∆[7:0] for this group
# Output: Dequantized partial sum (mGRP, eGRP) in FP16 range

def bit_serial_dequantize(mACC, eACC, delta_8bit):
    result_m = 0
    for i in range(8):  # one cycle per bit
        if delta_8bit[i] == 1:
            # shift accumulator by i + add
            term = (mACC << i, eACC)  # mantissa shift, same exponent
            result_m = result_m + term
        # else delta_8bit[i] == 0: skip (no-op cycle)
    
    # Normalize result to FP16
    (mGRP, eGRP) = normalize(result_m)
    return (mGRP, eGRP)

# Timing analysis:
# Group dot product: G/4 * terms_per_weight = 128/4 * 2 = 64 cycles
# Dequantization: 8 cycles ⊂ 64 cycles → no pipeline stall
# 同一PE column的accumulator在dequantization期间，PE继续处理下一group
```
DEQUANTIZATION pipeline嵌入PE的4-step流水线中：Step❶~❸处理group内各bit-serial dot product→Step❹在group完成时触发8-cycle dequantization→输出per-group FP16 partial sum到PE column accumulator→column accumulator汇总所有groups的partial sums→得到最终per-channel output activation。

术语一般如何实现？如何使用？
硬件实现：BitMoD PE data-path的Step❹——mACC寄存器 → barrel shifter (<<i, controlled by ∆[i]的位置) → adder tree (累加dequantization结果) → normalize (leading-one detector找mGRP最高非零位, 调整eGRP) → output register。面积开销极低（嵌入PE pipeline中，几乎无额外area）。与baseline dequantization对比：(1) GPU/AWQ: INT weight→FP16 multiply by ∆ + add z→消耗FP ALU cycles→打断Tensor Core pipeline；(2) ANT/OliVe: 仅支持per-channel dequantization (fuse到layer-norm)，不支持per-group；(3) MX: 需要FP pipeline处理shared exponent dequantization→额外energy。BitMoD的bit-serial dequantization是所有per-group方案中硬件开销最低的。

涉及论文标题：
- 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

## Mixed-Precision Bit-Serial Processing Element for LLM Inference

术语是什么？
Mixed-Precision Bit-Serial PE是BitMoD设计的核心计算单元，在统一bit-serial data-path上执行低精度权重（INT8/6, FP4/3）× FP16激活的混合精度矩阵乘法。每个PE每cycle执行4-way dot product between 4个bit-serial weight terms和4个FP16 activations。四步流水线：(1) Exponent Alignment——计算各对的(max(ae+we) - ae - we)作为delta exponent δe，同时生成每对乘积的符号；(2) Bit-Serial Multiplication——1-bit weight mantissa × 11-bit activation mantissa (含hidden bit) → 右移δe位对齐 → 保留3 extra bits for round-to-nearest-even；(3) Group Accumulation——adder tree求和4个bit-serial products → 左移wbsig位 → 与accumulator mantissa相加 → normalize更新累加器指数；(4) Bit-Serial Dequantization——累加器逐bit乘以INT8 scaling factor，shift-and-add。吞吐量：FP4/FP3=2 terms→2 cycles per 4 MAC→2× throughput over FP16 MAC；INT6=3 terms→1.33× throughput。PE面积较FP16 PE小24% (TSMC 28nm)。

从kernel调度角度拆解：
以处理FP3权重矩阵乘法为例（PE内4-way dot product）：
```
# 每cycle: 4个bit-serial weight terms (w0..w3) × 4个FP16 activations (a0..a3)
# Cycle 1 (term 0 of 2 for FP3):
# Step 1: Exponent Alignment
ae = [exponent(a0), exponent(a1), exponent(a2), exponent(a3)]  # 5-bit each
we = [w0.we, w1.we, w2.we, w3.we]                             # 2-bit each
e_sum = [ae[0]+we[0], ae[1]+we[1], ae[2]+we[2], ae[3]+we[3]]
max_exp = max(e_sum)
δe = [max_exp - e_sum[j] for j in 0..3]  # delta exponents

# Step 2: Bit-Serial Multiplication
for j in 0..3:
    # wm[j] is 1-bit → multiply = am[j] if wm[j]==1 else 0
    prod[j] = (wm[j] == 1) ? am[j] : 0  # 11-bit (含hidden bit)
    prod[j] = prod[j] >> δe[j]          # align to max exponent
    prod[j] = prod[j] with 3 extra bits  # rounding reservation

# Adder tree: sum 4 products
dot_product = prod[0] + prod[1] + prod[2] + prod[3]  # 14-bit

# Step 3: Group Accumulation
dot_product = dot_product << wbsig  # bit-significance shift
mACC_new = mACC + dot_product      # mantissa accumulation
(mACC, eACC) = normalize(mACC_new) # update accumulator

# Cycle 2 (term 1 of 2): repeat Step 1-3 with w term 1
# ...after all groups' terms processed...

# Step 4: Bit-Serial Dequantization (8 cycles, triggered after group completion)
# (see Bit-Serial Dequantization entry)

# Self-Attention: K,V quantized to INT8/INT4 → Booth decode → same PE
```
PE column: 8 PEs share one accumulator for per-column partial sum merging。PE tile: 8×8 PEs→output-stationary dataflow。Systolic: 4×4 tiles interconnected。

术语一般如何实现？如何使用？
RTL实现：SystemVerilog → Synopsys Design Compiler → TSMC 28nm @ 1GHz。PE area~1,500 µm² (tile 99,509 µm² / 64 PEs)。关键设计选择：(1) 4-way dot product balance——更大dot product size减少bit-serial cycles但增加硬件；4-way在FP3 (2 terms)下每PE每2 cycles完成4 MAC→与FP16 MAC 1 MAC/cycle相比2×加速。(2) Bit-serial dequantization≤8 cycles << group dot product 64 cycles→零stall。(3) Round-to-nearest-even的3 extra bits在shifter结果中保留，adder tree后round。(4) Booth encoder的3-bit Booth string→combinational truth table logic，零流水线开销。对比bit-parallel混合精度PE (FIGNA-like)：FP-INT4×2 PE因dual output需要double accumulator和output register→面积比FP-FP PE更大；BitMoD bit-serial PE面积反而更小→iso-area下更多PE→更高throughput。

涉及论文标题：
- 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf


## ReCoN (Redistribution and Coordination NoC) for Outlier Partial Sum Processing

术语解释
ReCoN 是 MicroScopiQ 加速器中的多级蝶形片上网络(Butterfly NoC)，负责在 systolic array 外部处理分布式离群值的 partial sum 重排序和 FP 累加，使得 PE 阵列保持 homogeneous INT 结构，从而以低硬件 overhead 支持离群值的高精度 FP 量化。

术语是什么？
ReCoN 是一个 {n(log₂n+1)} 个 {2-input, 2-output} switch 组成的多级蝶形 NoC（n为PE array列数），time-multiplexed across PE rows。每个 switch 通过 3-bit 配置信号执行三种操作：
- **Pass (=)**: 直通——inlier partial sum 直接通过，不做变换
- **Swap (×)**: 交换——将 Left/Right 输入端口的 partial sum 交换输出。用于将离群值 Lower Half 重定向到 Upper Half 对应的列
- **Merge (||)**: 合并——接收离群值的 Upper Half (O_Upper,Res) 和 Lower Half (O_Lower,Res) 两个 partial sum → 分离 Res 和 iAcc → mantissa bit 右移(Upper >>1, Lower >>2，对应FP mantissa的十进制位置) → 相加 → 加上 hidden-bit 贡献 (iAct值) → 输出完整 FP-outlier partial sum

ReCoN 的输入/输出 stages 也使用 {2-input, 2-output} switches，其中一个端口固定接 0。每级 ReCoN 同时接收与 PE 相同的 iActs（用于 hidden-bit 处理）。

从kernel调度角度拆解：
ReCoN 处理离群值 partial sum 的 kernel 级流程：

```
// ReCoN 调度伪代码 (per cycle)
// Input: partial sums from PE row i, with outlier_present flag per column
// Configuration: perm_list per μB specifying outlier distribution

for each column c in [0, PE_array_cols-1]:
    if outlier_present[row_i][c]:
        // Route partial sum to ReCoN via synchronization buffer
        sync_buffer[c].push(PE_output[row_i][c], timestamp)
    else:
        // Pass directly to next PE row (inlier path)
        PE_row[i+1].iAcc[c] = PE_output[row_i][c]

// Synchronization Buffer: 补偿 systolic array 的 skewed data flow
// 最快产出 partial sum 的列有最多 buffer stages
// 所有列同步后发送 ACK 给 PE row i
when all_columns_synced:
    // Column-wise arbiter: 多行并发访问 ReCoN 时仲裁
    grant = fair_arbitration(requesting_rows)

    // ReCoN 多级 butterfly 处理
    for stage s in [0, log2(n)]:
        for each switch in stage:
            switch_config = perm_list[μB_id][switch_id]
            if switch_config == PASS:
                out_L = in_L; out_R = in_R
            elif switch_config == SWAP:
                out_L = in_R; out_R = in_L
                // iAcc 通过另一端口传递
            elif switch_config == MERGE:
                // 分离 Res 和 iAcc (notated as //)
                Res_Upper, iAcc_Upper = split(in_L)
                Res_Lower, iAcc_Lower = split(in_R)
                // Mantissa 对齐 (FP decimal position)
                Res_Upper_shifted = Res_Upper >> 1   // Upper mantissa
                Res_Lower_shifted = Res_Lower >> 2   // Lower mantissa
                // FP 累加: mantissa + hidden-bit
                merged_Res = Res_Upper_shifted + Res_Lower_shifted + iAct
                out = {merged_Res, iAcc_Upper}  // iAcc_Upper 是正确的 accum

    // 重排序后的 partial sum 送入下一 PE row 或 oAct buffer
    PE_row[i+1].iAcc = ReCoN_output
```

关键设计特性：
- **Pipelined**: ReCoN 内部 pipelined → 每个 cycle 产出 reordered partial sum（pipeline depth 填充后）
- **Time-multiplexed**: 多个 PE rows 共享同一 ReCoN → 低面积 overhead。Column-wise arbiter 确保公平调度。N 个并发行中最后被服务的行增加 N-1 cycle 延迟
- **Access conflicts**: 对于 64×64 PE array，单 ReCoN unit 的 access conflicts <3% → 对性能影响可忽略
- **Scalability**: 8 ReCoN units 可达到 zero access conflicts → peak performance；128×128 PE array 时仅需 8 ReCoN units

术语一般如何实现？如何使用？
- **RTL 实现**: Synopsys Design Compiler + Innovus PnR, TSMC 7nm, 1GHz。1 ReCoN unit area = 204.68 μm² (64×64 PE array中)
- **Area overhead**: 随 PE array 规模增大而降低——128×128 时仅 3%（因 PE array dominates total area）。8 ReCoN units 时 overhead 为 11%
- **集成到现有 NoC-based accelerator**: MTIA (Meta) 和 Eyeriss-v2 中集成 ReCoN 功能→仅 3% 和 2.3% compute area 增加。因为这些 accelerator 已有 NoCs，ReCoN 功能可复用现有 NoC 架构
- **与 OliVe/GOBO 单 PE 处理对比**: OliVe 在每个 PE 内做 encoder/decoder 处理离群值→PE 阵列大时 overhead 大；GOBO 使用双 PE 类型→复杂调度；ReCoN 将处理外置到共享 NoC→PE 保持 simple homogeneous INT 结构→367.51 TOPS/mm² (2× vs OliVe, 14× vs GOBO)
- **Synchronization Buffer**: 在 ReCoN 输入侧，通过不同长度的 buffer chain (最快列 buffer 最长)补偿 systolic array 的 skewed data arrival time→同步所有列的 partial sum 到达

涉及论文标题：
- 84-MicroScopiQ- Accelerating Foundational Models through Outlier-Aware Microscaling Quantization.pdf

## Multi-Precision PE with MODE-based Bit-width Switching

术语解释
MicroScopiQ PE 采用 MODE 信号驱动的多精度乘积累加设计，通过同一 PE 内打包多个低精度权重或处理单个高精度权重实现 multi-precision support，而非传统方案（如 OliVe/GOBO）中 grouping 相邻 PE 导致 throughput 下降。

术语是什么？
传统 multi-precision accelerator (OliVe, ANT, BitFusion) 采用 bottom-up grouping——多个低精度 PE grouping 在一起支持更高精度→牺牲 parallelism 和 throughput。MicroScopiQ 采用 different strategy：通过 MODE 信号在同一 PE 内在 2-bit 和 4-bit 模式间切换。
- **MODE_2b**: PE 同时打包两个 2-bit 权重，共享同一 iAct→parallel 两个 2-bit 乘积累加
- **MODE_4b**: PE 处理单个 4-bit 权重→标准 4-bit 乘积累加

Mulitiplier-tree architecture: 4个 4-bit × 2-bit multipliers 计算 partial sums (P_00, P_01, P_10, P_11)；基于 MODE 信号通过 adder/shifter 组合产生最终结果。

从kernel调度角度拆解：
Multi-Precision PE 内部执行流程：

```
// Input: 2×2-bit weights W_0, W_1 (MODE_2b) 或 1×4-bit weight W (MODE_4b)
//        iAct (8-bit INT), iAcc (from prev PE row or ReCoN)

// === MUL Stage ===
// 4× (4-bit × 2-bit) multipliers 并行计算
P_00 = W[1:0]   × iAct[3:0]
P_01 = W[1:0]   × iAct[7:4]
P_10 = W[3:2]   × iAct[3:0]
P_11 = W[3:2]   × iAct[7:4]

// Combine based on MODE
if MODE_2b:
    // 两个独立 2-bit 乘积累: 并行产出两个结果
    Res_0 = (P_01 << 2) + P_00  // weight 0 × iAct
    Res_1 = (P_11 << 2) + P_10  // weight 1 × iAct
    Res = {Res_0, Res_1}  // 两个独立 partial sum
elif MODE_4b:
    // 单个 4-bit 乘积累加
    Res = (P_11 << 4 + P_00) + (P_01 << 2 + P_10 << 2)

// === ADD Stage ===
// Outlier_Present 信号控制累加行为
if Outlier_Present:
    if inlier:
        // 标准 INT 累加: Res + iAcc, carry propagation through MUX
        oAcc = add_with_carry(Res, iAcc, MODE)
    elif outlier (Upper/Lower half):
        // 离群值 partial sum offload 到 ReCoN
        // 通过 concatenation {Res, iAcc} 发送, 由 ReCoN 完成真实 FP 累加
        oAcc_NoC = concat(Res, iAcc)
else:
    // 纯 inlier row: 标准累加
    oAcc = add_with_carry(Res, iAcc, MODE)

// MODE_2b: 两个 adder 独立并行
// MODE_4b: 两个 adder 联合工作 (carry propagation)
```

Weight mapping 对 throughput 的影响：
- **MODE_4b**: 每个 PE column 处理一个 weight→充分利用 PE array parallelism
- **MODE_2b**: 每个 PE column 同时处理两个 weight→double throughput。同一 PE 打包的两个 2-bit weight 在传统方案中会被分配到同一 row 的不同 columns→MicroScopiQ 有效增加了并行度

术语一般如何实现？如何使用？
- **RTL 面积**: Base PE = 2.82 μm², Multi-precision support = 0.22 μm²/PE (TSMC 7nm)。与 OliVe 的 0.68 μm²/PE multi-precision area 对比→3× lower overhead（因 OliVe 需要 encoder/decoder 处理 exponent-integer pair）
- **MUL stage 设计**: 受 BISMO [Umuroglu et al., FPL 2018] 和 [Liu et al., IEEE ESL 2023] 的 multiplier-tree 启发
- **Throughput 优势**: MODE_2b 时 PE array 有效 throughput 翻倍（每个 PE 同时处理两个权重）。W4A4 配置下 MicroScopiQ v2 (大部分层使用 b_b=2) 达到 2.47× speedup vs baselines
- **到 ReCoN 的接口**: 含离群值的 PE 输出通过 OAcc_NoC/PE 信号路由到 ReCoN→离群值的 Upper/Lower halves 的 iAcc concatenation→ReCoN 在外部完成 FP partial sum 计算→保持 PE 内部简单

涉及论文标题：
- 84-MicroScopiQ- Accelerating Foundational Models through Outlier-Aware Microscaling Quantization.pdf

## GPU Register-Level Outlier Merging via shfl_sync for Quantized GEMM

术语解释
MicroScopiQ 在 GPU 上通过 CUDA warp-level primitive `shfl_sync` 实现离群值分布式 halves 的寄存器级合并，避免 shared memory 访问开销，并在 tensor core GEMM 前动态判断 tile 类型（mixed/纯 inlier）选择最优执行路径。

术语是什么？
当 MicroScopiQ 量化后的模型在 GPU 上执行 GEMM 时，离群值的 Upper/Lower halves 分布在不同线程的寄存器中。`shfl_sync(mask, register, thread_id)` 允许 warp 内线程直接交换寄存器值，实现零 shared memory 开销的离群值合并。基于 perm_list 的 warp 内重排后，再根据 tile 是否含离群值动态选择 GEMM 路径。

从kernel调度角度拆解：
GPU MicroScopiQ GEMM kernel 流程：

```
// GPU Thread Block 计算 T_m × T_n 输出 tile，沿 K 维度迭代
// Each warp (32 threads) = 4 μBs (B_μ=8)

for k in [0, K, K_tile]:
    // 1. 加载 weight tile 到 shared memory (含 perm_list metadata)
    load_weights_to_smem(weights[k], metadata[k])

    for each μB in warp (4 μBs × 8 threads):
        perm = metadata[μB].perm_list  // e.g., {(3,2)} for outlier at pos 3→Lower Half at pos 2

        // 2. Register-level outlier merging via shfl_sync
        for (outlier_pos, lower_pos) in perm:
            // 每个线程读取自己的 weight
            w_local = smem[thread_id]

            if thread_id == outlier_pos:
                // 持有 Upper Half {s, m1} 的线程
                upper_half = w_local
                // 从 lower_pos 线程读取 Lower Half {s, m0}
                lower_half = shfl_sync(FULL_MASK, w_smem[lower_pos], lower_pos)
                // 合并: recombine sign + mantissa bits → FP4 value
                merged_w = merge_halves(upper_half, lower_half)
                // Dequantize MX-FP4 → FP16
                w_fp16 = dequantize_MX_FP4(merged_w, MXScale[μB])

            elif thread_id == lower_pos:
                // 被剪枝位置 → 读入 Lower Half (但已完成共享给 outlier 线程)
                // 此线程在后续 GEMM 中贡献 0 (权重被剪枝)
                w_local = 0

    // 3. Dynamic GEMM dispatch
    if tile_has_outliers:
        // Mixed tile: FP16 GEMM on Tensor Cores (iteration 0, Figure 9)
        for each elem in tile:
            w_deq = dequantize_to_FP16(w_int, scale)  // MX-FP→FP16
        partial_sum = TC_FP16_GEMM(w_deq, iAct_FP16)  // Tensor Core FP16
    else:
        // Pure inlier tile: INT4 Tensor Core GEMM (iteration K-1, Figure 9)
        partial_sum = TC_INT4_GEMM(w_int, iAct_INT8)   // Tensor Core INT4
        partial_sum = dequantize_INT32_to_FP16(partial_sum)  // dequant for FP32 accum

    // 4. Accumulate along K dimension in FP32
    C_tile += FP32(partial_sum)
```

术语一般如何实现？如何使用？
- **CUDA 实现**: 基于 CUDA shfl_sync 和 PyTorch frontend。优化后的 kernel (MS optim.) 达到 2.06× TRT-LLM FP16 throughput (A100, LLaMA-2 13B, W4A4)
- **Modified Tensor Core**: 在 GPGPU-Sim/AccelSim 中添加 variable right shifter (Inliers: >>0, Upper: >>1, Lower: >>2) 处理 FP mantissa。Die overhead ~0.1% (RTX 2080 Ti sized GPU)
- **Unoptimized vs Optimized**: 未优化的实现使用 shared memory 进行 outlier merging→因 shared memory access overhead 和 FP16 GEMM 开销→低于 FP16 baseline；优化后通过 register caching 和动态 kernel dispatch→匹配或超越 SoTA INT4 kernel (Atom) 的 throughput
- **vs MicroScopiQ Accelerator**: GPU 通过 shfl_sync 在 warp 内完成离群值合并，而加速器通过 ReCoN NoC 完成——两者都是对分布式离群值的硬件/软件抽象，但实现层次不同

涉及论文标题：
- 84-MicroScopiQ- Accelerating Foundational Models through Outlier-Aware Microscaling Quantization.pdf

---

## SpMSpM (Sparse-Sparse Matrix Multiplication)

术语解释
SpMSpM（稀疏-稀疏矩阵乘法）是稀疏张量代数中的核心kernel，计算 C = A × B，其中A和B均为稀疏矩阵。以Einsum表示为 Cij = Aik × Bkj over (I, J, K)，k为contraction dimension。由于A和B均为稀疏，output C也是稀疏矩阵。

术语是什么？
SpMSpM广泛用于图处理（邻接矩阵乘法、BFS）、高性能计算（代数多重网格、量子化学）和机器学习（稀疏神经网络层）。由于稀疏矩阵的非零分布不规则，SpMSpM在通用处理器上性能极差——大量间接的随机数据访问导致memory hierarchy locality差，且各计算核间严重load imbalance。因此专用稀疏张量加速器（如ExTensor[12], GAMMA[40], SpArch[41], SIGMA[31], MatRaptor[33]）被提出以优化SpMSpM kernel。

SpMSpM的关键计算特征是：output tensor C的稀疏度和尺寸不仅取决于A和B的各自稀疏度，更取决于两者非零分布的correlation——例如若A的column k有非零元素但B的row k全是零，则该k贡献的MAC全部无效。

从kernel调度角度拆解：
以SpMSpM在HYTE Gust dataflow下的kernel执行伪代码为例（32 MAC PEs, CSR format, tiled）：
```
// Gust dataflow: i ⊲ k ⊲ j (outer → inner)
// Given: tiling scheme (Ti, Tj, Tk), inter-tile order, buffer allocation
for tile_i in 0..I/Ti-1:           // inter-tile i iteration
  for tile_j in 0..J/Tj-1:          // inter-tile j iteration
    for tile_k in 0..K/Tk-1:        // inter-tile k iteration (innermost)
      // Accessor fetch fiber segments into SRAM
      A_segs = fetch_CSR_rows(A, tile_i*Ti, Ti, tile_k*Tk, Tk)
      B_segs = fetch_CSR_rows_transposed(B, tile_k*Tk, Tk, tile_j*Tj, Tj)
      // Intra-tile PE execution
      for i' in 0..Ti-1:
        for k' in 0..Tk-1:
          a_val = A_segs[i'][k']
          if a_val == 0: continue
          for j' in 0..Tj-1:
            b_val = B_segs[k'][j']
            if b_val != 0:
              C_partial[i'][j'] += a_val * b_val
      // Partial sum merging (Gust/OP requires merger hardware)
      merge_to_C(C_partial, tile_i, tile_j)
```

三种dataflow的kernel执行核心差异：
- **IP (i ⊲ j ⊲ k)**：外层i/j定output坐标，内层k做向量内积→需要index intersector在A列和B行之间匹配k坐标→仅对匹配的非零对做MAC→A/C复用好，B访问量=nnzB×I
- **OP (k ⊲ i ⊲ j)**：外层k取A的一列×B的一行→产生I×J的partial product矩阵→通过partial sum merger累加→A/B复用好，C访问量=effMAC
- **Gust (i ⊲ k ⊲ j)**：外层i定output行，中层k做类似OP的外积，内层j遍历→C按行产出→partial sum buffer需求较OP小→A/C复用好，B访问量=effMAC

术语一般如何实现？如何使用？
- **加速器实现**：PE array (32-256 MAC units) + SRAM buffer hierarchy + 专用硬件（index intersector for IP, partial sum merger for Gust/OP）→GAMMA[40]基于Gust算法优化merge tree，SpArch[41]使用OP+partial product合并最大化数据复用，ExTensor[12]通过intersector支持通用稀疏张量代数
- **数据集**：SuiteSparse Matrix Collection[7]（University of Florida）是标准benchmark集→~2800+真实稀疏矩阵涵盖graph/optimization/structural/electrical等领域→本文选18个代表性矩阵评估（密度0.0006%-0.356%，非零1.5M-25M）
- **Tiling下的SpMSpM**：对大矩阵（非零数 > on-chip buffer），tiling限制每tile的计算在buffer容量内→但引入cross-tile repetitive access→optimal tiling取决于sparse pattern→HYTE等自动确定tiling scheme
- **其他variants**：FT×F（tall-skinny sparse × itself）、F×D（sparse × dense = SpMM）、FT×S（multi-source BFS）

涉及论文标题：
- 85-HYTE- Flexible Tiling for Sparse Accelerators via Hybrid Static-Dynamic Approaches.pdf

## In-Memory Activation Quantization

术语解释
In-Memory Activation Quantization（存内激活量化）是在PIM的DRAM bank内部直接对LLM的KV cache activation进行在线量化的技术——量化过程在数据所在的memory内完成，不将activation搬移到外部计算单元（GPU/CPU）进行处理。

术语是什么？
与传统"load到GPU→GPU上量化→写回"的offload量化不同，in-memory activation quantization在PIM bank内部完成全流程：(1) 聚类(DC/CA/CC)在BankPE+BufferPE上执行；(2) 量化结果(codebook+indices)直接留在bank memory中；(3) 后续attention计算（ATNK/ATNV）也在同一bank内对compressed data执行。这利用了PIM的核心优势——高内部带宽消除activation数据的host-device往返。AQPIM的key innovation是使PQ这种bandwidth-hungry的clustering方法在PIM上practical：PIM内部带宽足够支撑online k-means的迭代distance computation和centroid update，而传统GPU上这种clustering需要频繁的global memory access，延迟不可接受。

从kernel调度角度拆解：
PIM上在线activation量化的kernel执行伪代码（prefilling阶段，BankPE+BufferPE）：
```
// 输入: KV[head][N, d_head] 已分布在各Bank
//      hyperparams: m=32 subvectors, K=512 centroids, iter=4

// === Phase 1: Codebook Generation (并行于GPU attention) ===
for subvec s in range(m):  // 各subvector独立，可并行
    // 随机初始化centroids (选择N中K个点)
    centroids[s] = random_select(KV[s], K)
    
    for iter in range(4):
        // DC kernel (BankPE, all banks parallel)
        // 每个BankPE处理其bank中的tokens
        for token t in bank_tokens:
            for centroid c in range(K):
                // 欧氏距离: (a-b)^2 = a^2 + b^2 - 2ab
                dist[t][c] = sum(KV[t,s,:]^2) + sum(centroid[c]^2) 
                             - 2*sum(KV[t,s,:] * centroid[c])
                // MUL+ADD+SUM units in BankPE
        
        // CA kernel (BufferPE)
        // 接收所有bank的dist→argmin
        for token t:
            assignment[t] = argmin(dist[t])  // MIN unit in BufferPE
        
        // CC kernel (BankPE + BufferPE)
        // 分子: BankPE, weighted sum
        for centroid c in range(K):
            num[c] = 0
            for token t where assignment[t]==c:
                num[c] += weight[t] * KV[t,s,:]  // MUL+SUM in BankPE
        // 分母: BufferPE, sum of weights → 倒数
            den[c] = 1.0 / sum(weight[t] for t in cluster[c])  // SUM+DIV
        // centroids[c] = num[c] * den[c]   // MUL in BankPE (单次乘法)
    
    // 存储codebook和indices
    store(key_codebook[s])  // 在固定codebook区域
    store(key_indices[s])   // 逐层分配

// === Phase 2: Decode阶段 Append新token ===
// 新token qkv到达后
for subvec s in range(m):
    // 对新key/value子向量，找最近centroid
    dist_new = ||new_kv[s] - centroids[s][:]||^2  // DC kernel, BankPE
    new_assignment = argmin(dist_new)  // CA kernel, BufferPE
    append(key_indices[s], new_assignment)  // 追加到indices区域
```

术语一般如何实现？如何使用？
实现条件：(1) PIM硬件支持bank级FP16 MAC（BankPE）和跨bank通信（MV_BA/MV_BF命令）；(2) 聚类迭代次数限制（4次）以保证latency hidden behind GPU prefilling；(3) 逐层codebook生成以减少peak memory（一层的codebook生成后即可释放该层的原始KV）。与GPU-based online quantization（如Oaken的online per-token scheme）对比：Oaken在GPU上逐token做量化+编码，需要额外的GPU compute和memory bandwidth；AQPIM在PIM内完成，与GPU并行，不消耗GPU资源。与offload-based quantization（如PQCache在CPU上做PQ）对比：无需CPU-GPU KV传输，消除PCIe bottleneck。

涉及论文标题：
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

---

## PIM Command Set for Quantization and Attention

术语解释
PIM Command Set for Quantization and Attention（PIM量化与注意力命令集）是AQPIM在HBM-PIM标准命令路径上新增的一组专用命令，用于控制PQ codebook生成、PQ-based attention计算和数据搬运。

术语是什么？
AQPIM定义了两类PIM命令：(1) 计算命令：PIM_SET_CONFIG（广播PQ配置参数m、K等，一次配置）、PIM_MAC_AB（所有bank同时执行MAC操作，用于DC/ATNK/ATNV）、PIM_SFM（BufferPE执行softmax操作：ADD+SUM+MAX+DIV+EXP）、PIM_RET（intra-row indirection row buffer retrieval）；(2) 数据移动命令：PIM_MV_BA（BankPE→BufferPE数据传输）、PIM_MV_BF（BufferPE→BankPE数据传输）、PIM_ACT_AB（所有bank row激活）、PIM_RD/PIM_WR（I/O读写）。命令通过标准HBM command path issue，与常规DRAM命令（ACT/RD/WR/PRE）相同的地址/命令总线。

从kernel调度角度拆解：
AQPIM decode阶段单个token的attention计算命令序列：
```
// Pre-condition: query qkv已写入Bank (PIM_WR)
//               key_codebook/value_codebook在固定区域
//               key_indices/value_indices已存在

// Step 1: ATNK - Query × Key Codebook
PIM_SET_CONFIG {op: ATNK, m: 32, K: 512}  // 配置参数
PIM_ACT_AB {row_addr: key_codebook_rows}    // 激活codebook row
PIM_MAC_AB {act: query[head][s], weight: codebook[s][:]}  
// BankPE并行: query_sub × codebook_sub → inner_prod[s][k]

// Step 2: Data move to BufferPE for softmax
PIM_MV_BA {src: bank_inner_prod, dst: bufferpe_softmax_in}

// Step 3: SFM - Softmax
PIM_SFM {input: partial_qKT, output: attn_scores}
// BufferPE: EXP(partial) → SUM → DIV → softmax output

// Step 4: Data move back to BankPE
PIM_MV_BF {src: bufferpe_attn_scores, dst: bank_attn_buf}

// Step 5: ATNV - Intra-row Indirection + Attention Scores × Values
PIM_ACT_AB {row_addr: value_codebook_rows}  // 激活codebook row (1次)
PIM_RET {indices_src: GRF_value_indices, output_dst: GRF}  
// Column decoder streaming: GRF indices → row buffer lookup → values
PIM_MAC_AB {act: attn_scores, weight: looked_up_values}
// BankPE: attn_score × value_entry → accumulate output

// Step 6: Read back results
PIM_RD {src: bank_output}  // attention output → GPU

// 总DRAM activations: 2 (key + value codebook rows)
```

术语一般如何实现？如何使用？
命令实现基于AttAcc!的PIM command infrastructure：命令通过标准HBM command truth table中的reserved位编码（与常规DRAM ACT/RD/WR命令共用command/address bus），BankPE/BufferPE中的command decoder解析并执行。命令不为HBM-PIM商用产品支持——目前是simulator建模（基于Ramulator/AttAcc! simulator）。命令序列由host CPU或GPU的runtime scheduler编排，通过HBM controller的command queue issue。

涉及论文标题：
- 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

## Hardware-Orchestrated Kernel Launch (硬件编排的Kernel启动)

术语是什么？
Hardware-Orchestrated (HO) Kernel Launch是SN40L RDU中AGCU实现的自主kernel调度机制，与传统的Software-Orchestrated (SO) kernel launch相对。在SO模式下，host CPU通过PCIe逐一发送每个kernel的command序列（Program Load→Argument Load→Kernel Execute）→等待completion→发送下一kernel的commands。在HO模式下，整个kernel schedule（多个kernel的完整command序列）预加载到AGCU hardware command queue中，AGCU自主依次执行：完成kernel_0的Program Load/Arg Load/Execute后，无需host干预，立即开始kernel_1的command序列。关键优势：消除host-device PCIe round-trip latency、host software调度overhead（interrupt handling、queue management）、以及kernel间idle gap。论文数据显示HO对decode阶段（kernel执行时间极短、dominated by weight loading）带来1.4×-8× speedup，但对prefill/training（kernel执行时间长，overhead amortized）仅~1.1× speedup。

从kernel调度角度拆解术语：
Kernel执行流程在HO vs SO下的对比（以Llama2-7B decode, autoregressive loop为例）：
```
# SO (Software-Orchestrated) - 每次decode iteration
for token_i in range(num_tokens):
    host_cpu.send_command(AGC, PROGRAM_LOAD, kernel_bitstream_addr)
    wait(AGC_completion_interrupt)
    host_cpu.send_command(AGC, ARGUMENT_LOAD, weight_addr, kv_cache_addr, input_addr)
    wait(AGC_completion_interrupt)  
    host_cpu.send_command(AGC, KERNEL_EXECUTE)
    wait(kernel_completion_interrupt)
    # ~μs级PCIe round-trip + interrupt handling overhead per kernel
    # kernel执行时间仅~数十μs → overhead占比大

# HO (Hardware-Orchestrated) - 一次性预加载
# 初始化：加载kernel schedule到AGCU command queue
agcu.load_schedule([
    (PROGRAM_LOAD, kernel_bitstream_addr),
    (ARGUMENT_LOAD, weight_addr_0, kv_cache_addr_0, input_addr_0),
    (KERNEL_EXECUTE),
    (ARGUMENT_LOAD, weight_addr_0, kv_cache_addr_1, input_addr_1),  # reuse same program
    (KERNEL_EXECUTE),
    ...  # N次decode iterations的完整schedule
])
agcu.start_autonomous_execution()  # AGCU自主执行，host无需干预
# 所有kernel间zero gap，无PCIe/interrupt overhead
```
HO的适用条件：static kernel schedule（decode loop的iteration count已知，KV cache地址可compute ahead）。不适用条件：dynamic control flow（需要host判断router output决定下一个expert）。

术语一般如何实现？如何使用？
- 类似机制：NVIDIA GPU的CUDA Graphs可将多个kernel launch capture为static graph，GPU hardware replay（但CUDA Graphs仍需host发起replay，非完全autonomous）。AMD AI Engine的software-defined tile-to-tile streaming。Cerebras WSE的dataflow scheduling（但 Cerebras 无host-device分离，调度更简单）。
- SN40L HO实现：AGCU内hardware command queue + counter-based loop control。Kernel schedule由compiler生成（已知compile-time确定的loop iteration counts和memory addresses）。AGCU维护internal state machine：IDLE→PROGRAM_LOAD→ARG_LOAD→EXECUTE→(loop) next ARG_LOAD→...→DONE。
- 通用价值：适用于LLM decode（autoregressive loop）、multi-layer identical operators（所有decoder layers相同结构），以及其他存在static repeated execution pattern的场景。

涉及论文标题：
- 8-SambaNova_SN40L_Scaling_the_AI_Memory_Wall_with_Dataflow_and_Composition_of_Experts.pdf

## Spatial Operator Fusion on Dataflow Architecture (数据流架构上的空间算子融合)

术语是什么？
Spatial Operator Fusion是SN40L RDU compiler自动将20+ operators融合为单一fused dataflow kernel的技术，远超传统operator fusion（GPU通常融合1-5 operators）。关键特征：(1) **Spatial Fusion**：operators在PCU 2D array上空间分布为pipeline stages——而非time-multiplex到同一compute unit上顺序执行。(2) **Arbitrary Access Pattern Support**：融合包含transpose、shuffle、gather/scatter等复杂access patterns的operator chain（如Monarch FFT的Gemm0-Mul-Transpose-Gemm1）。Transpose通过PMU diagonal striped format实现（write-time layout transformation换取read-time zero-cost transpose），不增加额外operator。(3) **Automatic Compilation**：compiler从PyTorch-level model graph自动识别fusion opportunities、分配PCU/PMU资源、配置RDN routes，无需手写fused kernel。(4) **Operational Intensity提升**：融合将多个memory-bound operators转化为单一compute-bound kernel。如Monarch FFT的operational intensity从unfused的39.5 FLOPs/byte提升到fully fused的410.4 FLOPs/byte（A100 memory wall at ~150 FLOPs/byte, 意味着fusion是memory-bound→compute-bound的关键transformation）。

从kernel调度角度拆解术语：
Spatial fusion的kernel调度过程（以Mistral-7B decode, Figure 10中13× speedup的benchmark为例）：
```
# 传统Unfused（per-operator kernels）
ctx = load_qkv_weights(HBM)        # Kernel 1: Q/K/V projection GEMM
q, k, v = gemm(ctx, input)         # → write q,k,v to HBM
ctx = load_attn_weights(HBM)       # Kernel 2: Attention score
scores = gemm(q, k.T)              # → read q,k from HBM
attn = softmax(scores)             # → write attn to HBM
ctx = load_output_proj(HBM)        # Kernel 3: Attention output
out = gemm(attn, v)                # → write out to HBM
...                                 # + FFN GEMMs, activations, norms
# 总计 ~100+ kernel launches per decoder layer
# 大量HBM roundtrip, low operational intensity

# Spatial Fusion（单一fused kernel）
# Compiler分配：PCU[0:31] systolic=Q_proj, PCU[32:63] systolic=K_proj, 
# PCU[64:95] systolic=V_proj, PCU[96:127] SIMD=softmax,
# PCU[128:191] systolic=Attn_out, PCU[192:...] systolic=FFN, ...
# PMU[a:b] stage_buffer=QKV_outputs, PMU[b:c] stage_buffer=attn_scores, ...
# RDN routes: HBM→PMU[a]→PCU[0:95]→PMU[b]→PCU[96:127]→PMU[c]→...
# 单一AGCU kernel launch → pipeline stream开始
# 总计 1 kernel launch per decoder layer
# ZERO HBM roundtrip for intermediate tensors
```

术语一般如何实现？如何使用？
- GPU上的局限性：PyTorch2 [43] 和 TensorRT [42] 的fusion restrictions（不支持transpose、shuffle等patterns的chain fusion）。FlashAttention [47][48] 手写fused attention kernel（需要expert GPU programming）。相比之下，SN40L compiler自动处理所有patterns（因RDN + PMU hardware support）。
- Compiler实现：分析dataflow graph的依赖链→识别可融合的operator sequence→估算每operator的compute (FLOPs) 和 memory (bytes) 需求→分配PCU (systolic/SIMD mode) 和 PMU (capacity/bandwidth partitioning)→PnR到2D mesh→配置RDN routes→生成单一kernel binary。
- 通用价值：Spatial fusion对小型模型（如7B experts，operational intensity本就低于大模型）尤为重要——小型模型的memory-bound问题更严重，fusion的speedup更大（Mistral-7B decode: 13×）。

涉及论文标题：
- 8-SambaNova_SN40L_Scaling_the_AI_Memory_Wall_with_Dataflow_and_Composition_of_Experts.pdf

## Peer-to-Peer (P2P) Streaming Protocol for Inter-Accelerator Communication (加速器间点对点流式通信协议)

术语是什么？
P2P Streaming Protocol是SN40L RDU中AGCU实现的socket间直接数据流传输协议，使不同RDU socket的tile之间不经DDR或HBM直接streaming tensor data。关键特性：(1) **Streaming而非Message-Passing**：数据以dataflow streaming方式持续传输（而非DMA-based message passing），可与compute pipelined到同一fused kernel中。(2) **Collective Communication Primitives**：P2P protocol提供构建AllReduce等collective操作的基础——compiler将collective operators像任何其他operator一样fused and pipelined到kernel中。(3) **Avoiding HBM Hops**：跨socket通信跳过HBM（不经HBM write→read roundtrip），节省通信带宽并降低延迟。(4) **Uniform Abstraction**：Compiler对intra-socket (RDN) 和inter-socket (P2P) 通信使用统一的data/tensor/pipeline parallel mapping抽象——mapping across sockets与mapping within socket使用相同compiler pass。

从kernel调度角度拆解术语：
P2P streaming在8-socket TP=8的Llama2-7B decode中的执行流程：
```
# TP=8: 每socket计算partial results
for each_socket in [0..7]:
    # Intra-socket fused kernel执行
    partial_out = fused_decoder_layer(local_input_shard, local_weights_shard)
    # partial_out在PMU stage buffer中

# P2P AllReduce fused到同一kernel的尾部
# Compiler配置的P2P communication pattern (以Ring AllReduce为例):
socket_0: partial_out → AGCU[0] → P2P_link[0→1] → AGCU[1] (reduce) → ...
socket_1: partial_out + received_socket_0 → AGCU[1] → P2P_link[1→2] → ...
...
# After Ring completion: 每socket有完整reduced tensor
# 继续下游computation或写回HBM
```
P2P streaming的优势：(a) Data在传输过程中即被消费（streaming reduce——到达即累加，无需等全部到达）；(b) 无HBM bandwidth争抢（P2P用专用物理链路）；(c) pipelineable with compute（P2P传输可与下一tile的compute重叠）。

术语一般如何实现？如何使用？
- 类似技术：NVIDIA NVLink/NVSwitch（GPU-direct RDMA）、Google TPU ICI (Inter-Chip Interconnect)。差异：NVLink使用message-passing + RDMA（需显式通信kernel），P2P Protocol允许通信operator被fused into compute kernel（类似TPU的通信与计算重叠，但RDU的dataflow性质使pipeline更自然）。
- 物理实现：P2P interfaces（Figure 5中的"P2P" blocks）提供socket间物理连接。通过TLN路由到AGCU，再经P2P PHY到remote socket。Compiler配置P2P routes和bandwidth allocation（类似RDN routing，但在inter-socket scale）。
- 通用价值：Dataflow-native P2P streaming使多socket scaling的通信开销可被compute pipeline隐藏——对TP/PP distributed LLM inference至关重要。

涉及论文标题：
- 8-SambaNova_SN40L_Scaling_the_AI_Memory_Wall_with_Dataflow_and_Composition_of_Experts.pdf

## AU Bucket Mechanism (Accelerator Unit 离散化 Profiling 桶机制)

术语是什么？
AU Bucket 是 AUM Background AU Profiler 中用于离散化连续 AUV 空间以减少 profiling 开销的机制。由于 AU 使用模式、频率降低和资源需求在三维空间上连续变化（理论上需要无限次 profiling），AU Bucket 将其离散化为有限数量的 bucket：对 high/low/none AU usage 三种使用等级，各 profile 三种 processor division 配置 × 五种 resource configuration（LLC ways + Memory BW），每个 bucket 记录 50% 平均性能 P^a、90% tail 性能 P^t 和 CPU 功耗 W_CPU。总共 450 次执行（3×3×5×10 repetitions）即可收敛构建完整的 AUV Model。

从 kernel 调度角度拆解术语：
AU Bucket 的 profiling 矩阵和调度查询流程：
```
// Profiling 阶段：为每个 bucket 采集数据
for U_AU in {High, Low, None}:           // 3 种 AU 使用等级
    for C_div in {Div1, Div2, Div3}:    // 3 种核心划分配置
        for R_config in {R1..R5}:       // 5 种资源分配配置
            for rep in 1..10:           // 10 次重复
                record: P^a(U,C,R), P^t(U,C,R), W_CPU(U,C,R)

// Runtime 查询：根据当前 state 查表
// 输入：当前 U_AU (ARI 计算得出), C_region (当前核心区), SLO_slack
// 若 P^m < SLO → 查 M(P^a_H, P^a_L) → 得到 aggressive resource config
// 若 P^m > SLO → 查 M(P^t_H, P^t_L) → 得到 conservative resource config
// 输出：R_AU = {RL2C, RLLC, RBW}，即 L2 cache ways, LLC ways, Memory BW 配额
```

Profile 示例 bucket (Table III): U_High, C 0-11 cores, F 2.1GHz, RL2C 0-2 ways, RLLC 0-1 ways, RBW 50%, P^a 0.42, P^t 0.31。U_Low, C 12-15 cores, F 2.8GHz, RL2C 3-6 ways, RLLC 2-4 ways, RBW 40%, P^a 9.12, P^t 7.19。

术语一般如何实现？如何使用？
- Profiling 在专用节点上离线执行，每个 bucket 的 profiling 结果可重用给同一硬件型号的数千个生产节点（amortize profiling cost）。
- Runtime 查询是纯查表操作（<1ms），无学习/推理开销，适合 latency-sensitive 的 LLM serving 场景。
- 论文指出 AUM 的 limitation 是依赖 runtime 控制而非 online learning 来持续补充 AUV Model——AU Bucket 本身是静态的。

涉及论文标题：
- 90-AUM_Unleashing_HPCA-2026-Wang.pdf

## Collision-aware Allocation Tuner

术语是什么？
Collision-aware Allocation Tuner 是 AUM Runtime AU Controller 的第三阶段，负责在 AU 应用和共享负载之间动态调整硬件资源分配（LLC ways via CAT, Memory Bandwidth via MBA），同时检测和避免两者之间的性能碰撞（collision）。它通过持续监控 AU 应用 token 延迟 P^m，与 SLO 比较来决定 aggressive harvest（回收 AU 闲置资源给共享应用）或 conservative return（归还资源给 AU 应用）。关键机制：(1) Aggressive mode——P^m 满足 SLO 时，使用 AUV Model 中的平均性能 P^a 指导资源回收，优先回收对 AU 性能影响最小的资源（decode 的 LLC way 回收 <5% degradation）；(2) Conservative mode——P^m 超标时，使用 tail 性能 P^t 指导资源归还；(3) Collision detection——当 δ_AU = Σ U_AU × P^m / SLO > threshold (2) 时，说明仅靠资源调整已不够，触发 Core Switcher 重新划分核心区域。

从 kernel 调度角度拆解术语：
Allocation Tuner 的每控制迭代 kernel 资源调度伪代码：
```
Input: 当前 token latency P^m, SLO_H (dTTFT-t_wait), SLOL (dTPOT+LAG_i)
       AUV Model M, 当前核心划分 C={CH,CL,CN}

// Step 1: 检测 performance 状态
if P^m_H < SLO_H and P^m_L < SLOL:
    // AU 应用有 slack → aggressive harvest
    δ_AU ← Σ U_AU × SLO / P^m       // 正值，表示 slack 量
    R_AU ← M(P^a_H, P^a_L)          // 用平均性能指导：可回收更多资源
    // 资源回收优先级：LLC ways (AU affinity 低) → Memory BW (AU affinity 高)
    R_share ← R_total - R_AU
else:
    // AU 应用性能不足 → conservative return
    δ_AU ← Σ U_AU × P^m / SLO       // 负值或大正值，表示不足量
    R_AU ← M(P^t_H, P^t_L)          // 用 tail 性能指导：保守归还
    R_share ← R_total - R_AU

// Step 2: Collision check
if |δ_AU| > threshold (2):
    // 仅调资源不够，触发核心区重划分
    {C, F} ← M(δ_AU, P_AU, C_AU, F_AU)  // Core Switcher

// Step 3: 应用资源分配
pqos -e "LLC:0=LLC_mask(R_AU.LLC); mba:0=RBW"
```

术语一般如何实现？如何使用？
- Allocation Tuner 通过 Intel pqos 接口实现 CAT/MBA 的动态调整。LLC way 调整的 latency 可忽略（MSR 写入操作），Memory BW 调整同样轻量。
- 资源回收优先级基于 offline profiling 确定的 AU affinity：Prefill（高 AU）对 LLC way 敏感 (~20% degradation)，Decode（低 AU）对 LLC way 不敏感 (<5%) 但对 Memory BW 敏感 (DRAM bound ~31.2%)。
- 相比静态 RP（如 PARTIES 的固定资源分区），Collision-aware 实现了 demand-driven adaptive allocation。

涉及论文标题：
- 90-AUM_Unleashing_HPCA-2026-Wang.pdf
