## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出arithmetic intensity-aware operator scheduler，在PIM+GPU异构系统上动态调度speculative decoding的DLM FC和TLM attention operator。核心实现：(1) 初始映射：DLM attention每次迭代每请求仅生成1 token，算术强度低→固定映射到PIM；TLM FC因Shared Pool聚合token后变为compute-bound→固定映射到GPU/xPU。(2) DLM FC动态调度：每次prediction后Scheduler识别仍可继续drafting的请求数→估算effective micro-batch size→近似DLM FC算术强度→与预标定的PIM compute-bound和GPU memory-bound阈值比较→决定FC operator在PIM还是GPU执行。当effective batch size从12降至4时，DLM FC算术强度从GPU带宽限制区降到PIM计算限制区，optimal target从GPU变为PIM。(3) TLM attention动态调度：verification前Scheduler统计Shared Pool中每请求draft token数→估算TLM attention算术强度→同理与阈值比较决定PIM或GPU执行。draft length从1增至8时，TLM attention即使memory-bound在GPU上也优于PIM。实验：SADDLE变体Ssaddle-s（含动态operator mapping）相比Ssaddle-p（仅Shared Pool无动态mapping）再提升1.13×吞吐。无operator scheduling时SADDLE 9.51% ops在PIM、90.49%在GPU执行；启用后变为14.89%和85.11%，吞吐提升1.21×。对比baselines GPU-AD/GPU-SD/PIM-AD/PIM-SD，SADDLE平均吞吐提升3.36×/2.88×/1.94×/1.71×。

- 后端平台是什么，配置是什么。
  8个SADDLE PIM devices，每device含1×NVIDIA A100 GPU (80GB HBM2e, peak bandwidth 1555 GB/s) + 5×HBM3 stacks (各16GB, 5.2Gbps/pin)。HBM3 PIM chip每bank附1 PE (16 FP16 multipliers + 16 FP16 adders, 256-bit operands/cycle)，pCH内所有PE跨bank并行。buffer die集成SFU（softmax/layer norm/activation非矩阵运算）。PIM内部带宽144 TB/s (9× DGX system 16 TB/s)。GPU baselines在8×A100 DGX系统(DGX A100)上评估。

- 评估性能的软件/脚本是什么。修改了什么。
  构建cycle-accurate simulator，修改Ramulator2（DRAM simulator）和ATTACC（PIM accelerator simulator）来模拟GPU systems和SADDLE。输入系统配置和模型规格，输出execution time和energy consumption。PE面积/能耗通过Synopsys Design Compiler 28nm 1GHz综合并缩放到DRAM process评估。HBM energy参考prior work的activation/read energy值。GPU baselines (GPU-AD/GPU-SD)使用DeepSpeed Inference在A100 DGX上评估，PIM baselines (PIM-AD/PIM-SD) 基于AttAcc HBM-PIM架构。benchmark：FP16精度，TLM/DLM组合：Llama3.1-70B+Llama3.2-1B、OPT-66B+OPT-1.3B、OPT-175B+OPT-6.7B。数据集：Dolly instruction-following dataset。BS=16-128，max sequence length=1024 (OPT-175B=512)。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未提供SADDLE simulator开源链接（HPCA 2026）。Ramulator2为开源项目：https://github.com/CMU-SAFARI/ramulator2。ATTACC论文发表于ASPLOS'24。SADDLE operator scheduling使用流程：
  1. 预标定阶段：离线测量每个hardware device的peak compute performance和memory bandwidth→标定PIM compute-bound vs GPU memory-bound阈值线（roofline模型，Fig.6）
  2. Scheduler初始化：DLM attention固定→PIM（每iteration 1 token/request，低arithmetic intensity）；TLM FC固定→GPU（token pooling后compute-bound）；DLM FC和TLM attention标记为dynamic
  3. 每次prediction后：Scheduler统计仍活跃请求数→计算effective micro-batch size→估算DLM FC arithmetic intensity→与预标定阈值比较→若低于PIM compute-bound→remap到PIM；若高于GPU memory-bound→保留GPU
  4. 每次verification前：Scheduler统计Shared Pool每请求token数→估算TLM attention arithmetic intensity→同理remap决定PIM或GPU
  5. 例如：一个micro-batch初始12请求，8个短draft请求先完成→effective batch size从12降到4→DLM FC arithmetic intensity降低→Scheduler将DLM FC从GPU remap到PIM

## VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出VDHA（Vector-Driven Hash Aggregation），面向GPU的weighted SpMSpV kernel，核心设计：(1) Shared-memory hash aggregation：每个CTA维护私有shared-memory hash table（2048-entry，4B key+4B value），将partial product (row_idx, mat_val⊗vec_val)插入hash table做local accumulation。使用modulo hash (idx%table_size) + linear probing with fixed stride（(h+C)%table_size）降低冲突，probe次数超过FALLBACK_ITER阈值后fallback到global atomicAdd。(2) Short/long-column decomposition with reordering：按列长LEN_THRES=128分为short/long column，long column按SPLIT_SIZE=256切分为segment→按segment首非零row index排序（metadata排序非数据排序，O(S log S) cost）增强跨列locality→short column直接block-mapped。(3) Fetch-compute-writeback pipeline：将传统fetch→writeback两阶段重构为三阶段pipeline，使用double buffering通过cp.async异步fetch下一tile的同时对当前tile做hash aggregation，重叠hash计算与global memory访问延时，hash computation cost从16.7%降至12.3%，stall ratio从>45%降至~15%。(4) 轻量级预测模型：基于matrix structural features (num_rows, num_nnzs, bandwidth index B, variance index V, vector sparsity)训练decision tree classifier判断VDHA是否有利，91.3% accuracy（F1 score），结合fallback到BlockAtomic或best-of-7可将SuiteSparse geomean speedup从1.13×提升至1.16×（fallback atomic）或1.22×（best-of-7）。实验比较：在Konect/LAW (>100 web graphs) 和 SuiteSparse (>200 scientific matrices，均≥5M NNZ)上，对比7个baseline（cuSPARSE row-major SpMV、NaiveSpMSpV/HolaSpMSpV row-major SpMSpV with value validation、BlockSort/GlobalSort/BlockAtomic/GlobalAtomic column-major SpMSpV），4个vector sparsity (0.01/0.05/0.10/0.20)。VDHA实现Konect/LAW geomean 1.41× speedup (max 3.42×), SuiteSparse geomean 1.13× speedup (max 2.55×)。消融实验：hash only=0.689×, hash+split=0.947×, hash+split+reorder=1.000× normalized performance。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU (40GB HBM2e memory, peak bandwidth 1555 GB/s, SM80)，AMD EPYC 7742 CPU。编译：CUDA nvcc 12.5，-O3优化。

- 评估性能的软件/脚本是什么。修改了什么。
  自研CUDA kernel实现VDHA。baseline包括：cuSPARSE (row-major SpMV)、NaiveSpMSpV和HolaSpMSpV（基于NaiveSpMV[30]和HolaSpMV[31]开源代码添加bitmask value-validation实现row-major SpMSpV）、BlockSort/GlobalSort/BlockAtomic/GlobalAtomic（基于Adaptive SpMSpV[20]实现4种column-major kernel：BlockSort/GlobalSort采用sort-reduce避免atomics、BlockAtomic/GlobalAtomic采用atomic write-back、Block级按列分组/Global级按nonzero均匀分配）。所有kernel在同一软件和硬件环境下执行保证公平性。使用efficient performance metric（efficient NNZ/runtime，efficient NNZ=matrix NNZ×vector sparsity）评估性能。NVIDIA Nsight Compute用于profiling warp stall cycles和memory throughput。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未明确提供开源链接（PPoPP'26发表，可能pending release）。kernel使用流程：
  1. 预处理阶段：矩阵以CSC格式存储→Vector Processing扫描input vector识别active columns→按LEN_THRES=128分类short/long columns→long columns按SPLIT_SIZE=256切分为segments→segments按首非零row index排序增强locality→所有segments block-mapped分配CTAs
  2. Launch CUDA kernel：每个CTA含256 threads（8 warps），维护2048-entry shared-memory hash table（16KB），使用double buffering。Fetch stage通过cp.async异步加载当前segment indices/values到shared memory buffer→Compute stage执行hash aggregation：modulo hash计算起始位置→atomicCAS+linear probing插入(key,val)→更新hash table value→超过FALLBACK_ITER阈值的更新fallback到global atomicAdd→Writeback stage：hash table接近满时按bucket order flush到global memory（coalesced writes）
  3. 以it-2004 matrix（41.2M rows, 1.15B NNZ, vector sparsity=0.1, A100）为例：Vector Processing识别active columns→long columns (>128 NNZ)切分为256-NNZ segments→segments按首row index重排序→每个CTA处理一个segment在shared memory hash table局部聚合→flush时hash table bucket order提供partial order减少global atomic conflicts→fetch-compute-writeback pipeline重叠memory access与hash computation

## Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Drawloom，面向GPU Tensor Cores和Sparse Tensor Cores的SpMV优化库。核心kernel/runtime设计：(1) ArbitWeave mapping策略：根据TC shape的结构比V=mma_m/mma_n将稀疏矩阵切分为V-width row strip，按每strip的NNZ分为Long Mapping（nnz>T1，直接映射到TC block）、Medium Mapping（T2<nnz≤T1，多个row strip聚合到TC block）、Short Mapping（nnz≤T2，利用2:4 sparsity映射到SpTC block），支持任意TC shape（如m16n8k16、m16n8k8）。SpTC的2:4 sparsity通过remapping非零元到配对box并带metadata实现，节省50%内存。(2) Zig-zag Chained Format (ZCF)：long/medium part用longPtr/mediumPtr+Cid+Val存储、enforce vectorized memory access（zcf_value_stride=8 for FP16, 4 for FP32/FP64）、支持128-bit per-thread memory transaction；short part用shortCid+shortVal压缩存储、直接兼容SpTC计算。(3) Multi-stage Register Pipeline：将Fetch CID、Load X、TC Comp拆为FillSMEM→FillREG→Comp→EmptySMEM→EmptyREG五个阶段，通过async-copy实现GMEM-to-SMEM异步传输，设置delaySMEM控制GMEM-SMEM overlap、delayREG控制SMEM-REG+Compute overlap，消除warp stall。(4) Two-level load balancing：matrix-level按NNZ分组三类row、warp-level将Long Mapping TC blocks均匀分配给warp（WarpLoad参数）。实验比较：FP16/FP32/FP64三种精度，A100和H100双平台，对比cuSPARSE v12.0、Best-cuSPARSE（CSR/CSC/BSR/SELL选最优）、DASP（FP16/FP64）、Spaden（FP16/FP32）、FastLoad（FP64）、TileSpMV（FP64）。A100上FP16相对cuSPARSE 2.71×、DASP 1.26×；FP32相对cuSPARSE 2.95×；FP64相对cuSPARSE 2.47×、DASP 1.49×。H100上FP16相对cuSPARSE 1.90×、DASP 1.18×；FP64相对cuSPARSE 1.54×、DASP 1.56×。消融实验：v1(naive m8n8k4)→v2(+ArbitWeave)平均1.56×→v3(+ZCF)平均1.91×→v4(+Multi-stage Pipeline)平均1.46×。多级流水线warp stall改善3.02×、memory throughput改善2.61×（M1矩阵）。

- 后端平台是什么，配置是什么。
  NVIDIA A100（Ampere架构，Compute Capability 8.0，80GB global memory），Host CPU Intel Xeon Gold 6336Y。NVIDIA H100（Hopper架构，Compute Capability 9.0，80GB global memory），Host CPU Intel Xeon Platinum 8468。编译：NVCC 12.6，-O3优化。

- 评估性能的软件/脚本是什么。修改了什么。
  自研CUDA kernel实现Drawloom library。baseline包括：cuSPARSE v12.0（CSR格式、default algorithm）、Best-cuSPARSE（CSR/CSC/BSR/SELL中选择最优格式）、FastLoad（CSC格式FP64 memory-optimized）、TileSpMV（tile-based FP64，仅支持CUDA 11）、DASP（TC-based SpMV FP16/FP64）、Spaden（bitmap-based sparse format FP16/FP32）。benchmark执行1000次取平均值。矩阵预处理格式转换：CSR→CSC/BSR使用官方cuSPARSE函数、SELL使用PETSc。NVIDIA Nsight Compute用于profiling warp stall和memory throughput。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：Zenodo DOI: 10.5281/zenodo.17709956。kernel使用流程：
  1. 预处理阶段：读取CSR格式稀疏矩阵→按NNZ per row重排序→通过T1=(mma_m/v)*mma_k*WarpLoad和T2=mma_k阈值分类为long/medium/short三类→long/medium部分构建ZCF for TC（longPtr/mediumPtr+Cid+Val，zcf_value_stride对齐128-bit transaction）→short部分构建ZCF for SpTC（2:4 sparsity压缩+metadata编码非零位置）
  2. Launch CUDA kernel：每个thread block含4 warps→warp按ZCF row_ptr索引TC blocks→FillSMEM阶段async-copy将sparse A和Cid异步拷贝到SMEM→FillREG阶段从SMEM索引load vector X到寄存器→Comp阶段执行TC MMA指令（如m16n8k16）→Long Mapping结果intra-warp shuffle归约→inter-warp reduction kernel聚合最终Y
  3. 以pwtk矩阵（0.2M rows, 11.6M NNZ, A100 FP16）为例：ArbitWeave选择m16n8k16 TC shape→v=2 row strip→long/medium row strip经ZCF填充TC blocks→short row strip通过2:4 sparsity压缩映射到SpTC→multi-stage pipeline overlap memory与TC computation→最终输出Y向量。

## MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出MetaAttention attention runtime，将用户定义的attention变体自动生成跨后端优化kernel。核心kernel/runtime设计：(1) Parallel Pattern kernel：将relevance scoring实现为Q×K^T matmul、aggregation实现为scores×V matmul，采用online row-wise normalization technique（类似FlashAttention的online softmax）将score tile保持在on-chip memory，通过tiling遍历K/V sequence避免写出完整attention matrix到global memory；(2) Recurrent Pattern kernel：relevance scoring实现为Q×state matmul、aggregation实现为state += K[i]×V[i]状态更新，采用chunk parallelism将长序列切为可并行处理的chunk，在chunk内维护recurrent state，elementwise/reduction逻辑融合到recurrent kernel内；(3) RowNorm online接口：将行归一化拆为online_prologue（初始化row_max/row_sum状态）→online_forward（逐tile更新局部reduce结果、全局状态和对前tile rescale）→online_epilogue（完成最终输出），runtime在遍历K/V tile时同步更新归一化状态；(4) kernel templates实现：包含global→shared、global→register、shared→register三级数据搬移，以及输入位于shared/register时的matrix multiplication，调度计划确定后选择合适template并inline已lowering的customizable functions。实验比较：operator-level normalized latency，覆盖10种attention mechanism（Softmax Attention/DeepSeek-V2-Lite/LLAMA-3.1-8B/DiffTransformer-3B、Sigmoid Attention、ReLU Attention、Retention Parallel、Mamba2 SSM、Retention Recurrent、Gated Retention/RFA-Big/YOCO-13B、Multi-head Latent Attention/DeepSeek-V3、Sparse GQA/SeerAttention）。H100上：Softmax attention相对FlashAttention-3在Diff-Transformer-3B forward平均1.61× speedup（headdim_qk≠headdim_v无需padding）；customized parallel attention平均3.6× speedup（1.1×∼10.4× over FlashSigmoid/PyTorch）；recurrent attention forward/backward平均1.66×/1.78× over Flash-Linear-Attention；MLA性能接近FlashMLA且比MLA Triton快4.6×；Sparse GQA平均1.71× over SeerAttention Triton kernel。MI250上Softmax/ReLU/Mamba2/RetNet Recurrent subset forward 3.3×、backward 2.0× over baseline。

- 后端平台是什么，配置是什么。
  NVIDIA H100 SXM5（CUDA 12.4, Triton 2.3.1）：使用Tensor Memory Accelerator (TMA)异步加载、Tensor Cores MMA，基于TileLang和CUTE两种backend framework实现。AMD Instinct MI250（ROCm 6.2.4, Triton 3.1.0）：使用Matrix Cores + asynchronous copy units，基于TileLang backend实现。也支持MI300X。

- 评估性能的软件/脚本是什么。修改了什么。
  Operator-level benchmark脚本执行forward/backward latency测量。模型使用PyTorch/Transformers实现。end-to-end推理基于Transformers替换attention operator（H100单卡）。end-to-end训练基于TRL（H100单卡，seqlen 8k）。benchmark覆盖batch size 1/8，sequence length 2K/4K/8K，解码配置query seqlen=1+不同KV cache length。代码修改：MetaAttention runtime实现的attention kernel替代原有PyTorch/flash-attn等library的attention调用。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：GitHub https://github.com/SJTU-IPADS/MetaAttention，Zenodo DOI: 10.5281/zenodo.17701680。kernel使用流程：
  1. 用户定义attention模板（pattern+shape+customizable functions），MetaAttention frontend trace生成computation graph
  2. Scheduler搜索IntermediateTensor tile/mem/pipeline stage配置，生成scheduling plan
  3. Runtime根据plan选择kernel template（parallel/recurrent pattern），inline customized functions，生成后端代码
  4. 以MLA解码为例：Parallel Pattern，query seqlen=1，head=128，head_kv=1，dimqk=576，dimv=512，KV cache length 2048-8192→每个tile加载Q+一段K/V cache→计算relevance score→on-chip执行custom Mod/RowNorm online→聚合V得output→遍历后续KV tile。全部计算在单一fused kernel内完成

## Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  Bullet的kernel调度/运行时计算实现是动态SM分区下的prefill/decode kernel并发调度：通过libsmctrl_set_stream_mask修改CUDA stream metadata，将prefill和decode的kernel流绑定到GPU上不同的SM子集执行，实现同一GPU内的空间分区。prefill engine在分配的SM子集上逐layer发射QKV/O-proj/MLP/attention等kernel，decode engine在另一SM子集上以CUDA Graph step发射decode kernel。Resource manager在收到scheduler的repartition command后微秒级（平均4.1us）修改CUDA stream的SM mask，使后续kernel立即在新SM子集运行。实验对比Nsight Systems采集的SM active cycles、Tensor Core utilization和memory-bandwidth utilization，以及不同SM partitioning策略下的throughput、TTFT、TPOT。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80GB (108 SM/GPU, NVLink 600 GB/s)、NVIDIA H100 (132 SM/GPU, 600 GB/s)、NVIDIA H20 (78 SM/GPU, intra-node 400 GB/s)，CUDA 12.4。

- 评估性能的软件/脚本是什么。修改了什么。
  Nsight Systems采集SM/Tensor Core/memory bandwidth utilization。libsmctrl修改：使用libsmctrl_set_stream_mask()在运行时修改CUDA stream的SM mask（GPC配置掩码），使后续kernel launch限制在指定SM子集。CUDA MPS用于spatial sharing支持。CUDA Graph用于decode step的低开销一次性发射（减少kernel launch overhead）。实验中prefill和decode engine各自持有独立的CUDA stream，resource manager在scheduler下发repartition command后立即修改stream mask。SLO-aware scheduler周期性（每个prefill layer group或decode step后）读取全局状态并搜索新SM分区方案。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源仓库（https://github.com/zejia-lin/BulletServe）包含修改版libsmctrl。SM分区使用流程：
  1. 通过libsmctrl_get_gpc_info()获取GPU GPC/TPC拓扑
  2. 构建SM mask（以16 SM为粒度分组），论文在A100上定义6种SM配置，H100上7种
  3. 调用libsmctrl_set_stream_mask()将stream绑定到目标SM mask
  4. 后续kernel launch（通过PyTorch CUDA stream）自动限制在mask指定的SM子集执行
  5. scheduler根据实时队列状态和SLO压力下发repartition command，resource manager更新stream mask
  例如：prefill队列堆积时将prefill stream mask扩展到接近全部SM，decode SLO紧张时缩小prefill mask以释放更多SM给decode。SM mask更新开销平均4.1us，metadata传递平均0.21ms，performance prediction平均10.2us。

## V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出DRE（Dynamic KV Cache Retrieval Engine），一个硬件加速单元，负责streaming video LLM中KV cache retrieval的runtime计算和调度。DRE包含两大模块：(1) KVPU（KV Cache Prediction Unit）：集成HCU（Hash-bit Cluster Unit）和WTU（WiCSum Threshold Unit），加速ReSV中bit-level聚类和early-exit thresholding两类不规则计算——这些计算在GPU上因条件分支和数据依赖导致严重underutilization和延迟。(2) KVMU（KV Cache Management Unit）：管理分层KV cache memory（recent KV→V-Rex memory, old KV→CPU/storage offload, retrieved KV→prefetch回V-Rex memory），实现cluster-wise memory mapping使同cluster的token连续存储以最大化PCIe带宽利用。runtime pipeline：LXE生成hash-bit→HCU做Hamming distance clustering更新HC table→LXE计算Q×KeyCluster^T→WTU做early-exit WiCSum thresholding输出selected token indices→KVMU预取selected KV entries→attention与KV prediction/concurrent fetch重叠。实验比较：edge (V-Rex8)和server (V-Rex48)的latency/FPS/energy efficiency/bandwidth overlap across KV cache sizes (1K-40K)，对比AGX Orin和A100上的FlexGen/InfiniGen/InfiniGenP/ReKV。消融实验：AGX+ReSV (2.8× speedup但KV prediction仍占48% latency) → +KVPU (6.0× speedup, 9.2× energy reduction) → +KVMU (8.1× speedup, 10.2× energy saving)。Bandwidth分析显示KV prediction短时达600 GB/s但可hidden in attention，KV retrieval仅占~1% DRAM bandwidth可与attention/FFN并发。Roofline分析：V-Rex8达理论峰值71.5% throughput。

- 后端平台是什么，配置是什么。
  Edge: V-Rex8 (8 cores, 53.3 TFLOPS BF16, LPDDR5 204.8 GB/s 256-bit, PCIe 3.0 x4 4 GB/s, M.2 NVMe SSD)。Server: V-Rex48 (48 cores, 319.5 TFLOPS BF16, HBM2e 1935 GB/s 5120-bit, PCIe 4.0 x16 32 GB/s, DDR4 CPU memory)。Baseline GPU: NVIDIA Jetson AGX Orin (FP16 54 TFLOPS, 32 GB LPDDR5, ~40W)、NVIDIA A100 (FP16 312 TFLOPS, 80 GB HBM2e, ~300W)。V-Rex单核RTL 14nm Synopsys Design Compiler综合，0.8V 800MHz。

- 评估性能的软件/脚本是什么。修改了什么。
  自研custom cycle-level simulator + DRAMSim3 (DRAM) + MQSim (SSD)。模拟器建模GPU/CPU data movement bandwidth（A100和AGX Orin实测参数），集成DRAMSim3和MQSim进行系统级评估。V-Rex单核：Verilog RTL→Synopsys Design Compiler 14nm综合→PrimeTime PX功耗分析。GPU power通过NVIDIA-SMI和tegrastats实测。PCIe power按3W/lane，SSD power基于Kioxia BG6 spec。所有参数集成到custom simulator中。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未提供V-Rex simulator/RTL开源（HPCA 2026）。DRAMSim3开源 (https://github.com/umd-memsys/DRAMSim3)，MQSim开源 (https://github.com/CMU-SAFARI/MQSim)。runtime pipeline使用流程：
  1. Prefill开始：LXE的VPE做hash-bit generation（Key×Hyperplane→binarize），输出当前frame的hash-bit vector
  2. HCU clustering：HCU接收hash-bit→从key cache hash-bit memory读取已有KeyCluster hash-bit→NHCU_h个并行XOR accumulator计算Hamming distance→与Thhd=7比较→更新HC table (cluster id/token idx/KeyCluster/token count)，HC table存于8KB on-chip memory
  3. ScoreCluster计算：LXE的DPE做Query×KeyCluster^T矩阵乘法（BF16）→得ScoreCluster→送入WTU
  4. WTU thresholding：WTU cores并行处理→preprocess step预计算weighted sum/min/max/Th_wics→token selection step从高分bucket开始bucket sort→cumulative sum→与Th_wics比较→early-exit→输出selected cluster bitmask→汇总所有row→通过HC table映射为token indices
  5. KVMU prefetch：根据selected token indices通过PCIe从storage/CPU memory预取KV entries→cluster-wise memory mapping使同cluster token连续存放→batch fetch提高PCIe有效带宽
  6. Light Attention执行：仅对selected K/V token做attention→同时KV prediction for next layer与current attention/FFN重叠（bandwidth analysis证实prediction spike 600 GB/s可hidden, retrieval仅占1% DRAM BW可concurrent execution）

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出ZipGEMM，一个fused decompression-GEMM kernel，将TCA-TBE解压与Tensor Core矩阵乘法融合为单一CUDA kernel。关键设计：(1) 基于split-K tiling，每个thread block分4阶段：Tile Loading（LDGSTS.128异步加载压缩权重+激活到shared memory）→ Warp-Level Decoding（每个warp独立解压权重到register，利用bitwise OR/POPC/shfl指令做spatial indicator判断和dynamic addressing）→ Activation Register Transfer（LDSM.M88将激活从shared memory搬入register）→ Tensor Core Computation（mma.m16n8k16执行BF16 GEMM）；(2) 两级software pipeline：coarse-level tile double buffering重叠global→shared memory传输与计算；fine-level slice-wise interleaving重叠shared→register解压与Tensor Core计算；(3) TCA-TBE的三层tiling (FragTile 8×8, TensorCoreTile 16×16, BlockTile 64×64) 直接对齐Tensor Core mma operand layout，消除runtime坐标变换。Decompressor设计三阶段：spatial bitmap indicator（bitwise OR三bitmap得64-bit indicator mask，每bit标识元素压缩/fallback状态）→ dynamic addressing（POPC并行prefix sum计算每个线程的buffer offset）→ fast exponent reassembly（base_exp + codeword算术恢复exponent，无shared memory table lookup）。实验比较kernel-level speedup：对比cuBLAS_TC、DietGPU、nvCOMP、DFloat11，在RTX4090上平均1.31× (peak 1.71×)，L40S上平均1.36× (peak 2.21×)。还对比了standalone Decompression kernel性能：ZipServ-Decomp平均2.14×/1.83×/1.10× over DietGPU/nvCOMP/DFloat11。跨代GPU forward compatibility在RTX5090上验证（1.34×–1.87× over cuBLAS）。

- 后端平台是什么，配置是什么。
  NVIDIA RTX4090 (Ada Lovelace, 24GB, CC 8.9, SM频率2520 MHz)、NVIDIA L40S (Ada Lovelace, 48GB)、NVIDIA RTX5090 (Blackwell, 32GB, CC 12.0)。也对比了A100 (1410 MHz)和H800 (datacenter GPUs)。编译：NVCC 12.4 (RTX5090用NVCC 12.8)。

- 评估性能的软件/脚本是什么。修改了什么。
  Nsight Compute (NCU) profiler用于micro-level分析：测量DRAM reads (↓29.3%)、ALU utilization (66.0% LOP3/IADD/POPC指令)、Tensor Core utilization (保持cuBLAS的71.6%)、shared memory bank conflicts (仅~4.7K，vs DietGPU百万级)。benchmark脚本执行100 warm-up + 1000 timed iterations。ZipGEMM kernel通过PTX内联汇编实现mma.m16n8k16调用，使用LDGSTS.128 bypass L1 cache直接写入shared memory，cp.async.wait_group<0>() + __syncthreads()实现hierarchical barrier同步，__popc()和__shfl_sync()实现warp-level prefix sum。split-K配置和tile size针对不同matrix shape调优。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/HPMLL/ZipServ_ASPLOS26.git。ZipGEMM kernel编译为独立.so库（nvcc编译）。使用流程：
  1. 编译：`mkdir build && cd build && cmake .. && make` 生成libzipgemm.so
  2. kernel benchmark：通过C++ API调用，传入TCA-TBE格式的压缩权重buffer、激活tensor和matrix dimensions
  3. 性能分析：Nsight Compute `ncu --set full -o profile ./benchmark` 采集micro-architecture counters
  4. 例如LLaMA3.1-8B GateUp_proj layer (M=14336, K=4096, N=32)在RTX4090上，ZipGEMM 0.194ms vs cuBLAS_TC 0.275ms (1.42×)。DRAM读从~3.2GB降至~2.3GB (−29.3%)，ALU指令增加但被两级pipeline隐藏，Tensor Core利用率保持71.6%。shared memory bank conflict仅~4.7K（对DietGPU的百万级），因为TCA-TBE的triple bitmap layout确保coalesced access。

## DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出TreeSort-Verify机制，将tree-based speculative decoding的irregular causal attention masks转换为高效block-diagonal lower triangular矩阵，实现GPU上block-parallel的tree verification。核心设计：(1) Path-Packing Reordering：定义重排序函数π将token tree节点按ancestor关系排序（parent先于child），使重排后的causal mask矩阵M_reordered[i,j] = 1 iff π(t_j) ≤ π(t_i)且t_j是t_i的ancestor，形成block-diagonal lower triangular结构；(2) Block Decomposition：将重排序列分区为K个连续block，每个block内独立使用标准lower triangular mask，tree attention分解为Att_tree = ⊕_{k=1}^{K} Att_block(Q_Bk, K_Bk, V_Bk, M_Bk)，每个block直接调用高度优化的cuBLAS GEMM kernel；(3) Memory-Friendly Verification：连续block布局improves GPU memory locality，KV-cache compact存储，block-diagonal结构支持GPU SM间的pipelined parallel执行。此外FPGA侧：(4) Multi-Branch Mapping to Block Events：利用shared prefix，draft model将多分支映射到block event，在Linear阶段增加weight reuse（多分支共享前缀权重），在Q×K^T阶段复用shared prefix并在pipeline末尾仅改变loading address以产生额外token，在S×V阶段通过最后round accumulation将额外token归并回原sequence length；(5) Ping-Pong PE调度：KERload = PE_Num × Data_width / Bandwidth，IFMload = KERload + CAS_Latency，实现computation与data loading重叠。实验比较：(1) TreeSort-Verify ablation贡献（2.46× vs HW-Branch-only 2.21×）；(2) FPGA operator execution efficiency（matrix multiplication loading和computation均达86.2%-97.5%）；(3) 不同tree verification方式的mask结构对比（sequence-based vs tree-based vs TreeSort-Verify）。

- 后端平台是什么，配置是什么。
  GPU侧：NVIDIA RTX 4090 (512 Tensor Cores, 2230MHz, 24GB DRAM, 1008 GB/s BW) 和 NVIDIA A100 (432 Tensor Cores, 1410MHz, 80GB DRAM, 1935 GB/s BW)，CUDA 12.1，cuBLAS。FPGA侧：AMD V80 (300MHz, 10848 DSPs, HBM) 和 AMD U200 (300MHz, 6480 DSPs, DDR)，Xilinx Vivado 2024.1。CPU：Intel Xeon 4310。

- 评估性能的软件/脚本是什么。修改了什么。
  TreeSort-Verify在GPU上实现，通过path-packing对token tree节点排序后划分block，每个block调用cuBLAS GEMM。FPGA kernel使用Verilog HDL定制PE微架构，含branch concatenation（多weight buffer + 选择连线）和DSP packing（单DSP双BF16×BF16乘法）。性能测量：Xilinx xbutil (FPGA power)、nvidia-smi (GPU power)。TreeSort-Verify修改了tree attention的causal mask计算方式，从irregular per-path mask转为block-diagonal形式。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/ShaoqiangLu/DFVG。TreeSort-Verify使用流程：
  1. GPU kernel编译：`cd gpu/ && make all`
  2. TreeSort-Verify在verify阶段自动触发：FPGA生成的token tree通过PCIe传输到GPU→TreeSort-Verify对树节点做path-packing重排序→按block划分→每个block独立调用cuBLAS GEMM→结果按原始index顺序recombine
  3. 例如FPGA draft model生成一棵含D1-D6的token tree（深度γ，分支数k_max），TreeSort-Verify按ancestor关系重排序后划分为若干连续block，每个block内部使用标准lower triangular mask做attention，避免了传统tree-based方法中irregular sparse mask导致的GPU memory divergence和vectorized computing underutilization。

## PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  PAT实现了一套prefix-aware decode attention kernel，遵循pack-forward-merge执行范式：(1) Pack阶段：pack scheduler将vLLM block table转为prefix tree，用memory-centric profit model比较split/merge策略，生成CTA partition以最小化global KV cache访问，lazy update机制让调度结果在block table未变化时跨continuous-batching iterations复用并与pre-attention tasks重叠；(2) Forward阶段：multi-tile kernel为每个CTA从offline求解的可行tile set中选择合适的(m,n) tile配置（m为Q tile size，n为KV tile size），runtime tile selector以constant-time lookup选m（round-up规则覆盖当前CTA query数）和n（根据KV length和execution bubble trade-off选择），multi-stream forward为不同tile配置创建独立CUDA stream并行执行，long-KV split将KV超长CTA沿KV维拆分；(3) Merge阶段：lightweight merge kernel用online softmax合并每个query被多个CTA计算出的partial max score、log-sum-exp accumulator和partial value-weighted sum为最终输出。Kernel约3k行Cutlass/CuTe+C++实现，数据搬运使用cp_async+double buffering。实验比较attention latency和memory read/write量，对比FlashAttention v2.5.9（query-centric, tile固定(64,128)）、FlashInfer v0.2.5（query-centric, dynamic CTA partitioning, tile(16,128)）、FastTree（KV-centric, compute-oriented cost model, 两种tile configs）、RelayAttention（KV-centric, pack first-level prefix）、RelayAttention++（扩展版使用vLLM-style KV-cache reuse+L2 cache）、DeFT（KV-centric, fixed(32,16) tile）、Cascade Inference（KV-centric, fixed packing）。Synthetic batch下PAT相对FlashAttention最高21.5×、相对FlashInfer最高11.7×、相对FastTree最高3.2×加速，平均降低attention latency 53.5%。每配置重复20次取平均。

- 后端平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB (108 SM, 40MB L2, 80GB HBM)，NVIDIA H100-SXM4-80GB (132 SM)。CUDA 12.4, PyTorch 2.7.0, vLLM v0.9.0。Head configs: (#heads, #kv_heads) = (64,8), (32,8), (16,8), (32,32)，head dimension=128, FP16。端到端模型：Qwen3-8B、Llama-3-8B（A100单卡）；Qwen2.5-72B-Instruct（4×A100 TP=2/PP=2）；Qwen3-30B-A3B（单卡A100）。

- 评估性能的软件/脚本是什么。修改了什么。
  NCU profiling测量FlashAttention KV cache traffic（比理论最小值高4.3-8.7×）；PTX profiling分析CTA execution pipeline和execution bubble。Kernel benchmark脚本构造synthetic decode batch：B定义prefix tree结构和leaf数（如B=[1,4,16]表示两级shared prefix和16 leaves），L定义各层KV长度（如L=[128,256,1024]），共20种(𝐵,𝐿)配置。Multi-tile kernel基于Cutlass/CuTe实现MMA，tile配置通过offline solver基于三个约束导出：① register/shared memory约束（上界：m*h*b + n*h*b + 中间结果 ≤ S_smem，offline编译获取R_thr和R_CTA），② bandwidth lower bound（n ≥ LB/(S*C*h*b)确保inflight data覆盖memory latency），③ CUTLASS constraint（m,n为2的幂且≥16）。A100上得到11组可行(m,n)配置，H100上12组。Tile selector运行时constant-time lookup：m用round-up规则（选≥CTA query数的最小可行m避免维度padding），n根据KV length做profiled piecewise决策（长KV偏大n降低concurrency减少tail bubble，短KV偏小n避免compute bubble）。Multi-stream forward为每种active tile配置创建独立CUDA stream。Long-KV split在CTA KV length超batch均值时沿KV维拆分。Ablation对比：PAT-compute（FastTree compute-oriented cost model）、PAT-naive（简单每node独立pack）、PAT-fixed（固定(64,128) tile）、PAT-serial（串行multi-kernel执行）。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/flashserve/PAT（MIT License），Zenodo DOI: 10.5281/zenodo.18217189，Docker镜像 flashserve/pat:ae。使用流程：
  1. 环境：x86-64 Linux, ≥64GB RAM, 200GB disk, A100-80GB, CUDA driver ≥550
  2. 拉取镜像：`docker pull flashserve/pat:ae`
  3. 启动容器：`docker run -it --gpus all --shm-size=64g -v ${PWD}/PAT:/workspace/PAT -w /workspace/PAT flashserve/pat:ae /bin/bash`
  4. Kernel benchmark：`cd /workspace/PAT/benchmark && bash ./run_kernel_bench.sh`（约1.5小时，复现Figure 11）
  5. 端到端实验：`bash ./run_e2e_bench_part.sh`（快速验证，8-10 GPU-hours）或`bash ./run_e2e_bench_full.sh`（完整实验，>60 GPU-hours）
  6. 生成图：`cd /workspace/PAT/plot && python eval_kernel_perf.py --log-file ../benchmark/kernel_perf.json`
  在vLLM中启用PAT仅需设置环境变量：`VLLM_ATTENTION_BACKEND=PAT`（开源版README示例为`VLLM_ATTENTION_BACKEND="PREFIX_ATTN"`，接口名可能存在小幅演进）。Multi-tile kernel移植到新GPU需重新基于shared memory/register/bandwidth约束和CUTLASS requirement推导等价tile set（论文在H100上验证了该procedure的通用性）。

## JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现三类自定义CUDA kernel：(1) Fused smoothing + quantization kernel：整合smoothing transformation、scale/zero point计算与参数重排、KV cache INT2打包。FAVP技术将absmax计算限制在离线校准的稀疏channel集（<2% channels），使quantization kernel避免4.43× naive runtime smoothing overhead（64K seq len下，对比无smoothing baseline仅增加约1×开销）。(2) Memory-efficient ring buffer token cache kernel：预分配ring buffer替代KIVI/SKVQ的sliding-window tensor concatenation，以指针切换和分段量化避免decoding中频繁内存分配与拷贝。(3) Mixed-precision attention kernel：将INT2 dequantization与attention融合为单一kernel。包含两项优化：① INT2-to-FP16高效unpacking：利用FP16在[1024,2047]区间共享exponent=1024的特性，将2-bit值放入mantissa再bitwise OR设置exponent (R2=R2|0x64006400)，再减去1024得到FP16值，每条指令处理两个值（仅需lop3/or/sub三条指令 vs naive每值≥4条指令）。② Unified parameter block layout：将scale、zero point、smoothing factor等四类参数按thread block访问pattern合并对齐，减少memory transactions（示例从20次降至8次）。实验比较kernel runtime：对比SKVQ、KIVI、QServe、FA2 attention kernel。在128K seq、hidden size 4096、32 KV heads下，JanusQuant kernel speedup 6.17× over KIVI、1.69× over QServe、平均1.99× over FA2。Breakdown实验评估FAVP对quantization kernel的改善（64K seq下将4.43× runtime smoothing overhead降至接近无smoothing baseline）和unpacking/parameter reorg对attention kernel的改善（平均1.99×和3.05× over naive mixed-precision baseline）。

- 后端平台是什么，配置是什么。
  NVIDIA A100-PCIE-40GB（单卡）。CUDA 12.6。kernel编译为standalone shared library (.so)，通过Pybind和FlashInfer包装为Python extension。

- 评估性能的软件/脚本是什么。修改了什么。
  Nsight Compute用于Roofline分析（识别attention kernel在2-bit dequantization fused后从memory-bound变为compute-bound）。Kernel-level实验100 warm-up + 10000 runs取平均。量化kernel breakdown分析absmax calculation占quantization kernel >80% overhead（Figure 15a）。端到端实验10 warm-up + 100 runs取平均。修改：开发约3500行CUDA/C++和2500行Python，CUDA kernels包括token cache处理、fused smoothing+quantization、fused dequantization-attention三类。Python侧提供继承PyTorch nn.Module API的custom attention module，通过Pybind和FlashInfer调用CUDA kernels，支持Llama/Mistral/Vicuna/Qwen模型族。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文承诺artifact will be released但截至分析无法确认公开仓库。Kernel使用流程：
  1. FAVP离线校准：部署前在calibration dataset上运行一次（数分钟），为每个attention layer记录absmax频繁channel集（超过90%层仅需<2% channels）。
  2. Decoding时，fused quantization kernel执行：读取FAVP记录的sparse channel indices→仅扫描这些channel计算per-token absmax→smoothing factor = max(|K_i|)^0.5→smoothing transformation→per-channel/per-token group quantization→INT2 packing→追加到quantized KV cache。
  3. Mixed-precision attention kernel执行：每thread block加载unified parameter block→unpack INT2 values via bitwise ops (lop3/or/sub)→dequantize→与FP16 recent KV一起参与attention compute。Kernel利用task parallelism（不同thread block处理quantized/FP16 segments）和asynchronous execution重叠计算与访存。
  4. Kernel可编译为.so并通过Python extension调用，兼容PyTorch/Transformers serving框架。

## High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Quantix fused dequantization-matmul CUDA kernel，将non-uniform 3-bit weight的dequantization与Tensor Core matrix multiplication融合为单一GPU kernel。核心kernel设计：(1) 输入为hardware-aligned bit-shuffled weights（1-bit packed W1'和2-bit packed W2'）、activations A和centroids C，输出Y=A×Dequant(W1',W2',C)；(2) inter-tile层：用cp.async (128-bit width)异步预取future K-tile的W1'/W2'/A到shared memory，与当前tile计算重叠；(3) intra-tile层：从shared memory load subtile到registers→CUDA cores执行in-register dequantization（1-bit+2-bit bit concatenation重建3-bit index→shift+mask按qi=(R>>3i)&0x7提取→centroid lookup用3-bit index查row-specific centroids得FP16 W†）→Tensor Cores执行MMA (A×W†)；(4) in-register dequantization避免了中间结果写回global memory，消除cache-unfriendly pointer chasing和额外指令开销；(5) 两层double buffering：Smem0/Smem1实现inter-tile overlap，Reg0/Reg1实现intra-tile overlap；(6) Split-K将K维切分为多个独立slices并行计算partial sums，最后lightweight reduction kernel合并；(7) 128-bit vectorized memory access（UINT4 reinterpret）使global→shared和shared→register均以单指令传输128-bit chunk。实验比较kernel-level speedup vs FP16 cuBLAS、SqueezeLLM、Any-Precision LLM、GPTQ，Ablation study量化in-register dequantization/pipelining/vectorization/Split-K贡献，GPU utilization profiling (NVIDIA Nsight)分析compute/memory/ALU/Tensor Core/cache utilization。

- 后端平台是什么，配置是什么。
  NVIDIA L40 GPU（主要kernel benchmark平台，面向LLM inference，Compute Capability 8.9）、NVIDIA A100 GPU（对比平台，更高memory bandwidth降低memory-efficient kernel的相对优势）。NVIDIA Nsight用于profiling。

- 评估性能的软件/脚本是什么。修改了什么。
  NVIDIA Nsight分析GPU utilization：compute/memory utilization、ALU/Tensor Core utilization、cache hit rate和throughput。kernel benchmark从LLaMA/OPT linear layers提取真实weight matrix shapes并测试batch size 1-512。Quantix fused kernel通过CUDA实现，输入为预处理的1-bit和2-bit packed weights（经由offline bit shuffling），kernel内部通过PTX-level instructions调用Tensor Core MMA（mma.m16n8k16等）。修改：论文fused kernel集成进HuggingFace Transformers替换SqueezeLLM默认backend；对uniform baselines (GPTQ/Marlin)使用AutoGPTQ library。kernel benchmark 100 warm-up + timed iterations。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/yuang-chen/Quantix-PPoPP26。Quantix fused kernel使用流程：
  1. 离线bit shuffling：对已有non-uniform quantized weights执行bit dividing+bit mapping，生成W1'/W2'和reordered centroids C
  2. Kernel编译：nvcc编译fused dequantization-matmul CUDA kernel为.so
  3. 在线调用：输入W1' (32×1-bit elements/32-bit word)、W2' (32×2-bit elements/64-bit word)、activations A (FP16)和centroids C (FP16 per row)
  4. Kernel执行流程：`cp.async` prefetch→shared memory staging→register load→in-register dequantization→Tensor Core MMA→output Y
  5. L40上3-bit Quantix平均speedup：4.82× over FP16 cuBLAS、3.93× over Any-Precision、46.07× over SqueezeLLM、10.25× over GPTQ
  6. Ablation：移除in-register dequantization性能降至~40%（最显著），禁用pipelining降至~41%，移除vectorization降至~86%，Split-K主要帮助小矩阵增加parallelism
  7. 2-bit/4-bit变体：2-bit Quantix平均5.45× over 16-bit baseline（up to 8.59×），4-bit Quantix因更多centroids和更高memory bandwidth需求通常比3-bit慢但精度更高

## Accelerating Sparse Transformer Inference on GPU (STOF)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Unified MHA Module，包含row-wise和block-wise两种customized GPU kernel实现sparse MHA计算。关键设计：(1) Two-level sparse storage format：外层OuterTile (OT) 为64个8×8 InnerTile (IT)，用BSR (Block Compressed Sparse Row) 表示OT级稀疏——full OTs通过full_row_ptr/full_col_idx数组定位，part OTs通过part_row_ptr/part_col_idx定位+bitmap_mask (64×uint64) 表示IT内64个元素的精确mask pattern；(2) Block-wise kernel (Algorithm 1)：Q按OT_Size_M切分子块Q_i保持在register中，K/V按OT_Size_N切分子块K_Tj/V_j，根据load_row_ptr和load_col_idx仅加载需要计算的valid OTs，跳过无效块；async data copying (__async_memcpy)使V加载与GEMM重叠；对part OTs使用bitmap_mask做细粒度mask；(3) Row-wise kernel：Q按row sliced，warp内使用shuffle操作通信，消除warp间synchronization，适合小seq_len+高稀疏率场景；(4) Advanced optimizations：8×8 IT对齐Tensor Core mma数据粒度、OT行主序存储适配Softmax迭代计算、IT列主序存储消除bank conflict、Q_i register resident避免重复SMEM读写；(5) Kernel selection analytical model：公式1基于valid OT ratio和seq_len计算threshold，低于threshold选row-wise否则block-wise。实验比较MHA computation performance（RTX 4090/A100，causal/sliding window/Longformer/Bigbird，seq_len 128-4096，batch size 1/8/16），对比PyTorch Native、FA2、FlexAttention、ByteTransformer、MCFuser、SPLAT。STOF相对FlexAttention在RTX 4090上平均1.8×、A100上平均1.6×加速。sliding window上（93.8% sparsity）加速最显著，(batch=16, seq_len=4096)时STOF达到FA2的4.8×、FlexAttention的4.9×。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 4090 (Ada Lovelace, 24GB, 128 SM)、NVIDIA A100 (Ampere, 80GB, 108 SM)、NVIDIA H20 (Hopper, preliminary test)。CUDA v12.6, PyTorch 2.7.0。FP16精度。

- 评估性能的软件/脚本是什么。修改了什么。
  NVIDIA Nsight用于profiling。kernel通过CUDA/C++实现（约2,500 LOC），基于FA2的CuTe结构扩展，引入two-level storage format和对应优化。Kernel通过torch/cpp_extension接口封装为PyTorch native function，首次调用时ninja JIT编译为.so动态链接。实验100次warm-up + 100次timed iterations取平均。修改：SPLAT论文未开源，基于论文内容复现。所有方法统一FP16精度评估。MHA实验遵循BERT-Base配置。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：Zenodo artifact DOI: 10.5281/zenodo.17705801。STOF MHA kernel使用流程：
  1. Mask预处理：将任意mask pattern转换为two-level storage format——划分OT/IT网格→生成full_row_ptr/full_col_idx（full OTs的CSR索引）→生成part_row_ptr/part_col_idx（part OTs的CSR索引）→对每个part OT生成64×uint64 bitmap_mask
  2. Kernel Selection：analytical model（公式1）输入valid OT数量（通过load_row_ptr计算）和seq_len，输出threshold，小于0选row-wise kernel否则block-wise kernel
  3. Block-wise kernel执行（以BERT-Base, Bigbird mask, seq_len=4096为例）：
     - Q切分子块Q_i→保持在register→外层循环遍历Q_i
     - 内层循环：load_row_ptr确定当前row的valid OT数量→load_col_idx获取列索引→cp.async加载K_Tj→__async_memcpy加载V_j（与GEMM重叠）→Compute_GEMM(Q_i, K_Tj)得P_ij→检查part OTs→若为part OT则Apply_Mask(P_ij, bitmap_mask)做bitwise mask→Softmax→Compute_GEMM(S_ij, V_j)累加O_i→write back to HBM
  4. Row-wise kernel（小seq_len+高稀疏）：Q row sliced→warp内shuffle同步→集中式mask处理利用row locality
  5. 例如sliding window mask (93.8% sparsity)在A100上(batch=16, seq_len=4096)，block-wise kernel跳过绝大多数OT计算→STOF加速FA2 4.8×

## Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Difflow（原名ChituDiffusion）运行时kernel层优化，实现ragged batching kernel和冗余消除kernel。核心kernel/运行时实现：(1) Ragged Data-Independent Operation Kernels：实现四个ragged data-independent operation kernel (Triton + CUDA)，用于支持ragged batch请求的高效执行。对ragged data-independent操作（无跨请求共享数据，如transpose、reduce），采用每请求独立并行执行策略——基于已有regular operator的tiling plan和computing microkernels，将每个请求划分为tile集合，通过round-robin policy在batched execution时映射到GPU thread blocks；(2) Ragged Data-Sharing Operation Regularization Kernel：对ragged data-sharing操作（有共享权重，如convolution、linear），通过transpose+reshape/transpose+im2col等图变换kernel将ragged输入compact为regular shape→调用标准kernel库执行（如ragged Matmul通过fuse batch dim和ragged dim→reshape→regular Matmul kernel）；(3) Redundancy Memory Access Elimination Kernel：对attention操作中冗余K/V tensors，运行时压缩K/V沿redundant batch dimension、concat Q tensors from different requests into single one (Figure 6)→使用FlashAttention等标准attention kernel执行压缩后的计算；(4) Invariant Tensor Elimination Runtime：lightweight四态(constant/loop-invariant/loop-variant/unknown)检测→compile-time precompute constants→loop-invariants hoisted→multi-value constants selective fixing (trade off performance vs generation diversity)。实验比较：Ablation study在edit应用上隔离各优化的逐项throughput贡献 (ChituDiffusion-base→+SCH 1.29×→+COMP 1.56×→+IRE 1.71×)；图13(b) sequential execution (无batching)下IRE贡献1.3× speedup；raggedness ratio sweep (0%-100%)验证uniform/ragged/mixed dEngine选择效果。

- 后端平台是什么，配置是什么。
  NVIDIA A100 40GB PCIe GPU (CUDA 12.1) + NVIDIA H100 80GB PCIe GPU (CUDA 12.1 UNet / CUDA 12.8 DiT)。开源release PyTorch 2.9。

- 评估性能的软件/脚本是什么。修改了什么。
  基于Triton[59]和CUDA实现四个ragged data-independent operation kernel。性能模型基于OLS regression (R²=0.998)，profiling 16 samples (batch 1-16, shape 256-768)，96-sample evaluation set (R²=0.996, RMSE <3μs)。修改：在Triton/CUDA层新增ragged operation kernels——round-robin tile-to-thread-block mapping用于data-independent ops、transpose+reshape/im2col用于data-sharing ops的regularization。冗余消除kernel通过等价线性代数变换（compress K/V + concat Q）避免新建kernel直接复用FlashAttention。Invariant tensor detection算法从tensor definitions初始化→iterative propagation with priority hierarchy。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/thu-pacman/chitu/tree/Diffusion。kernel使用流程：
  1. Ragged batching kernel：当scheduler决定使用ragged dEngine执行一批不同shape的requests时→ragged data-sharing ops (Matmul/Conv) 通过transpose/reshape/im2col transform转为regular ops→regular Matmul/Conv kernel执行→ragged data-independent ops (transpose/reduce) 通过round-robin tile mapping在GPU thread blocks并行执行
  2. 以ragged Matmul为例：input [b, m̂, k] (m̂ ragged) + weight [k, n]→transpose+reshape fuse b和m̂→[b·m̂, k] regular Matmul→regular kernel
  3. 冗余内存消除：attention中K/V tensors有相同prompt→沿batch dim compress去重→concat所有请求的Q→标准FlashAttention计算→broadcast恢复
  4. Invariant tensor elimination：编译时detection→constant tensor precomputed/loop-invariant hoisted→multi-value constants在运行时selective fixing

Difflow kernel/运行时的作用：通过ragged operation regularization将异构shape请求转化为kernel-compatible形式（无需手写所有ragged kernel），通过冗余消除规则在tensor代数层面等价去除冗余计算和内存访问，使运行时能够高效利用现有优化kernel库同时支持数据属性感知的优化。

## MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  MixFusion的kernel/运行时核心是Patch Edge Stitcher和Compressed Sparse Patch (CSP)格式。关键设计：(1) Patch Edge Stitcher：将跨patch边界stitching操作fuse进GroupNorm kernel，消除额外memory movement。具体实现：每个GPU thread block (TB) normalizes一个patch的同时检查其boundary pixels是否被邻接patch需要→需要的boundary pixels暂存于shared memory→所有normalization完成后TB定位目标patch并将boundary写回global memory。此设计overlap了edge stitching与normalization，无需额外synchronization；(2) Fused GroupNorm+Stitcher kernel：TB内通过shared memory暂存boundary data实现column方向的irregular memory access局部化（column stitch按行方向内存布局时访问不连续→通过shared memory中转化解），row boundaries直接按内存layout对齐高效读写；(3) CSP格式：受CSR格式启发，通过resolution reorder→offset-based compression将patch mapping压缩为RequestOffset[]和RequestStart[]/RequestEnd[]三数组，O(1)定位任意patch所属请求及其邻接关系；(4) Batched Cache Operations：将patch级cache的query/delete/update/insert操作coalesce为batch——输入patch indices与cache entries比对→分出Common Set（需verification）/New Set（insert）/Expired Set（delete）→三集合并行处理。实验对比：(a) Patch Edge Stitcher vs. naive stitching latency overhead（Figure 5, stitcher overhead minimal vs. naive offsetting parallelism gains）；(b) PSNR/SSIM quality vs. Distrifusion across patch sizes（Table 4, Patch Size=512用4 Patches PSNR 28.82/SSIM 0.88）；(c) patched vs. whole-image caching latency savings（Figure 20, patch-level consistently outpeforms full image）；(d) cache overhead vs. batch size（Figure 17, cache overhead scales modestly）。

- 后端平台是什么，配置是什么。
  NVIDIA H100-80GB GPU (CUDA 12.3, PyTorch 2.2.2)，xformers用于加速attention算子。

- 评估性能的软件/脚本是什么。修改了什么。
  基于PyTorch + custom C++/CUDA实现（12.5K行总代码）。新增kernel/运行时：(1) Patch Edge Stitcher CUDA kernel：fused GroupNorm + boundary stitching，TB-level shared memory通信实现irregular memory access局部化；(2) CSP格式运行时：python-side resolution reorder + offset计算 + C++ tensor indexing；(3) CuML Random Forest用于GPU端cache prediction（避免CPU-GPU数据搬移开销）；(4) xformers集成用于加速Self-Attention中的batched attention。实验中对patch splitting overhead和cache management overhead分别ablation测量（Figure 17），patch size对throughput影响的sensitivity analysis（Figure 18, patch size 64/128/256）。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/desenSunUBW/mixfusion。kernel使用流程：
  1. Patch Edge Stitcher kernel：当patched convolution执行时，每个TB处理一个patch的GroupNorm→TB检查自身boundary pixels是否需要被邻接patch使用（依赖信息在patch splitting时记录）→需要的boundary存入TB的shared memory→完成所有normalization→TB根据metadata定位需要其boundary的目标patch→将boundary从shared memory写回global memory对应位置→目标patch的TB可直接读取准确边界值
  2. CSP查找：给定patch ID→二分查找RequestOffset[]确定所属请求→RequestStart[patch_id]和RequestEnd[patch_id]获取该patch在所属请求内的起止位置→ResolutionOffset[request_index]提供Self-Attention reconstruction时的分辨率偏移
  3. Batched Cache：block收到patch indices和intermediate results→cache system以map结构（patch unique ID为key）比对→Common Set中patch比较cached vs. new result决定update→New Set合并为batch insert→Expired Set（cache中存在但input indices无）的patch已退出→batch delete

MixFusion kernel/运行时的作用：通过fused Patch Edge Stitcher消除跨patch计算中的冗余memory movement（边stitching边normalization，利用shared memory localize irregular column access），通过CSP格式实现O(1) patch定位和高效批量cache操作，确保patch-level parallelism的计算收益不被management overhead侵蚀。

## ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出ASM-SpMM，一个面向ARM SME的高性能SpMM库。核心kernel/runtime设计：(1) OP-MCF (Outer-Product-friendly Masked Column-merging Format)：按SME vector length划分row window，window内做column compaction将非零位置不重叠的列合并为一个compressed slot，记录RowWindowOffset/ColumnOfRowWindow/SparseAtoB/ColumnPositionMaskBit四类数组，消除Tensor Core格式的硬性block padding；(2) SME outer-product SpMM microkernel：对每个compressed slot加载sparse vector和dense matrix B tile，用predicate mask控制有效非零位置，通过svmopa类outer-product指令累加到ZA tile，把稀疏值向量和dense tile外积写入ZA而非当作普通SIMD；(3) 多ZA tile并发与显式prefetch pipeline：多个independent outer products映射到不同ZA tile/slice，剩余Z register做operand streaming，_svprfw类prefetch指令显式预取sparse/dense operand；(4) SVE/Neon混合matrix-vector kernel：对低密度或碎片化block，将稀疏尾部交给vector unit处理，通过interleaved instruction scheduling与SME path重叠，要求vector工作量能隐藏在SME固定执行窗口内；(5) hetero-core动态work stealing调度：runtime先做hardware-aware task mapping，再用progress monitoring和work stealing动态再平衡。实验比较：(a) operator-level baseline对比ArmPL v24.10、Armadillo v14.6.0、SuiteSparse Cholmod v5.3.3、Eigen v3.4.0、MP-SpMM；(b) 单核ablation对比naive SME、multi-tile、format、prefetch pipeline、vector co-execution逐项贡献；(c) GNN case study将ASM-SpMM接入PyTorch CPU Extension实现ASM-GCN/ASM-GIN，对比PyG v2.6.1、DGL v1.1.2。M4上12个代表矩阵相对Cholmod取得3.5x-7.9x speedup；SuiteSparse分布实验相对ArmPL/Armadillo/Eigen/Cholmod/MP-SpMM的geomean speedup在LX2上为9.69/16.43/19.53/4.32/2.62，M4上为11.81/15.12/18.62/4.78/2.94。

- 后端平台是什么，配置是什么。
  Apple M4 CPU（最多10核：4 P-core + 6 E-core，2个SME compute unit分别服务P-core cluster和E-core cluster，512-bit vector length，一次处理8个double）；LX2 ARM processor（最多12核，所有core配备SME unit，512-bit vector length）。编译器Clang 16.0，使用ARM SME/SVE/Neon intrinsics实现。

- 评估性能的软件/脚本是什么。修改了什么。
  Benchmark脚本执行SpMM operator-level latency和GFLOPS测量。稀疏矩阵来自SuiteSparse（按规模/形状/稀疏度分层抽样80个矩阵）和12个代表性真实图/稀疏矩阵（含TC-GNN/SNAP/OGB/DGL图矩阵）。dense matrix B列数评估512和1024。ASM-SpMM kernel使用ARM SME intrinsics实现：svld1_f32加载sparse data、svmopa_za32_f32_m outer product accumulate、svst1_hor_za32写回ZA到output、_svprfw类prefetch指令显式预取。GNN case study中ASM-SpMM通过PyTorch CPU Extension接入，实现ASM-GCN和ASM-GIN的custom SpMM operator。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未找到明确开源仓库（正文无GitHub/artifact URL，PPoPP 2026未确认官方实现发布）。kernel使用流程（基于论文描述）：
  1. OP-MCF格式转换：输入sparse matrix A→按SME vector length划分row window→每个window内删除空列→非零位置不重叠的列合并为compressed slot→生成RowWindowOffset/ColumnOfRowWindow/SparseAtoB/ColumnPositionMaskBit四数组
  2. SME SpMM kernel执行：清空ZA tile→遍历row window的compressed slots→根据SparseAtoB找到B的对应列→svld1_f32加载compressed sparse values到Z register→ColumnPositionMaskBit转predicate register→vectorized load B的dense tile→svmopa_za32_f32_m执行outer product accumulate（sparse vector × dense tile→ZA tile/slice）→循环期间_svprfw预取下一slot的sparse data/mask/column index/dense B fragments→所有slots完成后svst1_hor_za32写回output matrix C
  3. hybrid路径：对低密度block分配SVE/Neon vector path→interleaved instruction scheduling隐藏vector工作量于SME执行窗口内→结果与SME累加路径合并
  4. 多核调度：每core初始获得若干row window→core提前完成时根据滑动窗口剩余非零元和进度信息→从负载较重core窃取row window→全局任务完成

## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出RoMeo的cross-precision混合精度GEMM kernel系统和辅助kernel。核心kernel/runtime设计：(1) Permutation-free Mixed Precision Computation：预分配dedicated outlier buffer（大小基于预定的outlier token数量），整个activation矩阵量化为INT4，outlier token embedding复制到outlier buffer量化为INT8。所有四个cross-precision矩阵乘法（W4A4/W4A8/W8A4/W8A8）各自操作dense uniform-precision矩阵，每个GPU thread block处理一种精度组合，完全避免permutation overhead。outlier token同时参与INT4和INT8两次计算（tolerate redundant computation以保证contiguous memory layout符合Tensor Core指令要求），高精度结果最终overwrite对应输出位置。(2) Separate-Kernels with Software Pipelining：采用separate-kernels而非fused-kernel实现，原因为不同精度组合的shared memory需求不同（INT8-INT8 kernel需2× shared memory vs INT4-INT4），separate允许compiler对每种kernel独立分配on-chip资源。Kernel内部使用software pipeline（Algorithm 2）：通过cp.async PTX指令异步加载global→shared memory（pipeline fill阶段），steady state每iteration等待oldest copy完成→Tensor Core mma计算→发射新async copy，最终drain阶段完成所有remaining mma。对于INT4→INT8的cross-precision计算，在shared memory内使用两个binary arithmetic指令做类型转换（而非昂贵的type conversion指令）。(3) Fused Triton Kernels：开发fused Triton kernel用于在线outlier identification（per-token row-max + top-k selection）、量化（round + scaling）和INT4 data packing，减少kernel launch次数和内存往返。(4) 在线动态Outlier Detection：因token-wise outlier来自输入语言特征，需要运行时在线检测；kernel执行在线row-max reduction然后top-k selection确定outlier set，而非离线静态分析。实验比较kernel-level speedup (normalized to BF16)，在Qwen3-8B/14B/32B和Llama-3.1-70B的QKV/O/UG/D四种linear layer matrix shape上，对比INT8 kernel、Atom group-wise INT4 kernel、QuaRot INT4 kernel。RoMeo geomean speedup 4.68× over BF16，与QuaRot 4.55×相当但额外计算5%高精度outlier。消融实验展示U-ker→U-ker+Pipe→S-ker→S-ker+Pipe→S-ker+Pipe+Async五组配置在batch=16/64下的layer-level latency breakdown。

- 后端平台是什么，配置是什么。
  NVIDIA GeForce RTX 4090 GPU (Ada Lovelace, 24GB memory, Compute Capability 8.9, peak INT4 Tensor Core throughput 8× over FP16)。Python 3.12, PyTorch 2.8.0, CUDA 12.8。Kernel编译：CUTLASS (cross-precision CUDA kernels), Triton (fused outlier detection/quantization/packing kernels), HadaCore (FWT Hadamard变换)。JIT编译机制：首次执行时编译对应模型维度的kernel并缓存compiled binary后续复用，auto-tune tiling size和pipeline stage数量。

- 评估性能的软件/脚本是什么。修改了什么。
  自研cross-precision CUDA kernels（基于CUTLASS）+ fused Triton kernels。Baseline kernels包括：BF16 (PyTorch half-precision matmul)、INT8 (CUTLASS INT8 matmul kernel)、Atom (group-wise INT4 mixed precision kernel)、QuaRot (INT4 matmul with fused dequantization kernel)。所有kernel使用CUDA Graph捕获消除launch overhead后，CUDA events测量平均latency。Kernel benchmark覆盖Qwen3和Llama-3.1模型实际weight tensor shapes（QKV_proj, O_proj, UpGate_proj, Down_proj），M dimension固定为4096，batch size可变。NVIDIA Nsight Compute用于profiling各kernel的shared memory/register使用和occupancy。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/thu-pacman/RoMeo。Kernel使用流程（以Qwen3-8B Down_proj layer [4096×4096] GEMM为例）：
  1. JIT编译阶段：RoMeo首次加载模型→根据weight shape (4096,4096)和outlier budget (5%, ~205 outlier tokens)→auto-tune确定tiling size (TM/TN)和pipeline stages (Nstage)→编译四个separate CUTLASS cross-precision kernels + Triton outlier detection/quantization/packing kernels→缓存compiled binaries
  2. 在线执行阶段（batch=64, seq_len=128→M=8192 tokens）：
     a. Triton outlier detection kernel：per-token row-max reduction→top-k outlier selection→生成outlier index mask (8192 entries, ~410 are outliers)
     b. Triton quantization kernels：normal tokens→INT4 quantize + pack（W4A4 kernel input）；outlier tokens→copy to outlier buffer→INT8 quantize（W8A4 kernel input）；weight offline已per-column mixed precision quantized
     c. Asynchronous concurrent GEMM：四个CUDA streams各自launch W4A4/W4A8/W8A4/W8A8 kernel→每个kernel内部cp.async pipeline异步加载global A/B tile到shared memory→INT4→INT8 casting in shared memory→mma指令计算→结果scale by per-token scaling factor→写回global memory
     d. Post-mul overwrite kernel：高精度outlier结果overwrite W4A4结果的对应位置→完成所有精度组合的结果合并
  3. 性能：4096×4096×4096 GEMM RoMeo kernel ~0.56ms，BF16 baseline ~2.62ms (4.68× speedup)。Pipeline+Async优化后batch=16 layer latency从6.73ms降至3.39ms (2.0× over BF16)

## AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  AUM在AU (Accelerator Unit) 选择和core分区层面进行kernel级调度：(1) **AU Selection**：根据ARI = 6(1/d + 3/BL)^(-1) (prefill) 和 6(1/d + 3/B)^(-1) (decode) 判定每个operator的最优AU——高ARI operator (prefill GEMM, dim=8192×4096×22016) 使用AMX TMUL达到40.57 TFLOPS，低ARI operator (decode GEMV, dim=16×4096×22016) 使用AVX达到更高效（仅3.87 TFLOPS via AMX），避免小矩阵AMX overhead（tile register配置开销>计算收益）；(2) **Frequency Region Division**：按AU使用率将物理核划分为High-AU region (2.1-2.5 GHz, 运行prefill)、Low-AU region (2.8-3.1 GHz, 运行decode)、None-AU region (3.2 GHz, 运行shared非AU应用)，利用AU功耗→频率反比关系避免decode phase被compute-intensive shared app频率拖累；(3) **Resource Affinity Profiling**：监控各AU operator的μarch资源bound——prefill为backend bound (92% backend), decode为memory bound (DRAM 59.9%)，指导kernel运行时资源分配。实验比较perf-per-watt效率、AU operator SLO guarantee、各频率区域划分下的AU perf degradation。

- 后端平台是什么，配置是什么。
  三台Intel Xeon AU-enabled CPU：GenA (SPR Xeon 8475B, 48核×2, AMX BF16 206.4 TFLOPS, DDR5 233.8 GB/s)；GenB (SPR Xeon Max 9468, 48核×2, AMX BF16 206.4 TFLOPS, HBM 588 GB/s)；GenC (GNR Xeon 6982P-C, 120核×1, AMX BF16+FP16 344 TFLOPS, MCR 600 GB/s)。AU单元详情：每物理核AMX单元含8×1KB TILECFG寄存器和TMUL加速器（1024 BF16 ops/cycle），m≤16, n≤64的矩阵乘。

- 评估性能的软件/脚本是什么。修改了什么。
  软件栈：xFasterTransformer (LLM serving框架) + Intel oneDNN (底层AMX算子)。表征工具：Linux perf (tma_amx_busy, tma_fp_amx, avx_insts等metrics)、pmu-tools (top-down分析)、turbostat (core频率记录)、pqos + Intel RDT CAT/MBA (LLC和内存带宽分区)。AUM Background Profiler收集operator-level AU behavior：AMX cycle ratio (prefill 14.4% vs decode 1.5%)、AMX μop ratio (prefill 3.7% vs decode 0.5%)、per-region频率下限(f_H/f_L/f_N)、per-region minimal资源需求(R_L2C, R_LLC, R_BW)。修改：通过ARI阈值判定AMX vs AVX选择→不同region设置不同U_AU threshold。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未明确说明开源。软件使用流程：
  1. 表征阶段：在GenA上运行xft llama2-7b prefill/decode→perf stat记录per-core AMX cycle ratio→turbostat记录per-core频率→绘制Figure 6a频率vs AU core count曲线
  2. Top-down分析：pmu-tools toplev采集prefill/GEMM/decode的cycle分布→Figure 7对比AU vs非AU应用的frontend/backend bound→发现AU frontend bound显著低(5%→1%)、decoder backend DRAM bound达59.9%
  3. AUV Model profiling：对High/Low/None三个频率区域×5种敏感资源配置各10次重复执行→记录P_a (50%-ile perf)和P_t (90%-ile tail perf)→例：High U_AU 0-11核, F=2.1 GHz, R_LLC=0-2 way, R_BW=50%
  4. Runtime控制：Controller查AUV Model→按Algorithm 1决策：if P^m_H < SLO_H and P^m_L < SLO_L → aggressive harvest (δ = U_AU × SLO/P^m) → 用P_a profile分配min resource给AU、剩余给shared

## GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出GyRot PE（Processing Element）微架构，实现fully integer-based dequantization datapath，支持W4A4 group quantization (G=32) + asymmetric quantization的高效运行时计算。核心PE设计：(1) 32-way INT4 dot product：每cycle执行32个4-bit activation (X0~31)与32个4-bit weight (W0~31)的点积，输出13-bit partial sum；(2) 重公式化integer dequantization pipeline：先乘activation scale SX (INT8)→加zero-point项 ZX×WSUM（WSUM=Σ_{i∈g} ŵ_i预计算）→乘weight scale SW (INT8)→全整数域完成dequantization，避免传统FP dequantization的type conversion和浮点开销；(3) 32-bit integer accumulator：全整数累加inter-group partial results→最终转FP16写output buffer；(4) FVU (Fused Vector Unit) 中FHT (Fast Hadamard Transform) unit：5-stage 32-way pipeline，含160 add/subtract units (32/stage)，支持O(n log₂ n)在线Hadamard rotation（当非线性层如SwiGLU/embedding介入时rotation无法fuse进weight），通过local register file + two-stage scheme支持scalable rotation up to 32×32=1024维，partial gating支持sub-32 power-of-two sizes (2/4/8/16/32)。实验比较PE级area和energy：GyRot-FP vs GyRot-INT vs Tender/MANT/LightRot PE在iso-throughput下28nm synthesis对比。GyRot-INT PE achieves 65.2% area和69.2% energy reduction over Tender。

- 后端平台是什么，配置是什么。
  GyRot custom accelerator：Samsung 28nm工艺，Synopsys Design Compiler综合，目标频率1GHz。PE array: 8×8×32 tensor organization (2048 parallel ops/cycle)。片上SRAM由commercial memory compiler生成。对比baseline（Tender/MANT/LightRot）均在相同28nm工艺、1GHz、iso-compute-area约束下综合评估。DRAM功耗由Micron DRAM Power Calculator (DDR4 model) 估算。

- 评估性能的软件/脚本是什么。修改了什么。
  RTL仿真验证PE功能正确性。Synopsys Design Compiler综合→area/power报告。对比分析：Tender PE（8-bit datapath，无group quantization）、MANT PE（G=64, FP16 SF, flexible data format）、LightRot PE（G=128, FP16 SF+FP16 ZP, floating-point dequantization）。GyRot的关键修改：(1) 将传统"先GEMM→后FP dequantize"的two-phase流程改为PE内部fused integer dequantization pipeline；(2) 预计算WSUM并broadcast到整行PE，消除per-PE重复计算；(3) FHT unit用add/subtract替代乘法实现Hadamard rotation，降低rotation硬件开销。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未明确说明开源。PE计算流程（以G=32 W4A4为例）：
  1. 每个PE cycle：从input buffer读取32个INT4 activation X0~31→从weight buffer读取32个INT4 weight W0~31→32-way dot product → 13-bit partial sum
  2. Dequantization stage：partial sum × SX (INT8)→ + ZX×WSUM (WSUM预计算并broadcast到整行)→ × SW (INT8)→ dequantized result
  3. 32-bit integer accumulator：累加intra-group partial results→group边界处转FP16写output buffer
  4. FHT unit（需online rotation时）：load activation→5-stage add/subtract pipeline (每stage 32 parallel units)→完成Hadamard rotation→输出到quantization unit→进入PE array
  对比baseline：Tender无group quantization/dequantization overhead；MANT需FP16 SF乘法（FP multiplier per PE）；LightRot需FP16 SF+FP16 ZP运算（更重的FP dequantization）。GyRot-INT以INT8 SF/ZP全整数pipeline实现最低PE area和energy。

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

## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出BitDecoding，一个利用Tensor Cores加速低比特KV Cache解码的GPU kernel系统。核心实现：
  (1) Residual Kernel：fuse computation、quantization、packing进单kernel——用ldmatrix将FP16 KV tensor加载到register（Tensor Cores interleaved layout）→执行矩阵运算（QK^T或PV）→每个thread在register内量化和pack→低比特packed data直接写global memory更新low-bit KV cache。利用ldmatrix的thread-to-register mapping自动induce与Tensor Cores兼容的packed layout，无需global reshape。
  (2) Packing Kernel：fuse dequantization与Tensor Cores计算——mirror Residual Kernel的ldmatrix/mma variant配置→ldmatrix加载packed low-bit data→lop3-based layout remapping (75316420 pattern)→INT4/INT2→FP16高效转换→直接参与mma。硬件指令配置（ldmatrix + mma variant）由GPU架构自动确定，residual block size Nr=Pn×Wn×R根据量化bit-width自动计算。
  (3) Warp-level parallelization：新颖warp layout——Wm=1（decode query length<16）→Wn增大→多个warps并行做dequantization→SM warp scheduler overlap dequantization与Tensor Cores mma。Cooperative softmax：利用shared memory buffer (sTMP + sAcc) 做cross-warp reduction和P矩阵重载，仅0.5% overhead。
  (4) 异步pipeline：register-level软件定义pipeline——ldmatrix加载+Dequant (CUDA Cores) 与mma (Tensor Cores) 异步重叠：第i个slice在Tensor Cores上做mma时，第(i+1)个slice同时从shared memory加载并dequantize。
  (5) 架构专用优化：Hopper上使用STSM PTX指令将dequantized FP16写入shared memory→支持wgmma_SS（B矩阵必须在shared memory）。Blackwell上绕过lop3 remapping，直接使用原生mxfp4/nvfp4 mma指令执行packed 4-bit GEMM。

  实验比较：kernel-level在Blackwell（RTX 5090, RTX PRO 6000）、Hopper（H100）、Ada（RTX 4090）、Ampere（A100）上评估，三种workload setting：Single (bs=1)、Batches (大batch)、Page (page management)。对比baselines：FlashDecoding-v2 (FP16)、Kivi（非fused low-bit kernel）、Atom和QServe（fused CUDA Cores-only kernel）。量化配置：4-bit/2-bit Key, tensor-wise (KT) + channel-wise (KC) scaling。

- 后端平台是什么，配置是什么。
  NVIDIA Blackwell: RTX 5090, RTX PRO 6000（原生MXFP4/NVFP4低精度格式支持，up to 20 PFLOPS）。NVIDIA Hopper: H100（WGMMA指令、warp-specialized pipeline、TMA异步数据加载、STSM指令）。NVIDIA Ada: RTX 4090（带宽受限，DRAM瓶颈显著）。NVIDIA Ampere: A100（高带宽，compute-bound场景）。多GPU: 8×A100 for LLaMA-3.1-70B。

- 评估性能的软件/脚本是什么。修改了什么。
  自研CUDA kernel实现BitDecoding，基于FlashAttention/FlashDecoding架构扩展。修改：(1) 新增Residual Kernel（fuse FP16 computation+quantization+packing）；(2) 新增Packing Kernel（fuse dequantization+Tensor Cores mma with async pipeline）；(3) 修改warp partitioning策略（Wm=1, 增大Wn）；(4) 新增cooperative softmax（shared memory cross-warp reduction）；(5) Hopper版本基于FA-3，用wgmma和tma.copy；Blackwell版本直接用原生mxfp4 mma。使用Nsight Compute profiling分析Tensor Cores utilization、memory throughput、stall cycles。baselines包括FlashDecoding v2/v3（开源，CUDA实现）、Kivi（Triton实现，非fused）、QServe和Atom（CUDA Cores-only fused实现）。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源: https://github.com/OpenBitSys/BitDecoding。BitDecoding kernel执行流程（以GQA、4-bit channel-wise量化、seq_len=128K、Hopper H100为例）：
  1. Query Transformation：Q tensor从[1, (gq, hkv)] reshape为[gq, hkv]，形成更大Q tile完全填充Tensor Cores fragment。
  2. Prefill阶段：Residual Kernel——用TMA/ldmatrix加载FP16 K/V→执行QK^T和PV mma (Tensor Cores)→每thread在register内用__shfl_xor_sync warp-reduce计算scale/zero→in-register INT4量化+pack→packed data写global memory更新low-bit KV cache。前Np = L - (L mod Nr) entries存packed low-bit cache，剩余res_len = L mod Nr存FP16 residual cache。
  3. Decode阶段（逐token）：
     a. 新生成K/V FP16 tensor追加到residual cache→residual cache≤Nr（通常<256）
     b. Packing Kernel: cp.async异步加载Q (global→shared mem)、加载Kpack/Vpack low-bit data和Kp/Vp scale/zero metadata→ldmatrix加载packed data到register→lop3-based 75316420 remapping→INT4→FP16 dequant (CUDA Cores)→异步overlap：slice i做mma (Tensor Cores)同时slice i+1做ldmatrix+dequant→softmax (cooperative cross-warp)→sAcc重载P via ldmatrix→PV mma→output writeback
     c. 每当residual cache达Nr时触发Residual Kernel将其量化写入packed cache
  4. Hopper优化：用STSM将dequantized FP16写入shared memory→wgmma_SS直接访问shared memory (B矩阵)。Blackwell：跳过dequant直接用mxfp4 mma。性能：vs FP16 FlashDecoding-v2: Blackwell 8.6×, Hopper 8.0×, Ada 7.5×, Ampere 4.8× speedup。

## AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出PQ-based attention kernel，在HBM-PIM上实现从GEMV到lookup+summation的运行时计算调度。核心kernel调度设计：(1) PQ-based attention kernel：将传统qK^T GEMV操作分解为query subvector splitting→BankPE query×codebook ATNK (inner product matrix计算)→BufferPE index lookup+softmax (SFM)→BankPE ATNV (attention reconstruction)，避免explicit dequantization，仅使用现有FP16 MAC units；(2) BankPE/BufferPE双PE runtime调度：BankPE（近bank, 高带宽, 面积受限）执行non-data-intensive操作 DC/ATNK/ATNV (ADD/MUL/SUM units)，BufferPE（buffer die, 低带宽, 面积充裕）执行data-intensive操作 CA/SFM (MIN/DIV/EXP units)，减少跨bank数据传输；(3) Intra-row indirection kernel：通过GRF中存储的lookup indices重定向到column decoder，直接从row buffer stream相应inner product values到BufferPE或GRF，单次row activation完成所有lookup；(4) Page-aware windowed clustering的kernel优化：限制每个window内512 centroids=512 inner product values，保证完整fit在1KB HBM row buffer (FP16)，每个window仅1次DRAM row activation即可完成所有indirect lookup；(5) Sequence-by-sequence pipelining：GPU生成每sequence的qkv并立即offload给PIM后继续处理下一sequence，隐藏GPU-PIM sequential processing的idling；(6) Head-wise + subvector-wise data mapping：每attention head映射到独立HBM stack（无跨HBM传输），每subvector映射到独立bank（无跨bank传输）。实验比较decoding per-step latency（图12），对比GPU baseline/AttAcc!/PQCache/SKVQ/SnapKV在4个sequence lengths (4096/8192/16384/32768)下的性能。AQPIM在S_len=32768达到最高8.33× speedup vs GPU baseline。

- 后端平台是什么，配置是什么。
  HBM-PIM架构：基于HBM3集成的PIM，3D-stacked DRAM dies通过TSV互联。BankPE位于DRAM bank旁（1KB row buffer per bank，高内部带宽，面积受限），BufferPE位于HBM buffer die（低带宽但面积充裕）。H100 GPU作为host processor。配置：1×H100 GPU core + 5×16GB HBM（GPU+HBMs），PIM系统将4×16GB HBM替换为4×16GB HBM-PIM用于KV cache存储，剩余HBM存储model parameters。

- 评估性能的软件/脚本是什么。修改了什么。
  构建customized GPU-PIM simulator，基于AttAcc! simulator ([55] https://github.com/scale-attacc-kr/attacc-sim) 和Ramulator2 ([20] https://github.com/CMU-SAFARI/ramulator2) 修改。输入：system configuration, model details, input configurations；输出：execution time和energy consumption。Timing和energy consumption基于AttAcc!的synthesis results。Added intra-row indirection logic area: 0.0565mm² per HBM (仅0.43% of BankPE area of HBM3 implementing AttAcc!)。PIM commands (PIM_SET_CONFIG/PIM_MAC_AB/PIM_SFM/PIM_RET/PIM_MV_BA/PIM_MV_BF/PIM_RD/PIM_WR/PIM_ACT_AB)通过标准HBM command path发出。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  AQPIM代码未直接开源。构建的simulator基于开源Ramulator2和AttAcc! simulator。使用例子：simulator接受system configuration (H100 GPU specs, HBM-PIM bank组织, BankPE/BufferPE microarchitecture参数)、model details (Mistral-7B layer数/head数/hidden dim/FFN dim)、input configurations (batch_size=16, input_length 4096-32768, output_length 128-1024)，输出timing breakdown (d_pq/d_attn/d_fc/d_comm/d_etc)和energy breakdown per decoding step。PE area/energy no new synthesis, reuse AttAcc! results; intra-row indirection synthesized with same PDK [9] as AttAcc! scaled to DRAM die density。

## Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Streaming Multilevel Concentration的运行时计算调度，将稀疏压缩嵌入systolic-array accelerator的GEMM tile执行流中。核心调度设计：(1) SEC Streaming top-k overlapping：SEC的a-way streaming bubble sorter执行M·a·k cycles，与image attention GEMM (M·(M+T)·h·n/(a·b) cycles)完全重叠。在典型配置下(h·n≈3584, b=32, k<M+T)，sorting操作远在Q(i)K^T完成前结束，SEC不在critical path上。(2) SIC on-chip tile-local compression：SIC不等整层或全序列token就绪，在每个GEMM m×n tile产生后立即on-chip压缩（m=1024, n=32, a=n=32），每个tile最多8×m cycles做similarity matching（7 pairwise comparisons + 1 L2-norm per vector）vs GEMM需要K/b×m=112×m cycles (K=3584, b=32)，matching远不在critical path。仅当K<256时matcher接近critical path，此时可scale多matcher并行。(3) Convolution-style layouter conflict-free scheduling：按Bank=f%2×4+r%2×2+c%2映射使2×2×2 block的8 vectors分散到8个不同SRAM bank，支持无复制无bank conflict的全并行读取。Offset=⌊r/2⌋×⌈W/2⌉+⌊c/2⌋计算地址。(4) Scatter-Gather循环：Similarity Scatter用2a-wide accumulator (64)做concurrent accumulation，根据similarity map将compact vectors的partial sums复制/distribute回原始token indices在output-stationary buffer中累加；完成所有⌈K/k⌉ outer loop iterations后Similarity Gather做一次性tile-level再次压缩。(5) SEC-to-SIC handoff：SEC的offset encoding随GEMM output stream传输给SIC，使SIC的convolution-style layouter可恢复prune后token的(Frame,Height,Width)坐标。实验比较：performance speedup和energy，对比vanilla SA、AdapTiV、CMC、GPU (A100)、GPU+FrameFusion。Focus实现4.47× speedup vs SA、7.90× vs GPU、2.37× vs GPU+FrameFusion。DRAM traffic分析：CMC虽有46% sparsity但仍有79% dense DRAM traffic，Focus达81% sparsity且仅需21% bandwidth。

- 后端平台是什么，配置是什么。
  Focus accelerator：32×32 PE array (FP16 multiply/FP32 accumulate, weight stationary dataflow)。On-chip buffer 734KB (128KB input + 78KB weight + 512KB output + 16KB layouter buffer for 256-vector window)。Off-chip memory: DDR4 4Gb×16, 2133R, 4 channels, 64GB/s。Target clock 1.32ns (≈757 MHz)，500MHz place-and-route目标下34% timing margin。TSMC N28HPC+工艺。GPU对照：NVIDIA A100 80GB (FP16), Jetson Orin Nano GPU。

- 评估性能的软件/脚本是什么。修改了什么。
  基于SCALEsim-v2构建cycle-accurate simulation framework。输入：PyTorch实现生成的layer-wise sparse traces（记录每GEMM tile的active/inactive token indices、similarity map、concentrated vector count）。修改SCALEsim-v2：添加Focus-specific runtime scheduling建模（SEC top-k sorter与attention GEMM的重叠调度、SIC similarity gather/scatter per tile的pipeline staging、convolution-style layouter bank conflict检测、scatter accumulator的2a-wide concurrent operation建模）。DRAM energy使用DRAMsim3建模。RTL (SystemVerilog) 用Synopsys Design Compiler综合。所有baseline accelerator的核心逻辑同样用SystemVerilog实现以公平评估面积和能耗。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/dubcyfor3/Focus（MIT License, algorithm/simulator/rtl/evaluation_scripts）。Zenodo DOI: https://doi.org/10.5281/zenodo.17851346。评估使用流程：
  1. 算法trace生成：在A100 GPU上运行PyTorch Focus实现，对LLaVA-Video-7B/LLaVA-OneVision-7B/MiniCPM-V-2.6在VideoMME/MVB/MLVU上推理，record per-layer per-tile的sparse traces（SEC token pruning decisions + SIC similarity map + concentrated vector indices）
  2. Simulator运行：`python simulator/run.py --config focus_arch.yaml --traces <trace_dir>` → 模拟32×32 PE array上GEMM tiling + SEC streaming top-k + SIC scatter/gather per tile → 输出cycles breakdown (GEMM/SEC/SIC/idle)、DRAM read/write bytes、on-chip buffer utilization
  3. RTL评估：`cd rtl/ && make synth` → Synopsys DC综合 → area/power report；Memory Compiler生成SRAM macros
  4. 对比baseline：同样配置下simulate vanilla SA/AdapTiV/CMC → compare speedup/energy/DRAM traffic
  5. 环境：Ubuntu 22.04, Python 3.11, PyTorch 2.6.0, CUDA, Transformers 4.48.2/4.49.0, FlashAttention 2.7.4.post1, A100 80GB

## RPU - A Reasoning Processing Unit

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Reasoning Core的decoupled memory-compute-network pipeline微架构和stripe-based VMM kernel执行，核心实现：(1) Stripe-based VMM kernel：将VMM O=V×W组织为stripe执行——一个stripe=8个垂直堆叠tile跨所有weight shard列。Activation shard（64 BF16 values）从network buffer加载到TMAC register file，在tile columns间复用后再retire。TMAC处理tile rows时先column-first遍历（内积风格需全activation on-chip，外积风格写回带宽高），处理完一列tile后用3-stage tree sum做列累积reduction，结果写回local register file供下一stripe复用。处理完当前stripe后network buffer已尽量收集下一stripe所需activation，重叠通信与计算。(2) Weight-streaming, output-stationary TMAC dataflow：8×8 TMAC阵列支持BF16 multiply+FP32 accumulate，activation broadcast跨8列，weight元素逐tile stream-in。Stream Decoder做on-the-fly dequantization（BFP/MxFP/NxFP 4-8 bit→BF16），通过1024-bit compute bus广播给TMAC。dequantization消除off-chip full-precision weight storage，压缩weight tile在memory中存储、on-chip解码后再计算。(3) Decoupled three-pipeline architecture：Memory DMA（HBM-CO ↔ memory buffer）、Compute DMA（buffer → Stream Decoder → TMAC/HP-VOPs）、Network DMA（inter-core activation forwarding/reduction），通过Pipeline Arbiter在SRAM buffer entry粒度用2-bit valid counter做数据驱动同步。Memory pipeline可在compute/network stall时继续预取weights/KV cache到on-chip buffer（每CU ~MB级），使compute后续"catch-up"消耗已预取数据。(4) NUMA at all scales：每core独立NUMA domain无共享内存，所有跨domain数据移动由software-programmable DMA显式管理，消除coherence overhead。(5) Layer smoothing via phase-imbalance absorption：BS=32时compute-bound weight layers和memory-bound KV cache layers自然交替，decoupled pipeline让memory在compute处理wUp/wGate时提前预取后续KV$和weights到buffer，吸收phase imbalance，无buffering时overall latency增加up to 1.6×。实验比较：(1) BS=1 vs BS=32 pipeline utilization trace（Fig.8），展示memory/compute/network utilization、buffer occupancy和power breakdown；(2) 强扩展latency vs CU count（Llama3-8B/70B/405B, Llama4-Maverick）；(3) Batch scaling下BW utilization (Fig.11 bottom right)；(4) Ablation：decoupling contributions (HBM-CO + aligned provisioning + decoupled pipelines) cumulative analysis。

- 后端平台是什么，配置是什么。
  RPU custom hardware（非GPU/CPU）。每Reasoning Core: 4×8×8 TMAC (1 TFLOP BF16/FP32)、1.0 MB on-chip SRAM、32 GB/s HBM-CO memory bandwidth、16 GB/s network bandwidth、0.25W。每Compute Unit: 16 cores (16 TFLOPs)、16 MB on-chip memory、512 GB/s memory bandwidth、256 GB/s network bandwidth。每Package: 4 CUs (64 TFLOPs)、64 MB on-chip memory、2 TB/s memory bandwidth。RTL target TSMC N16 projected to N2。H100 baseline: 4×H100 SXM (NVIDIA NVML profiling + PyTorch 2.2 compiled kernels)，H200 baseline data from Artificial Analysis [4]。

- 评估性能的软件/脚本是什么。修改了什么。
  自研event-driven simulator + RTL model。RTL model：SystemC + Catapult HLS实现proof-of-concept RPU单核/多CU/board-level集成→VCS/Design Compiler/PowerPro synthesize VMM/DMA microkernel→extract calibrated energy/area。Event-driven simulator：用symbolic transaction (address/size/type)代替真实tensor data，RTL-calibrated参数建模data transfer/stall/arbitration。支持全模型DSE：sweep model (Llama3-8B/70B/405B, Llama4-Maverick/Scout)、batch size (1-128)、sequence length (8K-128K)、CU count (36-500+)、HBM-CO config (BW/Cap variants)。输出per-kernel timeline、pipeline utilization、buffer occupancy、power traces和stall analysis。H100 baseline profiling: NVML power measurement + PyTorch 2.2 isolated kernel profiling (Fig.2)。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未提供simulator或RTL开源。kernel执行流程（论文Fig.7,8描述）：
  1. Compiler生成RPU ISA instruction stream：Python compiler trace PyTorch model→lower torch.nn.Linear to three-stage micro-kernel (Loading: config DMA, Looping: drive VMM pipeline, Launching: forward activation)→static order DMA/compute instructions→pre-shard and quantize weights→generate synchronized memory/compute/network instruction streams with Pipeline Arbiter flags
  2. Decode一个token的kernel执行（以Llama3-8B, 64-CU RPU, BS=1为例）：
     - Layer start: host transfer KV$ from prefill engine to RPU→trigger RPU autonomous execution
     - wQKV: network DMA broadcast activation across all CUs→memory DMA prefetch compressed QKV weights→Stream Decoder on-the-fly dequantize→TMAC compute VMM→memory pipeline keeps prefilling buffer while compute waits for activation
     - QK^T: network DMA gather Q/K shards across CUs (Q-vector spans 2 CUs, KV spans 8 CUs)→TMAC compute attention→memory pipeline prefetches KV$ entries
     - Softmax: distributed max collective + exp-sum reduction across CUs sharing GQA heads→compute stalls briefly for sync→memory pipeline continues prefetch
     - s(QK)V: TMAC compute attention output→memory pipeline prefilling K$/V$ for next layer
     - wO/wUp/wGate/wDown: similar VMM pattern, TMAC for wUp/wGate can become compute-bound at BS=32→memory prefetches deep ahead (~6MB/CU, ~384MB system-wide)
  3. BS=32执行：compute处理wUp/wGate时~4× longer than memory read→memory pipeline prefetches KV$+weights into buffer→进入attention后KV$stream from on-chip buffer→compute operates at compute-bound performance until buffer drained→returns to memory-bound
  4. Pipeline Arbiter同步：每个SRAM buffer entry含2-bit valid counter→生产者DMA write时set valid count=expected consumers→消费者check-valid stall until ready→读后可选decrement counter→hardware-enforced arbitration serialize multi-consumer access→software-configurable priority policy

## LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现multi-chiplet GPU上显式同步操作的硬件级调度机制，不是软件kernel调度而是硬件同步执行路径的优化。核心：(1) Lazy Release Consistency调度——在LLC的sync-val directory中记录每个同步变量的owner chiplet，acquire/release同步操作根据四种场景（invalid/local chiplet/remote chiplet/evicted）决定是否触发L1.5 cache coherence action，将同步开销从"每次同步操作都付费"变为"仅跨chiplet ownership迁移时才付费"，利用chiplet内连续同步的时间locality减少L1.5 invalidate/flush。具体：同chiplet内连续acquire/release不flush/invalidate L1.5；跨chiplet ownership迁移时才flush旧owner L1.5 + invalidate新owner L1.5。(2) In-Network Atomic Merge调度——AMU在网络中对跨chiplet atomic同步请求进行合并调度：检测同地址/同opcode atomic请求→在merge table中合并（如atomicAdd(addr,1)+atomicAdd(addr,1)→atomicAdd(addr,2)）→timer到期或SM list满后发送合并请求→响应通过multicast broadcast到所有参与SM。对于atomicCAS，仅在比较数据相同时合并（多线程竞争同一lock时最多一个成功）。支持atomicAdd/Sub/Max/Min/And/Or/Xor/CAS等操作类型。实验比较：vs MCM-GPU（传统每次acquire invalidate L1.5、atomic直接路由LLC）、hLRC（同步变量多级cache追踪+write-back）、HMG（cache coherence protocol）、AMU-only。microbenchmarks（atomicTreeBarr/lfTreeBarr/spinMutex/sleepMutex/faMutex/spinSem2/10/120）上LRM-GPU平均加速1.19× vs MCM-GPU。全局同步workload（reduce/scan/histogram/pagerank/barnes-hut/hash-table/MST）上平均加速1.33×。AMU-only平均贡献1.16×加速。

- 后端平台是什么，配置是什么。
  GPGPU-Sim模拟的4-chiplet GPU系统：256 SMs (64 SMs/chiplet), 64 warps/SM, 32 threads/warp。每SM 128KB L1 data cache (128B line, 4-way, write-through)。每chiplet 2MB L1.5 cache (128B line, 16-way, write-through, 仅缓存remote data)。全局8MB LLC (128B line, 16-way, write-back)。L1.5+LLC总容量16MB。Inter-chiplet network concentrated hierarchical crossbar, 768GB/s bandwidth, 32 cycles/hop。DRAM 64 channels, 3TB/s。4KB page size, first-touch page allocation, distributed CTA scheduling。LRM-GPU sync-val directory: 64 entries (16/chiplet)。AMU: 16 channels, 2K entries/16 banks merge table, SM list depth 8, data 32B。Workloads用CUDA 11.1 O3编译。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：扩展版GPGPU-Sim（集成BookSim 2.0 chiplet interconnection platform）。论文修改：(1) 实现multi-chiplet sync-val directory lookup/allocate/evict逻辑，处理acquire/release四种场景的L1.5 coherence action调度；(2) 实现AMU merge table CAM/SRAM查找、ALU合并计算、countdown timer调度、multicast响应广播；(3) 基于BookSim 2.0构建composable multi-chiplet interconnection network。workloads包含：同步microbenchmarks（HeteroSync: atomicTreeBarr/lfTreeBarr/spinMutex/sleepMutex/faMutex/spinSem2/10/120, inputs 8 512 2）；带全局同步的benchmarks（reduce 8192 32, scan 16384, histogram 262144 256, pagerank coAuthorsDBLP.graph, barnes-hut 262144 4 0, hash-table 65536 2048, MST USA-road-d.NY.gr）；不带全局同步的benchmarks（b+tree command.txt, backprop 65536, bfs graph65536.txt, dwt2d 192.bmp, nn filelist_32, lavaMD 10x10x10, VGG16 fw, GPT-2 fw, hotspot temp_512 power_512）。指标：normalized speedup, L1.5 cache invalidation count, inter-chiplet traffic, energy consumption。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：LRM-GPU模拟器修改未开源。GPGPU-Sim和BookSim 2.0为开源项目。使用流程（基于论文描述）：
  1. 同步microbenchmark执行流（以lock-based synchronization + LRM-GPU为例）：SM0 (chiplet0) acquire lock → atomicCAS路由到LLC → sync-val directory无X记录 → 分配entry记录owner=chiplet0 → invalidate chiplet0 L1.5 → SM0执行临界区load/store（L1.5持有最新A=1, LLC含stale A=0）→ SM0 release → directory发现owner=chiplet0 → 仅写同步变量到LLC，不flush L1.5（延迟coherence action）→ SM1 (同一chiplet0) acquire → directory命中owner=chiplet0 → 直接从LLC读同步变量，不invalidate L1.5（利用locality）→ SM1执行临界区 → SM2 (chiplet1) acquire → directory发现owner=chiplet0 ≠ chiplet1 → flush chiplet0 L1.5到LLC (write back A=2) → 更新owner=chiplet1 → invalidate chiplet1 L1.5 → SM2读最新A=2并执行临界区。相比MCM-GPU每次acquire都invalidate L1.5，LRM-GPU仅在SM0→SM2的跨chiplet handoff时才触发coherence action。
  2. AMU atomic merge执行流（以atomicAdd为例）：SM1发出atomicAdd(addr0,1)→跨chiplet请求→AMU merge table miss→分配新entry(status=valid, op=atomadd, addr=addr0, SM list={1}, data=1)+启动timer→SM0发出atomicAdd(addr0,1)→merge table命中→ALU合并为atomicAdd(addr0,2)，SM list更新为{0,1}→timer到期→合并请求发送到LLC→entry状态变为reserve→LLC执行atomicAdd(addr0,2)→响应返回AMU→AMU multicast broadcast结果到SM0和SM1→entry释放。对于不能合并的atomicCAS（比较值不同），请求单独建表项或直接发送。
  3. 无全局同步workload验证：运行VGG16/GPT-2/hotspot等无global synchronization的benchmarks→LRM-GPU与MCM-GPU性能差异仅~2%→验证新增同步路径对普通workload干扰极小。

## Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Swift，面向GPU上SpMM (Sparse-Dense Matrix Multiplication, C=A×B)的双路径kernel系统。核心设计：(1) Sparsity-based sorting：按稀疏矩阵A每列NNZ升序排序列，同步重排稠密矩阵B的对应行，使warp内线程处理相邻列时访问B中连续地址。(2) Blocking：按warpSize=32将排序后A划分为regular block（列宽=32的完整block）和irregular block（不足32列或长列残留），生成blkPtr/blkColIdx/value/rowIdx/positionIdx/offsetIdx（regular）和irrPtr/irrValue/irrRowIdx/colIdxIndex/blkStart/blkStop（irregular）两套索引结构。(3) Dual-kernel路径：regular kernel以32×8 thread block处理8个sparse block+B中连续32列，warp lane读sparse value/rowIdx后访问B连续位置，乘积写入shared memory，用positionIdx/offsetIdx做segment sum降低atomicAdd开销；irregular kernel将长短列统一按sub-column/block拆分，warpId先通过colIdxIndex判断独立短列还是长列子块，lane以步长32遍历范围做atomicAdd写回。实验比较：在2757个SuiteSparse矩阵上对比ASpT、cuSPARSE v12.2、RoDe、Sputnik四种SOTA baseline，覆盖FP32/FP64精度和dense矩阵B的N=32/128/48/96/182/384/768等列数。RTX 4080s上FP64/N=32相对ASpT/cuSPARSE/RoDe/Sputnik分别为2.22×/59.19×/5.16×/10.92×；FP64/N=128相对1.79×/27.02×/3.62×/6.53×；FP32/N=128相对ASpT 1.19×。消融实验：regular part coalesced B access带来1.32×(N=32)/1.38×(N=128) speedup；irregular part load-balancing优化带来2.26×(N=32)/2.69×(N=128) speedup。

- 后端平台是什么，配置是什么。
  RTX 4080 SUPER (i9-14900K CPU)、RTX 3090Ti (i9-12900K CPU)、Tesla V100 (Xeon Gold 6151 CPU)、NVIDIA A100 (Xeon Gold 5120 CPU)。软件环境：Ubuntu 22.04.4、GCC 9.5.0、CUDA 12.2（NVCC编译），Matplotlib+Numpy用于绘图脚本。

- 评估性能的软件/脚本是什么。修改了什么。
  自研CUDA kernel实现Swift SpMM双路径系统。baseline使用公开代码：ASpT (adaptive tiling SpMM)、cuSPARSE v12.2 (通用SpMM库)、RoDe (CSR row decomposition SpMM)、Sputnik (ROMA+vector memory instruction SpMM)。性能指标：相对baseline的几何平均speedup。使用NVIDIA profiling工具比较memory bandwidth utilization、memory coalescing、L2 hit rate、SM occupancy。实验覆盖2757个SuiteSparse矩阵，dense matrix B随机生成。所有kernel在同一软硬件环境下编译执行保证公平性。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/MinttHu/Swift.git（论文首页脚注和Artifact Appendix均确认）。仓库含CUDA源码、编译脚本、数据下载脚本、FigurePlot原始数据处理脚本和SOTA方法复现实验入口。CUDA kernel使用流程：
  1. 预处理阶段（CPU）：读取CSC格式稀疏矩阵A→统计每列NNZ→按NNZ升序排序A的列并同步重排B的行→按warpSize=32划分regular block（列宽=32，blkPtr记录value/rowIdx起点，blkColIdx记录起始列）→剩余不足32列或长列残留元素归入irregular part（irrPtr/irrValue/irrRowIdx+colIdxIndex/blkStart/blkStop描述拆分的sub-column范围）→为regular part生成positionIdx/offsetIdx（segment sum前缀和索引）。
  2. Regular kernel执行：thread block=32×8线程，每warp(32 lane)处理一个sparse block+B中连续32列。每个lane读一个sparse value/rowIdx→根据blkColIdx+colIdx访问B连续位置（column-major layout保证coalesced）→乘积写入shared memory→利用positionIdx/offsetIdx对rowIdx相同的partial sum做segment sum→用较少的atomicAdd写回C。
  3. Irregular kernel执行：warpId通过colIdxIndex判断任务类型（独立短列或长列子块）→blkStart/blkStop或irrPtr定位irrValue/irrRowIdx范围→lane以步长32遍历该范围→对每个非零元循环访问B的N个元素→atomicAdd写回C。
  4. 例如：对一个M=10K、K=10K、NNZ=500K的SuiteSparse矩阵，预处理排序后将相邻NNZ数相似的列聚集成regular block（如NNZ≈64的32列），同一warp的32个线程同时取B的连续地址；长列残余（如NNZ=512的单列）被拆为irregular sub-block，由warp内各lane分担处理。

## Uni-STC: Unified Sparse Tensor Core

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出UWMMA（Unified Warp-level Matrix Multiply-Accumulate）指令序列和软硬件协同数据流，将稀疏计算任务从"gather data"转为"gather tasks"模式。软件侧用BBC格式表达sparse matrix，硬件侧通过load/task generation/numeric execution三类指令驱动Uni-STC tensor core。指令生命周期：stc.load同步收集metadata和values；stc.task异步触发TMS/DPG生成task queue（T1→T3→T4 task decomposition）；stc.numeric检查READY/BUSY状态驱动SDPU执行，完成结果写回register file。TMS的task ordering在outer-product与row-major顺序间动态选择以提升A/B tile复用并降低write conflict。实验比较SpMV、SpMSpV、SpMM、SpGEMM四类kernel，Uni-STC相对DS-STC和RM-STC的几何平均speedup为3.35x和2.21x，energy reduction为1.97x和1.27x，energy efficiency gain为7.05x和2.96x。

- 后端平台是什么，配置是什么。
  后端平台为GPU SM内集成的Uni-STC coprocessor。GPU架构参考NVIDIA Ampere/A100类设计：432个Uni-STC单元，MAC array配置64 MAC@FP64或128 MAC@FP32。SM需要扩展现有instruction decoder解析UWMMA opcode，扩展warp scheduler分发指令。数据路径通过register file和operand collector：SM90+利用高带宽operand collector，Ampere需拓宽register-file port（每线程每周期最多16个FP64 source + 4个FP64 destination operands）。模拟器基于Accel-Sim扩展STC simulator，加入asynchronous memory access support。

- 评估性能的软件/脚本是什么。修改了什么。
  基于Accel-Sim扩展STC simulator，加入asynchronous memory access支持。论文用同一T1 task调用粒度比较GAMMA、SIGMA、Trapezoid、NV-DTC、DS-STC、RM-STC和Uni-STC。公平比较：按理论计算吞吐对齐MAC array（64 MAC@FP64或128 MAC@FP32），采用SIGMA的PE设计缩放不同accelerator。能耗根据register activity和Sparseloop方法外推。Artifact包含Python/Bash/C++ scripts，qrun自动化复现流程。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未提供GitHub代码仓库，但Artifact Appendix标注Publicly available并提供Google Drive Docker artifact。使用流程：
  1. 导入Docker镜像，配置SuiteSparse/DLMC数据集路径
  2. 运行qrun fast verification（约5小时）：执行SpMV/SpMSpV/SpMM/SpGEMM的kernel-level性能模拟，输出cycle count和energy
  3. 运行qrun complete verification（约75小时）：全量2893矩阵+DLMC+AMG应用级模拟
  4. 结果包含各STC architecture（DS-STC、RM-STC、NV-DTC、GAMMA、SIGMA、Trapezoid、Uni-STC）的performance/energy/area对比数据

## μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出half-plus blocksize shaping技术，在封闭的NVIDIA GPU硬件调度器之上通过非侵入式修改kernel launch参数实现intra-SM scattered kernel co-location。核心kernel调度技术：(1) Half-plus blocksize shaping：对A40 GPU（1536 threads/SM, CUDA max blocksize=1024）将kernel blocksize设为768+α（α最小32即800），确保同kernel任意两个block的thread和在1SM内超过1536上限→阻止stacked co-location→迫使同kernel blocks散布到不同SM→剩余threads可分配给其他kernel的小block实现scattered co-location；(2) 1/3-plus shaping for A800/A100/H200：对2048 threads/SM的GPU，将blocksize设为2048/3+α（最小704），允许同kernel两个1/3-plus block入1SM共占≤1364 threads，留下≤684 threads供互补资源的小block，实现2/3利用率上限；(3) α动态调整：当kernel launch slack sk positive时α=32（最小warp数）减少resource fragmentation，当sk negative时逐步增加α（+32/warp）加速kernel执行降低SLO violation；(4) Time-shifted launching：对blocksize不可修改的kernel（cuDNN/cuBLAS闭源wrapper、tiling kernel如Conv2d），检查与当前SM执行中kernel的6种hardware资源combined utilization ≤ 100%且shared memory/registers足够→满足直接launch→不满足delay β μs后重检→更新slack重排kernel set O→若进入top-x则升级为half-plus；(5) Kernel分类：modifiable kernel（cudaLaunchKernel syntactic sugar + blocksize/gridsize可暴露修改）占51.63%执行（3512/6802次），unmodifiable kernel（cuBLAS/cuDNN wrapper + tiling kernel）占48.37%（3290/6802次）；(6) GPU线程容量分析：61.85% kernel（max batch下）的线程数超过GPU总线程容量，占70.83%总执行时间，导致stacked co-location。实验比较μShare vs baselines在kernel-level scattered vs stacked co-location下的性能差异。消融实验：μShare shape 1024（固定blocksize）、μShare w/o shape（无shaping）、不同unmodifiable kernel比例对throughput的影响。Intra-SM co-location对比Tacker (kernel fusion)。NVIDIA Nsight Compute和Nsight Systems profiling数据。66+ kernel types across 10 models, 6802 total executions的统计分析。

- 后端平台是什么，配置是什么。
  (1) NVIDIA A40 GPU：84 SMs, 44.784GB memory, 每SM 1536 threads, 102,400B shared memory, 65,536 registers, CUDA 11.8, blocksize range for half-plus: {b | 768 < b ≤ 1024, b ≡ 0 (mod 32)}, 最小800。(2) NVIDIA A800 GPU：108 SMs, 80GB memory, 每SM 2048 threads, 167,936B shared memory, 65,536 registers, CUDA 12.1, blocksize range for 1/3-plus: {b | 683 < b ≤ 1024, b ≡ 0 (mod 32)}, 最小704。CPU：Intel Xeon Gold 6338 (128逻辑核, 2.00GHz)。PyTorch 2.2.0。

- 评估性能的软件/脚本是什么。修改了什么。
  自研μShare系统（C++ shared libraries），使用NVIDIA Nsight Compute测量6种low-level hardware utilization（FP32/FP64/INT32/LDST/SFU/Tensor cores），NVIDIA Nsight Systems记录kernel launch timing和execution timeline。CUDA inline assembly读取SM ID register和GPU clock counter register确定block的SM placement和start time。通过LD_PRELOAD劫持kernel launch函数、libdl (dlopen/dlsym)获取原始函数、libc (shm_open/mmap)实现共享内存inter-process通信。baseline INFless使用MPS+PyTorch memory control实现均匀SM/memory分配；Orion控制kernel launch time实现最多1 compute + 1 memory kernel per GPU共置。
  
  详细分析：(1) Block placement profiling：选择vectorized kernel（layer normalization, 主要用LDST）和roll kernel（displacement array, 主要用INT32）两个高频kernel（占总invocations 28.90%）→分别在exclusive和concurrent模式下发→CUDA inline PTX读取SM ID →确认stacked co-location模式。(2) Thread count analysis：从10模型6802次kernel执行统计→max batch下61.85% kernel超过GPU总thread capacity→70.83%总执行时间→导致stacked co-location。(3) Hardware utilization profiling：6802次执行→NVIDIA-SMI report 81.16% utilization → Nsight Compute report仅9.28% low-level hardware utilization→"1 more, 5 less" pattern（top 20 kernels: primary HW avg 30.19%, remaining 5 avg 5.07%）。(4) Blocksize effect experiment：roll kernel exclusive执行时blocksize 512最优（throughput 22863）→co-locate with vectorized kernel（blocksize 256）时roll最优blocksize变为1024（throughput 1.98× improvement）→证明static preset blocksize在co-location下不optimal。(5) Co-location pairing experiment：选不同dominant HW kernel配对→dominant resource不同时half-plus提升throughput 19.94%（前19 bars）、相同>10.37%下降（后4 bars）。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未明确提供开源代码仓库（HPCA 2026）。μShare kernel调度使用流程：
  1. 环境准备：CUDA 11.8/12.1 + PyTorch 2.2.0 → 编译μShare .so（kernel interceptor + block shaper + batch manager）→ 设置LD_PRELOAD
  2. Offline profiling：对每个模型，PyTorch运行max batch → Nsight Compute CLI: ncu --metrics gpu__time_active,fp32_active,fp64_active,int32_active,ldst_active,sfu_active,tensor_active → Nsight Systems CLI: nsys profile记录kernel launch timing → 输出per-kernel 9-tuple resource profile
  3. Kernel interceptor加载：dlopen("libcudart.so") → 获取cudaLaunchKernel原始函数指针 → 同名wrapper函数通过LD_PRELOAD先加载 → 拦截所有kernel launch → dlsym获取blocksize/gridsize/sharedMem/stream参数
  4. Half-plus shaping：读取SM thread capacity（A40=1536）→ 计算half+α = 768+32 = 800 → 或根据slack递增α → mmap into shared memory → 修改blocksize参数 → 调用原始cudaLaunchKernel
  5. Time-shifted launching：检查kernel资源需求 {rFP32, rFP64, rINT32, rLDST, rSFU, rTensor} 与当前SM中kernel combined → 若 ≤ 100%且rmem/rreg足够 → 直接launch → 否则usleep(β)后重试
  6. 例如co-locate vectorized kernel (FP32 13.43%/LDST 58.02%/SFU 11.03%, blocksize 1024 half-plus) + roll kernel (INT32 33.25%/SFU 24.94%, blocksize 512 default)：half-plus vectorized block占800+ threads → 剩余736 threads < 1024 = 不能放第二vectorized block → roll block (512 threads) 可放入剩余→ SM内同时执行LDST+SFU (vectorized) 和 INT32+SFU (roll) → 总HW utilization提升
  7. Monitoring：每个kernel执行完后Nsight Compute report utilization → 验证co-location效果 → 调度decision记录到log

## QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出QuCo firmware自动计算ATT（Asynchronous Tile Transfer）operand queue的最优配置参数——tile size、queue slots数量、LDS分配和synchronization barrier——无需程序员手动tuning。核心算法：(1) Optimal Tile Size Calculation（Algorithm 1）：遍历tile size范围64-8192 elements，对每个候选tile计算merit factor = processing time / memory transfer time（Algorithm 2），再乘以cost function得weighted merit，选最优tile后按CI（Compute Intensity）调整——低CI (<1, e.g. Elementwise/Dot-Product)放大tile提升memory throughput，高CI (>4, e.g. Matrix-Matrix)缩小tile平衡memory-computation overlap。(2) Processing time估算：bestLatency = TileSize / (SIMD Muls per cycle × consumer wavefronts)，加scheduling roundtrip overhead（min(consumerWfs-1, wavefrontPools)因子）。(3) Memory transfer time估算：latencyTotal = ATT cycles + DRAM latency + L2 latency，memTransferTime = TileSize×ElementSize / Bandwidth，cacheTransferTime = 2×TileSize×ElementSize / CacheLineSize（双向因子），总memTime = latencyTotal + memTransferTime + cacheTransferTime。(4) Optimal Number of Slots Calculation（Algorithm 3）：对streaming queues用hardware-aware Little's Law适配——理想slots = memory transfer time / tile compute time，然后CU-aware rounding（更多CU减少slots降低memory contention），LDS capacity check失败时fallback到CI-based scaling（低CI多slots提升throughput、高CI少slots降低pressure）；对stationary queues均分remaining LDS capacity后round to power-of-two + CU-aware rounding。QuCo根据user注册的queue类型（streaming vs stationary）和kernel CI自动决定最优参数。实验比较6种case：(i) NoATT/Non-Tuned（naive实现、小tile、无ATT）；(ii) NoATT/Fine-Tuned（extensive DSE优化tile/slots但无ATT）；(iii) ATT/Non-Tuned（用ATT但参数未tuned）；(iv) ATT/Informed-Tuned（heuristic-based配置，tile 64-256, slots 2-4，inspired by NVIDIA guidelines）；(v) ATT/Fine-Tuned（per-kernel exhaustive DSE找最优tile/slots，对Matrix-Matrix需2.6e+14次kernel launch）；(vi) QuCo（自动单次pass配置）。8个linear algebra kernels + 6个benchmarks（3个DNN models + 3个composite kernels）。消融实验：移除CU-aware slot scaling、Little's Law、CI-based scaling。portability测试在3 GPU（MI-100/R9 Nano/Radeon 530）同binary下评估。DVFS测试在MI-100 Whisper-Tiny上3种频率变化scenario（Decreasing/Decreasing-Increasing/Decreasing-Holding）下对比QuCo-HW（适应实际频率）vs QuCo-SW（假定默认频率）。

- 后端平台是什么，配置是什么。
  MGPUSim cycle-accurate GPU simulator，校准为AMD R9 Nano（GCN3 ISA, 64 CUs, 1.0 GHz, 64 SIMD Muls/cycle, 64×16KB L1V$, 16×32KB L1I$, 16×16KB L1S$, 16×256KB 16-way L2$, 8×512MB DRAM, memory latency L1/L2/DRAM = 190/300/450 cycles）。portability测试额外覆盖AMD MI-100（120 CUs, 1.5 GHz, 32×256KB 16-way L2$, 32×1GB DRAM, latency 100/250/300）和AMD Radeon 530（6 CUs, 1.0 GHz, 8×256KB 16-way L2$, 8×256MB DRAM, latency 80/200/400）。

- 评估性能的软件/脚本是什么。修改了什么。
  MGPUSim扩展支持ATT（global memory↔LDS asynchronous tile transfers）。线性代数kernel集合（ElementwiseK、Elementwise、Sumvectors、Dot-Product、Matrix-Vector、Matrix-Matrix、MM+Reduction、Batched MM）手动实现wavefront specialization + Operand Queues（基于producer-consumer模式：dedicated producer wavefront用Push/Wait_For_Push + consumer wavefronts用Peek/Pop + asynchronous transaction barriers）。DNN benchmarks（AlbertV2 74层、T5-Small 96层、Whisper Tiny 827层）和composite kernels（Norm-Project、Attention-Score、Residual-MLP）基于linear algebra kernel构建。性能归一化到ideal ATT scenario（unlimited LDS, 所有data可同时驻留）。DRAM activity traces用于分析memory-level effects。论文将Operand Queues实现类比于NVIDIA cuda::pipeline API + CUTLASS3+CuTe / ThunderKittens的高级ATT抽象。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：QuCo firmware和MGPUSim ATT扩展未明确提供开源链接（HPCA 2026）。MGPUSim开源（https://github.com/umd-memsys/mgpusim）。kernel调度使用流程：
  1. 环境准备：配置MGPUSim ATT扩展→设置GPU参数（CU数量、频率、memory latency/bandwidth、LDS容量）→GST数据预填充
  2. Host code编写：driver.InitQuCo(CI, WG_SIZE, #CUs)初始化→计算kernel CI（ratio of FP ops to global memory traffic without ATT）→driver.RegisterQueue(K_dim, element_size, TYPE_STREAMING/STATIONARY)注册queues
  3. Kernel launch：QuCo firmware执行tile size和slots计算→写入descriptors到LDS→ATT unit加载descriptors开始异步数据搬运
  4. 例如Matrix-Matrix（[512,2048]×[2048,128], High CI>4, 8+1 queues）：QuCo选tile 1024/512 + slots 2（CU-aware从4降为2降低memory contention）→streaming queues用Little's Law平衡queue depth→stationary queues用剩余LDS均分
  5. DNN models（如Whisper Tiny 827层）：QuCo per-layer reconfigure queue参数（tile sizes从256到1024不等，slots从2到4不等，因层而异）→无需per-layer manual tuning
  6. 输出：speedup vs ideal ATT、DRAM activity trace（QuCo维持持续高DRAM利用率vs NoATT/Fine-Tuned的bursty access）、ablation study per-heuristic贡献

## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出基于DSM的fused kernel dataflow调度，在NVIDIA H100 cluster架构上实现GEMM chain/Gated FFN/conv chain的跨SM数据流编排。核心kernel调度设计：(1) Cluster-level tiling：定义clsi（沿维度i的cluster内并行block数）和blki（block沿维度i计算的数据粒度）。cluster size四维(clsm, clsn, clsk, clsl)决定inter-block dataflow模式。(2) dsm_comm原语调度：GEMM0阶段K维spatial partition产生partial sum→dsm_all_exchange在cluster内沿K维All-Reduce聚合完整中间tile；GEMM1阶段dsm_shuffle在Shuffle Group内交换C矩阵切片供不同block计算不同E partial sum；Store阶段dsm_reduce_scatter cluster内scatter-reduce+inter_cluster_reduce TMA跨cluster原子聚合。对Gated FFN：dsm_all_exchange从Add变为Mul操作；两种spatial策略——cls_k=2空间划分两GEMM branch最大化并行（但增加DSM通信）vs 单block内顺序执行两GEMM最小化DSM通信。(3) Loop scheduling：统一codependent loop dimensions为独立集合X={x0,...,xJ-1}，通过permutation设置nesting order，partition为spatial(并行处理)或temporal(单processor顺序执行)维度。MNLK vs MLNK order影响需cached tensor大小和spilling策略。(4) Resource mapping：heuristic-driven贪心分配——按getFootprint确定单tile数据量→greedy分配到reg→SMEM→DSM→global层次→容量不足时逐层spill。(5) Kernel codegen调度：扩展CUTLASS prologue→mainloop→epilogue结构——prologue初始化DSM semaphore；mainloop插入DSM通信原语（生产者accumulation done→DSM mul/exchange→barrier；消费者循环→DSM shuffle ring communication）；epilogue执行DSM reduce后store global memory。实验比较GEMM chains/conv chains/Gated FFN kernel latency vs BOLT/Chimera/Relay/TASO/TensorRT/PyTorch；DSM primitive bandwidth utilization（shuffle/reduce/mul在不同cluster size下）；ablation隔离SE/DA/DC贡献；cost model搜索accuracy和top-K选择；brute-force vs search engine编译时间。

- 后端平台是什么，配置是什么。
  NVIDIA H100 SXM GPU（Hopper架构），CUDA 12.4。利用H100 Thread Block Cluster机制（最多16个SMs per cluster，DSM通过SM-to-SM NoC连接）。单SM SMEM上限约227KB。DSM bandwidth随cluster size增加逐渐降低（cluster size=2时最高，cluster size=16时最低但仍优于global memory bandwidth）。DSM latency始终低于global memory。

- 评估性能的软件/脚本是什么。修改了什么。
  FlashFuser自研Python search engine + CUTLASS CUDA codegen。评估使用NVIDIA Nsight Compute 2025.2.0 profiling memory access；PyTorch 2.6作为non-fusion baseline；BOLT/Chimera/Relay/TASO作为compiler baseline；SGLang用于端到端evaluation。修改：在CUTLASS kernel结构中嵌入DSM communication primitives；Python前端实现Dataflow Analyzer和Fusion Search Engine；内核扩展TMA数据移动+mbarrier many-to-many同步替代原有all-to-one cluster-sync。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未明确开源。Kernel调度使用流程：
  1. 子图定义：输入GEMM chain维度(M,N,K,L)或conv chain(IC,H,W,OC1,OC2,K1,K2)或Gated FFN参数
  2. Search space构建：枚举loop schedule（MNKL/MNLK等×spatial/temporal partition 24+12+4+1=41种组合）×tile size（block-level以16×16×16 MMA为最小粒度）×cluster size（{1,2,4,8,16}^4=54种）→初始约2.75×10^13候选
  3. Pruning：5条规则逐级过滤→Rule1可整除tile→Rule2 cluster product≤16→Rule3 accumulation dim innermost→Rule4 L维度不能spatial→Rule5 capacity≤lowest cache→约1.15×10^6候选
  4. Dataflow Analyzer：对每候选生成spilling plan和data movement volume→cost model C_l=V_l/B_l minimax评估→保留top-11
  5. H100 profiling：11个候选编译为CUDA kernel→on-device测实际latency→选最优
  6. 例如GPT-6.7B FFN fusion：cluster size(2,4,2,4)→blkm=128, blkk0=128, blkn=128, blkk1=64, blkl=128→GEMM0 M=128沿M spatial partition到2平行block→K=4096沿K partition到2平行block产生partial C→dsm_all_exchange沿clsk=2 All-Reduce→GEMM1 dsm_shuffle clsshuffle=2交换C→dsm_reduce_scatter clsreduce=2聚合→inter_cluster_reduce→store E
  7. Runtime：M维在inference时动态变化→binning策略按M范围预编译多kernel→查表选择

## VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出两个专用硬件dataflow：(1) Unified Attention Core的Row+OP可重构attention dataflow：Big Attention使用Row dataflow（PE Cluster按attention head分配，标准Row-stationary attention dataflow），Small Attention使用OP dataflow（PE Cell/Node改为output-stationary dataflow），通过Snooper配置不同PE Cell接收的packet ID，再由Fat Tree分发到各lane实现dataflow动态切换。Row+OP MAC通过Divide-and-Conquer和Fluid Zone Detection技术减少FP累加功耗。(2) Radix Sort Core的大K TopK dataflow：TP（Parallel-to-Sequential Converter）将并行confidence vector转串行→CountBin按radix digit分桶计数→PrefixSum计算前缀和定位candidate bin→SelectBin定位含第K大元素的bin→Filter从candidate bin筛选TopK元素并输出。加入Locality-aware Scheduling：根据mask map产生history table标记高置信空间区域，PE分组并行在不同区域各自执行Radix Select，利用confidence map空间偏斜优先处理靠近已解码token的区域。Radix Sort Core将大K TopK从通用排序问题（传统Bitonic Sort+Merge Sort在大K上需反复读写重排，N=4096时K=1936时TopK仅占3.5%操作数却占20.9%延迟）变为固定4阶段pipeline消除全局排序开销。实验比较：在VAR-Turbo accelerator上评估Radix Sort Core的延迟贡献降低效果；Attain Core Row vs OP dataflow的硬件效率对比（避免为两类attention分别放独立core造成的低利用率）。

- 后端平台是什么，配置是什么。
  VAR-Turbo accelerator (TSMC 28nm+HPC, 1P8M CMOS, TT 25C, 7.09 mm², 1.98 W, 目标频率论文未明确说明)。片外DRAM：2×64bit HBM2 channel @2GHz, 32GB/s。通用对比baseline：NVIDIA V100 GPU (14 TFLOPS FP32, 16 HBM2 channels 512GB/s)。

- 评估性能的软件/脚本是什么。修改了什么。
  自研cycle-accurate simulator + RTL仿真（Synopsys VCS + Design Compiler + PrimeTime PX）。RTL仿真通过测试用例校验Attention Core Row/OP dataflow切换功能正确性和Radix Sort Core四阶段pipeline功能正确性。模拟器与RTL延迟匹配率0.90。ViTCoD/AdapTiV baseline在相同工艺和simulator框架下重新评估。

## TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文实现Hierarchical KV Cache Manager，一个基于多CUDA stream并行pipeline的KV cache运行时管理子系统，用于支持请求的抢占和恢复。核心kernel调度实现包含：(1) 三并行CUDA stream pipeline：compute stream（GPU推理计算）、write/load stream（KV cache在GPU显存与CPU memory间的传输）、evict stream（GPU显存释放）；(2) Write-through KV cache策略：每次decode iteration后将新生成的KV chunk放入write buffer，在下一轮计算前根据compute duration预估选择合适大小的chunk，通过write stream同步到host memory；(3) Synchronous chunked writing：动态chunk sizing和batched transfer，用CUDA events协调compute、write、load、evict四类操作的非阻塞执行；(4) Load-evict overlap：preempted请求已同步的chunk直接释放，未同步的剩余chunk与load操作重叠传输，减少上下文切换延迟。实验比较：(a) 消融实验：完整系统 vs w/o offload（127.28s vs 66.00s完成时间）、w/o write-through、w/o evict-load overlap；(b) 系统端到端性能：burst/Poisson/real trace场景下的effective throughput和TTFT。

- 后端平台是什么，配置是什么。
  NVIDIA H200、NVIDIA RTX 4090、NVIDIA A6000 GPU。micro experiment中报告Huawei Ascend 910B支持。H200设置mem-frac=0.3。GPU显存作为CPU memory上大容量KV cache的高速cache。

- 评估性能的软件/脚本是什么。修改了什么。
  基于SGLang框架扩展，约3000行Python代码。Hierarchical KV Cache Manager使用Python multithreading + CUDA streams（PyTorch CUDA stream API），动态管理三类stream：compute stream（LLM推理forward pass）、write/load stream（KV cache chunk的device↔host传输）、evict stream（GPU显存block释放）。通过CUDA events在stream间建立同步点，实现非阻塞overlap。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  未开源（arXiv https://arxiv.org/abs/2510.02758 和 EuroSys 2026 DOI https://doi.org/10.1145/3767295.3769328 均未提供代码仓库）。KV cache数据流运行流程（以H200 + Llama3-8B为例）：
  1. GPU显存作为CPU memory上大容量KV cache的高速cache。不同于普通write-back策略（仅在真正抢占时写回），TokenFlow使用write-through。
  2. Decode iteration后：新生成的KV chunk放入write buffer → 下一轮compute前预估compute duration → 选择大小合适的chunk通过write stream同步到host memory → compute stream和write stream通过CUDA event并发执行（overlap compute和I/O）。
  3. Request preemption：scheduler决定抢占请求 → 已write-through同步的chunk可立即释放显存block → 未同步的剩余chunk与load stream加载新请求KV chunk的操作通过evict-load overlap重叠执行 → 显存block在evict stream中释放。
  4. Request resume：load stream从CPU memory加载请求的KV chunk回GPU → 恢复decode。后台write-through保证大部分KV cache已在host同步，resume时只需加载最近未同步的增量chunk。
  5. 消融效果：去掉offload时完成时间从66.00s恶化到127.28s（恶化93%），说明分层KV cache管理是TokenFlow性能收益的核心来源。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未开源RTL和simulator。Dataflow运行流程（以256×256 VAR, N=4096, K=1936为例）：
  1. Attention Core dataflow切换：浅层Learning Region→SA模式：Snooper配置PE Cell按OP dataflow接收packet ID→Fat Tree分发→local window内token经Small Attention聚合为representative token；随后→BA模式：PE Cluster按attention head分配→Row dataflow执行Big Attention全局建模
  2. Radix Sort Core TopK dataflow（PD/DB阶段）：confidence array (N=4096) 经TP串行化→CountBin按radix digit将4096个元素分到bins→PrefixSum计算每bin前缀和确定第1936大元素所在bin→SelectBin精确定位含K-th元素的candidate bin→Filter筛选TopK 1936元素的indices→Locality-aware Scheduling：history table标记已解码区域→PE分组优先处理高置信空间区域
  3. DB阶段TopK：类似流程但处理per-token importance scores→选TopK进入完整Transformer，其余bypass
  4. Divide-and-Conquer FP accumulation：Row和OP MAC共享FP accumulator，Fluid Zone Detection动态调整累加精度边界降低功耗

## High Throughput and Low Latency LLM Serving via Adaptive KV Caching

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文在eLLM系统的layer-level implementation中引入kernel调度优化，核心包括：(1) layer-wise kernel fusion：将下一层未缓存token的KV recomputation kernel K1与当前层当前token的decode attention kernel K2融合为单一fused kernel，K1是compute-intensive GEMM（对旧token重算K/V projection），K2是memory-intensive attention decode（对当前token做MHA/GQA），融合后减少kernel launch overhead并提升SM utilization；(2) dual CUDA stream异步执行：通过torch.cuda.stream启动两个CUDA stream，Stream A负责host↔GPU cached KV数据传输，Stream B负责K1+K2 fused computation，实现communication-computation overlapping；(3) 动态线程分配：预编译多组CUDA shared libraries，对recomputation和decoding kernels生成32到1024、步长32的线程配置，运行时根据layer-level K1/K2估计计算量比例选择合适.so，线程数调整为32的倍数匹配NVIDIA warp granularity；(4) layer-granular KV block管理：将vLLM的粗粒度KV block按F个连续layer划分为更小单元（默认F=4），维护map table用于runtime精确定位每个token在每个layer的KV cache状态，减少内存碎片。实验比较：(a) ablation study：禁用Kernel Fusion后TPOT和throughput退化，禁用Comm-Com Overlapping后退化更明显（因PCIe 4.0 x16带宽成为瓶颈）；(b) 融合vs非融合kernel的SM utilization对比，fused kernel显著减少GPU idle bubble；(c) different F values对性能影响，F=4在映射开销和碎片间取得最佳平衡。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80GB GPU，PCIe 4.0 x16连接（无NVLink），CUDA 12.4，NVIDIA Driver 550.107.02。Llama2-13B单卡MHA，Llama2-70B四卡tensor parallel GQA。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：基于vLLM框架的内置profiling机制和torch.cuda.event计时。修改：(1) 预编译CUDA shared libraries：对recompute K1和decode K2 kernel预生成32-1024 thread stepped by 32的多组.so文件；(2) torch.cuda.stream异步CUDA stream管理重构：替代vLLM原有同步执行路径；(3) fused kernel launch：将K1(GEMM)+K2(attention)合并为single kernel launch，内部按计算量比分配thread blocks；(4) layer-granular KV map table runtime lookup：O(1) hash-based查找替代原有粗粒度block遍历。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源状态：eLLM整体未找到官方GitHub仓库，kernel-level CUDA代码属于eLLM系统内约1,700行CUDA代码的一部分。kernel调度使用例子（以Llama2-13B单卡decode为例）：
  1. 编译阶段：对Llama2-13B的MHA attention，论文预compile K1(GEMM for old token K/V projection)和K2(MHA decode attention)的fused kernel variant，线程数从32到1024 step 32（共32组.so），覆盖不同layer维度（hidden_size=5120, num_heads=40）和不同uncached token count的workload组合。
  2. 调度阶段：request-level optimizer确定b和r后，layer-level读取每个layer需要重算的uncached token数（r × 历史token数），估算K1 FLOPs = 2 × hidden_size × head_dim × num_heads × num_uncached_tokens，K2 FLOPs = O(hidden_size × num_cached_tokens)，计算FLOP ratio = K1_FLOPs/K2_FLOPs。
  3. 线程分配：按FLOP ratio分配thread blocks——若K1占总FLOP 70%则分配约70% threads给K1子任务、30%给K2子任务，总threads取32的倍数。从预编译.so库中加载最接近目标thread数的variant。
  4. 运行时：Stream A异步执行cudaMemcpyAsync将host memory中swapped KV传输到GPU（对当前layer i+1），Stream B同时执行fused kernel：K1子kernel为layer i+1的uncached old token执行attention projection生成KV（临时存于workspace buffer），K2子kernel用layer i已准备好的完整历史KV（cached+recomputed）对current token执行decode attention。两个kernel共用一次launch，内部thread blocks按计算量比例分区，完成MHA输出。K1产生的临时KV在K2使用后立即释放。
  5. 同步：每layer结束时cudaStreamSynchronize对齐Stream A和Stream B。论文消融显示kernel fusion使TPOT降低而额外显存开销约1 layer的KV workspace。

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  在Qualcomm Hexagon NPU (HVX vector + HMX matrix混合架构)上实现两个核心kernel：(1) Hardware-aware tile-quantized dequantized GEMM kernel：离线将权重按HMX FP16 tile layout（tile级column-major，tile内2-row permutation）重排→在memory order上做group size 32的4-bit量化→8个group coalesce为128-byte super-group适配HVX 1024-bit vector register。运行时HVX用vlut16查表将INT4权重转为FP16并广播scale→HMX执行FP16 32×32 tile-level inner product→accumulate到FP32 accumulator。与传统layout相比加速9.65-19.04×，比仅用HMX layout版本加速1.82-3.45×，仅比no-dequantization上界慢27%。(2) LUT-based FP16 FlashAttention/Softmax kernel：利用safe softmax保证exp输入≤0，预计算64KiB FP16 exp LUT（32768个entry存于TCM）→通过vgather指令查表替代多项式exp2→减少VLIW顺序依赖。相比F32 exp加速1.26-2.19×，相比F16 exp加速最高1.60×。算子消融：对比F32 polynomial exp、F16 polynomial exp、no dequantization上界。Attention延迟分解显示Softmax占比随query length增加从39.2%(q=4)升至84.6%(q=32)。

- 后端平台是什么，配置是什么。
  Qualcomm Hexagon NPU (V73/V75/V79三代)：HVX vector units (1024-bit VRF, 4-6个, 单thread FP16 GEMM ~33 GFLOPS)，HMX matrix units (1-2个, FP16 GEMM ~12 TFLOPS)，1MiB L2 cache，8MiB TCM (software-managed)，DMA ~60GB/s，l2fetch 20-30GB/s。HVX scatter/gather操作和所有HMX指令仅可访问TCM。HMX FP16 tile为32×32=2KiB，每两行permuted。手机：OnePlus Ace3 (Snapdragon 8 Gen 2/V73)、OnePlus 12 (Snapdragon 8 Gen 3/V75)、OnePlus Ace5 Pro (Snapdragon 8 Elite/V79)。

- 评估性能的软件/脚本是什么。修改了什么。
  Hexagon NPU operator library (htp-ops-lib) + llama.cpp Hexagon NPU backend。约7K行C/C++和inline assembly，用Hexagon SDK 6.0.0.2 LLVM toolchain编译。修改：(1) GEMM kernel：从conventional column-major layout的scatter-based dequantization改为HMX tile layout + group coalesce的连续写入TCM方案；(2) Softmax kernel：从F32/F16 polynomial exp2改为64KiB LUT-based vgather exp，只存x≤0范围；(3) dequantization: vlut16直接INT4→FP16查表，并用LUT广播scale替代split-broadcast；(4) Attention：实现FP16 FlashAttention on NPU（Algorithm 1: tile-level Q/KV分块、FP16 online softmax、FP32 accumulation on critical ops）；(5) rpcmem/dmabuf共享内存通信替代FastRPC默认RPC降低延迟。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/haozixu/llama.cpp-npu，https://github.com/haozixu/htp-ops-lib。kernel使用例子（Qwen2.5-1.5B decode, batch size=8, OnePlus 12）：
  1. 初始化：CPU调用FastRPC启动Hexagon NPU remote session→NPU侧thread pool初始化→分配rpcmem/dmabuf共享内存区域（模型weight 1056MiB for 1.5B, 2090MiB for 3B under ctx=4096）。
  2. 线性层GEMM kernel：CPU将operator request写入共享内存（含activation pointer, weight pointer, dimensions）→cache flush→NPU thread轮询到请求→DMA将weight tile从DDR搬入TCM→HVX用vlut16查表将INT4→FP16（每个8-bit index映射为16-bit value, vlut16生成一对HVX register）→HVX用vlut16 LUT（scale of 4 groups as LUT content + constant indices）广播scale→HMX加载FP16 activation tile和weight tile→32×32 tile-level inner product→accumulator累加→output tile写回TCM→DMA搬回DDR。
  3. FlashAttention Softmax kernel：HMX执行QK^T tile multiplication（FP16 input, FP32 accumulate）→output S tile in FP16→HVX执行rowmax reduction→S - rowmax（safe softmax）→HVX vgather查64KiB exp LUT（input MSB忽略+left shift 1 bit作为byte offset, 一次vgather收集64个FP16）→HVX rowsum（FP32 accumulate）→HMX执行PV multiplication→online rescale + accumulate O。LUT存储在TCM固定64KiB区域（占总TCM 0.8%）。
  4. Test-time scaling集成：Best-of-N生成B个候选路径→CPU llama.cpp sample层维护B个序列的KV cache→每个transformer layer执行时B条路径的activation rows映射到HMX tile的B行→B=8时相比B=1无显著增加HMX延迟→lm_head/logits在CPU侧计算（B=16时CPU时间占比≥50%）。Beam Search在每step结束时由PRM skimmer评分并剪枝。

## TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文提出两个核心运行时调度机制：(1) Pipelined Parameter Restoration调度：将LLM inference的computation graph扩展为三类restoration operator（allocation/loading/decryption）+ computation operator的DAG pipeline。CPU上实现priority-based greedy scheduler：若computation operator ready则优先执行（最高优先级），否则执行与earliest computation operator关联的restoration operator。alloc/decr operator被切分为micro-operator支持preemptive scheduling（computation就绪时可抢占长restoration micro-operator），减少NPU/CPU idle bubble。Partial parameter caching按topological order缓存早期prefill参数，按reverse order释放。(2) Co-driver NPU job执行路径：将一次secure NPU job拆为REE control plane（scheduling/power/frequency）+ TEE data plane（secure job context/MMIO launch/interrupt completion）。Shadow job机制使REE统一调度REE NN job和TEE secure job，TEE通过initialized-not-issued状态机+monotonic sequence number校验防重放/重排序。NPU world switch通过TZPC（secure MMIO access）+ TZASC（secure memory access）+ GIC（secure interrupt routing）硬件配置完成。实验比较：pipeline scheduling overhead vs critical path lower bound、preemptive scheduling对TTFT的改善、不同cache proportion下TTFT变化、NPU sharing对REE NN应用throughput影响、CMA allocation对REE Geekbench干扰。

- 后端平台是什么，配置是什么。
  Orange Pi 5 Plus (RK3588 SoC)：CPU 4×Cortex-A76 @2.4GHz + 4×Cortex-A55 @1.8GHz，NPU 3-core ~6 TOPS，16GB LPDDR4X。TEE OS基于OpenHarmony TEE系统，REE OS为OpenHarmony v4.1 / Linux v5.10。LLM TA使用4×A76 core + 3-core NPU。

- 评估性能的软件/脚本是什么。修改了什么。
  基于llama.cpp（约1.2K LoC修改实现pipelined restoration，约1K LoC集成NPU data plane driver）+ Rockchip NPU driver v0.9.8（167 LoC shadow job scheduling）。TEE OS扩展约62 LoC（CMA page mapping）+ 约50 LoC（TZASC/TZPC动态配置）。OpenSSL做参数加解密。benchmark：UltraChat/PersonaChat/DroidTask评估LLM serving性能，YOLOv5/MobileNet评估NPU time-sharing对REE NN应用影响，stress-ng模拟memory pressure，Geekbench评估CMA对REE应用干扰。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：Zenodo artifact DOI 10.5281/zenodo.17213486（EuroSys '26 artifact, MIT License, 含prototype源码和复现脚本），arXiv:2511.13717。pipeline/kernel执行流例子（Llama-3-8B 512-token prompt on RK3588）：
  1. 冷启动：LLM TA提取llama.cpp computation graph拓扑顺序→为第一个prefill operator调用extend_allocated→REE Linux CMA返回连续物理内存→REE file system DMA加密参数入此内存（未TZASC-protected, 避免bounce buffer）→TA调用extend_protected→TEE OS扩展TZASC region映射入TA→OpenSSL AES解密参数→computation operator ready→CPU/NPU prefill开始。
  2. Pipeline调度：scheduler维护computation operator和restoration operator的就绪队列→priority policy：CPU computation优先→若computation未就绪则调度与earliest computation关联的restoration→large allocation/decryption被切为micro-operator→computation ready时抢占当前restoration micro-operator。
  3. Co-driver NPU secure job：LLM TA准备secure execution context（command/register sequence, I/O page table, buffers）→向REE driver提交paired shadow job→REE NPU scheduler按统一队列调度→shadow job选中时REE driver smc通知TEE→TEE data plane driver用TZPC阻止REE访问NPU MMIO+GIC路由NPU interrupt到TEE→等待non-secure job完成→TZASC允许NPU访问secure memory→验证sequence number（防重放/重排序）→写MMIO launch NPU job→secure interrupt到达后TEE driver收尾→NPU切回non-secure mode→REE标记shadow job完成继续调度。
  4. 结果：pipeline scheduling距critical path lower bound仅0.01%-9.9% overhead；preemptive scheduling进一步降低TTFT最多16.2%；vs strawman TTFT降低77.1%-91.1%。

## Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Infera runtime kernel调度框架，实现tile级micro-kernel的动态选择、warp-level binary fusion和device-side daemon kernel launch。核心kernel调度实现：(1) SelectKernels：两阶段kernel selection——第一步选data block：DAG中选zero in-degree且最大化asynchronous wavefront G(u)的节点（G(u)递归定义为∑G(v)/(d⁻(v))+2/|Γ⁺(u)|-1，transitive dependency propagation使children asynchrony gain均匀传播到parents）；第二步选kernel：优化目标min #inst/IPC，约束TLP≥4。IPC estimation通过data hazard (stall cycles+running cycles)、structure hazard (instruction density+hardware bandwidth)和online-learned lightweight regression model；(2) FuseKernels：warp-level horizontal kernel fusion at CUDA binary level——统一function signature global void kernel(void* args)→prologue为每个thread恢复special registers和arguments→shared memory base+index addressing加offset→barrier BAR.SYNC重组→insert preemption/locking/progress flags。Fusion发生在SASS/CUDA binary code level，使用thread pool design最大化throughput；(3) LaunchKernel：多级pipeline——Host Kernel Queue (HKQ, priority queue按launch timestamp排序)→GDRCopy (gdr_copy_to_mapping, <100ns small payloads/<5µs typical kernel size)→Device Kernel Queue (DKQ)→driver-level placeholder kernel slot覆盖（避免cuModuleLoad的global host-device synchronization）→daemon kernel通过CDP fire-and-forget launch（无需等待previous grids completion，<10µs latency，避免HoL问题）→shared-memory double-ended queue低延迟kernel fetch→completion时cudaGetLastError错误检查。Preemption：host端暂停HKQ→DKQ+保存HKQ kernel、device端保存DKQ+shared memory kernels、in-flight kernel通过preemption flag终止。实验对比Stream、MPS、Triton、Paella serving系统。GPU stall分析显示Infera降低scoreboard和throttle stall cycles。

- 后端平台是什么，配置是什么。
  NVIDIA A100-PCIE-40GB GPU (Ampere, Compute Capability 8.0, peak bandwidth 1555 GB/s)。Intel Xeon Gold 6330 CPU, 512 GB RAM。CUDA 12.0, Linux 6.1.0。GDRCopy用于low-latency host-device data transfer。CUDA Dynamic Parallelism用于device-side kernel launch。daemon kernel独占一个SM。

- 评估性能的软件/脚本是什么。修改了什么。
  Infera inference server约17k LoC C++ kernel-space module，从零开发。Kernel fuser基于SASS/CUDA binary level warp-level horizontal fusion实现。daemon kernel使用CDP device-side fire-and-forget launch。GDRCopy用于HKQ→DKQ传输（gdr_copy_to_mapping bypass DMA engines）。Driver-level修改实现placeholder kernel slot覆盖（避免cuModuleLoad global同步）。GPU stall analysis通过类似Nsight Compute的profiling工具测量scoreboard/throttle stall cycles占比。Host overhead测量CPU load和host memory usage。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  未找到明确开源仓库（EuroSys 2026, DOI: 10.1145/3767295.3769392）。Runtime kernel调度使用流程：
  1. 编译阶段：Infera compiler为每个micro operator生成多个kernel版本（不同ILP/TLP/intensity trade-off），记录kernel metadata（#inst、register/shared memory usage、grid/block launch configuration）供runtime使用
  2. Scheduling cycle开始：TEU接收VTB（Virtual Task with Budget），VTB中每个task有instruction budget限制
  3. SelectKernels：构建data dependency DAG（节点标记为completed/pending/running）→对每个zero in-degree节点计算G(u)（asynchronous wavefront metric，递归展开到特定深度，terminal G(v)=-1）→选G(u)最高的data blocks作为candidates→对每个data block枚举所有可用micro-kernel版本→online regression model基于data/structure hazard分析估计IPC→选择min #inst/IPC且满足TLP≥4（通过register/shared memory使用量保证理论TLP，动态调整grid tile size增加occupancy）的kernel
  4. FuseKernels：将选定primitive kernels在CUDA binary level做warp-level horizontal fusion→统一function signature kernel(void* args)→args指向global memory包含所有原始kernel arguments和每个thread的special registers→prologue阶段每个thread恢复自己的%tid等special registers和arguments→对每个原kernel的shared memory访问指令添加offset（物理级base+index寻址，base在SASS中省略需运行时动态生成）→重组BAR.SYNC barrier资源（因thread组织变化）→insert preemption/locking/progress flags
  5. LaunchKernel：fused kernel入HKQ（priority queue，按launch timestamp排序，标记为on-device）→host launcher通过GDRCopy gdr_copy_to_mapping将kernel code以<5µs延迟拷到device kernel slot（覆盖placeholder kernel）→arguments拷到global memory→kernel pointer+arg pointer+launch config入DKQ→daemon kernel从shared-memory double-ended queue取kernel→cudaLaunchDevice fire-and-forget launch（<10µs）→GPU scheduler立即调度无需等待前序grids完成
  6. 例如BERT+ViT concurrent inference：TEU选择BERT MatMul tile micro-kernel和ViT Conv tile micro-kernel→warp-level fusion将两个kernel的warps交错排布在同一SM→daemon launch→SM内BERT warps和ViT warps空间共享GPU pipeline units（FP unit、memory bandwidth等）→降低scoreboard和throttle stall。Infera-P preemption约10μs（暂停HKQ→DKQ并保存），Infera-R preemption约5μs（仅保存不暂停），比REEF-N快约2.5×
