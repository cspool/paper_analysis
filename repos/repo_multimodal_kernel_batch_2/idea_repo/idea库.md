# idea库

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

## Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

- baseline方法是什么？
  - **Baseline 方法**：Data-parallel tile-based GEMM decomposition——将输出矩阵C划分为BLK_M×BLK_N的tile，每个CTA独立产生一个output tile。所有CTA的output tile合起来覆盖整个C矩阵。当output tile数量不是SM数量的整数倍时，最后一波（wave）CTA部分空闲，造成处理器利用率低于100%（量化低效，quantization inefficiency）。
  - **Baseline 全栈执行例子（以384×384×128 GEMM, BLK_M=128, BLK_N=128, 4-SM GPU为例）**：
    - 算法层：标准GEMM C = A × B，FP64或FP16→32精度。使用cache-blocked formulation（Algorithm 1），外层循环遍历output tiles，内层循环沿k轴累加。
    - 系统框架层：cuBLAS（vendor library）或CUTLASS（开源模板库）。cuBLAS为每种精度提供20+预编译kernel specialization，通过复杂heuristics或ML模型（ISAAC）选择kernel。CUTLASS提供data-parallel和fixed-split kernel模板。
    - 编译框架层：论文未明确说明。使用标准CUDA C++编译，无特殊compiler pass。
    - Kernel调度层：Data-parallel kernel（Algorithm 2+3）——grid有⌈m/BLK_M⌉×⌈n/BLK_N⌉个CTA，每个CTA执行⌈k/BLK_K⌉个MAC-loop迭代。GPU SM调度器以wave形式dispatch CTA。384×384×128问题：3×3=9个128×128 output tiles → 在4-SM GPU上需要3波（wave 1: 4 CTA, wave 2: 4 CTA, wave 3: 1 CTA + 3 SM idle）→ 利用率上限75%。
    - 硬件架构层：NVIDIA A100 GPU（108 SMs），每SM含Tensor Cores（FP64/FP16→32），HBM2e memory。CTA通过shared memory做两级blocking（global→shared→register），使用software pipelining隐藏global/shared memory延迟。

  - **Baseline的缺陷**：
    - **量化低效（Quantization Inefficiency）**：当output tile数不能被SM数整除时，最后部分wave造成SM空闲。随着GPU SM数增加（宽的处理器），wave数减少，量化低效更显著。
    - **ensemble维护复杂**：为解决量化低效，cuBLAS需要为每种精度提供20+ kernel specialization（不同blocking factors），加在一起形成hundreds of MB的可执行代码。还需要复杂heuristics或ML模型做kernel selection，且这些heuristics在任意问题上难以始终选择最优配置。
    - **Fixed-split的overhead**：CUTLASS的fixed-split decomposition通过沿k轴split output tile来增加CTA数改善量化效率，但fixup overheads（partial sum通信、同步、临时存储）同时随问题规模和splitting factor增长。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **方法**：Stream-K——将GEMM计算的workload量子化单元从output tile改为MAC-loop iteration（单个CTA-wide BLK_M×BLK_N×BLK_K MAC操作），沿m→n→k线性化均匀分配给g个CTA。每个CTA执行连续的MAC-loop迭代范围，可跨越output tile边界。由覆盖每个tile的k=0迭代的CTA负责累积其他CTA计算的partial sums并写出最终结果。
  - **解决量化低效**：MAC-loop iteration的workload volume（BLK_M×BLK_N×BLK_K次MAC）比整个output tile（⌈k/BLK_K⌉个MAC-loop iterations，通常32-512个）小32-512×。因此即使最后一个iteration也不能被完美均分，方差可忽略不计（≤1个MAC-loop iteration的差异）。384×384×128问题上g=4 CTA各得72个MAC iterations → 100%利用率。
  - **解决ensemble维护复杂**：单个kernel、单个tile size配置即可覆盖全部GEMM shapes。无需维护20+ kernel specialization、heuristics或ML模型。可执行代码体积减少约20×（single kernel per precision vs cuBLAS 20+ kernels）。
  - **解决Fixed-split overhead**：Stream-K的communication/synchronization/global storage overheads仅与CTA数g（≈处理器宽度p）成正比（O(p)），与问题规模无关。当tile数>CTA数时，每个tile最多被2个CTA覆盖（而非fixed-split中s个CTA可被split到每个tile），且tile-processing的时间偏移（skew）自然隐藏inter-CTA同步等待。
  - **论文方法全栈执行例子（以384×384×128 GEMM, BLK_M=128, BLK_N=128, BLK_K=4, 4-SM GPU为例）**：
    - 算法层：标准GEMM计算不变，但work decomposition策略从tile-centric变为iteration-centric。total_iters = 9 tiles × (128/4) iters_per_tile = 9×32 = 288。g=4 CTA各分配72个MAC iterations。
    - 系统框架层：集成于CUTLASS 2.11。用户调用标准GEMM API，内部Stream-K decomposition对用户透明。Grid size选择由基于解析模型的启发式自动完成（参数a/b/c/d一次微基准per architecture静态确定）。
    - 编译框架层：论文未明确说明。单个CUDA kernel模板，通过runtime参数（grid size g）而非compile-time specialization适配不同shapes。
    - Kernel调度层：Stream-K kernel（Algorithm 5）——g=4 CTA启动。CTA_0执行iter [0,72)（覆盖tile 0 iter 0-31, tile 1 iter 0-31, tile 2 iter 0-7），CTA_1执行iter [72,144)（tile 2 iter 8-31, tile 3 iter 0-31, tile 4 iter 0-15），以此类推。每个CTA跨越tile边界时：若不在k=0起始则写partials到temporary global storage；若覆盖k=0则等待其他CTA的partials、累积、写最终输出tile。100% SM利用率——4个CTA各在4个SM上全时段执行72个MAC iterations。
    - 硬件架构层：同baseline（NVIDIA A100）。额外使用global memory的temporary partial sum storage和inter-CTA synchronization flags（atomic-based Signal/Wait），但这些overheads仅与CTA数量（4）成正比，而非tile数（9）或k维大小（128）。

  - **混合调度（Hybridization）增强**：基本Stream-K可能因tile-processing skew（各CTA的起始k-offset不同）影响cache reuse。引入"two-tile Stream-K + data-parallel"混合——仅对最后部分data-parallel wave的剩余tile做iteration balancing，确保每个output tile最多被2个（而非更多）CTA覆盖，改善cache locality和synchronization latency hiding。

## SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

- baseline方法是什么？
  - **Baseline 方法**：FlashAttention2 / xformers 使用 FP16 精度在 GPU 上通过 tiling + online softmax 做 exact attention，以及 SageAttention2 使用 INT8 per-block quantization 加速 attention。这些工作在推理中都受限于 FP16/INT8 Tensor Core 的理论吞吐上限（RTX5090 上 FP16 ≈ 200 TOPS，INT8 ≈ 800 TOPS）。
  - **Baseline 全栈执行例子（推理）**：
    - 算法层：FlashAttention2 的 FP16 QK^T → online softmax → FP16 PV，或 SageAttention2 的 INT8 QK^T → online softmax → INT8 PV（per-block 量化 + smoothing K/Q）
    - 系统框架层：论文未明确说明（plug-and-play 替换 attention 实现）
    - 编译框架层：论文未明确说明
    - Kernel 调度层：FlashAttention2 CUDA kernel 使用 warp-specialized tiling，FP16 MMA 指令；SageAttention2 使用 INT8 MMA + per-thread INT4 PV
    - 硬件架构层：RTX5090 Blackwell GPU，FP16 Tensor Core ~200 TOPS，INT8 Tensor Core ~800 TOPS。FP4 Tensor Core 达 ~1600 TOPS 但在 baseline 中未被利用
  - **Baseline 全栈执行例子（训练）**：
    - 算法层：FlashAttention2 的 FP16/BF16 前向 QK^T → softmax → PV，反向 dV = P^T dO, dP = dO V^T, dS = softmax_backward, dQ = dS K, dK = dS^T Q，全 FP16
    - 系统框架层：论文未明确说明
    - 编译框架层：论文未明确说明
    - Kernel 调度层：FlashAttention2 CUDA kernel forward+backward in FP16
    - 硬件架构层：RTX4090，FP16 Tensor Core

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **方法**：SageAttention3 提出两项创新——(1) FP4 microscaling attention 用于推理加速；(2) INT8 可训练 attention (SageBwd) 用于训练加速。
  - **解决 C1（FP4 值域限制）**：使用 NVFP4 microscaling quantization（1×16 块粒度，E2M1+E4M3 scale），相比 per-tensor/per-token 量化有效抑制 outlier 影响。选择 NVFP4 而非 MXFP4，因为 NVFP4 的 1×16 块大小和 E4M3 scale 在 attention 量化中精度更高。
  - **解决 C2（P 的 scale factor 精度损失）**：提出 two-level quantization——先 per-token 归一化 P 到 [0, 448×6]（level 1，在 FP32 中无损），再做 FP4 microscaling 量化（level 2），使 scale factor s_P 充分利用 E4M3 的 127 个有效表示值（vs 直接量化的 35 个），显著降低量化误差。
  - **解决 C3（训练中梯度敏感）**：识别反向 5 个 MatMul 中 dOV^T 的精度最关键（其误差在 FlashAttention 循环中沿序列长度累积到 dQ/dK），保持 dOV^T 在 FP16，其他 4 个 MatMul 量化到 INT8。选择 INT8 而非 FP8，因 INT8 在反向梯度精度更高且硬件支持更广泛。
  - **论文方法全栈执行例子（推理）**：
    - 算法层：FP4 microscaling QK^T（NVFP4, 1×16 块）→ Smoothing Q/K → online softmax → two-level FP4 quantization of P → FP4 microscaling PV
    - 系统框架层：论文未明确说明（plug-and-play 替换 existing attention）
    - 编译框架层：论文未明确说明
    - Kernel 调度层：CUTLASS+CUDA 实现，包含三项硬件优化：K permutation（对齐 FP4MMA accumulator 与 operand 寄存器布局）、Reuse shuffle（softmax 与 P 量化共享 rowmax）、Producer warp epilogue（双 producer warp ping-pong 实现 MatMul 与 global store 的 overlap）
    - 硬件架构层：RTX5090 Blackwell FP4 Tensor Core，NVFP4 FP4MMA 指令 ≈ 1600 TOPS，实测达 1038 TOPS

  - **论文方法全栈执行例子（训练）**：
    - 算法层：前向 INT8 per-block QK^T + per-token P + per-block V；反向保持 dOV^T 在 FP16，其余 dS K、dS^T Q、P^T dO 量化到 INT8 per-block
    - 系统框架层：论文未明确说明
    - 编译框架层：论文未明确说明
    - Kernel 调度层：OpenAI Triton 实现 forward+backward INT8 attention kernel
    - 硬件架构层：RTX4090 INT8 Tensor Core，前向 2× 加速，反向 1.2~1.6× 加速

## Modulated Diffusion (MoDiff): Accelerating Generative Modeling with Modulated Quantization

- baseline方法是什么？
  Baseline是两种独立的扩散模型加速方法：(1) Caching方法（如DeepCache）——利用扩散过程相邻时间步之间特征的相似性，每隔N步缓存一次high-level features并复用，跳过中间步的重计算。缺陷：重用历史计算结果引入approximation error，且该误差在迭代过程中不断累积（Figure 1a显示relative ℓ₂ distance在最终step达到40%），需要careful design of reuse schedules甚至retraining来弥补；(2) Post-Training Quantization (PTQ)方法（如Q-Diffusion、LCQ）——在训练无关的前提下估计scaling factor将网络参数量化到低位宽整数。缺陷：扩散模型中activation tensor范围在不同时间步之间变化显著，且每步内存在异常值（outliers with long-tailed distributions，Figure 1b），使得低bit activation量化时scaling factor难以同时最小化clipping error和rounding error——现有方法只能将activation quantize到8-bit，更低bit精度（<6-bit）质量急剧塌陷。

  全栈执行例子（以DDIM on CIFAR-10, T=100 denoising steps, Q-Diffusion W8A4 baseline为例）：
  - 算法层：DDIM采样过程的每一步中，U-Net的前向pass被量化为W8A4精度。每层线性operator A^{(l)}：量化激活 → Q(a_t^{(l)}) → 矩阵乘法 A^{(l)}(Q(a_t^{(l)})) → 输出 ô_t^{(l)}。不同时间步的计算相互独立，无法利用时序冗余。当activation降到4-bit时，activation range在时间步间大幅变化导致per-channel scale s无法覆盖全范围——一部分值被clip，另一部分值被round到过粗粒度。FID从W8A8的4.21快速恶化到W8A4的24.09（LCQ）甚至332.75（Q-Diff）。
  - 系统框架层：PyTorch + DeepSpeed inference（评估GBops而非实际加速）。Q-Diffusion通过MSE reconstruction loss校准量化参数。动态量化在每步运行时计算scale factor。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明——未实现实际GPU kernel加速，仅用DeepSpeed计算理论Bops。
  - 硬件架构层：论文未明确说明硬件平台——明确表示"Implementing acceleration on specialized hardware is beyond the scope of this work"。

  Baseline两大核心缺陷：
  1. **Caching methods' accumulated error**: 直接复用历史激活（approximate without error tracking）导致误差在时间维度上累积——每一步的偏差传到下一步后被放大，误差随时间步数线性以上增长。
  2. **PTQ methods' activation range diversity**: 不同时间步的activation分布不同（大的range variation + outliers），static/deep calibration得到的scaling factor无法在所有时间步都有效。低bit时clipping error或rounding error必定有一个占主导。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Modulated Diffusion (MoDiff)，通过**两个协同的数学设计**同时解决上述两个缺陷：

  **(1) Modulated Quantization**——解决"Activation range diversity"缺陷：
  重构扩散采样过程中每层线性算子的计算范式，从直接量化activation（基线：Q(a_t)）转为量化时序差分（MoDiff：Q(a_t - a_{t+1})）。时序差分 a_t - a_{t+1} 的数学性质：(a) 分布范围比原始activation小10×以上（Figure 1b：橙色vs蓝色violin plot高度对比）；(b) 分布更集中、异常值更少（Figure 1b：橙色violin plot宽度更窄、尾部更短）；(c) 在不同时间步之间范围更一致。根据Theorem 4.3，量化误差正比于输入范围的平方——范围缩小10× → 量化误差降低100× → 可用低3-4 bit达到同等精度。这是从数学上通过输入变换（subtraction of adjacent time steps）改变被量化量的统计特性，而非优化量化器本身——因此MoDiff与具体量化方法（Q-Diff/LCQ/LTQ）正交，可直接叠加。

  **(2) Error-Compensated Modulation**——解决"Caching/Modulation积累误差"缺陷：
  标准调制方法（Eq. 24-25）直接累加量化输出：ô_t = A(Q(a_t - a_{t+1})) + ô_{t+1}。但ô_{t+1}本身含有量化误差，叠加新的量化误差后误差传递方式为指数增长（Theorem 4.4, Eq. 30: Σ 2^{T-k-1} c ∥A∥² ∥a_k-a_{k+1}∥²）。MoDiff引入中间变量 â_t = Q(a_t - â_{t+1}) + â_{t+1}（Eq. 13），使当前步的量化误差 e_t = a_t - â_t 被显式追踪并通过输入反馈到下一时间步（因为下一时间步的差分基不再是 a_{t+1} 而是 â_{t+1} = a_{t+1} - e_{t+1}）。这使误差从"累积"变为"被吸收"——Theorem 4.4, Eq. 31证明误差以 (2c)^{T-k-1} 速率指数衰减（当 c<1/2 时），而非增长。直观理解：每一步的量化误差被显式记入误差追踪量ê，并在下一时间步通过输入被算子处理（补偿计算遗漏的A(ê_{t+1})）。

  全栈执行对比baseline（以DDIM on CIFAR-10, T=100 denoising steps, LCQ+MoDiff W8A4为例）：
  - 算法层：同U-Net架构，但每层linear operator的执行变为MoDiff范式：
    1. Step T（第一次迭代）：â_T = Q(a_T), ô_T = A(â_T)，使用全精度activation的量化计算（warm-up, 4-5步渐进收敛）
    2. Step T-1→1（后续迭代）：每层执行 â_t = Q(a_t - â_{t+1}) + â_{t+1} → ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}。时序差分 a_t - â_{t+1} 的range远小于 a_t 原始range → 4-bit量化误差大幅降低。误差 e_t = a_t - â_t 被自动追踪到 â_t 中，在t-1步作为输入被补偿。
    W8A4时LCQ+MoDiff FID保持4.38（vs LCQ baseline的24.09或33.97），甚至W8A3时FID=4.14（vs baseline的143.39/90.34）。10× computation savings (W8A3 154 GBops vs FP32 1636 GBops)。
  - 系统框架层：PyTorch + DeepSpeed Bops评估。Calibration phase: Q-Diff+MoDiff使用reconstructed calibration dataset（捕捉时序差分而非原始激活）。Bias removal（所有被MoDiff改造的层去除bias项保证线性）。Layer-wise reconstruction（逐步独立优化量化参数）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明——未实现实际GPU kernel加速。
  - 硬件架构层：论文未明确说明硬件平台。MoDiff引入的额外开销：(a) 1次matrix addition（a_t - â_{t+1}）；(b) 1次output addition（+ ô_{t+1}）；(c) 1次额外的dequantization of Q(a_t - â_{t+1})。但matrix multiplication占主导（dominant cost），这些额外操作与matrix multiplication相比overhead negligible。Memory overhead：需额外存储 â_t 和 ô_t 每层——单张图CIFAR-10上W8A4额外内存~4MB（Table 6: 36.4 MB vs baseline 35.09 MB W8A32）。

  设计思路核心：
  论文的核心洞察是：**乘法算子A的线性性质允许将"直接计算A(a_t)"转化为"增量计算A(a_t - a_{t+1}) + ô_{t+1}"，而时序差分a_t - a_{t+1}比原始激活a_t更易于量化的本质是其在统计上具有更小range和更少outliers**。这两个数学事实的结合催生了MoDiff——将扩散模型的加速从"优化量化器"（传统的PTQ做法）和"跳过计算"（传统的caching做法）两个独立方向，统一为"改变被量化量的统计特征来提高量化效率"的单一视角。而误差补偿则通过显式追踪和反馈量化误差，将caching的accumulated error问题转化为数学上被指数衰减控制的残余误差。Remark 4.1证明了caching方法是MoDiff在0-bit差分时的特例（当时序差分低于阈值时），使得MoDiff在概念上统一了caching和量化两个方向。该框架对具体量化方法（Q-Diff/LCQ/LTQ）、模型架构（U-Net/DiT/LDM）、采样器（DDIM/DDPM/DPM/PLMS）和数据分辨率（32×32到256×256到MS-COCO）都具有普适性。

## ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

- baseline方法是什么？
  现有多GPU AI kernel开发存在三类baseline方法，各有缺陷：
  
  (1) **算子特定手写kernel**（Flux, CUTLASS distributed GEMM, Comet, FlashDMoE, Ring Attention等）：针对特定operator（GEMM、attention、MoE）手工实现compute-communication overlap，使用host-triggered copy engine配合device kernel stream-level overlap，或高度优化的on-device scheduler。缺陷：实现复杂、缺乏可复用抽象（如FlashDMoE仅支持TF32精度，BF16/FP16支持在发布5个月后仍在开发中）。
  
  (2) **编译器方法**（Triton Distributed, TileLink）：扩展Triton DSL支持OpenSHMEM风格的单边操作，编译器自动生成多GPU kernel。缺陷：缺乏显式的负载分布控制（warp/SM specialization），无法实现最优overlap；跨硬件平台泛化差（Triton Distributed原为H800调优，在H100上有时慢于非overlap基线）。
  
  (3) **通信库方法**（NCCL+NCCLX, NVSHMEM）：NCCL使用host-initiated copy engine在独立CUDA stream上与compute kernel做stream-level overlap。缺陷：
  - NCCL强制双向同步（sender和receiver必须相互确认才能传输），即使点对点通信也如此，细粒度通信开销显著；
  - NCCL使用小型预分配中间缓冲区（communication channels），引入额外数据搬运；
  - NVSHMEM每次remote peer access执行__ldg获取peer地址并强制__syncthreads，导致高达4.5×的element-wise NVLink访问延迟；
  - 不支持TMA和in-network acceleration，仅使用copy engine或寄存器操作。
  
  全栈执行例子（以8×H100上cuBLAS GEMM + NCCL all-gather的数据并行前向为例）：
  - 算法层：数据并行下的AG+GEMM+Activation+GEMM+RS流程。输入按行分片，权重按列分片。AG收集完整输入 → GEMM1 → 激活 → GEMM2 → RS分散结果。
  - 系统框架层：PyTorch Distributed + Megatron-LM风格的parallelization。使用torchrun多进程管理（每GPU一进程）。
  - 编译框架层：Triton分布式或直接CUDA kernel。baseline使用独立kernel launch：NCCL all-gather → cuBLAS GEMM → NCCL reduce-scatter，每个操作是独立的kernel或stream。
  - kernel调度层：NCCL使用host-initiated copy engine在单独CUDA stream上执行all-gather/reduce-scatter，与cuBLAS GEMM在不同stream上做stream-level overlap。Host端管理同步（cudaStreamSynchronize或cudaEvent），无device-side tile级overlap。copy engine需要至少256MB消息才能饱和带宽；对小矩阵（N=2048），Triton Distributed和Flux可能慢于非overlap基线。
  - 硬件架构层：H100 GPU的copy engine独立于SM运作，但host发起、仅支持连续内存传输。NVSwitch fabric仅作为passive switch转发数据，未利用in-network加速能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ParallelKittens通过三大设计原则和8个核心原语+LCSC统一编程模板，系统化解决上述baseline的缺陷：
  
  **(A) 传输机制原则**：PK分析三种传输机制（Copy Engine、TMA、寄存器指令）的带宽利用率vs消息粒度的trade-off曲线，仅暴露每种功能最有效的机制：
  - TMA用于点对点通信：74%峰值利用率仅需2KB消息（vs copy engine需256MB），支持单线程异步启动，不增加寄存器压力，使intra-SM overlap成为可能。
  - 寄存器操作用于in-network reduction：虽然效率较低（70%利用率）需要更多SM占用，但提供了copy engine和TMA均不具备的NVSwitch in-network reduction功能（multimem.ld_reduce, multimem.red）。
  - 完全放弃host-initiated copy engine用于device-side通信叠加。
  
  **(B) 调度策略原则**：PK统一支持Intra-SM和Inter-SM两种overlap调度，通过LCSC模板自动选择最优策略：
  - Intra-SM overlapping：所有SM同时执行compute和communication，通过单线程TMA异步调用实现通信与tensor core计算并发。对GEMM+RS场景：T_comp_tile = 2mnK/R, T_comm_tile = s*m*n/B，当K ≥ sR/(2B)≈2197时通信完全隐藏。实测验证K=4096时通信占比<1%。适用于通信模式与计算模式对齐的场景。
  - Inter-SM overlapping：将部分SM专用于communication，其余SM全用于compute。关键优势：(i) 利用in-network reduction大幅减少通信量（GEMM+AR中T_comm降低约N倍）; (ii) 对于Ring Attention，通信SM将下一block的KV tensor批量传输到local HBM，避免remote L2 cache miss导致的重复传输。通过num_comm_sms参数运行时自动搜索最优SM分配。
  - 对比：intra-SM在GEMM+RS中比inter-SM快1.2×（更高compute利用、更低sync开销），inter-SM在GEMM+AR中通过in-network reduction实现3.62×提升。
  
  **(C) 设计开销消除**：
  - 使用预分配目标缓冲区实现单向TMA P2P传输，消除NCCL的双向同步和中间缓冲区overhead。纯all-reduce通信kernel由此获得1.79×加速。
  - 将peer地址保持在寄存器中并移除不必要的__syncthreads，使element-wise NVLink访问延迟降低4.5×，带宽利用率提升约20 GB/s。
  - PK的IPC utility通过VMM+POSIX fd机制透明处理多进程地址空间映射和multicast object创建，使kernel代码无需感知底层IPC复杂度。
  
  PK方法全栈执行例子（以8×H100上PK实现的fused GEMM+AR kernel为例）：
  - 算法层：同baseline的AG+GEMM+AR流程，但通过LCSC模板将GEMM和all-reduce融合为单个kernel。
  - 系统框架层：PK提供IPC和PyTorch utilities集成，使用VMM手动分配的multicast memory（通过cuMemCreate→cuMemExportToShareableHandle→Unix domain socket传输fd→cuMemImportFromShareableHandle→cuMulticastCreate流程），使各进程拥有local address和multicast address两个虚拟地址映射。
  - 编译框架层：PK本身是C++ embedded DSL编译为CUDA kernel，无额外编译层。LCSC模板编译时通过config struct确定SM数量、thread分配和warpgroup布局。
  - kernel调度层：Inter-SM overlapping模式。Compute SMs运行loader/consumer/storer流水线：(loader) TMA从local HBM异步加载A_tile和B_tile到SMEM → (consumer) warpgroup执行mma_AB累积C_accum → (storer) warpgroup::store写入local output，arrive信号量通知下一stage，signal原语原子加barrier通知communication SM。Communication SMs运行communicator：(wait) 等待所有compute SM的barrier达到NUM_DEVICES → __syncthreads → (all_reduce) 整个warp使用multimem.ld_reduce PTX指令从multicast memory读取各GPU的partial结果，通过NVSwitch in-network reduction直接归约，再写入multicast memory。每kernel的通信相关代码仅10行。
  - 硬件架构层：利用NVSwitch SHARP in-network reduction（multimem.red/multimem.ld_reduce PTX指令）将all-reduce的通信量从O(N) peer写入降为O(1)对multicast object的读取+归约。TMA单线程异步操作不占用tensor core，使所有SM的计算单元保持繁忙。

- baseline方法是什么？
  Baseline是已有的expert skipping方法（NAEE、MC-MoE、DiEP），这些方法最初为text-only LLMs设计（top-2 routing），论文中将其适配到MLLMs的top-k（k>2）setting。Baseline方法的核心决策方式：仅依赖per-layer内的局部routing probability π_i^{(l)}来判定expert去留。NAEE在top-2场景跳过top-2 expert若π_top-2 < β · π_top-1；MC-MoE在NAEE基础上增加attention-aware protection；DiEP联合考虑routing probability和expert similarity做可微分剪枝。所有baseline方法共享两个根本缺陷：(1) 忽略expert贡献在不同层之间的不对等性——浅层expert对最终输出的影响远大于深层expert，但跳过策略对所有层一视同仁；(2) 将text和vision token等同对待——未考虑不同模态token在FFN层中的行为差异（vision token在FFN前后的余弦相似度更高，说明FFN对vision token的更新幅度更小，vision expert冗余度更高）。

  全栈执行例子（以Kimi-VL-A3B-Instruct在8×H200上处理多模态推理请求为例，baseline NAEE方法，跳过67% experts）：
  - 算法层：标准MoE MLLM架构——Visual Encoder提取vision token → Projector对齐到text embedding空间 → LLM backbone（26层transformer，每层含self-attention + MoE FFN，64 experts/layer，k=6）。NAEE在每层FFN中：计算routing probability → 若sum_{u=i}^k π_top-u < β^{(l)} · sum_{v=1}^k π_top-v，跳过top-i到top-k experts。β^{(l)}通过genetic search在GQA上确定。所有层使用相同规则，不区分text/vision token。
  - 系统框架层：基于transformers库加载模型，PyTorch forward pass。router kernel执行routing logits → top-k selection → NAEE规则判断 → expert dispatch/gather。Baseline未对MoE层做特殊kernel优化（论文未明确说明baseline的具体kernel实现）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Baseline MoE dispatch/gather按NAEE规则跳过部分expert后，剩余active experts通过标准Group GEMM执行。NAEE的跳过决策逻辑在router kernel的top-k之后执行，仅增加少量比较操作。但由于baseline仅看局部π_i^{(l)}，在较高跳过率（>67%）时跳过过多浅层关键expert，造成严重准确率下降（83%跳过时avg accuracy从100%降至82.81%-88.32%）。
  - 硬件架构层：8×H200 GPU。Baseline与MoDES在相同跳过率下kernel wall-clock speedup几乎相同（<1%差异，因为skip操作本身的开销相似），但由于baseline的跳过决策质量差，无法在保持准确率的同时达到MLLMs所需的高跳过率。

  Baseline两大核心缺陷：
  1. **Layer-agnostic skipping**：不考虑层间expert贡献差异。Shallow layer experts对final output的影响大（error会在后续层被放大），deep layer experts影响小。Baseline对所有层使用同一类阈值规则，导致shallower layers中critical experts被不当跳过（error explosion），而deeper layers中redundant experts未充分跳过（opportunity waste）。
  2. **Modality-agnostic thresholding**：未区分text/vision token。经验证：vision token在FFN前后的余弦相似度更高（更接近1），即FFN对vision token的更新幅度小于text token。而且vision token的expert冗余度更高（降低k值对vision的性能影响小于text）。Baseline单一阈值要么对text过于激进（导致关键信息丢失），要么对vision过于保守（浪费计算资源）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出MoDES——第一个面向MoE MLLMs的训练无关expert skipping框架，通过三个协同组件解决baseline缺陷：

  **(1) GMLG（Globally-Modulated Local Gating）**——解决"Layer-agnostic skipping"缺陷：
  引入全局逐层重要性因子α^{(l)}，通过离线KL divergence校准量化每层expert对final output的贡献。Importance score s_i^{(l)} = α^{(l)} · π_i^{(l)}同时编码全局层重要性和局部token- expert匹配度。α^{(l)}大的浅层：s_i^{(l)}整体偏高 → 即使π_i^{(l)}较低也可能被保留；α^{(l)}小的深层：s_i^{(l)}整体偏低 → π_i^{(l)}中等也可能被跳过。Inference时无额外计算开销（α^{(l)}预计算）。

  **(2) DMT（Dual-Modality Thresholding）**——解决"Modality-agnostic thresholding"缺陷：
  为text token和vision token分别设置阈值τ_t和τ_v。基于验证：vision expert冗余度高 → τ_v < τ_t → vision token跳过更多expert。Modality-specific decision使text token保留关键专家保证质量，vision token更激进地跳过冗余专家节省计算。

  **(3) Frontier Search**——解决"Threshold optimization效率"问题：
  利用f（KL divergence，性能损失）和g（跳过比例，效率）关于(τ_t, τ_v)的单调性，用O(ND)的frontier search替代O(ND²)的exhaustive search。搜索时间~45×降低（从>2天降至<2小时），性能差异<0.01%（96.24% vs 96.25%）。

  全栈执行对比baseline（以Kimi-VL-A3B-Instruct在8×H200上处理同一多模态推理请求，MoDES方法，跳过83% experts）：
  - 算法层：相同MoE MLLM架构。每层MoE FFN中：router → softmax → 对每个top-k expert i计算s_i^{(l)} = α̃^{(l)} · π_i^{(l)} → 根据token类型选择阈值τ_t或τ_v → 跳过s_i^{(l)} < τ的expert → 仅active experts参与计算和aggregation。Vision token的τ_v < τ_t → vision token上的跳过率显著高于text token。Shallow层α̃^{(l)}大 → s_i整体高 → 跳过少；Deep层α̃^{(l)}小 → s_i整体低 → 跳过更多。MoDES在跳过83% experts时仍保持96.25%的平均准确率（vs baseline 82.81%-88.32%），在跳过88%时对Qwen3-VL-MoE-30B保持97.33%（vs baseline 86.66%）。
  - 系统框架层：基于transformers库加载模型 + 自定义CUDA extension。PyTorch forward pass中每个MoE层调用MoDES的custom kernel。Calibration和search离线完成于8×H200（20分钟至<4小时，depending on model）。
  - 编译框架层：论文未明确说明编译框架。
  - kernel调度层：custom CUDA kernels实现：(a) Router kernel内嵌DMT thresholding——branch-free masked comparison，跳过expert路由设为sentinel ID M+1；(b) Sentinel-aware dispatch/gather——自动过滤sentinel entries；(c) Group GEMM with offline-profiled tile sizes——根据动态expert activation pattern选择最优kernel配置。MoDES的额外开销：仅一次α̃^{(l)}乘法和一个masked comparison per expert（与baseline NAEE的β比较开销相当）。跳过率相同时speedup与baseline几乎相同（<1%差异），但MoDES可在高跳过率下保持准确率。
  - 硬件架构层：8×H200 GPU。Calibration每条数据需2次forward pass（original + layer-skipped），1024 samples over all layers。Frontier search对每对(τ_t, τ_v)需1次forward pass，O(ND)复杂度。Prefill speedup ~2.16×（batch=8, Kimi-VL），Decode speedup ~1.26×。Decode speedup较低原因：(i) decode为memory-bound，(ii) decode阶段仅处理text token，总体跳过率低于prefill。

  设计思路核心：MoDES的本质是将expert skipping从"只看局部（单层routing概率）"提升为"全局感知+模态感知"的决策。GMLG通过预校准的α^{(l)}将层重要性信息编码到重要性分数中，DMT通过双阈值区分text/vision token的不同特征。两个insight（层间贡献不均+模态行为差异）各对应一个设计组件，二者叠加产生非线性增益——在极高跳过率（>80%）下差距尤为显著（差距从~6%扩大到~10%）。同时Frontier Search保证了从"insight→设计→优化"全pipeline的效率可行性。

## Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs

- baseline方法是什么？
  baseline是kernel-per-operator LLM serving系统（SGLang、vLLM、PyTorch + CUDA Graphs + torch.compile），其执行方式为：将模型计算图（DAG of tensor operators）的每个节点作为独立GPU kernel launch。每个operator使用不同的专用kernel库：FlashInfer或FlashAttention处理attention、cuBLAS或cuTLASS处理MatMul、CUDA或Triton处理其余operators。GPU runtime在每个连续kernel launch之间隐式插入kernel barrier，强制前一个kernel的所有thread blocks完成后才能启动下一个kernel。CUDA Graphs用于减少kernel launch overhead但本质上是static capture，对dynamic batch size/shape变化需要重新实例化。

  全栈执行例子（以SGLang/vLLM在H100上执行Qwen3-8B单batch decode iteration为例）：
  - 算法层：标准Transformer decoder——Q/K/V projection (MatMul) → FlashAttention → O projection (MatMul) → RMSNorm → Gate/Up projection (MatMul) → SiLU → Down projection (MatMul) → RMSNorm → ... → LM Head (MatMul)。每个operator对应一个独立kernel。
  - 系统框架层：SGLang/vLLM通过PyTorch执行模型forward。CPU端执行：(1) continuous batching——从request queue取出batch，(2) page allocation——为KV cache分配物理页，(3) 逐operator dispatch kernel launch。CPU-GPU同步发生在每个iteration边界——CPU等待GPU完成上一步后才调度下一步。
  - 编译框架层：无统一编译框架。FlashInfer（Triton/CUDA手写attention kernel）、cuBLAS/cuTLASS（闭源GEMM库）、Triton（JIT编译element-wise/reduction ops）。各库互不感知，无法进行跨算子优化。
  - kernel调度层：每个kernel launch后GPU SM执行SPMD，kernel barrier强制所有SM完成当前kernel → CUDA runtime launch next kernel → 重复。关键瓶颈：(a) kernel launch overhead——每次decode iteration有数百个kernel launch（CUDA Graphs可降低但不能消除），(b) pipeline bubble——kernel barrier阻止跨算子pipelining，TMA/Tensor Cores/CUDA Cores在kernel边界产生pipeline bubble，(c) 粗粒度依赖——AllReduce必须等整个MatMul完成，即使每个AllReduce thread block只依赖一个MatMul thread block的输出，(d) CPU-side scheduling延迟——page allocation和request scheduling在CPU执行，每次iteration需要CPU→GPU dispatch round-trip。
  - 硬件架构层：NVIDIA H100 GPU（132 SMs, HBM 1.6TB/s）。TMA（Tensor Memory Accelerator）、Tensor Cores（989 TFLOPS FP16）、CUDA Cores（60 TFLOPS）三种异构计算单元。在kernel-per-operator模式下：单个MatMul kernel期间Tensor Cores工作但TMA在prefetch完成后闲置、CUDA Cores闲置；attention kernel期间情况类似；AllReduce期间NVLink/PCIe工作但SM闲置。硬件利用碎片化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Mirage Persistent Kernel (MPK)，第一个自动将多GPU模型推理编译为单个mega-kernel的compiler+runtime系统。核心理念是以SM-level graph representation (tGraph)替代kernel-level computation graph，从而暴露传统kernel barrier遮蔽的细粒度并行性。

  解决baseline缺陷的具体设计映射：

  **缺陷1: Kernel barrier阻止跨算子pipelining** → MPK方案：SM-level tGraph + Cross-task software pipelining
  - MPK将每个operator分解为SM级tasks，通过fine-grained event同步替代kernel barrier
  - Cross-task pipelining将每个task拆分为pre-loading phase和compute phase，当前task compute期间即启动下一task的pre-loading
  - Paged shared-memory abstraction让task按需acquire/release shared memory pages，使跨task数据prefetch成为可能
  - 效果：消除pipeline bubble，1.2-1.3x speedup（Qwen3-8B final linear layer on B200）

  **缺陷2: 粗粒度依赖阻止细粒度compute-communication overlap** → MPK方案：Fine-grained dependency analysis
  - MPK在task对级别而非operator级别分析数据依赖——仅当task t1的输出区域与task t2的输入区域重叠时才插入event
  - 例如MatMul→AllReduce：每个AllReduce task仅依赖一个对应MatMul task的输出，MPK通过per-task-pair events使compute和communication在不同SM上并发执行
  - 效果：1.1x speedup（Qwen3-1.7B on 4×H100 TP）

  **缺陷3: Kernel launch overhead + CPU-side scheduling延迟** → MPK方案：In-kernel parallel runtime
  - 整个模型编译为单个mega-kernel，仅一次kernel launch → 消除数百次launch/iteration
  - Page allocation和request scheduling逻辑全部嵌入mega-kernel内部作为单个task执行，消除CPU-GPU同步
  - Workers/schedulers使用circular buffer + atomicAdd实现轻量级task调度

  **缺陷4: 动态workload imbalance** → MPK方案：Hybrid JIT+AOT task launch
  - 对data-dependent执行时间的operator（如attention）使用JIT launch，runtime动态分配实现负载均衡
  - 对确定性的operator（如MatMul after barrier）使用AOT launch，预分配消除dispatch开销
  - MoE场景额外使用hybrid workload balancer：编译期静态分配expert tasks + 运行期根据router产生的global metadata动态调整

  **缺陷5: 分散的kernel库生态** → MPK方案：Compiler-based unified code generation
  - Mirage superoptimizer自动生成每个task的优化CUDA实现，无需手写不同库的kernel
  - NVSHMEM统一处理in-kernel inter-GPU通信
  - 用户只需 torch.compile(backend=MPK)，无需了解底层kernel实现

  全栈执行例子（以MPK在H100上执行Qwen3-8B单batch decode iteration为例，对比baseline）：
  - 算法层：与baseline相同模型架构。MPK不改变模型算法，仅改变执行方式。Attention仍使用paged attention算法，但attention tasks以JIT模式执行以适应sequence-length变异性。
  - 系统框架层：PyTorch + torch.compile(backend=MPK)。编译期：MPK compiler读入计算图 → operator decomposition (H100: ~132 tasks/MatMul) → dependency analysis (per-task-pair events) → event fusion (successor+predecessor) → tGraph normalization (fan-in/fan-out≤1) → tGraph linearization (BFS, 连续task索引) → task code generation (Mirage superoptimizer)。生成的mega-kernel为单个callable PyTorch function。执行期：单次kernel launch，in-kernel runtime持久运行直至所有decode iteration完成。Page allocation、request admission、KV-cache update均在mega-kernel内的start event task中执行——无CPU参与。
  - 编译框架层：MPK编译器完全自研。核心数据结构tGraph存储为GPU device memory compact格式：每个task 352 bytes（dependent_event index, trigger_event index, input/output tensor ptrs, config params）；每个event存储required trigger count + [first_task, last_task] index range。所有tasks和events以连续数组存储，enqueue/dequeue仅用atomicAdd。
  - kernel调度层（核心差异）：SM物理分区——128 workers (每SM一个) + 4 scheduler-SMs (16 warp-schedulers)。以Q_proj → K_proj → Attention → O_proj → RMSNorm → Gate/Up → SiLU → Down → RMSNorm为例：(a) Start event → scheduler dispatch所有Q/K/V projection tasks (AOT, 预分配到workers)，worker SM_i执行Q_proj: TMA preload weight tile → Tensor Core MMA (Q=X×W_Q) → 同时prefetch K_proj weight tile (cross-task pipelining) → 完成, notify event。(b) 所有Q/K/V tasks完成后event激活 → scheduler dispatch attention tasks (JIT, 因为attention执行时间data-dependent)。Worker SM_j执行attention task (JIT): 有long sequence的worker慢、short sequence的快 → 快的worker先完成先获得下游O_proj tasks (JIT) → dynamic load balance。(c) 所有attention tasks完成后barrier event → 后续MLP tasks全部AOT预分配，worker SM_k check AOT queue: event已激活? → execute Gate_proj GEMM + 同时prefetch Up_proj weight → SiLU → Down_proj GEMM + 同时prefetch下一层Q_proj weight → 流水线无缝衔接。(d) 跨全部层，TMA、Tensor Cores、CUDA Cores三种硬件持续饱和——任意时刻都有SM在进行计算、数据搬运或通信。
  - 硬件架构层：NVIDIA H100 GPU。与baseline key hardware utilization差异：Mega-kernel执行全程TMA持续prefetch（消除pipeline bubble），Tensor Cores几乎持续执行GEMM/Attention MMA（各SM间轮转），CUDA Cores持续执行element-wise ops和dequantization（若有），NVLink持续传输（fine-grained overlap with compute）。Kernel-per-operator的理论下限~10ms（纯粹模型参数加载时间），MPK达到12.5ms，只比理论下限高~25%。Baseline SGLang/vLLM为14.5ms，差距主要来自kernel launch overhead和pipeline bubble。

  总结：MPK通过将抽象层次从"kernel级"下沉到"SM级"（即用tGraph替代computation graph），暴露了传统GPU编程模型中被kernel barrier遮蔽的细粒度并行性，并通过compiler自动化和in-kernel runtime高效利用这些并行性，实现end-to-end 1.0-1.7x加速。

## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- baseline方法是什么？
  baseline是QServe [15]的W4A8 GEMM实现，使用QoQ（Quality over Quantity）dequantization算法。QoQ通过progressive quantization将INT8限制在[-119, 119]范围避免乘法溢出，再通过"先乘后减"策略（Q_u4 · s_i8 - s_i8 · z_i8）处理dequantization。但减法步骤依赖vadd伪指令——每条vadd被lowering为十余条底层指令——导致CUDA Cores上dequantization开销巨大（占warp stalls的21%）。baseline的GEMM执行采用简单的load→dequantize→MMA串行pipeline，dequantization无法被有效重叠，导致：(1) memory-bound场景下W4A8与W8A8性能相当（理论应更快）；(2) compute-bound场景下W4A8比W8A8慢2x（理论应相当）。

  全栈执行例子（以QServe W4A8 GEMM处理LLaMA2-7B FFN层，batch=256为例）：
  - 算法层：QServe两级量化——FP16→INT8 (per-channel, [-119,119]) → UINT4 (per-group, group_size=128)。激活SmoothQuant动态per-token量化FP16→INT8。
  - 系统框架层：QServe serving系统通过PyTorch调用QServe GEMM kernel。权重离线量化存储为UINT4，激活在线量化。KV cache 4-bit量化。
  - 编译框架层：未使用编译框架自动生成kernel。QServe hand-crafted GEMM kernel。
  - kernel调度层：QServe GEMM kernel执行简单pipeline——从GMEM加载UINT4 weight (LDG.32) → unpack 4-bit到8-bit → QoQ dequantization (CUDA Cores, vadd → dozen instructions) → WGMMA INT8 MMA (Tensor Cores)。dequantization与MMA串行，CUDA Cores成为瓶颈。LDS.32加载（非LDS.128）浪费一半带宽。
  - 硬件架构层：NVIDIA H800 GPU。CUDA Cores throughput远低于Tensor Cores（H100: 60 TFLOPS CUDA vs 990 TFLOPS TC INT8, 16.5x差距）。Tensor Cores在dequantization期间空闲，CUDA Cores在MMA期间空闲——两种硬件单元交替闲置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LiquidGEMM，包含两个co-designed技术解决W4A8 dequantization瓶颈：

  (1) **LiquidQuant (LQQ)量化算法**：通过rotation-based transformation将INT8先shift到UINT8域再量化为UINT4，利用two's complement同余性质（i ≡ j mod 2^8 → 相同binary representation）设计dequantization。dequantization仅需两条32-bit硬件指令（IMAD + XOR）处理四个元素，无overflow、无vadd伪指令开销。从QoQ的10+条指令/元素降至0.5条指令/元素，α=2（而非QServe的α≥10），远低于与memory load重叠所需的阈值α≤5.07。

  (2) **Implicit Fine-Grained Pipeline (ImFP)**：采用single-producer multiple-consumer模型替代ExCP的多WG显式同步。Load WG通过TMA加载weight到SMEM后切分为fine-grained tasks，多个Compute WG竞争获取task并各自完成dequantization+MMA，dequantization与MMA跨Compute WG自然重叠。消除ExCP的SMEM↔RF round-trip数据搬运和barrier同步开销。配合Dual-MMA packed layout让每个线程用单条LDS.128加载32个UINT4元素。

  全栈执行对比baseline（以LiquidGEMM处理同一LLaMA2-7B FFN层，batch=256为例）：
  - 算法层：LiquidQuant替代QoQ——FP16→INT8 (per-channel, [-119,119]) → shift到UINT8 → UINT4 (per-group, group_size=64)。Dequantization: Q_i8 = (Q_u4 * s_u8 + a) XOR 0x80，两条指令四元素。准确率通过WikiText2/zero-shot评估保持与QServe相当。
  - 系统框架层：LiquidServe自建serving系统。与QServe不同的是KV cache使用INT8（非4-bit），batch size可扩展到更大（如LLaMA2-70B batch=184 vs QServe batch=64），GEMM不再是瓶颈。
  - 编译框架层：基于CUTLASS/Cute构建，WGMMA/TMA/barrier用PTX包装，dequantization用CUDA直接实现。计算重构为Y=(WX^T)^T以利用WGMMA的m=64固定维度。
  - kernel调度层：ImFP替代串行pipeline。Load WG (TMA) → SMEM task queue → Compute WG_0 (LDS.128→unpack→IMAD+XOR dequantization→WGMMA MMA) 与 Compute WG_1 (同样pipeline, 不同task) 并发执行。CUDA Cores做dequantization期间Tensor Cores在另一Compute WG做MMA，反之亦然。无SMEM↔RF round-trip，无显式barrier同步（硬件task scheduling管理）。彻底消除CUDA Core瓶颈。
  - 硬件架构层：NVIDIA H800 GPU。TMA、CUDA Cores、Tensor Cores三种异构硬件通过ImFP实现pipeline-parallel执行——Weight loading (TMA) ∥ Dequantization (CUDA Cores, WG_0) ∥ MMA (Tensor Cores, WG_1)。从"交替闲置"变为"持续饱和"。2.90x kernel speedup vs QServe，4.94x system speedup。

## HipKittens: Fast and Furious AMD Kernels

- baseline方法是什么？
  baseline是AMD GPU上现有的高性能AI kernel开发方式：(1) AITER——AMD工程师手写汇编kernel，性能峰值高但开发困难、难以扩展到新的workload；(2) Composable Kernel (CK)——深度嵌套C++模板库，使用复杂；(3) PyTorch SDPA/torch.compile——编译器自动生成的kernel，性能远低于峰值（如Llama GQA backwards仅259 TFLOPS vs 峰值）；(4) ROCm Triton——将NVIDIA Triton移植到AMD，但寄存器生命周期追踪差、无法将访存lower到最优化指令（buffer load非默认），性能受限；(5) NVIDIA wave specialization模式直接移植到AMD——由于AMD静态寄存器分配，producer wave消耗寄存器但不参与计算，限制了output tile size和arithmetic intensity，在MI355X上仅达80%峰值BF16 GEMM性能。
  全栈执行例子（以GQA backwards为例）：
  - 算法层：Transformer模型执行GQA backward pass，需要计算dQ、dK、dV梯度。
  - 系统框架层：PyTorch调用SDPA backend或AITER的flash_attn_func。AITER仅支持有限的attention形状组合。
  - 编译框架层：Triton编译器将Python DSL lowering到LLVM IR，但无法精确控制AGPR寄存器使用，产生冗余v_accvgpr_read指令。
  - kernel调度层：NVIDIA wave specialization模式下，producer wave独占寄存器但只做memory搬运，AMD上512 registers/SIMD被静态分割为256 VGPR+256 AGPR，HIPCC不允许AGPR作为MFMA输入操作数，导致需插入v_accvgpr_read搬移数据。
  - 硬件架构层：AMD MI355X 8 XCD chiplet架构，naive row-major grid schedule仅达36% L2 hit rate，L2和LLC缓存复用未协同优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出HipKittens——一套最小化的C++ embedded tile-based编程原语，将ThunderKittens的tile抽象移植到AMD并重新设计底层实现：
  (1) **Pinned register tiles**：绕过HIPCC编译器，让开发者直接指定tile到物理VGPR/AGPR寄存器的映射，允许AGPR作为MFMA输入操作数，避免冗余v_accvgpr_read指令。MHA non-causal backwards从855 TFLOPS提升至1024 TFLOPS，匹配AITER汇编kernel。
  (2) **8-WAVE PING-PONG调度**：每SIMD 2个wave交替执行compute（MFMA）和memory（buffer_load/ds_read）角色，通过conditional barrier切换。解决wave specialization中producer wave浪费寄存器的问题，利用AMD更大的register file（2× NVIDIA）和更细粒度的MFMA指令（16×16×32）建立deep pipeline。
  (3) **4-WAVE INTERLEAVE调度**：每SIMD 1个wave精细交错发射compute和memory指令，用于compute/memory不平衡workload（如attention backwards），达到2.3× speedup vs baseline。
  (4) **Chiplet swizzle算法（Algorithm 1）**：通过XCD grouping（chunks of C blocks分配给同一XCD）和hierarchical windowed traversal（W高度垂直窗口遍历输出矩阵），联合优化L2和LLC hit rate，提升19%性能。
  (5) **AMD矩阵布局管理**：针对AMD各MFMA指令形状使用完全不同的thread-to-element mapping（无NVIDIA的16×16统一core matrix结构），HK自动处理不同指令的shared memory bank和phase ordering差异，提供bank-conflict-free的swizzle pattern。

  全栈执行对比baseline（以GQA attention backward为例）：
  - 算法层：同一Transformer GQA backward计算，HK kernel支持任意head dim（64/128）、causal/non-causal。
  - 系统框架层：HK通过Python bindings集成到PyTorch，替换SDPA backend，用户调用方式不变。
  - 编译框架层：HK不依赖编译器自动lowering，而是通过C++ template直接生成HIP/assembly指令，同时使用LLVM sched_barrier/sched_group_barrier hints指导编译器在cluster级别调度。
  - kernel调度层：8-WAVE PING-PONG下，每SIMD的两个wave交替：wave A发射MFMA（使用AK寄存器tile和BV寄存器tile），wave B同时发射buffer_load_dword从HBM预取下一tile到LDS。compute cluster内部通过sched_barrier_pairs交错MFMA和softmax vector ops（exp2/sub/max）。使用pinned AGPR作为MFMA的A/B输入，消除v_accvgpr_read开销。
  - 硬件架构层：Algorithm 1在launch前remap block坐标，使共享L2的XCD内block覆盖矩形输出区域（提升L2 hit rate至78-79%），同时跨XCD协调访问重叠的A/B行/列区域（提升LLC hit rate至55-93%），联合优化effective bandwidth。

## GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

- baseline方法是什么？
  baseline 是"将 GPU 视为黑盒、禁止并发 kernel 执行"的安全关键实时系统 GPU 管理方法。具体而言，GPU-using 程序需要 lock 整个 GPU 或 lock 单个 EE/CE（如 GPUSync [8]、RGEM [10]、TimeGraph [11]），即同一时刻只允许一个 kernel 在 GPU 上执行，禁止来自不同 stream/process 的 kernel 并发。在全栈执行例子中：
  - 算法层：一个图像处理 pipeline 包含多个 CUDA kernel（边缘检测 → 特征提取 → 目标识别），每个 kernel 串行执行。
  - 系统框架层：CPU task/process 通过单个 default stream 或手动串行化提交 GPU operations。
  - 编译框架层：论文未明确说明。
  - kernel调度层：GPU scheduler 内部机制未知，程序员仅知"同 stream 内 FIFO、不同 stream 间 may run concurrently"。为避免未知的并发干扰，实际做法是禁止多 stream 并发或 lock GPU。
  - 硬件架构层：NVIDIA TX2 的 2 个 SM 和 1 个 CE 的并行能力被浪费——即使有可用 SM 资源和 CE 带宽，也无法让多个 kernel 或 kernel+copy 同时执行。

  baseline 缺陷：(1) GPU 计算资源利用率低——未利用 SM 并行和 CE/EE 并发；(2) 无法进行实时可调度性分析（real-time schedulability analysis），因为调度行为未知；(3) 文档中未说明 NULL stream 和多 priority stream 的精确交互行为。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是通过系统化黑盒实验（合成 benchmark + GPU 端时间戳测量）和公开文档，逆向工程出 NVIDIA TX2 GPU 调度器在 task 共享地址空间下的完整调度规则（G1-G4, X1, R1-R3, C1-C4），以及 NULL stream (N1-N2) 和 stream priority (A1-A2) 的扩展规则。核心发现是：TX2 GPU scheduler 采用 **层次化 FIFO 调度**（hierarchical FIFO scheduling），虽然不完全是 work-conserving（存在 blocking delays），但具有可预测的 FIFO 特性，具备进行实时可调度性分析的可能。

  全栈执行对比 baseline：
  - 算法层：同一图像处理 pipeline 的多个 kernel 可以通过多 stream 实现并发。例如 K1（边缘检测，6 blocks × 768 threads）和 K4（特征提取，4 blocks × 256 threads × 32KB shared mem）可以同时在不同 SM 上执行，前提是 K4 在 EE queue 中排在 K1 之后、且资源（threads/shared_mem）允许。Kernel 执行期间，copy engine 可并发执行 copy 操作（如 K2 的输出 copy C2o），实现计算与数据传输重叠。
  - 系统框架层：CPU task 共享地址空间下，每个 task 可使用多个 user-specified stream 提交 GPU operations。论文定义了 stream queue → EE queue → SM assignment 的精确流转规则。避免使用 NULL stream（Rule N2 会阻塞其他 stream 的 kernel 入 EE queue），谨慎使用 stream priority（priority-high 可抢占 priority-low，可能导致饥饿）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文给出精确规则：
    - G1-G4: operations 从 stream queue → EE queue → dispatch → dequeue 的生命周期
    - X1: 非抢占——只有 EE queue 头部 kernel 的 block 可被分配
    - R1-R3: thread（≤2048/SM）、shared memory（≤64KB/SM）、register（≤65536/SM）资源约束下的 eligibility 判定
    - C1-C4: CE queue 的 FIFO 调度和 stream queue 解除阻塞规则
    - N1-N2: NULL stream 的同步语义——需等待其他 stream 的头部 kernel 先于自己 launch 的全部完成后才能入 EE queue
    - A1-A2: 两个 EE queue（priority-high 和 priority-low），高优先级 EE queue 非空时低优先级 block 不可分配
  - 硬件架构层：基于已知规则，GPU 的 2 SM 和 1 CE 可被安全地充分利用，实现 kernel-kernel 并发和 kernel-copy 并发。对 process 独立地址空间场景（附录 A），论文发现 TX2 使用 Pascal 架构的指令级 preemption 实现 time-slicing 多路复用，block 时间可能翻倍，且 stream priority 在多 process 场景下无效——因此推荐 task 共享地址空间模型。

## HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

- baseline方法是什么？
  Baseline是三种互不协作的GPU并发方案：(1) 仅kernel-level并发（CKE within single application）——如Taskflow要求用户显式指定kernel依赖并手动分配到stream，GrSched引入Python DSL的runtime动态调度但开销大且性能差；(2) 仅task-level并发（space-sharing across applications）——如CASE在multi-GPU系统上编译期静态分析资源需求并runtime动态调度task到GPU，但不做kernel内部并发优化，也不做memory management；(3) 手动编程的async并发——开发者手动使用CUDA stream/event API编写并发代码，需要大量专业知识和编码劳动。
  
  全栈执行例子（以CASE为例，运行W4 workload）：
  - 算法层：multi-kernel GPU程序（如M2包含多个activation和reduction kernel）以串行方式执行，每个kernel独占所有SM资源，但SM occupancy仅<10%（memory-intensive kernel的典型特征）。
  - 系统框架层：CASE runtime分析task资源需求，将task分配到有足够内存的GPU。但task内部kernel串行执行，无法利用kernel间并发。
  - 编译框架层：CASE在LLVM编译期分析资源需求，但不做DA分析也不修改kernel执行流。无memory management。
  - kernel调度层：每task内的kernel在同一default stream上串行执行，GPU硬件利用率低。仅靠不同task的kernel通过MPS space-sharing在同一GPU上并发，但内存容量限制了可并发task数。
  - 硬件架构层：4×A100 GPU，每GPU 40GB HBM。W4 workload中大量M1/M2/B&S应用（内存>8GB），仅能2个task同时运行。SM有大量idle周期。

  Baseline缺陷：(1) 仅kernel-level或仅task-level并发无法充分饱和GPU资源——即使GPU utilization=100%，SM occupancy仍可能<10%（memory-intensive kernel场景）；(2) 内存容量是task并发的瓶颈，不解决内存占用就无法增加并发task数；(3) 手动编程负担大——Taskflow需显式声明依赖（平均27 LoC），GrSched需重构到Python DSL（平均127 LoC），Async需手动管理stream/event（平均40 LoC）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出HuntKTm——一个结合kernel-level和task-level混合调度与自动内存管理的LLVM编译框架，包含三个协同组件：
  (1) **Stream Scheduler**：编译期自动分析kernel间RAW/WAR/WAW数据依赖，构建DFG后通过PP-Set启发式算法分配kernel到多stream实现kernel-level并发，并通过依赖传递性和串行隐式同步剪除冗余barrier。解决"需手动编写CKE代码"问题——仅需每kernel一行writable参数标注（平均15 LoC vs Async的40 LoC）。
  (2) **Task Scheduler**：编译期resource analyzer +运行时lazy engine收集task资源需求（threads/registers/shared memory/memory），task dispatcher基于三维度SM可用量评估选择最优GPU。解决"多task自动放置和负载均衡"问题——将task从编译期的资源分析到运行时的精确调度无缝衔接。
  (3) **Memory Manager**：在stream graph上执行memory object liveness分析，通过Algorithm 2推迟allocation到live range起点、提前free，将memory object lifetime缩短至live range，使非重叠lifetime的memory object可复用同一内存区域。解决"内存容量是并发瓶颈"问题——平均减少22.3% peak memory（M2从17.6GB→11.2GB，减少36.4%）。

  全栈执行对比baseline（以W4 workload运行M2为例）：
  - 算法层：同一M2应用（包含FasterTransformer的activation和reduction kernel），kernel间天然依赖可通过DFG自动发现后并行化。
  - 系统框架层：stream scheduler自动构建M2 DFG（宽度=6），分配kernel到6个stream并发执行，同步剪除后仅保留最少barrier。task scheduler同时将多个task（含其他应用的multi-stream版本）动态分发到4×A100 GPU上。
  - 编译框架层：LLVM pass自动转换M2源码：DFG constructor→kernel distributor→synchronization generator→memory manager（liveness分析+延迟allocation）→resource analyzer（nvcc获取register/shared memory）→function wrapper（注入lazy engine拦截逻辑）。
  - kernel调度层：M2的kernel从串行执行变为6-stream并发，同时与其他task（B&S、IMG等）的kernel通过MPS在同一GPU上space-sharing。lazy engine在task调度确定GPU后，顺序执行deferred CUDA操作（cudaMallocAsync→cudaMemcpyAsync→kernel launch），memory pool减少频繁alloc/free开销。
  - 硬件架构层：4×A100 GPU，HuntKTm在W4 workload下FP32 utilization提升3.54x、memory bandwidth utilization提升2.83x、SM occupancy提升2.47x（vs SA）。Memory management使system memory从232.3GB降至173.9GB，更多task可同时运行。最终HuntKTm比CASE throughput提升33.2%。

## KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

- baseline方法是什么？
  Baseline是**手动kernel开发**（manual kernel development）——由领域专家为每个算子-硬件平台组合手写优化kernel。在Meta的生产环境中，面临三维度的诅咒（Curse of Dimensionality）：模型架构多样性（≥1500个模型，从MLP到Transformer-based ranking）、kernel原语多样性（200+ data preprocessing operators + compute kernels）、硬件代际和架构异构性（NVIDIA Ampere/Hopper, AMD CDNA3/CDNA4, MTIA v2i/v3）。这种组合爆炸导致O(operators × hardware platforms)的实现矩阵，每个平台特定实现需要2-8周的专家优化工作，且12-18个月的硬件更新周期使已有优化失效。

  此外，baseline还包括现有的AI-powered kernel coding research prototypes（AutoTriton, KernelLLM, GEAK-agent, Kevin, KernelAgent, TritorX, AlphaEvolve），它们存在六个根本缺陷：(1) 窄优化范围——target isolated subproblems，无end-to-end lifecycle management；(2) synthetic evaluation——canonical operators with static shapes，非production dynamic batching/variable sequence lengths/domain-specific transformations；(3) 单平台focus——homogeneous NVIDIA environments；(4) limited agent capabilities——缺乏fully autonomous workflows（multi-level verification + hierarchical profiling + persistent knowledge bases）；(5) 无inference-time scaling——无大规模搜索策略（greedy/MCTS/evolutionary）；(6) 无checkpointing——失败后从零开始。

  全栈执行例子（以NVIDIA H100上conv1d为例，手动开发流程）：
  - 算法层：Convolutional Transformer模型需要1D convolution over user event sequences（production shape: B=2048, Cin=96, Cout=96, L=200）。
  - 系统框架层：PyTorch torch.nn.functional.conv1d直接执行，内部调用cuDNN implicit GEMM，但需要NCHW↔NHWC layout转换（5次kernel launch含多次format conversion）。或者用conv2d workaround——reshape到4D + channels_last → cuDNN NHWC optimized path（4次kernel launch）。
  - 编译框架层：cuDNN/Triton compiler执行固定的compilation passes——静态tiling heuristics、generic autotuning templates——无awareness of production shape distribution。
  - kernel调度层：多个独立kernel launches（layout transform + GEMM + post-processing），每个kernel独立从HBM读取输入、写入输出，中间tensor通过HBM传递（redundant global memory traffic）。
  - 硬件架构层：NVIDIA H100 TMA和warp specialization能力未被利用——kernel使用传统warp-centric模式，无async copy、无double-buffering、无differentiated cache modifiers。

  Baseline两大核心缺陷：
  (1) **Preprocessing kernel缺失导致disaggregated serving architecture**：当关键preprocessing operators缺少native accelerator实现时，生产系统被迫采用disaggregated topology——preprocessing在CPU server执行，neural network在accelerator执行——引入10-20ms pure network overhead（P99 latency从61ms增至97ms，25% degradation），消耗sub-100ms latency budget的显著部分。这不是增量性能损失，而是binary deployment constraint——单个缺失operator block整个模型在accelerator上的monolithic deployment。
  (2) **手动开发无法scale到组合爆炸空间**：O(operators × hardware platforms × model stages)的实现矩阵，每个平台2-8周专家开发时间，新硬件12-18个月更新周期导致已有优化失效。手动开发的经济成本和组织负担使完整kernel coverage成为不可能——直接限制模型创新和部署速度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出KernelEvolve——一个将kernel优化建模为**graph-based search with LLM agents**的自动化框架，通过四个关键设计解决baseline缺陷：

  **(1) 图搜索驱动的自动化kernel优化（Graph-Based Search & State Machine）**
  将kernel优化formalize为图搜索问题 G_t = (V_t, E_t)，specified by (F, π_sel, O, τ)。Fitness function F(v) = t_pytorch / t_triton；Selection policy π_sel支持greedy/MCTS/evolutionary三种策略；Universal Operator O 统一所有优化操作（Draft/Debug/Improve）为单个context-aware transformation；Termination rule τ基于时间/max artifacts/stall/fitness threshold。这替代了手动trial-and-error开发过程——将"人工数周探索"转化为"agent数小时搜索"（如conv1d 300 steps搜索自动发现fusion + tiling + autotuning + 3D grid + double-buffering + cache modifiers组合）。

  **(2) Universal Operator + Retrieval-Augmented Dynamic Prompting（Section 3.2）**
  用**单个通用算子**替代传统多算子框架（Draft/Debug/Improve各自有固定静态prompt template）。通过retrieval-augmented dynamic prompting机制，在每次迭代时基于实际runtime context（profiling结果、错误诊断、硬件约束）动态合成prompt——而非通过预定义的operator lens限制LLM推理。具体架构：
  - Context Memory Sub-Agent：分析runtime artifacts（kernel源码、profiling metrics、error diagnostics）→ 诊断bottleneck（如30% occupancy + high shared memory pressure → register spilling root cause）→ 生成优化指令
  - Deep Search Sub-Agent：根据诊断结果参数化知识库检索 → 检索目标hardware/optimization文档 → 注入LLM context window
  - Dynamic Prompt Synthesis：组合当前kernel + analysis report + retrieved knowledge + hardware constraints → 在token budget (64K-1M)内维持task-relevant information
  - Persistent Knowledge Base：hierarchical file system组织 → constraints/（correctness） + guidance/（platform-agnostic optimization） + hardware/{nvidia|amd|mtia}/（≥100 documents per platform）→ 通过index.md实现structured navigation

  **(3) MTIA Knowledge Injection for Proprietary Accelerators（Section 3.2.3）**
  针对MTIA（proprietary architecture absent from LLM pretraining corpora）的特殊挑战：系统化地将MTIA domain expertise注入持久化知识库。包括：libdevice API映射（SFU LUT operations如exp/gelu/sigmoid映射到专用硬件指令而非数学近似）、hardware-specific compilation options（cb_multiplier扩大circular buffer、use_dual_core分配DMA到core A+vector到core B实现pipeline parallelism）、compute helper functions（unary_elemwise_compute/binary_elemwise_compute/binary_elemwise_const_compute mapping到optimized vector instructions）、custom type system（TensorView/CoreID/ExecutionGrid via @core.struct_type decorator）、advanced synchronization（cross-PE broadcasting via tl.load direction attribute、cross-PE reduction via tl.store、runtime barriers via tl.pe_runtime_barrier、explicit tensor copies via tl.copy）。当LLM收到MTIA-targeted query时，retrieved documentation effectively teaches the model MTIA-specific idioms absent from pretraining——从"generate GPU-targeted Triton code that fails on MTIA"转变为"generate production-grade MTIA kernels leveraging hardware-specific features"。

  **(4) Multi-Granularity Evaluation & Profiling Integration（Section 3.4）**
  建立完整的多层次evaluation pipeline：TritonBench（correctness验证 + speedup测量）→ Torch Profiler（system-level timeline: CPU/GPU time, launch overhead）→ NCU（kernel-level: occupancy, memory throughput, instruction mix）→ Triton Proton/Triton MPP（intra-kernel: instruction-level pipeline behavior, async overlap）→ MTIA Insight（MTIA-specific: PE utilization, SFU/DPE/MLU utilization, cache hit rates, per-PE counters）。关键创新：Triton MPP作为unified profiling abstraction——通过compiler-centric job graph组合instrumentation、profiling passes、trace synthesis——解决"性能信号分散在DSL/compiler IR/CUDA/PTX/SASS/runtime/hardware counters多个abstraction layer"的fragmentation问题。Evaluation Code Generator（deterministic）自动将kernel artifact转换为instrumented evaluation scripts，通过FaaS platform dispatch到remote hardware worker——消除generation (CPU-bound) 和 evaluation (accelerator-bound) 之间的resource contention。

  **(5) Persistent Storage + Checkpointing for Scalable Search（Section 3.2.2）**
  Metadata store（关系数据库） + Object store（kernel文件）的两层存储架构支持：(a) distributed concurrent exploration——多个agent同时扩展不同节点，transaction isolation保证consistency；(b) complex contextual queries——通过SQL/recursive CTE实现graph traversal（如查找sibling outcomes、ancestor strategies、global best）；(c) cross-session knowledge reuse——历史optimized kernels作为新搜索的初始化（如在AMD MI350上类似GEMM变体：识别15个历史GEMM kernel、找到3个>1.5×speedup版本、以此为基础开始search）；(d) fault tolerance and checkpointing——persist每步搜索状态，crash后从last successful iteration恢复。这解决了"multi-hour optimization runs brittle and resource-inefficient"问题。

  全栈执行对比baseline（以NVIDIA H100上conv1d为例，KernelEvolve自动化流程）：
  - 算法层：同一conv1d计算，KernelEvolve在300步搜索中自动发现最优tiling和fusion策略组合。
  - 系统框架层：不再调用PyTorch conv1d/conv2d → cuDNN路径，而是使用KernelEvolve-generated fused Triton kernel，通过TritonBench BenchmarkOperator wrapper集成到模型inference pipeline。
  - 编译框架层：KernelEvolve的Universal Operator替代了固定compilation passes——通过retrieval-augmented prompting在每个search step动态调整优化策略，而非应用static tiling heuristics。Triton compiler接收已优化的Triton源码进行final compilation。
  - kernel调度层：从5次独立kernel launch（layout transform × 2 + GEMM + layout transform + post-process）→ 2次kernel launch（pack_conv1d_weight_kernel + conv1d_gemm_kernel）。跨operation fusion消除冗余layout转换和intermediate tensor materialization。使用3D grid launch并行化grouped convolution channels。使用double-buffered execution overlap memory access with Tensor Core operations。使用differentiated cache modifiers（.ca for streaming activations, .cg for reused weights）。
  - 硬件架构层：20+ autotune configurations探索BLOCK_M/N/K + num_warps + num_stages + WARP_SPECIALIZE组合空间。知识库驱动检索NVIDIA H100-specific文档（tensor cores + TMA + warp specialization + Hopper pipeline patterns）。Fitness score从~2000收敛到6889（300 steps），最终2.30× speedup vs PyTorch conv1d, 1.62× vs conv2d workaround。

  跨平台扩展（MTIA v3 conv1d）：
  - KernelEvolve的同一operator specification自动生成MTIA-specific kernel，知识库注入使LLM学习MTIA-specific idioms（libdevice API for SFU activation、cross-PE broadcasting for multi-PE kernels、cb_multiplier/use_dual_core for pipeline optimization），达到6.54× speedup vs PyTorch conv1d baseline——证明automated synthesis在vendor library coverage不成熟的custom accelerators上价值最大。
  - MapIdTransform在MTIA v2i上实现3.28-4.07× speedup——不仅优化性能，更是**唯一可行的on-device执行路径**（PyTorch baseline因缺少native ATen ops需CPU fallback）。实现从"missing kernel → disaggregated serving → 10-20ms network overhead"到"generated kernel → monolithic accelerator deployment → zero network tax"的架构转变。

## Kitsune: Enabling Dataflow Execution on GPUs

- baseline方法是什么？
  Baseline是GPU上两种现有的DL执行范式：(1) Bulk-Synchronous Programming (BSP)——每个DL算子映射为单个CUDA kernel，kernel独占GPU运行直到所有CTA完成后才launch下一个kernel，通过global barrier串行化执行；(2) Vertical Fusion（垂直融合）——将多个DL算子融合为单个"mega kernel"，在单个CTA内temporal multiplexing各算子的部分执行，通过shared memory/register file传递tile级中间数据。商业代表：TensorRT，学术代表：Welder、AStitch、Chimera。

  Baseline三大缺陷（对应Kitsune的三个untapped opportunities）：
  1. **资源闲置**：垂直融合的temporal multiplexing导致任一时间点只有TensorCore或SIMT core之一活跃，另一资源空闲。实测数据显示inference中20-25%、training中37-67%的runtime中SM和DRAM利用率均低于33%峰值。
  2. **大intermediate spilling到DRAM**：当intermediate的hidden dimension较大时（如MLP hidden dim ≥ 768 on A100），即使垂直融合的tile也超过shared memory capacity（192 KB/SM），不得不spill到off-chip DRAM。A100 round-trip DRAM latency ≈ 409ns (572 cycles @ 1.4GHz)。若通过多CTA/SM来增加并行度会进一步分割shared memory，加剧容量问题。
  3. **无法利用reduction/hidden维度并行**：Back-propagation中gradient reduction over batch dimension（split-K GEMM类似）产生少量CTA执行reduction，绝大多数SM空闲。垂直融合不支持back-propagation。

  全栈执行例子（以BSP执行MLP forward pass：Linear(768→3072) → ReLU → Linear(3072→768)为例）：
  - 算法层：三个DL算子顺序执行，每个算子的kernel独占所有SM资源。
  - 系统框架层：PyTorch eager mode依次dispatch kernel_Linear1 → kernel_barrier → kernel_ReLU → kernel_barrier → kernel_Linear2。每次barrier后中间结果由DRAM写入/读出。
  - 编译框架层：无融合优化，或TensorRT将ReLU作为Linear1的epilogue融合（垂直融合），但两个Linear因intermediate过大（3072×batch×4B ≈ 大tile）spill到DRAM。
  - kernel调度层：单个kernel独占GPU时，TensorCore执行GEMM期间SIMT core空闲，反之亦然。108 SM的A100上仅一种类型资源被利用。
  - 硬件架构层：intermediate数据在DRAM↔L2↔SM的shared memory间反复搬运，消耗大量DRAM bandwidth。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Kitsune——通过两个互补的SW/HW原语在GPU上实现synchronous dataflow执行：
  **(1) 软件Ring Buffer Queue（§4.1）**：纯软件实现的inter-CTA通信队列，通过L2 cache + global atomics在多CTA间传递tile级数据。Queue为双buffer设计，使用sequence number实现producer/consumer同步，metadata由atomic操作保护。Queue数据保持在L2 cache中（通过CUDA API pinning），避免了DRAM round-trip。
  **(2) Modified GPU Grid Scheduler（§4.2）**：将GPU的单round-robin arbiter扩展为双arbiter（SIMT和Tensor各一个），通过cudaPipeline API标注每个kernel的primary resource type，使grid scheduler能将不同类型的CTA配对到同一SM，实现fine-grained co-execution。
  基于这两个原语，Kitsune编译器将DL计算图自动lowering到spatial pipeline——不同算子映射到不同CTA，通过on-chip queues传递tile级中间数据，不同类型CTA在SM上co-locate，实现空间上的并发执行（而非时间上的multiplexing）。

  对应解决Baseline三大缺陷：
  1. **解决资源闲置**：Modified grid scheduler使SIMT-heavy CTA（如ReLU）和TensorCore-heavy CTA（如Linear GEMM）配对到同一SM，TensorCore执行矩阵乘的同时SIMT core执行elementwise。Kitsune将runtime中"both low utilization"从26%降至15%（inference），44%降至18%（training）。
  2. **解决大intermediate spilling**：Dataflow下，大hidden dimension被split到多个CTA并行处理，每个CTA仅需容纳一个tile（64-256KB），无需在shared memory中存储整个intermediate。Tiles通过on-chip queue直接传递，无需经DRAM。DRAM traffic减少41-98%（inference）、16-42%（training）。
  3. **解决reduction并行不足**：Dataflow下reduction操作通过queue构建reduction tree（多对一通信），将并行度从单CTA扩展到多CTA/SM。Back-propagation中的gradient reduction成为并行reduction pipeline stage。

  全栈执行对比baseline（以Kitsune执行同一MLP forward pass：Linear(768→3072) → ReLU → Linear(3072→768)为例）：
  - 算法层：同一MLP计算，Kitsune将其spatial fusion为一个sf-node（包含3个stage的spatial pipeline）。
  - 系统框架层：PyTorch Dynamo → Kitsune compiler backend → 生成cudaPipeline（3 kernels + 2 queues）。cudaPipeline启动后，3个kernel的CTAs co-resident在GPU上，不再需要中间barrier。
  - 编译框架层：Kitsune compiler执行：Subgraph Selection（模式匹配识别Linear→ReLU→Linear chain）→ Pipeline Design（在Linear1/ReLU间插入queue0，ReLU/Linear2间插入queue1）→ Load Balance（ILP求解：Linear1分配64 Tensor CTAs，ReLU分配44 SIMT CTAs，Linear2分配44 Tensor CTAs，利用SIMT/Tensor的重叠将152 CTAs压缩到108 SM预算内）。
  - kernel调度层：CUDA kernel改写——Linear1 kernel原来`store C[id]`到global memory → 改为`wr_acquire(queue0, tile_id)` → 写入queue → `wr_release(queue0)`。ReLU kernel原来从global memory读取 → 改为从queue0 `rd_acquire/release`获取tile。同一SM上，Linear1 CTA使用TensorCore执行GEMM的同时，ReLU CTA使用SIMT core执行elementwise。
  - 硬件架构层：Intermediate tile数据流：Linear1_CTA → L2-resident queue0 → ReLU_CTA → queue1 → Linear2_CTA。全程无DRAM访问。Modified grid scheduler的双arbiter机制确保每个SM同时有Tensor和SIMT CTA。当2× L2 bandwidth和2× SM时，Kitsune额外获得47%（inference）和27%（training）加速比，而baseline仅18-26%。

  设计思路核心：Kitsune的本质是将DL计算图中operator间的**时间串行**（temporal BSP/vertical fusion multiplexing）转换为**空间并行**（spatial dataflow pipeline）——仅需两个最小化原语（软件queue + 修改grid scheduler），无需clean-slate架构。这证明在现有GPU架构上的"modest adjustments"即可实现高效的dataflow执行。

## Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

- baseline方法是什么？
  Baseline是现有top-k sparse attention方法（Quest、Double Sparsity (DS)、H2O等），在LLM decoding阶段使用固定budget B选择top-k个"critical tokens"参与attention计算，以节省KV cache访问带宽。Baseline全栈执行例子（以Quest在A100上decode step为例）：

  - 算法层：Quest使用per-page (16 tokens/page) max-pooling估计token重要性——对K cache做max_pool得到K_pooled → 计算q @ K_pooled^T得approximate scores → 选择top-k pages（k=B/16）作为critical tokens的索引集合I。

  - 系统框架层：Quest kernel (CUDA/Triton) 直接替换PyTorch attention实现，可集成至vLLM/SGLang等serving框架。

  - 编译框架层：论文未明确说明。

  - Kernel调度层：Quest kernel执行q与pooled K的GEMV → topk selection → sparse FlashAttention仅对|I|=B个token计算精确attention。固定B值导致：(a) 对focused attention heads（权重集中于少数token），B过大→over-selection，加载和计算了不需要的token；(b) 对diffuse attention heads（权重均匀分布），B过小→under-selection，丢失重要context信息。从图2可见Quest/DS在不同budget(256→8192)下perplexity变化剧烈，说明B是高度敏感的hyperparameter。

  - 硬件架构层：NVIDIA A100 GPU。固定budget下，无论head是否需要，均加载相同数量的KV cache token，导致memory bandwidth浪费（over-selection heads）或accuracy损失（under-selection heads）。

  Baseline核心缺陷：**固定的token budget B无法适应不同attention head、不同layer、不同query、不同prompt下attention weight分布的动态性**。根本原因：top-k关心"选多少个"（|I|=B），但attention的数学目标是"累积足够的attention weight"（ΣW[i]≥p）。在focused分布下很小的B就能满足ΣW[i]≥p（B再大就是浪费），在diffuse分布下很大的B仍不够（精度不足）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Twilight，核心洞察：**将LLM text generation中的top-p (nucleus) sampling引入sparse attention**。top-p的本质是从"固定数量"（k个token）转为"固定累积概率"（p=Σattention weights），天然适应不同attention weight分布。

  **(1) Top-p Sparse Attention**——解决"固定budget无法适应分布动态性"缺陷：
  从数学上，attention近似误差界为‖o - ô‖ ≤ (1 - Σ_{i∈I} W[i]) · ‖V‖_F（Eq.2），因此最优策略是最小化|I|使ΣW[i]≥p——这正是top-p的定义（Definition 3.3）。对focused attention（peaked distribution），top-p自动选择很少的token（如B=97 for p=0.8，图4）；对diffuse attention（flat distribution），top-p自动选择更多token。B由分布决定而非预先固定。

  **(2) Hierarchical Select-then-Prune Architecture**——解决"如何将top-p应用于任意现有算法"问题：
  不是重新设计sparse attention算法，而是作为现有算法的**optimizer**。两层架构：(a) Token Selector——将现有算法（Quest/DS等）作为黑盒，使用保守的大budget B0≈N/4预选token子集（保证高recall，避免遗漏重要token）；(b) Twilight Pruner——在子集上用INT4 SpGEMV估计attention weights，然后top-p binary search精筛到B1<<B0。这种hierarchical design使Twilight可适配任何top-k sparse attention算法。

  **(3) INT4 K Cache + Efficient Kernels**——解决"top-p的精度要求高于top-k"的系统开销：
  top-k只需要序数正确（ordinality），top-p需要数值准确性（numerical accuracy）。实验发现4-bit是sweet spot（图6：2-bit累积attention weights显著下降，4/8-bit稳定）。基于FlashInfer实现：(a) INT4 SpGEMV——cp.async + 2-stage pipeline, memory access降至1/4；(b) Top-p binary search——O(log(range/ε))次并行reduction，避免O(N log N)排序；(c) Head-wise varlen attention——flatten head dim + load balancing处理不同head的不同budget。

  全栈执行对比baseline（以Quest-Twi, LLaMA-3.1-8B-Instruct, decode step, 32k context为例）：

  - 算法层：Twilight叠加于Quest之上。Token Selector: Quest用保守budget B0=8192（1/4 sparsity）预选token → Twilight Pruner: INT4 SpGEMV估计attention weights → softmax归一化 → top-p binary search (p=0.95) 精筛到B1≈446 → Sparse Attention仅对B1 token计算精确attention。从"固定8192个token"变为"自适应~446个token"，累积概率仍≥95%。

  - 系统框架层：基于FlashInfer构建的kernel library。支持PagedAttention，可集成至vLLM/SGLang。额外INT4 K cache内存开销为1/8 FP16 KV cache（可复用base algorithm已有的INT4 K cache）。

  - 编译框架层：论文未明确说明编译框架。使用CUDA/Triton直接编写kernel。

  - Kernel调度层：三阶段pipeline——(a) Quest Token Selector (SpGEMV on FP16 K cache, ~15% time)；(b) Twilight Pruner: INT4 SpGEMV + Softmax + Top-p Binary Search (~20% time)；(c) Sparse Attention: 仅对B1 token做FlashAttention (~65% time)。GQA下每query group取各head选择token的union后flatten head dim做load balancing。Quest-Twi vs Quest: 1.4× self-attention speedup, 1.35× end-to-end decoding speedup。up to 15.8× vs FlashAttention2。

  - 硬件架构层：单张A100 GPU。与baseline相比：Pruner引入额外INT4 SpGEMV开销，但Sparse Attention的token数从8192降至~446（18×减少），总时间减少（Quest 8192 budget vs Quest-Twi auto ~446 budget）。

  设计思路核心：论文的key insight是类比LLM token sampling中的top-k vs top-p问题。在text generation中，top-p (nucleus sampling) 替代top-k解决了"不同下一个词分布需要不同k"的问题；同理，在sparse attention中top-p替代top-k解决了"不同attention weight分布需要不同budget"的问题。Twilight的创新不在于提出全新的token selection策略，而在于认识到**budget selection本身应该是算法的一部分——且应该由sum of attention weights而非固定count来决定**。Hierarchical architecture使得这个insight可以作为任何现有top-k方法的"drop-in optimizer"叠加使用，而非重新设计整个sparse attention pipeline。

## Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs（来自BLAST repository Lee et al. 2024和Monarch repository Dao et al. 2022）。这些baseline实现虽然在理论上减少了FLOP和模型大小（2×-3×压缩），但在多token推理场景下性能反而退化——Monarch比dense慢1.14-1.68×，BLAST慢2.63-4.31×。退化根因是BLR的block结构产生了dense baseline中不存在的中间张量（Monarch: b×n×r, BLAST: 2个b×n×r中间量），这些中间数据通过global memory传递，加上block维度排列(b₂↔b₁, r'↔b₂)在contiguous维度上的uncoalesced access，将原本compute-bound的dense线性层推入memory-bound区域（roofline α从高于breakpoint降到低于breakpoint）。

  全栈执行例子（以Llama-7B QKVproj层Monarch baseline在A40上，n=1024, i=o=4096, r=1024, b=16为例）：
  - 算法层：Monarch BLR——dense权重W[4096×4096]替换为16×16块的块低秩分解，每块rank r'=64，参数从16.8M降至4.2M（4×压缩），FLOP从34.4G降至17.2G（2×减少）。但在n=1024多token时，b=16块结构产生中间张量Z[16×1024×1024]=64M个元素（BF16: 128MB），而dense线性层中间量为0。
  - 系统框架层：PyTorch eager mode dispatch Monarch forward的多个kernel：X_blocks reshape → bmm(X, V^T) → permutation kernel 1 (r'↔b₂) → permutation kernel 2 (b₂↔b₁) → bmm(Z_perm, U) → final permutation (b₂,n,q)→(n,q,b₂)。每个kernel都有独立launch overhead，中间张量全部通过global memory传递。
  - 编译框架层：torch.compile()尝试fuse操作但受限于BLR block结构——bmm和permutation的复杂index manipulation使compiler难以生成fused kernel，尤其permutation on innermost dimension导致uncoalesced memory access pattern，torch.compile()无法通过layout推导消除。
  - kernel调度层：A40上6MB L2 cache无法容纳128MB中间张量，导致频繁DRAM spill。Permutation kernel的uncoalesced loads使DRAM bandwidth利用率远低于峰值（696 GB/s）。bmm kernel本身是compute-bound但被permutation memory traffic拖累。
  - 硬件架构层：NVIDIA A40 GPU（6MB L2 cache），中间张量远大于L2容量→每步permutation都是DRAM round-trip。Jetson Orin Nano上更严重：DDR bandwidth仅68 GB/s，且L2仅4-6MB。

  Baseline核心缺陷总结：
  1. **中间数据移动**：BLR block结构产生的b×n×r中间张量在global memory中多次往返，dense baseline无此开销。
  2. **排列开销**：Monarch和BLAST的block维度重排需要独立的kernel launch，且排列在contiguous（innermost）维度上造成uncoalesced memory access。
  3. **编译器局限性**：即使torch.compile()也难以自动fuse bmm+permutation和优化BLR-specific memory layout。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过**硬件感知的Triton kernel设计**解决BLR多token推理的性能退化。核心策略：partial fusion（非全融合）、operation reordering、tailored memory layouts。不对BLR算法本身做任何修改（保持相同准确率），仅在kernel实现层面消除冗余数据移动。

  **Monarch优化①②③（联合使用）：**
  - ① V重排布（offline）：将V存储从contiguous along b₂ then r'改为r' first then b₂，消除推理时r'↔b₂ permutation kernel → 离线一次性操作消除运行时开销
  - ② 排列融合（kernel级）：将b₂↔b₁ permutation与第一个bmm融合为单个Triton kernel。每个thread block计算输出tile时直接计算目标b₂索引和调整后的r'偏移，用swapped indices写出 → 消除一个完整kernel launch + 一次global memory中间张量往返
  - ③ 避免最终permutation（offline）：当Monarch线性层输出被静态权重消费时，pre-permute该权重rows → 消除推理时的(b₂,n,q)→(n,q,b₂) kernel

  **BLAST优化⑤（推荐，TPC utilization高）：**
  - ⑤ 仅排列融合+Tensor Core：转置S和U为S^T/U^T从左侧乘，在每个kernel内transpose中间输出tiles。n保持contiguous，r/b₁/b₂依次作为三个kernel的batch维度 → 零permutation kernel launch，所有bmm保持tensor core执行（via Triton dot()），permutation开销完全吸收到bmm内部

  **BLAST优化④（备选）：**
  - ④ bmm部分融合：每个thread block内循环b₁维度计算S-weighted累加，消除V→S之间的中间permutation和第一个bmm输出的global memory物化。但第二个bmm用CUDA cores batched outer product → 牺牲tensor core 16×吞吐量优势，仅适用于小rank或极端memory-bound场景

  全栈执行对比baseline（以Llama-7B QKVproj层Monarch ①②③优化在A40上，同n=1024）：
  - 算法层：权重参数化和FLOP与baseline完全相同（Monarch BLR, 4.2M参数, 17.2G FLOP）——算法层面无变化，精度不变。
  - 系统框架层：从6-7个PyTorch kernel launches缩减为2-3个Triton kernel launches（fused perm+bmm → bmm → optional final perm/pre-permuted downstream matmul）。kernel launch overhead减少60-70%。
  - 编译框架层：不使用torch.compile()做高层fusion——直接用Triton编写fused kernel，对BLR-specific dataflow有完全控制。Triton compiler负责lower-level优化（shared memory allocation, warp scheduling, memory coalescing）。
  - kernel调度层：A40上：
    - Kernel 1 (fused perm+bmm): grid=(b₁×ceil(n/t_n)×ceil(r/t_r)), t_n=64, t_r=128, t_p=64。X和V tiles从global memory coalesced加载到shared memory → tensor core dot() → 直接写入permuted output layout。**消除了baseline中的两个独立permutation kernel和中间128MB张量的global memory往返**。
    - Kernel 2 (second bmm with U): 标准batched bmm with tensor cores。
    - (可选Kernel 3已消除): final permutation → 若接residual则仍需此kernel（论文承认无法避免），但大多数QKVproj→attention路径可通过pre-permute attention weight避免。
  - 硬件架构层：NVIDIA A40 GPU。优化后arithmetic intensity从memory-bound区域回升。关键数据流：X_tile [L2→shared] + V_tile [L2→shared] → tensor core MMA → result直接写入permuted layout [shared→L2]——无单独permutation kernel的数据移动。A40 6MB L2中仅需容纳正在处理的tile（64×256 BF16 ≈ 32KB），远小于baseline的128MB中间张量。

  **效果量化：**
  - Monarch ①②③综合：1.46-2.37× layer-wise speedup over Monarch baseline
  - BLAST ⑤：DiT-XL/2 QKVproj up to 7.15× over BLAST baseline on Jetson
  - BLAST ⑤ end-to-end：1.13-1.48× over dense baseline across models（注意：这是相对于dense的加速，不是相对BLR baseline）
  - 关键tradeoff：BLAST ⑤（tensor core）> BLAST ④（CUDA core）in >90% cases，因为tensor core 16× throughput优势远超消除permutation开销的收益
  - BLAST ⑤ > BLAST ④ 的例外：仅在极端memory-bound且b极小的场景（论文中④的GPT2-S on Jetson表现优于某些情况）

  设计思路核心：论文证明BLR压缩的"理论FLOP减少≠实际加速"的gap可以完全由软件/系统层面的kernel优化填补——**不改变压缩算法、不牺牲精度、不依赖新型硬件**，仅通过partial fusion、operation reordering和tailored memory layout三个策略，在现有GPU上实现BLR的理论加速变为实际加速。关键洞察是BLR的额外中间数据移动（而非额外计算）是瓶颈，且PyTorch compiler的通用优化无法处理BLR-specific的permutation-bmm交织pattern，需要手工Triton kernel设计来直接控制tile-level数据流。

## Marconi: Prefix Caching for the Era of Hybrid LLMs

- baseline方法是什么？
  Baseline是extended SGLang/vLLM prefix caching方案，针对Hybrid LLMs（Attention+SSM混合架构）采用**fine-grained checkpointing**策略。由于SSM层使用in-place recurrent state更新（无法像Attention KV cache那样通过切片回滚到任意前缀位置），baseline采用naive方案：每隔固定x个token保存一次SSM layer的完整recurrent state作为checkpoint，使用标准LRU eviction管理缓存容量。如页面25-28所示，该方案存在两个致命缺陷：
  - Catch 1 — cache entries are sparsely-hit：大量checkpoint位于无人复用的token位置，缓存命中率极低
  - Catch 2 — cache entries are huge：SSM state的固定大小虽不随序列长度增长，但比单token的KV cache大几个数量级（d_state × d_model × 4 bytes per layer），大量低价值checkpoint占满缓存，导致频繁thrashing和低命中率

  全栈执行例子（以vLLM+ extended serving Mamba2-Hybrid-7B，LMSys conversational workload）：
  - 算法层：Fine-grained checkpointing——每隔k token保存完整SSM states（d_state=128, d_model=4096 → ~2MB per layer checkpoint, 24 SSM layers → ~48MB per sequence checkpoint）。LRU eviction管理所有cached states（SSM states + KV caches）。
  - 系统框架层：vLLM+/SGLang+ serving框架。扩展vLLM原有prefix caching（仅支持KV cache）增加SSM state checkpoint save/restore逻辑。但admission策略简单粗暴——每个请求的所有state都被缓存（admit-all），eviction仅基于recency（LRU）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。Trace-based离线模拟不涉及实际GPU kernel调度。
  - 硬件架构层：论文未明确说明（Cloudlab CPU节点运行离线模拟）。

  Baseline缺陷：(1) Admission缺乏判断力——每个请求的SSM state都被无差别缓存，对sparsely-hit的checkpoint没有识别能力；(2) Eviction缺乏计算感知——LRU仅看recency，不考虑复用该state能节省多少计算量（FLOPs vs 内存的tradeoff），导致高计算价值的长前缀state可能被低计算价值的短前缀state挤出；(3) 混合架构下KV cache与SSM state统一管理的缺失——KV cache大小随序列长度线性增长而SSM state固定，两者有不同的memory-compute tradeoff特征，LRU无法感知这种差异。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Marconi——首个面向Hybrid LLMs的prefix caching系统，通过两个协同设计解决baseline缺陷：

  **(1) Judicious Admission（基于前缀复用模式分类）**：不再admit-all，而是通过radix tree bookkeeping将前缀复用分为两种模式——Purely Input（系统提示词、few-shot示例，被多请求共享的中间节点）和Input+Output（对话历史，从leaf node继续）。对Purely Input识别为高复用概率并缓存；对Input+Output仅缓存最后token的SSM state（因为对话总是从末尾继续）。限制每序列至多2个SSM state checkpoint，从源头上消除sparsely-hit checkpoints。

  **(2) FLOP-Aware Eviction（计算感知淘汰策略）**：定义FLOP Efficiency = Total FLOPs saved across all layers (Attention + SSM + MLP) / Memory bytes consumed by cached states。Utility Score = recency + α × flop_efficiency。关键洞察：SSM state大小固定（不随prefix length变化），但长前缀节省更多FLOPs（覆盖更多token的计算），因此长前缀的FLOP efficiency更高。Marconi优先保留长前缀的高FLOP efficiency entries，而非LRU-only仅基于时间。α参数由config_tuner自动根据workload模式调优。

  **(3) Unified Radix Tree管理**：KV caches（Attention层）和SSM states（SSM层）统一在单个radix tree中管理，因为所有layer states必须代表同一prefix才能被复用。避免disaggregated管理导致的prefix一致性问题和eviction策略不协调。

  全栈执行对比baseline（以Marconi serving Mamba2-Hybrid-7B，同一LMSys workload）：
  - 算法层：Judicious admission替代naive checkpointing——radix tree speculative insertion后仅缓存高复用概率节点（purely-input branches + leaf final states），每序列≤2个checkpoint。FLOP-aware eviction替代LRU——eviction决策综合考虑recency和该state节省的FLOPs/byte。Mamba2-Hybrid 7B (4Attn+24SSM+28MLP)中SSM layer的FLOP efficiency远高于Attention layer（因为SSM state固定大小但覆盖prefix全部token的计算节省），Marconi的eviction policy因此对SSM-heavy模型表现更优。
  - 系统框架层：Marconi核心逻辑在radix_cache_hybrid.py中实现，通过radix_cache_vllm.py适配到vLLM serving framework。与baseline vLLM+的admit-all+LRU不同，Marconi在每次请求到达时执行speculative admission判断，仅selectively缓存高价值state。config_tuner持续监控workload命中率反馈，动态调整α权重。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。离线trace-based模拟环境。
  - 硬件架构层：论文未明确说明具体GPU硬件。Trace-driven evaluation on CPU。

  结果：Marconi vs fine-grained checkpointing (naive admission) token hit rate提升4.5×–34.4×（取决于workload和SSM比例），P95 TTFT降低36.1%–71.1%。FLOP-aware eviction alone vs SGLang+ LRU提升19%–219% token hit rate。Marconi表现随longer contexts、higher SSM ratios、larger SSM state dimensions更好——这些趋势与Hybrid LLM架构发展方向（更高SSM比例、更大模型如Jamba 1.5 398B）一致。

  设计思路核心：Marconi的本质是将prefix caching的admission和eviction从"recency-only, admit-all"的粗粒度策略转变为"reuse-pattern-aware, FLOP-aware"的精细策略。关键洞察在于Hybrid LLMs中Attention和SSM层具有本质不同的memory-compute tradeoff特征（KV cache O(L) memory vs SSM state O(1) memory），前缀复用模式可以被分类为结构化的两种类型，以及radix tree的节点自然对应复用概率梯度（中间节点 → 高复用，叶子节点 → 仅末尾复用）。

## Mordal: Automated Pretrained Model Selection for Vision Language Models

- baseline方法是什么？
  Baseline是grid search（穷举搜索）——对给定的pretrained vision encoder和LLM模型库，穷举所有VE×LLM组合，对每个VLM候选用完整alignment数据训练feature projector并评估下游task性能，选择性能最优的组合。以论文中的49个候选（7 VE × 7 LLM）为例，grid search需要5439 GPU hours（每个候选~111 GPU hours，含feature projector training + LoRA fine-tuning + 6 task evaluation）。此外，开发者也会"cherrypick"——凭经验/直觉选择pretrained模型（如选最新、最大或最知名的），但这种方法不稳定且不可预测。

  全栈执行例子（以grid search在16×A40上搜索GQA最优VLM为例）：
  - 算法层：标准VLM架构——Vision Encoder（如CLIP-ViT-L/14）提取image embeddings → MLP Feature Projector对齐到LLM token space → LLM Decoder（如Vicuna-1.5-7B）生成文本。对于每个VE×LLM组合，训练MLP projector + LoRA fine-tune LLM。共49个candidate的完整训练和评估流程。
  - 系统框架层：基于LLaVA pipeline，使用HuggingFace Transformers + PEFT (LoRA) + Flash Attention-2。无搜索优化——每个candidate被独立、完整地训练。所有candidate串行或小批量并行在16×A40上执行。使用Adam optimizer（lr=1e-4, minibatch=4），LLaVA-1.5-Instruction作为alignment data。
  - 编译框架层：论文未明确说明（标准PyTorch eager execution）。
  - kernel调度层：论文未明确说明（标准A40 GPU kernel执行：matmul + attention + MLP kernels）。
  - 硬件架构层：16× NVIDIA A40 GPU (48 GB GDDR6 each)。Grid search通过数据并行或pipeline并行方式利用多GPU，但每个candidate的内部训练不涉及跨GPU分布式训练。

  Baseline根本缺陷：(1) **搜索空间爆炸**——HuggingFace上>150,000个LLM，即使精选到7×7=49个candidate也需要5439 GPU hours，扩展到更大模型库不可行；(2) **每个候选评估成本高**——每个candidate需要>100 GPU hours进行完整alignment训练，而大部分candidate最终会被淘汰（资源浪费）；(3) **无法利用候选间相似性**——grid search将每个candidate视为独立，不考虑相似VE/LLM组合可能产生相似性能，重复评估冗余候选；(4) **chicken-and-egg问题**——未alignment的VLM无法评估zero-shot性能（feature projector未训练时LLM不理解image embeddings），必须训练后才能知道性能，无法预先筛选。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Mordal——通过两个正交优化维度解决grid search的低效：(1) 减少候选数量（Candidate Clustering）; (2) 减少每个候选的评估时间（Efficient Evaluation）。两个维度各自包含协同设计：

  **(1) Candidate Clustering**——解决"搜索空间爆炸"和"候选间冗余"：
  Mordal使用CKA (Centered Kernel Alignment) 计算VE和LLM的表示相似度，通过两步聚类构建VLM候选cluster。关键insight：相似pretrained模型的组合会产生相似性能，因此只需评估每个cluster的representative candidate即可推断cluster整体的优劣。
  - 两步聚类策略：先对VE聚类（基于vision embeddings的CKA相似度）→ 再在每个VE cluster内基于medoid VE对LLM聚类（基于LLM last hidden state的CKA相似度）→ Cartesian product生成VLM候选cluster
  - Inter-cluster evaluation：仅评估每个cluster的medoid候选，淘汰性能差的cluster，大幅减少需评估的candidate数量
  - Intra-cluster evaluation：仅在remaining Top-K cluster内进行candidate-granularity评估
  - 为什么CKA有效：(a)可比较不同shape的表示（不同VE/LLM的输出维度不同，cosine similarity不可用）; (b)对MLP projection变换鲁棒（feature projector的MLP层不影响CKA的性质）

  **(2) Efficient Evaluation (Early Stopping + Scaling Prediction)**——解决"每个候选评估成本高"：
  两个互补机制减少单个candidate的评估时间：
  - **Early Stopping (SHA)**：在inter-cluster阶段使用Successive Halving Algorithm。每轮以固定budget (b)评估所有representative candidates，保留top 1/η，下一轮增加budget (b×η)，循环淘汰直到收敛。效果：差候选在少量数据训练后即被淘汰，资源集中于有潜力的候选。
  - **Scaling Prediction**：利用发现的observational scaling law——VLM alignment性能与训练数据量存在log-linear关系（即log(Err)与log(data_ratio)线性相关）。对每个intra-cluster candidate，从少量数据(如1/8)开始训练并逐步减少数据量，收集(log(r), log(Err))点对，拟合线性回归模型f_c，预测r=1（完整数据）时的性能。效果：无需完整训练即可预测最终性能。
  - 关键发现：log-linear scaling仅在训练数据量超过一定阈值后出现（consistent with prior work），因此Scaling Prediction从大ratio向小ratio递减（利用已有checkpoint节省计算）。

  全栈执行对比baseline（以Mordal在16×A40上搜索GQA最优VLM为例）：
  - 算法层：同一VLM架构和49 candidates。Mordal不改变模型训练本身，而是改变搜索策略——通过CKA-based clustering将49 candidates分为~10-15 clusters → Inter-cluster SHA评估representatives → 淘汰差cluster → 保留Top-3 cluster → Intra-cluster Scaling Prediction → 选出最优candidate。Search time从5439h降至483h（11.2× speedup），成功选出Top-1 candidate (SigLIP-Vicuna, 66.4% accuracy)。
  - 系统框架层：基于LLaVA + PyTorch + HuggingFace + PEFT + Flash Attention-2。Mordal提供统一Python接口 (`import mordal` → `mordal.query_for_model()`)，用户提供alignment data + target task data + model zoo。Mordal自动管理：clustering（CKA计算 + hierarchical clustering）→ inter-cluster training + SHA filtering → intra-cluster training + scaling prediction → best model selection。空闲GPU自动分配给未收敛candidate加速搜索。
  - 编译框架层：论文未明确说明（标准PyTorch eager execution）。
  - kernel调度层：论文未明确说明（标准A40 GPU kernel执行）。
  - 硬件架构层：16× NVIDIA A40 GPU (48 GB GDDR6)。Search time breakdown（Figure 11）：early stopping占大部分时间（因需实际训练部分数据），scaling prediction仅占小部分（仅对remaining candidates进行）。

  设计思路核心：Mordal的本质是将pretrained model selection从"穷举+完整训练"重构为"聚类粗筛+预测精排"的两阶段pipeline。CKA-based clustering利用表示相似度将候选分组（类似分层采样的思想），SHA在cluster级别做粗粒度淘汰，Scaling Prediction在candidate级别利用log-linear scaling做细粒度排序。两个关键科学发现支撑这个pipeline：(1) 相似pretrained模型的VLM性能也相似（使clustering可行）; (2) VLM alignment存在observational scaling law（使prediction可行）。Mordal证明了VLM pretrained model selection问题可以通过systematic algorithmic framework高效解决，而非依赖人工经验或暴力搜索。

## Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

- baseline方法是什么？
  Baseline是PyTorch（及TensorFlow）的默认eager execution模式——每个DL算子作为独立的GPU kernel被framework runtime逐个调度执行。框架执行流程：(1) Python/C++ operator dispatch —— 根据tensor types/shapes选择对应的kernel实现；(2) output shape inference —— 计算输出tensor shape用于后续operator；(3) GPU kernel selection —— 从多个candidate kernel中（如cuDNN的不同implementation）选择最优的；(4) kernel argument preparation —— 准备launch参数（grid/block dims, shared memory等）；(5) GPU kernel launch —— 通过CUDA driver API提交kernel到GPU；(6) memory allocation —— GPU memory的(de)allocation在每次operator执行时发生。这些overhead加起来导致GPU idle time高达91%（PyTorch）和71%（TensorFlow），尤其是当模型包含大量small GPU kernels（如mobile-optimized CNNs、NAS architectures）时。此外，baseline在单一default CUDA stream上串行执行所有kernel，忽视了DAG中独立算子间的并行机会。

  全栈执行例子（以PyTorch eager mode在V100上执行NASNet-A mobile inference，batch_size=1）：
  - 算法层：NASNet-A mobile CNN模型，forward pass包含~700个算子（separable conv、batch norm、ReLU、pooling、concat等），大部分为small kernels（计算量小，memory bound）。
  - 系统框架层：PyTorch eager mode。对每个operator，Python dispatcher: (a) 查找autograd Function对象，(b) 推断output tensor shape (meta-data computation on CPU)，(c) 调用对应的CUDA kernel wrapper，(d) 准备kernel launch parameters (grid/block dims, tensor strides) on CPU，(e) cudaLaunchKernel提交到default CUDA stream，(f) GPU执行kernel（可能仅有几十微秒的kernel执行时间，但CPU端的scheduling overhead累计上百微秒）。所有kernel在单一stream上串行执行——即使DAG中有多个独立分支（如NASNet cell中的多个separable conv分支），也无法在GPU上并行。Memory allocation按需执行，产生频繁的cudaMalloc/cudaFree开销。
  - 编译框架层：论文未明确说明（PyTorch eager mode不使用JIT编译）。
  - kernel调度层：单一default CUDA stream。GPU Scheduler从stream queue顺序launch每个kernel，kernel执行完成后GPU SM等待CPU提交下一个kernel（idle gap）。即使有多个kernel ready（DAG中无依赖），也无法并发。
  - 硬件架构层：NVIDIA V100 GPU + Intel Xeon CPU。Framework runtime开销主要消耗在CPU上，但CPU处理scheduling的时间远长于GPU执行small kernel的时间，导致GPU频繁idle。例如一个小separable conv的GPU执行时间~10µs但CPU scheduling开销~100µs，GPU idle比率高达90%。

  Baseline两大核心缺陷：
  1. **Framework scheduling overhead过大**：不是单纯的memory allocation，而是整个operator dispatch链（operator selection → shape inference → kernel selection → argument preparation → kernel launch）的累计开销，当模型包含大量small kernels时尤其严重。
  2. **不必要的串行执行**：所有kernel在同一CUDA stream上执行，忽视了DAG中独立算子间的并行性——DAG的logical concurrency未转化为GPU physical parallelism。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Nimble——一个轻量级DL执行引擎，通过Ahead-of-Time (AoT) scheduling消除框架runtime overhead，并通过自动多stream执行实现GPU kernel并行化。

  **(1) AoT Scheduling via CUDA Graph**——解决"Framework scheduling overhead"缺陷：
  Nimble在模型执行前完成全部scheduling：使用dummy input预运行一次PyTorch模型，通过CUDA Stream Capture API拦截所有GPU kernel calls和memory allocations，生成完整的CUDA Graph（包含kernel launches、arguments、submission order和stream assignment）。运行时仅通过cudaGraphLaunch重放CUDA Graph，GPU直接从预录制的trace执行所有kernel——完全绕开PyTorch的operator dispatch、shape inference、kernel selection和argument preparation流程。AoT preparation平均耗时0.35s，是一次性开销，在后续所有运行中摊销。

  **(2) Automatic Multi-Stream Execution**——解决"不必要的串行执行"缺陷：
  Nimble自动将DAG中的算子分配到多个CUDA stream，使独立kernel在GPU上并行执行。Stream assignment算法：(a) 计算DAG的Minimum Equivalent Graph (MEG) —— 消除冗余传递边；(b) 构建bipartite graph（MEG中的边成为bipartite节点）；(c) Ford-Fulkerson算法寻找maximum matching —— 每种matching代表一组可并行执行的算子；(d) 按matching结果将算子分配到不同stream，同时最小化跨stream同步（每个stream pair间仅保留必要的CUDA event synchronization）。理论证明实现maximum logical concurrency with minimum synchronizations。

  **(3) Integration simplicity**——与现有框架正交：
  用户仅需两行代码：`nimble_model = nimble.Nimble(model)`。Nimble在PyTorch之上运行，支持inference和training，与TorchScript兼容。Nimble的AoT scheduling和multi-stream执行与TensorRT/TVM的graph optimization和kernel tuning正交——Nimble专注于消除runtime scheduling overhead，可叠加operator fusion获得进一步加速。

  全栈执行例子（以Nimble在V100上执行同一NASNet-A mobile inference为例，对比baseline）：
  - 算法层：同一NASNet-A mobile模型。Nimble不改变模型算法，仅改变执行方式。支持operator fusion的子集（不如TensorRT aggressive）和Conv算子的basic kernel selection（cuDNN vs PyTorch native）。
  - 系统框架层：Nimble wrapping PyTorch model。AoT阶段：torch.jit.trace → Graph Rewriter (stream assignment) → CUDA Graph capture → CUDA Graph instantiation。Runtime阶段：直接cudaGraphLaunch，无PyTorch scheduler参与。整个Python runtime被bypass——只有GPU kernel执行。
  - 编译框架层：论文未明确说明（使用PyTorch TorchScript作为IR，但不引入新的编译器）。
  - kernel调度层（核心差异）：NASNet-A的DAG中多个separable conv分支被自动分配到不同CUDA stream并行执行。AoT生成的CUDA Graph保留了多stream拓扑和跨stream同步点。Runtime replay时：Stream 0执行branch A（sep_conv1→sep_conv2），Stream 1同时执行branch B（sep_conv3→sep_conv4），仅在concat点通过CUDA event同步。GPU SM得以持续工作，消除了baseline中kernel间idle gap。Framework overhead从CPU端完全消除——GPU kernel直接执行无CPU mediation。
  - 硬件架构层：NVIDIA V100 GPU。从"CPU调度驱动、GPU频繁idle"转变为"GPU自主执行预录制kernel图"。Max logial concurrency达15（NASNet-A的DAG中最多15个可并行执行的算子）。Multi-stream自身贡献up to 1.88× speedup。

  整体效果：vs PyTorch inference up to 22.34× speedup, vs TensorRT up to 2.81×, vs TVM up to 1.70×。Training speedup up to 3.61×（CIFAR-10 small models）。限制：(1) 仅支持static neural network model（不支持dynamic control flow），与TensorRT类似；(2) 大kernel模型（BERT、ResNet-50 ImageNet training）speedup有限——当kernel本身计算量大时，framework overhead占比小。

  设计思路核心：Nimble的本质是将DL执行的scheduling从"runtime per-operator dispatch"转变为"ahead-of-time whole-graph capture + replay"。这个转变的关键在于：CUDA Graph API提供了record-then-replay的能力，使GPU可以脱离CPU framework自主执行完整的计算图，而multi-stream算法自动为这个recorded graph找到最优的并行执行拓扑。Nimble证明在现有hardware和framework基础上，通过AoT scheduling和自动多stream并行即可消除DL框架的主要性能瓶颈——无需重写framework runtime或修改GPU硬件。

## ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

- baseline方法是什么？
  Baseline是vLLM [27] monolithic LMM serving部署。在monolithic架构下，LMM的image preprocessor、image encoder和LLM backend（prefill + decode）被打包为单个serving instance，使用统一的TP配置和batch size。所有instance完整副本部署，text-only和image-text请求都由同一个instance处理。如需扩容，整个instance（包括所有stage）一起扩容。

  Baseline的三个核心缺陷：(1) **Image encoding成为TTFT瓶颈但无法独立优化**——CroAttn模型中image encoding占TTFT的65–79%（Insight 1），但monolithic将其与LLM backend绑定，无法独立scale out encoder和并行化encoding；(2) **统一batch size和TP配置导致资源浪费**——image encoding是compute-bound，应使用小batch和TP-1；LLM prefill在DecOnly中compute-bound（需大TP），在CroAttn中更轻量（需小TP）；decode是memory-bound（需大batch）。Monolithic强制所有stage使用相同配置（Insight 4、5）；(3) **无法应对modality-specific bursts**——生产环境image-text和text-only traffic表现出独立的burst pattern（Insight 6），monolithic无法针对性扩缩容，image burst时只能整体扩容，导致LLM backend过度provisioning。

  全栈执行例子（以Llama3.2-11B (CroAttn) monolithic部署在4×A100 TP-4上，处理含4张896×896图像的请求）：
  - 算法层：ViT-H/14 (630M) image encoder + Llama 3.1 (8B) LLM backbone with 4 cross-attention layers（共40 layers，其中4层为CroAttn）。Image tokens仅在CroAttn layers参与cross-attention，自注意力层仅处理text tokens。Connector MLP将image tokens映射到LLM token space。
  - 系统框架层：vLLM v0.7.2，PD colocated模式（prefill和decode在同一instance）。Image preprocessing (CPU) → image encoding (GPU, TP-4, batching the 4 images' tiles) → LLM prefill (GPU, TP-4, batch_size=1) → LLM decode (GPU, TP-4, continuous batching via PagedAttention)。所有4张图的tiles串行或小batch编码在4个GPU上（630M模型被TP分到4 GPU，inter-GPU通信开销 > 计算节省）。
  - 编译框架层：论文未明确说明。PyTorch eager execution on CUDA。
  - kernel调度层：论文未明确说明。NVIDIA A100 GPU kernel执行：ViT forward（matmul + attention + MLP kernels）、LLM prefill（self-attention + cross-attention + MLP kernels）。
  - 硬件架构层：DGX-A100 server，8× A100 80GB via NVLINK 3.0（600 GB/s），内部4 GPU用于TP-4。AMD EPYC 7V12 96-core CPU for image preprocessing。1900 GiB DRAM。

  在此执行下，若同时到达大量image请求：(1) 所有请求排队等待TP-4 GPU资源，image encoding无法并行到更多GPU；(2) 大image请求造成HoL blocking——4张图的encoding阻塞后续小请求；(3) 扩容时TP-4整体扩容，即使只需更多encoder也要带4个GPUs。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出ModServe，将LMM serving拆分为Image Instances和Text Instances两个独立资源池，通过stage-aware profiling、token-aware autoscaling和modality-aware routing解决上述三个缺陷。

  **解决缺陷1（image encoding瓶颈）**：将Image Instances与Text Instances物理分离。Image Instances可以独立扩展到不同GPU，甚至多张GPU（如每个TP-1），实现单请求内多图像跨多个Image Instance并行编码（Insight 2：images within a request have no compute dependency）。ModServe调度将大请求的图像split到多个Image Instance，类似request chunking效果，减少HoL blocking。

  **解决缺陷2（统一配置浪费）**：Image Pool Manager和Text Pool Manager各自根据offline LMM profile独立决定各stage的TP度、max batch size。例如，ViT-H/14 (630M) encoder在TP-1时吞吐最高（因TP-4下inter-GPU communication > compute saving），LLaMA 3.1 (8B) LLM backend在TP-4时TTFT最低。ModServe允许Image Instance TP-1、Text Instance TP-4共存于同一physical GPU pool，灵活资源分配。

  **解决缺陷3（modality-specific bursts）**：Token-aware autoscaling基于modality-specific load（image tokens/sec for Image, prompt tokens/sec for Text）而非请求速率，独立计算各池需要的instance数（⌈ML/MC⌉）。Image burst时仅scale out Image Instances，LLM backend不受影响。CroAttn模型在image burst时LLM prefill不增加（因image tokens仅参与4/40 cross-attention layers），因此Text Instances无需扩容——这正是ModServe达到最高41.3% cost saving的来源。

  全栈执行例子（以Llama3.2-11B (CroAttn) ModServe部署在16-server 128-GPU集群上，处理含4张896×896图像的请求）：
  - 算法层：同上（ViT-H/14 encoder + Llama 3.1 8B with CroAttn layers）。但encoder和LLM backend现在运行在不同GPU上，通过RDMA传输image tokens。
  - 系统框架层：ModServe on vLLM v0.7.2 + HuggingFace Transformers。
    - Image Instances (TP-1, batch_size=1): 接收image-text请求 → CPU image preprocessing (numactl NUMA-pinned) → GPU encoding (ViT-H/14 forward, 4 tiles) → image tokens注册RDMA buffer
    - Image Pool Manager (gRPC server): 接收实时load metric → 计算autoscaling决策（⌈image_tokens_per_sec / per_instance_max_capacity⌉）→ 管理Image Instance生命周期
    - Text Pool Manager (gRPC server): 同上，基于prompt tokens/sec → 选择目标Text Instance
    - Pull-based RDMA: Text Instance收到RDMA地址 → NCCL+GPU Direct RDMA pull image tokens → Connector MLP（共置于Text Instance GPU）
    - Text Instance (TP-4, mixed batch): LLM prefill (image tokens仅在4 CroAttn layers) + decode (PagedAttention)
    - Modality-Aware Router: image-text→least-image-token-load Image Instance; text→least-pending-text-tokens Text Instance (CroAttn按text tokens路由)
    - SLO-driven Priority Scheduler: 优先短请求，防止长image-text请求HoL blocking短text请求
  - 编译框架层：论文未明确说明。PyTorch eager + NCCL backend。
  - kernel调度层：论文未明确说明具体kernel。A100 GPU上执行ViT matmul/attention kernels（Image Instance）和LLM self-attention/cross-attention/MLP kernels（Text Instance）。Image token transfer使用GPU Direct RDMA over InfiniBand。
  - 硬件架构层：16× DGX-A100 servers（128× A100-80GB total）。NVLINK 3.0 intra-server，InfiniBand inter-server（支持GPU Direct RDMA，P99 image token transfer latency 5ms，<0.5% TTFT for CroAttn）。Image和Text Instance可colocate同server（如1× TP-4 Text Instance + 2× TP-2 Image Instance on 8-GPU server），但instance配置独立。

  结果：vs vLLM monolithic baseline，ModServe取得3.3×（Llama3.2 CroAttn）–5.5×（InternVL DecOnly）higher throughput（static allocation），25–41.3% cost saving（autoscaling with production traces）。PD disaggregation之上额外2.8× average TTFT reduction（图19）。

  设计思路核心：ModServe的本质是将LMM serving从"model instance as atomic unit"的粗粒度资源管理转变为"pipeline stage as decoupled resource pool"的精细管理。这个转变的可行性依赖于三个发现：(1) 各stage对batch/TP的响应曲线截然不同（使独立配置有意义）；(2) image encoding tokens间无依赖（使并行编码可行）；(3) 不同modality的traffic burst pattern独立（使针对性autoscaling高效）。

## OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

- baseline方法是什么？
  Baseline是**Token Concatenation**方案：将视觉嵌入序列 $\mathbf{E}_v \in \mathbb{R}^{N_v \times C}$ 和音频嵌入序列 $\mathbf{E}_a \in \mathbb{R}^{N_a \times C}$ 简单拼接为 $[\mathbf{E}_v, \mathbf{E}_a]$ 送入LLM backbone。这种方案在三个层面存在缺陷：(1) **语义对齐缺失**：视觉和音频嵌入来自各自独立的projector，缺乏显式的跨模态对齐机制，导致LLM难以利用video-audio互补信息；(2) **时序关系丢失**：拼接序列中视觉和音频token的相对位置无法反映其真实时间戳关系，LLM的position embedding仅编码序列位置而非绝对时间；(3) **omni-modal数据稀缺**：缺乏高质量的视频-音频联合标注数据，现有video QA数据仅利用视觉信息，忽视了同步音频轨中的监督信号。

  全栈执行例子（Baseline - Token Concatenation处理一个带音频的视频问答请求）：
  - 算法层：ViT提取视频帧特征 → 2-layer MLP project → $\mathbf{E}_v$；AF-Whisper提取音频特征 → 2-layer MLP project → $\mathbf{E}_a$；直接拼接 $[\mathbf{E}_v, \mathbf{E}_a]$ 送Qwen2.5-7B LLM，无跨模态对齐loss，无时间编码
  - 系统框架层：论文未明确说明training framework具体名称，使用标准PyTorch分布式训练
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明，标准Transformer compute kernels在H100 GPU上运行
  - 硬件架构层：NVIDIA DGX H100集群

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文从**架构对齐**和**数据增强**两个维度解决baseline缺陷：

  **(1) 架构对齐三件套：**
  - **OmniAlignNet**：在视觉和音频projector之上增加一个共享omni-modal潜在空间，通过可学习query $\mathbf{Q}_v, \mathbf{Q}_a$ 将变长嵌入投影为固定维度 $1 \times C$，再经3层self-attention + L2归一化得到 $\mathbf{V}, \mathbf{A} \in \mathbb{R}^{K \times C}$，施加对称CLIP contrastive loss $\mathcal{L}_{\text{o-align}} = \frac{1}{2}(\mathcal{L}_{v \to a} + \mathcal{L}_{a \to v})$。这直接解决了语义对齐缺失问题——同一视频的视觉和音频嵌入被拉近，不同视频的被推远。
  - **Temporal Embedding Grouping (TEG)**：按固定时间窗口 $T_G$ 将视觉和音频嵌入分组，根据时间戳排序重组为 $[G_v^1, G_a^1, G_v^2, G_a^2, ...]$ 的omni-modal序列。这解决了相对时序关系丢失问题——LLM通过序列中embedding的位置即可感知跨模态的时间对应关系。
  - **Constrained Rotary Time Embedding (CRTE)**：基于几何级数频率 $\omega_i = 2\pi/(T_{\max}\theta^{i/C})$ 对嵌入向量进行元素级旋转变换，高频维度捕获细粒度时间差，低频维度编码粗粒度长时间关系。这解决了绝对时间戳编码问题——相比Learned Time Embedding和RoTE，CRTE的多尺度频率设计同时兼顾局部和全局时序。

  **(2) 数据增强：**
  - **Implicit Omni-Modal Learning**：利用现有video QA数据中自带的音频轨进行隐式omni-modal监督（先前的video LLM仅用视觉信息，浪费了同步音频中的监督信号）。
  - **Explicit Omni-Modal Learning (Data Engine)**：视觉captioning模型 + 音频captioning模型独立生成标注 → LLM进行跨模态纠错和总结（解决"modality-specific hallucination"：仅凭视觉误判场景，仅凭音频误判内容）→ Reasoning LLM合成QA对。生成3.6M omni-modal对话数据。

  **(3) GRPO Omni-Modal Reasoning Post-Training：**
  将GRPO的输入空间扩展为 $\{q_t, q_v, q_a\}$（文本+视觉+音频），18K omni-modal MCQ数据，rollout=8，证明了audio input对RL训练的boost效果（accuracy reward收敛+0.1高于video-only）。

  全栈执行例子（论文方法 - OmniVinci处理同一个带音频的视频问答请求）：
  - 算法层：ViT提取视频帧 → MLP project → $\mathbf{E}_v$；AF-Whisper提取音频(16kHz STFT→Conv+Transformer→750 tokens/30s)→Max Pooling压缩(375 tokens/30s)→MLP project → $\mathbf{E}_a$；**OmniAlignNet**对比学习对齐 $\mathbf{V}, \mathbf{A}$；**TEG**按 $T_G$ 窗口分组重排token序列；**CRTE**对每个embedding施加时间戳旋转编码；送入Qwen2.5-7B LLM autoregressive生成文本响应；可选TTS模块生成语音输出
  - 系统框架层：7阶段渐进式训练（Vision: 5 stages → Audio: 2 stages → Omni-Modal Joint Training: 200B tokens），Long-RL framework for GRPO post-training
  - 编译框架层：论文未明确说明。部署时使用AWQ (W4A16 LLM + W8A8 vision/audio towers) + TinyChat engine
  - kernel调度层：论文未明确说明。RTX 4090/A100/L40s上运行，TTFT ~160ms for 16-frame video+audio
  - 硬件架构层：NVIDIA DGX H100训练集群；部署支持RTX 4090 (24GB)、A100、L40s

  关键结果：OmniVinci用0.2T tokens（Qwen2.5-Omni的1/6）取得Dailyomni +19.05, MMAR +1.7, Video-MME +3.9的显著提升。Ablation验证：Token Concatenation Baseline平均45.51 → +TEG 47.72(+2.21) → +CRTE 50.25(+4.74) → +OmniAlignNet 52.59(+7.08)，每项技术均有显著增益。

## SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

- baseline方法是什么？
  Baseline是两类现有高效注意力方法：(1) **Sparse Attention方法**（VSA、VMoBa、SpargeAttn等）——通过mask跳过部分注意力分数的计算，仅保留top-k最重要的注意力权重。缺陷：实践中稀疏度很少超过90%，通常在40-60%（短序列）或80-85%（长序列100K-300K）。根本原因在于注意力权重分布中，约8.1%的权重大于平均值1/N（主导分布），而中间约47%的权重在1/(100N)到1/N之间——跳过这些"中间值"会导致显著误差（相对L1 error从3%跃升至33%），但保留它们又严重降低稀疏度。这形成了一个"稀疏度-准确率"的trade-off dilemma。(2) **Linear Attention方法**（SANA、DiG等）——通过解耦softmax将复杂度降至O(N)（如φ(Q)φ(K)^T替代softmax(QK^T)）。缺陷：在视频扩散模型中严重失效。根本原因在于full attention权重具有高rank（stable rank远大于d），而线性注意力本质上是rank≤d的低秩近似，无法准确逼近高秩的softmax注意力分布。

  全栈执行例子（以Wan2.1-1.3B视频生成，30K tokens，Sparse Only baseline [85% sparsity]为例）：
  - 算法层：Sparse Only保留top 15%注意力权重（按绝对值排序），mask掉85%的小权重。对每个Q block Q_i，仅对M[i,j]=1的K_j/V_j blocks执行完整FlashAttention计算。但85%稀疏度下：被保留的15%中大部分是"中间值"（在1/(100N)到1/N之间），这些值对最终输出的贡献不够大却被分配了完整的O(N²)计算资源（每个block 64×64 full GEMM + softmax normalization）；而被mask掉的85%中包含约45%极度小值（<1/(100N)），这些值的mask掉是正确的，但另外40%的mask掉导致不可忽略的信息损失——累积误差在逐denoising step中传播。
  - 系统框架层：PyTorch + 自定义sparse FlashAttention kernel。对每个attention层：计算full QK^T → 在Q block级别按行排序取top-k → 生成block mask → sparse FlashAttention执行masked计算。Mask生成需要额外的full QK^T计算或简化的pooling预测。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Sparse FlashAttention kernel在RTX 5090上执行。对N=30K、sparsity=85%：实际计算约15%的QK^T GEMM和PV MM，以及全部softmax normalization（因为需要正确归一化被保留的部分）。Kernel包含mask检查逻辑和稀疏迭代。
  - 硬件架构层：NVIDIA RTX 5090 GPU。Sparse attention的GPU利用率受限于非规则的mask pattern——被跳过的块产生warp divergence，且mask检查逻辑产生额外指令开销。

  Baseline核心缺陷：
  1. **稀疏度天花板**：注意力权重的幂律分布使85-90%成为实践中稀疏度的soft limit——超过此值，被mask的"中间值"累积误差使生成质量急剧下降（从<3% L1 error跃至>33%）。这限制了稀疏注意力能将计算量降到的最低水平。
  2. **线性注意力高秩不匹配**：在视频扩散模型中，full attention权重具有远高于d的stable rank，而线性注意力的表达能力上限为rank d。这一rank gap使线性注意力在视频生成场景中完全失效（Linear Only的VA=0.042 vs Full Attention=76.78）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SLA（Sparse-Linear Attention），通过对注意力权重的三级分类和混合计算策略，从根本上打破稀疏注意力的"稀疏度天花板"：

  **(1) 三级分类替代二级分类**——解决"稀疏度天花板"：
  SLA的核心洞察：注意力权重可以被分解为两部分——少量大值的sparse component（高rank，约8%）+大量小值的low-rank component（极低rank，约92%）。这解释了为什么sparse only和linear only都失败：sparse attention需要处理所有非极小的权重（包括那些"中间值"），而linear attention不需要处理高rank的大值部分。SLA将注意力权重分为三级而非传统的二级（保留/跳过）：
  - Critical（top k_h=5%）：高rank的大值 → 执行完整的O(N²) FlashAttention
  - Marginal（中间~85%）：低rank的小值 → 执行O(N)线性注意力（仅需d×d矩阵加法，占full attention <0.5% cost）
  - Negligible（bottom k_l=10%）：极度小值 → 完全跳过
  三级分类的关键意义：**sparsity从85%（仅保留critical）跃升至95%（跳过negligible），而marginal块用几乎免费的线性注意力替代，避免了"中间值"的dilemma**。对比baseline Sparse Only（85%稀疏度=跳过85%），SLA在95%稀疏度（跳过10% negligible + 用O(N)处理85% marginal）下实际计算量约为Sparse Only的一半（因为linear attention几乎免费），同时质量更好。

  **(2) 可学习融合而非简单叠加**——解决"线性注意力不匹配"：
  SLA不是简单地将稀疏和线性注意力的输出相加（L+S baseline的VA=29.65 vs SLA=76.96），而是设计了两个关键机制：
  - 可学习投影Proj(O^l)：对线性注意力输出应用可学习线性变换R^d→R^d，减少softmax和线性注意力之间的分布不匹配。这使线性注意力从"直接近似"转变为"learnable compensation"。
  - Fine-tuning：通过少量fine-tuning steps（2000步，<0.1% pretraining cost），模型参数自适应学习如何利用线性注意力作为稀疏注意力的补充。Fine-tuning使模型学会"信任"线性注意力的补偿，而非仅依赖稀疏部分。

  **(3) 预计算和单kernel融合**——实现前三级的实际加速：
  SLA通过预计算h_j = φ(K_j)^T V_j和z_j = rowsum(φ(K_j)^T)，使marginal块的线性注意力计算降为单次矩阵加法（而非每次重新计算φ(Q)φ(K)^T V）。这确保了95%稀疏度下的理论加速能转化为实际wall-clock speedup——13.7× kernel speedup vs FlashAttention2，2.2× end-to-end speedup。

  全栈执行对比baseline（以Wan2.1-1.3B同一视频生成，SLA 95% sparsity为例）：
  - 算法层：同一DiT架构，但每个注意力层替换为SLA。预测压缩mask P_c → 三级分类(M_c=1/0/-1) → critical块用FlashAttention（5%块数）→ marginal块用线性注意力precomputed H_i/Z_i加法（85%块数）→ negligible跳过（10%块数）→ Proj(O^l)融合。SLA @ 95% sparsity (2.73T FLOPs) vs Full Attention (52.75T FLOPs) = 19.3× FLOP reduction，视频质量VA=76.96 vs Full=76.78（无退化）。
  - 系统框架层：PyTorch + 自定义fused SLA CUDA kernel。Fine-tuning on 20K private videos × 2000 steps × batch 64。Fine-tuning cost << 0.1% pretraining。
  - 编译框架层：论文未明确说明（直接CUDA kernel实现，无编译框架层）。
  - kernel调度层：单CUDA kernel融合三种计算：
    1. Critical块：OnlineSoftmax FlashAttention（与FlashAttention2相同的GEMM+softmax+GEMM pipeline）
    2. Marginal块：单次矩阵加法（H_i += h_j）和向量加法（Z_i += z_j），预计算保证极低开销
    3. Forward: 13.7× vs FlashAttention2 forward; Backward: 6.8× vs FlashAttention2 backward
    4. 额外效率优化：Lookup table（预处理稀疏mask非零位置）、Pre-aggregation（减法替代加法）、Method of Four Russians（分组预计算子集和）
  - 硬件架构层：NVIDIA RTX 5090。SLA kernel的GPU利用率特征：critical块期间Tensor Cores活跃（GEMM），marginal块期间仅CUDA Cores做加法（几乎瞬时），negligible块完全跳过。由于marginal块计算量极低（<0.5% full attention），GPU大部分时间在critical块的Tensor Core密集型计算上，有效利用GPU算力。端到端attention时间从97s降至11s（8.8× reduction），非attention部分（MLP/RMSNorm/Conv）不变，总体2.2× end-to-end speedup。

  设计思路核心：
  论文的根本洞察是**注意力权重矩阵的谱分解**——P = (P ⊙ M) + (P ⊙ (1-M))，其中sparse component P⊙M保持高rank（需要完整O(N²)计算），而low-rank component P⊙(1-M)可用rank-d线性注意力准确近似。这解释了为什么单独使用sparse attention或linear attention都无法成功——它们各自试图用单一机制处理具有本质不同数学结构的两个成分。SLA的统一框架通过三级分类实现"对的结构处理对的成分"：高rank部分用sparse attention保持精度，低rank部分用linear attention换取效率。Proj层和fine-tuning进一步解决了两个成分输出分布不匹配的问题，使融合后的输出与full attention保持一致。关键实验证据：Figure 3直观展示了去除top 8%值后的剩余矩阵stable rank从~150降至~20，实证验证了low-rank分解假设。

## Task-Based Tensor Computations on Modern GPUs

- baseline方法是什么？
  Baseline是GPU上现有的三种高性能编程范式和系统：(1) CUTLASS模板库——用户在C++模板层面手动管理数据移动（TMA调用）、同步（barriers）、warp specialization（DMA warp vs compute warps）、pipelining（shared memory buffers深度），能达峰值性能但显式管理复杂且易错；(2) ThunderKittens——提供更简洁的API包装Tensor Core和TMA操作，但仍需程序员显式管理同步和通信；(3) Triton——block-level DSL，用户仅描述thread block级别的计算，编译器自动做thread-level decomposition、memory allocation、scheduling——但对Hopper架构，Triton的heuristic决策常常次优（如不使用TMA默认、heuristic将reduction accumulator放在SMEM而非register file、不自动overlap独立操作），且用户无法干预这些性能关键决策。

  全栈执行例子（以H100 Hopper GEMM在cuBLAS/CUTLASS中的执行）：
  - 算法层：标准GEMM C=A·B，tile-based decomposition，每个SM负责T_M×T_N输出tile
  - 系统框架层：cuBLAS（vendor闭源库）或CUTLASS（开源模板库）。cuBLAS包含hand-tuned assembly实现；CUTLASS C++ templates参数化tile size、pipeline depth、warp specialization策略。
  - 编译框架层：标准CUDA C++ compiler (NVCC)，无特殊编译pass。所有优化由程序员在C++ template level手工表达。
  - kernel调度层：Warp-specialized GEMM（Figure 1b）——DMA warp（32 threads）专门通过TMA异步加载A/B tiles到shared memory（单线程调用TMA_load），通过prod barrier通知compute warpgroup；compute warpgroup（128 threads=4 warps）通过warpgroup_sync→wgmma→warpgroup_wait序列驱动Tensor Core做GEMM，完成后通过cons barrier通知DMA warp buffer可重用。Pipelining (PIPE>1)使DMA预取隐藏global memory latency。程序员必须手动：(a) 插入所有barrier同步（prod/cons/copyout），(b) 管理pipeline buffer indexing [k%PIPE]，(c) 保证write-after-read anti-dependency（backwards edge），(d) 分配shared memory staging buffer (sC)用于TMA store。
  - 硬件架构层：NVIDIA H100 GPU，TMA异步copy单元（支持multicast, shared memory barriers），Tensor Core（wgmma指令，128线程cooperative launch，操作数跨registers+shared memory），named barriers用于warp间同步。

  Baseline缺陷：
  1. **编程复杂性爆炸**：从Ampere到Hopper，GEMM kernel结构根本性改变——从bulk-synchronous（所有线程参与load+compute）变为warp-specialized（DMA warp专做TMA copy，compute warpgroup专做Tensor Core MMA）。程序员必须理解和管理异步fixed-function units之间的producer-consumer同步，正确性难以保证。
  2. **Triton的heuristic次优**：Triton将性能关键决策（数据放置、TMA使用、操作overlap）完全委托给编译器heuristic，无法被程序员控制。Dual-GEMM中Triton未overlap B₂加载与A·B₁计算；GEMM+Reduction中Triton将reduction accumulator放在SMEM而非register file，且未overlap GEMM与reduction。
  3. **修改程序易引入bug**：在CUTLASS/ThunderKittens等低层模型中，添加新功能需要修改partition、同步和通信代码——遗漏任何一处都导致数据竞争或死锁。
  4. **性能调优受限**：Triton不允许程序员干预tile size、memory placement、pipelining等决策；CUTLASS允许但需要大量代码修改。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Cypress——task-based编程模型和编译器，核心设计是分离Logical Description（顺序语义的task树）和Mapping Specification（task→processor, tensor→memory的绑定），通过编译器自动管理异步和同步。

  **解决缺陷1（异步编程复杂性）**：
  Cypress程序无显式数据移动和同步。程序员编写顺序语义的task描述（含prange/srange loop和launch语句），mapping specification声明性能关键决策。编译器通过dependence analysis自动插入copy-in/copy-out维持coherence，通过event-based IR编码和调度所有依赖关系，最终生成正确的warp-specialized CUDA C++。

  **解决缺陷2（Triton heuristic次优）**：
  Mapping specification允许程序员显式控制：(a) 每个task在哪个processor级别执行（HOST/BLOCK/WARPGROUP/WARP/THREAD），(b) 每个tensor在哪种memory中物化（GLOBAL/SHARED/REGISTER/NONE），(c) tunable参数（tile sizes, warpgroup count, pipeline depth），(d) warpspecialize和pipeline flag。GEMM+Reduction中用户将reduction accumulator放在Register（NONE+partitioned）而非SMEM，Cypress自动利用asynchrony overlap GEMM与reduction——达到2.02-2.18x vs Triton。Dual-GEMM中Cypress自动overlap B₂加载与A·B₁计算——达到1.36-1.40x vs Triton。

  **解决缺陷3（程序修改安全性）**：
  Cypress的sequential semantics保证：任何mapping specification下，编译器保证生成的并发执行与顺序执行等效。添加功能时，partitioning/communication/synchronization由编译器管理，不会引入数据竞争。例子：从单warpgroup GEMM扩展到多warpgroup仅需添加新task variant和调整mapping——现有代码不变。

  **解决缺陷4（性能调优）**：
  Mapping specification隔离所有性能参数——tile sizes, pipeline depth, memory placement, warp specialization——可在不改逻辑代码的情况下独立调整。论文发现通过调整mapping（3 consumer warpgroups替代2个），Flash Attention 2可达到接近FA3的性能。

  论文方法全栈执行例子（以H100 GEMM在Cypress中）：
  - 算法层：同一GEMM算法，通过7个task variants层次化分解到HOST→BLOCK→WARPGROUP→WARP→THREAD各级别。每个variant使用blocks/mma partition operators和prange/srange loops。
  - 系统框架层：Cypress Python embedded DSL编写program + mapping specification。Compiler输出CUDA C++直接编译执行。
  - 编译框架层（核心差异）：Cypress compiler完全自研，6个pass：
    (1) Dependence Analysis: 从entrypoint遍历task tree→插入copy-in/copy-out + event dependencies维持coherence
    (2) Vectorization: flatten implicit pfor loops (warp/thread级别), event arrays保留dependencies
    (3) Copy Elimination: 四类pattern消除冗余copy→同时消除/sync保留
    (4) Resource Allocation: interference graph→最小aliasing allocation
    (5) Warp Specialization: dependence graph partition (DMA vs compute warps) + pipelining
    (6) CUDA C++ Generation: event→sync lowering (barriers, syncwarps)
  - kernel调度层：Generated代码 = DMA warp (TMA async copy) + compute warpgroup (WGMMA) + named barriers + 3-deep pipeline。与baseline CUTLASS实现等效，但所有同步和数据移动由compiler自动生成——程序员在source code中zero lines处理同步。
  - 硬件架构层：同baseline H100。Cypress kernel的TMA/Tensor Core利用率与手写代码相当（0.88x-1.06x cuBLAS GEMM, 0.80x-0.98x FA3）。

  设计思路核心：
  Cypress的本质洞察是**异步GPU编程的两个关注点——算法逻辑（what to compute）和性能策略（how to map）——应该被分离**。算法逻辑用顺序语义的task描述表达（保证正确性、可修改性），性能策略用mapping specification表达（保留下放控制权）。编译器填补两者之间的鸿沟——自动推断parallelism、插入数据移动维持coherence、生成正确同步、优化冗余通信。这避免了CUTLASS的"程序员做所有决定"和Triton的"编译器做所有决定"两个极端，实现在正确性保证、性能控制和编程可用性之间的平衡。关键证据：Cypress的Flash Attention 3实现无需程序员标注任何同步位置——仅需将main loop重写为pipelined形式，编译器推断所有interleaved TMA/Tensor Core通信和同步。

## ThunderKittens: Simple, Fast, and Adorable Kernels

- baseline方法是什么？
  Baseline是三种GPU kernel开发范式的代表：(1) CUTLASS/CuTe——NVIDIA的C++ template embedded library，通过大量nested templates提供极致性能（如FlashAttention-3的Hopper实现），但极其复杂（22MB include目录，用户需手动管理TMA调用、barrier同步、warp specialization、memory banking以避免bank conflict），FlashAttention-3的H100移植用了两年时间，且NCU profiling显示其仍存在9.6-way bank conflict；(2) Triton——编译器方法，用户编写block-level DSL，编译器自动分解为thread-level执行，但Triton无法使用H100的wgmma/TMA等特殊硬件指令（仅支持element-wise inline PTX），难以管理异步执行和register分配，编译器heuristic决策常常次优（如不使用TMA默认、将reduction accumulator放在SMEM而非register file）；(3) CuBLAS——NVIDIA闭源手写GEMM库，>600MB包含大量预调优kernel variant和runtime heuristic选择逻辑，性能极佳但只覆盖有限算子。

  全栈执行例子（以attention forward为例，baseline = CUTLASS FlashAttention-3写法）：
  - 算法层：标准scaled dot-product attention → Q@K^T → softmax → @V
  - 系统框架层：论文未明确说明（baseline为独立CUDA kernel，无框架封装）
  - 编译框架层：CUTLASS/CuTe模板 → nvcc编译 → CUDA binary。用户需手动指定TMA copy、warpgroup矩阵乘法wgmma指令、ping-pong scheduler协调DMA warp和compute warp的双buffer轮换、shared memory layout（手动padding或swizzle避免bank conflict）、barrier位置。
  - kernel调度层：H100上执行——DMA warp通过TMA异步load K/V tiles from HBM→SMEM，compute warpgroup（4个warp=128线程）通过wgmma指令执行Q@K^T和att@V的tensor core矩阵乘，中间在register中完成softmax（max/sub/exp/sum/div），ping-pong双buffer overlap load和compute。但FA3实现面临(1) shared memory 9.6-way bank conflict（增加C_Shared），(2) ping-pong scheduler增加代码复杂度，(3) occupancy tuning需手动调节。
  - 硬件架构层：NVIDIA H100 SXM，tensor cores通过wgmma指令提供989 TFLOPS BF16算力，TMA异步数据搬运，shared memory 227KB/33TB/s，L2 cache 50MB/12TB/s，HBM 80GB/3TB/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ThunderKittens提出**一个精简的opinionated抽象集合**，通过三层GPU抽象将kernel开发简化到PyTorch级别的易用性，同时保持峰值性能：
  (1) **Warp级：16×16 tile + 自动布局管理** → 解决CUTLASS需手动选择shared memory layout避免bank conflict的痛点。TK根据tile width编译期自动从32B/64B/128B swizzle中选择最优布局（width≤32→32B/4-way, ≤64→64B/2-way, >64→128B/0-way conflict），保证与wgmma/TMA硬件指令兼容。用户只需写PyTorch风格的tile操作（mma_ABt, exp, sub_row, div_row等），TK静态检查layout兼容性（如mma_AB要求A=row-major, B=col-major，编译期报错）。
  (2) **Block级：LCSF统一异步模板** → 解决CUTLASS需手动管理同步、pipeling和warp specialization的痛点。一个LCSF模板替代FA3的ping-pong scheduler：用户在Load/Compute/Store/Finish四个函数中填充逻辑，TK自动管理multi-stage pipeline buffer（用户只需设stage数量，如GEMM用4-stage pipeline从260 TFLOPS提升到760 TFLOPS）、同步barriers（arrive机制）、TMA descriptor创建。通过调节compute worker vs load/store worker数量控制occupancy，LCSF扩展Pareto前沿超出naive同步kernel。
  (3) **Grid级：persistent grid + block order调度** → 解决block launch/setup开销和L2 cache复用不足的痛点。Persistent grid在132个SM上常驻block，通过task iteration避免重复launch/setup开销。Block launch order可通过3D stride调度提升L2 cache hit rate——以16384×16384×16384 GEMM为例，优化block order从387 TFLOPS提升到797 TFLOPS（+106%）。

  全栈执行例子（以attention forward为例，TK方法）：
  - 算法层：同一scaled dot-product attention，但以<200行TK代码实现（vs FA3的2325行CUTLASS代码）
  - 系统框架层：论文未明确说明（TK为kernel开发框架，不提供上层Serving/训练框架集成，但可通过Python binding调用）
  - 编译框架层：用户include <kittens> → 定义tile类型(st_bf shared tile, rt_bf register tile, gl HBM descriptor) → 在LCSF模板中填充producer::load（TMA异步load K/V tiles到pipeline buffer）和consumer::compute（warpgroup::mm_ABt Q@K^T → warpgroup::mma_async_wait → sub_row/exp/div_row softmax → copy→bf16 → warpgroup::mma_AB att@V → arrive input_finished）→ TK自动选择shared memory swizzle、生成TMA descriptor、管理同步 → nvcc编译
  - kernel调度层：H100上执行——load worker warp通过TMA异步预取下一tile到pipeline buffer，compute warpgroup在register和SMEM中执行mma+softmax+mfma，通过multi-stage buffer（2-stage）隐藏TMA延迟。NCU profiling显示TK：(a) tensor core利用率58.2% vs FA3 61.2%（基本持平），(b) issue slot利用率34.8% vs FA3 25.1%（+39%，更好的occupancy tuning），(c) HBM吞吐490GB/s vs FA3 328GB/s（+49%），(d) shared memory stall仅0.14 cycles vs FA3 0.92 cycles（TK zero bank conflict vs FA3 9.6-way conflict）
  - 硬件架构层：同baseline H100
  关键证据：TK的NCU profiling数据直接验证了设计目标——自动layout管理消除了FA3的shared memory bank conflict（85% fewer stall cycles），LCSF模板通过更好occupancy tuning提升了issue slot利用率，pipeline buffer隐藏了HBM延迟。

## VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

- baseline方法是什么？
  Baseline是已有的VLM视觉交互评估框架和VLM的零样本视觉决策能力。具体而言：(1) 已有benchmark（OSWorld、LIBERO、VideoGameBench、LMGame-Bench、VLABench、VLM-Gym、KORGym、Visual Agent Bench、VAGEN等）存在三大缺陷——任务数量少（4-6个非robot任务）、缺乏跨领域覆盖（domain-specific而非domain-agnostic）、诊断能力弱（仅观察性"what fails"而非受控系统性"why fails"）；(2) 前沿VLM在多步视觉决策中表现出四种系统性失败模式——受限动作空间和动作循环（action looping）、状态管理失败（state mismanagement）、提前终止（early termination）、视觉/空间信息利用失败（failure to use visual or spatial information）。

  全栈执行例子（以GPT-5在VisGym评估前，使用现有benchmark的典型零样本VLM评估流程）：
  - 算法层：VLM zero-shot推理——模型接收单张图像或简单prompt，输出文本回复或单步动作，无多步交互历史管理。
  - 系统框架层：API-based evaluation（如通过OpenRouter调用GPT-5 API），每次请求独立，无状态追踪。评价指标为单步准确率或简单成功率。
  - 编译框架层：论文未明确说明（云端推理，无编译层）。
  - Kernel调度层：论文未明确说明（云端GPU推理）。
  - 硬件架构层：论文未明确说明（云端GPU）。

  Baseline核心缺陷：
  1. **缺乏受控诊断能力**：已有benchmark只能报告"模型在某任务上失败"，无法系统性地隔离导致失败的具体因素（如context length、representation modality、feedback design、goal visibility）。
  2. **缺乏训练支持**：大多数benchmark仅支持evaluation，不提供training pipeline（demonstration生成、SFT支持、RL接口），无法将评估洞察转化为模型改进。
  3. **跨领域覆盖不足**：已有benchmark各自聚焦单一领域（robotics、computer use、games），缺乏跨symbolic puzzles→real-image understanding→navigation→manipulation的统一框架。
  4. **VLM的多步视觉决策能力未被充分理解**：前端模型在static VQA benchmarks上表现优异（MMMU、MMBench），但在interactive multi-step视觉任务中的行为特征（如context length的倒U曲线效应、文本vs视觉表示的性能翻转、目标观察的逆火效应）未被系统化揭示。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出VisGym——一个包含17个环境的gymnasium框架，通过四个协同设计解决baseline缺陷：

  **(1) 17个跨领域环境 + 细粒度可控参数**——解决"跨领域覆盖不足"和"缺乏受控诊断能力"：
  VisGym覆盖四大领域：symbolic puzzles（Matchstick Equation/Rotation, Sliding Block, Maze 2D, Patch Reassembly）、real-image understanding（Colorization, Counting, Jigsaw, Mental Rotation 2D, Referring Dot-Pointing, Video Unshuffle, Zoom-In Puzzle）、navigation（Maze 3D）、manipulation（MuJoCo Fetch Pick-and-Place/Reach）。每个环境提供2-5个可控难度参数（如maze size、puzzle pieces count、angular tolerance），支持easy/hard两档。关键诊断开关：history length（1/2/4/∞ turns）、observation modality（image vs ASCII text）、textual feedback（on/off）、goal observation（with/without）——允许对同一任务进行受控对照实验，系统性隔离各因素对性能的影响。

  **(2) 多步求解器 + Demonstration生成**——解决"缺乏训练支持"：
  为每个任务实现启发式solver（BFS/DFS/图搜索/状态机oracle），支持多策略（如Jigsaw的reorder vs swap策略）和可选随机性（通过插入可逆padding动作），生成多样化demonstration轨迹。Demonstration预处理过滤失败轨迹和data leakage。Solver使VisGym从纯evaluation框架升级为evaluation+training框架。

  **(3) Function-Conditioned Action Space + 统一Step接口**——解决"多任务统一评估"：
  动作表示为函数调用格式（如('swap', ((0,0),(0,1))), ('rotate', (30.5,20.4,15.1))），充分发挥VLM的function-calling能力。统一Step函数处理解析、验证、执行、反馈（Algorithm 1），使所有17个任务共享相同的评估协议——多轮对话、完整历史（Eq.1）、统一步数限制。

  **(4) 系统性诊断 + 失败模式自动发现**——解决"VLM行为未被充分理解"：
  通过受控诊断实验发现五个关键insight：(i) 历史长度的倒U曲线——模型受益于有限历史（~4 turns），但无限历史反降性能；(ii) 文本vs视觉表示——GPT-5在文本表示下成功率高3-4×，瓶颈在视觉grounding而非推理；(iii) 文本反馈依赖——所有模型移除文本feedback后性能一致下降，无法从纯视觉变化推断action结果；(iv) 目标观察逆火效应——提供目标图提升大多数任务性能，但当VLM视觉感知不足时（如Zoom-In Puzzle/Matchstick Equation误判概率80%/57%），反而更差；(v) 信息揭示型demonstration——在partial observability/unknown dynamics下，暴露隐藏状态的demonstration（如Matchstick Rotation先做unit-scale探索步骤再对齐）比标准demonstration显著更有效（32.9%→70.0%）。

  全栈执行对比baseline（以Qwen2.5-VL-7B在VisGym上完成Maze 3D + SFT训练为例）：
  - 算法层：VisGym Maze 3D环境——部分可观察3D迷宫，agent通过move(0)/turn(d)动作导航到target。Solver基于图搜索生成最优路径demonstration。Qwen2.5-VL-7B全参数SFT微调1500步（单任务）或5000步（多任务），仅用easy难度demonstration训练。
  - 系统框架层：VisGym基于Gymnasium框架（与MuJoCo、Atari同一底层库），扩展了function-conditioned action space和function instructions。LlamaFactory处理训练数据预处理和训练编排。OpenRouter API统一proprietary模型评估接口。
  - 编译框架层：论文未明确说明（标准PyTorch eager execution）。
  - Kernel调度层：论文未明确说明（标准GPU kernel执行）。
  - 硬件架构层：论文未明确说明具体GPU硬件。训练使用bf16精度，full-parameter fine-tuning。

  效果：SFT后Qwen2.5-VL-7B在大多数任务上达到SOTA，验证环境的可学习性和solver demonstration的有效性。Qwen3-VL-8B比Qwen2.5-VL-7B在easy→hard泛化上表现更好（近2×成功率），说明更强的基座模型提供更好的分布外泛化。LLM backbone微调贡献大于vision encoder微调（尤其在partial observability任务上），说明时序推理和历史整合是当前VLM的主要瓶颈。

  设计思路核心：VisGym的本质是将VLM评估从"domain-specific observational benchmarking"转变为"domain-agnostic controlled diagnosis + training"。三个关键设计支撑这一转变：(1) 环境多样性+参数可控性使cross-domain controlled experiments成为可能（而非per-domain case study）；(2) solver-generated demonstrations使评估洞察可以直接转化为训练数据（闭环：发现failure→生成针对性demonstration→SFT改进）；(3) 信息揭示型demonstration的发现揭示了VLM在partial observability/unknown dynamics下需要的是structured exploration（暴露隐藏状态和dynamics），而非更多的标准demonstration——这对future VLM training data curation有重要指导意义。

## Welder Scheduling Deep Learning Memory Access via Tile-graph

- baseline方法是什么？
  Baseline是现有的DNN编译器和框架，它们将DNN视为计算密集型工作负载，采用计算中心化的优化策略：

  **PyTorch [10]**（eager execution）：每个operator独立执行，Python/C++ dispatch → shape inference → kernel selection → argument preparation → kernel launch，每个operator的中间结果通过global memory传递。小模型batch=1时Python overhead主导，大batch时依赖cuBLAS/cuDNN library kernel。

  **ONNXRuntime [8]**（graph optimization）：移除Python overhead，实现pattern-based graph optimizations（如operator fusion规则），但融合仅限于预定义模式（如Conv+ReLU）。

  **Ansor [50]**（search-based compiler）：通过ML-guided search生成高性能tensor program，支持register-level element-wise fusion（如Matmul+BiasAdd、Conv2D+ReLU），但无法exploit shared-memory-level inter-operator data reuse，也无法fuse两个reduction-based operator（如Matmul+Softmax）。

  **TensorRT [7]**（vendor-specific inference library）：NVIDIA手工优化的kernel库，含expert-designed fusion rules和in-house kernels。对popular models（如BERT、Swin-T）有专门优化，但对新模型（如NAFNet）无覆盖，依赖通用kernel。

  **Rammer [31]**（horizontal fusion）：将独立并行kernel通过multi-stream调度并发执行（horizontal fusion），但不支持通过shared memory复用中间数据的dependent kernel fusion（vertical fusion）。

  **BladeDISC/AStitch [51]**（rule-based shared memory fusion）：通过预定义fusion rules对特定operator组合做shared memory fusion，但遇到不支持的operator则fallback到PyTorch runtime。

  全栈执行例子（以Ansor在V100上执行BERT attention block：Matmul Q*K → Softmax → Matmul P*V为例）：
  - 算法层：标准Transformer attention——Q,K,V projection (Matmul) → Q*K^T (Matmul) → Softmax → P*V (Matmul) → O projection (Matmul)
  - 系统框架层：PyTorch trace → ONNX export → Ansor编译。Ansor对每个operator独立tune（800 trial per operator），生成optimized kernel。
  - 编译框架层：Ansor search-based compiler——program sampling → ML cost model training → top-k performance evaluation → final kernel selection。Register-level fusion rules：Matmul+BiasAdd、Conv2D+ReLU等element-wise融合。无法fuse Matmul+Softmax（两者都是reduction-based operator，tile shape冲突）。
  - Kernel调度层：每个operator独立kernel launch。Matmul Q*K^T kernel：从DRAM加载Q tile和K tile → shared memory → TensorCore MMA → 输出C [seq_len×seq_len] 写入DRAM。Softmax kernel：从DRAM读C → 执行softmax → 写回DRAM。Matmul P*V kernel：从DRAM读P和V → TensorCore MMA → 写入DRAM。中间tensor C和P在DRAM中完整物化，global memory traffic高。
  - 硬件架构层：NVIDIA V100 GPU (16GB, SIMT Core + TensorCore)。Global memory bandwidth 900 GB/s。Ansor kernel执行：Matmul Q*K^T 占用TensorCore但shared memory仅用于intra-operator tiling，无法与Softmax复用。Softmax受限于memory bandwidth（大中间结果从DRAM读取）。

  Baseline核心缺陷：
  1. **计算中心化思维**：将DNN视为compute-intensive，实际上现代DNN（ViT、Conformer、NeRF、NAFNet等）memory bandwidth utilization高达96.7%而compute utilization仅51.6%。基线优化器仍聚焦于加速计算而非减少内存访问。
  2. **缺乏跨算子内存复用**：operator间的中间tensor必须完整物化在global memory中，造成大量inter-operator DRAM traffic。Ansor/TVM的fusion仅覆盖register-level的element-wise算子对（如Conv+ReLU），无法处理含reduction的fusion（如Matmul+Softmax）。
  3. **融合规则vs通用性trade-off**：TensorRT/BladeDISC的expert-designed fusion rules对不支持的算子组合完全失效。缺乏一套通用的、以内存为中心的优化框架。
  4. **tile shape冲突未解决**：不同算子的最优tile shape不同（如Matmul的[32×64] vs Softmax的[4×128]），独立优化时无法在shared memory中复用中间数据。现有方法要么放弃shared memory复用，要么依赖有限的手工规则。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  WELDER提出以**内存访问优化为核心**的DNN编译框架，核心创新是**tile-graph抽象**——将DNN计算从operator级dataflow graph下沉到tile级dataflow graph，从而暴露operator内部tile间的细粒度数据复用机会。

  **方法一：Tile-Graph + Tile Propagation** → 解决"tile shape冲突"
  - 将DNN计算建模为operator-tile组成的数据流图。每个operator-tile处理一个输出数据tile。
  - 通过SetConnect接口在两个相邻operator-tile之间建立tile级连接（指定data reuse所在的memory level），通过Propagate接口从output tile shape链式推断整个tile-graph的所有tile配置。
  - 关键insight：给定operator的tensor expression（准确的输入→输出映射），可以从output tile shape反向推导其依赖的input tile shape。因此，只要两个相邻operator共享一个连接的output/input tile shape，它们的tile配置自动对齐——这是Matmul+Softmax得以在shared memory中复用的数学基础。

  **方法二：Inter-Layer Independence + Traffic Cost Model** → 解决"优化空间爆炸"
  - 发现total memory traffic仅由当前memory layer的output tile配置决定（inter-layer independence）。这允许将整个DNN的多层memory hierarchy优化空间解耦为多个独立子空间。
  - Traffic cost model：Σ(input_tile_sizes + output_tile_size) × num_tile_graphs。基于tile size和tensor shape解析计算，无需实际执行。
  - 两层调度算法：外层Graph Connecting枚举每条edge的连接层（L0/L1/L2），内层Sub-Graph Tiling基于traffic cost model搜索最优tile配置。MemFootprint检查容量约束，MemTraffic作为排序键。

  **方法三：Hierarchical Tile-Graph + 四条硬件抽象** → 解决"通用性"
  - 将优化后的执行计划递归编译为分层tile-graph——从最低memory level开始，递归分裂为上层sub-graph。
  - 仅需四条硬件接口（Allocate/LoadTiles/ComputeTile/StoreTiles）即可映射到任意有层级memory的加速器（CUDA GPU、ROCm GPU、GraphCore IPU）。
  - 无需per-operator-type fusion rules——所有operator只要能用tensor expression描述即可参与tile-graph优化。

  全栈执行例子（以WELDER在V100上执行同一BERT attention block：Matmul Q*K → Softmax fusion为例）：
  - 算法层：同baseline。WELDER不改变模型算法，仅改变数据在memory hierarchy中的流动方式。
  - 系统框架层：PyTorch trace → ONNX export → WELDER tile-graph compiler。编译过程：常量折叠 → operator-tile decomposition → SetConnect for each edge（枚举L0/L1/L2）→ Propagate tile shapes → Traffic cost evaluation → Profile best configs → Code Generation。
  - 编译框架层（核心差异）：WELDER替代了Ansor的per-operator tiling + rule-based fusion + ML cost model。Graph Connecting决定Matmul→Softmax edge在shared memory (L1)连接。Propagate从output tile [BM×BN] 反向推断Matmul的input tile [BM×BK] 和 [BK×BN]。Traffic cost model计算不同tile shape的global memory traffic，自动选择最优配置（如 [16×128] vs [4×128] vs [32×64]）。Hardware-aligned penalty过滤uncoalesced/inadequate parallel/over-capacity配置。最终选择 [16×128] output tile → 264MB total traffic（vs unfused 840MB）。
  - Kernel调度层（核心差异）：替代Ansor的两次独立kernel launch + DRAM中间物化。WELDER fused kernel：LoadTiles从DRAM加载Q tile [BM×BK] 和K tile [BK×BN] → shared memory buffer 0,1 → ComputeTile: Matmul operator-tile via TensorCore MMA (warp-level) → 中间结果 C_tile [BM×BN] 直接留在shared memory ← SetConnect at L1 → ComputeTile: Softmax operator-tile从shared memory读取C_tile → 执行softmax (SIMT) → StoreTiles将结果D_tile写回DRAM。循环覆盖全部tiles。消除了一次global memory write (C) + 一次global memory read (C) 的DRAM往返。
  - 硬件架构层：NVIDIA V100 GPU。从"Matmul TensorCore→DRAM→Softmax SIMT"的两段执行变为"Matmul TensorCore→Shared Memory→Softmax SIMT"的单kernel流水线。Global memory traffic 840MB→264MB (saving 69%)。Kernel launch count减半（单kernel fused vs 两个独立kernel）。1.26× speedup on Matmul-Softmax pair。

  扩展到更大范围：
  - BERT attention：Q*K fused with Softmax（seq_len=128时）。当seq_len=512（Conformer）时auto decision不fuse。
  - NeRF 7-layer MLP：full auto-fusion to single GPU kernel（前6层TensorCore + 输出层SIMT Core），全部中间结果存shared memory，5× speedup。
  - NAFNet：back-to-back pointwise convolutions fused with normalization。Auto decide fusion order（top layers: DWConv+PWConv cache feature map in shared memory；bottom layers: PWConv+DWConv cache full channel）。
  - 89种非常规fusion pattern自动发现：含Dual Matmul + Relu chain (13 ops)、48-operator fusion chain（DepthwiseConv+Broadcast+Divide+Erf+Multiply+Convolution × multiple cycles + Concat）。

  设计思路核心：WELDER将DNN内存优化问题从"为每对算子设计专门的fusion规则"转变为"在tile-graph上搜索最优tile连接配置"的单一通用优化问题。三个关键insight支撑这一转变：(1) tile propagation——给定tensor expression，output tile shape可以唯一确定所有input tile shape，自动对齐operators之间的tile配置；(2) traffic cost model——给定aligned tile config，traffic可通过解析计算（无需实际执行），且traffic仅依赖当前layer的tile config（inter-layer independence）；(3) 分层编译——通过SetConnect/Propagate两条原语和两条cost接口(MemFootprint/MemTraffic)，整个优化可递归分解为各memory layer的独立子问题。这本质上将operator fusion从"rule engineering problem"消解为"graph optimization problem"。

## vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

- baseline方法是什么？
  Baseline是HuggingFace Transformers的默认实现（Qwen-Omni系列），以及其他模型原始实现（BAGEL、MiMo-Audio）或Diffusers库（Diffusion模型）。核心问题：Existing LLM serving frameworks（vLLM、SGLang）使用**step-centric abstraction**——将模型推理抽象为单个forward function，框架内部封装iteration logic和KV cache management。这种抽象专为text-only LLM的单AR decoding设计。

  Baseline全栈执行例子（以Qwen3-Omni Thinker-Talker-Vocoder pipeline，2×80GB accelerators为例）：
  - 算法层：Thinker-Talker双AR LLM + Vocoder架构。Thinker (30B LLM)生成text tokens + hidden states → Talker (smaller LLM)生成audio codec tokens → Vocoder (DiT/CNN)生成audio waveforms。
  - 系统框架层：HuggingFace Transformers默认实现。开发者需手动实现pipeline：(1) 对Thinker实现custom generate()函数 → (2) 提取output hidden states → (3) 编排cross-stage transfer（将hidden states转为Talker input embeddings）→ (4) Talker custom generate() → (5) 提取codec tokens → (6) Vocoder generate waveforms。每个stage独立运行自己的generate loop。
  - 编译框架层：无execution graph compilation优化，kernel launch overhead大。
  - Kernel调度层：无continuous batching——每个request的Thinker和Talker分别独立运行generate loop。无chunked prefill——长prompt全量prefill。无paged attention——KV cache management低效。Thinker/Talker/Vocoder串行执行，后一stage必须等前一stage完全完成。
  - 硬件架构层：2× 80GB accelerator。所有stage co-located在单进程中，无法fine-grained resource allocation。Thinker(30B)占用大量memory和compute，Talker和Vocoder资源受限。计算资源利用不充分——Thinker compute-bound阶段GPU空闲时Talker无法开始。

  Baseline两大核心缺陷：
  1. **Step-centric抽象无法表示multi-stage pipeline**：vLLM/SGLang的step-centric interface只能封装单个AR decoding的forward+iteration loop。Qwen3-Omni的Thinker→Talker→Vocoder三层pipeline及其cross-stage dependency（Talker每步decoding需连接Thinker hidden states）超出了单forward function的表达能力。开发者被迫在serving framework外手动实现inter-stage transfer，丢失所有framework-level优化。
  2. **Monolithic execution导致资源分配低效**：所有stage co-located在同一进程中，computing resources无法按stage需求灵活分配。Thinker(30B)需大量memory，Talker compute-intensive需更多parallelism——两阶段需求相互冲突。且计算资源在stage间无法concurrent利用——pipeline串行，下一stage等待上一stage完全完成。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出vLLM-Omni，核心创新为**Stage Abstraction + Disaggregated Stage Execution**，将any-to-any pipeline从monolithic execution解耦为independently served stages：

  **(1) Stage Graph Abstraction**——解决"step-centric abstraction无法表示multi-stage pipeline"缺陷：
  将any-to-any model定义为stage graph G=(V,E)。V中每个node为独立stage，需实现：(a) forward function——step-centric batched forward（兼容vLLM的iteration loop优化）；(b) preprocess function——每iteration调用，modify stage input以incorporate upstream data。E中每条edge定义stage-transfer function——控制query states和intermediate data如何在stage间转换。Stage graph原生表达多stage pipeline（如Thinker→Talker→Vocoder），users仅实现per-stage forward/preprocess/transfer逻辑，无需手动管理batching/scheduling。

  **(2) Disaggregated Stage Execution Backend**——解决"monolithic execution资源分配低效"缺陷：
  每个stage由独立execution engine serving：
  - AR stages → vLLM engine（继承continuous batching、chunked prefill、PagedAttention KV cache management、execution graph compilation）
  - DiT stages → 专用diffusion engine（flash attention、SAGE attention、TurboAttention、TeaCache/cache-dit caching、RingAttention/Ulysses parallelism）
  - Per-stage independent scheduling、KV cache management、model execution
  - Flexible per-stage GPU allocation：Thinker(30B)分配更多memory + TP-2 across both accelerators；Talker分配更少memory但更高parallelism；Vocoder分配独立device
  - Stage间通过Unified Connector数据传输：shared memory（单节点, 5.49ms Thinker→Talker）或Mooncake RDMA（跨节点, 8.28ms）

  **(3) Streaming Stage Output**——解决pipeline串行等待问题：
  下游stage在上游未完全完成时即开始incremental处理。Talker产出initial tokens时Vocoder即可开始denoising，减少TTFT和enabling streaming responses。

  vLLM-Omni方法全栈执行例子（以Qwen3-Omni同一请求为例）：
  - 算法层：同Thinker-Talker-Vocoder架构，但执行方式完全不同。
  - 系统框架层：vLLM-Omni stage graph编程——users定义thinker_forward/talker_forward/dit_decode（forward functions）、mm_encode/process_input（preprocess functions）、Thinker2Talker/Talker2Vocoder（transfer functions）。Orchestrator管理请求routing through stage graph。每个engine独立配置parallelism policy和memory budget。
  - 编译框架层：vLLM engine复用execution graph compilation（Qwen3-Omni Thinker获12.97× speedup的主要来源之一）。Diffusion engine集成flash attention kernel compilation。
  - Kernel调度层：
    Thinker stage (TP-2, device-0+device-1):
      continuous batching + chunked prefill + paged attention
      → Thinker generate text tokens + hidden states (150.9 avg output tokens)
    Talker stage (device-1):
      每decode iteration: preprocess concatenate Thinker hidden states + Talker embeddings
      → Talker generate audio codec tokens (545.4 avg output tokens)
      → streaming to Vocoder
    Vocoder stage (device-0):
      incremental DiT denoising with TeaCache caching
      → final audio waveforms
    Stage间并行：当Request_1的Talker在device-1 decode时，Request_2的Thinker在device-0 prefill——compute和memory跨stage、跨请求自然重叠。
  - 硬件架构层：2× 80GB accelerator。Thinker TP-2 spread across both devices，Talker和Vocoder各占一个device。Per-stage independent GPU allocation最大化memory和compute利用率。Unified connector overhead negligible（<0.1% of total latency）。

  设计思路核心：
  vLLM-Omni的本质是将any-to-any model serving从"application-level manual orchestration"下沉为"framework-level automatic disaggregation"。关键洞察是**complex multimodal architectures can be decomposed into modular stages, each of which is just a standard AR or DiT component**——这些component本身可以被existing serving engines高效执行，难点在于stage graph的表达和执行。通过提供stage abstraction前端（表达任意topology）+ disaggregated execution后端（各stage独立优化），vLLM-Omni使得"支持任意any-to-any model"成为可能——从Thinker-Talker双AR（Qwen-Omni）到AR+DiT（BAGEL, GLM-Image）到纯DiT（Qwen-Image, Wan2.2），所有architecture均可用同一stage graph范式表达和执行。实验结果中Qwen3-Omni JCT降低91.4%的根本原因在于：baseline缺失的性能优化（continuous batching、chunked prefill、execution graph compilation）在vLLM-Omni中通过解耦stages自然获得，且30B Thinker相比7B Thinker(Qwen2.5-Omni)能更充分摊销优化pipeline，实现超线性加速比（91.4% vs 61.6%）。

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

## LMFusion: Adapting Pretrained Language Models for Multimodal Generation

- baseline方法是什么？
  Baseline是Transfusion（Zhou et al., 2024），一种从头训练的unified multimodal model。Transfusion使用标准Transformer架构（与主流LLM如Llama相同），通过end-to-end训练同时学习语言建模（next-token prediction with cross-entropy loss）和图像扩散（DDPM loss on continuous image latents）。架构特点：所有参数跨模态共享——同一套QKV、O、FFN同时处理文本token和图像patch。训练数据包含language-only text data（0.25T tokens）和image-caption pairs（0.25T image tokens）。虽然架构统一，但存在两个核心缺陷：
  1. **计算资源浪费**：训练从头开始需要大量language-only data维持语言能力，即使已有强大的预训练text-only LLM（如Llama-3 8B已训练15T+ tokens），仍需重新学习语言知识。Transfusion 7B在language-only benchmarks上比Llama-3 8B低11.6%（HellaSwag 51.0 vs 60.0），说明从头训练的多模态模型语言能力不如专用text-only LLM。
  2. **naive finetuning导致灾难性遗忘**：直接在Llama-3上继续用Transfusion recipe训练（dense finetuning），会导致语言能力显著退化——HellaSwag下降15%（初始阶段），即使后续有所恢复，仍存在~7%的永久性差距。

  Baseline全栈执行例子（以Transfusion 7B从头训练后生成"a cat with secrets to keep"对应图像为例）：
  - 算法层：Transfusion统一Transformer + U-Net下采样/上采样。文本token和图像patch交替排列为一个长序列，共享QKV/FFN/O参数处理。文本使用因果mask（autoregressive），图像使用双向mask（diffusion）。训练目标L = L_LM + λ·L_DDPM。从头训练需要language-only data + image-caption data混合。
  - 系统框架层：论文未明确说明训练框架。基于标准PyTorch + FSDP/DeepSpeed分布式训练，使用标准Transformer训练pipeline。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。标准Transformer attention kernel + U-Net convolution kernel，无特殊kernel优化。
  - 硬件架构层：论文未明确说明硬件平台。训练7B+规模多模态模型通常需要数百张H100/H800，FLOPs规模极大。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LMFusion，核心设计：(1) 模态特异性模块分离（modality-specific QKV/FFN/LayerNorm）将文本和图像处理路径解耦；(2) 共享自注意力层保持跨模态信息融合；(3) 文本模块冻结（η_text=0）仅训练图像模块。

  **解决"计算资源浪费"缺陷**：通过冻结Llama-3文本模块并排除训练数据中的language-only data，LMFusion用0.5× total FLOPs即达到甚至超越Transfusion的性能。具体而言，Transfusion需要0.25T text tokens + 0.25T image tokens训练，而LMFusion仅需0.25T image tokens（文本模块已预训练）。在匹配0.25T image data的情况下，LMFusion比Transfusion在image understanding上高20%（CIDEr 38.3 vs 32.0），image generation FID好3.6%（13.9 vs 14.4），language benchmarks保持Llama-3原水平。这证明**预训练LLM的语言知识可以通过冻结+模态分离无损复用于多模态任务**，无需重新学习。

  **解决"naive finetuning灾难性遗忘"缺陷**：通过deep modality separation（QKV+FFN都分离）+ 文本模块冻结（lr_ratio=0），LMFusion完全避免了语言能力的退化。Ablation实验证实：(a) 无分离（dense model）finetune时即使降低文本lr（lr_ratio=0.1），语言能力仍有2%退化且image性能受影响；(b) 浅层分离（仅FFN分离）能减轻但不足以消除退化；(c) 深层分离（QKV+FFN都分离）+ 冻结文本模块，在保持语言能力的同时image understanding/generation性能even超越全参数调优的dense模型（Figure 5 vs Figure 4）。深层原因：当文本和图像共享QKV参数时，image diffusion的梯度会通过attention层反向传播到文本参数的优化空间，干扰文本表征——文本token的Q/K/V被"拉向"适应图像噪声预测的方向，破坏了其在语言任务上的有效表征。

  **模态特异性QKV的深层作用**：看似是"增加参数"的朴素设计，实际解决了多模态训练中的**梯度冲突（gradient conflict）**问题。在dense模型中，一个token的QKV同时被LM loss和DDPM loss更新——这两个loss的梯度方向可能矛盾：LM loss要求text token的attention pattern保持语言上的coherence（更关注语义相关的token），而DDPM loss要求image patch的attention关注视觉上相关的区域。模态特异性QKV使这两种优化目标在独立的参数空间中完成，消除了梯度冲突。

  LMFusion方法全栈执行例子（以相同"a cat with secrets to keep"→图像生成为例）：
  - 算法层：Llama-3 8B text modules（冻结）+ 并行image modules（从Llama-3初始化，可训练）+ U-Net down/up（0.27B，从头训练）。文本token通过Proj_text → QKV_text → 在共享attention中text Q attend到[K_img, K_txt] → O_text → FFN_text → LM_Head_text。图像patch通过UNet_Down_img → QKV_img → 在共享attention中image Q attend到[K_txt, K_img] → O_img → FFN_img → UNet_Up_img。仅image路径参数有梯度更新。关键：文本和图像在attention层有双向cross-modal交互，但由于QKV分离，两者的attention计算是独立的——text的attention不改变image的QKV参数，反之亦然。这比完全独立的两个模型（no cross-modal at all）更优越，因为共享attention允许text context condition image diffusion（文本条件在去噪每一步都参与）。
  - 系统框架层：论文未明确说明。基于标准PyTorch训练，使用AdamW optimizer (β1=0.9, β2=0.95)，cosine decay LR schedule with 4000-step warmup，η_img=1e-4→1.5e-5。Image data 380M Shutterstock captions，80% caption→image顺序。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。与Transfusion相同——标准Transformer attention + U-Net卷积分支，无特殊kernel优化。注意：虽然LMFusion参数量是Transfusion的2倍（两套QKV/FFN），但每个token仅激活对应模态的模块（一半参数），因此每次前向的FLOPs与Transfusion相同。
  - 硬件架构层：论文未明确说明硬件平台。

  设计思路核心：
  LMFusion的本质洞察是**模态特异性参数化不是"增加参数量"的成本支出，而是"消除梯度冲突"的架构投资**。传统观点认为MoE或模态分离是computation-vs-capacity的trade-off（更多参数=更多计算），但LMFusion证明了在multimodal generation场景下，模态分离反而比dense model在**同等可训练参数量和同等FLOPs**下表现更好。深层原因：文本和图像的优化动态（optimization dynamics）在本质上是不同的——文本是离散token的自回归生成（需要精确的next-token probability），图像是连续latent的扩散去噪（需要smooth的噪声预测）。当两者共享参数时，这两种动态相互干扰（gradient interference）；分离参数使每种模态在其自身的优化空间中自由演化。LMFusion的扩展（LLaVAFusion）进一步验证了这一原则的普适性——同样的冻结+分离范式可以直接应用于已有多模态理解能力的VLM（LLaVA-NeXT），在不损失其已有能力的前提下赋予图像生成能力。

## TileLang: A Composable Tiled Programming Model for AI Systems

- baseline方法是什么？
  **Baseline方法有两类：编译器/DSL方法（Triton, TVM）和手写kernel方法（FlashAttention-3, FlashMLA, AITER, Marlin等）。**

  **Triton [20]**（tile-level compiler）：提供block-level编程原语但隐藏thread行为、memory layout和address-space annotations于自动生成的策略之后。对专家开发者存在三个关键痛点：(1) **无法自定义memory layout**——Triton的tl.dot等向量化操作不支持用户自定义PTX/custom tile operator注册，对于量化weight的GEMM kernel，无法实现硬件对齐的自定义data layout；(2) **Pipeline控制受限**——Triton仅暴露num_stages参数，不允许用户定义完全自定义的pipeline（如warp specialization）；(3) **低精度支持不完善**——sub-byte类型操作需要通过uint32 bitwise手工解包，且解包后的register layout与Tensor Core要求的MMA layout不兼容，需通过shared memory做layout conversion（成为性能瓶颈）。

  **TVM [7, 8]**（schedule-oriented compiler）：要求用户显式区分computation和schedule，需手动注册新tensor指令和指定buffer layout。schedule程序的编写和理解困难，且primitive-style scheduling不支持现代GPU的关键优化（如cp.async/TMA based software pipelining）。

  **手写kernel方法**：FlashAttention-3使用手写CUDA（TMA + wgmma.mma_async + warp specialization），但固定tile sizes导致对小sequence length suboptimal。FlashMLA同样是手写CUDA kernel。这些方法的code line数多（FlashMLA的手写实现远超TileLang的~70行Python），且通用性差。

  **Baseline全栈执行例子（以Triton FP16 GEMM, H100 GPU为例）：**
  - 算法层：标准FP16矩阵乘法 C[M,N] = A[M,K] × B[K,N]，FP32 accumulate
  - 系统框架层：Triton kernel，@triton.jit编译，auto-tuning搜索tile大小和num_stages
  - 编译框架层：Triton将Python kernel编译为PTX → SASS。Thread binding和memory layout由Triton编译器自动决定，用户无法干预。Pipeline仅通过num_stages参数控制overlap深度
  - Kernel调度层：Triton自动生成global→shared copy（cp.async）+ shared→register load + Tensor Core MMA。但thread-level的register分配、shared memory bank conflict避免策略均由compiler heuristics决定，无用户控制。对于量化GEMM：load packed uint32 → bitwise unpack in registers → store to shared memory for layout conversion → ldmatrix reload → MMA——shared memory layout conversion是额外开销
  - 硬件架构层：NVIDIA H100 (Hopper)，Tensor Core wgmma.mma_async，TMA hardware unit

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **TileLang方法：Python-embedded tiled DSL + JIT compiler，核心创新是解耦dataflow与scheduling space，将thread binding、memory layout、tensorization、pipeline封装为composable annotations和primitives。**

  **解决Triton的"隐式scheduling"限制**：TileLang显式暴露四种scheduling space为用户可控的annotations/primitives：
  1) **Thread Binding**: 通过Layout Inference Pass自动推断，但用户可通过T.Fragment手动指定thread→buffer映射。Fragment Layout (f: K^n → K²) 精确描述block级register file到thread的partitioning。
  2) **Memory Layout**: T.annotate_layout允许用户自定义shared/global memory layout（如自定义swizzle模式）。Layout抽象基于composable IterVar algebra (f: K^n → K^m)，支持stacking和composition（图5）。T.use_swizzle一键启用L2 cache友好的swizzle thread block ordering。
  3) **Tensorization**: 两种路径——Tile Library-based (CUTLASS cute/AMD CK, T.call_extern) 和 Direct PTX injection (T.ptx)。用户可直接注册custom tile operator（Python中定义Lower和InferLayout接口）。对于专家：可通过T.import_source + T.call_extern注入C++ template实现的DP4A/MMA等指令（图10a）。
  4) **Pipeline**: T.Pipelined(num_stages=N)自动推导pipeline schedule，同时允许用户显式指定producer/consumer order。自动支持Ampere cp.async、Hopper TMA+warp specialization、AMD CDNA async DMA。

  **解决Triton的"低精度layout conversion瓶颈"**：TileLang的Layout Inference Pass自动为GEMM的A_shared/B_shared应用MakeSwizzleLayout（消除bank conflict），为C_local应用MakeMMASTMatrixLayout（Tensor Core要求的register layout）。在Dequantized Matmul中（图17），weight以packed u8形式加载到register，通过View做零开销类型reinterpret（u8→i4）+ layout transform（tile layout→MMA layout），完全在寄存器内完成，消除Triton的shared memory layout conversion额外往返。

  **解决TVM的"schedule编程困难"**：TileLang采用dataflow-centric编程范式——用户仅需描述tile-level dataflow（T.copy, T.gemm, T.reduce, T.atomic），编译器自动完成所有scheduling推导。仅当默认优化不够时，用户才通过annotations精准控制。

  **解决手写kernel的"通用性缺失"**：所有kernel（GEMM, FlashAttention, Linear Attention, MLA, Dequantized Matmul）共享同一TileLang编程范式。FlashAttention仅需~70行Python代码即达FlashAttention-3的98%性能。Dequantized Matmul通过同一程序模板参数化支持INT2/INT4/NF4等多种量化格式。

  **TileLang方法全栈执行例子（FP16 GEMM, H100 GPU，对应图11）：**
  - 算法层：标准FP16矩阵乘法 C[M,N] = A[M,K] × B[K,N]，FP32 accumulate
  - 系统框架层：TileLang Python程序（~30行），@tilelang.jit decorator，tilelang.compile(program, target="cuda")
  - 编译框架层：五阶段pipeline——Parser (Python AST → TileLang AST) → IR Builder (→ TVM Tensor IR) → Optimization (Layout Inference + Thread Binding + Pipeline Derivation + Vectorization) → Codegen (→ CUDA C with TMA/wgmma.mma_async instructions) → nvcc → binary
  - Kernel调度层：
    a) T.Kernel(N//block_N, M//block_M, threads=128) → grid (N/128, M/128), block (128 threads)
    b) T.alloc_shared → shared memory buffers (A_shared[128,32] f16, B_shared[32,128] f16)
    c) T.alloc_fragment → register file C_local[128,128]（block-level allocation）
    d) Layout Inference: Gemm(priority=highest) → A_shared=SwizzleLayout, B_shared=SwizzleLayout, C_local=MMA_MatrixLayout → Copy(priority=lower) → auto parallelize + vectorize
    e) Thread Binding: C_local[128,128]通过Fragment Layout分发到128个threads（2 warps × 64 threads），每个thread持有部分register elements
    f) T.Pipelined(K // 32, num_stages=2): 推导Copy-GEMM interleaved pipeline → Hopper自动TMA + wgmma.mma_async + warp specialization（producer: TMA copy, consumer: wgmma.mma_async, mbarrier同步）
    g) Loop body执行: TMA load A/B tiles → mbarrier.arrive → mbarrier.wait → wgmma.mma_async(A_shared, B_shared, C_local) → 循环K维 → T.copy(C_local → C[global]) with thread binding + vectorized store
  - 硬件架构层：NVIDIA H100 (Hopper)，TMA hardware unit（异步global↔shared copy），wgmma.mma_async warp-group MMA，mbarrier同步

  对比Triton baseline：TileLang在同等简洁语法下（~30行 vs Triton ~25行 GEMM），实现了(vs Triton) 1.13× speedup on H100，通过custom swizzle layout消除shared memory bank conflict，通过TMA+warp specialization实现更高效的pipeline overlap。关键差异在于TileLang将scheduling从dataflow中解耦，让编译器在更大搜索空间中自动寻找最优调度。

