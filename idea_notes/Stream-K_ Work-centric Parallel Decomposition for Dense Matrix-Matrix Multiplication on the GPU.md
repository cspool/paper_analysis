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
