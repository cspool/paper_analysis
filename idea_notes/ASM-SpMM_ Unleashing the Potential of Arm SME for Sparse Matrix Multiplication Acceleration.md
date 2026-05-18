## ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

- baseline方法是什么？
  Baseline方法分为三类：(1) **ARM CPU通用SpMM库**：ArmPL v24.10、Armadillo v14.6.0、SuiteSparse Cholmod v5.3.3、Eigen v3.4.0、MP-SpMM，主要依赖通用多核和vector unit（SVE/Neon），无法使用SME matrix unit的outer-product算力。全栈执行例子：算法层稀疏矩阵乘法A×B→系统框架层调用ArmPL/Armadillo等BLAS库→编译框架层Clang编译但无SME指令生成→kernel调度层CSR/CSC等通用稀疏格式+SIMD vector kernel→硬件架构层ARM CPU P-core/E-core上的SVE/Neon vector unit。缺陷：SME matrix unit的ZA register和MOPA outer-product指令完全未被利用，导致理论算力浪费。(2) **GPU Tensor Core SpMM方法**：TCF、ME-TCF、DTC-LSH、AccOrder、TC-GNN等依赖inner-product、warp调度和GPU memory hierarchy，并常依赖固定block alignment、left-aligned tile和zero padding。全栈执行例子：算法层SpMM→系统框架层CUDA/cuSPARSE→编译框架层NVCC→kernel调度层Tensor Core MMA with block sparse format→硬件架构层GPU Tensor Cores。缺陷：这些格式和调度策略假设inner-product和GPU warp-level并行，与SME的predicate-driven vector outer-product不一致；固定block alignment导致冗余zero padding浪费SME的稀疏算力。(3) **静态负载均衡**：按行、非零元或core能力预分配任务。全栈执行例子：kernel调度层静态row partitioning将A的行均匀分配给各core→硬件架构层Apple M4上P-core和E-core共享SME unit。缺陷：在Apple M4这类P-core/E-core异构且SME单元按cluster共享的架构上，静态分配忽略非零分布不均和core速度差异，容易导致负载失衡。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**ASM-SpMM**：面向ARM SME的高性能SpMM库，通过OP-MCF稀疏格式+SME outer-product microkernel+multi-tile并发+prefetch pipeline+SVE/Neon混合执行+hetero-core动态work stealing调度六层协同设计。

  论文方法全栈执行例子（以Apple M4上GCN inference的SpMM为例）：
  - 算法pipeline：GCN layer中的稀疏邻接矩阵A × 稠密特征矩阵W的SpMM操作→ASM-SpMM替换PyG/DGL的默认SpMM实现。
  - 系统框架层：论文未明确说明（ASM-SpMM以library/kernel形式被GNN framework调用）。
  - 编译框架层：Clang 16.0编译，使用ARM SME intrinsics（svld1_f32/svmopa_za32_f32_m/svst1_hor_za32/_svprfw等）而非依赖compiler自动生成SME指令。
  - kernel调度层：OP-MCF格式转换→row window划分→compressed slot遍历→每个slot：svld1_f32加载sparse values到Z register→ColumnPositionMaskBit转predicate register→vectorized load dense B tile→svmopa_za32_f32_m outer product accumulate to ZA tile→循环间_svprfw预取下一slot的data/mask/column index/dense B fragments→多ZA tile并发→低密度block分配SVE/Neon vector path→interleaved scheduling隐藏vector latency。
  - 硬件架构层：Apple M4 CPU（512-bit SVL，8 double/window），ZA matrix register做outer-product累加，P-core cluster和E-core cluster各一个SME compute unit，predicate register控制有效非零计算。

  **对应解决Baseline缺陷的映射：**
  **(1) ARM CPU库无法使用SME outer-product算力 → OP-MCF格式+SME outer-product microkernel**：ArmPL/Armadillo等使用CSR+SIMD vector kernel，SME ZA register闲置。ASM-SpMM的OP-MCF按SME vector length切row window，column compaction将非重叠稀疏列合并为一个compressed slot，bitmask还原原始列有效行，使SME predicate register只参与有效非零计算；svmopa outer-product直接利用ZA tile做sparse value向量×dense tile外积累加，避免把SME当成普通SIMD。M4上geomean speedup 11.81× over ArmPL、15.12× over Armadillo、18.62× over Eigen。

  **(2) GPU Tensor Core格式不匹配SME → OP-MCF去掉硬性block padding**：TCF/ME-TCF等依赖fixed block alignment和left-aligned tile，SME上zero padding浪费算力。OP-MCF的masked multi-column merging消除padding——非重叠列合并为一个physical column无需对齐，mean NNZ per slot达到约4-6（vs CSR约1-2），显著提高SME外积的每slot有效计算密度。

  **(3) SpMM访存不规则导致cache miss → multi-tile并发+显式prefetch pipeline**：CSR/普通sparse kernel的LLC miss rate约30%-61%。ASM-SpMM用多ZA tile并发提高ZA/Z register占用率，_svprfw类prefetch指令显式预取sparse/dense operand将LLC miss rate降至23%-48%，prefetch pipeline隐藏不规则访存延迟。

  **(4) 低密度block在SME上利用率低 → SVE/Neon混合matrix-vector kernel**：稀疏尾部或碎片化block若强行走SME path会因padding浪费周期。ASM-SpMM将最稀疏block分配给SVE/Neon vector unit，通过interleaved instruction scheduling使vector工作量隐藏在SME固定执行窗口内。hybrid kernel在rCA、FY-RSR、ddi、ppi上比matrix-only快8%-18%。

  **(5) 异构多核静态负载均衡失衡 → hardware-aware task mapping+动态work stealing**：Apple M4有P-core/E-core异构且SME unit按cluster共享，静态分配忽略非零分布和core速度差异。ASM-SpMM runtime先做hardware-aware task mapping分配row window，再用progress monitoring检测core完成进度，从负载较重core窃取row window实现动态再平衡。LX2上12线程相对2线程达8x-11x scaling。

  Trade-off：(1) OP-MCF需要一次性格式转换，适合同一稀疏矩阵被重复执行的场景（GNN inference、迭代solver、超参搜索）；若矩阵频繁变化且复用次数很少，转换成本可能更难摊销。(2) 混合SME/SVE调度提升吞吐但引入register partition、资源竞争和workload partition开销，hybrid/theory仅0.78-0.90。(3) 设计依赖ARM SME硬件可用性，不同SME平台需调整tile size、resource allocation和调度参数。(4) ASM-SpMM面向FP64（论文在Apple M4和LX2上均以FP64为主评估），FP32/FP16等低精度下的OP-MCF和SME kernel行为需独立验证。
