## Composing Distributed Computations Through Task and Kernel Fusion

- baseline方法是什么？
  **标准分布式 task-based runtime 执行模型（不进行跨 task 融合）**：高层分布式库（如 cuPyNumeric, Legate Sparse）将每个库操作分解为独立的 index task 序列，每个 task 内部执行嵌套循环（kernel），task 通过 runtime 管理的分布式数据 collection（region）进行通信。每个中间操作结果分配为独立的分布式数组。不同库操作之间的 task 隔离开，不进行跨 task 融合。

  全栈执行例子（cuPyNumeric 5-point stencil，4 nodes 4 GPUs，Figure 1）：
  - **模型推理算法层**：论文未涉及 ML 推理（科学计算场景）。5-point stencil: avg = center + north + east + west + south; work = 0.2 * avg; center[:] = work。
  - **系统框架层**：cuPyNumeric 将每个 NumPy 操作（ADD, MULT, COPY）映射为独立 index task launch。每次调用 np.add/np.multiply 时，cuPyNumeric 创建临时 distributed array（如 t1, t2, t3, avg），然后发射独立的 index task 计算该操作。task 按程序顺序发射到 Legion runtime，Legion 负责动态发现 task 间的依赖关系并计算所需通信（data coherence）。
  - **编译框架层**：无明显编译框架层优化。cuPyNumeric 的每个操作独立 target Legion，不进行跨操作的编译优化。Legion 收到 task stream 后按序执行。
  - **kernel调度层**：每个 index task 内部为 element-wise 嵌套循环（Figure 1e: ADD 含一对 for 循环，MULT 含一对 for 循环，COPY 含一对 for 循环），各 kernel 独立执行。5-point stencil 一次迭代产生 6 个独立 kernel（4 ADD + 1 MULT + 1 COPY），每个 kernel 单独 launch，中间结果通过 HBM 传递，需要 6 次 pass over data（或至少需要 5 个临时数组的 HBM read/write）。
  - **硬件架构层**：NVIDIA A100 GPU，无自定义硬件修改。

  Baseline 缺陷：
  - (a) **临时分布式数据膨胀**：每个中间操作产生一个分布式临时数组（t1, t2, t3, avg, work），占用大量 GPU HBM 并产生额外 memory traffic。
  - (b) **数据局部性差**：多个 element-wise kernel 依次执行，每个 kernel 从其输入读取、计算、写入临时输出，下一个 kernel 再从临时输出读取。数据无法在 on-chip memory（register/SRAM）中复用。
  - (c) **kernel launch overhead**：大量小 kernel 的 launch overhead 累积（Black-Scholes: 67 tasks/iteration）。
  - (d) **跨库边界无法优化**：不同库（cuPyNumeric + Legate Sparse）的 task 互相独立，无法跨库进行融合。
  - (e) **分布式数据 aliasing 导致融合复杂**：aliasing views（center, north, east, west, south 均为 grid 的切片）使简单融合可能违反依赖关系。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Diffuse：通过 task fusion + kernel fusion 在分布式 runtime 上自动组合分布式计算**。核心设计：(i) 一个 scale-free IR，使分布式程序的表示大小与机器规模无关；(ii) 基于 4 个 fusion constraint 的动态 task fusion 算法；(iii) 基于 MLIR polyhedral compilation 的 kernel fusion，在融合 task 内部融合循环、消除临时分配；(iv) 在高层库（cuPyNumeric, Legate Sparse）和底层 runtime（Legion）之间作为中间层，对用户透明。

  全栈执行例子（同样 5-point stencil，Diffuse enabled）：
  - **模型推理算法层**：同一 5-point stencil 计算逻辑，用户代码不变。
  - **系统框架层**：修改后的 cuPyNumeric 动态生成 Diffuse IR 而非直接 target Legion。Diffuse 缓冲 task stream 到 window（window size=5），运行 fusion constraints 数据流分析：(a) launch-domain-equivalence 验证同一 launch domain；(b) true-dependence 检查 write→read/write 仅在相同 partition 时安全；(c) anti-dependence 检查 read→write 仅在相同 partition 时安全；(d) reduction 约束读写冲突。由于 aliasing（center/north/east/west/south 均为 grid 的不同 partition），COPY(work, center) 不满足 true-dependence constraint（work 写入的 partition 不同于各个 aliasing view 读取的 partition），因此 COPY 无法融入 fused task。而 4 ADD + MULT 操作使用相同的 partition 读写中间结果，满足所有 constraints → 融合为 FUSED_ADD_MULT。
  - **编译框架层**：Diffuse 的 MLIR JIT compiler（Section 6）将 FUSED_ADD_MULT 中的 5 个 task body 组合：(a) 顺序调用各 task 的 MLIR generator 生成 fragment → 组合为初始 fused kernel body；(b) temporary store elimination（Section 5.1）通过 dataflow 分析发现 avg 由 fused task 完全产生且不被 fused task 外的 task 或应用引用，降级为 task-local memref.alloca；(c) polyhedral fusion pass 将 4 个 ADD + 1 MULT 的 5 个独立 affine.for 循环融合为单个 affine.par 循环；(d) memref.alloca 消除，因为 temporary 被完全 inlined；(e) 最优 single-pass kernel（Figure 8d）：一次 pass 完成 5-way scaled add。
  - **kernel调度层**：单一 CUDA kernel launch 替代原来的 6 个 kernel。每次 GPU thread 处理一个 element 的全部 5 次加法和 1 次乘法，中间结果存储在 register 中（而非 HBM temporary array）。Arithmetic intensity 大幅提升，memory traffic 减少约 5×。
  - **硬件架构层**：标准 GPU，无自定义硬件修改。Memoization（Section 5.2）通过 canonical De-Bruijn index 表示检测 isomorphic task stream 并复用分析+编译结果，使编译开销可摊销（25–119 次迭代即可 breakeven，Figure 13）。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 临时分布式数据膨胀** → 方案：temporary store elimination（Section 5.1）。通过 dataflow 分析在 fused task 内识别仅被 fused task 内部产生和消费、且无外部引用的 store（满足 Definition 4 三约束），将其从分布式分配降级为 task-local allocation。MLIR 后续 pass 进一步 inlining 消除临时 allocation。
  - **defect: 数据局部性差** → 方案：kernel fusion（Section 6）。MLIR polyhedral compilation 将多个独立循环融合为单一循环，使中间结果保留在 register/SRAM 中，最大化数据复用。Black-Scholes 中 67 个 element-wise 操作融合为单 kernel 一 pass 计算，10.7× speedup。
  - **defect: kernel launch overhead** → 方案：task fusion 将多个 index task 合并为单个 index task，kernel fusion 将多个 kernel body 合并为单个 kernel。Black-Scholes: 67 tasks → 1 task（Figure 9）。即使 task granularity 已大于 Legion 的最小有效粒度（1ms/task），kernel fusion 通过提升 arithmetic intensity 产生实际加速。
  - **defect: 跨库边界无法优化** → 方案：Diffuse 的 IR 和 fusion analysis 基于 task 的 privilege 信息（R/W/Rd/RW）和 partition 结构（None/Tiling），而非任何特定库的语义。cuPyNumeric 和 Legate Sparse 的 task 在 Diffuse IR 中被统一表示，fusion constraints 在统一的 IR 层面运行，实现跨库（cross-library）融合。
  - **defect: aliasing 导致融合复杂** → 方案：Diffuse 的 fusion constraints 通过 partition equality check（而非计算 sub-store intersection）来检测 aliasing。由于 partition 的结构化表示（Tiling(shape, offset, proj)），equality check 为 O(1) 操作，且 scale-free（不随 processor 数增长）。constraints 的设计使得 aliasing 视图中非冲突的读写（如同时读不同的 aliasing view）可以安全融合，而冲突的读写（如 COPY 写 center 同时 ADD 通过不同 partition 读 center）被正确阻止融合。
  - **defect: 分布式依赖分析复杂度随规模增长** → 方案：scale-free IR。partition 的映射是隐式的（通过 Tiling 公式 sub-store-bounds(Tiling(shape, offset, proj), p) = [proj(p)*shape, proj(p+1)*shape) + offset），IR 大小仅由 task 数和 partition 数决定，与 GPU 数量无关。这使得 fusion analysis 可以在任意规模机器上执行。对比 Legion 直接表示 partition（显式存储每个 sub-store 边界），别名查询复杂度随 processor 数增长。
  - **额外设计：analysis 复用** → Memoization 将 alpha-equivalence 问题应用于 task stream 匹配（Figure 7），通过 canonical De-Bruijn index 表示消除 store ID 重命名的影响，使 isomorphic task stream 的 fusion decision 和编译结果可被复用。在循环中特别有效（同一 pattern 的 task stream 重复出现在每次迭代中）。
