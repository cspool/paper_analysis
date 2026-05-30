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
