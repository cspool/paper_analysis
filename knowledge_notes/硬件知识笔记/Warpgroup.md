## Warpgroup

术语解释
NVIDIA Hopper架构中引入的线程分组概念，指4个连续warp（128个线程）的集合，作为发起Tensor Core操作（WGMMA指令）的基本调度单元。Warpgroup是Hopper架构对Tensor Core编程模型的根本性改变——从Ampere时代的warp级Tensor Core操作变为warpgroup级cooperative launch。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Warpgroup是Hopper SM内部的一种逻辑线程组织方式，用于集体发起WGMMA（Warp Group Matrix Multiply-Accumulate）指令。Hopper架构中每个SM有一个大的Tensor Core（vs Ampere的4个独立Tensor Core），该Tensor Core需要128个线程（=4 warps）协同才能发起一次矩阵乘法操作。这就是warpgroup的根本原因：硬件要求。

关键特征：(1) Warpgroup是cooperative的——所有128线程必须在同一时间点执行相同的wgmma指令（通过`warpgroup_sync()`对齐）；(2) Warpgroup中每个线程持有输出矩阵的不同行和列（按照特定的swizzle pattern分布），操作数A和B也按架构规定的partition策略分布在线程的寄存器和shared memory之间——Cypress论文Figure 4展示了输出矩阵在寄存器中的64×n×8 swizzled partition pattern；(3) Warpgroup不等同于"任意4个连续warp"——而是编号为warp_id/4相同的4个warp组成（如warp 0-3、4-7等）；(4) Warpgroup允许warp specialization——DMA warp（一个额外warp）与compute warpgroup共存于同一SM，DMA warp的寄存器资源可被compute warpgroup利用存储更大accumulator。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Warpgroup在Hopper SM内部的运转流程：

```
SM资源配置：
  - 1个Tensor Core（每个SM）
  - 共享内存最大228KB（可配置）
  - 寄存器文件（65536个32-bit registers per SM）
  
Warpgroup组织（以5个warp的thread block为例）：
  - Warp 0-3：Compute Warpgroup（128 threads）→ 用于WGMMA
  - Warp 4：DMA Warp（32 threads）→ 用于TMA操作
    DMA warp的寄存器释放给compute warpgroup（warp specialization带来的寄存器复用）

一次WGMMA操作的生命周期：
  1. warpgroup_sync()：128线程barrier对齐
  2. wgmma.fence：确保操作数在寄存器/shared memory中就绪
  3. wgmma instruction：128线程同时发出WGMMA PTX指令驱动Tensor Core
     - 操作数A/B来自：部分线程寄存器 + 部分shared memory
     - 输出accumulator：分布在128线程的寄存器中（按Figure 4 swizzle pattern）
  4. warpgroup_wait()：等待Tensor Core完成异步计算
```

Cypress论文在machine model中显式引入了warpgroup作为processor level：
```
Processor levels: HOST → GPU → BLOCK → WARPGROUP → WARP → THREAD
```
这种层次化描述允许task variants针对不同级别定义不同的数据分解和compute分解策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在CUDA编程中使用warpgroup：
- PTX指令：`wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group`, `warpgroup.sync`（PTX ISA 8.0+）
- CUTLASS 3.x通过CuTe封装warpgroup操作
- ThunderKittens通过简洁的C++ API提供warpgroup-level tile操作（如 `warpgroup::mma_ABt`, `warpgroup::mma_AB`, `warpgroup::mma_async_wait`），在其LCSF模板中 compute worker 即由多个 warpgroup 组成
- 需要仔细管理寄存器分配——255 registers/thread的硬件上限意味着大tile需要split到多warpgroup（Cypress的gemm_tile使用WGS=2参数将输出行split到2个warpgroup）

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
- ThunderKittens: Simple, Fast, and Adorable Kernels
- TileLang: A Composable Tiled Programming Model for AI Systems
