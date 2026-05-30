## LCSC Programming Template

术语是什么？
LCSC (Load-Compute-Store-Communicate) Template是PK的统一多GPU kernel编程模板，分解kernel为4个worker：Loader (TMA HBM→SMEM)、Storer (TMA SMEM→HBM)、Consumer (tensor core compute)、Communicator (专用通信SM上的inter-GPU collective)。模板自动处理SM/warp分区、SMEM管理、barrier同步、TMA设置，用户仅需实现per-tile逻辑。每个kernel通信代码<50行。

从kernel调度角度拆解术语：
```
struct lcsc_template {
    static void loader(globals, comp_sem, comp_smem, comp_regs);
    static void storer(globals, comp_sem, comp_smem, comp_regs);      // 含signal barrier
    static void consumer(globals, comp_sem, comp_smem, comp_regs);    // warpgroup MMA
    static void communicator(globals, comm_sem, comm_smem, comm_regs); // wait+all_reduce
};
lcsc::launch_kernel<config, globals, lcsc_template>(G, stream);  // host launch
```
执行模型：compute SM运行loader+consumer+storer（producer-consumer semaphore同步），communication SM运行communicator。config编译时确定SM/thread/warpgroup配置，num_comm_sms控制inter-SM vs intra-SM切换（communicator为空即intra-SM模式）。

术语一般如何实现？如何使用？
PK所有6个kernel（AG+GEMM, GEMM+RS, GEMM+AR, Ring Attention, Ulysses, MoE dispatch+GEMM）均用LCSC模板实现。GEMM+AR的通信代码仅~10行。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels
