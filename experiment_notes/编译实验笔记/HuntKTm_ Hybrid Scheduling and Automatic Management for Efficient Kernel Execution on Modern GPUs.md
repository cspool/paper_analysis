## HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

- 属于编译框架的实现是什么？实验比较什么？
  实现是基于LLVM Compiler Infrastructure的编译框架，包含四个LLVM pass：(1) stream scheduler——自动分析kernel间数据依赖、构建DFG、将kernel分配到多stream、生成最小同步指令集；(2) memory manager——执行memory object liveness分析，通过推迟allocation（Algorithm 2: PostponeMalloc）和提前free来缩短memory object生命周期，实现跨object的memory reuse；(3) resource analyzer——通过nvcc获取kernel register/shared memory使用量，分析memory allocation size和kernel launch配置，汇总task资源需求；(4) function wrapper——拦截所有CUDA runtime函数调用（memory allocation、deallocation、kernel launch），注入lazy execution逻辑和resource信息收集代码。

  此外，编译框架还包括memory pool初始化代码插入和cudaTaskSchedule/cudaTaskScheduleLazy调度指令插入。整个compilation pipeline将串行CUDA源码自动转换为内存优化后的多stream并发程序。

  实验比较评估了：(1) compilation overhead——在7个benchmark上的平均编译时间从1.41s增加到2.40s；(2) runtime overhead——task scheduling约1ms，CUDA context初始化平均从81ms到118ms（但因overlap不增加额外执行时间）；(3) code modification effort——HuntKTm只需平均15 LoC和49 tokens的代码修改（vs Async需要手动管理stream/event/sync的大量代码，Taskflow需要显式声明依赖，GrSched需要切换到Python DSL）。

- 硬件平台是什么，配置是什么。
  编译平台：Debian 10.2.1，LLVM 14.0.6，CUDA 12.4.0，NVIDIA driver 555.42.06。编译目标：NVIDIA A100和RTX 4090 GPU。但不限定CUDA平台——设计可轻松应用于其他支持concurrent task queue和async memory management的框架（如HIP、SYCL）。

- 开源编译框架是什么。修改了什么。
  基于LLVM Compiler Infrastructure 14.0.6，新增四个自定义LLVM pass：
  1. stream scheduler pass：在host IR中定位`__cudaPushCallConfiguration`调用模式来识别kernel launch，对识别到的kernel集执行DFG构建→kernel分配→同步生成→冗余同步剪除。
  2. memory manager pass：在stream scheduler输出的stream graph上执行数据流分析——遍历所有kernel调用的GPU指针参数，通过use-def chain追踪每个memory address的allocation/deallocation指令，识别memory object与kernel的依赖关系，执行Algorithm 2延迟allocation到live range起点，转换memory操作为async版本（cudaMalloc→cudaMallocAsync, cudaMemcpy→cudaMemcpyAsync），提前free类似处理。
  3. resource analyzer pass：调用nvcc获取每个kernel的register和shared memory使用量，分析kernel launch参数和memory allocation指令大小，在资源需求完全确定的位置插入cudaTaskSchedule。
  4. function wrapper pass：包装所有CUDA memory和kernel launch函数，注入lazy engine拦截逻辑。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  开源链接：https://github.com/Gemini321/HuntKTm

  作用：HuntKTm编译框架的核心理念是"用编译器自动化的静态分析替代人工调优"，让开发者只需在kernel定义处添加一行writable参数标注（指示前N_out个参数为writable），编译器即可自动完成依赖分析→kernel并行化→内存优化→资源分析的全流程。

  全过程（以DL benchmark为例，包含多个深度学习中常见的kernel）：
  ```
  输入：用户编写的串行CUDA C++源码
    // 用户仅需添加一行标注：writable参数个数
    __global__ void dl_kernel1(const float* in, float* out, ...) { ... }  // 标注1个output
    调用序列: dl_kernel1(...); dl_kernel2(...); dl_kernel3(...);

  Step 1 - Stream Scheduler Pass:
    → 解析每个kernel调用的writable参数标注 → 区分read-only和writable参数
    → 逆序遍历kernel调用序列，BFS识别每个kernel的直接前驱：
      dl_kernel2 reads out → dl_kernel2 depends on dl_kernel1 (RAW)
      dl_kernel3 reads out, writes out2 → dl_kernel3 depends on dl_kernel1, dl_kernel2
    → DFG分层：Level 0: K1, Level 1: K2, Level 2: K3
    → Kernel分配：K1→Stream1, K2→Stream2(单前驱同stream或选最少后继前驱stream), K3→Stream2
    → 同步生成+剪除：创建跨stream event，利用隐式同步消除冗余barrier
    → 输出stream graph（多stream执行计划）

  Step 2 - Memory Manager Pass:
    → 遍历stream graph中所有kernel的GPU指针参数
    → use-def chain追踪：out指向的memory object M1 从cudaMalloc(&ptr, size)分配
    → liveness分析：M1的live range = [dl_kernel1, dl_kernel3]
    → Algorithm 2: 将cudaMalloc移到dl_kernel1之前，转换为cudaMallocAsync放入stream2
    → 将cudaMemcpy移到dl_kernel1之前同一stream
    → 添加同步指令确保M1在dl_kernel3使用前已分配
    → 同样对cudaFree提前处理
    → 输出内存优化后的stream graph

  Step 3 - Resource Analyzer Pass:
    → 调用nvcc --ptxas-options=-v编译每个kernel → 获取register数和shared memory使用量
    → 分析memory allocation大小 → 汇总task的memory需求
    → 在资源需求完全确定的位置插入cudaTaskSchedule()调用

  Step 4 - Function Wrapper Pass:
    → 包装所有CUDA调用：cudaMalocAsync → __huntktm_cudaMallocAsync (带lazy engine拦截)
    → 同样包装kernel launch calls

  编译输出：包含lazy engine逻辑、资源metadata、内存优化后的多stream执行计划的可执行程序

  运行时补充：
    lazy engine延迟执行CUDA操作 → 资源信息通过shared memory传给task dispatcher
    → task dispatcher选择目标GPU → 操作执行，memory pool管理
  ```
