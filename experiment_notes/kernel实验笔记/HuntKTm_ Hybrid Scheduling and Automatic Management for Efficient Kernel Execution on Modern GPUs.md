## HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是stream scheduler，它自动将multi-kernel程序中的kernel按数据依赖关系分配到多个CUDA stream（即多个hardware queue）上并发执行。核心包括三部分：(1) DFG constructor——通过轻量级代码标注（在kernel参数列表开头插入writable参数个数常量）自动分析kernel间的RAW/WAR/WAW数据依赖，构建数据流图（DFG）；(2) kernel distributor——将DFG分层（同一层内kernel无数据依赖），按PP-Set（preferred predecessor set）大小排序，通过三条规则（无前驱的round-robin分配、单前驱的同stream放置、多前驱的选最少未调度后继的前驱所在stream）将kernels分配到多stream；(3) synchronization generator——基于依赖传递性和同stream内串行执行的隐式同步，剪除冗余同步屏障。实验比较的baseline包括：Serial（串行执行）、Taskflow（静态调度）、GrSched（动态调度）。单任务场景下评估kernel执行加速比，多任务场景下评估system throughput和硬件资源利用率（SM occupancy, FP32 utilization, memory bandwidth utilization via DCGM）。

- 后端平台是什么，配置是什么。
  NVIDIA A100 (40GB HBM, 6912 CUDA cores) 和 NVIDIA RTX 4090 (24GB)。实验在配备4× NVIDIA A100 GPU、2× AMD EPYC 7742 64核处理器、256 GB DDR4的服务器上进行。操作系统Debian 10.2.1，NVIDIA driver 555.42.06。另一平台配备4× NVIDIA RTX 4090 24GB GPU、2× Intel Xeon Gold 6338N CPU、1024 GB DRAM。

- 评估性能的软件/脚本是什么。修改了什么。
  使用NVIDIA DCGM (Data Center GPU Manager) 低开销GPU系统监控工具周期性采集硬件指标（SM occupancy、FP32 utilization、memory bandwidth utilization）。benchmark包括7个代表性应用：VEC (Vector Square, DFG width=2), B&S (Black & Scholes, width=10), ML (Machine Learning, width=2), IMG (Image Processing, width=3), DL (Deep Learning, width=2), M1 (Micro-1, width=8), M2 (Micro-2, width=6)，其中M1/M2的kernel来自NVIDIA FasterTransformer。修改：HuntKTm通过LLVM pass自动将串行CUDA代码转换为多stream并发代码，仅需开发者在kernel参数列表添加一个常量标注writable参数个数（每kernel一行LoC），无需手动编写任何CUDA stream/event管理代码。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/Gemini321/HuntKTm

  评估原理：
  1. HuntKTm以LLVM pass形式编译时自动分析CUDA host IR，识别`__cudaPushCallConfiguration`调用点来定位kernel launch。
  2. DFG constructor通过遍历kernel参数中标注的writable参数信息，逆序遍历kernel调用序列，用BFS识别每个kernel的直接前驱（基于同数据对象的读写冲突判断RAW/WAR/WAW依赖）。
  3. kernel distributor对DFG分层后逐层分配kernel到stream，核心理念是最小化跨stream同步数量。
  4. synchronization generator遍历每个stream中的kernel，三步剪除冗余屏障：(i)为不在同stream的前驱创建屏障；(ii)每个stream仅保留来自最后前驱的同步；(iii)同stream内核执行顺序带来的隐式同步消除冗余屏障。
  5. 最终输出stream graph（含多stream执行信息和最小同步指令集），编译为可执行程序。

  全过程（以M2为例，包含多个activation和reduction kernel）：
  ```
  用户编写串行CUDA程序，在kernel定义处添加writable参数个数标注
    → LLVM pass: DFG constructor 自动识别kernel间数据依赖，构建DFG（M2的DFG宽度=6）
    → LLVM pass: kernel distributor 将kernel分层并分配到6个stream
       Level 1: round-robin分配无依赖kernel到stream 1-6
       Level 2-N: 按PP-Set大小排序，单前驱kernel跟随后继同stream，多前驱选最少未调度后继的前驱
    → LLVM pass: synchronization generator 创建跨stream event同步，剪除冗余barrier
    → LLVM pass: 编译生成多stream可执行程序
    → 运行时: 程序在多stream上并发执行kernel → NVIDIA MPS支持跨进程kernel在同一GPU上space-sharing
    → DCGM周期采集SM occupancy/FP32 utilization/memory bandwidth → 计算throughput和加速比
  ```
