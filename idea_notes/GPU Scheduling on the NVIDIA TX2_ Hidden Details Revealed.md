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
