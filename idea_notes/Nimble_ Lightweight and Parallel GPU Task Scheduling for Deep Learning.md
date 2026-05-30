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
