# cuDNN和TVM

- **cuDNN**
    - **适用场景**：NVIDIA GPU + 传统模型。
    - 优势:基于NVIDIA GPU架构的闭源手工优化，深度集成Tensor Core特性，对标准卷积（如3x3、5x5）和主流网络（ResNet、VGG）的优化成熟度极高。例如，使用Winograd算法时，cuDNN在TITAN X上的卷积速度比未优化的实现快5-8倍
    - 局限性:依赖固定算法候选集（如`cudnnConvolutionFwdAlgo_t`），无法动态适应新型算子。例如，MobileNet的深度卷积早期版本未被cuDNN支持，需框架自行实现
- **TVM优势**
    - **适用场景**：新兴硬件（如FPGA、AMD GPU）或自定义算子（如不规则卷积核）。
    - 优势:通过AutoTVM自动搜索调度参数，支持跨平台（如AMD GPU、ARM Mali）和非标准算子（如深度卷积、低精度运算）。例如，在MobileNet的深度卷积中，TVM生成的算子比TensorFlow的cuDNN实现快2-4倍
    - 灵活性:支持自动生成融合算子（如卷积+ReLU+池化），减少内存带宽消耗。在端到端模型中，TVM通过图优化和算子融合，整体推理速度比MXNet（依赖cuDNN）提升1.6-3.8倍