## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- 属于编译框架的实现是什么？实验比较什么？
  本文为综述论文，编译框架覆盖有限，无原创编译实验。论文仅在以下位置涉及编译框架：
  (i) 表5中列出MLC-LLM为"a machine learning compilation system for large language models...with compiler acceleration"，支持任意LLM的native部署。
  (ii) §4.4.3量化部分提及ggml利用CPU SIMD指令集进行整数矩阵乘法，Intel Extension for Transformers利用编译优化。
  (iii) §5.3.1提及FlashAttention等手写CUDA kernel通过nvcc编译，非编译框架自动生成。
  论文未深入编译框架的自动化kernel生成、IR设计或调度优化。

- 硬件平台是什么，配置是什么。
  论文未明确说明（编译框架相关内容有限）。

- 开源编译框架是什么。修改了什么。
  MLC-LLM（https://github.com/mlc-ai/mlc-llm）在表5中被列为编译加速LLM部署方案，论文未深入其内部实现。综述本身未修改任何编译框架。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  论文对该层次的覆盖不足以提供具体编译框架使用例子。以下根据MLC-LLM的一般流程（非论文原创内容）简述：
  ```
  输入: HuggingFace模型权重 (e.g., LLaMA-7B PyTorch checkpoint)
    → 模型导入: 将模型计算图转换为Relax IR（MLC的中间表示）
      → 图优化: 算子融合、常量折叠、内存规划
        → 代码生成: TVM targeting后端（CUDA/Metal/Vulkan/OpenCL）
          → 运行时: 平台native API对优化后模型进行推理
  输出: 针对目标平台的优化推理引擎
  ```
  论文在编译框架层次为近似匹配，核心贡献不在此层次。
