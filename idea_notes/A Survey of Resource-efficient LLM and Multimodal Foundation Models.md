## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- baseline方法是什么？
  本文为综述论文，不提出新的baseline或方法，而是对已有资源高效大型基础模型研究的系统性分类与梳理。Baseline即综述前文献中各独立研究点的分散状态——各方向（高效架构、训练算法、推理算法、模型压缩、分布式系统、端侧部署）彼此独立发表，缺乏统一的分类体系和纵向联系。论文识别的核心痛点：(i) 模型规模持续增长（scaling law），训练LLaMA-2-70B需1.7×10^6 GPU hours，碳排放291吨CO2；(ii) 资源需求集中在计算、存储、带宽、能耗；(iii) 资源壁垒阻碍模型民主化，仅少数巨头可训练部署SOTA模型；(iv) 各子方向成果分散，缺乏全栈视角。

  全栈执行例子（以LLaMA-7B推理为例，综述整合的全栈视角）：
  - **模型推理算法层**：自回归解码 + KV Cache（每个token需O(Td + d²)计算，KV cache占用2×B×S×D×L×4 bytes）；可通过Speculative Decoding加速2-3×、稀疏注意力降至O(Td)、量化降至4bit。
  - **系统框架层**：vLLM PagedAttention按需分配KV cache（block级管理消除碎片）、Orca迭代级批处理、SARATHI chunked-prefill与decode混合调度。
  - **编译框架层**：FlashAttention/CUDA kernel手写（nvcc编译）、MLC-LLM通过TVM编译加速多平台部署。论文未详细覆盖此层。
  - **kernel调度层**：FlashAttention-2 IO-aware tiling（HBM→SRAM分块计算避免N×N矩阵）、Flash-Decoding针对小seqlen大batch优化、FlashDecoding++优化softmax+flat GEMM。
  - **硬件架构层**：论文明确排除硬件设计（§1: "exclude hardware design"）。
  - **芯片设计层**：论文明确排除（同上）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  本综述的核心贡献是提出一个**多维度的分类框架**，将资源高效基础模型研究按三个维度组织：
  (1) **高效架构**（§3）——注意力机制（稀疏/近似/无注意力）、动态网络（MoE/Early Exiting）、扩散模型优化、ViT优化；
  (2) **高效算法**（§4）——按模型生命周期：预训练→微调→推理→压缩；
  (3) **高效系统**（§5）——分布式训练、联邦学习、云侧Serving、端侧Serving。
  这一分类框架解决了分散研究之间的纵向关联缺失问题，使得从算法到系统到部署的全栈优化路径可被系统性地追踪和对比。

  论文方法全栈执行例子（综述的分类导航能力）：
  - **模型推理算法层**：论文提供各类方法的统一cost分析框架——如用flops-profiler对GPT-2和Stable Diffusion 2.1各模块（Embedding/Attention/FFN/Im_head）的FLOPs和存储开销进行定量分解（图4-7），揭示Attention在长序列时的O(T²D)瓶颈和FFN在D增大时的O(TD²)瓶颈，为优化选择提供决策依据。
  - **系统框架层**：表5提供17个主流开源训练/推理框架的统一对比（DeepSpeed、Megatron-LM、vLLM、llama.cpp、MLC-LLM等），按Cloud/Edge/Training/Inference维度分类，为工程选型提供指导。
  - **编译框架层**：论文在该层覆盖有限，主要提及MLC-LLM的编译器加速部署方案。
  - **kernel调度层**：整合FlashAttention家族（FlashAttention/FA-2/Flash-Decoding/FlashDecoding++）的发展脉络和各自适用场景，明确prefill vs decode的不同kernel优化策略。
  - **硬件架构层**：论文明确排除（引用[192,185]为已有综述覆盖）。
  - **芯片设计层**：论文明确排除（同上）。

  对比baseline（分散研究），论文方法的独特价值在于：(i) 首次将LLM、ViT、扩散模型、多模态模型的资源高效技术统一在同一框架下；(ii) 跨越算法到系统到部署的全生命周期；(iii) 提供定量cost分析（flops-profiler）辅助技术选型；(iv) 指出6个未来方向（cloud-edge hybrid、model sparsity、FMaaS、agent optimization、privacy-preserving FM、scaling law understanding）。
