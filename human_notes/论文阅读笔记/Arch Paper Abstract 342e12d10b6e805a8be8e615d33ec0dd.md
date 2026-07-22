# Arch Paper Abstract

[https://github.com/fengbintu/Neural-Networks-on-Silicon](https://github.com/fengbintu/Neural-Networks-on-Silicon)

GPU和Tiled是两种架构形式（Tree分层和Tiled平铺），chiplet、SoC、3DIC和Wafer是不同物理封装或实现技术。

算法-软件栈/编译-架构-封装是算法执行的垂直截面，创新或优化可以依附或结合。

## HPCA24

### 数据移动、重排

- **SmartDIMM**: In-Memory Acceleration of Upper Layer Protocols **
    - 隐式的数据移动和转换
- Data **Motion** Acceleration: Chaining Cross-Domain Multi Accelerators **
    - 专用模块处理数据移动
- **RELIEF**: Relieving Memory Pressure In SoCs Via Data Movement-Aware Accelerator Scheduling **
    - 考虑数据移动需求的调度

### 动态精度、冷热权重

- **HotTiles**: Accelerating SpMM with Heterogeneous Accelerator Architectures **
    - 冷热Tile的异构架构
- **SPARK**: Scalable and Precision-Aware Acceleration of Neural Networks via Efficient Encoding **
    - 动态精度

### PIM架构

- **Pathfinding** Future PIM Architectures by Demystifying a Commercial PIM Technology **
    - PIM模拟器
- An **LPDDR**-based CXL-PNM Platform for TCO-Efficient GPT Inference **

### Memory

- **MIMDRAM**: An End-to-End Processing-Using-DRAM System for High-Throughput, Energy-Efficient and Programmer-Transparent Multiple-Instruction Multiple-Data Computing **
    - MIMD的存内计算
- **ASADI**: Accelerating Sparse Attention using Diagonal-based In-situ Computing

### 其他

- **Gemini**: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators
    - chiplet设计搜索

## ASPLOS24

### 推理pipeline

- **SpecInfer**: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification **
    - 树状token投机推理
- **8-bit** Transformer Inference and Fine-tuning for Edge Accelerators **
    - 边缘学习和推理
- **Fractal**: Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning **
    - 考虑所有层的联合稀疏

### 调度

- **ExeGPT**: Constraint-Aware Resource Scheduling for LLM Inference **
    - 约束感知的资源调度
- **Proteus**: A High-Throughput Inference-Serving System with Accuracy Scaling **
    - 精度换吞吐的（弹性）调度
- **SpotServe**: Serving Generative Large Language Models on Preemptible Instances **
    - 抢占式实例的动态调度、配置和恢复

### Memory

- **MAGIS**: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN **
    - 合并内存足迹小的算子，单卡推理
- **SmartMem**: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile **
    - layout变换消除
- **Cocco**: Hardware-Mapping Co-Exploration towards Memory Capacity-Communication Optimization **
    - 考虑内存容量、通信成本的Mapping
- **IANUS**: Integrated Accelerator based on NPU-PIM Unified Memory System **
    - NPU-PIM的统一内存
- **PIM-STM**: Software Transactional Memory for Processing-In-Memory Systems **
    - 软件事务内存，分布式PIM的一致性
- **PIM-DL**: Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization **
    - layout for PIM
- **Tandem** Processor: Grappling with Emerging Operators in Neural Networks **
    - 专用内存访问模块

### 动态NN

- Optimizing **Dynamic-Shape** Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization **
    - 在线微内核聚合
- **ACES**: Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations **
    - 动态稀疏，调整并行策略

### 静态分析

- **Explainable-DSE**: An Agile and Explainable Exploration of Efficient HW/SW Codesigns of Deep Learning Accelerators Using Bottleneck Analysis **
    - 性能瓶颈分析

### PIM推理

- **AttAcc**! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference **
    - 生成式模型在PIM
- **SpecPIM**: Accelerating Speculative Inference on PIM-Enabled System via Architecture-Dataflow Co-Exploration **
    - 投机推理在PIM
- **NeuPIMs**: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing **
    - LLM推理在NPU-PIM

### 数据压缩

- **Atalanta**: A Bit is Worth a “Thousand” Tensor Values
- **BeeZip**: Towards An Organized and Scalable Architecture for Data Compression

## ISCA24

### 推理pipeline、硬件加速器

- **Splitwise**: Efficient Generative LLM Inference Using Phase Splitting **
    - pipeline中PD分离到不同GPU cluster。
- **Cambricon-D**: Full-Network Differential Acceleration for Diffusion Models **
    - Diffusion加速器
- **Trapezoid**: A Versatile Accelerator for Dense and Sparse Matrix Multiplications **
    - dense和稀疏融合的GEMM单元
- **MECLA**: Memory-Compute-Efficient LLM Accelerator with Scaling Sub-matrix Partition **
    - 子矩阵划分

### 数据移动、重排

- A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip **Dataflow** **Switching ****
    - layout变换和reduction合并的tile加速器
- Mind the **Gap**: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms **
    - 数据搬运、计算密度的程序分析

### 动态

- **Pre-gated** MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference **
    - MoE的gate计算提前
- **ALISA**: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching
    - 稀疏感知

### Memory

- **UM-PIM**: DRAM-based PIM with Uniform & Shared Memory Space
    - PIM的统一共享内存
- **pSyncPIM**: Partially Synchronous Execution of Sparse Matrix Operations for All-bank PIM Architectures
    - 稀疏在Bank的分布不均，局部同步机制。

### 其他

- **PID-Comm**: A Fast and Flexible Collective Communication Framework for Commodity Processing-in-DIMMs
    - All-Reduce、Broadcast的PIM通信优化
- **PreSto**: An In-Storage Data Preprocessing System for Training Recommendation Models
- Enabling Efficient Large Recommendation Model Training with **Near CXL** Memory Processing
- **MAD** Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems

## MICRO24

### dataflow

- **SambaNova** SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
    - **数据流**架构计算CoE

### 加速器

- **BBS**: Bi-directional Bit-level Sparsity for Deep Learning Acceleration **
    - 位识别冗余
- **VGA**: Hardware Accelerator for Scalable Long Sequence Model Inference **
    - 硬件pipeline
- **FuseMax**: Leveraging Extended Einsums to Optimize Attention Accelerator Design
    - 新的数学抽象

### 数据重排、layout

- CamPU: A Multi-Camera Processing Unit for Deep Learning-based 3D Spatial Computing Systems **
    - 多摄像头数据预对齐，降低数据重排开销
- Stream-Based Data Placement for Near-Data Processing with Extended Memory

### 多任务

- **Duplex**: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching **
    - 连续批处理，动态负载
- **SCAR**: Scheduling Multi-Model AI Workloads on Heterogeneous Multi-Chiplet Module Accelereators **
    - 多模型负载

### 动态运行时

- **AdapTiV**: Sign-Similarity based Image-Adaptive Token Merging for Vision Transformer Acceleration **
    - ViT的token运行时合并
- **SOFA**: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling **
    - 稀疏数据对齐、同步

### Memory

- A **Mess** of Memory System Benchmarking, Simulation and Application Profiling **
    - 内存模拟器的高并发修正
- **PIM-MMU**: A Memory Management Unit for Accelerating Data Transfers in Commercial PIM Systems
    - PIM的MMU
- **Leviathan**: A Unified System for General-Purpose Near-Data Computing
    - NDP的一致性和可编程性

### 其他

- Fusion-3D: Integrated Acceleration for Instant 3D Reconstruction and Real-Time **Rendering**
    - 图形加速
- Stellar: An Automated Design Framework for Dense and Sparse Spatial Accelerators
    - 自动设计DSA
- LUCIE: A Universal **Chiplet**-Interposer Design Framework for Plug-and-Play Integration
- SRender: Boosting Neural Radiance Field Efficiency via Sensitivity-Aware Dynamic **Precision** Rendering
    - NeRF渲染
- EMP: Efficient 4-bit Matrix Unit via Primitivization
    - 4bit GEMM硬件原语
- SCALE: A Structure-Centric Accelerator for Message Passing **Graph** Neural Networks
- Low-overhead General-purpose Near-Data Processing in CXL Memory Expanders
- PIFS-Rec: **Process-In-Fabric-Switch** for Large-Scale Recommendation System Inferences
- Azul: An Accelerator for Sparse Iterative **Solvers** Leveraging Distributed On-Chip Memory
- FloatAP: Supporting High-Performance **Floating**-Point Arithmetic in Associative Processors
- COMPASS: SRAM-Based Computing-in-Memory **SNN** Accelerator with Adaptive Spike Speculation
- TMiner: A Vertex-Based Task Scheduling Architecture for **Graph** Pattern Mining
- PointCIM: A Computing-in-Memory Architecture for Accelerating Deep **Point** Cloud Analytics
- FlashLLM: A **Chiplet**-Based In-Flash Computing Architecture to Enable On-Device Inference of 70B LLM
- GauSPU: 3D Gaussian **Splatting** Processor for Real-Time SLAM Systems
- PyPIM: Integrating Digital Processing-in-Memory from Microarchitectural Design to Python Tensors
    - 存内计算的软件调用栈，支持python
- FiboCIM: a Fibonacci-coded Charge-domain SRAM-based CIM Accelerator for DNN Inference
- MeMCISA: Memristor-enabled **Memory-Centric** Instruction-Set Architecture for Database Systems

## HPCA25

### 量化方法

矢量量化

- **VQ**-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference **

混合量化

- **BitMoD**: Bit-serial Mixture-of-Datatype LLM Acceleration **
- **Anda**: Unlocking Efficient LLM Inference with a Variable-Length Grouped Activation Data Format **

### PIM+SoC的加速架构

- **PAISE**: PIM-Accelerated Inference Scheduling Engine for Transformer-based LLM **
- **FACIL**: Flexible DRAM Address Mapping for SoC-PIM Cooperative On-device LLM Inference **

### 动态性

- **LAD**: Efficient Accelerator for Generative Inference of LLM with Locality Aware Decoding **
    - token得分稳定后，后续不需KV计算
- **InstAttention**: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference **
    - 处理热token
- Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with **NDP-DIMM**
    - 冷热神经元参数

### Memory

- **SoMA**: Identifying, Exploring, and Understanding the DRAM Communication Scheduling Space for DNN Accelerators **
    - 通信开销的编译模型

### 其他

- eDKM: An Efficient and Accurate Train-Time Weight Clustering for Large Language Models
    - 基于聚类的训练量化
- **LUT**-DLA: Lookup Table as Efficient Extreme Low-Bit Deep Learning Accelerator
    - 矢量量化，LUT索引表示权重
- FIGLUT: An Energy-Efficient Accelerator Design for FP-INT GEMM Using Look-Up Tables
- MANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type
- Enhancing Large-Scale AI Training Efficiency: The C4 Solution for Real-Time Anomaly Detection and Communication Optimization
    - 训练实时异常检测
- Revisiting Reliability in Large-Scale Machine Learning Research Clusters
    - 训练可靠性
- **PIMnet**: A Domain-Specific Network for Efficient Collective Communication in Scalable PIM
    - chiplet NoC
- EIGEN: Enabling Efficient 3DIC Interconnect with Heterogeneous Dual-Layer Network-on-Active-Interposer
    - chiplet互联
- **Lincoln**: Real-Time 50~100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory
    - LPDDR的存内计算

## ASPLOS25

### NPU

- Fast **On-device** LLM Inference with NPUs **
    - LLM边缘端推理

### 多任务

- **Spindle**: Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling **
    - 多任务LLM的训练？
- **POD-Attention**: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference **
    - PD融合Attn kernel，SM运行时配置融合kernel的不同Op。

### 动态稀疏、量化

- **CoServe**: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory
    - CoE的MCM平台推理，小专家的动态激活调度。
- **Klotski**: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline **
    - MoE的专家动态稀疏的不均衡
- **COMET**: Towards Practical W4A4KV4 LLMs Serving **
    - KV（激活）量化
- **PAPI**: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System **
    - LLM动态并行的Mapping和调度，编译

### 通信、Memory

- Accelerating LLM Serving for **Multi-turn Dialogues** with Efficient Resource Management **
    - 多轮对话场景的KV缓存优化
- **Concerto**: Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning
    - 通信优化引入编译框架
- **Be CIM or Be Memory**: A Dual-mode-aware DNN Compiler for CIM Accelerators
    - Memory和CIM的动态配置，编译

### 其他

- DynaX: Sparse Attention Acceleration with Dynamic X:M Fine-Grained Structured Pruning
    - 模型剪枝
- ReCA: Integrated Acceleration for Real-Time and Efficient Cooperative Embodied Autonomous Agents
    - 具身智能
- **Helix**: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow
    - 异构GPU的负载调度
- FlexSP: Accelerating Large Language Model Training via Flexible Sequence Parallelism
- **MoE-Lightning**: High-Throughput MoE Inference on Memory-constrained GPUs
    - CPU+GPU部署MoE推理
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
    - MoE训练加速
- MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
    - 训练容错
- Past-Future Scheduler for LLM Serving under SLA Guarantees
    - 内存风险预测
- TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms

## ISCA25

### 加速器

- **SpecEE**: Accelerating Large Language Model Inference with Speculative Early Exiting **
    - 投机Decode
- Meta's Second Generation **AI Chip**: Model-Chip Co-Design and Productionization Experiences **
- **AMALI**: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs **
    - GPU分析模型，推理行为的cycle level建模
- **HeterRAG**: Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation **
    - RAG加速

### dataflow

- **H2-LLM**: Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference **
    - 硬件数据流
- **NUPEA**: Optimizing Critical Loads on **Spatial Dataflow** Architectures via Non-Uniform Processing-Element Access **

### 动态、多任务

- **MagiCache**: A Virtual In-Cache Computing Engine **
    - 多DNN推理
- **MicroScopiQ**: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization **
    - 离群值感知量化
- **HYTE**: Flexible Tiling for Sparse Accelerators via Hybrid Static-Dynamic Approaches **
    - 运行时稀疏调度Tiling策略
- **Oaken**: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization **
    - 混合量化

### 其他

- WSC-LLM: Efficient LLM Service and Architecture Co-exploration for **Wafer**-scale Chips
- FRED: A **Wafer**-scale Fabric for 3D Parallel DNN Training
- PD Constraint-aware Physical/Logical Topology Co-Design for Network on **Wafer**
- HiPER: Hierarchically-Composed Processing for Efficient Robot Learning-Based Control
    - 具身智能
- Dadu-Corki: Algorithm-Architecture Co-Design for Embodied AI-powered Robotic Manipulation
    - 具身智能
- **Chimera**: Communication Fusion for Hybrid Parallelism in Large Language Models
    - 通信融合
- **LUT** Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference
- AiF: Accelerating On-Device LLM Inference Using In-Flash Processing
    - 闪存计算
- LIA: A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading
    - GPU内存扩展
- Cramming a Data Center into One Cabinet: A Co-Exploration of Computing and Hardware Architecture of **Waferscale** Chip
- **Ecco**: Improving Memory Bandwidth and Capacity for LLMs via Entropy-Aware Cache Compression
    - 缓存压缩
- **Hybe**: GPU-NPU Hybrid System for Efficient LLM Inference with Million-Token Context Window
    - GPU（Prefill）+NPU（Decode）
- MeshSlice: Efficient 2D Tensor Parallelism for Distributed DNN **Training**
- AIM: Software and Hardware Co-design for Architecture-level IR-drop Mitigation in High-performance PIM
    - PIM缓解电压降
- OptiPIM: Optimizing Processing-in-Memory Acceleration Using Integer Linear Programming
    - 基于ILP的PIM的调度和Mapping
- ATiM: Autotuning Tensor Programs for Processing-in-DRAM
    - PIM的DRAM自动调优
- Hybrid SLC-MLC RRAM Mixed-Signal Processing-in-Memory Architecture for Transformer Acceleration via Gradient Redistribution
    - 存储权重的梯度重分布
- Scaling Llama 3 **Training** with Efficient Parallelism Strategies
- Insights into **DeepSeek**-V3: Scaling Challenges and Reflections on Hardware for AI Architectures
- BingoGCN: Towards Scalable and Efficient **GNN** Acceleration with Fine-Grained Partitioning and SLT

## MICRO25

### 新概念、工具

- MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging **Bit-Slice-enabled** Sparsity and Repetitiveness. (Tsinghua University, Shanghai Jiao Tong University) **
- **TAIDL**: Tensor Accelerator ISA Definition Language with Auto-generation of Scalable Test Oracles. (UIUC) **
    - DSA的ISA自动化开发

### 加速器、加速Core

- **S-DMA**: Sparse Diffusion Models Acceleration via Spatiality-Aware Prediction and Dimension-Adaptive Dataflow. (Southeast University) **
    - 稀疏Diffusion
- **Crane**: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture. (Rutgers University, Texas A&M University, NVIDIA) **
    - Tiled Acc同时支持推理、**训练**的编译，现有CNN调度专注layer内 Mapping。
- **Pimba**: A Processing-in-Memory Acceleration for **Post-Transformer** Large Language Model Serving. (KAIST, Uppsala University, Georgia Tech) **
    - 同时支持Mamba和Transformer架构模型
- Accelerating **Retrieval Augmented Language Model** via PIM and PNM Integration. (Yonsei University, Santa Clara University)
    - RAG加速
- **HLX**: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models. (KAIST) **
    - Mamba+Transformer
- **ORCHES**: Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System. (Georgia Institute of Technology) **
    - 测试时计算（思维链）
- **AxCore**: A Quantization-Aware Approximate GEMM Unit for LLM Inference. (HKUST(GZ))
    - 算力和能耗平衡
- **ReGate**: Enabling Power Gating in Neural Processing Units. (UIUC)
    - 功耗NPU

### dataflow、路由、ISA

- **StreamTensor**: Make Tensors Stream in Dataflow Accelerators for LLMs. (UIUC, Inspirit IoT) **
    - Convertor、DMA、kernel的编译
- **ELK**: Exploring the Efficiency of Inter-core Connected AI Chips with Deep Learning Compiler Techniques. (UIUC, Microsoft Research) **
    - Core间数据路由优化

### 多任务、动态量化

- **Chameleon**: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments. (UIUC, IBM Research) **
    - 多任务
- **MHE-TPE**: Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines. (USTC, University of Washington, Raytron Technology) **
    - 混合精度定点数TC的编码
- **Amove**: Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type. (Beihang University, Tsinghua University) **
    - 离群值量化

### 其他

- **PolymorPIC**: Embedding Polymorphic **Processing-in-Cache in RISC-V based Processor** for Full-stack Efficient AI Inference. (Shanghai Jiao Tong University, Shanghai AI Lab)
    - 静态缓存分配
- **Stratum**: System-Hardware Co-design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving. (UC San Diego, Georgia Tech, UIUC, Illinois Tech)
    - 3D DRAM的分层存放冷热参数
- **Kelle**: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing. (New York University)
    - eDRAM替换SRAM
- **LongSight**: Compute-Enabled Memory to Accelerate Large-Context LLMs via Sparse Attention. (Cornell University)
    - PIM
- ComPASS: A Compatible PIM Protocol Architecture and Scheduling Solution for Processor-PIM Collaboration. (Inha University)
    - PIM+Processor异构系统的协同
- **PIM-CCA**: An Efficient PIM Architecture with Optimized Integration of Configurable Functional Units. (Yonsei University, KAIST, Hanyang University)
    - PIM
- DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model. (Intel, UIUC)
    - 优化权重解压缩过程
- **Coruscant**: Co-Designing GPU Kernel and Sparse Tensor Core to Advocate Unstructured Sparsity in Efficient LLM Inference. (University of Maryland, d-Matrix)
    - GPU STC+kernel支持非结构稀疏
- SkipReduce: (Interconnection) Network Sparsity to Accelerate Distributed Machine Learning. (KAIST, NVIDIA, Hanyang University)
    - 分布式学习
- 3D-PATH: A Hierarchy LUT Processing-in-memory Accelerator with Thermal-aware Hybrid Bonding Integration. (Tsinghua University, Shanghai Jiao Tong University)
    - 3D堆叠的散热
- HEAT: NPU-NDP Heterogeneous Architecture for Transformer-Empowered **Graph** Neural Networks. (Shanghai Jiao Tong University, Chinese Academy of Sciences)
- RayN: Ray Tracing Acceleration with Near-memory Computing. (University of British Columbia)
- GateBleed: Exploiting On-Core Accelerator Power Gating for High Performance and Stealthy **Attacks** on AI. (North Carolina State University, Intel)
- Athena: Accelerating Quantized Convolutional Neural Networks under Fully Homomorphic **Encryption**. (Institute of Computing Technology CAS, University of Electronic Science and Technology of China)
- ccAI: A Compatible and **Confidential** System for AI Computing. (University of Science and Technology, The Hong Kong Polytechnic University, Ant Group, Southern University of Science and Technology)
- Ironman: Accelerating Oblivious Transfer Extension for **Privacy**-Preserving AI with Near-Memory Processing. (Peking University, State Key Laboratory of Cryptology, Alibaba Group, Tsinghua University)
- LLM.265: Video Codes are **Secretly** Tensor Codes. (Duke University, Carnegie Mellon University)
- NetZIP: Algorithm/Hardware Co-design of In-network Lossless Compression for Distributed Large Model **Training**. (UIUC, IBM Research)
    - 分布式训练
- Characterizing the Efficiency of Distributed **Training**: A Power, Performance, and Thermal Perspective. (Georgia Tech)
- Optimizing All-to-All Collective Communication with **Fault** Tolerance on Torus Networks. (HKUST(GZ), Huawei)
- **MX+**: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving. (Seoul National University)
    - 量化表示
- Multi-Dimensional ML-Pipeline Optimization in Cost-Effective Disaggregated Datacenter. (Pennsylvania State University, META, IBM, AMD)
- OASIS: A Commercial High Performance Terminal AI Processor Supporting RISCV Tensor Extension Instructions. (Beijing University of Posts and Telecommunications, Sophgo Technologies)
    - RV张量扩展指令集
- Empowering Vector Architectures for ML: The CAMP Architecture for Matrix Multiplication. (Barcelona Supercomputing Center, Polytechnic University of Catalonia)
    - 增加vector processor的GEMM能力
- Characterizing and Optimizing **Realistic** Workloads on a Commercial Compute-in-SRAM Device. (Cornell University, University of Southern California, MIT, GSI Inc.)
    - PIM
- SuperMesh: Energy-Efficient Collective Communications for Accelerators. (Texas A&M University)
    - 加速器互联优化
- **BitL**: A Hybrid Bit-Serial and Parallel Deep Learning Accelerator for Critical Path Reduction. (Yonsei University, Samsung Electronics)
    - 位串行架构
- **HiPACK**: Efficient Sub-8-Bit Direct Convolution with SIMD and Bitwise Management. (National University of Singapore, Tiangong University)
    - 亚8bit的SIMD
- SMX: Heterogeneous Architecture for Universal Sequence Alignment Acceleration. (Barcelona Supercomputing Center, UPC, Cornell University)
    - 生物信息学任务

## HPCA26

### 加速器、加速Core

- **Focus**: A Streaming Concentration Architecture for Efficient Vision-Language Models **
    - Stream执行
- **PADE**: A Predictor-Free Sparse Attention Accelerator via Unified Execution and Stage Fusion **
    - 无预测稀疏
- **Uni-STC**: Unified Sparse Tensor Core **
    - 跨算子的数据重用
- **RPU**: A Reasoning Processing Unit **
    - Agent设计
- **VAR-Turbo**: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy **
    - VAR模型
- **V-Rex**: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval **
    - 视频模型
- **BitDecoding**: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache **
    - Tensor Core稀疏的layout对齐

### 动态、多任务

- **ELORA**: Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving **
    - 多LoRA服务
- **AQPIM**: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization **
    - 激活量化的存内计算
- **AUM**: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving **
    - 动态调整AU+CPU系统

### Memory、通信

- **PIMphony**: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System **
    - PIM的内存管理
- **RoMe**: Row Granularity Access Memory System for Large Language Models **
    - Row level内存系统，而非Bank level内存系统
- **CoCoTree**: A Computation-Capable Architecture for Collective Communication in Scalable PIM **
    - PE互联的NoC

### 其他

- HR-DCIM: High-Reliability Floating-Point Digital CIM Architecture with Unified Low-Cost Iterative Error Correction
    - ECC技术
- WATOS: Efficient LLM Training Strategies and Architecture Co-exploration for Wafer-scale Chip
- **FACE**: Fully Overlapped PD Scheduling and Multi-Level Architecture Co-Exploration on Wafer
    - 晶圆级芯片的设计，传统是晶圆分割后设计。
- TEMP: A Memory Efficient Physical-aware Tensor Partition-Mapping Framework on Wafer-scale Chips
    - 晶圆级芯片的设计
- MoEntwine: Unleashing the Potential of Wafer-scale Chips for Large-scale Expert Parallel Inference
    - 晶圆级芯片的设计
- AutoGNN: End-to-End Hardware-Driven Graph Preprocessing for Enhanced GNN Performance