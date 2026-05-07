**ASPLOS26**

# **Large Language Models (LLMs)**

- LLM Inference
    - Prefill-Decode Multiplexing
        - Towards High-Goodput LLM Serving with Prefill-decode Multiplexing [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790236)] [[arXiv](https://arxiv.org/abs/2504.14489)]
            - SJTU & HKU & NUS
            - Propose **MuxWise**, an LLM serving framework built on intra-GPU prefill-decode multiplexing.
            - Integrate a bubble-less multiplex engine, a contention-tolerant estimator, and an SLO-aware dispatcher.
        - Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790135)] [[arXiv](https://arxiv.org/abs/2504.19516)] [[Code](https://github.com/zejia-lin/Bullet)]
            - SYSU
            - Enable concurrent execution of prefill and decode requests.
            - Dynamically provision GPU resources based on real-time performance modeling.
        - TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790237)]
            - PKU & Tencent YouTu Lab
            - Introduce tensor-parallel latent attention for disaggregated prefill/decode inference.
            - Combine latent attention with tensor parallelism to improve PD-disaggregated long-context serving.
    - Scheduling
        - QoServe: Breaking the Silos of LLM Inference Serving [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790206)] [[arXiv](https://arxiv.org/abs/2503.22562)]
            - MSR India
            - Introduce fine-grained QoS classification so applications can specify precise latency requirements, and adapt scheduling decisions to real-time system state.
            - Leverage the predictable execution characteristics of LLM inference to implement dynamic chunking for higher throughput under strict QoS guarantees.
            - Combine hybrid prioritization with selective request relegation to balance fairness, efficiency, and graceful degradation under overload.
        - Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790219)] [[arXiv](https://arxiv.org/abs/2509.16495)]
            - Snowflake
            - Introduce **Shift Parallelism**, a runtime that switches across inference parallelism strategies for dynamic workloads.
            - Turn parallelism selection into a runtime control decision to jointly improve latency and throughput.
        - XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads [[Paper](https://dl.acm.org/doi/10.1145/3760250.3762228)]
            - Huawei & THU & Shanghai AI Lab
            - Present **XY-Serve**, an end-to-end serving system for dynamic production LLM workloads.
            - Coordinate scheduling, batching, and runtime resource management to sustain serving efficiency under workload variation.
        - BlendServe: Optimizing Offline Inference with Resource-Aware Batching [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790133)]
            - UC Berkeley & UW & UC Davis & Rice
            - Present a resource-aware batching framework for offline inference.
            - Form batches against actual compute and memory bottlenecks to improve throughput.
    - MoE Inference
        - MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790187)]
            - SJTU & CUHK
            - Introduce an MoE inference system with adaptive-precision expert offloading.
            - Jointly tune expert offloading and precision to reduce memory pressure during serving.
    - Compression
        - ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790250)] [[arXiv](https://arxiv.org/abs/2603.17435)] [[Code](https://github.com/xxyux/ZipServ)]
            - HKUST-GZ & HIT-SZ & HKUST
            - Introduce hardware-aware lossless compression for LLM inference.
            - Reduce memory footprint while preserving exact model behavior and improving serving efficiency.
    - Speculative Decoding
        - DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790153)]
            - SJTU & Eastern Institute of Technology, Ningbo & Southeast University & Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo
            - Propose a heterogeneous speculative decoding architecture with FPGA draft generation and GPU verification.
            - Pipeline draft and verify across devices to reduce end-to-end decoding latency.
        - SwiftSpec: Disaggregated Speculative Decoding and Fused Kernels for Low-Latency LLM Inference [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790246)]
            - ByteDance Seed & UChicago
            - Introduce disaggregated speculative decoding together with fused kernels for low-latency LLM inference.
            - Combine system-level disaggregation and kernel-level optimization to make speculative decoding practical in deployment.
    - Sparsity
        - SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790224)]
            - SJTU & Infinigence-AI & SII & THU
            - Introduce speculative context sparsity for long-context reasoning in LLMs.
            - Avoid uniform full-context processing by speculating over sparse context usage during long-input inference.
    - Attention Mechanisms
        - I/O Analysis is All You Need: An I/O Analysis for Long-Sequence Attention [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790174)]
            - IIT & ICT, CAS & UCAS
            - Present an I/O-centric analysis framework for long-sequence attention.
            - Show that data movement, rather than FLOPs alone, dominates long-context attention cost.
        - PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790200)]
            - TJU & Stevens Institute of Technology
            - Introduce prefix-aware attention together with a multi-tile kernel for LLM decoding.
            - Reduce decode latency by exploiting shared prefixes while keeping GPU resource usage under control.
    - Value Level Parallelism (VLP)
        - Mugi: Value Level Parallelism For Efficient LLMs [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790189)]
            - CMU & UCF
            - Introduce value-level parallelism as a new execution dimension for LLM inference.
            - Exploit finer-grained parallel structure than conventional tensor or sequence parallelism.
    - KV Cache Offloading
        - REPA: Reconfigurable PIM for the Joint Acceleration of KV Cache Offloading and Processing [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790212)]
            - SJTU
            - Present a reconfigurable PIM architecture for jointly offloading and processing KV cache.
            - Co-design KV movement and KV computation to reduce host-memory bottlenecks during inference.
        - STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790226)]
            - RPI & UMass Amherst & IBM Research
            - Introduce selective token access with remapping and clustering for PIM-based LLM decoding.
            - Reduce unnecessary KV accesses and improve data locality during decoding.
- Language Processing Units (LPUs)
    - Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790169)]
        - ICT, CAS & USTC & IS, CAS & Cambricon Technologies
        - Propose Language Processing Units (LPUs) as a language-centric hardware substrate for general-purpose cognitive workloads.
        - Specialize the architecture around language processing primitives to improve efficiency on language-centric tasks.

# **Generative Recommenders (GRs)**

- GR Serving
    - BAT: Efficient Generative Recommender Serving with Bipartite Attention [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790131)]
        - ZJU & HKU & Alibaba & NUS & Aalto University
        - Introduce bipartite attention for generative recommender serving.
        - Tailor the serving design to recommendation-style generative workloads rather than generic LLM inference.

# **Diffusion Models**

- Video DiT Training
    - DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training [[Paper](https://dl.acm.org/doi/10.1145/3760250.3762216)] [[arXiv](https://arxiv.org/abs/2502.07590)]
        - CUHK & StepFun
        - Exploit dynamic sparsity to accelerate large-scale video DiT training.
        - Use hybrid sparsity-aware context parallelism to rebalance workloads under heterogeneous attention sparsity.
- Diffusion Model Serving
    - TetriServe: Efficiently Serving Mixed DiT Workloads [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790233)] [[arXiv](https://arxiv.org/abs/2602.05116)] [[Code](https://github.com/DiT-Serving/TetriServe)]
        - UMich & UW-Madison & NTU
        - Present a serving system for mixed DiT workloads.
        - Coordinate scheduling and batching across heterogeneous diffusion requests in a shared runtime.
- Mixture-of-Diffusion Models
    - MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models [[Paper](https://dl.acm.org/doi/10.1145/3760250.3762220)] [[Code](https://github.com/stsxxx/MoDM)]
        - UMich & Intel Labs
        - Introduce mixture-of-diffusion models for image generation serving.
        - Use specialization across diffusion sub-models to improve efficiency and quality-cost tradeoffs.

**PPoPP 2026**

# **LLM**

- LLM inference
    - JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference
        - WHU
    - Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving
        - SYSU
    - High-Throughput Non-Uniformly Quantized 3-bit LLM Inference
        - CUHK & HKUST
    - Accelerating Sparse Transformer Inference on GPU
        - CUP-Beijing & BUAA
- Attention
    - FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
        - USTC & ICT, CAS
    - MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends [[arXiv](https://arxiv.org/abs/2502.15349)] [[Code](https://github.com/microsoft/AttentionEngine)]
        - SJTU, IPADS & PKU & MSRA

# **Diffusion Models**

- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models
    - THU
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
    - UWaterloo & CMU & Rice

# **GNN**

- APERTURE: Algorithm-System Co-Optimization for Temporal Graph Network Inference
    - BUAA
- ElasGNN: An Elastic Training Framework for Distributed GNN Training
    - BUAA
- TAC: Cache-based System for Accelerating Billion-Scale GNN Training on Multi-GPU Platform
    - UCAS

# **Sparse Matrix**

- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration
    - SYSU
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
    - BUAA
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix–Sparse Vector Multiplication on GPUs
    - THU

# **Quantization**

- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization [[Artifact](https://github.com/thu-pacman/RoMeo)]
    - THU

# **Cache Management**

- Cacheman: A Comprehensive Last-Level Cache Management System for Multi-tenant Clouds
    - Alibaba Cloud

# **Misc**

- Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters
    - GaTech
- zBuffer: Zero-Copy and Metadata-Free Serialization for Fast RPC with Scatter-Gather Reflection
    - XMU & Alibaba & SJTU

**HPCA26**

# **LLM**

- LLM inference
    - AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving [[Paper](https://www.cs.sjtu.edu.cn/~lichao/publications/AUM_Unleashing_HPCA-2026-Wang.pdf)]
        - SJTU & Alibaba
    - GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference
        - KAIST
    - ELORA: Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving
        - SJTU & Huawei Cloud & HKUST
    - Towards Resource-Efficient Serverless LLM Inference with SLINFER [[arXiv](https://arxiv.org/abs/2507.00507)]
        - SJTU
    - LILo: Harnessing the On-chip Accelerators in Intel CPUs for Compressed LLM Inference Acceleration
        - UIUC & Seoul National University & Intel
    - PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System [[arXiv](https://arxiv.org/abs/2412.20166)]
        - Hanyang University & SK hynix & KAIST
- Speculative decoding
    - Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
        - HUST
- Quantization
    - BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache [[arXiv](https://arxiv.org/abs/2503.18773)]
        - Edinburgh & MSRA
    - AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization
        - Institute of Science Tokyo
- Reasoning
    - The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective [[arXiv](https://arxiv.org/abs/2506.04301)]
        - KAIST
    - PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models
        - KAIST
    - RPU - A Reasoning Processing Unit
        - Harvard
- RAG
    - VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG [[arXiv](https://arxiv.org/abs/2504.08930)]
        - GaTech
- VLM
    - Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models [[arXiv](https://arxiv.org/abs/2512.14661)] [[Code](https://github.com/dubcyfor3/Focus)]
        - Duke
- Video LLM
    - V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval [[arXiv](https://arxiv.org/abs/2512.12284)]
        - KAIST
- Misc
    - Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems
        - SJTU & Huawei
    - RoMe: Row Granularity Access Memory System for Large Language Models [[arXiv](https://arxiv.org/abs/2512.01541)]
        - Seoul National University & Meta
    - LEGO: Supporting LLM-enhanced Games with One Gaming GPU
        - SJTU & Tongji University

# **GPU**

- UVM
    - ARIADNE: Adaptive UVM Management for Efficient GPU Memory Oversubscription [[Artifact](https://zenodo.org/records/17852674)]
        - Yonsei University & DGIST
- Chiplet
    - COMET: Communication and Memory Co-Design for Fine-Grained AI Inference in MCM Accelerators
        - NUDT & PKU
    - Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem
        - NUDT
    - LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture
        - SYSU
- Sparsity
    - Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs
        - Hunan University
    - Uni-STC: Unified Sparse Tensor Core
        - CUP-Beijing & NUDT
- Misc
    - QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs
        - University of Murcia & William&Mary & NVIDIA
    - μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs
        - TJU
    - FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive operators via Inter-Core Connection [[arXiv](https://arxiv.org/abs/2512.12949)]
        - SJTU

# **VAR**

- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy
    - HKUST

**EuSYS26**

# **Large Language Models (LLMs)**

- LLM Inference
    - Speculative Decoding
        - AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding
            - CMU & Princeton & EPFL & AWS & Purdue
    - Request Scheduling
        - FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters [[Paper](https://doi.org/10.1145/3767295.3769316)] [[arXiv](https://arxiv.org/abs/2510.11938)]
            - SIAT, CAS & UCAS & UCSD & University of Macau
        - TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling
            - SJTU & GMU & China Telecom Shanghai
        - AdaGen: Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving
            - UVA & HPE Labs & UC Riverside
        - SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference
            - UC Berkeley & RUC & Rice
        - PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
            - ShanghaiTech & SenseTime & Beihang
    - KV Cache and Memory Management
        - Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading [[Paper](https://doi.org/10.1145/3767295.3769319)] [[arXiv](https://arxiv.org/abs/2502.05370)]
            - Stevens Institute of Technology & Waterloo & Rutgers
        - KUNSERVE: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving
            - SJTU
        - High Throughput and Low Latency LLM Serving via Adaptive KV Caching
            - University of Macau & SIAT, CAS & NTU
    - Multiplexing
        - MFS: An Efficient Model Family Serving System for LLMs
            - HKUST & USTC & Inspur
        - Efficient Multimodal Serving via Module Multiplexing
            - HKUST & SYSU & XJTU & MetaX
    - Sparsity
        - SAS: Sparse Attention Synthesizer for Efficient Language Model Inference
            - Amazon
    - Heterogeneous Environment
        - Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
            - THU & USTC & MSR & AIR, THU
        - TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation
            - BUPT
    - Trusted Execution
        - TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone
            - SJTU
    - LLM-based Applications
        - AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments
            - UVA & Microsoft
        - From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents
            - IS, CAS & UCAS & SJTU

# **Diffusion Models**

- Image Editing
    - FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling [[arXiv](https://arxiv.org/abs/2505.20600)] [[Code](https://github.com/Sylvia-16/FlashPS)]
        - HKUST & Alibaba

# **Model Serving**

- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
    - NJU & Hunan University

# **Resource Management**

- Serverless Computing
    - Efficient Data Passing for Serverless Inference Workflows: A GPU-Centric Approach
        - HUST & CUHK-Shenzhen & TeleAI & HKUST
    - iRoute: Local Routing Table-based Workflow Management in Serverless Computing
        - TJU & THU & IEIT Systems & Inspur
    - DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions
        - Waterloo & MSR & Microsoft
    - Squeezy: Rapid VM Memory Reclamation for Serverless Functions
        - NTUA & UIUC
    - Demystifying Serverless Costs on Public Platforms: Bridging Billing, Architecture, and OS Scheduling
        - UBC & Johns Hopkins
    - Fix: externalizing network I/O in serverless computing
        - Stanford
- GPU Cluster Management
    - Bridging the GPU Utilization Gap: Predictive Multi-Dimensional Resource Scheduling for AI Workloads
        - THU & Alibaba & SJTU
    - Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings [[Paper](https://hal.science/hal-05291033v1/file/GPU_power_Eurosys.pdf)]
        - ÉTS & Inria & OVHcloud & CNRS
        - Present practical job-level power estimation methods for GPUs under temporal sharing, spatial sharing, and passthrough deployment modes in cloud environments.
        - Show that GPU sharing can improve energy efficiency for small AI workloads, and identify substantial GPU underutilization in an IaaS GPU cluster.