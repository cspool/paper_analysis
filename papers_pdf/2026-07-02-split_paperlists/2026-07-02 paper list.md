
# LLM System

[https://github.com/AmberLJC/LLMSys-PaperList\]
## Serving

- [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://arxiv.org/abs/2505.11329) | [Code](https://github.com/microsoft/tokenweave) | MLSys' 26
- [AIConfigurator](https://arxiv.org/abs/2601.06288): Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving
- [SuperInfer](https://arxiv.org/abs/2601.20309): SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips | MLSys' 26
- [Scaling Up Efficient Small Language Models Serving](https://arxiv.org/abs/2510.22101): Serving and Deployment for Semantic Job Search | MLSys' 26
- [OptiKIT](https://arxiv.org/abs/2601.20408): Meeting SLOs, Slashing Hours - Automated Enterprise LLM Optimization | MLSys' 26
- [BlendServe](https://dl.acm.org/doi/abs/10.1145/3779212.3790133): Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching | ASPLOS' 26
- [SwiftSpec](https://dl.acm.org/doi/abs/10.1145/3779212.3790246): Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding with Disaggregated Pipeline and Fused Kernels | ASPLOS' 26
- [MuxWise](https://dl.acm.org/doi/abs/10.1145/3779212.3790236): Towards High-Goodput LLM Serving with Prefill-decode Multiplexing | ASPLOS' 26
- [MoEless](https://arxiv.org/abs/2603.06350): Efficient MoE LLM Serving via Serverless Computing
- [BiScale](https://arxiv.org/abs/2602.18755): Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS
- [Harvest](https://arxiv.org/abs/2602.00328): Opportunistic Peer-to-Peer GPU Caching for LLM Inference
- [MineDraft](https://arxiv.org/abs/2603.18016): A Framework for Batch Parallel Speculative Decoding — overlaps drafting and verification across two batches, hiding draft latency. Up to +75% throughput, -39% latency. Integrated into vLLM. | NUS & MIT
- [Foundry](https://arxiv.org/abs/2604.06664): Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start
- [AdaServe](https://arxiv.org/abs/2501.12162): Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding | EuroSys' 26
- [FlexPipe](https://arxiv.org/abs/2510.11938): Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters | EuroSys' 26
- [Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading](https://dl.acm.org/doi/10.1145/3767295.3769319) | EuroSys' 26
- [KunServe](https://arxiv.org/abs/2412.18169): Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving | EuroSys' 26
- [AdaGen](https://dl.acm.org/doi/10.1145/3767295.3769345): Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving | EuroSys' 26
- [SkyWalker](https://arxiv.org/abs/2505.24095): A Locality-Aware Cross-Region Load Balancer for LLM Inference | EuroSys' 26
- [High Throughput and Low Latency LLM Serving via Adaptive KV Caching](https://dl.acm.org/doi/10.1145/3767295.3803570) | EuroSys' 26
- [PARD](https://dl.acm.org/doi/10.1145/3767295.3803581): Enhancing Goodput for Inference Pipeline via Proactive Request Dropping | EuroSys' 26
- [PiLLM](https://dl.acm.org/doi/10.1145/3767295.3769393): Resource-Efficient LLM Inference Using Workload Prediction | EuroSys' 26
- [Automated End-to-End Model Serving with Cooperative Compilation and Scheduling](https://dl.acm.org/doi/10.1145/3767295.3769392) | EuroSys' 26
- [MFS](https://dl.acm.org/doi/10.1145/3767295.3769355): An Efficient Model Family Serving System for LLMs | EuroSys' 26
- [CRAFT](https://arxiv.org/abs/2603.28768): Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations for Efficient MoE Serving | MLSys' 26
- [MorphServe](https://arxiv.org/abs/2506.02006): Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing | MLSys' 26
- [FlexiCache](https://arxiv.org/abs/2511.00868): Leveraging Temporal Stability of Attention Heads for Efficient KV Cache Management | MLSys' 26
- [Kitty](https://arxiv.org/abs/2511.18643): Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost | MLSys' 26
- [SkipKV](https://arxiv.org/abs/2512.07993): Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models | MLSys' 26
- [BOute](https://arxiv.org/abs/2602.10729): Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization | MLSys' 26
- [From Tokens to Layers](https://arxiv.org/abs/2510.08055): Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill | MLSys' 26
- [HELIOS](https://arxiv.org/abs/2504.10724): Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving | MLSys' 26
- [BatchLLM](https://arxiv.org/abs/2412.03594): Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching | MLSys' 26
- [GhostServe](https://arxiv.org/abs/2605.00831): A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving | MLSys' 26
- [PRISM](https://arxiv.org/abs/2602.01762): Parametrically Refactoring Inference for Speculative Decoding Draft Models | MLSys' 26
- [FarSkip-Collective](https://arxiv.org/abs/2511.11505): Unhobbling Blocking Communication in Mixture of Experts Models | MLSys' 26
- [Efficient Data Passing for Serverless Inference Workflows: A GPU-Centric Approach](https://dl.acm.org/doi/10.1145/3767295.3769336) | EuroSys' 26
- [TrustWeave: Integrity Measurement and Attestation for Multi-Cloud LLMs](https://dl.acm.org/doi/10.1145/3767295.3803586) | EuroSys' 26
- [Stream2LLM: Overlapping Context Streaming and Prefill for Low-Latency LLM Serving](https://mlsys.org/virtual/2026/oral/3842) | MLSys' 26
- [Locality-Aware Beam Scheduling for Efficient Test-Time Compute](https://mlsys.org/virtual/2026/oral/3788) | MLSys' 26
- [Optimizing Deployment Configurations for LLM Inference](https://mlsys.org/virtual/2026/oral/3780) | MLSys' 26
- [ContextPilot: Fast Long-Context Inference via Context Reuse](https://mlsys.org/virtual/2026/oral/3810) | MLSys' 26
- [Speculative Decoding: Performance or Illusion?](https://mlsys.org/virtual/2026/oral/3782) | MLSys' 26
- [SHIP: SRAM-Based Huge Inference Pipelines for Fast LLM Serving](https://mlsys.org/virtual/2026/oral/3834) | MLSys' 26
- [BEAM: Joint Resource-Power Optimization for LLM Inference](https://mlsys.org/virtual/2026/oral/3849) | MLSys' 26
- [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://mlsys.org/virtual/2026/oral/3819) | MLSys' 26
- [PLA-Serve: Prefill-Length-Aware LLM Serving System](https://mlsys.org/virtual/2026/oral/3787) | MLSys' 26
- [Accelerating Reasoning Model Inference with Sparse Self-Speculative Decoding](https://mlsys.org/virtual/2026/oral/3733) | MLSys' 26
- [FaaScale: Unlocking Fast LLM Scaling for Serverless Inference](https://mlsys.org/virtual/2026/oral/3769) | MLSys' 26
- [Breaking the Ice: Analyzing Cold Start Latency in vLLM](https://mlsys.org/virtual/2026/oral/3784) | MLSys' 26
- [Demystifying the Mixture of Experts Serving Tax](https://mlsys.org/virtual/2026/oral/3764) | MLSys' 26
- [RaidServe: High-Performance Resilient LLM Serving](https://mlsys.org/virtual/2026/oral/3856) | MLSys' 26
- [Toward Principled LLM Safety Testing: Solving the Jailbreak Oracle Problem](https://mlsys.org/virtual/2026/oral/3739) | MLSys' 26
- [ZeRO-Prefill](https://arxiv.org/abs/2605.02960): Zero Redundancy Overheads in MoE Prefill Serving
- [SkyNomad](https://arxiv.org/abs/2601.06520): On Using Multi-Region Spot Instances to Minimize AI Batch Job Cost | UCB
- [Qrita](https://arxiv.org/abs/2602.01518): High-performance Top-k and Top-p Algorithm for GPUs using Pivot-based Truncation and Selection | UCB
- [The Time is Here for Just-in-Time Systems](https://arxiv.org/abs/2605.24096): Challenges and Opportunities for Adaptive AI/ML Serving | UCB
- [Event Tensor](https://arxiv.org/abs/2604.13327): A Unified Abstraction for Compiling Dynamic Megakernels for Low-Latency Serving | MLSys' 26
- [XGrammar-2](https://arxiv.org/abs/2601.04426): Efficient Dynamic Structured Generation Engine for Agentic LLMs | CMU
- [Prism (Superoptimizer)](https://arxiv.org/abs/2604.15272): Symbolic Superoptimization of Tensor Programs | CMU
- [Coral](https://arxiv.org/abs/2605.04357): Cost-Efficient Multi-LLM Serving over Heterogeneous Cloud GPUs | CMU
- [LAPS](https://arxiv.org/abs/2601.11589): A Length-Aware-Prefill LLM Serving System | CMU/MBZUAI
- [Where Do the Joules Go?](https://arxiv.org/abs/2601.22076): Diagnosing Inference Energy Consumption | UMich
- [SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal): Enabling Compute-Memory Disaggregation in LLM Serving Systems | NSDI' 26
- [OpenTela](https://systems.ethz.ch/news-and-events/news/2026/03/two-papers-accepted-at-osdi26.html): Unifying Decentralized Computing Resources for Heterogeneous LLM Serving | OSDI' 26
- [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251): Parallelizing the Speculate-then-Verify Dependency | ICLR' 26
- [WWW.Serve](https://arxiv.org/abs/2603.20661): Interconnecting Global LLM Services through Decentralization | CMU
- [Not All Prefills Are Equal](https://arxiv.org/abs/2603.13358): PPD Disaggregation for Multi-turn LLM Serving | UChicago
- [Hexcute](https://arxiv.org/abs/2504.16214): A Compiler Framework for Automating Layout Synthesis in GPU Programs | CGO' 26
- [AutoScout](https://arxiv.org/abs/2603.11603): Structured Optimization for Automating ML System Configuration | UT Austin
- [VoltanaLLM](https://arxiv.org/abs/2509.04827): Energy-Efficient and SLO-Aware Disaggregated LLM Serving | ISC' 26
- [Regulating Branch Parallelism in LLM Serving](https://arxiv.org/abs/2605.06914): Balancing Intra-Request Branch Admission against Co-batched Latency | Stanford
- [Strata](https://arxiv.org/abs/2508.18572): Hierarchical Context Caching for Long-Context LLM Serving | OSDI' 26


## agent system
- [DualPath](https://arxiv.org/abs/2602.21548): Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference | DeepSeek
- [AIMS](https://dl.acm.org/doi/10.1145/3767295.3803622): Cost-Efficient LLM-Based Agent Deployment in Hybrid Cloud-Edge Environments | EuroSys' 26
- [From Imperative to Declarative](https://dl.acm.org/doi/10.1145/3767295.3803576): Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents | EuroSys' 26
- [Hippocampus](https://arxiv.org/abs/2602.13594): An Efficient and Scalable Memory Module for Agentic AI | MLSys' 26
- [PROMPTS: Performance Optimization via Multi-Agent Planning for Test-time Compute Scaling](https://mlsys.org/virtual/2026/oral/3843) | MLSys' 26
- [TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval](https://mlsys.org/virtual/2026/poster/3573) | MLSys' 26
- [OpenHands Software Agent SDK](https://mlsys.org/virtual/2026/poster/3526) | MLSys' 26
- [FlashAgents: Accelerating Multi-Agent LLM Systems via Streaming Prefill Overlap](https://mlsys.org/virtual/2026/poster/3537) | MLSys' 26
- [AgenticCache: Cache-Driven Asynchronous Planning for Agentic LLM Systems](https://mlsys.org/virtual/2026/oral/3806) | MLSys' 26
- [Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation](https://mlsys.org/virtual/2026/oral/3753) | MLSys' 26
- [Ontology-Guided Long-Term Agent Memory for Conversational RAG](https://mlsys.org/virtual/2026/oral/3738) | MLSys' 26
- [OSWorld-Human: Benchmarking Efficiency of Computer-Use Agents](https://mlsys.org/virtual/2026/oral/3865) | MLSys' 26
- [KAIROS](https://arxiv.org/abs/2604.16682): Stateful, Context-Aware, Power-Efficient Agentic Inference Serving | UMich
- [ThunderAgent](https://arxiv.org/abs/2602.13692): A Simple, Fast and Program-Aware Agentic Inference System | CMU
- [Vortex](https://arxiv.org/abs/2606.06453): Efficient and Programmable Sparse Attention Serving for AI Agents | CMU
- [Nalar](https://arxiv.org/abs/2601.05109): An Agent Serving Framework Separating Workflow Specification from Execution | UT Austin
- [Concurrency without Model Changes](https://arxiv.org/abs/2605.15077): Future-based Asynchronous Function Calling for LLMs | UCB
- [Agent JIT Compilation](https://arxiv.org/abs/2605.21470): Latency-Optimizing Web Agent Planning and Scheduling | ICML' 26
- [SemWeave](https://dl.acm.org/doi/10.1145/3788853.3801593): Semantic Common Expressions for LLM-powered Query Processing | SIGMOD' 26

## edge serving
- [TZ-LLM](https://dl.acm.org/doi/10.1145/3767295.3769334): Protecting On-Device Large Language Models with Arm TrustZone | EuroSys' 26
- [TailorLLM](https://dl.acm.org/doi/10.1145/3767295.3769346): Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation | EuroSys' 26
- [Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices](https://dl.acm.org/doi/10.1145/3767295.3769329) | EuroSys' 26
- [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | EuroSys' 26
- [On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding](https://dl.acm.org/doi/10.1145/3767295.3803572) | EuroSys' 26
- [SwiftFL: Enabling Speculative Training for On-Device Federated Deep Learning](https://dl.acm.org/doi/10.1145/3767295.3803605) | EuroSys' 26
- [viNPU: Optimizing Vision Transformer Inference on Mobile NPUs](https://dl.acm.org/doi/10.1145/3767295.3803619) | EuroSys' 26
- [Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices](https://mlsys.org/virtual/2026/oral/3802) | MLSys' 26
- [Rethinking DVFS for Mobile LLMs: CORE for Energy-Efficient On-Device Inference](https://mlsys.org/virtual/2026/oral/3814) | MLSys' 26
- [IntAttention: Fully Integer Attention Pipeline for Edge LLM Inference](https://mlsys.org/virtual/2026/oral/3848) | MLSys' 26
- [OpenJarvis](https://arxiv.org/abs/2605.17172): Personal AI, On Personal Devices via LLM-Guided Spec Search and Local-Cloud Collaboration | Stanford

## system  model codesign
- [Reducing GPU Memory Fragmentation via Spatio-Temporal Allocation Planning](https://arxiv.org/abs/2507.16274) | EuroSys' 26
- [SAS](https://dl.acm.org/doi/10.1145/3767295.3769364): Sparse Attention Synthesizer for Efficient Language Model Inference | EuroSys' 26
- [LLMFolder](https://dl.acm.org/doi/10.1145/3767295.3769339): Revisiting Constant Folding in Large Language Models | EuroSys' 26
- [FlashAttention-4](https://arxiv.org/abs/2603.05451): Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling (Blackwell) | MLSys' 26
- [BLASST: Dynamic Blocked Attention Sparsity for Scalable Transformer Inference](https://mlsys.org/virtual/2026/poster/3631) | MLSys' 26
- [Attribution-based Sparse Activation in Large Language Models](https://mlsys.org/virtual/2026/poster/3556) | MLSys' 26
- [MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings](https://mlsys.org/virtual/2026/oral/3805) | MLSys' 26
- [MAC-Attention: Match-Amend-Complete Attention for Efficient Long-Context Inference](https://mlsys.org/virtual/2026/oral/3794) | MLSys' 26
- [Flashlight: PyTorch Compiler Extensions for Attention Variants](https://mlsys.org/virtual/2026/poster/3540) | MLSys' 26
- [CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training](https://mlsys.org/virtual/2026/oral/3841) | MLSys' 26
- [OPKV: Recallable Sparsity in Paged KV Cache for Efficient LLM Inference](https://mlsys.org/virtual/2026/poster/3621) | MLSys' 26
- [Using Span Queries to Optimize Cache and Attention Locality](https://mlsys.org/virtual/2026/oral/3747) | MLSys' 26
- [SLA2](https://arxiv.org/abs/2602.12675): Sparse-Linear Attention with Learnable Routing and QAT | UCB
- [SageBwd](https://arxiv.org/abs/2603.02170): A Trainable Low-bit Attention for Efficient Training | UCB
- [Inference Time Context Sparsity](https://arxiv.org/abs/2605.24168): Illusion or Opportunity? | UCB
- [Tilus](https://arxiv.org/abs/2504.12984): A Tile-Level GPGPU Programming Language for Low-Precision Computation | ASPLOS' 26
- [PuzzleMoE](https://arxiv.org/abs/2511.04805): Efficient Compression of Large MoE Models via Sparse Expert Merging and Bit-packed Inference | ICML' 26
- [Sieve](https://arxiv.org/abs/2605.11277): Dynamic Expert-Aware PIM Acceleration for Evolving Mixture-of-Experts Models | Stanford
- [Streaming Tensor Programs](https://arxiv.org/abs/2511.07776): A Streaming Abstraction for Dynamic Parallelism on Dataflow Accelerators | ASPLOS' 26
- [FuseFlow](https://arxiv.org/abs/2511.04768): A Fusion-Centric Compilation Framework for Sparse Deep Learning on Streaming Dataflow | ASPLOS' 26


## MultiModal Serving

- [xDiT](https://arxiv.org/abs/2411.01738): an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism
- [MOSEL](https://arxiv.org/pdf/2310.18481.pdf): Inference Serving Using Dynamic Modality Selection
- [Approximate Caching for Efficiently Serving Diffusion Models](https://arxiv.org/abs/2312.04429) | Adobe Research
- [Generative AI Beyond LLMs](https://arxiv.org/pdf/2312.14385): System Implications of Multi-Modal Generation | Meta
- [Characterizing and Efficiently Accelerating Multimodal Generation Model Inference](https://arxiv.org/abs/2410.00215) | Meta
- [DistriFusion:](https://arxiv.org/abs/2402.19481) Distributed Parallel Inference for High-Resolution Diffusion Models | MIT
- [LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://arxiv.org/abs/2408.10188) | NVIDIA
- [FlexCache: Flexible Approximate Cache System for Video Diffusion](https://arxiv.org/abs/2501.04012) | University of Waterloo
- [DDiT](https://arxiv.org/abs/2506.13497v1): Dynamic Resource Allocation for Diffusion Transformer Model Serving
- [PATCHEDSERVE](https://arxiv.org/pdf/2501.09253): A Patch Management Framework for SLO-Optimized Hybrid Resolution Diffusion Serving
- [ElasticMM](https://arxiv.org/abs/2507.10069): Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism
- [TetriServe](https://arxiv.org/abs/2510.01565): Efficient DiT Serving for Heterogeneous Image Generation
- [dInfer](https://arxiv.org/abs/2510.08666): An Efficient Inference Framework for Diffusion Language Models
- [Fast-dLLM v2](https://arxiv.org/abs/2509.26328): Efficient Block-Diffusion LLM
- [Argus](https://arxiv.org/abs/2511.06724): Quality-Aware High-Throughput Text-to-Image Inference Serving System
- [Cornserve](https://arxiv.org/abs/2512.14098): Efficiently Serving Any-to-Any Multimodal Models
- [HydraInfer](https://arxiv.org/abs/2505.12658): Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving
- [Enabling Disaggregated Multi-Stage MLLM Inference via GPU-Internal Scheduling and Resource Sharing](https://arxiv.org/abs/2512.17574)
- [VoxServe](https://arxiv.org/abs/2602.00269): Streaming-Centric Serving System for Speech Language Models
- [dLLM-Serve](https://arxiv.org/abs/2512.17077): Taming the Memory Footprint Crisis for Efficient Diffusion LLM Serving
- [HADIS](https://arxiv.org/abs/2509.00642): Hybrid Adaptive Diffusion Model Serving for Efficient Text-to-Image Generation
- [Efficient Multimodal Serving via Module Multiplexing](https://dl.acm.org/doi/10.1145/3767295.3769389) | EuroSys' 26
- [FlashPS](https://dl.acm.org/doi/10.1145/3767295.3769379): Efficient Generative Image Editing with Mask-aware Caching and Scheduling | EuroSys' 26
- [StreamDiffusionV2](https://arxiv.org/abs/2511.07399): A Streaming System for Dynamic and Interactive Video Generation | MLSys' 26
- [SpecDiff-2](https://arxiv.org/abs/2511.00606): Scaling Diffusion Drafter Alignment For Faster Speculative Decoding | MLSys' 26
- [Million-Scale Text-to-Video Retrieval with Hyperdimensional Computing](https://dl.acm.org/doi/10.1145/3767295.3803610) | EuroSys' 26
- [TriInfer: Hybrid Encode-Prefill-Decode Disaggregation for Multimodal LLM Inference](https://mlsys.org/virtual/2026/oral/3756) | MLSys' 26
- [CDLM: Consistency Diffusion Language Models for Faster Text Generation Sampling](https://mlsys.org/virtual/2026/oral/3785) | MLSys' 26
- [db-SP: Accelerating Sparse Attention for Visual Generative Models](https://mlsys.org/virtual/2026/poster/3575) | MLSys' 26
- [TiDAR: Think in Diffusion, Talk in Autoregression for Multimodal Generation](https://mlsys.org/virtual/2026/poster/3528) | MLSys' 26
- [SlackServe](https://arxiv.org/abs/2606.15319): Adaptive Resource Management and Quality Control for Streaming Video Generation | Peking University, Purdue
- [Quant VideoGen](https://arxiv.org/abs/2602.02958): Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization | UCB
- [MonarchRT](https://arxiv.org/abs/2602.12271): Efficient Attention for Real-Time Video Generation | CMU
- [SwiftFusion](https://arxiv.org/abs/2601.20273): Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers on GPUs | Univ. of Toronto
- [d3LLM](https://arxiv.org/abs/2601.07568): Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation | UCSD
- [VLA-Perf](https://arxiv.org/abs/2602.18397): Demystifying VLA Inference Performance | Stanford

## Industry Report
- [Falcon-H1R: Pushing the Reasoning Frontiers with a Hybrid Model for Efficient Test-Time Scaling](https://arxiv.org/abs/2601.02346) – TII (Jan 2026)
- [Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking](https://arxiv.org/abs/2601.04720) – Alibaba (Jan 2026)
- [Ministral 3](https://arxiv.org/abs/2601.08584) – Mistral AI (Jan 2026)
- [TranslateGemma Technical Report](https://arxiv.org/abs/2601.09012) – Google / DeepMind (Jan 2026)
- [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337) – Alibaba (Jan 2026)
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) – Zhipu AI (Feb 2026)
- [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729) – Alibaba (Feb 2026)
- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804) – Alibaba (Apr 2026)
- [Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence](https://arxiv.org/abs/2604.24954) – NVIDIA (Apr 2026)
- [Granite Embedding Multilingual R2 Models](https://arxiv.org/abs/2605.13521) – IBM (May 2026)


# Long Context LLM
\[https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling#month-papers\]
## News
- **[2026.06.30]**
    
    - Paper: [RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference](https://arxiv.org/abs/2606.31519) [![GitHub Repo stars](https://camo.githubusercontent.com/f96cab4da9a5249d8190cfe6ffd7456bc54ecde3c839c66ada80240947ab0857/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f53616b7572616161302f5261426974514361636865)](https://github.com/Sakuraaa0/RaBitQCache)
    - Paper: [SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference](https://arxiv.org/abs/2606.31145) [![GitHub Repo stars](https://camo.githubusercontent.com/b5064718baba45eb2a498cdea9e9d586c69fd9d9b0396dc47cf2727dec4dea9c/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f416d6972416261736b6f68692f53654b56)](https://github.com/AmirAbaskohi/SeKV)
    - Paper: [CoLT: Teaching Multi-Modal Models to Think with Chain of Latent Thoughts](https://arxiv.org/abs/2606.31986) [![GitHub Repo stars](https://camo.githubusercontent.com/396bfde9a98aa0217072a7334becc692aae6a8072379c12d93661e4a23b1ac2f/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f68756c69616e797579792f436f4c54)](https://github.com/hulianyuyy/CoLT)
    - Paper: [MemLearner: Learning to Query Context memory for Video World Models](https://arxiv.org/abs/2606.31734) [![GitHub Repo stars](https://camo.githubusercontent.com/81010dc72d334f6673a14ae90d063d59ac310c900af62957ef8c06adb9d1e08a/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f79756a6977656e2f6d656d6c6561726e6572)](https://github.com/yujiwen/memlearner)
- **[2026.06.29]**
    
    - Paper: [Predict, Reuse, and Repair: Accelerating Dynamic Sparse Attention for Long-Context LLM Decoding](https://arxiv.org/abs/2606.30389) [![GitHub Repo stars](https://camo.githubusercontent.com/1a2ca07872a10b8459325237bbc8c54508dfca7e5ed33ab9a5e26607d92099e5/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f5469616e7975393734382f496e6372656d656e74616c5f466c617368417474656e74696f6e)](https://github.com/Tianyu9748/Incremental_FlashAttention)
- **[2026.06.26]**
    
    - Paper: [Reflect-R1: Evidence-Driven Reflection for Self-Correction in Long Video Understanding](https://arxiv.org/abs/2606.27922)
- **[2026.06.24]**
    
    - Paper: [Towards a Dynamic and Fixed-budget Memory Bank for Efficient Streaming Video Understanding](https://arxiv.org/abs/2606.25658) [![GitHub Repo stars](https://camo.githubusercontent.com/e0306b9daa324024e56d76f69621fe5efc29beb998feabb7323aa74f3d1c6273/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f686b746b30372f43617573616c4d656d)](https://github.com/hktk07/CausalMem)
- **[2026.06.23]**
    
    - Paper: [Qwen-AgentWorld: Language World Models for General Agents](https://arxiv.org/abs/2606.24597) [![GitHub Repo stars](https://camo.githubusercontent.com/df039d73748a8edc722b220d5a97be28eec005eeb30d91e267b77b41bcaf5686/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f5177656e4c4d2f5177656e2d4167656e74576f726c64)](https://github.com/QwenLM/Qwen-AgentWorld)
- **[2026.06.22]**
    
    - Paper: [DynamicMem: A Long-Horizon Memory Benchmark in Real-World Settings](https://arxiv.org/abs/2606.22877) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://wenyaxie023.github.io/DynamicMem/)
    - Paper: [Tapered Language Models](https://arxiv.org/abs/2606.23670)
- **[2026.06.18]**
    
    - Paper: [Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning](https://arxiv.org/abs/2606.20002) [![GitHub Repo stars](https://camo.githubusercontent.com/9feeb18b6fa93f00bce422a3c6751a9e45992fb91d8ad739d4ece44a09121541/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6167656e7473636f70652d61692f5472696e6974792d524654)](https://github.com/agentscope-ai/Trinity-RFT/tree/research/cod/examples/research_cod)
    - Paper: [ADaPT: Token-Level Decoupling for Efficient Large Reasoning Models](https://arxiv.org/abs/2606.19919)
    - Paper: [CARE: Competence-Aware Reward Shaping for Adaptive Reasoning Length in Video-MLLMs](https://arxiv.org/abs/2606.19927) [![GitHub Repo stars](https://camo.githubusercontent.com/01493b93bca2d0e089b357c3f8363ce05f7922dd4557ad5b08208ffe5ed28896/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f3150616e73792f566964656f2d43415245)](https://github.com/1Pansy/Video-CARE)
- **[2026.06.17]**
    
    - Paper: [GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents](https://arxiv.org/abs/2606.18829) [![GitHub Repo stars](https://camo.githubusercontent.com/3afa97404e4469ebb50790da975fa43d6aacf7622a741b91525fa99e7ed2947b/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f727a6875622f476174654d656d)](https://github.com/rzhub/GateMem)
- **[2026.06.16]**
    
    - Paper: [LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents](https://arxiv.org/abs/2606.18388)
    - Paper: [Looped World Models](https://arxiv.org/abs/2606.18208)
- **[2026.06.12]**
    
    - Paper: [StreamMemBench: Streaming Evaluation of Agent Memory for Future-Oriented Assistance](https://arxiv.org/abs/2606.14571) [![GitHub Repo stars](https://camo.githubusercontent.com/65bb309cd056beab783e18d4e7cf132f14355e1e995f8d67499ab90efce1e85e/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6c616e6469616e36302f53747265616d4d656d42656e6368)](https://github.com/landian60/StreamMemBench)
- **[2026.06.11]**
    
    - Paper: [EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments](https://arxiv.org/abs/2606.13681)
    - Paper: [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392) [![GitHub Repo stars](https://camo.githubusercontent.com/917f1e0b928abb22bd0ac2b9a63ad021c012014a85da23269120ffe849e3a464/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f4d696e694d61782d41492f4d5341)](https://github.com/MiniMax-AI/MSA)
    - Paper: [Learning What to Remember: A Cognitively Grounded Multi-Factor Value Model for Agentic Memory](https://arxiv.org/abs/2606.12945)
    - Paper: [Can I Buy Your KV Cache?](https://arxiv.org/abs/2606.13361)
    - Paper: [Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning](https://arxiv.org/abs/2606.13106)
- **[2026.06.09]**
    
    - Paper: [Parallel Causal Associative Fields: Gated Sparse Memory for Long-Context Language Modeling](https://arxiv.org/abs/2606.10435) [![GitHub Repo stars](https://camo.githubusercontent.com/14be386410e85bc44e170e5068e4e3f509e42d0502d71bbf8ea989fe275a7546/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f61686d65643132336864732f50434146)](https://github.com/ahmed123hds/PCAF)
- **[2026.06.08]**
    
    - Paper: [IS-CoT: Breaking the Long-form Generation Collapse via Interleaved Structural Thinking](https://arxiv.org/abs/2606.09709)
    - Paper: [Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents](https://arxiv.org/abs/2606.09483)
    - Paper: [H2HMem: A Multimodal Memory Benchmark for Agents in Human-Human Interactions](https://arxiv.org/abs/2606.09461)
    - Paper: [FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention](https://arxiv.org/abs/2606.09079) [![GitHub Repo stars](https://camo.githubusercontent.com/366d347b8f8ba19fcc1498269eebc65a277ce64c0367e2f2e233b50dae830c96/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6c69626572747977696e672f466c6173684d656d6f72792d446565707365656b2d5634)](https://github.com/libertywing/FlashMemory-Deepseek-V4)
- **[2026.06.07]**
    
    - Paper: [Sparrow: Sparse Rollout for Stable and Efficient Long-context RL of Large Language Models](https://arxiv.org/abs/2606.08446)
    - Paper: [Look Less, Reason More: Block-wise Attention Skipping for Efficient Multimodal LLMs](https://arxiv.org/abs/2606.08511)
    - Paper: [From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](https://arxiv.org/abs/2606.08656) (ICML 2026)
- **[2026.06.06]**
    
    - Paper: [IntentKV: Cross-Turn Intent-Aware KV Cache Pruning for Agent Inference](https://arxiv.org/abs/2606.09916)
- **[2026.06.05]**
    
    - Paper: [How Much Dense Attention is Necessary? Oracle-Guided Sparse Prefill for Full/GQA Layers in Hybrid Long-Context Models](https://arxiv.org/abs/2606.07703)
    - Paper: [Rosetta Memory: Adaptive Memory for Cross-LLM Agents](https://arxiv.org/abs/2606.07711)
    - Paper: [SWE-Marathon: Can Agents Autonomously Complete Ultra-Long-Horizon Software Work?](https://arxiv.org/abs/2606.07682) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://swe-marathon.org/)
- **[2026.06.04]**
    
    - Paper: [TARPO: Token-Wise Latent-Explicit Reasoning via Action-Routing Policy Optimization](https://arxiv.org/abs/2606.05859) [![GitHub Repo stars](https://camo.githubusercontent.com/e4b25f821db1b07acc5967a7f512fe7ec19c767f520179265ca3c203a9e3b2f6/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f4e4b552d4c4954492f544152504f2d6d6173746572)](https://github.com/NKU-LITI/TARPO-master)
- **[2026.06.03]**
    
    - Paper: [Cartridges at Scale: Training Modular KV Caches over Large Document Collections](https://arxiv.org/abs/2606.04557)
    - Paper: [LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](https://arxiv.org/abs/2606.04302) (ICML 2026)
    - Paper: [SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511) [![GitHub Repo stars](https://camo.githubusercontent.com/b3c3435ddffdf900244c23d982c188b708994499ca85d5917db144c4e88d58db/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f4e566c6162732f537061724441)](https://github.com/NVlabs/SparDA)
    - Paper: [Depth-Attention: Cross-Layer Value Mixing for Language Models](https://arxiv.org/abs/2606.05014)
    - Paper: [Video2LoRA: Parametric Video Internalization for Vision-Language Models](https://arxiv.org/abs/2606.04351)
    - Paper: [Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance](https://arxiv.org/abs/2606.04970)
    - Paper: [Rethinking Continual Experience Internalization for Self-Evolving LLM Agents](https://arxiv.org/abs/2606.04703)
    - Paper: [Learning While Acting: A Skill-Enhanced Test-Time Co-Evolution Framework for Online Lifelong Learning Agents](https://arxiv.org/abs/2606.04815)
- **[2026.06.02]**
    
    - Paper: [KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks](https://arxiv.org/abs/2606.03458) [![GitHub Repo stars](https://camo.githubusercontent.com/16fea267d650ba8218d4d7030cebbcd046e06e9617ce5bdf929572d5ed522db0/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6875617765692d63736c2f4b5661724e)](https://github.com/huawei-csl/KVarN)
    - Paper: [Value-Aware Stochastic KV Cache Eviction for Reasoning Models](https://arxiv.org/abs/2606.03928) [![GitHub Repo stars](https://camo.githubusercontent.com/fae6fea92ce7c6fc15fa27b3d4ef35173edb7b504e485bfe6ab2dea9e6cc57de/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f7465726172616368616e672f56615345)](https://github.com/terarachang/VaSE)
- **[2026.06.01]**
    
    - Paper: [RESTORE: Improving Visual Token Reduction via Rectifying Distortions for Efficient Multimodal LLM Inference](https://arxiv.org/abs/2606.01711) (ICML 2026)
    - Paper: [PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning](https://arxiv.org/abs/2606.02443)
    - Paper: [LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning for Agentic Language Models](https://arxiv.org/abs/2606.01838)
    - Paper: [Attention-guided Fine-tuning of Multimodal Large Language Models Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2606.01558)
    - Paper: [Do Transformers Need Three Projections? Systematic Study of QKV Variants](https://arxiv.org/abs/2606.04032) [![GitHub Repo stars](https://camo.githubusercontent.com/3abb03587ce19865890d4f49923ad05aee20bdde9b6b5efd2fb383d7b8d2bb67/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f427261696e636869702d496e632f446f2d5472616e73666f726d6572732d4e6565642d332d50726f6a656374696f6e73)](https://github.com/Brainchip-Inc/Do-Transformers-Need-3-Projections)
- **[2026.05.30]**
    
    - Paper: [WaveFilter: Enhancing the Long-Context Capability of Diffusion LLMs via Wavelet-Guided KV Cache Filtering](https://arxiv.org/abs/2606.00724)
    - Paper: [ETC: Extreme Token Compression via Task-aware Visual Information Distillation in VLMs](https://arxiv.org/abs/2606.00543)
- **[2026.05.29]**
    
    - Paper: [LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards](https://arxiv.org/abs/2605.31584) [![GitHub Repo stars](https://camo.githubusercontent.com/dc74bb2ccfc5f8e311f97d8d3e04f13793c52a01ae0960b9546571e3241a7950/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f5448552d4b45472f4c6f6e675472616365524c)](https://github.com/THU-KEG/LongTraceRL)
    - Paper: [GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs](https://arxiv.org/abs/2605.31105)
    - Paper: [SlotMemory: Object-Centric KV Memory for Streaming Long-Video Generation](https://arxiv.org/abs/2605.31033) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://tj12323.github.io/SlotMemory/)
    - Paper: [SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling](https://arxiv.org/abs/2605.30750) (ICML 2026)
    - Paper: [PEEK: Picking Essential frames via Efficient Knowledge distillation](https://arxiv.org/abs/2605.31029) [![GitHub Repo stars](https://camo.githubusercontent.com/d5988fe566dea0a1bbad12d4fe3f52a498b23b96612a38db07f670223e658d67/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6d6f6d656e74736c61622f7065656b)](https://github.com/momentslab/peek)
    - Paper: [Towards Effective Long-Video Event Prediction via Multi-Level Event Semantics Mining](https://arxiv.org/abs/2605.31069)
    - Paper: [AdaptR1: Reinforcement Learning Based Adaptive Interleaved Thinking in Multi-hop Question Answering](https://arxiv.org/abs/2605.31062)
    - Paper: [Task-Focused Memorization for Multimodal Agents](https://arxiv.org/abs/2605.31075)
    - Paper: [ElasticMem: Latent Memory as a Learnable Resource for LLM Agents](https://arxiv.org/abs/2605.30690) [![GitHub Repo stars](https://camo.githubusercontent.com/ba6d51269da695c1fcced6e804ae51899e843412b3013859e436960ad3ae33f7/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f756c61622d756975632f456c61737469634d656d)](https://github.com/ulab-uiuc/ElasticMem)
    - Paper: [Beyond Static Dialogues: Benchmarking Realistic, Heterogeneous, and Evolving Long-Term Memory](https://arxiv.org/abs/2605.31086)
    - Paper: [EGOSTREAM: A Diagnostic Benchmark for Streaming Episodic Memory in Egocentric Vision](https://arxiv.org/abs/2605.31557)
    - Paper: [SAGE: A Novelty Gate for Efficient Memory Evolution in Agentic LLMs](https://arxiv.org/abs/2605.30711)
    - Paper: [ExpGraph: Model-Agnostic Experience Learning with Graph-Structured Memory for LLM Agents](https://arxiv.org/abs/2605.30712)
- **[2026.05.28]**
    
    - Paper: [OmniMem: Scalable and Adaptive Memory Retrieval for Long Video Generation](https://arxiv.org/abs/2605.30519) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://wuyushuwys.github.io/OmniMem/)
- **[2026.05.27]**
    
    - Paper: [Periodic RoPE for Infinite Context LLMs](https://arxiv.org/abs/2605.27980)
    - Paper: [ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay](https://arxiv.org/abs/2605.28069) [![GitHub Repo stars](https://camo.githubusercontent.com/db44cb8e9ee62f41975488a12f5211c9d705609317945607ac23bb720dbf0087/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f68757a686578696e2f5a6970524c)](https://github.com/huzhexin/ZipRL)
    - Paper: [CIVIC: End-to-End Sequence Compactness for Efficient Vision-Language Models](https://arxiv.org/abs/2605.28115)
    - Paper: [Self-Prophetic Decoding to Unlock Visual Search in LVLMs](https://arxiv.org/abs/2605.28741) (ICML 2026)
- **[2026.05.26]**
    
    - Paper: [OmniMem: Perturbation-aware Memory Compression for Streaming Audio-Visual LLMs](https://arxiv.org/abs/2606.07577)
    - Paper: [UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training](https://arxiv.org/abs/2605.27740)
    - Paper: [Hurwitz Quaternion Multiplicative Quantization for KV Cache Compression](https://arxiv.org/abs/2605.27646)
    - Paper: [Heterogeneous Parallelism for Multimodal Large Language Model Training](https://arxiv.org/abs/2605.27678)
    - Paper: [Tensor Memory: Fixed-Size Recurrent State for Long-Horizon Transformers](https://arxiv.org/abs/2605.27686) [![GitHub Repo stars](https://camo.githubusercontent.com/1bbad75e9f5dee0d07603ce8272c0b48b94e0e07774467c8999174d44763c98e/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6b737761696e39382f74656e736f722d6d656d6f7279)](https://github.com/kswain98/tensor-memory)
- **[2026.05.22]**
    
    - Paper: [Inference Time Context Sparsity: Illusion or Opportunity?](https://arxiv.org/abs/2605.24168)
    - Paper: [Parallel Context Compaction for Long-Horizon LLM Agent Serving](https://arxiv.org/abs/2605.23296)
- **[2026.05.21]**
    
    - Paper: [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791) [![GitHub Repo stars](https://camo.githubusercontent.com/b15f6444222b8d31cf5470d26a977e84be5a5446a23a2456b991409b0e2c5270/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f4e566c6162732f476174656444656c74614e65742d32)](https://github.com/NVlabs/GatedDeltaNet-2)
    - Paper: [ACC: Compiling Agent Trajectories for Long-Context Training](https://arxiv.org/abs/2605.21850)

### Month Papers

[](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling#month-papers)

Month Papers

- **[2026.05.14]**
    
    - Paper: [MemLens: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models](https://arxiv.org/abs/2605.14906) [![GitHub Repo stars](https://camo.githubusercontent.com/828ad0b9e832d677ec3ec7450070c5763ec0fb3e8683bbbdfdef46acbb9dfcba/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f7872656e61662f4d454d4c454e53)](https://github.com/xrenaf/MEMLENS)
    - Paper: [SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer](https://arxiv.org/abs/2605.15178) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://nvlabs.github.io/Sana/WM/)
    - Paper: [MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory](https://arxiv.org/abs/2605.15128) [![GitHub Repo stars](https://camo.githubusercontent.com/3cee47bd2edb73c62b0ab68259723ff0a4e29edd97a63469f64eeae608c9e9ea/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f4d696e67686f4b776f6b2f4d656d457965)](https://github.com/MinghoKwok/MemEye) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://minghokwok.github.io/MemEye/)
    - Paper: [GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations](https://arxiv.org/abs/2605.14498)
    - Paper: [Minimal-Intervention KV Retention: A Design-Space Study and a Diversity-Penalty Survivor](https://arxiv.org/abs/2605.14292)
    - Paper: [Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](https://arxiv.org/abs/2605.15184)
    - Paper: [Why Neighborhoods Matter: Traversal Context and Provenance in Agentic GraphRAG](https://arxiv.org/abs/2605.15109)
- **[2026.05.13]**
    
    - Paper: [KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving](https://arxiv.org/abs/2605.13734) (SIGCOMM 2026) [![GitHub Repo stars](https://camo.githubusercontent.com/bef6767e66400127de046ba0bf9abbfdec9f75225553c40597909e4814399663/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f68706470732d67726f75702f4b565365727665)](https://github.com/hpdps-group/KVServe)
    - Paper: [Thinking Ahead: Prospection-Guided Retrieval of Memory with Language Models](https://arxiv.org/abs/2605.14177) [![GitHub Repo stars](https://camo.githubusercontent.com/c15fbfbde014e3b6c9ee86ea8d39792311e9654a79aa1f96cfc6145e8aaeda16/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f68617273686974612d63686f7072612f5047522d6d656d)](https://github.com/harshita-chopra/PGR-mem)
    - Paper: [Cognifold: Always-On Proactive Memory via Cognitive Folding](https://arxiv.org/abs/2605.13438)
    - Paper: [Why Retrieval-Augmented Generation Fails: A Graph Perspective](https://arxiv.org/abs/2605.14192)
    - Paper: [Stop Overthinking: Unlocking Efficient Listwise Reranking with Minimal Reasoning](https://arxiv.org/abs/2605.14450)
- **[2026.05.12]**
    
    - Paper: [δ-mem: Efficient Online Memory for Large Language Models](https://arxiv.org/abs/2605.12357)
- **[2026.05.10]**
    
    - Paper: [Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction](https://arxiv.org/abs/2605.09649)
- **[2026.05.08]**
    
    - Paper: [Bridging Modalities, Spanning Time: Structured Memory for Ultra-Long Agentic Video Reasoning](https://arxiv.org/abs/2605.08271) [![GitHub Repo stars](https://camo.githubusercontent.com/2033226d4268726ceec5fc59b3542c639328a41183cf9442819649c94756cb10/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6c696a69617a68656e67303931372f4d414749432d766964656f)](https://github.com/lijiazheng0917/MAGIC-video)
    - Paper: [Reformulating KV Cache Eviction Problem for Long-Context LLM Inference](https://arxiv.org/abs/2605.07234)
- **[2026.05.07]**
    
    - Paper: [UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification](https://arxiv.org/abs/2605.06221) [![GitHub Repo stars](https://camo.githubusercontent.com/d991e6c938d78016472e060da5311009aefa79385615593d06db76c492b34e0d/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f716866616e2f556e6950726566696c6c)](https://github.com/qhfan/UniPrefill)
- **[2026.05.06]**
    
    - Paper: [Visual Text Compression as Measure Transport](https://arxiv.org/abs/2605.06708)
- **[2026.05.05]**
    
    - Paper: [Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving](https://arxiv.org/abs/2605.03375)
- **[2026.04.22]**
    
    - Paper: [LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction](https://arxiv.org/abs/2605.06676)
- **[2026.04.18]**
    
    - Paper: [GenericAgent: A Token-Efficient Self-Evolving LLM Agent via Contextual Information Density Maximization (V1.0)](https://arxiv.org/abs/2604.17091) [![GitHub Repo stars](https://camo.githubusercontent.com/a6fb82585f902a4542f11d186114935a99445eff2f1508ce965be402f934732f/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6c73646566696e652f47656e657269634167656e74)](https://github.com/lsdefine/GenericAgent)
- **[2026.04.17]**
    
    - Paper: [Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](https://arxiv.org/abs/2604.15809) (CVPR 2026) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://cxliu0.github.io/AIF/)
- **[2026.04.13]**
    
    - Paper: [Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning](https://arxiv.org/abs/2604.11462)
- **[2026.04.12]**
    
    - Paper: [IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs](https://arxiv.org/abs/2604.10539) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://yuzhenmao.github.io/IceCache/)
- **[2026.04.10]**
    
    - Paper: [MEMENTO: Teaching LLMs to Manage Their Own Context](https://arxiv.org/abs/2604.09852) [![GitHub Repo stars](https://camo.githubusercontent.com/f01dc93ce6e6062dbd53bc94c1e099ab779ba320fd04c268c0e9b30b51e23ad5/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f6d6963726f736f66742f6d656d656e746f)](https://github.com/microsoft/memento)
- **[2026.04.07]**
    
    - Paper: [In-Place Test-Time Training (In-Place TTT)](https://arxiv.org/abs/2604.06169) (ICLR 2026 Oral) [![GitHub Repo stars](https://camo.githubusercontent.com/4b4ecc991fa43cce4c1b63d9e844c6f6aa7b6df914e78293d00b5341ee15a0dd/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f4279746544616e63652d536565642f496e2d506c6163652d545454)](https://github.com/ByteDance-Seed/In-Place-TTT)
- **[2026.04.06]**
    
    - Paper: [Comparative Characterization of KV Cache Management Strategies for LLM Inference](https://arxiv.org/abs/2604.05012)
    - Paper: [TriAttention: Efficient Long Reasoning with Trigonometric KV Compression](https://arxiv.org/abs/2604.04921) [![GitHub Repo stars](https://camo.githubusercontent.com/ce5762ac43ad28803749d4d950c26c3393f14bfaf94f006f653f21e38149c13c/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f576569616e4d616f2f747269617474656e74696f6e)](https://github.com/WeianMao/triattention)
- **[2026.04.04]**
    
    - Paper: [LightThinker++: From Reasoning Compression to Memory Management](https://arxiv.org/abs/2604.03679)
- **[2026.04.03]**
    
    - Paper: [KiToke: Kernel-based Interval-aware Token Compression for Video Large Language Models](https://arxiv.org/abs/2604.03414)
    - Paper: [TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing](https://arxiv.org/abs/2604.03143)
- **[2026.04.02]**
    
    - Paper: [VideoZeroBench: Probing the Limits of Video MLLMs with Spatio-Temporal Evidence Verification](https://arxiv.org/abs/2604.01569)
    - Paper: [Brief Is Better: Non-Monotonic Chain-of-Thought Budget Effects in Function-Calling Language Agents](https://arxiv.org/abs/2604.02155)
    - Paper: [SimpleStream: A Simple Baseline for Streaming Video Understanding](https://arxiv.org/abs/2604.02317) [![GitHub Repo stars](https://camo.githubusercontent.com/b4c0f18358434d850bff2eb5a65fc6c1eee8c8e779b4ff9e00d7e34c9dbe2892/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f45766f6c76696e674c4d4d732d4c61622f53696d706c6553747265616d)](https://github.com/EvolvingLMMs-Lab/SimpleStream) [![Static Badge](https://camo.githubusercontent.com/32b02fb86de71e74228c1d504c1ccdacae0e9c16c54ece96c81e87eeef3db80e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f486f6d65706167652d626c7565)](https://simple-stream.github.io/)
    - Paper: [GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](https://arxiv.org/abs/2604.02093) (CVPR 2026) [![GitHub Repo stars](https://camo.githubusercontent.com/e62d8cc385ff5f91e144bab8e29441a4a41d194f31a428600c2b5d90792abb58/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f466c6f72656e63653336352f47726f756e64565453)](https://github.com/Florence365/GroundVTS)
    - Paper: [HieraVid: Hierarchical Token Pruning for Fast Video Large Language Models](https://arxiv.org/abs/2604.01881)
- **[2026.04.01]**
    
    - Paper: [Omni-SimpleMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory](https://arxiv.org/abs/2604.01007) [![GitHub Repo stars](https://camo.githubusercontent.com/569d41f4591566fe9a3a640020db6598d80b34f2d1f82519327081c05021f636/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f61696d696e672d6c61622f53696d706c654d656d)](https://github.com/aiming-lab/SimpleMem)
    - Paper: [Scaling Reasoning Tokens via RL and Parallel Thinking: Evidence From Competitive Programming](https://arxiv.org/abs/2604.01302) [![GitHub Repo stars](https://camo.githubusercontent.com/3d9e957b9f8281926c9f81e7a309d901603ac7db7e941f8e80c0300410ccd1ac/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f544855444d2f736c696d65)](https://github.com/THUDM/slime) 


    

# On-Device AI System
\[https://github.com/jeho-lee/Awesome-On-Device-AI-Systems\]

## Inference Engines

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#-inference-engines)

Frameworks and runtimes designed for deploying models on edge devices.

## General ML Workloads

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#general-ml-workloads)

- [LiteRT (formerly TensorFlow Lite)](https://ai.google.dev/edge/litert) - Google's framework for on-device inference.
- [ExecuTorch](https://github.com/pytorch/executorch) - PyTorch’s end-to-end solution for enabling on-device AI.
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference engine for ONNX models.
- [MNN](https://github.com/alibaba/MNN) - Lightweight deep learning framework by Alibaba.
- [NCNN](https://github.com/Tencent/ncnn) - High-performance NN inference framework by Tencent.

## Vendor-Specific SDKs

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#vendor-specific-sdks)

- [Qualcomm QNN](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) - Qualcomm AI Stack for Snapdragon NPUs/DSPs.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) - Framework to integrate ML models into iOS/macOS apps.
- [FluidAudio](https://github.com/FluidInference/FluidAudio) - Local audio AI SDK for Apple platforms with ASR, speaker diarization, VAD, and TTS optimized for Apple Neural Engine.
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - SDK for high-performance deep learning inference on NVIDIA GPUs (including Jetson).
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - Toolkit for optimizing and deploying AI inference on Intel hardware (CPU/GPU/NPU).
- [MediaTek NeuroPilot](https://neuropilot.mediatek.com/) - AI ecosystem and SDK for MediaTek NPUs.

## LLM & GenAI Specialized

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#llm--genai-specialized)

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference in C/C++ with minimal dependencies.
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) - Universal solution for deploying LLMs on any hardware (based on TVM).
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA GPU-optimized LLM inference library, relevant for Jetson-class edge devices.
- [mllm](https://github.com/UbiquitousLearning/mllm) - A fast and lightweight LLM inference engine for mobile and edge devices.
- [MLX LM](https://github.com/ml-explore/mlx-lm) - LLM inference and fine-tuning toolkit built on MLX for Apple silicon.
- [OmniInfer](https://github.com/omnimind-ai/OmniInfer-VLM) - High-performance, on-device VLM inference with hybrid NPU acceleration.
- [RunAnywhere](https://github.com/RunanywhereAI/runanywhere-sdks) - Open-source SDK for running LLMs and multimodal models on-device across iOS, Android, and cross-platform apps.
- [Off Grid](https://github.com/alichherawalla/off-grid-mobile-ai) - Open-source iOS/Android app running LLMs (Llama, Qwen, Gemma, Phi, DeepSeek) entirely on-device via llama.cpp. Includes voice (whisper.cpp), vision, on-device image generation, and tool calling.


## LLM Inference on Mobile SoCs

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#llm-inference-on-mobile-socs)

- [OSDI 2026] Inference in the Shadows: Taming Memory Bandwidth Contention in Mobile LLM Inference with Sereno
- [MobiSys 2026] [Agent-X: Full Pipeline Acceleration of On-device AI Agents](https://arxiv.org/pdf/2605.10380)
- [MLSys 2026] [Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE](https://arxiv.org/abs/2507.02135)
- [SenSys 2026] [LLM as a System Service on Mobile Devices](https://arxiv.org/pdf/2403.11805)
- [EuroSys 2026] [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/pdf/2509.23324v1)
- [SOSP 2025] [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://arxiv.org/abs/2501.14794)
- [ASPLOS 2025] [Neuralink: Fast on-Device LLM Inference with Neuron Co-Activation Linking](https://dl.acm.org/doi/10.1145/3676642.3736114)
- [ASPLOS 2025] [Fast On-device LLM Inference with NPUs](https://arxiv.org/abs/2407.05858)
- [arXiv 2024] [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](https://arxiv.org/abs/2406.06282)

## On-device AI Accelerators: Performance Characterization & Optimization

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#on-device-ai-accelerators-performance-characterization--optimization)

- [EuroSys 2026] [viNPU: Optimizing Vision Transformer Inference on Mobile NPUs](https://dl.acm.org/doi/10.1145/3767295.3803619)
- [ASPLOS 2026] [FlashMem: Supporting Modern DNN Workloads on Mobile with GPU Memory Hierarchy Optimizations](https://arxiv.org/abs/2602.15379)
- [ICS 2025] [TMModel: Modeling Texture Memory and Mobile GPU Performance to Accelerate DNN Computations](https://doi.org/10.1145/3721145.3725774)

## Compiler-based ML Optimization

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#compiler-based-ml-optimization)

- [ASPLOS 2024] [SmartMem: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile](https://dl.acm.org/doi/pdf/10.1145/3620666.3651384)
- [ASPLOS 2024] [SoD2: Statically Optimizing Dynamic Deep Neural Network Execution](https://dl.acm.org/doi/pdf/10.1145/3617232.3624869)
- [MICRO 2023] [Improving Data Reuse in NPU On-chip Memory with Interleaved Gradient Order for DNN Training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10411391)
- [MICRO 2022] [GCD2: A Globally Optimizing Compiler for Mapping DNNs to Mobile DSPs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9923837)
- [PLDI 2021] [DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)

## Attention Acceleration

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#attention-acceleration)

- [MLSys 2026] [IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference](https://arxiv.org/abs/2511.21513)
- [MobiSys 2026] [ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference](https://arxiv.org/abs/2508.16703)
- [MLSys 2025] [MAS-Attention: Memory-Aware Stream Processing for Attention Acceleration on Resource-Constrained Edge Devices](https://arxiv.org/pdf/2411.17720)
- [MLSys 2025] [TurboAttention: Efficient attention approximation for High Throughputs LLMs](https://arxiv.org/pdf/2412.08585)
- [ASPLOS 2023] [FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://dl.acm.org/doi/10.1145/3575693.3575747)
- [NeurIPS 2022] [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)

## Quantization/Sparsity

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#quantizationsparsity)

- [ASPLOS 2026] [oFFN: Outlier and Neuron-aware Structured FFN for Fast yet Accurate LLM Inference](https://dl.acm.org/doi/pdf/10.1145/3779212.3790194)
- [MLSys 2024] [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/pdf/2306.00978)
- [ISCA 2023] [OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization](https://arxiv.org/abs/2304.07493)

## Application-centric On-device AI Systems

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#application-centric-on-device-ai-systems)

- [MobiSys 2025] [ARIA: Optimizing Vision Foundation Model Inference on Heterogeneous Mobile Processors for Augmented Reality](https://dl.acm.org/doi/10.1145/3711875.3729161)
- [MobiCom 2024] [Panopticus: Omnidirectional 3D Object Detection on Resource-constrained Edge Devices](https://arxiv.org/pdf/2410.01270)
- [MobiCom 2024] [Perceptual-Centric Image Super-Resolution using Heterogeneous Processors on Mobile Devices](https://dl.acm.org/doi/10.1145/3636534.3690698)
- [IPSN 2023] [PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators](https://dl.acm.org/doi/pdf/10.1145/3583120.3587045)
- [MobiSys 2023] [OmniLive: Super-Resolution Enhanced 360° Video Live Streaming for Mobile Devices](https://dl.acm.org/doi/pdf/10.1145/3581791.3596851)
- [MobiCom 2022] [NeuLens: Spatial-based Dynamic Acceleration of Convolutional Neural Networks on Edge](https://dl.acm.org/doi/pdf/10.1145/3495243.3560528)
- [MobiCom 2021] [Flexible high-resolution object detection on edge devices with tunable latency](https://dl.acm.org/doi/abs/10.1145/3447993.3483274)

## Multi-DNN / Heterogeneous Runtime Scheduling

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#multi-dnn--heterogeneous-runtime-scheduling)

- [PPoPP 2024] [Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous SoCs](https://dl.acm.org/doi/pdf/10.1145/3627535.3638502)
- [RTSS 2024] [FLEX: Adaptive Task Batch Scheduling with Elastic Fusion in Multi-Modal Multi-View Machine Perception](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10844787)
- [MobiSys 2024] [Pantheon: Preemptible Multi-DNN Inference on Mobile Edge GPUs](https://dl.acm.org/doi/pdf/10.1145/3643832.3661878)
- [Sensys 2023] [Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU](https://dl.acm.org/doi/10.1145/3625687.3625789)
- [ATC 2023] [Decentralized Application-Level Adaptive Scheduling for Multi-Instance DNNs on Open Mobile Devices](https://www.usenix.org/system/files/atc23-sung.pdf)
- [MobiSys 2022] [Band: Coordinated Multi-DNN Inference on Heterogeneous Mobile Processors](https://dl.acm.org/doi/pdf/10.1145/3498361.3538948)
- [MobiSys 2022] [CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices](https://dl.acm.org/doi/pdf/10.1145/3498361.3538932)

## On-device Training, Model Adaptation

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#on-device-training-model-adaptation)

- [ASPLOS 2025] [Nazar: Monitoring and Adapting ML Models on Mobile Devices](https://dl.acm.org/doi/pdf/10.1145/3669940.3707246)
- [SenSys 2024] [AdaShadow: Responsive Test-time Model Adaptation in Non-stationary Mobile Environments](https://arxiv.org/pdf/2410.08256)
- [SenSys 2023] [EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge](https://dl.acm.org/doi/10.1145/3625687.3625793)
- [MobiCom 2023] [Cost-effective On-device Continual Learning over Memory Hierarchy with Miro](https://dl.acm.org/doi/pdf/10.1145/3570361.3613297)
- [MobiCom 2023] [AdaptiveNet: Post-deployment Neural Architecture Adaptation for Diverse Edge Environments](https://dl.acm.org/doi/pdf/10.1145/3570361.3592529)
- [MobiSys 2023] [ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection](https://dl.acm.org/doi/pdf/10.1145/3581791.3596852)
- [SenSys 2023] [On-NAS: On-Device Neural Architecture Search on Memory-Constrained Intelligent Embedded Systems](https://dl.acm.org/doi/10.1145/3625687.3625814)
- [MobiCom 2022] [Mandheling: mixed-precision on-device DNN training with DSP offloading](https://dl.acm.org/doi/abs/10.1145/3495243.3560545)
- [MobiSys 2022] [Memory-efficient DNN training on mobile devices](https://dl.acm.org/doi/abs/10.1145/3498361.3539765)

## Profilers

[](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems#profilers)

- [MobiCom 2024] [MELTing point: Mobile Evaluation of Language Transformers](https://arxiv.org/abs/2403.12844) [[code]](https://github.com/brave-experiments/MELT-public)
- [SenSys 2023] [nnPerf: Demystifying DNN Runtime Inference Latency on Mobile Platforms](https://dl.acm.org/doi/10.1145/3625687.3625797)
- [MobiSys 2021] [nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices](https://dl.acm.org/doi/10.1145/3458864.3467882)