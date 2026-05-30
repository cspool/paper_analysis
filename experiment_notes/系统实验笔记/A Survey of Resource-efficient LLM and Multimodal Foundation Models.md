## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- 属于Serving调度的实现是什么？实验比较什么？
  本文为综述论文，无原创Serving系统实验。§5.3-5.4系统梳理云侧与端侧LLM serving系统的优化技术：
  (i) **推理加速**（§5.3.1）——Kernel优化（FlashAttention/FlashAttention-2加速prefill、Flash-Decoding/FastGen加速decode）、Parallelism策略（TP+PP+DP+Expert Parallelism混合）、Request Batching与调度（Orca的iteration-level batching消除padding、SARATHI的chunked-prefill与decode混合调度、Splitwise的prefill-decode分离部署）；
  (ii) **内存节省**（§5.3.2）——vLLM的PagedAttention（block级按需分配KV cache消除碎片，up to 29×吞吐提升）、S-LoRA的Unified Paging支持多LoRA adapter、SGLang的RadixAttention支持跨请求KV cache复用、FlexGen的激活/参数offload到DRAM/NVMe；
  (iii) **新兴部署平台**（§5.3.3）——SpotServe on spot instances（动态调整并行策略应对抢占）、HexGen on heterogeneous GPUs（进化算法搜索placement和parallelism）；
  (iv) **端侧Serving**（§5.4）——Edge-cloud协作（EdgeFM）、端侧MoE（EdgeMoe的expert-wise bit-width adaptation、PC-MoE的参数委员会机制）、内存优化（LLMCad的speculative decoding + token tree、PowerInfer的热/冷神经元分离GPU/CPU计算）、I/O优化（STI的动态权重bit-width加载、LLM in a Flash的细粒度闪存管理）、Kernel优化（mllm-NPU利用移动NPU加速prefill）、LLMaaS范式（LMS的细粒度KV cache管理、ELMS的弹性SLO支持）。

- 硬件平台是什么，配置是什么。
  综述未进行统一实验。被引述系统的硬件平台包括：数据中心GPU（A100/H100/TPU）、消费级GPU、手机端（iPhone 12、安卓设备w/ NPU）、Raspberry Pi 5。

- 开源Serving框架是什么。修改了什么。
  综述表5总结开源框架：
  - **vLLM**（UC Berkeley）：PagedAttention block级KV cache管理，消除内存碎片。vAttention进一步直接依赖OS/CUDA做物理内存重分配，端到端吞吐再提升1.29×。
  - **DeepSpeed-Inference/MII**（Microsoft）：支持ZeRO优化、模型压缩、FastGen动态split-fuse schedule。
  - **TensorRT-LLM**（NVIDIA）：集成AWQ/GPTQ/SmoothQuant量化、speculative decoding、TP+PP并行、PagedAttention。
  - **HuggingFace TGI**：支持TP、bitsandbytes/GPTQ量化、PagedAttention。
  - **SGLang**：RadixAttention进行跨请求KV cache复用，prompt programming primitives。
  - **LightLLM**：token级别KV cache内存管理。
  - **MLC-LLM**：编译器加速的通用部署方案，支持native API。
  - **llama.cpp**：CPU端LLM推理，支持2-8bit整数量化（K-quant），3-4×加速。
  - **mnn-llm**（阿里）：将推理分为prefill/decoding两阶段分别优化。
  - **mllm**：面向多模态大模型的端侧推理引擎。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  综述全部材料开源：https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey（LaTeX源码+参考文献）。以下以vLLM为例说明Serving框架输入到硬件执行的全过程：

  以vLLM的PagedAttention为例（§5.3.2）：
  ```
  用户请求 → HTTP API Server
    → Scheduler（iteration-level调度，管理request queue）
      → Block Manager（为每个request的KV cache分配逻辑block，
          映射到物理GPU内存block，类似OS虚拟内存）
      → Model Runner（batching requests with PagedAttention kernel）
        → PagedAttention CUDA Kernel:
          for each query token q_i:
            for each physical_block b in block_table[req_id]:
              // 从GPU HBM读取该block的K/V cache
              K_block = KV_cache[b]  // shape [block_size, num_heads, head_dim]
              // 计算block内attention
              scores = q_i @ K_block^T / sqrt(head_dim)
              p = softmax(scores)
              o_i += p @ V_block
          → output token → 追加到KV cache（可能需要新block分配）
    → 返回generated token到用户
  ```
  PagedAttention消除KV cache碎片，相比vanilla KV cache（预分配max_seq_len连续空间），内存利用率从约20-30%提升至接近100%。
