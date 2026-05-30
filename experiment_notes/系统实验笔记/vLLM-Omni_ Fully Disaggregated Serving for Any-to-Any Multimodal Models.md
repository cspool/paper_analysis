## vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

- 属于Serving调度的实现是什么？实验比较什么？
  实现是vLLM-Omni——一个面向any-to-any多模态模型的fully disaggregated serving system。核心实现包括：(1) Stage Abstraction——用户将复杂的多模态模型架构分解为stage graph，node代表model stage（AR LLM、DiT等），edge代表stage间数据传输函数；(2) Disaggregated Stage Execution Backend——每个stage由独立的execution engine serving（vLLM engine用于AR stages，专用diffusion engine用于DiT stages），支持per-stage request batching、灵活GPU分配和intra-stage serving优化（continuous batching、chunked prefill、execution graph compilation）；(3) Unified Connector——负责stage间中间数据传输，支持inline control queues（小payload）、system shared memory（大payload，单节点）和Mooncake-based RDMA（跨节点）；(4) Streaming Stage Output——支持下游stage在上游未完全完成时就开始incremental处理。

  实验比较baseline：Qwen-Omni系列（Qwen2.5-Omni、Qwen3-Omni）使用HuggingFace Transformers默认实现；BAGEL使用原始实现；MiMo-Audio使用原始实现；Diffusion模型（Qwen-Image、Qwen-Image-Edit、Wan2.2系列）使用Diffusers库。评估指标：Real-Time Factor (RTF)、Job Completion Time (JCT)、Tokens Per Second (TPS) for Thinker和Talker components。关键结果：Qwen3-Omni JCT降低91.4%，RTF降低90.7%，Thinker TPS提升12.97×，Talker TPS提升7.98×。BAGEL T2I speedup 2.40×，I2I speedup 3.72×。MiMo-Audio RTF从1.39降至0.12（11.58× speedup）。Diffusion model整体1.26× speedup。

- 硬件平台是什么，配置是什么。
  服务器配备2块accelerator设备（每块80GB memory），24 CPU cores，192 GB system memory。使用virtual setup环境，vLLM version 0.12.0。BAGEL和MiMo-Audio在单accelerator（80GB）上评估。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.12.0。修改/新增内容：
  - Stage Abstraction Frontend：提供Python API让用户定义stage graph——每stage实现forward（step-centric batched forward）和preprocess（修改stage输入）函数，edges定义stage-transfer函数。
  - Orchestrator：管理stage执行和调度incoming requests，route requests through stage graph。
  - AR Stage Engine：基于vLLM engine，每stage独立运行自己的scheduler（batching）、KV-cache manager（per-stage KV管理）和model runner。新增per-request intermediate data dictionary（用户可在transform和preprocess函数中访问和更新）。preprocess在每个decode iteration调用，output processor负责执行transform function并传输数据到下游stage的设备。
  - Diffusion Engine：全新实现，集成flash attention、SAGE attention、TurboAttention等attention优化，TeaCache/cache-dit等caching策略，RingAttention context parallelism和Ulysses sequence parallelism。支持text-to-image（Z-Image、Qwen-Image、Flux）、image editing（Qwen-Image-Edit、LongCat-Image-Edit）和video generation（Wan2.2、HunyuanVideo）。
  - Unified Connector：单节点使用inline control queues（小payload）+ system shared memory（大payload）；多节点通过Ray orchestration + Mooncake-based connector（TCP/RDMA transport）。Connector同时处理intra-stage transfer（prefill-decode KV cache、encoder-prefill MM cache），兼容EPD disaggregation。
  - Streaming Stage Output：output processor异步stream partial outputs到下一stage，减少TTFT和enabling streaming responses。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接：https://github.com/vllm-project/vllm-omni（注意：URL带连字符）。vLLM-Omni v0.20.0（2026/05最新版本）。文档：https://docs.vllm.ai/projects/vllm-omni/en/stable/

  作用：vLLM-Omni解决的核心问题是"existing LLM serving frameworks（vLLM、SGLang）的step-centric abstraction无法表达multi-stage any-to-any pipeline"。传统框架仅支持单AR decoding或单DiT denoising stage，开发者需手动实现inter-stage transfer outside serving framework，导致：(1) 无法利用continuous batching、chunked prefill等serving优化；(2) 计算资源无法跨stage灵活分配，stage co-located as monolith导致资源浪费。

  全过程（以Qwen3-Omni音频生成请求为例，2×80GB accelerators）：
  ```
  用户请求到达vLLM-Omni server（含text + audio prompt，请求生成text + audio outputs）
    → Orchestrator接收请求，调度到stage graph执行:

    Stage 1 - Thinker (AR LLM, ~30B, TP-2 across both accelerators):
      vLLM engine接收stage input:
        multimodal encoder处理输入（audio→Whisper encoder, image→ViT, video→Vision encoder）
        → encoder embeddings + text token embeddings concatenated
      → vLLM model runner执行continuous batching:
        Prefill阶段: chunked prefill处理prompt tokens
          - PagedAttention管理KV cache
          - Thinker forward function + customized preprocess (concatenate MM embeddings)
        Decode阶段: 自回归生成text tokens + 产生hidden states
          - 每个decode iteration: preprocess → forward → 输出text token + hidden states
          - 使用execution graph compilation加速
      → Output Processor:
        对每个完成text生成的request:
          Thinker2Talker transform function:
            提取Thinker hidden states → 转换为Talker input embeddings
            → 通过unified connector传输到Talker stage设备(device-1)

    Stage 2 - Talker (AR LLM, smaller but compute-intensive, 置于device-1):
      vLLM engine接收streaming input:
        每iteration调用preprocess:
          连接Thinker hidden states + Talker input embeddings + original multimodal embeddings
      → AR decode生成audio codec tokens (平均545.4 tokens)
      → Output Processor:
        Talker2Vocoder transform:
          收集generated codec tokens → 通过unified connector传输到Vocoder(device-0)
        Streaming: 当Talker产出initial tokens时Vocoder即可开始处理

    Stage 3 - Vocoder (DiT/CNN, 置于device-0):
      Diffusion engine接收codec tokens:
        从partial tokens开始incremental denoising (streaming)
        → DiT denoising with flash attention + TeaCache caching
        → 生成audio waveforms
      → Final output: text response + audio waveforms返回client
  ```

  对比Baseline（HuggingFace Transformers monolithic执行）：
  ```
  请求到达 → 单进程加载所有model components:
    1. encoder处理multimodal input
    2. Thinker generate (自定义generate loop, 无可用的continuous batching/paged attention/chunked prefill)
    3. 等待Thinker完全完成 → 提取hidden states
    4. Talker generate (自定义generate loop)
    5. 等待Talker完全完成 → 提取codec tokens
    6. Vocoder generate waveforms
  → 返回结果
  关键性能损失:
    - Continuous batching不可用 → 多请求无法batch，GPU利用率低
    - Chunked prefill不可用 → 长prompt prefill latency高
    - Execution graph compilation不可用 → kernel launch overhead大
    - Stage co-located → Thinker(30B)和Talker竞争memory，无法独立扩缩容
    - 无streaming → Vocoder必须等Talker完全完成，增加end-to-end latency
  ```

  Qwen3-Omni关键数据流分析:
  ```
  Thinker TPS: baseline ≈ X tokens/s, vLLM-Omni = 12.97× X (execution graph compilation + chunked prefill)
  Talker TPS: baseline ≈ Y tokens/s, vLLM-Omni = 7.98× Y (continuous batching + KV cache management)
  Talker占总延迟大部分（生成545.4 audio tokens vs Thinker 150.9 text tokens）

  Thinker→Talker transfer: 5.49ms (shared memory) / 8.28ms (Mooncake RDMA) — negligible vs total latency (tens of seconds)
  Talker→Vocoder transfer: 0.53ms (shared memory) / 3.34ms (Mooncake) — negligible
  ```
