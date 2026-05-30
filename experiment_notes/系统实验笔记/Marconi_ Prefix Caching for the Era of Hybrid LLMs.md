## Marconi: Prefix Caching for the Era of Hybrid LLMs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是Marconi——首个面向Hybrid LLMs（Attention+SSM混合架构）的prefix caching系统。核心实现包括两部分：(1) Judicious Admission策略——通过radix tree记录历史请求，识别两种前缀复用模式（Purely Input: 系统提示词、few-shot示例等被多请求共享的前缀；Input and Output: 对话历史等从最后一个decoded token继续的前缀），仅缓存高复用概率的SSM状态，每个序列最多2个checkpoint；(2) FLOP-Aware Eviction策略——Utility Score = recency + α × flop_efficiency，其中flop_efficiency = 复用该state节省的总FLOPs / 该state占用的内存字节数，优先保留计算节省密度高的缓存条目。Marconi将SSM states和KV caches统一管理在单个radix tree中。

  实验比较baseline：fine-grained checkpointing（naive方案，每x token存一个SSM state checkpoint）、SGLang+（扩展SGLang支持Hybrid LLMs，使用LRU eviction）、vLLM+（扩展vLLM支持Hybrid LLMs）。评估指标：token hit rate (%)、Time To First Token (ms)、P95 TTFT reduction。消融实验：Marconi vs fine-grained checkpointing（评估judicious admission贡献）、FLOP-aware eviction vs LRU eviction（评估eviction policy贡献）。结果：Marconi vs fine-grained checkpointing token hit rate提升4.5×–34.4×，P95 TTFT降低36.1%–71.1%。FLOP-aware eviction单独（vs LRU）提升19%–219% token hit rate。

- 硬件平台是什么，配置是什么。
  Cloudlab节点，Ubuntu 22.04，32-core CPU，约20 GB磁盘空间，约7 GB traces数据。GPU硬件：论文未明确说明具体GPU型号（实验使用离线trace-based模拟而非实际GPU部署）。模型：NVIDIA Mamba2-Hybrid-7B（4 Attention + 24 SSM + 28 MLP layers），tokenizer使用meta-llama/Llama-2-7b-hf。

- 开源Serving框架是什么。修改了什么。
  基于radix-tree prefix cache架构（源自vLLM和SGLang的prefix caching设计），Marconi修改/扩展内容包括：
  - radix_cache_hybrid.py：核心caching逻辑，实现judicious admission和FLOP-aware eviction策略。在radix tree中统一管理Attention层的KV cache和SSM层的recurrent states。Tree节点分为intermediates（purely-input前缀，被多请求共享）和leaves（input-and-output前缀，对话末尾）。
  - radix_cache_vllm.py：vLLM适配版本，集成到vLLM serving framework。
  - policy_exploration.py：可插拔eviction policy框架，支持V1 (SGLang+ LRU)、V2 (Marconi)、V3 (offline-optimal oracle)。自定义policy可通过实现新的evict_policy_version加入。
  - config_tuner.py：自动调优α参数（FLOP efficiency的权重系数）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接：https://github.com/ruipeterpan/marconi。MLSys 2025 Outstanding Paper Honorable Mention。使用conda环境（environment.yml），Python 3.11.9。

  作用：Marconi解决Hybrid LLMs（Mamba2-Hybrid, Jamba等）中SSM层的prefix caching难题——SSM状态通过in-place recurrent更新，无法像Attention KV cache那样通过切片回滚到任意前缀位置。Naive checkpointing（每隔x token存一次SSM state）导致：(1) 缓存条目稀疏命中；(2) SSM state尺寸大（固定大小但比单token KV大几个数量级）；(3) 频繁缓存thrashing。Marconi通过judicious admission（仅缓存高复用概率状态，每序列至多2个checkpoint）和FLOP-aware eviction（权衡recency和compute savings）解决上述问题。

  全过程（以NVIDIA Mamba2-Hybrid-7B serving为例，LMSys conversational workload）：
  ```
  请求到达 → Marconi prefix cache处理:

    Step 1 - Speculative Admission:
      新请求token序列插入radix tree（tentative insertion）
      → 如果请求创建新branching point（intermediate node）:
         标记为purely-input前缀（如system prompt），高复用概率 → admit
      → 如果请求延伸到leaf node:
         标记为input-and-output前缀（如对话结束位置） → 仅缓存最后token的SSM state
      → 每个序列至多产生2个SSM state checkpoint

    Step 2 - Cache Lookup & Hit:
      radix tree从根节点匹配请求的token序列
      → 匹配到最深节点 → 获取已缓存的KV cache（Attention层）+ SSM states（SSM层）
      → 未匹配的tail tokens需重新prefill计算
      → Attention层: KV cache直接切片复用（传统prefix caching）
      → SSM层: 从checkpoint恢复recurrent state，从此state继续forward

    Step 3 - Eviction (当缓存容量满时):
      for each cached entry in radix tree:
        FLOP_efficiency = 复用该state节省的总FLOPs / state内存字节数
        Utility = recency_score + α × flop_efficiency_score
      淘汰Utility最低的entry
      → α由config_tuner根据workload模式自动调优

    Step 4 - 执行输出:
      Prefill剩余tail tokens（利用缓存的KV+SSM states加速） → 生成first token (TTFT)
      → Decoding阶段自回归生成后续tokens → 返回完整响应
  ```

  Evaluation配置：sweep各种cache size和request arrival patterns。Per-trace runtime: LMSys ~30s, ShareGPT ~5s, SWEBench ~5-10min（32-core CPU离线模拟）。完整实验sweep约12小时。图7（token_hit_rate.py）复现Marconi vs fine-grained checkpointing结果，图8（sglang_comparison.py）复现FLOP-aware eviction vs SGLang+ LRU比较。
