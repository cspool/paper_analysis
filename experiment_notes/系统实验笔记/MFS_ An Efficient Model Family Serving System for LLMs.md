## MFS: An Efficient Model Family Serving System for LLMs

- 属于Serving调度的实现是什么？实验比较什么？
  MFS实现了一个面向LLM model family的multi-tier serving system。核心Serving调度实现包含：(1) Multi-tier model structure：通过Knowledge Precipitation将model family中最大模型微调为嵌套式multi-tier模型，低tier对应小模型、高tier对应大模型，前若干层/tier之间共享参数和hidden states；(2) Group batching：tier-level scheduler将不同tier但共享公共前缀层的请求组成group batch一起执行，替代Orca的selective batching（后者假设batch内请求共享同一模型结构）；(3) Attention fusion：decode阶段将不同请求的QKV矩阵拼接后做统一attention计算，牺牲少量冗余attention换取GPU并行度，提升小batch decode吞吐；(4) Shareable KV cache：KV cache manager记录multi-tier可共享状态，tier切换时复用已有KV cache而不重复计算；(5) Tier-aware speculative sampling：低tier作为draft model快速提出token，高tier作为target model验证，直接继承低tier KV cache。分布式部署支持intra-layer parallelism和inter-layer parallelism，同一tier的层放在同一partition以减少同步开销。
  实验比较：(a) batching场景：MFS vs Orca的request execution time、JCT（MFS JCT最多提升31.2%）、per-token response latency（最多降低56.1%）；(b) KV cache sharing：GPU memory footprint对比（MFS最多降低47.8%）；(c) speculative sampling：GPU utilization对比（Orca约23.9% vs MFS约59.8%）；(d) end-to-end：合成generative request trace下的median end-to-end latency per generated token。

- 硬件平台是什么，配置是什么。
  训练：2台服务器，每台8×NVIDIA H800 SXM5 GPU (80GB)、2×56核Intel Xeon Gold CPU、2TB内存，8×400Gbps NDR InfiniBand互联。推理评估：(1) 2×NVIDIA A100 GPU服务器，2×48核Intel Xeon Gold CPU、512GB内存；(2) 8×NVIDIA 3090 GPU服务器，80核Intel Xeon Gold CPU、256GB内存。

- 开源Serving框架是什么。修改了什么。
  基于Orca scheduler。修改包括：(1) 替换Orca selective batching为group batching：将不同tier但共享公共前缀层的请求组成group batch，共同执行前缀层；(2) 新增tier-level scheduler：为不同tier维护独立请求队列，按优先级/公平队列/外部策略选择下一批执行，支持plug-and-play调度策略；(3) 新增attention fusion模块：decode阶段将group batch内各请求QKV拼接为一个larger attention operation，以冗余compute换GPU parallelism；(4) 新增multi-tier KV cache manager：管理跨tier可共享的KV cache，支持tier切换时KV cache复用；(5) 新增tier-aware speculative sampling模块：低tier作draft、高tier作target，共享lower-tier KV cache。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到MFS官方开源仓库（论文EuroSys 2026 DOI确认，但正文/参考文献未提供MFS代码入口）。公开页面：https://www.cse.ust.hk/~kaichen/papers/mfs-eurosys26.pdf。
  以Llama2-13B转3-tier MFS模型（tier-1=~7B等效/tier-2=~10B等效/tier-3=13B等效）在2×A100上serving为例：
  1. 离线：对Llama2-13B-chat做Knowledge Precipitation fine-tuning（数据guanaco-llama2 ~9.85k对话，AdamW lr=2e-5，1 epoch/2500 iterations，约24h on 16×H800），产出嵌套3-tier模型，tier边界通过测量各层推理latency确定（如tier-1取前24层对齐7B latency）。
  2. 部署：加载multi-tier模型到2×A100→tier-level scheduler初始化三个tier队列→KV cache manager初始化可共享缓存空间→front-end监听请求。
  3. 三类请求到达：R1（低latency需求，选tier-1）、R2（中等质量需求，选tier-2）、R3（高质量需求，选tier-3）→写入request pool→scheduler检测到三者都需要tier-1公共前缀层→组成group batch。
  4. Group batching执行tier-1公共前缀层：batch内合并R1/R2/R3→decode阶段attention fusion拼接三个请求的QKV→GPU并行度提升→产生的KV cache标记为multi-tier shareable。
  5. R1在tier-1完成后采样输出并返回；R2/R3继续tier-2→复用tier-1的hidden states和KV cache→R2完成后输出；R3继续tier-3→复用前两个tier的KV cache。
  6. 若启用speculative sampling：tier-1作为draft生成若干候选token→tier-3作为target验证→直接继承tier-1的KV cache（vs Orca独立部署需为target重新计算全部KV cache）。
  7. 效果：相比Orca独立部署Llama2-7B+13B两份模型，MFS单一嵌套模型：GPU显存降低（KV cache sharing最多-47.8%），group batching + attention fusion提升GPU利用率（speculative场景23.9%→59.8%），JCT提升31.2%，per-token latency降低56.1%。

