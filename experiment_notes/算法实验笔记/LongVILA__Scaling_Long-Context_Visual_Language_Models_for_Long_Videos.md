## LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) 五阶段训练pipeline：Stage1 多模态对齐（冻结LLM和视觉编码器，仅训练多模态投影器）→ Stage2 大规模预训练（冻结视觉编码器，训练LLM+投影器，使用VILA-1.5-40B重标注的COYO-25M）→ Stage3 短视频监督微调（全参数微调，混合图像+短视频数据如YouCook2/ShareGPTVideo）→ Stage4 LLM上下文扩展（文本only持续预训练，渐进式从8K→65K→262K上下文，使用SlimPajama 17B tokens，RoPE基频增大+LoRA微调，约336 H100 GPU hours）→ Stage5 长视频监督微调（全参数训练，使用LongVILA_SFT数据集15,292个长视频，每个视频paired caption+QA问答对，配合MM-SP系统）。
  (2) 长视频指令数据生成pipeline：长视频→切分为10秒短片段→VILA-1.5模型逐片段生成caption→LLM基于所有片段caption生成QA对（涵盖总结、空间、属性、动作、对象、OCR、时序等7类问题）。

  实验比较：
  (a) 9个视频benchmark（ActivityNet-QA/EgoSchema/EventBench/LongVideoBench/PerceptionTest/MVBench/NExT-QA/VNBench/VideoMME），对比GPT-4V/GPT-4o/Gemini-1.5-Pro和Video-LLaVA/Flash-VStream/ShareGPT4Video/VideoLLaMA2/VideoLLaMA2.1/Kangaroo/PLLaVA/LLaVA-OV。LongVILA-7B在VideoMME取得65.1% (w/ subtitle)。
  (b) VideoMME按时长细分(Short/Medium/Long)，256 frames，LongVILA-7B取得72.9/64.9/57.4 (w/ sub)，证明frame scaling在长视频上的收益。
  (c) 训练阶段顺序ablation (Table 1)：1-2-3-4-5 vs 1-2-4-(3&5) vs 4-1-2-3-5 vs 4-1-2-(3&5)，最优为原始顺序57.5 avg VideoMME w/o sub。
  (d) Needle-in-a-Haystack：LongVILA在2048 frames训练后，6000 frame (1M+ tokens)测试达99.8%准确率，远超32-frame baseline和LongVA（3000 frame 87.6%）。
  (e) LongVILA-Caption benchmark（100长视频）：8→128→256 frames，Correctness/Detailed/Contextual分数从1.87/1.85/2.27提升至3.23/3.11/3.43。
  (f) 10个图像VLM benchmark (Table 9)：证明长视频训练不损害图像理解能力，LongVILA-7B S3模型在多个图像benchmark上领先。

- 硬件平台是什么，配置是什么。
  训练系统：H100节点（每节点8×H100 80GB，NVLink 900 GB/s intra-node，InfiniBand 50 GB/s inter-node single path）。最大序列长度实验使用32 A100节点（每节点8×A100 80GB）。推理系统：单节点8×H100 80GB。模型复杂度Profiling：单张A100 GPU，FP16，Flash-Attention2。Stage4上下文扩展：约336 GPU hours on 80GB A100。

- 模型是什么。数据集和bench分别是什么。
  模型：基于VILA-1.5（Encoder-Decoder VLM架构），视觉编码器(ViT) → 多模态投影器(linear/MLP) → LLM解码器。LLM backbone为Qwen2-1.5B和Qwen2-7B，使用GQA（8 KV heads, 32 Q heads）。每帧产生约256个tokens，1400帧视频约274K tokens。训练后支持8→2048帧视频输入。
  数据集：COYO-25M（VILA-1.5-40B重标注）、YouCook2、ShareGPTVideo、Shot2Story20k→LongVILA_SFT（15,292长视频，12类别：Travel/Sports/Education/Pets/People/News/Music/Science/Comedy/Entertainment/Film/Gaming，每视频1 caption + 1 QA对）、SlimPajama（17B text tokens，Stage4使用）、LongVILA-Caption（100长视频人工校验）。
  Benchmarks：ActivityNet-QA, EgoSchema, EventBench, LongVideoBench, PerceptionTest, MVBench, NExT-QA, VNBench, VideoMME；图像benchmarks: VQAv2, GQA, VizWiz, SQA-I, VQA-T, MMB, MMB-CN, SEED, LLaVAW, MM-Vet。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源：github.com/NVlabs/VILA/tree/main/longvila，基于HuggingFace Transformers通过monkey-patching集成。
  
  训练pipeline伪代码：
  ```
  # Stage 1: Multi-modal Alignment
  freeze(vision_encoder); freeze(llm)
  for (img, text) in alignment_data:
      v = projector(vision_encoder(img))
      loss = CrossEntropy(llm([v; text_tokens]), labels)
      update(projector)

  # Stage 2: Large-scale Pre-training  
  freeze(vision_encoder)
  for (img, text) in coyo_relabeled:
      v = projector(vision_encoder(img))
      loss = CrossEntropy(llm([v; text]), labels)
      update(llm, projector)

  # Stage 3: Short Supervised Fine-Tuning
  for (img_or_frames, text) in mixed_short_data:
      v = projector(vision_encoder(img_or_frames))
      loss = CrossEntropy(llm([v; text]), labels)
      update(all_params)

  # Stage 4: Context Extension (text-only, LoRA)
  rope_base *= scale  # 增大RoPE基频
  for text in slimpajama_8k_65k_262k:  # progressive schedule
      loss = llm_lora(llm(text))
      update(lora_params)
  
  # Stage 5: Long Video SFT (full params, MM-SP)
  for (frames, text) in long_video_sft:
      # Two-stage sharding in MM-SP
      local_frames = distribute_evenly(frames, sp_ranks)  # Stage1: per-image balance
      vis_feats = vision_encoder(local_frames)  # balanced encoding
      all_tokens = gather_concat(vis_feats, text_tokens)  # global aggregation
      local_tokens = shard_by_token_count(all_tokens, sp_ranks)  # Stage2: per-token balance
      loss = llm_2d_attention(local_tokens)  # 2D-Attention: A2A intra + P2P inter
      update(all_params)
  ```
  张量计算示例（256 frames, 32 GPUs）：
  - 输入：B=1, 256 frames × ~256 tokens/frame = 65536 vision tokens + T text tokens
  - Stage1 sharding：32 GPUs各处理8 frames，视觉编码负载均衡
  - Stage2 sharding：全局tokens按sequence dim均匀分配，每GPU持有 (65536+T)/32 tokens
  - 2D-Attention (4×8 mesh)：intra-node 4 GPUs A2A交换head-dim分片 → inter-node 8 groups P2P传输KV → SDPA本地计算
