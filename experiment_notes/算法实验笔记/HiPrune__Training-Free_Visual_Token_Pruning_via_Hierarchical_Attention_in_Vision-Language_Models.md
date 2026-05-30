## HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出了 HiPrune，一种 training-free、model-agnostic 的视觉 token 剪枝方法。核心发现：视觉编码器（ViT）不同层有分层注意力模式——中间层关注图像中的 main object（object-centric），深层关注全局上下文信息（global representation）。基于此，HiPrune 从中间层选 Anchor Tokens（高注意力）+ Buffer Tokens（空间相邻），从输出层选 Register Tokens（高注意力），三者组合保留图像局部细节和全局信息。HiPrune++ 在此基础上额外加入与 text embedding 余弦相似度高的 token，增强指令跟随能力。
  实验比较 HiPrune 与 9 种 SOTA token 剪枝方法（ToMe、FastV、SparseVLM、HiRED、TRIM、VisionZip、PyramidDrop、DivPrune、VisPruner）在 4 个 VLM 上的准确率和效率。
  关键指标：在 LLaVA-1.5-7B 上，保留 1/3 token 时保持 99.3% 准确率，FLOPs 减少 58.7%；保留 1/9 token 时 HiPrune++ 仍保持 96.1%。在 LLaVA-NeXT-7B（高分辨率）上，保留 2/9 token 保持 99.7%。Qwen2.5-VL 上也达到 SOTA。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-PCIE 40GB，部分 overhead 分析在 RTX 5090 上测量 wall-clock latency。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5-7B（CLIP-ViT-L/14, 576 tokens/image）、LLaVA-NeXT-7B（CLIP-ViT-L/14, 2880 tokens/image）、Qwen2.5-VL-3B/7B/32B-Instruct（动态分辨率 ViT）、Video-LLaVA-7B（视频）、LLaVA-1.5-13B、LLaVA-NeXT-13B。
  数据集/Benchmark：GQA、MMB、MMB-CN、MME、POPE（幻觉）、SQA-IMG、VQA-V2、VQA-Text（TextVQA）、VizWiz、DocVQA（文本为主任务）、MVBench（视频）、Vinoground（视频密集时序推理）。COCO val2017 用于 IoU 分析。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/Danielement321/HiPrune
  工具：LMMs-Eval 工具包评估，calflops 计算 FLOPs。

  算法 pipeline（以 LLaVA-1.5 为例，l=9, α=0.1）：
  ```
  # Step 1: ViT 编码图像，获取各层注意力
  image_tokens, all_attns = encoder(image)  
  # image_tokens: (N=576, d), all_attns: list of (H, N+1, N+1)

  # Step 2: 从 object layer l 提取注意力分数
  mid_attn = all_attns[l].squeeze(0)         # (H, N+1, N+1)
  mid_attn = mid_attn.mean(0)                # 平均 multi-head -> (N+1, N+1)
  mid_attn = mid_attn.sum(0)[1:]             # 每个 token 收到总注意力 -> (N,)
  # a_i^{[l]} = (1/H) * sum_h sum_n A^{[l]}[h,n,i]

  # Step 3: 选择 Anchor Tokens（中间层 top-k 注意力）
  N_a = round(N' * α / 5)                    # α=0.1, 5 tokens per cluster
  anchor_idx = topk(mid_attn, k=N_a)

  # Step 4: 选择 Buffer Tokens（空间相邻，Cross scheme）
  # 每个 anchor 的上下左右 4 个邻居
  buffer_idx = cat([anchor_idx-1, anchor_idx+1, 
                    anchor_idx-p, anchor_idx+p])
  anchor_buffer_idx = unique(cat([anchor_idx, buffer_idx]))

  # Step 5: 从输出层选 Register Tokens
  deep_attn = all_attns[-1].squeeze(0).mean(0).sum(0)[1:]  # (N,)
  mask = zeros(N); mask[anchor_buffer_idx] = 1
  deep_attn -= mask
  r_sum = N' - len(anchor_buffer_idx)
  register_idx = topk(deep_attn, k=r_sum)

  # Step 6: HiPrune++ 可选 - 文本引导补充
  avg_text = text_encoder(text).mean(-2)     # 平均 text embedding
  avg_text /= avg_text.norm(-1)
  image_tokens_norm = image_tokens / image_tokens.norm(-1)
  similarity = avg_text @ image_tokens_norm.T  # (N,)
  t_sum = round(N' * β / 5)                 # β=0.1
  text_idx = topk(similarity, k=t_sum)

  # Step 7: 保留选中 token，丢弃其余
  retained_idx = cat([anchor_idx, buffer_idx, register_idx, text_idx])
  retained_tokens = image_tokens[retained_idx]  # (N', d)
  # retained_tokens 送入 projector → LLM
  ```

  张量计算关键点：
  - N=576 (LLaVA-1.5), N=2880 (LLaVA-NeXT 5 crops)，N 在 Qwen 动态分辨率下可变
  - d=1024 (CLIP-ViT-L/14 hidden dim)
  - 注意力维度：(H=16, N+1, N+1)，+1 for CLS token
  - HiPrune 在 ViT 输出后、projector 之前执行，与 FlashAttention 兼容
  - 排序开销 <1% prefill latency，HiPrune++ text encoder 开销 <10%
