## ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos

- baseline方法是什么？
  Baseline是**VTimeLLM**（一个标准的非递归VLM用于时序定位），结合**CONE的CLIP相似度排序方法**。VTimeLLM使用均匀帧采样（如从小时级视频中均匀采样100帧）提取CLIP特征后送入LLM预测事件边界，然后使用CLIP相似度（平均池化帧特征与文本特征的dot product）对所有候选段进行排序，选取Top-K预测。

  Baseline（VTimeLLM + CONE, MAD dataset, 约110分钟视频）全栈执行例子：
  - 算法层：输入小时级视频（约110分钟）→ 均匀采样100帧（丢失大量时序细节，尤其在moment-to-video比极低如4.1s/110min的场景下）→ Frozen CLIP ViT-L/14提取每帧CLS token (100×768) → 线性投影到LLM嵌入空间 (100×4096) → Vicuna-7B LLM预测事件边界 "From s to e" → 将视频分割为段，对每段重复预测 → CLIP相似度排序（每段mean pooled frame CLS dot product with text CLS）→ 选Top-K。问题：(1) 均匀100帧采样导致严重时序信息丢失（Table 2中R1@.1=0.0, 所有Recall=0）；(2) CLIP相似度排序置信度校准极差——ECA@IoU=0.1高达0.6231，大量高置信度假阳性；(3) 缺少对比训练，模型只能见过正样本段，从未被训练判断"事件不存在"，在长视频中产生大量误检。
  - 系统框架层：PyTorch + HuggingFace Transformers，标准VLM推理pipeline。无专用Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准PyTorch操作，无自定义kernel。
  - 硬件架构层：8×NVIDIA A100 GPUs用于训练。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ReVisionLLM通过**递归层次化视觉感知**将时序定位从"一次性全局预测"改为"由粗到细的递归聚焦"，解决了Baseline的三大缺陷：

  **(a) 缺陷1：均匀帧采样导致时序信息丢失（R1@.1=0.0）** → 递归层次化处理
  Baseline从110分钟视频均匀采样100帧，等价于每次采样间隔约66秒，4.1秒的事件可能被完全跳过。ReVisionLLM使用**三层递归**：
  - Top层（hierarchy=3）：用稀疏特征（每段压缩为1个768维token）扫描全视频（~150分钟→100个段，段长125秒，步长25秒），粗定位5分钟级的感兴趣区域。输入token数从100降至100个sparse tokens（每段1个）。
  - 中层（hierarchy=2）：在上层预测区域附近聚焦（约50分钟→33个段），进一步缩小到分钟级。
  - Bottom层（hierarchy=1）：在最终选定的少数段内，使用250帧密集特征（每帧都保留），精确定位起止时间（秒级精度）。
  通过sparse-to-dense的递归搜索策略，ReVisionLLM默认仅处理57%的视频帧（vs baseline 100%），却将R1@.1从0.0提升至15.0%。

  **(b) 缺陷2：置信度校准极差（ECE=0.62）导致大量高置信度假阳性** → Contrastive Segments + LLM内部置信度
  Baseline仅用正样本训练（只见过包含目标事件的段），且使用CLIP相似度排序，导致无法有效区分真假阳性。ReVisionLLM引入：
  - **Contrastive Segments训练**：Stage 1从小时级视频中随机采样不含目标事件的高迷惑性段（同视频内不重叠于ground truth的段），训练模型输出"Not Present."或对存在性判断"Does <event> happen? Answer yes or no." → 负样本回答"No"，直接训练模型辨识视觉输入的信心。
  - **LLM熵基置信度**：推理时计算LLM生成每个词的概率分布熵，取平均熵倒数作为置信度 $R^i = 1 / \text{mean}(H_k^i)$，而非依赖CLIP的跨模态相似度。ECE从0.6231降至0.4614（Table S1）。
  累积消融（Table 2）：+Contrastive Segments: R1@.1 1.4%→4.8%, +Calibration (-CONE): R1@.1 4.8%→8.4%。

  **(c) 缺陷3：单层处理无法应对极低moment-to-video比和长视频扩展性** → 渐进式训练 + 层次化适配器
  Baseline尝试在训练时直接处理完整长视频会因显存和计算资源爆炸而失败。ReVisionLLM的渐进式训练：
  - Stage 1：仅用短片段（~125秒段）训练模型识别事件存在性和精确边界，计算开销小。
  - Stage 2：冻结Hierarchical Adapter，引入稀疏特征压缩（段级压缩比高达250:1），仅微调新LoRA模块处理长视频。
  同时，Hierarchical Adapter设计为轻量级（2层Cross-Attn + 2层Self-Attn vs CLIP 24层），几乎不增加额外计算开销。Ablation on Video Length（Figure 5）证明递归方法可将性能从2h稳定扩展到10h，非递归方法在10h完全失败。

  对比baseline的全栈执行例子（ReVisionLLM, 默认Top-to-Bottom, MAD dataset）：
  - 算法层：输入110分钟视频→ Frozen CLIP ViT-L/14提取每帧CLS token (T×768) → 滑动窗口分段（125s段, 25s步长, 每段uniform采样250帧）→ Hierarchical Adapter生成稀疏特征（Cross-Attention对齐文本 + Self-Attention压缩为1×768 per segment）和密集特征（Linear Projection 768→4096）→ **Hierarchy 3**: LLM接收100个稀疏token + "when can we see <event> happening?" → 粗粒度预测 → **Hierarchy 2**: LLM接收33个稀疏token（聚焦区域）→ 中等粒度预测 → **Hierarchy 1**: LLM接收250个密集token（选定段）→ 精确边界秒级输出 "From 4562 to 4577" → 计算LLM输出熵的置信度排序 → Top-K最终预测。Epochs: 5(MAD)/1(VidChapters-7M) for Stage 1, 2 for Stage 2. LoRA r=64, α=128.
  - 系统框架层：PyTorch + HuggingFace Transformers, 8×A100 GPUs, AdamW optimizer, cosine LR decay。LoRA高效微调，无需全参数更新。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准PyTorch操作（nn.Linear, nn.MultiheadAttention），无自定义CUDA kernel。
  - 硬件架构层：8×NVIDIA A100 GPUs集群。每GPU batch size=16（total 128）for Stage 1短片段训练, batch size=1（total 8）for Stage 2长视频训练。推理仅需单GPU。
