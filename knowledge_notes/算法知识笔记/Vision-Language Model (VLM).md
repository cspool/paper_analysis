## Vision-Language Model (VLM)

术语是什么？通过联网搜索让回答具体和精准。
Vision-Language Model (VLM) 是一种多模态AI模型，能联合推理视觉和文本数据。现代VLM由vision encoder（如ViT）和Large Language Model (LLM) 组成：vision encoder将图像/视频帧切分为patches并tokenize为visual embeddings，经projection映射到LLM的word embedding space后与text tokens拼接，由LLM的Transformer自回归生成文本输出。代表性VLM包括LLaVA-OneVision、LLaVA-Video、MiniCPM-V、Qwen2.5-VL等。视频VLM中，视频被采样为帧，每帧独立tokenize；视觉tokens通常占输入的98-99%（如LLaVA-OneVision在VideoMME上平均6272 visual tokens vs 109 text tokens），LLM部分占99.35%参数和98.98%操作，因此VLM推理效率的核心瓶颈在LLM侧的视觉token处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VLM推理pipeline：`Video → Frame Sampling → Vision Encoder (per-frame patch tokenization) → Visual Embedding Projection → Concatenate with Text Tokens → LLM Transformer (Multi-Head Attention + FFN layers) → Text Generation`。以LLaVA-Video-7B为例：输入视频→采样为N帧→每帧被ViT编码为M_f个visual tokens→所有帧的visual tokens (M=N×M_f) 与text tokens (T) 拼接→送入LLM的32层Transformer→attention层计算QK^T softmax矩阵含四个block: image-to-image (M×M)、image-to-text、text-to-image (T×M)、text-to-text→FC/FFN层对拼接后的hidden states做GEMM→最终输出text tokens。VLM推理的计算量主要由M决定（M ≫ T），减少M是加速VLM的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VLM通常通过HuggingFace Transformers加载预训练权重实现推理，使用lmms-eval等多模态benchmarking框架评估。部署场景包括云端GPU (如NVIDIA A100/H100)、边缘设备 (如Jetson Orin) 和专用加速器。由于visual tokens数量远大于text tokens，VLM推理优化重点在视觉冗余消除：token pruning（移除不重要的visual tokens）、token merging（合并相似tokens）、sparse attention等。Focus论文指出VLM的LLM部分占绝大多数参数和计算，因此优化LLM处理visual tokens的效率是加速VLM的核心。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

