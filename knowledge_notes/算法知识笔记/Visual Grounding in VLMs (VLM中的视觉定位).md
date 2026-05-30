## Visual Grounding in VLMs (VLM中的视觉定位)

术语解释
Visual Grounding 在 VLM 中指模型基于自然语言描述（类别名、特征描述或指代表达），在图像中定位并输出对应目标物体的 bounding box 坐标的能力。DeepSeek-VL2 将视觉定位作为新增能力引入，通过 special tokens (<|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>) 在文本序列中编码定位信息。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Grounding 使 VLM 不仅能"看"图像和回答文字问题，还能"指"出图像中的具体位置，是通往 embodied AI 和 visual agent 应用的关键能力。DeepSeek-VL2 实现三类 grounding：(1) Referring Expression Comprehension (REC)——给定 "cat" 或 "the leftmost person"，输出 bounding box；(2) Grounded Conversation——在对话回复中引用具体目标位置（如 "Two <|ref|>dogs<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> are running"）；(3) In-context Visual Grounding——给定第一张图中参照目标（可能被 visual prompt 如红框高亮），在第二张图中定位同类目标。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== Visual Grounding Data Format (DeepSeek-VL2) ===
// Format A: Object Localization
Prompt:  "Locate <|ref|>car<|/ref|> in the given image."
Target:  "<|ref|>car<|/ref|><|det|>[[x1,y1,x2,y2],...]<|/det|>"

// Format B: Grounded Conversation
Prompt:  "<|grounding|>Can you describe the content of the image?"
Target:  "Two <|ref|>dogs<|/ref|><|det|>[[120,80,340,350]]<|/det|> are running..."

// Format C: In-context Grounding (2 images)
Prompt:  "<|grounding|>The first image shows an object within the red 
          bounding box. Please identify the same category in the 2nd image."
Target:  "<|ref|>cat<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>"

// Coordinate Normalization: [x1,y1,x2,y2] ∈ [0,999] 
// (top-left, bottom-right), normalized to image resolution
```

Grounding 输出的 bounding box 坐标归一化到 [0, 999]（共 1000 个 bin），模型通过 next-token prediction 学习生成坐标数字。训练时引入 negative samples（图中不含目标物体时 model 不应输出任何 box）增强鲁棒性。DeepSeek-VL2 实现了 emergent generalization——训练主要来自自然场景图像，却能在 meme、动漫等域做 grounding。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VLM 中实现视觉定位有几种主流范式：(a) 文本坐标范式（DeepSeek-VL2, Qwen2-VL, Kosmos-2, Shikra, Ferret）——将 bounding box 表示为 special token 包裹的文本坐标，利用 LLM 的 next-token prediction 自然生成；(b) 额外检测头范式（Grounding DINO, Florence-2）——额外训练 detection head 输出 box regression；(c) 定位 token 范式（Groma, Molmo）——引入额外的 <loc> 等定位专用 token。文本坐标范式的优势是无需额外参数量，但与 LLM 的自然分布（自然语言而非数字序列）存在 gap，需要大量 grounding 数据 training。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
