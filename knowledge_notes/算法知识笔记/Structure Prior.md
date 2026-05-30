## Structure Prior

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Structure Prior是Dimple提出的离散扩散模型输出控制机制。由于扩散模型在生成前已知完整序列长度，且任意位置token可独立预测，可在初始化时预置特定位置的token值，这些位置在后续迭代解码中始终保持不变（不被mask、不参与更新）。Structure Prior实现：(1) 精确输出格式控制（JSON/LaTeX），不依赖instruction prompt间接引导；(2) 推理步骤结构控制（如强制先描述image1再image2）；(3) 精确长度和结束位置控制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Structure Prior使用流程（以Structured Reasoning为例）：

```
Response Length: 64 tokens

1. 定义Structure Priors:
   Prior 1: position[0:5] = "In the first image, there "
   Prior 2: position[20:25] = "In the second image, there "
   Prior 3: position[50:58] = "The common item in the two images is"

2. 初始化:
   x_T[0:5] = tokenize("In the first image, there ")    # 固定
   x_T[20:25] = tokenize("In the second image, there ")  # 固定
   x_T[50:58] = tokenize("The common item...")           # 固定
   x_T[其他] = [MASK]  # 正常参与扩散

3. 迭代去噪: 每步仅对[MASK]位置预测/更新；固定位置永不被mask
   结果: 第10步解码"scissors"（最终答案）→ 答案先于完整推理步骤出现

4. Length Control变体:
   在position[L-12:L-4]预置 "Thus, the answer is \box{"
   强制模型在此位置输出最终答案，自动调整前序推理跨度
```

Annotations: Prior token值通过tokenizer映射后直接写入序列；固定位置在attention中正常参与（可被attend），但不被更新；Prior可放在任意位置。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

扩散模型bidirectional generation的独特能力——AR模型无法实现（从左到右生成，无法在生成前指定后序token）。实现：初始化时将prior位置设为目标token IDs，解码循环跳过这些位置。用途：格式控制（JSON/XML/LaTeX）、推理引导（三段式分析）、长度控制。局限性：需预知总序列长度和prior绝对位置；prior可能与模型实际推理冲突。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
