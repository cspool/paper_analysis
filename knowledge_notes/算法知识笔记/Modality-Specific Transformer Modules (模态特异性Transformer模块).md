## Modality-Specific Transformer Modules (模态特异性Transformer模块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modality-Specific Transformer Modules是LMFusion提出的多模态生成模型架构设计模式，核心思想：为每种模态（文本、图像）创建独立的Transformer计算模块（QKV投影、O投影、FFN和LayerNorm），各模态数据仅路由到其专用模块处理，而自注意力层的Q/K/V在拼接后跨模态共享。LMFusion使用两套并行Transformer参数：(1) 文本模块 Proj_text/QKV_text/O_text/FFN_text/LM_Head_text 从Llama-3 8B初始化；(2) 图像模块 UNet_Down_img/QKV_img/O_img/FFN_img/UNet_Up_img 也从Llama-3 8B初始化（除U-Net从头训练）。每个token仅激活其所在模态的模块（一半参数），因此虽总参数量是dense模型的2倍，每次前向FLOPs与dense模型相同。与Mixture of Experts中modality-aware expert routing有概念联系——每个模态有专属"expert"参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LMFusion单层前向（文本token数M，图像patch数N）：
```
# Step 1: 模态特异性输入投影
h_txt = Proj_text(x_txt)                # [M, d], text专用embedding
h_img = UNet_Down_img(x_img_t, t)       # [N, d], image专用下采样

# Step 2: 模态特异性QKV投影
Q_txt, K_txt, V_txt = QKV_text(h_txt)   # text专用QKV
Q_img, K_img, V_img = QKV_img(h_img)    # image专用QKV

# Step 3: 跨模态自注意力
# Text queries attend到所有keys: [K_img, K_txt]
A_txt = softmax(Q_txt @ [K_img, K_txt]^T / sqrt(d) + M)
h_O_txt = O_text(A_txt @ [V_img, V_txt])  # text专用O投影

# Image queries attend到所有keys: [K_txt, K_img]
A_img = softmax(Q_img @ [K_txt, K_img]^T / sqrt(d) + M)
h_O_img = O_img(A_img @ [V_txt, V_img])  # image专用O投影

# Step 4: 模态特异性FFN
h_FFN_txt = FFN_text(h_O_txt)
h_FFN_img = FFN_img(h_O_img)

# Step 5: 模态特异性输出
p_logits = LM_Head_text(h_FFN_txt)
ε_pred = UNet_Up_img(h_FFN_img, t, h_img)
```
文本和图像在attention层有双向cross-modal交互，但由于QKV分离，两者attention计算独立——text的attention不改变image的QKV参数，反之亦然。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
文本模块从Llama-3 8B加载后冻结（η_text=0），图像模块从Llama-3 8B初始化后可训练（η_img=1e-4）。实现要点：(1) 计算复用——预训练LLM语言知识通过冻结文本模块完整保留；(2) 知识迁移——图像模块从Llama-3权重初始化获得文本预训练的权重先验；(3) 梯度隔离——图像扩散的梯度不反向传播到文本模块，避免灾难性遗忘；(4) FLOPs效率——每token仅激活对应模态的模块（一半参数）。LLaVAFusion验证了相同范式可应用于已有VLM（LLaVA-NeXT）。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation
