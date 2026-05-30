## OmniAlignNet

术语是什么？
OmniAlignNet 是 NVIDIA OmniVinci 论文提出的跨模态对齐网络模块，用于在共享的 omni-modal 潜在空间中强化视觉嵌入和音频嵌入的对齐。其核心思想是：对于同一视频的视觉帧和同步音频轨，两者存在内在的语义互补关系（视觉提供空间/物体信息，音频提供语音/环境声信息），通过 CLIP-style 双向对比学习，使同一视频的视觉-音频嵌入对在共享空间中相互拉近，不同视频的拉远。

具体流程：给定视频的视觉嵌入序列 $\mathbf{E}_v \in \mathbb{R}^{N_v \times C}$ 和音频嵌入序列 $\mathbf{E}_a \in \mathbb{R}^{N_a \times C}$（$C$ 为潜在维度），初始化可学习 query $\mathbf{Q}_v, \mathbf{Q}_a \in \mathbb{R}^{1 \times C}$，通过 cross-attention 将变长序列投影为固定大小 $(1 \times C)$ 的表示；再经 3 层 self-attention + L2 归一化得到 $\mathbf{V}, \mathbf{A} \in \mathbb{R}^{K \times C}$（$K$ 为 batch 中视频数）。对比损失为对称交叉熵：$\mathcal{L}_{\text{o-align}} = \frac{1}{2}(\mathcal{L}_{v \to a} + \mathcal{L}_{a \to v})$，其中 $\mathcal{L}_{v \to a} = -\frac{1}{K}\sum_i \log\frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}$，$s_{ij} = \mathbf{V}_i^T \mathbf{A}_j$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
OmniAlignNet 位于视觉 projector 和音频 projector 之后、LLM backbone 之前，作为独立的对齐模块运行。在训练阶段，OmniAlignNet 的对比损失 $\mathcal{L}_{\text{o-align}}$ 作为辅助损失加入总训练目标，与 LM loss 联合优化。在推理阶段，OmniAlignNet 的对齐权重已固化在模型参数中，视觉和音频嵌入直接通过训练好的对齐空间进入 LLM，无需额外计算对比损失。核心计算流程：
```
# 训练阶段
E_v = VisualProjector(ViT(video_frames))    # [N_v, C]
E_a = AudioProjector(AF_Whisper(audio))     # [N_a, C]
V = L2Norm(SelfAttn3(CrossAttn(Q_v, E_v)))  # [K, C]
A = L2Norm(SelfAttn3(CrossAttn(Q_a, E_a)))  # [K, C]
L_align = CLIPContrastiveLoss(V, A)         # 辅助损失

# 推理阶段
# OmniAlignNet 对齐参数已固化，视觉/音频嵌入直接送入 LLM
omni_seq = [V_embeds, A_embeds]  # 按 TEG 重排后输入 LLM
output = LLM(omni_seq, text_prompt)
```

消融实验：+TEG+CRTE baseline 平均得分 50.25，加入 OmniAlignNet 后提升至 52.59 (+2.34)，其中 Omnibench 提升最显著 (+6.1)，证明跨模态对比对齐对 image-audio 联合理解尤为关键。

术语一般如何实现？如何使用？
受 ImageBind [Girdhar et al., CVPR 2023] 启发，使用共享嵌入空间绑定多模态。实现上基于 PyTorch，OmniAlignNet 模块包含：可学习 query 向量、cross-attention 层（将变长序列压缩为固定维度）、3 层 self-attention（增强模态内和跨模态交互）、L2 归一化、以及对称 CLIP 对比损失。训练时 batch 内需要正样本对（同一视频的视觉+音频），batch size 越大对比效果越好。开源实现：GitHub (NVlabs/OmniVinci)，社区 PyTorch 实现 (kyegomez/OmniAlignNet)。使用场景：任何需要对齐视觉和音频模态的多模态模型，尤其适用于视频理解任务。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM
