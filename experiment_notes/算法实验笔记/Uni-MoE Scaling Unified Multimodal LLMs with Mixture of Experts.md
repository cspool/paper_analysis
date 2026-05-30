## Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：首次将 MoE 架构应用于统一多模态大语言模型（MLLM），构建 **Uni-MoE**，可处理图像、视频、音频、语音和文本五种模态。核心设计包括：(1) **模态特定编码器+连接器**——使用 CLIP-V（视觉）、Whisper-small（语音）、BEATs（音频）作为冻结编码器，通过 MLP 线性投影（视觉）和四层 Q-Former+线性投影（音频/语音）将各模态映射到 LLM 语言空间；(2) **稀疏 MoE LLM 架构**——基于 Vicuna-7B（LLaMA-7B），将部分 FFN 层替换为稀疏 MoE 层，每层包含 4 或 8 个专家，使用线性 Router 进行 token 级别的 Top-k（k=2）专家选择；(3) **三阶段渐进式训练**——阶段一：跨模态对齐（仅训练连接器），阶段二：训练模态特定专家（对各模态数据分别训练对应专家 FFN），阶段三：使用 LoRA 在混合多模态数据上联合微调整个 Uni-MoE；(4) **LoRA 高效微调**——阶段二 rank=64/alpha=16，阶段三 rank=8/alpha=16，仅微调 LoRA 参数和投影层而不更新全部专家参数。
  - 实验比较：(a) Uni-MoE 与 Dense 统一多模态模型（Macaw-LLM、X-InstructBLIP）在语音-图像理解（A-OKVQA、OK-VQA、VQAv2 语音版、MMBench-Audio、RACE-Audio、EHSL）、音频理解（ClothoAQA、ClothoV1/V2 CIDEr）、图像-文本理解（A-OKVQA、OK-VQA、VQAv2、MMBench）、视频 QA（ActivityNet-QA、MSVD-QA）上的性能；(b) Uni-MoE MoE 变体 vs 单专家（Dense）变体在不同模态任务上的对比；(c) Top-k 值（1 vs 2）、专家数量（1/2/4/8, mixture vs pure）、MoE 层插入方式（First-Half/Second-Half/Interval/All）的消融实验；(d) 纯 MoE（相同初始专家）vs 混合 MoE（预训练模态特定专家）的对比；(e) 辅助平衡损失（auxiliary balancing loss）在 pure MoE 和 mixture MoE 场景下的效果。

- 硬件平台是什么，配置是什么。
  - 阶段一（跨模态对齐）：2 块 NVIDIA A100 GPU，global batch size=32
  - 阶段二（专家训练）：2 块 NVIDIA A100 GPU，global batch size=16
  - 阶段三（MoE 训练）：1/8/16 块（2 节点）NVIDIA A800 GPU，支持数据并行+专家并行+多节点多 GPU 并行训练

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 Vicuna-7B（LLaMA-7B），Uni-MoE-7B×4-Top2（4 专家，激活 8.9B/总 13.2B）、Uni-MoE-7B×4-Top2†（32 MoE 层，激活 11.1B/总 19.7B）、Uni-MoE-7B×8-Top2（8 专家，激活 8.9B/总 21.9B）、Uni-MoE-7B×8-Top2†（32 MoE 层，激活 11.1B/总 37.0B）。编码器：CLIP-V（视觉）、Whisper-small（语音）、BEATs（音频）。
  - 训练数据集：Common Voice（短语音 1.7M）、LLaVA-Instruct-150K（含 TTS 转换的语音-图像变体 I-A 和原始文本-图像 T-I）、LibriSpeech（长语音拼接）、RACE（文本转长语音 T-A）、WavCaps/AudioCaps/ClothoV1/ClothoAQA/MELD（音频字幕和问答）、Video-Instruct-Dataset（视频指令 100K）、LLaVA-v1.5-665K（扩展图像-文本）。
  - 评估 Benchmark：A-OKVQA、OK-VQA、VQAv2（语音版，EM 指标）；MMBench-Audio（语音-文本-图像长语音推理）；RACE-Audio（长语音多选）；English High School Listening Test/EHSL（真实高考英语听力）；ClothoV1/V2（CIDEr 指标）、ClothoAQA（EM 指标）；ActivityNet-QA、MSVD-QA（准确率+Score，Video-ChatGPT 评估方法）；图像-文本 sanity check：MMBench、POPE、SEED-Bench、MM-Vet。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源地址：https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs
  - 算法 pipeline（以视频理解为例，跨模态 tokens → MoE LLM 推理）：
    ```
    # 第一步：模态编码和投影（不同模态并行编码）
    I = MLP(CLIP-V(image))              # 图像 token，式(2)
    V = Mean(CLIP-V([I_1,...,I_8]))     # 视频 8 帧平均池化，式(3)
    A = Audio-QFormer(BEATs(audio))     # 音频 Q-Former 蒸馏，式(4)
    S = Speech-QFormer(Whisper(speech)) # 语音 Q-Former 蒸馏，式(5)
    T = Word-Embedding(text)            # 文本嵌入，式(6)
    
    # 第二步：拼接多模态序列
    x_0 = [V_1,...,V_N; T_1,...,T_z; A_1,...,A_k]  # 式(12)
    
    # 第三步：MoE LLM 逐层前向（以第 l 层为例）
    X_l^s = MSA(LN(X_{l-1})) + X_{l-1}             # 共享自注意力，式(13)
    X_l^M = MoE(LN(X_l^s)) + X_l^s                  # 稀疏 MoE FFN，式(14)
    x_l = LN(X_l^M)                                  # 层归一化，式(15)
    
    # MoE 层内部（token 级路由）：
    P(X_l^s)_i = softmax(W_router · X_l^s)_i        # Router 计算专家概率，式(16)
    MoE(X_l^s) = sum_{i=1}^{k} P(X_l^s)_i · e_i(X_l^s)_i  # Top-k 专家加权，式(17)
    
    # LoRA 微调每个专家（不更新全量参数）：
    h_e1_LoRA = LoRA-e_1(X_E1) = (W_0 + B·A) · X_E1  # 式(19)-(20)
    h_e1 = e_1(X_E1) + h_e1_LoRA                      # 式(21)
    ```
  - 训练 pipeline 三阶段详见 Algorithm 1（论文原文伪代码）。阶段一仅训练 Connector 参数（Q-Former + 线性投影层），loss 为交叉熵生成损失；阶段二复制阶段一权重，训练各模态对应的单个专家 FFN（LoRA + 投影层）；阶段三加载阶段二预训练专家权重到 MoE 层，使用 LoRA 在混合多模态数据上联合微调，冻结 LLM 全部参数（包括专家），仅更新 LoRA 参数、Router 和投影层。
