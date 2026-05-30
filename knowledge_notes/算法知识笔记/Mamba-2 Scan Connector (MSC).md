## Mamba-2 Scan Connector (MSC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2 Scan Connector (MSC) 是ML-Mamba论文提出的新型多模态连接器，用于在MLLM中桥接2D非因果视觉特征与1D因果状态空间模型（SSM）的处理能力。核心组件：(1) Mamba-2 Visual Selective Scanning (MVSS) 模块——将2D视觉patch序列通过Mamba-2层的selective scan进行空间上下文建模；(2) 可选的SwiGLU模块——对扫描后的特征进行gated feature extraction。MSC有三种变体：MLP（纯三层MLP，baseline）、MSC-MLP Basic（MSC不含SwiGLU + MLP）、MSC-MLP Advanced（MSC含SwiGLU + MLP）。MSC的设计motivation在于：传统SSM处理的是具有因果关系的1D序列（如语言），而视觉编码器产生的patch序列缺乏自然因果顺序，直接展平为1D序列送入SSM会丢失2D空间关系。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MSC-MLP Advanced（含BSM扫描 + SwiGLU模块）的前向计算：
```
Input: V_img ∈ R^{N_v×D_v}  (N_v=729个visual patches, 从DINOv2+SigLIP双编码器concat)

// Step 1: Mamba-2 Visual Selective Scan (MVSS) - BSM
// 前向扫描：沿原始patch展开顺序
V_f = Mamba2_Block(V_img)        // 1D SSM scan through Mamba-2 layer
// 后向扫描：反转patch顺序
V_b = Mamba2_Block(flip(V_img))
// 合并前后向信息
V_scan = V_f + flip(V_b)         // ∈ R^{729×D_v}

// 每个Mamba2_Block内部:
//   x_proj, z_proj = Linear_in(x)  // expand 2×
//   x_conv = CausalConv1d(x_proj, window=4)
//   x_act = SiLU(x_conv)
//   Δ, B, C = Linear_dt(x_act)  // data-dependent params
//   A_bar, B_bar = discretize(A, B, Δ)  // ZOH
//   h_t = A_bar ⊙ h_{t-1} + B_bar ⊗ x_act[t]  // recurrent update
//   y[t] = C ⊗ h_t
//   y = y ⊙ SiLU(z_proj)  // gating
//   output = Linear_out(y)

// Step 2: SwiGLU Feature Extraction
V_gate = Linear_gate(V_scan)     // gate projection
V_proj = Linear_proj(V_scan)     // value projection
V_swiglu = SiLU(V_gate) ⊙ V_proj  // gated activation ∈ R^{729×D_v}

// Step 3: MLP Projector (三层MLP)
V_final = MLP_3layer(V_swiglu)   // ∈ R^{729×D_llm}, 维度对齐至LLM embedding空间
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MSC实现为ML-Mamba代码库（https://github.com/WenjunHuang94/ML-Mamba, MIT License）中的核心模块。训练时MSC和MLP Projector首先在558K LAION-CC-SBU子集上做对齐训练（1 epoch，冻结视觉编码器和LLM），然后在665K Mixed Dataset上做全参数监督微调（1 epoch，解冻LLM）。消融实验（Table 6）证明MSC-MLP Advanced在VQAv2（75.26 vs MLP-only 73.42，+1.84）上优于纯MLP方案。MVSS模块使Mamba-2的selective mechanism（数据依赖的Δ/B/C参数）在visual token之间自适应分配注意力，弥补了纯MLP连接器无法建模空间关系的缺陷。使用场景：任何需要将视觉特征映射到Mamba/SSM-based LLM的多模态任务。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
