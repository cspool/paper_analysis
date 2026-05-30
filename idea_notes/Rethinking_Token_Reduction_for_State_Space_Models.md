## Rethinking_Token_Reduction_for_State_Space_Models

- baseline方法是什么？
  Baseline是将现有Transformer的SOTA Token Reduction方法（EViT的pruning、PuMer的bipartite merging）直接应用于Mamba SSM模型。这些方法原为Transformer设计但直接迁移到SSM时严重失败（如Mamba-2.8B在20% FLOPS reduction下，EViT准确率从63.3%降至43.6%，PuMer降至37.2%）。

  全栈执行例子——以EViT pruning方法处理一个输入token序列为例：

  **算法pipeline层**: Mamba-2.8B正常加载，EViT对当前层所有token按attention值（[CLS] token对其他token的attentiveness）排序，直接删除最不重要的20% token，进入下一层处理。
  **系统框架层**: PyTorch + HuggingFace Transformers的标准推理pipeline，无Serving框架修改，token reduction通过hook插入Mamba block之间。
  **编译框架层**: 论文未明确说明（无编译框架修改）。
  **kernel调度层**: 标准PyTorch CUDA kernel，无自定义kernel。Mamba中的SSM selective scan使用Mamba原生的triton kernel（来自mamba-ssm库），token reduction后调用相同的核函数处理更少的token。
  **硬件架构层**: NVIDIA A100 80GB GPU。未涉及硬件修改。

  Baseline的核心缺陷：
  - **Pruning缺陷**：pruning删除的"低重要性"token仍然包含不可恢复的信息，这些信息在SSM的序列化递推计算（h_t = A̅h_{t-1} + B̅x_t）中被逐token放大传播，删除任何一个token都会影响后续所有token的hidden state累积。
  - **Merging缺陷**：bipartite merging将token均匀分成两组，盲目将一组merge进另一组，完全忽视不同token的内在重要性差异。在SSM中，某些关键token对等式y = x * K̅的卷积结果有决定性影响，将其merge到不重要token中会导致输出发生根本偏移。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出了UTRC（Unified Token Reduction by token importance Classification），通过"重要性分类→相似度匹配→混合prune/merge"的三阶段pipeline解决baseline的缺陷：

  **（1）SSM原生Token重要性度量**：从SSM的hidden state y计算重要性，而非从attention值。具体地，`S_i = Σ_d max(0, y_{i,d}) / D'`，利用SSM高维通道空间（D'）中每个通道对各token的细粒度激活响应。对比发现clip操作（max(0,·)）优于ℓ1/ℓ2 norm，说明只关注正向激活的通道更有信息量。
  **（2）Token重要性分类**：按重要性将token分为M_A（低重要性50%）和M_B（高重要性50%），然后为M_A中每个token寻找M_B中最相似的对应token。只有相似度最高的p%连接对被保留处理，不相似的连接直接丢弃（即对应的不重要token不被保护）。
  **（3）混合Pruning + Merging（UTR）**：对保留的连接对，q比例的做pruning（仅删除M_A中token），(1-q)比例的做merging（将M_A token平均值融合进M_B对应token）。这精确平衡了"信息保留"（merging保留语义信息）和"冗余消除"（pruning清除纯粹冗余），q=0.5效果最优。
  **（4）Intra-layer分支解耦**：Hidden states上使用hybrid策略（q=0.5），Residual connections上仅用merging。原因是残差连接传递的是前一层原始信息，pruning会导致关键残差信息永久丢失，而merging可以将多路径信息融合保留。这解决了baseline中hidden和residual token不同步减少导致的index misalignment问题。
  **（5）层次化reduction**：不每层都reduction（相邻层重要性变化小），而是每5层应用一次，从第10~12层开始（前几层token representation尚未充分成熟）。

  全栈执行例子——以Mamba-2-2.7B在[12,17,22,27,32,37,42]层执行20% FLOPS reduction为例：

  **算法pipeline层**：在第12层SSM block的selective scan输出处，hook截取hidden state y ∈ R^{B×N×D'}，执行：(a) 计算S = max(0, y).sum(dim=-1) / D'得到每个token的重要程度；(b) 按S中位数二分，重要性低的进入M_A；(c) 计算M_A中每个token与M_B的cosine相似度矩阵；(d) 保留最相似的p%连接；(e) 对保留连接中前50%（q=0.5）执行pruning——仅从hidden states移除M_A token，M_B token保持；后50%执行merging——`T[f_i] = (T[a_i] + T[f_i]) / 2`。残差路径仅执行merging（保留所有残差信号的贡献）。第13~16层不执行reduction（间隔=5层逻辑），直到第17层再次执行。N从2048 tokens逐步减少至约1638 tokens（减少~20%FLOPs）。
  **系统框架层**: PyTorch 2.x + HuggingFace Transformers推理pipeline。UTRC作为hook注入每个Mamba block的SSM输出位置（在Linear投影和残差加法之前），不修改Mamba模型权重或架构本身。评估时用特殊logit裁剪：token数减少后PPL/Accuracy只在对应长度的非压缩token上计算。
  **编译框架层**: 论文未明确说明（无编译框架修改）。
  **kernel调度层**: 标准PyTorch CUDA kernel + Mamba原生的selective scan triton kernel（来自mamba-ssm库）。Token reduction后输入SSM scan的序列变短，直接享受更少的scan步数加速。无自定义kernel。GPU峰值内存因token数减少下降14.4%~40.0%（10%~30% FLOPS reduction对应），吞吐提升1.07×~1.37×。
  **硬件架构层**: NVIDIA A100 80GB GPU。未涉及硬件修改。
