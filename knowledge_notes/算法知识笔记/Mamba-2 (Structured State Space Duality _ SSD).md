## Mamba-2 (Structured State Space Duality / SSD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2由Dao和Gu(2024)提出，通过揭示SSM与attention之间的对偶性（Structured State Space Duality, SSD），将SSM重新表述为矩阵形式，使其可以利用类似FlashAttention的tiling策略进行高效计算。核心发现：SSM的序列变换可以写为半可分离矩阵（semiseparable matrix）M = L ◦ (C · B^T)，其中L是下三角矩阵，◦是逐元素乘法，使得SSM的前向传播等价于矩阵乘法Y = M · X。这一发现使Mamba-2能利用Tensor Core优化的矩阵乘法（而非Mamba的逐元素scan），训练速度达Mamba的8倍。SSD使用head_dim=64的多头结构（类似attention的多头），8个groups，expansion factor=2，conv window=4。论文中8B Mamba-2配置：56层，hidden dim 4096，state dim 128，8 groups，head dim 64，无位置编码。Mamba-2支持tensor parallelism仅需1次all-reduce（与Transformer持平），但需使用GroupNorm（而非LayerNorm）作为内部归一化，且group size需>256以保证统计量精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba-2 SSD层的前向传播（矩阵形式）：
```
Input: x ∈ R^{B×L×D}

// Step 1: Project to Q, K, V (analogous to attention)
Q, K, V = Linear_proj(RMSNorm(x))  // Q,K ∈ R^{B×L×H×P}, V ∈ R^{B×L×H×P}
// H = D/head_dim 个head, P = head_dim

// Step 2: Short convolution + activation
K = CausalConv1d(K, window=4)
Q = SiLU(Q)

// Step 3: SSM scan via matrix multiplication formulation
// M = L ◦ (Q · K^T) where L is lower triangular
// Equivalent computation via chunked parallel scan:
// Split L into chunks, process intra-chunk as MatMul, inter-chunk as recurrent

// Step 4: Output = M @ V (matrix multiply form)
Y = SSD_scan(Q, K, V, A)  // chunked scan using Tensor Core MatMuls

// Step 5: Gating + output
output = x + Linear_out(Y * SiLU(gate))
```
关键优势：Mamba-2的SSD scan通过chunked parallelism利用Tensor Core加速。将序列分为chunks，chunk内使用高效MatMul（Q·K^T和·V），chunk间使用recurrent state传递。论文中的Mamba-2使用head_dim=64、8 groups、state_dim=128、expansion=2、conv window=4。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba-2开源实现：https://github.com/state-spaces/mamba（包含SSD kernel）。论文中使用Megatron-LM的实现，支持tensor/sequence/pipeline parallelism。Mamba-2比Mamba训练快约3x（在8B规模），因SSD scan比Mamba的selective scan快8x。TP通信仅需1次all-reduce，与Transformer持平。适用场景：需要线性复杂度但追求比Mamba更高训练吞吐的大规模语言模型训练。

在 H-Net 中的用法：H-Net 使用 Mamba-2 作为 encoder/decoder 的构建模块。选择 Mamba-2 而非 Transformer 的原因：(1) SSM 的压缩归纳偏置——Mamba-2 将信息压缩为固定大小的 hidden state，天然适合 encoder 将多个输入 token 压缩为 richer representations 的角色；(2) 对 fine-grained 数据的处理能力——在字节级、DNA base-pair 级输入上 Mamba-2 显著优于 Transformer（消融中纯 Transformer E/D 表现最差）；(3) 效率——Mamba-2 的 O(L) 复杂度在操作未压缩长序列（L^0=8192）时至关重要。H-Net 的 E/D 使用 M4（4 层纯 Mamba-2 无 MLP），参数量约 6D²/layer（vs Transformer 的 12D²/layer）。消融还发现在 BPE token 级别输入上 Mamba E/D 也优于 Transformer E/D，说明优势不仅来自 fine-grained 输入处理，也来自 SSM 的压缩能力本身。

在 ML-Mamba 中的多模态用法：ML-Mamba 使用 Mamba-2 2.7B（在Pile数据集300B tokens上预训练）替换传统Transformer backbone（如LLaMA/Vicuna）作为MLLM的语言模型部分。关键设计优势：(1) RNN-like特性使推理时每token O(1)计算且内存恒定——即使处理729个visual tokens + 长文本生成，hidden state大小不变，无KV-Cache增长，实现171 tokens/s的生成速度（vs TinyLLaVA 38 tokens/s, MobileVLM v2 50 tokens/s）；(2) Mamba-2比Mamba-1快2-8倍，使ML-Mamba在推理性能上优于基于Mamba-1的MLLM（VL-Mamba、Cobra）。消融实验（Table 4）显示Mamba-2 2.7B在所有benchmark上全面超越780m和1.3b变体，验证了Mamba-2在MLLM中的scaling特性——更大SSM backbone类似Transformer scaling law持续提升多模态性能。Mamba-2的selective scan的input-dependent特性使模型能自适应地对visual token中的重要patch分配更高"注意力"，对不重要的background patches快速遗忘。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models
- Dynamic_Chunking_for_End-to-End_Hierarchical_Sequence_Modeling
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2
- Rethinking_Token_Reduction_for_State_Space_Models
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---
