## Rethinking_Token_Reduction_for_State_Space_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：面向SSM（Mamba系列）的统一后训练Token Reduction方法——UTRC（Unified Token Reduction by token importance Classification）。核心流程为：(1) 从SSM层的隐藏状态y提取token重要性度量 `S = Σ_{d=1}^{D'} max(0, y_{::d}) / D'`（每个token所有通道的正值求和并平均）；(2) 基于重要性将所有token分为两等份——集合M_A（重要性低的N/2个token）和集合M_B（重要性高的N/2个token）；(3) 为每个M_A中token a_i计算它到M_B中最相似的token f_i：`f_i = argmax_{b_j∈M_B} sim(a_i, b_j)`，得到最大相似度g_i；(4) 保留相似度最高的p%连接对；(5) 对保留的连接对执行UTR操作：对(p×q)%的连接进行pruning（删除a_i，保留f_i），对剩余[p×(1-q)]%的连接进行merging（`f_i = (a_i + f_i) / 2`，删除a_i）；(6) 重新组合token集。默认q=0.5时效果最佳。在hidden states上使用hybrid（q=0.5），在residual connections上只使用merging，避免去除关键残差信息。(7) 层次化应用：从第10~12层开始，每5层执行一次token reduction（如Mamba-2-2.7B在[12,17,22,27,32,37,42]层），使用固定的压缩率。
  - 实验比较：(1) 与baseline方法PuMer和EViT在Mamba-2-1.3B、Mamba-2-2.7B、Mamba-2.8B、Mamba-1.4B上比较，分别在10%/20%/30%的FLOPS Reduction下评估；(2) 消融实验：不同token重要性度量（ℓ1-norm、ℓ2-norm、无Clip、带Clip）；不同reduction位置配置（如[10,15,20,...]/[12,17,22,...]等6种配置）；不同design choices（P-only、M-only、不同q值组合）；(3) GPU峰值内存和吞吐量测量；(4) 附录中与LTMP方法对比。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB GPU
  - 所有实验均在单卡A100上完成（论文未明确说明多卡训练/推理）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba-2-1.3B、Mamba-2-2.7B、Mamba-2.8B（基于Mamba-2架构，Dao and Gu, 2024）、Mamba-1.4B（基于Mamba架构，Gu and Dao, 2023）
  - 数据集/benchmark：LAMBADA（Perplexity + Accuracy）、HellaSwag、PIQA、Arc-Easy、Arc-Challenge、WinoGrade（零样本评估，无微调）
  - 推理配置：生成2048 tokens、batch size 96（峰值内存测量）；prompt length 2048、batch size 16、生成100 tokens（吞吐量测量）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/wuyushuwys/ToR_SSM
  - 框架：PyTorch + HuggingFace Transformers
  - 算法pipeline伪代码（对Mamba第l层执行UTRC token reduction）：

```
# 输入: T_{l-1} ∈ R^{B×N×D} (上一层的token序列，B=batch, N=序列长度, D=特征维度)
#       reduction_rate: 目标压缩率 (如0.2表示减少20% FLOPs)
#       start_layer: 开始reduction的层号
#       interval: reduction间隔层数
#       q: hybrid比例参数 (默认0.5)

# === Step 0: 判断是否在当前层执行reduction ===
if layer_id < start_layer or (layer_id - start_layer) % interval != 0:
    return standard_mamba_layer(T_{l-1})  # 不执行reduction

# === Step 1: 获取SSM隐藏状态并计算Token重要性 ===
x = Linear_proj(T_{l-1})          # 投影到D'维: [B, N, D']
y = SSM(A, B, C)(x)              # 通过SSM, y∈[B, N, D']
# 计算token重要性 (Equation 5)
S_i = sum(max(0, y_{i,:})) / D'  # 对每个token i, 正通道值求平均
                                  # S ∈ R^{B×N×1}

# === Step 2: Token重要性分类 ===
# 按S降序排列所有N个token
sorted_idx = argsort(S, descending=True)  # [B, N]
M_B = sorted_idx[:, :N//2]               # 重要性高的前50% token
M_A = sorted_idx[:, N//2:]               # 重要性低的后50% token

# === Step 3: 建立相似度连接 ===
for each a_i in M_A:
    similarities = [cosine_sim(a_i, b_j) for b_j in M_B]  # cosine相似度
    f_i = M_B[argmax(similarities)]   # Equation 6: 最相似的M_B中token
    g_i = max(similarities)            # Equation 7: 最大相似度值

# === Step 4: 保留top-p%相似连接 ===
# 按g_i降序排序所有连接对
num_keep = int(p * len(M_A))
keep_pairs = sort_by_g_desc({(a_i, f_i, g_i)})[:num_keep]

# === Step 5: Unified Token Reduction (UTR) ===
num_prune = int(q * num_keep)    # q=0.5 by default
num_merge = num_keep - num_prune

# 取前num_prune对做pruning
for (a_i, f_i) in keep_pairs[:num_prune]:
    M_A = M_A \ {a_i}            # 删除a_i, f_i保持不变
# 取后num_merge对做merging  
for (a_i, f_i) in keep_pairs[num_prune:]:
    T_l[f_i] = (T_l[a_i] + T_l[f_i]) / 2   # 平均融合到f_i
    M_A = M_A \ {a_i}                       # 删除a_i

# === Step 6: 重新组装token序列 ===
# 将M_B和缩减后的M_A合并
if reduction_on_hidden_states:
    T_l_hidden = reassemble(M_B, M_A_reduced)
if reduction_on_residual:
    T_l_residual = merge_only(M_B, M_A)  # 残差只用merging

# === Step 7: 最终输出 ===
T_l = Linear(y) + T_l_residual    # 标准Mamba层输出的简化版
return T_l
```

  - 关键代码实现细节：
    - **Token重要性计算**：在Mamba block的SSM输出处（after selective scan），读取y hidden states，按`max(0, y).sum(dim=-1)`计算每个token的importance score，除以D'归一化
    - **Similarity计算**：使用余弦相似度 `cosine_similarity(a_i, b_j)`，通过矩阵乘法实现快速计算：`sim_matrix = norm(M_A) @ norm(M_B).T`
    - **残差连接处理**：论文发现在residual上只用merging（不删除任何残差信息）比hybrid/pruning-only效果更好（PPL 40.61 vs 42.61），因为残差保留了上一层的关键信息
    - **层次化reduction**：不在每层都做reduction（相邻层token重要性相似），每5层做一次；不从太早层开始（前几层尚未充分捕获token重要性），Mamba-2-2.7B从第12层开始
    - **PPL/Accuracy评估适配**：由于token数量减少，评估时需对应调整label logits，只取前(1-m%)的logits计算PPL和Accuracy
