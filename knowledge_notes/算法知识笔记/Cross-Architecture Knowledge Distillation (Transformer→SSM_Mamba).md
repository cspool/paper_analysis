## Cross-Architecture Knowledge Distillation (Transformer→SSM/Mamba)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
跨架构知识蒸馏是将知识从一种模型架构（通常为更强的Transformer teacher）迁移到另一种架构（如Mamba/SSM student）的技术。区别于传统同架构蒸馏（如DeepSeek-R1蒸馏到Qwen/Llama：仅需复制logits或hidden states），跨架构蒸馏面临额外挑战：teacher和student的token mixing机制根本不同（softmax attention vs selective SSM scan），直接复用或近似权重矩阵不可行。M1论文的解决方案是通过MambaInLlama方法：将Transformer attention层的Q/K/V/O投影权重映射为Mamba层的C/B/X/O投影，对GQA的KV heads扩展至full heads（因Mamba无KV cache），新增MLP（生成Δ_t）和A参数，然后通过reverse KL divergence蒸馏。M1发现直接跨架构蒸馏推理能力效果差（MATH500仅38%），创新性地采用分阶段策略：先蒸馏通用MATH能力，再SFT推理数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# M1跨架构蒸馏流程（完整三阶段）
# Teacher: Llama3.2-3B-Instruct (Transformer, GQA, 28 layers)
# Student: Hybrid Mamba (28 layers: 22 Mamba + 6 Attention保留)

# Stage 1: 权重初始化映射 (MambaInLlama Algorithm 1)
For each attention layer to convert to Mamba:
  W_C_student = W_Q_teacher       # Q投影→C投影
  # GQA扩展: 8 KV groups→28 full heads (因Mamba无KV cache)
  W_B_student = Linear_expand(W_K_teacher)  # head_dim*kv_head → head_dim*n_head
  W_X_student = Linear_expand(W_V_teacher)  # V投影→X投影 (同样扩展)
  W_O_student = W_O_teacher       # 输出投影直接复用
  MLP_Δ = random_init()           # 新增: 生成Δ_t的MLP
  A = random_init()               # 新增: dynamic parameter ∈ R^{N×N'}
  # MLP layers: 直接复用Transformer的MLP权重

# Stage 2: Reverse KL蒸馏 (token-level)
for input_ids, attention_mask in dataloader:
  # Chat template: mask user prompt, 仅计算assistant token loss
  p_teacher = Teacher(input_ids).logits  # [B, L, V]
  p_student = Student(input_ids).logits
  # Reverse KL: D_KL(p_student || p_teacher) = Σ p_student * log(p_student/p_teacher)
  loss = (p_student * (log(p_student) - log(p_teacher))).sum(dim=-1)
  loss = loss * assistant_mask  # 仅assistant token
  loss.backward()
# Optimizer: AdamW, LR=1e-5, cosine decay, β=(0.9,0.95), weight_decay=0.1
# Data packing: 合并多序列至max_len=8192

# Stage 3a: Math SFT (OpenMathInstruct-2, 2 epochs)
# Stage 3b: Reasoning SFT (10B tokens from R1-generated datasets, 5 epochs)
# Stage 3c: GRPO RL (50 steps, 128 batch, 8 rollouts/question)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
M1使用Axolotl框架（https://github.com/axolotl-ai-cloud/axolotl）实现蒸馏和SFT。关键设计决策：(1) 不追求近似softmax attention矩阵（与T2R策略相反），直接替换让Mamba学到自己的计算范式；(2) 先通用后专项的分阶段策略——先用OpenMathInstruct-2建立Mamba MATH基础（MATH500 45%→74%），再用10B reasoning tokens做推理SFT（74%→82%），克服了直接跨架构推理蒸馏数据不足问题；(3) 6/28=21%的attention层保留——完全去除attention会导致性能崩溃，少量attention层提供关键的长程信息路由能力。开源：https://github.com/jxiw/M1。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

---
