## Flash Multi-Head Feed-Forward Network

- 属于算法pipeline的实现是什么？实验比较什么？
  实现FlashMHF架构：将标准SwiGLU FFN替换为Multi-Head FFN + Parallel FFN Sub-Networks设计。核心：（1）多head分解——将FFN输入切分为H个head，每个head独立执行key-value style的FFN计算（公式8-9）；（2）并行子网络——每head内包含E个并行sub-network，通过sigmoid gating学习加权聚合（公式11-13），维持d_e ≈ 8/3·d_h的平衡ratio解决scaling imbalance；（3）最终concat所有head输出并做Wo投影（公式14）。整体类似dense MoE但不做sparse top-k selection。
  实验比较：（i）128M/370M/1.3B规模上FlashMHF vs SwiGLU baseline的validation loss（PG19）和perplexity（Table 1, Figure 4-7）；（ii）FlashMHF vs MH-FFN naive多head的scaling对比（128M vs 370M性能分化）；（iii）FlashMHF vs PKV baseline（Parametric KV——用multi-head attention替换FFN验证element-wise activation必要性）；（iv）FlashMHF vs Dense-MoE baseline（H=1验证多head必要性）；（v）head dimension ablations（d_h=64/128/256）at 370M和1.3B（Table 1, Figure 5-6）；（vi）下游任务：HellaSwag/SIQA/PIQA/OBQA/WinoGrande/RACE（Table 2）；（vii）memory和latency benchmark vs SwiGLU FFN和MH-FFN（Figure 8, Table 5）。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 GPU (Hopper架构)，用于memory/latency benchmark和efficiency实验
  - 训练：NVIDIA H100 GPU集群（PyTorch + bfloat16），128M/370M: 245K steps, 1.3B: 409K steps
  - 推理benchmark：单H100，batch size=8, sequence length从192到16128（Table 5）
  - 软件环境：PyTorch, Triton（consumer GPU kernel实现）, ThunderKittens（Hopper kernel实现）, cuBLAS

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-like architecture（RoPE + multi-head self-attention + SwiGLU FFN / FlashMHF），128M/370M/1.3B参数规模。使用GPT-NeoX tokenizer，vocab size=50,432。config详见Table 4。
  - 训练数据：THE PILE，128M/370M训练60B tokens，1.3B训练100B tokens。context length=4096, batch size=64。
  - 验证集：PG19 validation split（evaluation loss）
  - 下游benchmark：HellaSwag（commonsense reasoning）、SIQA（Social IQA）、PIQA（Physical IQA）、OpenBookQA、WinoGrande、RACE（reading comprehension）
  - Baseline模型：Llama-like SwiGLU FFN、PKV（Parametric KV attention替换FFN）、Dense-MoE（H=1）、Naïve MH-FFN

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明代码将公开于 https://anonymous.4open.science/r/FlashMHF-9395（当前为匿名审阅提交状态，403 Forbidden）。出版后将在该URL开源I/O-aware kernel实现、模型配置和训练脚本。

  算法pipeline——FlashMHF forward pass（以H=8 heads, E=7 sub-networks, d_h=128, d_e≈342为例）：
  
  ```
  输入: X ∈ R^{L × d_model}（如1024×1024）
  参数: W_in ∈ R^{d_model × d_model}, W_out ∈ R^{d_model × d_model}
        For h=1..H, e=1..E: K_e^h, U_e^h, V_e^h ∈ R^{d_e × d_h}
        For h=1..H: W^h ∈ R^{d_h × E}
  
  Step 1 - Head-wise split & projection:
    Q = split_H(X · W_in) ∈ R^{L × H × d_h}
    # d_model=1024, H=8, d_h=128
    # Q[:,h,:] 是第h个head的query (L×128)
  
  Step 2 - Per-head gating:
    For h = 1..H:
      P^h = Q[:,h,:] · W^h ∈ R^{L × E}          # 每token E个sub-network的logits
      R^h[:,e] = σ(P^h[:,e]) / (Σ_{e'} σ(P^h[:,e']) + ε)  # sigmoid归一化gating weights
  
  Step 3 - Per-head sub-network aggregation:
    For h = 1..H:
      S_h = 0 ∈ R^{L × d_h}
      For e = 1..E:                                # E=7个并行sub-network
        # 每个sub-network内做SwiGLU-style key-value计算:
        # FFÑ(Q_h; K_e^h, U_e^h, V_e^h) = (SiLU(Q_h · K_e^{hT}) ⊙ (Q_h · U_e^{hT})) · V_e^h
        gate = SiLU(Q[:,h,:] · K_e^{hT})         # ∈ R^{L × d_e}
        up   = Q[:,h,:] · U_e^{hT}               # ∈ R^{L × d_e}
        out  = (gate ⊙ up) · V_e^h                # ∈ R^{L × d_h}
        S_h += R^h[:,e:e+1] ⊙ out                 # gated aggregation
      # S_h ∈ R^{L × d_h}, 平衡的d_e ≈ 8/3·d_h ≈ 342
  
  Step 4 - Head concat & output projection:
    O = concat_H([S_1, S_2, ..., S_H]) · W_out ∈ R^{L × d_model}
  ```
  
  关键设计参数：d_e ≈ 8/3·d_h（维持SwiGLU ratio），E = round(d_ff / d_e)。128M: H=6, E=8; 370M: H=8, E=7; 1.3B: H=16, E=15。
  
  与baseline对比：标准SwiGLU FFN = (SiLU(X·W_gate) ⊙ (X·W_up)) · W_down，单个大中间激活(R^{L×d_ff})。FlashMHF将其分解为H×E个更小的sub-network计算，每sub-network仅需要d_e维中间激活。
