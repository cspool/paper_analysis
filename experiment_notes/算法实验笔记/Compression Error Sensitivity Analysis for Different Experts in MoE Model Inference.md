## Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是对 MoE 推理中不同专家（expert）进行压缩误差敏感性分析，提出使用 error-bounded lossy compression（如 SZ3、CuSZp）来压缩非激活专家以减少 PCIe offloading 开销。核心实验方法：随机生成服从正态分布 $N \sim (0, \hat{e})$ 的误差注入 expert 参数，模拟 SZ3/CuSZp 压缩引入的有界误差。从七个维度全面分析压缩误差对 MoE 推理精度的影响：
  - ① 单层单个 expert 注入误差（含低激活频率 expert 和最高频 expert）
  - ② 单个层最高频 expert 注入不同大小误差
  - ③ 不同层（L1/L13/L20/L26）最高频 expert 注入误差
  - ④ 单个层 Top-K（6个）最高频 expert 同时注入误差
  - ⑤ 单个层全部 64 个 expert 注入误差
  - ⑥ 一组层（每10层一组）最高频 expert 注入误差
  - ⑦ 不同 benchmark（GSM8K、Math dataset）上的泛化评估

  实验比较：Baseline（无误差）vs 不同 error bound（30%/50%/80% × 平均 L1 范数）下的 ICA（Instruction Compliance Accuracy）和 PIA（Pure Inference Accuracy），分析各层 expert 对误差的敏感性差异。

- 硬件平台是什么，配置是什么。
  论文为分析性研究，使用 error injection 模拟压缩误差，论文未明确说明具体 GPU/CPU 型号等硬件配置。

- 模型是什么。数据集和bench分别是什么。
  模型：**Moonlight**（16B 参数 MoE 模型），26 个 expert layer，每层 64 个 expert 子模块，top-6 routing。每次推理每层激活 6 个 expert。
  数据集和 Benchmarks：
  - **GSM8K**：数学文字推理题，主要 benchmark
  - **Math dataset**（Hendrycks et al. 2021）：更难的数学数据集，用于泛化验证
  评估指标：ICA（Instruction Compliance Accuracy，指令合规精度）和 PIA（Pure Inference Accuracy，纯推理精度，忽略格式要求）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供代码开源链接，SC'25 workshop 论文。以下基于论文描述的方法伪代码：

  ```
  输入: MoE模型权重 θ = {θ_{l,e} | l=1..L, e=1..E}, 数据集 D, error_bound_ratio r (e.g., 30%/50%/80%)
  输出: ICA, PIA for each perturbation scenario

  For each expert (l, e) in target set:
      # 计算该 expert 的平均 L1 范数作为误差基准
      avg_L1 = ||θ_{l,e}||_1 / n_{l,e}

      # 设置 error bound
      ê = r * avg_L1

      # 生成服从 N(0, ê) 的正态分布误差
      noise = Normal(mean=0, std=ê).sample(n_{l,e})

      # 将误差注入 expert 参数（模拟有损压缩-解压后的参数）
      θ'_{l,e} = θ_{l,e} + noise

  # 在注入误差的模型上推理
  For each sample x in D:
      For each token t in x:
          For each MoE layer l:
              # Router 选择 top-k experts
              weights, expert_ids = Router(token_embedding, k=6)
              # 计算加权输出（使用已注入误差的 expert 参数）
              output = Σ w_i * Expert_i(token_embedding; θ'_{l,expert_i})
          # 生成下一个 token
      # 计算 ICA: 检查输出格式（如 boxed{}）和内容正确性
      # 计算 PIA: 仅检查内容正确性
  ```

  ## Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是对 MoE 模型的统一压缩框架，包含两大视角：(1) **Expert Trimming**：通过结构化移除专家/层/块来压缩 MoE——Expert Drop（按重要性评分移除不重要专家，含 layer-wise 和 global 两种策略）、Layer Drop（基于 cosine similarity 移除整个 MoE 层和对应的 Norm 模块）、Block Drop（进一步移除包括 Attention 在内的整个 Transformer block）；(2) **Expert Slimming**：压缩单个专家内部权重——Pruning（Wanda / SparseGPT, unstructured 50% 和 2:4 semi-structured sparsity）和 Quantization（GPTQ / AWQ, 4-bit）；(3) **集成策略**：先 Expert Slimming 后 Expert Trimming（"S+T" order），结合 AWQ 量化 + Block Drop 在 Mixtral-8×7B 上实现 6.05× speedup + 77.1% 内存节省（20GB），维持 >92% 性能；(4) **Post-Finetuning**：在 Alpaca-GPT4 数据集上 full-finetune 3 epochs (lr=8e-6, cosine schedule, warmup ratio=0.03, global batch=32)，Block Drop 后模型性能差距从 5.5% 缩小至 0.6%。实验比较：(a) Expert Drop vs Layer Drop vs Block Drop 在不同压缩率下的 benchmark 性能和 speedup/memory；(b) Pruning (Wanda/SparseGPT, 50%/2:4) vs Quantization (GPTQ/AWQ, 4-bit) 的性能与效率对比；(c) Expert Trimming + Expert Slimming 集成在不同组合下的综合对比；(d) 压缩后 Post-Finetuning 的性能恢复能力；(e) MoE vs Dense 模型的冗余度对比（同深度 Mixtral-8×7B vs Mistral-7B 在 Layer/Block Drop 下的性能衰减差异）。

- 硬件平台是什么，配置是什么。
  NVIDIA GPU（论文提及部署目标为 NVIDIA RTX 3090 GPU）。AWQ 量化 speedup 在 5.08× (Mixtral-8×7B) 和 3.16× (DeepSeek-MoE-16B)。FLOPs、Memory 和 Speedup 通过 forward pass on input sequence of length 2,048 测量。论文未明确说明 eval 使用的 GPU 型号和数量。

- 模型是什么。数据集和bench分别是什么。
  模型：**Mixtral-8×7B**（32 层，8 experts/layer, top-2 routing, 47B total/13B activated, 87.7GB FP16 memory）和 **DeepSeek-MoE-16B**（28 层 / 27 MoE layers, 2 shared experts + 64 routed experts, top-6 routing, 30.8GB FP16 memory, 首个 block 使用 dense FFN）。压缩校准数据：C4 数据集 128 samples × seq_len=2048（用于相似度计算和 pruning 校准）；量化校准：128 samples from Alpaca (GPTQ) 和 Pile (AWQ)；GPTQ group_size=128 (Mixtral) / 64 (DeepSeek)；Post-finetuning：Alpaca-GPT4 3 epochs。Benchmarks（EleutherAI LM Harness）：ARC-C, BoolQ, HellaSwag, MMLU, OBQA, PIQA, RTE, WinoGrande（全部 zero-shot normalized accuracy）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源，代码发布于 https://github.com/CASE-Lab-UMD/Unified-MoE-Compression。以下为核心算法流程：

  **Layer Drop 和 Block Drop 的相似度度量（用于选择要移除的层/块）**：
  ```
  # S^{(M)}: MoE input-output cosine similarity
  # S^{(NM)}: Norm+MoE+residual 的整体相似度（论文采用此度量）
  # x: MoE input, y = MoE(x): MoE output
  # x': 残差连接前的输入 (block input)
  
  S^{(M)} = cos_sim(x, y)  # = (x·y) / (||x|| * ||y||)
  
  y' = x' + MoE(Norm(x'))  # 含 Norm + MoE + residual
  S^{(NM)} = cos_sim(x', y')
  
  # 对每个 layer/block，用 128 个 C4 样本计算平均 S^{(NM)}
  # 按 S^{(NM)} 从高到低排序（相似度越高 → 冗余越大）
  # 依次移除高冗余度的层/块
  ```

  **Expert Drop 重要性评分与移除**：
  ```
  # 专家重要性评分 S(E_i) = 批数据上的平均路由分
  S(E_i) = (1/|X|) * Σ_{x∈X} G_i(x)  # G_i(x) 为 router 对 expert i 的输出
  
  # Layer-wise dropping: 每层保留相同数量专家
  T'(l) = {E_t^(l)} where S(E_t^(l)) ∈ TopK({S(E_i^(l))}_{i=1..n}, n')
  
  # Global dropping: 全局跨层保留 Top experts
  T'(l) = {E_t^(l)} where S(E_t^(l)) ∈ TopK(∪_{j=1..L}{S(E_i^(j))}_{i=1..n}, n'*L)
  ```

  **统一压缩框架（Expert Trimming + Expert Slimming）**：
  ```
  # 通用形式: y = Σ_{i∈T'} G_i · E_i(x | f(W_i))
  # T': 保留的专家子集 (Expert Trimming)
  # f(W_i): 压缩后的专家权重 (Expert Slimming)
  
  # Expert Slimming → Expert Trimming 顺序 ("S+T"):
  # Step 1: 对所有 expert 应用 AWQ 4-bit 量化
  for each expert i in all experts:
      W_i_quant = AWQ_quantize(W_i, bits=4, calib=Pile_128samples)
  
  # Step 2: 基于量化后模型计算相似度，执行 Layer/Block Drop
  for each layer/block l:
      S_l = mean(cos_sim(x', x' + MoE_quant(Norm(x'))))  # 量化后计算
  sort layers by S_l descending
  remove top K layers/blocks  # Layer Drop / Block Drop
  
  # Step 3 (可选): Post-finetuning on Alpaca-GPT4
  for epoch in 1..3:
      for batch in Alpaca-GPT4 (global_bsz=32):
          loss = CrossEntropy(model_compressed(x), y)
          loss.backward()
          optimizer.step()  # Adam, lr=8e-6, cosine schedule
  ```

  **关键实验结果（Mixtral-8×7B, 综合最佳配置）**：
  ```
  # AWQ only:            5.08× speedup, 24.4GB, Avg=70.8 (vs 71.5 baseline)
  # AWQ + L8/32:         6.05× speedup, 20.0GB, Avg=66.1 (92.4% of baseline)
  # AWQ + B5/32:         5.94× speedup, 21.9GB, Avg=68.0 (95.1% of baseline)
  ```
