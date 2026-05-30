## Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **Context-Aware Mixed-Precision Quantization for NDP**，包含两个核心算法：
  
  **Expert Bitwidth Selector**：基于 prefill 阶段收集的 expert 激活频率 $P_{l,e}$ 和路由评分 $W_{l,e}$ 计算重要性分数 $S_{l,e} = \alpha \widetilde{P}_{l,e} + (1-\alpha) \widetilde{W}_{l,e}$。对 NDP-resident experts，离线预计算 1/2/3/4-bit GPTQ 量化版本和 per-bitwidth 量化损失表 $L_{l,e}(b)$（MSE vs FP16 reference）。运行时使用 prefix-structured mixed-precision allocation：按重要性降序排列 experts，枚举 $(n_4, n_3, n_2, n_1)$ 分配方案满足平均 bitwidth budget $\bar{b}$，最大化累积增益 $G(n_4,n_3,n_2) = C_4(n_4) + [C_3(n_4+n_3)-C_3(n_4)] + [C_2(n_4+n_3+n_2)-C_2(n_4+n_3)]$。时间复杂度 $O(LE_{\text{NDP}}^2)$。
  
  **Expert Placement Module**：同样基于 $S_{l,e}$ 选择 top-K experts 以 FP16 驻留 GPU HBM，其余 experts 分配至 CXL-NDP 执行。Placement 仅在 prefill 后执行一次，decoding 阶段不变。

  实验比较：
  - **Accuracy**：Ours-3bit/Ours-2bit vs Original (MoNDE, FP16 lossless) vs w/o Expert Bitwidth Selector variant，在 MMLU/MathQA/HellaSwag/ARC-E/ARC-C/BoolQ/WinoGrande/PIQA 八个 benchmark 上。Ours-3bit 仅 0.13% 平均精度下降，Ours-2bit 仅 3.4% 下降。
  - **Ablation**：w/ vs w/o Expert Bitwidth Selector → Ours-2bit 带 selector 比不带 selector 高 3.2% 平均精度，验证 context-aware bitwidth 选择的有效性。
  - **Performance**：Ours-3bit vs Ours-2bit vs MoNDE (same GPU-NDP) vs Hobbit (GPU-only mixed-precision offloading)，end-to-end latency 和 decoding throughput。

- 硬件平台是什么，配置是什么。
  系统：1× NVIDIA H100 GPU (132 SM, 989.4 TFLOP/s, 80GB HBM3) + 1× DDR-based NDP device (512 GB, 512 GB/s bandwidth, 64×(4×4) systolic arrays, 1 GHz clock)。PCIe Gen4 ×16 互联。NDP 模拟器：基于 Ramulator [19] 构建。

- 模型是什么。数据集和bench分别是什么。
  模型：**Mixtral-8×7B**（32 layers, 8 experts/layer, top-2, 46.7B params）和 **Mixtral-8×22B**（56 layers, 8 experts/layer, top-2, 140.6B params）。GPU-side：Mixtral-8×7B 每层 4 experts GPU + 4 NDP；Mixtral-8×22B 每层 2 experts GPU + 6 NDP。
  数据集：C4 (1024 samples for calibration), WikiText-2, TruthfulQA (activation analysis)。
  Benchmarks：MMLU (5-shot), MathQA, HellaSwag, ARC-Easy, ARC-Challenge, BoolQ, WinoGrande, PIQA (zero-shot)，使用 EleutherAI LM Evaluation Harness [10] 评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确说明代码开源。基础量化方法使用 GPTQ [9]，NDP 模拟器基于 Ramulator [19] (https://github.com/CMU-SAFARI/ramulator)。

  **Context-Aware Expert Bitwidth Selector 算法 Pipeline**：

  ```
  # === 离线阶段 (Offline Calibration) ===
  # 对每层 l 和每个 expert e，预计算 1/2/3/4-bit 量化损失
  for l in 1..L:
      for e in 1..E:
          for b in {1, 2, 3, 4}:
              W_q = GPTQ_quantize(W_{l,e}, bits=b, calib_data=D_cal)
              L_{l,e}(b) = MSE(W_q(x_calib), W_fp16(x_calib))
  
  # === 在线阶段 (Online Inference) ===
  # Step 1: Prefill 统计收集
  def prefill_with_stats(x_seq):
      for each MoE layer l:
          for each expert e:
              P_{l,e} = count(tokens routed to expert e)  # 激活频率
              W_{l,e} = sum(routing_scores for expert e)   # 累计路由分
      return {P, W}
  
  # Step 2: 重要性评分计算
  def compute_importance(P, W, alpha=0.5):
      for l in 1..L:
          P_tilde = P_{l,:} / sum(P_{l,:})     # 归一化
          W_tilde = W_{l,:} / sum(W_{l,:})
          S_{l,e} = alpha * P_tilde[e] + (1-alpha) * W_tilde[e]
      return S  # [L, E] importance scores
  
  # Step 3: Expert Placement
  def place_experts(S, K):
      for l in 1..L:
          sorted_experts = argsort(S[l,:], descending=True)
          H_l = sorted_experts[:K]    # GPU: FP16, hot experts
          C_l = sorted_experts[K:]    # NDP: quantized, cold experts
      return H, C
  
  # Step 4: Prefix-Structured Bitwidth Allocation
  def prefix_split_bitwidth(S_ndp, loss_table, b_bar):
      for l in 1..L:
          E_ndp = len(C_l)
          R = E_ndp * (b_bar - 1)  # bitwidth increment budget
          
          # 按重要性降序排列 NDP experts
          idx = argsort(S_ndp[l,:], descending=True)  # i=1..E_ndp
          
          # 预计算 prefix sums
          delta_2[i] = L_i(1) - L_i(2)
          delta_3[i] = L_i(1) - L_i(3)  
          delta_4[i] = L_i(1) - L_i(4)
          
          C_2(k) = sum_{i=1..k} delta_2[i]  # prefix sum
          C_3(k) = sum_{i=1..k} delta_3[i]
          C_4(k) = sum_{i=1..k} delta_4[i]
          
          # 枚举最优 (n4, n3, n2) 满足预算约束
          best_gain = -inf
          for n4 in 0..E_ndp where 3*n4 <= R:
              for n3 in 0..(E_ndp-n4) where 3*n4+2*n3 <= R:
                  n2 = R - 3*n4 - 2*n3
                  if n2 < 0 or n4+n3+n2 > E_ndp: continue
                  n1 = E_ndp - n4 - n3 - n2
                  
                  # 前缀结构: 最重要的 n4→4bit, 其次 n3→3bit, n2→2bit, n1→1bit
                  gain = C_4(n4) + (C_3(n4+n3)-C_3(n4)) + (C_2(n4+n3+n2)-C_2(n4+n3))
                  if gain > best_gain:
                      best_gain = gain
                      best_assignment = (n4, n3, n2, n1)
          
          # 分配 bitwidth: top-n4→4bit, next-n3→3bit, next-n2→2bit, rest→1bit
          b_{l, idx[0:n4]} = 4
          b_{l, idx[n4:n4+n3]} = 3
          b_{l, idx[n4+n3:n4+n3+n2]} = 2
          b_{l, idx[n4+n3+n2:]} = 1
      return b  # per-expert bitwidth for decoding
  
  # Step 5: Decoding with fixed placement + quantization
  def decoding_step(token, H, C, b):
      for l in 1..L:
          experts = router(token)
          for e in experts:
              if e in H_l:
                  out += expert_fp16_gpu(e, token)    # GPU FP16
              else:
                  out += expert_quant_ndp(e, token, b_{l,e})  # NDP with assigned bits
      return out
  ```

  **关键张量计算流程**（以 Mixtral-8×7B, K=4 GPU experts/layer, b_bar=3 为例）：
  - Prefill: [batch, seq_len, 4096] tokens → 32 MoE layers → 每层收集 8 experts 的 (P_{l,e}, W_{l,e}) → 计算 S_{l,1..8}
  - Placement: Top-4 by S → GPU FP16; Bottom-4 → NDP
  - Bitwidth: 4 NDP experts, b_bar=3 → R=4×(3-1)=8 increments → enumerate (n4,n3,n2) → e.g. (2,2,0,0): 最重要2个4bit, 其次2个3bit
  - Decoding: 每个 token 激活 top-2 experts → 若两者均在 GPU → 全 FP16；若一 GPU 一 NDP → GPU FP16 + NDP 3/4-bit

Error-bounded lossy compression pipeline（SZ3/CuSZp风格）：
  数据 → 预测（Lorenzo predictor/linear regression）→ 量化（基于 error bound ê 控制量化步长）→ 编码（Huffman/变长编码）→ 压缩数据。解压时逆过程：解码 → 反量化 → 反预测。关键特性：所有重建值与原值的绝对误差 ≤ ê（有界保证），压缩比由 ê 和数据分布决定。在 MoE offloading 场景中，expert 参数在传输前压缩（减少 PCIe 数据量），GPU 端解压后用于推理，参数中含 bounded error。本论文的 error injection 实验模拟了这一流程中解压后的参数状态。
