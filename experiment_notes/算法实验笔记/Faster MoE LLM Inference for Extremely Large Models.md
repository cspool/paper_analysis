## Faster MoE LLM Inference for Extremely Large Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文对 fine-grained MoE（DeepSeek-V2-Lite / DeepSeek-V3）提出两种推理阶段的算法优化策略：
    1. **Inference Time Expert Skipping（Section 5）**：在推理时按层级别动态减少每 token 的激活 expert 数 na。通过定义一个四元组 (b, h, e, p) 来描述跨层的 expert 分配——首层选 b 个、第 p 层选 h 个、末层选 e 个、其余层线性插值——实现 ascending/descending/peak/valley 等多种分配模式。探索了 softmax 激活（V2）和 sigmoid 激活（V3）两种路由机制下的不同行为。
    2. **Pre-Inference Expert Pruning（Section 6）**：在推理开始前减少总 expert 数 ne，通过多种选择策略（Random、Structured/Odd-Even-FirstHalf-LastHalf、Activate Count、Soft Count）从 ne 个 expert 中选择 ne' 个保留，其余丢弃不加载。
    3. **Roofline-based 效率分析（Section 4）**：从 Roofline 模型出发，推导 MoE 层的 I/O、FLOPS 和算术强度公式，分析 MoE 相比 FFN 的批次效应弱化原因——因 token 间很少复用同一 expert，增加 token 数反而增加额外 expert 参数加载开销。
  - 实验比较：
    - Expert skipping 效率：不同 na（2-8）在不同并发度（2-768）下的 throughput 和 speedup ratio
    - Expert skipping 性能：不同 (b,h,e,p) 四元组策略在 ARC-C, ARC-E, BoolQ, OBQA, RTE, WinoGrande 上的 Avg 得分
    - Expert pruning 效率：不同 ne（8-64）在不同并发度（2-784）下的 throughput 和 speedup
    - Expert pruning 性能：不同选择策略（Random/Structured/Activate Count/Soft Count）在不同 ne' (16/32/48) 下的 benchmark 得分

- 硬件平台是什么，配置是什么。
  - **DeepSeek-V2-Lite**：2× NVIDIA Tesla A800 80G PCI-e, Intel Xeon Silver 4314 CPU @ 2.40GHz
  - **DeepSeek-V3**：8× NVIDIA Tesla H200 141G SXM5, Intel Xeon Platinum 8558 CPU @ 2.10GHz
  - 效率测试固定 1024 input + 1024 output tokens，使用 sglang v0.4.4 post 1 + sglang.bench

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - DeepSeek-V2-Lite: 16B params, ne=64, na=6, d=2048, de=1408, ds=10944, da=8448, softmax routing
    - DeepSeek-V3: 671B params, ne=256, na=8, d=7168, de=2048, ds=18432, da=16384, sigmoid routing
  - **Benchmark 数据集**：ARC-Challenge, ARC-Easy, BoolQ, OpenBookQA (OBQA), RTE, WinoGrande（Avg 为 6 个 benchmark 的平均，baseline=36 为纯猜测基线）
  - 效率评测无特定 benchmark 数据集，使用随机生成 token 序列

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供独立开源代码仓库。基于开源 sglang 和 PyTorch 实现。
  - **算法 pipeline 解释**：

  **Expert Skipping 算法（Section 5.2）**：

  给定模型 L 层，原始每层激活 na 个 expert。定义四元组 (b, h, e, p) 控制跨层分配：

  ```
  # 输入: b (首层 expert 数), h (第 p 层 expert 数), e (末层 expert 数), p (峰值/谷值位置)
  # 输出: 每层的 na(l) for l in 0..L-1

  na[0] = b                      # 首层
  na[p] = h                      # 第 p 层
  na[L-1] = e                    # 末层

  # 线性插值填充中间层
  for l in 1..p-1:
      na[l] = b + (h - b) * l / p

  for l in p+1..L-2:
      na[l] = h + (e - h) * (l - p) / (L - 1 - p)
  ```

  **四种典型分配模式（Figure 6）**：
  - Ascending: b < h < e（expert 逐层递增，V3 最优）
  - Descending: b > h > e（expert 逐层递减，V2-Lite 最优）
  - Peak: h > b 且 h > e（中间层最多 expert）
  - Valley: h < b 且 h < e（中间层最少 expert）

  **MoE 推理效率模型（Section 3.1, Eq. 3）**：
  ```
  # FFN/MoE 的 Memory I/O, FLOPS, Arithmetic Intensity
  I/O(d, di, L) = 3 * di * d + 2 * L * (d + di)
  FLOPS(d, di, L) = 6 * L * (di * d)
  AI(d, di, L) = (6 * L * di * d) / (3 * di * d + 2 * L * (di + d))

  # 对于 MoE, di = da = de × na (激活 expert 的总中间维度)
  # 但实际 I/O 还包括所有被选中 expert 的参数加载
  ```

  **Expert Skipping 前向计算**（以 na'=2, 原始 na=6 为例）：
  ```
  # 输入: hidden_states h ∈ R^(d), batch_size B
  # 原始: na=6, router 选 top-6 expert
  # Skipping: na'=2, router 选 top-2 expert

  # Step 1: Router gate
  r' = W_r @ h                    # R^ne, router logits

  # DeepSeek-V2-Lite: softmax
  r = softmax(F_r(r'))            # 可选 load balancing modifier
  topk_indices = topk(r, k=2)     # 原 k=6, 改为 k=2
  topk_weights = softmax(r[topk_indices])  # renormalize

  # DeepSeek-V3: sigmoid
  r = sigmoid(F_r(r'))            # 无 softmax normalization
  topk_indices = topk(r, k=2)     # 原 k=8, 改为 k=2
  topk_weights = r[topk_indices]  # 直接用 sigmoid 值, 不 renormalize

  # Step 2: Expert FFN (仅 top-k)
  out = 0
  for idx, w in zip(topk_indices, topk_weights):
      # Expert FFN = GLU
      gate = W_g[idx] @ h         # R^(de) gate projection
      up = W_u[idx] @ h           # R^(de) up projection
      act = SiLU(gate) * up
      out += w * (W_d[idx] @ act) # down projection, R^(d)

  # Step 3: Shared Expert (always activated)
  gate_s = W_g_shared @ h
  up_s = W_u_shared @ h
  out += W_d_shared @ (SiLU(gate_s) * up_s)

  return out
  ```

  **Expert Pruning 算法（Section 6.2）**：

  在推理前从 ne 个 expert 中选择 ne' 个保留：

  ```
  # Soft Count 方法 (最佳方法):
  # 1. 在 calibration 数据上运行 forward pass
  expert_activation_count = zeros(ne, L)  # 记录每层每个 expert 被激活次数

  for batch in calibration_data:
      for layer in 0..L-1:
          gate_logits = router[layer](hidden_states)
          topk_idx = topk(gate_logits, k=na)
          expert_activation_count[layer][topk_idx] += 1

  # 2. 按激活次数排序, 选 top-ne' experts per layer
  for layer in 0..L-1:
      sorted_experts = argsort(expert_activation_count[layer], descending=True)
      selected[layer] = sorted_experts[:ne']  # 仅保留 ne' 个最活跃 expert

  # 3. 推理时仅加载 selected experts, 其余无视
  ```

  **V2 vs V3 行为差异（Section 5.2）**：
  - **V2 (softmax)**：低排名 expert 权重显著小于 top-1 expert → expert skipping 性能退化更平滑（descending 策略最优）
  - **V3 (sigmoid)**：expert 权重极化（趋于 0 或 1）→ 跳过权重接近 1 的 expert 会导致显著性能下降（ascending 策略最优）
  - 结论：不存在 universal skipping strategy，策略与模型强相关

  **关键性能数据**：
  - Expert skipping: na 从 6→2, V2-Lite 性能下降仅 7.5%（best 6%）；na 平均 3.3 时下降 <1%
  - V3: best method 可提高 throughput ≥10% 且零性能退化
  - Expert pruning (ne 64→48): best method (soft count) Avg 64.2 vs baseline 66.0（−2.7%）
  - Expert pruning (ne 64→32): best method Avg 57.8（−12.4%）
  - Expert pruning (ne 64→16): best method Avg 47.8（−27.6%），随机选择几乎丧失语言能力
