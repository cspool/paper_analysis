## Toward Inference-optimal Mixture-of-Expert Large Language Models

- baseline方法是什么？
  传统的 dense Transformer scaling law（Kaplan et al., 2020; Hoffmann et al., 2022）仅考虑 validation loss L(N, D) 与训练成本 C(N, D) 的关系，求解 $argmin L(N,D) \text{ s.t. } FLOPs(N,D) = C$ 得到 loss-optimal 配置。当扩展到 MoE 时，由于增加 expert 数量几乎不增加训练 FLOPs，从 loss-optimal 视角应无限扩展 expert 数量（直到饱和 E_max），但这在推理阶段会严重增加显存占用（更多 expert 参数挤占 KV-cache 可用显存），导致 batch size 下降、throughput 降低、cost per query 上升。现有 MoE scaling law（Clark et al., 2022）没有纳入训练数据量 D 的影响，无法给出具体的预算分配建议；Sardana & Frankle (2023) 虽然考虑了推理成本，但用恒定 MFU 估算，与实际 profiling 差异可达 10×。
  
  Baseline 全栈执行例子（loss-optimal 32-expert MoE，传统 scaling law 指导）：
  - **算法Pipeline层**：遵循 Hoffmann et al. 的 dense scaling law → 确定 loss-optimal (N_opt, D_opt) → 固定到所有 MoE 变体，不考虑 E 的影响 → 如果从训练成本看 E 越大越好，选择 E=32 → 推理时模型总参数 N_MoE = (1 + (32-1)*1/3) ≈ 11.33× N_dense，所有 32 个 expert 必须加载到 GPU 显存。
  - **系统框架层**：vLLM 部署 E=32 MoE → 每个 token 经 Top-2 gating 路由到 2 个 expert → 8×A100(40GB) 共 320GB 显存，但 32 个 expert 参数挤占后仅剩余小部分给 KV-cache → 最大 batch size b 极小 → throughput 低 → cost per token 高。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（使用 vLLM 默认 GEMM kernel）。
  - **硬件架构层**：8×40GB A100 GPU + NVLink。推理瓶颈：expert 参数占显存 → KV-cache 可用空间被压缩 → decode 阶段 batch size 受限（decode 是 memory-bound，batch size 直接影响 GPU 利用率）→ 更多 GPU 需求（单卡装不下）→ 更多通信开销（EP all-to-all）。

  Baseline 的核心痛点：**loss-optimal MoE 最大化 expert 数量导致推理成本不可控**，且传统 scaling law 无法提供同时考虑训练和推理的 budget allocation 建议。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法分为三步，层层推进：
  
  **Step 1: 建立包含 E 的 MoE Scaling Law（Section 3）**
  在 Kaplan/Hoffmann 的 dense scaling law 基础上，将 expert 数量 E 作为第三个独立变量纳入公式（公式 4），并引入 $E_{start}$ 和 $E_{max}$ 参数建模 expert 增长的饱和效应（与 Clark et al. 不同，本公式同时包含 N、D、E 三个变量且有 N-E 交互项）。通过在 100M-730M dense 模型上训练 4/8/16/32 expert 版本（SlimPajama 2.5B-20B tokens）拟合参数，RMSLE = 3.908e-3，Huber loss = 1.033e-3。
  
  **Step 2: 引入推理成本约束（Section 4）**
  在 8×40GB A100 + NVLink 上用 vLLM profiling 建立 MoE 模型的推理成本模型：$C_{Model,G} = GC_0 / T_{Model}(G)$。关键推导：MoE 总参数 $N_{MoE} = (1 + (E-1)c)N$（其中 c = MLP 占比 ≈ 1/3，因为每两层替换一层 MoE）。通过 profiling 发现推理成本与模型大小近似线性关系（Figure 2），从而将推理成本量化为可优化的指标。
  
  **Step 3: 提出 Over-training 策略（Section 5）**
  核心洞察：在图 3（middle）中，给定训练预算下，模型性能对模型大小的变化在 loss-optimal 附近相当"平坦"（loss 对 N 不敏感），但推理成本随 N 线性增长。因此文章提出**刻意训练比 loss-optimal 小很多（70-85% reduction）的模型，将节省的预算投入更多训练 token**。这种 "over-trained" 配置以微小的质量损失换取显著的推理成本降低。
  
  论文方法全栈执行例子（over-trained 16-expert MoE，以 loss-optimal 4-expert MoE 的 quality 为 target）：
  - **算法Pipeline层**：
    1. 用 scaling law（公式 4）计算 loss-optimal 4-expert 的 (N_4, D_4) 和质量 L_4_opt
    2. Algorithm 1: dichotomy search 找满足 L_16(N, B) = L_4_opt 的最小 N_16 → 仅为 N_4 的 ~15-30%（Figure 5 right）
    3. 对应推理成本 I_16_min = min_g Get_cost(N_16, E=16, g) → 仅为 I_4 的 48%-53%
    4. 节省的训练预算全投入 tokens: D_16 = B / (6 * N_16) >> D_4
  - **系统框架层**：vLLM 部署 over-trained 16-expert MoE → 模型小很多 → 更多显存留给 KV-cache → 更大 batch size → 更高 throughput → 更低 cost per token。虽然 expert 数量更多（16 vs 4），但模型本身大幅缩小更主导推理成本（因成本近似线性于 N，而非 E）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（vLLM 默认 kernel path）。
  - **硬件架构层**：8×40GB A100 + NVLink。差异：base model 因模型大可能需更多 GPU 或更大显存压力，over-trained 模型缩小后可在相同 GPU 数量下服务更多并发请求，或减少所需 GPU 数量。

  关键定量对比（方法 vs Baseline）：
  | 对比维度 | Baseline (loss-optimal) | 论文方法 (over-trained) |
  |---------|------------------------|------------------------|
  | 训练预算 | 固定 B | 相同 B |
  | 模型大小 | N_opt (loss-optimal) | 15%-30% of N_opt |
  | 训练数据量 | D_opt = B/(6N_opt) | >> D_opt (预算重分配) |
  | 推理成本 | 基准 I_base | 48%-53% of I_base (16-expert vs 4-expert) |
  | 模型质量 | L_opt | L_opt (锚定相同) 或略低 |
  | 训练成本 (同类质量) | 100% | 仅需 23.7%-42.8% FLOPs (16-expert vs 4-expert) |

  论文本质发现：**MoE 的"免费午餐"仅在训练侧成立**。要同时优化训练和推理，应在 loss-optimal 配置基础上"有意训练差一点（模型更小但数据更多）"，以推理效率换取可忽略的质量下降。这一思路颠覆了传统 scaling law 仅追求 loss-optimal 的单目标优化范式。
