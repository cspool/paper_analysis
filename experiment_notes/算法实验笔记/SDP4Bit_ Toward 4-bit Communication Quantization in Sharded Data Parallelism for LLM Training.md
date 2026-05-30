## SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出两种新颖的通信量化技术来压缩 ShardedDP 中的权值和梯度通信——(1) **Quantization on Weight Differences (qWD)**：不直接量化权值，而是对当前迭代与前次迭代间的权值差值做 INT4 量化，利用差值分布更均匀且范围更小的特性降低量化误差；(2) **Two-Level Gradient Smooth Quantization (TLq-HS)**：对 intra-node 梯度 all-to-all 通信使用 INT8 量化（降低误差），对 inter-node all-to-all 通信使用 INT4 量化（大幅压缩带宽），并在量化前施加 Hadamard Transform 平滑梯度中的 outlier。
  - 实验比较：Baseline（BF16/FP32 混合精度 Megatron-LM 全精度训练）vs ZeRO++ 类似策略（直接 4-bit 量化权值 qW + 两级均 4-bit 量化梯度 ULq）vs qWD 单独 vs TLq 单独 vs TLq-HS 单独 vs SDP4Bit（qWD + TLq-HS 组合），测量 validation loss（准确率）和 E2E TFLOPs throughput（加速比）。

- 硬件平台是什么，配置是什么。
  - **平台1**：16 节点，每节点 4× NVIDIA A100-SXM4-40GB，100 Gbps Slingshot10 互联（低带宽 inter-node）。
  - **平台2**：16 节点，每节点 8× NVIDIA H800-SXM5-80GB，8 条 InfiniBand 链路共 3.2 Tbps（高带宽 inter-node）。
  - 最大规模 128 GPUs。

- 模型是什么。数据集和bench分别是什么。
  - 模型：GPT 系列（125M, 350M, 1.3B, 2.7B, 6.7B, 13B, 18B 参数），配置详见 Table 7（hidden size 768→6144, layers 12→40）。
  - 数据集：The Pile（800GB），每轮 80,000 iterations（处理超 40B tokens），验证 loss 使用 The Pile 验证集。
  - Benchmark：E2E validation loss（准确率指标）和 E2E TFLOPs throughput（加速比指标），通过 Megatron-LM 内置 loss logging 和 throughput timer 收集。Wall-clock time vs. loss 曲线也作为综合指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源仓库：https://github.com/hanlin-lu/SDP4Bit（Apache-2.0），基于 Megatron-LM 实现，亦有 ByteDance-Seed 官方 fork：https://github.com/ByteDance-Seed/SDP4Bit。
  - 算法pipeline详细说明：

  **1) Quantization on Weight Differences (qWD)**：
  每轮迭代中，每个 GPU 持有完整模型权重 `w_model`（BF16）和分片的 main weights `w_main[p]`（FP32，用于优化器状态）。优化器更新 main weights 后：
  ```
  # 计算权值差值（FP32 -> BF16）
  d[p] = w_main[p] - w_model[p]

  # INT4 对称线性量化（group-wise, group_size=2048）
  for each group of 2048 elements in d[p]:
      s = max(abs(group))           # scale factor
      d_q = round(clip(group, -s, s) / s * 7)  # map to {-7, ..., +7}
      store(s, d_q)

  # AllGather 量化后的差值（通信量 ≈ 4 bit/elem）
  d_q_global = AllGather(d_q[p])

  # 反量化并更新模型权值
  for each group:
      d_deq = d_q_global * s / 7
      w_model = w_model + d_deq
  ```
  关键优势：(a) 权值差值的数值范围比权值本身小得多（`||δw|| < ||w||`），INT4 量化误差更小——论文通过直方图（Fig. 4）展示差值分布更均匀且范围更窄；(b) 理论保证：weight difference 量化兼容 biased compressor（如 top-k sparsifier），而直接量化权值时使用 biased compressor 会导致收敛失败（Counterexample 4.1 证明 ternary quantizer 直接量化权重时 SGD 卡在初始值不动）。

  **2) Two-Level Gradient Smooth Quantization (TLq-HS)**：
  梯度同步采用两次 all-to-all 替代传统的 reduce-scatter（沿用 ZeRO++ 的通信模式），但使用两级精度 + Hadamard 平滑：
  ```
  # Step 1: Hadamard Transform 平滑 outlier（32x32 矩阵在线旋转）
  g_hat = H @ grad @ H.T   # H 是 32x32 Walsh-Hadamard matrix

  # Step 2: INT8 量化 → Intra-node AlltoAll → 反量化 → 局部 reduce
  qg_8bit = round(clip(g_hat, -s8, s8) / s8 * 127)  # INT8, group_size=512
  list_qg8 = IntraAlltoAll(qg_8bit)   # 仅节点内通信（NVLink/NVSwitch 高带宽）
  # 反量化后做 local reduce（省略 Hadamard 逆向，因 H·H=I 自动抵消）
  g_local_reduced = sum([dequantize(x) for x in list_qg8])

  # Step 3: Hadamard → INT4 量化 → Inter-node AlltoAll → 反量化 → 最终 reduce
  g_hat_reduced = H @ g_local_reduced @ H.T
  qg_4bit = round(clip(g_hat_reduced, -s4, s4) / s4 * 7)  # INT4, group_size=128
  list_qg4 = InterAlltoAll(qg_4bit)   # 跨节点通信（InfiniBand/Slingshot 低带宽）
  g_reduced = sum([dequantize(x) for x in list_qg4])
  g_final = H @ g_reduced @ H.T        # 最终逆变换恢复原始梯度
  ```
  优化技巧（Section 3.3）：
  - 利用 `H·H=I` 在 Step 2 省略 Hadamard 逆向（intra-node dequant 后无需再 transform）
  - 利用 `Σ H·g_i = H·Σ g_i` 将 inter-node dequant 后的 Hadamard 移到最终 reduction 之后，将 transform 次数从 3 降低到 2
  - 将 Hadamard + quantization/dequantization 融合为单个 CUDA kernel

  **3) 训练运行时优化（Section 3.3）**：
  - **Buffer reuse**：Megatron-LM 维持完整 model weights，无需额外 buffer 存储历史权重用于差值计算
  - **Hadamard kernel fusion**：Hadamard transform 与 (de)quantization 融合为单个 CUDA kernel，利用 shared memory 局部性将 overhead 降低到近乎零

  **训练命令示例（GPT-1.3B on 32 A100）**：
  ```bash
  python pretrain_gpt.py \
    --num-layers 24 --hidden-size 2048 --num-attention-heads 16 \
    --seq-length 2048 --micro-batch-size 2 --global-batch-size 256 \
    --train-iters 80000 --lr 2e-4 --min-lr 2e-5 \
    --lr-decay-style cosine --lr-warmup-iters 2000 \
    --optimizer adam --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --fp16 --use-distributed-optimizer \
    --quantized-weights --weight-quantization-bits 4 --wq-group-size 2048 \
    --quantized-gradients --gradient-quantization-bits-intra 8 --gq-group-size-intra 512 \
    --gradient-quantization-bits-inter 4 --gq-group-size-inter 128 \
    --hadamard-transform --gradient-alltoall-pipeline 4 \
    --no-async-tensor-model-parallel-allreduce
  ```

  **核心结果**：
  - GPT-6.7B validation loss 与全精度 baseline 几乎重合（Fig. 1），最大 loss 增加仅 0.24%
  - GPT-18B on 128 H800: 4.08× E2E throughput speedup（59.2 vs 14.5 TFLOPs）
  - 低带宽网络下加速比更大：6.7B on A100 Slingshot 达 3.44×（37.1 vs 10.8 TFLOPs）
  - 收敛保证：Theorem 4.1 证明达到与标准 SGD 相同的 O(1/√T) 收敛率，且放宽了 QSDP 对 Polyak-Łojasiewicz 条件和对特定 quantizer 的依赖
