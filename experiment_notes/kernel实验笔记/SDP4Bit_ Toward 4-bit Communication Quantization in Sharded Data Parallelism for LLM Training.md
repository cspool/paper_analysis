## SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：(1) **Hadamard + Quantization/Dequantization CUDA Kernel Fusion**：将 Hadamard transform 与对称线性 (de)quantization 操作融合为单个 CUDA kernel，利用 shared memory 局部性使融合 kernel 运行速度与单独的量化操作几乎相同；(2) **Buffer Reuse**：利用 Megatron-LM 维持完整 model weights 的特性，复用 model weights buffer 用于权值差值计算，消除额外内存分配；(3) **Operation Pruning**：利用 Hadamard 正交性（H·H=I）和分配律（Σ Hg = HΣg）裁剪冗余的 Hadamard transform，将每轮迭代中的 transform 次数从 6 次（naive）减少到 2 次；(4) **梯度 AlltoAll Pipeline**：将梯度分 chunk 在 intra-node 和 inter-node all-to-all 之间流水线化，利用二者使用不同网络带宽（NVLink vs InfiniBand）的特性实现通信重叠。
  - 实验比较：w/ vs w/o fused Hadamard kernel 的 (de)quantization throughput（Table 5，GB/s）；w/ vs w/o fused Hadamard kernel 的 E2E gradient communication time 和 TFLOPs（Table 4）；TLq-HS vs ULq 的 grad communication time（Table 4）；不同输入大小下的 (de)quantization throughput（Table 5：8MB→2048MB）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100-SXM4-40GB（per-node 4×, NVLink intra-node + 100Gbps Slingshot10 inter-node）和 NVIDIA H800-SXM5-80GB（per-node 8×, NVLink + NVSwitch intra-node + 3.2Tbps InfiniBand inter-node）。
  - GPU 编程模型：CUDA kernel（自定义 Hadamard + quant kernel fusion），NCCL 用于集体通信（all-gather, all-to-all）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估框架：基于 Megatron-LM 的 `pretrain_gpt.py` 训练入口，通过在 Megatron-LM 的 distributed optimizer 和 model forward/backward 中插入量化/反量化操作来评估。
  - 修改内容：
    - `megatron/core/tensor_parallel/random.py` 等文件中增加 Hadamard transform CUDA kernel + (de)quantization CUDA kernel 的融合实现
    - 在 distributed optimizer 的 `all_gather` 和 `reduce_scatter` / `all_to_all` 调用前插入量化步骤
    - 添加权值差值计算逻辑：在 optimizer step 后计算 `w_main - w_model` 差值
    - 梯度 all-to-all pipeline 实现：将梯度分 chunk 在 intra/inter all-to-all 之间流水线化
  - 评估原理：warm-up 20 iterations 后进行 10 iterations 的 E2E throughput 测量（TFLOPs），单独记录 gradient communication time（ms）。Hadamard kernel 融合效果通过 (de)quantization throughput（GB/s）评估：对 8MB→2048MB 的数据进行 quantize/dequantize，测量带/不带 Hadamard 的 throughpout。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源仓库：https://github.com/hanlin-lu/SDP4Bit（Apache-2.0），核心 CUDA kernel 实现在 `megatron/core/tensor_parallel/` 目录下。
  - 评估流程详解：
    **Hadamard Kernel Fusion 评估**（Table 5）：
    ```
    # 输入：FP32/BF16 梯度张量 grad (size: 8MB ~ 2048MB)
    # 输出：INT4 量化数据 + scale factors
    
    # 无 Hadamard fusion (naive):
    # Step 1: Quantize kernel reads grad from global memory → compute scale per group → write INT4 to output
    # Step 2: (separate kernel) Hadamard kernel reads INT4 data → does H·x·H^T → writes back

    # 有 Hadamard fusion (SDP4Bit):
    # Single fused kernel:
    for each thread block (handling one group of 32×32 elements):
        # Load from global memory into shared memory (1 read)
        data_smem = load_global(grad[block_offsets])
        # Hadamard transform in shared memory (memory-bound at 32×32)
        data_smem = H_32 @ data_smem @ H_32.T
        # Quantize in shared memory (no extra global memory traffic)
        s = max(abs(data_smem))
        qdata = round(clip(data_smem, -s, s) / s * 7)
        # Write INT4 output to global memory (1 write)
        store_global(output[block_offsets], qdata, s)
    ```
    关键设计：必须保证 `group_size` 能被 Hadamard matrix size 整除，使得内存在 kernel block 内部保持局部性。论文选择 H=32×32，因为此时 transform 在 GPU 上是 memory-bound，几乎无计算开销，且 32×32 足以平滑梯度 outlier。Table 5 结果显示 w/ vs w/o Hadamard 的 throughput 差异 < 0.3%，证明融合理想。
    
    **AlltoAll Pipeline 评估**（Table 4）：
    梯度通信评估：记录从 backward pass 完成到梯度同步完成的时间。具体流程：
    ```
    # 原始梯度: Float32, collective via reduce-scatter (baseline)
    # SDP4Bit 量化梯度通信:
    t0 = timer()
    # 1. Hadamard + INT8 quantize (fused kernel)
    # 2. Intra-node all-to-all via NVLink (INT8 data)
    # 3. Local reduce
    # 4. Hadamard + INT4 quantize (fused kernel)
    # 5. Inter-node all-to-all via InfiniBand/Slingshot (INT4 data) — 与 step 2 流水线重叠
    # 6. Final reduce + Hadamard inverse
    t1 = timer()
    grad_comm_time = t1 - t0
    ```
    Table 4 结果：TLq-HS grad comm time 45.9ms vs Baseline 379.3ms（8.26× reduction）。ULq 45.0ms vs TLq-HS 45.9ms 几乎相同（因 intra-node 通信带宽高且与 inter-node 重叠）。Fused Hadamard kernel 相比于未融合版本（64.6ms），grad comm time 降低 29%。
