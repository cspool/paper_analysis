## Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**MoE 专家权重的 weight-only 量化（INT4/INT8）**——仅对 MoE 层的 expert weight matrices 进行对称、range-based per-channel 量化，不量化 activations 和 biases。所有非 expert 参数和中间激活保持 FP16。量化后使用 fused GEMM+Dequantize kernel（将 dequantize 融合进 CUTLASS Grouped GEMM），避免单独的 dequantize kernel 引入额外内存读写。INT4/INT8 均使用相同的量化方案：对形状为 (E, M, N) 的 expert weights 生成 scales (E, 1, N)，推理时将量化权重 dequantize 回 FP16 后进行浮点矩阵乘法。针对 INT 到 FP16 转换慢的问题，提出基于 FP16 位操作（mantissa 直接编码整数 + 0x6400 减法）的快速 I2F 转换替代原生的 IntToFloat 指令。
  - 实验比较：(a) INT8/INT4 fused GEMM+Dequantize vs FP16 GEMM 在不同 active experts 数量（1/4/8/16/24/32）下的归一化吞吐量（Table 1）；(b) INT8/INT4 量化 vs FP16 baseline 的 BLEU 分数差异（Table 2, EN-DE/DE-EN/10 语言对平均）；(c) 端到端推理吞吐对比：Torch-FP16 vs FT-FP16 vs FT-INT8 vs FT-INT4 在不同 batch size（1/8/20/32/64/96）和 beam（1/2）下的每秒处理输入 tokens 数（Table 3）。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA PCIE V100，Docker 容器运行 Ubuntu 20.04 + CUDA 11.6，代码由 nvcc + gcc/g++ 9.3 编译。

- 模型是什么。数据集和bench分别是什么。
  - 模型：encoder-decoder MoE Transformer（Deep encoder, shallow decoder），embedding dim 1024，FFN hidden dim 4096，24 encoder layers，12 decoder layers，32 experts，top-1 gating（Switch Transformer 风格），TUPE attention，总参数约 5B（FP16 下约 10GB）。
  - 数据集：WMT 公开数据集，10 语言对的 multilingual machine translation，vocabulary 128K（SentencePiece tokenizer），训练数据约 4B sentence pairs。
  - Benchmark：EN-DE 和 DE-EN 翻译，1000 tokenized English sentences（约 40K tokens），metric 为 BLEU 和 throughput（input tokens/sec）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文基于 NVIDIA FasterTransformer（https://github.com/NVIDIA/FasterTransformer）开源框架实现，CUTLASS（https://github.com/NVIDIA/cutlass）、CUB（https://github.com/NVIDIA/cub）和 Triton Inference Server（https://github.com/triton-inference-server/server）均是开源项目。论文自身未提供独立开源仓库。
  - 量化算法 pipeline（从训练后模型到 INT4/INT8 推理）：
    ```
    # === Step 1: 量化 Expert 权重（训练后，离线） ===
    # 输入：expert weights W_fp16，形状 (E, M, N)，E=32 experts
    # 对每个 expert e 的每个输出 channel n：
    for e in range(E):
        for n in range(N):
            # 对称 per-channel 量化
            max_abs = max(|W_fp16[e, :, n]|)
            scale[e, n] = max_abs / max_val    # max_val = 127 (INT8) or 7 (INT4)
            W_quant[e, :, n] = round(W_fp16[e, :, n] / scale[e, n])
            W_plus = W_quant + offset           # offset = 128 (INT8) or 8 (INT4), 转为无符号
    # 输出：W_plus (E, M, N) INT4/INT8, scales (E, 1, N) FP16

    # === Step 2: 推理时 Fused GEMM + Dequantize ===
    # 对于每个 MoE layer 的 Grouped GEMM 调用：
    # 输入 activation A 通过 CUB radix sort 路由到各 expert
    # 对每个 expert e（有该 expert 的 tokens）：
    for each expert e with active tokens:
        # 在 GEMM kernel 内部 fused 执行 dequantize：
        for each weight tile:
            # 加载 INT8 权重（4个 INT8 → 1个 32-bit reg）
            w_plus_int8 = load_4_int8(W_plus[e, tile_m:tile_m+4, tile_n])
            # 构造 FP16：fp16_repr = (0x6400 | w_plus_int8[i])
            # 即 (1024 + w_plus_int8[i]) 的 FP16 表示
            fp16_val = fp16_subtract(fp16_repr, 1152.0)  # 减去 1024+128
            # = w_original_float = int_to_float(w_quant)
            # 乘以 scale
            dequant_weight = fp16_val * scale[e, n]
            # 标准 FP16 GEMM
            C[e, :, :] += A_token @ dequant_weight
    # 输出：FP16 MoE layer output，与原 FP16 模型相同的激活精度
    ```
  - INT4 额外优化：权重 layout 重排 `[e0,e1,e2,e3,e4,e5,e6,e7] → [e0,e2,e4,e6,e1,e3,e5,e7]` 减少 bit 操作指令，减去的常数从 1152 变为 1032。
