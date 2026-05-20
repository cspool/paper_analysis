## ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出TCA-TBE（Tensor-Core-Aware Triple Bitmap Encoding），一种基于BF16权重指数字段冗余的固定长度无损压缩算法。核心设计：(1) 离线分析每层权重矩阵的exponent分布，识别top-7连续exponent值（覆盖>95%权重），编码为3-bit codeword (001-111)，超出top-7的outlier用codeword 000标记并全精度存储；(2) 将每个8×8 weight tile编码为三个独立64-bit bitmap（每个表示codeword的一个bit-plane）、一个PackedSignMantissa buffer（8-bit sign+mantissa）和一个FullValue buffer（完整BF16 fallback值）；(3) 平均每元素11.3 bits，压缩率约1.41×（16/11.3），接近理论下界10.6 bits。实验比较kernel-level speedup、end-to-end latency和throughput，对比cuBLAS_TC、DietGPU (rANS)、nvCOMP (rANS)、DFloat11 (Huffman)和vLLM/Transformers等baseline。算法保证bit-exact无损——解压缩后权重与原始BF16完全一致，无精度损失。

- 硬件平台是什么，配置是什么。
  三平台：(1) 4× NVIDIA RTX4090 (Ada Lovelace, 24GB, Compute Capability 8.9) + Intel Xeon Platinum 8352V (144 cores, 512GB DDR4)；(2) 4× NVIDIA L40S (Ada Lovelace, 48GB) + Intel Xeon Gold 6230R (104 cores, 512GB DDR4)；(3) NVIDIA RTX5090 (Blackwell, 32GB, CC 12.0)。编译：GCC 11.3 + NVCC 12.4 (RTX5090用NVCC 12.8)。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA3.1 (8B/70B/405B)、Qwen2.5 (7B/14B/32B/72B)、Gemma3 (12B/27B)、Mistral (24B/123B)。kernel benchmark使用这些模型真实weight矩阵的linear layer shapes（QKV_proj, O_proj, GateUp_proj, Down_proj, LM head），batch size 8/16/32。end-to-end evaluation使用LLaMA3.1-8B (RTX4090)、Mistral-24B (2×L40S)、LLaMA3.1-70B (4×L40S TP)，batch size 8/32，output length 128-2048 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  已开源：https://github.com/HPMLL/ZipServ_ASPLOS26.git。TCA-TBE算法pipeline：
  1. 离线压缩：对每个weight matrix，扫描exponent直方图→选择top-7连续exponent值→逐8×8 tile编码为triple bitmap + PackedSignMantissa + FullValue buffer。压缩LLaMA3.1-8B约2.5分钟（16核CPU）。
  2. 在线推理：解压时每个线程独立判断其负责元素的storage mode（通过bitwise OR三个bitmap得到64-bit spatial indicator mask），若indicator=1则从PackedSignMantissa读取sign+mantissa并通过base_exp+codeword算术恢复exponent重建BF16；若indicator=0则直接从FullValue读取完整BF16。codeword lookup通过base_exp + (bit2<<2 | bit1<<1 | bit0)算术实现，无需shared memory table lookup。
